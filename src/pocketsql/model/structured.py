"""V18 token-level schema linking and structured query planning heads."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import mlx.core as mx
import mlx.nn as nn

from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.model.factorized import FactorizedPocketSQLTransformer, FactorizedSchemaConfig
from pocketsql.model.transformer import ModelConfig


AGGREGATE_CLASSES = (None, "COUNT", "SUM", "AVG", "MIN", "MAX")
FILTER_OPERATOR_CLASSES = ("=", ">", "<", ">=", "<=")


@dataclass(frozen=True)
class StructuredQueryConfig:
    max_literal_span_tokens: int = 12
    max_limit_value: int = 100


def _masked_logits(logits: mx.array, mask: mx.array) -> mx.array:
    return mx.where(mask, logits, mx.full(logits.shape, -1e9))


class StructuredPocketSQLTransformer(FactorizedPocketSQLTransformer):
    """Use explicit token-to-schema attention plus operation and copy heads."""

    def __init__(
        self,
        config: ModelConfig,
        schema_config: FactorizedSchemaConfig | None = None,
        structured_config: StructuredQueryConfig | None = None,
    ) -> None:
        super().__init__(config, schema_config)
        self.structured_config = structured_config or StructuredQueryConfig()
        width = config.hidden_dim
        schema = self.schema_config
        structured = self.structured_config

        self.schema_cross_candidate_query = nn.Linear(width, width, bias=False)
        self.schema_cross_question_key = nn.Linear(width, width, bias=False)
        self.schema_cross_question_value = nn.Linear(width, width, bias=False)
        self.schema_cross_output = nn.Linear(width, width, bias=False)
        self.schema_cross_norm = nn.LayerNorm(width)
        self.schema_cross_table_roles = nn.Linear(width, 2)
        self.schema_cross_join_none = nn.Linear(width, 1)

        projection_roles = schema.max_projection_slots
        column_roles = (
            projection_roles
            + 1
            + 2
            + schema.max_filter_slots
            + schema.max_group_slots
            + 1
        )
        owner_roles = (
            projection_roles
            + 1
            + schema.max_filter_slots
            + schema.max_group_slots
            + 1
        )
        self.schema_cross_column_roles = nn.Linear(width, column_roles)
        self.schema_cross_owner_roles = nn.Linear(width, owner_roles)
        self.schema_cross_projection_star = nn.Linear(width, projection_roles)

        self.operation_selection_arity = nn.Linear(width, schema.max_projection_slots + 1)
        self.operation_aggregate = nn.Linear(width, len(AGGREGATE_CLASSES))
        self.operation_aggregate_target = nn.Linear(width, 2)
        self.operation_aggregate_position = nn.Linear(width, schema.max_projection_slots + 1)
        self.operation_distinct = nn.Linear(width, 2)
        self.operation_join_presence = nn.Linear(width, 2)
        self.operation_filter_count = nn.Linear(width, schema.max_filter_slots + 1)
        self.operation_filter_operator = nn.Linear(
            width, schema.max_filter_slots * len(FILTER_OPERATOR_CLASSES)
        )
        self.operation_filter_connector = nn.Linear(width, 2)
        self.operation_group_count = nn.Linear(width, schema.max_group_slots + 1)
        self.operation_order_presence = nn.Linear(width, 2)
        self.operation_descending = nn.Linear(width, 2)
        self.operation_limit = nn.Linear(width, structured.max_limit_value + 1)

        self.operation_norm = nn.LayerNorm(width)
        self.literal_filter_context = nn.Linear(width, width, bias=False)
        self.literal_filter_slots = nn.Embedding(schema.max_filter_slots, width)
        self.literal_start_query = nn.Linear(width, width, bias=False)
        self.literal_end_query = nn.Linear(width, width, bias=False)
        self.structured_query_planning = True

    @staticmethod
    def _gather(hidden: mx.array, positions: mx.array) -> mx.array:
        batch = hidden.shape[0]
        return hidden[mx.arange(batch)[:, None], positions]

    def _question_summary(self, hidden: mx.array, question_mask: mx.array) -> mx.array:
        active = question_mask.astype(hidden.dtype)[..., None]
        return mx.sum(hidden * active, axis=1) / mx.maximum(mx.sum(active, axis=1), 1)

    def _contextualize_candidates(
        self,
        candidates: mx.array,
        hidden: mx.array,
        question_mask: mx.array,
    ) -> mx.array:
        query = self.schema_cross_candidate_query(candidates)
        key = self.schema_cross_question_key(hidden)
        value = self.schema_cross_question_value(hidden)
        scores = (query @ key.transpose(0, 2, 1)) / math.sqrt(hidden.shape[-1])
        scores = _masked_logits(scores, question_mask[:, None, :])
        context = mx.softmax(scores, axis=-1) @ value
        return self.schema_cross_norm(candidates + self.schema_cross_output(context))

    def structured_logits_from_hidden(
        self,
        hidden: mx.array,
        layout: dict[str, mx.array],
    ) -> tuple[dict[str, mx.array], dict[str, mx.array], dict[str, mx.array]]:
        table_candidates = self._gather(hidden, layout["table_positions"])
        column_candidates = self._gather(hidden, layout["column_positions"])
        table_candidates = self._contextualize_candidates(
            table_candidates, hidden, layout["question_mask"]
        )
        column_candidates = self._contextualize_candidates(
            column_candidates, hidden, layout["question_mask"]
        )
        question = self._question_summary(hidden, layout["question_mask"])
        prompt = hidden[
            mx.arange(hidden.shape[0]), layout["prompt_positions"]
        ]
        summary = self.operation_norm(question + prompt)

        table_roles = self.schema_cross_table_roles(table_candidates)
        table_logits = _masked_logits(table_roles[:, :, 0], layout["table_mask"])
        join_logits = _masked_logits(table_roles[:, :, 1], layout["table_mask"])
        join_logits = mx.concatenate((join_logits, self.schema_cross_join_none(summary)), axis=-1)

        column_roles = self.schema_cross_column_roles(column_candidates).transpose(0, 2, 1)
        column_roles = _masked_logits(column_roles, layout["column_mask"][:, None, :])
        owner_roles = self.schema_cross_owner_roles(table_candidates).transpose(0, 2, 1)
        owner_roles = _masked_logits(owner_roles, layout["table_mask"][:, None, :])
        schema = self.schema_config
        offset = 0
        projection_column = column_roles[:, offset : offset + schema.max_projection_slots]
        offset += schema.max_projection_slots
        projection_star = self.schema_cross_projection_star(summary)[:, :, None]
        projection_column = mx.concatenate((projection_column, projection_star), axis=-1)
        aggregate_column = column_roles[:, offset]
        offset += 1
        join_column = column_roles[:, offset : offset + 2]
        offset += 2
        filter_column = column_roles[:, offset : offset + schema.max_filter_slots]
        offset += schema.max_filter_slots
        group_column = column_roles[:, offset : offset + schema.max_group_slots]
        offset += schema.max_group_slots
        order_column = column_roles[:, offset]

        owner_offset = 0
        projection_owner = owner_roles[
            :, owner_offset : owner_offset + schema.max_projection_slots
        ]
        owner_offset += schema.max_projection_slots
        aggregate_owner = owner_roles[:, owner_offset]
        owner_offset += 1
        filter_owner = owner_roles[:, owner_offset : owner_offset + schema.max_filter_slots]
        owner_offset += schema.max_filter_slots
        group_owner = owner_roles[:, owner_offset : owner_offset + schema.max_group_slots]
        owner_offset += schema.max_group_slots
        order_owner = owner_roles[:, owner_offset]

        schema_logits = {
            "table": table_logits,
            "join_table": join_logits,
            "projection_column": projection_column,
            "projection_owner": projection_owner,
            "aggregate_column": aggregate_column,
            "aggregate_owner": aggregate_owner,
            "join_column": join_column,
            "filter_column": filter_column,
            "filter_owner": filter_owner,
            "group_column": group_column,
            "group_owner": group_owner,
            "order_column": order_column,
            "order_owner": order_owner,
        }

        operation_logits = {
            "selection_arity": self.operation_selection_arity(summary),
            "aggregate": self.operation_aggregate(summary),
            "aggregate_target": self.operation_aggregate_target(summary),
            "aggregate_position": self.operation_aggregate_position(summary),
            "distinct": self.operation_distinct(summary),
            "join_presence": self.operation_join_presence(summary),
            "filter_count": self.operation_filter_count(summary),
            "filter_operator": self.operation_filter_operator(summary).reshape(
                hidden.shape[0], schema.max_filter_slots, len(FILTER_OPERATOR_CLASSES)
            ),
            "filter_connector": self.operation_filter_connector(summary),
            "group_count": self.operation_group_count(summary),
            "order_presence": self.operation_order_presence(summary),
            "descending": self.operation_descending(summary),
            "limit": self.operation_limit(summary),
        }

        # Each filter predicts a start/end span directly over question tokens.
        # Conditioning on the softly selected filter column helps distinguish
        # multiple values in requests such as ``status shipped and city Austin``.
        filter_weights = mx.softmax(filter_column, axis=-1)
        filter_context = filter_weights @ column_candidates
        slot_context = self.literal_filter_slots(
            mx.arange(schema.max_filter_slots)
        )[None, :, :]
        literal_seed = (
            summary[:, None, :]
            + self.literal_filter_context(filter_context)
            + slot_context
        )
        start_scores = self.literal_start_query(literal_seed) @ hidden.transpose(0, 2, 1)
        end_scores = self.literal_end_query(literal_seed) @ hidden.transpose(0, 2, 1)
        start_scores = _masked_logits(
            start_scores / math.sqrt(hidden.shape[-1]),
            layout["question_mask"][:, None, :],
        )
        end_scores = _masked_logits(
            end_scores / math.sqrt(hidden.shape[-1]),
            layout["question_mask"][:, None, :],
        )
        literal_logits = {
            "filter_start": start_scores,
            "filter_end": end_scores,
        }
        return schema_logits, operation_logits, literal_logits

    def structured_logits(
        self,
        prompt_tokens: mx.array,
        layout: dict[str, mx.array],
    ) -> tuple[dict[str, mx.array], dict[str, mx.array], dict[str, mx.array]]:
        return self.structured_logits_from_hidden(self.hidden_states(prompt_tokens), layout)


def decode_operation_logits(logits: dict[str, mx.array]) -> list[dict]:
    values = {name: mx.argmax(value, axis=-1).tolist() for name, value in logits.items()}
    records = len(values["aggregate"])
    return [
        {name: value[index] for name, value in values.items()}
        for index in range(records)
    ]


def decode_literal_logits(
    logits: dict[str, mx.array],
    prompts: list[str],
    layouts: list[dict],
    max_span_tokens: int,
) -> list[tuple[str | None, ...]]:
    """Decode the highest-scoring valid question span for every filter slot."""
    starts = logits["filter_start"].tolist()
    ends = logits["filter_end"].tolist()
    decoded = []
    for prompt, layout, start_rows, end_rows in zip(prompts, layouts, starts, ends):
        positions = [
            index for index, active in enumerate(layout["question_mask"]) if active
        ]
        values = []
        for start_scores, end_scores in zip(start_rows, end_rows):
            best = None
            for start in positions:
                for end in positions:
                    if end < start or end - start + 1 > max_span_tokens:
                        continue
                    score = start_scores[start] + end_scores[end]
                    if best is None or score > best[0]:
                        best = (score, start, end)
            if best is None:
                values.append(None)
                continue
            _, start, end = best
            left = layout["offsets"][start][0]
            right = layout["offsets"][end][1]
            value = prompt[left:right].strip().strip("\"'")
            value = value.rstrip("?.!,")
            values.append(value or None)
        decoded.append(tuple(values))
    return decoded


def structured_query_plan(
    operation: dict,
    links: dict,
    literal_values: tuple[str | None, ...],
    mapping,
    config: StructuredQueryConfig,
) -> QueryPlan | None:
    """Assemble independently predicted roles into one validated canonical plan."""
    aggregate = AGGREGATE_CLASSES[operation["aggregate"]]
    selection_arity = operation["selection_arity"]
    if not aggregate and selection_arity == 0:
        return None
    joined = bool(operation["join_presence"])
    table = f"table{links['table']}"
    join_table = f"table{links['join_table']}" if joined and links["join_table"] is not None else None
    if joined and join_table is None:
        return None

    def reference(owner: int, column: int, allow_star: bool = False) -> str:
        owner_slot = f"table{owner}"
        if allow_star and column == links["star_column"]:
            return f"{owner_slot}.*" if joined else "*"
        column_slot = f"column{column}"
        return f"{owner_slot}.{column_slot}" if joined else column_slot

    columns = tuple(
        reference(links["projection_owner"][index], links["projection_column"][index], True)
        for index in range(selection_arity)
    )
    aggregate_column = None
    if aggregate and (aggregate != "COUNT" or operation["aggregate_target"]):
        aggregate_column = reference(links["aggregate_owner"], links["aggregate_column"])

    filter_count = operation["filter_count"]
    filters = []
    for index in range(filter_count):
        value = literal_values[index]
        if value is None:
            return None
        if not re.fullmatch(r"value\d+", value):
            # A copied phrase can contain a canonical sub-value, for example
            # ``$value0 usd``. Restore that phrase before it is serialized as
            # one quoted semantic-plan literal.
            value = mapping.restore(value)
        if re.fullmatch(r"-?(?:0|[1-9]\d*)", value):
            value = int(value)
        elif re.fullmatch(r"-?(?:0|[1-9]\d*)\.\d+", value):
            value = float(value)
        filters.append(
            Filter(
                reference(links["filter_owner"][index], links["filter_column"][index]),
                FILTER_OPERATOR_CLASSES[operation["filter_operator"][index]],
                value,
            )
        )

    group_by = tuple(
        reference(links["group_owner"][index], links["group_column"][index])
        for index in range(operation["group_count"])
    )
    order_by = None
    if operation["order_presence"]:
        order_by = reference(links["order_owner"], links["order_column"])

    join_on = None
    if join_table:
        declared = mapping.declared_joins(table, join_table)
        if len(declared) == 1:
            join_on = declared[0]
        else:
            join_on = (
                reference(links["table"], links["join_column"][0]),
                reference(links["join_table"], links["join_column"][1]),
            )
            if not any(join_on in {item, item[::-1]} for item in declared):
                return None

    aggregate_position = min(operation["aggregate_position"], len(columns))
    return QueryPlan(
        family="structured_v18",
        table=table,
        columns=columns,
        aggregate=aggregate,
        aggregate_column=aggregate_column,
        distinct=bool(operation["distinct"]),
        filters=tuple(filters),
        filter_connector="OR" if operation["filter_connector"] else "AND",
        group_by=group_by,
        order_by=order_by,
        descending=bool(operation["descending"]) if order_by else False,
        limit=operation["limit"] or None,
        join_table=join_table,
        join_on=join_on,
        aggregate_position=aggregate_position,
    )


def _slot_number(slot: str, prefix: str) -> int:
    match = re.fullmatch(rf"{prefix}(\d+)", slot)
    if not match:
        raise ValueError(f"invalid {prefix} slot: {slot!r}")
    return int(match.group(1))


def prompt_layout(
    prompt: str,
    tokenizer,
    mapping,
    schema_config: FactorizedSchemaConfig,
    structured_config: StructuredQueryConfig,
) -> dict[str, list]:
    """Locate schema candidates, question tokens, and copy candidates in a prompt."""
    token_ids, offsets = tokenizer.encode_with_offsets(prompt)
    schema_start = prompt.index("<schema>") + len("<schema>")
    schema_end = prompt.index("</schema>")
    question_start = prompt.index("<question>") + len("<question>")
    question_end = prompt.index("</question>")
    prompt_start = prompt.index("<sql>")

    def token_for_span(start: int, end: int) -> int:
        matches = [
            index
            for index, (left, right) in enumerate(offsets)
            if left < end and start < right
        ]
        if not matches:
            raise ValueError(f"tokenizer produced no token for prompt span {start}:{end}")
        return matches[-1]

    def candidate_positions(slots: list[str], maximum: int) -> tuple[list[int], list[bool]]:
        positions = [0] * maximum
        mask = [False] * maximum
        schema_text = prompt[schema_start:schema_end]
        for slot in slots:
            index = _slot_number(slot, "table" if slot.startswith("table") else "column")
            if index >= maximum:
                raise ValueError(f"candidate slot {slot} exceeds configured capacity {maximum}")
            match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(slot)}(?![A-Za-z0-9_])", schema_text)
            if not match:
                raise ValueError(f"candidate slot {slot} is absent from canonical schema")
            positions[index] = token_for_span(
                schema_start + match.start(), schema_start + match.end()
            )
            mask[index] = True
        return positions, mask

    table_slots = sorted(mapping.table_to_slot.values(), key=lambda value: _slot_number(value, "table"))
    column_slots = sorted(mapping.column_to_slot.values(), key=lambda value: _slot_number(value, "column"))
    table_positions, table_mask = candidate_positions(
        table_slots, schema_config.max_table_slots
    )
    column_positions, column_mask = candidate_positions(
        column_slots, schema_config.max_column_slots
    )
    question_mask = [
        left < question_end and question_start < right
        for left, right in offsets
    ]
    return {
        "tokens": token_ids,
        "offsets": offsets,
        "prompt_position": token_for_span(prompt_start, prompt_start + len("<sql>")),
        "table_positions": table_positions,
        "table_mask": table_mask,
        "column_positions": column_positions,
        "column_mask": column_mask,
        "question_mask": question_mask,
    }


def literal_span_labels(
    prompt: str,
    tokenizer,
    values: list[str | int | float],
    max_filter_slots: int,
) -> tuple[dict[str, list[int]], dict[str, list[bool]]]:
    """Map every gold filter value to its exact question-token span."""
    _, offsets = tokenizer.encode_with_offsets(prompt)
    question_start = prompt.index("<question>") + len("<question>")
    question_end = prompt.index("</question>")
    question = prompt[question_start:question_end]
    starts = [0] * max_filter_slots
    ends = [0] * max_filter_slots
    masks = [False] * max_filter_slots
    if len(values) > max_filter_slots:
        raise ValueError("literal supervision exceeds max_filter_slots")

    for index, value in enumerate(values):
        raw = str(value)
        match = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(raw)}(?![A-Za-z0-9_])",
            question,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                f"filter literal {value!r} is absent from canonical question {question!r}"
            )
        left = question_start + match.start()
        right = question_start + match.end()
        token_positions = [
            token_index
            for token_index, (token_left, token_right) in enumerate(offsets)
            if token_left < right and left < token_right
        ]
        if not token_positions:
            raise ValueError(f"tokenizer produced no span for filter literal {value!r}")
        starts[index] = token_positions[0]
        ends[index] = token_positions[-1]
        masks[index] = True
    labels = {"filter_start": starts, "filter_end": ends}
    active = {"filter_start": masks, "filter_end": list(masks)}
    return labels, active
