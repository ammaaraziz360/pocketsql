"""Transformer with explicit role-specific schema-linking heads."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import mlx.core as mx
import mlx.nn as nn

from pocketsql.model.transformer import ModelConfig, PocketSQLTransformer


@dataclass(frozen=True)
class FactorizedSchemaConfig:
    max_table_slots: int = 16
    max_column_slots: int = 64
    max_projection_slots: int = 4
    max_filter_slots: int = 4
    max_group_slots: int = 2


class FactorizedPocketSQLTransformer(PocketSQLTransformer):
    """Retain autoregressive operation decoding while pointing to schema roles."""

    def __init__(self, config: ModelConfig, schema_config: FactorizedSchemaConfig | None = None) -> None:
        super().__init__(config)
        self.schema_config = schema_config or FactorizedSchemaConfig()
        width = config.hidden_dim
        tables = self.schema_config.max_table_slots
        columns = self.schema_config.max_column_slots
        projections = self.schema_config.max_projection_slots
        filters = self.schema_config.max_filter_slots
        groups = self.schema_config.max_group_slots

        self.schema_table = nn.Linear(width, tables)
        self.schema_join_table = nn.Linear(width, tables + 1)
        self.schema_projection_column = nn.Linear(width, projections * (columns + 1))
        self.schema_projection_owner = nn.Linear(width, projections * tables)
        self.schema_aggregate_column = nn.Linear(width, columns)
        self.schema_aggregate_owner = nn.Linear(width, tables)
        self.schema_join_column = nn.Linear(width, 2 * columns)
        self.schema_filter_column = nn.Linear(width, filters * columns)
        self.schema_filter_owner = nn.Linear(width, filters * tables)
        self.schema_group_column = nn.Linear(width, groups * columns)
        self.schema_group_owner = nn.Linear(width, groups * tables)
        self.schema_order_column = nn.Linear(width, columns)
        self.schema_order_owner = nn.Linear(width, tables)
        self.factorized_schema_linking = True

    def schema_link_logits_from_hidden(self, pooled: mx.array) -> dict[str, mx.array]:
        batch = pooled.shape[0]
        config = self.schema_config
        return {
            "table": self.schema_table(pooled),
            "join_table": self.schema_join_table(pooled),
            "projection_column": self.schema_projection_column(pooled).reshape(
                batch, config.max_projection_slots, config.max_column_slots + 1
            ),
            "projection_owner": self.schema_projection_owner(pooled).reshape(
                batch, config.max_projection_slots, config.max_table_slots
            ),
            "aggregate_column": self.schema_aggregate_column(pooled),
            "aggregate_owner": self.schema_aggregate_owner(pooled),
            "join_column": self.schema_join_column(pooled).reshape(batch, 2, config.max_column_slots),
            "filter_column": self.schema_filter_column(pooled).reshape(
                batch, config.max_filter_slots, config.max_column_slots
            ),
            "filter_owner": self.schema_filter_owner(pooled).reshape(
                batch, config.max_filter_slots, config.max_table_slots
            ),
            "group_column": self.schema_group_column(pooled).reshape(
                batch, config.max_group_slots, config.max_column_slots
            ),
            "group_owner": self.schema_group_owner(pooled).reshape(
                batch, config.max_group_slots, config.max_table_slots
            ),
            "order_column": self.schema_order_column(pooled),
            "order_owner": self.schema_order_owner(pooled),
        }

    def forward_with_schema_links(
        self,
        tokens: mx.array,
        prompt_positions: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        hidden = self.hidden_states(tokens)
        pooled = hidden[mx.arange(hidden.shape[0]), prompt_positions]
        return hidden @ self.embedding.weight.T, self.schema_link_logits_from_hidden(pooled)

    def schema_link_logits(self, prompt_tokens: mx.array) -> dict[str, mx.array]:
        hidden = self.hidden_states(prompt_tokens)
        return self.schema_link_logits_from_hidden(hidden[:, -1])


def decode_schema_link_logits(
    logits: dict[str, mx.array],
    mappings: list,
    config: FactorizedSchemaConfig,
) -> list[dict]:
    """Decode only physical table-column pairs and declared join relationships."""
    values = {name: tensor.tolist() for name, tensor in logits.items()}
    predictions = []

    def select(candidates):
        """Return the best constrained candidate and its normalized confidence."""
        candidates = tuple(candidates)
        if not candidates:
            raise ValueError("factorized schema decoding has no valid candidates")
        winner = max(candidates, key=lambda item: item[0])
        peak = winner[0]
        denominator = sum(math.exp(item[0] - peak) for item in candidates)
        return winner, 1.0 / denominator

    def slot_index(slot: str, prefix: str) -> int:
        match = re.fullmatch(rf"{prefix}(\d+)", slot)
        if not match:
            raise ValueError(f"invalid canonical slot: {slot!r}")
        return int(match.group(1))

    for row, mapping in enumerate(mappings):
        table_count = len(mapping.table_to_slot)
        column_count = len(mapping.column_to_slot)
        if not 0 < table_count <= config.max_table_slots:
            raise ValueError(f"factorized inference supports 1..{config.max_table_slots} tables")
        if not 0 < column_count <= config.max_column_slots:
            raise ValueError(f"factorized inference supports 1..{config.max_column_slots} columns")
        owner_columns = {index: set() for index in range(table_count)}
        for raw_column, raw_tables in mapping.column_to_tables.items():
            column = slot_index(mapping.column_to_slot[raw_column], "column")
            for raw_table in raw_tables:
                owner = slot_index(mapping.table_to_slot[raw_table], "table")
                owner_columns[owner].add(column)

        table_join_candidates = []
        for base in range(table_count):
            table_join_candidates.append(
                (
                    values["table"][row][base]
                    + values["join_table"][row][config.max_table_slots],
                    base,
                    config.max_table_slots,
                )
            )
            for joined in range(table_count):
                if base == joined or not mapping.declared_joins(f"table{base}", f"table{joined}"):
                    continue
                table_join_candidates.append(
                    (
                        values["table"][row][base] + values["join_table"][row][joined],
                        base,
                        joined,
                    )
                )
        (_, table, raw_join), table_join_confidence = select(table_join_candidates)
        join_table = None if raw_join == config.max_table_slots else raw_join
        owner_candidates = (table, join_table) if join_table is not None else (table,)

        def role_pairs(column_name: str, owner_name: str, allow_star: bool = False):
            column_rows = values[column_name][row]
            owner_rows = values[owner_name][row]
            if not column_rows or not isinstance(column_rows[0], list):
                column_rows = [column_rows]
                owner_rows = [owner_rows]
            decoded = []
            confidences = []
            for column_scores, owner_scores in zip(column_rows, owner_rows):
                candidates = []
                for owner in owner_candidates:
                    for column in owner_columns[owner]:
                        candidates.append((owner_scores[owner] + column_scores[column], owner, column))
                    if allow_star:
                        candidates.append(
                            (
                                owner_scores[owner] + column_scores[config.max_column_slots],
                                owner,
                                config.max_column_slots,
                            )
                        )
                (_, owner, column), confidence = select(candidates)
                decoded.append((owner, column))
                confidences.append(confidence)
            return tuple(decoded), tuple(confidences)

        projection, projection_confidence = role_pairs(
            "projection_column", "projection_owner", allow_star=True
        )
        aggregates, aggregate_confidences = role_pairs("aggregate_column", "aggregate_owner")
        aggregate = aggregates[0]
        aggregate_confidence = aggregate_confidences[0]
        filters, filter_confidence = role_pairs("filter_column", "filter_owner")
        groups, group_confidence = role_pairs("group_column", "group_owner")
        orders, order_confidences = role_pairs("order_column", "order_owner")
        order = orders[0]
        order_confidence = order_confidences[0]

        join_columns = (0, 0)
        join_column_confidence = 0.0
        if join_table is not None:
            join_candidates = []
            for left, right in mapping.declared_joins(f"table{table}", f"table{join_table}"):
                left_table, left_column = left.split(".")
                right_table, right_column = right.split(".")
                if left_table == f"table{table}":
                    base_column = slot_index(left_column, "column")
                    joined_column = slot_index(right_column, "column")
                else:
                    base_column = slot_index(right_column, "column")
                    joined_column = slot_index(left_column, "column")
                score = (
                    values["join_column"][row][0][base_column]
                    + values["join_column"][row][1][joined_column]
                )
                join_candidates.append((score, base_column, joined_column))
            (_, base_column, joined_column), join_column_confidence = select(join_candidates)
            join_columns = (base_column, joined_column)

        predictions.append(
            {
                "star_column": config.max_column_slots,
                "table": table,
                "join_table": join_table,
                "projection_column": tuple(column for _, column in projection),
                "projection_owner": tuple(owner for owner, _ in projection),
                "aggregate_column": aggregate[1],
                "aggregate_owner": aggregate[0],
                "join_column": join_columns,
                "filter_column": tuple(column for _, column in filters),
                "filter_owner": tuple(owner for owner, _ in filters),
                "group_column": tuple(column for _, column in groups),
                "group_owner": tuple(owner for owner, _ in groups),
                "order_column": order[1],
                "order_owner": order[0],
                "confidence": {
                    "table_join": table_join_confidence,
                    "projection": projection_confidence,
                    "aggregate": aggregate_confidence,
                    "join_column": join_column_confidence,
                    "filter": filter_confidence,
                    "group": group_confidence,
                    "order": order_confidence,
                },
            }
        )
    return predictions
