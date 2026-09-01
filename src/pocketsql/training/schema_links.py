"""Canonical supervision for factorized schema-linking heads."""
from __future__ import annotations

import re

from pocketsql.model.schema_grounding import (
    canonicalize_inputs,
    canonicalize_record,
    identifier_mapping,
)
from pocketsql.model.semantic_plan import query_plan_from_dict
from pocketsql.model.structured import (
    AGGREGATE_CLASSES,
    FILTER_OPERATOR_CLASSES,
    literal_span_labels,
    prompt_layout,
)


TABLE_SLOT = re.compile(r"^table(\d+)$")
COLUMN_SLOT = re.compile(r"^(?:table\d+\.)?column(\d+)$")


def _slot_index(value: str, pattern: re.Pattern[str], kind: str) -> int:
    match = pattern.fullmatch(value)
    if not match:
        raise ValueError(f"factorized schema targets require canonical {kind} slots, got {value!r}")
    return int(match.group(1))


def _owner_index(reference: str, plan, mapping) -> int:
    qualifier, separator, column_slot = reference.rpartition(".")
    if separator and TABLE_SLOT.fullmatch(qualifier):
        return _slot_index(qualifier, TABLE_SLOT, "table")
    if reference == "*":
        return _slot_index(plan.table, TABLE_SLOT, "table")
    if not separator:
        column_slot = reference
    raw_column = mapping.slot_to_raw.get(column_slot)
    selected_tables = {plan.table, plan.join_table} - {None}
    owners = [
        mapping.table_to_slot[raw_table]
        for raw_table in mapping.column_to_tables.get(raw_column, ())
        if mapping.table_to_slot.get(raw_table) in selected_tables
    ]
    if len(owners) == 1:
        return _slot_index(owners[0], TABLE_SLOT, "table")
    return _slot_index(plan.table, TABLE_SLOT, "table")


def _column_index(reference: str, max_columns: int) -> int:
    if reference == "*" or reference.endswith(".*"):
        return max_columns
    return _slot_index(reference, COLUMN_SLOT, "column")


def schema_link_batch(
    records: list[dict],
    *,
    identifier_slot_strategy: str,
    canonicalize_literals: bool,
    schema_linking_hints: bool,
    max_table_slots: int,
    max_column_slots: int,
    max_projection_slots: int,
    max_filter_slots: int,
    max_group_slots: int,
) -> tuple[dict[str, list], dict[str, list]]:
    """Return fixed-shape class labels and active masks for every pointer role."""
    labels = {
        "table": [],
        "join_table": [],
        "projection_column": [],
        "projection_owner": [],
        "aggregate_column": [],
        "aggregate_owner": [],
        "join_column": [],
        "filter_column": [],
        "filter_owner": [],
        "group_column": [],
        "group_owner": [],
        "order_column": [],
        "order_owner": [],
    }
    masks = {name: [] for name in labels}

    for record in records:
        mapping = identifier_mapping(record["schema_sql"], identifier_slot_strategy)
        if len(mapping.table_to_slot) > max_table_slots:
            raise ValueError(
                f"schema has {len(mapping.table_to_slot)} table slots, exceeding max_table_slots={max_table_slots}"
            )
        if len(mapping.column_to_slot) > max_column_slots:
            raise ValueError(
                f"schema has {len(mapping.column_to_slot)} column slots, exceeding max_column_slots={max_column_slots}"
            )
        canonical = canonicalize_record(
            record,
            identifier_slot_strategy,
            canonicalize_literals,
            schema_linking_hints,
        )
        plan = query_plan_from_dict(canonical["query_plan"])

        table = _slot_index(plan.table, TABLE_SLOT, "table")
        join_table = (
            _slot_index(plan.join_table, TABLE_SLOT, "table") if plan.join_table else max_table_slots
        )
        labels["table"].append(table)
        masks["table"].append(True)
        labels["join_table"].append(join_table)
        masks["join_table"].append(True)

        projection_columns = [0] * max_projection_slots
        projection_owners = [0] * max_projection_slots
        projection_mask = [False] * max_projection_slots
        if len(plan.columns) > max_projection_slots:
            raise ValueError(
                f"plan has {len(plan.columns)} projections, exceeding max_projection_slots={max_projection_slots}"
            )
        for index, reference in enumerate(plan.columns):
            projection_columns[index] = _column_index(reference, max_column_slots)
            projection_owners[index] = _owner_index(reference, plan, mapping)
            projection_mask[index] = True
        labels["projection_column"].append(projection_columns)
        labels["projection_owner"].append(projection_owners)
        masks["projection_column"].append(projection_mask)
        masks["projection_owner"].append(projection_mask)

        aggregate_active = plan.aggregate_column is not None
        labels["aggregate_column"].append(
            _column_index(plan.aggregate_column, max_column_slots) if aggregate_active else 0
        )
        labels["aggregate_owner"].append(
            _owner_index(plan.aggregate_column, plan, mapping) if aggregate_active else 0
        )
        masks["aggregate_column"].append(aggregate_active)
        masks["aggregate_owner"].append(aggregate_active)

        join_columns = [0, 0]
        join_mask = [False, False]
        if plan.join_on:
            for index, reference in enumerate(plan.join_on):
                join_columns[index] = _column_index(reference, max_column_slots)
                join_mask[index] = True
        labels["join_column"].append(join_columns)
        masks["join_column"].append(join_mask)

        filter_columns = [0] * max_filter_slots
        filter_owners = [0] * max_filter_slots
        filter_mask = [False] * max_filter_slots
        if len(plan.filters) > max_filter_slots:
            raise ValueError(
                f"plan has {len(plan.filters)} filters, exceeding max_filter_slots={max_filter_slots}"
            )
        for index, item in enumerate(plan.filters):
            filter_columns[index] = _column_index(item.column, max_column_slots)
            filter_owners[index] = _owner_index(item.column, plan, mapping)
            filter_mask[index] = True
        labels["filter_column"].append(filter_columns)
        labels["filter_owner"].append(filter_owners)
        masks["filter_column"].append(filter_mask)
        masks["filter_owner"].append(filter_mask)

        group_columns = [0] * max_group_slots
        group_owners = [0] * max_group_slots
        group_mask = [False] * max_group_slots
        if len(plan.group_by) > max_group_slots:
            raise ValueError(
                f"plan has {len(plan.group_by)} groups, exceeding max_group_slots={max_group_slots}"
            )
        for index, reference in enumerate(plan.group_by):
            group_columns[index] = _column_index(reference, max_column_slots)
            group_owners[index] = _owner_index(reference, plan, mapping)
            group_mask[index] = True
        labels["group_column"].append(group_columns)
        labels["group_owner"].append(group_owners)
        masks["group_column"].append(group_mask)
        masks["group_owner"].append(group_mask)

        order_active = plan.order_by is not None
        labels["order_column"].append(
            _column_index(plan.order_by, max_column_slots) if order_active else 0
        )
        labels["order_owner"].append(
            _owner_index(plan.order_by, plan, mapping) if order_active else 0
        )
        masks["order_column"].append(order_active)
        masks["order_owner"].append(order_active)

    return labels, masks


def structured_query_batch(
    records: list[dict],
    *,
    identifier_slot_strategy: str,
    canonicalize_literals: bool,
    schema_linking_hints: bool,
    max_projection_slots: int,
    max_filter_slots: int,
    max_group_slots: int,
    max_limit_value: int,
) -> tuple[dict[str, list], dict[str, list]]:
    """Return operation-skeleton and question-literal copy supervision."""
    operation_labels = {
        "selection_arity": [],
        "aggregate": [],
        "aggregate_target": [],
        "aggregate_position": [],
        "distinct": [],
        "join_presence": [],
        "filter_count": [],
        "filter_operator": [],
        "filter_connector": [],
        "group_count": [],
        "order_presence": [],
        "descending": [],
        "limit": [],
    }
    operation_masks = {name: [] for name in operation_labels}
    for record in records:
        canonical = canonicalize_record(
            record,
            identifier_slot_strategy,
            canonicalize_literals,
            schema_linking_hints,
        )
        plan = query_plan_from_dict(canonical["query_plan"])
        if len(plan.columns) > max_projection_slots:
            raise ValueError("operation target exceeds max_projection_slots")
        if len(plan.filters) > max_filter_slots:
            raise ValueError("operation target exceeds max_filter_slots")
        if len(plan.group_by) > max_group_slots:
            raise ValueError("operation target exceeds max_group_slots")
        if plan.limit is not None and plan.limit > max_limit_value:
            raise ValueError(
                f"LIMIT {plan.limit} exceeds max_limit_value={max_limit_value}"
            )

        aggregate = AGGREGATE_CLASSES.index(plan.aggregate)
        aggregate_active = plan.aggregate is not None
        order_active = plan.order_by is not None
        filter_count = len(plan.filters)
        scalar_values = {
            "selection_arity": len(plan.columns),
            "aggregate": aggregate,
            "aggregate_target": int(plan.aggregate_column is not None),
            "aggregate_position": plan.aggregate_position,
            "distinct": int(plan.distinct),
            "join_presence": int(plan.join_table is not None),
            "filter_count": filter_count,
            "filter_connector": int(plan.filter_connector == "OR"),
            "group_count": len(plan.group_by),
            "order_presence": int(order_active),
            "descending": int(plan.descending),
            "limit": plan.limit or 0,
        }
        for name, value in scalar_values.items():
            operation_labels[name].append(value)
            active = True
            if name in {"aggregate_target", "aggregate_position"}:
                active = aggregate_active
            elif name == "filter_connector":
                active = filter_count > 1
            elif name == "descending":
                active = order_active
            operation_masks[name].append(active)

        operators = [0] * max_filter_slots
        operator_mask = [False] * max_filter_slots
        for index, item in enumerate(plan.filters):
            operators[index] = FILTER_OPERATOR_CLASSES.index(item.operator)
            operator_mask[index] = True
        operation_labels["filter_operator"].append(operators)
        operation_masks["filter_operator"].append(operator_mask)

    return operation_labels, operation_masks


def structured_prompt_batch(
    records, tokenizer, model, config
) -> tuple[list[list[int]], dict, dict[str, list], dict[str, list]]:
    """Encode prompt-only rows and fixed-shape token layouts without target leakage."""
    layouts = []
    literal_labels = {"filter_start": [], "filter_end": []}
    literal_masks = {"filter_start": [], "filter_end": []}
    for record in records:
        hints_field = config.get("factorized_schema_linking_hints_field")
        use_hints = config.get("factorized_schema_linking_hints", False) and (
            bool(record.get(hints_field)) if hints_field else True
        )
        schema, question, mapping = canonicalize_inputs(
            record["schema_sql"],
            record["question"],
            config.get("identifier_slot_strategy", "ordered"),
            config.get("canonicalize_literals", False),
            use_hints,
            config.get("schema_linking_max_tables", 5),
            config.get("schema_linking_max_columns", 8),
        )
        prompt = f"<bos><schema>{schema}</schema><question>{question}</question><sql>"
        layout = prompt_layout(
            prompt,
            tokenizer,
            mapping,
            model.schema_config,
            model.structured_config,
        )
        canonical = canonicalize_record(
            record,
            config.get("identifier_slot_strategy", "ordered"),
            config.get("canonicalize_literals", False),
            use_hints,
            config.get("schema_linking_max_tables", 5),
            config.get("schema_linking_max_columns", 8),
        )
        plan = query_plan_from_dict(canonical["query_plan"])
        raw_filters = record.get("query_plan", {}).get("filters", ())
        def copy_value(raw_item, canonical_item):
            candidates = (
                str(canonical_item.value),
                mapping.canonicalize_question(str(raw_item["value"])),
                str(raw_item["value"]),
            )
            for candidate in candidates:
                if re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(candidate)}(?![A-Za-z0-9_])",
                    question,
                    re.IGNORECASE,
                ):
                    return candidate
            return candidates[0]

        copy_values = [
            copy_value(raw_item, canonical_item)
            for raw_item, canonical_item in zip(raw_filters, plan.filters)
        ]
        if len(copy_values) != len(plan.filters):
            raise ValueError("canonical and raw filter counts do not match")
        labels, masks = literal_span_labels(
            prompt,
            tokenizer,
            copy_values,
            model.schema_config.max_filter_slots,
        )
        for name in literal_labels:
            literal_labels[name].append(labels[name])
            literal_masks[name].append(masks[name])
        if len(layout["tokens"]) > config["context_length"]:
            raise ValueError("structured prompt exceeds context length")
        layouts.append(layout)

    width = max(len(item["tokens"]) for item in layouts)
    prompt_rows = [
        item["tokens"] + [tokenizer.pad_id] * (width - len(item["tokens"]))
        for item in layouts
    ]
    keys = (
        "prompt_position",
        "table_positions",
        "table_mask",
        "column_positions",
        "column_mask",
    )
    batched = {
        "prompt_positions" if key == "prompt_position" else key: [
            item[key] for item in layouts
        ]
        for key in keys
    }
    batched["question_mask"] = [
        item["question_mask"] + [False] * (width - len(item["question_mask"]))
        for item in layouts
    ]
    return prompt_rows, batched, literal_labels, literal_masks
