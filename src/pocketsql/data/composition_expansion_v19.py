"""Build V19's execution-checked, schema-disjoint composition curriculum.

The V18 filter curriculum taught individual schema pointers but rarely taught
the operation, schema, and literal heads to succeed together.  This generator
expands every V17 two-table schema into a fixed cross-product of projections,
comparisons, connectors, aggregates, grouping, ordering, limits, and joins.
Validation and fresh-gate schemas never appear in training, and every emitted
SQL statement must execute and return at least one row.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import tempfile

from pocketsql.data.composition import composition_signature, composition_tier
from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.data.render_sql import render_sql
from pocketsql.data.validate import validate_sql
from pocketsql.data.verbalize import humanize_identifier
from pocketsql.model.schema_grounding import canonicalize_inputs, canonicalize_record


TRAIN_TEMPLATE_VARIANTS = (0, 1, 2)
VALIDATION_TEMPLATE_VARIANT = 3
FRESH_GATE_TEMPLATE_VARIANT = 4
REQUIRED_INTENTS = {
    "v17_project_name",
    "v17_project_id_location",
    "v17_child_two_filter",
    "v17_join_amount_name",
}


@dataclass(frozen=True)
class SchemaContext:
    schema_id: str
    schema_sql: str
    database_sql: str
    parent_table: str
    child_table: str
    parent_id: str
    child_id: str
    name: str
    location: str
    amount: str
    status: str
    join_on: tuple[str, str]


@dataclass(frozen=True)
class CompositionCase:
    intent: str
    plan: QueryPlan
    contrast_group: str | None = None
    contrast_axis: str | None = None


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _grouped_sources(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in _load(path):
        if record.get("schema_id") and record.get("query_plan"):
            grouped[record["schema_id"]].append(record)
    if not grouped:
        raise ValueError(f"no structured source records found in {path}")
    return dict(grouped)


def _schema_context(schema_id: str, records: list[dict]) -> SchemaContext:
    by_intent = {}
    for record in records:
        intent = record.get("intent")
        if intent in REQUIRED_INTENTS and intent not in by_intent:
            by_intent[intent] = record
    missing = REQUIRED_INTENTS - set(by_intent)
    if missing:
        raise ValueError(f"{schema_id} is missing required V17 intents: {sorted(missing)}")

    parent_name = by_intent["v17_project_name"]["query_plan"]
    parent_pair = by_intent["v17_project_id_location"]["query_plan"]
    child_pair = by_intent["v17_child_two_filter"]["query_plan"]
    joined = by_intent["v17_join_amount_name"]["query_plan"]
    database_sql = next(
        (record.get("database_sql") for record in records if record.get("database_sql")),
        None,
    )
    if not database_sql:
        raise ValueError(f"{schema_id} has no database_sql for execution validation")
    if not joined.get("join_table") or len(joined.get("join_on") or ()) != 2:
        raise ValueError(f"{schema_id} has no usable declared join")

    return SchemaContext(
        schema_id=schema_id,
        schema_sql=records[0]["schema_sql"],
        database_sql=database_sql,
        parent_table=parent_name["table"],
        child_table=child_pair["table"],
        parent_id=parent_pair["columns"][0],
        child_id=child_pair["columns"][0],
        name=parent_name["columns"][0],
        location=parent_pair["columns"][1],
        amount=child_pair["columns"][1],
        status=child_pair["filters"][0]["column"],
        join_on=tuple(joined["join_on"]),
    )


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _reference(table: str, column: str) -> str:
    return f"{table}.{column}"


def _query_reference(table: str, column: str) -> str:
    return f"{_quote(table)}.{_quote(column)}"


def _database_dump(connection: sqlite3.Connection) -> str:
    return "\n".join(connection.iterdump())


def _prepare_database(connection: sqlite3.Connection, context: SchemaContext) -> dict[str, object]:
    """Plant a multi-row join/filter cell so aggregate choices are observable."""
    endpoints = []
    for reference in context.join_on:
        table, separator, column = reference.rpartition(".")
        if not separator:
            raise ValueError(f"unqualified join endpoint in {context.schema_id}: {reference}")
        endpoints.append((table, column))
    child_endpoint = next(
        (column for table, column in endpoints if table == context.child_table), None
    )
    parent_endpoint = next(
        (column for table, column in endpoints if table == context.parent_table), None
    )
    if not child_endpoint or not parent_endpoint:
        raise ValueError(f"join endpoints do not match schema tables in {context.schema_id}")

    location_marker = "Harbor"
    status_marker = "ready"
    existing_location = connection.execute(
        f"SELECT COUNT(*) FROM {_quote(context.parent_table)} "
        f"WHERE {_quote(context.location)} = ?",
        (location_marker,),
    ).fetchone()[0]
    existing_status = connection.execute(
        f"SELECT COUNT(*) FROM {_quote(context.child_table)} "
        f"WHERE {_quote(context.status)} = ?",
        (status_marker,),
    ).fetchone()[0]
    if existing_location or existing_status:
        raise ValueError(f"V19 marker collision in {context.schema_id}")

    candidates = connection.execute(
        f"SELECT {_quote(context.child_id)}, {_quote(child_endpoint)} "
        f"FROM {_quote(context.child_table)} "
        f"WHERE {_quote(child_endpoint)} IS NOT NULL "
        f"AND {_quote(context.amount)} IS NOT NULL "
        f"ORDER BY {_quote(context.amount)}, {_quote(context.child_id)}"
    ).fetchall()
    if len(candidates) < 4:
        raise ValueError(f"{context.schema_id} needs four child rows for planted composition cells")
    anchor_indices = (0, len(candidates) // 3, (2 * len(candidates)) // 3, len(candidates) - 1)
    anchors = [candidates[index] for index in anchor_indices]
    child_ids = [row[0] for row in anchors]
    parent_ids = list(dict.fromkeys(row[1] for row in anchors))
    connection.executemany(
        f"UPDATE {_quote(context.parent_table)} SET {_quote(context.location)} = ? "
        f"WHERE {_quote(parent_endpoint)} = ?",
        [(location_marker, value) for value in parent_ids],
    )
    connection.executemany(
        f"UPDATE {_quote(context.child_table)} SET {_quote(context.status)} = ? "
        f"WHERE {_quote(context.child_id)} = ?",
        [(status_marker, value) for value in child_ids],
    )
    connection.commit()
    name = connection.execute(
        f"SELECT {_query_reference(context.parent_table, context.name)} "
        f"FROM {_quote(context.child_table)} INNER JOIN {_quote(context.parent_table)} "
        f"ON {context.join_on[0]} = {context.join_on[1]} "
        f"WHERE {_query_reference(context.child_table, context.child_id)} = ?",
        (child_ids[0],),
    ).fetchone()[0]
    return {
        "parent_name": name,
        "parent_location": location_marker,
        "child_status": status_marker,
        "join_name": name,
        "join_location": location_marker,
        "join_status": status_marker,
    }


def _database_values(
    connection: sqlite3.Connection,
    context: SchemaContext,
    planted: dict[str, object],
) -> dict[str, object]:
    parent_query = (
        f"SELECT {_query_reference(context.parent_table, context.name)}, "
        f"{_query_reference(context.parent_table, context.location)} "
        f"FROM {_quote(context.parent_table)} "
        f"WHERE {_query_reference(context.parent_table, context.name)} IS NOT NULL "
        f"AND {_query_reference(context.parent_table, context.location)} IS NOT NULL"
    )
    parent_rows = connection.execute(parent_query).fetchall()
    if not parent_rows:
        raise ValueError(f"{context.schema_id} has no complete parent rows")

    joined_query = (
        f"SELECT {_query_reference(context.parent_table, context.name)}, "
        f"{_query_reference(context.parent_table, context.location)}, "
        f"{_query_reference(context.child_table, context.status)}, "
        f"{_query_reference(context.child_table, context.amount)}, "
        f"{_query_reference(context.child_table, context.child_id)} "
        f"FROM {_quote(context.child_table)} INNER JOIN {_quote(context.parent_table)} "
        f"ON {context.join_on[0]} = {context.join_on[1]} "
        f"WHERE {_query_reference(context.parent_table, context.name)} IS NOT NULL "
        f"AND {_query_reference(context.parent_table, context.location)} IS NOT NULL "
        f"AND {_query_reference(context.child_table, context.status)} IS NOT NULL "
        f"AND {_query_reference(context.child_table, context.amount)} IS NOT NULL "
        f"ORDER BY {_query_reference(context.child_table, context.child_id)}"
    )
    joined_rows = connection.execute(joined_query).fetchall()
    if len(joined_rows) < 3:
        raise ValueError(f"{context.schema_id} needs at least three complete joined rows")

    amounts = sorted({float(row[3]) for row in joined_rows})
    if len(amounts) < 3:
        raise ValueError(f"{context.schema_id} needs at least three distinct numeric values")
    threshold = amounts[len(amounts) // 2]

    def clean_number(value: float) -> int | float:
        return int(value) if value.is_integer() else value

    return {
        "threshold": clean_number(threshold),
        "limit": 3,
        **planted,
    }


def _case(
    intent: str,
    plan: QueryPlan,
    group: str | None = None,
    axis: str | None = None,
) -> CompositionCase:
    return CompositionCase(
        intent,
        replace(plan, family=composition_signature(plan)),
        group,
        axis,
    )


def _composition_cases(context: SchemaContext, values: dict[str, object]) -> list[CompositionCase]:
    p, c = context.parent_table, context.child_table
    pid, cid = context.parent_id, context.child_id
    name, location = context.name, context.location
    amount, status = context.amount, context.status
    p_name, p_location = _reference(p, name), _reference(p, location)
    c_id, c_amount, c_status = (
        _reference(c, cid),
        _reference(c, amount),
        _reference(c, status),
    )
    parent_name = values["parent_name"]
    parent_location = values["parent_location"]
    child_status = values["child_status"]
    threshold = values["threshold"]
    join_name = values["join_name"]
    join_location = values["join_location"]
    join_status = values["join_status"]
    limit = int(values["limit"])
    join = {"join_table": p, "join_on": context.join_on}

    cases = [
        _case("parent_project_name", QueryPlan("", p, (name,))),
        _case("parent_project_id_location", QueryPlan("", p, (pid, location))),
        _case("parent_location_by_name", QueryPlan("", p, (location,), filters=(Filter(name, "=", parent_name),))),
        _case("parent_name_by_location", QueryPlan("", p, (name,), filters=(Filter(location, "=", parent_location),))),
        _case("parent_count_location", QueryPlan("", p, (), aggregate="COUNT", filters=(Filter(location, "=", parent_location),))),
        _case(
            "parent_name_location_and",
            QueryPlan("", p, (pid, name), filters=(Filter(name, "=", parent_name), Filter(location, "=", parent_location))),
            "parent_connector",
            "filter_connector",
        ),
        _case(
            "parent_name_location_or",
            QueryPlan("", p, (pid, name), filters=(Filter(name, "=", parent_name), Filter(location, "=", parent_location)), filter_connector="OR"),
            "parent_connector",
            "filter_connector",
        ),
        _case("parent_distinct_location", QueryPlan("", p, (location,), distinct=True)),
        _case("parent_order_name_ascending", QueryPlan("", p, (name,), order_by=name)),
        _case("parent_order_name_descending_limit", QueryPlan("", p, (name,), order_by=name, descending=True, limit=limit)),
        _case("parent_id_name_by_location", QueryPlan("", p, (pid, name), filters=(Filter(location, "=", parent_location),))),
        _case("parent_id_location_by_name", QueryPlan("", p, (pid, location), filters=(Filter(name, "=", parent_name),))),

        _case("child_project_id_amount", QueryPlan("", c, (cid, amount))),
        _case("child_id_amount_by_status", QueryPlan("", c, (cid, amount), filters=(Filter(status, "=", child_status),))),
        _case("child_amount_greater", QueryPlan("", c, (cid,), filters=(Filter(amount, ">", threshold),)), "child_operator", "filter_operator"),
        _case("child_amount_less", QueryPlan("", c, (cid,), filters=(Filter(amount, "<", threshold),)), "child_operator", "filter_operator"),
        _case("child_amount_at_least", QueryPlan("", c, (cid,), filters=(Filter(amount, ">=", threshold),)), "child_operator", "filter_operator"),
        _case("child_amount_at_most", QueryPlan("", c, (cid,), filters=(Filter(amount, "<=", threshold),)), "child_operator", "filter_operator"),
        _case(
            "child_status_amount_and",
            QueryPlan("", c, (cid, amount), filters=(Filter(status, "=", child_status), Filter(amount, ">=", threshold))),
            "child_connector",
            "filter_connector",
        ),
        _case(
            "child_status_amount_or",
            QueryPlan("", c, (cid, amount), filters=(Filter(status, "=", child_status), Filter(amount, ">=", threshold)), filter_connector="OR"),
            "child_connector",
            "filter_connector",
        ),
    ]

    for aggregate in ("COUNT", "SUM", "AVG", "MIN", "MAX"):
        cases.append(
            _case(
                f"child_{aggregate.casefold()}_by_status",
                QueryPlan(
                    "",
                    c,
                    (),
                    aggregate=aggregate,
                    aggregate_column=None if aggregate == "COUNT" else amount,
                    filters=(Filter(status, "=", child_status),),
                ),
                "child_aggregate",
                "aggregate",
            )
        )
    cases.extend(
        [
            _case("child_group_count_status", QueryPlan("", c, (status,), aggregate="COUNT", group_by=(status,), aggregate_position=1)),
            _case("child_group_sum_status", QueryPlan("", c, (status,), aggregate="SUM", aggregate_column=amount, group_by=(status,), aggregate_position=1)),
            _case("child_group_count_above", QueryPlan("", c, (status,), aggregate="COUNT", filters=(Filter(amount, ">", threshold),), group_by=(status,), aggregate_position=1)),
            _case("child_order_amount_descending_limit", QueryPlan("", c, (cid, amount), order_by=amount, descending=True, limit=limit)),
            _case("child_distinct_status", QueryPlan("", c, (status,), distinct=True)),
        ]
    )

    cases.extend(
        [
            _case("join_amount_by_parent_name", QueryPlan("", c, (c_amount,), filters=(Filter(p_name, "=", join_name),), **join)),
            _case("join_projection_amount_by_status", QueryPlan("", c, (c_amount,), filters=(Filter(c_status, "=", join_status),), **join), "join_projection", "projection"),
            _case("join_projection_name_amount_by_status", QueryPlan("", c, (p_name, c_amount), filters=(Filter(c_status, "=", join_status),), **join), "join_projection", "projection"),
            _case("join_projection_id_by_status", QueryPlan("", c, (c_id,), filters=(Filter(c_status, "=", join_status),), **join), "join_projection", "projection"),
            _case("join_id_amount_by_location", QueryPlan("", c, (c_id, c_amount), filters=(Filter(p_location, "=", join_location),), **join)),
            _case("join_count_location", QueryPlan("", c, (), aggregate="COUNT", filters=(Filter(p_location, "=", join_location),), **join)),
            _case(
                "join_count_location_status_and",
                QueryPlan("", c, (), aggregate="COUNT", filters=(Filter(p_location, "=", join_location), Filter(c_status, "=", join_status)), **join),
                "join_connector",
                "filter_connector",
            ),
            _case(
                "join_count_location_status_or",
                QueryPlan("", c, (), aggregate="COUNT", filters=(Filter(p_location, "=", join_location), Filter(c_status, "=", join_status)), filter_connector="OR", **join),
                "join_connector",
                "filter_connector",
            ),
        ]
    )
    for aggregate in ("SUM", "AVG", "MIN", "MAX"):
        cases.append(
            _case(
                f"join_{aggregate.casefold()}_location_status",
                QueryPlan(
                    "",
                    c,
                    (),
                    aggregate=aggregate,
                    aggregate_column=c_amount,
                    filters=(Filter(p_location, "=", join_location), Filter(c_status, "=", join_status)),
                    **join,
                ),
                "join_aggregate",
                "aggregate",
            )
        )
    cases.extend(
        [
            _case("join_group_count_location", QueryPlan("", c, (p_location,), aggregate="COUNT", group_by=(p_location,), aggregate_position=1, **join)),
            _case("join_group_sum_status_location", QueryPlan("", c, (c_status,), aggregate="SUM", aggregate_column=c_amount, filters=(Filter(p_location, "=", join_location),), group_by=(c_status,), aggregate_position=1, **join)),
            _case("join_order_amount_location_limit", QueryPlan("", c, (p_name, c_amount), filters=(Filter(p_location, "=", join_location),), order_by=c_amount, descending=True, limit=limit, **join)),
            _case("join_distinct_parent_name_status", QueryPlan("", c, (p_name,), distinct=True, filters=(Filter(c_status, "=", join_status),), **join)),
            _case("join_sum_parent_name_location", QueryPlan("", c, (), aggregate="SUM", aggregate_column=c_amount, filters=(Filter(p_name, "=", join_name), Filter(p_location, "=", join_location)), **join)),
            _case("join_max_parent_name", QueryPlan("", c, (), aggregate="MAX", aggregate_column=c_amount, filters=(Filter(p_name, "=", join_name),), **join)),
        ]
    )
    return cases


def _column_label(reference: str, context: SchemaContext, joined: bool) -> str:
    qualifier, separator, column = reference.rpartition(".")
    label = humanize_identifier(column if separator else reference)
    table_labels = {
        humanize_identifier(context.parent_table, "singular").casefold(),
        humanize_identifier(context.parent_table, "plural").casefold(),
        humanize_identifier(context.child_table, "singular").casefold(),
        humanize_identifier(context.child_table, "plural").casefold(),
    }
    if label.casefold() in table_labels:
        # Suffix stripping can collapse ``donation_value`` to ``donation``,
        # which is ambiguous with the donations table.  The raw identifier is
        # still natural enough for direct-link supervision and is unambiguous.
        label = column if separator else reference
    if not joined or not separator:
        return label
    if qualifier == context.parent_table:
        return f"{humanize_identifier(context.parent_table, 'singular')} {label}"
    if qualifier == context.child_table:
        return f"{humanize_identifier(context.child_table, 'singular')} {label}"
    return label


def _subject(plan: QueryPlan, context: SchemaContext) -> str:
    parent_many = humanize_identifier(context.parent_table, "plural")
    child_many = humanize_identifier(context.child_table, "plural")
    if plan.join_table:
        return f"{child_many} and their related {parent_many}"
    return parent_many if plan.table == context.parent_table else child_many


def _metric(plan: QueryPlan, context: SchemaContext) -> str:
    subject = _subject(plan, context)
    if plan.aggregate == "COUNT":
        return f"the number of {subject.split(' and their related ')[0]}"
    label = _column_label(plan.aggregate_column or "", context, bool(plan.join_table))
    words = {"SUM": "total", "AVG": "average", "MIN": "minimum", "MAX": "maximum"}
    return f"the {words[plan.aggregate]} {label}"


def _request(plan: QueryPlan, context: SchemaContext) -> str:
    joined = bool(plan.join_table)
    if plan.group_by:
        groups = " and ".join(_column_label(item, context, joined) for item in plan.group_by)
        return f"each {groups} together with {_metric(plan, context)}"
    if plan.aggregate:
        return _metric(plan, context)
    columns = [_column_label(item, context, joined) for item in plan.columns]
    if len(columns) == 1:
        selected = columns[0]
    else:
        selected = ", ".join(columns[:-1]) + f" and {columns[-1]}"
    return f"the {'distinct ' if plan.distinct else ''}{selected}"


def _filter_phrase(item: Filter, context: SchemaContext, joined: bool) -> str:
    operator = {
        "=": "equals",
        ">": "is greater than",
        "<": "is less than",
        ">=": "is at least",
        "<=": "is at most",
    }[item.operator]
    return f"{_column_label(item.column, context, joined)} {operator} {item.value}"


def _question(plan: QueryPlan, context: SchemaContext, template_variant: int) -> str:
    request = _request(plan, context)
    subject = _subject(plan, context)
    conditions = ""
    if plan.filters:
        connector = f" {plan.filter_connector.casefold()} "
        conditions = " where " + connector.join(
            _filter_phrase(item, context, bool(plan.join_table)) for item in plan.filters
        )
    modifiers = ""
    if plan.order_by:
        direction = "highest to lowest" if plan.descending else "lowest to highest"
        modifiers += f", sorted by {_column_label(plan.order_by, context, bool(plan.join_table))} from {direction}"
    if plan.limit is not None:
        modifiers += f", returning at most {plan.limit} rows"
    templates = (
        f"show me {request} from {subject}{conditions}{modifiers}",
        f"for {subject}{conditions}, give me {request}{modifiers}",
        f"I need {request} for {subject}{conditions}{modifiers}",
        f"pull {request} from {subject}{conditions}{modifiers}",
        f"can you return {request} for {subject}{conditions}{modifiers}, please",
    )
    return templates[template_variant]


def _base_column(reference: str) -> str:
    return reference.rpartition(".")[2] if "." in reference else reference


def _explicit_roles(record: dict) -> tuple[int, int]:
    _, question, mapping = canonicalize_inputs(
        record["schema_sql"], record["question"], "permuted", True, False
    )
    plan = record["query_plan"]
    references = list(plan.get("columns", ()))
    references.extend(item["column"] for item in plan.get("filters", ()))
    references.extend(plan.get("group_by", ()))
    for key in ("aggregate_column", "order_by"):
        if plan.get(key):
            references.append(plan[key])
    slots = {
        mapping.column_to_slot[column]
        for column in map(_base_column, references)
        if column != "*" and column in mapping.column_to_slot
    }
    explicit = sum(
        bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(slot)}(?![A-Za-z0-9_])", question))
        for slot in slots
    )
    return explicit, len(slots)


def _records_for_split(
    grouped: dict[str, list[dict]],
    split: str,
    seed: int,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    result_groups: dict[str, list[tuple[tuple, ...]]] = defaultdict(list)
    explicit = 0
    explicit_total = 0
    cases_per_schema: set[int] = set()
    for schema_index, schema_id in enumerate(sorted(grouped)):
        context = _schema_context(schema_id, grouped[schema_id])
        connection = sqlite3.connect(":memory:")
        connection.executescript(context.database_sql)
        planted = _prepare_database(connection, context)
        values = _database_values(connection, context, planted)
        database_sql = _database_dump(connection)
        cases = _composition_cases(context, values)
        cases_per_schema.add(len(cases))
        for case_index, case in enumerate(cases):
            if split == "train":
                template_variant = TRAIN_TEMPLATE_VARIANTS[
                    (schema_index + case_index) % len(TRAIN_TEMPLATE_VARIANTS)
                ]
                training_use_allowed = True
            elif split == "validation":
                template_variant = VALIDATION_TEMPLATE_VARIANT
                training_use_allowed = False
            elif split == "fresh_gate":
                template_variant = FRESH_GATE_TEMPLATE_VARIANT
                training_use_allowed = False
            else:
                raise ValueError(f"unknown split: {split}")
            sql = render_sql(case.plan)
            valid, rows = validate_sql(connection, sql)
            if not valid:
                raise ValueError(f"{schema_id}/{case.intent} did not execute with rows: {sql}")
            record = {
                "id": f"v19:{split}:{schema_id}:{case_index:02d}",
                "schema_id": f"v19:{schema_id}",
                "schema_sql": context.schema_sql,
                "question": _question(case.plan, context, template_variant),
                "sql": sql,
                "query_plan": case.plan.normalized(),
                "intent": case.intent,
                "composition_signature": composition_signature(case.plan),
                "composition_tier": composition_tier(case.plan),
                "difficulty": 1
                + int(case.plan.join_table is not None)
                + int(bool(case.plan.filters))
                + int(len(case.plan.filters) > 1)
                + int(bool(case.plan.group_by))
                + int(case.plan.order_by is not None),
                "seed": seed,
                "source": {
                    "dataset": "PocketSQL V19 compositional expansion",
                    "human_authored": False,
                    "training_use_allowed": training_use_allowed,
                },
            }
            if case.contrast_group:
                group = f"{schema_id}:{case.contrast_group}"
                record["counterfactual_group"] = group
                record["counterfactual_axis"] = case.contrast_axis
                result_groups[group].append(tuple(tuple(row) for row in rows))
            if not training_use_allowed:
                record["database_sql"] = database_sql
                record["evaluation_track"] = f"v19_{split}"
            # Exercise the exact canonicalization path used by structured
            # training and fail during generation instead of at epoch start.
            canonicalize_record(record, "permuted", True)
            found, total = _explicit_roles(record)
            explicit += found
            explicit_total += total
            records.append(record)
        connection.close()

    if len(cases_per_schema) != 1:
        raise ValueError(f"inconsistent cases per schema: {sorted(cases_per_schema)}")
    multi_groups = {key: values for key, values in result_groups.items() if len(values) > 1}
    distinct_groups = sum(len(set(values)) == len(values) for values in multi_groups.values())
    if distinct_groups != len(multi_groups):
        raise ValueError(
            f"{split} has {len(multi_groups) - distinct_groups} counterfactual groups "
            "whose SQL results are not fully distinguishable"
        )
    report = {
        "records": len(records),
        "schemas": len(grouped),
        "cases_per_schema": next(iter(cases_per_schema), 0),
        "execution_checked": len(records),
        "counterfactual_groups": len(multi_groups),
        "counterfactual_groups_with_distinct_results": distinct_groups,
        "explicit_schema_roles": {
            "explicit": explicit,
            "total": explicit_total,
            "rate": explicit / max(explicit_total, 1),
        },
    }
    return records, report


def _write(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_composition_expansion_dataset(
    output: Path,
    train_source: Path,
    validation_source: Path,
    fresh_gate_source: Path,
    seed: int = 191919,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V19 composition dataset: {output}")
    grouped = {
        "train": _grouped_sources(train_source),
        "validation": _grouped_sources(validation_source),
        "fresh_gate": _grouped_sources(fresh_gate_source),
    }
    schema_sets = {split: set(records) for split, records in grouped.items()}
    if schema_sets["train"] & schema_sets["validation"]:
        raise ValueError("train and validation source schemas overlap")
    if schema_sets["train"] & schema_sets["fresh_gate"]:
        raise ValueError("train and fresh-gate source schemas overlap")
    if schema_sets["validation"] & schema_sets["fresh_gate"]:
        raise ValueError("validation and fresh-gate source schemas overlap")

    split_records = {}
    split_reports = {}
    for split in ("train", "validation", "fresh_gate"):
        split_records[split], split_reports[split] = _records_for_split(
            grouped[split], split, seed
        )
    rng = random.Random(seed)
    rng.shuffle(split_records["train"])
    split_records["validation"].sort(key=lambda record: record["id"])
    split_records["fresh_gate"].sort(key=lambda record: record["id"])

    question_sets = {
        split: {record["question"].strip().casefold() for record in records}
        for split, records in split_records.items()
    }
    all_records = [record for records in split_records.values() for record in records]
    report = {
        "profile": "composition_expansion_v19",
        "seed": seed,
        "training_use_allowed": {
            "train": True,
            "validation": False,
            "fresh_gate": False,
        },
        "splits": split_reports,
        "isolation": {
            "train_validation_schema_overlap": 0,
            "train_fresh_gate_schema_overlap": 0,
            "validation_fresh_gate_schema_overlap": 0,
            "train_validation_exact_question_overlap": len(
                question_sets["train"] & question_sets["validation"]
            ),
            "train_fresh_gate_exact_question_overlap": len(
                question_sets["train"] & question_sets["fresh_gate"]
            ),
        },
        "composition_tiers": dict(
            sorted(Counter(record["composition_tier"] for record in all_records).items())
        ),
        "joins": sum(bool(record["query_plan"]["join_table"]) for record in all_records),
        "multi_filter": sum(len(record["query_plan"]["filters"]) > 1 for record in all_records),
        "aggregates": dict(
            sorted(
                Counter(record["query_plan"]["aggregate"] or "NONE" for record in all_records).items()
            )
        ),
        "filter_operators": dict(
            sorted(
                Counter(
                    item["operator"]
                    for record in all_records
                    for item in record["query_plan"]["filters"]
                ).items()
            )
        ),
        "filter_connectors": dict(
            sorted(
                Counter(
                    record["query_plan"]["filter_connector"]
                    for record in all_records
                    if len(record["query_plan"]["filters"]) > 1
                ).items()
            )
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as name:
        temporary = Path(name)
        for split, records in split_records.items():
            _write(temporary / f"{split}.jsonl", records)
        report["sha256"] = {
            split: _sha256(temporary / f"{split}.jsonl") for split in split_records
        }
        (temporary / "quality_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--train-source",
        type=Path,
        default=Path("data/semantic-expansion-v17/train.jsonl"),
    )
    parser.add_argument(
        "--validation-source",
        type=Path,
        default=Path("data/semantic-expansion-v17/paired_validation.jsonl"),
    )
    parser.add_argument(
        "--fresh-gate-source",
        type=Path,
        default=Path("data/semantic-expansion-v17/fresh_gate.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=191919)
    args = parser.parse_args()
    print(
        json.dumps(
            build_composition_expansion_dataset(
                args.output,
                args.train_source,
                args.validation_source,
                args.fresh_gate_source,
                args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
