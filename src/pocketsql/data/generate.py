from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sqlite3

from .populate import populate
from .query_ast import Filter, QueryPlan
from .render_sql import render_sql
from .schemas import STATUS_VALUES, Schema, make_schema
from .validate import validate_sql
from .verbalize import verbalize


def _with_threshold(filters: tuple[Filter, ...], amount_column: str, threshold: int) -> tuple[Filter, ...]:
    """Replace any existing filter on amount_column with a new threshold instead of stacking redundant conditions."""
    updated = tuple(Filter(amount_column, item.operator, threshold) if item.column == amount_column else item for item in filters)
    return updated if any(item.column == amount_column for item in filters) else updated + (Filter(amount_column, ">", threshold),)


def plans_for(schema: Schema, count: int = 13) -> list[QueryPlan]:
    parent, child = schema.tables
    parent_id = parent.columns[0].name
    name_column, location_column = parent.columns[1].name, parent.columns[2].name
    child_fk, amount, status = (column.name for column in child.columns[1:])
    shipped, open_status = STATUS_VALUES[0], STATUS_VALUES[1]
    join_column_choices = ((name_column, amount), (location_column, amount), (name_column, status))
    base = [
        QueryPlan("select", parent.name, (name_column, location_column)),
        QueryPlan("distinct", parent.name, (location_column,), distinct=True),
        QueryPlan("filter", child.name, (child.columns[0].name, amount), filters=(Filter(status, "=", shipped),)),
        QueryPlan("and_filter", child.name, (child.columns[0].name,), filters=(Filter(amount, ">", 20), Filter(status, "=", shipped))),
        QueryPlan("or_filter", child.name, (child.columns[0].name,), filters=(Filter(status, "=", shipped), Filter(status, "=", open_status)), filter_connector="OR"),
        QueryPlan("count", child.name, (), aggregate="COUNT"),
        QueryPlan("sum", child.name, (), aggregate="SUM", aggregate_column=amount),
        QueryPlan("avg", child.name, (), aggregate="AVG", aggregate_column=amount),
        QueryPlan("min", child.name, (), aggregate="MIN", aggregate_column=amount),
        QueryPlan("max", child.name, (), aggregate="MAX", aggregate_column=amount),
        QueryPlan("group", child.name, (status,), aggregate="COUNT", group_by=(status,)),
        QueryPlan("order_limit", child.name, (child.columns[0].name, amount), order_by=amount, descending=True, limit=3),
        QueryPlan("join", parent.name, (f"{parent.name}.{name_column}", f"{child.name}.{amount}"), join_table=child.name, join_on=(f"{parent.name}.{parent_id}", f"{child.name}.{child_fk}")),
    ]
    plans: list[QueryPlan] = []
    thresholds = (10, 20, 40, 75, 100, 150, 200, 250)
    for variant in range((count + len(base) - 1) // len(base)):
        for plan in base:
            if len(plans) == count:
                return plans
            if variant == 0:
                plans.append(plan)
                continue
            threshold = thresholds[(variant - 1) % len(thresholds)]
            if plan.family == "join":
                left_column, right_column = join_column_choices[(variant - 1) % len(join_column_choices)]
                plans.append(replace(plan, columns=(f"{parent.name}.{left_column}", f"{child.name}.{right_column}"), limit=variant + 1))
            elif plan.family == "filter":
                cycled_status = STATUS_VALUES[variant % len(STATUS_VALUES)]
                plans.append(replace(plan, filters=(Filter(status, "=", cycled_status),)))
            elif plan.family == "or_filter":
                first, second = STATUS_VALUES[variant % len(STATUS_VALUES)], STATUS_VALUES[(variant + 1) % len(STATUS_VALUES)]
                plans.append(replace(plan, filters=(Filter(status, "=", first), Filter(status, "=", second))))
            elif plan.family == "and_filter":
                plans.append(replace(plan, filters=_with_threshold(plan.filters, amount, threshold)))
            elif plan.family == "order_limit":
                plans.append(replace(plan, limit=(variant % 5) + 2))
            elif plan.table == child.name:
                plans.append(replace(plan, filters=(Filter(amount, ">", threshold),)))
            else:
                plans.append(replace(plan, limit=variant + 1))
    return plans


def build_records(schemas: int, examples_per_schema: int, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    records: list[dict] = []
    for schema_index in range(schemas):
        schema = make_schema(schema_index, rng)
        connection = sqlite3.connect(":memory:")
        populate(connection, schema, rng)
        seen_questions: set[str] = set()
        for plan_index, plan in enumerate(plans_for(schema, examples_per_schema)):
            sql = render_sql(plan)
            valid, _ = validate_sql(connection, sql)
            question = verbalize(plan, rng)
            if valid and question not in seen_questions:
                seen_questions.add(question)
                records.append({"id": f"{schema.schema_id}_{plan_index:03d}", "schema_id": schema.schema_id, "schema_sql": schema.sql(), "database_sql": "\n".join(connection.iterdump()), "question": question, "sql": sql, "query_plan": plan.normalized(), "difficulty": 1 + int(plan.aggregate is not None) + int(plan.join_table is not None), "seed": seed})
        connection.close()
    schema_ids = [f"schema_{index:04d}" for index in range(schemas)]
    rng.shuffle(schema_ids)
    cut_train = max(1, int(schemas * .8))
    cut_val = max(cut_train + 1, int(schemas * .9))
    if schemas >= 3:
        cut_train = min(cut_train, schemas - 2)
        cut_val = min(max(cut_val, cut_train + 1), schemas - 1)
    split_ids = {"train": set(schema_ids[:cut_train]), "validation": set(schema_ids[cut_train:cut_val]), "test": set(schema_ids[cut_val:])}
    return {name: [record for record in records if record["schema_id"] in ids] for name, ids in split_ids.items()}


def write_dataset(output: Path, schemas: int, examples_per_schema: int, seed: int) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    splits = build_records(schemas, examples_per_schema, seed)
    for name, records in splits.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {name: len(records) for name, records in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schemas", type=int, default=100)
    parser.add_argument("--examples-per-schema", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(write_dataset(args.output, args.schemas, args.examples_per_schema, args.seed))


if __name__ == "__main__":
    main()