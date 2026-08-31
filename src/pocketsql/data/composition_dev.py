"""Build evaluation-only composition and counterfactual gates for v9."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sqlite3

from .composition import held_out_composition, held_out_plans_for
from .generate import dataset_quality_report
from .populate import populate
from .query_ast import Filter, QueryPlan
from .render_sql import render_sql
from .schemas import Schema, make_schema
from .validate import validate_sql
from .verbalize import compositional_verbalize


def counterfactual_groups_for(schema: Schema) -> list[tuple[str, tuple[QueryPlan, QueryPlan]]]:
    """Return minimal pairs where exactly one requested decision changes."""
    parent, child = schema.tables
    parent_id = schema.role("parent_id")[1].name
    child_id = schema.role("child_id")[1].name
    child_fk = schema.role("parent_fk")[1].name
    name = schema.role("name")[1].name
    location = schema.role("location")[1].name
    amount = schema.role("amount")[1].name
    status = schema.role("status")[1].name
    join_on = (f"{child.name}.{child_fk}", f"{parent.name}.{parent_id}")
    return [
        (
            "projection_column",
            (
                QueryPlan("counterfactual_projection_column", parent.name, (name,), filters=(Filter(location, "=", "Seattle"),)),
                QueryPlan("counterfactual_projection_column", parent.name, (location,), filters=(Filter(location, "=", "Seattle"),)),
            ),
        ),
        (
            "literal_value",
            (
                QueryPlan("counterfactual_literal_value", parent.name, (name,), filters=(Filter(location, "=", "Boston"),)),
                QueryPlan("counterfactual_literal_value", parent.name, (name,), filters=(Filter(location, "=", "Houston"),)),
            ),
        ),
        (
            "comparison_operator",
            (
                QueryPlan("counterfactual_comparison_operator", child.name, (child_id,), filters=(Filter(amount, ">", 150),)),
                QueryPlan("counterfactual_comparison_operator", child.name, (child_id,), filters=(Filter(amount, "<", 150),)),
            ),
        ),
        (
            "aggregate_operation",
            (
                QueryPlan("counterfactual_aggregate_operation", child.name, (), aggregate="COUNT", filters=(Filter(status, "=", "shipped"),)),
                QueryPlan("counterfactual_aggregate_operation", child.name, (), aggregate="SUM", aggregate_column=amount, filters=(Filter(status, "=", "shipped"),)),
            ),
        ),
        (
            "joined_literal_value",
            (
                QueryPlan(
                    "counterfactual_joined_literal_value",
                    child.name,
                    (f"{child.name}.*",),
                    filters=(Filter(f"{parent.name}.{location}", "=", "Boston"),),
                    join_table=parent.name,
                    join_on=join_on,
                ),
                QueryPlan(
                    "counterfactual_joined_literal_value",
                    child.name,
                    (f"{child.name}.*",),
                    filters=(Filter(f"{parent.name}.{location}", "=", "Houston"),),
                    join_table=parent.name,
                    join_on=join_on,
                ),
            ),
        ),
    ]


def _record(
    schema: Schema,
    database_sql: str,
    plan: QueryPlan,
    question: str,
    record_id: str,
    seed: int,
    **extra: object,
) -> dict:
    return {
        "id": record_id,
        "schema_id": schema.schema_id,
        "schema_sql": schema.sql(),
        "database_sql": database_sql,
        "question": question,
        "sql": render_sql(plan),
        "query_plan": plan.normalized(),
        "difficulty": 1 + int(plan.aggregate is not None) + int(plan.join_table is not None) + int(len(plan.filters) > 1),
        "seed": seed,
        **extra,
    }


def _composition_records(schemas: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for schema_index in range(schemas):
        schema = make_schema(schema_index, rng, schema_prefix="composition_dev")
        connection = sqlite3.connect(":memory:")
        populate(connection, schema, rng)
        database_sql = "\n".join(connection.iterdump())
        for plan_index, plan in enumerate(held_out_plans_for(schema)):
            sql = render_sql(plan)
            assert validate_sql(connection, sql)[0]
            # Keep language in-distribution so this gate isolates operation
            # composition. Held-out language is measured by casual_dev.
            question = compositional_verbalize(plan, rng)
            records.append(
                _record(
                    schema,
                    database_sql,
                    plan,
                    question,
                    f"{schema.schema_id}_{plan_index:03d}",
                    seed,
                    held_out_composition=held_out_composition(plan),
                )
            )
        connection.close()
    return records


def _counterfactual_records(schemas: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for schema_index in range(schemas):
        schema = make_schema(schema_index, rng, schema_prefix="counterfactual_dev")
        connection = sqlite3.connect(":memory:")
        populate(connection, schema, rng)
        database_sql = "\n".join(connection.iterdump())
        for group_index, (change, plans) in enumerate(counterfactual_groups_for(schema)):
            results = []
            for variant, plan in enumerate(plans):
                sql = render_sql(plan)
                assert validate_sql(connection, sql)[0]
                results.append(connection.execute(sql).fetchall())
                records.append(
                    _record(
                        schema,
                        database_sql,
                        plan,
                        compositional_verbalize(plan, rng),
                        f"{schema.schema_id}_{group_index:02d}_{variant}",
                        seed,
                        counterfactual_group=f"{schema.schema_id}_{group_index:02d}",
                        counterfactual_change=change,
                        counterfactual_variant=variant,
                    )
                )
            if results[0] == results[1]:
                raise RuntimeError(f"Counterfactual pair {change} is not discriminative for {schema.schema_id}")
        connection.close()
    return records


def write_v9_dev_datasets(
    output: Path,
    composition_schemas: int = 120,
    counterfactual_schemas: int = 120,
    seed: int = 9091,
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    composition = _composition_records(composition_schemas, seed)
    counterfactual = _counterfactual_records(counterfactual_schemas, seed + 1)
    for name, records in (("composition", composition), ("counterfactual", counterfactual)):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    report = dataset_quality_report({"composition": composition, "counterfactual": counterfactual})
    report.update(
        {
            "profile": "v9_composition_and_counterfactual_dev",
            "training_use_allowed": False,
            "language_distribution": "v9_training_style",
            "isolation_goal": "operation_composition_and_counterfactual_sensitivity",
            "held_out_compositions": dict(sorted(Counter(record["held_out_composition"] for record in composition).items())),
            "counterfactual_changes": dict(sorted(Counter(record["counterfactual_change"] for record in counterfactual).items())),
        }
    )
    (output / "quality_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"composition": len(composition), "counterfactual": len(counterfactual)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--composition-schemas", type=int, default=120)
    parser.add_argument("--counterfactual-schemas", type=int, default=120)
    parser.add_argument("--seed", type=int, default=9091)
    args = parser.parse_args()
    print(write_v9_dev_datasets(args.output, args.composition_schemas, args.counterfactual_schemas, args.seed))


if __name__ == "__main__":
    main()
