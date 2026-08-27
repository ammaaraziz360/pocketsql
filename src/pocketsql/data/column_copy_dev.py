"""Build an evaluation-only benchmark that requests every column in each schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sqlite3

from .populate import populate
from .query_ast import QueryPlan
from .schemas import make_schema
from .verbalize import humanize_identifier


def _question(table: str, column: str, variant: int) -> str:
    table_one = humanize_identifier(table, "singular")
    table_many = humanize_identifier(table, "plural")
    column_one = humanize_identifier(column, "singular")
    column_many = humanize_identifier(column, "plural")
    return (
        f"show me {table_one} {column_many}",
        f"what are the {column_many} of our {table_many}",
        f"list each {table_one}'s {column_one}",
    )[variant % 3]


def write_column_copy_dev_dataset(
    output: Path,
    schemas: int = 120,
    seed: int = 161803,
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[dict] = []
    position_counts: dict[str, int] = {}
    for schema_index in range(schemas):
        schema = make_schema(schema_index, rng, schema_prefix="column_copy")
        connection = sqlite3.connect(":memory:")
        populate(connection, schema, rng)
        database_sql = "\n".join(connection.iterdump())
        for table_index, table in enumerate(schema.tables):
            for column_index, column in enumerate(table.columns):
                sql = f"SELECT {column.name} FROM {table.name};"
                plan = QueryPlan("column_copy", table.name, (column.name,))
                position_key = f"table_{table_index}:position_{column_index}"
                position_counts[position_key] = position_counts.get(position_key, 0) + 1
                records.append(
                    {
                        "id": f"{schema.schema_id}_{table_index}_{column_index}",
                        "schema_id": schema.schema_id,
                        "schema_sql": schema.sql(),
                        "database_sql": database_sql,
                        "question": _question(table.name, column.name, schema_index + table_index + column_index),
                        "sql": sql,
                        "query_plan": plan.normalized(),
                        "difficulty": 1,
                        "seed": seed,
                        "column_role": column.role,
                        "column_position": column_index,
                    }
                )
        connection.close()

    path = output / "column_copy.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    report = {
        "profile": "all_column_copy_dev",
        "training_use_allowed": False,
        "records": len(records),
        "schemas": schemas,
        "roles": sorted({record["column_role"] for record in records}),
        "physical_position_counts": dict(sorted(position_counts.items())),
    }
    (output / "quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"column_copy": len(records), "schemas": schemas}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schemas", type=int, default=120)
    parser.add_argument("--seed", type=int, default=161803)
    args = parser.parse_args()
    print(write_column_copy_dev_dataset(args.output, args.schemas, args.seed))


if __name__ == "__main__":
    main()
