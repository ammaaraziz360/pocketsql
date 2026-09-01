"""Build training/dev data for schema linking from indirect human language.

The training split pairs an identifier-explicit question with a lexical-dropout
paraphrase for the same schema and query plan.  It uses only familiar PocketSQL
operation compositions; the anti-memorization gate remains evaluation-only and
uses a separate schema vocabulary.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import sqlite3

from .composition import held_out_composition
from .generate import schema_identifiers
from .populate import populate
from .query_ast import Filter, QueryPlan
from .render_sql import render_sql
from .schemas import CITY_VALUES, STATUS_VALUES, Schema, make_schema
from .validate import validate_sql
from .verbalize import humanize_identifier


_ALIASES = {
    "retail": {
        "parent_one": "person buying from us",
        "parent_many": "people buying from us",
        "child_one": "thing bought",
        "child_many": "things bought",
        "name": "person label",
        "location": "where they live",
        "amount": "money paid",
        "status": "buying progress",
        "child_id": "purchase reference number",
    },
    "library": {
        "parent_one": "person borrowing a book",
        "parent_many": "people borrowing books",
        "child_one": "borrowed item",
        "child_many": "borrowed items",
        "name": "person label",
        "location": "where they live",
        "amount": "money owed",
        "status": "return progress",
        "child_id": "borrowing reference number",
    },
    "school": {
        "parent_one": "person taking classes",
        "parent_many": "people taking classes",
        "child_one": "class participation",
        "child_many": "class participations",
        "name": "person label",
        "location": "where they live",
        "amount": "performance number",
        "status": "study progress",
        "child_id": "class reference number",
    },
    "restaurant": {
        "parent_one": "person eating here",
        "parent_many": "people eating here",
        "child_one": "dining occasion",
        "child_many": "dining occasions",
        "name": "person label",
        "location": "where they live",
        "amount": "money paid",
        "status": "meal progress",
        "child_id": "dining reference number",
    },
    "events": {
        "parent_one": "person attending",
        "parent_many": "people attending",
        "child_one": "access item",
        "child_many": "access items",
        "name": "person label",
        "location": "where they live",
        "amount": "money paid",
        "status": "entry progress",
        "child_id": "access reference number",
    },
}


def _labels(schema: Schema) -> dict[str, str]:
    parent, child = schema.tables
    return {
        "parent_one": humanize_identifier(parent.name, "singular"),
        "parent_many": humanize_identifier(parent.name, "plural"),
        "child_one": humanize_identifier(child.name, "singular"),
        "child_many": humanize_identifier(child.name, "plural"),
        "name": humanize_identifier(schema.role("name")[1].name),
        "location": humanize_identifier(schema.role("location")[1].name),
        "amount": humanize_identifier(schema.role("amount")[1].name),
        "status": humanize_identifier(schema.role("status")[1].name),
        "child_id": humanize_identifier(schema.role("child_id")[1].name),
    }


def _familiar_cases(schema: Schema, variant: int) -> list[tuple[str, QueryPlan, str, str]]:
    parent, child = schema.tables
    parent_id = schema.role("parent_id")[1].name
    child_id = schema.role("child_id")[1].name
    child_fk = schema.role("parent_fk")[1].name
    name = schema.role("name")[1].name
    location = schema.role("location")[1].name
    amount = schema.role("amount")[1].name
    status_column = schema.role("status")[1].name
    city = CITY_VALUES[variant % len(CITY_VALUES)]
    status_value = STATUS_VALUES[variant % len(STATUS_VALUES)]
    threshold = (50, 80, 100, 125)[variant % 4]
    direct = _labels(schema)
    semantic = _ALIASES[schema.domain]
    join_on = (f"{child.name}.{child_fk}", f"{parent.name}.{parent_id}")
    parent_location = f"{parent.name}.{location}"

    cases = [
        (
            "semantic_project_name",
            QueryPlan("semantic_project_name", parent.name, (name,)),
            f"List the {direct['name']} from all {direct['parent_many']}",
            f"Give me the {semantic['name']} for every {semantic['parent_one']}",
        ),
        (
            "semantic_count_location",
            QueryPlan(
                "semantic_count_location",
                parent.name,
                (),
                aggregate="COUNT",
                filters=(Filter(location, "=", city),),
            ),
            f"How many {direct['parent_many']} have {direct['location']} equal to {city}",
            f"How many {semantic['parent_many']} are based around {city}",
        ),
        (
            "semantic_join_location",
            QueryPlan(
                "semantic_join_location",
                child.name,
                (f"{child.name}.*",),
                filters=(Filter(parent_location, "=", city),),
                join_table=parent.name,
                join_on=join_on,
            ),
            f"Show all {direct['child_many']} whose {direct['parent_one']} has {direct['location']} equal to {city}",
            f"Show every {semantic['child_one']} connected to a {semantic['parent_one']} based around {city}",
        ),
        (
            "semantic_sum_status",
            QueryPlan(
                "semantic_sum_status",
                child.name,
                (),
                aggregate="SUM",
                aggregate_column=amount,
                filters=(Filter(status_column, "=", status_value),),
            ),
            f"Add up {direct['amount']} for {direct['child_many']} whose {direct['status']} is {status_value}",
            f"How much {semantic['amount']} do the {semantic['child_many']} marked {status_value} have combined",
        ),
        (
            "semantic_child_and_filter",
            QueryPlan(
                "semantic_child_and_filter",
                child.name,
                (child_id,),
                filters=(Filter(status_column, "=", status_value), Filter(amount, ">", threshold)),
            ),
            f"List {direct['child_id']} where {direct['status']} is {status_value} and {direct['amount']} is above {threshold}",
            f"Show the {semantic['child_id']} for {semantic['child_many']} marked {status_value} costing more than {threshold}",
        ),
        (
            "semantic_join_count_location",
            QueryPlan(
                "semantic_join_count_location",
                child.name,
                (),
                aggregate="COUNT",
                filters=(Filter(parent_location, "=", city),),
                join_table=parent.name,
                join_on=join_on,
            ),
            f"Count {direct['child_many']} where {direct['parent_one']} {direct['location']} is {city}",
            f"How many {semantic['child_many']} belong to {semantic['parent_many']} based around {city}",
        ),
        (
            "semantic_join_sum_location",
            QueryPlan(
                "semantic_join_sum_location",
                child.name,
                (),
                aggregate="SUM",
                aggregate_column=f"{child.name}.{amount}",
                filters=(Filter(parent_location, "=", city),),
                join_table=parent.name,
                join_on=join_on,
            ),
            f"Total {direct['amount']} for {direct['child_many']} where {direct['parent_one']} {direct['location']} is {city}",
            f"What is the combined {semantic['amount']} for {semantic['child_many']} connected to {semantic['parent_many']} based around {city}",
        ),
        (
            "semantic_group_status",
            QueryPlan(
                "semantic_group_status",
                child.name,
                (status_column,),
                aggregate="COUNT",
                group_by=(status_column,),
            ),
            f"Count {direct['child_many']} for each {direct['status']}",
            f"Break down the {semantic['child_one']} count by {semantic['status']}",
        ),
        (
            "semantic_distinct_location",
            QueryPlan(
                "semantic_distinct_location",
                parent.name,
                (location,),
                distinct=True,
            ),
            f"List the distinct {direct['location']} from {direct['parent_many']}",
            f"Which different places are represented among {semantic['parent_many']}",
        ),
    ]
    if any(held_out_composition(plan) for _, plan, _, _ in cases):
        raise RuntimeError("semantic-link training accidentally contains a held-out composition")
    return cases


def _unique_schemas(count: int, rng: random.Random) -> list[Schema]:
    schemas: list[Schema] = []
    seen: set[str] = set()
    candidate = 0
    while len(schemas) < count:
        schema = make_schema(candidate, rng, schema_prefix="semantic_link")
        candidate += 1
        if schema.sql() in seen:
            continue
        seen.add(schema.sql())
        schemas.append(replace(schema, schema_id=f"semantic_link_{len(schemas):04d}"))
    return schemas


def _records_for(schemas: list[Schema], seed: int, rng: random.Random) -> list[dict]:
    records: list[dict] = []
    for schema_index, schema in enumerate(schemas):
        connection = sqlite3.connect(":memory:")
        populate(connection, schema, rng)
        database_sql = "\n".join(connection.iterdump())
        for case_index, (intent, plan, direct, paraphrase) in enumerate(
            _familiar_cases(schema, schema_index)
        ):
            sql = render_sql(plan)
            valid, reason = validate_sql(connection, sql)
            if not valid:
                raise RuntimeError(f"Invalid semantic-link SQL for {schema.schema_id}/{intent}: {reason}")
            pair = f"{schema.schema_id}_{case_index:02d}"
            for track, question in (("direct_identifier", direct), ("semantic_paraphrase", paraphrase)):
                records.append(
                    {
                        "id": f"{pair}_{track}",
                        "schema_id": schema.schema_id,
                        "schema_sql": schema.sql(),
                        "database_sql": database_sql,
                        "question": question,
                        "sql": sql,
                        "query_plan": plan.normalized(),
                        "difficulty": 1
                        + int(plan.aggregate is not None)
                        + int(plan.join_table is not None)
                        + int(len(plan.filters) > 1),
                        "seed": seed,
                        "semantic_linking_pair": pair,
                        "semantic_linking_track": track,
                        "intent": intent,
                    }
                )
        connection.close()
    return records


def _write(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_semantic_linking_dataset(
    output: Path,
    train_schemas: int = 400,
    validation_schemas: int = 40,
    seed: int = 515151,
    replay_data: Path | None = None,
    replay_records: int = 7200,
    heldout_data: Path | None = None,
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    schemas = _unique_schemas(train_schemas + validation_schemas, rng)
    train = _records_for(schemas[:train_schemas], seed, rng)
    paired_validation = _records_for(schemas[train_schemas:], seed, rng)
    validation = [
        record for record in paired_validation if record["semantic_linking_track"] == "semantic_paraphrase"
    ]
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(paired_validation)
    _write(output / "train.jsonl", train)
    _write(output / "validation.jsonl", validation)
    _write(output / "paired_validation.jsonl", paired_validation)

    replay: list[dict] = []
    if replay_data is not None:
        replay_source = [json.loads(line) for line in replay_data.read_text(encoding="utf-8").splitlines() if line]
        if replay_records > len(replay_source):
            raise ValueError(f"replay_records={replay_records} exceeds {len(replay_source)} available records")
        replay = rng.sample(replay_source, replay_records)
    mixed_train = [*train, *replay]
    rng.shuffle(mixed_train)
    _write(output / "mixed_train.jsonl", mixed_train)

    train_schema_sql = {record["schema_sql"] for record in train}
    validation_schema_sql = {record["schema_sql"] for record in validation}
    train_identifiers = set().union(*(schema_identifiers(record["schema_sql"]) for record in train))
    heldout_identifiers: set[str] = set()
    if heldout_data is not None:
        heldout = [json.loads(line) for line in heldout_data.read_text(encoding="utf-8").splitlines() if line]
        heldout_identifiers = set().union(*(schema_identifiers(record["schema_sql"]) for record in heldout))
    report = {
        "profile": "semantic_schema_linking_v15",
        "seed": seed,
        "training_use_allowed": {"train": True, "mixed_train": True, "validation": False, "paired_validation": False},
        "records": {
            "train": len(train),
            "replay": len(replay),
            "mixed_train": len(mixed_train),
            "validation": len(validation),
            "paired_validation": len(paired_validation),
        },
        "schemas": {"train": train_schemas, "validation": validation_schemas},
        "schema_overlap_train_validation": len(train_schema_sql & validation_schema_sql),
        "track_counts": dict(sorted(Counter(record["semantic_linking_track"] for record in train).items())),
        "intent_counts": dict(sorted(Counter(record["intent"] for record in train).items())),
        "held_out_compositions_in_training": sum(
            held_out_composition(QueryPlan(**record["query_plan"])) is not None for record in train
        ),
        "replay_data": str(replay_data) if replay_data else None,
        "heldout_data": str(heldout_data) if heldout_data else None,
        "identifier_overlap_with_heldout": len(train_identifiers & heldout_identifiers) if heldout_data else None,
    }
    (output / "quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "train": len(train),
        "mixed_train": len(mixed_train),
        "validation": len(validation),
        "paired_validation": len(paired_validation),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-schemas", type=int, default=400)
    parser.add_argument("--validation-schemas", type=int, default=40)
    parser.add_argument("--seed", type=int, default=515151)
    parser.add_argument("--replay-data", type=Path)
    parser.add_argument("--replay-records", type=int, default=7200)
    parser.add_argument("--heldout-data", type=Path)
    args = parser.parse_args()
    print(
        write_semantic_linking_dataset(
            args.output,
            args.train_schemas,
            args.validation_schemas,
            args.seed,
            args.replay_data,
            args.replay_records,
            args.heldout_data,
        )
    )


if __name__ == "__main__":
    main()
