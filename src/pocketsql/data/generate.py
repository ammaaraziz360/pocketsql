from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import random
import re
import sqlite3
from typing import Mapping

from .populate import populate
from .query_ast import Filter, QueryPlan
from .render_sql import render_sql
from .schemas import DOMAIN_VOCAB, STATUS_VALUES, Schema, make_opaque_schema, make_schema
from .validate import validate_sql
from .verbalize import verbalize


def _with_threshold(filters: tuple[Filter, ...], amount_column: str, threshold: int) -> tuple[Filter, ...]:
    """Replace any existing filter on amount_column with a new threshold instead of stacking redundant conditions."""
    updated = tuple(Filter(amount_column, item.operator, threshold) if item.column == amount_column else item for item in filters)
    return updated if any(item.column == amount_column for item in filters) else updated + (Filter(amount_column, ">", threshold),)


def base_plans_for(schema: Schema) -> list[QueryPlan]:
    parent, child = schema.tables
    parent_id = schema.role("parent_id")[1].name
    name_column = schema.role("name")[1].name
    location_column = schema.role("location")[1].name
    child_id = schema.role("child_id")[1].name
    child_fk = schema.role("parent_fk")[1].name
    amount = schema.role("amount")[1].name
    status = schema.role("status")[1].name
    shipped, open_status = STATUS_VALUES[0], STATUS_VALUES[1]
    join_column_choices = ((name_column, amount), (location_column, amount), (name_column, status))
    base = [
        QueryPlan("select", parent.name, (parent_id,)),
        QueryPlan("distinct", parent.name, (location_column,), distinct=True),
        QueryPlan("filter", child.name, (child_id, amount), filters=(Filter(status, "=", shipped),)),
        QueryPlan("and_filter", child.name, (child_id,), filters=(Filter(amount, ">", 20), Filter(status, "=", shipped))),
        QueryPlan("or_filter", child.name, (child_id,), filters=(Filter(status, "=", shipped), Filter(status, "=", open_status)), filter_connector="OR"),
        QueryPlan("count", child.name, (), aggregate="COUNT"),
        QueryPlan("sum", child.name, (), aggregate="SUM", aggregate_column=amount),
        QueryPlan("avg", child.name, (), aggregate="AVG", aggregate_column=amount),
        QueryPlan("min", child.name, (), aggregate="MIN", aggregate_column=amount),
        QueryPlan("max", child.name, (), aggregate="MAX", aggregate_column=amount),
        QueryPlan("group", child.name, (status,), aggregate="COUNT", group_by=(status,)),
        QueryPlan("order_limit", child.name, (child_id, amount), order_by=amount, descending=True, limit=3),
        QueryPlan("join", parent.name, (f"{parent.name}.{name_column}", f"{child.name}.{amount}"), join_table=child.name, join_on=(f"{parent.name}.{parent_id}", f"{child.name}.{child_fk}")),
    ]
    return base


def plan_variant(plan: QueryPlan, schema: Schema, variant: int) -> QueryPlan:
    """Create a distinct variant of a query family for one schema."""
    if plan.family == "select":
        targets = [(table.name, column.name) for table in schema.tables for column in table.columns]
        if variant < len(targets):
            table, column = targets[variant]
            return replace(plan, table=table, columns=(column,), limit=None)
        # Once every physical column has appeared as a direct copy target, use
        # the remaining projection budget for multi-column requests.  This
        # preserves complete copy coverage without teaching that SELECT always
        # contains exactly one column.
        extra = variant - len(targets)
        selected_table = schema.tables[extra % len(schema.tables)]
        columns = [column.name for column in selected_table.columns]
        start = (extra // len(schema.tables)) % len(columns)
        width = 2 + ((extra // (len(schema.tables) * len(columns))) % min(2, len(columns) - 1))
        selected = tuple(columns[(start + offset) % len(columns)] for offset in range(width))
        return replace(plan, table=selected_table.name, columns=selected, limit=None)
    if variant == 0:
        return plan
    parent, child = schema.tables
    parent_id = schema.role("parent_id")[1].name
    name_column = schema.role("name")[1].name
    location_column = schema.role("location")[1].name
    child_fk = schema.role("parent_fk")[1].name
    amount = schema.role("amount")[1].name
    status = schema.role("status")[1].name
    join_column_choices = ((name_column, amount), (location_column, amount), (name_column, status))
    # The base AND-filter plan already uses 20.  Keep it out of the variant
    # schedule so later examples cannot recreate the base query.
    thresholds = (10, 30, 40, 50, 60, 75, 90, 100, 120, 140, 160, 180, 200, 225, 250)
    threshold = thresholds[(variant - 1) % len(thresholds)]
    if plan.family == "join":
        left_column, right_column = join_column_choices[(variant - 1) % len(join_column_choices)]
        join_plan = replace(plan, columns=(f"{parent.name}.{left_column}", f"{child.name}.{right_column}"), limit=(variant % 5) + 2, join_on=(f"{parent.name}.{parent_id}", f"{child.name}.{child_fk}"))
        if variant % 3 == 1:
            return replace(join_plan, filters=(Filter(status, "=", STATUS_VALUES[variant % len(STATUS_VALUES)]),))
        if variant % 3 == 2:
            return replace(join_plan, filters=(Filter(amount, ">", threshold),), order_by=f"{child.name}.{amount}", descending=True)
        return join_plan
    if plan.family == "filter":
        cycled_status = STATUS_VALUES[variant % len(STATUS_VALUES)]
        return replace(plan, filters=(Filter(status, "=", cycled_status),))
    if plan.family == "or_filter":
        # Six pairs, four triples, and the full set provide eleven distinct OR
        # predicates before any family repeats.  This avoids identical natural
        # questions carrying different SQL labels in larger balanced corpora.
        status_groups = tuple(combinations(STATUS_VALUES, 2)) + tuple(combinations(STATUS_VALUES, 3)) + (STATUS_VALUES,)
        return replace(plan, filters=tuple(Filter(status, "=", value) for value in status_groups[variant % len(status_groups)]))
    if plan.family == "and_filter":
        return replace(plan, filters=_with_threshold(plan.filters, amount, threshold))
    if plan.family == "order_limit":
        # Variant zero is the base LIMIT 3 query, so start subsequent variants
        # at 4 rather than immediately recreating it.
        return replace(plan, limit=(variant % 5) + 3)
    if plan.family == "distinct":
        columns = (name_column,) if variant % 2 else (location_column,)
        return replace(plan, columns=columns, limit=None if variant == 1 else variant + 1)
    if plan.table == child.name:
        return replace(plan, filters=(Filter(amount, ">", threshold),))
    return replace(plan, limit=variant + 1)


def plans_for(schema: Schema, count: int = 13, family_weights: Mapping[str, int] | None = None) -> list[QueryPlan]:
    base = base_plans_for(schema)
    allowed_families = {plan.family for plan in base}
    weights = {family: 1 for family in allowed_families}
    if family_weights:
        unknown = set(family_weights) - allowed_families
        if unknown:
            raise ValueError(f"Unknown query families in family_weights: {sorted(unknown)}")
        if any(not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0 for weight in family_weights.values()):
            raise ValueError("Every query family weight must be a positive integer")
        weights.update(family_weights)
    plans_by_family = {plan.family: plan for plan in base}
    family_order = [plan.family for plan in base]
    total_weight = sum(weights.values())
    current_weights = {family: 0 for family in family_order}
    variants: dict[str, int] = defaultdict(int)
    plans: list[QueryPlan] = []
    while len(plans) < count:
        for family in family_order:
            current_weights[family] += weights[family]
        family = max(family_order, key=lambda item: current_weights[item])
        current_weights[family] -= total_weight
        variant = variants[family]
        plans.append(plan_variant(plans_by_family[family], schema, variant))
        variants[family] += 1
    return plans


def build_records(
    schemas: int,
    examples_per_schema: int,
    seed: int,
    family_weights: Mapping[str, int] | None = None,
    vocabularies: Mapping | None = None,
    schema_prefix: str = "schema",
    generation_stats: dict | None = None,
    identifier_mode: str = "natural",
    question_style: str = "classic",
) -> dict[str, list[dict]]:
    if identifier_mode not in {"natural", "opaque"}:
        raise ValueError("identifier_mode must be 'natural' or 'opaque'")
    rng = random.Random(seed)
    records: list[dict] = []
    schema_ids: list[str] = []
    stats = {"planned_examples": schemas * examples_per_schema, "retained_examples": 0, "discarded_invalid_sql": 0, "discarded_duplicate_questions": 0}
    for schema_index in range(schemas):
        schema = (
            make_opaque_schema(schema_index, rng, schema_prefix)
            if identifier_mode == "opaque"
            else make_schema(schema_index, rng, dict(vocabularies or DOMAIN_VOCAB), schema_prefix)
        )
        schema_ids.append(schema.schema_id)
        connection = sqlite3.connect(":memory:")
        populate(connection, schema, rng)
        seen_questions: set[str] = set()
        for plan_index, plan in enumerate(plans_for(schema, examples_per_schema, family_weights)):
            sql = render_sql(plan)
            valid, _ = validate_sql(connection, sql)
            question = verbalize(plan, rng, question_style)
            if not valid:
                stats["discarded_invalid_sql"] += 1
            elif question in seen_questions:
                stats["discarded_duplicate_questions"] += 1
            else:
                seen_questions.add(question)
                stats["retained_examples"] += 1
                records.append({"id": f"{schema.schema_id}_{plan_index:03d}", "schema_id": schema.schema_id, "schema_sql": schema.sql(), "database_sql": "\n".join(connection.iterdump()), "question": question, "sql": sql, "query_plan": plan.normalized(), "difficulty": 1 + int(plan.aggregate is not None) + int(plan.join_table is not None) + int(len(plan.filters) > 1), "seed": seed})
        connection.close()
    rng.shuffle(schema_ids)
    cut_train = max(1, int(schemas * .8))
    cut_val = max(cut_train + 1, int(schemas * .9))
    if schemas >= 3:
        cut_train = min(cut_train, schemas - 2)
        cut_val = min(max(cut_val, cut_train + 1), schemas - 1)
    split_ids = {"train": set(schema_ids[:cut_train]), "validation": set(schema_ids[cut_train:cut_val]), "test": set(schema_ids[cut_val:])}
    if generation_stats is not None:
        generation_stats.update(stats)
    return {name: [record for record in records if record["schema_id"] in ids] for name, ids in split_ids.items()}


def schema_identifiers(schema_sql: str) -> set[str]:
    reserved = {"CREATE", "TABLE", "INTEGER", "PRIMARY", "KEY", "TEXT", "REAL", "REFERENCES"}
    return {token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", schema_sql) if token.upper() not in reserved}


def dataset_quality_report(
    splits: Mapping[str, list[dict]], reference_identifiers: set[str] | None = None
) -> dict:
    """Summarize split hygiene and identifier novelty.

    By default, validation and test identifier novelty is measured relative to the
    training split.  A separate reference set can be provided for a challenge
    corpus, where there intentionally is no training split in the output.
    """
    train_identifiers = set().union(*(schema_identifiers(record["schema_sql"]) for record in splits.get("train", [])))
    comparison_identifiers = reference_identifiers if reference_identifiers is not None else train_identifiers
    report = {"splits": {}, "family_counts": {}, "identifier_reference": "provided" if reference_identifiers is not None else "train"}
    all_records = [record for records in splits.values() for record in records]
    family_counts = defaultdict(int)
    schema_sets = {split_name: {record["schema_id"] for record in records} for split_name, records in splits.items()}
    overlaps = {
        f"{left}:{right}": len(schema_sets[left] & schema_sets[right])
        for index, left in enumerate(schema_sets)
        for right in list(schema_sets)[index + 1 :]
    }
    for split_name, records in splits.items():
        schemas = {record["schema_id"] for record in records}
        duplicates = len(records) - len({(record["schema_id"], record["question"]) for record in records})
        identifiers = set().union(*(schema_identifiers(record["schema_sql"]) for record in records)) if records else set()
        report["splits"][split_name] = {
            "records": len(records),
            "schemas": len(schemas),
            "duplicate_schema_question_pairs": duplicates,
            "unseen_identifier_rate_vs_reference": (len(identifiers - comparison_identifiers) / len(identifiers)) if identifiers else 0.0,
        }
        for record in records:
            family_counts[record["query_plan"]["family"]] += 1
    report["family_counts"] = dict(sorted(family_counts.items()))
    report["schema_split_overlap"] = overlaps
    report["schema_disjoint"] = not any(overlaps.values())
    report["total_records"] = len(all_records)
    return report


def write_dataset(
    output: Path,
    schemas: int,
    examples_per_schema: int,
    seed: int,
    family_weights: Mapping[str, int] | None = None,
    vocabularies: Mapping | None = None,
    schema_prefix: str = "schema",
    identifier_mode: str = "natural",
    question_style: str = "classic",
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    generation_stats: dict = {}
    splits = build_records(
        schemas,
        examples_per_schema,
        seed,
        family_weights,
        vocabularies,
        schema_prefix,
        generation_stats,
        identifier_mode,
        question_style,
    )
    for name, records in splits.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    report = dataset_quality_report(splits)
    report["generation"] = generation_stats
    (output / "quality_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {name: len(records) for name, records in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schemas", type=int, default=100)
    parser.add_argument("--examples-per-schema", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--family-weights", help='JSON object of positive per-family weights, e.g. \'{"join": 4, "and_filter": 3, "or_filter": 3}\'.')
    parser.add_argument("--identifier-mode", choices=("natural", "opaque"), default="natural")
    parser.add_argument("--question-style", choices=("classic", "casual", "mixed", "heldout"), default="classic")
    args = parser.parse_args()
    family_weights = json.loads(args.family_weights) if args.family_weights else None
    if family_weights is not None and not isinstance(family_weights, dict):
        raise SystemExit("--family-weights must be a JSON object mapping family names to positive integers.")
    print(
        write_dataset(
            args.output,
            args.schemas,
            args.examples_per_schema,
            args.seed,
            family_weights,
            identifier_mode=args.identifier_mode,
            question_style=args.question_style,
        )
    )


if __name__ == "__main__":
    main()
