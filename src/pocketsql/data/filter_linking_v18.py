"""Build schema-disjoint contrastive supervision for V18 filter-column links.

Every counterfactual group keeps the schema, projection, operator, and literal
fixed.  Only the filter column and its explicit phrase in the question change.
The accompanying SQLite rows make each target column return a different result,
so evaluation cannot reward a wrong filter through an empty-result coincidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import tempfile

from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.data.render_sql import render_sql
from pocketsql.data.validate import validate_sql
from pocketsql.data.verbalize import humanize_identifier
from pocketsql.model.schema_grounding import canonicalize_inputs


TEXT_MARKERS = ("Amber", "Cobalt", "Ivory", "Marigold", "Teal")
NUMERIC_MARKERS = (777, 913, 1049, 1223, 1427)
TRAIN_TEMPLATES = (0, 1, 2)
VALIDATION_TEMPLATES = (3, 4)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _source_schemas(path: Path) -> list[dict]:
    selected = {}
    for record in _load(path):
        if record.get("database_sql") and record["schema_id"] not in selected:
            selected[record["schema_id"]] = record
    if not selected:
        raise ValueError(f"no records with database_sql found in {path}")
    return [selected[key] for key in sorted(selected)]


def _affinity(declared_type: str) -> str | None:
    upper = declared_type.upper()
    if any(token in upper for token in ("CHAR", "CLOB", "TEXT")):
        return "text"
    if any(token in upper for token in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return "numeric"
    return None


def _table_columns(connection: sqlite3.Connection, table: str) -> tuple[str | None, dict[str, list[str]]]:
    rows = connection.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
    primary = next((row[1] for row in rows if row[5]), None)
    groups: dict[str, list[str]] = defaultdict(list)
    for _, name, declared_type, _, _, is_primary in rows:
        affinity = _affinity(declared_type or "")
        if affinity and not is_primary:
            groups[affinity].append(name)
    return primary, {name: values for name, values in groups.items() if len(values) >= 2}


def _marker_absent(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    marker: str | int,
) -> bool:
    return all(
        connection.execute(
            f"SELECT COUNT(*) FROM {_quote(table)} WHERE {_quote(column)} = ?",
            (marker,),
        ).fetchone()[0]
        == 0
        for column in columns
    )


def _choose_marker(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    affinity: str,
) -> str | int:
    candidates = TEXT_MARKERS if affinity == "text" else NUMERIC_MARKERS
    for marker in candidates:
        if _marker_absent(connection, table, columns, marker):
            return marker
    raise ValueError(f"could not find a novel {affinity} marker for {table}")


def _question(
    template: int,
    table: str,
    projection: str,
    target: str,
    value: str | int,
    target_phrase: str | None = None,
) -> str:
    table_label = humanize_identifier(table)
    projection_label = humanize_identifier(projection)
    target_label = target_phrase or humanize_identifier(target)
    templates = (
        f"show {projection_label} from {table_label} where {target_label} is {value}",
        f"which {table_label} rows have {target_label} set to {value}; give me {projection_label}",
        f"find the {projection_label} for records whose {target_label} equals {value}",
        f"I need {projection_label} from {table_label} with {value} in {target_label}",
        f"using {target_label} as the filter, return {projection_label} when it matches {value}",
    )
    return templates[template]


def _database_dump(connection: sqlite3.Connection) -> str:
    return "\n".join(connection.iterdump())


def _records_for_schema(
    source: dict,
    templates: tuple[int, ...],
    training_use_allowed: bool,
    seed: int,
) -> list[dict]:
    connection = sqlite3.connect(":memory:")
    connection.executescript(source["database_sql"])
    schema_sql = source["schema_sql"]
    schema_id = source["schema_id"]
    table_names = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    prepared = []
    for table in table_names:
        primary, groups = _table_columns(connection, table)
        if primary is None:
            continue
        primary_values = [
            row[0]
            for row in connection.execute(
                f"SELECT {_quote(primary)} FROM {_quote(table)} ORDER BY {_quote(primary)}"
            ).fetchall()
        ]
        for affinity, columns in sorted(groups.items()):
            if len(primary_values) < len(columns):
                continue
            marker = _choose_marker(connection, table, columns, affinity)
            for column, primary_value in zip(columns, primary_values):
                connection.execute(
                    f"UPDATE {_quote(table)} SET {_quote(column)} = ? WHERE {_quote(primary)} = ?",
                    (marker, primary_value),
                )
            prepared.append((table, primary, affinity, tuple(columns), marker))
    connection.commit()
    database_sql = _database_dump(connection)

    records = []
    for group_index, (table, primary, affinity, columns, marker) in enumerate(prepared):
        result_sets = {}
        for target in columns:
            plan = QueryPlan(
                family="v18_filter_column_contrast",
                table=table,
                columns=(primary,),
                filters=(Filter(target, "=", marker),),
            )
            sql = render_sql(plan)
            valid, rows = validate_sql(connection, sql)
            if not valid:
                raise ValueError(f"contrastive SQL returned no rows: {sql}")
            result_sets[target] = tuple(rows)
        if len(set(result_sets.values())) != len(columns):
            raise ValueError(f"contrastive filters do not produce distinct results for {schema_id}/{table}")

        for template in templates:
            pair_id = f"{schema_id}:{group_index}:{template}"
            for target in columns:
                plan = QueryPlan(
                    family="v18_filter_column_contrast",
                    table=table,
                    columns=(primary,),
                    filters=(Filter(target, "=", marker),),
                )
                question = _question(template, table, primary, target, marker)
                _, grounded_question, mapping = canonicalize_inputs(
                    schema_sql, question, "permuted", True, False
                )
                target_slot = mapping.column_to_slot[target]
                if not re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(target_slot)}(?![A-Za-z0-9_])",
                    grounded_question,
                ):
                    # Generic suffixes such as ``_value`` can be removed by
                    # humanization and collide with a table alias. Fall back
                    # to the exact identifier for an unambiguous direct-link
                    # example instead of silently labeling an implicit one.
                    question = _question(
                        template, table, primary, target, marker, target
                    )
                    _, grounded_question, mapping = canonicalize_inputs(
                        schema_sql, question, "permuted", True, False
                    )
                    target_slot = mapping.column_to_slot[target]
                    if not re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(target_slot)}(?![A-Za-z0-9_])",
                        grounded_question,
                    ):
                        raise ValueError(
                            f"target {target!r} is not explicit after grounding: {grounded_question!r}"
                        )
                record = {
                    "id": f"v18_filter:{pair_id}:{target}",
                    "schema_id": f"v18_filter:{schema_id}",
                    "schema_sql": schema_sql,
                    "question": question,
                    "sql": render_sql(plan),
                    "query_plan": plan.normalized(),
                    "difficulty": 1,
                    "seed": seed,
                    "counterfactual_group": pair_id,
                    "counterfactual_change": "filter_column",
                    "filter_link_target": target,
                    "filter_link_distractors": [column for column in columns if column != target],
                    "filter_link_affinity": affinity,
                    "filter_link_explicit": True,
                    "source": {
                        "dataset": "PocketSQL V18 contrastive filter linking",
                        "human_authored": False,
                        "training_use_allowed": training_use_allowed,
                    },
                }
                if not training_use_allowed:
                    record["database_sql"] = database_sql
                    record["evaluation_track"] = "filter_linking_validation"
                records.append(record)
    connection.close()
    return records


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


def build_filter_linking_dataset(
    output: Path,
    train_source: Path,
    validation_source: Path,
    seed: int = 181818,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite filter-linking dataset: {output}")
    train_sources = _source_schemas(train_source)
    validation_sources = _source_schemas(validation_source)
    train_schema_ids = {record["schema_id"] for record in train_sources}
    validation_schema_ids = {record["schema_id"] for record in validation_sources}
    overlap = train_schema_ids & validation_schema_ids
    if overlap:
        raise ValueError(f"train/validation source schemas overlap: {sorted(overlap)[:3]}")

    train = [
        item
        for source in train_sources
        for item in _records_for_schema(source, TRAIN_TEMPLATES, True, seed)
    ]
    validation = [
        item
        for source in validation_sources
        for item in _records_for_schema(source, VALIDATION_TEMPLATES, False, seed)
    ]
    rng = random.Random(seed)
    rng.shuffle(train)
    validation.sort(key=lambda record: record["id"])

    train_questions = {record["question"].casefold() for record in train}
    validation_questions = {record["question"].casefold() for record in validation}
    group_sizes = Counter(record["counterfactual_group"] for record in validation)
    report = {
        "profile": "v18_filter_column_contrast",
        "seed": seed,
        "training_use_allowed": {"train": True, "validation": False},
        "records": {"train": len(train), "validation": len(validation)},
        "schemas": {"train": len(train_schema_ids), "validation": len(validation_schema_ids)},
        "schema_overlap": len(overlap),
        "exact_question_overlap": len(train_questions & validation_questions),
        "counterfactual_groups": {
            "train": len({record["counterfactual_group"] for record in train}),
            "validation": len(group_sizes),
            "validation_complete": sum(size >= 2 for size in group_sizes.values()),
        },
        "affinities": {
            "train": dict(sorted(Counter(record["filter_link_affinity"] for record in train).items())),
            "validation": dict(sorted(Counter(record["filter_link_affinity"] for record in validation).items())),
        },
        "all_targets_explicit": all(record["filter_link_explicit"] for record in train + validation),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary_name:
        temporary = Path(temporary_name)
        _write(temporary / "train.jsonl", train)
        _write(temporary / "validation.jsonl", validation)
        report["sha256"] = {
            "train": _sha256(temporary / "train.jsonl"),
            "validation": _sha256(temporary / "validation.jsonl"),
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
        "--train-source", type=Path, default=Path("data/semantic-expansion-v17/train.jsonl")
    )
    parser.add_argument(
        "--validation-source",
        type=Path,
        default=Path("data/semantic-expansion-v17/paired_validation.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=181818)
    args = parser.parse_args()
    print(
        json.dumps(
            build_filter_linking_dataset(
                args.output, args.train_source, args.validation_source, args.seed
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
