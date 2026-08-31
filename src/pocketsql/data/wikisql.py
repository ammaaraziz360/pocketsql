"""Import an execution-validated, grounded subset of WikiSQL v1.1."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import ssl
import tarfile
from typing import Iterable
import unicodedata
from urllib.request import urlopen

import yaml

from pocketsql.data.generate import dataset_quality_report
from pocketsql.data.gretel import mix_records
from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.data.render_sql import render_sql
from pocketsql.model.schema_grounding import canonicalize_inputs, canonicalize_record
from pocketsql.model.tokenizer import load_tokenizer
from pocketsql.training.audit import audit_sequences


DATASET_ID = "salesforce/WikiSQL"
DATASET_VERSION = "1.1"
DATASET_REVISION = "a9e07caff1472ed242bf101c0b6fc6cd5a6fbabf"
DATASET_LICENSE = "BSD-3-Clause"
ARCHIVE_SHA256 = "755c728ab188e364575705c8641f3fafd86fb089cb8b08e8c03f01832aae0881"
ARCHIVE_URL = f"https://raw.githubusercontent.com/salesforce/WikiSQL/{DATASET_REVISION}/data.tar.bz2"
SOURCE_SPLITS = {"train": "train", "validation": "dev", "test": "test"}
AGGREGATES = (None, "MAX", "MIN", "COUNT", "SUM", "AVG")
OPERATORS = ("=", ">", "<")
FAMILY_WEIGHTS = {"wikisql_filter": 2, "wikisql_aggregate": 1}
SLOT = re.compile(r"(?<![A-Za-z0-9_])column\d+(?![A-Za-z0-9_])")
SQLITE_RESERVED = {
    "alter",
    "analyze",
    "attach",
    "case",
    "create",
    "delete",
    "detach",
    "distinct",
    "drop",
    "exists",
    "from",
    "group",
    "having",
    "in",
    "index",
    "insert",
    "join",
    "limit",
    "not",
    "null",
    "on",
    "or",
    "order",
    "pragma",
    "references",
    "returning",
    "select",
    "table",
    "trigger",
    "union",
    "update",
    "vacuum",
    "view",
    "where",
    "with",
}


class WikiSQLRejected(ValueError):
    """A WikiSQL row cannot be used safely by PocketSQL's current model."""


@dataclass
class TableAssets:
    table_name: str
    columns: tuple[str, ...]
    schema_sql: str
    database_sql: str
    connection: sqlite3.Connection


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(cache: Path) -> Path:
    try:
        import certifi
    except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
        raise RuntimeError("Install PocketSQL with the external-data extra to download WikiSQL") from exc
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "wikisql-v1.1.tar.bz2"
    if destination.exists() and _sha256_file(destination) == ARCHIVE_SHA256:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".partial")
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(ARCHIVE_URL, context=ssl_context) as response, temporary.open("wb") as handle:  # noqa: S310
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    if _sha256_file(temporary) != ARCHIVE_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Checksum mismatch for WikiSQL v1.1 archive")
    os.replace(temporary, destination)
    return destination


def _archive_jsonl(archive: tarfile.TarFile, name: str) -> list[dict]:
    member = archive.getmember(f"data/{name}")
    handle = archive.extractfile(member)
    if handle is None:
        raise RuntimeError(f"Missing {name} in WikiSQL archive")
    return [json.loads(line) for line in handle.read().decode("utf-8").splitlines() if line]


def read_source_split(archive_path: Path, source_split: str) -> tuple[list[dict], list[dict]]:
    if _sha256_file(archive_path) != ARCHIVE_SHA256:
        raise RuntimeError("Checksum mismatch for WikiSQL v1.1 archive")
    with tarfile.open(archive_path, "r:bz2") as archive:
        records = _archive_jsonl(archive, f"{source_split}.jsonl")
        tables = _archive_jsonl(archive, f"{source_split}.tables.jsonl")
    return records, tables


def _identifier(value: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or fallback
    if normalized[0].isdigit():
        normalized = "c_" + normalized
    if normalized in SQLITE_RESERVED:
        normalized = "field_" + normalized
    return normalized


def _column_names(headers: list[str], table_name: str) -> tuple[str, ...]:
    used: Counter = Counter()
    names = []
    for index, header in enumerate(headers):
        base = _identifier(header, f"column_{index}")
        if base == table_name:
            base = f"{base}_column"
        used[base] += 1
        names.append(base if used[base] == 1 else f"{base}_{used[base]}")
    return tuple(names)


def _number(value) -> int | float:
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        raw = str(value).replace(",", "")
        try:
            result = float(raw)
        except ValueError:
            match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", raw)
            if not match:
                raise WikiSQLRejected("numeric_conversion_error")
            result = float(match.group())
    if result.is_integer() and -(2**63) <= result <= 2**63 - 1:
        return int(result)
    return result


def _coerce_cell(value, kind: str):
    if value is None:
        return None
    return _number(value) if kind == "real" else str(value)


def _build_table(table: dict) -> TableAssets:
    headers = table["header"]
    kinds = table["types"]
    if not headers or len(headers) != len(kinds):
        raise WikiSQLRejected("invalid_table_schema")
    table_name = _identifier(table.get("name") or table["id"], "wikisql_table")
    columns = _column_names(headers, table_name)
    definitions = [f"{name} {'REAL' if kind == 'real' else 'TEXT'}" for name, kind in zip(columns, kinds)]
    schema_sql = f"CREATE TABLE {table_name} ({', '.join(definitions)});"
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(schema_sql)
        placeholders = ", ".join("?" for _ in columns)
        for row in table["rows"]:
            if len(row) != len(columns):
                raise WikiSQLRejected("invalid_table_row")
            connection.execute(
                f"INSERT INTO {table_name} VALUES ({placeholders})",
                tuple(_coerce_cell(value, kind) for value, kind in zip(row, kinds)),
            )
        connection.commit()
        database_sql = "\n".join(connection.iterdump())
    except Exception:
        connection.close()
        raise
    return TableAssets(table_name, columns, schema_sql, database_sql, connection)


def _mentioned(question: str, value: str | int | float) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(str(value))}(?![A-Za-z0-9_])",
            question,
            re.IGNORECASE,
        )
    )


def _text_value(table: dict, column: int, raw) -> str:
    value = str(raw)
    for row in table["rows"]:
        candidate = row[column]
        if candidate is not None and str(candidate).casefold() == value.casefold():
            return str(candidate)
    return value


def _fits_configs(record: dict, configs: list[tuple[dict, object]]) -> bool:
    for config, tokenizer in configs:
        report = audit_sequences(
            [record],
            tokenizer,
            config["context_length"],
            config.get("generation_max_tokens", 128),
            config.get("canonicalize_identifiers", False),
            config.get("identifier_slot_strategy", "ordered"),
            config.get("canonicalize_literals", False),
        )
        if report["complete_sequences"] != 1 or report["generation_targets_over_cap"]:
            return False
    return True


def _convert_with_assets(
    table: dict,
    source: dict,
    source_split: str,
    assets: TableAssets,
    configs: list[tuple[dict, object]],
) -> dict:
    query = source["sql"]
    if query["agg"] not in range(len(AGGREGATES)):
        raise WikiSQLRejected("unsupported_aggregate")
    if any(condition[1] not in range(len(OPERATORS)) for condition in query["conds"]):
        raise WikiSQLRejected("unsupported_operator")
    if query["sel"] not in range(len(assets.columns)):
        raise WikiSQLRejected("invalid_select_column")

    filters = []
    for column, operator, raw_value in query["conds"]:
        if column not in range(len(assets.columns)):
            raise WikiSQLRejected("invalid_filter_column")
        value = _number(raw_value) if table["types"][column] == "real" else _text_value(table, column, raw_value)
        if not _mentioned(source["question"], value):
            raise WikiSQLRejected("literal_not_mentioned_in_question")
        filters.append(Filter(assets.columns[column], OPERATORS[operator], value))

    aggregate = AGGREGATES[query["agg"]]
    selected = assets.columns[query["sel"]]
    if not aggregate and not filters:
        raise WikiSQLRejected("unsupported_unfiltered_select")
    aggregate_column = None if aggregate == "COUNT" else selected if aggregate else None
    family = "wikisql_aggregate" if aggregate else "wikisql_filter"
    plan = QueryPlan(
        family,
        assets.table_name,
        () if aggregate else (selected,),
        aggregate=aggregate,
        aggregate_column=aggregate_column,
        filters=tuple(filters),
    )
    sql = render_sql(plan)
    try:
        result = assets.connection.execute(sql).fetchmany(1001)
    except sqlite3.Error as exc:
        raise WikiSQLRejected("sqlite_error") from exc
    if not result:
        raise WikiSQLRejected("empty_result")
    if len(result) > 1000:
        raise WikiSQLRejected("result_too_large")

    record = {
        "id": f"wikisql_{source_split}_{source['table_id']}_{source['_index']}",
        "schema_id": f"wikisql_{source['table_id']}",
        "schema_sql": assets.schema_sql,
        "database_sql": assets.database_sql,
        "question": source["question"].strip(),
        "sql": sql,
        "query_plan": plan.normalized(),
        "difficulty": 1 + int(aggregate is not None) + int(len(filters) > 1),
        "seed": None,
        "source": {
            "dataset": DATASET_ID,
            "version": DATASET_VERSION,
            "revision": DATASET_REVISION,
            "split": source_split,
            "table_id": source["table_id"],
            "logical_form": query,
        },
    }

    training = canonicalize_record(record, "permuted", True)
    inference_schema, inference_question, mapping = canonicalize_inputs(
        assets.schema_sql, record["question"], "permuted", True
    )
    inference_sql = mapping.canonicalize_sql(sql)
    if (
        training["schema_sql"] != inference_schema
        or training["question"] != inference_question
        or training["sql"] != inference_sql
        or not mapping.accepts_sql(inference_sql)
    ):
        raise WikiSQLRejected("training_inference_grounding_mismatch")

    required_columns = {item.column for item in filters}
    if aggregate != "COUNT":
        required_columns.add(selected)
    grounded_slots = set(SLOT.findall(inference_question))
    if any(mapping.column_to_slot[column] not in grounded_slots for column in required_columns):
        raise WikiSQLRejected("unresolved_schema_link")
    if not _fits_configs(record, configs):
        raise WikiSQLRejected("sequence_too_long")
    return record


def convert_example(table: dict, source: dict, source_split: str, config_paths: list[Path] | None = None) -> dict:
    """Convert one source example; primarily useful for focused validation and tests."""
    configs = _load_configs(config_paths or [])
    assets = _build_table(table)
    try:
        return _convert_with_assets(table, {**source, "_index": source.get("_index", 0)}, source_split, assets, configs)
    finally:
        assets.connection.close()


def _load_configs(paths: list[Path]) -> list[tuple[dict, object]]:
    loaded = []
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        loaded.append((config, load_tokenizer(config.get("tokenizer_path"))))
    return loaded


def curate_split(
    source_records: list[dict],
    tables: list[dict],
    source_split: str,
    config_paths: list[Path],
) -> tuple[list[dict], Counter]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for index, source in enumerate(source_records):
        grouped[source["table_id"]].append({**source, "_index": index})
    configs = _load_configs(config_paths)
    accepted = []
    rejected: Counter = Counter()
    for table in tables:
        sources = grouped.get(table["id"], ())
        if not sources:
            continue
        try:
            assets = _build_table(table)
        except WikiSQLRejected as exc:
            rejected[str(exc)] += len(sources)
            continue
        except sqlite3.Error:
            rejected["sqlite_schema_error"] += len(sources)
            continue
        try:
            for source in sources:
                try:
                    accepted.append(_convert_with_assets(table, source, source_split, assets, configs))
                except WikiSQLRejected as exc:
                    rejected[str(exc)] += 1
        finally:
            assets.connection.close()
    return accepted, rejected


def _quota(total: int) -> dict[str, int]:
    weight_total = sum(FAMILY_WEIGHTS.values())
    exact = {family: total * weight / weight_total for family, weight in FAMILY_WEIGHTS.items()}
    quota = {family: int(value) for family, value in exact.items()}
    for family in sorted(exact, key=lambda name: exact[name] - quota[name], reverse=True)[: total - sum(quota.values())]:
        quota[family] += 1
    return quota


def balanced_schema_take(
    candidates: list[dict], count: int, seed: int, excluded_schemas: set[str] | None = None
) -> list[dict]:
    excluded_schemas = set(excluded_schemas or ())
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in candidates:
        if record["schema_id"] not in excluded_schemas:
            buckets[record["query_plan"]["family"]].append(record)
    rng = random.Random(seed)
    for records in buckets.values():
        rng.shuffle(records)
    used = set(excluded_schemas)
    selected = []
    for family, family_count in _quota(count).items():
        for record in buckets.get(family, ()):
            if family_count <= 0:
                break
            if record["schema_id"] in used:
                continue
            selected.append(record)
            used.add(record["schema_id"])
            family_count -= 1
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} balanced schema-unique records available for requested {count}")
    rng.shuffle(selected)
    return selected


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_pilot(
    output: Path,
    candidates: dict[str, list[dict]],
    rejected: dict[str, Counter],
    pilot_records: int,
    gate_records: int,
    seed: int,
    mix_with: Path | None = None,
    external_fraction: float = 0.1,
    mixed_train_records: int | None = None,
    mixed_validation_records: int | None = None,
) -> dict[str, int]:
    if pilot_records < 10 or gate_records < 1:
        raise ValueError("pilot_records must be at least 10 and gate_records must be positive")
    output.mkdir(parents=True, exist_ok=True)
    gate = balanced_schema_take(candidates["test"], gate_records, seed)
    gate_schemas = {record["schema_id"] for record in gate}
    test_count = max(1, round(pilot_records * 0.1))
    validation_count = max(1, round(pilot_records * 0.1))
    train_count = pilot_records - test_count - validation_count
    test = balanced_schema_take(candidates["test"], test_count, seed + 1, gate_schemas)
    validation = balanced_schema_take(candidates["validation"], validation_count, seed + 2)
    train = balanced_schema_take(candidates["train"], train_count, seed + 3)
    splits = {"train": train, "validation": validation, "test": test, "external_gate": gate}
    for name, records in splits.items():
        _write_jsonl(output / f"{name}.jsonl", records)

    report = dataset_quality_report(splits)
    report.update(
        {
            "source": {
                "dataset": DATASET_ID,
                "version": DATASET_VERSION,
                "revision": DATASET_REVISION,
                "license": DATASET_LICENSE,
                "archive_sha256": ARCHIVE_SHA256,
                "official_split_used": True,
                "official_split_mapping": SOURCE_SPLITS,
            },
            "selection": {
                "seed": seed,
                "pilot_records": pilot_records,
                "external_gate_records": gate_records,
                "one_record_per_schema": True,
                "family_weights": FAMILY_WEIGHTS,
                "training_use_allowed": {name: name == "train" for name in splits},
            },
            "curation": {
                "accepted_before_balancing": {name: len(records) for name, records in candidates.items()},
                "accepted_by_family": {
                    name: dict(sorted(Counter(record["query_plan"]["family"] for record in records).items()))
                    for name, records in candidates.items()
                },
                "rejected": {name: dict(sorted(counts.items())) for name, counts in rejected.items()},
                "requirements": [
                    "supported_single_table_query_plan",
                    "successful_nonempty_sqlite_execution",
                    "literal_mentioned_in_question",
                    "training_inference_grounding_equivalence",
                    "all_required_columns_grounded",
                    "configured_sequence_limits",
                ],
            },
        }
    )
    if mix_with is not None:
        mixed_train = mix_records(
            _load_jsonl(mix_with / "train.jsonl"), train, external_fraction, seed + 4, mixed_train_records
        )
        mixed_validation = mix_records(
            _load_jsonl(mix_with / "validation.jsonl"),
            validation,
            external_fraction,
            seed + 5,
            mixed_validation_records,
        )
        _write_jsonl(output / "mixed_train.jsonl", mixed_train)
        _write_jsonl(output / "mixed_validation.jsonl", mixed_validation)
        report["mixed"] = {
            "reference_directory": str(mix_with),
            "external_fraction": external_fraction,
            "train_records": len(mixed_train),
            "validation_records": len(mixed_validation),
        }
    (output / "quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {name: len(records) for name, records in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".tmp/wikisql"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--pilot-records", type=int, default=5000)
    parser.add_argument("--gate-records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9292)
    parser.add_argument("--compatibility-config", type=Path, action="append", default=[])
    parser.add_argument("--mix-with", type=Path)
    parser.add_argument("--external-fraction", type=float, default=0.1)
    parser.add_argument("--mixed-train-records", type=int)
    parser.add_argument("--mixed-validation-records", type=int)
    args = parser.parse_args()
    archive_path = args.archive or download_archive(args.cache)
    config_paths = args.compatibility_config or [Path("configs/base_position_robust_v8.yaml")]
    candidates = {}
    rejected = {}
    for output_split, source_split in SOURCE_SPLITS.items():
        source_records, tables = read_source_split(archive_path, source_split)
        candidates[output_split], rejected[output_split] = curate_split(
            source_records, tables, source_split, config_paths
        )
    result = write_pilot(
        args.output,
        candidates,
        rejected,
        args.pilot_records,
        args.gate_records,
        args.seed,
        args.mix_with,
        args.external_fraction,
        args.mixed_train_records,
        args.mixed_validation_records,
    )
    print(
        json.dumps(
            {
                "splits": result,
                "accepted_before_balancing": {name: len(records) for name, records in candidates.items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
