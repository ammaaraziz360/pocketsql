"""Build a frozen, evaluation-only PocketSQL benchmark from Spider 1.0."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import ssl
import tempfile
from typing import Iterable
import unicodedata
from urllib.request import Request, urlopen
import zipfile

import yaml

from pocketsql.data.gretel import GretelRejected, query_plan_from_sql
from pocketsql.data.render_sql import render_sql
from pocketsql.evaluation.normalize import normalize_sql
from pocketsql.model.schema_grounding import canonicalize_inputs, canonicalize_record
from pocketsql.model.tokenizer import load_tokenizer
from pocketsql.training.audit import audit_sequences


DATASET_ID = "Spider 1.0"
DATASET_REVISION = "official-release-2024-09-11"
DATASET_LICENSE = "CC BY-SA 4.0"
ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
)
ARCHIVE_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
CONTRACT_VERSION = "pocketsql-human-alpha-v1"
SPLITS = {
    "train": (("train_spider.json", "train_others.json"), "tables.json", "database"),
    "dev": (("dev.json",), "tables.json", "database"),
    "test": (("test.json",), "test_tables.json", "test_database"),
}
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TYPE_AFFINITY = {
    "boolean": "INTEGER",
    "number": "REAL",
    "text": "TEXT",
    "time": "TEXT",
    "others": "TEXT",
}


class SpiderRejected(ValueError):
    """A Spider example falls outside the frozen PocketSQL alpha contract."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def download_archive(cache: Path) -> Path:
    """Download the official release once and require its pinned checksum."""
    try:
        import certifi
    except ImportError as exc:  # pragma: no cover - optional dependency failure
        raise RuntimeError("Install PocketSQL with the external-data extra to download Spider") from exc
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "spider_data-2024-09-11.zip"
    if destination.exists() and _sha256_file(destination) == ARCHIVE_SHA256:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = Request(ARCHIVE_URL, headers={"User-Agent": "PocketSQL/0.1 Spider benchmark builder"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, context=context) as response, temporary.open("wb") as handle:  # noqa: S310
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    if _sha256_file(temporary) != ARCHIVE_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Checksum mismatch for the official Spider 1.0 archive")
    os.replace(temporary, destination)
    return destination


def _safe_db_id(db_id: str) -> str:
    if not IDENTIFIER.fullmatch(db_id):
        raise SpiderRejected("unsafe_database_identifier")
    return db_id


def read_source_split(
    archive_path: Path,
    split: str,
    database_cache: Path,
) -> tuple[list[dict], dict[str, dict], dict[str, Path]]:
    """Read annotations and extract only the requested split's SQLite files."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}")
    if _sha256_file(archive_path) != ARCHIVE_SHA256:
        raise RuntimeError("Checksum mismatch for the official Spider 1.0 archive")
    annotation_names, tables_name, database_directory = SPLITS[split]
    with zipfile.ZipFile(archive_path) as archive:
        rows = [
            row
            for annotation_name in annotation_names
            for row in json.loads(archive.read(f"spider_data/{annotation_name}"))
        ]
        tables = json.loads(archive.read(f"spider_data/{tables_name}"))
        table_by_id = {item["db_id"]: item for item in tables}
        database_cache.mkdir(parents=True, exist_ok=True)
        database_paths = {}
        for db_id in sorted({row["db_id"] for row in rows}):
            _safe_db_id(db_id)
            member = f"spider_data/{database_directory}/{db_id}/{db_id}.sqlite"
            destination = database_cache / f"{split}-{db_id}.sqlite"
            temporary = destination.with_suffix(".sqlite.partial")
            try:
                source = archive.open(member)
            except KeyError as exc:
                raise RuntimeError(f"Official archive is missing SQLite database {db_id!r}") from exc
            with source, temporary.open("wb") as handle:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
            os.replace(temporary, destination)
            database_paths[db_id] = destination
    missing_metadata = sorted(set(database_paths) - set(table_by_id))
    if missing_metadata:
        raise RuntimeError(f"Official archive is missing table metadata for: {missing_metadata}")
    return rows, table_by_id, database_paths


def schema_sql_from_metadata(metadata: dict) -> str:
    """Render concise, canonicalization-compatible DDL from Spider table metadata."""
    tables = metadata["table_names_original"]
    columns = metadata["column_names_original"]
    types = metadata["column_types"]
    if len(columns) != len(types) or not tables:
        raise SpiderRejected("invalid_schema_metadata")
    if any(not IDENTIFIER.fullmatch(name) for name in tables):
        raise SpiderRejected("unsupported_schema_identifier")
    if len({name.casefold() for name in tables}) != len(tables):
        raise SpiderRejected("case_colliding_table_identifiers")

    by_table: dict[int, list[tuple[int, str, str]]] = {index: [] for index in range(len(tables))}
    spellings: dict[str, set[str]] = {}
    for index, ((table_index, name), kind) in enumerate(zip(columns, types)):
        if table_index < 0:
            continue
        if table_index not in by_table or not IDENTIFIER.fullmatch(name):
            raise SpiderRejected("unsupported_schema_identifier")
        by_table[table_index].append((index, name, TYPE_AFFINITY.get(kind, "TEXT")))
        spellings.setdefault(name.casefold(), set()).add(name)
    if any(len(values) > 1 for values in spellings.values()):
        raise SpiderRejected("case_colliding_column_identifiers")
    if set(name.casefold() for name in tables) & set(spellings):
        raise SpiderRejected("table_column_identifier_overlap")
    for fields in by_table.values():
        if not fields or len({name.casefold() for _, name, _ in fields}) != len(fields):
            raise SpiderRejected("duplicate_or_missing_columns")

    primary_keys = set(metadata.get("primary_keys", ()))
    primary_by_table: Counter = Counter(
        columns[index][0] for index in primary_keys if 0 <= index < len(columns) and columns[index][0] >= 0
    )
    foreign_by_source: dict[int, tuple[int, str]] = {}
    for source, target in metadata.get("foreign_keys", ()):
        if not (0 <= source < len(columns) and 0 <= target < len(columns)):
            raise SpiderRejected("invalid_foreign_key_metadata")
        target_table, target_column = columns[target]
        if target_table < 0 or source in foreign_by_source:
            raise SpiderRejected("unsupported_foreign_key_metadata")
        foreign_by_source[source] = (target_table, target_column)

    statements = []
    for table_index, table_name in enumerate(tables):
        definitions = []
        for column_index, column_name, affinity in by_table[table_index]:
            definition = f"{column_name} {affinity}"
            if column_index in primary_keys and primary_by_table[table_index] == 1:
                definition += " PRIMARY KEY"
            if column_index in foreign_by_source:
                target_table, target_column = foreign_by_source[column_index]
                definition += f" REFERENCES {tables[target_table]}({target_column})"
            definitions.append(definition)
        statements.append(f"CREATE TABLE {table_name} ({', '.join(definitions)});")
    schema_sql = "\n".join(statements)
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(schema_sql)
    except sqlite3.Error as exc:
        raise SpiderRejected("unexecutable_prompt_schema") from exc
    finally:
        connection.close()
    return schema_sql


def _foreign_key_edges(metadata: dict) -> set[frozenset[str]]:
    tables = metadata["table_names_original"]
    columns = metadata["column_names_original"]

    def reference(index: int) -> str:
        table_index, column = columns[index]
        return f"{tables[table_index]}.{column}".casefold()

    return {frozenset((reference(left), reference(right))) for left, right in metadata.get("foreign_keys", ())}


def _mentioned(question: str, value: str | int | float) -> bool:
    raw = str(value)
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(raw)}(?![A-Za-z0-9_])",
            question,
            re.IGNORECASE,
        )
    )


def _family(plan) -> str:
    if plan.join_table and plan.aggregate:
        return "spider_joined_aggregate"
    if plan.join_table:
        return "spider_join"
    if plan.group_by:
        return "spider_group"
    if plan.aggregate:
        return "spider_aggregate"
    if plan.distinct:
        return "spider_distinct"
    if plan.filters:
        return "spider_filter"
    if plan.order_by or plan.limit is not None:
        return "spider_order_limit"
    return "spider_select"


def _execute(connection: sqlite3.Connection, sql: str) -> list[tuple]:
    progress_calls = 0

    def progress() -> int:
        nonlocal progress_calls
        progress_calls += 1
        return int(progress_calls > 500)

    connection.set_progress_handler(progress, 10_000)
    try:
        rows = connection.execute(sql).fetchmany(1001)
    except sqlite3.Error as exc:
        raise SpiderRejected("sqlite_execution_error") from exc
    finally:
        connection.set_progress_handler(None, 0)
    if len(rows) > 1000:
        raise SpiderRejected("result_too_large")
    return rows


def _equivalent(left: list[tuple], right: list[tuple], ordered: bool) -> bool:
    return left == right if ordered else Counter(left) == Counter(right)


def _fits_configs(record: dict, configs: list[tuple[dict, object]]) -> bool:
    for config, tokenizer in configs:
        try:
            report = audit_sequences(
                [record],
                tokenizer,
                config["context_length"],
                config.get("generation_max_tokens", 128),
                config.get("canonicalize_identifiers", False),
                config.get("identifier_slot_strategy", "ordered"),
                config.get("canonicalize_literals", False),
                config.get("target_format", "sql"),
                config.get("schema_linking_hints", False),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SpiderRejected("schema_grounding_error") from exc
        if report["complete_sequences"] != 1 or report["generation_targets_over_cap"]:
            return False
    return True


def convert_example(
    source: dict,
    source_index: int,
    metadata: dict,
    database_path: Path,
    split: str,
    configs: list[tuple[dict, object]] | None = None,
    training_use_allowed: bool = False,
) -> dict:
    """Convert and execution-check one human-authored Spider example."""
    configs = configs or []
    schema_sql = schema_sql_from_metadata(metadata)
    try:
        plan = query_plan_from_sql(source["query"])
    except GretelRejected as exc:
        raise SpiderRejected(str(exc)) from exc
    plan = replace(plan, family=_family(plan))
    if plan.join_on and frozenset(value.casefold() for value in plan.join_on) not in _foreign_key_edges(metadata):
        raise SpiderRejected("join_is_not_declared_foreign_key")
    if any(not _mentioned(source["question"], item.value) for item in plan.filters):
        raise SpiderRejected("literal_not_mentioned_in_question")
    if plan.limit is not None and not _mentioned(source["question"], plan.limit):
        raise SpiderRejected("limit_not_mentioned_in_question")

    normalized_sql = render_sql(plan)
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise SpiderRejected("sqlite_open_error") from exc
    try:
        connection.execute("PRAGMA query_only = ON")
        original_rows = _execute(connection, source["query"])
        normalized_rows = _execute(connection, normalized_sql)
    finally:
        connection.close()
    if not original_rows:
        raise SpiderRejected("empty_result")
    if not _equivalent(original_rows, normalized_rows, plan.order_by is not None):
        raise SpiderRejected("normalization_changed_result")

    db_id = _safe_db_id(source["db_id"])
    record = {
        "id": f"spider_{split}_{source_index:04d}",
        "schema_id": f"spider_{db_id}",
        "schema_sql": schema_sql,
        "database_path": f"databases/{db_id}.sqlite",
        "question": source["question"].strip(),
        "sql": normalized_sql,
        "query_plan": plan.normalized(),
        "difficulty": 1
        + int(plan.aggregate is not None)
        + int(plan.join_table is not None)
        + int(len(plan.filters) > 1)
        + int(bool(plan.group_by)),
        "seed": None,
        "source": {
            "dataset": DATASET_ID,
            "revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            "split": split,
            "index": source_index,
            "db_id": db_id,
            "original_sql": source["query"].strip(),
            "human_authored": True,
            "training_use_allowed": training_use_allowed,
        },
    }

    try:
        training = canonicalize_record(record, "permuted", True)
        inference_schema, inference_question, mapping = canonicalize_inputs(
            schema_sql, record["question"], "permuted", True
        )
        inference_sql = mapping.canonicalize_sql(normalized_sql)
    except ValueError as exc:
        raise SpiderRejected("schema_grounding_error") from exc
    if (
        training["schema_sql"] != inference_schema
        or training["question"] != inference_question
        or training["sql"] != inference_sql
        or not mapping.accepts_sql(inference_sql)
    ):
        raise SpiderRejected("training_inference_grounding_mismatch")
    if not _fits_configs(record, configs):
        raise SpiderRejected("sequence_too_long")
    return record


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _schema_key(record: dict) -> str:
    return _normalize_text(record.get("schema_sql", ""))


def _sql_key(record: dict) -> str:
    return normalize_sql(record.get("sql", "")).casefold()


def discover_training_files(data_root: Path, excluded: Path | None = None) -> list[Path]:
    """Find every local JSONL that is explicitly named as a training split."""
    excluded = excluded.resolve() if excluded else None
    paths = []
    if not data_root.exists():
        return paths
    for path in data_root.rglob("*.jsonl"):
        if excluded and (path.resolve() == excluded or excluded in path.resolve().parents):
            continue
        if path.name in {"train.jsonl", "mixed_train.jsonl"} or path.name.endswith("_train.jsonl"):
            paths.append(path)
    return sorted(paths)


def training_contamination_index(paths: Iterable[Path]) -> tuple[dict[str, dict[tuple, list[str]]], dict[str, str]]:
    """Index exact record-level overlaps without treating common SQL shapes as leakage."""
    indexes: dict[str, dict[tuple, list[str]]] = {
        "schema_question": {},
        "schema_sql": {},
        "question_sql": {},
    }
    file_hashes = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        file_hashes[str(path)] = _sha256_file(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not all(key in record for key in ("schema_sql", "question", "sql")):
                    continue
                label = f"{path}:{line_number}:{record.get('id', 'unknown')}"
                schema = _schema_key(record)
                question = _normalize_text(record["question"])
                sql = _sql_key(record)
                for name, key in (
                    ("schema_question", (schema, question)),
                    ("schema_sql", (schema, sql)),
                    ("question_sql", (question, sql)),
                ):
                    indexes[name].setdefault(key, []).append(label)
    return indexes, file_hashes


def contamination_matches(record: dict, indexes: dict[str, dict[tuple, list[str]]]) -> dict[str, list[str]]:
    schema = _schema_key(record)
    question = _normalize_text(record["question"])
    sql = _sql_key(record)
    keys = {
        "schema_question": (schema, question),
        "schema_sql": (schema, sql),
        "question_sql": (question, sql),
    }
    return {name: indexes[name][key] for name, key in keys.items() if key in indexes[name]}


def _load_configs(paths: list[Path]) -> list[tuple[dict, object]]:
    loaded = []
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        loaded.append((config, load_tokenizer(config.get("tokenizer_path"))))
    return loaded


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def build_benchmark(
    output: Path,
    source_rows: list[dict],
    table_by_id: dict[str, dict],
    database_paths: dict[str, Path],
    split: str,
    config_paths: list[Path],
    training_files: list[Path],
    archive_path: Path,
) -> dict:
    """Freeze every contract-compatible row before any model is evaluated."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen benchmark directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    configs = _load_configs(config_paths)
    contamination_index, training_hashes = training_contamination_index(training_files)
    accepted = []
    rejection_rows = []
    rejection_counts: Counter = Counter()
    contamination_counts: Counter = Counter()
    for index, source in enumerate(source_rows):
        try:
            metadata = table_by_id[source["db_id"]]
            record = convert_example(
                source,
                index,
                metadata,
                database_paths[source["db_id"]],
                split,
                configs,
            )
            matches = contamination_matches(record, contamination_index)
            if matches:
                contamination_counts.update(matches)
                reason = "training_contamination:" + ",".join(sorted(matches))
                raise SpiderRejected(reason)
            accepted.append(record)
        except (KeyError, SpiderRejected) as exc:
            reason = "missing_schema_metadata" if isinstance(exc, KeyError) else str(exc)
            rejection_counts[reason] += 1
            rejection_rows.append(
                {
                    "id": f"spider_{split}_{index:04d}",
                    "db_id": source.get("db_id"),
                    "question": source.get("question"),
                    "original_sql": source.get("query"),
                    "reason": reason,
                }
            )
    if not accepted:
        raise RuntimeError("No Spider examples satisfy the PocketSQL alpha contract")

    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary_name:
        temporary = Path(temporary_name)
        databases_output = temporary / "databases"
        databases_output.mkdir()
        database_hashes = {}
        for db_id in sorted({record["source"]["db_id"] for record in accepted}):
            destination = databases_output / f"{db_id}.sqlite"
            shutil.copyfile(database_paths[db_id], destination)
            database_hashes[db_id] = _sha256_file(destination)

        benchmark_path = temporary / "benchmark.jsonl"
        _write_jsonl(benchmark_path, accepted)
        _write_jsonl(temporary / "rejections.jsonl", rejection_rows)
        audit = audit_sequences(
            accepted,
            configs[0][1],
            configs[0][0]["context_length"],
            configs[0][0].get("generation_max_tokens", 128),
            configs[0][0].get("canonicalize_identifiers", False),
            configs[0][0].get("identifier_slot_strategy", "ordered"),
            configs[0][0].get("canonicalize_literals", False),
            configs[0][0].get("target_format", "sql"),
        )
        accepted_ids = [record["id"] for record in accepted]
        freeze_payload = {
            "contract_version": CONTRACT_VERSION,
            "source_archive_sha256": ARCHIVE_SHA256,
            "split": split,
            "accepted_ids": accepted_ids,
            "compatibility_configs": {str(path): _sha256_file(path) for path in config_paths},
            "training_files": training_hashes,
        }
        freeze_sha256 = _sha256_text(json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")))
        report = {
            "benchmark": {
                "name": "PocketSQL Human Alpha v1",
                "contract_version": CONTRACT_VERSION,
                "records": len(accepted),
                "schemas": len({record["schema_id"] for record in accepted}),
                "benchmark_jsonl_sha256": _sha256_file(benchmark_path),
                "freeze_sha256": freeze_sha256,
                "training_use_allowed": False,
            },
            "source": {
                "dataset": DATASET_ID,
                "revision": DATASET_REVISION,
                "license": DATASET_LICENSE,
                "official_url": "https://yale-lily.github.io/spider",
                "archive_url": ARCHIVE_URL,
                "archive_sha256": ARCHIVE_SHA256,
                "official_split": split,
                "source_records": len(source_rows),
                "source_schemas": len({row["db_id"] for row in source_rows}),
                "human_authored": True,
            },
            "selection": {
                "accepted": len(accepted),
                "rejected": len(rejection_rows),
                "accepted_by_family": dict(sorted(Counter(record["query_plan"]["family"] for record in accepted).items())),
                "accepted_by_schema": dict(sorted(Counter(record["schema_id"] for record in accepted).items())),
                "rejected_by_reason": dict(sorted(rejection_counts.items())),
                "selected_before_model_evaluation": True,
                "model_outputs_consulted": False,
                "sampling": "none; every compatible official-split row is retained",
            },
            "contract": {
                "statement": "one read-only SELECT",
                "maximum_tables": 2,
                "maximum_joins": 1,
                "join_requirement": "declared foreign-key equality",
                "filter_operators": ["=", ">", "<", ">=", "<="],
                "filter_connectors": ["AND", "OR"],
                "aggregates": ["COUNT", "SUM", "AVG", "MIN", "MAX"],
                "unsupported": ["subqueries", "CTEs", "set operations", "window functions", "HAVING", "mixed Boolean trees"],
                "literal_policy": "filter and LIMIT literals must occur verbatim in the question",
                "result_policy": "gold and normalized queries must execute equivalently and return 1-1000 rows",
            },
            "contamination": {
                "training_files_scanned": training_hashes,
                "rejected_by_match_type": dict(sorted(contamination_counts.items())),
                "accepted_overlap_count": 0,
            },
            "sequence_audit": audit,
            "database_sha256": database_hashes,
            "freeze_payload": freeze_payload,
        }
        (temporary / "quality_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "ATTRIBUTION.md").write_text(
            "# PocketSQL Human Alpha v1\n\n"
            "This evaluation-only derivative contains a filtered subset of Spider 1.0, "
            "created by Tao Yu et al. for EMNLP 2018. Spider is licensed under "
            "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).\n\n"
            "Official dataset: https://yale-lily.github.io/spider\n\n"
            "Do not use `benchmark.jsonl` for training, validation-based checkpoint selection, "
            "interpolation, prompt tuning, or data generation.\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".tmp/spider"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--split", choices=sorted(SPLITS), default="test")
    parser.add_argument(
        "--compatibility-config",
        type=Path,
        action="append",
        default=[],
        help="Require every accepted sequence to fit this model configuration.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--training-data",
        type=Path,
        action="append",
        default=[],
        help="Training JSONL to scan for contamination; defaults to all local training splits.",
    )
    args = parser.parse_args()
    archive_path = args.archive or download_archive(args.cache)
    config_paths = args.compatibility_config or [Path("configs/base_semantic_v11_composed.yaml")]
    training_files = args.training_data or discover_training_files(args.data_root, args.output)
    source_rows, table_by_id, database_paths = read_source_split(
        archive_path,
        args.split,
        args.cache / "databases",
    )
    report = build_benchmark(
        args.output,
        source_rows,
        table_by_id,
        database_paths,
        args.split,
        config_paths,
        training_files,
        archive_path,
    )
    print(json.dumps({"output": str(args.output), **report["benchmark"]}, sort_keys=True))


if __name__ == "__main__":
    main()
