"""Import a conservative, SQLite-compatible subset of Gretel's text-to-SQL data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import ssl
from typing import Iterable, Iterator
from urllib.request import urlopen

import yaml

from pocketsql.data.generate import dataset_quality_report
from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.data.render_sql import render_sql
from pocketsql.model.schema_grounding import canonicalize_inputs, canonicalize_record
from pocketsql.model.tokenizer import load_tokenizer
from pocketsql.training.audit import audit_sequences


DATASET_ID = "gretelai/synthetic_text_to_sql"
DATASET_REVISION = "740ab236e64503fba51be1101df7a1be83bf455d"
DATASET_LICENSE = "Apache-2.0"
SOURCE_FILES = {
    "train": (
        "synthetic_text_to_sql_train.snappy.parquet",
        "2bee9ac07cf5057d36b5ea30fb47d948697e882f42bd1cc661185396287c0180",
    ),
    "test": (
        "synthetic_text_to_sql_test.snappy.parquet",
        "f2056edcd897db59c89d12bde149a36d4e242ac0d2c6d4d42b1f2bc764318993",
    ),
}
FAMILY_WEIGHTS = {
    "gretel_select": 2,
    "gretel_filter": 4,
    "gretel_aggregate": 3,
    "gretel_group": 2,
    "gretel_join": 3,
    "gretel_distinct": 1,
    "gretel_order_limit": 1,
}
_COMPARISONS: dict[type, str] | None = None
_CONTEXT_FORBIDDEN = re.compile(r"\b(?:ATTACH|DETACH|PRAGMA|VACUUM|TRIGGER|VIRTUAL|LOAD_EXTENSION)\b", re.I)


class GretelRejected(ValueError):
    """A source row cannot be represented safely by PocketSQL's current IR."""


def _sqlglot():
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
        raise RuntimeError("Install PocketSQL with the external-data extra to import Gretel data") from exc
    global _COMPARISONS
    if _COMPARISONS is None:
        _COMPARISONS = {exp.EQ: "=", exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<="}
    return sqlglot, exp


def _unwrap(node):
    _, exp = _sqlglot()
    while isinstance(node, (exp.Paren, exp.Alias)):
        node = node.this
    return node


def _plain_table(node) -> tuple[str, str]:
    _, exp = _sqlglot()
    if not isinstance(node, exp.Table) or node.catalog or node.db or not node.name:
        raise GretelRejected("qualified_or_computed_table")
    if node.this.args.get("quoted"):
        raise GretelRejected("quoted_identifier")
    return node.name, node.alias or node.name


def _column_name(node, aliases: dict[str, str]) -> str:
    _, exp = _sqlglot()
    node = _unwrap(node)
    if not isinstance(node, exp.Column):
        raise GretelRejected("column_expression")
    if getattr(node.this, "args", {}).get("quoted"):
        raise GretelRejected("quoted_identifier")
    name = "*" if isinstance(node.this, exp.Star) else node.name
    if not name:
        raise GretelRejected("missing_column")
    if not node.table:
        return name
    table = aliases.get(node.table.casefold())
    if table is None:
        raise GretelRejected("unknown_table_alias")
    return f"{table}.{name}"


def _literal_value(node) -> str | int | float:
    _, exp = _sqlglot()
    node = _unwrap(node)
    if not isinstance(node, exp.Literal):
        raise GretelRejected("literal_expression")
    if node.is_string:
        return node.this
    try:
        return int(node.this)
    except ValueError:
        try:
            return float(node.this)
        except ValueError as exc:
            raise GretelRejected("non_numeric_literal") from exc


def _filters(node, aliases: dict[str, str]) -> tuple[tuple[Filter, ...], str]:
    _, exp = _sqlglot()
    connectors: set[str] = set()
    found: list[Filter] = []

    def visit(item) -> None:
        item = _unwrap(item)
        if isinstance(item, (exp.And, exp.Or)):
            connectors.add("AND" if isinstance(item, exp.And) else "OR")
            visit(item.left)
            visit(item.right)
            return
        operator = next((symbol for kind, symbol in (_COMPARISONS or {}).items() if isinstance(item, kind)), None)
        if operator is None:
            raise GretelRejected("unsupported_predicate")
        found.append(Filter(_column_name(item.left, aliases), operator, _literal_value(item.right)))

    visit(node)
    if len(connectors) > 1:
        raise GretelRejected("mixed_filter_connectors")
    return tuple(found), next(iter(connectors), "AND")


def query_plan_from_sql(sql: str) -> QueryPlan:
    """Losslessly lower the supported SQL subset to PocketSQL's typed query plan."""
    sqlglot, exp = _sqlglot()
    try:
        trees = sqlglot.parse(sql)
    except Exception as exc:
        raise GretelRejected("query_parse_error") from exc
    if len(trees) != 1 or not isinstance(trees[0], exp.Select):
        raise GretelRejected("not_one_select")
    tree = trees[0]
    if len(list(tree.find_all(exp.Select))) != 1:
        raise GretelRejected("nested_select")
    if tree.args.get("with_") or tree.args.get("having") or tree.find(exp.Window) or tree.find(exp.SetOperation):
        raise GretelRejected("advanced_clause")

    source = tree.args.get("from_")
    if source is None or source.this is None:
        raise GretelRejected("missing_from")
    table, table_alias = _plain_table(source.this)
    aliases = {table.casefold(): table, table_alias.casefold(): table}
    joins = list(tree.args.get("joins") or ())
    if len(joins) > 1:
        raise GretelRejected("multiple_joins")
    join_table = None
    join_on = None
    if joins:
        join = joins[0]
        if (join.args.get("side") or "").upper() or (join.args.get("kind") or "").upper() not in {"", "INNER"}:
            raise GretelRejected("non_inner_join")
        join_table, join_alias = _plain_table(join.this)
        aliases.update({join_table.casefold(): join_table, join_alias.casefold(): join_table})
        condition = _unwrap(join.args.get("on"))
        if not isinstance(condition, exp.EQ):
            raise GretelRejected("complex_join_condition")
        join_on = (_column_name(condition.left, aliases), _column_name(condition.right, aliases))

    columns: list[str] = []
    aggregate = None
    aggregate_column = None
    aggregate_position = None
    output_aliases: dict[str, str] = {}
    aggregate_types = ((exp.Count, "COUNT"), (exp.Sum, "SUM"), (exp.Avg, "AVG"), (exp.Min, "MIN"), (exp.Max, "MAX"))
    for position, expression in enumerate(tree.expressions):
        alias = expression.alias if isinstance(expression, exp.Alias) else ""
        item = _unwrap(expression)
        if isinstance(item, (exp.Column, exp.Star)):
            column = "*" if isinstance(item, exp.Star) else _column_name(item, aliases)
            columns.append(column)
            if alias:
                output_aliases[alias.casefold()] = column
            continue
        operation = next((name for kind, name in aggregate_types if isinstance(item, kind)), None)
        if operation is None or aggregate is not None:
            raise GretelRejected("unsupported_projection")
        target = _unwrap(item.this)
        if not isinstance(target, (exp.Column, exp.Star)):
            raise GretelRejected("aggregate_expression")
        aggregate = operation
        aggregate_column = None if isinstance(target, exp.Star) else _column_name(target, aliases)
        if operation != "COUNT" and aggregate_column is None:
            raise GretelRejected("star_non_count_aggregate")
        aggregate_position = position

    distinct_node = tree.args.get("distinct")
    if distinct_node is not None and distinct_node.args.get("on"):
        raise GretelRejected("distinct_on")

    where = tree.args.get("where")
    filters, connector = _filters(where.this, aliases) if where is not None else ((), "AND")
    group = tree.args.get("group")
    group_by = tuple(_column_name(item, aliases) for item in group.expressions) if group is not None else ()
    if group_by:
        if aggregate is None or tuple(columns) != group_by:
            raise GretelRejected("unsupported_group_projection")
    elif aggregate is not None and columns:
        raise GretelRejected("mixed_aggregate_projection")

    order_by = None
    descending = False
    order = tree.args.get("order")
    if order is not None:
        if len(order.expressions) != 1:
            raise GretelRejected("multiple_order_columns")
        ordered = order.expressions[0]
        order_item = _unwrap(ordered.this)
        if isinstance(order_item, exp.Column) and not order_item.table and order_item.name.casefold() in output_aliases:
            order_by = output_aliases[order_item.name.casefold()]
        else:
            order_by = _column_name(order_item, aliases)
        descending = bool(ordered.args.get("desc"))

    limit = None
    limit_node = tree.args.get("limit")
    if limit_node is not None:
        value = _literal_value(limit_node.expression)
        if not isinstance(value, int) or value <= 0:
            raise GretelRejected("non_positive_integer_limit")
        limit = value
    if tree.args.get("offset"):
        raise GretelRejected("offset")

    if join_table:
        family = "gretel_join"
    elif group_by:
        family = "gretel_group"
    elif aggregate:
        family = "gretel_aggregate"
    elif distinct_node is not None:
        family = "gretel_distinct"
    elif filters:
        family = "gretel_filter"
    elif order_by or limit is not None:
        family = "gretel_order_limit"
    else:
        family = "gretel_select"
    return QueryPlan(
        family,
        table,
        tuple(columns),
        aggregate=aggregate,
        aggregate_column=aggregate_column,
        distinct=distinct_node is not None,
        filters=filters,
        filter_connector=connector,
        group_by=group_by,
        order_by=order_by,
        descending=descending,
        limit=limit,
        join_table=join_table,
        join_on=join_on,
        aggregate_position=aggregate_position or 0,
    )


def _safe_context(context: str) -> bool:
    sqlglot, exp = _sqlglot()
    if _CONTEXT_FORBIDDEN.search(context):
        return False
    try:
        statements = sqlglot.parse(context)
    except Exception:
        return False
    if not statements:
        return False
    for statement in statements:
        if isinstance(statement, exp.Insert):
            continue
        if isinstance(statement, exp.Create) and (statement.args.get("kind") or "").upper() in {"TABLE", "VIEW"}:
            continue
        return False
    return True


def _equivalent(left: list[tuple], right: list[tuple], ordered: bool) -> bool:
    return left == right if ordered else Counter(left) == Counter(right)


def _question_mentions_value(question: str, value: str | int | float) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(str(value))}(?![A-Za-z0-9_])",
            question,
            re.IGNORECASE,
        )
    )


def _schema_sql(connection: sqlite3.Connection) -> str:
    statements = [
        row[0]
        for row in connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "AND sql IS NOT NULL ORDER BY name"
        )
    ]
    return ";\n".join(statements) + (";" if statements else "")


def convert_row(row: dict, source_split: str) -> dict:
    """Convert one source row, rejecting anything unsafe, ambiguous, or lossy."""
    if row.get("sql_task_type") not in {"analytics and reporting", "data retrieval"}:
        raise GretelRejected("non_retrieval_task")
    plan = query_plan_from_sql(row["sql"])
    values = [item.value for item in plan.filters]
    if plan.limit is not None:
        values.append(plan.limit)
    if any(not _question_mentions_value(row["sql_prompt"], value) for value in values):
        raise GretelRejected("literal_not_mentioned_in_question")
    normalized_sql = render_sql(plan)
    context = row["sql_context"]
    if not _safe_context(context):
        raise GretelRejected("unsafe_or_unparseable_context")

    connection = sqlite3.connect(":memory:")
    progress_calls = 0

    def progress() -> int:
        nonlocal progress_calls
        progress_calls += 1
        return int(progress_calls > 200)

    connection.set_progress_handler(progress, 10_000)
    try:
        connection.executescript(context)
        schema_sql = _schema_sql(connection)
        if not schema_sql:
            raise GretelRejected("no_tables")
        connection.execute("PRAGMA query_only = ON")
        original_rows = connection.execute(row["sql"]).fetchmany(1001)
        normalized_rows = connection.execute(normalized_sql).fetchmany(1001)
        if len(original_rows) > 1000 or len(normalized_rows) > 1000:
            raise GretelRejected("result_too_large")
        if not original_rows:
            raise GretelRejected("empty_result")
        if not _equivalent(original_rows, normalized_rows, plan.order_by is not None):
            raise GretelRejected("normalization_changed_result")
        database_sql = "\n".join(connection.iterdump())
    except GretelRejected:
        raise
    except sqlite3.Error as exc:
        raise GretelRejected("sqlite_error") from exc
    finally:
        connection.close()

    schema_hash = sha256(" ".join(schema_sql.casefold().split()).encode("utf-8")).hexdigest()
    record = {
        "id": f"gretel_{row['id']}",
        "schema_id": f"gretel_{schema_hash[:20]}",
        "schema_sql": schema_sql,
        "database_sql": database_sql,
        "question": row["sql_prompt"].strip(),
        "sql": normalized_sql,
        "query_plan": plan.normalized(),
        "difficulty": 1 + int(plan.aggregate is not None) + int(plan.join_table is not None) + int(len(plan.filters) > 1),
        "seed": None,
        "source": {
            "dataset": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": source_split,
            "id": row["id"],
            "domain": row["domain"],
            "declared_complexity": row["sql_complexity"],
            "original_sql": row["sql"].strip(),
        },
    }

    # Training obtains literal slots from query_plan; inference obtains them
    # from the question. Reject examples where those two paths disagree.
    try:
        training_record = canonicalize_record(record, "permuted", True)
        inference_schema, inference_question, mapping = canonicalize_inputs(
            schema_sql, record["question"], "permuted", True
        )
        inference_sql = mapping.canonicalize_sql(normalized_sql)
    except ValueError as exc:
        raise GretelRejected("schema_grounding_error") from exc
    if (
        training_record["schema_sql"] != inference_schema
        or training_record["question"] != inference_question
        or training_record["sql"] != inference_sql
        or not mapping.accepts_sql(inference_sql)
    ):
        raise GretelRejected("training_inference_grounding_mismatch")
    return record


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_sources(cache: Path) -> dict[str, Path]:
    try:
        import certifi
    except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
        raise RuntimeError("Install PocketSQL with the external-data extra to download Gretel data") from exc
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, (filename, expected) in SOURCE_FILES.items():
        destination = cache / filename
        if not destination.exists() or _sha256_file(destination) != expected:
            temporary = destination.with_suffix(destination.suffix + ".partial")
            url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{filename}"
            with urlopen(url, context=ssl_context) as response, temporary.open("wb") as handle:  # noqa: S310 - pinned URL and checksum
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if _sha256_file(temporary) != expected:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"Checksum mismatch for {filename}")
            os.replace(temporary, destination)
        paths[split] = destination
    return paths


def parquet_rows(paths: dict[str, Path]) -> Iterator[tuple[str, dict]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
        raise RuntimeError("Install PocketSQL with the external-data extra to read Parquet files") from exc
    for split, path in paths.items():
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                yield split, row


def _load_compatibility_configs(paths: list[Path]) -> list[tuple[dict, object]]:
    loaded = []
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        loaded.append((config, load_tokenizer(config.get("tokenizer_path"))))
    return loaded


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


def curate_rows(rows: Iterable[tuple[str, dict]], configs: list[Path]) -> tuple[list[dict], Counter]:
    candidates = []
    rejected: Counter = Counter()
    seen = set()
    compatibility = _load_compatibility_configs(configs)
    for split, row in rows:
        try:
            record = convert_row(row, split)
            if not _fits_configs(record, compatibility):
                raise GretelRejected("sequence_too_long")
            key = (record["schema_id"], record["question"].casefold())
            if key in seen:
                raise GretelRejected("duplicate_schema_question")
        except GretelRejected as exc:
            rejected[str(exc)] += 1
            continue
        seen.add(key)
        candidates.append(record)
    return candidates, rejected


def _quota(total: int) -> dict[str, int]:
    weight_total = sum(FAMILY_WEIGHTS.values())
    exact = {family: total * weight / weight_total for family, weight in FAMILY_WEIGHTS.items()}
    quota = {family: int(value) for family, value in exact.items()}
    for family in sorted(exact, key=lambda name: exact[name] - quota[name], reverse=True)[: total - sum(quota.values())]:
        quota[family] += 1
    return quota


def balanced_take(
    candidates: list[dict],
    count: int,
    seed: int,
    excluded_schemas: set[str],
) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in candidates:
        if record["schema_id"] not in excluded_schemas:
            buckets[record["query_plan"]["family"]].append(record)
    rng = random.Random(seed)
    for records in buckets.values():
        rng.shuffle(records)
    selected = []
    used = set(excluded_schemas)
    for family, family_count in _quota(count).items():
        for record in buckets.get(family, ()):
            if family_count <= 0:
                break
            if record["schema_id"] in used:
                continue
            selected.append(record)
            used.add(record["schema_id"])
            family_count -= 1
    if len(selected) < count:
        remaining = [
            record
            for records in buckets.values()
            for record in records
            if record["schema_id"] not in used
        ]
        rng.shuffle(remaining)
        for record in remaining:
            if len(selected) >= count:
                break
            if record["schema_id"] in used:
                continue
            selected.append(record)
            used.add(record["schema_id"])
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} schema-unique records available for requested {count}")
    rng.shuffle(selected)
    return selected


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def mix_records(
    reference: list[dict],
    external: list[dict],
    external_fraction: float,
    seed: int,
    total_records: int | None = None,
) -> list[dict]:
    if not 0 < external_fraction < 1:
        raise ValueError("external_fraction must be between zero and one")
    external_needed = round(total_records * external_fraction) if total_records is not None else len(external)
    reference_needed = (
        total_records - external_needed
        if total_records is not None
        else round(len(external) * (1 - external_fraction) / external_fraction)
    )
    if external_needed > len(external):
        raise ValueError(f"Need {external_needed} external records but only {len(external)} are available")
    if reference_needed > len(reference):
        raise ValueError(f"Need {reference_needed} reference records but only {len(reference)} are available")
    rng = random.Random(seed)
    chosen = rng.sample(reference, reference_needed) + rng.sample(external, external_needed)
    rng.shuffle(chosen)
    return chosen


def write_pilot(
    output: Path,
    candidates: list[dict],
    rejected: Counter,
    pilot_records: int,
    gate_records: int,
    seed: int,
    mix_with: Path | None = None,
    external_fraction: float = 0.2,
    mixed_train_records: int | None = None,
    mixed_validation_records: int | None = None,
) -> dict[str, int]:
    if pilot_records < 10 or gate_records < 1:
        raise ValueError("pilot_records must be at least 10 and gate_records must be positive")
    output.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    gate = balanced_take(candidates, gate_records, seed, used)
    used.update(record["schema_id"] for record in gate)
    test_count = max(1, round(pilot_records * 0.1))
    validation_count = max(1, round(pilot_records * 0.1))
    train_count = pilot_records - test_count - validation_count
    test = balanced_take(candidates, test_count, seed + 1, used)
    used.update(record["schema_id"] for record in test)
    validation = balanced_take(candidates, validation_count, seed + 2, used)
    used.update(record["schema_id"] for record in validation)
    train = balanced_take(candidates, train_count, seed + 3, used)
    splits = {"train": train, "validation": validation, "test": test, "external_gate": gate}
    for name, records in splits.items():
        _write_jsonl(output / f"{name}.jsonl", records)

    report = dataset_quality_report(splits)
    report.update(
        {
            "source": {
                "dataset": DATASET_ID,
                "revision": DATASET_REVISION,
                "license": DATASET_LICENSE,
                "checksums": {split: checksum for split, (_, checksum) in SOURCE_FILES.items()},
                "official_split_used": False,
                "resplit_key": "normalized_schema_sha256",
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
                "accepted_before_balancing": len(candidates),
                "accepted_by_family": dict(
                    sorted(Counter(record["query_plan"]["family"] for record in candidates).items())
                ),
                "rejected": dict(sorted(rejected.items())),
                "requirements": [
                    "read_only_current_query_plan_shape",
                    "safe_sqlite_context",
                    "nonempty_original_result",
                    "normalized_execution_equivalence",
                    "training_inference_grounding_equivalence",
                    "configured_sequence_limits",
                ],
            },
            "domains": len({record["source"]["domain"] for records in splits.values() for record in records}),
        }
    )

    if mix_with is not None:
        mixed_train = mix_records(
            _load_jsonl(mix_with / "train.jsonl"),
            train,
            external_fraction,
            seed + 4,
            mixed_train_records,
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
    parser.add_argument("--cache", type=Path, default=Path(".tmp/gretel"))
    parser.add_argument("--train-parquet", type=Path)
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument("--pilot-records", type=int, default=5000)
    parser.add_argument("--gate-records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9191)
    parser.add_argument(
        "--compatibility-config",
        type=Path,
        action="append",
        default=[],
        help="Reject records that do not fit this training config; may be repeated.",
    )
    parser.add_argument("--mix-with", type=Path, help="Reference dataset directory used to create mixed splits.")
    parser.add_argument("--external-fraction", type=float, default=0.2)
    parser.add_argument("--mixed-train-records", type=int)
    parser.add_argument("--mixed-validation-records", type=int)
    args = parser.parse_args()
    if bool(args.train_parquet) != bool(args.test_parquet):
        raise SystemExit("Provide both --train-parquet and --test-parquet, or neither.")
    paths = (
        {"train": args.train_parquet, "test": args.test_parquet}
        if args.train_parquet
        else download_sources(args.cache)
    )
    configs = args.compatibility_config or [Path("configs/base_position_robust_v8.yaml")]
    candidates, rejected = curate_rows(parquet_rows(paths), configs)
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
    print(json.dumps({"splits": result, "accepted_before_balancing": len(candidates)}, sort_keys=True))


if __name__ == "__main__":
    main()
