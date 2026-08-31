"""Curate Spider human training data without touching the frozen test benchmark."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Iterable

from pocketsql.data.spider import (
    ARCHIVE_SHA256,
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    SpiderRejected,
    _load_configs,
    _sha256_file,
    convert_example,
    download_archive,
    read_source_split,
    schema_sql_from_metadata,
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def curate_human_split(
    rows: list[dict],
    table_by_id: dict[str, dict],
    database_paths: dict[str, Path],
    split: str,
    config_paths: list[Path],
    forbidden_db_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict], Counter]:
    """Keep all supported rows from one official split and log every rejection."""
    forbidden_db_ids = forbidden_db_ids or set()
    configs = _load_configs(config_paths)
    schema_errors = {}
    for db_id in sorted({row["db_id"] for row in rows}):
        if db_id in forbidden_db_ids:
            schema_errors[db_id] = "frozen_benchmark_schema_overlap"
            continue
        try:
            schema_sql_from_metadata(table_by_id[db_id])
        except (KeyError, SpiderRejected) as exc:
            schema_errors[db_id] = "missing_schema_metadata" if isinstance(exc, KeyError) else str(exc)

    accepted = []
    rejections = []
    counts: Counter = Counter()
    seen = set()
    for index, source in enumerate(rows):
        reason = schema_errors.get(source["db_id"])
        try:
            if reason:
                raise SpiderRejected(reason)
            record = convert_example(
                source,
                index,
                table_by_id[source["db_id"]],
                database_paths[source["db_id"]],
                split,
                configs,
                training_use_allowed=split == "train",
            )
            key = (record["schema_id"], record["question"].casefold(), record["sql"].casefold())
            if key in seen:
                raise SpiderRejected("duplicate_schema_question_sql")
            seen.add(key)
            accepted.append(record)
        except (KeyError, SpiderRejected) as exc:
            reason = "missing_schema_metadata" if isinstance(exc, KeyError) else str(exc)
            counts[reason] += 1
            rejections.append(
                {
                    "id": f"spider_{split}_{index:04d}",
                    "db_id": source.get("db_id"),
                    "question": source.get("question"),
                    "original_sql": source.get("query"),
                    "reason": reason,
                    "split": split,
                }
            )
    return accepted, rejections, counts


def _sample(records: list[dict], count: int, rng: random.Random, source: str) -> list[dict]:
    """Deterministically cycle small sources so their curriculum weight is explicit."""
    if not records and count:
        raise ValueError(f"source {source!r} has no records")
    selected = []
    cycle = 0
    while len(selected) < count:
        candidates = list(records)
        rng.shuffle(candidates)
        for record in candidates[: count - len(selected)]:
            selected.append(
                {
                    **record,
                    "id": f"{source}:{cycle}:{record.get('id', len(selected))}",
                    "schema_id": f"{source}:{record.get('schema_id', 'unknown')}",
                    "semantic_source": source,
                }
            )
        cycle += 1
    return selected


def mix_human_curriculum(
    base: list[dict],
    human: list[dict],
    total: int,
    human_fraction: float,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    if total < 1 or not 0.0 < human_fraction < 1.0:
        raise ValueError("total must be positive and human_fraction must be between zero and one")
    human_count = round(total * human_fraction)
    base_count = total - human_count
    rng = random.Random(seed)
    mixed = _sample(base, base_count, rng, "v11") + _sample(human, human_count, rng, "spider_human")
    rng.shuffle(mixed)
    return mixed, {"v11": base_count, "spider_human": human_count}


def build_training_corpus(
    output: Path,
    train_rows: list[dict],
    train_tables: dict[str, dict],
    train_databases: dict[str, Path],
    dev_rows: list[dict],
    dev_tables: dict[str, dict],
    dev_databases: dict[str, Path],
    config_paths: list[Path],
    base_train_path: Path,
    base_validation_path: Path,
    forbidden_benchmark_path: Path,
    mixed_train_records: int,
    mixed_validation_records: int,
    human_train_fraction: float,
    human_validation_fraction: float,
    seed: int,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite training corpus directory: {output}")
    forbidden_records = _load_jsonl(forbidden_benchmark_path)
    forbidden_db_ids = {record["source"]["db_id"] for record in forbidden_records}
    human_train, train_rejections, train_counts = curate_human_split(
        train_rows, train_tables, train_databases, "train", config_paths, forbidden_db_ids
    )
    human_validation, validation_rejections, validation_counts = curate_human_split(
        dev_rows, dev_tables, dev_databases, "dev", config_paths, forbidden_db_ids
    )
    train_schemas = {record["source"]["db_id"] for record in human_train}
    validation_schemas = {record["source"]["db_id"] for record in human_validation}
    if train_schemas & validation_schemas or (train_schemas | validation_schemas) & forbidden_db_ids:
        raise RuntimeError("Spider train, validation, and frozen benchmark schemas must be disjoint")

    base_train = _load_jsonl(base_train_path)
    base_validation = _load_jsonl(base_validation_path)
    mixed_train, train_mix = mix_human_curriculum(
        base_train, human_train, mixed_train_records, human_train_fraction, seed
    )
    mixed_validation, validation_mix = mix_human_curriculum(
        base_validation,
        human_validation,
        mixed_validation_records,
        human_validation_fraction,
        seed + 1,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary_name:
        temporary = Path(temporary_name)
        databases_output = temporary / "databases"
        databases_output.mkdir()
        database_hashes = {}
        accepted_databases = {**train_databases, **dev_databases}
        for db_id in sorted(train_schemas | validation_schemas):
            destination = databases_output / f"{db_id}.sqlite"
            shutil.copyfile(accepted_databases[db_id], destination)
            database_hashes[db_id] = _sha256_file(destination)
        _write_jsonl(temporary / "human_train.jsonl", human_train)
        _write_jsonl(temporary / "human_validation.jsonl", human_validation)
        _write_jsonl(temporary / "mixed_train.jsonl", mixed_train)
        _write_jsonl(temporary / "mixed_validation.jsonl", mixed_validation)
        _write_jsonl(temporary / "rejections.jsonl", [*train_rejections, *validation_rejections])
        report = {
            "source": {
                "dataset": DATASET_ID,
                "revision": DATASET_REVISION,
                "license": DATASET_LICENSE,
                "archive_sha256": ARCHIVE_SHA256,
                "official_training_split": "train_spider + train_others",
                "official_validation_split": "dev",
            },
            "isolation": {
                "frozen_benchmark": str(forbidden_benchmark_path),
                "frozen_benchmark_sha256": _sha256_file(forbidden_benchmark_path),
                "frozen_benchmark_schemas": len(forbidden_db_ids),
                "train_validation_schema_overlap": 0,
                "training_benchmark_schema_overlap": 0,
                "benchmark_training_use_allowed": False,
            },
            "curation": {
                "train_source_records": len(train_rows),
                "train_accepted": len(human_train),
                "train_schemas": len(train_schemas),
                "train_rejected_by_reason": dict(sorted(train_counts.items())),
                "validation_source_records": len(dev_rows),
                "validation_accepted": len(human_validation),
                "validation_schemas": len(validation_schemas),
                "validation_rejected_by_reason": dict(sorted(validation_counts.items())),
                "accepted_train_by_family": dict(
                    sorted(Counter(record["query_plan"]["family"] for record in human_train).items())
                ),
                "accepted_validation_by_family": dict(
                    sorted(Counter(record["query_plan"]["family"] for record in human_validation).items())
                ),
            },
            "mix": {
                "seed": seed,
                "base_train": str(base_train_path),
                "base_train_sha256": _sha256_file(base_train_path),
                "base_validation": str(base_validation_path),
                "base_validation_sha256": _sha256_file(base_validation_path),
                "mixed_train_records": len(mixed_train),
                "mixed_train_counts": train_mix,
                "mixed_validation_records": len(mixed_validation),
                "mixed_validation_counts": validation_mix,
            },
            "database_sha256": database_hashes,
        }
        (temporary / "quality_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "ATTRIBUTION.md").write_text(
            "# Spider human training curriculum\n\n"
            "This training derivative uses only the official Spider 1.0 train split. The official dev "
            "split is used for validation. The separately frozen test benchmark is never used here.\n\n"
            "Spider was created by Tao Yu et al. and is licensed under "
            "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".tmp/spider"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--compatibility-config", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--base-train", type=Path, default=Path("data/semantic-v11-composed/mixed_train.jsonl")
    )
    parser.add_argument(
        "--base-validation", type=Path, default=Path("data/semantic-v11-composed/mixed_validation.jsonl")
    )
    parser.add_argument(
        "--frozen-benchmark",
        type=Path,
        default=Path("data/spider-human-alpha-v1/benchmark.jsonl"),
    )
    parser.add_argument("--mixed-train-records", type=int, default=20000)
    parser.add_argument("--mixed-validation-records", type=int, default=640)
    parser.add_argument("--human-train-fraction", type=float, default=0.30)
    parser.add_argument("--human-validation-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    archive_path = args.archive or download_archive(args.cache)
    config_paths = args.compatibility_config or [Path("configs/base_semantic_v11_composed.yaml")]
    train_rows, train_tables, train_databases = read_source_split(
        archive_path, "train", args.cache / "databases"
    )
    dev_rows, dev_tables, dev_databases = read_source_split(
        archive_path, "dev", args.cache / "databases"
    )
    report = build_training_corpus(
        args.output,
        train_rows,
        train_tables,
        train_databases,
        dev_rows,
        dev_tables,
        dev_databases,
        config_paths,
        args.base_train,
        args.base_validation,
        args.frozen_benchmark,
        args.mixed_train_records,
        args.mixed_validation_records,
        args.human_train_fraction,
        args.human_validation_fraction,
        args.seed,
    )
    print(json.dumps({"output": str(args.output), **report["curation"], **report["mix"]}, sort_keys=True))


if __name__ == "__main__":
    main()
