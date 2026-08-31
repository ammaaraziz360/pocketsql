"""Build a hard-example curriculum for explicit schema linking."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile


LINKING_WEIGHTS = {
    "replay": 0.75,
    "filter": 3.0,
    "group": 2.0,
    "join": 3.0,
    "joined_aggregate": 4.0,
}
SYNTHETIC_WEIGHTS = {
    "replay": 1.0,
    "filter": 1.5,
    "group": 2.0,
    "join": 1.5,
    "joined_aggregate": 2.0,
}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linking_family(record: dict) -> str:
    """Classify by the schema decision that the example supervises."""
    plan = record["query_plan"]
    if plan.get("join_table") and plan.get("aggregate"):
        return "joined_aggregate"
    if plan.get("join_table"):
        return "join"
    if plan.get("group_by"):
        return "group"
    if plan.get("filters"):
        return "filter"
    return "replay"


def _weighted_quotas(records: list[dict], total: int, weights: dict[str, float]) -> dict[str, int]:
    counts = Counter(linking_family(record) for record in records)
    missing = sorted(family for family in counts if weights.get(family, 0) <= 0)
    if missing:
        raise ValueError(f"positive weights are required for present families: {missing}")
    weighted = {family: count * weights[family] for family, count in counts.items()}
    denominator = sum(weighted.values())
    if not records or total < 1 or denominator <= 0:
        raise ValueError("records and total must be positive")
    raw = {family: total * value / denominator for family, value in weighted.items()}
    quotas = {family: math.floor(value) for family, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(raw, key=lambda family: (raw[family] - quotas[family], family), reverse=True)
    for family in order[:remaining]:
        quotas[family] += 1
    return quotas


def weighted_resample(
    records: list[dict],
    total: int,
    weights: dict[str, float],
    seed: int,
    source: str,
) -> tuple[list[dict], dict[str, int]]:
    """Cycle each family deterministically according to frequency-adjusted weights."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for record in records:
        buckets.setdefault(linking_family(record), []).append(record)
    quotas = _weighted_quotas(records, total, weights)
    selected: list[dict] = []
    for family in sorted(quotas):
        candidates = buckets[family]
        count = quotas[family]
        offset = 0
        cycle = 0
        while offset < count:
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            for record in shuffled[: count - offset]:
                selected.append(
                    {
                        **record,
                        "id": f"{source}:{family}:{cycle}:{record.get('id', offset)}",
                        "schema_id": f"{source}:{record.get('schema_id', 'unknown')}",
                        "semantic_source": source,
                        "schema_linking_family": family,
                    }
                )
                offset += 1
            cycle += 1
    rng.shuffle(selected)
    return selected, quotas


def build_schema_linking_curriculum(
    output: Path,
    human_train_path: Path,
    human_validation_path: Path,
    synthetic_train_path: Path,
    synthetic_validation_path: Path,
    frozen_benchmark_path: Path,
    train_records: int = 24000,
    validation_records: int = 640,
    human_fraction: float = 0.5,
    seed: int = 20260902,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite schema-linking corpus directory: {output}")
    if not 0.0 < human_fraction < 1.0:
        raise ValueError("human_fraction must be between zero and one")
    human_train = _load_jsonl(human_train_path)
    human_validation = _load_jsonl(human_validation_path)
    synthetic_train = _load_jsonl(synthetic_train_path)
    synthetic_validation = _load_jsonl(synthetic_validation_path)
    frozen = _load_jsonl(frozen_benchmark_path)
    if any(not record.get("source", {}).get("training_use_allowed") for record in human_train):
        raise ValueError("human training input contains a record not licensed for training use")

    frozen_db_ids = {record.get("source", {}).get("db_id") for record in frozen}
    train_db_ids = {record.get("source", {}).get("db_id") for record in human_train}
    validation_db_ids = {record.get("source", {}).get("db_id") for record in human_validation}
    if train_db_ids & validation_db_ids or (train_db_ids | validation_db_ids) & frozen_db_ids:
        raise ValueError("human train, validation, and frozen benchmark schemas must remain disjoint")

    human_train_count = round(train_records * human_fraction)
    synthetic_train_count = train_records - human_train_count
    selected_human_train, human_train_quotas = weighted_resample(
        human_train, human_train_count, LINKING_WEIGHTS, seed, "spider_link"
    )
    selected_synthetic_train, synthetic_train_quotas = weighted_resample(
        synthetic_train, synthetic_train_count, SYNTHETIC_WEIGHTS, seed + 1, "v11_link"
    )
    human_validation_count = round(validation_records * human_fraction)
    synthetic_validation_count = validation_records - human_validation_count
    neutral_weights = {family: 1.0 for family in LINKING_WEIGHTS}
    selected_human_validation, human_validation_quotas = weighted_resample(
        human_validation, human_validation_count, neutral_weights, seed + 2, "spider_dev"
    )
    selected_synthetic_validation, synthetic_validation_quotas = weighted_resample(
        synthetic_validation,
        synthetic_validation_count,
        neutral_weights,
        seed + 3,
        "v11_validation",
    )
    rng = random.Random(seed + 4)
    mixed_train = selected_human_train + selected_synthetic_train
    mixed_validation = selected_human_validation + selected_synthetic_validation
    rng.shuffle(mixed_train)
    rng.shuffle(mixed_validation)

    report = {
        "contract": "pocketsql-schema-linking-v1",
        "seed": seed,
        "inputs": {
            str(path): _sha256(path)
            for path in (
                human_train_path,
                human_validation_path,
                synthetic_train_path,
                synthetic_validation_path,
                frozen_benchmark_path,
            )
        },
        "isolation": {
            "human_train_validation_schema_overlap": len(train_db_ids & validation_db_ids),
            "human_training_benchmark_schema_overlap": len(train_db_ids & frozen_db_ids),
            "human_validation_benchmark_schema_overlap": len(validation_db_ids & frozen_db_ids),
        },
        "source_records": {
            "human_train": len(human_train),
            "human_validation": len(human_validation),
            "synthetic_train": len(synthetic_train),
            "synthetic_validation": len(synthetic_validation),
        },
        "train": {
            "records": len(mixed_train),
            "human_records": human_train_count,
            "synthetic_records": synthetic_train_count,
            "human_family_counts": human_train_quotas,
            "synthetic_family_counts": synthetic_train_quotas,
            "combined_family_counts": dict(sorted(Counter(linking_family(record) for record in mixed_train).items())),
            "human_weights": LINKING_WEIGHTS,
            "synthetic_weights": SYNTHETIC_WEIGHTS,
        },
        "validation": {
            "records": len(mixed_validation),
            "human_records": human_validation_count,
            "synthetic_records": synthetic_validation_count,
            "human_family_counts": human_validation_quotas,
            "synthetic_family_counts": synthetic_validation_quotas,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary_name:
        temporary = Path(temporary_name)
        source_databases = human_train_path.parent / "databases"
        if source_databases.exists():
            shutil.copytree(source_databases, temporary / "databases")
        _write_jsonl(temporary / "mixed_train.jsonl", mixed_train)
        _write_jsonl(temporary / "mixed_validation.jsonl", mixed_validation)
        _write_jsonl(temporary / "human_validation.jsonl", human_validation)
        (temporary / "quality_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--human-train", type=Path, default=Path("data/spider-human-train-v1/human_train.jsonl")
    )
    parser.add_argument(
        "--human-validation",
        type=Path,
        default=Path("data/spider-human-train-v1/human_validation.jsonl"),
    )
    parser.add_argument(
        "--synthetic-train",
        type=Path,
        default=Path("data/semantic-v11-composed/mixed_train.jsonl"),
    )
    parser.add_argument(
        "--synthetic-validation",
        type=Path,
        default=Path("data/semantic-v11-composed/mixed_validation.jsonl"),
    )
    parser.add_argument(
        "--frozen-benchmark",
        type=Path,
        default=Path("data/spider-human-alpha-v1/benchmark.jsonl"),
    )
    parser.add_argument("--train-records", type=int, default=24000)
    parser.add_argument("--validation-records", type=int, default=640)
    parser.add_argument("--human-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    report = build_schema_linking_curriculum(
        args.output,
        args.human_train,
        args.human_validation,
        args.synthetic_train,
        args.synthetic_validation,
        args.frozen_benchmark,
        args.train_records,
        args.validation_records,
        args.human_fraction,
        args.seed,
    )
    print(json.dumps({"output": str(args.output), **report["train"], "validation": report["validation"]}))


if __name__ == "__main__":
    main()
