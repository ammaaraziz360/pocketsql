"""Build the human-heavy replay mixture for the v16 residual-linking experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from .schema_linking import linking_family, weighted_resample


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def _uniform_weights(records: list[dict]) -> dict[str, float]:
    return {linking_family(record): 1.0 for record in records}


def _sample(
    records: list[dict],
    count: int,
    seed: int,
    source: str,
    evaluation_track: str | None = None,
) -> list[dict]:
    if count > len(records):
        raise ValueError(f"{source} count {count} exceeds {len(records)} available records")
    rng = random.Random(seed)
    selected = rng.sample(records, count)
    return [
        {
            **record,
            "id": f"v16:{source}:{index}:{record.get('id', index)}",
            "v16_source": source,
            **({"evaluation_track": evaluation_track} if evaluation_track else {}),
        }
        for index, record in enumerate(selected)
    ]


def build_v16_mixture(
    output: Path,
    semantic_train_path: Path,
    semantic_validation_path: Path,
    human_train_path: Path,
    human_validation_path: Path,
    synthetic_train_path: Path,
    synthetic_validation_path: Path,
    frozen_human_path: Path,
    human_train_records: int = 6000,
    synthetic_train_records: int = 6800,
    human_validation_records: int = 320,
    synthetic_validation_records: int = 320,
    seed: int = 161616,
) -> dict:
    """Create a 20k train/1k validation mix without touching frozen anti data."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite v16 corpus directory: {output}")

    semantic_train = _load(semantic_train_path)
    semantic_validation = _load(semantic_validation_path)
    human_train = _load(human_train_path)
    human_validation = _load(human_validation_path)
    synthetic_train = _load(synthetic_train_path)
    synthetic_validation = _load(synthetic_validation_path)
    frozen_human = _load(frozen_human_path)

    if any(not record.get("source", {}).get("training_use_allowed") for record in human_train):
        raise ValueError("human replay contains a record not licensed for training use")

    train_db_ids = {record.get("source", {}).get("db_id") for record in human_train}
    validation_db_ids = {
        record.get("source", {}).get("db_id") for record in human_validation
    }
    frozen_db_ids = {record.get("source", {}).get("db_id") for record in frozen_human}
    if train_db_ids & validation_db_ids:
        raise ValueError("human train and validation schemas overlap")
    if (train_db_ids | validation_db_ids) & frozen_db_ids:
        raise ValueError("human replay overlaps the frozen human benchmark")

    selected_human_train, human_train_families = weighted_resample(
        human_train,
        human_train_records,
        _uniform_weights(human_train),
        seed,
        "v16_human",
    )
    selected_human_train = [
        {**record, "v16_source": "human_replay"} for record in selected_human_train
    ]
    selected_synthetic_train = _sample(
        synthetic_train,
        synthetic_train_records,
        seed + 1,
        "synthetic_replay",
    )
    selected_semantic_train = [
        {
            **record,
            "id": f"v16:semantic:{record.get('id')}",
            "v16_source": "paired_semantic",
        }
        for record in semantic_train
    ]

    selected_human_validation, human_validation_families = weighted_resample(
        human_validation,
        human_validation_records,
        _uniform_weights(human_validation),
        seed + 2,
        "v16_human_validation",
    )
    selected_human_validation = [
        {
            **record,
            "v16_source": "human_validation",
            "evaluation_track": "human_validation",
        }
        for record in selected_human_validation
    ]
    selected_synthetic_validation = _sample(
        synthetic_validation,
        synthetic_validation_records,
        seed + 3,
        "synthetic_validation",
        "synthetic_validation",
    )
    selected_semantic_validation = [
        {
            **record,
            "id": f"v16:semantic_validation:{record.get('id')}",
            "v16_source": "semantic_paraphrase_validation",
            "evaluation_track": "semantic_paraphrase_validation",
        }
        for record in semantic_validation
    ]

    rng = random.Random(seed + 4)
    train = [
        *selected_semantic_train,
        *selected_human_train,
        *selected_synthetic_train,
    ]
    validation = [
        *selected_semantic_validation,
        *selected_human_validation,
        *selected_synthetic_validation,
    ]
    rng.shuffle(train)
    rng.shuffle(validation)

    report = {
        "profile": "semantic_schema_linking_v16_human_replay",
        "seed": seed,
        "training_use_allowed": {"mixed_train": True, "mixed_validation": False},
        "inputs": {
            str(path): _sha256(path)
            for path in (
                semantic_train_path,
                semantic_validation_path,
                human_train_path,
                human_validation_path,
                synthetic_train_path,
                synthetic_validation_path,
                frozen_human_path,
            )
        },
        "isolation": {
            "human_train_validation_schema_overlap": len(train_db_ids & validation_db_ids),
            "human_train_frozen_schema_overlap": len(train_db_ids & frozen_db_ids),
            "human_validation_frozen_schema_overlap": len(validation_db_ids & frozen_db_ids),
            "anti_memorization_data_loaded": False,
        },
        "train": {
            "records": len(train),
            "source_counts": dict(sorted(Counter(record["v16_source"] for record in train).items())),
            "human_unique_records": len(human_train),
            "human_family_counts": human_train_families,
        },
        "validation": {
            "records": len(validation),
            "source_counts": dict(
                sorted(Counter(record["evaluation_track"] for record in validation).items())
            ),
            "human_unique_records": len(human_validation),
            "human_family_counts": human_validation_families,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as name:
        temporary = Path(name)
        databases = human_train_path.parent / "databases"
        if databases.exists():
            shutil.copytree(databases, temporary / "databases")
        _write(temporary / "mixed_train.jsonl", train)
        _write(temporary / "mixed_validation.jsonl", validation)
        (temporary / "quality_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--semantic-train",
        type=Path,
        default=Path("data/semantic-linking-v15/train.jsonl"),
    )
    parser.add_argument(
        "--semantic-validation",
        type=Path,
        default=Path("data/semantic-linking-v15/validation.jsonl"),
    )
    parser.add_argument(
        "--human-train",
        type=Path,
        default=Path("data/spider-human-train-v1/human_train.jsonl"),
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
        "--frozen-human",
        type=Path,
        default=Path("data/spider-human-alpha-v1/benchmark.jsonl"),
    )
    parser.add_argument("--human-train-records", type=int, default=6000)
    parser.add_argument("--synthetic-train-records", type=int, default=6800)
    parser.add_argument("--human-validation-records", type=int, default=320)
    parser.add_argument("--synthetic-validation-records", type=int, default=320)
    parser.add_argument("--seed", type=int, default=161616)
    args = parser.parse_args()
    report = build_v16_mixture(
        args.output,
        args.semantic_train,
        args.semantic_validation,
        args.human_train,
        args.human_validation,
        args.synthetic_train,
        args.synthetic_validation,
        args.frozen_human,
        args.human_train_records,
        args.synthetic_train_records,
        args.human_validation_records,
        args.synthetic_validation_records,
        args.seed,
    )
    print(json.dumps({"output": str(args.output), **report["train"], "validation": report["validation"]}))


if __name__ == "__main__":
    main()
