"""Build an evaluation-only corpus with random, semantically meaningless identifiers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .challenge import CHALLENGE_FAMILY_WEIGHTS, _load_reference_identifiers
from .generate import build_records, dataset_quality_report


def write_grounding_dev_dataset(
    output: Path,
    schemas: int = 120,
    examples_per_schema: int = 75,
    seed: int = 314159,
    reference_data: Path | None = None,
) -> dict[str, int]:
    """Write a stable opaque-name set that must never be included in training."""
    output.mkdir(parents=True, exist_ok=True)
    generation_stats: dict = {}
    generated_splits = build_records(
        schemas,
        examples_per_schema,
        seed,
        CHALLENGE_FAMILY_WEIGHTS,
        schema_prefix="opaque",
        generation_stats=generation_stats,
        identifier_mode="opaque",
    )
    records = [record for split in generated_splits.values() for record in split]
    records.sort(key=lambda record: record["id"])
    path = output / "opaque.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    report = dataset_quality_report(
        {"opaque": records},
        _load_reference_identifiers(reference_data),
    )
    report.update(
        {
            "profile": "opaque_identifier_grounding_dev",
            "reference_data": str(reference_data) if reference_data else None,
            "training_use_allowed": False,
            "generation": generation_stats,
        }
    )
    (output / "quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"opaque": len(records), "schemas": len({record["schema_id"] for record in records})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schemas", type=int, default=120)
    parser.add_argument("--examples-per-schema", type=int, default=75)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--reference-data", type=Path)
    args = parser.parse_args()
    print(
        write_grounding_dev_dataset(
            args.output,
            args.schemas,
            args.examples_per_schema,
            args.seed,
            args.reference_data,
        )
    )


if __name__ == "__main__":
    main()
