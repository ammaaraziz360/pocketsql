"""Build an evaluation-only benchmark using conversational, held-out phrasing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .challenge import CHALLENGE_FAMILY_WEIGHTS
from .generate import build_records, dataset_quality_report


def write_casual_dev_dataset(
    output: Path,
    schemas: int = 120,
    examples_per_schema: int = 50,
    seed: int = 271828,
) -> dict[str, int]:
    """Write conversational questions whose template family is absent from training."""
    output.mkdir(parents=True, exist_ok=True)
    generation_stats: dict = {}
    splits = build_records(
        schemas,
        examples_per_schema,
        seed,
        CHALLENGE_FAMILY_WEIGHTS,
        schema_prefix="casual",
        generation_stats=generation_stats,
        question_style="heldout",
    )
    records = [record for split in splits.values() for record in split]
    records.sort(key=lambda record: record["id"])
    with (output / "casual.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    report = dataset_quality_report({"casual": records})
    report.update(
        {
            "profile": "held_out_casual_language_dev",
            "training_use_allowed": False,
            "generation": generation_stats,
        }
    )
    (output / "quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"casual": len(records), "schemas": len({record["schema_id"] for record in records})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schemas", type=int, default=120)
    parser.add_argument("--examples-per-schema", type=int, default=50)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()
    print(write_casual_dev_dataset(args.output, args.schemas, args.examples_per_schema, args.seed))


if __name__ == "__main__":
    main()
