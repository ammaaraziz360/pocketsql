"""Build an identifier-held-out text-to-SQL challenge set.

The normal generator is for model training.  This module instead emits one
challenge JSONL file using business domains and identifiers that never occur in
the standard vocabulary.  It is intended for evaluation only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate import build_records, dataset_quality_report, schema_identifiers
from .schemas import CHALLENGE_VOCAB


CHALLENGE_FAMILY_WEIGHTS = {"join": 4, "and_filter": 3, "or_filter": 3}


def _load_reference_identifiers(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return set().union(*(schema_identifiers(record["schema_sql"]) for record in records))


def write_challenge_dataset(
    output: Path,
    schemas: int = 120,
    examples_per_schema: int = 75,
    seed: int = 2026,
    reference_data: Path | None = None,
) -> dict[str, int]:
    """Write a schema-disjoint, lexically held-out evaluation corpus."""
    output.mkdir(parents=True, exist_ok=True)
    generation_stats: dict = {}
    generated_splits = build_records(
        schemas,
        examples_per_schema,
        seed,
        CHALLENGE_FAMILY_WEIGHTS,
        CHALLENGE_VOCAB,
        "challenge",
        generation_stats,
    )
    records = [record for split in generated_splits.values() for record in split]
    records.sort(key=lambda record: record["id"])
    with (output / "challenge.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    reference_identifiers = _load_reference_identifiers(reference_data)
    report = dataset_quality_report({"challenge": records}, reference_identifiers)
    report.update(
        {
            "profile": "identifier_held_out_challenge",
            "reference_data": str(reference_data) if reference_data else None,
            "lexically_isolated_from_standard_vocabulary": True,
            "generation": generation_stats,
        }
    )
    (output / "quality_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"challenge": len(records), "schemas": len({record["schema_id"] for record in records})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schemas", type=int, default=120)
    parser.add_argument("--examples-per-schema", type=int, default=75)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--reference-data",
        type=Path,
        help="Training JSONL used to quantify held-out identifier coverage in quality_report.json.",
    )
    args = parser.parse_args()
    print(write_challenge_dataset(args.output, args.schemas, args.examples_per_schema, args.seed, args.reference_data))


if __name__ == "__main__":
    main()
