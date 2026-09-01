"""Mix V19 compositional examples with ordinary V17 replay supervision."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def build_composition_replay_mixture(
    output: Path,
    composition_path: Path,
    replay_path: Path,
    replay_records: int = 28800,
    seed: int = 191920,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V19 replay mixture: {output}")
    composition = _load(composition_path)
    replay = _load(replay_path)
    if not composition:
        raise ValueError("composition dataset is empty")
    compatible_replay = []
    canonical_sql_by_pair: dict[tuple[str, str], str] = {}
    excluded_conflicts = 0
    for record in replay:
        key = (record["schema_sql"], record["question"].strip().casefold())
        sql = record.get("sql", "")
        if key in canonical_sql_by_pair and canonical_sql_by_pair[key] != sql:
            excluded_conflicts += 1
            continue
        canonical_sql_by_pair.setdefault(key, sql)
        compatible_replay.append(record)
    if replay_records < 1 or replay_records > len(compatible_replay):
        raise ValueError(
            f"replay_records must be between 1 and {len(compatible_replay)}, got {replay_records}"
        )

    rng = random.Random(seed)
    selected_replay = rng.sample(compatible_replay, replay_records)
    mixed = [
        {
            **record,
            "id": f"v19_composition:{record.get('id', index)}",
            "v19_replay_kind": "composition",
        }
        for index, record in enumerate(composition)
    ]
    mixed.extend(
        {
            **record,
            "id": f"v19_ordinary:{record.get('id', index)}",
            "v19_replay_kind": "ordinary",
        }
        for index, record in enumerate(selected_replay)
    )
    rng.shuffle(mixed)

    pair_counts = Counter(
        (record["schema_sql"], record["question"].strip().casefold()) for record in mixed
    )
    conflicting_pairs = 0
    sql_by_pair: dict[tuple[str, str], set[str]] = {}
    for record in mixed:
        key = (record["schema_sql"], record["question"].strip().casefold())
        sql_by_pair.setdefault(key, set()).add(record.get("sql", ""))
    conflicting_pairs = sum(len(values) > 1 for values in sql_by_pair.values())
    report = {
        "profile": "composition_expansion_v19_with_replay",
        "seed": seed,
        "records": {
            "composition": len(composition),
            "ordinary_replay": len(selected_replay),
            "total": len(mixed),
        },
        "ratio": {
            "composition": len(composition) / len(mixed),
            "ordinary_replay": len(selected_replay) / len(mixed),
        },
        "ordinary_sources": dict(
            sorted(
                Counter(record.get("v17_source", "unknown") for record in selected_replay).items()
            )
        ),
        "ordinary_families": dict(
            sorted(Counter(record["query_plan"]["family"] for record in selected_replay).items())
        ),
        "duplicate_schema_question_pairs": sum(count - 1 for count in pair_counts.values()),
        "conflicting_schema_question_pairs": conflicting_pairs,
        "excluded_conflicting_replay_records": excluded_conflicts,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as name:
        temporary = Path(name)
        _write(temporary / "train.jsonl", mixed)
        report["sha256"] = _sha256(temporary / "train.jsonl")
        (temporary / "quality_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--composition",
        type=Path,
        default=Path("data/composition-expansion-v19/train.jsonl"),
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("data/semantic-expansion-v17-mixture/mixed_train.jsonl"),
    )
    parser.add_argument("--replay-records", type=int, default=28800)
    parser.add_argument("--seed", type=int, default=191920)
    args = parser.parse_args()
    print(
        json.dumps(
            build_composition_replay_mixture(
                args.output,
                args.composition,
                args.replay,
                args.replay_records,
                args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
