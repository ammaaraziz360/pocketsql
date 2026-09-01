"""Mix V18 contrastive filter examples with ordinary schema-role replay."""
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


def build_filter_replay_mixture(
    output: Path,
    contrast_path: Path,
    replay_path: Path,
    replay_records: int = 12600,
    seed: int = 181819,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite filter replay mixture: {output}")
    contrast = _load(contrast_path)
    replay = _load(replay_path)
    if replay_records < 1 or replay_records > len(replay):
        raise ValueError(
            f"replay_records must be between 1 and {len(replay)}, got {replay_records}"
        )
    if not contrast:
        raise ValueError("contrast dataset is empty")

    rng = random.Random(seed)
    selected_replay = rng.sample(replay, replay_records)
    mixed = [
        {**record, "id": f"v18_contrast:{record.get('id', index)}", "v18_replay_kind": "contrast"}
        for index, record in enumerate(contrast)
    ]
    mixed.extend(
        {
            **record,
            "id": f"v18_replay:{record.get('id', index)}",
            "v18_replay_kind": "ordinary",
        }
        for index, record in enumerate(selected_replay)
    )
    rng.shuffle(mixed)

    duplicate_pairs = len(mixed) - len(
        {(record["schema_sql"], record["question"].strip().casefold()) for record in mixed}
    )
    report = {
        "profile": "v18_filter_contrast_with_replay",
        "seed": seed,
        "records": {
            "contrast": len(contrast),
            "ordinary_replay": len(selected_replay),
            "total": len(mixed),
        },
        "ratio": {
            "contrast": len(contrast) / len(mixed),
            "ordinary_replay": len(selected_replay) / len(mixed),
        },
        "ordinary_sources": dict(
            sorted(Counter(record.get("v17_source", "unknown") for record in selected_replay).items())
        ),
        "ordinary_families": dict(
            sorted(Counter(record["query_plan"]["family"] for record in selected_replay).items())
        ),
        "duplicate_schema_question_pairs": duplicate_pairs,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary_name:
        temporary = Path(temporary_name)
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
        "--contrast",
        type=Path,
        default=Path("data/filter-linking-v18/train.jsonl"),
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("data/semantic-expansion-v17-mixture/mixed_train.jsonl"),
    )
    parser.add_argument("--replay-records", type=int, default=12600)
    parser.add_argument("--seed", type=int, default=181819)
    args = parser.parse_args()
    print(
        json.dumps(
            build_filter_replay_mixture(
                args.output,
                args.contrast,
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
