import json
from pathlib import Path

import pytest

from pocketsql.data.filter_replay_v18 import build_filter_replay_mixture


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _record(identifier: str, source: str) -> dict:
    return {
        "id": identifier,
        "schema_sql": f"CREATE TABLE {identifier} (id INTEGER);",
        "question": f"show {identifier}",
        "query_plan": {"family": "select"},
        "v17_source": source,
    }


def test_filter_replay_mixture_preserves_all_contrast_and_samples_replay(tmp_path: Path):
    contrast_path = tmp_path / "contrast.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    _write(contrast_path, [_record("contrast_0", "contrast"), _record("contrast_1", "contrast")])
    _write(
        replay_path,
        [_record(f"replay_{index}", "ordinary") for index in range(6)],
    )
    output = tmp_path / "mixture"

    report = build_filter_replay_mixture(
        output, contrast_path, replay_path, replay_records=4, seed=7
    )

    records = [json.loads(line) for line in (output / "train.jsonl").open()]
    assert report["records"] == {"contrast": 2, "ordinary_replay": 4, "total": 6}
    assert sum(record["v18_replay_kind"] == "contrast" for record in records) == 2
    assert sum(record["v18_replay_kind"] == "ordinary" for record in records) == 4


def test_filter_replay_mixture_rejects_oversampling(tmp_path: Path):
    contrast_path = tmp_path / "contrast.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    _write(contrast_path, [_record("contrast", "contrast")])
    _write(replay_path, [_record("replay", "ordinary")])

    with pytest.raises(ValueError, match="replay_records"):
        build_filter_replay_mixture(
            tmp_path / "mixture", contrast_path, replay_path, replay_records=2
        )
