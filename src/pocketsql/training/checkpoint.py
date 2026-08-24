from __future__ import annotations

import json
from pathlib import Path

def save_checkpoint(path: Path, model, metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path / "weights.safetensors"))
    (path / "metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")


def load_checkpoint(path: Path, model) -> dict:
    model.load_weights(str(path / "weights.safetensors"), strict=True)
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))