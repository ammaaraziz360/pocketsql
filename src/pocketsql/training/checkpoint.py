from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def save_checkpoint(path: Path, model, metadata: dict, optimizer=None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path / "weights.safetensors"))
    if optimizer is not None:
        mx.save_safetensors(str(path / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    (path / "metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")


def load_checkpoint(path: Path, model, optimizer=None) -> dict:
    model.load_weights(str(path / "weights.safetensors"), strict=True)
    optimizer_path = path / "optimizer.safetensors"
    if optimizer is not None and optimizer_path.exists():
        optimizer.state = tree_unflatten(list(mx.load(str(optimizer_path)).items()))
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))