from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def save_checkpoint(path: Path, model, metadata: dict, optimizer=None, tokenizer=None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path / "weights.safetensors"))
    if optimizer is not None:
        mx.save_safetensors(str(path / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    if tokenizer is not None:
        tokenizer.save(path / "tokenizer.json")
    (path / "metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")


def load_checkpoint(path: Path, model, optimizer=None) -> dict:
    model.load_weights(str(path / "weights.safetensors"), strict=True)
    optimizer_path = path / "optimizer.safetensors"
    if optimizer is not None and optimizer_path.exists():
        optimizer.state = tree_unflatten(list(mx.load(str(optimizer_path)).items()))
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def initialize_model(path: Path, model) -> dict:
    """Load compatible weights while allowing only position-table resizing."""
    loaded = dict(mx.load(str(path / "weights.safetensors")))
    expected = dict(tree_flatten(model.parameters()))
    if loaded.keys() != expected.keys():
        missing = sorted(expected.keys() - loaded.keys())
        extra = sorted(loaded.keys() - expected.keys())
        raise ValueError(f"Initialization parameter names differ; missing={missing}, extra={extra}")
    for name in loaded:
        if loaded[name].shape == expected[name].shape:
            continue
        if name != "position.weight" or loaded[name].ndim != 2 or expected[name].ndim != 2:
            raise ValueError(
                f"Initialization shape differs for {name}: expected {expected[name].shape}, got {loaded[name].shape}"
            )
        if loaded[name].shape[1] != expected[name].shape[1]:
            raise ValueError(
                f"Position width differs: expected {expected[name].shape[1]}, got {loaded[name].shape[1]}"
            )
        shared = min(loaded[name].shape[0], expected[name].shape[0])
        if expected[name].shape[0] > shared:
            loaded[name] = mx.concatenate((loaded[name][:shared], expected[name][shared:]), axis=0)
        else:
            loaded[name] = loaded[name][:shared]
    model.load_weights(list(loaded.items()), strict=True)
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))
