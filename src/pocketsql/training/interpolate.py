"""Interpolate two compatible PocketSQL checkpoints for inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import mlx.core as mx


COMPATIBILITY_KEYS = (
    "layers",
    "hidden_dim",
    "heads",
    "ffn_dim",
    "context_length",
    "canonicalize_identifiers",
    "identifier_slot_strategy",
    "canonicalize_literals",
    "target_format",
)


def _metadata(path: Path) -> dict:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def interpolate_checkpoints(base: Path, fine_tuned: Path, output: Path, fine_tuned_weight: float) -> None:
    """Write ``(1 - weight) * base + weight * fine_tuned`` to a new directory."""
    if not 0.0 <= fine_tuned_weight <= 1.0:
        raise ValueError("fine_tuned_weight must be between zero and one")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    base_metadata = _metadata(base)
    fine_tuned_metadata = _metadata(fine_tuned)
    base_config = base_metadata["config"]
    fine_tuned_config = fine_tuned_metadata["config"]
    mismatches = [key for key in COMPATIBILITY_KEYS if base_config.get(key) != fine_tuned_config.get(key)]
    if mismatches:
        raise ValueError(f"Checkpoint configs differ for: {', '.join(mismatches)}")
    if (base / "tokenizer.json").read_bytes() != (fine_tuned / "tokenizer.json").read_bytes():
        raise ValueError("Checkpoint tokenizers differ")

    base_weights = mx.load(str(base / "weights.safetensors"))
    fine_tuned_weights = mx.load(str(fine_tuned / "weights.safetensors"))
    if base_weights.keys() != fine_tuned_weights.keys():
        raise ValueError("Checkpoint parameter names differ")
    for name in base_weights:
        if base_weights[name].shape != fine_tuned_weights[name].shape:
            raise ValueError(f"Checkpoint parameter shapes differ for {name}")

    interpolated = {
        name: (1.0 - fine_tuned_weight) * base_weights[name] + fine_tuned_weight * fine_tuned_weights[name]
        for name in base_weights
    }
    mx.eval(*interpolated.values())
    output.mkdir(parents=True)
    mx.save_safetensors(str(output / "weights.safetensors"), interpolated)
    shutil.copy2(fine_tuned / "tokenizer.json", output / "tokenizer.json")
    metadata = {
        "config": fine_tuned_config,
        "epoch": fine_tuned_metadata.get("epoch"),
        "train_loss": None,
        "val_loss": None,
        "resumable": False,
        "weight_interpolation": {
            "base": str(base),
            "fine_tuned": str(fine_tuned),
            "fine_tuned_weight": fine_tuned_weight,
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fine-tuned", type=Path, required=True)
    parser.add_argument("--fine-tuned-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    interpolate_checkpoints(args.base, args.fine_tuned, args.output, args.fine_tuned_weight)
    print(json.dumps({"output": str(args.output), "fine_tuned_weight": args.fine_tuned_weight}))


if __name__ == "__main__":
    main()
