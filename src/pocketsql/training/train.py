from __future__ import annotations

import argparse
from pathlib import Path
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import yaml

from pocketsql.model.tokenizer import ByteTokenizer
from pocketsql.model.transformer import ModelConfig, PocketSQLTransformer
from pocketsql.training.checkpoint import save_checkpoint
from pocketsql.training.dataset import load_jsonl, make_batch


def masked_loss(model: PocketSQLTransformer, tokens: mx.array, mask: mx.array) -> mx.array:
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    active = mask[:, 1:]
    safe_targets = mx.where(active, targets, mx.zeros_like(targets))
    losses = nn.losses.cross_entropy(logits, safe_targets)
    return mx.sum(losses * active) / mx.maximum(mx.sum(active), 1)


def train_step(model, optimizer, tokens, mask) -> float:
    loss_and_grad = nn.value_and_grad(model, masked_loss)
    loss, gradients = loss_and_grad(model, tokens, mask)
    gradients = tree_map(lambda value: mx.clip(value, -1.0, 1.0), gradients)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state)
    return float(loss.item())


def train(records: list[dict], config: dict, checkpoint_dir: Path | None = None) -> tuple[PocketSQLTransformer, list[float]]:
    random.seed(config["seed"])
    mx.random.seed(config["seed"])
    tokenizer = ByteTokenizer()
    model = PocketSQLTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, layers=config["layers"], hidden_dim=config["hidden_dim"], heads=config["heads"], ffn_dim=config["ffn_dim"], context_length=config["context_length"]))
    optimizer = optim.AdamW(learning_rate=config["learning_rate"])
    losses: list[float] = []
    for _ in range(config.get("epochs", 1)):
        for offset in range(0, len(records), config["batch_size"]):
            batch = records[offset : offset + config["batch_size"]]
            tokens, mask = make_batch(batch, tokenizer, config["context_length"])
            losses.append(train_step(model, optimizer, mx.array(tokens), mx.array(mask)))
    if checkpoint_dir:
        save_checkpoint(checkpoint_dir, model, {"config": config, "steps": len(losses)})
    return model, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest"))
    parser.add_argument("--overfit", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    records = load_jsonl(args.data)
    if args.overfit:
        records = records[: args.overfit]
    _, losses = train(records, config, args.checkpoint)
    print({"steps": len(losses), "final_loss": losses[-1]})


if __name__ == "__main__":
    main()