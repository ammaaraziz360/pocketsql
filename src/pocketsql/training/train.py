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
from pocketsql.training.checkpoint import load_checkpoint, save_checkpoint
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


def accumulated_train_step(model, optimizer, micro_batches: list[tuple[mx.array, mx.array]]) -> float:
    loss_and_grad = nn.value_and_grad(model, masked_loss)
    total_loss = 0.0
    accumulated = None
    for tokens, mask in micro_batches:
        loss, gradients = loss_and_grad(model, tokens, mask)
        total_loss += float(loss.item())
        accumulated = gradients if accumulated is None else tree_map(lambda a, b: a + b, accumulated, gradients)
    count = len(micro_batches)
    averaged = tree_map(lambda value: mx.clip(value / count, -1.0, 1.0), accumulated)
    optimizer.update(model, averaged)
    mx.eval(model.parameters(), optimizer.state)
    return total_loss / count


def evaluate_loss(model, records: list[dict], tokenizer: ByteTokenizer, config: dict) -> float:
    if not records:
        return float("nan")
    losses = []
    for offset in range(0, len(records), config["batch_size"]):
        batch = records[offset : offset + config["batch_size"]]
        tokens, mask = make_batch(batch, tokenizer, config["context_length"])
        losses.append(float(masked_loss(model, mx.array(tokens), mx.array(mask)).item()))
    return sum(losses) / len(losses)


def make_optimizer(config: dict) -> optim.AdamW:
    lr = config["learning_rate"]
    warmup_steps = config.get("warmup_steps", 0)
    if warmup_steps and warmup_steps > 0:
        schedule = optim.join_schedules(
            [optim.linear_schedule(0.0, lr, warmup_steps), lambda _step: mx.array(lr)],
            [warmup_steps],
        )
    else:
        schedule = lr
    return optim.AdamW(learning_rate=schedule)


def build_model(config: dict, vocab_size: int) -> PocketSQLTransformer:
    return PocketSQLTransformer(
        ModelConfig(
            vocab_size=vocab_size,
            layers=config["layers"],
            hidden_dim=config["hidden_dim"],
            heads=config["heads"],
            ffn_dim=config["ffn_dim"],
            context_length=config["context_length"],
        )
    )


def train(
    records: list[dict],
    config: dict,
    checkpoint_dir: Path | None = None,
    val_records: list[dict] | None = None,
    resume_from: Path | None = None,
) -> tuple[PocketSQLTransformer, list[float]]:
    random.seed(config["seed"])
    mx.random.seed(config["seed"])
    tokenizer = ByteTokenizer()
    model = build_model(config, tokenizer.vocab_size)
    optimizer = make_optimizer(config)
    start_epoch = 0
    if resume_from is not None:
        metadata = load_checkpoint(resume_from, model, optimizer)
        start_epoch = metadata.get("epoch", 0)
    grad_accum_steps = max(1, config.get("grad_accum_steps", 1))
    losses: list[float] = []
    for epoch in range(start_epoch, config.get("epochs", 1)):
        epoch_losses: list[float] = []
        for offset in range(0, len(records), config["batch_size"] * grad_accum_steps):
            micro_batches = []
            for micro_offset in range(offset, min(offset + config["batch_size"] * grad_accum_steps, len(records)), config["batch_size"]):
                batch = records[micro_offset : micro_offset + config["batch_size"]]
                tokens, mask = make_batch(batch, tokenizer, config["context_length"])
                micro_batches.append((mx.array(tokens), mx.array(mask)))
            if grad_accum_steps == 1:
                step_loss = train_step(model, optimizer, *micro_batches[0])
            else:
                step_loss = accumulated_train_step(model, optimizer, micro_batches)
            epoch_losses.append(step_loss)
        losses.extend(epoch_losses)
        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        val_loss = evaluate_loss(model, val_records, tokenizer, config) if val_records else float("nan")
        print({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if checkpoint_dir:
            save_checkpoint(checkpoint_dir, model, {"config": config, "epoch": epoch + 1, "steps": len(losses)}, optimizer)
    return model, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overfit", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    records = load_jsonl(args.data)
    if args.overfit:
        records = records[: args.overfit]
    val_records = load_jsonl(args.val_data) if args.val_data else None
    _, losses = train(records, config, args.checkpoint, val_records=val_records, resume_from=args.resume)
    print({"steps": len(losses), "final_loss": losses[-1]})


if __name__ == "__main__":
    main()