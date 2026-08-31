from __future__ import annotations

import argparse
from pathlib import Path
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import yaml

from pocketsql.model.tokenizer import TokenizerProtocol, load_tokenizer
from pocketsql.model.transformer import ModelConfig, PocketSQLTransformer
from pocketsql.evaluation.evaluate import evaluate_model, write_prediction_diagnostics
from pocketsql.training.checkpoint import initialize_model, load_checkpoint, save_checkpoint
from pocketsql.training.audit import audit_sequences, require_complete_sequences
from pocketsql.training.dataset import load_jsonl, make_batch
from pocketsql.training.tensorboard import TensorBoardLogger


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


def evaluate_loss(model, records: list[dict], tokenizer: TokenizerProtocol, config: dict) -> float:
    if not records:
        return float("nan")
    losses = []
    for offset in range(0, len(records), config["batch_size"]):
        batch = records[offset : offset + config["batch_size"]]
        tokens, mask = make_batch(
            batch,
            tokenizer,
            config["context_length"],
            config.get("canonicalize_identifiers", False),
            config.get("identifier_slot_strategy", "ordered"),
            config.get("canonicalize_literals", False),
            config.get("target_format", "sql"),
            config.get("schema_linking_hints", False),
        )
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
    model = PocketSQLTransformer(
        ModelConfig(
            vocab_size=vocab_size,
            layers=config["layers"],
            hidden_dim=config["hidden_dim"],
            heads=config["heads"],
            ffn_dim=config["ffn_dim"],
            context_length=config["context_length"],
        )
    )
    model.canonicalize_identifiers = config.get("canonicalize_identifiers", False)
    model.identifier_slot_strategy = config.get("identifier_slot_strategy", "ordered")
    model.canonicalize_literals = config.get("canonicalize_literals", False)
    model.generation_max_tokens = config.get("generation_max_tokens", 128)
    model.target_format = config.get("target_format", "sql")
    model.constrain_semantic_plan = config.get("constrain_semantic_plan", model.target_format == "semantic_plan")
    model.schema_linking_hints = config.get("schema_linking_hints", False)
    return model


def default_best_checkpoint_dir(checkpoint_dir: Path) -> Path:
    """Return the sibling directory that holds the lowest-validation-loss model."""
    return checkpoint_dir.with_name(f"{checkpoint_dir.name}-best")


def default_best_execution_checkpoint_dir(checkpoint_dir: Path) -> Path:
    """Return the sibling directory for the strongest execution-scored checkpoint."""
    return checkpoint_dir.with_name(f"{checkpoint_dir.name}-best-execution")


def default_tensorboard_log_dir(checkpoint_dir: Path) -> Path:
    """Keep TensorBoard event files outside checkpoint directories."""
    return Path("runs") / checkpoint_dir.name


def tokenizer_for_training(
    config: dict,
    resume_from: Path | None = None,
    initialize_from: Path | None = None,
) -> TokenizerProtocol:
    source = resume_from or initialize_from
    if source is not None and (source / "tokenizer.json").exists():
        return load_tokenizer(source)
    return load_tokenizer(config.get("tokenizer_path"))


def diagnostic_markdown(records: list[dict], predictions: list[str], limit: int = 8) -> str:
    def clean(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    lines = ["| Family | Question | Gold SQL | Predicted SQL |", "|---|---|---|---|"]
    for record, prediction in list(zip(records, predictions))[:limit]:
        family = record.get("query_plan", {}).get("family", "unknown")
        lines.append(f"| {clean(family)} | {clean(record['question'])} | `{clean(record['sql'])}` | `{clean(prediction or '<empty>')}` |")
    return "\n".join(lines)


def validation_subset(records: list[dict], max_examples: int) -> list[dict]:
    """Take a deterministic, schema- and family-stratified validation subset."""
    if max_examples <= 0 or len(records) <= max_examples:
        return records
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        key = (record.get("schema_id", "unknown"), record["query_plan"]["family"])
        groups.setdefault(key, []).append(record)
    selected: list[dict] = []
    offsets = {key: 0 for key in groups}
    schemas = sorted({schema_id for schema_id, _ in groups})
    families = sorted({family for _, family in groups})
    round_index = 0
    while len(selected) < max_examples:
        added = False
        for schema_index, schema_id in enumerate(schemas):
            for family_offset in range(len(families)):
                family = families[(round_index + schema_index + family_offset) % len(families)]
                key = (schema_id, family)
                offset = offsets.get(key, 0)
                if offset >= len(groups.get(key, ())):
                    continue
                selected.append(groups[key][offset])
                offsets[key] += 1
                added = True
                break
            if len(selected) == max_examples:
                break
        if not added:
            break
        round_index += 1
    return selected


def train(
    records: list[dict],
    config: dict,
    checkpoint_dir: Path | None = None,
    val_records: list[dict] | None = None,
    resume_from: Path | None = None,
    initialize_from: Path | None = None,
    best_checkpoint_dir: Path | None = None,
    best_execution_checkpoint_dir: Path | None = None,
    tensorboard_log_dir: Path | None = None,
) -> tuple[PocketSQLTransformer, list[float]]:
    random.seed(config["seed"])
    mx.random.seed(config["seed"])
    if resume_from is not None and initialize_from is not None:
        raise ValueError("resume_from and initialize_from are mutually exclusive")
    tokenizer = tokenizer_for_training(config, resume_from, initialize_from)
    generation_max_tokens = config.get("generation_max_tokens", 128)
    if generation_max_tokens < 1:
        raise ValueError("generation_max_tokens must be positive")
    if config.get("require_full_sequences", True):
        training_audit = audit_sequences(
            records,
            tokenizer,
            config["context_length"],
            generation_max_tokens,
            config.get("canonicalize_identifiers", False),
            config.get("identifier_slot_strategy", "ordered"),
            config.get("canonicalize_literals", False),
            config.get("target_format", "sql"),
            config.get("schema_linking_hints", False),
        )
        require_complete_sequences(training_audit, "training")
        print({"sequence_audit": "training", **training_audit})
        if val_records:
            validation_audit = audit_sequences(
                val_records,
                tokenizer,
                config["context_length"],
                generation_max_tokens,
                config.get("canonicalize_identifiers", False),
                config.get("identifier_slot_strategy", "ordered"),
                config.get("canonicalize_literals", False),
                config.get("target_format", "sql"),
                config.get("schema_linking_hints", False),
            )
            require_complete_sequences(validation_audit, "validation")
            print({"sequence_audit": "validation", **validation_audit})
    model = build_model(config, tokenizer.vocab_size)
    optimizer = make_optimizer(config)
    start_epoch = 0
    previous_metadata: dict = {}
    if resume_from is not None:
        previous_metadata = load_checkpoint(resume_from, model, optimizer)
        start_epoch = previous_metadata.get("epoch", 0)
    elif initialize_from is not None:
        initialize_model(initialize_from, model)
    if checkpoint_dir and val_records and best_checkpoint_dir is None:
        best_checkpoint_dir = default_best_checkpoint_dir(checkpoint_dir)
    if checkpoint_dir and val_records and best_execution_checkpoint_dir is None:
        best_execution_checkpoint_dir = default_best_execution_checkpoint_dir(checkpoint_dir)

    best_val_loss = previous_metadata.get("best_val_loss")
    if best_val_loss is None:
        best_val_loss = float("inf")
    best_epoch = previous_metadata.get("best_epoch")
    epochs_without_improvement = previous_metadata.get("epochs_without_improvement") or 0
    if resume_from is not None and best_checkpoint_dir and best_checkpoint_dir.exists() and best_val_loss == float("inf"):
        best_metadata = load_checkpoint(best_checkpoint_dir, build_model(config, tokenizer.vocab_size))
        best_val_loss = best_metadata.get("val_loss", best_metadata.get("best_val_loss", float("inf")))
        best_epoch = best_metadata.get("epoch")

    semantic_every = config.get("validation_execution_every", 0)
    if semantic_every < 0:
        raise ValueError("validation_execution_every must be non-negative")
    semantic_max_examples = config.get("validation_execution_max_examples", 0)
    if semantic_max_examples < 0:
        raise ValueError("validation_execution_max_examples must be non-negative")
    semantic_batch_size = config.get("validation_execution_batch_size", 1)
    if semantic_batch_size < 1:
        raise ValueError("validation_execution_batch_size must be positive")
    semantic_metric = config.get("validation_checkpoint_metric", "execution_accuracy")
    supported_semantic_metrics = {"syntactically_valid", "executable", "exact_match", "execution_accuracy"}
    if semantic_metric not in supported_semantic_metrics:
        raise ValueError(f"validation_checkpoint_metric must be one of {sorted(supported_semantic_metrics)}")
    best_semantic_score = previous_metadata.get("best_validation_score", float("-inf"))
    if best_semantic_score is None:
        best_semantic_score = float("-inf")
    best_semantic_epoch = previous_metadata.get("best_validation_epoch")
    if resume_from is not None and best_execution_checkpoint_dir and best_execution_checkpoint_dir.exists() and best_semantic_score == float("-inf"):
        execution_metadata = load_checkpoint(best_execution_checkpoint_dir, build_model(config, tokenizer.vocab_size))
        best_semantic_score = execution_metadata.get("validation_score", float("-inf"))
        best_semantic_epoch = execution_metadata.get("epoch")

    patience = config.get("early_stopping_patience")
    if patience is not None and patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    min_delta = config.get("early_stopping_min_delta", 0.0)
    if min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")

    grad_accum_steps = max(1, config.get("grad_accum_steps", 1))
    losses: list[float] = []
    tensorboard = TensorBoardLogger(tensorboard_log_dir) if tensorboard_log_dir else None
    initial_steps = previous_metadata.get("steps", 0)
    for epoch in range(start_epoch, config.get("epochs", 1)):
        epoch_losses: list[float] = []
        for offset in range(0, len(records), config["batch_size"] * grad_accum_steps):
            micro_batches = []
            for micro_offset in range(offset, min(offset + config["batch_size"] * grad_accum_steps, len(records)), config["batch_size"]):
                batch = records[micro_offset : micro_offset + config["batch_size"]]
                tokens, mask = make_batch(
                    batch,
                    tokenizer,
                    config["context_length"],
                    config.get("canonicalize_identifiers", False),
                    config.get("identifier_slot_strategy", "ordered"),
                    config.get("canonicalize_literals", False),
                    config.get("target_format", "sql"),
                    config.get("schema_linking_hints", False),
                )
                micro_batches.append((mx.array(tokens), mx.array(mask)))
            if grad_accum_steps == 1:
                step_loss = train_step(model, optimizer, *micro_batches[0])
            else:
                step_loss = accumulated_train_step(model, optimizer, micro_batches)
            epoch_losses.append(step_loss)
            if tensorboard:
                tensorboard.add_scalar("loss/train_step", step_loss, initial_steps + len(losses) + len(epoch_losses))
        losses.extend(epoch_losses)
        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        val_loss = evaluate_loss(model, val_records, tokenizer, config) if val_records else float("nan")
        if tensorboard:
            tensorboard.add_scalar("loss/train_epoch", train_loss, epoch + 1)
            if val_records:
                tensorboard.add_scalar("loss/validation", val_loss, epoch + 1)
        improved = bool(val_records) and val_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if best_checkpoint_dir:
                save_checkpoint(
                    best_checkpoint_dir,
                    model,
                    {
                        "config": config,
                        "epoch": epoch + 1,
                        "steps": previous_metadata.get("steps", 0) + len(losses),
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                    },
                    optimizer,
                    tokenizer,
                )
        elif val_records:
            epochs_without_improvement += 1

        semantic_result = None
        if val_records and semantic_every and (epoch + 1) % semantic_every == 0:
            semantic_records = validation_subset(val_records, semantic_max_examples)
            semantic_evaluation = evaluate_model(
                model,
                semantic_records,
                tokenizer,
                batch_size=semantic_batch_size,
                max_tokens=generation_max_tokens,
                return_predictions=True,
            )
            if isinstance(semantic_evaluation, tuple):
                semantic_result, semantic_predictions = semantic_evaluation
            else:  # Keeps lightweight mocked evaluators usable in tests.
                semantic_result, semantic_predictions = semantic_evaluation, None
            semantic_score = semantic_result[semantic_metric]
            if tensorboard:
                for metric_name, value in semantic_result.items():
                    if isinstance(value, (int, float)):
                        tensorboard.add_scalar(f"validation_execution/{metric_name}", value, epoch + 1)
                for failure_name, values in semantic_result.get("failure_counts", {}).items():
                    tensorboard.add_scalar(f"validation_failures/{failure_name}", values["rate"], epoch + 1)
                for family_name, values in semantic_result.get("by_family", {}).items():
                    for metric_name, value in values.items():
                        tensorboard.add_scalar(f"validation_family/{family_name}/{metric_name}", value, epoch + 1)
                if semantic_predictions is not None:
                    tensorboard.add_text("validation/examples", diagnostic_markdown(semantic_records, semantic_predictions), epoch + 1)
                    write_prediction_diagnostics(
                        tensorboard.log_dir / f"validation_predictions_epoch_{epoch + 1:04d}.jsonl",
                        semantic_records,
                        semantic_predictions,
                    )
            if semantic_score > best_semantic_score:
                best_semantic_score = semantic_score
                best_semantic_epoch = epoch + 1
                if best_execution_checkpoint_dir:
                    save_checkpoint(
                        best_execution_checkpoint_dir,
                        model,
                        {
                            "config": config,
                            "epoch": epoch + 1,
                            "steps": previous_metadata.get("steps", 0) + len(losses),
                            "validation_metric": semantic_metric,
                            "validation_score": semantic_score,
                            "validation_metrics": semantic_result,
                        },
                        optimizer,
                        tokenizer,
                    )

        stopped_early = bool(val_records) and patience is not None and epochs_without_improvement >= patience
        if tensorboard:
            tensorboard.add_scalar("loss/best_validation", best_val_loss, epoch + 1)
            tensorboard.add_scalar("training/epochs_without_improvement", epochs_without_improvement, epoch + 1)
            if semantic_every and best_semantic_score != float("-inf"):
                tensorboard.add_scalar(f"validation_execution/best_{semantic_metric}", best_semantic_score, epoch + 1)
            tensorboard.flush()
        print(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "validation_score": semantic_result[semantic_metric] if semantic_result else None,
            }
        )
        if checkpoint_dir:
            save_checkpoint(
                checkpoint_dir,
                model,
                {
                    "config": config,
                    "epoch": epoch + 1,
                    "steps": previous_metadata.get("steps", 0) + len(losses),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "stopped_early": stopped_early,
                    "best_validation_metric": semantic_metric if semantic_every else None,
                    "best_validation_score": best_semantic_score if semantic_every else None,
                    "best_validation_epoch": best_semantic_epoch if semantic_every else None,
                    "validation_metrics": semantic_result,
                },
                optimizer,
                tokenizer,
            )
        if stopped_early:
            print({"early_stopping": True, "epoch": epoch + 1, "best_epoch": best_epoch, "best_val_loss": best_val_loss})
            break
    if tensorboard:
        tensorboard.close()
    return model, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest"))
    parser.add_argument("--best-checkpoint", type=Path, default=None, help="Directory for the lowest-validation-loss checkpoint (default: <checkpoint>-best).")
    parser.add_argument("--best-execution-checkpoint", type=Path, default=None, help="Directory for the best validation execution checkpoint (default: <checkpoint>-best-execution).")
    parser.add_argument("--log-dir", type=Path, default=None, help="TensorBoard event directory (default: runs/<checkpoint-name>).")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=None,
        help="Load model/tokenizer weights only, starting a fresh optimizer and epoch counter.",
    )
    parser.add_argument("--overfit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch count in the YAML config.")
    parser.add_argument("--execution-gate", type=float, default=None, help="After an --overfit run, fail unless training execution accuracy reaches this value.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        if args.epochs < 1:
            raise SystemExit("--epochs must be positive")
        config["epochs"] = args.epochs
    if args.execution_gate is not None and not 0.0 <= args.execution_gate <= 1.0:
        raise SystemExit("--execution-gate must be between 0 and 1")
    if args.execution_gate is not None and not args.overfit:
        raise SystemExit("--execution-gate requires --overfit so it cannot accidentally score the full training split")
    if args.resume and args.initialize_from:
        raise SystemExit("--resume and --initialize-from are mutually exclusive")
    records = load_jsonl(args.data)
    if args.overfit:
        records = records[: args.overfit]
    val_records = load_jsonl(args.val_data) if args.val_data else None
    model, losses = train(
        records,
        config,
        args.checkpoint,
        val_records=val_records,
        resume_from=args.resume,
        initialize_from=args.initialize_from,
        best_checkpoint_dir=args.best_checkpoint,
        best_execution_checkpoint_dir=args.best_execution_checkpoint,
        tensorboard_log_dir=args.log_dir or default_tensorboard_log_dir(args.checkpoint),
    )
    print({"steps": len(losses), "final_loss": losses[-1]})
    if args.execution_gate is not None:
        tokenizer = load_tokenizer(args.checkpoint)
        result = evaluate_model(
            model,
            records,
            tokenizer,
            batch_size=config.get("validation_execution_batch_size", config["batch_size"]),
            max_tokens=config.get("generation_max_tokens", 128),
        )
        print({"execution_gate": args.execution_gate, "metrics": result})
        if result["execution_accuracy"] < args.execution_gate:
            raise SystemExit(
                f"Execution gate failed: {result['execution_accuracy']:.1%} < {args.execution_gate:.1%}"
            )


if __name__ == "__main__":
    main()
