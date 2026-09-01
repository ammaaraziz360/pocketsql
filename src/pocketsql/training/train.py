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
from pocketsql.model.factorized import FactorizedPocketSQLTransformer, FactorizedSchemaConfig
from pocketsql.model.structured import (
    StructuredPocketSQLTransformer,
    StructuredQueryConfig,
)
from pocketsql.model.transformer import ModelConfig, PocketSQLTransformer
from pocketsql.evaluation.evaluate import evaluate_model, write_prediction_diagnostics
from pocketsql.training.checkpoint import initialize_model, load_checkpoint, save_checkpoint
from pocketsql.training.audit import audit_sequences, require_complete_sequences
from pocketsql.training.dataset import encode_record, load_jsonl, make_batch
from pocketsql.training.schema_links import (
    schema_link_batch,
    structured_prompt_batch,
    structured_query_batch,
)
from pocketsql.training.tensorboard import TensorBoardLogger


def _language_loss(logits: mx.array, tokens: mx.array, mask: mx.array) -> mx.array:
    targets = tokens[:, 1:]
    active = mask[:, 1:]
    safe_targets = mx.where(active, targets, mx.zeros_like(targets))
    losses = nn.losses.cross_entropy(logits, safe_targets)
    return mx.sum(losses * active) / mx.maximum(mx.sum(active), 1)


def masked_loss(model: PocketSQLTransformer, tokens: mx.array, mask: mx.array) -> mx.array:
    return _language_loss(model(tokens[:, :-1]), tokens, mask)


def factorized_masked_loss(
    model: FactorizedPocketSQLTransformer,
    tokens: mx.array,
    mask: mx.array,
    schema_tokens: mx.array | None,
    prompt_positions: mx.array,
    schema_labels: dict[str, mx.array],
    schema_masks: dict[str, mx.array],
    schema_link_loss_weight: float,
    language_loss_weight: float = 1.0,
) -> mx.array:
    if schema_tokens is None:
        language_logits, schema_logits = model.forward_with_schema_links(tokens[:, :-1], prompt_positions)
    else:
        language_logits = model(tokens[:, :-1])
        schema_hidden = model.hidden_states(schema_tokens[:, :-1])
        pooled = schema_hidden[mx.arange(schema_hidden.shape[0]), prompt_positions]
        schema_logits = model.schema_link_logits_from_hidden(pooled)
    language = _language_loss(language_logits, tokens, mask)
    schema_loss = _classification_loss(schema_logits, schema_labels, schema_masks)
    return language_loss_weight * language + schema_link_loss_weight * schema_loss


def _classification_loss(
    logits_by_name: dict[str, mx.array],
    labels_by_name: dict[str, mx.array],
    masks_by_name: dict[str, mx.array],
    role_weights: dict[str, float] | None = None,
) -> mx.array:
    total_loss = mx.array(0.0)
    active_labels = mx.array(0.0)
    for name, logits in logits_by_name.items():
        weight = float((role_weights or {}).get(name, 1.0))
        if weight < 0:
            raise ValueError(f"classification role weight for {name!r} must be non-negative")
        if weight == 0:
            continue
        targets = labels_by_name[name]
        active = masks_by_name[name]
        classes = logits.shape[-1]
        losses = nn.losses.cross_entropy(logits.reshape(-1, classes), targets.reshape(-1))
        flat_active = active.reshape(-1)
        total_loss = total_loss + weight * mx.sum(losses * flat_active)
        active_labels = active_labels + weight * mx.sum(flat_active)
    return total_loss / mx.maximum(active_labels, 1)


def structured_masked_loss(
    model: StructuredPocketSQLTransformer,
    tokens: mx.array,
    mask: mx.array,
    supervision: dict,
    schema_link_loss_weight: float,
    language_loss_weight: float,
    operation_loss_weight: float,
    literal_loss_weight: float,
) -> mx.array:
    language = (
        _language_loss(model(tokens[:, :-1]), tokens, mask)
        if language_loss_weight
        else mx.array(0.0)
    )
    schema_logits, operation_logits, literal_logits = model.structured_logits(
        supervision["prompt_tokens"], supervision["layout"]
    )
    schema_loss = (
        _classification_loss(
            schema_logits,
            supervision["schema_labels"],
            supervision["schema_masks"],
            supervision.get("schema_role_loss_weights"),
        )
        if schema_link_loss_weight
        else mx.array(0.0)
    )
    operation_loss = (
        _classification_loss(
            operation_logits,
            supervision["operation_labels"],
            supervision["operation_masks"],
        )
        if operation_loss_weight
        else mx.array(0.0)
    )
    literal_loss = (
        _classification_loss(
            literal_logits,
            supervision["literal_labels"],
            supervision["literal_masks"],
        )
        if literal_loss_weight
        else mx.array(0.0)
    )
    return (
        language_loss_weight * language
        + schema_link_loss_weight * schema_loss
        + operation_loss_weight * operation_loss
        + literal_loss_weight * literal_loss
    )


def _schema_supervision(model, records, tokens, tokenizer, config):
    if not isinstance(model, FactorizedPocketSQLTransformer):
        return None
    labels, masks = schema_link_batch(
        records,
        identifier_slot_strategy=config.get("identifier_slot_strategy", "ordered"),
        canonicalize_literals=config.get("canonicalize_literals", False),
        schema_linking_hints=config.get("schema_linking_hints", False),
        max_table_slots=model.schema_config.max_table_slots,
        max_column_slots=model.schema_config.max_column_slots,
        max_projection_slots=model.schema_config.max_projection_slots,
        max_filter_slots=model.schema_config.max_filter_slots,
        max_group_slots=model.schema_config.max_group_slots,
    )
    if isinstance(model, StructuredPocketSQLTransformer):
        prompt_rows, layout, literal_labels, literal_masks = structured_prompt_batch(
            records, tokenizer, model, config
        )
        operation_labels, operation_masks = structured_query_batch(
            records,
            identifier_slot_strategy=config.get("identifier_slot_strategy", "ordered"),
            canonicalize_literals=config.get("canonicalize_literals", False),
            schema_linking_hints=config.get("schema_linking_hints", False),
            max_projection_slots=model.schema_config.max_projection_slots,
            max_filter_slots=model.schema_config.max_filter_slots,
            max_group_slots=model.schema_config.max_group_slots,
            max_limit_value=model.structured_config.max_limit_value,
        )
        return {
            "prompt_tokens": mx.array(prompt_rows),
            "layout": {name: mx.array(values) for name, values in layout.items()},
            "schema_labels": {name: mx.array(values) for name, values in labels.items()},
            "schema_masks": {name: mx.array(values) for name, values in masks.items()},
            "schema_role_loss_weights": config.get("schema_role_loss_weights", {}),
            "operation_labels": {
                name: mx.array(values) for name, values in operation_labels.items()
            },
            "operation_masks": {
                name: mx.array(values) for name, values in operation_masks.items()
            },
            "literal_labels": {
                name: mx.array(values) for name, values in literal_labels.items()
            },
            "literal_masks": {
                name: mx.array(values) for name, values in literal_masks.items()
            },
        }
    schema_tokens = None
    prompt_rows = tokens
    if config.get("factorized_schema_linking_hints", False):
        hints_field = config.get("factorized_schema_linking_hints_field")
        encoded = [
            encode_record(
                record,
                tokenizer,
                config["context_length"],
                config.get("canonicalize_identifiers", False),
                config.get("identifier_slot_strategy", "ordered"),
                config.get("canonicalize_literals", False),
                config.get("target_format", "sql"),
                bool(record.get(hints_field)) if hints_field else True,
                config.get("schema_linking_max_tables", 5),
                config.get("schema_linking_max_columns", 8),
            )[0]
            for record in records
        ]
        width = max(len(row) for row in encoded)
        prompt_rows = [row + [tokenizer.pad_id] * (width - len(row)) for row in encoded]
        schema_tokens = mx.array(prompt_rows)
    prompt_positions = []
    for row in prompt_rows:
        try:
            prompt_positions.append(row.index(tokenizer.sql_start_id))
        except ValueError as error:
            raise ValueError("encoded factorized training record is missing <sql>") from error
    return (
        schema_tokens,
        mx.array(prompt_positions),
        {name: mx.array(values) for name, values in labels.items()},
        {name: mx.array(values) for name, values in masks.items()},
    )


def train_step(
    model,
    optimizer,
    tokens,
    mask,
    schema_supervision=None,
    schema_link_loss_weight: float = 1.0,
    language_loss_weight: float = 1.0,
    operation_loss_weight: float = 1.0,
    literal_loss_weight: float = 1.0,
) -> float:
    if schema_supervision is None:
        loss_and_grad = nn.value_and_grad(model, masked_loss)
        loss, gradients = loss_and_grad(model, tokens, mask)
    elif isinstance(schema_supervision, dict):
        def loss_function(current_model):
            return structured_masked_loss(
                current_model,
                tokens,
                mask,
                schema_supervision,
                schema_link_loss_weight,
                language_loss_weight,
                operation_loss_weight,
                literal_loss_weight,
            )

        loss_and_grad = nn.value_and_grad(model, loss_function)
        loss, gradients = loss_and_grad(model)
    else:
        schema_tokens, prompt_positions, schema_labels, schema_masks = schema_supervision

        def loss_function(current_model):
            return factorized_masked_loss(
                current_model,
                tokens,
                mask,
                schema_tokens,
                prompt_positions,
                schema_labels,
                schema_masks,
                schema_link_loss_weight,
                language_loss_weight,
            )

        loss_and_grad = nn.value_and_grad(model, loss_function)
        loss, gradients = loss_and_grad(model)
    gradients = tree_map(lambda value: mx.clip(value, -1.0, 1.0), gradients)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state)
    return float(loss.item())


def accumulated_train_step(
    model,
    optimizer,
    micro_batches,
    schema_link_loss_weight: float = 1.0,
    language_loss_weight: float = 1.0,
    operation_loss_weight: float = 1.0,
    literal_loss_weight: float = 1.0,
) -> float:
    total_loss = 0.0
    accumulated = None
    for tokens, mask, schema_supervision in micro_batches:
        if schema_supervision is None:
            loss_and_grad = nn.value_and_grad(model, masked_loss)
            loss, gradients = loss_and_grad(model, tokens, mask)
        elif isinstance(schema_supervision, dict):
            def loss_function(current_model):
                return structured_masked_loss(
                    current_model,
                    tokens,
                    mask,
                    schema_supervision,
                    schema_link_loss_weight,
                    language_loss_weight,
                    operation_loss_weight,
                    literal_loss_weight,
                )

            loss_and_grad = nn.value_and_grad(model, loss_function)
            loss, gradients = loss_and_grad(model)
        else:
            schema_tokens, prompt_positions, schema_labels, schema_masks = schema_supervision

            def loss_function(current_model):
                return factorized_masked_loss(
                    current_model,
                    tokens,
                    mask,
                    schema_tokens,
                    prompt_positions,
                    schema_labels,
                    schema_masks,
                    schema_link_loss_weight,
                    language_loss_weight,
                )

            loss_and_grad = nn.value_and_grad(model, loss_function)
            loss, gradients = loss_and_grad(model)
        # MLX evaluates lazily. Without forcing each micro-batch here, the
        # gradient trees retain all forward/backward graphs until the final
        # optimizer update, defeating the memory benefit of accumulation.
        mx.eval(loss, gradients)
        total_loss += float(loss.item())
        accumulated = (
            gradients
            if accumulated is None
            else tree_map(lambda a, b: a + b, accumulated, gradients)
        )
        mx.eval(accumulated)
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
            config.get("schema_linking_max_tables", 5),
            config.get("schema_linking_max_columns", 8),
        )
        token_array = mx.array(tokens)
        mask_array = mx.array(mask)
        schema_supervision = _schema_supervision(model, batch, tokens, tokenizer, config)
        if schema_supervision is None:
            loss = masked_loss(model, token_array, mask_array)
        elif isinstance(schema_supervision, dict):
            loss = structured_masked_loss(
                model,
                token_array,
                mask_array,
                schema_supervision,
                config.get("schema_link_loss_weight", 1.0),
                config.get("language_loss_weight", 1.0),
                config.get("operation_loss_weight", 1.0),
                config.get("literal_loss_weight", 1.0),
            )
        else:
            loss = factorized_masked_loss(
                model,
                token_array,
                mask_array,
                *schema_supervision,
                config.get("schema_link_loss_weight", 1.0),
                config.get("language_loss_weight", 1.0),
            )
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def evaluate_schema_links(model, records: list[dict], tokenizer: TokenizerProtocol, config: dict) -> dict | None:
    """Measure pointer, operation, and literal heads without SQL decoding."""
    if not isinstance(model, FactorizedPocketSQLTransformer) or not records:
        return None
    correct: dict[str, int] = {}
    active_counts: dict[str, int] = {}
    exact_pointer_records = 0
    exact_structured_records = 0
    total_records = 0
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
            config.get("schema_linking_max_tables", 5),
            config.get("schema_linking_max_columns", 8),
        )
        supervision = _schema_supervision(model, batch, tokens, tokenizer, config)
        if isinstance(supervision, dict):
            schema_logits, operation_logits, literal_logits = model.structured_logits(
                supervision["prompt_tokens"], supervision["layout"]
            )
            groups = (
                ("", schema_logits, supervision["schema_labels"], supervision["schema_masks"], True),
                (
                    "operation_",
                    operation_logits,
                    supervision["operation_labels"],
                    supervision["operation_masks"],
                    False,
                ),
                (
                    "literal_",
                    literal_logits,
                    supervision["literal_labels"],
                    supervision["literal_masks"],
                    False,
                ),
            )
        else:
            schema_tokens, prompt_positions, labels, masks = supervision
            link_tokens = schema_tokens if schema_tokens is not None else mx.array(tokens)
            link_hidden = model.hidden_states(link_tokens[:, :-1])
            pooled = link_hidden[mx.arange(link_hidden.shape[0]), prompt_positions]
            schema_logits = model.schema_link_logits_from_hidden(pooled)
            groups = (("", schema_logits, labels, masks, True),)
        mx.eval(*[value for _, logits, _, _, _ in groups for value in logits.values()])
        batch_pointer_exact = [True] * len(batch)
        batch_structured_exact = [True] * len(batch)
        for prefix, logits, labels, masks, pointer_group in groups:
            for name, values in logits.items():
                metric_name = f"{prefix}{name}"
                correct.setdefault(metric_name, 0)
                active_counts.setdefault(metric_name, 0)
                predictions = mx.argmax(values, axis=-1).tolist()
                expected = labels[name].tolist()
                active = masks[name].tolist()
                if values.ndim == 2:
                    predictions = [[item] for item in predictions]
                    expected = [[item] for item in expected]
                    active = [[item] for item in active]
                for row, (predicted_row, expected_row, active_row) in enumerate(
                    zip(predictions, expected, active)
                ):
                    for predicted, target, enabled in zip(
                        predicted_row, expected_row, active_row
                    ):
                        if not enabled:
                            continue
                        active_counts[metric_name] += 1
                        matched = predicted == target
                        correct[metric_name] += int(matched)
                        batch_structured_exact[row] = (
                            batch_structured_exact[row] and matched
                        )
                        if pointer_group:
                            batch_pointer_exact[row] = (
                                batch_pointer_exact[row] and matched
                            )
        exact_pointer_records += sum(batch_pointer_exact)
        exact_structured_records += sum(batch_structured_exact)
        total_records += len(batch)
    result = {
        **{
            f"{name}_accuracy": correct[name] / max(active_counts[name], 1)
            for name in correct
        },
        "all_active_pointers_accuracy": exact_pointer_records / max(total_records, 1),
    }
    if isinstance(model, StructuredPocketSQLTransformer):
        result["all_active_structured_accuracy"] = exact_structured_records / max(
            total_records, 1
        )
    return result


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
    model_config = ModelConfig(
        vocab_size=vocab_size,
        layers=config["layers"],
        hidden_dim=config["hidden_dim"],
        heads=config["heads"],
        ffn_dim=config["ffn_dim"],
        context_length=config["context_length"],
    )
    architecture = config.get("model_architecture", "autoregressive")
    if architecture == "autoregressive":
        model = PocketSQLTransformer(model_config)
    elif architecture == "factorized_schema":
        model = FactorizedPocketSQLTransformer(
            model_config,
            FactorizedSchemaConfig(
                max_table_slots=config.get("max_table_slots", 16),
                max_column_slots=config.get("max_column_slots", 64),
                max_projection_slots=config.get("max_projection_slots", 4),
                max_filter_slots=config.get("max_filter_slots", 4),
                max_group_slots=config.get("max_group_slots", 2),
            ),
        )
    elif architecture == "structured_v18":
        model = StructuredPocketSQLTransformer(
            model_config,
            FactorizedSchemaConfig(
                max_table_slots=config.get("max_table_slots", 16),
                max_column_slots=config.get("max_column_slots", 64),
                max_projection_slots=config.get("max_projection_slots", 4),
                max_filter_slots=config.get("max_filter_slots", 4),
                max_group_slots=config.get("max_group_slots", 2),
            ),
            StructuredQueryConfig(
                max_literal_span_tokens=config.get("max_literal_span_tokens", 12),
                max_limit_value=config.get("max_limit_value", 100),
            ),
        )
    else:
        raise ValueError(
            "model_architecture must be 'autoregressive', 'factorized_schema', or 'structured_v18'"
        )
    model.model_architecture = architecture
    model.factorized_schema_mode = config.get("factorized_schema_mode", "fallback")
    model.factorized_schema_confidence_threshold = config.get(
        "factorized_schema_confidence_threshold", 0.9
    )
    model.canonicalize_identifiers = config.get("canonicalize_identifiers", False)
    model.identifier_slot_strategy = config.get("identifier_slot_strategy", "ordered")
    model.canonicalize_literals = config.get("canonicalize_literals", False)
    model.generation_max_tokens = config.get("generation_max_tokens", 128)
    model.target_format = config.get("target_format", "sql")
    model.constrain_semantic_plan = config.get("constrain_semantic_plan", model.target_format == "semantic_plan")
    model.schema_linking_hints = config.get("schema_linking_hints", False)
    model.factorized_schema_linking_hints = config.get("factorized_schema_linking_hints", False)
    model.schema_linking_max_tables = config.get("schema_linking_max_tables", 5)
    model.schema_linking_max_columns = config.get("schema_linking_max_columns", 8)
    model.structured_plan_mode = config.get("structured_plan_mode", "fallback")
    return model


def configure_trainable_parameters(model, config: dict) -> None:
    """Optionally isolate the schema heads for the first adaptation phase."""
    if not config.get("train_schema_heads_only", False):
        return
    if not isinstance(model, FactorizedPocketSQLTransformer):
        raise ValueError(
            "train_schema_heads_only requires model_architecture: factorized_schema or structured_v18"
        )
    model.freeze()
    allowed_prefixes = ("schema_", "operation_", "literal_")
    requested_prefixes = tuple(
        config.get("train_head_prefixes", allowed_prefixes)
    )
    if not requested_prefixes or any(
        prefix not in allowed_prefixes for prefix in requested_prefixes
    ):
        raise ValueError(
            f"train_head_prefixes must contain values from {allowed_prefixes}"
        )
    model.apply_to_modules(
        lambda key, module: module.unfreeze()
        if key.startswith(requested_prefixes)
        else None
    )
    if config.get("train_schema_prompt_positions", False):
        model.position.unfreeze()


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
    memory_limit = config.get("mlx_memory_limit_bytes")
    cache_limit = config.get("mlx_cache_limit_bytes")
    if memory_limit is not None:
        if memory_limit < 1:
            raise ValueError("mlx_memory_limit_bytes must be positive")
        mx.set_memory_limit(int(memory_limit))
    if cache_limit is not None:
        if cache_limit < 0:
            raise ValueError("mlx_cache_limit_bytes must be non-negative")
        mx.set_cache_limit(int(cache_limit))
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
            config.get("schema_linking_max_tables", 5),
            config.get("schema_linking_max_columns", 8),
        )
        require_complete_sequences(training_audit, "training")
        print({"sequence_audit": "training", **training_audit})
        if config.get("factorized_schema_linking_hints", False):
            hints_field = config.get("factorized_schema_linking_hints_field")
            schema_audit_records = (
                [record for record in records if record.get(hints_field)]
                if hints_field
                else records
            )
            schema_training_audit = audit_sequences(
                schema_audit_records,
                tokenizer,
                config["context_length"],
                generation_max_tokens,
                config.get("canonicalize_identifiers", False),
                config.get("identifier_slot_strategy", "ordered"),
                config.get("canonicalize_literals", False),
                config.get("target_format", "sql"),
                True,
                config.get("schema_linking_max_tables", 5),
                config.get("schema_linking_max_columns", 8),
            )
            require_complete_sequences(schema_training_audit, "schema-link training")
            print({"sequence_audit": "schema-link training", **schema_training_audit})
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
                config.get("schema_linking_max_tables", 5),
                config.get("schema_linking_max_columns", 8),
            )
            require_complete_sequences(validation_audit, "validation")
            print({"sequence_audit": "validation", **validation_audit})
            if config.get("factorized_schema_linking_hints", False):
                hints_field = config.get("factorized_schema_linking_hints_field")
                schema_validation_records = (
                    [record for record in val_records if record.get(hints_field)]
                    if hints_field
                    else val_records
                )
                schema_validation_audit = audit_sequences(
                    schema_validation_records,
                    tokenizer,
                    config["context_length"],
                    generation_max_tokens,
                    config.get("canonicalize_identifiers", False),
                    config.get("identifier_slot_strategy", "ordered"),
                    config.get("canonicalize_literals", False),
                    config.get("target_format", "sql"),
                    True,
                    config.get("schema_linking_max_tables", 5),
                    config.get("schema_linking_max_columns", 8),
                )
                require_complete_sequences(schema_validation_audit, "schema-link validation")
                print({"sequence_audit": "schema-link validation", **schema_validation_audit})
    model = build_model(config, tokenizer.vocab_size)
    configure_trainable_parameters(model, config)
    optimizer = make_optimizer(config)
    start_epoch = 0
    previous_metadata: dict = {}
    if resume_from is not None:
        previous_metadata = load_checkpoint(resume_from, model, optimizer)
        start_epoch = previous_metadata.get("epoch", 0)
    elif initialize_from is not None:
        initialize_model(
            initialize_from,
            model,
            allow_new_schema_heads=isinstance(model, FactorizedPocketSQLTransformer),
        )
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
                    config.get("schema_linking_max_tables", 5),
                    config.get("schema_linking_max_columns", 8),
                )
                schema_supervision = _schema_supervision(model, batch, tokens, tokenizer, config)
                micro_batches.append((mx.array(tokens), mx.array(mask), schema_supervision))
            if grad_accum_steps == 1:
                step_loss = train_step(
                    model,
                    optimizer,
                    *micro_batches[0][:2],
                    schema_supervision=micro_batches[0][2],
                    schema_link_loss_weight=config.get("schema_link_loss_weight", 1.0),
                    language_loss_weight=config.get("language_loss_weight", 1.0),
                    operation_loss_weight=config.get("operation_loss_weight", 1.0),
                    literal_loss_weight=config.get("literal_loss_weight", 1.0),
                )
            else:
                step_loss = accumulated_train_step(
                    model,
                    optimizer,
                    micro_batches,
                    config.get("schema_link_loss_weight", 1.0),
                    config.get("language_loss_weight", 1.0),
                    config.get("operation_loss_weight", 1.0),
                    config.get("literal_loss_weight", 1.0),
                )
            epoch_losses.append(step_loss)
            if tensorboard:
                tensorboard.add_scalar("loss/train_step", step_loss, initial_steps + len(losses) + len(epoch_losses))
        losses.extend(epoch_losses)
        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        val_loss = evaluate_loss(model, val_records, tokenizer, config) if val_records else float("nan")
        schema_link_result = (
            evaluate_schema_links(model, val_records, tokenizer, config) if val_records else None
        )
        if tensorboard:
            tensorboard.add_scalar("loss/train_epoch", train_loss, epoch + 1)
            if val_records:
                tensorboard.add_scalar("loss/validation", val_loss, epoch + 1)
            if schema_link_result:
                for metric_name, value in schema_link_result.items():
                    tensorboard.add_scalar(f"schema_link/{metric_name}", value, epoch + 1)
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
                        "validation_schema_links": schema_link_result,
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
                            "validation_schema_links": schema_link_result,
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
                "schema_link_score": (
                    schema_link_result["all_active_pointers_accuracy"] if schema_link_result else None
                ),
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
                    "validation_schema_links": schema_link_result,
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
