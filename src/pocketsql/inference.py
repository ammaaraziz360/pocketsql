from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import mlx.core as mx

from pocketsql.data.validate import is_read_only_select
from pocketsql.model.schema_grounding import IdentifierMapping, canonicalize_inputs
from pocketsql.model.tokenizer import TokenizerProtocol, load_tokenizer
from pocketsql.model.transformer import ModelConfig, PocketSQLTransformer
from pocketsql.training.checkpoint import load_checkpoint


def _generation_limit(model, max_tokens: int | None) -> int:
    limit = max_tokens if max_tokens is not None else getattr(model, "generation_max_tokens", 128)
    if limit < 1:
        raise ValueError("max_tokens must be positive")
    return limit


def _grounded_inputs(model, schema: str, question: str) -> tuple[str, str, IdentifierMapping | None]:
    if not getattr(model, "canonicalize_identifiers", False):
        return schema, question, None
    return canonicalize_inputs(schema, question, getattr(model, "identifier_slot_strategy", "ordered"))


def _valid_for_schema(schema_sql: str, sql: str) -> bool:
    """Ask SQLite to resolve every table and column without running the query."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(schema_sql)
        connection.execute(f"EXPLAIN QUERY PLAN {sql}")
        return True
    except sqlite3.Error:
        return False
    finally:
        connection.close()


def _finish_sql(sql: str, mapping: IdentifierMapping | None, schema_sql: str | None = None) -> str:
    if mapping is not None:
        if not mapping.accepts_sql(sql):
            return ""
        sql = mapping.restore(sql)
    if not is_read_only_select(sql):
        return ""
    if schema_sql is not None and not _valid_for_schema(schema_sql, sql):
        return ""
    return sql


def generate_sql(model, schema: str, question: str, tokenizer: TokenizerProtocol | None = None, max_tokens: int | None = None) -> str:
    tokenizer = tokenizer or load_tokenizer()
    source_schema = schema
    schema, question, mapping = _grounded_inputs(model, schema, question)
    prompt = f"<bos><schema>{schema}</schema><question>{question}</question><sql>"
    tokens = tokenizer.encode(prompt)
    if len(tokens) >= model.config.context_length:
        raise ValueError(f"prompt uses {len(tokens)} tokens but model context length is {model.config.context_length}")
    budget = min(_generation_limit(model, max_tokens), model.config.context_length - len(tokens))
    for _ in range(budget):
        logits = model(mx.array([tokens]))
        next_token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(next_token)
        if next_token in (tokenizer.eos_id, tokenizer.sql_end_id):
            break
    text = tokenizer.decode(tokens)
    sql = text.split("<sql>", 1)[-1].split("</sql>", 1)[0].replace("<eos>", "").strip()
    finished = _finish_sql(sql, mapping, source_schema)
    if not finished:
        raise ValueError(f"model output is not a single read-only SELECT: {sql!r}")
    return finished


def generate_sql_batch(
    model,
    schemas: list[str],
    questions: list[str],
    tokenizer: TokenizerProtocol | None = None,
    max_tokens: int | None = None,
) -> list[str]:
    """Generate safe SQL for a batch of prompts using right padding under causal attention.

    Each row's next-token logits are read at its last non-padding position, so future
    right-padding cannot influence the generated result. Invalid generations are
    returned as empty strings, matching evaluation's existing failure behavior.
    """
    if len(schemas) != len(questions):
        raise ValueError("schemas and questions must have the same length")
    if not schemas:
        return []
    tokenizer = tokenizer or load_tokenizer()
    max_tokens = _generation_limit(model, max_tokens)
    grounded = [_grounded_inputs(model, schema, question) for schema, question in zip(schemas, questions)]
    prompts = [
        tokenizer.encode(f"<bos><schema>{schema}</schema><question>{question}</question><sql>")
        for schema, question, _ in grounded
    ]
    sequences = [list(prompt) for prompt in prompts]
    groups: dict[int, list[int]] = {}
    for index, prompt in enumerate(prompts):
        groups.setdefault(len(prompt), []).append(index)
    for prompt_length, indices in groups.items():
        available = model.config.context_length - prompt_length
        if available > 0:
            _generate_with_cache(model, sequences, indices, tokenizer, min(max_tokens, available))

    outputs = []
    for tokens, source_schema, (_, _, mapping) in zip(sequences, schemas, grounded):
        text = tokenizer.decode(tokens)
        sql = text.split("<sql>", 1)[-1].split("</sql>", 1)[0].replace("<eos>", "").strip()
        outputs.append(_finish_sql(sql, mapping, source_schema))
    return outputs


def _generate_with_cache(model, sequences: list[list[int]], indices: list[int], tokenizer: TokenizerProtocol, max_tokens: int) -> None:
    prompt_tokens = mx.array([sequences[index] for index in indices])
    logits, cache = model.forward_with_cache(prompt_tokens)
    mx.eval(logits, *[value for block_cache in cache for value in block_cache])
    active = [True] * len(indices)
    for _ in range(max_tokens):
        next_tokens = [int(mx.argmax(logits[row, -1]).item()) for row in range(len(indices))]
        inputs = []
        for row, (index, next_token) in enumerate(zip(indices, next_tokens)):
            if active[row]:
                sequences[index].append(next_token)
                if next_token in (tokenizer.eos_id, tokenizer.sql_end_id):
                    active[row] = False
            inputs.append(next_token if active[row] else tokenizer.pad_id)
        if not any(active):
            break
        logits, cache = model.forward_with_cache(mx.array(inputs)[:, None], cache)
        mx.eval(logits, *[value for block_cache in cache for value in block_cache])


def _generate_without_cache(model, sequences: list[list[int]], indices: list[int], tokenizer: TokenizerProtocol, max_tokens: int) -> None:
    """Preserve sliding-window behavior for unusually long prompts."""
    active = list(indices)
    for _ in range(max_tokens):
        if not active:
            break
        windows = [sequences[index][-model.config.context_length :] for index in active]
        width = max(len(window) for window in windows)
        batch = [window + [tokenizer.pad_id] * (width - len(window)) for window in windows]
        logits = model(mx.array(batch))
        next_tokens = [int(mx.argmax(logits[row, len(window) - 1]).item()) for row, window in enumerate(windows)]
        remaining = []
        for index, next_token in zip(active, next_tokens):
            sequences[index].append(next_token)
            if next_token not in (tokenizer.eos_id, tokenizer.sql_end_id):
                remaining.append(index)
        active = remaining


def load_model_from_checkpoint(checkpoint: str, tokenizer: TokenizerProtocol | None = None) -> PocketSQLTransformer:
    tokenizer = tokenizer or load_tokenizer(Path(checkpoint))
    metadata_path = Path(checkpoint)
    config = json.loads((metadata_path / "metadata.json").read_text(encoding="utf-8"))["config"]
    model = PocketSQLTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, layers=config["layers"], hidden_dim=config["hidden_dim"], heads=config["heads"], ffn_dim=config["ffn_dim"], context_length=config["context_length"]))
    model.generation_max_tokens = config.get("generation_max_tokens", 128)
    model.canonicalize_identifiers = config.get("canonicalize_identifiers", False)
    model.identifier_slot_strategy = config.get("identifier_slot_strategy", "ordered")
    load_checkpoint(metadata_path, model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-tokens", type=int, default=None, help="Override the generation cap stored in the checkpoint.")
    args = parser.parse_args()
    tokenizer = load_tokenizer(Path(args.checkpoint))
    model = load_model_from_checkpoint(args.checkpoint, tokenizer)
    print(generate_sql(model, args.schema, args.question, tokenizer, args.max_tokens))


if __name__ == "__main__":
    main()
