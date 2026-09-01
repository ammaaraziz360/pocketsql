from __future__ import annotations

import json
from pathlib import Path

from pocketsql.model.schema_grounding import canonicalize_record
from pocketsql.model.semantic_plan import serialize_semantic_plan
from pocketsql.model.tokenizer import TokenizerProtocol


TARGET_FORMATS = {"sql", "semantic_plan"}


def target_text(record: dict, target_format: str = "sql") -> str:
    if target_format == "sql":
        return record["sql"]
    if target_format == "semantic_plan":
        if "query_plan" not in record:
            raise ValueError("semantic_plan targets require record['query_plan']")
        return serialize_semantic_plan(record["query_plan"])
    raise ValueError(f"target_format must be one of {sorted(TARGET_FORMATS)}")


def format_record(
    record: dict,
    canonicalize_identifiers: bool = False,
    identifier_slot_strategy: str = "ordered",
    canonicalize_literals: bool = False,
    target_format: str = "sql",
    schema_linking_hints: bool = False,
    schema_linking_max_tables: int = 5,
    schema_linking_max_columns: int = 8,
) -> str:
    if canonicalize_identifiers:
        record = canonicalize_record(
            record,
            identifier_slot_strategy,
            canonicalize_literals,
            schema_linking_hints,
            schema_linking_max_tables,
            schema_linking_max_columns,
        )
    target = target_text(record, target_format)
    return f"<bos><schema>{record['schema_sql']}</schema><question>{record['question']}</question><sql>{target}</sql><eos>"


def encode_record(
    record: dict,
    tokenizer: TokenizerProtocol,
    context_length: int,
    canonicalize_identifiers: bool = False,
    identifier_slot_strategy: str = "ordered",
    canonicalize_literals: bool = False,
    target_format: str = "sql",
    schema_linking_hints: bool = False,
    schema_linking_max_tables: int = 5,
    schema_linking_max_columns: int = 8,
) -> tuple[list[int], list[bool]]:
    text = format_record(
        record,
        canonicalize_identifiers,
        identifier_slot_strategy,
        canonicalize_literals,
        target_format,
        schema_linking_hints,
        schema_linking_max_tables,
        schema_linking_max_columns,
    )
    ids = tokenizer.encode(text)[:context_length]
    sql_start = text.index("<sql>")
    prefix_length = len(tokenizer.encode(text[:sql_start]))
    mask = [index >= prefix_length for index in range(len(ids))]
    return ids, mask


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records:
        if record.get("database_path") and not Path(record["database_path"]).is_absolute():
            record["_database_base_dir"] = str(path.parent.resolve())
    return records


def make_batch(
    records: list[dict],
    tokenizer: TokenizerProtocol,
    context_length: int,
    canonicalize_identifiers: bool = False,
    identifier_slot_strategy: str = "ordered",
    canonicalize_literals: bool = False,
    target_format: str = "sql",
    schema_linking_hints: bool = False,
    schema_linking_max_tables: int = 5,
    schema_linking_max_columns: int = 8,
) -> tuple[list[list[int]], list[list[bool]]]:
    encoded = [
        encode_record(
            record,
            tokenizer,
            context_length,
            canonicalize_identifiers,
            identifier_slot_strategy,
            canonicalize_literals,
            target_format,
            schema_linking_hints,
            schema_linking_max_tables,
            schema_linking_max_columns,
        )
        for record in records
    ]
    width = max(len(ids) for ids, _ in encoded)
    tokens, masks = [], []
    for ids, mask in encoded:
        padding = width - len(ids)
        tokens.append(ids + [tokenizer.pad_id] * padding)
        masks.append(mask + [False] * padding)
    return tokens, masks
