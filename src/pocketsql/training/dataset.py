from __future__ import annotations

import json
from pathlib import Path

from pocketsql.model.schema_grounding import canonicalize_record
from pocketsql.model.tokenizer import TokenizerProtocol


def format_record(
    record: dict,
    canonicalize_identifiers: bool = False,
    identifier_slot_strategy: str = "ordered",
) -> str:
    if canonicalize_identifiers:
        record = canonicalize_record(record, identifier_slot_strategy)
    return f"<bos><schema>{record['schema_sql']}</schema><question>{record['question']}</question><sql>{record['sql']}</sql><eos>"


def encode_record(
    record: dict,
    tokenizer: TokenizerProtocol,
    context_length: int,
    canonicalize_identifiers: bool = False,
    identifier_slot_strategy: str = "ordered",
) -> tuple[list[int], list[bool]]:
    text = format_record(record, canonicalize_identifiers, identifier_slot_strategy)
    ids = tokenizer.encode(text)[:context_length]
    sql_start = text.index("<sql>")
    prefix_length = len(tokenizer.encode(text[:sql_start]))
    mask = [index >= prefix_length for index in range(len(ids))]
    return ids, mask


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_batch(
    records: list[dict],
    tokenizer: TokenizerProtocol,
    context_length: int,
    canonicalize_identifiers: bool = False,
    identifier_slot_strategy: str = "ordered",
) -> tuple[list[list[int]], list[list[bool]]]:
    encoded = [
        encode_record(record, tokenizer, context_length, canonicalize_identifiers, identifier_slot_strategy)
        for record in records
    ]
    width = max(len(ids) for ids, _ in encoded)
    tokens, masks = [], []
    for ids, mask in encoded:
        padding = width - len(ids)
        tokens.append(ids + [tokenizer.pad_id] * padding)
        masks.append(mask + [False] * padding)
    return tokens, masks
