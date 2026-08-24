from __future__ import annotations

import json
from pathlib import Path

from pocketsql.model.tokenizer import ByteTokenizer


def format_record(record: dict) -> str:
    return f"<bos><schema>{record['schema_sql']}</schema><question>{record['question']}</question><sql>{record['sql']}</sql><eos>"


def encode_record(record: dict, tokenizer: ByteTokenizer, context_length: int) -> tuple[list[int], list[bool]]:
    text = format_record(record)
    ids = tokenizer.encode(text)[:context_length]
    sql_start = text.index("<sql>")
    prefix_length = len(tokenizer.encode(text[:sql_start]))
    mask = [index >= prefix_length for index in range(len(ids))]
    return ids, mask


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_batch(records: list[dict], tokenizer: ByteTokenizer, context_length: int) -> tuple[list[list[int]], list[list[bool]]]:
    encoded = [encode_record(record, tokenizer, context_length) for record in records]
    width = max(len(ids) for ids, _ in encoded)
    tokens, masks = [], []
    for ids, mask in encoded:
        padding = width - len(ids)
        tokens.append(ids + [tokenizer.pad_id] * padding)
        masks.append(mask + [False] * padding)
    return tokens, masks