from __future__ import annotations

import argparse
import json
from pathlib import Path

from pocketsql.model.tokenizer import BPETokenizer
from pocketsql.training.dataset import format_record


def record_texts(
    paths: list[Path],
    canonicalize_identifiers: bool = False,
    identifier_slot_strategy: str = "ordered",
    canonicalize_literals: bool = False,
):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield format_record(
                        json.loads(line),
                        canonicalize_identifiers,
                        identifier_slot_strategy,
                        canonicalize_literals,
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local byte-level BPE tokenizer on PocketSQL JSONL records.")
    parser.add_argument("--data", type=Path, action="append", required=True, help="JSONL input; repeat to add another split.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument(
        "--canonicalize-identifiers",
        action="store_true",
        help="Train on reversible tableN/columnN schema slots instead of raw identifiers.",
    )
    parser.add_argument(
        "--identifier-slot-strategy",
        choices=("ordered", "permuted"),
        default="ordered",
        help="How canonical tableN/columnN slots are assigned.",
    )
    parser.add_argument(
        "--canonicalize-literals",
        action="store_true",
        help="Train on reversible valueN slots for filter and limit literals.",
    )
    args = parser.parse_args()
    if args.vocab_size < 512:
        raise SystemExit("--vocab-size must be at least 512 to retain byte fallback plus useful merges.")
    tokenizer = BPETokenizer.train(
        record_texts(
            args.data,
            args.canonicalize_identifiers,
            args.identifier_slot_strategy,
            args.canonicalize_literals,
        ),
        args.vocab_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output)
    print({"output": str(args.output), "vocab_size": tokenizer.vocab_size})


if __name__ == "__main__":
    main()
