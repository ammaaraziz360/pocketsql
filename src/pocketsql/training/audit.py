from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pocketsql.model.tokenizer import TokenizerProtocol, load_tokenizer
from pocketsql.training.dataset import format_record, load_jsonl


def audit_sequences(
    records: list[dict],
    tokenizer: TokenizerProtocol,
    context_length: int,
    generation_max_tokens: int,
    canonicalize_identifiers: bool = False,
    identifier_slot_strategy: str = "ordered",
) -> dict:
    zero_target = 0
    partial_target = 0
    complete = 0
    generation_too_long = 0
    max_prompt_tokens = 0
    max_target_tokens = 0
    max_sequence_tokens = 0
    for record in records:
        tokens = tokenizer.encode(format_record(record, canonicalize_identifiers, identifier_slot_strategy))
        sql_marker = tokens.index(tokenizer.sql_start_id)
        prefix_tokens = sql_marker
        prompt_tokens = sql_marker + 1
        full_tokens = len(tokens)
        # Exclude <sql> and the trailing <eos>; include SQL plus </sql>.
        generation_tokens = full_tokens - sql_marker - 2
        max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
        max_target_tokens = max(max_target_tokens, generation_tokens)
        max_sequence_tokens = max(max_sequence_tokens, full_tokens)
        if prefix_tokens >= context_length:
            zero_target += 1
        elif full_tokens > context_length:
            partial_target += 1
        else:
            complete += 1
        if generation_tokens > generation_max_tokens:
            generation_too_long += 1
    count = len(records)
    return {
        "records": count,
        "context_length": context_length,
        "generation_max_tokens": generation_max_tokens,
        "complete_sequences": complete,
        "partial_targets": partial_target,
        "zero_targets": zero_target,
        "generation_targets_over_cap": generation_too_long,
        "max_prompt_tokens": max_prompt_tokens,
        "max_generation_tokens": max_target_tokens,
        "max_sequence_tokens": max_sequence_tokens,
    }


def require_complete_sequences(report: dict, label: str = "training") -> None:
    if report["partial_targets"] or report["zero_targets"] or report["generation_targets_over_cap"]:
        raise ValueError(
            f"{label} sequence audit failed: {report['partial_targets']} partial SQL targets, "
            f"{report['zero_targets']} missing SQL targets, and {report['generation_targets_over_cap']} "
            f"generation targets over the {report['generation_max_tokens']}-token cap. "
            f"Need context_length >= {report['max_sequence_tokens']} and generation_max_tokens >= "
            f"{report['max_generation_tokens']} for this tokenizer."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PocketSQL sequence and generation lengths before training.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allow-truncation", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(config.get("tokenizer_path"))
    report = audit_sequences(
        load_jsonl(args.data),
        tokenizer,
        config["context_length"],
        config.get("generation_max_tokens", 128),
        config.get("canonicalize_identifiers", False),
        config.get("identifier_slot_strategy", "ordered"),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.allow_truncation:
        require_complete_sequences(report)


if __name__ == "__main__":
    main()
