from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from pocketsql.data.validate import is_read_only_select
from pocketsql.model.tokenizer import ByteTokenizer
from pocketsql.model.transformer import ModelConfig, PocketSQLTransformer
from pocketsql.training.checkpoint import load_checkpoint


def generate_sql(model, schema: str, question: str, tokenizer: ByteTokenizer | None = None, max_tokens: int = 128) -> str:
    tokenizer = tokenizer or ByteTokenizer()
    prompt = f"<bos><schema>{schema}</schema><question>{question}</question><sql>"
    tokens = tokenizer.encode(prompt)
    for _ in range(max_tokens):
        logits = model(mx.array([tokens[-model.config.context_length :]]))
        next_token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(next_token)
        if next_token == tokenizer.eos_id or tokenizer.decode([next_token]) == "</sql>":
            break
    text = tokenizer.decode(tokens)
    sql = text.split("<sql>", 1)[-1].split("</sql>", 1)[0].replace("<eos>", "").strip()
    if not is_read_only_select(sql):
        raise ValueError(f"model output is not a single read-only SELECT: {sql!r}")
    return sql


def load_model_from_checkpoint(checkpoint: str, tokenizer: ByteTokenizer | None = None) -> PocketSQLTransformer:
    tokenizer = tokenizer or ByteTokenizer()
    metadata_path = Path(checkpoint)
    config = json.loads((metadata_path / "metadata.json").read_text(encoding="utf-8"))["config"]
    model = PocketSQLTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, layers=config["layers"], hidden_dim=config["hidden_dim"], heads=config["heads"], ffn_dim=config["ffn_dim"], context_length=config["context_length"]))
    load_checkpoint(metadata_path, model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    tokenizer = ByteTokenizer()
    model = load_model_from_checkpoint(args.checkpoint, tokenizer)
    print(generate_sql(model, args.schema, args.question, tokenizer, args.max_tokens))


if __name__ == "__main__":
    main()