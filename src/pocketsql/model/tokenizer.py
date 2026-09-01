from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<schema>", "</schema>", "<question>", "</question>", "<sql>", "</sql>")


class TokenizerProtocol(Protocol):
    pad_id: int
    bos_id: int
    eos_id: int
    sql_start_id: int
    sql_end_id: int

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]: ...

    def decode(self, ids: list[int]) -> str: ...

    def save(self, path: Path) -> None: ...


class ByteTokenizer:
    special_tokens = SPECIAL_TOKENS

    def __init__(self) -> None:
        self.token_to_id = {token: 256 + index for index, token in enumerate(self.special_tokens)}
        self.id_to_token = {identifier: token for token, identifier in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.sql_start_id = self.token_to_id["<sql>"]
        self.sql_end_id = self.token_to_id["</sql>"]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.special_tokens)

    def encode(self, text: str) -> list[int]:
        ids, _ = self.encode_with_offsets(text)
        return ids

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        position = 0
        while position < len(text):
            token = next((item for item in self.special_tokens if text.startswith(item, position)), None)
            if token:
                ids.append(self.token_to_id[token])
                offsets.append((position, position + len(token)))
                position += len(token)
            else:
                char = text[position]
                encoded = char.encode("utf-8")
                ids.extend(encoded)
                offsets.extend([(position, position + 1)] * len(encoded))
                position += 1
        return ids, offsets

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        byte_buffer = bytearray()
        for identifier in ids:
            if identifier < 256:
                byte_buffer.append(identifier)
            else:
                if byte_buffer:
                    parts.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                parts.append(self.id_to_token.get(identifier, ""))
        if byte_buffer:
            parts.append(byte_buffer.decode("utf-8", errors="replace"))
        return "".join(parts)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"type": "byte", "special_tokens": self.special_tokens}), encoding="utf-8")


class BPETokenizer:
    """Fast local byte-level BPE with complete byte fallback."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self.pad_id = self._required_id("<pad>")
        self.bos_id = self._required_id("<bos>")
        self.eos_id = self._required_id("<eos>")
        self.sql_start_id = self._required_id("<sql>")
        self.sql_end_id = self._required_id("</sql>")

    def _required_id(self, token: str) -> int:
        identifier = self._tokenizer.token_to_id(token)
        if identifier is None:
            raise ValueError(f"Tokenizer is missing required special token {token!r}")
        return identifier

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self._tokenizer.encode(text, add_special_tokens=False)
        return encoded.ids, encoded.offsets

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, path: Path) -> None:
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def train(cls, texts, vocab_size: int = 4096) -> "BPETokenizer":
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer

        tokenizer = Tokenizer(BPE())
        byte_level = ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.pre_tokenizer = byte_level
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)


def load_tokenizer(path: Path | str | None = None) -> TokenizerProtocol:
    """Load an embedded/configured tokenizer, falling back to legacy bytes."""
    if path is None:
        return ByteTokenizer()
    tokenizer_path = Path(path)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
        if not tokenizer_path.exists():
            return ByteTokenizer()
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if payload.get("type") == "byte":
        return ByteTokenizer()
    if payload.get("model", {}).get("type") == "BPE":
        return BPETokenizer.load(tokenizer_path)
    raise ValueError(f"Unsupported tokenizer file: {tokenizer_path}")
