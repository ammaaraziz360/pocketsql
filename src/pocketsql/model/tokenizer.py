from __future__ import annotations


class ByteTokenizer:
    special_tokens = ("<pad>", "<bos>", "<eos>", "<schema>", "</schema>", "<question>", "</question>", "<sql>", "</sql>")

    def __init__(self) -> None:
        self.token_to_id = {token: 256 + index for index, token in enumerate(self.special_tokens)}
        self.id_to_token = {identifier: token for token, identifier in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.special_tokens)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        position = 0
        while position < len(text):
            token = next((item for item in self.special_tokens if text.startswith(item, position)), None)
            if token:
                ids.append(self.token_to_id[token])
                position += len(token)
            else:
                char = text[position]
                ids.extend(char.encode("utf-8"))
                position += 1
        return ids

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