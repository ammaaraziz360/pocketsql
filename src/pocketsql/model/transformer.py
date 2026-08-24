from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 265
    layers: int = 6
    hidden_dim: int = 384
    heads: int = 6
    ffn_dim: int = 1536
    context_length: int = 512


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.hidden_dim // config.heads
        self.qkv = nn.Linear(config.hidden_dim, config.hidden_dim * 3, bias=False)
        self.output = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def __call__(self, values: mx.array) -> mx.array:
        batch, length, width = values.shape
        qkv = self.qkv(values).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, value = (qkv[:, :, index].transpose(0, 2, 1, 3) for index in range(3))
        scores = (query @ key.transpose(0, 1, 3, 2)) / (self.head_dim**0.5)
        mask = mx.triu(mx.full((length, length), -1e9), k=1)
        weights = mx.softmax(scores + mask, axis=-1)
        attended = (weights @ value).transpose(0, 2, 1, 3).reshape(batch, length, width)
        return self.output(attended)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        self.attention = Attention(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(config.hidden_dim, config.ffn_dim), nn.GELU(), nn.Linear(config.ffn_dim, config.hidden_dim))

    def __call__(self, values: mx.array) -> mx.array:
        values = values + self.attention(self.attention_norm(values))
        return values + self.ffn(self.ffn_norm(values))


class PocketSQLTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position = nn.Embedding(config.context_length, config.hidden_dim)
        self.blocks = [Block(config) for _ in range(config.layers)]
        self.norm = nn.LayerNorm(config.hidden_dim)

    def __call__(self, tokens: mx.array) -> mx.array:
        _, length = tokens.shape
        if length > self.config.context_length:
            raise ValueError("sequence exceeds context length")
        positions = mx.arange(length)[None, :]
        values = self.embedding(tokens) + self.position(positions)
        for block in self.blocks:
            values = block(values)
        return self.norm(values) @ self.embedding.weight.T