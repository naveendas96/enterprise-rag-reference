"""Embedders. Default is a dependency-free hashing embedder (deterministic, CPU, no download).
Set EMBEDDER=st to use sentence-transformers (all-MiniLM-L6-v2, ~90 MB, CPU)."""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class HashEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])


class STEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.m.encode(texts, normalize_embeddings=True))


def get_embedder() -> HashEmbedder | STEmbedder:
    return STEmbedder() if os.getenv("EMBEDDER", "hash") == "st" else HashEmbedder()
