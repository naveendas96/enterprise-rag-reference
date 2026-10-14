"""Vector store. Uses a local persistent Chroma collection when VECTOR_STORE=chroma
(and chromadb is installed); otherwise an in-memory numpy store with the same interface."""
from __future__ import annotations

import os

import numpy as np

from .ingest import Chunk


class NumpyStore:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.mat = np.zeros((0, 1), dtype=np.float32)
        self.roles: list[set[str]] = []

    def add(self, chunks: list[Chunk], vecs: np.ndarray) -> None:
        self.ids = [c.id for c in chunks]
        self.roles = [set(c.roles) for c in chunks]
        self.mat = vecs

    def query(self, vec: np.ndarray, k: int, role: str) -> list[tuple[str, float]]:
        sims = self.mat @ vec
        order = np.argsort(-sims)
        out = [(self.ids[i], float(sims[i])) for i in order if role in self.roles[i] or role == "admin"]
        return out[:k]


class ChromaStore:
    def __init__(self, path: str = "data/chroma") -> None:
        import chromadb
        self.col = chromadb.PersistentClient(path=path).get_or_create_collection(
            "docs", metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], vecs: np.ndarray) -> None:
        self.col.upsert(ids=[c.id for c in chunks], embeddings=vecs.tolist(),
                        metadatas=[{r: True for r in c.roles} for c in chunks])

    def query(self, vec: np.ndarray, k: int, role: str) -> list[tuple[str, float]]:
        where = None if role == "admin" else {role: True}
        r = self.col.query(query_embeddings=[vec.tolist()], n_results=k, where=where)
        return [(i, 1 - d) for i, d in zip(r["ids"][0], r["distances"][0])]


def get_store() -> NumpyStore | ChromaStore:
    return ChromaStore() if os.getenv("VECTOR_STORE") == "chroma" else NumpyStore()
