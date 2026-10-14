"""Hybrid retrieval (BM25 + vector, fused with RRF), reranking, RBAC filter and grounded answering."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .embed import get_embedder, tokenize
from .ingest import Chunk, load_chunks
from .llm import LLM, get_llm
from .store import get_store

DOCS = Path(__file__).resolve().parents[2] / "data" / "docs"
STOP = set("a an the of to for in on is are do does i my we our how what when who which many much must be by with and or per can".split())
SYSTEM = ("Answer ONLY from the context. Cite sources as [n]. If the context does not contain "
          "the answer, reply exactly: I don't know based on the available documents.")
REFUSAL = "I don't know based on the available documents."


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    citations: list[dict]
    grounded: bool
    hits: list[Hit]


def content_terms(text: str) -> set[str]:
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokenize(text) if t not in STOP}


def mock_answer(prompt: str, _system: str) -> str:
    """Deterministic extractive 'LLM': returns the context sentence with most query overlap."""
    q = prompt.split("QUESTION:")[-1]
    qt = content_terms(q)
    best, best_s, best_n = "", 0, 0
    for m in re.finditer(r"\[(\d+)\] (.*)", prompt.split("QUESTION:")[0]):
        for s in re.split(r"(?<=[.!?])\s+", m.group(2)):
            sc = len(qt & content_terms(s))
            if sc > best_s:
                best, best_s, best_n = s, sc, int(m.group(1))
    return f"{best} [{best_n}]" if best_s >= 1 else REFUSAL


class RAG:
    def __init__(self, docs: Path = DOCS, llm: LLM | None = None, min_overlap: float = 0.5) -> None:
        self.chunks = load_chunks(docs)
        self.by_id = {c.id: c for c in self.chunks}
        self.emb = get_embedder()
        self.store = get_store()
        self.store.add(self.chunks, self.emb.encode([f"{c.title} {c.section} {c.text}" for c in self.chunks]))
        self.bm25 = BM25Okapi([tokenize(f"{c.section} {c.text}") for c in self.chunks])
        self.llm = llm or get_llm(mock_answer)
        self.min_overlap = min_overlap

    def allowed(self, c: Chunk, role: str) -> bool:
        return role == "admin" or role in c.roles

    def retrieve(self, q: str, role: str, k: int = 4, rrf_k: int = 60) -> list[Hit]:
        # RBAC is applied BEFORE fusion so restricted text never reaches the prompt.
        vec = self.store.query(self.emb.encode([q])[0], 20, role)
        bm = self.bm25.get_scores(tokenize(q))
        bm_rank = sorted((i for i, c in enumerate(self.chunks) if self.allowed(c, role) and bm[i] > 0),
                         key=lambda i: -bm[i])[:20]
        fused: dict[str, float] = {}
        for r, (cid, _) in enumerate(vec):
            fused[cid] = fused.get(cid, 0) + 1 / (rrf_k + r)
        for r, i in enumerate(bm_rank):
            cid = self.chunks[i].id
            fused[cid] = fused.get(cid, 0) + 1 / (rrf_k + r)
        cands = sorted(fused, key=lambda c: -fused[c])[:12]
        return self.rerank(q, cands)[:k]

    def rerank(self, q: str, ids: list[str]) -> list[Hit]:
        """Lightweight lexical cross-scorer: query-term coverage + section-title bonus.
        Swap for a cross-encoder (e.g. ms-marco-MiniLM) in production."""
        qt = content_terms(q)
        hits = []
        for cid in ids:
            c = self.by_id[cid]
            cov = len(qt & content_terms(c.text)) / max(len(qt), 1)
            sec = len(qt & content_terms(c.section + " " + c.title)) / max(len(qt), 1)
            hits.append(Hit(c, round(cov + 0.5 * sec, 4)))
        return sorted(hits, key=lambda h: -h.score)

    def ask(self, q: str, role: str = "employee") -> Answer:
        hits = self.retrieve(q, role)
        if not hits or hits[0].score < self.min_overlap:
            return Answer(REFUSAL, [], False, hits)
        ctx = "\n".join(f"[{i + 1}] {h.chunk.text}" for i, h in enumerate(hits))
        text = self.llm.complete(f"CONTEXT:\n{ctx}\nQUESTION: {q}", SYSTEM).strip()
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
        grounded = REFUSAL not in text and bool(cited)
        cites = [{"n": n, "doc": hits[n - 1].chunk.doc_id, "section": hits[n - 1].chunk.section}
                 for n in cited]
        return Answer(text if grounded else REFUSAL, cites if grounded else [], grounded, hits)
