from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, Header
from pydantic import BaseModel

from .pipeline import RAG

app = FastAPI(title="Enterprise RAG Reference")


@lru_cache
def rag() -> RAG:
    return RAG()


class Q(BaseModel):
    question: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Q, x_user_role: str = Header("employee")) -> dict:
    """Role comes from a header here; in production derive it from the verified IdP token."""
    a = rag().ask(q.question, x_user_role)
    return {"answer": a.text, "grounded": a.grounded, "citations": a.citations}
