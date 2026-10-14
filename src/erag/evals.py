"""Golden-set evaluation: retrieval hit@k, answer correctness, citation accuracy, refusal and RBAC leakage."""
from __future__ import annotations

import time
from pathlib import Path

import yaml

from .pipeline import RAG

GOLDEN = Path(__file__).resolve().parents[2] / "data" / "golden.yaml"


def run_eval(rag: RAG | None = None) -> dict[str, float]:
    rag = rag or RAG()
    cases = yaml.safe_load(GOLDEN.read_text())
    m = {"hit@4": 0, "answer_correct": 0, "citation_correct": 0, "refusal_correct": 0, "rbac_leaks": 0}
    n_ans = n_ref = 0
    t0 = time.perf_counter()
    for c in cases:
        a = rag.ask(c["q"], c.get("role", "employee"))
        leak = any(not rag.allowed(h.chunk, c.get("role", "employee")) for h in a.hits)
        m["rbac_leaks"] += int(leak)
        if c.get("expect_refusal"):
            n_ref += 1
            m["refusal_correct"] += int(not a.grounded)
            continue
        n_ans += 1
        m["hit@4"] += int(any(h.chunk.doc_id == c["doc"] for h in a.hits))
        m["answer_correct"] += int(a.grounded and c["contains"].lower() in a.text.lower())
        m["citation_correct"] += int(any(ci["doc"] == c["doc"] for ci in a.citations))
    res = {k: (v / n_ans if k in ("hit@4", "answer_correct", "citation_correct") else v) for k, v in m.items()}
    res["refusal_correct"] = m["refusal_correct"] / max(n_ref, 1)
    res["n_cases"] = len(cases)
    res["avg_latency_ms"] = (time.perf_counter() - t0) * 1000 / len(cases)
    return res
