from erag.evals import run_eval
from erag.ingest import chunk_doc
from erag.llm import MockLLM
from erag.pipeline import REFUSAL, RAG

rag = RAG()


def test_chunking_respects_headings():
    cs = chunk_doc("d", {"title": "T", "roles": ["employee"]}, "# A\none two\n## B\nthree")
    assert [c.section for c in cs] == ["A", "B"]


def test_answer_has_citation():
    a = rag.ask("How many days of paid annual leave do employees get?")
    assert a.grounded and "24 days" in a.text
    assert a.citations[0]["doc"] == "hr-leave-policy"


def test_rbac_blocks_restricted_docs():
    a = rag.ask("What was Q3 revenue?", role="employee")
    assert not a.grounded and a.text == REFUSAL
    assert all("employee" in h.chunk.roles for h in a.hits)


def test_rbac_allows_entitled_role():
    assert "412" in rag.ask("What was Q3 revenue?", role="finance").text


def test_refuses_out_of_scope():
    assert rag.ask("Who won the cricket world cup?", role="admin").text == REFUSAL


def test_uncited_llm_output_is_refused():
    r = RAG(llm=MockLLM(lambda p, s: "Employees get 99 days."))
    assert not r.ask("How many days of paid annual leave do employees get?").grounded


def test_eval_thresholds():
    m = run_eval(rag)
    assert m["hit@4"] >= 0.9 and m["answer_correct"] >= 0.8
    assert m["refusal_correct"] == 1.0 and m["rbac_leaks"] == 0


def test_api():
    from fastapi.testclient import TestClient
    from erag.api import app
    r = TestClient(app).post("/ask", json={"question": "How many characters must passwords have?"})
    assert r.status_code == 200 and "14" in r.json()["answer"]
