from fastapi.testclient import TestClient

from src.agent.graph import AgentResult
from src.api import app as api_module

client = TestClient(api_module.app)


def make_result(answer: str = "Grounded answer.") -> AgentResult:
    return AgentResult(
        answer=answer,
        sources=[("item1_business.txt", "Item 1 – Business")],
        route="retrieve",
        route_reason="Factual corpus question.",
        rewritten_queries=[],
        retrieval_attempts=[("query", ["item1_business.txt:0"])],
        grading_attempts=[("query", True, "Chunks answer the question.")],
        retrieved_contexts=["chunk text"],
        retrieved_chunk_ids=["item1_business.txt:0"],
    )


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_sources_and_route(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "run", lambda question: make_result())

    response = client.post("/ask", json={"question": "What is AirCover for Hosts?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Grounded answer.",
        "sources": [{"source": "item1_business.txt", "section": "Item 1 – Business"}],
        "route": "retrieve",
    }


def test_ask_rejects_blank_question() -> None:
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_rejects_missing_question() -> None:
    response = client.post("/ask", json={})

    assert response.status_code == 422


def test_ask_maps_agent_failure_to_502(monkeypatch) -> None:
    def failing_run(question: str) -> AgentResult:
        raise RuntimeError("provider down")

    monkeypatch.setattr(api_module, "run", failing_run)

    response = client.post("/ask", json={"question": "What is AirCover?"})

    assert response.status_code == 502
    assert "RuntimeError" in response.json()["detail"]
    assert "provider down" not in response.json()["detail"]
