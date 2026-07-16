"""Unit tests for the agent graph logic, using fake LLMs so no API keys are needed."""

from langchain_core.documents import Document
from pydantic import BaseModel

from src.agent import graph
from src.agent.graph import (
    MAX_REWRITES,
    GradeDecision,
    RewriteDecision,
    RouteDecision,
    build_initial_state_update,
    grade_node,
    route_after_grade,
    route_after_router,
)
from src.generation.answer import AnswerResult


class FakeStructuredLLM:
    def __init__(self, decision: BaseModel) -> None:
        self.decision = decision

    def invoke(self, messages: object) -> BaseModel:
        return self.decision


class FakeLLM:
    """Returns a canned decision per structured-output model type."""

    def __init__(self, decisions: dict[type[BaseModel], BaseModel]) -> None:
        self.decisions = decisions

    def with_structured_output(self, model: type[BaseModel]) -> FakeStructuredLLM:
        return FakeStructuredLLM(self.decisions[model])


def make_document(chunk_id: str) -> Document:
    return Document(
        page_content=f"content for {chunk_id}",
        metadata={"chunk_id": chunk_id, "source": "source.txt", "section": "section"},
    )


def base_state(**overrides: object) -> dict:
    state = {
        "question": "What is AirCover?",
        "search_query": "What is AirCover?",
        "documents": [make_document("a")],
        "answer": "",
        "sources": [],
        "rewrite_count": 0,
        "route": "retrieve",
        "rewritten_queries": [],
        "retrieval_attempts": [],
        "grading_attempts": [],
    }
    state.update(overrides)
    return state


def test_build_initial_state_update_resets_state_and_keeps_route() -> None:
    decision = RouteDecision(route="retrieve", reason="Factual corpus question.")

    update = build_initial_state_update("What is AirCover?", decision)

    assert update["search_query"] == "What is AirCover?"
    assert update["route"] == "retrieve"
    assert update["route_reason"] == "Factual corpus question."
    assert update["rewrite_count"] == 0
    assert update["documents"] == []


def test_route_after_router_returns_route_from_state() -> None:
    assert route_after_router(base_state(route="out_of_scope")) == "out_of_scope"


def test_route_after_grade_generates_when_sufficient() -> None:
    assert route_after_grade(base_state(sufficient=True)) == "generate"


def test_route_after_grade_rewrites_while_budget_remains() -> None:
    state = base_state(sufficient=False, rewrite_count=MAX_REWRITES - 1)

    assert route_after_grade(state) == "rewrite"


def test_route_after_grade_stops_when_budget_exhausted() -> None:
    state = base_state(sufficient=False, rewrite_count=MAX_REWRITES)

    assert route_after_grade(state) == "insufficient_evidence"


def test_grade_node_marks_empty_retrieval_insufficient_without_llm() -> None:
    state = base_state(documents=[])

    update = grade_node(state)

    assert update["sufficient"] is False
    assert update["grading_attempts"] == [
        ("What is AirCover?", False, "No documents were retrieved for this query.")
    ]


def test_grade_node_records_llm_decision(monkeypatch) -> None:
    decision = GradeDecision(sufficient=True, reason="Chunks answer the question.")
    monkeypatch.setattr(graph, "_llm", lambda: FakeLLM({GradeDecision: decision}))

    update = grade_node(base_state())

    assert update["sufficient"] is True
    assert update["grading_attempts"] == [("What is AirCover?", True, "Chunks answer the question.")]


def test_rewrite_node_tracks_budget_and_query_history(monkeypatch) -> None:
    decision = RewriteDecision(query="aircover host protections coverage")
    monkeypatch.setattr(graph, "_llm", lambda: FakeLLM({RewriteDecision: decision}))

    update = graph.rewrite_node(base_state(grading_attempts=[("q", False, "missing facts")]))

    assert update["search_query"] == "aircover host protections coverage"
    assert update["rewrite_count"] == 1
    assert update["rewritten_queries"] == ["aircover host protections coverage"]


def test_graph_routes_retrieve_grade_generate_end_to_end(monkeypatch) -> None:
    decisions = {
        RouteDecision: RouteDecision(route="retrieve", reason="Factual corpus question."),
        GradeDecision: GradeDecision(sufficient=True, reason="Chunks answer the question."),
    }
    monkeypatch.setattr(graph, "_llm", lambda: FakeLLM(decisions))
    monkeypatch.setattr(graph.hybrid, "retrieve", lambda query, k=4: [make_document("a")])
    monkeypatch.setattr(
        graph.generation,
        "generate",
        lambda question, documents: AnswerResult(
            answer="Grounded answer.", sources=[("source.txt", "section")]
        ),
    )

    final_state = graph.build_graph().invoke({"question": "What is AirCover?"})

    assert final_state["answer"] == "Grounded answer."
    assert final_state["sources"] == [("source.txt", "section")]
    assert final_state["route"] == "retrieve"
    assert final_state["retrieval_attempts"] == [("What is AirCover?", ["a"])]


def test_graph_exhausts_rewrites_and_refuses_end_to_end(monkeypatch) -> None:
    decisions = {
        RouteDecision: RouteDecision(route="retrieve", reason="Factual corpus question."),
        GradeDecision: GradeDecision(sufficient=False, reason="Chunks miss the needed facts."),
        RewriteDecision: RewriteDecision(query="rewritten query"),
    }
    monkeypatch.setattr(graph, "_llm", lambda: FakeLLM(decisions))
    monkeypatch.setattr(graph.hybrid, "retrieve", lambda query, k=4: [make_document("a")])

    final_state = graph.build_graph().invoke({"question": "What is AirCover?"})

    assert final_state["answer"] == graph.INSUFFICIENT_EVIDENCE_ANSWER
    assert final_state["sources"] == []
    assert final_state["rewrite_count"] == MAX_REWRITES
    # Initial attempt plus one retrieval per rewrite.
    assert len(final_state["retrieval_attempts"]) == MAX_REWRITES + 1


def test_graph_out_of_scope_skips_retrieval_end_to_end(monkeypatch) -> None:
    decisions = {
        RouteDecision: RouteDecision(route="out_of_scope", reason="Different company."),
    }
    monkeypatch.setattr(graph, "_llm", lambda: FakeLLM(decisions))

    def fail_retrieve(query: str, k: int = 4) -> list[Document]:
        raise AssertionError("retrieval must not run for out_of_scope questions")

    monkeypatch.setattr(graph.hybrid, "retrieve", fail_retrieve)

    final_state = graph.build_graph().invoke({"question": "What is Tesla's revenue?"})

    assert "can only answer questions about the indexed corpus" in final_state["answer"]
    assert final_state["sources"] == []
    assert final_state["retrieval_attempts"] == []
