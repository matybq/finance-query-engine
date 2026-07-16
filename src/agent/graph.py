"""LangGraph agent for routed, self-correcting corpus Q&A."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.config import COLLECTION_NAME, get_settings
from src.generation import answer as generation
from src.generation.llm import build_llm
from src.observability.langsmith import configure_langsmith
from src.retrieval import hybrid

MAX_REWRITES = 2
# The agent retrieves deeper than the plain-RAG baseline (k=4) because measured fused
# ranks put answering chunks at 5-7 after query rewriting. The k=4 hallucination risk is
# handled structurally here by the router: out-of-scope questions never reach retrieval.
RETRIEVE_K = 6
INSUFFICIENT_EVIDENCE_ANSWER = (
    "I do not know based on the retrieved context. "
    "The indexed sources do not provide enough evidence to answer this question."
)
AGENT_RUN_CONFIG: RunnableConfig = {
    "run_name": "finance_query_agent",
    "tags": ["finance-query-engine", "agent"],
    "metadata": {"corpus": COLLECTION_NAME},
}


Route = Literal["retrieve", "direct", "out_of_scope"]
GradeRoute = Literal["generate", "rewrite", "insufficient_evidence"]
Source = tuple[str, str]
GradingAttempt = tuple[str, bool, str]
AgentEvent = dict[str, object]

_FINAL_NODES = {"generate", "direct", "out_of_scope", "insufficient_evidence"}
_TOKEN_NODES = {"generate", "direct"}


class AgentInput(TypedDict):
    question: str


class AgentState(TypedDict):
    question: str  # original question
    search_query: str  # the search query used to retrieve documents
    documents: list[Document]  # the retrieved documents
    answer: str  # the generated answer
    sources: list[Source]  # the sources of the retrieved documents
    rewrite_count: int  # the number of times the question has been rewritten
    route: Route  # the next route for the user's question
    route_reason: NotRequired[str]  # why the router chose the route
    sufficient: NotRequired[bool]  # whether the documents can answer the original question
    rewritten_queries: NotRequired[list[str]]  # the rewritten queries used to retrieve documents
    retrieval_attempts: NotRequired[list[tuple[str, list[str]]]]  # retrieval attempts and their results
    grading_attempts: NotRequired[list[GradingAttempt]]  # the grading decisions for each retrieval attempt


class RouteDecision(BaseModel):
    route: Route = Field(description="The next route for the user's question.")
    reason: str = Field(description="A brief explanation for the chosen route.")


class GradeDecision(BaseModel):
    sufficient: bool = Field(
        description="Whether the retrieved chunks contain enough information to answer the original question."
    )
    reason: str = Field(description="A brief explanation for the sufficiency decision.")


class RewriteDecision(BaseModel):
    query: str = Field(description="A concise rewritten retrieval query.")


@dataclass(frozen=True)
class AgentResult:
    answer: str
    sources: list[Source]
    route: Route
    route_reason: str
    rewritten_queries: list[str]
    retrieval_attempts: list[tuple[str, list[str]]]
    grading_attempts: list[GradingAttempt]
    retrieved_contexts: list[str]
    retrieved_chunk_ids: list[str]


def _parse_structured_output[StructuredOutput: BaseModel](
    model: type[StructuredOutput],
    value: object,
) -> StructuredOutput:
    return model.model_validate(value)


def _llm():
    return build_llm(get_settings())


def build_initial_state_update(question: str, decision: RouteDecision) -> dict:
    return {
        "search_query": question,
        "documents": [],
        "answer": "",
        "sources": [],
        "rewrite_count": 0,
        "route": decision.route,
        "route_reason": decision.reason,
        "rewritten_queries": [],
        "retrieval_attempts": [],
        "grading_attempts": [],
    }


def router_node(state: AgentState) -> dict:
    structured_llm = _llm().with_structured_output(RouteDecision)
    decision = _parse_structured_output(
        RouteDecision,
        structured_llm.invoke(
            [
                (
                    "system",
                    "Set the `route` field for a corpus-grounded finance assistant and set `reason` "
                    "to a brief explanation. Use direct ONLY for meta/help/greetings about the assistant "
                    "itself, such as 'hello' or 'what can you do?'. NEVER use direct for factual questions. "
                    "Use retrieve for factual questions about Airbnb or the indexed Airbnb 10-K FY2025 "
                    "corpus. The corpus is the Airbnb 10-K FY2025; questions may reference products, "
                    "programs, metrics, or terms from those documents without naming the company, and those "
                    "questions must route to retrieve. Do not require the question to include Airbnb or "
                    "10-K when it asks what a named item is; named items may be corpus products, programs, "
                    "metrics, or terms. Use out_of_scope ONLY when the question is clearly about a "
                    "different company or a topic unrelated to the indexed documents. When uncertain between "
                    "retrieve and out_of_scope, choose retrieve; downstream grading and grounding handle "
                    "irrelevant results.",
                ),
                ("human", state["question"]),
            ]
        ),
    )
    return build_initial_state_update(state["question"], decision)


def route_after_router(state: AgentState) -> Route:
    return state["route"]


def out_of_scope_node(state: AgentState) -> dict:
    return {
        "answer": (
            "I can only answer questions about the indexed corpus: Airbnb 10-K FY2025, Items 1, 1A, and 7."
        ),
        "sources": [],
    }


def direct_node(state: AgentState) -> dict:
    response = _llm().invoke(
        [
            (
                "system",
                "You are a corpus-grounded financial Q&A assistant. Answer briefly. "
                "Explain that you can answer questions about the indexed Airbnb 10-K FY2025 "
                "Items 1, 1A, and 7, and that factual answers use retrieved evidence and sources. "
                "Do not mention model training data, cutoff dates, or general web knowledge.",
            ),
            ("human", state["question"]),
        ]
    )
    return {"answer": str(response.content), "sources": []}


def retrieve_node(state: AgentState) -> dict:
    documents = hybrid.retrieve(state["search_query"], k=RETRIEVE_K)
    chunk_ids = [str(document.metadata.get("chunk_id", "")) for document in documents]
    attempts = list(state.get("retrieval_attempts", []))
    attempts.append((state["search_query"], chunk_ids))
    return {"documents": documents, "retrieval_attempts": attempts}


def _format_grade_context(documents: list[Document]) -> str:
    return "\n\n".join(f"Chunk {index}: {doc.page_content}" for index, doc in enumerate(documents, start=1))


def _format_grading_feedback(grading_attempts: list[GradingAttempt]) -> str:
    if not grading_attempts:
        return "- No previous grading feedback."

    return "\n".join(
        f"- Query: {query}\n  Sufficient: {sufficient}\n  Reason: {reason}"
        for query, sufficient, reason in grading_attempts
    )


def grade_node(state: AgentState) -> dict:
    if not state["documents"]:
        grading_attempts = list(state.get("grading_attempts", []))
        grading_attempts.append(
            (
                state["search_query"],
                False,
                "No documents were retrieved for this query.",
            )
        )
        return {"sufficient": False, "grading_attempts": grading_attempts}

    structured_llm = _llm().with_structured_output(GradeDecision)
    decision = _parse_structured_output(
        GradeDecision,
        structured_llm.invoke(
            [
                (
                    "system",
                    "Set the `sufficient` field to indicate whether the retrieved chunks contain enough "
                    "information to answer the original question, and set `reason` to a brief explanation. "
                    "Judge against the original question, not the retrieval query. Use sufficient=true only "
                    "when the chunks include the specific facts needed for a grounded answer.",
                ),
                (
                    "human",
                    f"Original question: {state['question']}\n\n"
                    f"Retrieved chunks:\n{_format_grade_context(state['documents'])}",
                ),
            ]
        ),
    )
    grading_attempts = list(state.get("grading_attempts", []))
    grading_attempts.append((state["search_query"], decision.sufficient, decision.reason))
    return {"sufficient": decision.sufficient, "grading_attempts": grading_attempts}


def route_after_grade(state: AgentState) -> GradeRoute:
    if state.get("sufficient"):
        return "generate"
    if state["rewrite_count"] < MAX_REWRITES:
        return "rewrite"
    return "insufficient_evidence"


def rewrite_node(state: AgentState) -> dict:
    previous_queries = [state["question"], *state.get("rewritten_queries", [])]
    grading_feedback = _format_grading_feedback(state.get("grading_attempts", []))
    structured_llm = _llm().with_structured_output(RewriteDecision)
    decision = _parse_structured_output(
        RewriteDecision,
        structured_llm.invoke(
            [
                (
                    "system",
                    "Set the `query` field to a concise keyword-style search query: a short noun phrase, "
                    "not a question. Use the grading feedback to add missing factual terms from the original "
                    "information need. Prefer specific filing/business terms over broad generic terms. "
                    "For business-model questions, include concrete earning-mechanism terms such as fees, "
                    "payments, revenue recognition, customers, hosts, or guests when relevant. Do not use "
                    "interrogative words or full sentences. Produce a query that is more specific than, and "
                    "meaningfully different from, all previous search attempts.",
                ),
                (
                    "human",
                    "Original question: "
                    f"{state['question']}\nCurrent search query: {state['search_query']}\n"
                    "Previous search attempts:\n"
                    + "\n".join(f"- {query}" for query in previous_queries)
                    + "\n\nGrading feedback:\n"
                    + grading_feedback,
                ),
            ]
        ),
    )
    rewritten_queries = list(state.get("rewritten_queries", []))
    rewritten_queries.append(decision.query)
    return {
        "search_query": decision.query,
        "rewrite_count": state["rewrite_count"] + 1,
        "rewritten_queries": rewritten_queries,
    }


def generate_node(state: AgentState) -> dict:
    result = generation.generate(state["question"], state["documents"])
    return {"answer": result.answer, "sources": result.sources}


def insufficient_evidence_node(state: AgentState) -> dict:
    return {"answer": INSUFFICIENT_EVIDENCE_ANSWER, "sources": []}


def build_graph():
    graph = StateGraph(AgentState, input_schema=AgentInput)

    graph.add_node("router", router_node)
    graph.add_node("direct", direct_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("insufficient_evidence", insufficient_evidence_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, ["retrieve", "direct", "out_of_scope"])
    graph.add_edge("direct", END)
    graph.add_edge("out_of_scope", END)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        ["generate", "rewrite", "insufficient_evidence"],
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("insufficient_evidence", END)

    return graph.compile()


_GRAPH = build_graph()


def _retrieve_event(update: dict) -> AgentEvent:
    documents = update.get("documents", [])
    attempts = update.get("retrieval_attempts", [])
    return {
        "type": "retrieve",
        "query": attempts[-1][0] if attempts else "",
        "count": len(documents),
        "sections": [
            {"source": source, "section": section} for source, section in generation.unique_sources(documents)
        ],
    }


def run_stream(question: str) -> Iterator[AgentEvent]:
    """Yield typed agent events while the graph executes, ending with a `done` event.

    Node-level events (`route`, `retrieve`, `grade`, `rewrite`) surface the agent's
    decisions as they happen; `token` events stream the answer text from the
    generation and direct nodes.
    """
    configure_langsmith(get_settings())
    route: Route | None = None
    answer_text = ""
    sources: list[Source] = []

    for mode, payload in _GRAPH.stream(
        {"question": question},
        config=AGENT_RUN_CONFIG,
        stream_mode=["updates", "messages"],
    ):
        if mode == "messages":
            chunk, metadata = payload
            content = chunk.content
            if metadata.get("langgraph_node") in _TOKEN_NODES and isinstance(content, str) and content:
                yield {"type": "token", "text": content}
            continue

        for node, update in payload.items():
            if node == "router":
                route = update["route"]
                yield {"type": "route", "route": route, "reason": update.get("route_reason", "")}
            elif node == "retrieve":
                yield _retrieve_event(update)
            elif node == "grade":
                grading_attempts = update.get("grading_attempts", [])
                yield {
                    "type": "grade",
                    "sufficient": bool(update.get("sufficient")),
                    "reason": grading_attempts[-1][2] if grading_attempts else "",
                }
            elif node == "rewrite":
                yield {"type": "rewrite", "query": update["search_query"]}
            elif node in _FINAL_NODES:
                answer_text = update.get("answer", "")
                sources = update.get("sources", [])

    yield {
        "type": "done",
        "answer": answer_text,
        "sources": [{"source": source, "section": section} for source, section in sources],
        "route": route,
    }


def run(question: str) -> AgentResult:
    configure_langsmith(get_settings())
    final_state = _GRAPH.invoke({"question": question}, config=AGENT_RUN_CONFIG)
    documents = final_state.get("documents", [])
    return AgentResult(
        answer=final_state["answer"],
        sources=final_state["sources"],
        route=final_state["route"],
        route_reason=final_state.get("route_reason", ""),
        rewritten_queries=final_state.get("rewritten_queries", []),
        retrieval_attempts=final_state.get("retrieval_attempts", []),
        grading_attempts=final_state.get("grading_attempts", []),
        retrieved_contexts=[document.page_content for document in documents],
        retrieved_chunk_ids=[str(document.metadata.get("chunk_id", "")) for document in documents],
    )
