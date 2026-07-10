"""LangGraph agent for routed, self-correcting corpus Q&A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypeVar, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.config import get_settings
from src.generation import answer as generation
from src.generation.llm import build_llm
from src.retrieval import hybrid


MAX_REWRITES = 2
# The agent retrieves deeper than the plain-RAG baseline (k=4) because measured fused ranks put answering chunks at 5–7 after query rewriting.
# The Fase 2 k=4 hallucination risk is handled structurally here by the router: out-of-scope questions never reach retrieval.
RETRIEVE_K = 6


Route = Literal["retrieve", "direct", "out_of_scope"]
Source = tuple[str, str]
GradingAttempt = tuple[str, bool, str]
StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


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
    sufficient: NotRequired[bool]  # whether the retrieved documents contain enough information to answer the original question
    rewritten_queries: NotRequired[list[str]]  # the rewritten queries used to retrieve documents
    retrieval_attempts: NotRequired[list[tuple[str, list[str]]]]  # the retrieval attempts and their results
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


def _parse_structured_output(
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
                    "Use retrieve for factual questions about Airbnb or the indexed Airbnb 10-K FY2025 corpus. "
                    "The corpus is the Airbnb 10-K FY2025; questions may reference products, programs, "
                    "metrics, or terms from those documents without naming the company, and those questions "
                    "must route to retrieve. Do not require the question to include Airbnb or 10-K when it "
                    "asks what a named item is; named items may be corpus products, programs, metrics, or terms. "
                    "Use out_of_scope ONLY when the question is clearly about a "
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
            "I can only answer questions about the indexed corpus: Airbnb 10-K FY2025, "
            "Items 1, 1A, and 7."
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
                "Items 1, 1A, and 7, and that factual answers use retrieved evidence and sources.",
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
    return "\n\n".join(
        f"Chunk {index}: {doc.page_content}" for index, doc in enumerate(documents, start=1)
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
                    "Set the `sufficient` field to indicate whether the retrieved chunks contain enough information "
                    "to answer the original question, and set `reason` to a brief explanation. Judge against "
                    "the original question, not the retrieval query. Use sufficient=true only when the chunks "
                    "include the specific facts needed for a grounded answer.",
                ),
                (
                    "human",
                    f"Original question: {state['question']}\n\nRetrieved chunks:\n{_format_grade_context(state['documents'])}",
                ),
            ]
        ),
    )
    grading_attempts = list(state.get("grading_attempts", []))
    grading_attempts.append((state["search_query"], decision.sufficient, decision.reason))
    return {"sufficient": decision.sufficient, "grading_attempts": grading_attempts}


def route_after_grade(state: AgentState) -> Literal["generate", "rewrite"]:
    if state.get("sufficient"):
        return "generate"
    if state["rewrite_count"] < MAX_REWRITES:
        return "rewrite"
    return "generate"


def rewrite_node(state: AgentState) -> dict:
    previous_queries = [state["question"], *state.get("rewritten_queries", [])]
    structured_llm = _llm().with_structured_output(RewriteDecision)
    decision = _parse_structured_output(
        RewriteDecision,
        structured_llm.invoke(
            [
                (
                    "system",
                    "Set the `query` field to a concise keyword-style search query: a short noun phrase, "
                    "not a question. Do not use interrogative words or full sentences. Use formal terminology "
                    "as found in SEC 10-K filings instead of colloquial wording. Produce a query that differs "
                    "meaningfully from all previous search attempts.",
                ),
                (
                    "human",
                    "Original question: "
                    f"{state['question']}\nCurrent search query: {state['search_query']}\n"
                    "Previous search attempts:\n"
                    + "\n".join(f"- {query}" for query in previous_queries),
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


def build_graph():
    graph = StateGraph(AgentState, input_schema=AgentInput)

    graph.add_node("router", router_node)
    graph.add_node("direct", direct_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, ["retrieve", "direct", "out_of_scope"])
    graph.add_edge("direct", END)
    graph.add_edge("out_of_scope", END)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, ["generate", "rewrite"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


_GRAPH = build_graph()


def run(question: str) -> AgentResult:
    final_state = _GRAPH.invoke({"question": question})
    return AgentResult(
        answer=final_state["answer"],
        sources=final_state["sources"],
        route=final_state["route"],
        route_reason=final_state.get("route_reason", ""),
        rewritten_queries=final_state.get("rewritten_queries", []),
        retrieval_attempts=final_state.get("retrieval_attempts", []),
        grading_attempts=final_state.get("grading_attempts", []),
    )
