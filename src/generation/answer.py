"""Grounded answer generation from retrieved document chunks."""

from dataclasses import dataclass

from langchain_core.documents import Document

from src.config import get_settings
from src.generation.llm import build_llm
from src.retrieval import hybrid


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[tuple[str, str]]


def format_context(documents: list[Document]) -> str:
    """Format retrieved chunks as source-labeled context for the LLM."""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')} ({doc.metadata.get('section', 'unknown')})\n{doc.page_content}"
        for doc in documents
    )


def build_messages(context: str, question: str) -> list[tuple[str, str]]:
    """Build the grounded generation prompt."""
    return [
        (
            "system",
            "Answer only from the provided context. If the context is insufficient, say you do not know.",
        ),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]


def unique_sources(documents: list[Document]) -> list[tuple[str, str]]:
    """Return unique source-section pairs while preserving retrieval order."""
    sources = []
    seen_sources = set()

    for doc in documents:
        source_ref = (doc.metadata.get("source", "unknown"), doc.metadata.get("section", "unknown"))

        if source_ref not in seen_sources:
            seen_sources.add(source_ref)
            sources.append(source_ref)

    return sources


def generate(question: str, documents: list[Document]) -> AnswerResult:
    """Generate a grounded answer from already-retrieved documents."""
    settings = get_settings()
    context = format_context(documents)
    messages = build_messages(context, question)

    llm = build_llm(settings)
    response = llm.invoke(messages)

    return AnswerResult(answer=str(response.content), sources=unique_sources(documents))


def answer(question: str, k: int = 4) -> AnswerResult:
    """Retrieve documents and generate an answer for the plain RAG baseline."""
    documents = hybrid.retrieve(question, k=k)
    return generate(question, documents)
