"""Grounded answer generation from retrieved document chunks."""

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.retrieval import hybrid


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[tuple[str, str]]


def build_llm(settings):
    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
        )

    if settings.llm_provider == "openrouter":
        return ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def answer(question: str, k: int = 4) -> AnswerResult:
    settings = get_settings()
    documents = hybrid.retrieve(question, k=k)
    context = "\n\n".join(
        f"Source: {doc.metadata['source']} ({doc.metadata['section']})\n{doc.page_content}"
        for doc in documents
    )
    messages = [
        (
            "system",
            "Answer only from the provided context. If the context is insufficient, say you do not know.",
        ),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    llm = build_llm(settings)
    response = llm.invoke(messages)

    sources = []
    seen_sources = set()
    for doc in documents:
        source_ref = (doc.metadata["source"], doc.metadata["section"])

        if source_ref not in seen_sources:
            seen_sources.add(source_ref)
            sources.append(source_ref)

    return AnswerResult(answer=str(response.content), sources=sources)
