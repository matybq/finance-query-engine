"""Grounded answer generation from retrieved document chunks."""

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.retrieval import hybrid


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[tuple[str, str]]


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

    if settings.llm_provider == "openai":
        llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)
    else:
        llm = ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
    response = llm.invoke(messages)

    sources = []
    seen = set()
    for doc in documents:
        source = (doc.metadata["source"], doc.metadata["section"])
        if source not in seen:
            seen.add(source)
            sources.append(source)

    return AnswerResult(answer=str(response.content), sources=sources)
