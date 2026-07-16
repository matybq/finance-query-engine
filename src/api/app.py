"""FastAPI serving layer over the LangGraph agent."""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.graph import Route, run

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Finance Query Engine",
    description="Evidence-grounded Q&A over the indexed financial filing corpus.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="Natural-language question over the indexed corpus.")


class SourceRef(BaseModel):
    source: str
    section: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    route: Route


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be blank")

    try:
        result = run(question)
    except Exception as error:
        logger.exception("Agent failed to process question")
        raise HTTPException(
            status_code=502,
            detail=f"The agent failed to process the question ({error.__class__.__name__}).",
        ) from error

    return AskResponse(
        answer=result.answer,
        sources=[SourceRef(source=source, section=section) for source, section in result.sources],
        route=result.route,
    )
