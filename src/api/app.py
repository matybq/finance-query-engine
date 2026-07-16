"""FastAPI serving layer over the LangGraph agent."""

import json
import logging
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent.graph import Route, run, run_stream

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


def _validated_question(request: AskRequest) -> str:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be blank")
    return question


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    question = _validated_question(request)

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


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    question = _validated_question(request)

    def event_source() -> Iterator[str]:
        try:
            for event in run_stream(question):
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        except Exception as error:
            logger.exception("Agent failed while streaming")
            detail = f"The agent failed to process the question ({error.__class__.__name__})."
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'detail': detail})}\n\n"

    # X-Accel-Buffering tells nginx not to buffer the event stream.
    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
