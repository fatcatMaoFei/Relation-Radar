from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.db import init_db  # noqa: E402
from backend.core.ingest import ingest_manual  # noqa: E402
from backend.core.models import Event, Person  # noqa: E402
from backend.core.repositories import (  # noqa: E402
    EventRepository,
    PersonRepository,
)
from backend.rag.chains import get_qa_chain  # noqa: E402


app = FastAPI(
    title="Relation Radar API",
    description="HTTP API for querying persons and events in Relation Radar.",
    version="0.2.0",
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database on application startup."""
    init_db()


class AskRequest(BaseModel):
    """Request payload for /persons/{id}/ask."""

    question: str = Field(..., description="Natural language question to ask")
    top_k: Optional[int] = Field(
        default=None,
        description="Number of context records to retrieve (optional, default from backend).",
    )


class AskResponse(BaseModel):
    """Response payload for /persons/{id}/ask."""

    question: str
    person_id: int
    answer: str
    used_context_event_ids: List[int] = Field(default_factory=list)


class CreateEventRequest(BaseModel):
    """Request payload for creating a new text event for a person."""

    text: str = Field(..., description="Raw text describing the event")


@app.get("/persons", response_model=List[Person])
def list_persons(tag: Optional[str] = Query(default=None, description="Filter by tag")) -> List[Person]:
    """
    List all persons, optionally filtered by tag.
    """
    repo = PersonRepository()
    persons = repo.list_all()

    if tag:
        tag = tag.strip()
        persons = [p for p in persons if tag in (p.tags or [])]

    return persons


@app.get("/persons/{person_id}", response_model=Person)
def get_person(person_id: int) -> Person:
    """
    Get a single person by ID.
    """
    repo = PersonRepository()
    person = repo.get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@app.get("/persons/{person_id}/events", response_model=List[Event])
def list_person_events(
    person_id: int,
    start: Optional[str] = Query(default=None, description="Start datetime (ISO)"),
    end: Optional[str] = Query(default=None, description="End datetime (ISO)"),
    tag: Optional[str] = Query(default=None, description="Filter by event tag"),
    limit: Optional[int] = Query(default=None, description="Max number of events"),
) -> List[Event]:
    """
    List events for a specific person, with optional time and tag filters.
    """
    person_repo = PersonRepository()
    if person_repo.get(person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")

    event_repo = EventRepository()
    events = event_repo.list_for_person(
        person_id=person_id,
        start=start,
        end=end,
        limit=limit,
    )

    if tag:
        tag = tag.strip()
        events = [e for e in events if tag in (e.tags or [])]

    return events


@app.post("/persons/{person_id}/ask", response_model=AskResponse)
def ask_person_question(person_id: int, payload: AskRequest) -> AskResponse:
    """
    Ask a question about a specific person using the RAG + LLM chain.

    This endpoint is a thin wrapper over the backend QAChain. It is designed
    so that Web / 移动端可以直接调用，无需了解内部实现细节。
    """
    person_repo = PersonRepository()
    person = person_repo.get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    qa_chain = get_qa_chain()
    top_k = payload.top_k if payload.top_k is not None else 5

    result = qa_chain.ask(question=question, person_id=person_id, top_k=top_k)
    used_ids = [doc.event_id for doc in result.retrieved_contexts]

    return AskResponse(
        question=result.question,
        person_id=person_id,
        answer=result.answer,
        used_context_event_ids=used_ids,
    )


@app.post("/persons/{person_id}/events", response_model=Event)
def create_person_event(person_id: int, payload: CreateEventRequest) -> Event:
    """
    Create a new text-based event for a specific person.

    这是 Web / 移动端录入的最小入口：给定一个朋友和一段自然语言文本，
    由后端负责抽取结构化字段并写入数据库 + 向量库。
    """
    person_repo = PersonRepository()
    if person_repo.get(person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        event = ingest_manual([person_id], text, auto_index=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest event: {exc}",
        ) from exc

    return event


def get_app() -> FastAPI:
    """Convenience accessor used by scripts/tests."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.service:app", host="127.0.0.1", port=8000, reload=False)

