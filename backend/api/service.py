from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.db import init_db  # noqa: E402
from backend.core.models import Event, Person  # noqa: E402
from backend.core.repositories import (  # noqa: E402
    EventRepository,
    PersonRepository,
)


app = FastAPI(
    title="Relation Radar API",
    description="HTTP API for querying persons and events in Relation Radar.",
    version="0.2.0",
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database on application startup."""
    init_db()


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


def get_app() -> FastAPI:
    """Convenience accessor used by scripts/tests."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.service:app", host="127.0.0.1", port=8000, reload=False)

