#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.ingest import ingest_manual  # noqa: E402
from backend.core.models import Event, Person  # noqa: E402
from backend.core.repositories import EventRepository, PersonRepository  # noqa: E402
from backend.rag.chains import get_qa_chain  # noqa: E402


def add_person(args) -> None:
    """Add a new person to the database."""
    tags = []
    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(",")]
    
    person = Person(
        name=args.name,
        nickname=args.nickname,
        birthday=args.birthday,
        gender=args.gender,
        tags=tags,
        notes=args.notes,
    )
    
    repo = PersonRepository()
    created_person = repo.create(person)
    print(f"Created person: {created_person.name} (ID: {created_person.id})")


def add_event(args) -> None:
    """Add a new event associated with one or more persons."""
    person_ids = [int(pid.strip()) for pid in args.person_ids.split(",")]
    
    # Verify all person IDs exist
    repo = PersonRepository()
    for pid in person_ids:
        person = repo.get(pid)
        if person is None:
            print(f"Error: Person with ID {pid} does not exist")
            return
    
    tags = []
    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(",")]
    
    preferences = []
    if args.preferences:
        preferences = [pref.strip() for pref in args.preferences.split(",")]
    
    taboos = []
    if args.taboos:
        taboos = [taboo.strip() for taboo in args.taboos.split(",")]
    
    event = Event(
        person_ids=person_ids,
        occurred_at=args.occurred_at,
        raw_time_text=args.raw_time_text,
        event_type=args.event_type,
        summary=args.summary,
        raw_text=args.raw_text,
        emotion=args.emotion,
        preferences=preferences,
        taboos=taboos,
        tags=tags,
    )
    
    event_repo = EventRepository()
    created_event = event_repo.create(event)
    print(f"Created event: {created_event.summary or 'Untitled'} (ID: {created_event.id})")
    print(f"Associated with persons: {', '.join(map(str, created_event.person_ids))}")


def list_events(args) -> None:
    """List events for a specific person."""
    repo = EventRepository()
    events = repo.list_for_person(
        person_id=args.person_id,
        start=args.start_date,
        end=args.end_date,
        limit=args.limit,
    )
    
    if not events:
        print(f"No events found for person ID {args.person_id}")
        return
    
    print(f"Found {len(events)} event(s) for person ID {args.person_id}:")
    print()
    
    for event in events:
        print(f"ID: {event.id}")
        print(f"Summary: {event.summary or 'N/A'}")
        print(f"Type: {event.event_type or 'N/A'}")
        print(f"Time: {event.occurred_at or event.raw_time_text or 'N/A'}")
        print(f"Emotion: {event.emotion or 'N/A'}")
        if event.preferences:
            print(f"Preferences: {', '.join(event.preferences)}")
        if event.taboos:
            print(f"Taboos: {', '.join(event.taboos)}")
        if event.tags:
            print(f"Tags: {', '.join(event.tags)}")
        print(f"Person IDs: {', '.join(map(str, event.person_ids))}")
        print("-" * 40)


def ask_question(args) -> None:
    """Ask a question about a person using RAG."""
    qa_chain = get_qa_chain()
    
    # Get person name if person_id is provided
    person_name = None
    if args.person_id:
        repo = PersonRepository()
        person = repo.get(args.person_id)
        if person:
            person_name = person.name
        else:
            print(f"Warning: Person with ID {args.person_id} not found")
    
    # Ask the question
    result = qa_chain.ask(
        question=args.question,
        person_id=args.person_id,
        top_k=args.top_k or 5
    )
    
    # Display results
    print("\n" + "=" * 60)
    print(f"🔍 问题: {result.question}")
    if args.person_id:
        print(f"👤 查询对象: {person_name or 'Unknown'} (ID: {args.person_id})")
    print("=" * 60)
    
    # Show retrieved contexts if verbose
    if args.verbose:
        print("\n📄 检索到的相关记录:")
        print("-" * 40)
        
        if result.retrieved_contexts:
            for i, doc in enumerate(result.retrieved_contexts, 1):
                print(f"\n[{i}] 相关度: {doc.score:.2%}")
                print(doc.to_context_string())
        else:
            print("未找到相关记录")
        
        print("\n" + "-" * 40)
    
    # Show answer
    print("\n🤖 回答:")
    print(result.answer)
    print("=" * 60)


def ingest_text(args) -> None:
    """Ingest raw text and create an event."""
    person_ids = [int(pid.strip()) for pid in args.person_ids.split(",")]
    
    # Verify all person IDs exist
    repo = PersonRepository()
    for pid in person_ids:
        person = repo.get(pid)
        if person is None:
            print(f"Error: Person with ID {pid} does not exist")
            return
    
    # Get person names for display
    person_names = [repo.get(pid).name for pid in person_ids]
    
    try:
        event = ingest_manual(
            person_ids=person_ids,
            raw_text=args.text,
            auto_index=True
        )
        
        print("\n✅ 事件录入成功!")
        print("=" * 40)
        print(f"ID: {event.id}")
        print(f"关联人物: {', '.join(person_names)}")
        print(f"摘要: {event.summary}")
        if event.emotion:
            print(f"情绪: {event.emotion}")
        if event.event_type:
            print(f"类型: {event.event_type}")
        if event.preferences:
            print(f"偏好: {', '.join(event.preferences)}")
        if event.taboos:
            print(f"忌讳: {', '.join(event.taboos)}")
        if event.tags:
            print(f"标签: {', '.join(event.tags)}")
        print(f"向量索引: {event.embedding_id}")
        print("=" * 40)
        print("💡 现在可以通过 'ask' 命令查询这条记录了!")
        
    except ValueError as e:
        print(f"Error: {e}")


def list_persons(args) -> None:
    """List all persons in the database."""
    repo = PersonRepository()
    persons = repo.list_all()
    
    if not persons:
        print("No persons found in the database")
        return
    
    print(f"Found {len(persons)} person(s):")
    print()
    
    for person in persons:
        print(f"ID: {person.id}")
        print(f"Name: {person.name}")
        if person.nickname:
            print(f"Nickname: {person.nickname}")
        if person.birthday:
            print(f"Birthday: {person.birthday}")
        if person.gender:
            print(f"Gender: {person.gender}")
        if person.tags:
            print(f"Tags: {', '.join(person.tags)}")
        if person.notes:
            print(f"Notes: {person.notes}")
        print("-" * 40)


def ask(args) -> None:
    """Ask a question using the RAG QA chain."""
    question = args.question.strip()
    if not question:
        print("Error: question cannot be empty")
        return

    person_id = args.person_id
    verbose = bool(args.verbose)

    try:
        answer = ask_question(
            question=question,
            person_id=person_id,
            top_k=args.top_k,
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error while running QA chain: {exc}")
        return

    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="relation-radar",
        description="Relation Radar CLI - Manage your personal relationships",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # add-person command
    person_parser = subparsers.add_parser("add-person", help="Add a new person")
    person_parser.add_argument("name", help="Person's name")
    person_parser.add_argument("--nickname", help="Person's nickname")
    person_parser.add_argument("--birthday", help="Person's birthday (YYYY-MM-DD format)")
    person_parser.add_argument("--gender", help="Person's gender")
    person_parser.add_argument("--tags", help="Comma-separated tags (e.g., 'friend,colleague')")
    person_parser.add_argument("--notes", help="Additional notes")
    person_parser.set_defaults(func=add_person)
    
    # add-event command
    event_parser = subparsers.add_parser("add-event", help="Add a new event")
    event_parser.add_argument("person_ids", help="Comma-separated person IDs")
    event_parser.add_argument("--summary", help="Event summary")
    event_parser.add_argument("--occurred-at", help="Event time (ISO format: YYYY-MM-DDTHH:MM:SS)")
    event_parser.add_argument("--raw-time-text", help="Raw time description (e.g., 'yesterday evening')")
    event_parser.add_argument("--event-type", help="Event type (e.g., 'chat', 'meeting', 'gift')")
    event_parser.add_argument("--raw-text", help="Raw event text/description")
    event_parser.add_argument("--emotion", help="Emotional state (e.g., 'happy', 'stressed')")
    event_parser.add_argument("--preferences", help="Comma-separated preferences")
    event_parser.add_argument("--taboos", help="Comma-separated taboos")
    event_parser.add_argument("--tags", help="Comma-separated tags")
    event_parser.set_defaults(func=add_event)
    
    # list-events command
    events_parser = subparsers.add_parser("list-events", help="List events for a person")
    events_parser.add_argument("person_id", type=int, help="Person ID")
    events_parser.add_argument("--start-date", help="Start date filter (ISO format)")
    events_parser.add_argument("--end-date", help="End date filter (ISO format)")
    events_parser.add_argument("--limit", type=int, help="Maximum number of events to return")
    events_parser.set_defaults(func=list_events)
    
    # list-persons command
    persons_parser = subparsers.add_parser("list-persons", help="List all persons")
    persons_parser.set_defaults(func=list_persons)

    # ask command
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question based on recorded events",
    )
    ask_parser.add_argument(
        "question",
        help="Question to ask (in natural language)",
    )
    ask_parser.add_argument(
        "--person-id",
        type=int,
        help="Optional person ID to focus the question on a specific person",
    )
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of context records to retrieve (default: 5)",
    )
    ask_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed context and formatted answer",
    )
    ask_parser.set_defaults(func=ask)
    
    
    # ingest command (text ingestion)
    ingest_parser = subparsers.add_parser("ingest", help="Ingest raw text and create an event")
    ingest_parser.add_argument("person_ids", help="Comma-separated person IDs")
    ingest_parser.add_argument("text", help="Raw text to ingest")
    ingest_parser.set_defaults(func=ingest_text)
    
    args = parser.parse_args()
    
    if not hasattr(args, "func"):
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
