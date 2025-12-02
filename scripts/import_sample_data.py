from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.db import init_db  # noqa: E402
from backend.core.models import Event, Person, Relationship  # noqa: E402
from backend.core.repositories import (  # noqa: E402
    EventRepository,
    PersonRepository,
    RelationshipRepository,
)


def main() -> None:
    init_db()

    person_repo = PersonRepository()
    event_repo = EventRepository()
    relationship_repo = RelationshipRepository()

    # Sample persons
    alice = person_repo.create(
        Person(name="猫", nickname="Cat", tags=["朋友", "爱猫"])
    )
    bob = person_repo.create(
        Person(name="阿B", nickname="B", tags=["朋友", "健身"])
    )

    # Sample events
    event_repo.create(
        Event(
            person_ids=[alice.id or 0, bob.id or 0],
            occurred_at="2025-01-01T19:00:00",
            raw_time_text="元旦晚上",
            event_type="聚餐",
            summary="元旦一起吃日料，猫喜欢安静环境，阿B爱吃鱼",
            emotion="轻松愉快",
            preferences=["猫喜欢安静", "阿B喜欢吃鱼"],
            tags=["聚餐", "日料"],
        )
    )

    # Sample relationship
    relationship_repo.create(
        Relationship(
            person_a_id=alice.id or 0,
            person_b_id=bob.id or 0,
            score=0.9,
            relation_type="朋友",
            description="都爱宠物，一起健身和吃饭的好搭子",
        )
    )

    print("Inserted persons:")
    for person in person_repo.list_all():
        print(" -", person)

    print("\nEvents for 猫:")
    events_for_alice = event_repo.list_for_person(alice.id or 0)
    for event in events_for_alice:
        print(" -", event)

    print("\nRelationships for 猫:")
    for rel in relationship_repo.list_for_person(alice.id or 0):
        print(" -", rel)


if __name__ == "__main__":
    main()
