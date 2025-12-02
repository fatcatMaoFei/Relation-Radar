from __future__ import annotations

import json
from typing import Iterable, List, Optional

from .db import get_connection
from .models import Event, Person, Relationship


def _encode_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _decode_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item) for item in data]
    except Exception:
        pass
    return []


class PersonRepository:
    def create(self, person: Person) -> Person:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO persons (name, nickname, birthday, gender, tags, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    person.name,
                    person.nickname,
                    person.birthday,
                    person.gender,
                    _encode_list(person.tags) if person.tags else None,
                    person.notes,
                ),
            )
            conn.commit()
            person_id = int(cursor.lastrowid)
            return person.copy(update={"id": person_id})
        finally:
            conn.close()

    def get(self, person_id: int) -> Optional[Person]:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return Person(
                id=row["id"],
                name=row["name"],
                nickname=row["nickname"],
                birthday=row["birthday"],
                gender=row["gender"],
                tags=_decode_list(row["tags"]),
                notes=row["notes"],
            )
        finally:
            conn.close()

    def list_all(self) -> List[Person]:
        conn = get_connection()
        persons: List[Person] = []
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM persons ORDER BY id ASC")
            for row in cursor.fetchall():
                persons.append(
                    Person(
                        id=row["id"],
                        name=row["name"],
                        nickname=row["nickname"],
                        birthday=row["birthday"],
                        gender=row["gender"],
                        tags=_decode_list(row["tags"]),
                        notes=row["notes"],
                    )
                )
        finally:
            conn.close()
        return persons

    def update(self, person: Person) -> Person:
        if person.id is None:
            raise ValueError("person.id is required for update")

        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE persons
                SET name = ?, nickname = ?, birthday = ?, gender = ?, tags = ?, notes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    person.name,
                    person.nickname,
                    person.birthday,
                    person.gender,
                    _encode_list(person.tags) if person.tags else None,
                    person.notes,
                    person.id,
                ),
            )
            conn.commit()
            return person
        finally:
            conn.close()

    def delete(self, person_id: int) -> None:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            conn.commit()
        finally:
            conn.close()


class EventRepository:
    def create(self, event: Event) -> Event:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO events (
                    occurred_at,
                    raw_time_text,
                    event_type,
                    summary,
                    raw_text,
                    emotion,
                    preferences,
                    taboos,
                    tags,
                    embedding_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.occurred_at,
                    event.raw_time_text,
                    event.event_type,
                    event.summary,
                    event.raw_text,
                    event.emotion,
                    _encode_list(event.preferences) if event.preferences else None,
                    _encode_list(event.taboos) if event.taboos else None,
                    _encode_list(event.tags) if event.tags else None,
                    event.embedding_id,
                ),
            )
            event_id = int(cursor.lastrowid)

            for person_id in event.person_ids:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO event_persons (event_id, person_id)
                    VALUES (?, ?)
                    """,
                    (event_id, person_id),
                )

            conn.commit()

            return event.copy(update={"id": event_id})
        finally:
            conn.close()

    def _row_to_event(self, row, person_ids: List[int]) -> Event:
        return Event(
            id=row["id"],
            person_ids=person_ids,
            occurred_at=row["occurred_at"],
            raw_time_text=row["raw_time_text"],
            event_type=row["event_type"],
            summary=row["summary"],
            raw_text=row["raw_text"],
            emotion=row["emotion"],
            preferences=_decode_list(row["preferences"]),
            taboos=_decode_list(row["taboos"]),
            tags=_decode_list(row["tags"]),
            embedding_id=row["embedding_id"],
        )

    def get(self, event_id: int) -> Optional[Event]:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            if row is None:
                return None

            cursor.execute(
                "SELECT person_id FROM event_persons WHERE event_id = ?",
                (event_id,),
            )
            person_ids = [r["person_id"] for r in cursor.fetchall()]

            return self._row_to_event(row, person_ids)
        finally:
            conn.close()

    def list_for_person(
        self,
        person_id: int,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        conn = get_connection()
        events: List[Event] = []
        try:
            cursor = conn.cursor()
            query = """
                SELECT e.*
                FROM events e
                JOIN event_persons ep ON e.id = ep.event_id
                WHERE ep.person_id = ?
            """
            params: List[object] = [person_id]

            if start is not None:
                query += " AND e.occurred_at >= ?"
                params.append(start)
            if end is not None:
                query += " AND e.occurred_at <= ?"
                params.append(end)

            query += " ORDER BY e.occurred_at DESC, e.id DESC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, tuple(params))
            event_rows = cursor.fetchall()

            for row in event_rows:
                cursor.execute(
                    "SELECT person_id FROM event_persons WHERE event_id = ?",
                    (row["id"],),
                )
                person_ids = [r["person_id"] for r in cursor.fetchall()]
                events.append(self._row_to_event(row, person_ids))
        finally:
            conn.close()
        return events


class RelationshipRepository:
    def create(self, relationship: Relationship) -> Relationship:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO relationships (
                    person_a_id,
                    person_b_id,
                    score,
                    relation_type,
                    description
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    relationship.person_a_id,
                    relationship.person_b_id,
                    relationship.score,
                    relationship.relation_type,
                    relationship.description,
                ),
            )
            conn.commit()
            rel_id = int(cursor.lastrowid)
            return relationship.copy(update={"id": rel_id})
        finally:
            conn.close()

    def get(self, relationship_id: int) -> Optional[Relationship]:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM relationships WHERE id = ?",
                (relationship_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return Relationship(
                id=row["id"],
                person_a_id=row["person_a_id"],
                person_b_id=row["person_b_id"],
                score=row["score"],
                relation_type=row["relation_type"],
                description=row["description"],
            )
        finally:
            conn.close()

    def list_for_person(self, person_id: int) -> List[Relationship]:
        conn = get_connection()
        relationships: List[Relationship] = []
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM relationships
                WHERE person_a_id = ? OR person_b_id = ?
                ORDER BY id ASC
                """,
                (person_id, person_id),
            )
            for row in cursor.fetchall():
                relationships.append(
                    Relationship(
                        id=row["id"],
                        person_a_id=row["person_a_id"],
                        person_b_id=row["person_b_id"],
                        score=row["score"],
                        relation_type=row["relation_type"],
                        description=row["description"],
                    )
                )
        finally:
            conn.close()
        return relationships

    def update(self, relationship: Relationship) -> Relationship:
        if relationship.id is None:
            raise ValueError("relationship.id is required for update")

        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE relationships
                SET person_a_id = ?,
                    person_b_id = ?,
                    score = ?,
                    relation_type = ?,
                    description = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    relationship.person_a_id,
                    relationship.person_b_id,
                    relationship.score,
                    relationship.relation_type,
                    relationship.description,
                    relationship.id,
                ),
            )
            conn.commit()
            return relationship
        finally:
            conn.close()

    def delete(self, relationship_id: int) -> None:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM relationships WHERE id = ?",
                (relationship_id,),
            )
            conn.commit()
        finally:
            conn.close()
