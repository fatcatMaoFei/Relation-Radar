from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from backend.core.repositories import EventRepository, PersonRepository

"""
Tool: get_person_summary

为远端大模型提供“某个朋友的结构化画像”：
- 基于 Person 基本信息 + 最近若干事件；
- 汇总偏好 / 忌讳 / 情绪等字段；
- 不返回完整原始记录，只返回必要摘要。
"""


def get_person_summary_tool(person_id: int, *, max_events: int = 20) -> Dict[str, Any]:
    """
    Build a structured summary for a given person.

    Returns
    -------
    dict with keys:
      - person: {id, name, tags, notes}
      - preferences: List[str]
      - taboos: List[str]
      - recent_events: List[{occurred_at, summary, emotion}]
    """
    person_repo = PersonRepository()
    event_repo = EventRepository()

    person = person_repo.get(person_id)
    if person is None:
        raise ValueError(f"Person {person_id} not found")

    events = event_repo.list_for_person(person_id=person_id, limit=max_events)

    pref_counter: Counter[str] = Counter()
    taboo_counter: Counter[str] = Counter()

    recent_events: List[Dict[str, Any]] = []
    for ev in events:
        for p in ev.preferences or []:
            pref_counter[p] += 1
        for t in ev.taboos or []:
            taboo_counter[t] += 1

        recent_events.append(
            {
                "event_id": ev.id,
                "occurred_at": ev.occurred_at or ev.raw_time_text,
                "summary": ev.summary or "",
                "emotion": ev.emotion,
            },
        )

    preferences = [item for item, _ in pref_counter.most_common(10)]
    taboos = [item for item, _ in taboo_counter.most_common(10)]

    return {
        "person": {
            "id": person.id,
            "name": person.name,
            "tags": person.tags,
            "notes": person.notes,
        },
        "preferences": preferences,
        "taboos": taboos,
        "recent_events": recent_events,
    }
