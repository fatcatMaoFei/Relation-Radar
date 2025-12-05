"""
Reminder scanning logic for Relation Radar.

This module inspects existing Person / Event data and produces
human‑readable "reminder" items, such as:

- upcoming birthdays or anniversaries;
- friends that have not出现ed在最近一段时间的记录中；
- friends with连续负向情绪的事件记录。

当前实现专注于本地 CLI / 日志输出，不做任何调度或推送，
方便后续在 Web UI / 移动端复用相同的扫描逻辑。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.models import Event, Person  # noqa: E402
from backend.core.repositories import (  # noqa: E402
    EventRepository,
    PersonRepository,
)


NEGATIVE_EMOTION_KEYWORDS = [
    "sad",
    "angry",
    "upset",
    "anxious",
    "tired",
    "depressed",
    "低落",
    "难过",
    "生气",
    "焦虑",
    "委屈",
    "压力大",
    "不开心",
]


@dataclass
class Reminder:
    """
    Lightweight reminder item.

    kind: one of "birthday", "inactive", "emotion".
    message: human‑readable summary for CLI / logs / UI.
    related_events: subset of events that triggered this reminder.
    """

    kind: str
    person: Person
    message: str
    related_events: List[Event]


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        # Expect YYYY-MM-DD
        return date.fromisoformat(value.strip())
    except Exception:
        return None


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except Exception:
        return None


def _is_negative_emotion(emotion: Optional[str]) -> bool:
    if not emotion:
        return False
    text = emotion.lower()
    for keyword in NEGATIVE_EMOTION_KEYWORDS:
        if keyword in text:
            return True
    return False


def scan_upcoming_birthdays(
    person_repo: PersonRepository,
    *,
    today: Optional[date] = None,
    window_days: int = 14,
) -> List[Reminder]:
    """
    Find persons whose birthday is within the next N days.

    生日字段约定为 `YYYY-MM-DD`，只使用月/日进行对比。
    """
    today = today or date.today()
    reminders: List[Reminder] = []

    for person in person_repo.list_all():
        birthday_date = _parse_iso_date(person.birthday)
        if not birthday_date:
            continue

        # Move birthday to this year, or next year if already passed.
        try:
            next_birthday = birthday_date.replace(year=today.year)
        except ValueError:
            # Invalid day for this year (e.g., 02-29 in non‑leap year)
            continue

        if next_birthday < today:
            next_birthday = next_birthday.replace(year=today.year + 1)

        delta_days = (next_birthday - today).days
        if 0 <= delta_days <= window_days:
            message = (
                f"生日将在 {delta_days} 天后（{next_birthday.isoformat()}）。"
            )
            reminders.append(
                Reminder(
                    kind="birthday",
                    person=person,
                    message=message,
                    related_events=[],
                )
            )

    return reminders


def scan_inactive_friends(
    person_repo: PersonRepository,
    event_repo: EventRepository,
    *,
    now: Optional[datetime] = None,
    inactive_days: int = 90,
) -> List[Reminder]:
    """
    Find friends whose last event is older than `inactive_days`.
    """
    now = now or datetime.utcnow()
    reminders: List[Reminder] = []

    for person in person_repo.list_all():
        if person.id is None:
            continue

        events = event_repo.list_for_person(person.id, limit=1)
        if not events:
            # 没有任何记录的朋友先跳过，避免一上来就全部提醒。
            continue

        last_event = events[0]
        last_time = _parse_iso_datetime(last_event.occurred_at)
        if not last_time:
            continue

        days_since = (now - last_time).days
        if days_since >= inactive_days:
            message = f"已经 {days_since} 天没有记录与 TA 有新的互动，可以考虑主动联系一下。"
            reminders.append(
                Reminder(
                    kind="inactive",
                    person=person,
                    message=message,
                    related_events=[last_event],
                )
            )

    return reminders


def scan_negative_emotions(
    person_repo: PersonRepository,
    event_repo: EventRepository,
    *,
    now: Optional[datetime] = None,
    window_days: int = 30,
    min_events: int = 2,
) -> List[Reminder]:
    """
    Find friends with multiple negative‑emotion events in recent history.
    """
    now = now or datetime.utcnow()
    cutoff = now - timedelta(days=window_days)
    start_iso = cutoff.isoformat()

    reminders: List[Reminder] = []

    for person in person_repo.list_all():
        if person.id is None:
            continue

        events = event_repo.list_for_person(person.id, start=start_iso)
        negative_events = [e for e in events if _is_negative_emotion(e.emotion)]

        if len(negative_events) < min_events:
            continue

        message = (
            f"最近 {window_days} 天内有 {len(negative_events)} 条情绪偏负面的记录，"
            "可以多关心一下 TA 的状态。"
        )
        reminders.append(
            Reminder(
                kind="emotion",
                person=person,
                message=message,
                related_events=negative_events,
            )
        )

    return reminders


def collect_reminders(
    person_repo: PersonRepository,
    event_repo: EventRepository,
    *,
    today: Optional[date] = None,
    now: Optional[datetime] = None,
) -> List[Reminder]:
    """
    Run all reminder scanners and return a combined list.
    """
    today = today or date.today()
    now = now or datetime.utcnow()

    reminders: List[Reminder] = []
    reminders.extend(
        scan_upcoming_birthdays(person_repo, today=today, window_days=14)
    )
    reminders.extend(
        scan_inactive_friends(person_repo, event_repo, now=now, inactive_days=90)
    )
    reminders.extend(
        scan_negative_emotions(
            person_repo,
            event_repo,
            now=now,
            window_days=30,
            min_events=2,
        )
    )

    # Sort by kind and then person name for stable output.
    reminders.sort(key=lambda r: (r.kind, r.person.name))
    return reminders
