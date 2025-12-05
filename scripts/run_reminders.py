from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.db import init_db  # noqa: E402
from backend.core.reminders import (  # noqa: E402
    Reminder,
    collect_reminders,
)
from backend.core.repositories import (  # noqa: E402
    EventRepository,
    PersonRepository,
)


def format_reminder(reminder: Reminder) -> str:
    prefix = {
        "birthday": "[BIRTHDAY]",
        "inactive": "[INACTIVE]",
        "emotion": "[EMOTION]",
    }.get(reminder.kind, "[REMINDER]")

    return f"{prefix} {reminder.person.name}: {reminder.message}"


def main() -> None:
    """
    Run reminder scan and print suggestions to stdout.

    这一脚本用于本地手动/定时任务，例如：
        python scripts/run_reminders.py
    """
    init_db()

    person_repo = PersonRepository()
    event_repo = EventRepository()

    reminders: List[Reminder] = collect_reminders(person_repo, event_repo)

    if not reminders:
        print("No reminders for now.")
        return

    print("Relation Radar Reminders")
    print("========================")
    for item in reminders:
        print(format_reminder(item))


if __name__ == "__main__":
    main()
