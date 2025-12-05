#!/usr/bin/env python3
"""
Test script for PR-0.2-09: Feedback flow.

This script does not drive the interactive CLI. Instead it verifies that:
- the feedback table exists in SQLite;
- FeedbackRepository.create / list_recent work end-to-end.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.db import init_db  # noqa: E402
from backend.core.models import Feedback  # noqa: E402
from backend.core.repositories import FeedbackRepository  # noqa: E402


def main() -> bool:
    print("Testing feedback flow (PR-0.2-09)")
    print("=" * 60)

    init_db()

    repo = FeedbackRepository()

    sample = Feedback(
        person_id=None,
        question="测试问题：猫喜欢吃什么？",
        answer="根据记录，猫喜欢麻辣和日料。",
        used_context_event_ids=[1, 2, 3],
        rating="accurate",
    )
    saved = repo.create(sample)
    print(f"Created feedback id={saved.id}")

    recent = repo.list_recent(limit=5)
    print(f"Recent feedback count: {len(recent)}")
    for fb in recent:
        print(
            f"- id={fb.id}, rating={fb.rating}, "
            f"person_id={fb.person_id}, context_ids={fb.used_context_event_ids}",
        )

    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
