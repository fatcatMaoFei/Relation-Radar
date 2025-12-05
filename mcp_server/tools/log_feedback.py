from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.core.models import Feedback
from backend.core.repositories import FeedbackRepository

"""
Tool: log_feedback

把远端 LLM 或用户对某次回答的评价写入 Feedback 表，
用于后续统计与本地小模型微调。
"""


def log_feedback_tool(
    *,
    person_id: Optional[int],
    question: str,
    answer: str,
    rating: str,
    used_context_event_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Log feedback for a QA answer.

    Parameters
    ----------
    person_id:
        Optional person id the answer is about.
    question:
        Original question text.
    answer:
        Answer text that is being evaluated.
    rating:
        One of: "accurate", "inaccurate", "risky".
    used_context_event_ids:
        Optional list of event ids that were used as context.

    Returns
    -------
    Dict representing the saved Feedback row.
    """
    feedback = Feedback(
        person_id=person_id,
        question=question,
        answer=answer,
        used_context_event_ids=used_context_event_ids or [],
        rating=rating,
    )
    repo = FeedbackRepository()
    saved = repo.create(feedback)

    return {
        "id": saved.id,
        "person_id": saved.person_id,
        "rating": saved.rating,
        "used_context_event_ids": saved.used_context_event_ids,
        "created_at": saved.created_at,
    }
