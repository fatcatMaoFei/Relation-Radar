from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.rag.retriever import get_retriever

"""
Tool: search_events

为远端大模型提供安全的“查事实”接口：
- 可以按人 / 问句检索本地事件；
- 只返回必要摘要，不暴露整段原始文本。
"""


def _truncate(text: str, max_len: int = 160) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def search_events_tool(
    *,
    person_id: Optional[int] = None,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search events using the existing RAG retriever.

    Parameters
    ----------
    person_id:
        Optional person id to filter events. If None, search across all persons.
    query:
        Natural language query.
    top_k:
        Number of results to return.

    Returns
    -------
    List of dicts, each containing:
        - event_id: int
        - score: float
        - occurred_at: str | None
        - event_type: str | None
        - emotion: str | None
        - person_ids: list[int]
        - snippet: short text snippet summarising the event
    """
    retriever = get_retriever()
    if person_id is not None:
        docs = retriever.retrieve_for_person(
            query=query,
            person_id=person_id,
            top_k=top_k,
        )
    else:
        docs = retriever.retrieve(query=query, top_k=top_k)

    results: List[Dict[str, Any]] = []
    for doc in docs:
        results.append(
            {
                "event_id": doc.event_id,
                "score": float(doc.score),
                "occurred_at": doc.occurred_at,
                "event_type": doc.event_type,
                "emotion": doc.emotion,
                "person_ids": list(doc.person_ids),
                "snippet": _truncate(doc.to_context_string()),
            },
        )
    return results
