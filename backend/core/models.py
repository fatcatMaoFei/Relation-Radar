
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Person(BaseModel):
    id: Optional[int] = None
    name: str
    nickname: Optional[str] = None
    birthday: Optional[str] = None
    gender: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class Event(BaseModel):
    id: Optional[int] = None
    person_ids: List[int] = Field(default_factory=list)
    occurred_at: Optional[str] = None
    raw_time_text: Optional[str] = None
    event_type: Optional[str] = None
    summary: Optional[str] = None
    raw_text: Optional[str] = None
    emotion: Optional[str] = None
    preferences: List[str] = Field(default_factory=list)
    taboos: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    embedding_id: Optional[str] = None


class Relationship(BaseModel):
    id: Optional[int] = None
    person_a_id: int
    person_b_id: int
    score: Optional[float] = None
    relation_type: Optional[str] = None
    description: Optional[str] = None


class Feedback(BaseModel):
    """
    User feedback on a single QA answer.

    rating 建议使用固定取值：
    - "accurate"   表示回答准确、有帮助；
    - "inaccurate" 表示回答不准或误导；
    - "risky"      表示回答有潜在风险（隐私 / 伦理等）。
    """

    id: Optional[int] = None
    person_id: Optional[int] = None
    question: str
    answer: str
    used_context_event_ids: List[int] = Field(default_factory=list)
    rating: str
    created_at: Optional[str] = None

