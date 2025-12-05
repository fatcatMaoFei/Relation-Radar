"""
RAG Retriever for Relation Radar.

This module provides intelligent retrieval of events based on
semantic similarity, person filtering, time ranges, and tags.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.models import Event  # noqa: E402
from backend.core.repositories import EventRepository  # noqa: E402
from backend.rag.embeddings import get_embedding_client  # noqa: E402
from backend.rag.vector_store import get_vector_store  # noqa: E402


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""
    
    event_id: int
    content: str
    score: float
    person_ids: List[int]
    event_type: Optional[str] = None
    occurred_at: Optional[str] = None
    emotion: Optional[str] = None
    
    def to_context_string(self) -> str:
        """Convert to a formatted context string."""
        parts = []
        
        if self.occurred_at:
            parts.append(f"[时间: {self.occurred_at}]")
        if self.event_type:
            parts.append(f"[类型: {self.event_type}]")
        if self.emotion:
            parts.append(f"[情绪: {self.emotion}]")
        
        header = ' '.join(parts)
        if header:
            return f"{header}\n{self.content}"
        return self.content


class EventRetriever:
    """
    Intelligent retriever for events using vector similarity search
    with filtering capabilities.
    """
    
    def __init__(self):
        """Initialize the retriever with embedding client and vector store."""
        self.embedding_client = get_embedding_client()
        self.vector_store = get_vector_store()
        self.event_repo = EventRepository()
    
    def retrieve(
        self,
        query: str,
        person_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant events based on query and filters.
        
        Args:
            query: The search query text
            person_id: Filter by person ID (only retrieve events for this person)
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            tags: Filter by event tags
            top_k: Number of top results to return (will be clamped to [1, 50])
            
        Returns:
            List of retrieved documents with scores
        """
        # Clamp top_k to a reasonable range
        if top_k <= 0:
            top_k = 5
        elif top_k > 50:
            top_k = 50

        # Step 1: Get all events matching the filters from database
        candidate_events = self._get_candidate_events(
            person_id=person_id,
            start_date=start_date,
            end_date=end_date,
            tags=tags
        )
        
        if not candidate_events:
            return []
        
        # Step 2: Compute query embedding
        query_vector = self.embedding_client.encode(query)
        
        # Step 3: Search vector store with metadata filter
        # Build metadata filter for ChromaDB
        where_filter = None
        if person_id is not None:
            # ChromaDB metadata filter
            where_filter = {"person_id": str(person_id)}
        
        # Get more results to filter later
        ids, distances, metadatas = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k * 3,  # Get more to filter
            where=where_filter
        )
        
        # Step 4: Build retrieved documents
        retrieved_docs = []
        candidate_event_ids = {e.id for e in candidate_events if e.id}
        
        for doc_id, distance, metadata in zip(ids, distances, metadatas):
            # Try to parse event_id from doc_id
            try:
                event_id = int(doc_id.split('_')[-1]) if '_' in doc_id else int(doc_id)
            except (ValueError, IndexError):
                continue
            
            # Check if this event is in our filtered candidates
            if candidate_event_ids and event_id not in candidate_event_ids:
                continue
            
            # Get full event details
            event = self.event_repo.get(event_id)
            if not event:
                continue
            
            # Calculate similarity score (convert distance to similarity)
            # ChromaDB returns cosine distance, so similarity = 1 - distance
            similarity_score = 1.0 - min(distance, 1.0)
            
            # Build content string
            content = self._build_event_content(event)
            
            retrieved_docs.append(RetrievedDocument(
                event_id=event_id,
                content=content,
                score=similarity_score,
                person_ids=event.person_ids,
                event_type=event.event_type,
                occurred_at=event.occurred_at or event.raw_time_text,
                emotion=event.emotion
            ))
            
            if len(retrieved_docs) >= top_k:
                break
        
        # Sort by score (highest first)
        retrieved_docs.sort(key=lambda x: x.score, reverse=True)
        
        return retrieved_docs[:top_k]
    
    def retrieve_for_person(
        self,
        query: str,
        person_id: int,
        top_k: int = 5
    ) -> List[RetrievedDocument]:
        """
        Convenience method to retrieve events for a specific person.
        
        Args:
            query: The search query text
            person_id: The person ID to filter by
            top_k: Number of top results to return
            
        Returns:
            List of retrieved documents
        """
        return self.retrieve(query=query, person_id=person_id, top_k=top_k)

    def retrieve_for_persons(
        self,
        query: str,
        person_ids: Iterable[int],
        top_k: int = 5,
    ) -> List[RetrievedDocument]:
        """
        Retrieve events for multiple persons at once.

        当前实现策略：
        - 在向量空间里全局检索 top_k * 5 条候选；
        - 仅保留 person_ids 有交集的事件；
        - 按相似度排序后返回前 top_k 条。
        """
        person_id_set: Set[int] = {int(pid) for pid in person_ids}
        if not person_id_set:
            return []

        # Clamp top_k
        if top_k <= 0:
            top_k = 5
        elif top_k > 50:
            top_k = 50

        query_vector = self.embedding_client.encode(query)

        ids, distances, metadatas = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k * 5,
            where=None,
        )

        retrieved_docs: List[RetrievedDocument] = []

        for doc_id, distance, metadata in zip(ids, distances, metadatas):
            try:
                event_id = int(doc_id.split("_")[-1]) if "_" in doc_id else int(doc_id)
            except (ValueError, IndexError):
                continue

            event = self.event_repo.get(event_id)
            if not event:
                continue

            if not any(pid in person_id_set for pid in event.person_ids):
                continue

            similarity_score = 1.0 - min(distance, 1.0)
            content = self._build_event_content(event)

            retrieved_docs.append(
                RetrievedDocument(
                    event_id=event_id,
                    content=content,
                    score=similarity_score,
                    person_ids=event.person_ids,
                    event_type=event.event_type,
                    occurred_at=event.occurred_at or event.raw_time_text,
                    emotion=event.emotion,
                ),
            )

            if len(retrieved_docs) >= top_k:
                break

        retrieved_docs.sort(key=lambda x: x.score, reverse=True)
        return retrieved_docs[:top_k]
    
    def _get_candidate_events(
        self,
        person_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Event]:
        """
        Get candidate events from database based on filters.
        """
        if person_id is not None:
            # Get events for specific person
            events = self.event_repo.list_for_person(
                person_id=person_id,
                start=start_date,
                end=end_date
            )
        else:
            # Get all events (with optional time filter)
            events = self.event_repo.list_for_person(
                person_id=0,  # This might need adjustment based on repo implementation
                start=start_date,
                end=end_date
            )
        
        # Filter by tags if specified
        if tags:
            events = [
                e for e in events
                if any(tag in e.tags for tag in tags)
            ]
        
        return events
    
    def _build_event_content(self, event: Event) -> str:
        """
        Build a searchable content string from an event.
        """
        parts = []
        
        if event.summary:
            parts.append(event.summary)
        
        if event.raw_text:
            parts.append(event.raw_text)
        
        if event.preferences:
            parts.append(f"偏好：{'、'.join(event.preferences)}")
        
        if event.taboos:
            parts.append(f"忌讳：{'、'.join(event.taboos)}")
        
        return '\n'.join(parts) if parts else "无详细内容"
    
    def index_event(self, event: Event) -> str:
        """
        Index an event into the vector store.
        
        Args:
            event: The event to index
            
        Returns:
            The document ID in the vector store
        """
        if not event.id:
            raise ValueError("Event must have an ID to be indexed")
        
        # Build content for embedding
        content = self._build_event_content(event)
        
        # Create document ID
        doc_id = f"event_{event.id}"
        
        # Build metadata
        metadata = {
            "event_id": str(event.id),
            "event_type": event.event_type or "",
            "emotion": event.emotion or "",
        }
        
        # Add person_id for filtering (use first person if multiple)
        if event.person_ids:
            metadata["person_id"] = str(event.person_ids[0])
        
        # Add to vector store
        self.vector_store.add_documents(
            texts=[content],
            ids=[doc_id],
            metadatas=[metadata]
        )
        
        return doc_id


# Global instance
_retriever = None


def get_retriever() -> EventRetriever:
    """
    Get a global singleton instance of the retriever.
    
    Returns:
        Global EventRetriever instance
    """
    global _retriever
    if _retriever is None:
        _retriever = EventRetriever()
    return _retriever
