"""
QA Chain for Relation Radar.

This module implements the question-answering pipeline that combines
retrieval with LLM generation.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm.local_client import Message, get_llm_client  # noqa: E402
from backend.llm.prompts import build_qa_rag_prompt  # noqa: E402
from backend.rag.retriever import RetrievedDocument, get_retriever  # noqa: E402


@dataclass
class QAResult:
    """Result of a QA query."""
    
    question: str
    answer: str
    retrieved_contexts: List[RetrievedDocument]
    person_id: Optional[int] = None
    
    def format_full_response(self) -> str:
        """Format the complete response with contexts."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append(f"🔍 问题: {self.question}")
        if self.person_id:
            lines.append(f"👤 查询对象: Person ID {self.person_id}")
        lines.append("=" * 60)
        
        # Retrieved contexts
        lines.append("\n📄 检索到的相关记录:")
        lines.append("-" * 40)
        
        if self.retrieved_contexts:
            for i, doc in enumerate(self.retrieved_contexts, 1):
                lines.append(f"\n[{i}] 相关度: {doc.score:.2%}")
                lines.append(doc.to_context_string())
        else:
            lines.append("未找到相关记录")
        
        # Answer
        lines.append("\n" + "-" * 40)
        lines.append("🤖 回答:")
        lines.append(self.answer)
        lines.append("=" * 60)
        
        return '\n'.join(lines)


class QAChain:
    """
    Question-Answering chain that combines retrieval with LLM generation.
    
    Pipeline: question -> retriever -> context assembly -> LLM -> answer
    """
    
    def __init__(self):
        """Initialize the QA chain with retriever and LLM client."""
        self.retriever = get_retriever()
        self.llm_client = get_llm_client()
    
    def ask(
        self,
        question: str,
        person_id: Optional[int] = None,
        top_k: int = 5
    ) -> QAResult:
        """
        Ask a question and get an answer based on retrieved context.
        
        Args:
            question: The question to ask
            person_id: Optional person ID to filter events
            top_k: Number of context documents to retrieve
            
        Returns:
            QAResult with answer and retrieved contexts
        """
        # Step 1: Retrieve relevant documents
        if person_id is not None:
            retrieved_docs = self.retriever.retrieve_for_person(
                query=question,
                person_id=person_id,
                top_k=top_k
            )
        else:
            retrieved_docs = self.retriever.retrieve(
                query=question,
                top_k=top_k
            )
        
        # Step 2: Build context from retrieved documents
        context = self._build_context(retrieved_docs)
        
        # Step 3: Build prompt
        prompt = self._build_prompt(question, context)
        
        # Step 4: Generate answer using LLM
        answer = self.llm_client.generate(prompt)
        
        # Return result
        return QAResult(
            question=question,
            answer=answer,
            retrieved_contexts=retrieved_docs,
            person_id=person_id
        )
    
    def chat(
        self,
        question: str,
        person_id: Optional[int] = None,
        top_k: int = 5
    ) -> QAResult:
        """
        Chat-style question answering with system context.
        
        Args:
            question: The user's question
            person_id: Optional person ID to filter events
            top_k: Number of context documents to retrieve
            
        Returns:
            QAResult with answer and retrieved contexts
        """
        # Step 1: Retrieve relevant documents
        if person_id is not None:
            retrieved_docs = self.retriever.retrieve_for_person(
                query=question,
                person_id=person_id,
                top_k=top_k
            )
        else:
            retrieved_docs = self.retriever.retrieve(
                query=question,
                top_k=top_k
            )
        
        # Step 2: Build context from retrieved documents
        context = self._build_context(retrieved_docs)
        
        # Step 3: Build chat messages
        messages = self._build_chat_messages(question, context)
        
        # Step 4: Generate answer using LLM chat
        answer = self.llm_client.chat(messages)
        
        # Return result
        return QAResult(
            question=question,
            answer=answer,
            retrieved_contexts=retrieved_docs,
            person_id=person_id
        )
    
    def _build_context(self, retrieved_docs: List[RetrievedDocument]) -> str:
        """
        Build context string from retrieved documents.
        """
        if not retrieved_docs:
            return "没有找到相关的记录。"
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[记录{i}] {doc.to_context_string()}")
        
        return '\n\n'.join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build the prompt for the LLM.
        """
        return build_qa_rag_prompt(question=question, context=context)
    
    def _build_chat_messages(
        self,
        question: str,
        context: str
    ) -> List[Message]:
        """
        Build chat messages for the LLM.
        """
        system_message = Message(
            role="system",
            content=f"""你是一个帮助用户管理人际关系的助手。

相关记录：
{context}

请根据这些记录回答用户的问题。如果记录中没有相关信息，请如实说明。"""
        )
        
        user_message = Message(
            role="user",
            content=question
        )
        
        return [system_message, user_message]


# Global instance
_qa_chain = None


def get_qa_chain() -> QAChain:
    """
    Get a global singleton instance of the QA chain.
    
    Returns:
        Global QAChain instance
    """
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = QAChain()
    return _qa_chain


def ask_question(
    question: str,
    person_id: Optional[int] = None,
    top_k: int = 5,
    verbose: bool = False
) -> str:
    """
    Convenience function to ask a question.
    
    Args:
        question: The question to ask
        person_id: Optional person ID to filter events
        top_k: Number of context documents to retrieve
        verbose: If True, return full formatted response
        
        Returns:
        Answer string (or full response if verbose)
    """
    chain = get_qa_chain()
    result = chain.ask(question, person_id=person_id, top_k=top_k)
    
    if verbose:
        return result.format_full_response()
    return result.answer
