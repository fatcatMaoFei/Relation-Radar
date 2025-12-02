#!/usr/bin/env python3
"""
Test script for PR-0.1-06: RAG Retriever & QA Chain
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.db import init_db  # noqa: E402
from backend.core.models import Event, Person  # noqa: E402
from backend.core.repositories import EventRepository, PersonRepository  # noqa: E402
from backend.rag.chains import ask_question, get_qa_chain  # noqa: E402
from backend.rag.retriever import get_retriever  # noqa: E402


def setup_test_data():
    """Set up test data for RAG testing."""
    print("📦 Setting up test data...")
    
    # Initialize database
    init_db()
    
    person_repo = PersonRepository()
    event_repo = EventRepository()
    retriever = get_retriever()
    
    # Create test persons
    cat = person_repo.create(Person(name="猫", nickname="Cat", tags=["朋友", "爱猫"]))
    ab = person_repo.create(Person(name="阿B", nickname="B", tags=["朋友", "健身"]))
    
    print(f"  Created: {cat.name} (ID: {cat.id})")
    print(f"  Created: {ab.name} (ID: {ab.id})")
    
    # Create test events
    events_data = [
        Event(
            person_ids=[cat.id or 0],
            occurred_at="2025-01-15T12:00:00",
            event_type="聚餐",
            summary="猫今天心情很好，我们一起去吃了川菜，她很喜欢麻辣的味道",
            emotion="开心",
            preferences=["麻辣食物", "川菜"],
        ),
        Event(
            person_ids=[cat.id or 0],
            occurred_at="2025-01-20T19:00:00",
            event_type="聊天",
            summary="和猫聊天时发现她最近压力有点大，工作上的事情让她有些焦虑",
            emotion="焦虑",
        ),
        Event(
            person_ids=[ab.id or 0],
            occurred_at="2025-01-18T08:00:00",
            event_type="健身",
            summary="阿B昨天健身很累，不过他说感觉很有成就感，下次还想一起去",
            emotion="满足",
            preferences=["健身", "运动"],
        ),
        Event(
            person_ids=[ab.id or 0],
            occurred_at="2025-01-22T10:00:00",
            event_type="推荐",
            summary="阿B推荐了一家新的健身房，设备很好，环境也不错",
            emotion="积极",
            preferences=["新健身房"],
        ),
        Event(
            person_ids=[cat.id or 0, ab.id or 0],
            occurred_at="2025-01-25T19:00:00",
            event_type="聚餐",
            summary="元旦一起吃日料，猫喜欢安静环境，阿B爱吃鱼",
            emotion="轻松愉快",
            preferences=["猫喜欢安静", "阿B喜欢吃鱼"],
        ),
    ]
    
    created_events = []
    for event in events_data:
        created_event = event_repo.create(event)
        created_events.append(created_event)
        print(f"  Created event: {created_event.summary[:30]}... (ID: {created_event.id})")
        
        # Index event into vector store
        doc_id = retriever.index_event(created_event)
        print(f"    Indexed as: {doc_id}")
    
    return cat, ab, created_events


def test_retriever(cat, ab):
    """Test the retriever functionality."""
    print("\n" + "=" * 60)
    print("🔧 Testing Retriever")
    print("=" * 60)
    
    retriever = get_retriever()
    
    # Test 1: Retrieve events for 猫
    print("\n📍 Test 1: Retrieve events for 猫 (query: '心情')")
    docs = retriever.retrieve_for_person(query="心情", person_id=cat.id, top_k=3)
    
    print(f"Found {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. Score: {doc.score:.2%}")
        print(f"     Content: {doc.content[:60]}...")
        print(f"     Person IDs: {doc.person_ids}")
    
    # Validate: All should be for 猫
    all_for_cat = all(cat.id in doc.person_ids for doc in docs)
    print(f"\n✅ All results for 猫: {all_for_cat}")
    
    # Test 2: Retrieve events for 阿B
    print("\n📍 Test 2: Retrieve events for 阿B (query: '健身')")
    docs = retriever.retrieve_for_person(query="健身", person_id=ab.id, top_k=3)
    
    print(f"Found {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. Score: {doc.score:.2%}")
        print(f"     Content: {doc.content[:60]}...")
    
    # Validate: All should be for 阿B
    all_for_ab = all(ab.id in doc.person_ids for doc in docs)
    print(f"\n✅ All results for 阿B: {all_for_ab}")
    
    return all_for_cat and all_for_ab


def test_qa_chain(cat, ab):
    """Test the QA chain functionality."""
    print("\n" + "=" * 60)
    print("🔧 Testing QA Chain")
    print("=" * 60)
    
    qa_chain = get_qa_chain()
    
    # Test 1: Ask about 猫's mood
    print("\n📍 Test 1: Ask about 猫's mood")
    result = qa_chain.ask("猫最近心情怎么样？", person_id=cat.id, top_k=3)
    print(result.format_full_response())
    
    # Test 2: Ask about 阿B's activities
    print("\n📍 Test 2: Ask about 阿B's activities")
    result = qa_chain.ask("阿B最近在做什么？", person_id=ab.id, top_k=3)
    print(result.format_full_response())
    
    # Test 3: Ask general question (no person filter)
    print("\n📍 Test 3: General question (no person filter)")
    result = qa_chain.ask("最近有什么聚餐？", top_k=5)
    print(result.format_full_response())
    
    return True


def test_convenience_function(cat):
    """Test the convenience function."""
    print("\n" + "=" * 60)
    print("🔧 Testing Convenience Function")
    print("=" * 60)
    
    # Simple answer
    print("\n📍 Simple answer:")
    answer = ask_question("猫喜欢吃什么？", person_id=cat.id)
    print(f"Answer: {answer}")
    
    # Verbose answer
    print("\n📍 Verbose answer:")
    full_response = ask_question("猫喜欢吃什么？", person_id=cat.id, verbose=True)
    print(full_response)
    
    return True


def main():
    """Run all RAG tests."""
    print("🚀 Testing RAG Retriever & QA Chain (PR-0.1-06)")
    print("=" * 60)
    
    try:
        # Setup test data
        cat, ab, events = setup_test_data()
        
        # Run tests
        retriever_passed = test_retriever(cat, ab)
        qa_passed = test_qa_chain(cat, ab)
        convenience_passed = test_convenience_function(cat)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        print(f"  Retriever Tests: {'✅ Passed' if retriever_passed else '❌ Failed'}")
        print(f"  QA Chain Tests: {'✅ Passed' if qa_passed else '❌ Failed'}")
        print(f"  Convenience Function: {'✅ Passed' if convenience_passed else '❌ Failed'}")
        
        if retriever_passed and qa_passed and convenience_passed:
            print("\n✅ All tests completed successfully!")
            return True
        else:
            print("\n❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
