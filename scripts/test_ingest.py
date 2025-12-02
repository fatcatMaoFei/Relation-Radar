#!/usr/bin/env python3
"""
Test script for PR-0.1-07: Manual Text Ingestion
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.db import init_db  # noqa: E402
from backend.core.ingest import (  # noqa: E402
    TextExtractor,
    extract_events,
    ingest_manual,
)
from backend.core.models import Person  # noqa: E402
from backend.core.repositories import EventRepository, PersonRepository  # noqa: E402
from backend.rag.chains import ask_question  # noqa: E402


def test_text_extractor():
    """Test the TextExtractor class."""
    print("=" * 60)
    print("🔧 Testing TextExtractor")
    print("=" * 60)
    
    extractor = TextExtractor()
    
    # Test cases
    test_texts = [
        "今天和猫一起去吃火锅，她很开心，说喜欢麻辣口味",
        "昨天阿B健身很累，但他很满足，不喜欢有氧运动",
        "上周和小张聊天，发现她最近工作压力很大，有些焦虑",
        "刚才和朋友看电影，感觉很放松",
        "猫生日快到了，她喜欢安静的环境，讨厌太吵闹的地方",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text}")
        print("-" * 40)
        
        draft = extractor.extract(text)
        
        print(f"  时间: {draft.raw_time_text} → {draft.occurred_at}")
        print(f"  情绪: {draft.emotion}")
        print(f"  类型: {draft.event_type}")
        print(f"  偏好: {draft.preferences}")
        print(f"  忌讳: {draft.taboos}")
        print(f"  标签: {draft.tags}")
    
    print("\n✅ TextExtractor tests completed!")
    return True


def test_ingest_manual():
    """Test the ingest_manual function."""
    print("\n" + "=" * 60)
    print("🔧 Testing ingest_manual")
    print("=" * 60)
    
    # Initialize database
    init_db()
    
    # Create test person
    person_repo = PersonRepository()
    cat = person_repo.create(Person(name="测试猫", nickname="TestCat", tags=["测试"]))
    print(f"\n✅ Created test person: {cat.name} (ID: {cat.id})")
    
    # Test ingestion
    test_texts = [
        "今天和测试猫一起去吃川菜，她很开心，说喜欢麻辣的味道",
        "昨天测试猫说工作压力有点大，我们聊了很久",
    ]
    
    event_repo = EventRepository()
    created_events = []
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📥 Ingesting text {i}: {text[:40]}...")
        
        event = ingest_manual(person_ids=[cat.id], raw_text=text)
        created_events.append(event)
        
        print(f"  ✅ Event created (ID: {event.id})")
        print(f"     Summary: {event.summary[:50]}...")
        print(f"     Emotion: {event.emotion}")
        print(f"     Type: {event.event_type}")
        print(f"     Embedding ID: {event.embedding_id}")
    
    # Verify in database
    print("\n📊 Verifying database...")
    events = event_repo.list_for_person(cat.id)
    print(f"  Found {len(events)} events for {cat.name}")
    
    return len(created_events) == 2


def test_rag_integration():
    """Test that ingested events can be retrieved via RAG."""
    print("\n" + "=" * 60)
    print("🔧 Testing RAG Integration")
    print("=" * 60)
    
    # Get the test person
    person_repo = PersonRepository()
    persons = [p for p in person_repo.list_all() if p.name == "测试猫"]
    
    if not persons:
        print("❌ Test person not found, skipping RAG test")
        return False
    
    cat = persons[-1]  # Get the most recent one
    print(f"\n🔍 Querying for person: {cat.name} (ID: {cat.id})")
    
    # Ask questions
    questions = [
        "测试猫最近心情怎么样？",
        "测试猫喜欢吃什么？",
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        answer = ask_question(question, person_id=cat.id, top_k=3)
        print(f"🤖 Answer: {answer[:200]}...")
    
    print("\n✅ RAG integration test completed!")
    return True


def test_extract_events():
    """Test the extract_events function."""
    print("\n" + "=" * 60)
    print("🔧 Testing extract_events (AI placeholder)")
    print("=" * 60)
    
    text = "今天和朋友吃火锅，很开心，她喜欢麻辣锅底"
    
    drafts = extract_events(text)
    
    print(f"\n📝 Input text: {text}")
    print(f"📋 Extracted {len(drafts)} event(s):")
    
    for i, draft in enumerate(drafts, 1):
        print(f"\n  Event {i}:")
        print(f"    Summary: {draft.summary}")
        print(f"    Emotion: {draft.emotion}")
        print(f"    Type: {draft.event_type}")
        print(f"    Preferences: {draft.preferences}")
    
    print("\n✅ extract_events test completed!")
    print("📝 Note: Future AI integration will extract multiple events from a single text")
    return True


def main():
    """Run all ingestion tests."""
    print("🚀 Testing Manual Text Ingestion (PR-0.1-07)")
    print("=" * 60)
    
    try:
        extractor_passed = test_text_extractor()
        extract_passed = test_extract_events()
        ingest_passed = test_ingest_manual()
        rag_passed = test_rag_integration()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        print(f"  TextExtractor: {'✅ Passed' if extractor_passed else '❌ Failed'}")
        print(f"  extract_events: {'✅ Passed' if extract_passed else '❌ Failed'}")
        print(f"  ingest_manual: {'✅ Passed' if ingest_passed else '❌ Failed'}")
        print(f"  RAG Integration: {'✅ Passed' if rag_passed else '❌ Failed'}")
        
        if all([extractor_passed, extract_passed, ingest_passed, rag_passed]):
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
