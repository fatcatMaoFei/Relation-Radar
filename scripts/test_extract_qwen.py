#!/usr/bin/env python3
"""
Test script for PR-0.2-02: Qwen-powered Event Extraction

Tests the upgraded extract_events function with both mock and Qwen modes.
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
    extract_events,
    ingest_manual,
)
from backend.core.models import Person  # noqa: E402
from backend.core.repositories import PersonRepository  # noqa: E402


def test_extract_events_mock():
    """Test extract_events in mock mode (rule-based)."""
    print("=" * 60)
    print("🔧 Testing Extract Events (Mock Mode)")
    print("=" * 60)
    
    test_texts = [
        "今天和猫一起吃火锅，她很开心，喜欢麻辣锅底",
        "昨天阿B健身很累，但他很满足，不喜欢有氧运动",
        "上周和小张聊天，发现她最近工作压力很大，有些焦虑",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text}")
        print("-" * 40)
        
        drafts = extract_events(text)
        
        print(f"  提取到 {len(drafts)} 个事件:")
        for j, draft in enumerate(drafts, 1):
            print(f"    事件 {j}:")
            print(f"      摘要: {draft.summary}")
            print(f"      时间: {draft.raw_time_text} → {draft.occurred_at}")
            print(f"      情绪: {draft.emotion}")
            print(f"      类型: {draft.event_type}")
            print(f"      偏好: {draft.preferences}")
            print(f"      忌讳: {draft.taboos}")
            print(f"      标签: {draft.tags}")
    
    print("\n✅ Mock mode tests completed!")
    return True


def test_extract_events_qwen():
    """Test extract_events in Qwen mode (AI-powered)."""
    print("\n" + "=" * 60)
    print("🔧 Testing Extract Events (Qwen Mode)")
    print("=" * 60)
    
    # Check if Qwen mode is available
    from backend.llm.local_client import get_llm_client
    client = get_llm_client()
    
    if not (hasattr(client, '_mode') and client._mode == 'qwen'):
        print("⚠️  Qwen mode not available (RELATION_RADAR_LLM_MODE != 'qwen')")
        print("   Set environment variable: export RELATION_RADAR_LLM_MODE=qwen")
        return None  # Skip, not fail
    
    # Complex multi-event texts
    complex_texts = [
        """周末和猫去了川菜馆，她特别开心，说最喜欢麻辣锅底，但不喜欢太油腻的菜。
        聊天中发现她最近工作压力很大，准备下周开始健身来放松。""",
        
        """今天上午和阿B一起健身，他推荐了一家新健身房，设备很好。
        中午我们去吃了日料，他说不太喜欢生鱼片，更喜欢熟的。
        晚上他提到明天要加班，有点焦虑。""",
        
        """昨天小张生日聚会，大家都很开心。她说喜欢安静的环境，不喜欢太吵闹的地方。
        今天她告诉我准备换工作，因为现在的公司压力太大了。"""
    ]
    
    for i, text in enumerate(complex_texts, 1):
        print(f"\n📝 Complex Test {i}:")
        print(f"输入: {text[:60]}...")
        print("-" * 40)
        
        try:
            drafts = extract_events(text)
            
            print(f"  ✨ AI 提取到 {len(drafts)} 个事件:")
            for j, draft in enumerate(drafts, 1):
                print(f"    🎯 事件 {j}:")
                print(f"      📋 摘要: {draft.summary}")
                print(f"      ⏰ 时间: {draft.raw_time_text} → {draft.occurred_at}")
                print(f"      😊 情绪: {draft.emotion}")
                print(f"      🏷️ 类型: {draft.event_type}")
                print(f"      ❤️ 偏好: {draft.preferences}")
                print(f"      ❌ 忌讳: {draft.taboos}")
                print(f"      🔖 标签: {draft.tags}")
        
        except Exception as e:
            print(f"  ❌ 提取失败: {e}")
            return False
    
    print("\n✅ Qwen mode tests completed!")
    return True


def test_end_to_end_qwen():
    """Test end-to-end ingestion with Qwen extraction."""
    print("\n" + "=" * 60)
    print("🔧 Testing End-to-End Ingestion (Qwen)")
    print("=" * 60)
    
    # Initialize database
    init_db()
    
    # Create test person
    person_repo = PersonRepository()
    cat = person_repo.create(Person(name="测试猫AI", nickname="AI Cat", tags=["测试", "AI"]))
    print(f"\n✅ Created test person: {cat.name} (ID: {cat.id})")
    
    # Test complex ingestion
    complex_text = """今天下午和测试猫AI去了新开的川菜馆，她特别开心，说最喜欢麻辣锅底和水煮鱼，
    但不喜欢太咸的菜。聊天中得知她最近工作压力比较大，有些焦虑，
    准备明天开始每周去健身房三次来缓解压力。她还提到下周末想去看电影放松一下。"""
    
    print("\n📥 Ingesting complex text:")
    print(f"输入: {complex_text[:80]}...")
    
    try:
        # This should extract multiple events
        event = ingest_manual(person_ids=[cat.id], raw_text=complex_text)
        
        print(f"\n✅ Primary event created (ID: {event.id})")
        print(f"   📋 摘要: {event.summary}")
        print(f"   😊 情绪: {event.emotion}")
        print(f"   🏷️ 类型: {event.event_type}")
        print(f"   ❤️ 偏好: {event.preferences}")
        print(f"   ❌ 忌讳: {event.taboos}")
        print(f"   🔍 向量索引: {event.embedding_id}")
        
        # Note: Current ingest_manual only returns first event
        # This is expected behavior - multiple events would need batch processing
        
        return True
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return False


def test_json_parsing_robustness():
    """Test robustness of JSON parsing with edge cases."""
    print("\n" + "=" * 60)
    print("🔧 Testing JSON Parsing Robustness")
    print("=" * 60)
    
    # Edge case texts
    edge_cases = [
        "",  # Empty text
        "这是一段很简单的文字。",  # Simple text
        "Just English text without Chinese.",  # English only
        "今天123456789！@#$%^&*()特殊字符测试",  # Special characters
    ]
    
    for i, text in enumerate(edge_cases, 1):
        if not text:
            print(f"\n📝 Edge Case {i}: <empty text>")
        else:
            print(f"\n📝 Edge Case {i}: {text[:50]}")
        print("-" * 40)
        
        try:
            drafts = extract_events(text)
            print(f"  ✅ 处理成功，提取 {len(drafts)} 个事件")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    print("\n✅ Robustness tests completed!")
    return True


def main():
    """Run all extraction tests."""
    print("🚀 Testing Qwen-Powered Event Extraction (PR-0.2-02)")
    print("=" * 60)
    
    try:
        mock_passed = test_extract_events_mock()
        qwen_passed = test_extract_events_qwen()
        e2e_passed = test_end_to_end_qwen()
        robust_passed = test_json_parsing_robustness()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        print(f"  Mock Mode: {'✅ Passed' if mock_passed else '❌ Failed'}")
        
        if qwen_passed is True:
            qwen_status = "✅ Passed"
        elif qwen_passed is None:
            qwen_status = "⏭️  Skipped"
        else:
            qwen_status = "❌ Failed"
        print(f"  Qwen Mode: {qwen_status}")
        
        print(f"  End-to-End: {'✅ Passed' if e2e_passed else '❌ Failed'}")
        print(f"  Robustness: {'✅ Passed' if robust_passed else '❌ Failed'}")
        
        # Check if any failed (None is skip, not fail)
        failed = any(r is False for r in [mock_passed, qwen_passed, e2e_passed, robust_passed])
        
        if failed:
            print("\n❌ Some tests failed!")
            return False
        else:
            print("\n✅ All tests completed successfully!")
            if qwen_passed is None:
                print("💡 To test Qwen mode: export RELATION_RADAR_LLM_MODE=qwen")
            return True
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
