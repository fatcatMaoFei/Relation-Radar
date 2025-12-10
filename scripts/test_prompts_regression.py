#!/usr/bin/env python3
"""
Prompt Regression Test Script (PR-0.3-03)

用于测试和对比不同场景下的提示词效果。
固定一组典型用例，方便人工评估提示词质量。

使用方法:
    # 测试所有场景（mock模式）
    python scripts/test_prompts_regression.py
    
    # 使用真实Qwen测试
    RELATION_RADAR_LLM_MODE=qwen python scripts/test_prompts_regression.py
    
    # 只测试特定场景
    python scripts/test_prompts_regression.py --scenario gift
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm.prompts import (  # noqa: E402
    build_qa_rag_prompt,
    build_multi_person_qa_prompt,
    build_gift_suggestion_prompt,
    build_emotion_care_prompt,
    build_person_summary_prompt,
    build_teacher_qa_prompt,
    get_prompt_version,
    get_prompt_stats,
    list_available_prompts,
)
from backend.llm.local_client import get_llm_client  # noqa: E402


# ==================== 测试用例定义 ====================

class TestCase:
    """测试用例"""
    def __init__(
        self,
        name: str,
        scenario: str,
        prompt_builder: callable,
        prompt_args: dict,
        expected_keywords: List[str],
        avoid_keywords: List[str] = None,
        description: str = ""
    ):
        self.name = name
        self.scenario = scenario
        self.prompt_builder = prompt_builder
        self.prompt_args = prompt_args
        self.expected_keywords = expected_keywords
        self.avoid_keywords = avoid_keywords or []
        self.description = description


# 测试数据：模拟的上下文记录
SAMPLE_CONTEXTS = {
    "cat_food": """
事件1 [2025-12-01] 聚餐：和猫一起吃川菜，她很开心，说最喜欢麻辣锅底和水煮鱼，但不喜欢太油腻的菜。
事件2 [2025-11-28] 聊天：猫提到她最近在减肥，尽量少吃甜食和油炸食品。
事件3 [2025-11-20] 生日聚会：猫的生日，大家送了蛋糕，她说虽然在减肥但生日可以例外。
""",
    "cat_emotion": """
事件1 [2025-12-08] 聊天：猫说最近工作压力很大，经常加班到很晚，有些焦虑。
事件2 [2025-12-05] 微信：猫发消息说感觉很累，想找时间放松一下。
事件3 [2025-12-01] 聚餐：虽然工作压力大，但和朋友聚餐时她还是很开心的。
""",
    "multi_person": """
【关于猫的记录】
事件1：猫喜欢吃川菜和麻辣口味，不喜欢太油腻的食物。
事件2：猫最近在减肥，尽量少吃甜食。

【关于阿B的记录】
事件1：阿B喜欢清淡口味，不太能吃辣。
事件2：阿B对海鲜过敏，不能吃虾蟹。
""",
    "gift_context": """
事件1：猫喜欢看书，最近在读心理学相关的书籍。
事件2：猫喜欢安静的环境，周末经常去咖啡馆。
事件3：猫提到想学瑜伽放松身心。
事件4：猫不喜欢太花哨的东西，偏好简约风格。
""",
}

# 定义测试用例
TEST_CASES = [
    # 场景1：饮食偏好问答
    TestCase(
        name="饮食偏好查询",
        scenario="food",
        prompt_builder=build_qa_rag_prompt,
        prompt_args={
            "question": "猫喜欢吃什么？有什么忌口？",
            "context": SAMPLE_CONTEXTS["cat_food"]
        },
        expected_keywords=["麻辣", "川菜", "不喜欢", "油腻", "减肥"],
        avoid_keywords=["我认为", "可能喜欢"],
        description="测试基于记录的饮食偏好回答"
    ),
    
    # 场景2：送礼建议
    TestCase(
        name="生日送礼建议",
        scenario="gift",
        prompt_builder=build_gift_suggestion_prompt,
        prompt_args={
            "person_name": "猫",
            "context": SAMPLE_CONTEXTS["gift_context"],
            "occasion": "生日",
            "budget": "200-500元"
        },
        expected_keywords=["书", "瑜伽", "简约"],
        avoid_keywords=["花哨", "不确定"],
        description="测试基于偏好的礼物推荐"
    ),
    
    # 场景3：情绪关怀
    TestCase(
        name="情绪关怀建议",
        scenario="emotion",
        prompt_builder=build_emotion_care_prompt,
        prompt_args={
            "person_name": "猫",
            "context": SAMPLE_CONTEXTS["cat_emotion"],
            "recent_emotion": "焦虑"
        },
        expected_keywords=["压力", "放松", "关心"],
        avoid_keywords=["抑郁症", "心理治疗"],
        description="测试情绪关怀建议的温和度"
    ),
    
    # 场景4：多人场景
    TestCase(
        name="多人聚餐建议",
        scenario="multi",
        prompt_builder=build_multi_person_qa_prompt,
        prompt_args={
            "question": "想约猫和阿B一起吃饭，去什么餐厅比较合适？",
            "context": SAMPLE_CONTEXTS["multi_person"],
            "person_names": ["猫", "阿B"]
        },
        expected_keywords=["冲突", "辣", "清淡", "海鲜"],
        avoid_keywords=[],
        description="测试多人需求平衡"
    ),
    
    # 场景5：人物画像
    TestCase(
        name="人物画像生成",
        scenario="summary",
        prompt_builder=build_person_summary_prompt,
        prompt_args={
            "person_name": "猫",
            "events_summary": "最近和猫聚餐了3次，她都选择川菜馆。工作比较忙，偶尔会抱怨压力大。",
            "preferences": ["麻辣口味", "安静环境", "阅读"],
            "taboos": ["油腻食物", "甜食（减肥中）"]
        },
        expected_keywords=["川菜", "压力", "安静"],
        avoid_keywords=["我猜", "应该是"],
        description="测试人物画像的准确性"
    ),
    
    # 场景6：Teacher问答
    TestCase(
        name="Teacher专业回答",
        scenario="teacher",
        prompt_builder=build_teacher_qa_prompt,
        prompt_args={
            "question": "猫最近心情不好，我应该怎么关心她？",
            "facts": "猫最近工作压力大，经常加班，有焦虑情绪。她喜欢安静的环境和阅读。",
            "local_answer": "可以约她出来聊聊天，听她倾诉。"
        },
        expected_keywords=["压力", "关心", "倾听"],
        avoid_keywords=["抑郁", "看医生"],
        description="测试Teacher模型的专业度"
    ),
]


# ==================== 测试执行 ====================

def run_test_case(test_case: TestCase, client) -> dict:
    """执行单个测试用例"""
    # 构建prompt
    prompt = test_case.prompt_builder(**test_case.prompt_args)
    
    # 调用LLM
    response = client.generate(prompt, max_tokens=512)
    
    # 检查关键词
    found_expected = [kw for kw in test_case.expected_keywords if kw in response]
    found_avoid = [kw for kw in test_case.avoid_keywords if kw in response]
    
    # 计算得分
    expected_score = len(found_expected) / len(test_case.expected_keywords) if test_case.expected_keywords else 1.0
    avoid_penalty = len(found_avoid) / len(test_case.avoid_keywords) if test_case.avoid_keywords else 0.0
    final_score = max(0, expected_score - avoid_penalty * 0.5)
    
    return {
        "name": test_case.name,
        "scenario": test_case.scenario,
        "description": test_case.description,
        "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
        "response": response,
        "expected_keywords": test_case.expected_keywords,
        "found_expected": found_expected,
        "avoid_keywords": test_case.avoid_keywords,
        "found_avoid": found_avoid,
        "score": final_score,
    }


def print_result(result: dict, verbose: bool = False):
    """打印测试结果"""
    score = result["score"]
    if score >= 0.8:
        status = "✅"
    elif score >= 0.5:
        status = "⚠️"
    else:
        status = "❌"
    
    print(f"\n{'='*60}")
    print(f"{status} {result['name']} (场景: {result['scenario']})")
    print(f"{'='*60}")
    print(f"📝 {result['description']}")
    print(f"📊 得分: {score:.0%}")
    print(f"   - 期望关键词: {result['expected_keywords']}")
    print(f"   - 找到: {result['found_expected']}")
    if result["avoid_keywords"]:
        print(f"   - 应避免: {result['avoid_keywords']}")
        if result["found_avoid"]:
            print(f"   - ⚠️ 发现应避免的: {result['found_avoid']}")
    
    print("\n🤖 AI回答:")
    print("-" * 40)
    # 截断过长的回答
    response = result["response"]
    if len(response) > 500:
        print(response[:500] + "...")
    else:
        print(response)
    print("-" * 40)
    
    if verbose:
        print("\n📜 Prompt预览:")
        print(result["prompt_preview"])


def run_all_tests(
    scenarios: Optional[List[str]] = None,
    verbose: bool = False
) -> dict:
    """运行所有测试"""
    print("🚀 Prompt回归测试 (PR-0.3-03)")
    print(f"📌 提示词版本: {get_prompt_version()}")
    print(f"📁 可用模板: {list_available_prompts()}")
    
    # 获取LLM客户端
    client = get_llm_client()
    print(f"🤖 LLM模式: {client._mode}")
    
    # 筛选测试用例
    if scenarios:
        test_cases = [tc for tc in TEST_CASES if tc.scenario in scenarios]
    else:
        test_cases = TEST_CASES
    
    print(f"\n📋 将运行 {len(test_cases)} 个测试用例")
    
    # 执行测试
    results = []
    for tc in test_cases:
        try:
            result = run_test_case(tc, client)
            results.append(result)
            print_result(result, verbose)
        except Exception as e:
            print(f"\n❌ 测试失败: {tc.name}")
            print(f"   错误: {e}")
            results.append({
                "name": tc.name,
                "scenario": tc.scenario,
                "score": 0,
                "error": str(e)
            })
    
    # 汇总统计
    print("\n" + "=" * 60)
    print("📊 测试汇总")
    print("=" * 60)
    
    total_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0
    passed = sum(1 for r in results if r.get("score", 0) >= 0.8)
    warned = sum(1 for r in results if 0.5 <= r.get("score", 0) < 0.8)
    failed = sum(1 for r in results if r.get("score", 0) < 0.5)
    
    print(f"总体得分: {total_score:.0%}")
    print(f"✅ 通过: {passed}")
    print(f"⚠️ 警告: {warned}")
    print(f"❌ 失败: {failed}")
    
    # 按场景统计
    print("\n按场景统计:")
    scenarios_scores = {}
    for r in results:
        s = r["scenario"]
        if s not in scenarios_scores:
            scenarios_scores[s] = []
        scenarios_scores[s].append(r.get("score", 0))
    
    for s, scores in scenarios_scores.items():
        avg = sum(scores) / len(scores)
        print(f"  - {s}: {avg:.0%}")
    
    return {
        "version": get_prompt_version(),
        "total_score": total_score,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "results": results
    }


def show_prompt_stats():
    """显示提示词统计信息"""
    stats = get_prompt_stats()
    print("\n📊 提示词模板统计")
    print("=" * 60)
    print(f"版本: {stats['version']}")
    print("\n模板详情:")
    for name, info in stats["templates"].items():
        placeholder = "✓" if info["has_placeholders"] else "✗"
        print(f"  - {name}: {info['lines']}行, {info['size']}字节, 占位符:{placeholder}")


def main():
    parser = argparse.ArgumentParser(description="Prompt回归测试")
    parser.add_argument(
        "--scenario", "-s",
        choices=["food", "gift", "emotion", "multi", "summary", "teacher"],
        help="只测试特定场景"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细信息（包括prompt）"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="显示提示词统计信息"
    )
    
    args = parser.parse_args()
    
    if args.stats:
        show_prompt_stats()
        return
    
    scenarios = [args.scenario] if args.scenario else None
    results = run_all_tests(scenarios=scenarios, verbose=args.verbose)
    
    # 返回码
    if results["failed"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
