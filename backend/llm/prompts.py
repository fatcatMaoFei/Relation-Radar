"""
Prompt helpers for Relation Radar.

提示词体系分为三大类：
1. 抽取类 (Extract): 文本 → 结构化 JSON
2. 问答类 (QA): RAG 检索 + 回答生成
3. Teacher类: 远端大模型调用

约定：
- 所有基础 prompt 文本放在 `config/prompts/*.txt` 中，使用 UTF-8 编码
- 这里提供加载和格式化的辅助函数
- 所有 prompt 遵循"只基于事实，不自行揣测"原则
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"

# ==================== 提示词版本管理 ====================
PROMPT_VERSION = "0.3.03"

# 提示词类型常量
class PromptType:
    """提示词类型枚举"""
    EXTRACT_EVENT = "extract_event"      # 事件抽取
    QA_RAG = "qa_rag"                    # RAG问答
    QA_MULTI_PERSON = "qa_multi_person"  # 多人问答
    PERSON_SUMMARY = "person_summary"    # 人物画像
    SCENE_ADVICE = "scene_advice"        # 场景建议
    TEACHER_QA = "teacher_qa"            # Teacher问答
    GIFT_SUGGESTION = "gift_suggestion"  # 送礼建议
    EMOTION_CARE = "emotion_care"        # 情绪关怀


def _load_prompt_file(name: str) -> str:
    """
    Load a prompt template from config/prompts.

    Args:
        name: Prompt file base name without extension (e.g. "qa_rag").

    Returns:
        Prompt template string (stripped). If file is missing or empty,
        returns an empty string and lets caller fall back to defaults.
    """
    path = PROMPTS_DIR / f"{name}.txt"
    try:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8")
        return text.strip()
    except OSError:
        return ""


def get_prompt_version() -> str:
    """获取当前提示词版本"""
    return PROMPT_VERSION


def list_available_prompts() -> List[str]:
    """列出所有可用的提示词模板"""
    if not PROMPTS_DIR.exists():
        return []
    return [p.stem for p in PROMPTS_DIR.glob("*.txt")]


def build_qa_rag_prompt(question: str, context: str) -> str:
    """
    Build prompt for RAG-based QA.

    优先使用 `config/prompts/qa_rag.txt` 中的模版：
    - 若包含 `{context}` / `{question}` 占位符，则用 format 替换。
    - 否则追加 context / question 到模版末尾。
    """
    template = _load_prompt_file("qa_rag")
    if template:
        if "{context}" in template or "{question}" in template:
            return template.format(context=context, question=question)
        # 模版中没有占位符时，附加上下文和问题
        return f"{template.rstrip()}\n\n相关记录：\n{context}\n\n问题：{question}"

    # 回退到内置 prompt（与 v0.1 时期等价）
    return (
        "你是一个帮助用户管理人际关系的助手。根据以下记录回答问题。\n\n"
        f"相关记录：\n{context}\n\n"
        f"问题：{question}\n\n"
        "请根据上述记录，给出详细、有帮助的回答。如果记录中没有相关信息，请如实说明。"
    )


def build_extract_event_prompt(text: str) -> str:
    """
    Build prompt for event extraction (Qwen-driven, v0.2-02 使用).

    当前仅负责加载模版并替换 `{text}`，调用方负责具体解析 JSON。
    """
    template = _load_prompt_file("extract_event")
    if template:
        if "{text}" in template:
            return template.format(text=text)
        return f"{template.rstrip()}\n\n原始文本：\n{text}"

    # 简单回退模版，方便在缺失文件时也可开发调试
    return (
        "你是一个信息抽取助手，请从下面的对话或描述中提取人物事件列表，"
        "输出 JSON 数组，每个元素包含：人物、时间、事件类型、摘要、情绪、偏好、忌讳、标签。\n\n"
        f"文本：\n{text}"
    )


# ==================== 问答类提示词 ====================

def build_multi_person_qa_prompt(
    question: str,
    context: str,
    person_names: List[str]
) -> str:
    """
    Build prompt for multi-person QA scenario.
    
    Args:
        question: User's question
        context: Retrieved context from multiple persons
        person_names: List of person names involved
    """
    template = _load_prompt_file("qa_multi_person")
    if template:
        names_str = "、".join(person_names)
        if all(p in template for p in ["{context}", "{question}", "{persons}"]):
            return template.format(
                context=context,
                question=question,
                persons=names_str
            )
    
    # 回退模版
    names_str = "、".join(person_names)
    return f"""你是一个帮助用户管理人际关系的智能助手。

现在需要综合分析以下朋友的记录来回答问题：{names_str}

【重要原则】
- 只依据给出的记录进行推理，不要自行揣测
- 当需要平衡多人需求时，要明确指出冲突点
- 给出建议时要考虑所有相关人的偏好和忌讳

【相关记录】
{context}

【用户问题】
{question}

请综合分析后给出回答。如果记录中存在冲突或不足，请明确说明。"""


def build_gift_suggestion_prompt(
    person_name: str,
    context: str,
    occasion: Optional[str] = None,
    budget: Optional[str] = None
) -> str:
    """
    Build prompt for gift suggestion scenario.
    
    Args:
        person_name: Name of the person to give gift
        context: Retrieved context about the person
        occasion: Optional occasion (birthday, holiday, etc.)
        budget: Optional budget range
    """
    template = _load_prompt_file("gift_suggestion")
    if template and "{context}" in template:
        return template.format(
            person=person_name,
            context=context,
            occasion=occasion or "日常",
            budget=budget or "不限"
        )
    
    # 回退模版
    occasion_text = f"场合：{occasion}" if occasion else ""
    budget_text = f"预算：{budget}" if budget else ""
    
    return f"""你是一个帮助用户选择礼物的智能助手。

【送礼对象】{person_name}
{occasion_text}
{budget_text}

【关于 TA 的记录】
{context}

【重要原则】
- 只基于记录中提到的偏好和忌讳推荐
- 避免推荐 TA 明确不喜欢的东西
- 如果记录不足，请说明并给出通用建议
- 不要推荐过于私密或可能引起误会的礼物

请给出 2-3 个礼物建议，并说明推荐理由。"""


def build_emotion_care_prompt(
    person_name: str,
    context: str,
    recent_emotion: Optional[str] = None
) -> str:
    """
    Build prompt for emotional care scenario.
    
    Args:
        person_name: Name of the person
        context: Retrieved context about the person
        recent_emotion: Recently detected emotion (焦虑/压力/难过 etc.)
    """
    template = _load_prompt_file("emotion_care")
    if template and "{context}" in template:
        return template.format(
            person=person_name,
            context=context,
            emotion=recent_emotion or "未知"
        )
    
    # 回退模版
    emotion_text = f"最近情绪：{recent_emotion}" if recent_emotion else ""
    
    return f"""你是一个帮助用户关心朋友的智能助手。

【关心对象】{person_name}
{emotion_text}

【关于 TA 的记录】
{context}

【重要原则】
- 只基于记录中的信息给出建议
- 建议要温和、具体、可执行
- 避免过度解读或心理诊断
- 如果情况严重，建议寻求专业帮助

请分析 TA 最近的状态，并给出 1-2 个关心 TA 的具体建议。"""


def build_person_summary_prompt(
    person_name: str,
    events_summary: str,
    preferences: List[str],
    taboos: List[str]
) -> str:
    """
    Build prompt for person summary/profile generation.
    
    Args:
        person_name: Name of the person
        events_summary: Summary of recent events
        preferences: List of known preferences
        taboos: List of known taboos
    """
    template = _load_prompt_file("person_summary")
    if template and len(template) > 10:  # 非空模板
        prefs_str = "、".join(preferences) if preferences else "暂无记录"
        tabs_str = "、".join(taboos) if taboos else "暂无记录"
        return template.format(
            person=person_name,
            events=events_summary,
            preferences=prefs_str,
            taboos=tabs_str
        )
    
    # 回退模版
    prefs_str = "\n".join(f"- {p}" for p in preferences) if preferences else "- 暂无记录"
    tabs_str = "\n".join(f"- {t}" for t in taboos) if taboos else "- 暂无记录"
    
    return f"""请为【{person_name}】生成一份简要的人物画像。

【近期事件摘要】
{events_summary}

【已知偏好】
{prefs_str}

【已知忌讳】
{tabs_str}

请用 3-5 句话概括这个人的特点，包括：性格印象、主要兴趣、需要注意的点。
只基于以上记录，不要自行推测。"""


# ==================== Teacher类提示词 ====================

def build_teacher_qa_prompt(
    question: str,
    facts: str,
    local_answer: Optional[str] = None
) -> str:
    """
    Build prompt for teacher model (remote large model).
    
    Args:
        question: User's original question
        facts: Anonymized facts from local data
        local_answer: Optional local model's answer for comparison
    """
    template = _load_prompt_file("teacher_qa")
    if template and "{facts}" in template:
        return template.format(
            question=question,
            facts=facts,
            local_answer=local_answer or "（未提供）"
        )
    
    # 回退模版
    local_ref = f"\n\n【本地模型回答（参考）】\n{local_answer}" if local_answer else ""
    
    return f"""你是一个人际关系管理专家，正在帮助用户处理朋友相关的问题。

【用户问题】
{question}

【已知事实】（来自用户的记录，已脱敏）
{facts}
{local_ref}

【回答要求】
1. 只基于提供的事实进行推理，不要编造
2. 回答结构：先给结论，再解释理由
3. 如果涉及敏感建议（如感情、金钱），要给出风险提示
4. 当事实不足时，明确说明并建议用户补充信息

请给出你的专业回答。"""


# ==================== 安全提示词包装器 ====================

def wrap_with_safety_guidelines(prompt: str) -> str:
    """
    Wrap any prompt with safety guidelines.
    
    Used for high-risk scenarios (relationship advice, financial decisions, etc.)
    """
    safety_prefix = """【安全提示】
在回答以下问题时，请特别注意：
- 不要给出可能伤害他人感情的建议
- 涉及金钱、感情决策时要提醒用户谨慎
- 如果问题涉及心理健康，建议寻求专业帮助
- 始终尊重隐私，不要过度揣测

"""
    return safety_prefix + prompt


def get_prompt_stats() -> dict:
    """
    Get statistics about prompt templates.
    
    Returns:
        Dict with prompt names, sizes, and version info
    """
    stats = {
        "version": PROMPT_VERSION,
        "templates": {}
    }
    
    for name in list_available_prompts():
        path = PROMPTS_DIR / f"{name}.txt"
        if path.exists():
            content = path.read_text(encoding="utf-8")
            stats["templates"][name] = {
                "size": len(content),
                "lines": content.count("\n") + 1,
                "has_placeholders": "{" in content
            }
    
    return stats
