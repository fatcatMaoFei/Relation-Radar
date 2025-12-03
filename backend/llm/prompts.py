"""
Prompt helpers for Relation Radar.

约定：
- 所有基础 prompt 文本放在 `config/prompts/*.txt` 中，使用 UTF-8 编码。
- 这里提供加载和简单格式化的辅助函数，供 RAG QA 链、信息抽取链等使用。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"


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
