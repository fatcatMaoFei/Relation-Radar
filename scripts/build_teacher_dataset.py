#!/usr/bin/env python3
"""
Build teacher dataset for v0.3-02.

功能：
- 从 JSONL 输入文件中读取一组「问题 + 关联 person_ids」；
- 使用现有的本地工具（人物画像 + RAG 搜索）构造事实摘要；
- 调用远端 teacher 模型（backend.llm.remote_client）生成理想答案；
- 将 (question, facts, ideal_answer, person_ids) 写入 JSONL 输出文件。

注意：
- 远端模型的 API Key 不会写在代码里，必须通过环境变量提供：
    REMOTE_LLM_API_KEY
    REMOTE_LLM_PROVIDER   (可选，默认 "openai")
    REMOTE_LLM_MODEL      (可选，默认 "gpt-4o" / "gemini-1.5-flash")
- 该脚本只在你手动运行时才会触发网络调用，CI 不会执行它。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from backend.core.db import init_db
from backend.llm.prompts import build_qa_rag_prompt
from backend.llm.remote_client import get_remote_llm_client
from mcp_server.tools.get_person_summary import get_person_summary_tool
from mcp_server.tools.search_events import search_events_tool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "teacher_dataset.jsonl"


@dataclass
class TeacherSampleInput:
    question: str
    person_ids: List[int]
    top_k: int = 5
    sample_id: Optional[str] = None


def load_inputs(path: Path) -> List[TeacherSampleInput]:
    """
    Load teacher sample inputs from a JSONL file.

    每一行 JSON 至少需要：
      - question: str
      - person_ids: list[int]
    可选：
      - id: str
      - top_k: int
    """
    items: List[TeacherSampleInput] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            question = str(data["question"]).strip()
            if not question:
                raise ValueError(f"Line {line_no}: empty question")
            person_ids_raw = data.get("person_ids") or []
            person_ids = [int(pid) for pid in person_ids_raw]
            if not person_ids:
                raise ValueError(f"Line {line_no}: person_ids is empty")
            top_k = int(data.get("top_k", 5))
            sample_id = data.get("id")
            items.append(
                TeacherSampleInput(
                    question=question,
                    person_ids=person_ids,
                    top_k=top_k,
                    sample_id=sample_id,
                ),
            )
    return items


def _format_person_section(person_summary: Dict[str, Any]) -> str:
    person = person_summary["person"]
    lines: List[str] = []
    lines.append(
        f"- {person.get('name', 'Unknown')} (ID {person.get('id')}) "
        f"tags={','.join(person.get('tags') or [])}",
    )
    notes = person.get("notes")
    if notes:
        lines.append(f"  Notes: {notes}")

    prefs = person_summary.get("preferences") or []
    taboos = person_summary.get("taboos") or []
    if prefs:
        lines.append(f"  Likes: {', '.join(prefs)}")
    if taboos:
        lines.append(f"  Avoid: {', '.join(taboos)}")

    recent = person_summary.get("recent_events") or []
    if recent:
        lines.append("  Recent events:")
        for ev in recent:
            lines.append(
                f"    - [{ev.get('occurred_at')}] {ev.get('summary')} "
                f"(emotion={ev.get('emotion')})",
            )

    return "\n".join(lines)


def _format_search_results(results: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in results:
        lines.append(
            f"- event_id={item['event_id']} score={item['score']:.2f} "
            f"at {item.get('occurred_at')} type={item.get('event_type')} "
            f"emotion={item.get('emotion')}",
        )
        if item.get("snippet"):
            lines.append(f"  {item['snippet']}")
    return "\n".join(lines)


def build_facts_for_sample(sample: TeacherSampleInput) -> str:
    """
    Use MCP-style tools to build a human-readable facts block for teacher.
    """
    sections: List[str] = []
    for pid in sample.person_ids:
        summary = get_person_summary_tool(person_id=pid, max_events=10)
        sections.append(_format_person_section(summary))

    # 针对问题做一次补充搜索（使用多人的检索更接近实际问答场景）
    search_results = search_events_tool(
        person_id=None,
        query=sample.question,
        top_k=sample.top_k,
    )
    if search_results:
        sections.append("Extra relevant events across all persons:")
        sections.append(_format_search_results(search_results))

    return "\n\n".join(sections)


def build_teacher_example(
    sample: TeacherSampleInput,
    *,
    facts: str,
) -> Dict[str, Any]:
    """
    Call remote teacher model and build one training example.
    """
    client = get_remote_llm_client()
    prompt = build_qa_rag_prompt(question=sample.question, context=facts)
    answer = client.generate(prompt, max_tokens=512, temperature=0.2)
    return {
        "id": sample.sample_id,
        "question": sample.question,
        "person_ids": sample.person_ids,
        "facts": facts,
        "ideal_answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build teacher dataset JSONL using remote LLM and local MCP tools.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL with {question, person_ids[, top_k, id]} per line.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[teacher-dataset] loading inputs from {input_path}")
    samples = load_inputs(input_path)
    print(f"[teacher-dataset] loaded {len(samples)} samples")

    # Ensure DB schema and repositories are ready
    init_db()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = get_remote_llm_client()  # noqa: F841 - force config check early

    with output_path.open("w", encoding="utf-8") as f_out:
        for idx, sample in enumerate(samples, 1):
            print(f"[teacher-dataset] building sample {idx}/{len(samples)} …")
            facts = build_facts_for_sample(sample)
            example = build_teacher_example(sample, facts=facts)
            if example.get("id") is None:
                example["id"] = str(idx)
            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"[teacher-dataset] done. Wrote {len(samples)} examples to {output_path}")


if __name__ == "__main__":
    main()

