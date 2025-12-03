#!/usr/bin/env python3
"""
Test script for PR-0.2-01: Local LLM (Qwen via Ollama).

用于本地验证：
- 当前 LLM 模式（mock / qwen）。
- generate(prompt) 和 chat(messages) 是否能正常返回结果。

CI 不会调用本脚本，主要面向开发者本地自测。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm.local_client import Message, get_llm_client  # noqa: E402


def main() -> int:
    mode = os.getenv("RELATION_RADAR_LLM_MODE", "mock").lower()
    model_name = os.getenv("RELATION_RADAR_LLM_MODEL", "qwen2.5:3b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    print("=== Local LLM smoke test ===")
    print(f"RELATION_RADAR_LLM_MODE = {mode}")
    if mode == "qwen":
        print(f"  -> Using Qwen via Ollama: model={model_name}, base_url={ollama_url}")
    else:
        print("  -> Using mock LLM implementation")

    client = get_llm_client()

    # Test generate (prompt-style)
    prompt = "简单介绍一下你是谁，你可以帮我做什么？"
    print("\n[generate] Prompt:")
    print(prompt)
    try:
        answer = client.generate(prompt, max_tokens=256)
    except Exception as exc:  # pragma: no cover - runtime env issue
        print(f"\n❌ generate() failed: {exc}")
        return 1

    print("\n[generate] Answer (first 300 chars):")
    print(answer[:300])

    # Test chat (Message-style)
    messages = [
        Message(role="system", content="你是一个友好的关系助手，用简短中文回答。"),
        Message(role="user", content="如果朋友最近工作压力很大，我可以怎么关心他？"),
    ]
    print("\n[chat] Messages:")
    for m in messages:
        print(f"- {m.role}: {m.content}")

    try:
        chat_answer = client.chat(messages, max_tokens=256)
    except Exception as exc:  # pragma: no cover
        print(f"\n❌ chat() failed: {exc}")
        return 1

    print("\n[chat] Answer (first 300 chars):")
    print(chat_answer[:300])

    print("\n✅ Local LLM smoke test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

