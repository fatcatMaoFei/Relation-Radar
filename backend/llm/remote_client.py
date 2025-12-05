from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

"""
Remote LLM client for teacher mode.

说明：
- 仅在 v0.3-02 的离线脚本中使用，用于调用远端“大模型老师”生成理想答案；
- 不在核心链路（CLI / Web / API）中自动调用，避免无意间走到外网；
- 不在代码里写死任何密钥，所有配置都从环境变量读取：

  - REMOTE_LLM_PROVIDER   : "openai" / "google" / 其它（默认 "openai"）
  - REMOTE_LLM_API_KEY    : 必填，远端 LLM 的 API Key
  - REMOTE_LLM_MODEL      : 远端模型名，如 "gpt-4o" / "gemini-1.5-flash"
  - REMOTE_LLM_BASE_URL   : 可选，自定义接口地址（默认为常见官方地址）

当前仅实现两种 provider：
- openai  : 假定兼容 Chat Completions 接口（/v1/chat/completions）；
- google  : 使用 Gemini `generateContent` 风格接口。

如果你使用的是其它兼容接口，可以设置 REMOTE_LLM_BASE_URL 并复用 "openai" 分支。
"""


@dataclass
class RemoteLLMConfig:
    provider: str
    api_key: str
    model: str
    base_url: str
    timeout: int = 60


def _load_config_from_env() -> RemoteLLMConfig:
    provider = os.getenv("REMOTE_LLM_PROVIDER", "openai").strip().lower()
    api_key = os.getenv("REMOTE_LLM_API_KEY")
    model = os.getenv("REMOTE_LLM_MODEL") or ("gpt-4o" if provider == "openai" else "gemini-1.5-flash")

    if not api_key:
        raise RuntimeError(
            "REMOTE_LLM_API_KEY is not set. "
            "Please export REMOTE_LLM_API_KEY before running teacher scripts.",
        )

    if provider == "openai":
        base_url = os.getenv("REMOTE_LLM_BASE_URL", "https://api.openai.com/v1")
    elif provider == "google":
        # For Gemini-style endpoints; model name is appended in client.
        base_url = os.getenv(
            "REMOTE_LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/models",
        )
    else:
        base_url = os.getenv("REMOTE_LLM_BASE_URL")
        if not base_url:
            raise RuntimeError(
                f"Unsupported REMOTE_LLM_PROVIDER '{provider}' and REMOTE_LLM_BASE_URL is not set.",
            )

    return RemoteLLMConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url.rstrip("/"),
    )


class RemoteLLMClient:
    """
    Minimal HTTP client for calling a remote teacher model.
    """

    def __init__(self, config: Optional[RemoteLLMConfig] = None):
        self.config = config or _load_config_from_env()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        if self.config.provider == "openai":
            return self._generate_openai(prompt, max_tokens=max_tokens, temperature=temperature)
        if self.config.provider == "google":
            return self._generate_google(prompt, max_tokens=max_tokens, temperature=temperature)
        raise RuntimeError(f"Unsupported provider: {self.config.provider}")

    # --- Provider implementations -------------------------------------------------

    def _generate_openai(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI-like API error {resp.status_code}: {resp.text}")

        data = resp.json()
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenAI response format: {json.dumps(data)[:500]}") from exc

        return str(content)

    def _generate_google(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        # Gemini generateContent endpoint.
        url = f"{self.config.base_url}/{self.config.model}:generateContent"
        params = {"key": self.config.api_key}
        body: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        resp = requests.post(url, params=params, json=body, timeout=self.config.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Google Gemini API error {resp.status_code}: {resp.text}")

        data = resp.json()
        try:
            first_candidate = data["candidates"][0]
            parts = first_candidate["content"]["parts"]
            text_parts = [p.get("text", "") for p in parts]
            content = "\n".join(text_parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini response format: {json.dumps(data)[:500]}") from exc

        return str(content)


_remote_client: Optional[RemoteLLMClient] = None


def get_remote_llm_client() -> RemoteLLMClient:
    """
    Global singleton access helper.
    """
    global _remote_client
    if _remote_client is None:
        _remote_client = RemoteLLMClient()
    return _remote_client
