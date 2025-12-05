from __future__ import annotations

from .server import list_tools  # noqa: F401

"""
MCP-style tool layer for Relation Radar.

This package exposes a small set of pure-Python functions that wrap the
core backend capabilities (搜索事件 / 人物画像 / 记录反馈)。

真正的 Model Context Protocol 适配层（如 modelcontextprotocol.Server）
可以在其它项目中引用这些函数并完成协议绑定，本仓库只负责：

- 定义工具函数的签名和返回结构；
- 提供一个简易 CLI（`python -m mcp_server.server ...`）方便本地自测。
"""
