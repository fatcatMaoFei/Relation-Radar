from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List

from mcp_server.tools.get_person_summary import get_person_summary_tool
from mcp_server.tools.log_feedback import log_feedback_tool
from mcp_server.tools.search_events import search_events_tool

"""
Lightweight tool registry / CLI wrapper for Relation Radar MCP tools.

说明：
- 这里不直接实现完整的 Model Context Protocol。
- 真实的 MCP server 可以在另外的项目里使用诸如
  `modelcontextprotocol.Server`，并引入本模块中注册好的工具函数。

在本仓库中，我们只提供：
- 一个 `TOOLS` 映射（名称 -> 可调用）；
- 一个极简 CLI，便于本地用 `python -m mcp_server.server ...`
  手动验证工具行为。
"""


ToolFn = Callable[..., Any]


TOOLS: Dict[str, ToolFn] = {
    "search_events": search_events_tool,
    "get_person_summary": get_person_summary_tool,
    "log_feedback": log_feedback_tool,
}


def list_tools() -> List[str]:
    """
    Return the list of registered tool names.

    外部的 MCP 适配层可以通过这个函数发现有哪些工具需要暴露给远端 LLM。
    """
    return sorted(TOOLS.keys())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relation-radar-mcp",
        description="Minimal CLI wrapper around Relation Radar MCP tools.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # search-events
    search_parser = subparsers.add_parser(
        "search-events",
        help="Search events for a person or globally.",
    )
    search_parser.add_argument("--person-id", type=int, help="Optional person id")
    search_parser.add_argument("--query", required=True, help="Search query text")
    search_parser.add_argument("--top-k", type=int, default=5, help="Max results")
    search_parser.set_defaults(func=_cli_search_events)

    # get-person-summary
    summary_parser = subparsers.add_parser(
        "get-person-summary",
        help="Get a structured summary for a person.",
    )
    summary_parser.add_argument("person_id", type=int, help="Person id")
    summary_parser.set_defaults(func=_cli_get_person_summary)

    # log-feedback
    feedback_parser = subparsers.add_parser(
        "log-feedback",
        help="Log feedback for a QA answer.",
    )
    feedback_parser.add_argument("--person-id", type=int, help="Optional person id")
    feedback_parser.add_argument("--question", required=True, help="Question text")
    feedback_parser.add_argument("--answer", required=True, help="Answer text")
    feedback_parser.add_argument(
        "--rating",
        required=True,
        choices=["accurate", "inaccurate", "risky"],
        help="Feedback rating label",
    )
    feedback_parser.add_argument(
        "--context-ids",
        help="Comma-separated event ids used as context",
    )
    feedback_parser.set_defaults(func=_cli_log_feedback)

    return parser


def _cli_search_events(args: argparse.Namespace) -> None:
    results = search_events_tool(
        person_id=args.person_id,
        query=args.query,
        top_k=args.top_k,
    )
    print(f"Found {len(results)} results:")
    for item in results:
        print(f"- event_id={item['event_id']}, score={item['score']:.2f}")
        print(f"  occurred_at={item.get('occurred_at')}")
        print(f"  type={item.get('event_type')}, emotion={item.get('emotion')}")
        print(f"  snippet={item.get('snippet')}")


def _cli_get_person_summary(args: argparse.Namespace) -> None:
    summary = get_person_summary_tool(person_id=args.person_id)
    print(f"Summary for person {summary['person']['id']}: {summary['person']['name']}")
    print("Tags:", ", ".join(summary["person"].get("tags") or []))
    if summary["person"].get("notes"):
        print("Notes:", summary["person"]["notes"])
    if summary.get("preferences"):
        print("Preferences:", ", ".join(summary["preferences"]))
    if summary.get("taboos"):
        print("Taboos:", ", ".join(summary["taboos"]))
    if summary.get("recent_events"):
        print("Recent events:")
        for ev in summary["recent_events"]:
            print(
                f"- [{ev.get('occurred_at')}] {ev.get('summary')} "
                f"(emotion={ev.get('emotion')})",
            )


def _cli_log_feedback(args: argparse.Namespace) -> None:
    if args.context_ids:
        used_ids = [
            int(part.strip())
            for part in args.context_ids.split(",")
            if part.strip()
        ]
    else:
        used_ids = []

    result = log_feedback_tool(
        person_id=args.person_id,
        question=args.question,
        answer=args.answer,
        rating=args.rating,
        used_context_event_ids=used_ids,
    )
    print(f"Saved feedback id={result['id']} rating={result['rating']}")


def main(argv: List[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
