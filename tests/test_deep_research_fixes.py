#!/usr/bin/env python3
"""
Regression test for deep_research/quick_search fixes:

1. Progress notifications: long-running calls now report MCP progress via
   ProgressLogHandler, so a client doesn't sit with no signal until the call
   finishes or its idle timeout aborts the connection. This covers both of
   gpt_researcher's independent progress paths -- log_handler (coarse
   macro-checkpoints) and websocket (the fine-grained per-sub-query/
   per-source progress emitted during the actual search/scrape loop) --
   sharing one handler instance so both paths keep one monotonic counter.
2. Report synthesis: deep_research now also calls write_report() and
   returns it as "report" by default (synthesize_report=True), instead of
   only ever returning raw context that the caller had to know to pass to
   the separate write_report tool.
3. content_length: format_sources_for_response() now reads the "raw_content"
   key GPTResearcher.get_research_sources() actually populates, instead of
   a "content" key that was never present -- content_length was 0 for every
   source regardless of how much text was actually scraped.
4. write_report() is called with the handler's websocket cleared, so
   gpt_researcher's report-generation LLM call keeps its normal retry
   budget and doesn't re-stream the whole report back as progress
   messages (see server.py's deep_research for the full rationale).

Does not require the real gpt_researcher package: patches server.GPTResearcher
with a lightweight fake so this runs fast and without network/LLM access.
Collectible by pytest; also runnable directly:

    python3 tests/test_deep_research_fixes.py
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "unused-dummy-value-for-tests")

# Populated by _FakeGPTResearcher.__init__/write_report so assertions can
# check what server.py actually passed in, independent of the fake's own
# simulated event counts.
captured: Dict[str, Any] = {}


class _FakeGPTResearcher:
    def __init__(self, query, log_handler=None, websocket=None, **kwargs):
        self.query = query
        self.log_handler = log_handler
        self.websocket = websocket
        captured["log_handler"] = log_handler
        captured["websocket"] = websocket
        self._sources = [
            {"title": "Example", "url": "http://example.com", "raw_content": "x" * 250},
            {"title": "NoContent", "url": "http://example.com/2"},
        ]

    async def conduct_research(self):
        if self.log_handler:
            await self.log_handler.on_research_step("start", {})
            await self.log_handler.on_tool_start("web_search")
        # Mirrors gpt_researcher's real stream_output() calls
        # (gpt_researcher/actions/utils.py), made throughout the actual
        # sub-query search/scrape loop -- the real bottleneck. That's a
        # separate mechanism from log_handler, gated on `websocket` alone --
        # this path was never wired to MCP progress before this fix.
        if self.websocket:
            await self.websocket.send_json({"type": "logs", "output": "running sub-query 1"})
            await self.websocket.send_json({"type": "logs", "output": "running sub-query 2"})
        if self.log_handler:
            await self.log_handler.on_research_step("research_completed", {})

    async def quick_search(self, query):
        if self.log_handler:
            await self.log_handler.on_research_step("quick_search_start", {})
        return [{"title": "quick", "url": "http://example.com", "snippet": "..."}]

    async def write_report(self, custom_prompt=None):
        # server.py must clear researcher.websocket between
        # conduct_research() and write_report() -- see that call site for
        # why (retry budget + duplicate report streaming).
        captured["websocket_at_write_report"] = self.websocket
        return f"# Report for {self.query}\n\nSynthesized findings."

    def get_research_context(self):
        return f"context for {self.query}"

    def get_research_sources(self):
        return self._sources

    def get_source_urls(self):
        return [s["url"] for s in self._sources]

    def get_costs(self):
        return 0.0


async def _run() -> None:
    import server
    from fastmcp import Client

    with patch("server.GPTResearcher", _FakeGPTResearcher):
        progress_events = []

        async def on_progress(progress, total, message):
            progress_events.append((progress, message))

        async with Client(server.mcp, progress_handler=on_progress) as client:
            # Fix 1 + 2: progress reported, report synthesized by default.
            result = await client.call_tool("deep_research", {"query": "test query"})
            data = result.data
            assert data["status"] == "success", data
            assert "report" in data, "synthesize_report defaults True, report must be present"
            assert data["report"].startswith("# Report for test query"), data["report"]
            # 2 log_handler checkpoints + 2 websocket-path sub-query events;
            # the websocket-path count catches the regression this fake
            # simulates: without websocket= wired, log_handler alone only
            # covers the coarse macro-checkpoints, not the actual
            # search/scrape loop where these events would really fire.
            assert len(progress_events) >= 4, f"expected >=4 progress events (log_handler + websocket path), got {progress_events}"

            # The construction itself: both paths must be wired, and to the
            # *same* handler instance -- two separate instances would each
            # keep their own step counter and emit colliding, non-monotonic
            # progress values.
            assert captured["websocket"] is not None, "websocket= must be wired for search-loop progress"
            assert captured["websocket"] is captured["log_handler"], "both progress paths must share one handler instance"

            # Fix 4: websocket must be cleared before write_report() runs.
            assert captured["websocket_at_write_report"] is None, (
                "write_report() must be called with the researcher's websocket "
                "cleared, or the report LLM call loses its retry budget and "
                "the whole report gets re-streamed back as progress"
            )

            # Fix 3: content_length reflects raw_content, missing key doesn't crash.
            assert data["sources"][0]["content_length"] == 250, data["sources"]
            assert data["sources"][1]["content_length"] == 0, data["sources"]

            # synthesize_report=False must omit the report.
            progress_events.clear()
            result2 = await client.call_tool(
                "deep_research", {"query": "no report please", "synthesize_report": False}
            )
            assert "report" not in result2.data, "synthesize_report=False must omit report"

            # quick_search also reports progress.
            progress_events.clear()
            result3 = await client.call_tool("quick_search", {"query": "fast query"})
            assert result3.data["status"] == "success", result3.data
            assert len(progress_events) >= 1, "quick_search must report progress too"


def test_deep_research_and_quick_search_progress_and_report_fixes() -> None:
    asyncio.run(_run())


async def _run_send_json_fallbacks() -> None:
    from utils import ProgressLogHandler

    events = []

    class _FakeCtx:
        async def report_progress(self, progress: int, message: str) -> None:
            events.append((progress, message))

    handler = ProgressLogHandler(_FakeCtx())

    # Each of send_json's fallback branches, in priority order.
    await handler.send_json({"output": "from output", "content": "ignored", "type": "ignored"})
    await handler.send_json({"content": "from content", "type": "ignored"})
    await handler.send_json({"type": "from type"})
    await handler.send_json({})

    assert [message for _, message in events] == [
        "from output",
        "from content",
        "from type",
        "progress",
    ], events


def test_progress_log_handler_send_json_covers_all_fallback_branches() -> None:
    asyncio.run(_run_send_json_fallbacks())


async def _run_report_progress_failure_does_not_abort() -> None:
    from utils import ProgressLogHandler

    class _FailingCtx:
        async def report_progress(self, progress: int, message: str) -> None:
            raise RuntimeError("client disconnected")

    handler = ProgressLogHandler(_FailingCtx())

    # gpt_researcher's websocket-shaped stream_output() awaits send_json()
    # unguarded (unlike the defensive log_handler path in agent.py's
    # _log_event), so a broken transport here must not propagate and abort
    # whatever research is still in progress.
    await handler.send_json({"output": "progress update"})
    await handler.on_research_step("some_step", {})


def test_progress_log_handler_swallows_report_progress_failures() -> None:
    asyncio.run(_run_report_progress_failure_does_not_abort())


if __name__ == "__main__":
    asyncio.run(_run())
    asyncio.run(_run_send_json_fallbacks())
    asyncio.run(_run_report_progress_failure_does_not_abort())
    print("OK: deep_research/quick_search progress, report synthesis, retry-budget, and content_length fixes verified")
