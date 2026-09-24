"""Where a model reply's first word actually goes, measured from the worker itself.

A model swap is only worth doing if the model is what the caller is waiting for.
Measured from a workstation in India, a bare HTTPS request to the model's host
already cost 544 ms of the 746 ms first token: the prompt and six tool schemas
added 25 ms, and the rest was distance. The worker runs in us-east, where that
distance is small -- so the split has to be measured there, not here, before
anyone changes models.

This runs inside a job, on a room named `bench-...`, and logs one JSON line per
measurement. Nothing here ever touches a call: :func:`stacy.agent.entrypoint`
returns before a session exists.

    python scripts/bench_models.py            # dispatches the job and reads the log
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

log = logging.getLogger("stacy.bench")

BENCH_PREFIX = "bench-"
"""Rooms that exist only to time the models. See :mod:`stacy.agent.entrypoint`."""

CANDIDATES: tuple[tuple[str, str], ...] = (
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("livekit", "google/gemma-4-31b-it"),
    ("livekit", "google/gemini-3-flash"),
    ("livekit", "google/gemini-3.1-flash-lite"),
    ("livekit", "google/gemini-2.5-flash"),
    ("livekit", "google/gemini-2.5-flash-lite"),
    ("livekit", "openai/gpt-5.4-mini"),
    ("livekit", "openai/gpt-5.4-nano"),
    ("livekit", "openai/gpt-5.5"),
    ("livekit", "openai/chat-latest"),
    ("livekit", "moonshotai/kimi-k2.6"),
    ("livekit", "deepseek-ai/deepseek-v3.2"),
    ("livekit", "zai/glm-5.1"),
)
"""What to time. Each is asked the same thing, with the same prompt and tools."""

ROUNDS = 3
ASK = "The caller just said: 'I'm in Wallingford.' Reply as you would on the call."


async def _time_transport(url: str) -> float:
    """A round trip to the host with no inference in it: what distance alone costs."""
    import aiohttp  # noqa: PLC0415

    started = time.perf_counter()
    try:
        async with aiohttp.ClientSession() as session, session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            await r.read()
    except Exception as exc:  # noqa: BLE001 - a 401 still measures the round trip
        log.info("bench transport %s: %s", url, type(exc).__name__)
    return (time.perf_counter() - started) * 1000


async def _time_model(vendor: str, model: str, instructions: str, tools: Any, ask: str) -> dict[str, Any]:
    """Time to the model's first word, and to the end of its first sentence."""
    from livekit.agents import APIConnectOptions  # noqa: PLC0415
    from livekit.agents.llm import ChatContext  # noqa: PLC0415

    if vendor == "anthropic":
        from livekit.plugins import anthropic  # noqa: PLC0415

        llm = anthropic.LLM(model=model, api_key=os.environ.get("LLM_API_KEY", ""),
                            caching="ephemeral", _strict_tool_schema=False)
    else:
        from livekit.agents import inference  # noqa: PLC0415

        llm = inference.LLM(model=model)

    ctx = ChatContext.empty()
    ctx.add_message(role="system", content=instructions)
    ctx.add_message(role="user", content=ask)

    first = sentence = tool_at = None
    text, called = "", []
    started = time.perf_counter()
    async with llm.chat(chat_ctx=ctx, tools=tools, conn_options=APIConnectOptions(timeout=30, max_retry=0)) as stream:
        async for chunk in stream:
            if not chunk.delta:
                continue
            for call in chunk.delta.tool_calls or []:
                if tool_at is None:
                    tool_at = (time.perf_counter() - started) * 1000
                called.append(call.name)
            if not chunk.delta.content:
                continue
            if first is None:
                first = (time.perf_counter() - started) * 1000
            text += chunk.delta.content
            if sentence is None and any(end in text for end in (".", "?", "!")):
                sentence = (time.perf_counter() - started) * 1000
    return {
        "first_word_ms": round(first) if first else None,
        "first_sentence_ms": round(sentence) if sentence else None,
        # A model that writes its tool call before speaking makes the caller wait
        # for the whole pass; the prompt asks for words first. See stacy.agent.nodes.
        "tool_first": bool(tool_at and (first is None or tool_at < first)),
        "tools": called,
        "chars": len(text),
        "said": text[:120],
    }


async def run(runtime: Any, tools: Any, metadata: str = "") -> None:
    """Time every candidate from this worker and log the numbers.

    `metadata` is the dispatch's own, so a run can name its models and rounds
    without a redeploy: {"models": ["livekit/openai/gpt-5.4-mini"], "rounds": 1,
    "gap_s": 2}. LiveKit Inference answers 429 when a burst is too wide, so a
    narrow list with a gap is how the rate-limited ones get retried.
    """
    import asyncio  # noqa: PLC0415

    candidates, rounds, gap = CANDIDATES, ROUNDS, 0.0
    try:
        asked = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        asked = {}
    if asked.get("models"):
        candidates = tuple(tuple(m.split("/", 1)) for m in asked["models"])
    rounds = int(asked.get("rounds", rounds))
    gap = float(asked.get("gap_s", gap))
    instructions = runtime.instructions or runtime.system_prompt
    for url in ("https://api.anthropic.com/v1/models", "https://agent-gateway.livekit.cloud/v1"):
        trips = [round(await _time_transport(url)) for _ in range(rounds)]
        log.info("bench %s", json.dumps({"transport": url, "ms": trips, "best": min(trips)}))

    for vendor, model in candidates:
        runs = []
        for _ in range(rounds):
            if gap:
                await asyncio.sleep(gap)
            try:
                runs.append(await _time_model(vendor, model, instructions, tools, ASK))
            except Exception as exc:  # noqa: BLE001 - one model failing is a result too
                runs.append({"error": f"{type(exc).__name__}: {exc}"[:200]})
        firsts = [r["first_word_ms"] for r in runs if r.get("first_word_ms")]
        log.info("bench %s", json.dumps({
            "vendor": vendor, "model": model,
            "first_word_ms": firsts, "best_first_word_ms": min(firsts) if firsts else None,
            "first_sentence_ms": [r.get("first_sentence_ms") for r in runs],
            "runs": runs,
        }))
    log.info("bench done")
