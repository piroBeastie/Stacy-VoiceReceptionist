"""The instrumented agent: where timings are actually taken, and where a slow
model gets covered with speech instead of silence.

Timings are taken *here*, inside the pipeline nodes, rather than from the
session's `metrics_collected` event. That event reports durations after the fact
and fires once the LLM and TTS streams have completed -- which is after the turn
has already been emitted, so the two stages the README calls the biggest levers
were never recorded at all. First token and first audio frame are observable
exactly once each, at the moment they pass through, and that is what these
overrides capture.

One honesty note on `egress_ms`: measured here it is in-process framing only,
from the vendor's first frame to handing that frame downstream. It does not
include the LiveKit-to-SIP-to-PSTN leg, which is not observable from inside the
worker. The README's 50ms figure for that leg is an estimate and should be
labelled as one.

VERIFY AGAINST THE INSTALLED SDK before the first live call: the node method
names and their yielded types are the surface most likely to have moved.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterable

from dana.agent.runtime import CallRuntime
from dana.agent.tools import DeskOperations, build_function_tools
from dana.metrics import Stage

log = logging.getLogger("dana.agent.nodes")


def build_agent(runtime: CallRuntime, ops: DeskOperations | None = None) -> Any:
    """Construct the instrumented Agent for one call.

    The class is defined inside the function so that importing
    :mod:`dana.agent` costs nothing when the SDK is absent, which is what keeps
    the test suite fast and vendor-free.
    """
    from livekit.agents import Agent  # noqa: PLC0415

    ops = ops or DeskOperations(runtime)
    tools = build_function_tools(ops)
    soft_timeout = runtime.pipeline.filler.soft_timeout_s

    class InstrumentedAgent(Agent):
        """Agent that times its own critical path and covers its own pauses."""

        def __init__(self) -> None:
            super().__init__(instructions=runtime.system_prompt, tools=tools)

        async def stt_node(self, audio: Any, model_settings: Any) -> Any:
            """Pass audio through, marking the first final transcript."""
            async for event in super().stt_node(audio, model_settings):
                if getattr(event, "type", None) is not None and _is_final(event):
                    runtime.mark(Stage.STT_FINAL)
                yield event

        async def llm_node(self, chat_ctx: Any, tools: Any, model_settings: Any) -> Any:
            """Stream the model, marking first token and covering a slow start.

            The filler is yielded into the model's own output stream, so it is
            synthesized by the same TTS connection and lands in the transcript
            as something the agent said -- which it is. That does mean it enters
            the chat context; the platform's soft timeout behaves the same way.
            """
            upstream = super().llm_node(chat_ctx, tools, model_settings)
            iterator = upstream.__aiter__()
            pending: asyncio.Task[Any] | None = None
            first = True

            try:
                while True:
                    if pending is None:
                        pending = asyncio.ensure_future(iterator.__anext__())

                    if first and soft_timeout > 0:
                        done, _ = await asyncio.wait({pending}, timeout=soft_timeout)
                        if not done:
                            phrase = runtime.filler.soft_timeout_phrase()
                            if phrase:
                                runtime.annotate(soft_timeout_filler=True)
                                runtime.record_agent(phrase)
                                yield phrase

                    try:
                        chunk = await pending
                    except StopAsyncIteration:
                        return
                    finally:
                        pending = None

                    if first:
                        runtime.mark(Stage.LLM_FIRST_TOKEN)
                        first = False
                    yield chunk
            finally:
                if pending is not None:
                    pending.cancel()

        async def tts_node(self, text: AsyncIterable[str], model_settings: Any) -> Any:
            """Synthesize, marking the first frame from the vendor and out the door."""
            first = True
            async for frame in super().tts_node(text, model_settings):
                if first:
                    runtime.mark(Stage.TTS_FIRST_BYTE)
                    runtime.mark(Stage.AUDIO_OUT)
                    first = False
                yield frame

    return InstrumentedAgent()


def _is_final(event: Any) -> bool:
    """Whether an STT event carries a final transcript.

    Written against the shape rather than an exact enum member so a renamed
    event type degrades to a missing column instead of a raised exception on a
    live call.
    """
    event_type = getattr(event, "type", None)
    name = getattr(event_type, "name", None) or str(event_type)
    return "FINAL_TRANSCRIPT" in name.upper()
