"""The instrumented agent: where timings are actually taken, and where the
model's words are made safe to speak.

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

import logging
from typing import Any, AsyncIterable

from stacy.agent.runtime import CallRuntime
from stacy.agent.speech import strip_audio_tags_stream
from stacy.agent.tools import DeskOperations, build_function_tools
from stacy.metrics import Stage

log = logging.getLogger("stacy.agent.nodes")


def build_agent(runtime: CallRuntime, ops: DeskOperations | None = None) -> Any:
    """Construct the instrumented Agent for one call.

    The class is defined inside the function so that importing
    :mod:`stacy.agent` costs nothing when the SDK is absent, which is what keeps
    the test suite fast and vendor-free.
    """
    from livekit.agents import Agent  # noqa: PLC0415

    ops = ops or DeskOperations(runtime)
    tools = build_function_tools(ops)

    class InstrumentedAgent(Agent):
        """Agent that times its own critical path and keeps tags out of the voice."""

        def __init__(self) -> None:
            super().__init__(instructions=runtime.system_prompt, tools=tools)

        async def stt_node(self, audio: Any, model_settings: Any) -> Any:
            """Pass audio through, marking the first final transcript."""
            async for event in super().stt_node(audio, model_settings):
                if getattr(event, "type", None) is not None and _is_final(event):
                    runtime.mark(Stage.STT_FINAL)
                yield event

        async def llm_node(self, chat_ctx: Any, tools: Any, model_settings: Any) -> Any:
            """Stream the model, marking when it commits and when it first speaks.

            Nothing is spoken here but the model's own words. A filler used to be
            yielded into this stream on a timer -- and the SDK runs this node a
            second time after every tool call, a generation that on this prompt
            almost always says nothing because the model already answered. The
            timer fired on that one, so "One sec" arrived after the answer, on
            nearly every turn. Slow tools are covered where they run instead;
            see :func:`stacy.agent.tools.build_function_tools`.

            Before anything else: the gas/CO override. If it has latched, the
            model is not called at all. That is what makes it a control rather
            than an instruction -- there is no sampling step to go wrong, no
            tool that can fire, and no token budget between the caller saying
            "I smell gas" and hearing what to do.
            """
            # The pipeline has decided the caller finished and is committing to
            # a reply. Marking it here separates "still waiting to be sure you
            # stopped talking" from "the model is thinking". Marks keep their
            # first timestamp, so the post-tool generation cannot move it.
            runtime.mark(Stage.ENDPOINT_COMMITTED)

            spoken = _last_user_text(chat_ctx)
            runtime.safety.scan(spoken)
            if runtime.safety.latched:
                script = runtime.safety.response(quiet=not spoken)
                if script:
                    runtime.mark(Stage.LLM_FIRST_TOKEN, safety_override=True)
                    runtime.record_agent(script)
                    log.warning("call %s: speaking safety override", runtime.call_id)
                    yield script
                return

            first = True
            async for chunk in super().llm_node(chat_ctx, tools, model_settings):
                if first:
                    runtime.mark(Stage.LLM_FIRST_TOKEN)
                    first = False
                yield chunk

        async def tts_node(self, text: AsyncIterable[str], model_settings: Any) -> Any:
            """Synthesize, marking the first frame from the vendor and out the door.

            Audio tags are removed on the way in. The prompt asks for them and the
            voice model reads them out: "[slow] two-zero-three" was spoken as
            "slow, two-zero-three". See :mod:`stacy.agent.speech`.
            """
            first = True
            async for frame in super().tts_node(strip_audio_tags_stream(text), model_settings):
                if first:
                    runtime.mark(Stage.TTS_FIRST_BYTE)
                    runtime.mark(Stage.AUDIO_OUT)
                    first = False
                yield frame

    return InstrumentedAgent()


def _last_user_text(chat_ctx: Any) -> str:
    """The caller's most recent words, whatever shape the SDK holds them in.

    Read from the context rather than from the transcript event, because
    preemptive generation can start a turn on interim text before that event
    fires. Written against several plausible attribute names so a rename
    degrades to an empty string -- and an empty string is the *quiet* branch of
    the override, which is the safe direction to fail in.
    """
    items = getattr(chat_ctx, "items", None) or getattr(chat_ctx, "messages", None) or []
    for item in reversed(list(items)):
        if getattr(item, "role", None) != "user":
            continue
        text = getattr(item, "text_content", None)
        if callable(text):
            text = text()
        if text is None:
            content = getattr(item, "content", None)
            if isinstance(content, str):
                text = content
            elif isinstance(content, (list, tuple)):
                text = " ".join(c for c in content if isinstance(c, str))
        return (text or "").strip()
    return ""


def _is_final(event: Any) -> bool:
    """Whether an STT event carries a final transcript.

    Written against the shape rather than an exact enum member so a renamed
    event type degrades to a missing column instead of a raised exception on a
    live call.
    """
    event_type = getattr(event, "type", None)
    name = getattr(event_type, "name", None) or str(event_type)
    return "FINAL_TRANSCRIPT" in name.upper()
