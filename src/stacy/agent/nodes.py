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

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterable, AsyncIterator

from stacy.agent.fastlane import desk_reply, name_spelled
from stacy.agent.runtime import CallRuntime
from stacy.agent.slots import (
    foreign_number_note,
    heard_booking_details,
    number_with_country,
    partial_number_note,
    phone_across_turns,
    phone_in_speech,
    readback_note,
)
from stacy.agent.speech import push_with_flush, strip_audio_tags_stream
from stacy.bench.shadow_stt import build as build_shadows
from stacy.bench.shadow_stt import specs as shadow_specs
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
            from livekit.agents.llm import ChatContext  # noqa: PLC0415

            # The stable prompt as instructions, and everything per-call in a
            # first message: see CallRuntime.instructions and call_briefing.
            briefing = ChatContext.empty()
            briefing.add_message(role="user", content=ops.call_briefing())
            super().__init__(
                instructions=runtime.instructions or runtime.system_prompt,
                tools=tools,
                chat_ctx=briefing,
            )

        async def on_user_turn_completed(self, turn_ctx: Any, new_message: Any) -> None:
            """Catch a phone number the caller read out, before the model answers.

            On a live call the model skipped `record_details` and turned "double
            9, double 8, triple 4, 5 2 0" into digits itself: an extra nine,
            eleven digits, read back wrong four times while the caller corrected
            it. The number is parsed here instead, saved to the booking, and its
            read-back is attached to the caller's message, so the model reads out
            digits it was given rather than digits it worked out. Editing the
            message discards a preemptive reply started without the note, which
            on this one kind of turn is the point.
            """
            text = getattr(new_message, "text_content", None) or ""
            runtime.desk_reply = None
            early, runtime.early_desk = runtime.early_desk, None
            if not text:
                return
            turn = str(getattr(new_message, "id", "") or "")
            runtime.settled_turn = turn
            voiced_early = early is not None and _same_words(early[0], text)
            reply = desk_reply(runtime, text)
            if reply is not None and reply.action == "book":
                # A yes to the read-back. Booked here, and the desk's line is
                # spoken only if the calendar took it; if not, the model tells
                # them what the calendar said instead.
                started = time.perf_counter()
                result = await asyncio.to_thread(ops.book_appointment)
                log.info(
                    "tool %s book_appointment() by the desk -> %r in %.0fms",
                    runtime.call_id,
                    result[:200],
                    (time.perf_counter() - started) * 1000,
                )
                if not result.startswith("Booked"):
                    # This also discards an early "You're all set", voiced
                    # before the calendar had its say.
                    runtime.last_caller_turn = text
                    new_message.content.append(
                        "(Not said by the caller: they said yes to the read-back, so the booking "
                        f"was tried, and the calendar said: {result})"
                    )
                    return
            if reply is not None:
                # A turn with one right reply; see stacy.agent.fastlane.
                runtime.slots.update(**reply.record)
                runtime.last_caller_turn = text
                runtime.latency.on_desk_reply()
                # Also what a pass the SDK regenerates speaks, if it drops the
                # early one: the same words get the same line, never the model.
                runtime.desk_reply = (turn, reply.speech)
                if voiced_early and early is not None and early[1] == reply.speech:
                    # Decided and voiced before the turn was confirmed: its audio
                    # is ready and plays now. The message is left alone, because
                    # changing it is what makes the SDK throw an early reply away;
                    # its note goes on once the SDK has filed it (entrypoint).
                    runtime.kept_note = (text, reply.note)
                    runtime.mark(Stage.LLM_FIRST_TOKEN, desk_reply=True)
                    log.info("call %s: desk reply ready early, recorded %s", runtime.call_id, reply.record)
                    return
                # Otherwise the note changes the message, which discards whatever
                # was started early, and the desk's reply is spoken instead.
                new_message.content.append(reply.note)
                log.info("call %s: desk reply, recorded %s", runtime.call_id, reply.record)
                return
            if voiced_early:
                # A desk reply was voiced early that the confirmed turn no longer
                # supports. Changing the message throws that audio away.
                new_message.content.append("(Not said by the caller: answer this turn yourself.)")
            notes: list[str] = []
            spelled = name_spelled(runtime, text)
            if spelled:
                # Heard as "Nak" while the letters said N-A-N-A-K.
                runtime.slots.update(customer_name=spelled)
                notes.append(
                    f"(Not said by the caller: they spelled their name out, so it's {spelled}. "
                    "Use that spelling.)"
                )
            number = phone_in_speech(text, caller_id=runtime.caller_id)
            if not number and not runtime.slots.phone:
                # A number read out or corrected over two turns; see phone_across_turns.
                number = phone_across_turns(
                    runtime.last_caller_turn, text, caller_id=runtime.caller_id
                ) or number_with_country(runtime.last_caller_turn, text)
            runtime.last_caller_turn = text
            if number:
                runtime.slots.update(phone=number)
                notes.append(readback_note(number))
                log.info("call %s: caller read out a number, saved %s", runtime.call_id, number)
            elif not runtime.slots.phone and (partial := partial_number_note(text)):
                notes.append(partial)
            elif not runtime.slots.phone and (foreign := foreign_number_note(text)):
                notes.append(foreign)

            # The same backstop for the town, day and window, which the model
            # records in the same breath as speaking and occasionally doesn't.
            # Only empty slots are filled, and nothing is added to the message
            # unless the calendar refuses what was said.
            heard = heard_booking_details(text, runtime.tenant)
            fill = {k: v for k, v in heard.items() if not getattr(runtime.slots, k)}
            if fill:
                newly = runtime.slots.update(**fill)
                log.info("call %s: heard %s from the caller", runtime.call_id, fill)
                refusal = ops.check_when(newly)
                if refusal:
                    notes.append(f"(Not said by the caller: {refusal})")

            new_message.content.extend(notes)

        async def stt_node(self, audio: Any, model_settings: Any) -> Any:
            """Pass audio through, marking the first final transcript.

            When `STACY_SHADOW_STT` names any, the same frames also go to shadow
            transcribers whose transcripts are only logged -- the way to compare
            recognisers on the caller's own voice rather than on synthesised
            speech. See :mod:`stacy.bench.shadow_stt`.
            """
            shadows: list[Any] = []
            if shadow_specs():
                try:
                    shadows = await build_shadows(runtime.pipeline.stt, runtime.call_id)
                except Exception:  # noqa: BLE001 - a shadow never disturbs the call
                    log.exception("call %s: shadow transcribers could not start", runtime.call_id)

            async def heard(frames: Any) -> Any:
                async for frame in frames:
                    for shadow in shadows:
                        try:
                            shadow.push(frame)
                        except Exception:  # noqa: BLE001, S110 - same
                            pass
                    yield frame

            try:
                async for event in super().stt_node(heard(audio) if shadows else audio, model_settings):
                    if getattr(event, "type", None) is not None and _is_final(event):
                        runtime.mark(Stage.STT_FINAL)
                    yield event
            finally:
                for shadow in shadows:
                    await shadow.aclose()

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

            # A reply the desk already wrote for this turn: spoken as it is, and
            # the model is not called. Matched to the caller's message itself, not
            # its words: review found a kept "You're all set" replayed on a later
            # "Yes." that no VAD event had separated from the first.
            pending = runtime.desk_reply
            if pending is not None and pending[0] and pending[0] == _last_caller_id(chat_ctx):
                runtime.desk_reply = None
                runtime.mark(Stage.LLM_FIRST_TOKEN, desk_reply=True)
                yield pending[1]
                return

            # Earlier still: the SDK runs this node on the caller's words before
            # it is sure they have finished, and builds the voice from what comes
            # out. A turn with one right reply is decided here, so its audio is
            # ready the moment they are judged done -- no model, no wait for the
            # voice. Only on a pass that answers the caller directly (never the
            # pass after a tool call, which would say it twice), and only before
            # the hook has settled this turn. The hook checks it against the
            # confirmed words and throws it away if anything changed.
            if spoken and _ends_with_caller(chat_ctx) and _last_caller_id(chat_ctx) != runtime.settled_turn:
                early = desk_reply(runtime, spoken)
                if early is not None:
                    # Not marked on the trace here: this can run before the
                    # turn's trace opens, and the reply may yet be thrown away.
                    runtime.early_desk = (spoken, early.speech)
                    yield early.speech
                    return

            # Every model pass is logged with what came first. A pass that writes a
            # tool call before any words costs its whole length before the caller
            # hears anything, and the SDK's ttft only times the pass after it.
            first = True
            started = time.perf_counter()
            text_ms: float | None = None
            tool_ms: float | None = None
            chars = 0
            called: list[str] = []
            usage: Any = None
            async for chunk in super().llm_node(chat_ctx, tools, model_settings):
                if first:
                    runtime.mark(Stage.LLM_FIRST_TOKEN)
                    first = False
                delta = getattr(chunk, "delta", None)
                content = chunk if isinstance(chunk, str) else getattr(delta, "content", None)
                calls = getattr(delta, "tool_calls", None) or []
                usage = getattr(chunk, "usage", None) or usage
                elapsed = (time.perf_counter() - started) * 1000
                if content:
                    chars += len(content)
                    text_ms = elapsed if text_ms is None else text_ms
                if calls:
                    tool_ms = elapsed if tool_ms is None else tool_ms
                    called += [getattr(c, "name", "") for c in calls]
                yield chunk
            order = (
                "tool-first" if tool_ms is not None and (text_ms is None or tool_ms < text_ms)
                else "text-first" if text_ms is not None
                else "empty"
            )
            log.info("llm_pass %s", json.dumps({
                "call": runtime.call_id,
                "order": order,
                "started_at": round(time.time() - (time.perf_counter() - started), 3),
                "text_ms": round(text_ms) if text_ms is not None else None,
                "tool_ms": round(tool_ms) if tool_ms is not None else None,
                "total_ms": round((time.perf_counter() - started) * 1000),
                "chars": chars,
                "tools": called,
                # A cold cache is the first suspect when a reply is slow.
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "cached_tokens": getattr(usage, "prompt_cached_tokens", None),
            }))

        async def tts_node(self, text: AsyncIterable[str], model_settings: Any) -> Any:
            """Synthesize, marking the first frame from the vendor and out the door.

            Audio tags are removed on the way in. The prompt asks for them and the
            voice model reads them out: "[slow] two-zero-three" was spoken as
            "slow, two-zero-three". See :mod:`stacy.agent.speech`.
            """
            clean = strip_audio_tags_stream(text)
            tts = self.session.tts
            if tts is not None and tts.capabilities.streaming:
                # The SDK's streaming synthesis, but with a short reply flushed
                # the moment the model's text stalls on a tool call.
                frames = _synthesize(tts, clean, self.session.conn_options.tts_conn_options)
            else:
                frames = super().tts_node(clean, model_settings)
            first = True
            async for frame in frames:
                if first:
                    runtime.mark(Stage.TTS_FIRST_BYTE)
                    runtime.mark(Stage.AUDIO_OUT)
                    first = False
                yield frame

    return InstrumentedAgent()


async def _synthesize(tts: Any, text: AsyncIterable[str], conn_options: Any) -> AsyncIterator[Any]:
    """Streaming synthesis as the SDK does it, with the text pushed by push_with_flush."""
    from livekit.agents import utils  # noqa: PLC0415

    async with tts.stream(conn_options=conn_options) as stream:
        push = asyncio.create_task(push_with_flush(text, stream))
        try:
            async for ev in stream:
                yield ev.frame
        finally:
            await utils.aio.cancel_and_wait(push)


_NOTE = re.compile(r"\(Not said by the caller:.*?\)", re.S)


def _same_words(a: str, b: str) -> bool:
    """Whether two transcripts say the same words, ignoring case, punctuation and notes.

    Notes the hook attached to a message are not the caller's words. Anything
    else is: "Tomorrow." and "Tomorrow, 11 to 2." are different turns.
    """
    def key(s: str) -> str:
        return " ".join(re.findall(r"[a-z0-9']+", _NOTE.sub(" ", s or "").lower()))

    ka = key(a)
    return bool(ka) and ka == key(b)


def _ends_with_caller(chat_ctx: Any) -> bool:
    """Whether the context ends on the caller's message -- the pass that answers them.

    The pass after a tool call ends on the tool's output instead.
    """
    items = getattr(chat_ctx, "items", None) or getattr(chat_ctx, "messages", None) or []
    items = list(items)
    if not items:
        return False
    last = items[-1]
    return getattr(last, "type", "message") == "message" and getattr(last, "role", None) == "user"


def _last_caller_id(chat_ctx: Any) -> str:
    """The id of the message the context ends on, when that is the caller's; '' otherwise.

    The SDK files a preemptive pass's message under a new id and the confirmed
    turn's under another, which it keeps when it regenerates after the hook. So
    the id tells the hook's own turn apart from an early pass, and from any later
    turn with the same words.
    """
    if not _ends_with_caller(chat_ctx):
        return ""
    items = list(getattr(chat_ctx, "items", None) or getattr(chat_ctx, "messages", None) or [])
    return str(getattr(items[-1], "id", "") or "")


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
