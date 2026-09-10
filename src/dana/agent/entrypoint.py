"""LiveKit worker entrypoint: route an inbound call to its tenant and run it.

This module is deliberately the thinnest thing in the repo. Everything with
judgement in it -- prompt, tenant routing, backchannel policy, filler policy,
tools, vendor choice, metrics -- lives elsewhere and is unit-testable without a
SIP trunk. What is left here is glue, and glue is what breaks when an SDK
version moves.

Two things that used to happen inside this file and no longer do:

**Model loading.** Silero and the end-of-utterance transformer were constructed
per call, which put a disk read and a session build between the caller hearing a
ring and the agent saying anything. They are loaded once per worker process in
:func:`prewarm` now, along with the tenant store, which was also being re-read
and re-validated on every single call.

**Timing.** Stage marks were derived from the session's `metrics_collected`
event, which reports durations after the fact and fires after the turn had
already been emitted. They are taken inside the pipeline nodes instead -- see
:mod:`dana.agent.nodes`.

VERIFY AGAINST THE INSTALLED SDK before the first live call. The shape below
(AgentSession keyword arguments, event names, participant attribute keys for the
dialed number) is the part most likely to have shifted.
:func:`dana.providers.registry.supported` already guards the plugin
constructors; the session arguments here are guarded the same way.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

from dana.agent import greeting
from dana.agent.nodes import build_agent
from dana.agent.runtime import CallRuntime
from dana.agent.tools import DeskOperations
from dana.agent.turns import NoiseHeldTurnGuard
from dana.delivery import CallArchive, LoggingProvider, SmsSender, WebhookSender
from dana.delivery.sms import ConsentMissing
from dana.desk import CalendarStore, MessageStore
from dana.metrics import JsonlSink
from dana.providers.base import PipelineConfig
from dana.providers.registry import (
    build_llm,
    build_noise_cancellation,
    build_stt,
    build_tts,
    build_turn_detector,
    build_vad,
    prewarm as prewarm_models,
    register_plugins,
    supported,
)
from dana.tenants import TenantStore

log = logging.getLogger("dana.agent")

SIP_DIALED_ATTR = "sip.trunkPhoneNumber"
"""Participant attribute holding the number the caller dialed. Tenant routing key."""

SIP_CALLER_ATTR = "sip.phoneNumber"
"""Participant attribute holding the caller's own number, for {{system__caller_id}}."""


def prewarm(proc: Any) -> None:
    """Load everything reusable once per worker process.

    Registered as `WorkerOptions(prewarm_fnc=...)`, so it runs before any job is
    assigned. Anything constructed here is paid for while the process is idle
    instead of while a caller is listening to silence.
    """
    prewarm_models(proc.userdata)
    try:
        proc.userdata["tenants"] = TenantStore(
            os.getenv("DANA_TENANT_DIR", "src/dana/tenants/data")
        )
    except Exception:  # noqa: BLE001 - a bad tenant file should surface, not crash the worker
        log.exception("tenant store failed to load at prewarm; will retry per call")


def _stores(proc_userdata: dict[str, Any]) -> tuple[TenantStore, CalendarStore, MessageStore]:
    """Fetch the prewarmed stores, building any that prewarm could not."""
    tenants = proc_userdata.get("tenants") or TenantStore(
        os.getenv("DANA_TENANT_DIR", "src/dana/tenants/data")
    )
    data_dir = os.getenv("DANA_DATA_DIR", "data")
    calendar = proc_userdata.setdefault("calendar", CalendarStore(f"{data_dir}/calendar"))
    messages = proc_userdata.setdefault("messages", MessageStore(f"{data_dir}/messages"))
    return tenants, calendar, messages


def _looks_dialed(value: Any) -> bool:
    """Is this actually a number somebody dialled?

    Emptiness is not the only way for it to be absent. `console` mode hands the
    entrypoint a MagicMock participant, so `attributes.get(...)` returns a mock
    -- which is truthy, sailed straight past an `if dialed:` guard, and was
    passed to the tenant lookup as if it were a phone number. Anything that is
    not a string shaped like E.164 is treated as "no dialed number", which is
    what the fallback exists for.
    """
    return isinstance(value, str) and bool(re.fullmatch(r"\+?[0-9]{7,15}", value.strip()))


def _route(tenants: TenantStore, dialed: Any) -> Any:
    """Pick the tenant for an inbound call.

    Normally the dialed number decides, and an unclaimed number is a loud
    failure -- a caller reaching the wrong business is the worst possible place
    to discover a routing mistake.

    `console` and `dev` modes have no SIP attributes at all, so the dialed
    number is empty and there is nothing to route on. `DANA_DEFAULT_TENANT`
    covers that case, because otherwise the fastest way to hear the agent --
    talking to it through your laptop mic -- raises KeyError before it says a
    word. It logs at warning level every time it is used: on a real trunk this
    firing means the SIP attribute name has changed, and silently answering as
    the wrong business would be much worse than a stack trace.
    """
    if _looks_dialed(dialed):
        return tenants.for_number(dialed.strip())

    fallback = os.getenv("DANA_DEFAULT_TENANT")
    if not fallback:
        raise KeyError(
            "no dialed number on the participant and DANA_DEFAULT_TENANT is unset. "
            "On a live trunk, check the SIP attribute name; for local testing, set "
            "DANA_DEFAULT_TENANT to a tenant_id."
        )
    log.warning(
        "no dialed number -- falling back to tenant %r. Expected in console/dev "
        "mode; on a live trunk this means SIP attributes are missing.",
        fallback,
    )
    return tenants.get(fallback)


def _turn_detector(cfg: PipelineConfig) -> Any | None:
    """The semantic end-of-utterance model, or None to fall back to raw VAD.

    Never fatal. Its weights are baked in at image build time and have been
    unreachable at runtime once already -- fetched as root, read as a non-root
    user -- and a missing model must not take the call down with it.

    Losing it is a real degradation, so it says so loudly: VAD alone endpoints
    on silence, which means the agent starts talking during the pause in the
    middle of "fourteen... Elm Street". Worth a warning on every call until it
    is fixed.
    """
    try:
        return build_turn_detector(cfg.turn)
    except Exception:  # noqa: BLE001
        log.exception(
            "semantic endpointing unavailable -- falling back to raw VAD, which "
            "will interrupt callers mid-sentence. Check that the model weights "
            "are readable by the user the worker runs as."
        )
        return None


def _webhook() -> WebhookSender | None:
    """The post-call consumer, if one is configured."""
    url, secret = os.getenv("DANA_WEBHOOK_URL"), os.getenv("DANA_WEBHOOK_SECRET")
    if not url or not secret:
        log.warning(
            "no DANA_WEBHOOK_URL/SECRET configured -- the agent's closing promise "
            "that someone gets the caller's details will not be kept"
        )
        return None
    return WebhookSender(url=url, secret=secret)


async def entrypoint(ctx: Any) -> None:
    """LiveKit job entrypoint. One invocation per inbound call."""
    from livekit.agents import AgentSession, RoomInputOptions  # noqa: PLC0415

    await ctx.connect()
    participant = await ctx.wait_for_participant()
    # Coerced to a real mapping: in console mode `participant` is a MagicMock,
    # so this attribute is a mock whose .get() returns another mock rather than
    # the default. Anything that is not already a dict is no attributes at all.
    raw_attrs = getattr(participant, "attributes", None)
    attrs: dict[str, Any] = raw_attrs if isinstance(raw_attrs, dict) else {}

    userdata = getattr(ctx.proc, "userdata", {}) if hasattr(ctx, "proc") else {}
    tenants, calendar, messages = _stores(userdata)

    runtime = CallRuntime.build(
        tenant=_route(tenants, attrs.get(SIP_DIALED_ATTR, "")),
        call_id=ctx.room.name,
        caller_id=attrs.get(SIP_CALLER_ATTR, ""),
        sink=JsonlSink(os.getenv("DANA_TRACE_DIR", "data/traces")),
        calendar=calendar,
        messages=messages,
    )
    cfg: PipelineConfig = runtime.pipeline

    session_kwargs: dict[str, Any] = {
        "vad": userdata.get("vad") or build_vad(),
        "stt": build_stt(cfg.stt),
        "llm": build_llm(cfg.llm),
        "tts": build_tts(cfg.tts),
        "turn_detection": userdata.get("turn_detector") or _turn_detector(cfg),
        "min_endpointing_delay": cfg.turn.min_endpointing_delay_s,
        "max_endpointing_delay": cfg.turn.max_endpointing_delay_s,
        "preemptive_generation": cfg.turn.preemptive_generation,
        # Held only in the backchannel filter, this classified correctly and
        # suppressed nothing -- the session is what owns the interruption.
        "min_interruption_words": cfg.turn.interrupt_min_words,
        "false_interruption_timeout": cfg.turn.false_interruption_timeout_s,
        "resume_false_interruption": True,
    }
    session = AgentSession(**supported(AgentSession, session_kwargs))

    ops = DeskOperations(runtime)
    # Words, not audio, decide whether a noisy line is the caller or the room;
    # see dana.agent.turns.
    guard = NoiseHeldTurnGuard(commit_after_s=cfg.turn.noise_commit_after_s)
    _wire(session, runtime, guard)
    watchdog = asyncio.ensure_future(_silence_watchdog(session, runtime, guard))
    ctx.add_shutdown_callback(lambda: _on_shutdown(runtime, watchdog))

    await session.start(
        agent=build_agent(runtime, ops),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            **supported(
                RoomInputOptions,
                # Isolates the caller from whoever else is in the room. Without
                # it a television counts as a participant in the conversation:
                # its speech is transcribed, endpointed and answered exactly as
                # the caller's is, because nothing downstream can tell them
                # apart. `supported` keeps an older SDK degrading to unfiltered
                # audio rather than raising while a call is connecting.
                {"noise_cancellation": build_noise_cancellation(cfg.turn)},
            )
        ),
    )

    # The greeting is verbatim tenant config, not model output: the AI
    # disclosure and the recording notice have to be exact every single time.
    # It stays interruptible by default -- a caller who talks over "thanks for
    # calling" was previously both ignored and untranscribed, losing their
    # opening sentence entirely.
    # Pre-rendered if we have it. The greeting is the one utterance fully known
    # before the phone rings, and synthesising it live puts 330-570ms of TTS
    # first-byte latency in the gap between the caller saying "hello?" and the
    # business answering -- the one stall no filler can cover, because it comes
    # before the first word. From cache it is under a millisecond.
    audio = await greeting.prepare(
        runtime.tenant, cfg.tts, f"{os.getenv('DANA_DATA_DIR', 'data')}/greetings", build_tts
    )
    say_kwargs: dict[str, Any] = {"allow_interruptions": cfg.turn.first_message_interruptible}
    if audio is not None:
        say_kwargs["audio"] = audio

    # Not recorded here as well: conversation_item_added records it, and doing
    # both counted a repeated line on every call before the caller spoke.
    await session.say(runtime.tenant.greeting, **supported(session.say, say_kwargs))


def _wire(session: Any, runtime: CallRuntime, guard: NoiseHeldTurnGuard) -> None:
    """Attach conversation bookkeeping, and the events that explain a bad call.

    Only events that carry information the pipeline nodes cannot see are
    handled here. Timing is not among them.

    The logging is diagnostic, and specific. When a caller reports that the
    agent did not speak, there are three different ways that happens, with
    opposite fixes: noise interrupting the agent (an overlap judged to be an
    interruption, or a reply marked cut off), noise holding the caller's turn
    open (a stretch of "speech" with no words in it), and the input filter
    removing the caller (turns with no transcript at all). Each leaves a line.
    """
    state: dict[str, float | int] = {}

    @session.on("user_state_changed")  # type: ignore[attr-defined]
    def _on_user_state(ev: Any) -> None:
        new = getattr(ev, "new_state", None)
        now = time.monotonic()
        if new == "speaking":
            state["speaking_since"] = now
            state["words"] = 0
        elif getattr(ev, "old_state", None) == "speaking" and "speaking_since" in state:
            held = now - float(state.pop("speaking_since"))
            if held >= 1.5 and not state.get("words"):
                log.info(
                    "call %s: %.1fs of speech with no words in it -- the room, not the caller",
                    runtime.call_id,
                    held,
                )
        if new == "listening":
            runtime.begin_turn()

    @session.on("user_input_transcribed")  # type: ignore[attr-defined]
    def _on_transcript(ev: Any) -> None:
        text = getattr(ev, "transcript", "") or ""
        final = bool(getattr(ev, "is_final", False))
        if text.strip():
            state["words"] = int(state.get("words", 0)) + 1
            guard.on_words(time.monotonic(), text=text, final=final)
        if final:
            runtime.record_user(text)

    @session.on("agent_state_changed")  # type: ignore[attr-defined]
    def _on_agent_state(ev: Any) -> None:
        if getattr(ev, "new_state", None) in ("thinking", "speaking"):
            guard.on_agent_reply()

    @session.on("conversation_item_added")  # type: ignore[attr-defined]
    def _on_item(ev: Any) -> None:
        item = getattr(ev, "item", None)
        if getattr(item, "role", None) != "assistant":
            return
        text = getattr(item, "text_content", None) or ""
        if not text:
            return
        runtime.record_agent(text)
        cut = " (CUT OFF)" if getattr(item, "interrupted", False) else ""
        log.info("said %s%s: %r", runtime.call_id, cut, " ".join(text.split())[:160])

    @session.on("overlapping_speech")  # type: ignore[attr-defined]
    def _on_overlap(ev: Any) -> None:
        log.info(
            "call %s: caller audio over the agent -- interruption=%s p=%.2f over %.2fs",
            runtime.call_id,
            getattr(ev, "is_interruption", None),
            float(getattr(ev, "probability", 0.0) or 0.0),
            float(getattr(ev, "total_duration", 0.0) or 0.0),
        )

    @session.on("agent_false_interruption")  # type: ignore[attr-defined]
    def _on_false_interruption(ev: Any) -> None:
        log.info(
            "call %s: false interruption -- agent %s",
            runtime.call_id,
            "resumed" if getattr(ev, "resumed", False) else "did not resume",
        )

    @session.on("close")  # type: ignore[attr-defined]
    def _on_close(_ev: Any) -> None:
        runtime.finish()


_AGENT_BUSY = frozenset({"initializing", "thinking", "speaking"})
"""Agent states that are not silence. `AgentState` is
initializing | idle | listening | thinking | speaking, and only the last two
mean nobody is doing anything."""


def _log_commit_failure(fut: Any) -> None:
    """Surface a failed turn commit without letting it end the call."""
    if fut.cancelled():
        return
    exc = fut.exception()
    if exc is not None:
        log.warning("committing a noise-held turn failed: %s", exc)


async def _silence_watchdog(
    session: Any, runtime: CallRuntime, guard: NoiseHeldTurnGuard | None = None
) -> None:
    """Re-engage a quiet line, close an abandoned one, and answer through noise.

    The prompt says "if the line goes quiet mid-call, ask once, 'Still there?'"
    -- an instruction the agent could not follow, because a silent caller never
    produces a turn in which to follow it.

    Two things decide what counts as quiet, and both were wrong once. The agent
    thinking or speaking is not silence. And the caller "speaking" is only the
    caller if words are arriving: background noise holds voice detection on
    indefinitely, which both kept this timer from ever firing and kept a
    finished turn from ever ending. When a transcript is waiting and only the
    noise is still "talking", the turn is committed so the agent answers what
    was said. See :mod:`dana.agent.turns`.
    """
    turn_timeout = runtime.pipeline.turn.turn_timeout_s
    end_after = runtime.pipeline.turn.silence_end_call_s
    if turn_timeout <= 0:
        return

    asked = False
    quiet_for = 0.0
    tick = 0.5
    try:
        while True:
            await asyncio.sleep(tick)
            now = time.monotonic()
            agent_state = getattr(session, "agent_state", None)
            user_state = getattr(session, "user_state", None)

            if guard is not None and guard.should_commit(
                now, user_state=user_state, agent_state=agent_state
            ):
                guard.mark_committed()
                log.info(
                    "call %s: noise held the turn open %.1fs after the last word -- answering",
                    runtime.call_id,
                    now - (guard.last_words_at or now),
                )
                try:
                    fut = session.commit_user_turn(transcript_timeout=1.0, stt_flush_duration=0.5)
                    if hasattr(fut, "add_done_callback"):
                        fut.add_done_callback(_log_commit_failure)
                except Exception:  # noqa: BLE001 - a failed commit must not end the call
                    log.exception("call %s: could not commit the noise-held turn", runtime.call_id)

            if agent_state in _AGENT_BUSY:
                quiet_for, asked = 0.0, False
                continue
            caller_talking = user_state == "speaking" and (
                guard is None or guard.heard_words_within(now, 2.0)
            )
            if caller_talking:
                quiet_for, asked = 0.0, False
                continue
            quiet_for += tick

            if not asked and quiet_for >= turn_timeout:
                asked = True
                runtime.annotate(reengaged=True)
                await session.say("Still there?", allow_interruptions=True)
            elif 0 < end_after <= quiet_for:
                log.info("call %s closing after %.0fs of silence", runtime.call_id, quiet_for)
                runtime.outcome = "abandoned" if not runtime.transcript else runtime.outcome
                await session.aclose()
                return
    except asyncio.CancelledError:
        return


async def _on_shutdown(runtime: CallRuntime, watchdog: asyncio.Future[Any]) -> None:
    """Flush the last turn, text the confirmation, and deliver the call.

    Ordering matters: the trace is flushed first so the latency summary in the
    payload includes the final turn, and the SMS is attempted before the webhook
    so a consent refusal shows up in the delivered record.
    """
    watchdog.cancel()
    runtime.finish()

    sender = SmsSender(provider=LoggingProvider())
    for appt in runtime.appointments:
        try:
            sender.send_confirmation(runtime.tenant, appt)
        except ConsentMissing:
            # Expected whenever the caller declined. The booking still stands;
            # only the text is withheld. Consent is per booking, so a call that
            # produced two is evaluated twice.
            log.info("no SMS for %s: consent was not captured", appt.appointment_id)

    payload = runtime.post_call_payload()

    # Archive before delivering. The webhook can be down, misconfigured, or not
    # set up yet; none of those should mean the call went unrecorded, and the
    # eval harness reads its corpus from here.
    CallArchive(f"{os.getenv('DANA_DATA_DIR', 'data')}/calls").write(payload)

    webhook = _webhook()
    if webhook is not None:
        await webhook.send(payload)


def main() -> None:
    """Run the worker. `python -m dana.agent.entrypoint`."""
    from dotenv import load_dotenv  # noqa: PLC0415
    from livekit.agents import WorkerOptions, cli  # noqa: PLC0415

    load_dotenv()
    logging.basicConfig(level=os.getenv("DANA_LOG_LEVEL", "INFO"))

    # Must happen here, on the main thread. livekit-agents registers a plugin
    # when its module is imported and refuses to do so from a worker thread, so
    # leaving it to the first lazy import inside prewarm made prewarm fail and
    # pushed model loading into the first call, where the caller hears it.
    register_plugins()

    opts: dict[str, Any] = {"entrypoint_fnc": entrypoint, "prewarm_fnc": prewarm}

    # Naming the worker switches LiveKit from automatic dispatch (join every
    # room in the project) to explicit (join only rooms whose dispatch rule
    # names you). Explicit is correct once a project runs more than one kind of
    # agent, and is a silent failure if the rule does not match: the call
    # connects, the room is created, and nobody joins it. `scripts/sip_config.py`
    # reads the same variable so the rule it emits cannot disagree.
    agent_name = os.getenv("DANA_AGENT_NAME", "").strip()
    if agent_name:
        opts["agent_name"] = agent_name
        log.info("registering as %r -- only rooms dispatching this name get an agent", agent_name)

    cli.run_app(WorkerOptions(**supported(WorkerOptions, opts)))


if __name__ == "__main__":
    main()
