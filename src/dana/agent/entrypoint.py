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
from typing import Any

from dana.agent import greeting
from dana.agent.nodes import build_agent
from dana.agent.runtime import CallRuntime
from dana.agent.tools import DeskOperations
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
    _wire(session, runtime)
    watchdog = asyncio.ensure_future(_silence_watchdog(session, runtime))
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

    await session.say(runtime.tenant.greeting, **supported(session.say, say_kwargs))
    runtime.record_agent(runtime.tenant.greeting)


def _wire(session: Any, runtime: CallRuntime) -> None:
    """Attach conversation bookkeeping to session events.

    Only events that carry information the pipeline nodes cannot see are
    handled here. Timing is not among them.
    """

    @session.on("user_state_changed")  # type: ignore[attr-defined]
    def _on_user_state(ev: Any) -> None:
        if getattr(ev, "new_state", None) == "listening":
            runtime.begin_turn()

    @session.on("user_input_transcribed")  # type: ignore[attr-defined]
    def _on_transcript(ev: Any) -> None:
        text = getattr(ev, "transcript", "") or ""
        if getattr(ev, "is_final", False):
            runtime.record_user(text)

    @session.on("conversation_item_added")  # type: ignore[attr-defined]
    def _on_item(ev: Any) -> None:
        item = getattr(ev, "item", None)
        if getattr(item, "role", None) == "assistant":
            text = getattr(item, "text_content", None) or ""
            if text:
                runtime.record_agent(text)

    @session.on("close")  # type: ignore[attr-defined]
    def _on_close(_ev: Any) -> None:
        runtime.finish()


_AGENT_BUSY = frozenset({"initializing", "thinking", "speaking"})
"""Agent states that are not silence. `AgentState` is
initializing | idle | listening | thinking | speaking, and only the last two
mean nobody is doing anything."""


async def _silence_watchdog(session: Any, runtime: CallRuntime) -> None:
    """Re-engage a quiet line, then close it.

    The prompt says "if the line goes quiet mid-call, ask once, 'Still there?'"
    -- an instruction the agent could not follow, because a silent caller never
    produces a turn in which to follow it. `turn_timeout_s` and
    `silence_end_call_s` were config fields nothing read.
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
            # "thinking" and "initializing" are the agent working, not the line
            # going quiet. Counting them meant a slow turn walked its own way
            # into "Still there?" -- the agent interrupting its own pause to ask
            # the caller whether they were still on the phone.
            if getattr(session, "agent_state", None) in _AGENT_BUSY:
                quiet_for, asked = 0.0, False
                continue
            if getattr(session, "user_state", None) == "speaking":
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
