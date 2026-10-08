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

**Timing.** Caller-heard latency comes from the SDK's timestamps on each
conversation item, logged once per reply and summarised once per call -- see
:mod:`stacy.metrics.live`. The stage marks in :mod:`stacy.agent.nodes` still
feed the JSONL trace, but they start late and mislabel tool time; do not read
latency from them.

VERIFY AGAINST THE INSTALLED SDK before the first live call. The shape below
(AgentSession keyword arguments, event names, participant attribute keys for the
dialed number) is the part most likely to have shifted.
:func:`stacy.providers.registry.supported` already guards the plugin
constructors; the session arguments here are guarded the same way.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

from stacy.agent import greeting
from stacy.agent.fastlane import answer_kind
from stacy.agent.hangup import farewell_said, hang_up
from stacy.agent.nodes import _same_words, build_agent
from stacy.agent.runtime import CallRuntime
from stacy.agent.speech import TTS_TEXT_TRANSFORMS
from stacy.agent.tools import DeskOperations, build_function_tools
from stacy.agent.turns import NoiseHeldTurnGuard
from stacy.bench import BENCH_PREFIX
from stacy.delivery import CallArchive, LoggingProvider, SmsSender, WebhookSender
from stacy.delivery.sms import ConsentMissing
from stacy.desk import CalendarStore, MessageStore
from stacy.metrics import JsonlSink
from stacy.metrics.live import format_reply
from stacy.providers.base import PipelineConfig
from stacy.providers.registry import (
    build_llm,
    build_noise_cancellation,
    build_stt,
    build_tts,
    build_turn_detector,
    build_vad,
    llm_conn_options,
    register_plugins,
    supported,
    turn_handling,
    warm_llm_cache,
)
from stacy.providers.registry import (
    prewarm as prewarm_models,
)
from stacy.tenants import TenantStore

log = logging.getLogger("stacy.agent")

_BACKGROUND: set[asyncio.Future[Any]] = set()
"""Fire-and-forget tasks, held so they aren't collected before they finish."""

SIP_DIALED_ATTR = "sip.trunkPhoneNumber"
"""Participant attribute holding the number the caller dialed. Tenant routing key."""

SIP_CALLER_ATTR = "sip.phoneNumber"
"""Participant attribute holding the caller's own number, for {{system__caller_id}}."""

DEFAULT_TENANT_DIR = "src/stacy/tenants/data"

KEEP_WARM_PREFIX = "keepwarm-"
"""Rooms that exist only to keep the worker awake; see scripts/keep_warm.py."""


def _tenant_dir() -> str:
    """Where tenant records are read from. `STACY_TENANT_DIR` overrides it."""
    return os.getenv("STACY_TENANT_DIR", DEFAULT_TENANT_DIR)


def prewarm(proc: Any) -> None:
    """Load everything reusable once per worker process.

    Registered as `WorkerOptions(prewarm_fnc=...)`, so it runs before any job is
    assigned. Anything constructed here is paid for while the process is idle
    instead of while a caller is listening to silence.
    """
    prewarm_models(proc.userdata)
    try:
        proc.userdata["tenants"] = TenantStore(_tenant_dir())
    except Exception:  # noqa: BLE001 - a bad tenant file should surface, not crash the worker
        log.exception("tenant store failed to load at prewarm; will retry per call")


def _stores(proc_userdata: dict[str, Any]) -> tuple[TenantStore, CalendarStore, MessageStore]:
    """Fetch the prewarmed stores, building any that prewarm could not."""
    tenants = proc_userdata.get("tenants") or TenantStore(_tenant_dir())
    data_dir = os.getenv("STACY_DATA_DIR", "data")
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


def _runtime_for(ctx: Any, room_name: str) -> CallRuntime:
    """A runtime for a job that is not a call: the tenant's own prompt and tools.

    Used by the model bench, which needs exactly what a call would send and
    nothing that a call would record.
    """
    userdata = getattr(ctx.proc, "userdata", {}) if hasattr(ctx, "proc") else {}
    tenants, calendar, messages = _stores(userdata)
    first = os.getenv("STACY_DEFAULT_TENANT") or next(iter(tenants.all()))
    return CallRuntime.build(
        tenant=tenants.get(first) if isinstance(first, str) else first,
        call_id=room_name,
        caller_id="",
        calendar=calendar,
        messages=messages,
    )


def _route(tenants: TenantStore, dialed: Any) -> Any:
    """Pick the tenant for an inbound call.

    Normally the dialed number decides, and an unclaimed number is a loud
    failure -- a caller reaching the wrong business is the worst possible place
    to discover a routing mistake.

    `console` and `dev` modes have no SIP attributes at all, so the dialed
    number is empty and there is nothing to route on. `STACY_DEFAULT_TENANT`
    covers that case, because otherwise the fastest way to hear the agent --
    talking to it through your laptop mic -- raises KeyError before it says a
    word. It logs at warning level every time it is used: on a real trunk this
    firing means the SIP attribute name has changed, and silently answering as
    the wrong business would be much worse than a stack trace.
    """
    if _looks_dialed(dialed):
        return tenants.for_number(dialed.strip())

    fallback = os.getenv("STACY_DEFAULT_TENANT")
    if not fallback:
        raise KeyError(
            "no dialed number on the participant and STACY_DEFAULT_TENANT is unset. "
            "On a live trunk, check the SIP attribute name; for local testing, set "
            "STACY_DEFAULT_TENANT to a tenant_id."
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
    url, secret = os.getenv("STACY_WEBHOOK_URL"), os.getenv("STACY_WEBHOOK_SECRET")
    if not url or not secret:
        log.warning(
            "no STACY_WEBHOOK_URL/SECRET configured -- the agent's closing promise "
            "that someone gets the caller's details will not be kept"
        )
        return None
    return WebhookSender(url=url, secret=secret)


async def entrypoint(ctx: Any) -> None:
    """LiveKit job entrypoint. One invocation per inbound call."""
    room_name = getattr(getattr(getattr(ctx, "job", None), "room", None), "name", "") or ""
    if room_name.startswith(KEEP_WARM_PREFIX):
        # Not a call: a ping that stops the worker scaling to zero. A sleeping
        # worker takes 20-30s to wake, and the call that wakes it rings out.
        log.info("keep-warm ping %s", room_name)
        return
    if room_name.startswith(BENCH_PREFIX):
        # Not a call either: timing the models from the worker, where the
        # network distance is production's. See stacy.bench.llm_latency.
        from stacy.bench import run as bench_run  # noqa: PLC0415

        runtime = _runtime_for(ctx, room_name)
        metadata = getattr(getattr(ctx, "job", None), "metadata", "") or ""
        await bench_run(runtime, build_function_tools(DeskOperations(runtime)), metadata)
        return

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
        sink=JsonlSink(os.getenv("STACY_TRACE_DIR", "data/traces")),
        calendar=calendar,
        messages=messages,
    )
    cfg: PipelineConfig = runtime.pipeline

    session_kwargs: dict[str, Any] = {
        "vad": userdata.get("vad") or build_vad(),
        "stt": build_stt(cfg.stt),
        "llm": build_llm(cfg.llm),
        "tts": build_tts(cfg.tts),
        # Endpointing, interruption and preemptive generation -- now voiced
        # early as well as written early -- in the SDK's current shape. The
        # session owns interruption; see registry.turn_handling.
        "turn_handling": turn_handling(
            cfg.turn, userdata.get("turn_detector") or _turn_detector(cfg)
        ),
        # A stuck model request is retried after seconds, not the SDK's ten.
        "conn_options": llm_conn_options(cfg.llm),
        # Without the SDK's markdown filter, which held each reply's last word
        # until the model had written its tool call. See stacy.agent.speech.
        "tts_text_transforms": TTS_TEXT_TRANSFORMS,
    }
    session = AgentSession(**supported(AgentSession, session_kwargs))

    ops = DeskOperations(runtime)
    # Started before anything else so it finishes while the greeting plays.
    warm = asyncio.ensure_future(_warm_model(runtime, ops))
    _BACKGROUND.add(warm)
    warm.add_done_callback(_BACKGROUND.discard)
    # Words, not audio, decide whether a noisy line is the caller or the room;
    # see stacy.agent.turns.
    guard = NoiseHeldTurnGuard(commit_after_s=cfg.turn.noise_commit_after_s)
    _wire(session, runtime, guard, session_kwargs["vad"], ctx)
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
    # before the first word. From cache it is under a millisecond. On a miss it
    # is spoken live and recorded in the background for the next call.
    audio = await greeting.prepare(
        runtime.tenant, cfg.tts, f"{os.getenv('STACY_DATA_DIR', 'data')}/greetings", build_tts
    )
    say_kwargs: dict[str, Any] = {"allow_interruptions": cfg.turn.first_message_interruptible}
    if audio is not None:
        say_kwargs["audio"] = audio

    # Not recorded here as well: conversation_item_added records it, and doing
    # both counted a repeated line on every call before the caller spoke.
    await session.say(runtime.tenant.greeting, **supported(session.say, say_kwargs))


async def _warm_model(runtime: CallRuntime, ops: DeskOperations) -> None:
    """Warm the model's prompt cache for this call. Never allowed to break the call."""
    try:
        # In a thread: building the client loads a CA bundle, and the tool schemas
        # are parsed, both synchronous -- on the event loop that is audio held up.
        warmed = await asyncio.to_thread(
            lambda: warm_llm_cache(
                runtime.pipeline.llm,
                runtime.instructions or runtime.system_prompt,
                build_function_tools(ops),
            )
        )
    except Exception as exc:  # noqa: BLE001 -- a failed warm-up only costs the old cold start
        log.warning("call %s: model warm-up failed: %s", runtime.call_id, exc)
        return
    if warmed:
        log.info(
            "call %s: model cache warmed in %dms (wrote %d tokens, read %d)",
            runtime.call_id, warmed["ms"], warmed["written"], warmed["read"],
        )


def _pace(vad: Any, runtime: CallRuntime, state: dict[str, Any]) -> None:
    """Wait longer for a number or an address, less for a yes or no. See fastlane.answer_kind."""
    update = getattr(vad, "update_options", None)
    if update is None:
        return
    turn = runtime.pipeline.turn
    kind = answer_kind(runtime)
    pause = {"patient": turn.vad_patient_silence_s, "eager": turn.vad_eager_silence_s}.get(
        kind, turn.vad_min_silence_s
    )
    if state.get("pause") == pause:
        return
    try:
        update(min_silence_duration=pause)
    except Exception as exc:  # noqa: BLE001 -- the default wait is still a working call
        log.warning("call %s: could not change the silence wait: %s", runtime.call_id, exc)
        return
    state["pause"] = pause
    log.info("call %s: waiting %.2fs of silence for a %s answer", runtime.call_id, pause, kind or "normal")


def _wire(
    session: Any,
    runtime: CallRuntime,
    guard: NoiseHeldTurnGuard,
    vad: Any = None,
    ctx: Any = None,
) -> None:
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
    state: dict[str, Any] = {}
    # The VAD belongs to the worker process and outlives the call, so a wait
    # set on the last call is put back first.
    state["pause"] = None
    _pace(vad, runtime, state)

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
        # Latency comes from the SDK's timestamps on the items themselves: the
        # caller's message says when they stopped, the reply says when its
        # audio started. One log line per reply, because nothing on disk
        # survives the worker. See stacy.metrics.live.
        item = getattr(ev, "item", None)
        role = getattr(item, "role", None)
        metrics = getattr(item, "metrics", None)
        metrics = metrics if isinstance(metrics, dict) else None
        if role == "user":
            runtime.latency.on_user(metrics)
            kept, runtime.kept_note = runtime.kept_note, None
            content = getattr(item, "content", None)
            if kept and isinstance(content, list) and _same_words(kept[0], getattr(item, "text_content", "") or ""):
                # The desk's note for a kept early reply, added now the SDK has
                # filed the message. Later passes read the history from here.
                content.append(kept[1])
            return
        if role != "assistant":
            return
        text = getattr(item, "text_content", None) or ""
        record = runtime.latency.on_reply(
            metrics, interrupted=bool(getattr(item, "interrupted", False)), chars=len(text)
        )
        if record is not None:
            log.info("latency %s", format_reply(runtime.call_id, record))
        if not text:
            return
        runtime.record_agent(text)
        cut = " (CUT OFF)" if getattr(item, "interrupted", False) else ""
        log.info("said %s%s: %r", runtime.call_id, cut, " ".join(text.split())[:160])
        _pace(vad, runtime, state)

        # A receptionist says bye and puts the phone down. See stacy.agent.hangup:
        # decided from the transcript, never by the model, and only once the
        # caller has said they are done.
        reason = ""
        if ctx is not None:
            # The intent is consumed either way. A goodbye the caller talked
            # over is not a goodbye, and leaving the intent set would hang up
            # a turn later, after they had started a new sentence.
            reason, runtime.end_call = runtime.end_call, ""
            reason = "" if cut else (reason or farewell_said(runtime))
        if reason:
            ending = asyncio.ensure_future(hang_up(ctx, session, runtime, reason))
            _BACKGROUND.add(ending)
            ending.add_done_callback(_BACKGROUND.discard)

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
    was said. See :mod:`stacy.agent.turns`.
    """
    turn_timeout = runtime.pipeline.turn.turn_timeout_s
    end_after = runtime.pipeline.turn.silence_end_call_s
    if turn_timeout <= 0:
        return

    asked = False
    quiet_for = 0.0
    tick = 0.25  # the noise-held turn is committed within a quarter second of its deadline
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

    Ordering matters: the SMS is attempted before the webhook so a consent
    refusal shows up in the delivered record. The latency summary is logged as
    one line as well as put in the payload: with no webhook configured, the log
    is the only place it outlives the worker.
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

    latency = runtime.latency.summary()
    log.info(
        "call_summary %s",
        json.dumps(
            {
                "call": runtime.call_id,
                "tenant": runtime.tenant.tenant_id,
                "outcome": runtime.outcome,
                "duration_s": round(runtime.duration_s, 1),
                **latency,
            },
            separators=(",", ":"),
        ),
    )
    payload = runtime.post_call_payload(latency=latency)

    # Archive before delivering. The webhook can be down, misconfigured, or not
    # set up yet; none of those should mean the call went unrecorded, and the
    # eval harness reads its corpus from here.
    CallArchive(f"{os.getenv('STACY_DATA_DIR', 'data')}/calls").write(payload)

    webhook = _webhook()
    if webhook is not None:
        await webhook.send(payload)


def _require_tenants() -> int:
    """Refuse to start a worker that cannot route a single call.

    Every call is routed by the number dialed, so a worker with no tenants still
    accepts each job and then crashes on it, and the caller hears ringing until
    the carrier gives up. The dana-to-stacy rename did exactly that:
    STACY_TENANT_DIR still named src/dana/tenants/data, the image had no such
    folder, the store loaded nothing, and every call rang out while the agent
    reported Running. Exiting here turns that into a failed deploy instead.
    """
    tenant_dir = _tenant_dir()
    try:
        count = len(TenantStore(tenant_dir).all())
    except Exception as exc:  # noqa: BLE001 - any load failure means nothing routes
        raise SystemExit(
            f"cannot load tenants from {tenant_dir!r} (STACY_TENANT_DIR): {exc}"
        ) from exc
    if not count:
        raise SystemExit(
            f"no tenants in {tenant_dir!r} (STACY_TENANT_DIR) -- every call would ring out"
        )
    return count


def main() -> None:
    """Run the worker. `python -m stacy.agent.entrypoint`."""
    from dotenv import load_dotenv  # noqa: PLC0415
    from livekit.agents import WorkerOptions, cli  # noqa: PLC0415

    load_dotenv()
    logging.basicConfig(level=os.getenv("STACY_LOG_LEVEL", "INFO"))
    log.info("routing calls for %d tenant(s) from %s", _require_tenants(), _tenant_dir())

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
    agent_name = os.getenv("STACY_AGENT_NAME", "").strip()
    if agent_name:
        opts["agent_name"] = agent_name
        log.info("registering as %r -- only rooms dispatching this name get an agent", agent_name)

    cli.run_app(WorkerOptions(**supported(WorkerOptions, opts)))


if __name__ == "__main__":
    main()
