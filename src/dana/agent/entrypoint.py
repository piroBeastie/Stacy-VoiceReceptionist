"""LiveKit worker entrypoint: route an inbound call to its tenant and run it.

This module is deliberately the thinnest thing in the repo. Everything with
judgement in it -- prompt, tenant routing, backchannel policy, vendor choice,
metrics -- lives elsewhere and is unit-testable without a SIP trunk. What is
left here is glue, and glue is what breaks when an SDK version moves.

VERIFY AGAINST THE INSTALLED SDK before the first live call. The shape below
(AgentSession, on-event names, participant attribute keys for the dialed number)
is the part most likely to have shifted.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dana.agent.backchannel import BackchannelFilter
from dana.metrics import JsonlSink, Stage, TraceSink, TurnTrace
from dana.prompt import PromptBuilder
from dana.providers.registry import (
    build_llm,
    build_stt,
    build_tts,
    build_turn_detector,
    build_vad,
    pipeline_for,
)
from dana.tenants import TenantStore

log = logging.getLogger("dana.agent")

SIP_DIALED_ATTR = "sip.trunkPhoneNumber"
"""Participant attribute holding the number the caller dialed. Tenant routing key."""

SIP_CALLER_ATTR = "sip.phoneNumber"
"""Participant attribute holding the caller's own number, for {{system__caller_id}}."""


class CallRuntime:
    """Per-call state: which tenant, which prompt, and the turn traces so far.

    Built once per call rather than per turn, because the prompt render and the
    tenant lookup are pure overhead on the critical path if repeated.
    """

    def __init__(
        self,
        *,
        store: TenantStore,
        dialed_number: str,
        caller_id: str,
        call_id: str,
        sink: TraceSink,
    ) -> None:
        self.tenant = store.for_number(dialed_number)
        self.call_id = call_id
        self.caller_id = caller_id
        self.sink = sink
        self.turn_index = 0
        self._trace: TurnTrace | None = None

        builder = PromptBuilder()
        now = datetime.now(ZoneInfo(self.tenant.timezone)).strftime("%A, %B %d, %Y at %I:%M %p")
        self.system_prompt = builder.fill_runtime(
            builder.render(self.tenant), now=now, caller_id=caller_id
        )
        self.pipeline = pipeline_for(self.tenant)
        self.backchannel = BackchannelFilter(max_words=self.pipeline.turn.interrupt_min_words)

        log.info(
            "call %s -> tenant=%s prompt=%d chars caller=%s",
            call_id,
            self.tenant.tenant_id,
            len(self.system_prompt),
            caller_id or "<unknown>",
        )

    def begin_turn(self) -> TurnTrace:
        """Open a new trace at the moment the caller stops speaking."""
        self._trace = TurnTrace(
            call_id=self.call_id,
            tenant_id=self.tenant.tenant_id,
            turn_index=self.turn_index,
        )
        self.turn_index += 1
        self._trace.mark(Stage.USER_STOPPED)
        return self._trace

    def mark(self, stage: Stage, **meta: object) -> None:
        """Mark a stage on the open turn. A no-op if no turn is open."""
        if self._trace is None:
            return
        self._trace.mark(stage)
        if meta:
            self._trace.annotate(**meta)

    def end_turn(self) -> None:
        """Emit the open trace and close it."""
        if self._trace is None:
            return
        self.sink.emit(self._trace)
        self._trace = None


async def entrypoint(ctx: object) -> None:
    """LiveKit job entrypoint. One invocation per inbound call.

    Args:
        ctx: The livekit-agents JobContext. Typed loosely so this module imports
            without the SDK present, which keeps the test suite fast.
    """
    from livekit.agents import Agent, AgentSession, RoomInputOptions  # noqa: PLC0415

    await ctx.connect()
    participant = await ctx.wait_for_participant()
    attrs = getattr(participant, "attributes", {}) or {}

    dialed = attrs.get(SIP_DIALED_ATTR, "")
    caller_id = attrs.get(SIP_CALLER_ATTR, "")

    runtime = CallRuntime(
        store=TenantStore(os.getenv("DANA_TENANT_DIR", "src/dana/tenants/data")),
        dialed_number=dialed,
        caller_id=caller_id,
        call_id=ctx.room.name,
        sink=JsonlSink(os.getenv("DANA_TRACE_DIR", "data/traces")),
    )
    cfg = runtime.pipeline

    session = AgentSession(
        vad=build_vad(),
        stt=build_stt(cfg.stt),
        llm=build_llm(cfg.llm),
        tts=build_tts(cfg.tts),
        turn_detection=build_turn_detector(cfg.turn),
        min_endpointing_delay=cfg.turn.min_endpointing_delay_s,
        max_endpointing_delay=cfg.turn.max_endpointing_delay_s,
        preemptive_generation=cfg.turn.preemptive_generation,
    )

    _wire_metrics(session, runtime)

    await session.start(
        agent=Agent(instructions=runtime.system_prompt),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

    # The greeting is verbatim tenant config, not model output: the AI
    # disclosure and the recording notice have to be exact every single time.
    await session.say(runtime.tenant.greeting, allow_interruptions=False)


def _wire_metrics(session: object, runtime: CallRuntime) -> None:
    """Attach stage marks to session events.

    Event names are the part of this file most likely to drift between SDK
    versions; each handler is defensive so a renamed event degrades to a missing
    column in the report rather than a dropped call.
    """

    @session.on("user_state_changed")  # type: ignore[attr-defined]
    def _on_user_state(ev: object) -> None:
        if getattr(ev, "new_state", None) == "listening":
            runtime.begin_turn()

    @session.on("user_input_transcribed")  # type: ignore[attr-defined]
    def _on_transcript(ev: object) -> None:
        if getattr(ev, "is_final", False):
            runtime.mark(Stage.STT_FINAL)
        else:
            # Interim text is where the backchannel decision belongs: by the
            # time a transcript is final the agent has already stopped talking.
            text = getattr(ev, "transcript", "") or ""
            if text and runtime.backchannel.is_backchannel(text):
                log.debug("ignoring backchannel %r", text)

    @session.on("agent_state_changed")  # type: ignore[attr-defined]
    def _on_agent_state(ev: object) -> None:
        state = getattr(ev, "new_state", None)
        if state == "thinking":
            runtime.mark(Stage.ENDPOINT_COMMITTED)
        elif state == "speaking":
            runtime.mark(Stage.AUDIO_OUT)
            runtime.end_turn()

    @session.on("metrics_collected")  # type: ignore[attr-defined]
    def _on_metrics(ev: object) -> None:
        m = getattr(ev, "metrics", None)
        ttft = getattr(m, "ttft", None)
        if ttft is not None:
            runtime.mark(Stage.LLM_FIRST_TOKEN, ttft_reported=ttft)
        ttfb = getattr(m, "ttfb", None)
        if ttfb is not None:
            runtime.mark(Stage.TTS_FIRST_BYTE, ttfb_reported=ttfb)


def main() -> None:
    """Run the worker. `python -m dana.agent.entrypoint`."""
    from dotenv import load_dotenv  # noqa: PLC0415
    from livekit.agents import WorkerOptions, cli  # noqa: PLC0415

    load_dotenv()
    logging.basicConfig(level=os.getenv("DANA_LOG_LEVEL", "INFO"))
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()
