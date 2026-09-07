"""Per-call state, with no SDK in sight.

Everything with judgement in it lives here so it can be tested without a SIP
trunk: which tenant, which prompt, what has been said, what was booked, and
whether the agent should cover the next wait with speech. :mod:`entrypoint` is
left holding only the glue.

The turn lifecycle is the part that changed most from the original. A trace used
to be emitted the moment the agent started speaking, which meant every mark that
arrives after that point -- which on a cascade is most of the interesting ones --
landed on a closed trace and was silently dropped. A turn is now closed by the
*next* turn beginning, or by the call ending, so late marks still find their
home.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from dana.agent.backchannel import BackchannelFilter
from dana.agent.filler import FillerPolicy, RepetitionTracker
from dana.delivery.webhook import call_payload
from dana.desk.calendar import Appointment, CalendarStore
from dana.desk.messages import Message, MessageStore
from dana.metrics import NullSink, Stage, TraceSink, TurnTrace
from dana.prompt import PromptBuilder
from dana.providers.base import PipelineConfig
from dana.providers.registry import pipeline_for
from dana.tenants.models import Tenant

log = logging.getLogger("dana.agent")

Outcome = Literal["answered", "booked", "dispatched", "message", "abandoned"]


@dataclass
class CallRuntime:
    """One call's state: tenant, prompt, traces, transcript and outcome.

    Built once per call rather than per turn -- the prompt render and the tenant
    lookup are pure overhead on the critical path if repeated.
    """

    tenant: Tenant
    call_id: str
    caller_id: str = ""
    sink: TraceSink = field(default_factory=NullSink)
    calendar: CalendarStore | None = None
    messages: MessageStore | None = None
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    system_prompt: str = ""
    started_at: float = field(default_factory=time.monotonic)
    turn_index: int = 0
    outcome: Outcome = "answered"
    appointments: list[Appointment] = field(default_factory=list)
    """Every booking made on this call, in order.

    Plural because a single call really can produce two -- "and can you look at
    the AC while you're here Thursday?" -- and a scalar silently dropped the
    first one, taking its SMS consent flag with it.
    """

    message: Message | None = None
    transcript: list[dict[str, str]] = field(default_factory=list)

    backchannel: BackchannelFilter = field(default_factory=BackchannelFilter)
    filler: FillerPolicy = field(default_factory=FillerPolicy)
    repetition: RepetitionTracker = field(default_factory=RepetitionTracker)

    _trace: TurnTrace | None = None

    @classmethod
    def build(
        cls,
        *,
        tenant: Tenant,
        call_id: str,
        caller_id: str = "",
        sink: TraceSink | None = None,
        calendar: CalendarStore | None = None,
        messages: MessageStore | None = None,
        overrides: PipelineConfig | None = None,
        now: datetime | None = None,
    ) -> "CallRuntime":
        """Assemble a runtime, rendering the prompt and resolving vendors once."""
        cfg = pipeline_for(tenant, overrides=overrides)
        builder = PromptBuilder()
        stamp = (now or datetime.now(ZoneInfo(tenant.timezone))).strftime(
            "%A, %B %d, %Y at %I:%M %p"
        )
        prompt = builder.fill_runtime(
            builder.render(tenant), now=stamp, caller_id=caller_id
        )
        runtime = cls(
            tenant=tenant,
            call_id=call_id,
            caller_id=caller_id,
            sink=sink or NullSink(),
            calendar=calendar,
            messages=messages,
            pipeline=cfg,
            system_prompt=prompt,
            backchannel=BackchannelFilter(max_words=cfg.turn.interrupt_min_words),
            filler=FillerPolicy(cfg=cfg.filler),
        )
        log.info(
            "call %s -> tenant=%s prompt=%d chars caller=%s",
            call_id,
            tenant.tenant_id,
            len(prompt),
            caller_id or "<unknown>",
        )
        return runtime

    # ---- turn lifecycle -------------------------------------------------

    def begin_turn(self) -> TurnTrace:
        """Open a trace at the moment the caller stops speaking.

        Closes the previous turn first. Deferring the flush this way is what
        lets a stage that reports late still land on the turn it belongs to.
        """
        self.flush_turn()
        self._trace = TurnTrace(
            call_id=self.call_id,
            tenant_id=self.tenant.tenant_id,
            turn_index=self.turn_index,
        )
        self.turn_index += 1
        self._trace.mark(Stage.USER_STOPPED)
        self.filler.begin_turn()
        return self._trace

    def mark(self, stage: Stage, **meta: Any) -> None:
        """Mark a stage on the open turn. A no-op if no turn is open."""
        if self._trace is None:
            return
        self._trace.mark(stage)
        if meta:
            self._trace.annotate(**meta)

    def annotate(self, **meta: Any) -> None:
        """Attach context to the open turn without marking a stage."""
        if self._trace is not None:
            self._trace.annotate(**meta)

    def flush_turn(self) -> None:
        """Emit the open trace, if any, and close it."""
        if self._trace is None:
            return
        self.sink.emit(self._trace)
        self._trace = None

    @property
    def open_trace(self) -> TurnTrace | None:
        """The turn currently being measured, for tests and node instrumentation."""
        return self._trace

    # ---- conversation ---------------------------------------------------

    def record_user(self, text: str) -> None:
        """Log a caller utterance and unlock fillers."""
        if not text.strip():
            return
        self.transcript.append({"role": "user", "text": text})
        self.filler.on_user_message()

    def record_agent(self, text: str) -> None:
        """Log an agent utterance and count it if it repeats an earlier one."""
        if not text.strip():
            return
        self.transcript.append({"role": "agent", "text": text})
        if self.repetition.observe(text):
            log.info("call %s repeated a line: %r", self.call_id, text[:60])
            self.annotate(repeat_detected=True)

    def should_interrupt(self, partial: str) -> bool:
        """Whether a partial transcript is a real interruption or a 'mhm'."""
        return self.backchannel.should_interrupt(partial)

    # ---- outcome --------------------------------------------------------

    @property
    def appointment(self) -> Appointment | None:
        """The most recent booking, for the common single-booking call."""
        return self.appointments[-1] if self.appointments else None

    def note_booking(self, appt: Appointment) -> None:
        """Record a completed booking as the call's outcome.

        An emergency dispatch outranks a routine booking made earlier in the
        same call: it is the thing the office needs to see first.
        """
        self.appointments.append(appt)
        if appt.kind == "emergency" or self.outcome != "dispatched":
            self.outcome = "dispatched" if appt.kind == "emergency" else "booked"

    def note_message(self, msg: Message) -> None:
        """Record a taken message, unless a booking already happened."""
        self.message = msg
        if self.outcome == "answered":
            self.outcome = "message"

    @property
    def duration_s(self) -> float:
        return time.monotonic() - self.started_at

    def post_call_payload(self, latency: dict[str, float] | None = None) -> dict[str, Any]:
        """Everything a consumer needs to act on the call, ready to sign and send."""
        return call_payload(
            call_id=self.call_id,
            tenant_id=self.tenant.tenant_id,
            caller_id=self.caller_id,
            transcript=list(self.transcript),
            outcome=self.outcome,
            duration_s=self.duration_s,
            appointments=[
                {
                    "appointment_id": a.appointment_id,
                    "day": a.day,
                    "window": a.window_label,
                    "kind": a.kind,
                    "customer_name": a.customer_name,
                    "phone": a.phone,
                    "street_address": a.street_address,
                    "town": a.town,
                    "problem": a.problem,
                    "sms_consent": a.sms_consent,
                }
                for a in self.appointments
            ],
            message=(
                {
                    "message_id": self.message.message_id,
                    "caller_name": self.message.caller_name,
                    "phone": self.message.phone,
                    "summary": self.message.summary,
                    "priority": self.message.priority,
                    "for_staff": self.message.for_staff,
                }
                if self.message
                else None
            ),
            latency=latency,
        )

    def finish(self) -> None:
        """Close the last open turn. Call once, when the room closes."""
        self.flush_turn()
