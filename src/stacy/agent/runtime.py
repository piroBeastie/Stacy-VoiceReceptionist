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

from stacy.agent.backchannel import BackchannelFilter
from stacy.agent.filler import FillerPolicy, RepetitionTracker
from stacy.agent.safety import SafetyInterceptor
from stacy.agent.slots import BookingSlots
from stacy.delivery.webhook import call_payload
from stacy.desk.calendar import Appointment, CalendarStore
from stacy.desk.messages import Message, MessageStore
from stacy.metrics import CallLatency, NullSink, Stage, TraceSink, TurnTrace
from stacy.prompt import PromptBuilder
from stacy.providers.base import PipelineConfig
from stacy.providers.registry import pipeline_for
from stacy.tenants.models import Tenant

log = logging.getLogger("stacy.agent")

Outcome = Literal[
    "answered", "booked", "dispatched", "message", "abandoned", "safety_override"
]


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

    instructions: str = ""
    """The system prompt minus its `# This call` section: byte-identical on every call.

    Sent as the agent's instructions. Measured against the live API, a prompt
    carrying the caller's number and the minute read 0 of 7,692 tokens from cache
    on every new call; without them the stable prefix is shared between calls.
    """

    call_section: str = ""
    """The filled `# This call` section, sent in the call's first message instead."""

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
    safety: SafetyInterceptor = field(default_factory=SafetyInterceptor)
    slots: BookingSlots = field(default_factory=BookingSlots)
    """Working memory for the booking. Callers give three fields in one breath;
    this is where they land so the agent never asks for one of them twice."""

    latency: CallLatency = field(default_factory=CallLatency)
    """Caller-heard response times, from the SDK's own timestamps. See
    :mod:`stacy.metrics.live` for why these replace the stage marks."""

    last_caller_turn: str = ""
    """What the caller said in their previous turn, for a number read over two."""

    early_desk: tuple[str, str] | None = None
    """(the caller's words, the reply) a desk reply voiced before the turn was confirmed.

    The SDK starts replying on the caller's words before it is sure they have
    finished; a desk reply decided then has its audio built by the time they
    are. Checked against the confirmed turn in the hook, and thrown away if it
    no longer fits.
    """

    kept_note: tuple[str, str] | None = None
    """(the caller's words, the note) for a kept early desk reply.

    Keeping the early reply means leaving the caller's message untouched, so its
    note is added once the SDK has filed the message; see entrypoint._on_item.
    Without it the model never saw that the desk had booked, and could book again.
    """

    settled_turn: str = ""
    """The id of the caller's message the hook last ran for: a pass on it is not early any more."""

    desk_reply: tuple[str, str] | None = None
    """(the caller's message id, the reply) for a turn answered without the model.

    Set when the caller's turn is committed and spoken by the next generation;
    see :mod:`stacy.agent.fastlane`.
    """

    fixed_now: datetime | None = None
    """The clock the desk tools book against, pinned. None is the real time.

    Tests pin it; production never does. Before this, the tools read the wall
    clock while the tests assumed a Monday, so every booking test that said
    "tomorrow" failed on Fridays -- when tomorrow is a Saturday nothing can be
    booked on.
    """

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
    ) -> CallRuntime:
        """Assemble a runtime, rendering the prompt and resolving vendors once."""
        cfg = pipeline_for(tenant, overrides=overrides)
        builder = PromptBuilder()
        stamp = (now or datetime.now(ZoneInfo(tenant.timezone))).strftime(
            "%A, %B %d, %Y at %I:%M %p"
        )
        prompt = builder.fill_runtime(
            builder.render(tenant), now=stamp, caller_id=caller_id
        )
        stable, marker, per_call = prompt.partition("\n# This call")
        runtime = cls(
            tenant=tenant,
            call_id=call_id,
            caller_id=caller_id,
            sink=sink or NullSink(),
            calendar=calendar,
            messages=messages,
            pipeline=cfg,
            system_prompt=prompt,
            instructions=stable.rstrip() + "\n",
            call_section=(marker + per_call).strip(),
            backchannel=BackchannelFilter(max_words=cfg.turn.interrupt_min_words),
            filler=FillerPolicy(cfg=cfg.filler),
            latency=CallLatency(call_id=call_id),
            fixed_now=now,
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
        """Emit the open trace to the sink, if any, and close it.

        This no longer logs a headline number. The `turn ... response_ms=` line
        it used to print started 0.55s after the caller stopped and filed tool
        round trips under text-to-speech; the caller-heard figures are logged
        from :attr:`latency` instead.
        """
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
        """Log a caller utterance, run the safety scan, and unlock fillers.

        The scan happens here *and* at generation time. They are independent on
        purpose: preemptive generation can start the model on interim text
        before this event fires, so neither path may assume it ran first. The
        interceptor latches, which makes the double scan harmless.
        """
        if not text.strip():
            return
        # Logged, not just stored. The transcript went to a JSONL file inside an
        # ephemeral container, so when a caller says "it cannot hear me" there
        # was no way to tell a mis-transcription from a mis-understanding --
        # which are different bugs with different fixes.
        log.info("heard %s: %r", self.call_id, text)
        self.transcript.append({"role": "user", "text": text})
        self.filler.on_user_message()
        if self.safety.scan(text) and self.outcome != "safety_override":
            log.warning(
                "call %s: safety override latched on %r", self.call_id, self.safety.trigger_phrase
            )
            self.outcome = "safety_override"
            self.annotate(safety_override=True)

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
        same call: it is the thing the office needs to see first. A latched
        safety override outranks both, and should be unreachable anyway --
        the interceptor bypasses the model, so no tool can be called after it
        fires. The guard is here because "should be unreachable" is not a
        property worth trusting on this particular path.
        """
        self.appointments.append(appt)
        if self.outcome == "safety_override":
            log.error(
                "call %s: a booking was recorded after the safety override latched",
                self.call_id,
            )
            return
        if appt.kind == "emergency" or self.outcome != "dispatched":
            self.outcome = "dispatched" if appt.kind == "emergency" else "booked"

    def note_message(self, msg: Message) -> None:
        """Record a taken message, unless something louder already happened."""
        self.message = msg
        if self.outcome == "answered":
            self.outcome = "message"

    def now(self) -> datetime:
        """The time the desk tools book against, in the tenant's timezone."""
        return self.fixed_now or datetime.now(ZoneInfo(self.tenant.timezone))

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
            safety=(
                {
                    "override_fired": True,
                    "trigger_phrase": self.safety.trigger_phrase,
                    "turns_repeated": self.safety.triggers,
                }
                if self.safety.latched
                else None
            ),
        )

    def finish(self) -> None:
        """Close the last open turn. Call once, when the room closes."""
        self.flush_turn()
