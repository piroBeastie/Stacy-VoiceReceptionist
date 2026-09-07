"""Stage timings for a single conversational turn.

The number that matters is `response_ms`: the wall time between the caller
stopping speaking and the first byte of our audio leaving the process. Every
stage below is a slice of it, and each slice is owned by a different decision:

    endpoint_ms   turn-detector policy  (biggest lever -- see docs/latency.md)
    stt_ms        speech-to-text vendor
    llm_ttft_ms   model + host + prompt size (second biggest lever)
    tts_ttfb_ms   text-to-speech vendor
    egress_ms     our own framing and the codec path

Timings use a monotonic clock so they survive NTP steps mid-call.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Stage(str, Enum):
    """Ordered marks on the critical path of one turn.

    The order of declaration is the order of the pipeline, and
    :meth:`TurnTrace.spans` relies on it.
    """

    USER_STOPPED = "user_stopped"
    """VAD saw the caller's speech end. t=0 for the whole measurement."""

    ENDPOINT_COMMITTED = "endpoint_committed"
    """Turn detector decided the utterance was complete. Wait time ends here."""

    STT_FINAL = "stt_final"
    """Final transcript in hand. Skipped entirely when preemptive generation wins."""

    LLM_FIRST_TOKEN = "llm_first_token"
    """First token off the model. The only LLM number that affects felt latency."""

    TTS_FIRST_BYTE = "tts_first_byte"
    """First synthesized audio byte received from the TTS vendor."""

    AUDIO_OUT = "audio_out"
    """First byte handed to the transport. End of the measurement."""


_SPAN_NAMES: dict[tuple[Stage, Stage], str] = {
    (Stage.USER_STOPPED, Stage.ENDPOINT_COMMITTED): "endpoint_ms",
    (Stage.ENDPOINT_COMMITTED, Stage.STT_FINAL): "stt_ms",
    (Stage.STT_FINAL, Stage.LLM_FIRST_TOKEN): "llm_ttft_ms",
    (Stage.LLM_FIRST_TOKEN, Stage.TTS_FIRST_BYTE): "tts_ttfb_ms",
    (Stage.TTS_FIRST_BYTE, Stage.AUDIO_OUT): "egress_ms",
}


@dataclass
class TurnTrace:
    """Collects stage marks for one turn and renders them as a flat record.

    Marking is cheap and non-fatal: a stage that never fires is simply absent
    from the output rather than raising, because a half-measured turn is more
    useful than a crashed call.
    """

    call_id: str
    tenant_id: str
    turn_index: int
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    marks: dict[str, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def mark(self, stage: Stage, *, at: int | None = None) -> None:
        """Record `stage` as having happened now.

        Re-marking a stage is ignored rather than overwriting: the first
        occurrence is the true one, and barge-in can legitimately replay a
        stage within a single turn.
        """
        if stage.value in self.marks:
            return
        self.marks[stage.value] = at if at is not None else time.perf_counter_ns()

    def annotate(self, **kwargs: Any) -> None:
        """Attach non-timing context (model name, token counts, barge-in flags)."""
        self.meta.update(kwargs)

    def spans(self) -> dict[str, float]:
        """Durations in ms for every consecutive pair of stages we captured.

        Gaps are bridged rather than dropped. When preemptive generation skips
        STT_FINAL, the llm_ttft_ms span is measured from ENDPOINT_COMMITTED
        instead, so the row still sums to `response_ms`.
        """
        present = [s for s in Stage if s.value in self.marks]
        out: dict[str, float] = {}
        for earlier, later in zip(present, present[1:]):
            name = _SPAN_NAMES.get((earlier, later))
            if name is None:
                # A stage was skipped; name the span after where it landed.
                name = _SPAN_NAMES.get(
                    next((k for k in _SPAN_NAMES if k[1] == later), (later, later)),
                    f"{earlier.value}_to_{later.value}_ms",
                )
            delta = self.marks[later.value] - self.marks[earlier.value]
            out[name] = round(delta / 1_000_000, 2)
        return out

    @property
    def response_ms(self) -> float | None:
        """Caller stopped talking -> first audio out. The headline number."""
        start = self.marks.get(Stage.USER_STOPPED.value)
        end = self.marks.get(Stage.AUDIO_OUT.value)
        if start is None or end is None:
            return None
        return round((end - start) / 1_000_000, 2)

    def to_record(self) -> dict[str, Any]:
        """Flatten to one JSON-serializable row for the sink."""
        return {
            "turn_id": self.turn_id,
            "call_id": self.call_id,
            "tenant_id": self.tenant_id,
            "turn_index": self.turn_index,
            "started_at": self.started_at,
            "response_ms": self.response_ms,
            **self.spans(),
            **{f"meta_{k}": v for k, v in self.meta.items()},
        }
