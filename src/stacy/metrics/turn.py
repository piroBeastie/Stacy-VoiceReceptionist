"""Stage timings for a single conversational turn.

The number that matters is `response_ms`: the wall time between the caller
stopping speaking and the first byte of our audio leaving the process. Every
stage below is a slice of it, and each slice is owned by a different decision:

    endpoint_ms   turn-detector policy  (biggest lever -- see README)
    stt_ms        speech-to-text vendor
    llm_ttft_ms   model + host + prompt size (second biggest lever)
    tts_ttfb_ms   text-to-speech vendor
    egress_ms     our own framing and the codec path

Timings use a monotonic clock so they survive NTP steps mid-call.

A span is named after the stage it *ends at*, and spans are computed in
timestamp order rather than in the order the stages are declared. Both of those
are load-bearing. The pipeline is nominal, not guaranteed: a final transcript
can land after the turn detector has already committed, preemptive generation
can skip STT_FINAL entirely, and a barge-in can reorder the tail. Deriving spans
from declaration order produced negative `stt_ms` on every single turn, and
mislabelled most of the response as `egress_ms`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Any


class Stage(StrEnum):
    """Marks on the critical path of one turn.

    Declaration order is the *nominal* pipeline and is used only for stable
    column ordering in reports. Real marks are ordered by their timestamps.
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
    """First synthesized audio frame out of the TTS node."""

    AUDIO_OUT = "audio_out"
    """First frame handed to the transport. End of the measurement."""


SPAN_NAME: dict[Stage, str] = {
    Stage.ENDPOINT_COMMITTED: "endpoint_ms",
    Stage.STT_FINAL: "stt_ms",
    Stage.LLM_FIRST_TOKEN: "llm_ttft_ms",
    Stage.TTS_FIRST_BYTE: "tts_ttfb_ms",
    Stage.AUDIO_OUT: "egress_ms",
}
"""Each span is 'time since the previous mark', named for the stage it ends at.

USER_STOPPED has no entry because nothing precedes it: it is t=0.
"""


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

    def has(self, stage: Stage) -> bool:
        """Whether `stage` has been marked on this turn."""
        return stage.value in self.marks

    def annotate(self, **kwargs: Any) -> None:
        """Attach non-timing context (model name, token counts, barge-in flags)."""
        self.meta.update(kwargs)

    def _ordered(self) -> list[tuple[Stage, int]]:
        """Present marks sorted by timestamp, then by nominal pipeline position.

        The tiebreak keeps output stable when two stages land in the same
        nanosecond, which happens on a cached greeting or a mocked clock.
        """
        nominal = {s: i for i, s in enumerate(Stage)}
        present = [(s, self.marks[s.value]) for s in Stage if s.value in self.marks]
        return sorted(present, key=lambda pair: (pair[1], nominal[pair[0]]))

    def spans(self) -> dict[str, float]:
        """Durations in ms between consecutive marks, in timestamp order.

        Named for the stage each span ends at, so a skipped stage widens its
        neighbour rather than dropping time on the floor. When every mark
        between USER_STOPPED and AUDIO_OUT is present the values sum exactly to
        `response_ms`, which is what makes the report's columns trustworthy.
        """
        ordered = self._ordered()
        out: dict[str, float] = {}
        for (_, earlier), (stage, later) in pairwise(ordered):
            name = SPAN_NAME.get(stage)
            if name is None:
                continue
            out[name] = round((later - earlier) / 1_000_000, 2)
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
