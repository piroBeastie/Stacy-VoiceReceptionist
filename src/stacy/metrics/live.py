"""Caller-heard latency, from the numbers the SDK already measures.

The stage marks in :mod:`stacy.metrics.turn` misreport every live call. They
start the clock when voice detection reports silence, which is already 0.55s
after the caller stopped. They stamp the voice's first byte and the audio going
out on the same frame. And on a turn that runs a tool, the "first token" is the
tool call, so the tool and the second model pass get counted as text-to-speech.
A 1.9s "TTS" spike on a live call was a `record_details` round trip.

LiveKit already times each of these on the conversation items themselves, from
the moment speech actually ended: the caller's message carries when they stopped
and how long the turn took to commit, and the reply carries its model and voice
timings and when its audio started. This module turns those into one record per
reply and one summary per call, and both are logged as single lines, because a
worker on LiveKit Cloud keeps no disk between restarts.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from stacy.metrics.report import percentile

LOG_MARK = "latency "
"""Prefix of the per-reply log line. :func:`parse_log` finds records by it."""

_REPORT_ROWS = (
    ("wait_ms", "caller wait"),
    ("eot_ms", "end of turn"),
    ("stt_ms", "transcript"),
    ("llm_ttft_ms", "model 1st token"),
    ("tts_ttfb_ms", "voice 1st byte"),
    ("tools_ms", "tools"),
)


def _ms(value: Any) -> float | None:
    """Seconds from the SDK to rounded milliseconds; anything else is absent."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value) * 1000, 1)
    return None


def _numbers(records: Iterable[Mapping[str, Any]], name: str) -> list[float]:
    return [r[name] for r in records if isinstance(r.get(name), (int, float))]


@dataclass
class CallLatency:
    """One call's replies, timed against the caller turn in front of each.

    Fed from session events and from the tool wrapper; pure otherwise, so the
    rules about what counts as a response time are tested without the SDK.
    """

    call_id: str = ""
    replies: list[dict[str, Any]] = field(default_factory=list)
    _user: dict[str, Any] | None = None
    _tools: list[tuple[str, float]] = field(default_factory=list)
    _desk: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def on_user(self, metrics: Mapping[str, Any] | None) -> None:
        """The caller's turn was committed. The next reply is timed against it."""
        self._user = dict(metrics or {})
        # Not reset here: the SDK adds the caller's message after the reply has
        # started, so a desk reply is flagged before this runs. on_reply clears it.
        with self._lock:
            self._tools.clear()

    def on_desk_reply(self) -> None:
        """The reply to this turn was written by the desk, not the model."""
        self._desk = True

    def on_tool(self, name: str, elapsed_ms: float) -> None:
        """A tool finished. Called from a worker thread, hence the lock."""
        with self._lock:
            self._tools.append((name, round(float(elapsed_ms), 1)))

    def on_reply(
        self,
        metrics: Mapping[str, Any] | None,
        *,
        interrupted: bool = False,
        chars: int = 0,
    ) -> dict[str, Any] | None:
        """Time the first thing the caller hears after their turn.

        Anything else the agent says has no caller turn in front of it and is
        not a response time: the greeting, a "Still there?" into a quiet line.
        """
        user, self._user = self._user, None
        if user is None:
            return None
        m = metrics or {}
        started, stopped = m.get("started_speaking_at"), user.get("stopped_speaking_at")
        wait = (
            started - stopped
            if isinstance(started, (int, float)) and isinstance(stopped, (int, float))
            else None
        )
        with self._lock:
            tools, self._tools = list(self._tools), []
        desk, self._desk = self._desk, False
        record: dict[str, Any] = {
            "turn": len(self.replies),
            "at": round(started, 3) if isinstance(started, (int, float)) else None,
            "wait_ms": _ms(wait),
            "eot_ms": _ms(user.get("end_of_turn_delay")),
            "stt_ms": _ms(user.get("transcription_delay")),
            "llm_ttft_ms": _ms(m.get("llm_node_ttft")),
            "tts_ttfb_ms": _ms(m.get("tts_node_ttfb")),
            "playback_ms": _ms(m.get("playback_latency")),
            "tools": [f"{name}:{ms:.0f}ms" for name, ms in tools],
            "tools_ms": round(sum(ms for _, ms in tools), 1) if tools else None,
            "interrupted": bool(interrupted),
            "chars": int(chars),
            "desk": desk,
        }
        self.replies.append(record)
        return record

    def summary(self) -> dict[str, float]:
        """The call in a handful of numbers, for the log and the post-call payload."""
        out: dict[str, float] = {"replies": float(len(self.replies))}
        for name in ("wait_ms", "eot_ms", "llm_ttft_ms", "tts_ttfb_ms"):
            values = _numbers(self.replies, name)
            if values:
                stem = name[: -len("_ms")]
                out[f"{stem}_p50_ms"] = percentile(values, 50)
                out[f"{stem}_max_ms"] = max(values)
        waits = _numbers(self.replies, "wait_ms")
        if waits:
            out["wait_p90_ms"] = percentile(waits, 90)
        out["tool_replies"] = float(sum(1 for r in self.replies if r["tools"]))
        desk = sum(1 for r in self.replies if r.get("desk"))
        if desk:
            out["desk_replies"] = float(desk)
        return out


def format_reply(call_id: str, record: Mapping[str, Any]) -> str:
    """One reply as a compact JSON object, the body of the log line."""
    return json.dumps({"call": call_id, **record}, separators=(",", ":"), ensure_ascii=False)


def parse_log(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Reply records out of captured worker logs.

    The worker prints each record twice, plain and wrapped in a JSON log line;
    only the plain one parses. A reply that shows up in two overlapping captures
    is counted once.
    """
    seen: set[tuple[Any, Any]] = set()
    out: list[dict[str, Any]] = []
    for line in lines:
        at = line.find(LOG_MARK + "{")
        if at < 0:
            continue
        try:
            record = json.loads(line[at + len(LOG_MARK) :].strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        key = (record.get("call"), record.get("turn"))
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def render_report(records: list[dict[str, Any]]) -> str:
    """p50 / p90 / max per stage, and the caller wait split by whether a tool ran."""
    calls = len({r.get("call") for r in records})
    head = f"{'':<18}{'p50':>8}{'p90':>8}{'max':>8}{'n':>6}"
    rows = [f"{len(records)} replies across {calls} calls", "", head, "-" * len(head)]
    for name, label in _REPORT_ROWS:
        values = _numbers(records, name)
        if values:
            rows.append(
                f"{label:<18}{percentile(values, 50):>8.0f}{percentile(values, 90):>8.0f}"
                f"{max(values):>8.0f}{len(values):>6}"
            )
    plain = _numbers([r for r in records if not r.get("tools")], "wait_ms")
    tooled = _numbers([r for r in records if r.get("tools")], "wait_ms")
    if plain and tooled:
        rows += [
            "",
            (
                f"caller wait p50: {percentile(plain, 50):.0f}ms with no tool ({len(plain)}), "
                f"{percentile(tooled, 50):.0f}ms after a tool ({len(tooled)})"
            ),
        ]
    desk = _numbers([r for r in records if r.get("desk")], "wait_ms")
    model = _numbers([r for r in records if not r.get("desk")], "wait_ms")
    if desk and model:
        rows.append(
            f"caller wait p50: {percentile(desk, 50):.0f}ms answered by the desk ({len(desk)}), "
            f"{percentile(model, 50):.0f}ms by the model ({len(model)})"
        )
    return "\n".join(rows)
