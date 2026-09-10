"""Where finished turn traces go.

JSONL on local disk is deliberate for now: it is greppable, it survives a
crashed process, and `scripts/latency_report.py` reads it with no service
running. Swap in a real store once Phase 4 gates are passing.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Protocol

from stacy.metrics.turn import TurnTrace

log = logging.getLogger(__name__)


class TraceSink(Protocol):
    """Anything that can accept a completed turn trace."""

    def emit(self, trace: TurnTrace) -> None: ...


class NullSink:
    """Discards traces. For tests and for anyone benchmarking the harness itself."""

    def emit(self, trace: TurnTrace) -> None:  # noqa: D102
        return None


class JsonlSink:
    """Appends one JSON object per turn, one file per day.

    Writes are mutex-guarded and flushed per line. Instrumentation must never
    be the thing that breaks a live call, so an IO failure here is logged and
    swallowed rather than raised.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, trace: TurnTrace) -> Path:
        day = __import__("time").strftime("%Y-%m-%d", __import__("time").gmtime(trace.started_at))
        return self._dir / f"turns-{day}.jsonl"

    def emit(self, trace: TurnTrace) -> None:
        """Append `trace` as a single line. Never raises."""
        try:
            line = json.dumps(trace.to_record(), separators=(",", ":"))
            with self._lock, self._path(trace).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
        except Exception:  # noqa: BLE001 - instrumentation must not kill a call
            log.exception("failed to emit turn trace %s", trace.turn_id)
