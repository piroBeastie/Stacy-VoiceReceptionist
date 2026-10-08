"""Keeping a local copy of every finished call.

The webhook is the delivery path; this is the record. They are separate because
they fail separately: a consumer being down, misconfigured, or simply not set up
yet should not also mean the call never happened as far as this repo is
concerned.

It also supplies the eval corpus. Grading model choice needs a pile of real
calls sitting somewhere greppable, and the post-call payload already contains
exactly what a judge needs -- the transcript, the outcome, and what was booked.

Retention is the obvious open question here and is deliberately not answered:
these are plaintext transcripts on disk. See the README before real traffic.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger("stacy.delivery.archive")


class CallArchive:
    """Append-only store of post-call payloads, one file per day."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, when: float | None = None) -> Path:
        day = time.strftime("%Y-%m-%d", time.gmtime(when or time.time()))
        return self._dir / f"calls-{day}.jsonl"

    def write(self, payload: dict[str, Any]) -> None:
        """Append one call. Never raises -- losing the archive is not worth a
        crash on the way out of a call that already happened."""
        try:
            line = json.dumps(payload, separators=(",", ":"), default=str)
            with self._lock, self._path().open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
        except Exception:  # noqa: BLE001
            log.exception("failed to archive call %s", payload.get("call_id"))

    def read(self) -> Iterator[dict[str, Any]]:
        """Every archived call, oldest file first, skipping unparseable lines."""
        for path in sorted(self._dir.glob("calls-*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
