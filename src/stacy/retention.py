"""How long the plaintext on disk is kept, and the sweep that enforces it.

Every finished call leaves a transcript in `data/calls`, a timing trace in
`data/traces`, and sometimes a message in `data/messages`. All three are
plaintext, which is deliberate -- they are greppable after a bad call, they
survive a crashed worker, and they need no service running to read. What was not
deliberate was keeping them forever: a year of HVAC calls is a year of names,
numbers and addresses sitting in a container volume, and nobody chose that.

So the policy is explicit, it has defaults a person can defend, and a sweep
enforces it:

* **calls, 30 days.** The transcripts. The eval corpus wants a pile of real
  calls; it does not want last winter's.
* **traces, 14 days.** Timings, no caller detail beyond the call id. They are
  read while tuning and never again.
* **messages, 90 days.** A message is a promise until somebody acts on it, and
  "we lost it" is worse than keeping it a quarter.
* **appointments, kept.** The business's own book. Deleting it to satisfy a
  retention policy would be deleting the product. Set a window explicitly if a
  client's own policy requires one.

Each window is overridable per deployment:

    STACY_RETAIN_CALLS_DAYS=7 STACY_RETAIN_TRACES_DAYS=3 python scripts/retention_sweep.py --apply

Nothing here runs on its own. Deleting a customer's records is not something to
do as a side effect of answering a phone call, so the sweep is a script an
operator schedules, and it prints what it would delete unless told to apply.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger("stacy.retention")

KEEP_FOREVER = 0
"""A window of zero days means keep, not delete immediately."""

ENV_PREFIX = "STACY_RETAIN_"


@dataclass(frozen=True)
class RetentionPolicy:
    """How many days of each kind to keep. Zero means keep."""

    calls_days: int = 30
    traces_days: int = 14
    messages_days: int = 90
    appointments_days: int = KEEP_FOREVER

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> tuple[RetentionPolicy, list[str]]:
        """The policy, and the names of the variables that set it.

        The second value is what tells `preflight` whether an operator chose
        these windows or inherited them, which is a different thing to report.
        """
        source = env if env is not None else dict(os.environ)
        chosen: list[str] = []
        values: dict[str, int] = {}
        for name in ("calls", "traces", "messages", "appointments"):
            key = f"{ENV_PREFIX}{name.upper()}_DAYS"
            raw = (source.get(key) or "").strip()
            if not raw:
                continue
            try:
                days = int(raw)
            except ValueError:
                log.warning("%s is not a number (%r); using the default", key, raw)
                continue
            if days < 0:
                log.warning("%s is negative (%r); using the default", key, raw)
                continue
            values[f"{name}_days"] = days
            chosen.append(key)
        return cls(**values), chosen

    def days_for(self, kind: str) -> int:
        return int(getattr(self, f"{kind}_days"))


@dataclass
class Sweep:
    """What a sweep found, and what it did or would do."""

    applied: bool
    files_removed: list[str] = field(default_factory=list)
    lines_removed: dict[str, int] = field(default_factory=dict)
    kept_forever: list[str] = field(default_factory=list)

    @property
    def nothing_to_do(self) -> bool:
        return not self.files_removed and not any(self.lines_removed.values())

    def spoken(self) -> str:
        """One line a human reads in a cron log."""
        verb = "removed" if self.applied else "would remove"
        lines = sum(self.lines_removed.values())
        parts = [f"{verb} {len(self.files_removed)} file(s)"]
        if lines:
            parts.append(f"{lines} record(s) from {len(self.lines_removed)} file(s)")
        if self.kept_forever:
            parts.append(f"kept {', '.join(self.kept_forever)} (no window set)")
        return "; ".join(parts)


_DAILY = {"calls": "calls-*.jsonl", "traces": "turns-*.jsonl"}
"""Kinds written one file per day, where the filename carries the date."""

_PER_TENANT = {"messages": "messages-*.jsonl", "appointments": "appointments-*.jsonl"}
"""Kinds written one file per tenant, where each line carries its own date."""

_DIRS = {"calls": "calls", "traces": "traces", "messages": "messages", "appointments": "calendar"}


def sweep(
    data_dir: str | Path,
    policy: RetentionPolicy | None = None,
    *,
    now: datetime | None = None,
    apply: bool = False,
) -> Sweep:
    """Apply the policy under `data_dir`. Without `apply`, only reports."""
    policy = policy or RetentionPolicy()
    today = (now or datetime.now().astimezone()).date()
    root = Path(data_dir)
    result = Sweep(applied=apply)

    for kind, pattern in _DAILY.items():
        days = policy.days_for(kind)
        if days == KEEP_FOREVER:
            result.kept_forever.append(kind)
            continue
        cutoff = today - timedelta(days=days)
        for path in sorted((root / _DIRS[kind]).glob(pattern)):
            stamp = _date_in(path.stem)
            if stamp is None or stamp >= cutoff:
                continue
            result.files_removed.append(str(path))
            if apply:
                path.unlink(missing_ok=True)

    for kind, pattern in _PER_TENANT.items():
        days = policy.days_for(kind)
        if days == KEEP_FOREVER:
            result.kept_forever.append(kind)
            continue
        cutoff = today - timedelta(days=days)
        for path in sorted((root / _DIRS[kind]).glob(pattern)):
            dropped = _prune_lines(path, cutoff, apply=apply)
            if dropped:
                result.lines_removed[str(path)] = dropped

    log.info("retention sweep: %s", result.spoken())
    return result


def _date_in(stem: str) -> date | None:
    """The date in `calls-2026-09-09`, or None if the name does not carry one."""
    tail = stem.rsplit("-", 3)[-3:]
    if len(tail) != 3:
        return None
    try:
        return date(int(tail[0]), int(tail[1]), int(tail[2]))
    except ValueError:
        return None


def _prune_lines(path: Path, cutoff: date, *, apply: bool) -> int:
    """Drop records created before `cutoff`. Returns how many went.

    A record with no readable date is kept: the point of the sweep is to honour
    a policy, and guessing at an unparseable line would delete a customer's
    appointment because a field was renamed.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0

    keep: list[str] = []
    dropped = 0
    for line in lines:
        if not line.strip():
            continue
        stamp = _created_date(line)
        if stamp is not None and stamp < cutoff:
            dropped += 1
            continue
        keep.append(line)

    if dropped and apply:
        # Written beside the original and moved over it, so a sweep interrupted
        # halfway leaves the records intact rather than truncated.
        spare = path.with_suffix(".jsonl.sweeping")
        spare.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        spare.replace(path)
    return dropped


def _created_date(line: str) -> date | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    raw = record.get("created_at") or record.get("at") or ""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return None
