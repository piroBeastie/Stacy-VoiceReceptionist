"""Turn a pile of JSONL traces into the table that gates each build phase.

Reports p50 and p95 per stage. p95 is the one that matters -- a p50 of 600ms
with a p95 of 2.4s is a receptionist that feels broken one call in twenty, and
the caller remembers that call.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAGE_COLUMNS = (
    "endpoint_ms",
    "stt_ms",
    "llm_ttft_ms",
    "tts_ttfb_ms",
    "egress_ms",
    "response_ms",
)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. No numpy dependency for a six-line function."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered))))
    return ordered[rank - 1]


@dataclass
class Report:
    """Aggregate latency across a set of turns."""

    turns: int
    calls: int
    stages: dict[str, dict[str, float]]

    def render(self) -> str:
        """Fixed-width table, safe to paste into a phase-gate note."""
        head = f"{'stage':<16}{'p50':>10}{'p95':>10}{'max':>10}{'n':>8}"
        rows = [head, "-" * len(head)]
        for name in STAGE_COLUMNS:
            s = self.stages.get(name)
            if not s:
                continue
            rows.append(
                f"{name:<16}{s['p50']:>10.0f}{s['p95']:>10.0f}{s['max']:>10.0f}{int(s['n']):>8}"
            )
        rows.append("")
        rows.append(f"{self.turns} turns across {self.calls} calls")
        return "\n".join(rows)


def load_traces(directory: str | Path) -> list[dict[str, Any]]:
    """Read every turns-*.jsonl in `directory`, skipping unparseable lines."""
    out: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("turns-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def summarize(records: Iterable[dict[str, Any]]) -> Report:
    """Collapse trace records into a :class:`Report`."""
    records = list(records)
    stages: dict[str, dict[str, float]] = {}
    for name in STAGE_COLUMNS:
        values = [r[name] for r in records if isinstance(r.get(name), (int, float))]
        if not values:
            continue
        stages[name] = {
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "max": max(values),
            "n": len(values),
        }
    return Report(
        turns=len(records),
        calls=len({r.get("call_id") for r in records}),
        stages=stages,
    )
