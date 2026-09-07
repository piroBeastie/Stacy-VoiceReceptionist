"""Print the per-stage latency table that gates each build phase.

    python scripts/latency_report.py [trace_dir] [--tenant millbrook] [--since-calls 50]

Phase gates from the plan:
    Phase 2 baseline   -- record whatever it is, expect 1200-1800ms
    Phase 4 target     -- p50 <= 700ms, p95 <= 1100ms over >= 50 real calls
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dana.metrics import load_traces, summarize  # noqa: E402

PHASE4_P50_MS = 700
PHASE4_P95_MS = 1100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace_dir", nargs="?", default=os.getenv("DANA_TRACE_DIR", "data/traces"))
    ap.add_argument("--tenant", help="Filter to one tenant_id")
    ap.add_argument("--since-calls", type=int, help="Only the N most recent calls")
    ap.add_argument("--gate", action="store_true", help="Exit non-zero if Phase 4 targets miss")
    args = ap.parse_args()

    records = load_traces(args.trace_dir)
    if args.tenant:
        records = [r for r in records if r.get("tenant_id") == args.tenant]
    if args.since_calls:
        recent = list(dict.fromkeys(r.get("call_id") for r in reversed(records)))[
            : args.since_calls
        ]
        records = [r for r in records if r.get("call_id") in set(recent)]

    if not records:
        print(f"no traces in {args.trace_dir!r}", file=sys.stderr)
        return 1

    report = summarize(records)
    print(report.render())

    resp = report.stages.get("response_ms")
    if resp:
        print()
        p50_ok = resp["p50"] <= PHASE4_P50_MS
        p95_ok = resp["p95"] <= PHASE4_P95_MS
        print(f"  p50 {resp['p50']:.0f}ms vs {PHASE4_P50_MS}ms target  {'PASS' if p50_ok else 'MISS'}")
        print(f"  p95 {resp['p95']:.0f}ms vs {PHASE4_P95_MS}ms target  {'PASS' if p95_ok else 'MISS'}")
        if report.calls < 50:
            print(f"  NOTE: {report.calls} calls; the gate wants at least 50 real ones.")
        if args.gate and not (p50_ok and p95_ok):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
