"""Grade archived calls against the 8 criteria, and print the table.

    EVAL_API_KEY=... python scripts/eval_calls.py
    EVAL_API_KEY=... python scripts/eval_calls.py --save baselines/qwen.json
    python scripts/eval_calls.py --compare baselines/qwen.json --save baselines/haiku.json

This is the other half of a model decision. `scripts/latency_report.py` says how
fast a stack is; this says whether it still follows the rules. Swapping the LLM
without both is choosing on how one call felt.

The judge should be the strongest model available and does not need to be fast.
Grading a model's output with the same model is self-marking, so point
EVAL_MODEL somewhere other than LLM_MODEL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stacy.delivery import CallArchive  # noqa: E402
from stacy.evals import Judge, openai_completer, summarize  # noqa: E402
from stacy.evals.judge import CallGrade, CriterionResult  # noqa: E402
from stacy.evals.report import EvalReport, regressions  # noqa: E402
from stacy.tenants import TenantStore  # noqa: E402


def _load_baseline(path: Path) -> EvalReport:
    """Rebuild a report from saved grades, so two runs are comparable."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    grades = [
        CallGrade(
            call_id=g["call_id"],
            tenant_id=g["tenant_id"],
            results=tuple(CriterionResult(**r) for r in g["results"]),
            fields=g.get("fields", {}),
        )
        for g in raw["grades"]
    ]
    return summarize(grades)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calls", default=f"{os.getenv('STACY_DATA_DIR', 'data')}/calls")
    ap.add_argument("--tenant-dir", default="src/stacy/tenants/data")
    ap.add_argument("--tenant", help="Only grade calls for this tenant_id")
    ap.add_argument("--limit", type=int, help="Grade at most N calls")
    ap.add_argument("--save", help="Write the grades to this path as a baseline")
    ap.add_argument("--compare", help="Report regressions against a saved baseline")
    ap.add_argument(
        "--gate",
        action="store_true",
        help="Exit non-zero if any criterion regressed against --compare",
    )
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    records = list(CallArchive(args.calls).read())
    if args.tenant:
        records = [r for r in records if r.get("tenant_id") == args.tenant]
    if args.limit:
        records = records[-args.limit :]
    if not records:
        print(f"no archived calls in {args.calls!r}", file=sys.stderr)
        return 1

    # Never borrowed from LLM_API_KEY. The judge talks to EVAL_BASE_URL, a
    # different vendor by default, and falling back sent the agent's Anthropic
    # key to Groq as a bearer token.
    api_key = os.getenv("EVAL_API_KEY")
    if not api_key:
        print("EVAL_API_KEY not set (it is never borrowed from LLM_API_KEY)", file=sys.stderr)
        return 1
    model = os.getenv("EVAL_MODEL", "claude-haiku-4-5")
    base_url = os.getenv("EVAL_BASE_URL", "https://api.groq.com/openai/v1")
    if model == os.getenv("LLM_MODEL"):
        print(
            f"warning: judging {model} with itself -- set EVAL_MODEL to something else",
            file=sys.stderr,
        )

    store = TenantStore(args.tenant_dir)
    judge = Judge(
        complete=openai_completer(base_url=base_url, api_key=api_key, model=model), model=model
    )

    grades = []
    for i, record in enumerate(records, 1):
        tenant = store.get(record["tenant_id"])
        print(f"  grading {i}/{len(records)} {record.get('call_id', '')}", file=sys.stderr)
        grades.append(
            judge.grade(
                tenant=tenant,
                transcript=record.get("transcript", []),
                call_id=record.get("call_id", ""),
            )
        )

    report = summarize(grades)
    print()
    print(report.render())

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "model": model,
                    "agent_model": os.getenv("LLM_MODEL", ""),
                    "grades": [
                        {
                            "call_id": g.call_id,
                            "tenant_id": g.tenant_id,
                            "results": [r.__dict__ for r in g.results],
                            "fields": g.fields,
                        }
                        for g in grades
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nbaseline written to {out}")

    if args.compare:
        worse = regressions(_load_baseline(Path(args.compare)), report)
        print()
        if worse:
            print("REGRESSED against " + args.compare + ":")
            for line in worse:
                print(f"  {line}")
        else:
            print(f"no regressions against {args.compare}")
        if args.gate and worse:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
