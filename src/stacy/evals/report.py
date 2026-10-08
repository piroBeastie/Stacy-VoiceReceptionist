"""Aggregating grades into the table that decides a model swap.

Pass rate is computed over the calls a criterion *applies to*. Most criteria are
`unknown` on most calls -- the taste is never asked for, nobody mentions gas --
and folding those into the denominator would make every number drift toward
whatever mix of calls happened to be in the corpus, which is the opposite of
what a comparison needs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from stacy.evals.judge import CallGrade


@dataclass(frozen=True)
class CriterionSummary:
    """One criterion's record across a corpus."""

    id: str
    success: int = 0
    failure: int = 0
    unknown: int = 0

    @property
    def applicable(self) -> int:
        """Calls where the criterion actually had something to say."""
        return self.success + self.failure

    @property
    def pass_rate(self) -> float | None:
        """None when nothing applied -- distinct from a 0% pass rate."""
        return None if not self.applicable else self.success / self.applicable


@dataclass(frozen=True)
class EvalReport:
    """Every criterion's record, plus the calls that failed anything."""

    calls: int
    criteria: tuple[CriterionSummary, ...]
    failing_calls: tuple[str, ...]

    @property
    def clean_rate(self) -> float | None:
        """Share of calls with no failure on any criterion."""
        if not self.calls:
            return None
        return (self.calls - len(self.failing_calls)) / self.calls

    def render(self) -> str:
        """Fixed-width table, safe to paste next to a latency table."""
        head = f"{'criterion':<26}{'pass':>8}{'fail':>8}{'n/a':>8}{'rate':>9}"
        rows = [head, "-" * len(head)]
        for c in self.criteria:
            rate = "     —" if c.pass_rate is None else f"{c.pass_rate:>8.0%}"
            rows.append(f"{c.id:<26}{c.success:>8}{c.failure:>8}{c.unknown:>8}{rate:>9}")
        rows.append("")
        clean = "—" if self.clean_rate is None else f"{self.clean_rate:.0%}"
        rows.append(f"{self.calls} calls, {clean} with no failure on any criterion")
        if self.failing_calls:
            shown = ", ".join(self.failing_calls[:5])
            more = "" if len(self.failing_calls) <= 5 else f" (+{len(self.failing_calls) - 5} more)"
            rows.append(f"failed: {shown}{more}")
        return "\n".join(rows)


def summarize(grades: Iterable[CallGrade]) -> EvalReport:
    """Collapse per-call grades into a corpus-level report."""
    grades = list(grades)
    tallies: dict[str, dict[str, int]] = {}
    order: list[str] = []

    for grade in grades:
        for result in grade.results:
            if result.id not in tallies:
                tallies[result.id] = {"success": 0, "failure": 0, "unknown": 0}
                order.append(result.id)
            tallies[result.id][result.verdict] += 1

    return EvalReport(
        calls=len(grades),
        criteria=tuple(CriterionSummary(id=cid, **tallies[cid]) for cid in order),
        failing_calls=tuple(g.call_id for g in grades if not g.passed),
    )


def regressions(
    before: EvalReport, after: EvalReport, *, tolerance: float = 0.0
) -> Sequence[str]:
    """Criteria whose pass rate dropped between two runs.

    The point of the harness: a faster model that quietly stops following the
    two-sentence rule shows up here rather than three weeks later in a call
    somebody complains about.
    """
    prior = {c.id: c.pass_rate for c in before.criteria}
    worse = []
    for c in after.criteria:
        old, new = prior.get(c.id), c.pass_rate
        if old is None or new is None:
            continue
        if new < old - tolerance:
            worse.append(f"{c.id}: {old:.0%} -> {new:.0%}")
    return worse
