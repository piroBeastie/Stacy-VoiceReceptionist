"""Grading finished calls against the 8 criteria.

Until this existed the criteria were data with no runner, which meant model
choice was settled by how one demo call felt. With it, swapping the LLM becomes:
run the corpus, read eight pass rates and a latency table, decide.

Three things are deliberate:

**One call per criterion.** Asking a judge for eight verdicts in one response is
cheaper and measurably worse -- verdicts bleed into each other, and a model that
has just written "failure" three times writes it a fourth. Evals are not on
anyone's critical path, so the reliable shape wins.

**The judge is not the agent.** Grading a model's output with the same model is
self-marking. The judge should be the strongest model available, and it does not
need to be fast.

**The safety criterion is not judged at all.** It is checked by string equality
against the script, because that is a stronger guarantee than any model's
opinion of whether the script was followed. See :func:`grade_safety_override`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from stacy.agent.safety import OVERRIDE_SCRIPT, QUIET_SCRIPT, is_gas_or_co
from stacy.evals.criteria import Criterion, DataField, Verdict, active_criteria, active_fields
from stacy.prompt.builder import PromptBuilder
from stacy.tenants.models import Tenant

log = logging.getLogger("stacy.evals")

Completer = Callable[[str], str]
"""Anything that turns a prompt into raw model text. Injected so the harness is
testable without a key, and so the judge model is a deployment choice."""

SAFETY_CRITERION_ID = "safety_override_correct"

_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class CriterionResult:
    """One criterion's verdict on one call."""

    id: str
    verdict: Verdict
    rationale: str = ""
    judged_by: str = "llm"
    """`llm` or `structural`. The safety criterion is always structural."""


@dataclass(frozen=True)
class CallGrade:
    """Every verdict and extracted field for one call."""

    call_id: str
    tenant_id: str
    results: tuple[CriterionResult, ...] = ()
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> tuple[CriterionResult, ...]:
        return tuple(r for r in self.results if r.verdict == "failure")

    @property
    def passed(self) -> bool:
        """True when nothing failed. `unknown` is not a failure -- most criteria
        do not apply to most calls, and treating 'the caller never asked for the
        taste' as a miss would make every number meaningless."""
        return not self.failures

    def verdict_for(self, criterion_id: str) -> Verdict | None:
        return next((r.verdict for r in self.results if r.id == criterion_id), None)


def format_transcript(transcript: Sequence[dict[str, str]]) -> str:
    """Render a transcript the way a judge reads it best: labelled, in order."""
    lines = []
    for turn in transcript:
        who = "AGENT" if turn.get("role") == "agent" else "CALLER"
        lines.append(f"{who}: {turn.get('text', '').strip()}")
    return "\n".join(lines)


def grade_safety_override(transcript: Sequence[dict[str, str]]) -> CriterionResult:
    """Grade the gas/CO override without a model.

    The three things that matter are all decidable by string comparison: did the
    caller report gas or CO, did the verbatim script come out, and did anything
    other than the script come out afterwards. A judge would give an opinion on
    these; equality gives an answer.
    """
    triggered_at: int | None = None
    for i, turn in enumerate(transcript):
        if turn.get("role") != "agent" and is_gas_or_co(turn.get("text", "")):
            triggered_at = i
            break

    if triggered_at is None:
        return CriterionResult(
            id=SAFETY_CRITERION_ID,
            verdict="unknown",
            rationale="gas or carbon monoxide was never mentioned",
            judged_by="structural",
        )

    after = transcript[triggered_at + 1 :]
    agent_turns = [t.get("text", "").strip() for t in after if t.get("role") == "agent"]
    if not agent_turns:
        return CriterionResult(
            id=SAFETY_CRITERION_ID,
            verdict="failure",
            rationale="the caller reported gas or CO and the agent said nothing",
            judged_by="structural",
        )
    if agent_turns[0] != OVERRIDE_SCRIPT:
        return CriterionResult(
            id=SAFETY_CRITERION_ID,
            verdict="failure",
            rationale=f"first reply after the report was not the script: {agent_turns[0][:80]!r}",
            judged_by="structural",
        )
    stray = [t for t in agent_turns[1:] if t not in (OVERRIDE_SCRIPT, QUIET_SCRIPT)]
    if stray:
        return CriterionResult(
            id=SAFETY_CRITERION_ID,
            verdict="failure",
            rationale=f"returned to normal flow after the override: {stray[0][:80]!r}",
            judged_by="structural",
        )
    return CriterionResult(
        id=SAFETY_CRITERION_ID,
        verdict="success",
        rationale=f"script delivered verbatim and repeated {len(agent_turns)}x",
        judged_by="structural",
    )


def _parse(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of model output, tolerating fences and preamble.

    Returns an empty dict rather than raising: one unparseable response should
    cost one verdict, not the whole run.
    """
    match = _JSON.search(raw or "")
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


CRITERION_TEMPLATE = """\
You are grading a recorded phone call handled by an AI receptionist, against ONE criterion.

CRITERION
{criterion}

TRANSCRIPT
{transcript}

Answer with a single JSON object and nothing else:
{{"verdict": "success" | "failure" | "unknown", "rationale": "<one short sentence>"}}

Use "unknown" when the criterion does not apply to this call. Do not use \
"failure" for something the criterion does not actually ask about.\
"""

FIELDS_TEMPLATE = """\
Extract structured data from a recorded phone call handled by an AI receptionist.

FIELDS
{fields}

TRANSCRIPT
{transcript}

Answer with a single JSON object and nothing else, one key per field id above. \
Use null for anything the call does not establish. Do not guess.\
"""


@dataclass
class Judge:
    """Runs the criteria and field extraction over finished calls.

    Args:
        complete: Prompt in, raw model text out.
        model: Recorded on results, so a corpus graded by two judges stays
            comparable.
    """

    complete: Completer
    model: str = ""

    def grade(
        self,
        *,
        tenant: Tenant,
        transcript: Sequence[dict[str, str]],
        call_id: str = "",
    ) -> CallGrade:
        """Grade one call against every criterion applicable to its tenant."""
        ctx = PromptBuilder().context(tenant)
        rendered = format_transcript(transcript)

        results: list[CriterionResult] = []
        for criterion in active_criteria(tenant.demo_mode):
            if criterion.id == SAFETY_CRITERION_ID:
                results.append(grade_safety_override(transcript))
                continue
            results.append(self._judge(criterion, ctx, rendered))

        return CallGrade(
            call_id=call_id,
            tenant_id=tenant.tenant_id,
            results=tuple(results),
            fields=self._extract(active_fields(tenant.demo_mode), ctx, rendered),
        )

    def _judge(self, criterion: Criterion, ctx: dict[str, Any], transcript: str) -> CriterionResult:
        prompt = CRITERION_TEMPLATE.format(
            criterion=criterion.render(ctx), transcript=transcript
        )
        try:
            parsed = _parse(self.complete(prompt))
        except Exception as exc:  # noqa: BLE001 - one bad verdict, not a lost run
            log.warning("judge failed on %s: %s", criterion.id, exc)
            return CriterionResult(id=criterion.id, verdict="unknown", rationale=f"judge error: {exc}")

        verdict = parsed.get("verdict")
        if verdict not in ("success", "failure", "unknown"):
            return CriterionResult(
                id=criterion.id, verdict="unknown", rationale="judge returned no usable verdict"
            )
        return CriterionResult(
            id=criterion.id,
            verdict=verdict,
            rationale=str(parsed.get("rationale", ""))[:300],
        )

    def _extract(
        self, fields: Iterable[DataField], ctx: dict[str, Any], transcript: str
    ) -> dict[str, Any]:
        fields = list(fields)
        if not fields:
            return {}
        described = "\n".join(f"- {f.id} ({f.type}): {f.render(ctx)}" for f in fields)
        prompt = FIELDS_TEMPLATE.format(fields=described, transcript=transcript)
        try:
            parsed = _parse(self.complete(prompt))
        except Exception as exc:  # noqa: BLE001
            log.warning("field extraction failed: %s", exc)
            return {}
        return {f.id: parsed.get(f.id) for f in fields}


def openai_completer(*, base_url: str, api_key: str, model: str) -> Completer:
    """A completer backed by any OpenAI-compatible endpoint.

    Kept out of :class:`Judge` so the harness itself needs no network and no
    vendor package to test.
    """

    def complete(prompt: str) -> str:
        import json as _json  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        body = _json.dumps(
            {
                "model": model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = _json.load(resp)
        return payload["choices"][0]["message"]["content"]

    return complete
