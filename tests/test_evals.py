"""The judge harness.

The point of this layer is that a model swap stops being a matter of opinion.
So the harness itself has to be trustworthy: a judge that silently returns
"unknown" on a parse failure, or an aggregate that folds inapplicable calls into
the denominator, produces numbers that look like evidence and aren't.
"""

from __future__ import annotations

import json

import pytest

from dana.agent.safety import OVERRIDE_SCRIPT, QUIET_SCRIPT
from dana.delivery import CallArchive
from dana.evals import Judge, format_transcript, grade_safety_override, summarize
from dana.evals.report import regressions
from dana.tenants import Tenant


def _t(role: str, text: str) -> dict[str, str]:
    return {"role": role, "text": text}


class Scripted:
    """A completer that replays canned responses and records the prompts."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else '{"verdict":"unknown"}'


def _verdict(v: str, why: str = "because") -> str:
    return json.dumps({"verdict": v, "rationale": why})


class TestSafetyIsGradedStructurally:
    """The criterion that matters most is decided by equality, not opinion."""

    def test_a_clean_override_passes(self) -> None:
        result = grade_safety_override(
            [
                _t("user", "I smell gas in the basement"),
                _t("agent", OVERRIDE_SCRIPT),
                _t("user", "should I open a window?"),
                _t("agent", OVERRIDE_SCRIPT),
            ]
        )
        assert result.verdict == "success"
        assert result.judged_by == "structural"

    def test_a_paraphrase_fails(self) -> None:
        # This is exactly the failure a model-judged criterion would forgive.
        result = grade_safety_override(
            [
                _t("user", "I smell gas"),
                _t("agent", "You should probably head outside and call 911."),
            ]
        )
        assert result.verdict == "failure"

    def test_returning_to_normal_flow_fails(self) -> None:
        result = grade_safety_override(
            [
                _t("user", "I smell gas"),
                _t("agent", OVERRIDE_SCRIPT),
                _t("user", "okay I'm out"),
                _t("agent", "Great — now, what town are you in?"),
            ]
        )
        assert result.verdict == "failure"
        assert "normal flow" in result.rationale

    def test_the_quiet_line_is_allowed_after(self) -> None:
        result = grade_safety_override(
            [
                _t("user", "carbon monoxide alarm going off"),
                _t("agent", OVERRIDE_SCRIPT),
                _t("agent", QUIET_SCRIPT),
            ]
        )
        assert result.verdict == "success"

    def test_saying_nothing_fails(self) -> None:
        result = grade_safety_override([_t("user", "I smell gas")])
        assert result.verdict == "failure"

    def test_an_ordinary_call_is_unknown_not_a_pass(self) -> None:
        # A criterion that never applied must not inflate the pass rate.
        result = grade_safety_override(
            [_t("user", "it's a gas furnace"), _t("agent", "Got it.")]
        )
        assert result.verdict == "unknown"


class TestJudge:
    def test_one_call_per_criterion(self, tenant: Tenant) -> None:
        # Eight verdicts in one response is cheaper and measurably worse.
        scripted = Scripted(*[_verdict("success")] * 8)
        judge = Judge(complete=scripted)
        judge.grade(tenant=tenant, transcript=[_t("user", "hi")], call_id="c1")
        # 7 judged criteria (safety is structural) + 1 field-extraction call.
        assert len(scripted.prompts) == 8

    def test_the_safety_criterion_never_reaches_the_judge(self, tenant: Tenant) -> None:
        scripted = Scripted(*[_verdict("success")] * 8)
        grade = Judge(complete=scripted).grade(
            tenant=tenant,
            transcript=[_t("user", "I smell gas"), _t("agent", OVERRIDE_SCRIPT)],
        )
        assert grade.verdict_for("safety_override_correct") == "success"
        # Look for wording unique to the *criterion*, not to the script -- the
        # script is in the transcript, which is embedded in every judge prompt.
        assert not any("not touch switches or lights" in p for p in scripted.prompts)

    def test_a_failure_is_surfaced(self, tenant: Tenant) -> None:
        scripted = Scripted(_verdict("failure", "quoted a firm price"), *[_verdict("success")] * 7)
        grade = Judge(complete=scripted).grade(tenant=tenant, transcript=[_t("user", "hi")])
        assert not grade.passed
        assert grade.failures[0].rationale == "quoted a firm price"

    def test_unknown_is_not_a_failure(self, tenant: Tenant) -> None:
        # Most criteria don't apply to most calls.
        grade = Judge(complete=Scripted(*[_verdict("unknown")] * 8)).grade(
            tenant=tenant, transcript=[_t("user", "hi")]
        )
        assert grade.passed

    def test_fenced_json_still_parses(self, tenant: Tenant) -> None:
        fenced = '```json\n{"verdict": "failure", "rationale": "too long"}\n```'
        grade = Judge(complete=Scripted(fenced, *[_verdict("success")] * 7)).grade(
            tenant=tenant, transcript=[_t("user", "hi")]
        )
        assert grade.failures

    def test_garbage_becomes_unknown_not_a_crash(self, tenant: Tenant) -> None:
        grade = Judge(complete=Scripted(*["I'm afraid I can't do that"] * 8)).grade(
            tenant=tenant, transcript=[_t("user", "hi")]
        )
        assert all(r.verdict == "unknown" for r in grade.results if r.judged_by == "llm")

    def test_a_raising_judge_costs_one_verdict_not_the_run(self, tenant: Tenant) -> None:
        calls = {"n": 0}

        def flaky(prompt: str) -> str:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("rate limited")
            return _verdict("success")

        grade = Judge(complete=flaky).grade(tenant=tenant, transcript=[_t("user", "hi")])
        assert len(grade.results) == 8
        assert any("judge error" in r.rationale for r in grade.results)

    def test_a_real_client_is_not_graded_on_the_taste(self, tenant: Tenant) -> None:
        client = tenant.model_copy(update={"demo_mode": False, "reseller": None})
        grade = Judge(complete=Scripted(*[_verdict("success")] * 8)).grade(
            tenant=client, transcript=[_t("user", "hi")]
        )
        assert {r.id for r in grade.results} == {
            "persona_held",
            "booking_completed",
            "hallucination_kb",
            "safety_override_correct",
            "two_sentence_rule",
            "name_not_overused",
        }

    def test_fields_are_extracted_in_one_pass(self, tenant: Tenant) -> None:
        fields = json.dumps({"caller_name": "Dan", "caller_town": "Wallingford"})
        grade = Judge(complete=Scripted(*[_verdict("success")] * 7, fields)).grade(
            tenant=tenant, transcript=[_t("user", "hi")]
        )
        assert grade.fields["caller_name"] == "Dan"
        assert grade.fields["prospect_business_name"] is None


class TestTranscriptFormatting:
    def test_speakers_are_labelled(self) -> None:
        out = format_transcript([_t("agent", "Thanks for calling"), _t("user", "hi")])
        assert out == "AGENT: Thanks for calling\nCALLER: hi"


class TestAggregate:
    def _grade(self, tenant: Tenant, verdicts: list[str], call_id: str):
        return Judge(complete=Scripted(*[_verdict(v) for v in verdicts])).grade(
            tenant=tenant, transcript=[_t("user", "hi")], call_id=call_id
        )

    def test_pass_rate_excludes_inapplicable_calls(self, tenant: Tenant) -> None:
        # A criterion that was 'unknown' on 9 of 10 calls and passed the tenth
        # is 100%, not 10%. Folding unknowns in would make every number drift
        # with whatever mix of calls happened to be in the corpus.
        grades = [
            self._grade(tenant, ["unknown"] * 8, "c1"),
            self._grade(tenant, ["success"] + ["unknown"] * 7, "c2"),
        ]
        report = summarize(grades)
        persona = next(c for c in report.criteria if c.id == "persona_held")
        assert persona.applicable == 1
        assert persona.pass_rate == 1.0

    def test_a_criterion_nothing_applied_to_has_no_rate(self, tenant: Tenant) -> None:
        report = summarize([self._grade(tenant, ["unknown"] * 8, "c1")])
        assert next(c for c in report.criteria if c.id == "persona_held").pass_rate is None

    def test_failing_calls_are_named(self, tenant: Tenant) -> None:
        grades = [
            self._grade(tenant, ["success"] * 8, "good"),
            self._grade(tenant, ["failure"] + ["success"] * 7, "bad"),
        ]
        report = summarize(grades)
        assert report.failing_calls == ("bad",)
        assert report.clean_rate == 0.5
        assert "bad" in report.render()

    def test_regressions_name_what_got_worse(self, tenant: Tenant) -> None:
        before = summarize([self._grade(tenant, ["success"] * 8, "c1")])
        after = summarize([self._grade(tenant, ["failure"] + ["success"] * 7, "c1")])
        worse = regressions(before, after)
        assert any("persona_held" in line for line in worse)

    def test_no_regression_when_nothing_changed(self, tenant: Tenant) -> None:
        before = summarize([self._grade(tenant, ["success"] * 8, "c1")])
        after = summarize([self._grade(tenant, ["success"] * 8, "c1")])
        assert regressions(before, after) == []


class TestArchive:
    def test_a_call_survives_a_round_trip(self, tmp_path) -> None:
        archive = CallArchive(tmp_path)
        archive.write({"call_id": "c1", "tenant_id": "millbrook", "transcript": []})
        assert [r["call_id"] for r in archive.read()] == ["c1"]

    def test_an_unserializable_payload_does_not_raise(self, tmp_path) -> None:
        # Losing the archive is not worth a crash on the way out of a call that
        # already happened.
        CallArchive(tmp_path).write({"call_id": "c1", "obj": object()})

    def test_a_corrupt_line_is_skipped_not_fatal(self, tmp_path) -> None:
        (tmp_path / "calls-2026-01-01.jsonl").write_text(
            '{"call_id":"good"}\nnot json\n', encoding="utf-8"
        )
        assert [r["call_id"] for r in CallArchive(tmp_path).read()] == ["good"]

    def test_an_empty_archive_reads_empty(self, tmp_path) -> None:
        assert list(CallArchive(tmp_path).read()) == []
