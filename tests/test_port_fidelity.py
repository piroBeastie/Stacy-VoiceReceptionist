"""Guards on the port from ElevenLabs.

These are not style checks. Every assertion here corresponds to a line in the
v4 prompt that exists because something went wrong on a real call, or to a
promise the agent makes to a caller. If one of these fails, the rebuild has
regressed against the demo that prospects have already heard.
"""

from __future__ import annotations

import pytest

from dana.evals import active_criteria, active_fields
from dana.prompt import PromptBuilder
from dana.providers.base import TELEPHONY_FORMAT
from dana.providers.registry import pipeline_for
from dana.tenants import TenantStore

TENANT_DIR = "src/dana/tenants/data"


@pytest.fixture(scope="module")
def store() -> TenantStore:
    return TenantStore(TENANT_DIR)


@pytest.fixture(scope="module")
def millbrook(store: TenantStore):
    return store.for_number("+18603989358")


@pytest.fixture(scope="module")
def prompt(millbrook) -> str:
    return PromptBuilder().render(millbrook)


class TestSafetyOverride:
    """The one failure mode that can hurt a caller. Checked first, always."""

    def test_script_is_verbatim(self, prompt: str) -> None:
        assert (
            "Please leave the house right now, don't flip any switches or light "
            "anything, and call nine-one-one from outside."
        ) in prompt

    def test_override_stops_everything(self, prompt: str) -> None:
        assert "From then on: no booking, no questions, no goodbye." in prompt

    def test_override_repeats_verbatim(self, prompt: str) -> None:
        assert "repeat the same instruction in the same words" in prompt

    def test_quiet_line_fallback(self, prompt: str) -> None:
        assert '"Please go outside now."' in prompt


class TestPromisesToCallers:
    """Lines where the agent commits to something. Each has burned someone."""

    def test_sms_consent_language_is_verbatim(self, prompt: str) -> None:
        # TCPA. The words are the disclosure; paraphrase is non-compliance.
        assert "Reply STOP anytime to opt out, or HELP for help." in prompt
        assert "Message and data rates may apply." in prompt
        assert "Is it okay if we text that number?" in prompt

    def test_no_lead_claim_without_spoken_digits(self, prompt: str) -> None:
        assert (
            'Never say "I\'ve sent that over to Mister Miles" unless you have spoken '
            "the caller's phone number digits out loud in this call." in prompt
        )

    def test_never_promises_a_callback_within_minutes(self, prompt: str) -> None:
        assert 'will call "in the next few minutes" or "right away."' in prompt

    def test_never_promises_transfer(self, prompt: str) -> None:
        assert "You have no transfer tool on this line." in prompt

    def test_never_promises_exact_arrival_time(self, prompt: str) -> None:
        assert "Never promise an exact time." in prompt


class TestConversationDiscipline:
    """The rules that make it sound like a person rather than a chatbot."""

    def test_two_sentence_ceiling_stated_twice(self, prompt: str) -> None:
        # Once in Tone, once in Guardrails. The repetition is intentional.
        assert prompt.count("Keep responses to two sentences at a time.") == 2

    def test_name_used_exactly_three_times(self, prompt: str) -> None:
        assert "Three times in a call, never seven." in prompt

    def test_banned_corporate_phrases(self, prompt: str) -> None:
        assert '"I understand your frustration"' in prompt
        assert '"I apologize for the inconvenience."' in prompt

    def test_one_question_per_turn(self, prompt: str) -> None:
        assert "Ask one question at a time" in prompt

    def test_address_mandatory_before_confirmation(self, prompt: str) -> None:
        assert "Never confirm a booking without a street address" in prompt

    def test_fuel_asked_only_for_furnace_or_boiler(self, prompt: str) -> None:
        assert "never ask gas-or-electric about an air conditioner" in prompt

    def test_step_importance_markers_survive(self, prompt: str) -> None:
        # 23 in the rendered demo prompt. Each marks a rule that failed once.
        assert prompt.count("This step is important") == 23


class TestTasteIsBounded:
    """The demo's most original mechanic, and the easiest to over-deliver."""

    def test_taste_is_one_greeting_and_one_question(self, prompt: str) -> None:
        assert "say exactly one greeting" in prompt
        assert "reply with exactly one short intake question" in prompt

    def test_taste_stops_itself(self, prompt: str) -> None:
        assert "Never a second greeting, never a second scenario" in prompt


class TestTenantParameterization:
    """A second client must not inherit Millbrook's facts or QuickFlip's pitch."""

    def test_demo_blocks_absent_for_a_real_client(self, millbrook) -> None:
        client = millbrook.model_copy(update={"demo_mode": False, "reseller": None})
        rendered = PromptBuilder().render(client)
        for demo_artifact in ("Mister Miles", "QuickFlip", "the taste", "prospect"):
            assert demo_artifact not in rendered, f"{demo_artifact!r} leaked to a real client"

    def test_safety_override_survives_without_demo_mode(self, millbrook) -> None:
        client = millbrook.model_copy(update={"demo_mode": False, "reseller": None})
        rendered = PromptBuilder().render(client)
        assert "call nine-one-one from outside" in rendered

    def test_demo_mode_requires_a_reseller(self, millbrook) -> None:
        # Otherwise the close has nobody to hand the lead to.
        with pytest.raises(ValueError, match="reseller"):
            millbrook.model_copy(update={"reseller": None}).model_validate(
                {**millbrook.model_dump(), "reseller": None}
            )

    def test_business_facts_come_from_tenant_data(self, prompt: str, millbrook) -> None:
        assert millbrook.business_name in prompt
        assert millbrook.agent_name in prompt
        for town in millbrook.service_area:
            assert town in millbrook.service_area_spoken

    def test_no_unrendered_template_slots(self, prompt: str) -> None:
        assert "${" not in prompt

    def test_runtime_placeholders_survive_render(self, prompt: str) -> None:
        assert "{{system__time}}" in prompt
        assert "{{system__caller_id}}" in prompt

    def test_the_volatile_lines_stay_at_the_very_end(self, prompt: str) -> None:
        # Prompt caching keys on a prefix: everything before the first byte
        # that varies per call is cacheable, everything after it is not. These
        # two placeholders used to sit ~200 tokens in, leaving 5,200 tokens of
        # stable rules re-prefilled on every turn. Anything added to the
        # template that changes per call belongs below them.
        first_volatile = min(
            prompt.index("{{system__time}}"), prompt.index("{{system__caller_id}}")
        )
        cacheable = first_volatile / len(prompt)
        assert cacheable > 0.95, (
            f"only {cacheable:.1%} of the prompt is a stable prefix; something "
            "per-call was added above the 'This call' section"
        )

    def test_runtime_fill_blanks_unknown_caller(self, prompt: str) -> None:
        # The prompt branches on "if nothing appears after 'is'", so an unknown
        # caller must produce an empty string, never the literal "None".
        filled = PromptBuilder().fill_runtime(prompt, now="Monday", caller_id="")
        assert "None" not in filled.split("The caller's number is")[1][:40]


class TestEvalParity:
    """Main grades on 8 criteria and 11 fields. So must this."""

    def test_demo_tenant_matches_main(self, millbrook) -> None:
        assert len(active_criteria(millbrook.demo_mode)) == 8
        assert len(active_fields(millbrook.demo_mode)) == 11

    def test_all_eight_criterion_ids_present(self, millbrook) -> None:
        assert {c.id for c in active_criteria(millbrook.demo_mode)} == {
            "persona_held",
            "booking_completed",
            "hallucination_kb",
            "safety_override_correct",
            "taste_bounded",
            "callback_captured",
            "two_sentence_rule",
            "name_not_overused",
        }

    def test_criteria_render_without_leaks(self, millbrook) -> None:
        ctx = PromptBuilder().context(millbrook)
        for item in (*active_criteria(True), *active_fields(True)):
            assert "${" not in item.render(ctx), item.id

    def test_criteria_render_for_a_real_client_too(self, millbrook) -> None:
        # This is the check that was missing. `persona_held` referenced
        # ${reseller_short_company} without being demo-only, so rendering the
        # suite against a client raised KeyError -- the eval harness could not
        # run at all for the exact case this repo exists to support.
        client = millbrook.model_copy(update={"demo_mode": False, "reseller": None})
        ctx = PromptBuilder().context(client)
        for item in (*active_criteria(False), *active_fields(False)):
            rendered = item.render(ctx)
            assert "${" not in rendered, item.id
            assert "reseller" not in rendered.lower(), item.id

    def test_a_shared_criterion_id_resolves_to_exactly_one_wording(self) -> None:
        for demo in (True, False):
            ids = [c.id for c in active_criteria(demo)]
            assert len(ids) == len(set(ids)), f"duplicate criterion id for demo={demo}"

    def test_real_client_drops_demo_only_criteria(self) -> None:
        ids = {c.id for c in active_criteria(demo_mode=False)}
        assert "taste_bounded" not in ids
        assert "callback_captured" not in ids
        assert "safety_override_correct" in ids


class TestAudioPath:
    """A hidden transcode makes every latency number downstream a lie."""

    def test_mulaw_end_to_end(self, millbrook) -> None:
        cfg = pipeline_for(millbrook)
        assert cfg.stt.audio_format == TELEPHONY_FORMAT
        assert cfg.tts.audio_format == TELEPHONY_FORMAT

    def test_voice_and_keywords_carry_over(self, millbrook) -> None:
        cfg = pipeline_for(millbrook)
        assert cfg.tts.voice_id == "eVItLK1UvXctxuaRV2Oq"
        assert len(cfg.stt.keywords) == 17

    def test_interim_results_on_for_preemptive_generation(self, millbrook) -> None:
        cfg = pipeline_for(millbrook)
        assert cfg.turn.preemptive_generation
        assert cfg.stt.interim_results, "preemptive generation needs interim transcripts"


class TestTenantRouting:
    def test_unknown_number_is_a_loud_failure(self, store: TenantStore) -> None:
        with pytest.raises(KeyError, match="no tenant configured"):
            store.for_number("+15550000000")

    def test_phone_numbers_must_be_e164(self, millbrook) -> None:
        with pytest.raises(ValueError, match="E.164"):
            millbrook.model_validate({**millbrook.model_dump(), "phone_number": "860-398-9358"})
