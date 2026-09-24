"""Guards on the port from ElevenLabs.

These are not style checks. Every assertion here corresponds to a line in the
v4 prompt that exists because something went wrong on a real call, or to a
promise the agent makes to a caller. If one of these fails, the rebuild has
regressed against the demo that prospects have already heard.
"""

from __future__ import annotations

import pytest

from stacy.evals import active_criteria, active_fields
from stacy.prompt import PromptBuilder
from stacy.providers.base import TELEPHONY_FORMAT
from stacy.providers.registry import pipeline_for
from stacy.tenants import TenantStore

TENANT_DIR = "src/stacy/tenants/data"


@pytest.fixture(scope="module")
def store() -> TenantStore:
    return TenantStore(TENANT_DIR)


@pytest.fixture(scope="module")
def millbrook(store: TenantStore):
    # By id, not by number. The tenant_id is the stable identity; the phone
    # number is deployment configuration and changes when a trunk changes --
    # which should never take the test suite down with it.
    return store.get("millbrook")


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
        # 23 ported from the original demo prompt, each marking a rule that
        # failed once, plus one added here for a failure observed on this stack:
        # the model announcing its own tool calls ("let me check if you've been
        # in before"). A count below 24 means a ported rule was lost.
        assert prompt.count("This step is important") == 24


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
        # Asserted against the tenant, not a literal. Pinning the id here meant
        # that recasting the voice -- an ordinary business decision -- failed
        # the suite, which teaches people to edit the test rather than read it.
        cfg = pipeline_for(millbrook)
        assert cfg.tts.voice_id == millbrook.voice_id
        # Whatever the tenant lists, unabridged. "Casey" is deliberately not in
        # it: boosted, it turned a caller's "AC is not cooling" into "Casey is
        # not cooling".
        assert len(cfg.stt.keywords) == len(millbrook.asr_keywords)
        assert "Casey" not in cfg.stt.keywords
        # The words a caller says on every HVAC call, boosted by name: live calls
        # heard "furnace" as "a button" and as "Furnished".
        assert {"furnace", "boiler", "thermostat", "water heater"} <= set(cfg.stt.keywords)

    def test_delivery_comes_from_the_tenant_too(self, millbrook) -> None:
        # One client wants a brisk receptionist, another an unhurried one. That
        # is not an infrastructure default to be shared between them.
        cfg = pipeline_for(millbrook)
        assert cfg.tts.speed == millbrook.voice_speed
        assert cfg.tts.stability == millbrook.voice_stability
        assert cfg.tts.similarity_boost == millbrook.voice_similarity

    def test_speed_past_the_telephony_ceiling_is_refused(self, millbrook) -> None:
        # Past 1.2 consonants clip at 8kHz, costing comprehension faster than it
        # buys energy -- so it is a validation error, not a judgement call.
        from stacy.tenants import Tenant

        with pytest.raises(ValueError):
            Tenant.model_validate({**millbrook.model_dump(), "voice_speed": 1.6})

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


class TestTheCascadeIsTwoAccounts:
    """The primary and the fallback are different vendors on different bills.

    `_one_llm` read a fixed `LLM_API_KEY` for both, so "smart primary, fast
    fallback" was not expressible: the fallback was handed the primary's
    credential and would have failed authentication at the exact moment it was
    needed -- four seconds into a timeout, with a caller waiting.
    """

    def test_the_fallback_carries_its_own_key_variable(
        self, millbrook: Tenant, monkeypatch
    ) -> None:
        monkeypatch.setenv("LLM_FALLBACK_MODEL", "llama-3.3-70b-versatile")
        monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "https://api.groq.com/openai/v1")
        cfg = pipeline_for(millbrook)

        assert cfg.llm.fallback is not None
        assert cfg.llm.api_key_env != cfg.llm.fallback.api_key_env

    def test_the_fallback_is_a_different_vendor_from_the_primary(
        self, millbrook: Tenant, monkeypatch
    ) -> None:
        monkeypatch.setenv("LLM_FALLBACK_MODEL", "llama-3.3-70b-versatile")
        cfg = pipeline_for(millbrook)

        assert cfg.llm.vendor == "anthropic"
        assert cfg.llm.fallback.vendor == "openai"

    def test_no_fallback_configured_is_still_no_fallback(
        self, millbrook: Tenant, monkeypatch
    ) -> None:
        monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)
        assert pipeline_for(millbrook).llm.fallback is None

    def test_a_missing_key_names_the_variable_it_wanted(self, monkeypatch) -> None:
        # The failure a person has to act on is "which variable", not "401".
        from stacy.providers.base import LLMConfig
        from stacy.providers.registry import _one_llm

        monkeypatch.delenv("LLM_FALLBACK_API_KEY", raising=False)
        cfg = LLMConfig(vendor="openai", model="whatever", api_key_env="LLM_FALLBACK_API_KEY")
        with pytest.raises(KeyError, match="LLM_FALLBACK_API_KEY"):
            _one_llm(cfg)

    def test_a_blank_key_is_a_missing_key(self, monkeypatch) -> None:
        # An empty value in a .env file is the usual way this goes wrong, and it
        # is not the same failure as an absent variable unless we say so.
        from stacy.providers.base import LLMConfig
        from stacy.providers.registry import _one_llm

        monkeypatch.setenv("LLM_FALLBACK_API_KEY", "   ")
        cfg = LLMConfig(vendor="openai", model="whatever", api_key_env="LLM_FALLBACK_API_KEY")
        with pytest.raises(KeyError, match="LLM_FALLBACK_API_KEY"):
            _one_llm(cfg)

    def test_prompt_caching_is_on_unless_switched_off(
        self, millbrook: Tenant, monkeypatch
    ) -> None:
        # ~5,700 tokens of tenant knowledge, 99.2% byte-identical between turns.
        # Anthropic does not cache unless asked, so the default has to ask.
        monkeypatch.delenv("LLM_CACHING", raising=False)
        assert pipeline_for(millbrook).llm.caching is True

        monkeypatch.setenv("LLM_CACHING", "0")
        assert pipeline_for(millbrook).llm.caching is False


class TestWhoseVoiceTheAgentAnswers:
    """A second voice in the caller's room is not a second caller.

    Nothing downstream can tell them apart: a television's speech is
    transcribed, endpointed and answered exactly as the caller's is. The filter
    is the only place that distinction can be made.
    """

    def test_telephony_background_voice_cancellation_is_the_default(self) -> None:
        from stacy.providers.base import TurnConfig

        assert TurnConfig().noise_cancellation == "bvc_telephony"

    def test_it_can_be_turned_off_for_debugging(self) -> None:
        # When a difficult caller is being cut up, you need to know whether the
        # filter is what is doing it.
        from stacy.providers.base import TurnConfig
        from stacy.providers.registry import build_noise_cancellation

        assert build_noise_cancellation(TurnConfig(noise_cancellation="off")) is None

    def test_the_vad_runs_at_the_rate_the_audio_actually_is(self) -> None:
        # Silero defaults to 16kHz. Phone audio is 8kHz, so the default ran the
        # detector on upsampled narrowband -- no extra information, one extra
        # resample per frame.
        from stacy.providers.base import TELEPHONY_SAMPLE_RATE, TurnConfig

        assert TurnConfig().vad_sample_rate == TELEPHONY_SAMPLE_RATE == 8000

    def test_the_bar_for_counting_as_speech_is_above_silero_defaults(self) -> None:
        # Background speech sits lower in the mix than the caller, and 50ms is
        # short enough that a door closing counts as someone starting to talk.
        from stacy.providers.base import TurnConfig

        cfg = TurnConfig()
        assert cfg.vad_activation_threshold > 0.5
        assert cfg.vad_min_speech_s > 0.05


class TestTheFirstTurnDoesNotPayForImports:
    """Lazy imports on the event loop are not a slow first turn.

    The SDK measured 607ms spent importing httpcore's SSL machinery *on the
    event loop*, inside a job, at the moment of the first LLM or TTS request. A
    blocked loop cannot read from a websocket, so the symptom was not latency:
    it was `Could not synthesize` and `ReadTimeout` against vendors that were
    healthy, on a call where the greeting played and nothing else ever did.
    """

    def test_warming_is_idempotent_and_cheap_to_repeat(self) -> None:
        from stacy.providers.registry import warm_imports

        warm_imports()
        warm_imports()  # modules are cached; the second call must not raise

    def test_warming_actually_loads_the_ssl_stack(self) -> None:
        import sys

        from stacy.providers.registry import warm_imports

        warm_imports()
        assert "ssl" in sys.modules

    def test_the_input_filter_can_be_switched_without_a_build(
        self, millbrook, monkeypatch
    ) -> None:
        # When a caller is being cut up, "is the filter doing it?" should be one
        # environment variable and a restart, not a deploy.
        monkeypatch.setenv("STACY_NOISE_CANCELLATION", "off")
        assert pipeline_for(millbrook).turn.noise_cancellation == "off"

        monkeypatch.delenv("STACY_NOISE_CANCELLATION", raising=False)
        assert pipeline_for(millbrook).turn.noise_cancellation == "bvc_telephony"
