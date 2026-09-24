"""The latency overhaul, after its review: waits by kind of answer, days and windows
picked in code once the model has offered them, the fuel question asked in turn, early
desk replies, and the model's cache warmed at call start."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from stacy.agent import CallRuntime, DeskOperations
from stacy.agent.fastlane import SMS_CONSENT_LINE, answer_kind, desk_reply
from stacy.agent.tools import build_function_tools
from stacy.desk import CalendarStore, MessageStore
from stacy.providers.base import LLMConfig
from stacy.providers.registry import warm_llm_cache
from stacy.tenants import Tenant


def _runtime(tenant: Tenant, tmp_path, now: datetime) -> CallRuntime:
    rt = CallRuntime.build(
        tenant=tenant,
        call_id="call-1",
        caller_id="+12035550147",
        calendar=CalendarStore(tmp_path / "cal"),
        messages=MessageStore(tmp_path / "msg"),
        now=now,
    )
    rt.slots.update(problem="furnace not heating", town="Wallingford")
    return rt


@pytest.fixture
def runtime(tenant: Tenant, tmp_path, monday_9am: datetime) -> CallRuntime:
    return _runtime(tenant, tmp_path, monday_9am)


@pytest.fixture
def ready(runtime) -> CallRuntime:
    """Everything recorded but the day and window."""
    runtime.slots.update(
        customer_name="Nanak", phone="+12035550199", sms_consent=True, street_address="123 Coffee Street"
    )
    return runtime


def _turn(runtime: CallRuntime, agent: str, caller: str):
    runtime.record_agent(agent)
    runtime.record_user(caller)
    return desk_reply(runtime, caller)


class TestHowLongToWait:
    @pytest.mark.parametrize(
        ("line", "kind"),
        [
            ("Thanks, Nanak. What's the best number to reach you?", "patient"),
            ("And what's the street address there?", "patient"),
            ("Can I get your name?", "patient"),
            ("Is [slow] two-zero-three, five-five-five, zero-one-four-seven the best number to reach you?", "patient"),
            (f"That's [slow] two-zero-three, five-five-five, zero-one-nine-nine. {SMS_CONSENT_LINE}", ""),
            (f"Perfect. {SMS_CONSENT_LINE}", "eager"),
            ("You're all set. Anything else before I let you go?", "eager"),
            ("Anything else?", "eager"),
            ("Is there anything else going on with it, like a noise or a smell?", ""),
            ("Sorry, what was the number again?", "patient"),
            ("Got it. And what's the house number?", "patient"),
            ("Oh no. Can you tell me what's going on with it?", ""),
            ("Is it running but not getting cold, or not coming on at all?", ""),
            ("Today I've got two to five, or tomorrow eight to eleven. Which works better?", ""),
            ("Got it, we service Wallingford.", ""),
        ],
    )
    def test_the_wait_follows_the_question(self, runtime, line: str, kind: str) -> None:
        runtime.record_agent(line)
        assert answer_kind(runtime) == kind

    def test_a_read_back_is_heard_out_not_rushed(self, ready) -> None:
        ready.slots.update(day="tomorrow", window="eleven to two")
        ready.record_agent(
            "So that's Nanak, 123 Coffee Street in Wallingford, furnace not heating, tomorrow, "
            "eleven to two. Does that all sound right?"
        )
        assert answer_kind(ready) == ""


class TestTheTimeIsTheModelsToOffer:
    def test_after_the_last_detail_the_model_offers_the_time(self, runtime) -> None:
        runtime.slots.update(customer_name="Nanak", phone="+12035550199", street_address="123 Coffee Street")
        line = f"That's [slow] two-zero-three, five-five-five, zero-one-nine-nine. {SMS_CONSENT_LINE}"
        assert _turn(runtime, line, "Yes.") is None


class TestPickingFromAnOffer:
    def test_a_window_goes_with_the_day_named_before_it(self, ready) -> None:
        line = "Today I've got two to five, or tomorrow eight to eleven. Which works better?"
        reply = _turn(ready, line, "Eight to eleven.")
        assert reply is not None and reply.record == {"day": "tomorrow", "window": "eight to eleven"}
        assert reply.speech.startswith("So that's Nanak,")

    def test_a_window_goes_with_the_day_named_after_it(self, ready) -> None:
        line = "I can do eleven to two today, or eight to eleven tomorrow. Which works?"
        reply = _turn(ready, line, "Eight to eleven.")
        assert reply is not None and reply.record == {"day": "tomorrow", "window": "eight to eleven"}

    @pytest.mark.parametrize(
        "line",
        [
            "Today or tomorrow, I've got two to five. Which works?",
            "Today two to five tomorrow, or Thursday. Which works?",
        ],
    )
    def test_a_window_with_no_clear_day_goes_to_the_model(self, ready, line: str) -> None:
        assert _turn(ready, line, "Two to five.") is None


class TestADayOnItsOwn:
    def test_that_days_windows(self, ready) -> None:
        reply = _turn(ready, "Would today or tomorrow work better?", "Tomorrow.")
        assert reply is not None and reply.record == {"day": "tomorrow"}
        assert reply.speech == "Tomorrow I've got eight to eleven, eleven to two, or two to five. Which works best?"

    def test_only_the_morning_ones_when_they_say_morning(self, ready) -> None:
        reply = _turn(ready, "Would today or tomorrow work better?", "Tomorrow morning.")
        assert reply is not None
        assert reply.speech == "Tomorrow I've got eight to eleven or eleven to two. Which works best?"

    def test_the_one_window_offered_for_that_day_is_picked(self, ready) -> None:
        reply = _turn(ready, "I've got eight to eleven tomorrow, or would Thursday be better?", "Tomorrow.")
        assert reply is not None and reply.record == {"day": "tomorrow", "window": "eight to eleven"}
        assert reply.speech.startswith("So that's Nanak,")

    def test_a_day_and_window_together_are_read_back(self, ready) -> None:
        reply = _turn(ready, "Would today or tomorrow work better?", "Tomorrow, 11 to 2.")
        assert reply is not None and reply.record == {"day": "tomorrow", "window": "eleven to two"}

    def test_a_day_that_answers_something_else_is_not_a_booking_day(self, ready) -> None:
        assert _turn(ready, "Got it. When did it stop working?", "Today.") is None

    def test_a_day_before_the_other_details_goes_to_the_model(self, runtime) -> None:
        assert _turn(runtime, "Would today or tomorrow work better?", "Tomorrow.") is None

    def test_a_window_nearly_over_is_not_offered(self, tenant, tmp_path) -> None:
        rt = _runtime(tenant, tmp_path, datetime(2026, 9, 14, 10, 0, tzinfo=ZoneInfo(tenant.timezone)))
        rt.slots.update(customer_name="Nanak", phone="+12035550199", sms_consent=True, street_address="1 Elm Street")
        reply = _turn(rt, "Would today or tomorrow work better?", "Today.")
        assert reply is not None
        assert reply.speech == "Today I've got eleven to two or two to five. Which works best?"


class TestFuel:
    @pytest.fixture
    def furnace(self, runtime) -> CallRuntime:
        runtime.slots.update(customer_name="Bubba", phone="+12035550199", sms_consent=True, system_type="furnace")
        return runtime

    def test_asked_after_the_street_for_a_furnace(self, furnace) -> None:
        reply = _turn(furnace, "And what's the street address there?", "12 Abuba Street.")
        assert reply is not None and reply.speech == "Okay, 12 Abuba Street. Is that a gas furnace or an electric one?"

    @pytest.mark.parametrize(("system", "said"), [("gas furnace", ""), ("boiler", "My oil boiler won't fire.")])
    def test_never_asked_when_they_already_said_it(self, furnace, system: str, said: str) -> None:
        furnace.slots.system_type = system
        if said:
            furnace.record_user(said)
        assert _turn(furnace, "And what's the street address there?", "12 Abuba Street.") is None

    @pytest.mark.parametrize(
        ("line", "caller"),
        [
            ("Is that a gas furnace or an electric one?", "Gas."),
            ("Does it smell like gas, or more like something electric?", "Gas."),
            ("Is that a gas furnace or an electric one?", "Not gas."),
        ],
    )
    def test_the_answer_is_the_models(self, furnace, line: str, caller: str) -> None:
        furnace.slots.street_address = "12 Abuba Street"
        assert _turn(furnace, line, caller) is None


class TestReadBack:
    @pytest.fixture
    def picked(self, runtime) -> CallRuntime:
        runtime.slots.update(customer_name="Bubba", phone="+12035550199", sms_consent=True, street_address="12 Abuba Street")
        return runtime

    def _read_back(self, rt: CallRuntime) -> str:
        line = "Tomorrow I've got eight to eleven, eleven to two, or two to five. Which works best?"
        reply = _turn(rt, line, "8 to 11.")
        assert reply is not None
        return reply.speech

    def test_a_problem_without_equipment_gets_the_system(self, picked) -> None:
        picked.slots.update(system_type="furnace", fuel="gas")
        picked.slots.problem = "not working"
        assert "Wallingford, furnace not working, tomorrow" in self._read_back(picked)

    @pytest.mark.parametrize("problem", ["AC not cooling", "no heat", "furnace won't start"])
    def test_a_problem_that_names_it_is_left_alone(self, picked, problem: str) -> None:
        picked.slots.update(system_type="air conditioner")
        picked.slots.problem = problem
        speech = self._read_back(picked)
        assert "air conditioner" not in speech and problem[1:] in speech


def test_a_second_booking_of_the_same_visit_is_refused(ready) -> None:
    ready.slots.update(day="tomorrow", window="eleven to two")
    ready.record_agent("So that's Nanak, 123 Coffee Street in Wallingford, tomorrow, eleven to two. Does that all sound right?")
    ready.record_user("Yes.")
    ops = DeskOperations(ready)
    assert ops.book_appointment().startswith("Booked:")
    again = ops.book_appointment()
    assert again.startswith("Booked already") and len(ready.appointments) == 1


class TestEarlyDeskReplies:
    """A desk reply voiced before the turn is confirmed: kept only for the same words."""

    def test_the_same_words_despite_case_punctuation_and_notes(self) -> None:
        from stacy.agent.nodes import _same_words

        assert _same_words("Yeah. Perfect.", "yeah perfect")
        assert _same_words("Tomorrow.", "Tomorrow. (Not said by the caller: the desk recorded tomorrow.)")
        assert not _same_words("Tomorrow.", "Tomorrow, 11 to 2.")
        assert not _same_words("", "anything")

    def test_only_the_pass_that_answers_the_caller(self) -> None:
        from stacy.agent.nodes import _ends_with_caller

        caller = SimpleNamespace(type="message", role="user")
        agent = SimpleNamespace(type="message", role="assistant")
        tool_output = SimpleNamespace(type="function_call_output", role=None)
        assert _ends_with_caller(SimpleNamespace(items=[agent, caller]))
        assert not _ends_with_caller(SimpleNamespace(items=[caller, tool_output]))
        assert not _ends_with_caller(SimpleNamespace(items=[]))


class TestWarmUp:
    def _capture(self, monkeypatch) -> dict:
        import anthropic

        captured: dict = {}

        class _Messages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    usage=SimpleNamespace(cache_creation_input_tokens=7000, cache_read_input_tokens=0)
                )

        class _Client:
            def __init__(self, **kwargs) -> None:
                self.messages = _Messages()

            def close(self) -> None:
                return None

        monkeypatch.setattr(anthropic, "Anthropic", _Client)
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        return captured

    def test_it_primes_exactly_what_the_agent_sends(self, monkeypatch, runtime) -> None:
        captured = self._capture(monkeypatch)
        tools = build_function_tools(DeskOperations(runtime))
        result = warm_llm_cache(LLMConfig(), runtime.instructions, tools)
        assert result is not None and result["written"] == 7000
        assert captured["max_tokens"] == 0 and "stream" not in captured
        assert captured["model"] == LLMConfig().model
        assert captured["system"] == [
            {"type": "text", "text": runtime.instructions, "cache_control": {"type": "ephemeral"}}
        ]
        assert captured["tools"][-1]["cache_control"] == {"type": "ephemeral"}
        assert all("cache_control" not in t for t in captured["tools"][:-1])
        assert len(captured["tools"]) == len(tools)

    def test_only_for_a_cached_anthropic_model(self, monkeypatch, runtime) -> None:
        self._capture(monkeypatch)
        assert warm_llm_cache(LLMConfig(vendor="openai"), runtime.instructions, []) is None
        assert warm_llm_cache(LLMConfig(caching=False), runtime.instructions, []) is None


class TestSecondReview:
    """What the second review confirmed, each fixed."""

    def test_the_demo_close_is_not_a_booking_question(self, ready) -> None:
        reseller = ready.tenant.reseller
        assert reseller is not None
        ready.record_agent(
            "And that's how I'd handle it. Since you're checking me out -- want me to have "
            f"{reseller.contact_spoken_name} call you?"
        )
        ready.record_user("Sure.")
        assert _turn(ready, "Today, or tomorrow morning?", "Tomorrow morning.") is None

    def test_the_same_bare_question_in_a_booking_still_gets_the_desk(self, ready) -> None:
        reply = _turn(ready, "Today, or tomorrow morning?", "Tomorrow morning.")
        assert reply is not None and reply.record == {"day": "tomorrow"}

    @pytest.mark.parametrize(
        "line",
        [
            "Did it quit today, or has it been out since Sunday?",
            "Is it something that started today, or yesterday?",
            "Do you need someone today, or can it wait until tomorrow?",
        ],
    )
    def test_a_question_about_the_problem_is_not_about_the_visit(self, ready, line: str) -> None:
        assert _turn(ready, line, "Just today.") is None

    def test_a_yes_to_an_offer_with_another_day_goes_to_the_model(self, ready) -> None:
        assert _turn(ready, "Tomorrow I've got eight to eleven. Or would Thursday work better?", "Yes.") is None

    def test_a_yes_to_a_plain_offer_is_still_a_pick(self, ready) -> None:
        reply = _turn(ready, "Tomorrow I've got eight to eleven open. Does that work?", "Yes.")
        assert reply is not None and reply.record == {"day": "tomorrow", "window": "eight to eleven"}

    def test_the_day_offered_instead_gets_its_own_windows(self, ready) -> None:
        reply = _turn(ready, "I've got eight to eleven, or would Thursday be better?", "Thursday.")
        assert reply is not None and reply.record == {"day": "thursday"}
        assert reply.speech.startswith("Thursday I've got")

    def test_two_days_at_once_go_to_the_model(self, ready) -> None:
        assert _turn(ready, "Would today or tomorrow work better?", "Tomorrow and Thursday.") is None

    def test_the_same_list_is_not_read_out_twice(self, ready) -> None:
        line = "For tomorrow I have eight to eleven, eleven to two, or two to five -- which one works?"
        assert _turn(ready, line, "Tomorrow.") is None

    def test_evening_is_the_models(self, ready) -> None:
        # No window is an evening one: the model says what the latest is.
        assert _turn(ready, "Would today or tomorrow work better?", "Tomorrow evening.") is None

    def test_fuel_is_asked_once(self, runtime) -> None:
        runtime.slots.update(
            customer_name="Bubba", phone="+12035550199", sms_consent=True, system_type="furnace",
            street_address="12 Abuba Street",
        )
        caller = "Honestly I don't know, it's pretty old."
        assert _turn(runtime, "Is that a gas furnace or an electric one?", caller) is None
        line = "That's fine. Tomorrow I've got eight to eleven, eleven to two, or two to five. Which works best?"
        assert _turn(runtime, line, "Eight to eleven.") is None  # the model records "unknown" and reads back
        runtime.slots.update(fuel="unknown")
        reply = _turn(runtime, line, "Eight to eleven.")
        assert reply is not None and reply.speech.startswith("So that's Bubba,")

    def test_a_symptom_reads_back_with_its_system(self, runtime) -> None:
        runtime.slots.update(
            customer_name="Bubba", phone="+12035550199", sms_consent=True, street_address="12 Abuba Street",
            system_type="boiler", fuel="gas",
        )
        runtime.slots.problem = "not heating"
        line = "Tomorrow I've got eight to eleven, eleven to two, or two to five. Which works best?"
        reply = _turn(runtime, line, "8 to 11.")
        assert reply is not None and "Wallingford, boiler not heating, tomorrow" in reply.speech

    def test_a_second_visit_on_another_day_is_booked(self, ready) -> None:
        ready.slots.update(day="tomorrow", window="eleven to two")
        ready.record_agent(
            "So that's Nanak, 123 Coffee Street in Wallingford, tomorrow, eleven to two. Does that all sound right?"
        )
        ready.record_user("Yes.")
        ops = DeskOperations(ready)
        assert ops.book_appointment().startswith("Booked:")
        ready.slots.update(day="thursday", problem="AC not cooling")
        ready.record_agent(
            "So that's Nanak, 123 Coffee Street in Wallingford, Thursday, eleven to two. Does that all sound right?"
        )
        ready.record_user("Yes.")
        assert ops.book_appointment().startswith("Booked:")
        assert len(ready.appointments) == 2


class TestAKeptReplyBelongsToItsMessage:
    """The desk's reply is tied to the caller's message, so the same words later can't replay it."""

    @pytest.fixture
    def agent(self, runtime, monkeypatch):
        pytest.importorskip("livekit.agents")
        from livekit.agents import Agent

        from stacy.agent.nodes import build_agent

        async def model(self, chat_ctx, tools, model_settings):
            yield "MODEL"

        monkeypatch.setattr(Agent, "llm_node", model)
        runtime.slots.update(customer_name="Nanak", phone="+12035550199")
        runtime.record_agent(f"That's [slow] two-zero-three, five-five-five, zero-one-nine-nine. {SMS_CONSENT_LINE}")
        return build_agent(runtime)

    @staticmethod
    def _first(agent, message) -> str:
        import asyncio

        from livekit.agents.llm import ChatContext

        async def run() -> str:
            ctx = ChatContext.empty()
            ctx.insert(message)
            async for chunk in agent.llm_node(ctx, [], None):
                return chunk if isinstance(chunk, str) else str(chunk)
            return ""

        return asyncio.run(run())

    @staticmethod
    def _hook(agent, message) -> None:
        import asyncio

        from livekit.agents.llm import ChatContext

        asyncio.run(agent.on_user_turn_completed(ChatContext.empty(), message))

    def test_a_kept_early_reply_is_not_replayed_on_the_next_yes(self, agent, runtime) -> None:
        from livekit.agents.llm import ChatMessage

        early = self._first(agent, ChatMessage(role="user", content=["Yes."]))
        assert early == "Perfect. And what's the street address there?"
        confirmed = ChatMessage(role="user", content=["Yes."])
        self._hook(agent, confirmed)
        assert confirmed.content == ["Yes."]  # kept: the message is left alone
        runtime.record_agent(early)  # the early audio played; no regenerated pass
        # Another "Yes." with no VAD event between: a new message, so the model's.
        assert self._first(agent, ChatMessage(role="user", content=["Yes."])) == "MODEL"

    def test_a_regenerated_pass_on_the_kept_message_says_the_same(self, agent) -> None:
        from livekit.agents.llm import ChatMessage

        early = self._first(agent, ChatMessage(role="user", content=["Yes."]))
        confirmed = ChatMessage(role="user", content=["Yes."])
        self._hook(agent, confirmed)
        # The SDK may still drop the early pass and regenerate on the confirmed message.
        assert self._first(agent, confirmed) == early

    def test_a_pending_reply_is_spoken_on_its_own_message_only(self, agent, runtime) -> None:
        from livekit.agents.llm import ChatMessage

        confirmed = ChatMessage(role="user", content=["Yes."])
        self._hook(agent, confirmed)
        assert runtime.desk_reply is not None and runtime.desk_reply[0] == confirmed.id
        runtime.record_agent(runtime.desk_reply[1])
        assert self._first(agent, ChatMessage(role="user", content=["Yes."])) == "MODEL"


class TestThirdReview:
    """What the review of the second round confirmed, each fixed."""

    def test_a_yes_takes_the_day_the_offer_named_not_the_one_recorded(self, ready) -> None:
        ready.slots.update(day="today")
        line = "Tomorrow morning I've got eight to eleven. Does that work better than today?"
        reply = _turn(ready, line, "Yes.")
        assert reply is not None and reply.record == {"day": "tomorrow", "window": "eight to eleven"}

    def test_another_part_of_the_day_is_not_a_yes_to_the_window_offered(self, ready) -> None:
        assert _turn(ready, "Eight to eleven tomorrow?", "Tomorrow evening.") is None
        reply = _turn(ready, "Eight to eleven tomorrow?", "Tomorrow afternoon.")
        assert reply is not None and reply.record == {"day": "tomorrow"}
        assert "eight to eleven" not in reply.speech

    def test_a_day_whose_windows_were_listed_together_is_not_a_pick(self, ready) -> None:
        line = (
            "Today we've got two to five open, or I can get you tomorrow morning eight to eleven, "
            "eleven to two, or two to five -- which works better?"
        )
        reply = _turn(ready, line, "Tomorrow.")
        assert reply is None or "window" not in reply.record

    def test_a_day_that_came_with_its_window_is_a_pick(self, ready) -> None:
        reply = _turn(ready, "Today I've got two to five, or tomorrow eight to eleven. Which works better?", "Tomorrow.")
        assert reply is not None and reply.record == {"day": "tomorrow", "window": "eight to eleven"}

    def test_a_customer_with_the_resellers_first_name_keeps_the_desk(self, runtime) -> None:
        from stacy.agent.fastlane import _in_the_close

        first = runtime.tenant.reseller.contact_full_name.split()[0]
        runtime.slots.update(customer_name=first)
        line = f"Thanks, {first}. Is [slow] two-zero-three, five-five-five, zero-one-four-seven the best number to reach you?"
        reply = _turn(runtime, line, "Yes.")
        assert not _in_the_close(runtime)
        assert reply is not None and reply.record == {"phone": "+12035550147"}

    def test_a_system_they_dont_know_is_not_read_back(self, runtime) -> None:
        runtime.slots.update(
            customer_name="Bubba", phone="+12035550199", sms_consent=True, street_address="12 Abuba Street",
            system_type="unknown",
        )
        runtime.slots.problem = "not working"
        line = "Tomorrow I've got eight to eleven, eleven to two, or two to five. Which works best?"
        reply = _turn(runtime, line, "8 to 11.")
        assert reply is not None and "unknown" not in reply.speech and "Wallingford, not working, tomorrow" in reply.speech

    def test_booked_already_names_the_day_as_said(self, ready) -> None:
        ready.slots.update(day="tomorrow", window="eleven to two")
        ready.record_agent("So that's Nanak, 123 Coffee Street in Wallingford, tomorrow, eleven to two. Does that all sound right?")
        ready.record_user("Yes.")
        ops = DeskOperations(ready)
        ops.book_appointment()
        again = ops.book_appointment()
        assert again.startswith("Booked already") and "tomorrow" in again and "2026" not in again

    @pytest.mark.parametrize(
        "turn",
        [
            ["Is that Wallingford?", "Great. And what's the street address there?"],
            ["Sorry, I didn't catch that?", "And can I get your name?"],
        ],
    )
    def test_the_last_question_sets_the_wait_even_after_another(self, runtime, turn) -> None:
        for line in turn:
            runtime.record_agent(line)
        assert answer_kind(runtime) == "patient"

    def test_the_closes_offer_of_their_own_number_is_waited_for(self, runtime) -> None:
        name = runtime.tenant.reseller.contact_spoken_name
        runtime.record_agent(f"Want me to have {name} call you at [slow] two-zero-three, five-five-five, zero-one-four-seven?")
        assert answer_kind(runtime) == "patient"


class TestTheDeskAnswersWhatTheFileKnows:
    """Price, hours, towns and the tenant's own FAQs: read out, never composed.

    The model answered a price right and a staffing question wrong on the same
    call ("we send whoever's available", where the file lists one technician).
    What the file states outright is answered from it; everything else still
    goes to the model, which is the only thing allowed to say "I'm not sure".
    """

    @pytest.fixture
    def asked_for_a_name(self, runtime) -> CallRuntime:
        runtime.record_agent("And can I get your name?")
        return runtime

    def test_the_price_comes_from_the_service_list(self, asked_for_a_name) -> None:
        reply = desk_reply(asked_for_a_name, "How much do you charge for a service call?")
        assert reply is not None
        assert "89 to 129 dollars" in reply.speech and "waived if you book the repair" in reply.speech
        assert reply.speech.endswith("And can I get your name?")  # the booking carries on

    def test_the_hours_come_from_the_hours(self, asked_for_a_name) -> None:
        reply = desk_reply(asked_for_a_name, "What are your hours?")
        assert reply is not None
        assert "Monday to Friday eight to five" in reply.speech
        assert "Saturday nine to one" in reply.speech and "Closed Sunday" in reply.speech

    def test_the_towns_come_from_the_service_area(self, asked_for_a_name, tenant) -> None:
        reply = desk_reply(asked_for_a_name, "Do you cover Cheshire?")
        assert reply is not None
        for town in tenant.service_area:
            assert town in reply.speech

    def test_a_tenant_faq_is_answered_in_its_own_words(self, asked_for_a_name, tenant) -> None:
        reply = desk_reply(asked_for_a_name, "Do you work on mini-splits?")
        assert reply is not None
        answer = next(f.answer for f in tenant.faqs if "mini-split" in f.question)
        assert answer in reply.speech

    @pytest.mark.parametrize(
        "question",
        [
            "Who's the technician coming out?",
            "Is the repair under warranty?",
            "Can you give me a firm price for the part?",
            "How much longer will the tech be?",
            "What time is my appointment?",
        ],
    )
    def test_anything_the_file_does_not_say_goes_to_the_model(self, asked_for_a_name, question) -> None:
        assert desk_reply(asked_for_a_name, question) is None

    def test_an_answer_alone_when_nothing_is_left_to_ask(self, ready) -> None:
        ready.slots.update(day="tomorrow", window="eleven to two")
        ready.record_agent("Anything else before I let you go?")
        reply = desk_reply(ready, "Actually, how much is the visit?")
        assert reply is not None and reply.speech.endswith("waived if you book the repair.")


class TestBookingWhenTheCallerAsksForIt:
    """"Go ahead and book it", a few turns after the read-back.

    Replayed against the model: the caller asked the price between the read-back
    and her yes, and the model answered "you're all set -- you're booked for
    tomorrow" having called nothing. Nothing was in the calendar and she had no
    way to know. The desk books it instead, and the line is spoken only after
    the calendar has taken it.
    """

    @pytest.fixture
    def read_back(self, ready) -> CallRuntime:
        ready.slots.update(day="tomorrow", window="eleven to two")
        ready.record_agent(
            "So that's Nanak, 123 Coffee Street in Wallingford, furnace not heating, "
            "tomorrow, eleven to two. Does that all sound right?"
        )
        ready.record_user("Yeah, but how much is the visit?")
        ready.record_agent("Diagnostic visit runs 89 to 129 dollars, waived if you book the repair.")
        return ready

    def test_the_caller_asking_for_it_books(self, read_back) -> None:
        reply = desk_reply(read_back, "Yeah, go ahead and book it.")
        assert reply is not None and reply.action == "book"
        assert reply.speech.startswith("You're all set.")

    def test_a_yes_to_the_agent_offering_books(self, read_back) -> None:
        read_back.record_agent("Ready to get you booked?")
        reply = desk_reply(read_back, "Yes.")
        assert reply is not None and reply.action == "book"

    def test_a_yes_to_something_else_books_nothing(self, read_back) -> None:
        read_back.record_agent("Is that a gas furnace or an electric one?")
        reply = desk_reply(read_back, "Yes.")
        assert reply is None or reply.action != "book"

    def test_nothing_is_booked_before_it_was_read_back(self, ready) -> None:
        ready.slots.update(day="tomorrow", window="eleven to two")
        ready.record_agent("And what time works for you?")
        reply = desk_reply(ready, "Just book it.")
        assert reply is None or reply.action != "book"

    def test_a_no_books_nothing(self, read_back) -> None:
        read_back.record_agent("Ready to get you booked?")
        assert desk_reply(read_back, "No, not yet.") is None
