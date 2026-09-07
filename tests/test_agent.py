"""The call runtime, the filler policy, and the tools -- all without an SDK.

Everything the agent decides lives in these three modules, so everything the
agent decides is tested here. The SDK-facing files (`nodes`, `entrypoint`)
contain no branches worth asserting on; if a behaviour matters it belongs in
one of the modules below instead.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from dana.agent import CallRuntime, DeskOperations, FillerPolicy, RepetitionTracker
from dana.desk import CalendarStore, MessageStore
from dana.metrics import Stage, TurnTrace
from dana.providers.base import FillerConfig
from dana.tenants import Tenant


class _CollectingSink:
    def __init__(self) -> None:
        self.emitted: list[TurnTrace] = []

    def emit(self, trace: TurnTrace) -> None:
        self.emitted.append(trace)


@pytest.fixture
def sink() -> _CollectingSink:
    return _CollectingSink()


@pytest.fixture
def runtime(tenant: Tenant, sink: _CollectingSink, tmp_path, monday_9am: datetime) -> CallRuntime:
    return CallRuntime.build(
        tenant=tenant,
        call_id="call-1",
        caller_id="+12035550147",
        sink=sink,
        calendar=CalendarStore(tmp_path / "cal"),
        messages=MessageStore(tmp_path / "msg"),
        now=monday_9am,
    )


class TestTurnLifecycle:
    """A stage that reports late has to land on the turn it belongs to."""

    def test_a_turn_is_closed_by_the_next_one(self, runtime: CallRuntime, sink) -> None:
        runtime.begin_turn()
        assert sink.emitted == []
        runtime.begin_turn()
        assert len(sink.emitted) == 1

    def test_marks_arriving_after_audio_still_land(self, runtime: CallRuntime, sink) -> None:
        # This is the bug the old wiring had: the trace was emitted the instant
        # the agent started speaking, so every later mark hit a closed trace.
        runtime.begin_turn()
        runtime.mark(Stage.AUDIO_OUT)
        runtime.mark(Stage.LLM_FIRST_TOKEN)
        runtime.finish()
        assert sink.emitted[0].has(Stage.LLM_FIRST_TOKEN)

    def test_finish_flushes_the_last_turn(self, runtime: CallRuntime, sink) -> None:
        runtime.begin_turn()
        runtime.finish()
        assert len(sink.emitted) == 1

    def test_finish_is_idempotent(self, runtime: CallRuntime, sink) -> None:
        runtime.begin_turn()
        runtime.finish()
        runtime.finish()
        assert len(sink.emitted) == 1

    def test_marking_with_no_open_turn_is_harmless(self, runtime: CallRuntime, sink) -> None:
        runtime.mark(Stage.AUDIO_OUT)
        assert sink.emitted == []

    def test_turn_indices_increment(self, runtime: CallRuntime, sink) -> None:
        runtime.begin_turn()
        runtime.begin_turn()
        runtime.finish()
        assert [t.turn_index for t in sink.emitted] == [0, 1]


class TestPromptAndTranscript:
    def test_runtime_placeholders_are_filled(self, runtime: CallRuntime) -> None:
        assert "{{system__time}}" not in runtime.system_prompt
        assert "{{system__caller_id}}" not in runtime.system_prompt
        assert "+12035550147" in runtime.system_prompt

    def test_an_unknown_caller_leaves_the_slot_empty_not_none(
        self, tenant: Tenant, monday_9am: datetime
    ) -> None:
        # The prompt branches on "if nothing appears after 'is'". The literal
        # string "None" would read as a phone number to the model.
        rt = CallRuntime.build(tenant=tenant, call_id="c", caller_id="", now=monday_9am)
        assert "None" not in rt.system_prompt.split("The caller's number is")[1][:40]

    def test_transcript_records_both_sides(self, runtime: CallRuntime) -> None:
        runtime.record_user("my furnace is out")
        runtime.record_agent("Oh no. What town are you in?")
        assert [t["role"] for t in runtime.transcript] == ["user", "agent"]

    def test_blank_utterances_are_not_recorded(self, runtime: CallRuntime) -> None:
        runtime.record_user("   ")
        runtime.record_agent("")
        assert runtime.transcript == []


class TestFillerPolicy:
    def test_nothing_before_the_caller_speaks(self) -> None:
        # Otherwise the agent mutters "one sec" over its own greeting.
        p = FillerPolicy()
        assert p.soft_timeout_phrase() is None
        assert p.should_speak_before_tool("book_appointment") is False

    def test_unlocked_once_the_caller_speaks(self) -> None:
        p = FillerPolicy()
        p.on_user_message()
        assert p.soft_timeout_phrase() is not None

    def test_one_filler_per_turn(self) -> None:
        p = FillerPolicy()
        p.on_user_message()
        assert p.soft_timeout_phrase() is not None
        assert p.soft_timeout_phrase() is None
        p.begin_turn()
        assert p.soft_timeout_phrase() is not None

    def test_phrases_rotate_rather_than_repeat(self) -> None:
        # At temperature 0 the model will not vary this for us.
        p = FillerPolicy()
        p.on_user_message()
        heard = []
        for _ in range(3):
            p.begin_turn()
            heard.append(p.soft_timeout_phrase())
        assert len(set(heard)) == 3

    def test_an_unseen_tool_is_assumed_slow(self) -> None:
        # The first call is the cold one; staying quiet through it is the worst
        # available choice.
        p = FillerPolicy()
        p.on_user_message()
        assert p.should_speak_before_tool("check_availability") is True

    def test_a_tool_learned_to_be_fast_stops_being_covered(self) -> None:
        p = FillerPolicy()
        p.on_user_message()
        for _ in range(8):
            p.observe_tool("check_availability", 0.01)
        assert p.should_speak_before_tool("check_availability") is False

    def test_a_tool_that_turns_slow_starts_being_covered(self) -> None:
        p = FillerPolicy()
        p.on_user_message()
        for _ in range(8):
            p.observe_tool("book_appointment", 0.01)
        assert p.should_speak_before_tool("book_appointment") is False
        for _ in range(8):
            p.observe_tool("book_appointment", 2.0)
        assert p.should_speak_before_tool("book_appointment") is True

    def test_disabling_the_policy_silences_everything(self) -> None:
        p = FillerPolicy(cfg=FillerConfig(enabled=False))
        p.on_user_message()
        assert p.soft_timeout_phrase() is None
        assert p.should_speak_before_tool("anything") is False


class TestRepetitionTracker:
    def test_short_acknowledgments_may_recur(self) -> None:
        # "Got it." twice in a call is a person, not a loop.
        r = RepetitionTracker()
        assert r.observe("Got it.") is False
        assert r.observe("Got it.") is False
        assert r.repeats == 0

    def test_a_repeated_sentence_is_counted(self) -> None:
        r = RepetitionTracker()
        line = "The tech will give you a call about thirty minutes before they head over."
        assert r.observe(line) is False
        assert r.observe(line) is True
        assert r.repeats == 1

    def test_punctuation_and_case_do_not_hide_a_repeat(self) -> None:
        r = RepetitionTracker()
        r.observe("Anything else before I let you go?")
        assert r.observe("anything else before i let you go") is True


class TestDeskOperations:
    @pytest.fixture
    def ops(self, runtime: CallRuntime) -> DeskOperations:
        return DeskOperations(runtime)

    def test_availability_is_spoken_not_serialized(self, ops: DeskOperations) -> None:
        said = ops.check_availability("tomorrow")
        assert "{" not in said and "[" not in said
        assert "eight to eleven" in said

    def test_an_unreadable_day_asks_rather_than_guesses(self, ops: DeskOperations) -> None:
        assert "didn't catch" in ops.check_availability("whenever suits you")

    def test_booking_returns_a_read_back(self, ops: DeskOperations) -> None:
        said = ops.book_appointment(
            customer_name="Dan",
            phone="+12035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="no heat upstairs",
            day="tomorrow",
            window="eleven to two",
            sms_consent=True,
        )
        assert "14 Elm Street" in said and "eleven to two" in said
        assert ops.runtime.outcome == "booked"
        assert ops.runtime.appointment is not None

    def test_a_full_window_offers_the_alternatives_in_the_same_breath(
        self, ops: DeskOperations, tenant: Tenant
    ) -> None:
        # A dead end here is where a caller decides they're talking to a machine.
        for i in range(tenant.capacity_per_window):
            ops.book_appointment(
                customer_name=f"C{i}",
                phone="+1203555010{i}",
                street_address="1 Main Street",
                town="Wallingford",
                problem="tune up",
                day="tomorrow",
                window="eleven to two",
            )
        said = ops.book_appointment(
            customer_name="Late",
            phone="+12035550199",
            street_address="2 Main Street",
            town="Wallingford",
            problem="tune up",
            day="tomorrow",
            window="eleven to two",
        )
        assert "just filled" in said
        assert "eight to eleven" in said or "two to five" in said

    def test_an_invented_window_is_corrected_with_the_real_ones(
        self, ops: DeskOperations
    ) -> None:
        said = ops.book_appointment(
            customer_name="Dan",
            phone="+12035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="no heat",
            day="tomorrow",
            window="seven to nine",
        )
        assert "eight to eleven" in said

    def test_a_missing_address_is_asked_for_not_worked_around(
        self, ops: DeskOperations
    ) -> None:
        said = ops.book_appointment(
            customer_name="Dan",
            phone="+12035550147",
            street_address="",
            town="Wallingford",
            problem="no heat",
            day="tomorrow",
            window="eleven to two",
        )
        assert "street address" in said
        assert ops.runtime.appointment is None

    def test_emergency_dispatch_sets_the_outcome(self, ops: DeskOperations) -> None:
        said = ops.dispatch_emergency(
            customer_name="Ruth",
            phone="+12035550188",
            street_address="9 Oak Lane",
            town="Meriden",
            problem="water heater leaking",
        )
        assert "9 Oak Lane" in said
        assert ops.runtime.outcome == "dispatched"

    def test_a_returning_caller_is_recognised(self, ops: DeskOperations) -> None:
        ops.book_appointment(
            customer_name="Dan",
            phone="+12035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="no heat upstairs",
            day="tomorrow",
            window="two to five",
        )
        said = ops.lookup_customer()
        assert "Dan" in said and "14 Elm Street" in said

    def test_an_unknown_caller_is_told_so_plainly(self, ops: DeskOperations) -> None:
        assert "No record" in ops.lookup_customer("+12035559999")

    def test_a_message_is_taken_with_a_priority(self, ops: DeskOperations) -> None:
        said = ops.take_message(
            caller_name="Pat",
            phone="+12035550101",
            summary="wants a quote on a new furnace",
            for_staff="Lisa",
        )
        assert "Lisa" in said
        assert ops.runtime.outcome == "message"

    def test_a_booking_outranks_a_later_message(self, ops: DeskOperations) -> None:
        ops.book_appointment(
            customer_name="Dan",
            phone="+12035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="no heat",
            day="tomorrow",
            window="two to five",
        )
        ops.take_message(caller_name="Dan", phone="+12035550147", summary="also asked about filters")
        assert ops.runtime.outcome == "booked"

    def test_operations_degrade_rather_than_raise_without_stores(
        self, tenant: Tenant, monday_9am: datetime
    ) -> None:
        # A tool that raises mid-call drops the call. One that answers in words
        # lets the agent fall back to taking a message.
        bare = CallRuntime.build(tenant=tenant, call_id="c", now=monday_9am)
        ops = DeskOperations(bare)
        assert "message" in ops.check_availability("tomorrow")
        assert "message" in ops.book_appointment(
            customer_name="Dan",
            phone="+1",
            street_address="1 Main",
            town="Wallingford",
            problem="x",
            day="tomorrow",
            window="two to five",
        )


class TestPostCallPayload:
    def test_a_booked_call_carries_the_appointment(self, runtime: CallRuntime) -> None:
        ops = DeskOperations(runtime)
        ops.book_appointment(
            customer_name="Dan",
            phone="+12035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="no heat",
            day="tomorrow",
            window="two to five",
            sms_consent=True,
        )
        payload = runtime.post_call_payload()
        assert payload["outcome"] == "booked"
        assert payload["appointments"][0]["street_address"] == "14 Elm Street"
        assert payload["appointments"][0]["sms_consent"] is True

    def test_an_answered_call_carries_neither(self, runtime: CallRuntime) -> None:
        runtime.record_user("what are your hours?")
        payload = runtime.post_call_payload()
        assert payload["outcome"] == "answered"
        assert payload["appointments"] == [] and payload["message"] is None

    def test_consent_travels_so_a_consumer_can_audit_it(self, runtime: CallRuntime) -> None:
        ops = DeskOperations(runtime)
        ops.book_appointment(
            customer_name="Dan",
            phone="+12035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="no heat",
            day="tomorrow",
            window="two to five",
            sms_consent=False,
        )
        assert runtime.post_call_payload()["appointments"][0]["sms_consent"] is False

    def test_a_second_booking_does_not_erase_the_first(self, runtime: CallRuntime) -> None:
        # "And can you look at the AC while you're here?" -- a scalar dropped
        # the first booking, and took its consent flag with it.
        ops = DeskOperations(runtime)
        ops.book_appointment(
            customer_name="Dan",
            phone="+12035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="no heat",
            day="tomorrow",
            window="two to five",
            sms_consent=True,
        )
        ops.book_appointment(
            customer_name="Dan",
            phone="+12035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="ac making a noise",
            day="tomorrow",
            window="eight to eleven",
            sms_consent=False,
        )
        got = runtime.post_call_payload()["appointments"]
        assert [a["sms_consent"] for a in got] == [True, False]

    def test_an_emergency_outranks_an_earlier_booking(self, runtime: CallRuntime) -> None:
        ops = DeskOperations(runtime)
        ops.book_appointment(
            customer_name="Dan",
            phone="+12035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="tune up",
            day="tomorrow",
            window="two to five",
        )
        ops.dispatch_emergency(
            customer_name="Dan",
            phone="+12035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="water heater started leaking",
        )
        assert runtime.post_call_payload()["outcome"] == "dispatched"
