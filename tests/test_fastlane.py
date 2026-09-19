"""The fast lane: turns with one right reply, answered without the model.

Every reply here is one the prompt already scripts. What matters as much is
what is *not* answered: anything with a question in it, anything extra, the
demo close, a line asking two things -- all of it goes to the model.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from stacy.agent import CallRuntime, DeskOperations
from stacy.agent.fastlane import SMS_CONSENT_LINE, desk_reply, name_spelled
from stacy.desk import CalendarStore, MessageStore
from stacy.metrics.live import CallLatency, render_report
from stacy.tenants import Tenant

OWN_NUMBER_OFFER = (
    "Is [slow] two-zero-three, five-five-five, zero-one-four-seven the best number to reach you?"
)


def _runtime(tenant: Tenant, tmp_path, now: datetime, caller_id: str = "+12035550147") -> CallRuntime:
    return CallRuntime.build(
        tenant=tenant,
        call_id="call-1",
        caller_id=caller_id,
        calendar=CalendarStore(tmp_path / "cal"),
        messages=MessageStore(tmp_path / "msg"),
        now=now,
    )


@pytest.fixture
def runtime(tenant: Tenant, tmp_path, monday_9am: datetime) -> CallRuntime:
    """A call where the caller has described the problem and named their town."""
    rt = _runtime(tenant, tmp_path, monday_9am)
    rt.slots.update(problem="furnace not heating", town="Wallingford")
    return rt


def _turn(runtime: CallRuntime, agent: str, caller: str):
    runtime.record_agent(agent)
    runtime.record_user(caller)
    return desk_reply(runtime, caller)


class TestAPhoneNumber:
    def test_a_number_read_out_is_read_back_with_the_consent_line(self, runtime) -> None:
        runtime.slots.update(customer_name="Nanak")
        reply = _turn(runtime, "Thanks, Nanak. What's the best number to reach you?", "It's 2 0 3 5 5 5 0 1 9 9.")
        assert reply is not None
        assert reply.speech.startswith("That's [slow] two-zero-three, five-five-five, zero-one-nine-nine.")
        assert SMS_CONSENT_LINE in reply.speech
        assert reply.record == {"phone": "+12035550199"}

    def test_a_number_from_abroad_keeps_its_country_code(self, tenant, tmp_path, monday_9am) -> None:
        rt = _runtime(tenant, tmp_path, monday_9am, caller_id="+919988444520")
        rt.slots.update(problem="no heat", town="Meriden", customer_name="Nanak")
        reply = _turn(rt, "What's the best number to reach you?", "99884 44520")
        assert reply is not None
        assert "plus nine-one" in reply.speech
        assert reply.record == {"phone": "+919988444520"}

    @pytest.mark.parametrize(
        "caller",
        [
            "It's 203 555 0199, do you do boilers?",
            "203 555 0199 and the furnace is leaking too",
            "203 555",
        ],
    )
    def test_anything_more_or_less_than_a_number_goes_to_the_model(self, runtime, caller) -> None:
        assert _turn(runtime, "What's the best number to reach you?", caller) is None

    def test_yes_to_their_own_number_saves_it(self, runtime) -> None:
        reply = _turn(runtime, OWN_NUMBER_OFFER, "Yeah, that works.")
        assert reply is not None
        assert reply.record == {"phone": "+12035550147"}
        assert SMS_CONSENT_LINE in reply.speech

    def test_no_to_their_own_number_asks_for_another(self, runtime) -> None:
        reply = _turn(runtime, OWN_NUMBER_OFFER, "No.")
        assert reply is not None
        assert reply.speech == "No problem. What's the best number to reach you?"
        assert reply.record == {}

    def test_a_different_number_instead_of_their_own_is_read_back(self, runtime) -> None:
        reply = _turn(runtime, OWN_NUMBER_OFFER, "No, it's 203 555 0199.")
        assert reply is not None
        assert reply.record == {"phone": "+12035550199"}


class TestTextConsent:
    @pytest.fixture
    def asked(self, runtime) -> CallRuntime:
        runtime.slots.update(customer_name="Nanak", phone="+12035550199")
        runtime.record_agent(f"That's [slow] two-zero-three, five-five-five, zero-one-nine-nine. {SMS_CONSENT_LINE}")
        return runtime

    def test_yes_is_recorded_and_the_street_asked_for(self, asked) -> None:
        asked.record_user("Yes, that's fine.")
        reply = desk_reply(asked, "Yes, that's fine.")
        assert reply is not None
        assert reply.record == {"sms_consent": True}
        assert reply.speech.endswith("And what's the street address there?")

    def test_no_is_an_answer_too(self, asked) -> None:
        asked.record_user("No thanks.")
        reply = desk_reply(asked, "No thanks.")
        assert reply is not None
        assert reply.record == {"sms_consent": False}
        assert reply.speech.startswith("No problem.")

    def test_a_question_about_the_texts_goes_to_the_model(self, asked) -> None:
        asked.record_user("What kind of texts?")
        assert desk_reply(asked, "What kind of texts?") is None

    def test_a_street_already_given_is_never_asked_for(self, asked) -> None:
        asked.record_user("I'm at 14 Elm Street.")
        asked.record_user("Yes.")
        assert desk_reply(asked, "Yes.") is None


class TestANameOrAddress:
    def test_a_name_is_thanked_and_their_own_number_offered(self, runtime) -> None:
        reply = _turn(runtime, "Ugh, okay. Can I get your name?", "It's Nanak.")
        assert reply is not None
        assert reply.speech == f"Thanks, Nanak. {OWN_NUMBER_OFFER}"
        assert reply.record == {"customer_name": "Nanak"}

    @pytest.mark.parametrize("caller", ["Why do you need it", "Hold on", "N A", "Wallingford"])
    def test_what_isnt_a_name_goes_to_the_model(self, runtime, caller) -> None:
        assert _turn(runtime, "Can I get your name?", caller) is None

    def test_a_street_without_a_town_asks_for_the_town(self, runtime) -> None:
        runtime.slots.town = ""
        runtime.slots.update(customer_name="Nanak", phone="+12035550199", sms_consent=True)
        reply = _turn(runtime, "And what's the street address there?", "123 Coffey Street.")
        assert reply is not None
        assert reply.record == {"street_address": "123 Coffey Street"}
        assert reply.speech == "Perfect. And what town is that in?"

    def test_a_street_and_served_town_are_both_recorded(self, runtime) -> None:
        runtime.slots.town = ""
        reply = _turn(runtime, "And what's your address?", "It's 1 23 Coffey Street in Wallingford.")
        assert reply is not None
        assert reply.record == {"street_address": "123 Coffey Street", "town": "Wallingford"}
        assert reply.speech.endswith("And can I get your name?")

    def test_a_town_we_dont_serve_goes_to_the_model(self, runtime) -> None:
        runtime.slots.town = ""
        assert _turn(runtime, "And what's your address?", "12 Main Street in New Haven") is None

    def test_the_window_and_the_read_back_stay_with_the_model(self, runtime) -> None:
        runtime.slots.update(customer_name="Nanak", phone="+12035550199", sms_consent=True)
        assert _turn(runtime, "And what's the street address there?", "123 Coffey Street.") is None


class TestLeftToTheModel:
    def test_nothing_before_a_booking_is_under_way(self, tenant, tmp_path, monday_9am) -> None:
        rt = _runtime(tenant, tmp_path, monday_9am)
        assert _turn(rt, "And who am I talking to?", "Nanak.") is None

    def test_the_demo_close_takes_its_own_numbers(self, runtime) -> None:
        line = "Want me to have Mister Miles call you at [slow] two-zero-three, five-five-five, zero-one-four-seven?"
        assert _turn(runtime, line, "Yes.") is None

    def test_a_line_asking_two_things(self, runtime) -> None:
        assert _turn(runtime, "What's your name? And the best number to reach you?", "Nanak.") is None

    def test_anything_once_the_booking_is_made(self, runtime) -> None:
        runtime.appointments.append(object())  # type: ignore[arg-type]
        assert _turn(runtime, "Can I get your name?", "Nanak.") is None

    def test_a_line_already_said_in_this_call(self, runtime) -> None:
        runtime.record_agent("No problem. What's the best number to reach you?")
        assert _turn(runtime, OWN_NUMBER_OFFER, "No.") is None

    def test_after_the_safety_override(self, runtime) -> None:
        runtime.record_user("I smell gas.")
        assert _turn(runtime, "Can I get your name?", "Nanak.") is None


def test_a_desk_reply_is_marked_in_the_latency_record() -> None:
    latency = CallLatency(call_id="call-1")
    latency.on_user({"stopped_speaking_at": 10.0})
    latency.on_desk_reply()
    record = latency.on_reply({"started_speaking_at": 10.6})
    assert record is not None and record["desk"] is True
    latency.on_user({"stopped_speaking_at": 20.0})
    assert latency.on_reply({"started_speaking_at": 21.0})["desk"] is False
    assert latency.summary()["desk_replies"] == 1.0


def test_the_report_splits_desk_replies_from_the_model() -> None:
    records = [
        {"call": "a", "turn": 0, "wait_ms": 700.0, "tools": [], "desk": True},
        {"call": "a", "turn": 1, "wait_ms": 1500.0, "tools": [], "desk": False},
    ]
    assert "700ms answered by the desk (1), 1500ms by the model (1)" in render_report(records)


class TestTheWindowAndTheBooking:
    OFFER = "Tomorrow I've got eight to eleven, eleven to two, or two to five. Which works?"

    @pytest.fixture
    def ready(self, runtime) -> CallRuntime:
        runtime.slots.update(
            customer_name="Nanak",
            phone="+12035550199",
            sms_consent=True,
            street_address="123 Coffey Street",
            problem="Furnace not heating",
        )
        return runtime

    def test_a_window_chosen_is_read_back(self, ready) -> None:
        reply = _turn(ready, self.OFFER, "Eleven to two, please.")
        assert reply is not None
        assert reply.record == {"day": "tomorrow", "window": "eleven to two"}
        assert reply.speech == (
            "So that's Nanak, 123 Coffey Street in Wallingford, furnace not heating, "
            "tomorrow, eleven to two. Does that all sound right?"
        )

    def test_hours_said_as_numbers_pick_the_window(self, ready) -> None:
        reply = _turn(ready, self.OFFER, "Yeah. Just do it 11 to 2.")
        assert reply is not None and reply.record["window"] == "eleven to two"

    def test_yes_to_the_one_window_offered(self, ready) -> None:
        reply = _turn(ready, "Eight to eleven tomorrow?", "Yeah, sure.")
        assert reply is not None
        assert reply.record == {"day": "tomorrow", "window": "eight to eleven"}

    @pytest.mark.parametrize(
        "caller", ["Can you do today, like 5 to 8?", "Two to five, but call first", "Eight to eleven."]
    )
    def test_anything_else_about_a_window_goes_to_the_model(self, ready, caller) -> None:
        assert _turn(ready, "Tomorrow I've got eleven to two, or two to five. Which works?", caller) is None

    def test_an_offer_naming_two_days_needs_the_caller_to_pick_one(self, ready) -> None:
        line = "Today I've got two to five, or tomorrow eight to eleven. Which works?"
        assert _turn(ready, line, "Two to five.") is None

    def test_a_window_the_calendar_doesnt_have_goes_to_the_model(self, ready) -> None:
        # Offered by mistake: the office is closed on Sundays.
        assert _turn(ready, "Sunday I've got eight to eleven. Does that work?", "Eight to eleven.") is None

    def test_a_town_already_said_is_never_asked_for(self, ready) -> None:
        ready.slots.town = ""
        ready.record_user("I live in office suite 1 2 3 Wallingford.")
        line = "Today I've got eleven to two, or two to five. Which works?"
        assert _turn(ready, line, "Two to five.") is None

    def test_yes_to_the_read_back_books_it(self, ready) -> None:
        first = _turn(ready, self.OFFER, "Eleven to two.")
        assert first is not None
        ready.slots.update(**first.record)
        reply = _turn(ready, first.speech, "Yeah, perfect.")
        assert reply is not None and reply.action == "book"
        assert reply.speech.startswith("You're all set. The office will call in the morning")
        # The booking tool's own read-back check accepts the desk's read-back.
        assert DeskOperations(ready).book_appointment().startswith("Booked")

    def test_a_visit_today_gets_the_same_day_line(self, ready) -> None:
        first = _turn(ready, "Today I've got eleven to two, or two to five. Which works?", "Two to five.")
        assert first is not None and "today, two to five" in first.speech
        ready.slots.update(**first.record)
        reply = _turn(ready, first.speech, "Yes.")
        assert reply is not None
        assert "thirty minutes before they head over" in reply.speech

    def test_a_read_back_they_correct_goes_to_the_model(self, ready) -> None:
        first = _turn(ready, self.OFFER, "Eleven to two.")
        assert first is not None
        ready.slots.update(**first.record)
        assert _turn(ready, first.speech, "No, that's not right, it's 124.") is None


class TestFromTheLiveCalls:
    """Turns from two live calls that went to the model and shouldn't have."""

    def test_a_number_read_out_is_read_back_whatever_the_question(self, tenant, tmp_path, monday_9am) -> None:
        rt = _runtime(tenant, tmp_path, monday_9am, caller_id="")
        rt.slots.update(problem="Heater not heating", town="Wallingford", customer_name="Lanak", street_address="123")
        reply = _turn(rt, "Sorry, say that number once more for me, slow?", "2 0 3 5 5 5 0 1 9 9.")
        assert reply is not None
        assert reply.record == {"phone": "+12035550199"}
        assert reply.speech.startswith("That's [slow] two-zero-three, five-five-five, zero-one-nine-nine.")

    def test_ten_digits_that_arent_a_us_number_go_to_the_model(self, tenant, tmp_path, monday_9am) -> None:
        # Caller ID withheld: +1 998 is no US area code, so the model asks which country.
        rt = _runtime(tenant, tmp_path, monday_9am, caller_id="")
        rt.slots.update(problem="Heater not heating", town="Wallingford", customer_name="Lanak")
        assert _turn(rt, "What's the best number to reach you?", "Double 9, double 8, triple 4, 5 2 0.") is None

    @pytest.mark.parametrize("caller", ["This 1 only.", "It's the same.", "this number", "The one I'm calling from."])
    def test_this_one_means_the_number_they_called_from(self, tenant, tmp_path, monday_9am, caller) -> None:
        rt = _runtime(tenant, tmp_path, monday_9am, caller_id="+919988444520")
        rt.slots.update(problem="AC not cooling", town="Wallingford", customer_name="Nanak")
        reply = _turn(rt, "Thanks, Nanak. What's the best number for us to reach you at?", caller)
        assert reply is not None
        assert reply.record == {"phone": "+919988444520"}
        assert reply.speech.startswith("That's [slow] plus nine-one")

    def test_this_one_without_a_caller_id_goes_to_the_model(self, tenant, tmp_path, monday_9am) -> None:
        rt = _runtime(tenant, tmp_path, monday_9am, caller_id="")
        rt.slots.update(problem="AC not cooling", town="Wallingford", customer_name="Nanak")
        assert _turn(rt, "What's the best number to reach you?", "This 1 only.") is None

    def test_a_name_said_with_a_rising_voice_is_still_a_name(self, runtime) -> None:
        reply = _turn(runtime, "Perfect — that's in our service area. And who am I talking to?", "You're talking with Nanak?")
        assert reply is not None and reply.record == {"customer_name": "Nanak"}

    def test_a_real_question_is_not_a_name(self, runtime) -> None:
        assert _turn(runtime, "And who am I talking to?", "Who is asking?") is None

    def test_a_street_name_given_before_the_number(self, runtime) -> None:
        runtime.slots.update(customer_name="Lanak", street_address="123")
        reply = _turn(runtime, "Lanak — got it. And the street name at that address?", "It the Coffee Street 1 2 3 Wallingford.")
        assert reply is not None
        assert reply.record == {"street_address": "123 Coffee Street", "town": "Wallingford"}
        assert reply.speech.endswith(OWN_NUMBER_OFFER)

    def test_a_read_back_split_around_a_tool_call_is_still_a_read_back(self, runtime) -> None:
        runtime.slots.update(
            customer_name="Lanak", phone="+12035550199", sms_consent=True, street_address="123 Coffee Street",
            problem="Heater not heating properly, tried turning it off and back on", day="tomorrow", window="eleven to two",
        )
        runtime.record_agent(
            "Perfect. So to confirm: Lanak, 123 Coffee Street in Wallingford, your heater's not heating "
            "properly, and we'll see you tomorrow between eleven and two."
        )
        reply = _turn(runtime, "Does that sound right?", "Yeah. Perfect.")
        assert reply is not None and reply.action == "book"

    def test_which_works_better_is_answered_with_better(self, runtime) -> None:
        runtime.slots.update(
            customer_name="Lanak", phone="+12035550199", sms_consent=True, street_address="123 Coffee Street",
            problem="Heater not heating properly, tried turning it off and back on",
        )
        line = "Tomorrow morning we've got eight to eleven, or eleven to two — which works better for you?"
        reply = _turn(runtime, line, "11 to 2 is better.")
        assert reply is not None
        assert reply.record == {"day": "tomorrow", "window": "eleven to two"}
        assert "heater not heating properly, tomorrow, eleven to two" in reply.speech


class TestTheThirdLiveCall:
    def test_a_spelled_name_is_the_name(self, runtime) -> None:
        reply = _turn(runtime, "Perfect. And can I get your name?", "It's n a n a k, Nak.")
        assert reply is not None
        assert reply.record == {"customer_name": "Nanak"}
        assert reply.speech.startswith("Thanks, Nanak.")

    def test_a_spelling_with_something_else_in_it_goes_to_the_model(self, runtime) -> None:
        assert _turn(runtime, "Can I get your name?", "N a n a k, and my AC is broken") is None

    def test_yes_to_the_read_back_after_a_hold(self, runtime) -> None:
        runtime.slots.update(
            customer_name="Nanak", phone="+12035550199", sms_consent=True, street_address="123 Coffee Street",
            day="today", window="two to five",
        )
        runtime.record_agent(
            "Alright, Nanak — just to confirm: one-two-three Coffee Street in Wallingford, your AC's not "
            "cooling properly, and we'll see you today between two and five. Does that sound right?"
        )
        runtime.record_user("Yeah. But the")
        reply = _turn(runtime, "Sure, take your time.", "Yeah. It works.")
        assert reply is not None and reply.action == "book"

    def test_just_book_the_window(self, runtime) -> None:
        runtime.slots.update(customer_name="Nanak", phone="+12035550199", sms_consent=True, street_address="123 Coffee Street")
        line = "Today we've got three windows open: eight to eleven, eleven to two, or two to five. Which one works for you?"
        reply = _turn(runtime, line, "Just book the 2 to 5 1.")
        assert reply is not None and reply.record == {"day": "today", "window": "two to five"}

    def test_yes_to_texts_with_a_stray_word(self, runtime) -> None:
        runtime.slots.update(customer_name="Nanak", phone="+12035550199")
        line = f"That's [slow] two-zero-three, five-five-five, zero-one-nine-nine. {SMS_CONSENT_LINE}"
        reply = _turn(runtime, line, "Yeah. Sure. Book.")
        assert reply is not None and reply.record == {"sms_consent": True}


def test_a_desk_reply_flagged_before_the_callers_message_is_logged() -> None:
    latency = CallLatency(call_id="call-1")
    latency.on_desk_reply()
    latency.on_user({"stopped_speaking_at": 10.0})
    assert latency.on_reply({"started_speaking_at": 10.55})["desk"] is True


def test_yes_to_a_read_back_said_without_a_question(runtime) -> None:
    runtime.slots.update(
        customer_name="Nanak", phone="+12035550199", sms_consent=True, street_address="123 Coffee Street",
        day="today", window="two to five",
    )
    line = "So that's Nanak, 123 Coffee Street in Wallingford, furnace not heating, today, two to five."
    reply = _turn(runtime, line, "Yeah. It works.")
    assert reply is not None and reply.action == "book"


class TestTheFirstCallAt300ms:
    @pytest.fixture
    def ready(self, runtime) -> CallRuntime:
        runtime.slots.update(
            customer_name="Nanak", phone="+919988444520", sms_consent=True, street_address="123 Coffee Street"
        )
        return runtime

    def test_a_spelled_name_after_a_lead_in(self, runtime) -> None:
        reply = _turn(runtime, "A week — that's a while. And who am I talking to?", "You're talking with Nana n a n a k.")
        assert reply is not None and reply.record == {"customer_name": "Nanak"}

    def test_the_model_hears_the_spelling_even_without_the_word_name(self, runtime) -> None:
        runtime.record_agent("A week — that's a while. And who am I talking to?")
        assert name_spelled(runtime, "You're talking with Nana n a n a k, and my AC is out") == "Nanak"

    def test_a_read_back_using_only_the_first_name_is_confirmed(self, ready) -> None:
        ready.slots.update(customer_name="Nana Nanak", day="tomorrow", window="eleven to two")
        line = (
            "Perfect. Let me get that booked for you. So here's what I've got for you, Nana: tomorrow, "
            "eleven to two, at one-twenty-three Coffee Street in Wallingford, for your air conditioner. "
            "Does that sound right?"
        )
        reply = _turn(ready, line, "Yeah. Perfect.")
        assert reply is not None and reply.action == "book"

    def test_a_day_and_window_named_outright_are_read_back(self, ready) -> None:
        line = "Got it — one-two-three Coffee Street in Wallingford. And you're looking to get this fixed today, or tomorrow?"
        reply = _turn(ready, line, "Just do it tomorrow morning, 11 to 2.")
        assert reply is not None
        assert reply.record == {"day": "tomorrow", "window": "eleven to two"}
        assert reply.speech.startswith("So that's Nanak,")

    def test_a_window_without_its_day_still_needs_an_offer(self, ready) -> None:
        assert _turn(ready, "And you're looking to get this fixed today, or tomorrow?", "11 to 2.") is None

    def test_yes_to_a_window_already_recorded_is_not_read_back_again(self, ready) -> None:
        ready.slots.update(day="tomorrow", window="eleven to two")
        assert _turn(ready, "Eleven to two tomorrow?", "Yes.") is None

    def test_yes_to_texts_with_a_garbled_tail(self, runtime) -> None:
        runtime.slots.update(customer_name="Nanak", phone="+919988444520")
        line = f"That's [slow] plus nine-one, nine-nine-eight-eight-four, four-four-five-two-zero. {SMS_CONSENT_LINE}"
        reply = _turn(runtime, line, "Yeah. Perfect. You then.")
        assert reply is not None and reply.record == {"sms_consent": True}


def test_the_booking_check_accepts_a_first_name_read_back(runtime) -> None:
    runtime.slots.update(
        customer_name="Nana Nanak", phone="+919988444520", sms_consent=True, street_address="123 Coffee Street",
        day="tomorrow", window="eleven to two",
    )
    runtime.record_agent("So here's what I've got for you, Nana: tomorrow, eleven to two, at 123 Coffee Street. Sound right?")
    runtime.record_user("Yeah. Perfect.")
    assert DeskOperations(runtime).book_appointment().startswith("Booked")


def test_yes_to_a_long_read_back_the_model_never_recorded_is_not_a_window_pick(runtime) -> None:
    runtime.slots.update(phone="+919988444520", street_address="123 Coffee Street")
    line = (
        "Perfect — eleven to two today works. Before I lock that in, let me read this back: Nana, "
        "one-twenty-three Coffee Street in Wallingford, heater not putting out heat, today eleven to two. "
        "Does that sound right?"
    )
    assert _turn(runtime, line, "Yeah. It's perfect.") is None
