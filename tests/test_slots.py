"""Working memory for a booking.

Callers do not answer one question at a time. "Yeah hi, this is Dan over at
fourteen Elm in Wallingford and my furnace is making a noise" is a name, an
address, a town and a problem in one breath. Being asked for your own name
after you have just given it is one of the sharpest tells there is, so these
tests are mostly about *not asking twice*.
"""

from __future__ import annotations

import pytest

from stacy.agent import CallRuntime, DeskOperations
from stacy.agent.slots import (
    BookingSlots,
    foreign_number_note,
    heard_booking_details,
    normalize_phone,
    number_with_country,
    partial_number_note,
    phone_across_turns,
    phone_in_speech,
    readback_note,
    spelled_name,
    split_town,
    spoken_digits,
)
from stacy.desk import CalendarStore, MessageStore
from stacy.tenants import Tenant


@pytest.fixture
def runtime(tenant: Tenant, tmp_path, monday_9am) -> CallRuntime:
    return CallRuntime.build(
        tenant=tenant,
        call_id="c1",
        caller_id="+12035550147",
        calendar=CalendarStore(tmp_path / "cal"),
        messages=MessageStore(tmp_path / "msg"),
        now=monday_9am,
    )


@pytest.fixture
def ops(runtime: CallRuntime) -> DeskOperations:
    return DeskOperations(runtime)


class TestPhoneNormalization:
    @pytest.mark.parametrize(
        "raw",
        [
            "2035550147",
            "+12035550147",
            "1-203-555-0147",
            "(203) 555-0147",
            "two zero three five five five oh one four seven",
        ],
    )
    def test_every_way_a_number_arrives(self, raw: str) -> None:
        assert normalize_phone(raw) == "+12035550147"

    @pytest.mark.parametrize("raw", ["", "555-0147", "not a number", "12345678901234"])
    def test_things_that_are_not_numbers(self, raw: str) -> None:
        assert normalize_phone(raw) == ""

    @pytest.mark.parametrize(
        "raw",
        [
            "+91 99884 44520",
            "plus nine one nine nine eight eight four four four five two zero",
            "0091 9988444520",
        ],
    )
    def test_a_number_from_abroad_keeps_its_country_code(self, raw: str) -> None:
        assert normalize_phone(raw) == "+919988444520"

    def test_doubles_and_triples_are_spelled_out(self) -> None:
        # Exactly how the number was said on the live call that looped.
        said = "double 9, double 8, triple 4, five two zero"
        assert normalize_phone(said, caller_id="+919988444520") == "+919988444520"

    def test_the_callers_own_mobile_without_its_code_is_their_number(self) -> None:
        assert normalize_phone("99884 44520", caller_id="+919988444520") == "+919988444520"

    def test_a_different_ten_digit_number_is_still_north_american(self) -> None:
        assert normalize_phone("2035550147", caller_id="+919988444520") == "+12035550147"

    def test_a_leading_oh_is_a_word_not_a_digit(self) -> None:
        assert normalize_phone("oh, it's 203 555 0147") == "+12035550147"

    def test_read_back_of_a_number_from_abroad_keeps_its_code(self) -> None:
        assert spoken_digits("+919988444520") == (
            "plus nine-one, nine-nine-eight-eight-four, four-four-five-two-zero"
        )

    def test_read_back_is_spoken_in_groups(self) -> None:
        # The prompt reads digits back with [slow], country code left off.
        assert spoken_digits("+12035550147") == (
            "two-zero-three, five-five-five, zero-one-four-seven"
        )


class TestSlots:
    def test_three_fields_from_one_utterance(self) -> None:
        s = BookingSlots()
        newly = s.update(
            customer_name="Dan", street_address="14 Elm Street", town="Wallingford"
        )
        assert set(newly) == {"customer_name", "street_address", "town"}

    def test_missing_follows_the_prompts_order(self) -> None:
        s = BookingSlots(customer_name="Dan", street_address="14 Elm Street")
        assert s.missing() == ["phone", "town", "problem", "day", "window"]

    def test_a_blank_never_clears_a_slot(self) -> None:
        # A model passing street_address="" means "I didn't hear one", not
        # "forget the one you have".
        s = BookingSlots(street_address="14 Elm Street")
        s.update(street_address="")
        assert s.street_address == "14 Elm Street"

    def test_a_correction_overwrites(self) -> None:
        s = BookingSlots(street_address="14 Elm Street")
        assert s.update(street_address="16 Elm Street") == ["street_address"]
        assert s.street_address == "16 Elm Street"

    def test_restating_the_same_value_is_not_new(self) -> None:
        s = BookingSlots(customer_name="Dan")
        assert s.update(customer_name="Dan") == []

    def test_fuel_is_only_needed_for_a_furnace_or_boiler(self) -> None:
        # The prompt: "never ask gas-or-electric about an air conditioner, a
        # mini-split, or a heat pump."
        assert BookingSlots(system_type="gas furnace").needs_fuel()
        assert BookingSlots(system_type="Boiler").needs_fuel()
        assert not BookingSlots(system_type="mini-split").needs_fuel()
        assert not BookingSlots(system_type="heat pump").needs_fuel()
        assert not BookingSlots(system_type="air conditioner").needs_fuel()

    def test_fuel_is_asked_before_the_window_not_after_everything(self) -> None:
        # The script asks fuel right after the system type. Appending it to the
        # end had the agent asking "gas or electric?" after the window was set.
        s = BookingSlots(customer_name="Dan", system_type="furnace")
        assert s.missing() == ["phone", "street_address", "town", "problem", "fuel", "day", "window"]

    def test_fuel_appears_in_missing_only_when_it_applies(self) -> None:
        base = {
            "customer_name": "Dan",
            "phone": "+12035550147",
            "street_address": "14 Elm",
            "town": "Wallingford",
            "problem": "no heat",
            "day": "tomorrow",
            "window": "two to five",
        }
        assert BookingSlots(**base, system_type="furnace").missing() == ["fuel"]
        assert BookingSlots(**base, system_type="heat pump").missing() == []

    def test_complete_when_every_required_field_is_in(self) -> None:
        s = BookingSlots(
            customer_name="Dan",
            phone="+12035550147",
            street_address="14 Elm",
            town="Wallingford",
            problem="no heat",
            day="tomorrow",
            window="two to five",
        )
        assert s.complete
        assert "Dan" in s.readback() and "two to five" in s.readback()

    def test_an_unserved_town_is_rejected_and_named(self, tenant: Tenant) -> None:
        s = BookingSlots(town="Hartford")
        assert s.check_town(tenant) is False
        assert s.town == ""
        assert s.rejected_town == "Hartford"

    def test_a_served_town_passes_regardless_of_case(self, tenant: Tenant) -> None:
        s = BookingSlots(town="wallingford")
        assert s.check_town(tenant) is True
        assert s.town == "wallingford"


class TestANumberReadOutLoud:
    """Caught in code, because on a live call the model got the digits wrong four times.

    The inputs are the caller's turns from that call, as the SDK handed them over.
    """

    def test_doubles_and_triples_in_one_turn(self) -> None:
        said = "It's double 9, double 8, triple 4, 5 2 0."
        assert phone_in_speech(said, caller_id="+919988444520") == "+919988444520"

    def test_digits_split_by_pauses_within_one_turn(self) -> None:
        said = "It's 9 9\n8 8\n4 4 4 5 2 0."
        assert phone_in_speech(said, caller_id="+919988444520") == "+919988444520"

    @pytest.mark.parametrize(
        "said",
        [
            "I'm in coffee street 1 2 3, Wallingford.",
            "Can you do 11 to 2?",
            "It's the same. Double 9, double 8, triple 4, 5 to do.",
            "",
        ],
    )
    def test_anything_short_of_a_whole_number_is_left_alone(self, said: str) -> None:
        assert phone_in_speech(said, caller_id="+919988444520") == ""

    def test_the_note_gives_the_exact_read_back(self) -> None:
        note = readback_note("+919988444520")
        assert "plus nine-one, nine-nine-eight-eight-four, four-four-five-two-zero" in note
        assert "Not said by the caller" in note


class TestOnlyNumbersTheCallerSaid:
    """Replayed: nine digits said, a complete number of the model's own saved."""

    def test_a_number_made_up_from_part_of_one_is_refused(self, ops: DeskOperations) -> None:
        ops.runtime.record_user("And my number is double 9, double 8, triple 4, 5 0.")
        said = ops.record_details(phone="+919988444450")
        assert "isn't the number they said" in said
        assert ops.runtime.slots.phone == ""

    def test_the_number_they_read_out_is_saved(self, ops: DeskOperations) -> None:
        ops.runtime.record_user("My number is 2 0 3 5 5 5 0 1 9 9.")
        ops.record_details(phone="203-555-0199")
        assert ops.runtime.slots.phone == "+12035550199"

    def test_their_own_number_needs_no_digits_said(self, ops: DeskOperations) -> None:
        ops.runtime.record_user("Yes, this one.")
        ops.record_details(phone="+12035550147")
        assert ops.runtime.slots.phone == "+12035550147"

    def test_part_of_a_number_gets_a_note_and_an_address_does_not(self) -> None:
        assert "only part of a phone number" in partial_number_note(
            "And my number is double 9, double 8, triple 4, 5 0."
        )
        assert partial_number_note("I'm at 123 Coffey Street in Wallingford.") == ""


class TestANumberOverTwoTurns:
    """Only ever the number they are calling from, so nothing is guessed."""

    def test_a_tail_corrected_in_the_next_turn(self) -> None:
        # Exactly the live call: "5 0", then "5 2 0" as a turn of its own.
        assert phone_across_turns(
            "And my number is double 9, double 8, triple 4, 5 0.", "5 2 0.", caller_id="+919988444520"
        ) == "+919988444520"

    def test_a_number_split_by_a_pause(self) -> None:
        assert phone_across_turns("It's 99884", "44520", caller_id="+919988444520") == "+919988444520"

    def test_digits_that_are_not_their_own_number_are_left_alone(self) -> None:
        assert phone_across_turns("It's 20355", "50199", caller_id="+919988444520") == ""

    def test_no_caller_id_means_no_guessing(self) -> None:
        assert phone_across_turns("It's 99884", "44520", caller_id="Restricted") == ""


class TestDetailsHeardInSpeech:
    """The backstop for a model that speaks first and sometimes forgets to record."""

    @pytest.mark.parametrize(
        "said, expected",
        [
            ("Today works, two to five.", {"day": "today", "window": "two to five"}),
            ("You can come tomorrow, 11 to 2.", {"day": "tomorrow", "window": "eleven to two"}),
            ("Tomorrow.", {"day": "tomorrow"}),
            ("It's in Wallingford.", {"town": "Wallingford"}),
            ("Let's do today, actually no, tomorrow, 8 to 11.", {"day": "tomorrow", "window": "eight to eleven"}),
        ],
    )
    def test_what_a_caller_chose_is_heard(self, tenant: Tenant, said: str, expected: dict) -> None:
        assert heard_booking_details(said, tenant) == expected

    @pytest.mark.parametrize(
        "said",
        [
            "What time are you available today and tomorrow?",
            "Can you do today like 5 to 8?",
            "I think I need someone out sometime today because the house is getting warm",
            "My number is double 9 double 8 triple 4 5 2 0.",
        ],
    )
    def test_a_question_or_a_passing_mention_is_not_a_choice(self, tenant: Tenant, said: str) -> None:
        assert "day" not in heard_booking_details(said, tenant)
        assert "window" not in heard_booking_details(said, tenant)

    @pytest.mark.parametrize(
        "street, expected",
        [
            ("office suite 1 2 3 Wallingford", ("office suite 1 2 3", "Wallingford")),
            ("14 Elm Street in Wallingford.", ("14 Elm Street", "Wallingford")),
            ("14 Elm Street", ("14 Elm Street", "")),
        ],
    )
    def test_a_town_on_the_end_of_a_street_is_the_town(self, tenant: Tenant, street, expected) -> None:
        assert split_town(street, tenant) == expected


class TestRecordDetails:
    def test_everything_in_one_breath_is_kept(self, ops: DeskOperations) -> None:
        said = ops.record_details(
            customer_name="Dan",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="furnace is making a noise",
        )
        assert "Still need" in said
        # The four it already has must not be asked for again.
        for already in ("their name", "the street address", "the town", "what's going on"):
            assert already not in said

    def test_a_number_that_did_not_parse_is_said_not_dropped(self, ops: DeskOperations) -> None:
        # Reported live: dropped silently, the agent asked for the number five
        # times, then read back a number it had made up.
        said = ops.record_details(phone="double nine double eight")
        assert "didn't come through" in said
        assert "Read the number back" not in said

    def test_a_caller_from_abroad_reading_their_own_mobile_is_saved(
        self, tenant: Tenant, tmp_path, monday_9am
    ) -> None:
        rt = CallRuntime.build(
            tenant=tenant,
            call_id="c-in",
            caller_id="+919988444520",
            calendar=CalendarStore(tmp_path / "cal"),
            messages=MessageStore(tmp_path / "msg"),
            now=monday_9am,
        )
        said = DeskOperations(rt).record_details(phone="double 9 double 8 triple 4 5 2 0")
        assert rt.slots.phone == "+919988444520"
        assert "plus nine-one, nine-nine-eight-eight-four, four-four-five-two-zero" in said

    def test_it_names_only_what_is_outstanding(self, ops: DeskOperations) -> None:
        said = ops.record_details(customer_name="Dan", phone="2035550147")
        assert "the street address" in said and "the town" in said
        assert "their name" not in said

    def test_a_number_comes_back_ready_to_read_aloud(self, ops: DeskOperations) -> None:
        # The prompt: read a number back in the very next turn, digits spoken.
        said = ops.record_details(phone="two zero three five five five oh one four seven")
        assert "two-zero-three, five-five-five" in said

    def test_an_out_of_area_town_stops_the_booking_there(self, ops: DeskOperations) -> None:
        said = ops.record_details(customer_name="Dan", town="Hartford")
        assert "outside the service area" in said
        assert "Don't offer a callback" in said
        assert ops.runtime.slots.town == ""

    def test_a_non_fuel_system_says_not_to_ask(self, ops: DeskOperations) -> None:
        said = ops.record_details(system_type="mini-split")
        assert "Don't ask about fuel" in said

    def test_a_complete_set_asks_for_the_read_back(self, ops: DeskOperations) -> None:
        said = ops.record_details(
            customer_name="Dan",
            phone="2035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="no heat",
            day="tomorrow",
            window="two to five",
        )
        assert "That's everything" in said
        assert "Read it back" in said


class TestBookingFromSlots:
    def _fill(self, ops: DeskOperations) -> None:
        ops.record_details(
            customer_name="Dan",
            phone="2035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="no heat upstairs",
            day="tomorrow",
            window="two to five",
            sms_consent=True,
        )

    def test_booking_takes_no_arguments(self, ops: DeskOperations) -> None:
        self._fill(ops)
        said = ops.book_appointment()
        assert "14 Elm Street" in said
        assert ops.runtime.outcome == "booked"

    def test_an_incomplete_set_books_nothing_and_says_what_is_missing(
        self, ops: DeskOperations
    ) -> None:
        ops.record_details(customer_name="Dan", phone="2035550147")
        said = ops.book_appointment()
        assert "still need" in said
        assert ops.runtime.appointment is None

    def test_consent_recorded_earlier_survives_to_the_booking(
        self, ops: DeskOperations
    ) -> None:
        self._fill(ops)
        ops.book_appointment()
        assert ops.runtime.appointment is not None
        assert ops.runtime.appointment.sms_consent is True

    def test_declining_the_text_is_a_real_answer(self, ops: DeskOperations) -> None:
        self._fill(ops)
        ops.record_details(sms_consent=False)
        ops.book_appointment()
        assert ops.runtime.appointment.sms_consent is False

    def test_front_loading_every_field_into_the_booking_still_works(
        self, ops: DeskOperations
    ) -> None:
        # Two interaction styles, one destination.
        said = ops.book_appointment(
            customer_name="Ruth",
            phone="2035550188",
            street_address="9 Oak Lane",
            town="Meriden",
            problem="no cooling",
            day="tomorrow",
            window="eleven to two",
        )
        assert "9 Oak Lane" in said
        assert ops.runtime.outcome == "booked"

    def test_a_returning_caller_does_not_get_asked_for_their_address(
        self, ops: DeskOperations
    ) -> None:
        self._fill(ops)
        ops.book_appointment()

        second = DeskOperations(
            CallRuntime.build(
                tenant=ops.runtime.tenant,
                call_id="c2",
                caller_id="+12035550147",
                calendar=ops.runtime.calendar,
                messages=ops.runtime.messages,
            )
        )
        assert "Confirm the address" in second.lookup_customer()
        assert second.runtime.slots.street_address == "14 Elm Street"
        assert "the street address" not in second.record_details(problem="no heat again")


class TestEmergencyUsesTheSameMemory:
    def test_details_collected_before_the_emergency_are_kept(
        self, ops: DeskOperations
    ) -> None:
        # An emergency surfacing mid-booking must not restart the intake on the
        # worst possible call.
        ops.record_details(
            customer_name="Ruth",
            phone="2035550188",
            street_address="9 Oak Lane",
            town="Meriden",
        )
        said = ops.dispatch_emergency(problem="water heater is leaking")
        assert "9 Oak Lane" in said
        assert ops.runtime.outcome == "dispatched"

    def test_it_says_what_it_still_needs(self, ops: DeskOperations) -> None:
        said = ops.dispatch_emergency(problem="water heater is leaking")
        assert "Before I can send anyone" in said
        assert ops.runtime.appointment is None


class TestUnsuppliedToolArguments:
    """What the model sends when the caller did not say something.

    livekit-agents renders every tool parameter into the schema's `required`
    list whatever its Python default, and makes the unsupplied case expressible
    by widening the type to ["string", "null"]. So the model must send all ten
    keys on every call, and needs a legal way to say "not mentioned".

    With `str = ""` defaults there was no such value, and against the real API
    the model filled unknown slots with fragments of its own tool-call
    scaffolding -- a phone number of "</parameter>\n" went into the booking as
    the caller's callback number. Null is the honest answer.
    """

    def test_null_and_blank_both_mean_not_mentioned(self) -> None:
        from stacy.agent.tools import _said

        assert _said(None) == ""
        assert _said("") == ""
        assert _said("   ") == ""

    def test_a_real_value_survives_untouched(self) -> None:
        from stacy.agent.tools import _said

        assert _said("14 Elm Street") == "14 Elm Street"
        assert _said("  Wallingford  ") == "Wallingford"

    def test_an_unmentioned_field_does_not_clear_what_is_known(
        self, ops: DeskOperations
    ) -> None:
        from stacy.agent.tools import _said

        ops.record_details(street_address="14 Elm Street")
        # The next turn: the model must send street_address, and sends null.
        ops.record_details(problem="no heat", street_address=_said(None))
        assert ops.runtime.slots.street_address == "14 Elm Street"


class TestASpelledName:
    """Live: "It's n a n a k, Nak." was booked as Nak."""

    @pytest.mark.parametrize(
        ("said", "name"),
        [("It's n a n a k, Nak.", "Nanak"), ("N-A-N-A-K", "Nanak"), ("my name is N. A. N. A. K.", "Nanak")],
    )
    def test_the_letters_make_the_name(self, said: str, name: str) -> None:
        assert spelled_name(said) == name

    @pytest.mark.parametrize("said", ["My A C is not cooling.", "It's Dan.", "I'm in Wallingford."])
    def test_ordinary_speech_spells_nothing(self, said: str) -> None:
        assert spelled_name(said) == ""


class TestANumberThatIsntAmerican:
    """Live, caller ID withheld: "double 9, double 8, triple 4, 5 2 0" was saved as +1 998 844 4520."""

    def test_ten_digits_that_cant_be_a_us_number_are_not_saved_as_one(self) -> None:
        assert normalize_phone("double 9, double 8, triple 4, 5 2 0") == ""
        assert normalize_phone("2035550147") == "+12035550147"
        assert normalize_phone("860-398-9358") == "+18603989358"

    def test_the_caller_is_asked_which_country(self) -> None:
        assert "which country" in foreign_number_note("double 9, double 8, triple 4, 5 2 0")
        assert foreign_number_note("2035550147") == ""
        assert foreign_number_note("+91 99884 44520") == ""

    @pytest.mark.parametrize("said", ["India.", "It's an Indian number", "from india"])
    def test_naming_the_country_completes_it(self, said: str) -> None:
        assert number_with_country("double 9, double 8, triple 4, 5 2 0", said) == "+919988444520"

    def test_a_country_after_a_us_number_changes_nothing(self) -> None:
        assert number_with_country("203 555 0147", "India") == ""

    def test_the_booking_tool_says_why_it_saved_nothing(self, ops: DeskOperations) -> None:
        said = ops.record_details(phone="9988444520")
        assert "isn't a US number" in said
        assert ops.runtime.slots.phone == ""
