"""Corrections that arrive after the booking is already made.

A caller who says "wait, sorry, it's forty Elm, not fourteen" after hearing
"you're all set" used to get a *second* appointment. The wrong one stayed on
the board, a technician could drive to it, and it held a slot nobody was
coming to. Every test here is that call, or one of the calls that look like it
and are not -- someone adding a second job, which must never be swallowed.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime

import pytest

from stacy.agent import CallRuntime, DeskOperations
from stacy.desk import CalendarStore, MessageStore
from stacy.tenants import Tenant


@pytest.fixture
def calendar(tmp_path) -> CalendarStore:
    return CalendarStore(tmp_path / "cal")


@pytest.fixture
def runtime(tenant: Tenant, calendar: CalendarStore, tmp_path, monday_9am: datetime) -> CallRuntime:
    return CallRuntime.build(
        tenant=tenant,
        call_id="call-1",
        caller_id="+12035550147",
        calendar=calendar,
        messages=MessageStore(tmp_path / "msg"),
        now=monday_9am,
    )


def _book(runtime: CallRuntime, **over: str) -> str:
    """Get one visit on the books, the way a real call does: read back, then yes."""
    fields = {
        "customer_name": "Dan", "phone": "+12035550147", "street_address": "14 Elm Street",
        "town": "Wallingford", "problem": "no heat", "day": "tomorrow",
        "window": "eleven to two",
    }
    fields.update(over)
    runtime.slots.update(**fields)
    runtime.record_agent(
        f"So that's {fields['customer_name']}, {fields['street_address']} in {fields['town']}, "
        f"{fields['problem']}, {fields['day']}, {fields['window']}. Right?"
    )
    runtime.record_user("Yes.")
    return DeskOperations(runtime).book_appointment()


def _confirm(runtime: CallRuntime) -> str:
    """Read the details back as they now stand, and take the yes.

    Built from the slots rather than hardcoded, because the whole point of
    the guard being tested is that the read-back has to carry the corrected
    value -- a generic "is that right?" must not be able to move a visit.
    """
    s = runtime.slots
    runtime.record_agent(
        f"So that's {s.customer_name}, {s.street_address} in {s.town}, {s.problem}, "
        f"{s.day}, {s.window}. Right?"
    )
    runtime.record_user("Yes.")
    return DeskOperations(runtime).book_appointment()


class TestACorrectionReplacesTheVisit:
    def test_the_wrong_address_does_not_stay_on_the_board(self, runtime, tenant, calendar) -> None:
        assert _book(runtime).startswith("Booked:")
        runtime.record_user("Wait, sorry — it's 40 Elm Street, not 14.")
        DeskOperations(runtime).record_details(street_address="40 Elm Street")
        assert _confirm(runtime).startswith("Changed, not booked twice:")

        live = calendar._live(tenant.tenant_id)
        assert [a.street_address for a in live] == ["40 Elm Street"]
        assert len(runtime.appointments) == 1
        assert runtime.appointments[0].street_address == "40 Elm Street"

    def test_it_does_not_hold_two_slots(self, runtime, tenant, calendar) -> None:
        _book(runtime)
        runtime.record_user("Sorry, that's wrong — it's 40 Elm Street.")
        DeskOperations(runtime).record_details(street_address="40 Elm Street")
        _confirm(runtime)
        day = Date.fromisoformat(runtime.appointments[0].day)
        assert calendar.booked_count(tenant.tenant_id, day, "eleven to two") == 1

    def test_both_versions_stay_on_disk(self, runtime, tenant, calendar) -> None:
        # "I told her forty, not fourteen" has to be answerable afterwards.
        _book(runtime)
        runtime.record_user("No, the address is 40 Elm Street.")
        DeskOperations(runtime).record_details(street_address="40 Elm Street")
        _confirm(runtime)
        rows = calendar._load(tenant.tenant_id)
        assert [a.street_address for a in rows] == ["14 Elm Street", "40 Elm Street"]
        assert rows[1].supersedes == rows[0].appointment_id

    def test_the_history_shows_the_corrected_visit(self, runtime, tenant, calendar) -> None:
        _book(runtime)
        runtime.record_user("Sorry, it's 40 Elm Street, I mixed up the number.")
        DeskOperations(runtime).record_details(street_address="40 Elm Street")
        _confirm(runtime)
        seen = calendar.history_for(tenant.tenant_id, "+12035550147")
        assert [a.street_address for a in seen] == ["40 Elm Street"]

    def test_a_reschedule_moves_the_visit_instead_of_adding_one(self, runtime) -> None:
        _book(runtime)
        runtime.record_user("Actually, can you make it Thursday instead?")
        runtime.slots.update(day="thursday")
        assert _confirm(runtime).startswith("Changed, not booked twice:")
        assert len(runtime.appointments) == 1
        assert runtime.appointments[0].day == "2026-09-17"


class TestASecondJobIsNotACorrection:
    """The failure that would be worse: silently cancelling a real visit."""

    def test_while_youre_here_books_a_second_visit(self, runtime) -> None:
        _book(runtime)
        runtime.record_user("Can you also look at the AC while you're here?")
        runtime.slots.update(problem="ac making a noise", window="two to five")
        assert _confirm(runtime).startswith("Booked:")
        assert len(runtime.appointments) == 2

    def test_another_visit_for_a_second_address(self, runtime) -> None:
        _book(runtime)
        runtime.record_user("And another visit for my mother's place at 12 Oak Street.")
        runtime.slots.update(street_address="12 Oak Street", day="thursday")
        assert _confirm(runtime).startswith("Booked:")
        assert len(runtime.appointments) == 2

    def test_also_beats_actually_in_the_same_sentence(self, runtime) -> None:
        # The sentence that needs both halves of the rule: it opens with a
        # word that means a fix and asks for a second job anyway. Reading it
        # as a correction would quietly cancel the heating visit.
        _book(runtime)
        runtime.record_user("Actually, can you also take a look at the AC while you're here?")
        runtime.slots.update(problem="ac making a noise", window="two to five")
        assert _confirm(runtime).startswith("Booked:")
        assert len(runtime.appointments) == 2
        assert {a.problem for a in runtime.appointments} == {"no heat", "ac making a noise"}

    def test_a_changed_detail_alone_is_not_enough(self, runtime) -> None:
        # Deliberate: with no correction said, the conservative read is a second
        # visit. Replacing a booking on a silent slot change is the one mistake
        # the caller would not hear and could not undo.
        _book(runtime)
        runtime.slots.update(day="thursday", problem="AC not cooling")
        assert _confirm(runtime).startswith("Booked:")
        assert len(runtime.appointments) == 2


class TestACorrectionIsStillABooking:
    def test_it_waits_for_the_read_back(self, runtime) -> None:
        _book(runtime)
        runtime.record_user("Sorry, it's 40 Elm Street.")
        DeskOperations(runtime).record_details(street_address="40 Elm Street")
        said = DeskOperations(runtime).book_appointment()
        assert said.startswith("Not yet:") and "old visit stands" in said

    def test_a_no_to_the_read_back_moves_nothing(self, runtime) -> None:
        _book(runtime)
        runtime.record_user("Sorry, it's 40 Elm Street.")
        DeskOperations(runtime).record_details(street_address="40 Elm Street")
        runtime.record_agent("So that's Dan at 40 Elm Street, tomorrow, eleven to two. Right?")
        runtime.record_user("No, that's not right either.")
        said = DeskOperations(runtime).book_appointment()
        assert said.startswith("Not yet:") and "Nothing has moved" in said
        assert runtime.appointments[0].street_address == "14 Elm Street"

    def test_a_full_window_leaves_the_original_standing(self, runtime, tenant, calendar) -> None:
        _book(runtime)
        # Fill the window they want to move to.
        for i in range(tenant.capacity_per_window):
            calendar.book(
                tenant, day=Date(2026, 9, 15), window_label="two to five",
                customer_name=f"C{i}", phone=f"+1203555010{i}", street_address="1 Main Street",
                town="Wallingford", problem="tune up", now=runtime.now(),
            )
        runtime.record_user("Actually, can you make it the afternoon instead?")
        runtime.slots.update(window="two to five")
        said = _confirm(runtime)
        assert "full" in said and "nothing has changed" in said
        assert runtime.appointments[0].window_label == "eleven to two"
