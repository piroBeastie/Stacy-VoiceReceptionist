"""The calendar and the message pad.

Every assertion here is a promise the agent makes out loud. "You're booked for
eleven to two" is only true if the record survives and the slot can't be sold
twice; "we'll text you a confirmation" is only true if consent was captured.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from stacy.delivery import ConsentMissing, LoggingProvider, SmsSender
from stacy.desk import (
    BookingError,
    CalendarStore,
    MessageStore,
    SlotUnavailable,
    UnknownWindow,
    classify_priority,
    resolve_day,
)
from stacy.tenants import Tenant


@pytest.fixture
def cal(tmp_path) -> CalendarStore:
    return CalendarStore(tmp_path)


def _book(cal: CalendarStore, tenant: Tenant, day, window: str, *, name="Dan", now=None):
    return cal.book(
        tenant,
        day=day,
        window_label=window,
        customer_name=name,
        phone="+12035550147",
        street_address="14 Elm Street",
        town="Wallingford",
        problem="no heat upstairs",
        sms_consent=True,
        now=now,
    )


class TestResolveDay:
    """Callers say 'Thursday', never '2026-09-17'."""

    def test_today_and_tomorrow(self, tenant: Tenant, monday_9am: datetime) -> None:
        assert resolve_day(tenant, "today", now=monday_9am) == date(2026, 9, 14)
        assert resolve_day(tenant, "tomorrow", now=monday_9am) == date(2026, 9, 15)

    def test_weekday_names_never_resolve_backwards(
        self, tenant: Tenant, monday_9am: datetime
    ) -> None:
        # Said on a Monday, "Friday" is this week; "Monday" is next week, not today.
        assert resolve_day(tenant, "friday", now=monday_9am) == date(2026, 9, 18)
        assert resolve_day(tenant, "monday", now=monday_9am) == date(2026, 9, 21)

    def test_iso_passes_through(self, tenant: Tenant, monday_9am: datetime) -> None:
        assert resolve_day(tenant, "2026-10-01", now=monday_9am) == date(2026, 10, 1)

    def test_nonsense_is_a_loud_failure(self, tenant: Tenant, monday_9am: datetime) -> None:
        with pytest.raises(BookingError):
            resolve_day(tenant, "sometime next quarter", now=monday_9am)


class TestAvailability:
    def test_open_day_offers_every_window(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        # 9am Monday: the eight-to-eleven window is still running, so all three.
        assert cal.availability(tenant, date(2026, 9, 15), now=monday_9am) == [
            "eight to eleven",
            "eleven to two",
            "two to five",
        ]

    def test_closed_day_offers_nothing(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        sunday = date(2026, 9, 20)
        assert cal.availability(tenant, sunday, now=monday_9am) == []

    def test_windows_that_already_ended_today_are_not_offered(
        self, cal: CalendarStore, tenant: Tenant
    ) -> None:
        # A caller at 3pm cannot be booked into the eight-to-eleven window, and
        # offering it is how an agent gets caught not knowing what time it is.
        afternoon = datetime(2026, 9, 14, 15, 0, tzinfo=ZoneInfo(tenant.timezone))
        assert cal.availability(tenant, date(2026, 9, 14), now=afternoon) == ["two to five"]

    def test_full_window_drops_out(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        day = date(2026, 9, 15)
        for i in range(tenant.capacity_per_window):
            _book(cal, tenant, day, "eleven to two", name=f"Caller {i}", now=monday_9am)
        assert "eleven to two" not in cal.availability(tenant, day, now=monday_9am)
        assert "two to five" in cal.availability(tenant, day, now=monday_9am)

    def test_every_refusal_carries_its_real_reason(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        tuesday, saturday, sunday = date(2026, 9, 15), date(2026, 9, 19), date(2026, 9, 20)
        assert cal.why_unavailable(tenant, sunday, "eleven to two", now=monday_9am) == "closed"
        # Saturday runs nine to one, so no window fits inside it. A live caller
        # was told this one had "just filled up".
        assert (
            cal.why_unavailable(tenant, saturday, "eleven to two", now=monday_9am) == "outside_hours"
        )
        afternoon = datetime(2026, 9, 14, 15, 0, tzinfo=ZoneInfo(tenant.timezone))
        assert (
            cal.why_unavailable(tenant, date(2026, 9, 14), "eight to eleven", now=afternoon)
            == "passed"
        )
        assert cal.why_unavailable(tenant, tuesday, "seven to nine", now=monday_9am) == "unknown"
        for i in range(tenant.capacity_per_window):
            _book(cal, tenant, tuesday, "eleven to two", name=f"Caller {i}", now=monday_9am)
        assert cal.why_unavailable(tenant, tuesday, "eleven to two", now=monday_9am) == "full"
        assert cal.why_unavailable(tenant, tuesday, "two to five", now=monday_9am) == ""

    def test_next_available_skips_closed_days(
        self, cal: CalendarStore, tenant: Tenant
    ) -> None:
        # Saturday afternoon: Saturday's windows have passed, Sunday is closed,
        # so the soonest real offer is Monday.
        saturday_pm = datetime(2026, 9, 19, 16, 0, tzinfo=ZoneInfo(tenant.timezone))
        days = [d for d, _ in cal.next_available(tenant, now=saturday_pm)]
        assert date(2026, 9, 20) not in days
        assert days[0] == date(2026, 9, 21)


class TestBooking:
    def test_a_booking_survives_and_reads_back(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        appt = _book(cal, tenant, date(2026, 9, 15), "eleven to two", now=monday_9am)
        assert appt.appointment_id
        summary = appt.spoken_summary()
        assert "14 Elm Street" in summary and "eleven to two" in summary

    def test_capacity_is_enforced_not_advertised(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        day = date(2026, 9, 15)
        for i in range(tenant.capacity_per_window):
            _book(cal, tenant, day, "two to five", name=f"C{i}", now=monday_9am)
        with pytest.raises(SlotUnavailable):
            _book(cal, tenant, day, "two to five", name="one too many", now=monday_9am)

    def test_unknown_window_is_refused(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        with pytest.raises(UnknownWindow):
            _book(cal, tenant, date(2026, 9, 15), "seven to nine", now=monday_9am)

    def test_no_street_address_no_booking(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        # The prompt says never confirm without one. This makes it structural,
        # so a bad turn cannot produce a visit nobody can drive to.
        with pytest.raises(BookingError):
            cal.book(
                tenant,
                day=date(2026, 9, 15),
                window_label="eleven to two",
                customer_name="Dan",
                phone="+12035550147",
                street_address="   ",
                town="Wallingford",
                problem="no heat",
                now=monday_9am,
            )

    def test_closed_day_cannot_be_booked(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        with pytest.raises(SlotUnavailable):
            _book(cal, tenant, date(2026, 9, 20), "eleven to two", now=monday_9am)


class TestEmergencyDispatch:
    def test_dispatch_ignores_capacity(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        day = monday_9am.date()
        for i in range(tenant.capacity_per_window):
            _book(cal, tenant, day, "two to five", name=f"C{i}", now=monday_9am)
        appt = cal.dispatch(
            tenant,
            customer_name="Ruth",
            phone="+12035550188",
            street_address="9 Oak Lane",
            town="Meriden",
            problem="water heater leaking",
            now=monday_9am,
        )
        assert appt.kind == "emergency"
        assert appt.window_label == "dispatch"

    def test_dispatch_does_not_consume_a_window(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        cal.dispatch(
            tenant,
            customer_name="Ruth",
            phone="+12035550188",
            street_address="9 Oak Lane",
            town="Meriden",
            problem="water heater leaking",
            now=monday_9am,
        )
        assert cal.availability(tenant, date(2026, 9, 15), now=monday_9am) == [
            "eight to eleven",
            "eleven to two",
            "two to five",
        ]


class TestReturningCallers:
    def test_history_matches_on_the_last_ten_digits(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        # Caller ID arrives as +1203..., a caller says "203...". Same person.
        _book(cal, tenant, date(2026, 9, 15), "eleven to two", now=monday_9am)
        assert cal.history_for(tenant.tenant_id, "2035550147")
        assert cal.history_for(tenant.tenant_id, "+1 (203) 555-0147")

    def test_unknown_number_has_no_history(self, cal: CalendarStore, tenant: Tenant) -> None:
        assert cal.history_for(tenant.tenant_id, "+12035559999") == []

    def test_blank_caller_id_does_not_match_everything(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        _book(cal, tenant, date(2026, 9, 15), "eleven to two", now=monday_9am)
        assert cal.history_for(tenant.tenant_id, "") == []


class TestMessages:
    def test_priority_comes_from_the_words(self) -> None:
        assert classify_priority("I smell gas in the basement") == "emergency"
        assert classify_priority("need to reschedule tomorrow morning") == "urgent"
        assert classify_priority("just wondering about your hours") == "normal"

    def test_emergency_wins_over_urgent(self) -> None:
        # Both lists match; under-calling an emergency is not symmetric with
        # over-calling one.
        assert classify_priority("no heat and I need someone today") == "emergency"

    def test_messages_persist_newest_first(self, tmp_path, tenant: Tenant) -> None:
        store = MessageStore(tmp_path)
        store.take(
            tenant_id=tenant.tenant_id,
            caller_name="Pat",
            phone="+12035550101",
            summary="wants a quote on a new furnace",
        )
        store.take(
            tenant_id=tenant.tenant_id,
            caller_name="Sam",
            phone="+12035550102",
            summary="asking for Lisa",
            for_staff="Lisa",
        )
        got = store.all_for(tenant.tenant_id)
        assert [m.caller_name for m in got] == ["Sam", "Pat"]
        assert got[0].for_staff == "Lisa"


class TestSmsConsentIsAControlNotAScript:
    def test_send_refused_without_consent(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        appt = cal.book(
            tenant,
            day=date(2026, 9, 15),
            window_label="eleven to two",
            customer_name="Dan",
            phone="+12035550147",
            street_address="14 Elm Street",
            town="Wallingford",
            problem="no heat",
            sms_consent=False,
            now=monday_9am,
        )
        sender = SmsSender(provider=LoggingProvider())
        with pytest.raises(ConsentMissing):
            sender.send_confirmation(tenant, appt)

    def test_send_allowed_with_consent(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        appt = _book(cal, tenant, date(2026, 9, 15), "eleven to two", now=monday_9am)
        sender = SmsSender(provider=LoggingProvider())
        assert sender.send_confirmation(tenant, appt) == "logged"

    def test_confirmation_carries_the_opt_out(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        # The consent script promises "Reply STOP anytime to opt out", so the
        # message has to actually say it.
        appt = _book(cal, tenant, date(2026, 9, 15), "eleven to two", now=monday_9am)
        body = SmsSender(provider=LoggingProvider()).confirmation_body(tenant, appt)
        assert "STOP" in body
        assert tenant.business_name in body

    def test_the_date_reads_like_a_person_wrote_it(
        self, cal: CalendarStore, tenant: Tenant, monday_9am: datetime
    ) -> None:
        # "confirmed for 2026-09-15" looks like a database receipt, which is
        # the impression the whole call was spent avoiding.
        appt = _book(cal, tenant, date(2026, 9, 15), "eleven to two", now=monday_9am)
        body = SmsSender(provider=LoggingProvider()).confirmation_body(tenant, appt)
        assert "Tuesday, September 15" in body
        assert "2026-09-15" not in body
