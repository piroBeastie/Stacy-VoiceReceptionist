"""Appointments, availability, and the capacity check that makes them true.

This is a real store, not a placeholder. The distinction matters more here than
it looks: the README's own rule is that "a tool pointed at a placeholder is
worse than no tool, because the agent will call it mid-call and fall apart in
front of a prospect." An agent that says "you're booked for eleven to two" has
made a promise, and the only thing that makes the promise true is a record that
survives the call and refuses to double-book.

Storage is append-only JSONL per tenant. Same reasoning as the metrics sink: it
is greppable, it survives a crashed worker, and it needs no service running to
inspect after a bad call.

The capacity check and the append are one critical section, held by a thread
lock *and* by a lock file the operating system arbitrates, so a second worker
process cannot take the same last slot. Two callers reaching two workers in
the same second is not a hypothetical: it is a Tuesday morning after an ad
runs, and the symptom is two vans at two houses in one window. Postgres behind
:class:`CalendarStore` is still the better answer at volume -- this is the
honest version of a file store, not a replacement for one.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO, Literal
from zoneinfo import ZoneInfo

from stacy.tenants.models import Tenant

WEEKDAY_NAMES = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)


LOCK_WAIT_S = 5.0
"""Longest wait for another process to leave the calendar's critical section."""

if os.name == "nt":  # pragma: no cover - one branch per host
    import msvcrt

    def _take_lock(handle: IO[bytes]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _drop_lock(handle: IO[bytes]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:  # pragma: no cover - one branch per host
    import fcntl

    def _take_lock(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _drop_lock(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class BookingError(Exception):
    """Base for anything that stops a booking from being real."""


class SlotUnavailable(BookingError):
    """The requested window is full or the office is closed that day."""


class UnknownWindow(BookingError):
    """The window label isn't one this tenant offers."""


@dataclass(frozen=True)
class Appointment:
    """One booked visit. Immutable; a change is a new record."""

    appointment_id: str
    tenant_id: str
    day: str
    """ISO date in the tenant's timezone."""

    window_label: str
    """Spoken window, or 'dispatch' for an emergency with no window."""

    customer_name: str
    phone: str
    street_address: str
    town: str
    problem: str
    kind: Literal["appointment", "emergency"] = "appointment"
    system_type: str = ""
    fuel: str = ""
    sms_consent: bool = False
    """Captured during booking. The SMS sender refuses without it -- the prompt
    asks for consent, but only this field can withhold the message."""

    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())

    def spoken_summary(self) -> str:
        """What the agent reads back. Windows spoken, never clock times."""
        where = f"{self.street_address}, {self.town}"
        when = "today" if self.kind == "emergency" else f"{self.day}, {self.window_label}"
        return f"{self.customer_name} at {where} — {self.problem} — {when}"


def resolve_day(tenant: Tenant, spoken: str, *, now: datetime | None = None) -> Date:
    """Turn what a caller said into a date in the tenant's timezone.

    Accepts 'today', 'tomorrow', a weekday name (the next such day, never one in
    the past), or an ISO date. Callers say "Thursday", not "2026-09-11", and a
    tool that only takes ISO pushes that conversion onto the model, which is the
    one participant with no clock.
    """
    now = now or datetime.now(ZoneInfo(tenant.timezone))
    today = now.date()
    text = spoken.strip().lower()

    if text in ("today", "this afternoon", "this morning", "now"):
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)
    if text in WEEKDAY_NAMES:
        target = WEEKDAY_NAMES.index(text)
        ahead = (target - today.weekday()) % 7
        return today + timedelta(days=ahead or 7)
    try:
        return Date.fromisoformat(text)
    except ValueError as exc:
        raise BookingError(f"could not read {spoken!r} as a day") from exc


class CalendarStore:
    """Availability and booking for every tenant, backed by JSONL on disk."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, tenant_id: str) -> Path:
        return self._dir / f"appointments-{tenant_id}.jsonl"

    @contextmanager
    def _across_processes(self, tenant_id: str) -> Iterator[None]:
        """Hold this tenant's calendar against every other process.

        Gives up after `LOCK_WAIT_S` rather than holding a caller on the line:
        a refusal the agent can speak is better than silence, and a window
        another worker has been sitting inside for five seconds is gone anyway.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"appointments-{tenant_id}.lock"
        with path.open("a+b") as handle:
            deadline = time.monotonic() + LOCK_WAIT_S
            while True:
                try:
                    _take_lock(handle)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise BookingError(
                            "the calendar is busy -- take a message and call them back"
                        ) from None
                    time.sleep(0.05)
            try:
                yield
            finally:
                _drop_lock(handle)

    def _load(self, tenant_id: str) -> list[Appointment]:
        path = self._path(tenant_id)
        if not path.exists():
            return []
        out: list[Appointment] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(Appointment(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def _append(self, appt: Appointment) -> None:
        with self._path(appt.tenant_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(appt), separators=(",", ":")) + "\n")
            fh.flush()

    def booked_count(self, tenant_id: str, day: Date, window_label: str) -> int:
        """How many visits are already on the books for that window."""
        iso = day.isoformat()
        return sum(
            1
            for a in self._load(tenant_id)
            if a.day == iso and a.window_label == window_label and a.kind == "appointment"
        )

    def availability(
        self, tenant: Tenant, day: Date, *, now: datetime | None = None
    ) -> list[str]:
        """Window labels still bookable on `day`.

        Three filters, in order: the office has to be open, the window has to
        not have already passed if the day is today, and it has to have capacity
        left. Offering a window that fails any of these produces a promise the
        business then has to break by hand.
        """
        now = now or datetime.now(ZoneInfo(tenant.timezone))
        return [
            w.label
            for w in tenant.booking_windows
            if not self.why_unavailable(tenant, day, w.label, now=now)
        ]

    def why_unavailable(
        self, tenant: Tenant, day: Date, window_label: str, *, now: datetime | None = None
    ) -> str:
        """Why a window can't be booked: 'closed', 'unknown', 'outside_hours', 'passed' or 'full'.

        An empty string means it can. The reasons are kept apart because they
        are said differently on the phone: a Saturday afternoon the office never
        runs used to come back as "just filled up", and a live caller went round
        the same impossible booking twice.
        """
        now = now or datetime.now(ZoneInfo(tenant.timezone))
        weekday = WEEKDAY_NAMES[day.weekday()]
        if not tenant.is_open_on(weekday):
            return "closed"
        window = next((w for w in tenant.booking_windows if w.label == window_label), None)
        if window is None:
            return "unknown"
        hours = tenant.hours_for(weekday)
        # A window has to fit inside the day's hours. Zero-padded 24h strings
        # compare correctly as strings, so no parsing is needed. Without this the
        # Saturday that closes at one still offered a two-to-five window, which is
        # a visit nobody would have shown up for.
        if (
            hours is not None
            and not hours.closed
            and (window.start < (hours.open_at or "") or window.end > (hours.close_at or ""))
        ):
            return "outside_hours"
        if day == now.date():
            end_h, _, end_m = window.end.partition(":")
            window_end = now.replace(hour=int(end_h), minute=int(end_m or 0), second=0, microsecond=0)
            if window_end <= now:
                return "passed"
        if self.booked_count(tenant.tenant_id, day, window.label) >= tenant.capacity_per_window:
            return "full"
        return ""

    def next_available(
        self, tenant: Tenant, *, within_days: int = 14, now: datetime | None = None
    ) -> list[tuple[Date, list[str]]]:
        """The next few days that have any window open, soonest first."""
        now = now or datetime.now(ZoneInfo(tenant.timezone))
        found: list[tuple[Date, list[str]]] = []
        for offset in range(within_days):
            day = now.date() + timedelta(days=offset)
            windows = self.availability(tenant, day, now=now)
            if windows:
                found.append((day, windows))
            if len(found) >= 3:
                break
        return found

    def book(
        self,
        tenant: Tenant,
        *,
        day: Date,
        window_label: str,
        customer_name: str,
        phone: str,
        street_address: str,
        town: str,
        problem: str,
        system_type: str = "",
        fuel: str = "",
        sms_consent: bool = False,
        now: datetime | None = None,
    ) -> Appointment:
        """Reserve a window. Raises rather than overbooking.

        The capacity check and the append happen under one lock, so two calls
        landing on the same worker cannot both take the last slot.
        """
        if window_label not in {w.label for w in tenant.booking_windows}:
            raise UnknownWindow(
                f"{window_label!r} is not a window {tenant.business_name} offers"
            )
        if not street_address.strip():
            # The prompt says never confirm without one; this makes it structural.
            raise BookingError("a street address is required before a booking is real")

        with self._lock, self._across_processes(tenant.tenant_id):
            if window_label not in self.availability(tenant, day, now=now):
                raise SlotUnavailable(f"{window_label} on {day.isoformat()} is not available")
            appt = Appointment(
                appointment_id=uuid.uuid4().hex[:12],
                tenant_id=tenant.tenant_id,
                day=day.isoformat(),
                window_label=window_label,
                customer_name=customer_name,
                phone=phone,
                street_address=street_address,
                town=town,
                problem=problem,
                system_type=system_type,
                fuel=fuel,
                sms_consent=sms_consent,
            )
            self._append(appt)
        return appt

    def dispatch(
        self,
        tenant: Tenant,
        *,
        customer_name: str,
        phone: str,
        street_address: str,
        town: str,
        problem: str,
        sms_consent: bool = False,
        now: datetime | None = None,
    ) -> Appointment:
        """Send the on-call tech now. No window, no capacity check.

        An emergency does not queue behind the day's bookings, which is the
        whole point of the emergency branch in the prompt.
        """
        now = now or datetime.now(ZoneInfo(tenant.timezone))
        with self._lock, self._across_processes(tenant.tenant_id):
            appt = Appointment(
                appointment_id=uuid.uuid4().hex[:12],
                tenant_id=tenant.tenant_id,
                day=now.date().isoformat(),
                window_label="dispatch",
                customer_name=customer_name,
                phone=phone,
                street_address=street_address,
                town=town,
                problem=problem,
                kind="emergency",
                sms_consent=sms_consent,
            )
            self._append(appt)
        return appt

    def history_for(self, tenant_id: str, phone: str) -> list[Appointment]:
        """Every past visit for a phone number, newest first.

        This is what turns "can I get your address?" into "is this about the
        unit at fourteen Elm?" -- the cheapest single thing that makes an agent
        sound like it works there.
        """
        digits = "".join(c for c in phone if c.isdigit())[-10:]
        if not digits:
            return []
        matches = [
            a
            for a in self._load(tenant_id)
            if "".join(c for c in a.phone if c.isdigit())[-10:] == digits
        ]
        # See MessageStore.all_for: append order breaks timestamp ties.
        return sorted(  # noqa: C414 - reversed() is what breaks ties by append order
            reversed(matches), key=lambda a: a.created_at, reverse=True
        )
