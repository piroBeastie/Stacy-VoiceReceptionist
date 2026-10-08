"""What the agent can actually do, and the words it gets back.

Split deliberately in two. :class:`DeskOperations` is plain Python with no SDK
import -- every branch, including every failure branch, is unit-tested without a
SIP trunk. :func:`build_function_tools` is the mechanical adapter that exposes
those methods to the model. When the livekit tool-calling API moves, only the
adapter moves.

Every method returns a sentence the agent can say, not a JSON blob. A model
handed `{"available": ["eleven to two"]}` will sometimes read the braces out
loud; a model handed "I've got eleven to two, or two to five" will not. The
return value is the agent's next line, so it is written as one.

Failure strings are the important ones. "That window just filled -- I've got
eleven to two or two to five" keeps the call moving; a bare error makes the
agent apologise and stall, which is the exact moment a caller decides they are
talking to a machine.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from datetime import date as Date
from datetime import timedelta
from typing import Any

from stacy.agent.runtime import CallRuntime
from stacy.agent.slots import (
    foreign_number_note,
    normalize_phone,
    number_was_said,
    split_town,
    spoken_digits,
)
from stacy.desk.calendar import BookingError, SlotUnavailable, UnknownWindow, resolve_day

try:
    # At module scope on purpose. The SDK resolves tool annotations with
    # get_type_hints, which looks names up in this module's globals -- and every
    # annotation here is a string, because of `from __future__ import annotations`.
    # Imported inside build_function_tools, `ctx: RunContext` would not resolve,
    # and the failure would surface as a crash when a call starts.
    from livekit.agents import RunContext
except ImportError:  # the desk logic and its tests do not need the SDK
    RunContext = Any  # type: ignore[assignment,misc]

log = logging.getLogger("stacy.agent.tools")

_CORRECTION = re.compile(
    r"\b(?:wrong|not right|isn'?t right|actually|instead|change|different|mistake|mixed up"
    r"|that'?s not|it'?s not)\b",
    re.I,
)
"""Words that answer a read-back with a fix rather than a yes."""
_STREET_WORDS = re.compile(
    r"\b(street|st|road|rd|avenue|ave|lane|ln|drive|court|ct|place|way|boulevard|blvd"
    r"|circle|terrace|highway|route|parkway|pike)\b",
    re.IGNORECASE,
)
"""Words that make a caller's line sound like it carried a street."""


def _describe_call(
    name: str, args: tuple[Any, ...], kwargs: dict[str, Any], result: object, elapsed_ms: float
) -> str:
    """One log line for a tool call: what the model passed, and what it got back.

    A caller was asked for a street they had already given, and nothing in the
    log could say whether the model never recorded it or recorded it and asked
    anyway -- two different fixes. Empty arguments are left out: they are the
    model saying it heard nothing. A declined text consent is False, and stays.
    """
    given = [repr(a) for a in args if a not in (None, "")]
    given += [f"{k}={v!r}" for k, v in kwargs.items() if v not in (None, "")]
    return f"{name}({', '.join(given)}) -> {str(result)[:200]!r} in {elapsed_ms:.0f}ms"


def _day_name(day: Date, today: Date) -> str:
    """'today', 'tomorrow', or the weekday -- what a caller would say.

    A weekday alone misled on a live call: told "soonest is Friday" on a Friday,
    the agent offered the caller today's windows as tomorrow's.
    """
    if day == today:
        return "today"
    if day == today + timedelta(days=1):
        return "tomorrow"
    return day.strftime("%A")


def _on(day: Date, today: Date) -> str:
    """The same inside a sentence: 'today', 'tomorrow', 'on Saturday'."""
    name = _day_name(day, today)
    return name if name in ("today", "tomorrow") else f"on {name}"


_HOUR_WORDS = ("twelve", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven")


def _spoken_time(hhmm: str) -> str:
    """'13:00' -> 'one', '17:30' -> 'five-thirty'. Opening hours, the way they're said."""
    hour, _, minute = hhmm.partition(":")
    word = _HOUR_WORDS[int(hour) % 12]
    if minute in ("", "00"):
        return word
    return f"{word}-thirty" if minute == "30" else f"{word} {minute}"


def _sentence(text: str) -> str:
    """Capitalise the first letter only; `str.capitalize` would lowercase 'Saturday'."""
    return text[:1].upper() + text[1:]


def _spoken_list(items: list[str], conjunction: str = "or") -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"


class DeskOperations:
    """The receptionist's actual capabilities, as testable callables."""

    def __init__(self, runtime: CallRuntime) -> None:
        self.runtime = runtime

    @property
    def _tenant(self):
        return self.runtime.tenant

    # ---- reading ---------------------------------------------------------

    def check_availability(self, day: str) -> str:
        """Windows still open on a day the caller named.

        Args:
            day: What the caller said -- "today", "tomorrow", "Thursday", or a
                date. Never require ISO from the model; it has no clock.
        """
        cal = self.runtime.calendar
        if cal is None:
            return "I can't see the calendar right now, so let me take a message instead."
        now = self.runtime.now()
        try:
            target = resolve_day(self._tenant, day, now=now)
        except BookingError:
            return "I didn't catch which day you meant. Today, or tomorrow?"

        windows = cal.availability(self._tenant, target, now=now)
        if windows:
            return f"{_sentence(_on(target, now.date()))} I've got {_spoken_list(windows)}."
        return self._unbookable(target)

    def call_briefing(self) -> str:
        """What the agent knows before the caller says a word, as the call's first message.

        Most calls opened with round trips: `lookup_customer`, then
        `record_details`, then `check_availability` -- each a model pass that
        wrote a tool call, waited, and ran again before the caller heard a
        word. Measured live, those replies took two to three and a half seconds
        against about one second for a plain answer. Everything they fetched is
        known when the phone rings, so it is worked out here, once.

        It is a message, not part of the system prompt, because it changes on
        every call, and anything that changes inside the system prompt stops
        the prompt cache from being shared between calls.
        """
        runtime, cal = self.runtime, self.runtime.calendar
        now = runtime.now()
        today = now.date()
        lines = ["(Call details, not said aloud. This is not the caller speaking.)"]
        section = runtime.call_section.removeprefix("# This call").strip()
        if section:
            lines.append(section)
        lines.append(self.lookup_customer())

        caller = normalize_phone(runtime.caller_id)
        if caller:
            # Offered, not saved. Saved up front, a replayed call skipped the
            # callback number altogether, and the caller never confirmed it.
            lines.append(
                f"The number they're calling from is [slow] {spoken_digits(caller)}. When you "
                "ask for their callback number, offer that one, read exactly like that; if they "
                "say yes, record it with record_details."
            )
        else:
            lines.append("There's no usable caller ID, so ask for their number.")

        if cal is None:
            lines.append("The calendar can't be reached: take a message instead of booking.")
            return "\n".join(lines)
        opened = []
        for offset in (0, 1):
            day = today + timedelta(days=offset)
            windows = cal.availability(self._tenant, day, now=now)
            name = _day_name(day, today) + ("" if offset == 0 else f" ({day.strftime('%A')})")
            opened.append(f"{_sentence(name)}: {_spoken_list(windows) if windows else 'nothing open'}.")
        later = [(d, w) for d, w in cal.next_available(self._tenant, now=now) if d > today + timedelta(days=1)]
        if later:
            day, windows = later[0]
            opened.append(f"Next open after that: {day.strftime('%A')}, {_spoken_list(windows)}.")
        lines.append(
            "Open windows. " + " ".join(opened) + " For any other day, use check_availability."
        )
        return "\n".join(lines)

    def _unbookable(self, target: Date, window: str = "") -> str:
        """Why nothing (or `window`) can be booked on `target`, and what can, in one breath.

        The reason is the real one, and the alternative is never a dead end.
        """
        cal, now = self.runtime.calendar, self.runtime.now()
        today = now.date()
        on_day = _on(target, today)
        probe = window or (self._tenant.booking_windows[0].label if self._tenant.booking_windows else "")
        reason = cal.why_unavailable(self._tenant, target, probe, now=now) if probe else "closed"
        hours = self._tenant.hours_for(target.strftime("%A").lower())
        if reason == "closed":
            lead = f"We're closed {on_day}."
        elif not window:
            lead = f"There's nothing open {on_day}."
        elif reason == "outside_hours" and hours is not None and hours.open_at and hours.close_at:
            # The hours, not just "we don't run it": told only that, the model
            # filled in a reason of its own -- "that window's already booked".
            lead = (
                f"{_sentence(on_day)} we're only open {_spoken_time(hours.open_at)} to "
                f"{_spoken_time(hours.close_at)}, so there's no {window}."
            )
        else:
            lead = {
                "outside_hours": f"We don't run {window} {on_day}.",
                "passed": f"{_sentence(window)} {on_day} has already gone by.",
                "full": f"{_sentence(window)} {on_day} just filled.",
            }.get(reason, f"{_sentence(window)} {on_day} isn't open.")

        left = cal.availability(self._tenant, target, now=now)
        if left:
            return f"{lead} {_sentence(on_day)} I've still got {_spoken_list(left)}."
        upcoming = [(d, w) for d, w in cal.next_available(self._tenant, now=now) if d != target]
        if upcoming:
            soonest, labels = upcoming[0]
            return (
                f"{lead} The soonest I've got is {_day_name(soonest, today)}: "
                f"{_spoken_list(labels)}."
            )
        return f"{lead} Nothing's open in the next couple of weeks. Let me take their details."

    def check_when(self, newly: list[str]) -> str:
        """Refuse a day or window the calendar can't book, the moment it is recorded.

        Checked here as well as at booking, because a read-back is a promise. A
        live caller said "tomorrow, eleven to two" on a Friday -- a Saturday
        afternoon the office never runs -- had it read back, confirmed it, and
        only then heard it refused. Twice. A refused day or window is cleared,
        so it cannot be read back or booked again.
        """
        slots, cal = self.runtime.slots, self.runtime.calendar
        if cal is None or not slots.day or not ({"day", "window"} & set(newly)):
            return ""
        now = self.runtime.now()
        try:
            target = resolve_day(self._tenant, slots.day, now=now)
        except BookingError:
            slots.day = ""
            return "I didn't catch which day they meant. Ask: today, or tomorrow?"
        open_windows = cal.availability(self._tenant, target, now=now)
        if not slots.window:
            if open_windows:
                return ""
            slots.day = ""
            return f"{self._unbookable(target)} Offer that instead."
        if slots.window in open_windows:
            return ""
        refused, slots.window = slots.window, ""
        if not open_windows:
            slots.day = ""
        if refused not in {w.label for w in self._tenant.booking_windows}:
            offered = _spoken_list(open_windows or [w.label for w in self._tenant.booking_windows])
            return f"{refused!r} isn't a window we run. Offer {offered}."
        return f"{self._unbookable(target, refused)} Offer one of those; don't read anything back yet."

    def lookup_customer(self, phone: str = "") -> str:
        """Whether this caller has been here before, and for what.

        Defaults to the caller ID. Recognising a returning caller is the
        cheapest thing on this list that makes the agent sound like it works
        there rather than like it was hired this morning.

        A hit pre-fills the slots, so the agent confirms an address instead of
        collecting one. It says so in the return value: people move, and the
        record is a starting point rather than a fact.
        """
        cal = self.runtime.calendar
        if cal is None:
            return "No record on this number."
        number = phone or self.runtime.caller_id
        if not number:
            return "No record on this number."
        history = cal.history_for(self._tenant.tenant_id, number)
        if not history:
            return "No record on this number — treat them as a new customer."
        last = history[0]
        self.runtime.slots.update(
            customer_name=last.customer_name,
            phone=last.phone,
            street_address=last.street_address,
            town=last.town,
        )
        return (
            f"Returning customer: {last.customer_name} at {last.street_address}, "
            f"{last.town}. Last visit was {last.day} for {last.problem}. "
            f"Confirm the address is still right rather than asking for it again."
        )

    def record_details(
        self,
        *,
        customer_name: str = "",
        phone: str = "",
        street_address: str = "",
        town: str = "",
        problem: str = "",
        system_type: str = "",
        fuel: str = "",
        day: str = "",
        window: str = "",
        sms_consent: bool | None = None,
    ) -> str:
        """Write down whatever the caller just gave, however many things it was.

        Callers answer three questions in one breath. Everything heard goes in
        here, and the return value names only what is still outstanding -- so
        the agent asks for the next missing thing rather than marching through a
        list it has already been given half of.
        """
        if street_address and not town:
            street_address, town = split_town(street_address, self._tenant)
        slots = self.runtime.slots
        heard_phone = phone
        # Against the number they are calling from first, so a caller reading
        # out their own mobile without its country code gets that number.
        phone = normalize_phone(phone, caller_id=self.runtime.caller_id) if phone else ""
        # Only a number the caller said, or the one they are calling from. On a
        # call, the tool is fed from recorded speech; driven directly, with no
        # conversation recorded, there is nothing to check against.
        said = " ".join(
            t.get("text", "") for t in self.runtime.transcript if t.get("role") == "user"
        )
        invented = bool(
            phone
            and said
            and phone != normalize_phone(self.runtime.caller_id)
            and not number_was_said(phone, said)
        )
        if invented:
            phone = ""
        newly = slots.update(
            customer_name=customer_name,
            phone=phone,
            street_address=street_address,
            town=town,
            problem=problem,
            system_type=system_type,
            fuel=fuel,
            day=day,
            window=window,
            sms_consent=sms_consent,
        )

        if not slots.check_town(self._tenant):
            # "Never book outside the service area", checked where the town
            # arrives rather than after six more fields have been collected.
            return (
                f"{slots.rejected_town} is outside the service area. Say so in one line "
                "and suggest they look for a local company. Don't offer a callback."
            )

        refusal = self.check_when(newly)
        parts = [refusal] if refusal else []
        if invented:
            parts.append(
                f"{heard_phone!r} isn't the number they said, so nothing was saved. Ask "
                "them to read it once more, and don't read any number back until one is saved."
            )
        elif heard_phone and not phone and foreign_number_note(heard_phone):
            parts.append(
                f"{heard_phone!r} isn't a US number, so nothing was saved. Ask which country "
                "it's from, and don't read it back until you know."
            )
        elif heard_phone and not phone:
            # Dropped silently, this looped: the reply only ever said "still need
            # a callback number", so the agent asked again and again, and with
            # nothing to read back it read out a number it made up.
            parts.append(
                f"{heard_phone!r} didn't come through as a full phone number, so nothing "
                "was saved. Ask for it once more, digit by digit. Don't read any number "
                "back until one is saved."
            )
        elif "phone" in newly:
            # The prompt: read a number back the very next turn, digits spoken. Only
            # a number new here: one the caller's message carried was read back
            # from its note, and a second read-back after the tool call is a tell.
            parts.append(f"Read the number back as [slow] {spoken_digits(slots.phone)}.")
        if slots.complete:
            # Live, told only "read it back", the model said "Let me get that booked
            # for you" first -- announcing a booking the caller hadn't agreed to.
            parts.append(
                "That's everything. Read it back and ask if it's right; don't book it, or say "
                f"you're booking it, until they say yes. Read it back: {slots.readback()}."
            )
        else:
            parts.append(f"Still need {slots.missing_spoken()}.")
            said = self._street_already_said() if "street_address" in slots.missing() else ""
            if said:
                parts.append(
                    f'They may already have given the street: "{said}". '
                    "Record it from that instead of asking again."
                )
        if not slots.needs_fuel() and slots.system_type:
            parts.append("Don't ask about fuel for this system type.")
        recorded = ", ".join(newly) if newly else "nothing new"
        return f"Recorded {recorded}. " + " ".join(parts)

    def _street_already_said(self) -> str:
        """The caller's latest line that sounds like it named a street, if any.

        A live caller gave "Coffey Street 1 2 3, Wallingford" when asked for
        their town. The town was checked, the street was never recorded, and
        they were asked for it again later -- the exact thing the slots exist to
        prevent. The model only writes down what it records, so the tool that
        reports the gap also points at the line that already filled it.
        """
        for turn in reversed(self.runtime.transcript):
            text = turn.get("text", "")
            if turn.get("role") == "user" and _STREET_WORDS.search(text):
                return text.strip()
        return ""

    def _read_back_answered(self) -> bool:
        """Whether the caller has spoken since hearing this booking read back.

        Replayed with speak-first recording, the agent read the details back and
        called book_appointment in the same breath -- booked before the caller
        could say yes. So one of the agent's last few lines before the caller's
        latest one has to be the read-back: it names the caller and the current
        window's hours, so a read-back of a window since changed does not count.

        A few lines, not just the last one. On a live call the yes came out
        garbled, the agent asked "Sorry, say that once more?", the caller said
        yes again -- and a last-line-only check refused a booking the caller had
        clearly confirmed.

        With no conversation recorded at all -- the tool driven directly, not
        on a call -- there is nothing to check.
        """
        transcript = self.runtime.transcript
        if not any(t.get("role") == "agent" for t in transcript):
            return True
        last_user = max((i for i, t in enumerate(transcript) if t.get("role") == "user"), default=-1)
        if last_user < 0:
            return False
        recent = [t.get("text", "").lower() for t in transcript[:last_user] if t.get("role") == "agent"][-3:]
        slots = self.runtime.slots
        hours = [w for w in slots.window.lower().split() if w != "to"]

        # Any part of the name will do: a read-back says "Nana", not "Nana Nanak".
        names = [n for n in re.findall(r"[a-z']+", slots.customer_name.lower()) if len(n) > 1]

        def is_read_back(line: str) -> bool:
            named = not names or any(re.search(rf"\b{re.escape(n)}\b", line) for n in names)
            return named and all(re.search(rf"\b{re.escape(w)}\b", line) for w in hours)

        return any(is_read_back(line) for line in recent)

    # ---- writing ---------------------------------------------------------

    def _booked_already(self) -> Any:
        """The appointment this call already made for the visit now recorded, if any.

        The same visit is the same name, street, day and window. Review: matching
        without the day refused a real second visit, Thursday after tomorrow.
        """
        slots = self.runtime.slots
        if not slots.day:
            return None
        try:
            day = resolve_day(self._tenant, slots.day, now=self.runtime.now()).isoformat()
        except BookingError:
            return None
        visit = (slots.customer_name, slots.street_address, day, slots.window)
        for booked in self.runtime.appointments:
            fields = ("customer_name", "street_address", "day", "window_label")
            if tuple(getattr(booked, f, None) for f in fields) == visit:
                return booked
        return None

    def _turned_the_read_back_down(self) -> str:
        """The caller's last words, when they are a no or a correction rather than a yes.

        A booking is the one thing that cannot be taken back on the call, so it
        waits for a plain yes. Anything unclear is left to the model, which has
        the conversation; only a refusal stops it here.
        """
        from stacy.agent.fastlane import _yes_no  # noqa: PLC0415 - avoids an import cycle

        said = [t.get("text", "") for t in self.runtime.transcript if t.get("role") == "user"]
        last = (said[-1] if said else "").strip()
        if not last:
            return ""
        if _yes_no(last) is False or _CORRECTION.search(last):
            return last[:80]
        return ""

    def book_appointment(self, **heard: object) -> str:
        """Commit the booking that has been assembled so far.

        Takes no required arguments: it books what :meth:`record_details` wrote
        down, which means the address that gets booked is the one that was read
        back, not a re-statement the model may have drifted on. Anything passed
        here is folded in first, so a model that front-loads every field into
        one call still works.
        """
        cal = self.runtime.calendar
        if cal is None:
            return "I can't reach the calendar. Let me take a message instead."

        slots = self.runtime.slots
        slots.update(**heard)
        booked = self._booked_already()
        if booked is not None:
            # The desk books on a yes to the read-back. A model that books the
            # same visit again would double it, or be told the window just filled.
            # A second visit on another day is a real second booking.
            now = self.runtime.now()
            when = _day_name(Date.fromisoformat(booked.day), now.date())
            return (
                f"Booked already, earlier in this call: {booked.customer_name}, "
                f"{booked.street_address}, {when}, {booked.window_label}. "
                "It's done -- don't book it again."
            )
        if not slots.check_town(self._tenant):
            return f"{slots.rejected_town} is outside the service area — don't book it."
        if not slots.complete:
            return f"Not yet — still need {slots.missing_spoken()}."
        if not self._read_back_answered():
            return (
                "Not yet: read it all back first -- name, address, problem, day and window -- "
                "and call book_appointment again once they say yes. Nothing is booked yet, so "
                "don't tell them they're booked or all set."
            )
        turned_down = self._turned_the_read_back_down()
        if turned_down:
            # Replayed against the model: the read-back was answered "No that's
            # all" and it booked anyway, saying "You're all set".
            return (
                f"Not yet: their answer to the read-back was {turned_down!r}, which isn't a yes. "
                "Ask what needs changing, fix it, and read it back again. Nothing is booked, so "
                "don't tell them they're booked or all set."
            )

        now = self.runtime.now()
        try:
            target = resolve_day(self._tenant, slots.day, now=now)
        except BookingError:
            return "I didn't catch the day. Today, or tomorrow?"

        try:
            appt = cal.book(
                self._tenant,
                day=target,
                window_label=slots.window,
                customer_name=slots.customer_name,
                phone=slots.phone,
                street_address=slots.street_address,
                town=slots.town,
                problem=slots.problem,
                system_type=slots.system_type,
                fuel=slots.fuel,
                sms_consent=bool(slots.sms_consent),
                now=now,
            )
        except UnknownWindow:
            offered = _spoken_list([w.label for w in self._tenant.booking_windows])
            return f"That's not a window we run. I've got {offered}."
        except SlotUnavailable:
            # Never a dead end, and never the wrong reason. The refused window is
            # cleared so the same impossible slot cannot be read back and tried
            # again; a new one has to go through record_details.
            refused, slots.window = slots.window, ""
            if not cal.availability(self._tenant, target, now=now):
                slots.day = ""
            return self._unbookable(target, refused)
        except BookingError as exc:
            log.warning("booking refused: %s", exc)
            return "I still need the street address before I can lock that in."

        self.runtime.note_booking(appt)
        return (
            f"Booked: {appt.customer_name}, {appt.street_address} in {appt.town}, "
            f"{_day_name(target, now.date())} {appt.window_label}, for {appt.problem}."
        )

    def dispatch_emergency(self, **heard: object) -> str:
        """Send the on-call tech now, ahead of the day's bookings.

        Slot-backed like :meth:`book_appointment`, so an emergency that surfaces
        halfway through an ordinary booking keeps everything already collected
        instead of starting the intake over on the worst possible call.
        """
        cal = self.runtime.calendar
        if cal is None:
            return "I can't reach dispatch. Take their details and flag it urgent."

        slots = self.runtime.slots
        sent = next((a for a in self.runtime.appointments if a.window_label == "dispatch"), None)
        if sent is not None:
            # Replayed: the model called this twice in one turn and two techs
            # were sent to the same house.
            return (
                f"Already sent, earlier in this call: the on-call tech is on the way to "
                f"{sent.street_address}, {sent.town}. Don't send anyone again."
            )
        slots.update(**heard)
        needed = [f for f in ("customer_name", "phone", "street_address", "town", "problem")
                  if not getattr(slots, f)]
        if needed:
            from stacy.agent.slots import SPOKEN  # noqa: PLC0415

            return "Before I can send anyone: " + ", ".join(SPOKEN[f] for f in needed) + "."

        appt = cal.dispatch(
            self._tenant,
            customer_name=slots.customer_name,
            phone=slots.phone,
            street_address=slots.street_address,
            town=slots.town,
            problem=slots.problem,
            now=self.runtime.now(),
        )
        self.runtime.note_booking(appt)
        return f"On-call tech dispatched to {appt.street_address}, {appt.town}."

    def end_call(self, reason: str = "") -> str:
        """Hang up, for a call that was never going to be a booking.

        A wrong number, a sales call, somebody who will not stop swearing: a
        person on a desk ends those, and so can this. What it refuses is the
        case that would be unforgivable -- hanging up holding half of
        somebody's address -- and anything at all while the gas instruction
        is latched, where the caller leaving the line is the whole point.
        """
        if self.runtime.safety.latched:
            return (
                "No. The instruction stands until they are out of the house -- repeat it, and let them go."
            )
        slots = self.runtime.slots
        collected = [f for f in ("customer_name", "phone", "street_address", "town", "problem")
                     if getattr(slots, f)]
        finished = bool(self.runtime.appointments) or self.runtime.outcome == "message"
        if collected and not finished and not self._caller_signed_off():
            return (
                f"Not yet: you have {', '.join(collected)} written down and nothing booked. "
                "Finish it or take a message. If they want to go, let them say so first."
            )
        self.runtime.end_call = reason.strip() or "the agent ended the call"
        log.info("call %s: end_call -- %s", self.runtime.call_id, self.runtime.end_call)
        return "Alright. Say your last line now; the line closes once you stop speaking."

    def _caller_signed_off(self) -> bool:
        """Whether the caller's last words were them ending the call."""
        from stacy.agent.hangup import caller_signed_off  # noqa: PLC0415 - import cycle

        said = [t.get("text", "") for t in self.runtime.transcript if t.get("role") == "user"]
        return bool(said) and caller_signed_off(said[-1])

    def take_message(
        self,
        *,
        caller_name: str,
        phone: str,
        summary: str,
        for_staff: str = "",
    ) -> str:
        """Write it on the pad. The fallback when a booking can't complete."""
        store = self.runtime.messages
        if store is None:
            return "Message noted."
        msg = store.take(
            tenant_id=self._tenant.tenant_id,
            caller_name=caller_name,
            phone=phone,
            summary=summary,
            for_staff=for_staff,
        )
        self.runtime.note_message(msg)
        who = f" for {for_staff}" if for_staff else ""
        return f"Message taken{who} — marked {msg.priority}."


def _said(value: str | None) -> str:
    """Normalise an unsupplied argument to the blank the desk layer expects.

    Optional arguments are declared `str | None` rather than `str = ""` because
    of how the schema reaches the model. livekit-agents renders every parameter
    into `required`, whatever its Python default, and makes the unsupplied case
    expressible by widening the type to `["string", "null"]`. So the model is
    obliged to send all ten keys on every call and needs a legal way to say "the
    caller did not tell me this".

    With `str = ""` there was no such value. Asked for a phone number nobody had
    given, the model filled the slot with whatever came to hand -- in testing,
    fragments of its own tool-call scaffolding -- and that went into the booking
    as the caller's callback number. Null is the honest answer, and this turns it
    back into the blank that `BookingSlots.update` already treats as "no change".
    """
    return (value or "").strip()


def build_function_tools(ops: DeskOperations) -> list[Any]:
    """Wrap :class:`DeskOperations` as livekit function tools.

    Imported lazily and kept mechanical: the desk logic lives in
    :class:`DeskOperations`, so the tests that matter never need the SDK.

    What does live here is how a tool call *feels* on the phone. Every operation
    runs off the event loop, one at a time, and a slow one is covered with a
    line that fits it -- but only once the line has actually gone quiet. See
    :mod:`stacy.agent.filler` for why the model itself is never covered.
    """
    import time  # noqa: PLC0415

    from livekit.agents import function_tool  # noqa: PLC0415

    filler = ops.runtime.filler
    delay = ops.runtime.pipeline.filler.tool_delay_s
    one_at_a_time = asyncio.Lock()

    async def run(ctx: Any, name: str, fn: Callable[..., str], *args: Any, **kwargs: Any) -> str:
        """Run a desk operation off the event loop, covering it only if it is slow.

        Off the loop, because the desk reads and appends files, and the SDK has
        already measured what synchronous work on the agent loop costs: audio
        and turn handling stall behind it. One at a time, because operations
        share this call's booking state and the model can ask for several at
        once.

        The filler source is a callable, so whether to speak -- and what -- is
        decided when the idle dwell actually expires. A tool that finishes
        first never asks, and consumes nothing.
        """

        def work() -> str:
            start = time.perf_counter()
            result = ""
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                ops.runtime.annotate(**{f"tool_{name}_ms": elapsed_ms})
                ops.runtime.latency.on_tool(name, elapsed_ms)
                log.info(
                    "tool %s %s",
                    ops.runtime.call_id,
                    _describe_call(name, args, kwargs, result, elapsed_ms),
                )

        async with one_at_a_time:
            if not filler.covers(name):
                return await asyncio.to_thread(work)
            async with ctx.with_filler(lambda _step: filler.tool_phrase(name), delay=delay):
                return await asyncio.to_thread(work)

    @function_tool
    async def check_availability(ctx: RunContext, day: str) -> str:
        """Check which arrival windows are open on a day the call details don't list.

        Real work: say one short line about it first ("Let me see what Thursday
        looks like."), then wait for this reply before naming any window.

        Args:
            day: The day the caller named, e.g. "today", "tomorrow", "Thursday".
        """
        return await run(ctx, "check_availability", ops.check_availability, day)

    @function_tool
    async def lookup_customer(ctx: RunContext, phone: str | None = None) -> str:
        """Look up whether a number has been a customer before.

        Only for a number other than the one they called from: the call details
        already cover that one. Real work: say one short line first, then wait
        for this reply.

        Args:
            phone: Their number, or null to use the number they called from.
        """
        return await run(ctx, "lookup_customer", ops.lookup_customer, _said(phone))

    @function_tool
    async def record_details(
        ctx: RunContext,
        customer_name: str | None = None,
        phone: str | None = None,
        street_address: str | None = None,
        town: str | None = None,
        problem: str | None = None,
        system_type: str | None = None,
        fuel: str | None = None,
        day: str | None = None,
        window: str | None = None,
        sms_consent: bool | None = None,
    ) -> str:
        """Write down everything the caller just told you, however many things it was.

        Say your whole spoken reply first -- the acknowledgement and your next
        question together -- then call this in the same response, so the caller
        never waits on it. Never stop at "Got it." to wait for this reply: live,
        that left a second of silence before every question. A phone number is
        no exception: the digits to read back are in the note on the caller's
        message, so never call this first to get them. Call it whenever the caller gives you any
        detail, even a single one like just their name, and including several at
        once. Pass null for anything they did not say. If the reply names
        something still missing that you already asked for, say nothing more.

        Args:
            customer_name: Their name, spelled the way they spelled it if they did, or null.
            phone: Their callback number, digits or spoken, or null.
            street_address: House number and street, or null.
            town: The town, or null.
            problem: One line on what is wrong, or null.
            system_type: Furnace, boiler, air conditioner, mini-split, heat pump, or null
                (null too when they don't know it).
            fuel: Gas, electric, oil or propane; "unknown" if they don't know; or null.
                Only ever for a furnace or a boiler.
            day: "today", "tomorrow", a weekday name, or null.
            window: One of the arrival windows you have offered, or null.
            sms_consent: True only if they said yes to the text consent question,
                null if it has not been asked.
        """
        return await run(
            ctx,
            "record_details",
            ops.record_details,
            customer_name=_said(customer_name),
            phone=_said(phone),
            street_address=_said(street_address),
            town=_said(town),
            problem=_said(problem),
            system_type=_said(system_type),
            fuel=_said(fuel),
            day=_said(day),
            window=_said(window),
            sms_consent=sms_consent,
        )

    @function_tool
    async def book_appointment(ctx: RunContext) -> str:
        """Commit the booking, after you have read the details back and they confirmed.

        Takes no arguments -- it books exactly what `record_details` recorded,
        which is what you read back. If anything is still missing it tells you
        what, and nothing is booked. Real work, unlike record_details: say one
        short line first ("Let me get that booked for you."), then wait for this
        reply. Only a reply that starts "Booked" means they're booked; anything
        else means nothing is, so never say they're all set.
        """
        return await run(ctx, "book_appointment", ops.book_appointment)

    @function_tool
    async def dispatch_emergency(ctx: RunContext) -> str:
        """Send the on-call technician now, for an emergency under the policy.

        Real work: say one short line first ("I'm getting the on-call tech for
        you now."), then wait for this reply. Takes no arguments, like `book_appointment`. Record the name, number,
        street address, town and problem with `record_details` first; this tells
        you if any of them is still missing.
        """
        return await run(ctx, "dispatch_emergency", ops.dispatch_emergency)

    @function_tool
    async def take_message(
        ctx: RunContext, caller_name: str, phone: str, summary: str, for_staff: str | None = None
    ) -> str:
        """Take a message for a person at the business.

        Real work: say one short line first, then wait for this reply before
        telling them it's passed on.

        Args:
            caller_name: The caller's name.
            phone: Their callback number.
            summary: One line on what they want.
            for_staff: The person they asked for, or null if they named nobody.
        """
        return await run(
            ctx,
            "take_message",
            ops.take_message,
            caller_name=caller_name,
            phone=phone,
            summary=summary,
            for_staff=for_staff,
        )

    @function_tool
    async def end_call(ctx: RunContext, reason: str) -> str:
        """End the call, for someone who is not a customer.

        A wrong number, a sales call, or somebody abusive after you have
        already said you are happy to help when they are ready. Say your
        line first -- the call ends when you stop speaking, not before.

        Never for a caller you are still collecting details from: finish the
        booking or take a message. Never during the gas instruction.

        Args:
            reason: Why, in a few words. It goes in the call record.
        """
        return await run(ctx, "end_call", ops.end_call, reason=reason)

    return [
        check_availability,
        lookup_customer,
        record_details,
        book_appointment,
        dispatch_emergency,
        take_message,
        end_call,
    ]
