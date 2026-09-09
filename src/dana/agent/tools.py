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

import logging
from typing import Any, Callable

from dana.agent.runtime import CallRuntime
from dana.agent.slots import spoken_digits
from dana.desk.calendar import BookingError, SlotUnavailable, UnknownWindow, resolve_day

log = logging.getLogger("dana.agent.tools")


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
        try:
            target = resolve_day(self._tenant, day)
        except BookingError:
            return "I didn't catch which day you meant. Today, or tomorrow?"

        windows = cal.availability(self._tenant, target)
        if windows:
            return f"On {target.strftime('%A')} I've got {_spoken_list(windows)}."

        upcoming = cal.next_available(self._tenant)
        if not upcoming:
            return (
                "I don't have anything open in the next couple of weeks. "
                "Let me take your details and the office will call you."
            )
        soonest, labels = upcoming[0]
        return (
            f"{target.strftime('%A')} is full. The soonest I've got is "
            f"{soonest.strftime('%A')} — {_spoken_list(labels)}."
        )

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
        slots = self.runtime.slots
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

        parts = []
        if phone and slots.phone:
            # The prompt: read a number back the very next turn, digits spoken.
            parts.append(f"Read the number back as [slow] {spoken_digits(slots.phone)}.")
        if slots.complete:
            parts.append(f"That's everything. Read it back: {slots.readback()}.")
        else:
            parts.append(f"Still need {slots.missing_spoken()}.")
        if not slots.needs_fuel() and slots.system_type:
            parts.append("Don't ask about fuel for this system type.")
        recorded = ", ".join(newly) if newly else "nothing new"
        return f"Recorded {recorded}. " + " ".join(parts)

    # ---- writing ---------------------------------------------------------

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
        if not slots.check_town(self._tenant):
            return f"{slots.rejected_town} is outside the service area — don't book it."
        if not slots.complete:
            return f"Not yet — still need {slots.missing_spoken()}."

        try:
            target = resolve_day(self._tenant, slots.day)
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
            )
        except UnknownWindow:
            offered = _spoken_list([w.label for w in self._tenant.booking_windows])
            return f"That's not a window we run. I've got {offered}."
        except SlotUnavailable:
            # Never a dead end: the caller is mid-booking and needs the next
            # option in the same breath, not an apology.
            left = cal.availability(self._tenant, target)
            if left:
                return f"That one just filled. I've still got {_spoken_list(left)}."
            upcoming = cal.next_available(self._tenant)
            if upcoming:
                soonest, labels = upcoming[0]
                return (
                    f"That day just filled up. Soonest is {soonest.strftime('%A')} — "
                    f"{_spoken_list(labels)}."
                )
            return "That day is full and I don't have anything soon. Let me take your details."
        except BookingError as exc:
            log.warning("booking refused: %s", exc)
            return "I still need the street address before I can lock that in."

        self.runtime.note_booking(appt)
        return (
            f"Booked: {appt.customer_name}, {appt.street_address} in {appt.town}, "
            f"{target.strftime('%A')} {appt.window_label}, for {appt.problem}."
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
        slots.update(**heard)
        needed = [f for f in ("customer_name", "phone", "street_address", "town", "problem")
                  if not getattr(slots, f)]
        if needed:
            from dana.agent.slots import SPOKEN  # noqa: PLC0415

            return "Before I can send anyone: " + ", ".join(SPOKEN[f] for f in needed) + "."

        appt = cal.dispatch(
            self._tenant,
            customer_name=slots.customer_name,
            phone=slots.phone,
            street_address=slots.street_address,
            town=slots.town,
            problem=slots.problem,
        )
        self.runtime.note_booking(appt)
        return f"On-call tech dispatched to {appt.street_address}, {appt.town}."

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

    Imported lazily and kept mechanical: no behaviour lives in here, so the
    tests that matter never need the SDK installed.

    Each wrapper times its own call and feeds the result to the filler policy,
    which is how `pre_tool_speech: auto` learns which tools are slow without
    anyone annotating them.
    """
    import time  # noqa: PLC0415

    from livekit.agents import function_tool  # noqa: PLC0415

    filler = ops.runtime.filler

    def timed(name: str, fn: Callable[..., str]) -> Callable[..., str]:
        def wrapper(*args: Any, **kwargs: Any) -> str:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                filler.observe_tool(name, elapsed)
                ops.runtime.annotate(**{f"tool_{name}_ms": round(elapsed * 1000, 2)})

        return wrapper

    @function_tool
    async def check_availability(day: str) -> str:
        """Check which arrival windows are open on a given day.

        Args:
            day: The day the caller named, e.g. "today", "tomorrow", "Thursday".
        """
        return timed("check_availability", ops.check_availability)(day)

    @function_tool
    async def lookup_customer(phone: str | None = None) -> str:
        """Look up whether this caller has been a customer before.

        Args:
            phone: Their number, or null to use the number they called from.
        """
        return timed("lookup_customer", ops.lookup_customer)(_said(phone))

    @function_tool
    async def record_details(
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

        Call this whenever the caller gives you any detail, including several at
        once. Pass null for anything they did not say. The reply tells you what
        is still missing, so ask for that and nothing you already have.

        Args:
            customer_name: Their name, or null.
            phone: Their callback number, digits or spoken, or null.
            street_address: House number and street, or null.
            town: The town, or null.
            problem: One line on what is wrong, or null.
            system_type: Furnace, boiler, air conditioner, mini-split, heat pump, or null.
            fuel: Gas or electric, or null. Only ever for a furnace or a boiler.
            day: "today", "tomorrow", a weekday name, or null.
            window: One of the arrival windows you have offered, or null.
            sms_consent: True only if they said yes to the text consent question,
                null if it has not been asked.
        """
        return timed("record_details", ops.record_details)(
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
    async def book_appointment() -> str:
        """Commit the booking, after you have read the details back and they confirmed.

        Takes no arguments -- it books exactly what `record_details` recorded,
        which is what you read back. If anything is still missing it tells you
        what, and nothing is booked.
        """
        return timed("book_appointment", ops.book_appointment)()

    @function_tool
    async def dispatch_emergency() -> str:
        """Send the on-call technician now, for an emergency under the policy.

        Takes no arguments, like `book_appointment`. Record the name, number,
        street address, town and problem with `record_details` first; this tells
        you if any of them is still missing.
        """
        return timed("dispatch_emergency", ops.dispatch_emergency)()

    @function_tool
    async def take_message(
        caller_name: str, phone: str, summary: str, for_staff: str | None = None
    ) -> str:
        """Take a message for a person at the business.

        Args:
            caller_name: The caller's name.
            phone: Their callback number.
            summary: One line on what they want.
            for_staff: The person they asked for, or null if they named nobody.
        """
        return timed("take_message", ops.take_message)(
            caller_name=caller_name, phone=phone, summary=summary, for_staff=for_staff
        )

    return [
        check_availability,
        lookup_customer,
        record_details,
        book_appointment,
        dispatch_emergency,
        take_message,
    ]
