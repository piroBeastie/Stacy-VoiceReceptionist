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
        return (
            f"Returning customer: {last.customer_name} at {last.street_address}, "
            f"{last.town}. Last visit was {last.day} for {last.problem}."
        )

    # ---- writing ---------------------------------------------------------

    def book_appointment(
        self,
        *,
        customer_name: str,
        phone: str,
        street_address: str,
        town: str,
        problem: str,
        day: str,
        window: str,
        sms_consent: bool = False,
        system_type: str = "",
        fuel: str = "",
    ) -> str:
        """Reserve a window and make it real.

        Called once, at the confirmation step -- not once per field. The prompt
        collects the details one question at a time; this records the result.
        """
        cal = self.runtime.calendar
        if cal is None:
            return "I can't reach the calendar. Let me take a message instead."
        try:
            target = resolve_day(self._tenant, day)
        except BookingError:
            return "I didn't catch the day. Today, or tomorrow?"

        try:
            appt = cal.book(
                self._tenant,
                day=target,
                window_label=window,
                customer_name=customer_name,
                phone=phone,
                street_address=street_address,
                town=town,
                problem=problem,
                system_type=system_type,
                fuel=fuel,
                sms_consent=sms_consent,
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

    def dispatch_emergency(
        self,
        *,
        customer_name: str,
        phone: str,
        street_address: str,
        town: str,
        problem: str,
    ) -> str:
        """Send the on-call tech now, ahead of the day's bookings."""
        cal = self.runtime.calendar
        if cal is None:
            return "I can't reach dispatch. Take their details and flag it urgent."
        appt = cal.dispatch(
            self._tenant,
            customer_name=customer_name,
            phone=phone,
            street_address=street_address,
            town=town,
            problem=problem,
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
    async def lookup_customer(phone: str = "") -> str:
        """Look up whether this caller has been a customer before.

        Args:
            phone: Their number. Leave empty to use the number they called from.
        """
        return timed("lookup_customer", ops.lookup_customer)(phone)

    @function_tool
    async def book_appointment(
        customer_name: str,
        phone: str,
        street_address: str,
        town: str,
        problem: str,
        day: str,
        window: str,
        sms_consent: bool = False,
        system_type: str = "",
        fuel: str = "",
    ) -> str:
        """Book a service visit once every detail has been collected and read back.

        Args:
            customer_name: The caller's name.
            phone: Their callback number.
            street_address: House number and street. Required.
            town: The town, which must be in the service area.
            problem: One line on what is wrong.
            day: "today", "tomorrow", or a weekday name.
            window: One of the arrival windows offered, spoken exactly.
            sms_consent: True only if the caller said yes to the text consent script.
            system_type: Furnace, boiler, air conditioner, mini-split, heat pump.
            fuel: Only for a furnace or boiler.
        """
        return timed("book_appointment", ops.book_appointment)(
            customer_name=customer_name,
            phone=phone,
            street_address=street_address,
            town=town,
            problem=problem,
            day=day,
            window=window,
            sms_consent=sms_consent,
            system_type=system_type,
            fuel=fuel,
        )

    @function_tool
    async def dispatch_emergency(
        customer_name: str, phone: str, street_address: str, town: str, problem: str
    ) -> str:
        """Send the on-call technician immediately, for an emergency under the policy.

        Args:
            customer_name: The caller's name.
            phone: Their callback number.
            street_address: House number and street.
            town: The town.
            problem: What is happening.
        """
        return timed("dispatch_emergency", ops.dispatch_emergency)(
            customer_name=customer_name,
            phone=phone,
            street_address=street_address,
            town=town,
            problem=problem,
        )

    @function_tool
    async def take_message(
        caller_name: str, phone: str, summary: str, for_staff: str = ""
    ) -> str:
        """Take a message for a person at the business.

        Args:
            caller_name: The caller's name.
            phone: Their callback number.
            summary: One line on what they want.
            for_staff: The person they asked for, if they named one.
        """
        return timed("take_message", ops.take_message)(
            caller_name=caller_name, phone=phone, summary=summary, for_staff=for_staff
        )

    return [
        check_availability,
        lookup_customer,
        book_appointment,
        dispatch_emergency,
        take_message,
    ]
