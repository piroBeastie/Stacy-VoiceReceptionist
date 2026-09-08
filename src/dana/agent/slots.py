"""What the agent has learned so far about a booking.

Callers do not answer one question at a time. "Yeah hi, this is Dan over at
fourteen Elm in Wallingford and my furnace is making a noise" is three fields and
a town in one breath, and the agent has to keep all of it.

The prompt's "one item per turn" rule governs what the agent *asks*, never what
it can *absorb* -- but read literally it marches through name, phone, address,
problem, system, fuel, window in order regardless of what was already said. The
taste section already has the right instruction ("if they've already said
either, don't ask again -- use it"); the booking flow did not. Being asked for
your own name after you have just given it is one of the sharpest tells there is.

So the slots are the agent's working memory, and :meth:`BookingSlots.missing`
answers "what do I still need?" deterministically, in the prompt's own order.
The model does not have to remember; it has to record.

Two of the prompt's rules become structural here rather than instructed:

* **Fuel is only asked for a furnace or a boiler.** The prompt says "never ask
  gas-or-electric about an air conditioner, a mini-split, or a heat pump", and
  :meth:`needs_fuel` means the model cannot ask it anyway.
* **The town has to be in the service area.** "Never book outside the service
  area" is checked when the town is recorded, not discovered at booking time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields as dataclass_fields

from dana.tenants.models import Tenant

FUEL_SYSTEMS = ("furnace", "boiler")
"""Only these are ever asked about fuel. See the module docstring."""

ASK_ORDER: tuple[str, ...] = (
    "customer_name",
    "phone",
    "street_address",
    "town",
    "problem",
    "system_type",
    "fuel",
    "day",
    "window",
)
"""The prompt's own order. `missing()` returns fields in this sequence so the
agent still works through them the way the script intends."""

REQUIRED: frozenset[str] = frozenset(
    {"customer_name", "phone", "street_address", "town", "problem", "day", "window"}
)
"""What a booking cannot be confirmed without. System type is useful and fuel is
conditional, so neither blocks."""

SPOKEN = {
    "customer_name": "their name",
    "phone": "a callback number",
    "street_address": "the street address",
    "town": "the town",
    "problem": "what's going on",
    "system_type": "the system type",
    "fuel": "gas or electric",
    "day": "which day",
    "window": "which window",
}

_DIGITS = re.compile(r"\d")
_WORD_DIGITS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def normalize_phone(raw: str) -> str:
    """Reduce anything phone-shaped to E.164, or return '' if it isn't one.

    Handles spelled-out digits because a caller reading their number aloud is
    the common case and STT does not always render them as numerals.
    """
    if not raw:
        return ""
    text = raw.lower().replace("-", " ")
    for word, digit in _WORD_DIGITS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    digits = "".join(_DIGITS.findall(text))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return f"+1{digits}" if len(digits) == 10 else ""


def spoken_digits(e164: str) -> str:
    """'+12035550147' -> 'two-zero-three, five-five-five, oh-one-four-seven'.

    The prompt reads numbers back as words, with [slow] in front, and leaves off
    the country code.
    """
    names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    digits = "".join(_DIGITS.findall(e164))[-10:]
    if len(digits) != 10:
        return ""
    groups = (digits[:3], digits[3:6], digits[6:])
    return ", ".join("-".join(names[int(d)] for d in group) for group in groups)


@dataclass
class BookingSlots:
    """Everything gathered toward one booking, and what is still outstanding."""

    customer_name: str = ""
    phone: str = ""
    street_address: str = ""
    town: str = ""
    problem: str = ""
    system_type: str = ""
    fuel: str = ""
    day: str = ""
    window: str = ""
    sms_consent: bool | None = None
    """None means never asked. False means asked and declined -- a real answer."""

    rejected_town: str = field(default="", repr=False)
    """A town outside the service area, kept so the agent can name it."""

    def update(self, **values: object) -> list[str]:
        """Record whatever was heard. Returns the field names newly filled.

        Blank strings are ignored rather than clearing a slot: a model passing
        `street_address=""` means "I didn't hear one", not "forget the one you
        have".
        """
        known = {f.name for f in dataclass_fields(self)}
        newly: list[str] = []
        for name, value in values.items():
            if name not in known or value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
                if name == "phone":
                    normalized = normalize_phone(value)
                    if not normalized:
                        continue
                    value = normalized
            if getattr(self, name) != value:
                newly.append(name)
            setattr(self, name, value)
        return newly

    def check_town(self, tenant: Tenant) -> bool:
        """Whether the recorded town is served. Clears it if not.

        "Never book outside the service area" checked where the town arrives,
        rather than discovered after the caller has given six more fields.
        """
        if not self.town:
            return True
        served = {t.casefold() for t in tenant.service_area}
        if not served or self.town.casefold() in served:
            return True
        self.rejected_town, self.town = self.town, ""
        return False

    def needs_fuel(self) -> bool:
        """Fuel is only ever asked about a furnace or a boiler."""
        return any(s in self.system_type.casefold() for s in FUEL_SYSTEMS)

    def missing(self) -> list[str]:
        """Outstanding required fields, in the prompt's asking order.

        Fuel is walked in place rather than appended, so it lands after the
        system type and before the window -- where the script asks for it --
        instead of after everything else.
        """
        out: list[str] = []
        for name in ASK_ORDER:
            if name == "fuel":
                if self.needs_fuel() and not self.fuel:
                    out.append(name)
            elif name in REQUIRED and not getattr(self, name):
                out.append(name)
        return out

    def missing_spoken(self) -> str:
        """The outstanding fields as the agent would name them."""
        names = [SPOKEN.get(f, f) for f in self.missing()]
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        return f"{', '.join(names[:-1])}, and {names[-1]}"

    @property
    def complete(self) -> bool:
        return not self.missing()

    def readback(self) -> str:
        """The confirmation line: name, address and town, problem, window.

        Exactly the four things the prompt says to repeat back, and no more.
        """
        return (
            f"{self.customer_name}, {self.street_address} in {self.town}, "
            f"{self.problem}, {self.day} {self.window}"
        )
