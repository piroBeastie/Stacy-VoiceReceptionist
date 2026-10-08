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
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields

from stacy.tenants.models import Tenant

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


_REPEATED = re.compile(r"\b(double|triple)\s+(\d)")


def _said_digits(raw: str) -> tuple[str, bool]:
    """The digits in what was said, words and doubles expanded, and whether a '+' was."""
    text = raw.lower().replace("-", " ")
    text = re.sub(r"\bplus\b", "+", text)
    for word, digit in _WORD_DIGITS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    text = _REPEATED.sub(lambda m: m.group(2) * (2 if m.group(1) == "double" else 3), text)
    return "".join(_DIGITS.findall(text)), re.search(r"\+\s*\d", text) is not None


_NANP = re.compile(r"^(?![2-9]11)[2-9][0-8]\d[2-9]\d{6}$")
"""A North American number: area code and exchange both start 2-9, and no area
code has 9 as its middle digit or is an N11 service code."""

_COUNTRY_CODES = {
    "india": "91", "indian": "91", "pakistan": "92", "pakistani": "92", "bangladesh": "880",
    "philippines": "63", "filipino": "63", "mexico": "52", "mexican": "52", "uk": "44",
    "united kingdom": "44", "england": "44", "british": "44", "ireland": "353", "irish": "353",
    "australia": "61", "australian": "61", "new zealand": "64", "germany": "49", "german": "49",
    "france": "33", "french": "33", "nigeria": "234", "nigerian": "234", "china": "86",
    "chinese": "86", "brazil": "55", "brazilian": "55", "uae": "971", "dubai": "971",
    "singapore": "65", "south africa": "27",
}


def normalize_phone(raw: str, *, caller_id: str = "") -> str:
    """Reduce anything phone-shaped to E.164, or return '' if it isn't one.

    Handles spelled-out digits because a caller reading their number aloud is
    the common case and STT does not always render them as numerals -- and
    "double nine, triple four", which is how much of the world says them.

    Ten digits are a North American number, unless they are the end of the
    number the caller is ringing from. A caller from India reading out their
    own mobile says "99884 44520", no country code; this used to reject it,
    the booking tool said "still need a callback number" after every attempt,
    and one live caller was asked for their number five times. A number with
    "+", "plus" or "00" in front keeps its own country code.
    """
    if not raw:
        return ""
    digits, international = _said_digits(raw)
    if digits.startswith("00"):
        digits, international = digits[2:], True

    own = "".join(_DIGITS.findall(caller_id or ""))
    if own and len(digits) >= 10 and own.endswith((digits, digits.removeprefix("0"))):
        return f"+{own}"

    if len(digits) == 11 and digits.startswith("0") and not international:
        # "Oh, it's two-zero-three..." -- the "oh" was a word, not a digit.
        digits = digits[1:]
    if digits.startswith("1"):
        return f"+{digits}" if len(digits) == 11 and _NANP.match(digits[1:]) else ""
    if international:
        return f"+{digits}" if 8 <= len(digits) <= 15 else ""
    # Only a number that can exist in North America gets +1. Live, with the
    # caller ID withheld, "double 9, double 8, triple 4, 5 2 0" was saved as
    # +1 998 844 4520 -- no such area code -- and read back as a US number.
    return f"+1{digits}" if len(digits) == 10 and _NANP.match(digits) else ""


def spoken_digits(e164: str) -> str:
    """'+12035550147' -> 'two-zero-three, five-five-five, zero-one-four-seven'.

    North American numbers are read without the country code, the way the prompt
    always read them. Any other number is read whole, code first --
    '+919988444520' -> 'plus nine-one, nine-nine-eight-eight-four,
    four-four-five-two-zero' -- because leaving the code off a number from
    abroad reads back a different number.
    """
    names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    def say(group: str) -> str:
        return "-".join(names[int(d)] for d in group)

    digits = "".join(_DIGITS.findall(e164))
    north_american = (len(digits) == 11 and digits.startswith("1")) or (
        len(digits) == 10 and not e164.strip().startswith("+")
    )
    if north_american:
        national = digits[-10:]
        return ", ".join(say(g) for g in (national[:3], national[3:6], national[6:]))
    if not 8 <= len(digits) <= 15:
        return ""
    if len(digits) > 10:
        groups = [digits[:-10], digits[-10:-5], digits[-5:]]
    else:
        groups = [digits[i : i + 3] for i in range(0, len(digits), 3)]
    return "plus " + ", ".join(say(g) for g in groups)


def phone_in_speech(text: str, *, caller_id: str = "") -> str:
    """The phone number a caller read out in one turn, or '' if they didn't read one.

    Only a turn carrying a whole number counts -- at least ten digits once words
    and doubles are expanded -- so a house number, "eleven to two", or a number
    cut off halfway is left to the model and the booking tool.
    """
    if not text or len(_said_digits(text)[0]) < 10:
        return ""
    return normalize_phone(text, caller_id=caller_id)


_NUMBER_WORDS = {
    **{word: digit for word, digit in _WORD_DIGITS.items() if word != "oh"},
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}
_DAYS = ("today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_HOUR_RANGE = re.compile(r"\b(\d{1,2})\s*(?:to|-|till|until|through)\s*(\d{1,2})\b")


def _hour12(hhmm: str) -> int:
    return (int(hhmm.split(":")[0]) % 12) or 12


def heard_booking_details(text: str, tenant: Tenant) -> dict[str, str]:
    """The town, day and arrival window a caller named in one turn, if they named them.

    A backstop for the booking tool, which the model can skip: on a replayed
    call it answered "Today works, two to five" and never recorded either. A
    question is not a choice ("what have you got today and tomorrow?"), so it
    yields nothing; a day on its own counts only in a short answer ("Tomorrow."),
    not in a sentence that merely mentions one.
    """
    if not text or "?" in text:
        return {}
    lowered = text.lower()
    heard: dict[str, str] = {}

    town = next(
        (t for t in tenant.service_area if re.search(rf"\b{re.escape(t.lower())}\b", lowered)), ""
    )
    if town:
        heard["town"] = town

    window = next((w.label for w in tenant.booking_windows if w.label in lowered), "")
    if not window:
        spoken = lowered
        for word, digit in _NUMBER_WORDS.items():
            spoken = re.sub(rf"\b{word}\b", digit, spoken)
        by_hours = {(_hour12(w.start), _hour12(w.end)): w.label for w in tenant.booking_windows}
        window = next(
            (by_hours[k] for m in _HOUR_RANGE.finditer(spoken) if (k := (int(m[1]), int(m[2]))) in by_hours),
            "",
        )
    if window:
        heard["window"] = window

    mentions = [(m.start(), m[0]) for d in _DAYS for m in re.finditer(rf"\b{d}\b", lowered)]
    if mentions and (window or len(lowered.split()) <= 5):
        heard["day"] = max(mentions)[1]  # the last one said: "today -- actually tomorrow"
    return heard


def split_town(street: str, tenant: Tenant) -> tuple[str, str]:
    """'office suite 1 2 3 Wallingford' -> ('office suite 1 2 3', 'Wallingford').

    A live call recorded the whole of that as the street and no town, then asked
    for the town twice -- once before the read-back and once when the booking
    refused. A served town at the end of a street is the town.
    """
    stripped = street.strip().rstrip(".,")
    for town in tenant.service_area:
        if len(stripped) > len(town) and stripped.lower().endswith(town.lower()):
            head = re.sub(r"(?:,|\s)+(?:in)?\s*$", "", stripped[: -len(town)])
            if head:
                return head, town
    return street, ""


def phone_across_turns(previous: str, current: str, *, caller_id: str) -> str:
    """The caller's own number, read out or corrected over two turns, or ''.

    On a live call the caller said "... triple 4, 5 0." and then, as a turn of
    its own, "5 2 0." Neither turn held a whole number, nothing was caught, and
    the agent read digits back from memory -- wrong. Only the number they are
    calling from is recognised this way: joined straight, or with the second
    turn replacing the end of the first, the digits have to end in that number,
    so nothing is ever guessed.
    """
    own = "".join(_DIGITS.findall(caller_id or ""))
    before, after = _said_digits(previous or "")[0], _said_digits(current or "")[0]
    if len(own) < 10 or not before or not after:
        return ""
    joined = [before + after] + [
        before[:-k] + after for k in range(1, min(len(before), len(after)) + 1)
    ]
    if any(10 <= len(d) <= len(own) and d[-10:] == own[-10:] for d in joined):
        return f"+{own}"
    return ""


def number_was_said(e164: str, said: str) -> bool:
    """Whether a number's digits are digits the caller actually spoke, in a row.

    Replayed: the caller had said nine digits, and the model passed a complete
    twelve-digit number of its own making. It was well formed, so it was saved
    and read back. The last ten digits (all of them, for a shorter number) have
    to appear, in order, in what the caller said.
    """
    tail = "".join(_DIGITS.findall(e164))[-10:]
    return bool(tail) and tail in _said_digits(said)[0]


def foreign_number_note(text: str) -> str:
    """A note for ten digits said without a country code that can't be a US number."""
    digits, international = _said_digits(text or "")
    if international or len(digits) != 10 or _NANP.match(digits):
        return ""
    return (
        "(Not said by the caller: those ten digits aren't a US number, so nothing was saved. "
        "Ask which country the number is from, and don't read it back until you know.)"
    )


def number_with_country(previous: str, current: str) -> str:
    """The foreign number from the caller's last turn, once this turn names its country.

    "double 9, double 8, triple 4, 5 2 0" and then "India" -> +919988444520.
    """
    digits, international = _said_digits(previous or "")
    if international or len(digits) != 10 or _NANP.match(digits):
        return ""
    low = (current or "").lower()
    for country, code in _COUNTRY_CODES.items():
        if re.search(rf"\b{re.escape(country)}\b", low):
            return f"+{code}{digits}"
    return ""


def partial_number_note(text: str) -> str:
    """A note for a turn that carries most of a phone number but not all of it.

    Six to nine digits is a number cut short, never an address or a window.
    Without the note the model filled in the rest itself and read it back.
    """
    count = len(_said_digits(text)[0]) if text else 0
    if 6 <= count < 10:
        return (
            "(Not said by the caller: that is only part of a phone number. Ask for the "
            "rest, and don't read any digits back yet.)"
        )
    return ""


_SPELLED = re.compile(r"(?<![\w'])(?:[a-z][\s,.-]+){2,}[a-z](?![\w'])", re.I)


def spelled_name(text: str) -> str:
    """A name spelled out letter by letter -- 'n a n a k' -> 'Nanak' -- or ''.

    Speech-to-text gets an unusual name wrong on nearly every call ("Nanook",
    "Lanak", "Nak") but hears the letters. Three letters or more, each standing
    alone, so "A C" or the "s" of "it's" is never taken for one.
    """
    for m in _SPELLED.finditer(text or ""):
        letters = re.findall(r"[a-z]", m[0], re.I)
        if len(letters) >= 3:
            word = "".join(letters)
            return word[0].upper() + word[1:].lower()
    return ""


def readback_note(e164: str) -> str:
    """Attached to the caller's message when their number was caught in code."""
    return (
        "(Not said by the caller: the number above is saved as their callback number. "
        f"Read it back exactly as [slow] {spoken_digits(e164)} -- don't work the digits "
        "out yourself.)"
    )


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
