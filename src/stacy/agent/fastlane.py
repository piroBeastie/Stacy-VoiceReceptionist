"""Replies the desk gives without waiting on the model.

Measured against the live API with 99% of the prompt read from cache, Claude
Haiku took 650 to 1,400 ms to its first token, and 2.6 s at worst. On a phone
number it said nothing at all and called the booking tool first, so the caller
waited through two passes. Some turns have exactly one right reply, and the
prompt already scripts it: the number read back digit for digit, the text
consent line, the next question on the list, the read-back once the window is
chosen, and the booking once the read-back is confirmed. For those, the reply is
written here the moment the caller finishes, and the model is not asked.

Narrow on purpose. A turn is answered here only when all of these hold:

* a service booking is under way (a problem, town or address is recorded) and
  nothing has been booked yet;
* the agent's last line asked one thing this module recognises;
* the caller answered that and nothing else: no question, and no words beyond
  "yeah", "it's", "okay";
* whatever comes next is something this module can word without judgment, and
  the calendar agrees with it.

Anything else goes to the model exactly as before. A turn answered here leaves
a note on the caller's message, so the model still sees what was recorded.
"""

from __future__ import annotations

import copy
import difflib
import re
from dataclasses import dataclass, field
from datetime import date as Date
from typing import TYPE_CHECKING

from stacy.agent.slots import (
    _DAYS,
    _SPELLED,
    BookingSlots,
    _said_digits,
    spelled_name,
    heard_booking_details,
    normalize_phone,
    phone_in_speech,
    spoken_digits,
)
from stacy.agent.tools import _day_name
from stacy.desk.calendar import BookingError, resolve_day

if TYPE_CHECKING:
    from stacy.agent.runtime import CallRuntime

SMS_CONSENT_LINE = (
    "We'll text you a confirmation, and one reminder before your appointment if needed — "
    "that's up to two texts for this booking. Message and data rates may apply. Reply STOP "
    "anytime to opt out, or HELP for help. Is it okay if we text that number?"
)
"""The prompt's consent line, word for word: it is the one line it says to say exactly."""

SAME_DAY_NEXT = "The tech will give you a call about thirty minutes before they head over."
LATER_NEXT = (
    "The office will call in the morning to confirm, and the tech calls about thirty minutes "
    "before arriving."
)
"""What happens next, in the prompt's words, for a visit today and for any later day."""


@dataclass(frozen=True)
class DeskReply:
    """A reply decided in code: what to say, what to record, and what to tell the model.

    `action` is "book" when the reply stands only if the booking goes through;
    the caller of :func:`desk_reply` books first and speaks it only then.
    """

    speech: str
    record: dict[str, object] = field(default_factory=dict)
    note: str = ""
    action: str = ""


_TAG = re.compile(r"\[[a-z ]+\]")
_BETWEEN = re.compile(r"\bbetween (\w+) and (\w+)\b")
_STREET_SUFFIX = (
    r"street|st|road|rd|avenue|ave|lane|ln|drive|dr|court|ct|place|pl|way|boulevard|blvd"
    r"|circle|terrace|highway|parkway|pike"
)
_STREET_WORDS = re.compile(rf"\b(?:{_STREET_SUFFIX})\b", re.I)

_SOFT = frozenset(
    "um uh er erm oh so well okay ok alright yeah yes yep sure huh mhm hmm like just".split()
)
"""Words that carry no answer of their own, and may surround one."""

_DIGIT_WORDS = frozenset(
    "zero oh one two three four five six seven eight nine double triple plus".split()
)
_AROUND_A_NUMBER = _SOFT | frozenset(
    "it's its it is my number the you can reach me at on call and that's thats i'm im "
    "cell mobile phone no sorry actually".split()
)
_YES = frozenset(
    "yes yeah yep yup sure okay ok correct fine absolutely definitely perfect great alright "
    "right exactly".split()
)
_NO = frozenset("no nope nah don't dont not".split())
_AROUND_YES_NO = _SOFT | frozenset(
    "that's thats it's its is go ahead works sounds good of course please thanks thank you "
    "do that i'd id rather text texts me all let's lets it cool nice look book then".split()
)
_AROUND_A_WINDOW = (
    _SOFT
    | frozenset(_DAYS)
    | frozenset(
        "do it let's lets go with the that one works please can you for window i'll ill take "
        "sounds good fine perfect great then is between and to put me us in better prefer rather i'd id book "
        "morning afternoon "
        "one two three four five six seven eight nine ten eleven twelve".split()
    )
)
_NOT_NAMES = frozenset(
    """yes yeah yep no nope sure okay ok hello hi hey sorry what why who how um uh hmm the a an
    and my it's is this that not i you we me here there just still wait hold on one second
    minute moment bye thanks thank please today tomorrow morning afternoon furnace boiler heat
    heater heating ac air conditioner cooling broken leak leaking gas oil electric street road
    avenue lane drive none nothing rather prefer don't dont can't cant need number""".split()
)
_ASKS = {
    "phone": re.compile(
        r"\b(?:best number|phone number|callback number|good number|number to reach|reach you)\b"
    ),
    "name": re.compile(r"\b(?:your name|get a name|who am i (?:talking|speaking) (?:to|with))\b"),
    "street": re.compile(r"\b(?:street address|street name|the address|your address)\b"),
    "town": re.compile(r"\b(?:what|which) town\b"),
}
_LEAD = r"^(?:(?:yeah|yes|sure|okay|ok|oh|um|uh|so|hi|hey)\b[,.]?\s*)*"
_NAME_LEAD = re.compile(
    _LEAD + r"(?:(?:it's|it is|this is|my name is|my name's|name's|i'm|i am|call me"
    r"|you're (?:talking|speaking) (?:to|with)|you are (?:talking|speaking) (?:to|with))\b[,.]?\s*)?",
    re.I,
)
_STREET_LEAD = re.compile(
    _LEAD
    + r"(?:(?:it's|it is|it|i'm at|i am at|i'm in|i live at|i live in|we're at|we are at|that's"
    r"|the address is|my address is|address is)\b[,.]?\s*)?(?:the\s+)?",
    re.I,
)
_STREET_ANSWER = re.compile(
    r"^(?P<house>\d+[a-z]?(?:\s+\d+)*)\s+"
    r"(?P<name>[a-z][a-z'.-]*(?:\s+[a-z][a-z'.-]*){0,3}?)\s+"
    rf"(?P<suffix>{_STREET_SUFFIX})\.?"
    r"(?:,?\s+(?:in\s+)?(?P<town>[a-z][a-z ]*?))?$",
    re.I,
)
_STREET_NAME_FIRST = re.compile(
    r"^(?P<name>[a-z][a-z'.-]*(?:\s+[a-z][a-z'.-]*){0,3}?)\s+"
    rf"(?P<suffix>{_STREET_SUFFIX})\.?,?\s+"
    r"(?P<house>\d+[a-z]?(?:\s+\d+)*)"
    r"(?:,?\s+(?:in\s+)?(?P<town>[a-z][a-z ]*?))?$",
    re.I,
)
"""'Coffee Street 1 2 3 Wallingford': the order one caller gives it in, every time."""
_SAME_NUMBER = re.compile(
    _LEAD
    + r"(?:(?:it's|its|it is|use|you can use|just)\s+)?(?:the\s+)?"
    r"(?:(?:this|same)(?:\s+(?:same\s+)?(?:1|one|number))?|that\s+(?:1|one|number)"
    r"|(?:1|one|number) i'?m calling (?:from|on))"
    r"(?:\s+(?:only|is fine|works|please))?$",
    re.I,
)
_HOLDS = re.compile(
    r"^(?:sure,? take your time|no rush|still there|sorry,? say that once more for me)[.!?]*$", re.I
)
"""Lines that hold the conversation rather than move it: the question before them still stands."""
_AROUND_A_SPELLING = _SOFT | frozenset("it's its it is my name name's spelled that's thats like i'm im".split())
_INTRODUCED = re.compile(r"\b(?:my name is|my name's|this is|i'm|i am)\s+([A-Z][a-z]+)")


def desk_reply(runtime: CallRuntime, text: str) -> DeskReply | None:
    """The reply to this caller turn, if the desk can give it without the model."""
    slots = runtime.slots
    if not text.strip() or runtime.appointments or runtime.safety.latched:
        return None
    if slots.rejected_town or not (slots.problem or slots.town or slots.street_address):
        return None
    line = _last_agent_turn(runtime)
    if _for_the_reseller(runtime, line):
        return None
    asked = _asked(runtime, line)

    # A whole number and nothing else is read back whatever the last line asked:
    # the prompt's own rule. Live, the model asked "say that number once more
    # for me, slow?" -- which matched nothing here -- and read ten digits back as nine.
    reply = _on_phone(runtime, text)
    if reply is None and asked in ("phone", "offer_phone"):
        reply = _on_same_number(runtime, text, asked)
    if reply is None:
        handler = {
            "offer_phone": _on_offer,
            "sms": _on_sms,
            "name": _on_name,
            "street": _on_street,
            "town": _on_town,
            "window": _on_window,
            "readback": _on_readback,
        }.get(asked)
        reply = handler(runtime, text) if handler else None
    if reply is None and not asked:
        # Live: asked "today, or tomorrow?", the caller said "tomorrow morning, 11
        # to 2" -- a day and a window, nothing to guess.
        reply = _on_window(runtime, text)
    if reply is None:
        return None
    # "Never say the same sentence twice in one call." The consent line is the
    # exception: a corrected number has to be asked about again.
    if SMS_CONSENT_LINE not in reply.speech and _said_before(runtime, reply.speech):
        return None
    return reply


# ---- what was asked ------------------------------------------------------


def _last_agent_turn(runtime: CallRuntime) -> str:
    """Everything the agent said since the caller last spoke, as one line.

    One reply often reaches the transcript as two lines either side of a tool
    call. Live: "Perfect. So to confirm: Lanak, 123 Coffee Street ... between
    eleven and two." then "Does that sound right?" Read alone, the last one is a
    question about nothing, and the yes that followed took four seconds to book.
    """
    lines: list[str] = []
    for turn in reversed(runtime.transcript):
        if turn.get("role") == "agent":
            lines.append(turn.get("text", ""))
            continue
        if not lines:
            continue
        said = " ".join(reversed(lines))
        # Live: the read-back, "Yeah. But the", "Sure, take your time.", then
        # "Yeah. It works." The yes answers the read-back, not the hold.
        if not _HOLDS.match(_TAG.sub("", said).strip()):
            return said
        lines = []
    said = " ".join(reversed(lines))
    return "" if _HOLDS.match(_TAG.sub("", said).strip()) else said


def name_spelled(runtime: CallRuntime, text: str) -> str:
    """A spelled-out name, when a name is what's being talked about; '' otherwise."""
    name = spelled_name(text)
    if not name:
        return ""
    asked = _plain(_last_agent_turn(runtime))
    # Live, the question was "And who am I talking to?" -- no "name" in it -- and
    # "Nana n a n a k" was booked as Nana Nanak.
    if "name" in text.lower() or "name" in asked or _ASKS["name"].search(asked):
        return name
    return ""


def _for_the_reseller(runtime: CallRuntime, line: str) -> bool:
    """A line naming the reseller belongs to the demo close, where a number is for a callback."""
    reseller = runtime.tenant.reseller
    if reseller is None:
        return False
    low = line.lower()
    names = {reseller.contact_spoken_name.lower(), reseller.contact_full_name.split()[0].lower()}
    return any(re.search(rf"\b{re.escape(n)}\b", low) for n in names if n)


def _plain(line: str) -> str:
    low = _TAG.sub("", line).lower().replace("’", "'").strip()
    return _BETWEEN.sub(r"\1 to \2", low)


def _asked(runtime: CallRuntime, line: str) -> str:
    """What the agent's last line asked for, or '' unless it is one thing we know.

    A question has to end the line and be its only one, and match exactly one
    kind. A line that names open windows without asking anything else is an
    offer of those windows. A line naming the reseller belongs to the demo
    close, where a number is taken for a callback, not a booking.
    """
    low = _plain(line)
    if not low or low.count("?") > 1 or _for_the_reseller(runtime, low):
        return ""
    # A read-back said as a statement is still confirmed by a yes; without this
    # the yes looked like a pick of the window it names, and got a second read-back.
    if (low.endswith("?") or "?" not in low) and _is_read_back(runtime, low):
        return "readback"
    if low.endswith("?"):
        question = re.split(r"(?<=[.!])\s+", low)[-1]
        if "okay if we text" in question:
            return "sms"
        own = "".join(ch for ch in normalize_phone(runtime.caller_id) if ch.isdigit())
        if own and own[-10:] in _said_digits(question)[0]:
            return "offer_phone"
        found = [kind for kind, pattern in _ASKS.items() if pattern.search(question)]
        if found:
            return found[0] if len(found) == 1 else ""
    if _offered(runtime, low):
        return "window"
    return ""


def _offered(runtime: CallRuntime, low: str) -> list[str]:
    return [w.label for w in runtime.tenant.booking_windows if w.label in low]


def _is_read_back(runtime: CallRuntime, low: str) -> bool:
    """Whether a line reads the whole booking back: the name, the street, the window."""
    s = runtime.slots
    if not s.complete:
        return False
    # The first name is enough: a read-back says "Nana", not "Nana Nanak". Live,
    # requiring the whole name missed one, and the yes to it got a second read-back.
    names = [n for n in re.findall(r"[a-z']+", s.customer_name.lower()) if len(n) > 1]
    if names and not any(re.search(rf"\b{re.escape(n)}\b", low) for n in names):
        return False
    if not all(re.search(rf"\b{re.escape(w)}\b", low) for w in s.window.lower().split() if w != "to"):
        return False
    street = [w for w in s.street_address.lower().split() if not w.isdigit() and not _STREET_WORDS.fullmatch(w)]
    return not street or any(w in low for w in street)


# ---- answers ---------------------------------------------------------------


def _on_phone(runtime: CallRuntime, text: str) -> DeskReply | None:
    """A whole number read out, and nothing else."""
    if "?" in text:
        return None
    number = phone_in_speech(text, caller_id=runtime.caller_id)
    if not number:
        return None
    if any(w not in _AROUND_A_NUMBER and w not in _DIGIT_WORDS and not w.isdigit() for w in _words(text)):
        return None
    return _read_number_back(runtime, number)


def _on_same_number(runtime: CallRuntime, text: str, asked: str) -> DeskReply | None:
    """"This one", "the same": the number they're calling from, when a number was asked for.

    Live, "This 1 only." went to the model, which called the tool before saying
    anything and took 2.1 s to read the number back.
    """
    number = normalize_phone(runtime.caller_id)
    if not number or not _SAME_NUMBER.match(text.strip().replace("’", "'").rstrip(".!")):
        return None
    if asked == "offer_phone":
        return _on_offer(runtime, "yes")  # the digits were just read to them
    return _read_number_back(runtime, number)


def _read_number_back(runtime: CallRuntime, number: str) -> DeskReply | None:
    """The number back, then the consent line or the next question."""
    digits = spoken_digits(number)
    if not digits:
        return None
    after = _after(runtime, phone=number)
    readback = f"That's [slow] {digits}."
    if _consent_needed(runtime, after):
        speech = f"{readback} {SMS_CONSENT_LINE}"
    else:
        nxt = _next_question(runtime, after)
        if not nxt:
            return None
        speech = f"{readback} {nxt}"
    return DeskReply(speech, {"phone": number}, _note(f"saved {number} as their callback number"))


def _on_offer(runtime: CallRuntime, text: str) -> DeskReply | None:
    """A yes or no to the number they're calling from."""
    answer = _yes_no(text)
    if answer is None:
        return None
    if not answer:
        return DeskReply(
            "No problem. What's the best number to reach you?",
            {},
            _note("they don't want the number they're calling from used"),
        )
    number = normalize_phone(runtime.caller_id)
    if not number:
        return None
    after = _after(runtime, phone=number)
    if _consent_needed(runtime, after):
        speech = f"{_ack(runtime)} {SMS_CONSENT_LINE}"
    else:
        nxt = _next_question(runtime, after)
        if not nxt:
            return None
        speech = f"{_ack(runtime)} {nxt}"
    return DeskReply(speech, {"phone": number}, _note(f"saved {number} as their callback number"))


def _on_sms(runtime: CallRuntime, text: str) -> DeskReply | None:
    """A yes or no to the text consent line."""
    answer = _yes_no(text)
    if answer is None:
        return None
    nxt = _next_question(runtime, _after(runtime, sms_consent=answer))
    if not nxt:
        return None
    lead = _ack(runtime) if answer else "No problem."
    said = "yes" if answer else "no, so no texts"
    return DeskReply(f"{lead} {nxt}", {"sms_consent": answer}, _note(f"text consent: {said}"))


def _on_name(runtime: CallRuntime, text: str) -> DeskReply | None:
    """Just a name, when the name was asked for and isn't known yet."""
    if runtime.slots.customer_name:
        return None
    # Speech-to-text marks a rising voice as a question: "You're talking with
    # Nanak?" is an answer. Only a mark at the very end is let through.
    name = _name_answer(runtime, text.strip().removesuffix("?"))
    if not name:
        return None
    nxt = _next_question(runtime, _after(runtime, customer_name=name))
    if not nxt:
        return None
    return DeskReply(f"Thanks, {name}. {nxt}", {"customer_name": name}, _note(f"recorded their name, {name}"))


def _on_street(runtime: CallRuntime, text: str) -> DeskReply | None:
    """A house number and street, with a served town or none."""
    got = _street_answer(runtime, text)
    if not got:
        return None
    street, town = got
    record: dict[str, object] = {"street_address": street}
    if town:
        record["town"] = town
    nxt = _next_question(runtime, _after(runtime, **record))
    if not nxt:
        return None
    return DeskReply(f"{_ack(runtime)} {nxt}", record, _note(f"recorded the street, {street}"))


def _on_town(runtime: CallRuntime, text: str) -> DeskReply | None:
    """A served town and nothing else. Anywhere else is the model's to decline."""
    if "?" in text:
        return None
    low = text.lower()
    towns = [t for t in runtime.tenant.service_area if re.search(rf"\b{re.escape(t.lower())}\b", low)]
    if len(towns) != 1:
        return None
    rest = re.sub(rf"\b{re.escape(towns[0].lower())}\b", " ", low)
    allowed = _SOFT | {"it's", "its", "in", "we're", "i'm", "we", "are", "i", "am", "the", "town", "of", "that's"}
    if any(w not in allowed for w in _words(rest)):
        return None
    nxt = _next_question(runtime, _after(runtime, town=towns[0]))
    if not nxt:
        return None
    return DeskReply(f"{_ack(runtime)} {nxt}", {"town": towns[0]}, _note(f"recorded the town, {towns[0]}"))


def _on_window(runtime: CallRuntime, text: str) -> DeskReply | None:
    """One of the windows just offered, on a day that's clear, and still open.

    The day is the one the caller said, or else the only day the offer named,
    or else the day already recorded. Two days in the offer and none in the
    answer is a guess, and goes to the model; so does a window that wasn't
    offered, or one the calendar no longer has.
    """
    if "?" in text:
        return None
    line = _plain(_last_agent_turn(runtime))
    offered = _offered(runtime, line)
    heard = heard_booking_details(text, runtime.tenant)
    if "town" in heard:
        return None
    window = heard.get("window", "")
    if window:
        # Named outright with its day -- "tomorrow, 11 to 2" -- a window needs no
        # offer; otherwise it has to be one of the windows just offered.
        if window not in offered and (offered or "day" not in heard):
            return None
        if any(not w.isdigit() and w not in _AROUND_A_WINDOW for w in _words(text)):
            return None
    elif len(offered) == 1 and _yes_no(text) is True and len(line.split()) <= 15:
        # Only a short offer ("Eight to eleven tomorrow?") is picked with a yes.
        # Replayed: a read-back of details the model never recorded named one
        # window too, and the yes to it was taken as a pick -- then the name asked for.
        window = offered[0]
        if runtime.slots.window == window and runtime.slots.day:
            return None  # a yes to what is already recorded is not a new pick
    else:
        return None

    day = heard.get("day", "")
    if not day:
        named = {d for d in _DAYS if re.search(rf"\b{d}\b", line)}
        if len(named) == 1:
            day = named.pop()
        elif not named and runtime.slots.day:
            day = runtime.slots.day
        else:
            return None
    target = _bookable(runtime, day, window)
    if target is None:
        return None

    record: dict[str, object] = {"day": day, "window": window}
    after = _after(runtime, **record)
    if after.complete:
        speech = _read_back(runtime, after, target)
    else:
        nxt = _next_question(runtime, after)
        if not nxt:
            return None
        speech = f"{_ack(runtime)} {nxt}"
    return DeskReply(speech, record, _note(f"recorded {day}, {window}"))


def _on_readback(runtime: CallRuntime, text: str) -> DeskReply | None:
    """A plain yes to the read-back: book it, and say what happens next.

    Returned with action "book", so it is spoken only once the calendar has
    taken the booking. Only a booking that went through is ever called one.
    """
    if _yes_no(text) is not True:
        return None
    slots = runtime.slots
    target = _bookable(runtime, slots.day, slots.window)
    if target is None:
        return None
    nxt = SAME_DAY_NEXT if target == runtime.now().date() else LATER_NEXT
    return DeskReply(
        f"You're all set. {nxt} Anything else before I let you go?",
        {},
        _note("booked it"),
        action="book",
    )


# ---- what comes next -----------------------------------------------------


def _next_question(runtime: CallRuntime, after: BookingSlots) -> str:
    """The next question on the prompt's list, if it is one that needs no judgment.

    The problem, the system, fuel and the window offer are the model's: each
    depends on what the caller has described. The read-back is written by
    :func:`_on_window`. And a thing the caller seems to have said already is
    never asked for -- the model records it instead.
    """
    missing = after.missing()
    if not missing:
        return ""
    nxt = missing[0]
    said = [t.get("text", "") for t in runtime.transcript if t.get("role") == "user"]
    if nxt == "customer_name":
        towns = {t.lower() for t in runtime.tenant.service_area}
        for line in said:
            for m in _INTRODUCED.finditer(line):
                if m[1].lower() not in towns and m[1].lower() not in _NOT_NAMES:
                    return ""
        return "And can I get your name?"
    if nxt == "phone":
        if any(len(_said_digits(line)[0]) >= 6 for line in said):
            return ""
        caller = normalize_phone(runtime.caller_id)
        digits = spoken_digits(caller) if caller else ""
        if digits:
            return f"Is [slow] {digits} the best number to reach you?"
        return "What's the best number to reach you?"
    if nxt == "street_address":
        if any(_STREET_WORDS.search(line) and re.search(r"\d", line) for line in said):
            return ""
        return "And what's the street address there?"
    if nxt == "town":
        # Replayed: the caller had said Wallingford twice and the model never
        # recorded it; asking a third time is the tell the slots exist to stop.
        towns = [t.lower() for t in runtime.tenant.service_area]
        if after.rejected_town or any(
            re.search(rf"\b{re.escape(t)}\b", line.lower()) for line in said for t in towns
        ):
            return ""
        return "And what town is that in?"
    return ""


def _read_back(runtime: CallRuntime, after: BookingSlots, target: Date) -> str:
    """Name, street and town, the problem, the day and window -- the prompt's four things."""
    # The complaint, not the story: "Heater not heating properly, tried turning it off".
    problem = after.problem.split(",")[0].strip().rstrip(".")
    if problem[:2] != problem[:2].upper():
        problem = problem[:1].lower() + problem[1:]
    when = _day_name(target, runtime.now().date())
    return (
        f"So that's {after.customer_name}, {after.street_address} in {after.town}, "
        f"{problem}, {when}, {after.window}. Does that all sound right?"
    )


# ---- helpers ---------------------------------------------------------------


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("’", "'"))


def _yes_no(text: str) -> bool | None:
    """True for a plain yes, False for a plain no, None for anything else."""
    if "?" in text:
        return None
    words = _words(text)
    if not words or len(words) > 6:
        return None
    if any(w not in _YES and w not in _NO and w not in _AROUND_YES_NO for w in words):
        return None
    yes, no = any(w in _YES for w in words), any(w in _NO for w in words)
    return None if yes == no else yes


def _name_answer(runtime: CallRuntime, text: str) -> str:
    """'Yeah, it's Nanak.' -> 'Nanak'. One to three plain words, or ''.

    Spelled out, the letters win over whatever the caller's spoken attempt was
    heard as: "It's n a n a k, Nak." is Nanak.
    """
    if "?" in text:
        return ""
    spelled = spelled_name(text)
    if spelled:
        # "You're talking with Nana n a n a k." -- the lead-in is not part of the name.
        lead_free = _NAME_LEAD.sub("", text.strip().replace("’", "'"), count=1)
        rest = _words(_SPELLED.sub(" ", lead_free))
        close = lambda w: difflib.SequenceMatcher(None, w, spelled.lower()).ratio() >= 0.5  # noqa: E731
        return spelled if all(w in _AROUND_A_SPELLING or close(w) for w in rest) else ""
    stripped = _NAME_LEAD.sub("", text.strip().replace("’", "'"), count=1).strip(" .,!")
    words = stripped.split()
    if not 1 <= len(words) <= 3:
        return ""
    towns = {w for t in runtime.tenant.service_area for w in t.lower().split()}
    for w in words:
        if not re.fullmatch(r"[A-Za-z][A-Za-z'-]+", w):
            return ""  # digits, or a name spelled out letter by letter
        if w.lower() in _NOT_NAMES or w.lower() in towns or w.lower() in _DIGIT_WORDS:
            return ""
    return " ".join(w[0].upper() + w[1:] for w in words)


def _street_answer(runtime: CallRuntime, text: str) -> tuple[str, str] | None:
    """'It's 1 23 Coffey Street in Wallingford.' -> ('123 Coffey Street', 'Wallingford')."""
    if "?" in text:
        return None
    stripped = _STREET_LEAD.sub("", text.strip().replace("’", "'"), count=1).strip(" .,!")
    m = _STREET_ANSWER.match(stripped) or _STREET_NAME_FIRST.match(stripped)
    if not m:
        return None
    town = ""
    if m["town"]:
        town = next((t for t in runtime.tenant.service_area if t.casefold() == m["town"].strip().casefold()), "")
        if not town:
            return None
    house = re.sub(r"\s+", "", m["house"])
    return f"{house} {m['name']} {m['suffix']}", town


def _bookable(runtime: CallRuntime, day: str, window: str) -> Date | None:
    """The date a day names, if that window is still open on it."""
    cal = runtime.calendar
    if cal is None or not day or not window:
        return None
    now = runtime.now()
    try:
        target = resolve_day(runtime.tenant, day, now=now)
    except BookingError:
        return None
    return target if window in cal.availability(runtime.tenant, target, now=now) else None


def _after(runtime: CallRuntime, **values: object) -> BookingSlots:
    """The booking as it would stand with these recorded, without recording them."""
    after = copy.copy(runtime.slots)
    after.update(**values)
    return after


def _consent_needed(runtime: CallRuntime, after: BookingSlots) -> bool:
    return bool(runtime.tenant.sms_consent_required and after.sms_consent is None)


def _agent_lines(runtime: CallRuntime) -> list[str]:
    return [_TAG.sub("", t.get("text", "")).strip() for t in runtime.transcript if t.get("role") == "agent"]


def _ack(runtime: CallRuntime) -> str:
    """A short acknowledgment the agent hasn't opened a line with yet."""
    used = _agent_lines(runtime)
    for word in ("Perfect.", "Great.", "Okay."):
        if not any(line.startswith(word) for line in used):
            return word
    return "Okay."


def _said_before(runtime: CallRuntime, speech: str) -> bool:
    def norm(s: str) -> str:
        return " ".join(_words(_TAG.sub("", s)))

    target = norm(speech)
    return any(norm(line) == target for line in _agent_lines(runtime))


def _note(what: str) -> str:
    return f"(Not said by the caller: the desk {what}, and has already answered them.)"
