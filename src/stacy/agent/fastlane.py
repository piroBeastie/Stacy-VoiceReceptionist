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
    FUEL_SYSTEMS,
    _SPELLED,
    BookingSlots,
    _said_digits,
    spelled_name,
    heard_booking_details,
    normalize_phone,
    phone_in_speech,
    spoken_digits,
)
from stacy.agent.tools import _day_name, _sentence, _spoken_list
from stacy.desk.calendar import BookingError, resolve_day

if TYPE_CHECKING:
    from stacy.agent.runtime import CallRuntime
    from stacy.tenants.models import Service, Tenant

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
    "day": re.compile(
        rf"\b(?:{'|'.join(_DAYS)})\b[^?]{{0,40}}\bor\b[^?]{{0,40}}\b(?:{'|'.join(_DAYS)}|later in the week|next week)\b"
        r"|\b(?:what|which) day (?:works|would work|is best|is good|suits)\b|\bwhen would (?:work|suit)\b|\bwhen works\b"
    ),
}
_NOT_THE_VISIT = re.compile(
    r"\b(?:since|start\w*|stop\w*|quit|went|gone|go out|has it|had it|did it|was it|were|happen\w*"
    r"|last|ago|wait|call\w*|reach|hear)\b"
)
"""Words that make a two-day question about something other than the visit: "Did it
quit today, or has it been out since Sunday?", "Today, or tomorrow morning?" for a callback."""
_CLOSE = re.compile(r"\b(?:call you|reach you|talk to|set that up)\b")
_SEPARATOR = re.compile(r",|;|—|\.|\bor\b|\band\b")
"""What ends the phrase a day and a window share: "today two to five, or tomorrow eight to eleven"."""
_EQUIPMENT = re.compile(
    r"\b(?:ac|a/c|air condition\w*|heat|heater|heat pump|hot water|furnace|boiler|pump|mini|split|unit"
    r"|system|thermostat)\b",
    re.I,
)
"""Words that already name the equipment in a problem: the read-back adds no system to these.

Not a symptom: "not heating" names nothing, and reads back as "boiler not heating"."""
_FUEL_SAID = re.compile(r"\b(?:gas|propane|oil|electric)\b", re.I)
_CLOSING = re.compile(r"^(?:is there )?anything else(?: i can (?:help|do)[^?]*| before i let you go)?\?$")
"""The closing question alone: not "Is there anything else going on with it?"."""
_A_NUMBER = re.compile(r"\b(?:number|house number)\b")
SAME_DAY_LEAD_MIN = 90
"""A same-day window is offered only with at least this many minutes left in it."""
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
    if _for_the_reseller(runtime, line) or _in_the_close(runtime):
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
            "day": _on_day,
        }.get(asked)
        reply = handler(runtime, text) if handler else None
    if reply is None and asked != "readback":
        # "Go ahead and book it", a couple of turns after the read-back. Live,
        # the model answered that with "you're all set" and never called the
        # tool: the caller rang off believing she had an appointment that did
        # not exist. The desk books it instead, and only says so afterwards.
        reply = _on_book_now(runtime, text)
    if reply is None:
        # A question about the business itself, answered from its own record.
        reply = _on_question(runtime, text)
    if reply is None and asked == "window":
        # An offer of windows answered with just a day: that day's windows, or
        # the one window the offer paired with it.
        reply = _on_day(runtime, text)
    if reply is None and not asked:
        # Live: asked "today, or tomorrow?", the caller said "tomorrow morning, 11
        # to 2" -- a day and a window, nothing to guess. A day on its own after
        # some other question ("When did it stop working?" "Today.") is not one.
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


def answer_kind(runtime: CallRuntime) -> str:
    """How long to wait for the answer to the agent's last line: "patient", "eager" or "".

    Patient for a phone number, a street or a name -- and for the offer of their
    own number, often answered with a different one -- because callers read
    those out in groups with pauses between them. Eager only where committing
    early is cheap: the text-consent question and the closing "Anything else?".
    Not a read-back: "Yeah, ... but the street's wrong" has to be heard out,
    and a yes there books. ElevenLabs calls this turn eagerness.
    """
    line = _last_agent_turn(runtime)
    low = _plain(line)
    if not low.endswith("?"):
        return ""
    asked = _asked(runtime, line)
    question = re.split(r"(?<=[.!?])\s+", low)[-1]
    # The final question is judged on its own as well: _asked gives up on a turn
    # with two questions in it (a retry after a cut-off line), and on any line
    # naming the reseller -- rules that keep the desk's answers right, not waits.
    own = "".join(ch for ch in normalize_phone(runtime.caller_id) if ch.isdigit())
    if (
        asked in ("phone", "street", "name", "offer_phone")
        or "spell" in low
        or any(_ASKS[k].search(question) for k in ("phone", "name", "street"))
        or (own and own[-10:] in _said_digits(question)[0])
    ):
        return "patient"
    if asked == "sms" or "okay if we text" in question:
        # Not straight after digits read back: "No. [pause] It's zero-one-nine-
        # eight" is a correction, and cut at the "No." it was recorded as no texts.
        before = low.split("we'll text")[0]
        return "" if len(_said_digits(before)[0]) >= 7 else "eager"
    if _A_NUMBER.search(question):
        return "patient"  # "Sorry, what was the number again?", "And the house number?"
    if _CLOSING.match(question):
        return "eager"
    return ""


def _for_the_reseller(runtime: CallRuntime, line: str) -> bool:
    """A line naming the reseller belongs to the demo close, where a number is for a callback."""
    reseller = runtime.tenant.reseller
    if reseller is None:
        return False
    low = line.lower()
    names = {reseller.contact_spoken_name.lower(), reseller.contact_full_name.split()[0].lower()}
    # Not the bare first name when it is the caller's own: "Thanks, Miles."
    caller = set(re.findall(r"[a-z']+", runtime.slots.customer_name.lower()))
    names = {n for n in names if n and not (" " not in n and n in caller)}
    return any(re.search(rf"\b{re.escape(n)}\b", low) for n in names)


def _in_the_close(runtime: CallRuntime) -> bool:
    """Whether the demo close has begun: the agent has offered the reseller's callback.

    Its "Today, or tomorrow morning?" is about the callback, not a visit, and
    names no reseller for the line check to catch. Review: the desk answered it
    with booking windows.
    """
    if runtime.tenant.reseller is None:
        return False
    reseller = runtime.tenant.reseller
    names = [n.lower() for n in (reseller.contact_spoken_name, reseller.contact_full_name) if n]
    # By the name as said, "Mister Miles": a customer called Miles hears "Thanks,
    # Miles. ...to reach you?" from the desk itself, and that switched it off.
    return any(
        turn.get("role") == "agent"
        and _CLOSE.search(_plain(turn.get("text", "")))
        and any(re.search(rf"\b{re.escape(n)}\b", turn.get("text", "").lower()) for n in names)
        for turn in runtime.transcript
    )


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
        if "day" in found and _NOT_THE_VISIT.search(question):
            found.remove("day")
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
    return DeskReply(f"Thanks, {_said_back(text, name)}. {nxt}", {"customer_name": name},
                     _note(f"recorded their name, {name}"))


def _said_back(text: str, name: str) -> str:
    """The name, and its letters after it when the caller spelled them.

    Three live calls in a row had a spelled name heard wrong -- "Ananak",
    "Nanat", "Nank" for Nanak -- and each one was booked and texted that way,
    because nothing in the call ever showed the caller the spelling. Letters are
    read back separated by commas, which is how the voice reads them as letters
    rather than as a word, and it costs no turn: the next question follows in
    the same breath, and a caller whose name came back wrong corrects it there.
    """
    if not spelled_name(text):
        return name
    return f"{name} -- {', '.join(name.upper())}"


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
    # Said back, not just acknowledged: "Okay, 123 Coffee Street." A reviewer
    # heard the bare "Perfect." before every question as curt.
    return DeskReply(f"Okay, {street}. {nxt}", record, _note(f"recorded the street, {street}"))


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

    The day is the one the caller said, or else the day the offer paired with
    that window, or else the day already recorded. A window whose day isn't
    clear, one that wasn't offered, or one the calendar no longer has goes to
    the model.
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
    elif len(offered) == 1 and _yes_no(text) is True and len(line.split()) <= 15 and not re.search(r"\bor\b", line):
        # Only a short offer ("Eight to eleven tomorrow?") is picked with a yes.
        # Replayed: a read-back of details the model never recorded named one
        # window too, and the yes to it was taken as a pick -- then the name asked for.
        # Never an offer with an "or": a yes to "eight to eleven tomorrow, or
        # would Thursday be better?" could mean either.
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
        elif named:
            # "Tomorrow I've got eight to eleven. Does that beat today?" -- the
            # day beside the window, never the day recorded before the offer.
            day = _day_for(line, window, offered)
        elif runtime.slots.day:
            day = runtime.slots.day
        if not day:
            return None
    return _pick(runtime, day, window)


# ---- what the business already knows --------------------------------------

_ABOUT = {
    "price": re.compile(
        r"\b(?:how much|what(?:'s| is| does| do)?[^?]{0,20}\b(?:cost|charge|run)|price|pricing"
        r"|fee|rate|expensive)\b"
    ),
    "hours": re.compile(r"\b(?:what time|hours|open|close[sd]?)\b"),
    "area": re.compile(r"\b(?:service area|do you (?:come|cover|service|go|serve)|how far)\b"),
    "soon": re.compile(
        r"\b(?:how soon|how (?:quickly|fast)|earliest|first (?:available|opening)"
        r"|when can (?:you|someone|somebody|a tech))\b"
    ),
}
_THEIR_OWN = re.compile(
    r"\bhow (?:much )?long(?:er)?\b"
    r"|\bmy (?:appointment|booking|visit|slot)\b"
    r"|\bthe (?:appointment|booking|slot)\b"
    r"|\bwhat time (?:is|are|will|does|do) (?:my|the|they|he|she|i)\b"
    r"|\b(?:when|what time) (?:is|are) (?:they|he|she|someone|the tech) (?:coming|getting)\b"
)
"""Asked about their own booking rather than about the business."""

"""Questions whose answer is in the tenant file, so the model never has to invent one.

Live, the model answered a price question correctly and a staffing question
wrongly -- "we send whoever's available that day", where the file lists one
technician. An answer read out of the file cannot be wrong, and it arrives in
the time a desk reply takes rather than the time a model takes.
"""


def _on_question(runtime: CallRuntime, text: str) -> DeskReply | None:
    """Answer a question about the business from its own record, then carry on.

    Only what the file states outright: the priced service, the opening hours,
    the towns, the first open window, and any FAQ whose wording the caller
    echoes. Anything else -- a warranty, who the technician will be, whether
    they can make an exception -- is the model's, because the file does not say.
    """
    if "?" not in text and not _ABOUT["price"].search(_plain(text)):
        return None  # a question, or a plain "how much" the transcriber left unmarked
    low = _plain(text)
    if _THEIR_OWN.search(low):
        # "How much longer?", "What time is my appointment?" -- about this
        # caller's own visit, not about the business. The model has the call.
        return None
    answer = _faq_answer(runtime, low) or _fact(runtime, low)
    if not answer:
        return None
    after = _after(runtime)
    nxt = _next_question(runtime, after) if after.missing() else ""
    return DeskReply(f"{answer} {nxt}".strip(), {}, _note("answered that from the tenant record"))


def _fact(runtime: CallRuntime, low: str) -> str:
    tenant = runtime.tenant
    if _ABOUT["price"].search(low):
        priced = _priced(tenant, low)
        if priced is None:
            return ""
        note = f", {priced.note}" if priced.note else ""
        return f"{_sentence(priced.name)} runs {priced.price_low} to {priced.price_high} dollars{note}."
    if _ABOUT["area"].search(low):
        return f"We cover {_spoken_list(list(tenant.service_area), 'and')}."
    if _ABOUT["hours"].search(low):
        return _hours_line(tenant)
    if _ABOUT["soon"].search(low):
        return _soonest(runtime)
    return ""


def _priced(tenant: Tenant, low: str) -> Service | None:
    """The service the caller asked the price of, or None to leave it to the model.

    A price is only ever read out for the thing the file prices by name. "What
    does a service call run?" is the diagnostic visit; "a firm price for the
    part" is nothing in the file, and a canned range in its place would answer a
    question the caller did not ask.
    """
    priced = [s for s in tenant.services if s.price_low and s.price_high]
    asked = set(_words(low))
    scored = sorted(((len(_service_words(s.name) & asked), s) for s in priced), key=lambda p: -p[0])
    if scored and scored[0][0] and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        # "a furnace repair" beats "air conditioning repair" on the word furnace;
        # a bare "a repair" ties them, and a tie is the model's to sort out.
        return scored[0][1]
    if scored and scored[0][0]:
        return None
    visit = next((s for s in priced if "visit" in s.name.lower()), None)
    return visit if _A_VISIT.search(low) else None


_A_VISIT = re.compile(
    r"\b(?:service call|call ?out|come out|coming out|visit|diagnostic|send someone|trip charge"
    r"|to look at it|look at it)\b"
)
"""How a caller asks what it costs to get someone there at all."""


def _service_words(name: str) -> set[str]:
    return {w for w in _words(name.lower()) if w not in ("visit", "service", "system")}


def _faq_answer(runtime: CallRuntime, low: str) -> str:
    """A tenant FAQ the caller has asked in their own words, answered verbatim.

    Matched on the uncommon words the question and the FAQ share, so "do you do
    mini-splits?" finds the mini-split entry and nothing finds the wrong one.
    """
    asked = {w for w in _words(low) if len(w) > 3 and w not in _COMMON}
    if not asked:
        return ""
    for faq in runtime.tenant.faqs:
        theirs = {w for w in _words(faq.question.lower()) if len(w) > 3 and w not in _COMMON}
        if theirs and len(asked & theirs) >= max(2, len(theirs) // 2):
            return faq.answer
    return ""


_COMMON = frozenset(
    "does your this that they them with have what when will would could should about there here "
    "much many take does your from into been were they".split()
)


def _hours_line(tenant: Tenant) -> str:
    """The week in one sentence, read off the tenant's own hours."""
    open_days = [d for d in tenant.hours if d.open_at and d.close_at]
    if not open_days:
        return ""
    weekdays = [d for d in open_days if d.day not in ("saturday", "sunday")]
    parts = []
    if weekdays:
        first, last = weekdays[0], weekdays[-1]
        same = all((d.open_at, d.close_at) == (first.open_at, first.close_at) for d in weekdays)
        when = f"{_oclock(first.open_at)} to {_oclock(first.close_at)}"
        parts.append(f"{first.day.title()} to {last.day.title()} {when}" if same else "weekdays")
    for name in ("saturday", "sunday"):
        day = next((d for d in open_days if d.day == name), None)
        if day:
            parts.append(f"{name.title()} {_oclock(day.open_at)} to {_oclock(day.close_at)}")
    shut = [d.day.title() for d in tenant.hours if not d.open_at]
    line = f"We're open {_spoken_list(parts, 'and')}."
    return f"{line} Closed {_spoken_list(shut, 'and')}." if shut else line


def _oclock(clock: str) -> str:
    """'08:00' as a person says it: 'eight', 'five-thirty'."""
    hour, _, minute = clock.partition(":")
    said = _HOURS[int(hour) % 12 or 12]
    return said if minute in ("00", "") else f"{said}-{_MINUTES.get(minute, minute)}"


_HOURS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
          9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
_MINUTES = {"15": "fifteen", "30": "thirty", "45": "forty-five"}


def _soonest(runtime: CallRuntime) -> str:
    """The first window the calendar actually has, today or tomorrow."""
    for day in ("today", "tomorrow"):
        windows = _open_windows(runtime, day)
        if windows:
            when = "today" if day == "today" else "tomorrow"
            return f"The earliest I've got is {when}, {windows[0]}."
    return ""


def _pick(runtime: CallRuntime, day: str, window: str) -> DeskReply | None:
    """A day and window the caller chose: the read-back if that's everything, else the next question."""
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


def _day_for(line: str, window: str, offered: list[str]) -> str:
    """The day an offer paired with a window, or '' if that isn't clear.

    A day belongs to a window when they share a phrase: nothing between them
    but plain words -- no comma, "or", or other window. "Today two to five, or
    tomorrow eight to eleven" and "eleven to two today, or eight to eleven
    tomorrow" both pair cleanly; "today or tomorrow, I've got two to five"
    pairs with neither, and so does a window with a day on both sides. Review:
    taking the last day before the window read "eleven to two today, or eight
    to eleven tomorrow" as eight to eleven today.
    """
    at = line.find(window)
    if at < 0 or line.count(window) > 1:
        return ""
    end = at + len(window)
    others = [line.find(w) for w in offered if w != window and w in line]
    days = [(m.start(), m.end(), m[0]) for d in _DAYS for m in re.finditer(rf"\b{d}\b", line)]

    def clear(a: int, b: int) -> bool:
        return not _SEPARATOR.search(line[a:b]) and not any(a <= s < b for s in others)

    left = [d for s, e, d in days if e <= at and clear(e, at)]
    right = [d for s, e, d in days if s >= end and clear(end, s)]
    return (left + right)[0] if len(left) + len(right) == 1 else ""


def _only_day(line: str) -> str:
    named = {d for d in _DAYS if re.search(rf"\b{d}\b", line)}
    return named.pop() if len(named) == 1 else ""


def _part_of_day(text: str) -> str:
    """Morning or afternoon, if the caller said so. "Evening" is the model's: no window is one."""
    low = text.lower()
    if re.search(r"\bmorning\b", low):
        return "morning"
    if re.search(r"\bafternoon\b", low):
        return "afternoon"
    return ""


def _in_part(runtime: CallRuntime, window: str, part: str) -> bool:
    """Whether a window is a morning or an afternoon one: starts before noon, or ends at two or later."""
    if not part:
        return True
    bounds = {w.label: (w.start, w.end) for w in runtime.tenant.booking_windows}
    start, end = bounds.get(window, ("00:00", "23:59"))
    return start < "12:00" if part == "morning" else end >= "14:00"


def _on_day(runtime: CallRuntime, text: str) -> DeskReply | None:
    """A day -- "Tomorrow." -- once only the window is left.

    The one window the offer paired with that day is a pick of it; otherwise
    that day's open windows, only the morning or afternoon ones if they said
    so. Only ever after a question about days or an offer of windows: "Today."
    to "When did it stop working?" is not a booking day.
    """
    if "?" in text:
        return None
    heard = heard_booking_details(text, runtime.tenant)
    if heard.get("window"):
        return _on_window(runtime, text)  # "tomorrow, 11 to 2": a day and a window
    day = heard.get("day", "")
    if not day or "town" in heard:
        return None
    if any(not w.isdigit() and w not in _AROUND_A_WINDOW for w in _words(text)):
        return None
    if len({d for d in _DAYS if re.search(rf"\b{d}\b", text.lower())}) > 1:
        return None  # "Tomorrow and Thursday.": two days is a choice for the model
    if _after(runtime, day=day).missing() != ["window"]:
        return None
    line = _plain(_last_agent_turn(runtime))
    offered = _offered(runtime, line)
    part = _part_of_day(text)
    # "...or would Thursday be better?" "Thursday.": the day offered instead of
    # the windows, so its own windows, never the ones offered for another day.
    # Not "...or tomorrow eight to eleven": that day comes with its window.
    instead = re.search(
        rf"\bor\s+(?:(?:would|does|is|how about|what about|maybe)\s+)?{day}\b", line
    ) and not any(_day_for(line, w, offered) == day for w in offered)
    pairs = {} if instead else {w: _day_for(line, w, offered) or _only_day(line) for w in offered}
    mine = [w for w, d in pairs.items() if d == day]
    if (
        len(mine) == 1
        and all(d not in ("", day) for w, d in pairs.items() if w != mine[0])
        and _in_part(runtime, mine[0], part)
    ):
        # "I've got eight to eleven tomorrow, or would Thursday be better?"
        # "Tomorrow." is a yes to eight to eleven: the one window offered for
        # that day, every other one clearly another day's. Review: "Tomorrow."
        # to "...or tomorrow morning eight to eleven, eleven to two, or two to
        # five" took the first of the three; "Tomorrow evening." to "Eight to
        # eleven tomorrow?" took eight to eleven.
        return _pick(runtime, day, mine[0])
    windows = _open_windows(runtime, day, part)
    if not windows:
        return None  # nothing open that day, or that part of it: the model says so
    if _only_day(line) == day and set(windows) <= set(offered):
        return None  # that day's windows were just offered: listing them again helps no one
    return DeskReply(_windows_line(runtime, day, windows), {"day": day}, _note(f"recorded {day}"))


_BOOK_ASK = re.compile(
    r"\b(?:book|booked|booking|lock (?:it|that) in|set (?:it|that) up|get you (?:in|booked))\b"
)
"""The agent offering to book, or the caller asking for it."""
_GO_AHEAD = re.compile(
    r"\b(?:go ahead|book it|book me|book that|do it|lock it in|let'?s do it|sounds good)\b"
)


def _on_book_now(runtime: CallRuntime, text: str) -> DeskReply | None:
    """Book what was already read back, when the caller now asks for it.

    Everything must already be recorded and already have been read back to them
    in this call -- this is never the first time they hear the details. Then
    either they say to book it, or the agent offered and they said yes. A bare
    yes to anything else is not this: it books nothing.
    """
    if runtime.slots.missing() or not _read_back_said(runtime):
        return None
    # The question they are answering, not the whole turn: an earlier sentence
    # of it said "waived if you book the repair", and a yes to "is that a gas
    # furnace?" was booking the visit.
    question = re.split(r"(?<=[.!?])\s+", _plain(_last_agent_turn(runtime)))[-1]
    asked_to_book = question.endswith("?") and bool(_BOOK_ASK.search(question)) and _yes_no(text) is True
    told_to_book = bool(_GO_AHEAD.search(_plain(text))) and _yes_no(text) is not False
    if not (asked_to_book or told_to_book):
        return None
    return _on_readback(runtime, "yes")


def _read_back_said(runtime: CallRuntime) -> bool:
    """Whether the booking now recorded has been read back to the caller already."""
    agent_lines = [t.get("text", "") for t in runtime.transcript if t.get("role") == "agent"]
    return any(_is_read_back(runtime, _plain(line)) for line in agent_lines[-6:])


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

    The problem and the system type are the model's, and so is the first offer
    of a time: that is where an emergency is dispatched instead of booked, and
    the review caught the desk offering a routine window to a freezing house.
    The read-back is written by _pick. A thing the caller seems to have said
    already is never asked for -- the model records it instead.
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
    if nxt == "fuel":
        # Only for a furnace or a boiler (missing() sees to that), and never when
        # they have said it already: "my oil boiler won't fire". Never twice
        # either: review found "I don't know" to it answered with the same
        # question, the loop the prompt's don't-know rule exists to stop.
        if _FUEL_SAID.search(after.system_type) or any(_FUEL_SAID.search(line) for line in said):
            return ""
        asked = [_plain(t.get("text", "")) for t in runtime.transcript if t.get("role") == "agent"]
        if any(re.search(r"\bgas\b", line) and re.search(r"\belectric\b", line) for line in asked):
            return ""
        system = next((s for s in FUEL_SYSTEMS if s in after.system_type.lower()), "furnace")
        return f"Is that a gas {system} or an electric one?"
    return ""


def _windows_on(runtime: CallRuntime, day: str, *, part: str = "") -> str:
    """The offer of a day's open windows, or '' if none."""
    windows = _open_windows(runtime, day, part)
    return _windows_line(runtime, day, windows) if windows else ""


def _windows_line(runtime: CallRuntime, day: str, windows: list[str]) -> str:
    now = runtime.now()
    when = _sentence(_day_name(resolve_day(runtime.tenant, day, now=now), now.date()))
    if len(windows) == 1:
        return f"{when} I've got {windows[0]} open. Does that work?"
    return f"{when} I've got {_spoken_list(windows)}. Which works best?"


def _open_windows(runtime: CallRuntime, day: str, part: str = "") -> list[str]:
    """The windows still worth offering on a day the caller picked.

    Today's windows need an hour and a half left in them: the calendar keeps a
    window open until it ends, and "the tech will call thirty minutes before
    they head over" can't be kept at ten to eleven for eight to eleven.
    """
    cal = runtime.calendar
    if cal is None:
        return []
    now = runtime.now()
    try:
        target = resolve_day(runtime.tenant, day, now=now)
    except BookingError:
        return []
    bounds = {w.label: (w.start, w.end) for w in runtime.tenant.booking_windows}
    windows = []
    for label in cal.availability(runtime.tenant, target, now=now):
        start, end = bounds.get(label, ("00:00", "23:59"))
        if target == now.date():
            hour, minute = (int(x) for x in end.split(":"))
            if (hour * 60 + minute) - (now.hour * 60 + now.minute) < SAME_DAY_LEAD_MIN:
                continue
        if not _in_part(runtime, label, part):
            continue
        windows.append(label)
    return windows


def _read_back(runtime: CallRuntime, after: BookingSlots, target: Date) -> str:
    """Name, street and town, the problem, the day and window -- the prompt's four things."""
    # The complaint, not the story: "Heater not heating properly, tried turning it off".
    problem = after.problem.split(",")[0].strip().rstrip(".")
    if problem[:2] != problem[:2].upper():
        problem = problem[:1].lower() + problem[1:]
    # Live: "...in Wallingford, not working, tomorrow...". A problem that names
    # no equipment gets the system: "furnace not working". One that does is
    # left alone -- "air conditioner AC not cooling" is what a label sounds like.
    system = after.system_type.strip()
    # Only equipment: a system recorded as "unknown" would be read back as "unknown not working".
    if system and _EQUIPMENT.search(system) and not _EQUIPMENT.search(problem) and len(problem.split()) <= 3:
        problem = f"{system} {problem}"
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
