"""Ending the call, which until now only the caller could do.

A receptionist says "bye", waits a beat, and puts the phone down. Stacy said
"bye" and then sat there: the prompt admitted it outright -- "you have no way to
hang up on anyone" -- and the line stayed open until the caller hung up or the
silence watchdog gave up thirty seconds later. On a mobile that half-minute of
nothing is what the caller remembers, and on a demo it reads as a bug.

Two ways to end it, and the difference matters:

* **The farewell both people have already said.** This is the normal one, and it
  is decided in code from the transcript, not by the model: the agent said a
  goodbye, the caller had said they were done, and the goodbye has finished
  playing. Nothing is inferred and nothing is asked of the model.
* **:meth:`DeskOperations.end_call`, for a call that is not a customer.** A
  wrong number, a sales call, somebody abusive. A human ends those; so can
  Stacy, after saying one line. It refuses while a booking is half-collected,
  because hanging up on someone mid-address is the worst version of this
  feature, and it refuses outright while the gas-leak script is latched.

Both paths drain the speech queue first. Cutting the SIP leg while the goodbye
is still being spoken is the same bug as not hanging up at all, heard from the
other end.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger("stacy.agent")

_COURTESY = re.compile(
    r"(?:\b(?:thanks?(?: you)?(?: so much| very much)?|cheers|appreciate it|much appreciated)\b"
    r"[\s,.!]*)+$",
    re.I,
)
"""A thank-you on the end of a sign-off. Stripped before the sign-off is read.

"No, that's it. Thanks." is one of the commonest ways a call ends, and anchoring
the whole turn missed every one of them.
"""

_SIGN_OFF = re.compile(
    r"(?:bye(?:[- ]?bye)?|goodbye|good\s?bye|take care|see (?:you|ya)"
    r"|that'?s (?:it|all)|nothing else|i'?m (?:good|all set)|we'?re (?:good|all set)|all good"
    r"|forget it|not interested|have a good (?:one|day|night|evening))",
    re.I,
)
"""What a caller says to end a call, matched against their whole last clause.

The clause has to *be* the sign-off, not merely end with one. Transcription on
an 8 kHz line hears names as all sorts of things, and "My name is Bye" has to
not hang up on somebody who is introducing themselves -- which is exactly what
matching the tail alone did.
"""

_LEAD = re.compile(
    r"^(?:(?:okay|ok|alright|right|well|yeah|yep|yes|no|nope|nah|um|uh|oh|so)\b[\s,.!]*)+",
    re.I,
)
"""Filler in front of a sign-off: "okay, bye", "no, that's it"."""

_AGENT_FAREWELL = re.compile(
    r"\b(?:bye|goodbye|take care|good luck with|have a good (?:one|day|night|evening)"
    r"|stay warm|talk soon)\b",
    re.I,
)
"""The agent's own sign-off, which the prompt only ever says at the very end."""


def caller_signed_off(said: str) -> bool:
    """Whether this caller turn is them ending the call rather than answering.

    Read from the last clause, and the clause has to be nothing but the
    sign-off. Three turns say why it is drawn this finely:

    * "No, that's it. Thanks." -- a sign-off, with the thanks stripped first.
    * "Bye, but first -- how much is it?" -- not one. The last clause is a
      question, and answering it is the job.
    * "My name is Bye" -- not one either. Deepgram hears names as all sorts of
      things on an 8 kHz line, and hanging up on somebody introducing
      themselves is the worst thing this function could do.
    """
    spoken = (said or "").strip()
    if not spoken:
        return False
    without_thanks = _COURTESY.sub("", spoken).strip()
    if not without_thanks:
        return True  # "Thanks." on its own, after a goodbye, is the goodbye

    clause = _last_clause(without_thanks)
    return bool(clause) and bool(_SIGN_OFF.fullmatch(_LEAD.sub("", clause).strip(" ,.!-")))


def _last_clause(spoken: str) -> str:
    """The final clause of a turn: what comes after the last stop or comma."""
    pieces = [piece.strip(" -—") for piece in re.split(r"[.!?,;]+", spoken)]
    said = [piece for piece in pieces if piece]
    return said[-1] if said else ""


BOOKING_SLOTS = ("customer_name", "phone", "street_address", "town", "problem")
"""The details a caller only gives when they want something booked."""


def business_unfinished(runtime: Any) -> list[str]:
    """What this call started and has not finished, as a list of slot names.

    Holding somebody's name and address with nothing booked and no message
    taken means the call had a purpose that was not served. Both ways of
    ending a call ask this question, so they are guaranteed to agree on the
    answer -- the version where `end_call` refused and the automatic farewell
    cut anyway would be a bug nobody would find until a caller complained.

    A caller who gave nothing -- a wrong number, somebody asking the opening
    hours -- is finished the moment they say so, and gets an empty list.
    """
    if runtime.appointments or runtime.outcome == "message":
        return []
    return [field for field in BOOKING_SLOTS if getattr(runtime.slots, field, "")]


def farewell_said(runtime: Any) -> str:
    """Why this call is over, or '' if it is not.

    Every one of these has to hold, and each one is a call that went wrong
    before the check existed:

    1. The caller said they were done, in their own last turn.
    2. The agent's reply was a sign-off, and the agent spoke last.
    3. Neither of those two turns left a question hanging.
    4. The call finished what it started -- booked, dispatched, or a message
       taken -- or never started anything.

    Four is the one worth stating plainly: **the line is never cut before the
    booking is made.** A caller who gave their name and address and has no
    appointment did not get what they rang for, and dropping them at that
    point loses the job and the customer. If they genuinely want to leave
    anyway, the model can say so with `end_call`, which is allowed to let a
    caller go once they have signed off. That is a decision with a sentence
    behind it; this function will not make it by accident.
    """
    if getattr(runtime.safety, "latched", False):
        return ""  # the gas script owns the line until the caller leaves it

    agent_lines = [t.get("text", "") for t in runtime.transcript if t.get("role") == "agent"]
    caller_lines = [t.get("text", "") for t in runtime.transcript if t.get("role") == "user"]
    if not agent_lines or not caller_lines:
        return ""
    if runtime.transcript[-1].get("role") != "agent":
        return ""  # the caller spoke last; they are still on the call
    if not _AGENT_FAREWELL.search(agent_lines[-1]):
        return ""
    if not caller_signed_off(caller_lines[-1]):
        return ""

    # Nothing is cut over an open question. The agent saying "take care" in
    # the same breath as asking something still wants an answer, and a caller
    # who tacked a question onto their goodbye still wants it answered.
    if agent_lines[-1].rstrip().endswith("?"):
        return ""
    if "?" in caller_lines[-1]:
        return ""

    # And nothing is cut before the thing the caller rang about is done.
    if business_unfinished(runtime):
        return ""
    return "both said goodbye"


async def hang_up(ctx: Any, session: Any, runtime: Any, reason: str) -> bool:
    """Finish speaking, then drop the call. True if the line was actually cut.

    `drain` is what makes this polite: it waits for the goodbye to play. The
    caller gets one last chance to interrupt it -- if they start talking while
    it drains, the hangup is abandoned, because "oh wait, one more thing" is a
    real thing people say and cutting it off is unforgivable.
    """
    said_before = len(runtime.transcript)
    try:
        await session.drain()
    except Exception:  # noqa: BLE001 - a failed drain must not leave the call open
        log.exception("call %s: could not drain before hanging up", runtime.call_id)

    if len(runtime.transcript) > said_before and runtime.transcript[-1].get("role") == "user":
        log.info("call %s: not hanging up -- the caller spoke again", runtime.call_id)
        return False

    log.info("call %s: hanging up (%s)", runtime.call_id, reason)
    runtime.annotate(hung_up=reason)
    try:
        await ctx.delete_room()
        return True
    except Exception:  # noqa: BLE001 - fall back to closing our own side
        log.exception("call %s: delete_room failed; closing the session instead", runtime.call_id)
        try:
            await session.aclose()
            return True
        except Exception:  # noqa: BLE001 - nothing left to try
            log.exception("call %s: could not close the session either", runtime.call_id)
            return False
