"""The gas and carbon monoxide override, taken away from the model.

Every other rule in the prompt is an instruction the model is asked to follow.
This one is different in kind: if it slips, somebody stays in a house filling
with gas. An instruction that is followed 99.5% of the time is an excellent
instruction and an unacceptable safety control.

So this path does not reach the model at all. When a caller's words match, the
script below is spoken verbatim and the LLM is never called for that turn --
which makes the single most safety-critical path in the product also the
fastest one in it, since it costs no inference.

Two properties matter more than the detection itself:

**It latches.** Once triggered, every subsequent turn returns the same words.
The prompt says "whatever they say next, repeat the same instruction in the same
words", and a latch is the only way that is true even if the caller says
something that would otherwise be a perfectly ordinary booking request.

**It removes the tools.** Because the model is never invoked, no tool can fire.
There is no path from a triggered override to a booking, and that is structural
rather than instructed.

The matching is deliberately narrow on the word "gas" alone. The booking flow
asks "gas or electric?" about furnaces on ordinary calls, so a bare mention of
gas must not trigger. What triggers is gas plus a *smell* or a *leak*, or carbon
monoxide in any form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

OVERRIDE_SCRIPT = (
    "Please leave the house right now, don't flip any switches or light anything, "
    "and call nine-one-one from outside. Call us back once they've cleared the house."
)
"""Verbatim from the prompt. `tests/test_port_fidelity.py` pins the same string
in the template, so the two cannot drift apart silently."""

QUIET_SCRIPT = "Please go outside now."
"""What the prompt says to use when the line goes quiet after a trigger."""

_GAS = re.compile(
    r"""
    smell(?:s|ing|ed)?\s+(?:like\s+|of\s+)?(?:some\s+)?gas
  | gas\s+(?:smell|odou?r|leak)
  | (?:smell|smelling|smells)\s+(?:like\s+)?(?:rotten\s+eggs?|sulf|sulph)
  | leak(?:s|ing)?\s+gas
  | gas\s+(?:is\s+)?leak(?:s|ing)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CO = re.compile(
    r"""
    carbon\s*monoxide
  | \bco\s*(?:alarm|detector|monitor)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_gas_or_co(text: str) -> bool:
    """Whether a caller utterance is a gas-smell or carbon-monoxide report.

    Asymmetric by design: a false positive costs one wrongly-handled call, a
    false negative costs a house. But not *so* asymmetric that a bare "gas"
    fires -- "it's a gas furnace" is an answer to a question the agent itself
    asked, and triggering on it would break every furnace booking.
    """
    return bool(_GAS.search(text) or _CO.search(text))


@dataclass
class SafetyInterceptor:
    """Latching override for one call.

    Args:
        latched: Whether the override has fired. Never returns to False.
        triggers: How many turns have been answered with the script, for the
            trace -- a caller who keeps talking is a caller who has not left yet.
    """

    latched: bool = False
    triggers: int = 0
    _first_match: str = field(default="", repr=False)

    def scan(self, text: str) -> bool:
        """Check a caller utterance and latch if it matches. Returns the latch state.

        Safe to call repeatedly with the same text, and safe to call after the
        latch has already closed -- both happen, because the transcript event
        and the generation path check independently rather than relying on the
        order they fire in.
        """
        if not self.latched and text and is_gas_or_co(text):
            self.latched = True
            self._first_match = text.strip()[:120]
        return self.latched

    def response(self, *, quiet: bool = False) -> str | None:
        """The words to speak, or None if the override has not fired.

        Args:
            quiet: True when the caller has said nothing since the last script.
                The prompt uses a shorter line there rather than repeating the
                full instruction into silence.
        """
        if not self.latched:
            return None
        self.triggers += 1
        return QUIET_SCRIPT if quiet else OVERRIDE_SCRIPT

    @property
    def trigger_phrase(self) -> str:
        """What the caller actually said, for the post-call record."""
        return self._first_match
