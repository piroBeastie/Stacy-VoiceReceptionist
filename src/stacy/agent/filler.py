"""Covering a wait with speech -- only when there is a wait to cover.

A person at a desk does not say "one sec" before telling you their hours. They
say it when they turn to the screen and the screen is slow. Getting that wrong
in either direction is audible: silence through a slow lookup sounds like a
dead line, and a filler in front of an instant answer sounds like a machine
reading from a script.

The previous design got it wrong in the second direction, and measurably. A
timer on the model's first token spoke "One sec" whenever a generation was slow
-- and the SDK starts a second generation after every tool call, which on this
prompt almost always says nothing because the model already answered in the
first. So the filler landed *after* the answer, on nearly every turn. It was
also yielded into the model's own output stream, so it entered the chat history
as something the agent had chosen to say.

What replaces it:

**Fillers cover tools, never the model.** The model's first words arrive in
about a second. What can genuinely stall is a lookup, and that is where a
receptionist would speak.

**They fire on measured idleness, not a guess.** The SDK's `with_filler` waits
until the session has been continuously idle -- agent not speaking or thinking,
caller not talking -- for a set time while the tool is still running. A tool
that returns in a millisecond, which every tool here currently does, never
produces a filler at all.

**The words fit the moment.** "Let me pull up the schedule" before checking
availability, not "one moment" before everything. Some tools are never covered:
`record_details` runs on nearly every caller turn, and covering it would put the
original problem straight back; `dispatch_emergency` is no moment for filler.

**At most one per caller turn, never the same line twice running, and nothing
before the caller has spoken** -- otherwise the agent mutters over its own
greeting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stacy.providers.base import FillerConfig


@dataclass
class FillerPolicy:
    """Decides whether, and in what words, to cover a slow tool.

    One instance per call. The decision is taken at the moment a filler would
    actually be spoken, not when the tool starts, so a filler that never fires
    consumes nothing.
    """

    cfg: FillerConfig = field(default_factory=FillerConfig)
    heard_user: bool = False
    _spoken_this_turn: int = 0
    _last_phrase: str | None = None
    _next_index: dict[str, int] = field(default_factory=dict)

    def on_user_message(self) -> None:
        """The caller has said something. Fillers are unlocked from here."""
        self.heard_user = True

    def begin_turn(self) -> None:
        """A new caller turn: its filler allowance starts over."""
        self._spoken_this_turn = 0

    @property
    def _unlocked(self) -> bool:
        if not self.cfg.enabled:
            return False
        return self.heard_user or not self.cfg.suppress_until_first_user_message

    def covers(self, tool_name: str) -> bool:
        """Whether this tool may ever be covered with speech at all."""
        return bool(self.cfg.tool_phrases.get(tool_name))

    def tool_phrase(self, tool_name: str) -> str | None:
        """Words to speak because `tool_name` is running slow, or None to stay quiet.

        Calling it is the decision: a returned phrase counts against this turn's
        allowance. It is called by the SDK only once the line has actually gone
        quiet, so a tool that finished first never reaches here.
        """
        if not self._unlocked:
            return None
        if self._spoken_this_turn >= self.cfg.max_per_turn:
            return None
        phrases = self.cfg.tool_phrases.get(tool_name) or ()
        if not phrases:
            return None

        start = self._next_index.get(tool_name, 0)
        for offset in range(len(phrases)):
            candidate = phrases[(start + offset) % len(phrases)]
            if candidate != self._last_phrase or len(phrases) == 1:
                self._next_index[tool_name] = start + offset + 1
                self._last_phrase = candidate
                self._spoken_this_turn += 1
                return candidate
        return None


@dataclass
class RepetitionTracker:
    """Notices when the agent says the same sentence twice in one call.

    The prompt forbids it in as many words, and temperature 0 -- which the
    safety override requires -- is precisely the setting that causes it. This
    does not prevent a repeat; it makes one countable, so `frequency_penalty`
    can be tuned against a number instead of a hunch.
    """

    _seen: set[str] = field(default_factory=set)
    repeats: int = 0

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split()).strip(" .!?,")

    def observe(self, text: str) -> bool:
        """Record an agent utterance. True if it repeats an earlier one."""
        key = self._normalize(text)
        # Short acknowledgments are supposed to recur; "Got it." twice in a call
        # is a person, not a loop. Only sentence-length repeats are a tell.
        if len(key.split()) < 4:
            return False
        if key in self._seen:
            self.repeats += 1
            return True
        self._seen.add(key)
        return False
