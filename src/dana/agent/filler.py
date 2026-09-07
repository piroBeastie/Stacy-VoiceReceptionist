"""Covering a wait with speech, so silence never has to.

A person at a desk does not go quiet for two seconds while they look something
up; they say "let me check that" and then go quiet. The platform ships this as
two separate features -- a soft timeout on slow model responses, and pre-tool
speech gated on observed tool latency -- and both are rebuilt here because the
gap they cover is the most audible difference between a cascade and a person.

Two decisions in here are worth stating rather than discovering later:

**An unknown tool is assumed slow.** The first call to a tool is the one most
likely to be slow -- cold connection, cold cache, cold lambda -- and it is also
the one with no latency history. Staying quiet through it is the worst available
choice, so the default is to speak and then learn.

**Fillers are suppressed until the caller has spoken.** Otherwise the agent
mutters "one sec" over its own greeting, which reads as broken rather than
thoughtful.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dana.providers.base import FillerConfig

EWMA_ALPHA = 0.3
"""Weight on the newest sample. High enough that a tool that just got slow is
treated as slow, low enough that one outlier doesn't make the agent chatty."""


@dataclass
class FillerPolicy:
    """Decides when the agent should say something to cover a wait.

    One instance per call. Tool latency is learned within the call rather than
    shared across calls: a tenant's endpoint being slow right now matters more
    than what its average looked like yesterday.
    """

    cfg: FillerConfig = field(default_factory=FillerConfig)
    heard_user: bool = False
    _tool_latency_s: dict[str, float] = field(default_factory=dict)
    _soft_used_this_turn: int = 0
    _soft_idx: int = 0
    _pre_tool_idx: int = 0

    def on_user_message(self) -> None:
        """The caller has said something. Fillers are unlocked from here."""
        self.heard_user = True

    def begin_turn(self) -> None:
        """Reset per-turn state. More than one filler per turn reads as a stall."""
        self._soft_used_this_turn = 0

    @property
    def _unlocked(self) -> bool:
        if not self.cfg.enabled:
            return False
        return self.heard_user or not self.cfg.suppress_until_first_user_message

    def soft_timeout_phrase(self) -> str | None:
        """A phrase to speak because the model is slow, or None to stay quiet.

        Consumes one of this turn's allowance, so calling it is the decision.
        """
        if not self._unlocked or self.cfg.soft_timeout_s < 0:
            return None
        if self._soft_used_this_turn >= self.cfg.max_soft_timeouts_per_turn:
            return None
        if not self.cfg.soft_timeout_phrases:
            return None
        self._soft_used_this_turn += 1
        phrase = self.cfg.soft_timeout_phrases[self._soft_idx % len(self.cfg.soft_timeout_phrases)]
        self._soft_idx += 1
        return phrase

    def should_speak_before_tool(self, tool_name: str) -> bool:
        """Whether to cover this tool call with speech.

        Mirrors the platform's `pre_tool_speech: auto` -- decided from observed
        latency rather than annotated by hand on every tool, so a tool that
        becomes slow in production starts getting covered without a redeploy.
        """
        if not self._unlocked:
            return False
        observed = self._tool_latency_s.get(tool_name)
        if observed is None:
            return True  # never seen it; assume the cold path is slow
        return observed >= self.cfg.pre_tool_threshold_s

    def pre_tool_phrase(self) -> str:
        """The next pre-tool phrase, rotated.

        Rotation is deterministic rather than random: at temperature 0 the model
        will not vary this for us, and hearing "let me check that" four times in
        one booking is exactly the tell this whole module exists to remove.
        """
        phrases = self.cfg.pre_tool_phrases or ("One moment.",)
        phrase = phrases[self._pre_tool_idx % len(phrases)]
        self._pre_tool_idx += 1
        return phrase

    def observe_tool(self, tool_name: str, seconds: float) -> None:
        """Fold a measured tool latency into the running estimate."""
        previous = self._tool_latency_s.get(tool_name)
        self._tool_latency_s[tool_name] = (
            seconds if previous is None else EWMA_ALPHA * seconds + (1 - EWMA_ALPHA) * previous
        )

    def observed_latency(self, tool_name: str) -> float | None:
        """Current estimate for a tool, for the trace annotation."""
        return self._tool_latency_s.get(tool_name)


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
