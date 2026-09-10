"""Deciding whether a caller's noise is actually an interruption.

ElevenLabs gets this from `interruption_ignore_terms` with the English presets
merged in. Rebuilt here because without it the agent stops talking every time
someone says "mhm" while it reads a booking back, which is the single most
common way a voice agent reveals it is not a person.

The rule is narrow on purpose: only a short utterance made *entirely* of
acknowledgment tokens is ignored. "yeah, but wait" contains a real word and
must interrupt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_IGNORE_TERMS_EN: frozenset[str] = frozenset(
    {
        # Acknowledgment
        "mhm", "mm", "mmm", "mmhmm", "hmm", "hm", "uhhuh", "uhuh", "ahuh",
        "yeah", "yea", "yep", "yup", "yes", "ya", "sure", "right", "ok", "okay",
        "gotcha", "got", "it", "alright", "aight",
        # Filler / hesitation
        "uh", "um", "er", "ah", "oh", "eh", "hah", "huh", "well", "so", "like",
        # Continuers
        "and", "then", "wow", "nice", "cool", "great", "good", "true", "exactly",
        "totally", "definitely", "absolutely", "sounds", "makes", "sense",
    }
)
"""Rough equivalent of the merged `en` presets. Tune against real call audio."""

_WORD = re.compile(r"[a-z']+")


@dataclass
class BackchannelFilter:
    """Classifies a partial transcript as backchannel or genuine interruption.

    Args:
        ignore_terms: Tokens that never, alone, constitute an interruption.
        max_words: Utterances longer than this always interrupt, however
            agreeable the words. Someone saying "yeah yeah yeah yeah okay okay"
            for five words is trying to stop you.
        extra_terms: Per-tenant additions, e.g. a town name people mumble.
    """

    ignore_terms: frozenset[str] = DEFAULT_IGNORE_TERMS_EN
    max_words: int = 3
    extra_terms: frozenset[str] = field(default_factory=frozenset)

    def _tokens(self, text: str) -> list[str]:
        return _WORD.findall(text.lower().replace("-", ""))

    def is_backchannel(self, text: str) -> bool:
        """True if `text` should NOT interrupt the agent.

        Empty or punctuation-only input counts as backchannel: a stray VAD blip
        with no words in it is noise, not a turn.
        """
        tokens = self._tokens(text)
        if not tokens:
            return True
        if len(tokens) > self.max_words:
            return False
        allowed = self.ignore_terms | self.extra_terms
        return all(tok in allowed for tok in tokens)

    def should_interrupt(self, text: str) -> bool:
        """Inverse of :meth:`is_backchannel`, for readability at the call site."""
        return not self.is_backchannel(text)
