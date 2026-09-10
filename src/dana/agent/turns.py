"""When background noise holds a caller's turn open, answer what they said anyway.

livekit ends a user turn on a final transcript only once voice activity
detection says the caller has stopped speaking: in audio_recognition, a final
transcript runs end-of-turn detection behind `if not self._speaking`. Silero
releases "speaking" only once speech probability falls below its deactivation
threshold, which by default sits 0.15 under activation -- 0.45 at this
pipeline's 0.6. A television, a fan or a room of people hovering above that
keeps "speaking" true for as long as it lasts. Reported from a live call: the
caller finished, Deepgram transcribed them, and the agent did not speak.

The signal that survives any amount of noise is words. Speech produces
transcripts, interim ones every few hundred milliseconds; noise produces none.
So when a transcript is waiting for a reply, the line still reads as "speaking",
and no word has arrived for a couple of seconds, it is the noise that is
talking, and the turn is committed.

Nothing here acts while voice detection is working: if "speaking" has already
ended, the SDK's own endpointing owns the turn and this stays out of the way.
"""

from __future__ import annotations

from dataclasses import dataclass

BUSY_AGENT_STATES = frozenset({"thinking", "speaking"})


@dataclass
class NoiseHeldTurnGuard:
    """Tracks words, not audio, to tell a finished caller from a noisy room."""

    commit_after_s: float = 2.0
    last_words_at: float | None = None
    _waiting_since: float | None = None
    _committed: bool = False

    def on_words(self, now: float, *, text: str, final: bool) -> None:
        """A transcript arrived. Interim results count: they are what speech looks like."""
        if not text.strip():
            return
        self.last_words_at = now
        if final and self._waiting_since is None:
            self._waiting_since = now
            self._committed = False

    def on_agent_reply(self) -> None:
        """The agent has started answering, so nothing is waiting any more."""
        self._waiting_since = None
        self._committed = False

    def heard_words_within(self, now: float, seconds: float) -> bool:
        """Whether "speaking" right now is plausibly the caller rather than the room."""
        return self.last_words_at is not None and now - self.last_words_at < seconds

    def should_commit(
        self, now: float, *, user_state: str | None, agent_state: str | None
    ) -> bool:
        """Whether noise, not the caller, is what is keeping this turn open."""
        if self._waiting_since is None or self._committed:
            return False
        if agent_state in BUSY_AGENT_STATES:
            return False
        if user_state != "speaking":
            return False
        return self.last_words_at is not None and now - self.last_words_at >= self.commit_after_s

    def mark_committed(self) -> None:
        """Commit once per waiting transcript, however long the noise goes on."""
        self._committed = True
