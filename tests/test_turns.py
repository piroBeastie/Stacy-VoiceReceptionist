"""A finished caller in a noisy room still gets an answer.

Reported from a live call: with background noise, the agent did not speak.
livekit only ends a turn on a final transcript once voice detection says the
caller stopped, and noise above Silero's release threshold keeps that from
ever happening. Words are the one signal noise cannot fake.
"""

from __future__ import annotations

from stacy.agent.turns import NoiseHeldTurnGuard
from stacy.providers.base import TurnConfig


def _guard() -> NoiseHeldTurnGuard:
    return NoiseHeldTurnGuard(commit_after_s=2.0)


class TestAnswersWhenNoiseHoldsTheTurn:
    def test_a_finished_caller_under_noise_is_answered(self) -> None:
        g = _guard()
        g.on_words(10.0, text="my furnace is out", final=True)
        assert g.should_commit(12.5, user_state="speaking", agent_state="listening")

    def test_a_short_pause_is_not_cut_off(self) -> None:
        g = _guard()
        g.on_words(10.0, text="my furnace is", final=True)
        assert not g.should_commit(11.5, user_state="speaking", agent_state="listening")

    def test_interim_words_mean_the_caller_is_still_talking(self) -> None:
        # Interim results every few hundred milliseconds are what speech looks like.
        g = _guard()
        g.on_words(10.0, text="my furnace", final=True)
        g.on_words(11.9, text="is making a noise", final=False)
        assert not g.should_commit(13.0, user_state="speaking", agent_state="listening")

    def test_it_commits_once_however_long_the_noise_lasts(self) -> None:
        g = _guard()
        g.on_words(10.0, text="hello", final=True)
        assert g.should_commit(12.5, user_state="speaking", agent_state="listening")
        g.mark_committed()
        assert not g.should_commit(40.0, user_state="speaking", agent_state="listening")


class TestStaysOutOfTheWay:
    def test_noise_alone_is_not_a_turn(self) -> None:
        # Nothing was said, so there is nothing to answer.
        assert not _guard().should_commit(30.0, user_state="speaking", agent_state="listening")

    def test_working_endpointing_is_left_to_the_sdk(self) -> None:
        # Voice detection already released the turn, so noise is not holding it.
        g = _guard()
        g.on_words(10.0, text="my furnace is out", final=True)
        assert not g.should_commit(15.0, user_state="listening", agent_state="listening")

    def test_never_while_the_agent_is_already_answering(self) -> None:
        g = _guard()
        g.on_words(10.0, text="my furnace is out", final=True)
        for busy in ("thinking", "speaking"):
            assert not g.should_commit(15.0, user_state="speaking", agent_state=busy)

    def test_an_answer_clears_what_was_waiting(self) -> None:
        g = _guard()
        g.on_words(10.0, text="my furnace is out", final=True)
        g.on_agent_reply()
        assert not g.should_commit(15.0, user_state="speaking", agent_state="listening")

    def test_blank_transcripts_are_not_words(self) -> None:
        g = _guard()
        g.on_words(10.0, text="   ", final=True)
        assert g.last_words_at is None
        assert not g.should_commit(15.0, user_state="speaking", agent_state="listening")


class TestNoiseIsNotTheCallerToTheWatchdog:
    def test_recent_words_make_speaking_real(self) -> None:
        g = _guard()
        g.on_words(10.0, text="hi", final=False)
        assert g.heard_words_within(11.0, 2.0)

    def test_speaking_with_no_recent_words_is_the_room(self) -> None:
        # Otherwise noise holds "speaking" on and the agent never re-engages.
        g = _guard()
        assert not g.heard_words_within(11.0, 2.0)
        g.on_words(10.0, text="hi", final=False)
        assert not g.heard_words_within(13.0, 2.0)


def test_it_waits_no_longer_than_the_semantic_detector_would() -> None:
    cfg = TurnConfig()
    assert 0 < cfg.noise_commit_after_s <= cfg.max_endpointing_delay_s
