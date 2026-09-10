"""Backchannel policy.

Getting this wrong is the most common way a voice agent reveals itself: it stops
talking every time the caller says "mhm" while it reads a booking back.
"""

from __future__ import annotations

import pytest

from stacy.agent import BackchannelFilter


@pytest.fixture
def f() -> BackchannelFilter:
    return BackchannelFilter()


@pytest.mark.parametrize(
    "text",
    ["mhm", "yeah", "uh huh", "okay", "okay sure", "right", "gotcha", "mm hmm", "", "..."],
)
def test_acknowledgments_do_not_interrupt(f: BackchannelFilter, text: str) -> None:
    assert f.is_backchannel(text)


@pytest.mark.parametrize(
    "text",
    [
        "yeah but wait",
        "no stop",
        "my furnace is broken",
        "two zero three",
        "hold on",
        "actually",
        "yeah yeah yeah yeah",  # length beats agreeableness
    ],
)
def test_real_speech_interrupts(f: BackchannelFilter, text: str) -> None:
    assert f.should_interrupt(text)


def test_hold_on_interrupts_because_the_prompt_answers_it(f: BackchannelFilter) -> None:
    # The prompt has a specific response to "hold on" -- swallowing it as
    # backchannel would lose that behavior entirely.
    assert f.should_interrupt("hold on")


def test_tenant_extra_terms_are_honored() -> None:
    f = BackchannelFilter(extra_terms=frozenset({"wallingford"}))
    assert f.is_backchannel("wallingford")
    assert BackchannelFilter().should_interrupt("wallingford")


def test_max_words_is_configurable() -> None:
    assert BackchannelFilter(max_words=1).should_interrupt("okay sure")
    assert BackchannelFilter(max_words=3).is_backchannel("okay sure")
