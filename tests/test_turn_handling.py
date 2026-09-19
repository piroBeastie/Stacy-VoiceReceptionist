"""The session's turn settings, checked against the SDK that takes them.

Moved off the deprecated keyword arguments so the reply can be voiced
preemptively, not only written preemptively. The move itself must change
nothing else: the same endpointing, the same interruption rules, and VAD when
there is no semantic detector.
"""

from __future__ import annotations

import asyncio

import pytest

from stacy.providers.base import TurnConfig
from stacy.providers.registry import turn_handling


def test_the_values_are_the_ones_the_old_arguments_carried() -> None:
    cfg = TurnConfig()
    th = turn_handling(cfg, None)
    assert th["endpointing"] == {
        "min_delay": cfg.min_endpointing_delay_s,
        "max_delay": cfg.max_endpointing_delay_s,
    }
    assert th["interruption"] == {
        "min_words": cfg.interrupt_min_words,
        "false_interruption_timeout": cfg.false_interruption_timeout_s,
        "resume_false_interruption": True,
    }


def test_turn_detection_is_present_even_when_there_is_no_detector() -> None:
    # Left out, the SDK substitutes its own detector instead of falling back to VAD.
    assert "turn_detection" in turn_handling(TurnConfig(), None)


def test_the_reply_is_voiced_early_as_well_as_written_early() -> None:
    assert turn_handling(TurnConfig(), None)["preemptive_generation"] == {
        "enabled": True,
        "preemptive_tts": True,
    }


def test_the_sdk_accepts_them_as_given() -> None:
    pytest.importorskip("livekit.agents")
    from livekit.agents import AgentSession

    cfg = TurnConfig()

    async def build():
        return AgentSession(turn_handling=turn_handling(cfg, None)).options

    opts = asyncio.run(build())
    assert opts.preemptive_generation["preemptive_tts"] is True
    assert opts.endpointing["max_delay"] == cfg.max_endpointing_delay_s
    assert opts.interruption["min_words"] == cfg.interrupt_min_words


def test_a_stuck_model_request_is_retried_after_seconds_not_ten() -> None:
    # Live: a stalled request waited out the SDK's 10s default -- 13s of silence.
    pytest.importorskip("livekit.agents")
    from livekit.agents import AgentSession

    from stacy.providers.base import LLMConfig
    from stacy.providers.registry import llm_conn_options

    cfg = LLMConfig()

    async def build():
        return AgentSession(conn_options=llm_conn_options(cfg)).conn_options

    opts = asyncio.run(build())
    assert opts.llm_conn_options.timeout == cfg.timeout_s < 10
    assert opts.llm_conn_options.max_retry >= 1
