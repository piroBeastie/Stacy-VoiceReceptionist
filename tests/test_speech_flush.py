"""Flushing a finished sentence to the voice when the model's text stalls."""

from __future__ import annotations

import asyncio

from stacy.agent.speech import push_with_flush


class _Stream:
    def __init__(self) -> None:
        self.events: list[tuple[str, ...]] = []

    def push_text(self, text: str) -> None:
        self.events.append(("text", text))

    def flush(self) -> None:
        self.events.append(("flush",))

    def end_input(self) -> None:
        self.events.append(("end",))


async def _chunks(parts: list[str], *, pause_after: int, pause_s: float):
    for i, part in enumerate(parts):
        yield part
        if i == pause_after:
            await asyncio.sleep(pause_s)


def _push(parts: list[str], *, pause_after: int, pause_s: float = 0.3) -> list[tuple[str, ...]]:
    stream = _Stream()
    asyncio.run(push_with_flush(_chunks(parts, pause_after=pause_after, pause_s=pause_s), stream, stall_s=0.05))
    return stream.events


def test_a_reply_that_stalls_on_a_tool_call_is_flushed() -> None:
    events = _push(["Got it.", " What town", " are you in?"], pause_after=2)
    assert events == [
        ("text", "Got it."), ("text", " What town"), ("text", " are you in?"), ("flush",), ("end",),
    ]


def test_text_that_keeps_coming_is_never_flushed() -> None:
    events = _push(["Got it.", " What town", " are you in?"], pause_after=-1)
    assert ("flush",) not in events
    assert events[-1] == ("end",)


def test_a_stall_mid_sentence_waits_for_the_rest() -> None:
    events = _push(["What town", " are you in?"], pause_after=0)
    assert ("flush",) not in events


def test_a_stall_between_sentences_releases_the_first() -> None:
    events = _push(["Got it.", " What town are you in?"], pause_after=0)
    assert events[:2] == [("text", "Got it."), ("flush",)]
