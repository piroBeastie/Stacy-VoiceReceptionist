"""Flushing a finished sentence to the voice when the model's text stalls, or ends a question.

Driven through the SDK's own SynthesizeStream, recording what reaches a plugin's
input channel: a stand-in stream accepted text after a flush that the real one
silently drops, and the tests passed while "Or electric?" was never spoken.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("livekit.agents")

from livekit.agents import APIConnectOptions  # noqa: E402
from livekit.agents import tts as sdk_tts  # noqa: E402

from stacy.agent.speech import push_with_flush  # noqa: E402

FLUSH = "<flush>"


class _RecordingTTS(sdk_tts.TTS):
    def __init__(self) -> None:
        super().__init__(capabilities=sdk_tts.TTSCapabilities(streaming=True), sample_rate=8000, num_channels=1)

    def synthesize(self, text, *, conn_options=None):  # pragma: no cover - never called
        raise NotImplementedError

    def stream(self, *, conn_options=APIConnectOptions(max_retry=0)):
        return _RecordingStream(tts=self, conn_options=conn_options)


class _RecordingStream(sdk_tts.SynthesizeStream):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.seen: list[str] = []

    async def _run(self, emitter) -> None:
        emitter.initialize(request_id="r", sample_rate=8000, num_channels=1, mime_type="audio/pcm", stream=True)
        emitter.start_segment(segment_id="s")
        async for item in self._input_ch:
            self.seen.append(FLUSH if isinstance(item, self._FlushSentinel) else item)
        emitter.push(b"\x00\x00" * 800)
        emitter.end_segment()


async def _chunks(parts: list[str], *, pause_after: int, pause_s: float):
    for i, part in enumerate(parts):
        yield part
        if i == pause_after:
            await asyncio.sleep(pause_s)


def _push(parts: list[str], *, pause_after: int, pause_s: float = 0.3) -> list[str]:
    """What the plugin receives, in order, up to the end of input."""

    async def run() -> list[str]:
        stream = _RecordingTTS().stream()
        push = asyncio.create_task(
            push_with_flush(_chunks(parts, pause_after=pause_after, pause_s=pause_s), stream, stall_s=0.05)
        )
        async for _ in stream:
            pass
        await push
        return stream.seen

    seen = asyncio.run(run())
    assert seen and seen[-1] == FLUSH  # end_input always sends one
    seen.pop()
    return seen


def _text(seen: list[str]) -> str:
    return "".join(s for s in seen if s != FLUSH)


def test_a_reply_that_stalls_on_a_tool_call_is_flushed() -> None:
    assert _push(["Got it.", " What town", " are you in?"], pause_after=2) == [
        "Got it.", " What town", " are you in?", FLUSH,
    ]


def test_text_that_keeps_coming_is_never_flushed() -> None:
    seen = _push(["Got it.", " It's on the way", " now."], pause_after=-1)
    assert FLUSH not in seen and _text(seen) == "Got it. It's on the way now."


def test_a_question_is_sent_the_moment_it_ends() -> None:
    assert _push(["Got it.", " What town", " are you in?"], pause_after=-1) == [
        "Got it.", " What town", " are you in?", FLUSH,
    ]


def test_a_stall_mid_sentence_waits_for_the_rest() -> None:
    assert _push(["What town", " are you in?"], pause_after=0) == ["What town", " are you in?", FLUSH]


def test_text_after_a_flush_still_reaches_the_voice() -> None:
    # The live bug: SynthesizeStream.flush() ends the segment and drops the rest.
    seen = _push(["Is it gas?", " Or electric?", " I just need that one detail."], pause_after=-1)
    assert _text(seen) == "Is it gas? Or electric? I just need that one detail."
    assert seen[:2] == ["Is it gas?", FLUSH]


def test_text_after_a_stall_flush_still_reaches_the_voice() -> None:
    seen = _push(["Got it.", " What town are you in?"], pause_after=0)
    assert seen[:2] == ["Got it.", FLUSH]
    assert _text(seen) == "Got it. What town are you in?"
