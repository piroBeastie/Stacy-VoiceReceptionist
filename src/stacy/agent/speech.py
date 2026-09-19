"""Text on its way to the voice: what the TTS model must never read aloud.

The prompt was ported from an ElevenLabs agent running an expressive model, and
it asks for audio tags -- [slow] before every phone number read-back, [chuckles]
when asked whether the agent is a person. The pipeline runs eleven_flash_v2_5,
which does not interpret them. It reads them. Synthesised and transcribed back,
"[slow] Fourteen Elm Street" came out as "Slow 14 Elm Street", and
"[chuckles] Nope" as "Chuckles. Nope." Every caller heard the word "slow" in
front of their own phone number.

So tags are removed on the way into TTS rather than out of the prompt. The
prompt stays diffable against the original, and a tag the model puts somewhere
odd -- it put [slow] on an address in testing -- becomes harmless.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncIterable, AsyncIterator

_TAG = re.compile(r"\[[A-Za-z][A-Za-z ]{0,30}\]")
"""A voice cue: a bracketed word or two. "[1]" is not one and is left alone."""

_RUNS_OF_SPACES = re.compile(r"[ \t]{2,}")

_LONGEST_TAG = 32
"""Past this many characters an unclosed "[" cannot be a tag, so it is spoken."""


def strip_audio_tags(text: str) -> str:
    """Remove every audio tag from a complete string."""
    return _RUNS_OF_SPACES.sub(" ", _TAG.sub("", text))


async def strip_audio_tags_stream(chunks: AsyncIterable[str]) -> AsyncIterator[str]:
    """Remove audio tags from streamed text, including a tag split across chunks.

    Model output arrives a few characters at a time, so "[sl" and "ow] two" can
    land in separate chunks. Anything from an unclosed "[" onward is held back
    until it either closes -- and is dropped, if it was a tag -- or grows past
    the longest possible tag, at which point it cannot be one and is released.
    The hold is bounded, so a stray bracket never stalls speech.
    """
    pending = ""
    async for chunk in chunks:
        pending = _TAG.sub("", pending + chunk)
        cut = pending.rfind("[")
        if cut != -1 and "]" not in pending[cut:] and len(pending) - cut <= _LONGEST_TAG:
            ready, pending = pending[:cut], pending[cut:]
        else:
            ready, pending = pending, ""
        if ready:
            yield _RUNS_OF_SPACES.sub(" ", ready)
    if pending:
        yield _RUNS_OF_SPACES.sub(" ", pending)


SENTENCE_END = (".", "?", "!")


async def push_with_flush(chunks: AsyncIterable[str], stream: Any, *, stall_s: float = 0.12) -> None:
    """Push the model's text into a TTS stream, flushing a finished sentence when the text stalls.

    A short reply followed by a tool call -- "Got it. What town are you in?" and
    then record_details -- sat unspoken while the model wrote the tool call: the
    voice only starts a sentence once it has the next one, or the end of the
    text. Measured against ElevenLabs, first audio came 630 ms after the text;
    flushed after a 120 ms stall, 300 ms. Text that keeps coming is never
    flushed, so a long reply is unchanged.
    """
    it = chunks.__aiter__()
    tail = ""
    while True:
        nxt = asyncio.ensure_future(anext(it))
        try:
            if tail.endswith(SENTENCE_END):
                done, _ = await asyncio.wait({nxt}, timeout=stall_s)
                if not done:
                    stream.flush()
                    tail = ""
            chunk = await nxt
        except StopAsyncIteration:
            break
        except BaseException:
            nxt.cancel()
            raise
        stream.push_text(chunk)
        if chunk.strip():
            tail = chunk.rstrip()
    stream.end_input()
