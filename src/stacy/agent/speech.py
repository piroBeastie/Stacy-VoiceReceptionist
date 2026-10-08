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
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

_TAG = re.compile(r"\[[A-Za-z][A-Za-z ]{0,30}\]")
"""A voice cue: a bracketed word or two. "[1]" is not one and is left alone."""

_RUNS_OF_SPACES = re.compile(r"[ \t]{2,}")

_MARKDOWN = re.compile(r"[*#`~]")
"""Markdown the model may slip into speech. Removed character by character, so nothing is held.

The SDK's own filter_markdown is not used on the voice (see TTS_TEXT_TRANSFORMS):
it holds back the last word of the text until more arrives, and after a reply's
closing "...are you in?" what arrives next is the record_details call. Live, every
such reply was heard only once the tool call had been written -- up to half a
second after its last word."""

TTS_TEXT_TRANSFORMS = ["filter_emoji"]
"""The SDK's transforms on text headed for the voice: its default minus filter_markdown."""

_LONGEST_TAG = 32
"""Past this many characters an unclosed "[" cannot be a tag, so it is spoken."""


def strip_audio_tags(text: str) -> str:
    """Remove every audio tag, and any markdown, from a complete string."""
    return _RUNS_OF_SPACES.sub(" ", _MARKDOWN.sub("", _TAG.sub("", text)))


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
        ready = _MARKDOWN.sub("", ready)
        if ready:
            yield _RUNS_OF_SPACES.sub(" ", ready)
    pending = _MARKDOWN.sub("", pending)
    if pending:
        yield _RUNS_OF_SPACES.sub(" ", pending)


SENTENCE_END = (".", "?", "!")


def flush_voice(stream: Any) -> None:
    """Have the voice start on the text it holds, and keep the stream open for more.

    Not ``stream.flush()``: in livekit-agents 1.8 that ends the stream's segment,
    and every ``push_text`` after it is dropped with a "multiple segments ...
    deprecated" warning. Review found "Is it gas? Or electric?" spoken as "Is it
    gas?" once a flush landed between them. The flush marker is put on the
    stream's input channel directly -- the one thing a plugin acts on; the
    ElevenLabs plugin flushes its sentence tokenizer on it -- and the segment
    stays open. On an SDK without those internals nothing is flushed: speech
    starts later, and no word is lost.
    """
    channel = getattr(stream, "_input_ch", None)
    marker = getattr(stream, "_FlushSentinel", None)
    if channel is None or marker is None or getattr(channel, "closed", True):
        return
    channel.send_nowait(marker())


async def push_with_flush(chunks: AsyncIterable[str], stream: Any, *, stall_s: float = 0.12) -> None:
    """Push the model's text into a TTS stream, flushing a finished sentence when the text stalls.

    A short reply followed by a tool call -- "Got it. What town are you in?" and
    then record_details -- sat unspoken while the model wrote the tool call: the
    voice only starts a sentence once it has the next one, or the end of the
    text. Measured against ElevenLabs, first audio came 630 ms after the text;
    flushed after a 120 ms stall, 300 ms. Text that keeps coming is never
    flushed, so a long reply is unchanged.

    A question is sent at once, with no stall: one question per turn, and it
    goes last, so it is nearly always the end of what will be said. Measured,
    the model's text comes in bursts up to a quarter second apart, so the stall
    alone often held the question back. A sentence ending in a full stop still
    waits the stall, which keeps a short "Got it." joined to what follows it.
    """
    it = chunks.__aiter__()
    tail = ""
    while True:
        nxt = asyncio.ensure_future(anext(it))
        try:
            if tail.endswith(SENTENCE_END):
                done, _ = await asyncio.wait({nxt}, timeout=stall_s)
                if not done:
                    flush_voice(stream)
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
            if tail.endswith("?"):
                flush_voice(stream)
                tail = ""
    stream.end_input()
