"""Pre-rendered greeting audio.

The first thing a caller hears is the same sentence every time. It is verbatim
tenant configuration -- it has to be, because it carries the AI disclosure and
the recording notice -- so it is the one utterance in the call that is fully
known before the phone rings, and it was being synthesised from scratch on every
single call.

That put TTS connection setup and first-byte latency at the most-judged moment
there is: the gap between the caller finishing "hello?" and the business
answering. It is also a failure mode with no cover. Every later stall can be
papered over by a filler; a stall *before the first word* just sounds like a
dead line, and a TTS hiccup there loses the disclosure entirely.

So it is rendered once and kept on disk, keyed by the things that change it. A
greeting edit or a voice setting change invalidates itself because both are
part of the key.

The disk is the worker's own, and LiveKit Cloud wipes it on every restart and
every scale-to-zero, so a miss is the normal first call after a quiet spell. A
miss is spoken live -- which streams, so audio starts at the voice's first byte
-- and recorded in the background for the calls that follow. It used to render
the whole greeting before playing any of it, which made that first caller wait
for the entire recording.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

log = logging.getLogger("stacy.greeting")

FRAME_MS = 20
"""Playback chunk. Twenty milliseconds is the ordinary telephony frame; handing
the session one enormous frame works but gives it nothing to pace against."""

_rendering: set[asyncio.Task[None]] = set()
"""Background renders in flight, held so a task is not collected mid-render."""


def _key(text: str, voice_id: str, model: str, encoding: str, *settings: object) -> str:
    """Everything that changes the audio, and nothing that does not.

    The text is in here so that editing a tenant's greeting invalidates its
    cache by construction. A cache that needs a human to remember to clear it
    will one day answer the phone with last month's disclosure. Voice settings
    are in here for the same reason: a speed change kept playing the recording
    made at the old speed.
    """
    parts = (text, voice_id, model, encoding, *(repr(s) for s in settings))
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:32]


def key_for(text: str, cfg: Any) -> str:
    """The cache key for `text` spoken with this TTS config."""
    return _key(
        text,
        cfg.voice_id,
        cfg.model,
        cfg.audio_format,
        getattr(cfg, "speed", None),
        getattr(cfg, "stability", None),
        getattr(cfg, "similarity_boost", None),
    )


def cache_paths(cache_dir: str | Path, key: str) -> tuple[Path, Path]:
    base = Path(cache_dir)
    return base / f"{key}.pcm", base / f"{key}.json"


def load(cache_dir: str | Path, key: str) -> tuple[bytes, int, int] | None:
    """Return (pcm, sample_rate, channels) if this greeting was rendered before."""
    raw, meta = cache_paths(cache_dir, key)
    if not (raw.exists() and meta.exists()):
        return None
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
        return raw.read_bytes(), int(info["sample_rate"]), int(info["num_channels"])
    except Exception:  # noqa: BLE001 - a corrupt cache is a cache miss, not an outage
        log.warning("greeting cache at %s is unreadable; re-rendering", raw)
        return None


def _replace(path: Path, data: bytes) -> None:
    """Write through a temporary file, so a reader never sees half of one."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def store(cache_dir: str | Path, key: str, pcm: bytes, sample_rate: int, channels: int) -> None:
    """Keep a rendered greeting. Every job process on a worker shares this directory."""
    raw, meta = cache_paths(cache_dir, key)
    raw.parent.mkdir(parents=True, exist_ok=True)
    _replace(raw, pcm)
    _replace(
        meta,
        json.dumps({"sample_rate": sample_rate, "num_channels": channels}).encode("utf-8"),
    )


async def render(tts: Any, text: str) -> tuple[bytes, int, int]:
    """Synthesise once, and flatten to raw PCM."""
    stream = tts.synthesize(text)
    audio = await stream.collect()
    frame = getattr(audio, "frame", audio)
    return (
        bytes(frame.data),
        int(frame.sample_rate),
        int(frame.num_channels),
    )


async def frames(pcm: bytes, sample_rate: int, channels: int) -> AsyncIterator[Any]:
    """Yield the cached bytes as audio frames the session can play."""
    from livekit import rtc  # noqa: PLC0415

    bytes_per_sample = 2  # 16-bit PCM
    per_frame = max(1, (sample_rate * FRAME_MS) // 1000)
    stride = per_frame * channels * bytes_per_sample

    for start in range(0, len(pcm), stride):
        chunk = pcm[start : start + stride]
        if not chunk:
            break
        samples = len(chunk) // (channels * bytes_per_sample)
        if samples == 0:
            break
        yield rtc.AudioFrame(
            data=chunk,
            sample_rate=sample_rate,
            num_channels=channels,
            samples_per_channel=samples,
        )


async def _render_and_store(
    tenant_id: str, cfg: Any, cache_dir: str | Path, key: str, text: str, build_tts: Any
) -> None:
    """Record the greeting for the next call. Never raises into the call."""
    try:
        pcm, rate, channels = await render(build_tts(cfg), text)
        store(cache_dir, key, pcm, rate, channels)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a failed render costs the next caller a live greeting
        log.exception("greeting render failed; the next call will speak it live and retry")
        return
    log.info("rendered greeting for %s (%d bytes)", tenant_id, len(pcm))


async def prepare(tenant: Any, cfg: Any, cache_dir: str | Path, build_tts: Any) -> Any | None:
    """Get the recorded greeting, or None to have the session speak it live.

    Returns something suitable for `session.say(text, audio=...)`. On a miss it
    returns None at once and records the greeting in the background, so the
    caller is never waiting on a full render.
    """
    text = (tenant.greeting or "").strip()
    if not text:
        return None

    key = key_for(text, cfg)
    cached = load(cache_dir, key)
    if cached is not None:
        return frames(*cached)

    task = asyncio.ensure_future(
        _render_and_store(tenant.tenant_id, cfg, cache_dir, key, text, build_tts)
    )
    _rendering.add(task)
    task.add_done_callback(_rendering.discard)
    return None
