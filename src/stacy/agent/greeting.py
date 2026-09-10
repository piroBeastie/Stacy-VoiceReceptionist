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
worker restart does not re-synthesise, and a greeting edit invalidates itself
because the text is part of the key.

Falling back is always safe: if anything here fails, the caller gets the
greeting synthesised live, exactly as before.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator

log = logging.getLogger("stacy.greeting")

FRAME_MS = 20
"""Playback chunk. Twenty milliseconds is the ordinary telephony frame; handing
the session one enormous frame works but gives it nothing to pace against."""


def _key(text: str, voice_id: str, model: str, encoding: str) -> str:
    """Everything that changes the audio, and nothing that does not.

    The text is in here so that editing a tenant's greeting invalidates its
    cache by construction. A cache that needs a human to remember to clear it
    will one day answer the phone with last month's disclosure.
    """
    digest = hashlib.sha256(
        "\x00".join((text, voice_id, model, encoding)).encode("utf-8")
    ).hexdigest()
    return digest[:32]


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


def store(cache_dir: str | Path, key: str, pcm: bytes, sample_rate: int, channels: int) -> None:
    raw, meta = cache_paths(cache_dir, key)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(pcm)
    meta.write_text(
        json.dumps({"sample_rate": sample_rate, "num_channels": channels}), encoding="utf-8"
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


async def prepare(tenant: Any, cfg: Any, cache_dir: str | Path, build_tts: Any) -> Any | None:
    """Get playable greeting audio, rendering it only if it is not already on disk.

    Returns something suitable for `session.say(text, audio=...)`, or None, in
    which case the caller should just let the session synthesise normally.
    """
    text = (tenant.greeting or "").strip()
    if not text:
        return None

    key = _key(text, cfg.voice_id, cfg.model, cfg.audio_format)
    cached = load(cache_dir, key)
    if cached is None:
        try:
            pcm, rate, channels = await render(build_tts(cfg), text)
        except Exception:  # noqa: BLE001 - never let a cache miss cost the call
            log.exception("greeting pre-render failed; it will be synthesised live")
            return None
        store(cache_dir, key, pcm, rate, channels)
        log.info("rendered greeting for %s (%d bytes)", tenant.tenant_id, len(pcm))
    else:
        pcm, rate, channels = cached

    return frames(pcm, rate, channels)
