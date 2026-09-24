"""A second transcriber listening to the same call, so two can be compared on real audio.

Accuracy arguments about speech recognition cannot be settled on synthesised
speech: what the recogniser gets wrong is the caller's own voice, line and
accent. Three live calls had "Nanak", spelled out, come back as "Ananak",
"Nanat" and "Nank", and "123 Oak Street" as "1 2 3 0 Street" -- each one costing
a correction turn, and one of them booked to the wrong address.

So a shadow transcriber is fed the same audio frames as the real one and its
transcripts are logged beside them. It never speaks, never reaches the agent,
and never blocks the call: a shadow that errors is dropped and the call goes on.
Switched on per deploy with `STACY_SHADOW_STT`, a semicolon-separated list of

    <label>=<key>:<value>,<key>:<value>

for example::

    STACY_SHADOW_STT="multi=language:multi;india=language:en-IN"
    STACY_SHADOW_STT="terms=keyterm:furnace|boiler|thermostat|Nanak"

Every key not named is the tenant's own setting, so each shadow differs from the
live transcriber in exactly the ways it names. Read the results with::

    lk agent logs | grep shadow_stt
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import replace
from typing import Any

log = logging.getLogger("stacy.bench")

ENV_VAR = "STACY_SHADOW_STT"
_CASTS = {"endpointing_ms": int, "utterance_end_ms": int, "sample_rate": int,
          "numerals": lambda v: v == "true", "filler_words": lambda v: v == "true",
          "interim_results": lambda v: v == "true"}


def specs(raw: str = "") -> list[tuple[str, dict[str, Any]]]:
    """Parse `STACY_SHADOW_STT` into (label, overrides) pairs. Anything unparseable is skipped."""
    out: list[tuple[str, dict[str, Any]]] = []
    for chunk in (raw or os.getenv(ENV_VAR, "")).split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        label, _, rest = chunk.partition("=")
        fields: dict[str, Any] = {}
        for pair in rest.split(","):
            key, _, value = pair.partition(":")
            key, value = key.strip(), value.strip()
            if not key:
                continue
            if key in ("keyterm", "keywords"):
                # A leading "+" adds to the tenant's own terms instead of replacing them.
                terms = tuple(v for v in value.lstrip("+").split("|") if v)
                fields["keywords"] = ("+", *terms) if value.startswith("+") else terms
            else:
                fields[key] = _CASTS.get(key, str)(value)
        if fields:
            out.append((label.strip(), fields))
    return out


class Shadow:
    """One extra transcriber on the same audio, logging what it hears."""

    def __init__(self, label: str, stt: Any, call_id: str) -> None:
        self.label, self._stt, self._call = label, stt, call_id
        self._stream: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._started = time.monotonic()

    async def start(self) -> None:
        self._stream = self._stt.stream()
        self._reader = asyncio.create_task(self._read(), name=f"shadow-stt-{self.label}")

    def push(self, frame: Any) -> None:
        if self._stream is not None:
            self._stream.push_frame(frame)

    async def _read(self) -> None:
        from livekit.agents import stt as stt_types  # noqa: PLC0415

        try:
            async for event in self._stream:
                if event.type != stt_types.SpeechEventType.FINAL_TRANSCRIPT:
                    continue
                text = event.alternatives[0].text if event.alternatives else ""
                if text.strip():
                    log.info("shadow_stt %s", json.dumps({
                        "call": self._call, "shadow": self.label,
                        "at_s": round(time.monotonic() - self._started, 2), "text": text,
                    }))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a shadow never disturbs the call
            log.info("shadow_stt %s failed: %s", self.label, type(exc).__name__)

    async def aclose(self) -> None:
        for closer in (getattr(self._stream, "end_input", None), getattr(self._stream, "aclose", None)):
            try:
                result = closer() if closer else None
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001, S110 - closing a shadow cannot matter
                pass
        if self._reader is not None:
            self._reader.cancel()


async def build(cfg: Any, call_id: str, raw: str = "") -> list[Shadow]:
    """Every shadow named by the environment, started and ready for frames."""
    from stacy.providers.registry import build_stt  # noqa: PLC0415

    shadows: list[Shadow] = []
    for label, fields in specs(raw):
        try:
            if fields.get("keywords", ())[:1] == ("+",):
                fields = {**fields, "keywords": tuple(cfg.keywords) + fields["keywords"][1:]}
            shadow = Shadow(label, build_stt(replace(cfg, **fields)), call_id)
            await shadow.start()
            shadows.append(shadow)
            log.info("shadow_stt %s listening with %s", label, fields)
        except Exception as exc:  # noqa: BLE001 - a shadow that cannot start is skipped
            log.warning("shadow_stt %s could not start: %s: %s", label, type(exc).__name__, exc)
    return shadows
