"""Post-call delivery.

The agent's closing line is "he gets your details the second we hang up." Until
this module existed, that sentence was false on every call -- the single most
consequential gap in the repo, because it is the one promise the caller can
check.

The signature and retry semantics deliberately copy the platform's, so a
consumer written against ElevenLabs' post-call webhook needs no changes:

* ``Stacy-Signature: t=<unix>,v0=<hex>`` over the exact bytes ``{t}.{body}``
* five attempts -- immediate, 30s, 2m, 8m, 30m -- with +/-10% jitter
* only 5xx, 429 and 408 are retried; a 4xx is a bug in the payload, and
  retrying it just delivers the same bug four more times
* the endpoint disables itself after enough consecutive failures, so a dead
  consumer stops generating an unbounded retry backlog

Comparison of the signature uses :func:`hmac.compare_digest`. A webhook
receiver that compares with ``==`` leaks the signature a byte at a time.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger("stacy.delivery")

RETRY_DELAYS_S: tuple[float, ...] = (0.0, 30.0, 120.0, 480.0, 1800.0)
"""Five attempts. The first is immediate; the rest are the platform's ladder."""

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

SIGNATURE_HEADER = "Stacy-Signature"
MAX_CONSECUTIVE_FAILURES = 10


def sign(body: bytes, secret: str, *, timestamp: int | None = None) -> str:
    """Build the signature header value for `body`.

    The timestamp is inside the signed payload, not just alongside it, so a
    captured request cannot be replayed later with a fresh header.
    """
    ts = int(time.time()) if timestamp is None else timestamp
    payload = f"{ts}.".encode() + body
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"t={ts},v0={digest}"


def verify(
    body: bytes, header: str, secret: str, *, tolerance_s: int = 300, now: int | None = None
) -> bool:
    """Check a signature header. Used by consumers and by the tests.

    Shipping the verifier next to the signer is what keeps the two from drifting
    -- and it means a customer integrating against this has a reference that is
    known to agree with the sender.
    """
    parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
    ts, sig = parts.get("t"), parts.get("v0")
    if not ts or not sig:
        return False
    try:
        ts_int = int(ts)
    except ValueError:
        return False
    current = int(time.time()) if now is None else now
    if abs(current - ts_int) > tolerance_s:
        return False
    expected = hmac.new(
        secret.encode(), f"{ts_int}.".encode() + body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, sig)


class Transport(Protocol):
    """Anything that can POST bytes and report a status code."""

    async def post(self, url: str, body: bytes, headers: dict[str, str]) -> int: ...


class AiohttpTransport:
    """Default transport. Imported lazily so tests need no HTTP stack."""

    def __init__(self, timeout_s: float = 10.0) -> None:
        self._timeout_s = timeout_s

    async def post(self, url: str, body: bytes, headers: dict[str, str]) -> int:
        import aiohttp  # noqa: PLC0415

        timeout = aiohttp.ClientTimeout(total=self._timeout_s)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(url, data=body, headers=headers) as resp,
        ):
            return resp.status


@dataclass
class WebhookSender:
    """Delivers one JSON payload per finished call, with retries.

    Args:
        url: Consumer endpoint. HTTPS in anything but a test.
        secret: Shared HMAC secret.
        transport: Injectable so the retry ladder can be tested without a server.
        sleep: Injectable so the tests do not actually wait thirty minutes.
    """

    url: str
    secret: str
    transport: Transport = field(default_factory=AiohttpTransport)
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    jitter: Callable[[], float] = lambda: random.uniform(-0.1, 0.1)  # noqa: E731

    consecutive_failures: int = 0
    disabled: bool = False

    async def send(self, payload: dict[str, Any]) -> bool:
        """Deliver `payload`, retrying transient failures. True if accepted.

        Never raises. A post-call webhook that can take down the worker is worse
        than one that drops a delivery, because the call it would take down is
        the *next* one.
        """
        if self.disabled:
            log.warning("webhook %s is auto-disabled; dropping payload", self.url)
            return False

        body = json.dumps(payload, separators=(",", ":")).encode()
        for attempt, base_delay in enumerate(RETRY_DELAYS_S):
            if base_delay:
                await self.sleep(base_delay * (1 + self.jitter()))
            try:
                status = await self.transport.post(
                    self.url,
                    body,
                    {
                        "Content-Type": "application/json",
                        SIGNATURE_HEADER: sign(body, self.secret),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - a network error is a retry, not a crash
                log.warning("webhook attempt %d raised %s", attempt + 1, exc)
                continue

            if 200 <= status < 300:
                self.consecutive_failures = 0
                return True
            if status not in RETRYABLE_STATUS:
                # A 4xx means the consumer read the payload and rejected it.
                # Sending it four more times delivers the same rejection.
                log.error("webhook %s returned %d; not retrying", self.url, status)
                self._record_failure()
                return False
            log.warning("webhook attempt %d got %d; retrying", attempt + 1, status)

        self._record_failure()
        return False

    def _record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self.disabled = True
            log.error(
                "webhook %s disabled after %d consecutive failures",
                self.url,
                self.consecutive_failures,
            )


def call_payload(
    *,
    call_id: str,
    tenant_id: str,
    caller_id: str,
    transcript: list[dict[str, str]],
    outcome: str,
    duration_s: float,
    appointments: list[dict[str, Any]] | None = None,
    message: dict[str, Any] | None = None,
    latency: dict[str, float] | None = None,
    safety: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the post-call payload.

    `outcome` is one of the three the prompt allows -- answered, booked,
    message -- plus `dispatched` for the emergency branch, so a consumer can
    route on it without parsing the transcript.

    `appointments` is a list even though nearly every call produces at most
    one. A scalar meant a caller who booked twice silently lost the first
    booking, and with it that booking's SMS consent flag.
    """
    return {
        "type": "post_call",
        "call_id": call_id,
        "tenant_id": tenant_id,
        "caller_id": caller_id,
        "outcome": outcome,
        "duration_s": round(duration_s, 2),
        "transcript": transcript,
        "appointments": appointments or [],
        "message": message,
        "latency": latency or {},
        # Present only when the gas/CO override fired. A consumer should treat
        # its presence as the headline of the call regardless of `outcome`.
        "safety": safety,
        "sent_at": int(time.time()),
    }
