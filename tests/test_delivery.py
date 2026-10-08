"""Post-call delivery.

The agent tells every prospect "he gets your details the second we hang up."
These tests are what stand between that sentence and a lie.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from stacy.delivery import RETRY_DELAYS_S, SIGNATURE_HEADER, WebhookSender, call_payload, sign, verify
from stacy.delivery.webhook import MAX_CONSECUTIVE_FAILURES

SECRET = "whsec_not_a_real_secret"


class RecordingTransport:
    """Returns a scripted sequence of statuses and remembers what it was sent."""

    def __init__(self, statuses: list[int | Exception]) -> None:
        self._statuses = list(statuses)
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []

    async def post(self, url: str, body: bytes, headers: dict[str, str]) -> int:
        self.calls.append((url, body, headers))
        outcome = self._statuses.pop(0) if self._statuses else 200
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _sender(statuses: list[int | Exception]) -> tuple[WebhookSender, RecordingTransport, list[float]]:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    transport = RecordingTransport(statuses)
    sender = WebhookSender(
        url="https://example.test/hook",
        secret=SECRET,
        transport=transport,
        sleep=fake_sleep,
        jitter=lambda: 0.0,
    )
    return sender, transport, slept


class TestSignature:
    def test_sign_and_verify_agree(self) -> None:
        body = b'{"call_id":"abc"}'
        assert verify(body, sign(body, SECRET), SECRET)

    def test_a_tampered_body_fails(self) -> None:
        header = sign(b'{"amount":10}', SECRET)
        assert not verify(b'{"amount":1000}', header, SECRET)

    def test_the_wrong_secret_fails(self) -> None:
        body = b"{}"
        assert not verify(body, sign(body, SECRET), "whsec_someone_elses")

    def test_an_old_signature_is_rejected(self) -> None:
        # The timestamp is inside the signed payload, so a captured request
        # cannot be replayed an hour later with its original header.
        body = b"{}"
        old = int(time.time()) - 3600
        assert not verify(body, sign(body, SECRET, timestamp=old), SECRET)

    def test_malformed_headers_do_not_raise(self) -> None:
        for header in ("", "garbage", "t=notanumber,v0=abc", "v0=abc"):
            assert not verify(b"{}", header, SECRET)


class TestRetryLadder:
    def test_a_2xx_sends_once(self) -> None:
        sender, transport, slept = _sender([200])
        assert asyncio.run(sender.send({"hello": "world"})) is True
        assert len(transport.calls) == 1
        assert slept == []

    def test_transient_failures_climb_the_ladder_then_succeed(self) -> None:
        sender, transport, slept = _sender([503, 429, 200])
        assert asyncio.run(sender.send({})) is True
        assert len(transport.calls) == 3
        assert slept == list(RETRY_DELAYS_S[1:3])

    def test_a_4xx_is_not_retried(self) -> None:
        # A 422 means the consumer read the payload and rejected it. Sending it
        # four more times delivers the same rejection four more times.
        sender, transport, _ = _sender([422])
        assert asyncio.run(sender.send({})) is False
        assert len(transport.calls) == 1

    def test_network_errors_are_retried(self) -> None:
        sender, transport, _ = _sender([ConnectionError("reset"), 200])
        assert asyncio.run(sender.send({})) is True
        assert len(transport.calls) == 2

    def test_exhausting_the_ladder_gives_up(self) -> None:
        sender, transport, slept = _sender([500] * len(RETRY_DELAYS_S))
        assert asyncio.run(sender.send({})) is False
        assert len(transport.calls) == len(RETRY_DELAYS_S)
        assert slept == list(RETRY_DELAYS_S[1:])

    def test_send_never_raises(self) -> None:
        sender, _, _ = _sender([RuntimeError("boom")] * len(RETRY_DELAYS_S))
        assert asyncio.run(sender.send({})) is False


class TestAutoDisable:
    def test_a_dead_consumer_stops_the_backlog(self) -> None:
        sender, _, _ = _sender([])
        sender.transport = RecordingTransport([404] * MAX_CONSECUTIVE_FAILURES * 2)
        for _ in range(MAX_CONSECUTIVE_FAILURES):
            asyncio.run(sender.send({}))
        assert sender.disabled
        before = len(sender.transport.calls)
        asyncio.run(sender.send({}))
        assert len(sender.transport.calls) == before  # dropped, not attempted

    def test_a_success_resets_the_counter(self) -> None:
        sender, _, _ = _sender([500, 500, 500, 500, 500, 200])
        asyncio.run(sender.send({}))  # exhausts the ladder -> one failure
        assert sender.consecutive_failures == 1
        sender.transport = RecordingTransport([200])
        asyncio.run(sender.send({}))
        assert sender.consecutive_failures == 0


class TestPayload:
    def test_outcome_is_routable_without_reading_the_transcript(self) -> None:
        payload = call_payload(
            call_id="c1",
            tenant_id="millbrook",
            caller_id="+12035550147",
            transcript=[{"role": "agent", "text": "Thanks for calling"}],
            outcome="booked",
            duration_s=61.4,
        )
        assert payload["outcome"] == "booked"
        assert payload["type"] == "post_call"
        assert payload["duration_s"] == 61.4

    def test_payload_is_json_serializable(self) -> None:
        payload = call_payload(
            call_id="c1",
            tenant_id="millbrook",
            caller_id="",
            transcript=[],
            outcome="message",
            duration_s=12.0,
            message={"priority": "urgent"},
            latency={"p50": 640.0},
        )
        assert json.loads(json.dumps(payload))["message"]["priority"] == "urgent"

    def test_the_signature_header_travels_with_the_body(self) -> None:
        sender, transport, _ = _sender([200])
        payload = call_payload(
            call_id="c1",
            tenant_id="millbrook",
            caller_id="+12035550147",
            transcript=[],
            outcome="answered",
            duration_s=8.0,
        )
        asyncio.run(sender.send(payload))
        _, body, headers = transport.calls[0]
        assert verify(body, headers[SIGNATURE_HEADER], SECRET)
        assert json.loads(body)["call_id"] == "c1"


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_every_retryable_status_actually_retries(status: int) -> None:
    sender, transport, _ = _sender([status, 200])
    assert asyncio.run(sender.send({})) is True
    assert len(transport.calls) == 2
