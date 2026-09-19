"""What counts as a response time, measured the way the caller hears it.

The stage marks this replaces logged a typical 572ms on live calls. That clock
started 0.55s after the caller stopped talking, and it filed tool round trips
under text-to-speech. These tests pin the replacement to the SDK's own
timestamps, which start at the real end of speech.
"""

from __future__ import annotations

from stacy.metrics.live import CallLatency, format_reply, parse_log, render_report


def _turn(lat: CallLatency, *, stopped: float = 100.0, eot: float = 0.8, stt: float = 0.3) -> None:
    lat.on_user(
        {"stopped_speaking_at": stopped, "end_of_turn_delay": eot, "transcription_delay": stt}
    )


class TestAReply:
    def test_the_wait_runs_from_the_end_of_speech_to_the_first_audio(self) -> None:
        lat = CallLatency(call_id="c")
        _turn(lat, stopped=100.0)
        record = lat.on_reply(
            {"started_speaking_at": 101.2, "llm_node_ttft": 0.25, "tts_node_ttfb": 0.3}
        )
        assert record is not None
        assert record["wait_ms"] == 1200.0
        assert (record["eot_ms"], record["stt_ms"]) == (800.0, 300.0)
        assert (record["llm_ttft_ms"], record["tts_ttfb_ms"]) == (250.0, 300.0)

    def test_tool_time_lands_on_the_reply_it_delayed(self) -> None:
        lat = CallLatency()
        _turn(lat)
        lat.on_tool("record_details", 4.24)
        record = lat.on_reply({"started_speaking_at": 102.0})
        assert record["tools"] == ["record_details:4ms"]
        assert record["tools_ms"] == 4.2

    def test_a_turn_without_timestamps_still_reports_what_it_has(self) -> None:
        # A turn committed by the noise guard can arrive without speech timing.
        lat = CallLatency()
        lat.on_user({})
        record = lat.on_reply({"llm_node_ttft": 0.2})
        assert record["wait_ms"] is None
        assert record["llm_ttft_ms"] == 200.0


class TestNotAResponseTime:
    def test_the_greeting_has_no_caller_turn_in_front_of_it(self) -> None:
        assert CallLatency().on_reply({"started_speaking_at": 5.0}) is None

    def test_still_there_into_a_quiet_line_is_not_timed(self) -> None:
        lat = CallLatency()
        _turn(lat)
        assert lat.on_reply({"started_speaking_at": 101.0}) is not None
        assert lat.on_reply({"started_speaking_at": 111.0}) is None

    def test_a_tool_from_an_earlier_turn_does_not_leak_forward(self) -> None:
        lat = CallLatency()
        _turn(lat)
        lat.on_tool("lookup_customer", 3.0)
        lat.on_reply({"started_speaking_at": 101.0})
        _turn(lat, stopped=110.0)
        assert lat.on_reply({"started_speaking_at": 110.5})["tools"] == []


class TestTheCallSummary:
    def test_percentiles_over_the_replies(self) -> None:
        lat = CallLatency()
        for wait in (0.6, 0.8, 1.0, 2.5):
            _turn(lat, stopped=100.0)
            lat.on_reply({"started_speaking_at": 100.0 + wait})
        s = lat.summary()
        assert s["replies"] == 4.0
        assert s["wait_p50_ms"] == 800.0
        assert s["wait_max_ms"] == 2500.0
        assert s["tool_replies"] == 0.0

    def test_a_call_with_no_replies_is_still_a_summary(self) -> None:
        assert CallLatency().summary() == {"replies": 0.0, "tool_replies": 0.0}


class TestReadingItBack:
    def test_plain_lines_parse_and_the_json_wrapped_copy_is_skipped(self) -> None:
        body = format_reply("call_x", {"turn": 0, "wait_ms": 900.0, "tools": []})
        lines = [
            f"INFO:stacy.agent:latency {body}",
            '{"message": "latency ' + body.replace('"', '\\"') + '", "level": "INFO"}',
            f"INFO:stacy.agent:latency {body}",  # the same reply from an overlapping capture
            "INFO:stacy.agent:heard call_x: 'hello'",
        ]
        records = parse_log(lines)
        assert records == [{"call": "call_x", "turn": 0, "wait_ms": 900.0, "tools": []}]

    def test_the_report_splits_waits_by_whether_a_tool_ran(self) -> None:
        records = [
            {"call": "a", "turn": 0, "wait_ms": 800.0, "tools": []},
            {"call": "a", "turn": 1, "wait_ms": 2000.0, "tools": ["record_details:3ms"]},
        ]
        text = render_report(records)
        assert "2 replies across 1 calls" in text
        assert "800ms with no tool" in text
        assert "2000ms after a tool" in text
