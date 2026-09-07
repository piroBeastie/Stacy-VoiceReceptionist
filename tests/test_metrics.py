"""The measurement harness has to be right before anything it measures is."""

from __future__ import annotations

from dana.metrics import JsonlSink, Stage, TurnTrace, load_traces, summarize
from dana.metrics.report import percentile

BASE = 1_000_000_000


def _trace(stages: list[Stage], step_ms: int = 100, **kw) -> TurnTrace:
    t = TurnTrace(call_id=kw.get("call_id", "c1"), tenant_id="millbrook", turn_index=0)
    for i, s in enumerate(stages):
        t.mark(s, at=BASE + i * step_ms * 1_000_000)
    return t


class TestTurnTrace:
    def test_full_path_spans_sum_to_response(self) -> None:
        t = _trace(list(Stage))
        assert t.response_ms == 500
        assert sum(t.spans().values()) == t.response_ms

    def test_preemptive_generation_skips_stt_without_losing_the_total(self) -> None:
        # When the LLM starts on interim transcript there is no STT_FINAL mark.
        # The row must still sum to response_ms or the report silently undercounts.
        t = _trace(
            [
                Stage.USER_STOPPED,
                Stage.ENDPOINT_COMMITTED,
                Stage.LLM_FIRST_TOKEN,
                Stage.TTS_FIRST_BYTE,
                Stage.AUDIO_OUT,
            ],
            step_ms=50,
        )
        spans = t.spans()
        assert "stt_ms" not in spans
        assert sum(spans.values()) == t.response_ms == 200

    def test_remarking_keeps_the_first_occurrence(self) -> None:
        # Barge-in can replay a stage within one turn; the first is the true one.
        t = TurnTrace(call_id="c", tenant_id="m", turn_index=0)
        t.mark(Stage.USER_STOPPED, at=BASE)
        t.mark(Stage.USER_STOPPED, at=BASE + 999_000_000)
        assert t.marks[Stage.USER_STOPPED.value] == BASE

    def test_incomplete_turn_reports_none_rather_than_raising(self) -> None:
        t = _trace([Stage.USER_STOPPED, Stage.ENDPOINT_COMMITTED])
        assert t.response_ms is None
        assert t.to_record()["response_ms"] is None

    def test_annotations_are_namespaced(self) -> None:
        t = _trace(list(Stage))
        t.annotate(model="qwen", barge_in=True)
        rec = t.to_record()
        assert rec["meta_model"] == "qwen"
        assert rec["meta_barge_in"] is True


class TestReport:
    def test_percentile_is_nearest_rank(self) -> None:
        assert percentile([10, 20, 30, 40], 50) == 20
        assert percentile([10, 20, 30, 40], 95) == 40
        assert percentile([], 50) != percentile([], 50)  # nan

    def test_roundtrip_through_the_sink(self, tmp_path) -> None:
        sink = JsonlSink(tmp_path)
        sink.emit(_trace(list(Stage), step_ms=100))
        sink.emit(_trace(list(Stage), step_ms=200))
        report = summarize(load_traces(tmp_path))
        assert report.turns == 2
        assert report.calls == 1
        assert report.stages["response_ms"]["p50"] == 500
        assert report.stages["response_ms"]["max"] == 1000
        assert "response_ms" in report.render()

    def test_sink_never_raises_on_a_bad_record(self, tmp_path) -> None:
        # Instrumentation must not be the thing that kills a live call.
        sink = JsonlSink(tmp_path)
        t = _trace(list(Stage))
        t.annotate(unserializable=object())
        sink.emit(t)  # logged and swallowed
