"""Per-turn latency instrumentation.

Built before the pipeline on purpose: every later decision in this repo is a
comparison against a number, and a number you cannot see is a number you cannot
beat.
"""

from stacy.metrics.turn import Stage, TurnTrace
from stacy.metrics.sink import JsonlSink, NullSink, TraceSink
from stacy.metrics.report import Report, load_traces, summarize

__all__ = [
    "Stage",
    "TurnTrace",
    "TraceSink",
    "JsonlSink",
    "NullSink",
    "Report",
    "load_traces",
    "summarize",
]
