"""The one failure mode that can hurt a caller.

These run first in spirit if not in order. Everything else in the suite guards a
promise; this guards a person. The bar is different: a detection miss here is not
a regression, it is an incident.
"""

from __future__ import annotations

import pytest

from dana.agent.safety import (
    OVERRIDE_SCRIPT,
    QUIET_SCRIPT,
    SafetyInterceptor,
    is_gas_or_co,
)


@pytest.mark.parametrize(
    "said",
    [
        "I smell gas",
        "i smell gas in the basement",
        "there's a gas smell in here",
        "it smells like gas",
        "smells of gas near the furnace",
        "I think there's a gas leak",
        "the gas is leaking",
        "gas leak in the kitchen",
        "something smells like rotten eggs",
        "my carbon monoxide detector is going off",
        "carbon monoxide alarm won't stop",
        "the CO alarm is beeping",
        "co detector went off an hour ago",
        "CARBON MONOXIDE",
    ],
)
def test_reports_that_must_trigger(said: str) -> None:
    assert is_gas_or_co(said), f"missed: {said!r}"


@pytest.mark.parametrize(
    "said",
    [
        # The booking flow asks "gas or electric?" about furnaces on ordinary
        # calls. Triggering on a bare "gas" would break every furnace booking.
        "it's a gas furnace",
        "gas heat, not electric",
        "we run on natural gas",
        "my gas bill went way up",
        "the gas company came out last week",
        "I need a gas line moved",
        "it's a gas boiler, about ten years old",
        "no heat upstairs",
        "the AC is making a noise",
        "can someone come tomorrow",
        "",
    ],
)
def test_ordinary_calls_that_must_not_trigger(said: str) -> None:
    assert not is_gas_or_co(said), f"false positive: {said!r}"


class TestLatching:
    def test_the_script_is_verbatim(self) -> None:
        s = SafetyInterceptor()
        s.scan("I smell gas")
        assert s.response() == OVERRIDE_SCRIPT

    def test_nothing_before_a_trigger(self) -> None:
        s = SafetyInterceptor()
        assert s.response() is None
        s.scan("my furnace is out")
        assert s.response() is None

    def test_it_never_unlatches(self) -> None:
        # The prompt: "From then on: no booking, no questions, no goodbye."
        # A caller who follows a gas report with an ordinary sentence still
        # gets the script -- that is the whole point of a latch.
        s = SafetyInterceptor()
        s.scan("I smell gas")
        s.scan("okay but can you still send someone tomorrow at two")
        assert s.latched
        assert s.response() == OVERRIDE_SCRIPT

    def test_every_later_turn_repeats_the_same_words(self) -> None:
        s = SafetyInterceptor()
        s.scan("carbon monoxide alarm is going off")
        assert [s.response() for _ in range(3)] == [OVERRIDE_SCRIPT] * 3

    def test_a_quiet_line_gets_the_shorter_line(self) -> None:
        s = SafetyInterceptor()
        s.scan("I smell gas")
        assert s.response(quiet=True) == QUIET_SCRIPT

    def test_scanning_is_idempotent(self) -> None:
        # The transcript event and the generation path both scan, and they do
        # not fire in a guaranteed order.
        s = SafetyInterceptor()
        assert s.scan("I smell gas") is True
        assert s.scan("I smell gas") is True
        assert s.latched

    def test_triggers_are_counted_for_the_record(self) -> None:
        # A caller still talking is a caller who has not left the house yet.
        s = SafetyInterceptor()
        s.scan("gas leak")
        s.response()
        s.response()
        assert s.triggers == 2

    def test_the_trigger_phrase_is_kept(self) -> None:
        s = SafetyInterceptor()
        s.scan("I think I smell gas in the basement")
        assert "smell gas" in s.trigger_phrase

    def test_a_later_report_does_not_overwrite_the_first(self) -> None:
        s = SafetyInterceptor()
        s.scan("I smell gas")
        s.scan("carbon monoxide too")
        assert s.trigger_phrase == "I smell gas"
