"""What the caller hears when a desk operation fails in a way nobody planned.

Every desk operation returns a sentence, including for the failures it knows
about -- a full window, a town out of area, a calendar that is busy. The ones
it does not know about are the problem: a full disk, a permission on the lock
file, a plain bug. Those used to raise straight through the SDK, and what the
caller gets then is not an apology, it is dead air in the middle of giving
someone their address.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("livekit.agents")

from stacy.agent import CallRuntime, DeskOperations  # noqa: E402
from stacy.agent.tools import build_function_tools  # noqa: E402
from stacy.desk import CalendarStore, MessageStore  # noqa: E402

FAILS = [
    ("record_details", {"customer_name": "Dan"}),
    ("book_appointment", {}),
    ("take_message", {"caller_name": "Dan", "phone": "+12035550147", "summary": "no heat"}),
    ("check_availability", {"day": "tomorrow"}),
    ("lookup_customer", {}),
    ("dispatch_emergency", {}),
    ("end_call", {"reason": "wrong number"}),
]


class _Ctx:
    """Enough RunContext for the filler wrapper; the SDK supplies the real one."""

    def with_filler(self, *_args, **_kwargs):
        class _Nothing:
            async def __aenter__(self_inner) -> None:
                return None

            async def __aexit__(self_inner, *_exc) -> bool:
                return False

        return _Nothing()


@pytest.fixture
def runtime(tenant, tmp_path, monday_9am) -> CallRuntime:
    return CallRuntime.build(
        tenant=tenant,
        call_id="fail",
        caller_id="+12035550147",
        calendar=CalendarStore(tmp_path / "cal"),
        messages=MessageStore(tmp_path / "msg"),
        now=monday_9am,
    )


@pytest.mark.parametrize(("name", "kwargs"), FAILS)
def test_an_unexpected_failure_becomes_a_sentence(runtime, name: str, kwargs: dict) -> None:
    ops = DeskOperations(runtime)

    def full_disk(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    setattr(ops, name, full_disk)
    tool = {t.info.name: t for t in build_function_tools(ops)}[name]

    said = asyncio.run(tool(_Ctx(), **kwargs))

    assert isinstance(said, str) and said
    assert "failed" in said
    # The one thing it must never let the agent say.
    assert "don't say it's done" in said


def test_the_sentence_cannot_be_read_as_success(runtime) -> None:
    # "Booked" is the word the prompt keys on: only a reply starting with it
    # means the caller is booked. A failure that began with it would put a
    # visit in the agent's mouth that is in no calendar.
    ops = DeskOperations(runtime)

    def boom(*_args, **_kwargs):
        raise RuntimeError("something nobody wrote a branch for")

    ops.book_appointment = boom
    tool = {t.info.name: t for t in build_function_tools(ops)}["book_appointment"]

    said = asyncio.run(tool(_Ctx()))
    assert not said.startswith(("Booked", "Changed", "Alright"))
    assert "call them straight back" in said


def test_a_keyboard_interrupt_still_stops_the_worker(runtime) -> None:
    # The catch is for Exception, not BaseException: a cancellation or a
    # shutdown has to keep travelling, or the worker cannot be stopped.
    ops = DeskOperations(runtime)

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    ops.lookup_customer = interrupted
    tool = {t.info.name: t for t in build_function_tools(ops)}["lookup_customer"]

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(tool(_Ctx()))
