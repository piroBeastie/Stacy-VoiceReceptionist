"""Ending the call: when it happens by itself, and when the agent may ask for it.

Hanging up is the one action with no undo on a phone call. Every test here is
either "this call is over and the caller should not be left listening to
nothing" or "this call is not over and cutting it would be unforgivable".
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from stacy.agent import CallRuntime, DeskOperations
from stacy.agent.hangup import (
    business_unfinished,
    caller_signed_off,
    farewell_said,
    hang_up,
)
from stacy.desk import CalendarStore, MessageStore
from stacy.tenants import Tenant


@pytest.fixture
def runtime(tenant: Tenant, tmp_path, monday_9am: datetime) -> CallRuntime:
    return CallRuntime.build(
        tenant=tenant,
        call_id="call-1",
        caller_id="+12035550147",
        calendar=CalendarStore(tmp_path / "cal"),
        messages=MessageStore(tmp_path / "msg"),
        now=monday_9am,
    )


def _exchange(runtime: CallRuntime, *turns: tuple[str, str]) -> None:
    for role, text in turns:
        if role == "agent":
            runtime.record_agent(text)
        else:
            runtime.record_user(text)


class TestTheCallEndsItself:
    def test_both_said_goodbye(self, runtime) -> None:
        _exchange(
            runtime,
            ("agent", "You're all set. Anything else before I let you go?"),
            ("user", "No, that's it. Thanks."),
            ("agent", "Good luck with the furnace, Dan — bye."),
        )
        assert farewell_said(runtime) == "both said goodbye"

    @pytest.mark.parametrize("last", ["Bye.", "Bye bye.", "Yeah, bye.", "Nope, that's all.", "I'm good, thanks."])
    def test_the_ways_a_caller_signs_off(self, runtime, last: str) -> None:
        _exchange(runtime, ("agent", "Anything else?"), ("user", last), ("agent", "Take care — bye."))
        assert farewell_said(runtime)

    def test_the_agent_has_to_have_said_goodbye_too(self, runtime) -> None:
        # The caller said bye to a question that is still open. She answers it.
        _exchange(runtime, ("agent", "And what's the street address there?"), ("user", "Bye."))
        assert farewell_said(runtime) == ""

    def test_not_while_the_caller_is_still_talking(self, runtime) -> None:
        _exchange(
            runtime,
            ("agent", "Good luck with the furnace — bye."),
            ("user", "Oh wait, one more thing."),
        )
        assert farewell_said(runtime) == ""

    def test_not_mid_booking(self, runtime) -> None:
        _exchange(runtime, ("agent", "Perfect. And can I get your name?"), ("user", "Dan Whitaker."))
        assert farewell_said(runtime) == ""

    def test_never_during_the_gas_instruction(self, runtime) -> None:
        runtime.record_user("I can smell gas in the kitchen.")
        assert runtime.safety.latched
        _exchange(runtime, ("agent", "Please go outside now. Bye."), ("user", "Okay, bye."))
        # The caller leaving the line is the point; the script keeps repeating.
        assert farewell_said(runtime) == ""

    def test_a_cut_off_goodbye_is_not_a_goodbye(self, runtime) -> None:
        _exchange(runtime, ("agent", "Anything else?"), ("user", "No thanks."))
        assert farewell_said(runtime) == ""  # nothing said since


class TestTheAgentAsksToEnd:
    def test_a_wrong_number_can_be_ended(self, runtime) -> None:
        _exchange(runtime, ("agent", "Millbrook Heating and Cooling."), ("user", "Sorry, wrong number."))
        said = DeskOperations(runtime).end_call(reason="wrong number")
        assert said.startswith("Alright") and runtime.end_call == "wrong number"

    def test_not_holding_half_an_address(self, runtime) -> None:
        runtime.slots.update(customer_name="Dan", street_address="14 Elm Street", problem="no heat")
        _exchange(runtime, ("agent", "And what town is that in?"), ("user", "Hang on."))
        said = DeskOperations(runtime).end_call(reason="taking too long")
        assert said.startswith("Not yet") and runtime.end_call == ""
        assert "customer_name" in said  # it says what it is holding

    def test_a_finished_booking_can_be_ended(self, runtime, tenant) -> None:
        runtime.slots.update(
            customer_name="Dan", phone="+12035550147", street_address="14 Elm Street",
            town="Wallingford", problem="no heat", day="tomorrow", window="eleven to two",
        )
        runtime.record_agent("So that's Dan, 14 Elm Street in Wallingford, no heat, tomorrow, eleven to two. Right?")
        runtime.record_user("Yes.")
        assert DeskOperations(runtime).book_appointment().startswith("Booked")
        assert DeskOperations(runtime).end_call(reason="done").startswith("Alright")

    def test_a_caller_who_wants_to_go_can_go(self, runtime) -> None:
        runtime.slots.update(customer_name="Dan", problem="no heat")
        _exchange(runtime, ("agent", "What's the best number for you?"), ("user", "Actually, forget it. Bye."))
        assert DeskOperations(runtime).end_call(reason="caller rang off").startswith("Alright")

    def test_never_during_the_gas_instruction(self, runtime) -> None:
        runtime.record_user("There's a smell of gas.")
        said = DeskOperations(runtime).end_call(reason="they are shouting")
        assert said.startswith("No.") and runtime.end_call == ""

    def test_the_reason_is_recorded_even_when_the_model_gives_none(self, runtime) -> None:
        _exchange(runtime, ("agent", "Hello?"), ("user", "Not interested."))
        DeskOperations(runtime).end_call(reason="")
        assert runtime.end_call == "the agent ended the call"


class TestHangingUpPolitely:
    class _Session:
        def __init__(self, runtime: CallRuntime | None = None, interrupt_with: str = "") -> None:
            self.drained = False
            self.closed = False
            self._runtime = runtime
            self._interrupt = interrupt_with

        async def drain(self) -> None:
            self.drained = True
            if self._runtime is not None and self._interrupt:
                # The caller gets a last word in while the goodbye plays.
                self._runtime.record_user(self._interrupt)

        async def aclose(self) -> None:
            self.closed = True

    class _Ctx:
        def __init__(self, fails: bool = False) -> None:
            self.deleted = False
            self._fails = fails

        async def delete_room(self) -> None:
            if self._fails:
                raise RuntimeError("no such room")
            self.deleted = True

    def test_the_goodbye_finishes_before_the_line_drops(self, runtime) -> None:
        session, ctx = self._Session(), self._Ctx()
        assert asyncio.run(hang_up(ctx, session, runtime, "both said goodbye")) is True
        assert session.drained and ctx.deleted

    def test_a_caller_who_speaks_again_keeps_the_call(self, runtime) -> None:
        session = self._Session(runtime, interrupt_with="Oh wait, one more thing!")
        ctx = self._Ctx()
        assert asyncio.run(hang_up(ctx, session, runtime, "both said goodbye")) is False
        assert not ctx.deleted  # "one more thing" is a real thing people say

    def test_a_failed_delete_still_closes_our_side(self, runtime) -> None:
        session, ctx = self._Session(), self._Ctx(fails=True)
        assert asyncio.run(hang_up(ctx, session, runtime, "wrong number")) is True
        assert session.closed


class TestNotBeforeTheJobIsDone:
    """The line is not cut before the caller got what they rang for.

    Every test here is a caller who said goodbye, and an agent who said
    goodbye back, where cutting the line would still be wrong.
    """

    def _book(self, runtime) -> None:
        runtime.slots.update(
            customer_name="Dan", phone="+12035550147", street_address="14 Elm Street",
            town="Wallingford", problem="no heat", day="tomorrow", window="eleven to two",
        )
        runtime.record_agent(
            "So that's Dan, 14 Elm Street in Wallingford, no heat, tomorrow, eleven to two. Right?"
        )
        runtime.record_user("Yes.")
        assert DeskOperations(runtime).book_appointment().startswith("Booked")

    def test_it_cuts_once_the_visit_is_booked(self, runtime) -> None:
        self._book(runtime)
        _exchange(
            runtime,
            ("agent", "You're all set for tomorrow, eleven to two. Anything else?"),
            ("user", "Nope, that's it."),
            ("agent", "Good luck with the furnace, Dan — bye."),
        )
        assert farewell_said(runtime) == "both said goodbye"

    def test_it_does_not_cut_with_the_details_and_no_booking(self, runtime) -> None:
        runtime.slots.update(customer_name="Dan", street_address="14 Elm Street", problem="no heat")
        _exchange(
            runtime,
            ("agent", "Got it. Let me find you a window — bye for now?"),
            ("user", "Okay, bye."),
            ("agent", "Take care, Dan — bye."),
        )
        # She has his address and no appointment. He rang to get a tech out.
        assert farewell_said(runtime) == ""
        assert business_unfinished(runtime) == ["customer_name", "street_address", "problem"]

    def test_a_message_taken_counts_as_finished(self, runtime) -> None:
        runtime.slots.update(customer_name="Dan", phone="+12035550147", problem="wants a quote")
        assert DeskOperations(runtime).take_message(
            caller_name="Dan", phone="+12035550147", summary="wants a quote",
        )
        _exchange(runtime, ("agent", "I'll get that to Mike. Anything else?"),
                  ("user", "No, I'm good."), ("agent", "Take care — bye."))
        assert farewell_said(runtime) == "both said goodbye"

    def test_a_caller_who_gave_nothing_can_still_go(self, runtime) -> None:
        # The opening-hours question. Nothing started, so nothing is unfinished.
        _exchange(runtime, ("agent", "We're open till five today."), ("user", "Great, bye."),
                  ("agent", "Bye."))
        assert farewell_said(runtime) == "both said goodbye"

    def test_not_while_she_is_asking_something(self, runtime) -> None:
        _exchange(
            runtime,
            ("agent", "Anything else?"),
            ("user", "No, that's it."),
            ("agent", "Take care, Dan — did you want me to put the filter on the ticket too?"),
        )
        assert farewell_said(runtime) == ""  # she asked; he answers

    def test_not_while_his_goodbye_carries_a_question(self, runtime) -> None:
        self._book(runtime)
        _exchange(
            runtime,
            ("agent", "You're all set. Anything else?"),
            ("user", "How much will it run me? Anyway, bye."),
            ("agent", "Bye, Dan."),
        )
        assert farewell_said(runtime) == ""  # the price question is still open

    def test_a_dispatch_counts_as_finished(self, runtime) -> None:
        runtime.slots.update(
            customer_name="Dan", phone="+12035550147", street_address="14 Elm Street",
            town="Wallingford", problem="furnace is dead and the pipes are at risk",
        )
        assert DeskOperations(runtime).dispatch_emergency()
        _exchange(runtime, ("agent", "He's on his way now."), ("user", "Thank you, bye."),
                  ("agent", "Stay warm, Dan — bye."))
        assert farewell_said(runtime) == "both said goodbye"

    def test_the_tool_can_still_let_a_leaving_caller_go(self, runtime) -> None:
        # The escape hatch: unfinished, but he said he is going. The automatic
        # path refuses this; the model has to ask for it in so many words.
        runtime.slots.update(customer_name="Dan", street_address="14 Elm Street")
        _exchange(runtime, ("agent", "And what town is that?"), ("user", "Forget it, bye."))
        assert DeskOperations(runtime).end_call(reason="caller rang off").startswith("Alright")


class TestTheWiringThatActuallyFires:
    """The entrypoint half: nothing else in the suite touches `_wire`.

    `farewell_said` being right is only half of it. The handler that calls it
    runs on every agent turn, and the bug worth a test is the one where an
    intent set by the model survives the turn it was set in.
    """

    class _Session:
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}
            self.drained = False

        def on(self, event: str):  # the SDK's decorator form
            def keep(fn):
                self.handlers[event] = fn
                return fn

            return keep

        async def drain(self) -> None:
            self.drained = True

        async def aclose(self) -> None:
            pass

    class _Ctx:
        def __init__(self) -> None:
            self.deleted = False

        async def delete_room(self) -> None:
            self.deleted = True

    class _Item:
        def __init__(self, text: str, interrupted: bool = False) -> None:
            self.role, self.interrupted = "assistant", interrupted
            self.text_content = text

    def _said(self, session, text: str, *, interrupted: bool = False) -> None:
        handler = session.handlers["conversation_item_added"]
        handler(SimpleNamespace(item=self._Item(text, interrupted)))

    @pytest.fixture
    def wired(self, runtime):
        from stacy.agent.entrypoint import _wire
        from stacy.agent.turns import NoiseHeldTurnGuard

        session, ctx = self._Session(), self._Ctx()
        _wire(session, runtime, NoiseHeldTurnGuard(), None, ctx)
        return session, ctx

    def test_an_interrupted_goodbye_does_not_hang_up_later(self, runtime, wired) -> None:
        session, ctx = wired
        runtime.record_user("Sorry, wrong number.")
        DeskOperations(runtime).end_call(reason="wrong number")

        # The caller talks over the line, so it never finished.
        self._said(session, "No problem at all — bye.", interrupted=True)
        assert runtime.end_call == ""  # consumed, not left armed

        # The next turn is a new conversation, and must not inherit the intent.
        runtime.record_user("Actually, is this Millbrook Heating?")
        self._said(session, "It is — what can I do for you?")
        assert not ctx.deleted

    def test_the_farewell_path_fires_from_a_real_event(self, runtime, wired) -> None:
        session, ctx = wired
        runtime.record_agent("Anything else before I let you go?")
        runtime.record_user("No, that's it. Thanks.")

        async def the_last_turn() -> None:
            # The handler is sync and schedules the hangup; the loop has to
            # exist for it, exactly as it does in the worker.
            self._said(session, "Good luck with the furnace, Dan — bye.")
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            await asyncio.gather(*pending)

        asyncio.run(the_last_turn())
        assert session.drained and ctx.deleted


def test_a_plain_sign_off_is_recognised_without_a_runtime() -> None:
    assert caller_signed_off("Bye bye.")
    assert caller_signed_off("no, that's all")
    assert not caller_signed_off("Bye, but first — how much is it?")
