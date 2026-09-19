"""The tools as the model sees them, built against the real SDK.

Every tool takes a RunContext, which the SDK injects and must keep out of the
schema the model sees. It resolves that annotation with get_type_hints against
the tool module's globals -- and every annotation in tools.py is a string,
because of `from __future__ import annotations`. Imported anywhere but module
scope, the name would not resolve, and the failure would surface as a crash when
a call starts rather than here.
"""

from __future__ import annotations

import pytest

pytest.importorskip("livekit.agents")

from stacy.agent import CallRuntime, DeskOperations  # noqa: E402
from stacy.agent.tools import build_function_tools  # noqa: E402
from stacy.desk import CalendarStore, MessageStore  # noqa: E402

TOOLS = {
    "check_availability",
    "lookup_customer",
    "record_details",
    "book_appointment",
    "dispatch_emergency",
    "take_message",
}


@pytest.fixture
def schemas(tenant, tmp_path) -> dict[str, dict]:
    from livekit.agents.llm import utils as llm_utils

    runtime = CallRuntime.build(
        tenant=tenant,
        call_id="schema",
        caller_id="+12035550147",
        calendar=CalendarStore(tmp_path / "cal"),
        messages=MessageStore(tmp_path / "msg"),
    )
    out = {}
    for tool in build_function_tools(DeskOperations(runtime)):
        fn = llm_utils.build_legacy_openai_schema(tool)
        fn = fn.get("function", fn)
        out[fn["name"]] = fn["parameters"]
    return out


def test_every_tool_builds(schemas) -> None:
    assert set(schemas) == TOOLS


def test_the_run_context_never_reaches_the_model(schemas) -> None:
    for name, params in schemas.items():
        assert "ctx" not in params.get("properties", {}), name
        assert "ctx" not in params.get("required", []), name


class TestANumberCaughtBeforeTheModelAnswers:
    """The hook against the SDK's real message types, not a stand-in."""

    def _agent(self, tenant, tmp_path, caller_id: str):
        from stacy.agent.nodes import build_agent

        runtime = CallRuntime.build(
            tenant=tenant,
            call_id="hook",
            caller_id=caller_id,
            calendar=CalendarStore(tmp_path / "cal"),
            messages=MessageStore(tmp_path / "msg"),
        )
        return runtime, build_agent(runtime)

    def test_a_read_out_number_is_saved_and_its_read_back_attached(self, tenant, tmp_path) -> None:
        import asyncio

        from livekit.agents.llm import ChatContext, ChatMessage

        runtime, agent = self._agent(tenant, tmp_path, "+919988444520")
        message = ChatMessage(role="user", content=["It's double 9, double 8, triple 4, 5 2 0."])
        asyncio.run(agent.on_user_turn_completed(ChatContext.empty(), message))

        assert runtime.slots.phone == "+919988444520"
        assert "plus nine-one, nine-nine-eight-eight-four" in (message.text_content or "")

    def test_an_ordinary_turn_is_left_exactly_as_said(self, tenant, tmp_path) -> None:
        import asyncio

        from livekit.agents.llm import ChatContext, ChatMessage

        _, agent = self._agent(tenant, tmp_path, "+919988444520")
        message = ChatMessage(role="user", content=["My AC is not cooling properly."])
        asyncio.run(agent.on_user_turn_completed(ChatContext.empty(), message))

        assert message.content == ["My AC is not cooling properly."]

    def test_the_agent_starts_with_the_briefing_not_the_per_call_prompt(self, tenant, tmp_path) -> None:
        _, agent = self._agent(tenant, tmp_path, "+919988444520")
        assert "+919988444520" not in agent.instructions
        first = agent.chat_ctx.items[0]
        assert first.role == "user" and "not said aloud" in (first.text_content or "")

    def test_a_day_and_window_the_calendar_refuses_get_a_note(self, tenant, tmp_path) -> None:
        import asyncio
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from livekit.agents.llm import ChatContext, ChatMessage

        from stacy.agent.nodes import build_agent

        friday = datetime(2026, 9, 11, 10, 30, tzinfo=ZoneInfo(tenant.timezone))
        runtime = CallRuntime.build(
            tenant=tenant,
            call_id="fri",
            caller_id="+919988444520",
            calendar=CalendarStore(tmp_path / "cal"),
            messages=MessageStore(tmp_path / "msg"),
            now=friday,
        )
        agent = build_agent(runtime)
        message = ChatMessage(role="user", content=["You can come tomorrow, 11 to 2."])
        asyncio.run(agent.on_user_turn_completed(ChatContext.empty(), message))

        assert "Tomorrow we're only open nine to one, so there's no eleven to two." in (
            message.text_content or ""
        )
        assert (runtime.slots.day, runtime.slots.window) == ("", "")


class TestANumberCorrectedOverTwoTurns:
    def test_the_callers_own_number_is_caught_across_two_turns(self, tenant, tmp_path) -> None:
        import asyncio

        from livekit.agents.llm import ChatContext, ChatMessage

        from stacy.agent.nodes import build_agent

        runtime = CallRuntime.build(
            tenant=tenant,
            call_id="split",
            caller_id="+919988444520",
            calendar=CalendarStore(tmp_path / "cal"),
            messages=MessageStore(tmp_path / "msg"),
        )
        agent = build_agent(runtime)
        first = ChatMessage(role="user", content=["And my number is double 9, double 8, triple 4, 5 0."])
        asyncio.run(agent.on_user_turn_completed(ChatContext.empty(), first))
        assert runtime.slots.phone == ""

        second = ChatMessage(role="user", content=["5 2 0."])
        asyncio.run(agent.on_user_turn_completed(ChatContext.empty(), second))
        assert runtime.slots.phone == "+919988444520"
        assert "plus nine-one, nine-nine-eight-eight-four" in (second.text_content or "")


def test_the_context_name_resolves_at_module_scope() -> None:
    import stacy.agent.tools as module
    from livekit.agents import RunContext

    assert module.RunContext is RunContext
