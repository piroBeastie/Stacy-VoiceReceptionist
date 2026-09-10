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


def test_the_context_name_resolves_at_module_scope() -> None:
    import stacy.agent.tools as module
    from livekit.agents import RunContext

    assert module.RunContext is RunContext
