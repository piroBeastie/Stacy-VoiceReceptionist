"""The LLM client as it is built for a live call.

The plugin marks every tool `strict: true` by default, which makes the API
compile a grammar for constrained decoding on each request. Measured through
the real plugin, interleaved over three rounds: median 3762ms to first token
with strict, one request at 9185ms, against 809ms without. That was the whole of
the latency on live calls, so the default is a regression guard.
"""

from __future__ import annotations

import pytest

from dana.providers.base import LLMConfig
from dana.providers.registry import pipeline_for


def test_strict_tool_schemas_are_off_by_default() -> None:
    assert LLMConfig().strict_tool_schema is False


def test_a_call_is_built_with_strict_off(tenant, monkeypatch) -> None:
    monkeypatch.delenv("LLM_STRICT_TOOLS", raising=False)
    assert pipeline_for(tenant).llm.strict_tool_schema is False


def test_strict_can_be_switched_back_on_without_a_build(tenant, monkeypatch) -> None:
    # If a model ever needs it, that should be an environment variable and a
    # restart -- measured again first -- not a code change.
    monkeypatch.setenv("LLM_STRICT_TOOLS", "1")
    assert pipeline_for(tenant).llm.strict_tool_schema is True


def test_the_real_plugin_receives_it(monkeypatch) -> None:
    # Guards the handoff, not just the field: a misspelled keyword would be
    # dropped by `supported()` with only a warning, and strict would quietly
    # come back on through the plugin's own default.
    pytest.importorskip("livekit.plugins.anthropic")
    from dana.providers.registry import _one_llm

    monkeypatch.setenv("LLM_API_KEY", "sk-ant-test-not-a-real-key")
    model = _one_llm(LLMConfig(vendor="anthropic", model="claude-haiku-4-5-20251001"))
    assert model._opts.strict_tool_schema is False
