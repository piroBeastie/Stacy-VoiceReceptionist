"""Caching the one thing said on every call.

The greeting is verbatim tenant config -- it carries the AI disclosure and the
recording notice -- so it is fully known before the phone rings, and it was
being synthesised from scratch every time. Measured against the live API that
cost 334-572ms before the caller heard a syllable: the one stall no filler can
cover, because it happens before the first word.

These tests are about the cache being *safe*, not fast. A stale greeting is
worse than a slow one -- it means the disclosure a caller hears is not the one
the tenant configured.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from dana.agent import greeting
from dana.tenants import Tenant


class TestTheCacheKey:
    def test_editing_the_greeting_invalidates_it(self) -> None:
        # By construction, not by anyone remembering to clear a directory.
        a = greeting._key("Thanks for calling.", "v1", "flash", "pcm_8000")
        b = greeting._key("Thanks for calling!", "v1", "flash", "pcm_8000")
        assert a != b

    @pytest.mark.parametrize(
        "changed",
        [("v2", "flash", "pcm_8000"), ("v1", "turbo", "pcm_8000"), ("v1", "flash", "pcm_16000")],
    )
    def test_a_different_voice_model_or_format_is_a_different_recording(self, changed) -> None:
        base = greeting._key("Hello.", "v1", "flash", "pcm_8000")
        assert greeting._key("Hello.", *changed) != base

    def test_the_same_inputs_are_stable_across_runs(self) -> None:
        # A key that moved between restarts would re-synthesise forever.
        args = ("Hello.", "v1", "flash", "pcm_8000")
        assert greeting._key(*args) == greeting._key(*args)


class TestTheCacheOnDisk:
    def test_a_stored_greeting_comes_back_intact(self, tmp_path) -> None:
        greeting.store(tmp_path, "k", b"\x01\x02\x03\x04", 8000, 1)
        assert greeting.load(tmp_path, "k") == (b"\x01\x02\x03\x04", 8000, 1)

    def test_nothing_stored_is_a_miss_not_a_crash(self, tmp_path) -> None:
        assert greeting.load(tmp_path, "never-written") is None

    def test_a_corrupt_sidecar_is_a_miss_not_an_outage(self, tmp_path) -> None:
        # Half-written cache on a killed worker must not take the next call down.
        raw, meta = greeting.cache_paths(tmp_path, "k")
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"\x00\x00")
        meta.write_text("{not json", encoding="utf-8")
        assert greeting.load(tmp_path, "k") is None

    def test_audio_without_its_sidecar_is_a_miss(self, tmp_path) -> None:
        raw, _ = greeting.cache_paths(tmp_path, "k")
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"\x00\x00")
        assert greeting.load(tmp_path, "k") is None


class TestPrepare:
    def test_a_tenant_with_no_greeting_is_not_an_error(self, tenant: Tenant, tmp_path) -> None:
        blank = tenant.model_copy(update={"greeting": "   "})
        result = asyncio.run(greeting.prepare(blank, object(), tmp_path, lambda _cfg: None))
        assert result is None

    def test_a_synthesis_failure_falls_back_instead_of_failing_the_call(
        self, tenant: Tenant, tmp_path
    ) -> None:
        # Returning None means "let the session synthesise it live". A cache
        # that could drop the greeting entirely would be worse than no cache.
        class Cfg:
            voice_id, model, audio_format = "v1", "flash", "pcm_8000"

        def explode(_cfg):
            raise RuntimeError("elevenlabs is down")

        assert asyncio.run(greeting.prepare(tenant, Cfg(), tmp_path, explode)) is None

    def test_a_cache_hit_never_calls_the_vendor(self, tenant: Tenant, tmp_path) -> None:
        class Cfg:
            voice_id, model, audio_format = "v1", "flash", "pcm_8000"

        key = greeting._key(tenant.greeting.strip(), "v1", "flash", "pcm_8000")
        greeting.store(tmp_path, key, b"\x00\x00" * 80, 8000, 1)

        def must_not_be_called(_cfg):
            raise AssertionError("synthesised despite a warm cache")

        assert asyncio.run(greeting.prepare(tenant, Cfg(), tmp_path, must_not_be_called)) is not None
