"""Turning provider config into live plugin instances.

The only module in the repo that imports vendor SDKs, and it does so inside the
factory functions rather than at module scope -- so config, prompts, tenants and
evals all stay importable and testable with no vendor packages installed.

VERIFY BEFORE FIRST CALL: the livekit-agents plugin constructor signatures below
are the surface most likely to have moved between SDK versions. Check them
against the installed version once, then trust this file.
"""

from __future__ import annotations

import os
from typing import Any

from dana.providers.base import LLMConfig, PipelineConfig, STTConfig, TTSConfig, TurnConfig
from dana.tenants.models import Tenant


def pipeline_for(tenant: Tenant, *, overrides: PipelineConfig | None = None) -> PipelineConfig:
    """Build a tenant's pipeline config from env defaults plus tenant specifics.

    Tenant data wins over env for anything business-shaped (voice, keywords);
    env wins for anything infrastructure-shaped (endpoints, model names), so
    moving every tenant to a new LLM host is one environment change.
    """
    base = overrides or PipelineConfig()
    return PipelineConfig(
        stt=STTConfig(
            vendor=base.stt.vendor,
            model=os.getenv("STT_MODEL", base.stt.model),
            language=base.stt.language,
            interim_results=base.stt.interim_results,
            keywords=tuple(tenant.asr_keywords),
        ),
        llm=LLMConfig(
            base_url=os.getenv("LLM_BASE_URL", base.llm.base_url),
            model=os.getenv("LLM_MODEL", base.llm.model),
            temperature=float(os.getenv("LLM_TEMPERATURE", base.llm.temperature)),
            max_tokens=base.llm.max_tokens,
            timeout_s=base.llm.timeout_s,
            fallback=base.llm.fallback,
        ),
        tts=TTSConfig(
            vendor=base.tts.vendor,
            model=os.getenv("TTS_MODEL", base.tts.model),
            voice_id=tenant.voice_id,
            stability=base.tts.stability,
            similarity_boost=base.tts.similarity_boost,
            speed=base.tts.speed,
            streaming_latency=base.tts.streaming_latency,
        ),
        turn=base.turn,
        max_call_duration_s=base.max_call_duration_s,
    )


def build_stt(cfg: STTConfig) -> Any:
    """Instantiate the STT plugin named by `cfg.vendor`."""
    if cfg.vendor == "deepgram":
        from livekit.plugins import deepgram

        return deepgram.STT(
            model=cfg.model,
            language=cfg.language,
            interim_results=cfg.interim_results,
            keywords=[(k, 1.0) for k in cfg.keywords] or None,
        )
    if cfg.vendor == "assemblyai":
        from livekit.plugins import assemblyai

        return assemblyai.STT()
    if cfg.vendor == "cartesia":
        from livekit.plugins import cartesia

        return cartesia.STT(model=cfg.model, language=cfg.language)
    raise ValueError(f"unknown STT vendor {cfg.vendor!r}")


def build_llm(cfg: LLMConfig) -> Any:
    """Instantiate the LLM client. Any OpenAI-compatible host works."""
    from livekit.plugins import openai

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "base_url": cfg.base_url,
        "api_key": os.environ["LLM_API_KEY"],
        "temperature": cfg.temperature,
    }
    if cfg.max_tokens is not None:
        kwargs["max_completion_tokens"] = cfg.max_tokens
    return openai.LLM(**kwargs)


def build_tts(cfg: TTSConfig) -> Any:
    """Instantiate the TTS plugin named by `cfg.vendor`.

    Both branches request mu-law 8k explicitly. Do not let this default: a PCM
    stream converted locally is a hidden transcode and it will not announce
    itself.
    """
    if cfg.vendor == "elevenlabs":
        from livekit.plugins import elevenlabs

        return elevenlabs.TTS(
            model=cfg.model,
            voice_id=cfg.voice_id,
            encoding="ulaw_8000",
            streaming_latency=cfg.streaming_latency,
            voice_settings=elevenlabs.VoiceSettings(
                stability=cfg.stability,
                similarity_boost=cfg.similarity_boost,
                speed=cfg.speed,
            ),
        )
    if cfg.vendor == "cartesia":
        from livekit.plugins import cartesia

        return cartesia.TTS(model=cfg.model, voice=cfg.voice_id, encoding="pcm_mulaw", sample_rate=8000)
    raise ValueError(f"unknown TTS vendor {cfg.vendor!r}")


def build_turn_detector(cfg: TurnConfig) -> Any | None:
    """The semantic end-of-utterance model, or None to fall back to raw VAD."""
    if not cfg.use_semantic_endpointing:
        return None
    from livekit.plugins.turn_detector.english import EnglishModel

    return EnglishModel()


def build_vad() -> Any:
    """Silero VAD. Runs in-process, sub-millisecond, no network hop."""
    from livekit.plugins import silero

    return silero.VAD.load()
