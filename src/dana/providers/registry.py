"""Turning provider config into live plugin instances.

The only module in the repo that imports vendor SDKs, and it does so inside the
factory functions rather than at module scope -- so config, prompts, tenants and
evals all stay importable and testable with no vendor packages installed.

The README used to say "verify the plugin constructor signatures against the
installed SDK before the first call, then trust this file." That instruction has
never been carried out, and it is the wrong shape of instruction anyway: it asks
a human to re-check on every dependency bump and fails silently when they don't.
:func:`supported` does it at construction time instead -- unsupported keyword
arguments are dropped with a warning naming the plugin and the argument, so a
version skew shows up in the first line of the worker log rather than as a
TypeError on the first inbound call.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from typing import Any, Callable

from dana.providers.base import (
    TELEPHONY_FORMAT,
    FillerConfig,
    LLMConfig,
    PipelineConfig,
    STTConfig,
    TTSConfig,
    TurnConfig,
)
from dana.tenants.models import Tenant

log = logging.getLogger("dana.providers")


def supported(factory: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keyword arguments `factory` does not accept, loudly.

    A plugin that gained or lost a parameter between versions then degrades to
    "that tuning is not applied", which is visible and recoverable, instead of
    "the worker raised TypeError while a caller was on the line".
    """
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):  # C extension, no introspectable signature
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    ok = {k: v for k, v in kwargs.items() if k in params}
    for k in kwargs.keys() - ok.keys():
        log.warning(
            "%s does not accept %r on this SDK version -- tuning not applied",
            getattr(factory, "__qualname__", factory),
            k,
        )
    return ok


def pipeline_for(tenant: Tenant, *, overrides: PipelineConfig | None = None) -> PipelineConfig:
    """Build a tenant's pipeline config from env defaults plus tenant specifics.

    Tenant data wins over env for anything business-shaped (voice, keywords);
    env wins for anything infrastructure-shaped (endpoints, model names), so
    moving every tenant to a new LLM host is one environment change.
    """
    base = overrides or PipelineConfig()

    fallback = base.llm.fallback
    if fallback is None and os.getenv("LLM_FALLBACK_MODEL"):
        # The fallback is normally a different vendor on a different account --
        # a fast open-weight host standing behind a frontier primary -- so it
        # gets its own vendor, endpoint and key variable rather than inheriting
        # the primary's.
        fallback = LLMConfig(
            vendor=os.getenv("LLM_FALLBACK_VENDOR", "openai"),  # type: ignore[arg-type]
            base_url=os.getenv("LLM_FALLBACK_BASE_URL", ""),
            model=os.environ["LLM_FALLBACK_MODEL"],
            api_key_env=os.getenv("LLM_FALLBACK_API_KEY_ENV", "LLM_FALLBACK_API_KEY"),
            temperature=_opt_float("LLM_FALLBACK_TEMPERATURE", 0.0),
            frequency_penalty=base.llm.frequency_penalty,
            caching=base.llm.caching,
            max_tokens=base.llm.max_tokens,
            timeout_s=base.llm.timeout_s,
        )

    return PipelineConfig(
        stt=STTConfig(
            vendor=base.stt.vendor,
            model=os.getenv("STT_MODEL", base.stt.model),
            language=tenant.stt_language,
            interim_results=base.stt.interim_results,
            keywords=tuple(tenant.asr_keywords),
            sample_rate=base.stt.sample_rate,
            endpointing_ms=base.stt.endpointing_ms,
            utterance_end_ms=base.stt.utterance_end_ms,
            numerals=base.stt.numerals,
            filler_words=base.stt.filler_words,
        ),
        llm=LLMConfig(
            vendor=os.getenv("LLM_VENDOR", base.llm.vendor),  # type: ignore[arg-type]
            base_url=os.getenv("LLM_BASE_URL", base.llm.base_url),
            model=os.getenv("LLM_MODEL", base.llm.model),
            api_key_env=os.getenv("LLM_API_KEY_ENV", base.llm.api_key_env),
            caching=os.getenv("LLM_CACHING", "1") not in {"0", "false", "no"},
            temperature=_opt_float("LLM_TEMPERATURE", base.llm.temperature),
            frequency_penalty=float(
                os.getenv("LLM_FREQUENCY_PENALTY", base.llm.frequency_penalty)
            ),
            max_tokens=base.llm.max_tokens,
            timeout_s=float(os.getenv("LLM_TIMEOUT_S", base.llm.timeout_s)),
            fallback=fallback,
        ),
        tts=TTSConfig(
            vendor=base.tts.vendor,
            model=os.getenv("TTS_MODEL", base.tts.model),
            voice_id=tenant.voice_id,
            # Delivery is business-shaped, like the voice itself: one client
            # wants a brisk receptionist, another wants an unhurried one, and
            # that is not an infrastructure setting to be shared between them.
            stability=tenant.voice_stability,
            similarity_boost=tenant.voice_similarity,
            speed=tenant.voice_speed,
            first_chunk_chars=base.tts.first_chunk_chars,
            chunk_length_schedule=base.tts.chunk_length_schedule,
        ),
        turn=base.turn,
        filler=base.filler,
        max_call_duration_s=base.max_call_duration_s,
    )


def build_stt(cfg: STTConfig) -> Any:
    """Instantiate the STT plugin named by `cfg.vendor`.

    Format is passed explicitly. Deepgram's plugin defaults to 16kHz PCM, so
    leaving it alone means LiveKit resamples 8k mu-law up on the way in -- a
    hidden transcode on the leg the README claims is transcode-free.
    """
    if cfg.vendor == "deepgram":
        from livekit.plugins import deepgram  # noqa: PLC0415

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            # Passed explicitly rather than left to the plugin's own env lookup.
            # Each vendor's plugin has its own idea of what the variable is
            # called, and one of them disagreed with this repo (see build_tts).
            "api_key": _key("DEEPGRAM_API_KEY"),
            "language": cfg.language,
            "interim_results": cfg.interim_results,
            "sample_rate": cfg.sample_rate,
            "endpointing_ms": cfg.endpointing_ms,
            "numerals": cfg.numerals,
            "filler_words": cfg.filler_words,
        }
        # Nova-3 replaced `keywords` with keyterm prompting and rejects the old
        # parameter, so the tenant's boosted terms were silently doing nothing.
        boost = cfg.boost_param
        if boost == "keyterm":
            kwargs["keyterm"] = list(cfg.keywords)
        elif boost == "keywords":
            kwargs["keywords"] = [(k, 1.0) for k in cfg.keywords]
        return deepgram.STT(**supported(deepgram.STT, kwargs))

    if cfg.vendor == "assemblyai":
        from livekit.plugins import assemblyai  # noqa: PLC0415

        return assemblyai.STT()

    if cfg.vendor == "cartesia":
        from livekit.plugins import cartesia  # noqa: PLC0415

        return cartesia.STT(**supported(cartesia.STT, {"model": cfg.model, "language": cfg.language}))

    raise ValueError(f"unknown STT vendor {cfg.vendor!r}")


def register_plugins() -> None:
    """Import the vendor plugins on the main thread, before any worker starts.

    livekit-agents registers a plugin as a side effect of importing its module,
    and refuses to do that off the main thread. Every import in this file is
    lazy and inside a function -- which is what keeps the test suite vendor-free
    -- so the first import happened inside `prewarm`, on a job-runner thread,
    and raised. Prewarm then fell back to loading the models inline on the first
    call, which is the exact stall it was written to remove: the caller waits
    through a disk read and a session build after the line connects.

    Called from `main()` before `cli.run_app`, so registration happens once, on
    the main thread, while nobody is listening.
    """
    for module in (
        "livekit.plugins.silero",
        "livekit.plugins.turn_detector",
        "livekit.plugins.deepgram",
        "livekit.plugins.elevenlabs",
        "livekit.plugins.openai",
        "livekit.plugins.anthropic",
        "livekit.plugins.cartesia",
        "livekit.plugins.assemblyai",
    ):
        try:
            importlib.import_module(module)
        except ImportError:
            # Optional vendors. A tenant configured for one that is not
            # installed fails later, in build_*, naming the vendor.
            log.debug("%s not installed", module)


def _opt_float(name: str, default: float | None) -> float | None:
    """Read a float that is allowed to be absent.

    An empty value means "do not send this parameter", which is a different
    instruction from "send zero" and the distinction matters: some hosts reject
    the parameter entirely.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return float(raw) if raw else None


def _key(name: str) -> str:
    """Read a required credential, naming the variable if it is missing.

    "ElevenLabs API key is required" does not tell you which of the several
    plausible variable names this deployment expects. This does.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise KeyError(f"{name} is unset or empty")
    return value


def _one_llm(cfg: LLMConfig) -> Any:
    """Build one client. The cascade calls this twice, for two different vendors.

    The key is looked up through `cfg.api_key_env` rather than read from a fixed
    `LLM_API_KEY`. That indirection is the whole reason a cross-vendor cascade
    is possible: a frontier primary and a fast open-weight fallback are two
    accounts, and sharing one variable between them silently sent the fallback
    the primary's credential.
    """
    key = os.environ.get(cfg.api_key_env, "").strip()
    if not key:
        raise KeyError(
            f"{cfg.api_key_env} is unset, so {cfg.model!r} has no credentials. "
            "Each model in the cascade names its own key variable."
        )

    if cfg.vendor == "anthropic":
        from livekit.plugins import anthropic  # noqa: PLC0415

        kwargs: dict[str, Any] = {"model": cfg.model, "api_key": key}
        # Sent only when set. The newest Anthropic models reject `temperature`
        # with a 400 marked non-retryable, and because it is a perfectly valid
        # *plugin* parameter, `supported()` cannot see the problem -- the
        # rejection comes from the API. Every turn would have failed instantly.
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.caching:
            # Anthropic does not cache unless asked. Unasked, every turn pays
            # for the full ~5,700-token tenant prompt in both money and
            # time-to-first-token. If `supported` drops this, the warning in the
            # worker log is the only sign -- check the first line on startup.
            kwargs["caching"] = "ephemeral"
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        # frequency_penalty is deliberately not passed: the Anthropic API has no
        # such parameter, and sending it would log a dropped-argument warning on
        # every single call for something that was never going to apply.
        return anthropic.LLM(**supported(anthropic.LLM, kwargs))

    from livekit.plugins import openai  # noqa: PLC0415

    kwargs = {"model": cfg.model, "api_key": key}
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    # livekit-plugins-openai 1.8 has no `frequency_penalty` parameter, so this
    # tuning was being dropped by `supported()` on every call -- the config
    # documented it as the reason temperature could stay at 0, and it had never
    # once been applied. `extra_body` is the plugin's pass-through to the host,
    # which is where the parameter actually lives.
    if cfg.frequency_penalty:
        kwargs["extra_body"] = {"frequency_penalty": cfg.frequency_penalty}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    if cfg.max_tokens is not None:
        kwargs["max_completion_tokens"] = cfg.max_tokens
    return openai.LLM(**supported(openai.LLM, kwargs))


def build_llm(cfg: LLMConfig) -> Any:
    """Instantiate the LLM client, wrapped in a fallback if one is configured.

    `LLMConfig.fallback` and `timeout_s` existed as fields for a while without
    anything reading them, which meant a rate-limited primary was a silent
    ten-second hole in the call rather than a slightly worse answer.
    """
    primary = _one_llm(cfg)
    if cfg.fallback is None:
        return primary
    try:
        from livekit.agents import llm as agents_llm  # noqa: PLC0415

        adapter = agents_llm.FallbackAdapter
    except (ImportError, AttributeError):
        log.warning(
            "installed livekit-agents has no llm.FallbackAdapter -- "
            "running on %s alone with no cascade",
            cfg.model,
        )
        return primary
    return adapter(
        [primary, _one_llm(cfg.fallback)],
        **supported(adapter, {"attempt_timeout": cfg.timeout_s}),
    )


def build_tts(cfg: TTSConfig) -> Any:
    """Instantiate the TTS plugin named by `cfg.vendor`.

    Both branches pin the sample rate to 8k explicitly. Do not let this default:
    the ElevenLabs plugin defaults to `mp3_22050_32`, which is an MP3 decode and
    a 22050-to-8000 resample on every utterance, and it will not announce
    itself.

    `optimize_streaming_latency` is deliberately absent. It is a documented
    no-op on current ElevenLabs models; the parameter that actually governs
    time-to-first-audio is the chunk schedule.
    """
    if cfg.vendor == "elevenlabs":
        from livekit.plugins import elevenlabs  # noqa: PLC0415

        schedule = (cfg.first_chunk_chars, *cfg.chunk_length_schedule[1:])
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            # The plugin's own fallback reads ELEVEN_API_KEY, which is not the
            # name used anywhere else in this repo, in .env.example, or in
            # preflight. Setting the documented variable and getting "API key is
            # required" is a genuinely baffling five minutes; pass it explicitly.
            "api_key": _key("ELEVENLABS_API_KEY"),
            "voice_id": cfg.voice_id,
            "encoding": TELEPHONY_FORMAT,
            "chunk_length_schedule": list(schedule),
            "voice_settings": elevenlabs.VoiceSettings(
                stability=cfg.stability,
                similarity_boost=cfg.similarity_boost,
                speed=cfg.speed,
            ),
        }
        return elevenlabs.TTS(**supported(elevenlabs.TTS, kwargs))

    if cfg.vendor == "cartesia":
        from livekit.plugins import cartesia  # noqa: PLC0415

        kwargs = {
            "model": cfg.model,
            "api_key": _key("CARTESIA_API_KEY"),
            "voice": cfg.voice_id,
            "encoding": "pcm_mulaw",
            "sample_rate": 8000,
            "speed": cfg.speed,
        }
        return cartesia.TTS(**supported(cartesia.TTS, kwargs))

    raise ValueError(f"unknown TTS vendor {cfg.vendor!r}")


def build_turn_detector(cfg: TurnConfig) -> Any | None:
    """The semantic end-of-utterance model, or None to fall back to raw VAD.

    Loading this costs hundreds of milliseconds, so callers should prefer the
    prewarmed instance in `proc.userdata` -- see :func:`prewarm`.
    """
    if not cfg.use_semantic_endpointing:
        return None
    from livekit.plugins.turn_detector.english import EnglishModel  # noqa: PLC0415

    return EnglishModel()


def build_vad(cfg: TurnConfig | None = None) -> Any:
    """Silero VAD. Runs in-process, sub-millisecond, no network hop.

    Tuned rather than defaulted. Silero assumes 16kHz and a 50ms minimum
    utterance, which on a phone line means running on upsampled narrowband and
    treating a door closing as the caller starting to speak.
    """
    from livekit.plugins import silero  # noqa: PLC0415

    cfg = cfg or TurnConfig()
    kwargs: dict[str, Any] = {
        "sample_rate": cfg.vad_sample_rate,
        "activation_threshold": cfg.vad_activation_threshold,
        "min_speech_duration": cfg.vad_min_speech_s,
        "min_silence_duration": cfg.vad_min_silence_s,
    }
    return silero.VAD.load(**supported(silero.VAD.load, kwargs))


def build_noise_cancellation(cfg: TurnConfig) -> Any | None:
    """The input filter that decides whose voice the agent is listening to.

    Returned rather than applied here so the entrypoint can hand it to
    RoomInputOptions, and so a version of the SDK without the parameter degrades
    to no filtering instead of a TypeError mid-call.
    """
    if cfg.noise_cancellation == "off":
        return None
    try:
        from livekit.plugins import noise_cancellation  # noqa: PLC0415
    except ImportError:
        log.warning(
            "livekit-plugins-noise-cancellation is not installed -- background "
            "voices will be transcribed and answered alongside the caller"
        )
        return None

    factory = {
        "bvc_telephony": "BVCTelephony",
        "bvc": "BVC",
        "nc": "NC",
    }[cfg.noise_cancellation]
    made = getattr(noise_cancellation, factory, None)
    if made is None:
        log.warning("noise cancellation %r unavailable on this plugin version", factory)
        return None
    return made()


def prewarm(userdata: dict[str, Any], *, turn: TurnConfig | None = None) -> None:
    """Load Silero once per worker process, before any call arrives.

    Silero reads weights off disk and builds a session. Constructing it inside
    the job entrypoint -- which is where it used to live -- put that cost
    between the caller hearing a ring and the agent saying anything, on every
    cold process.

    The end-of-utterance detector is deliberately **not** prewarmed here, though
    it used to be attempted. On livekit-agents 1.8 its weights live in the
    worker's shared inference executor, which is started once and serves every
    job; the object built per call is only a handle to it, and constructing that
    handle outside a job raises "no job context found". So the expensive half is
    already warm by the time a call lands, and trying to hoist the cheap half
    produced a stack trace on every job runner at startup and warned about a
    stall that was not happening.

    Failures here are logged and swallowed: a worker that cannot prewarm should
    still take calls, just slower on the first one.
    """
    turn = turn or TurnConfig()
    try:
        userdata["vad"] = build_vad(turn)
    except Exception:  # noqa: BLE001 - a cold VAD is better than a dead worker
        log.exception("VAD prewarm failed; the first call will load it inline")


__all__ = [
    "FillerConfig",
    "build_llm",
    "build_stt",
    "build_tts",
    "build_turn_detector",
    "build_noise_cancellation",
    "build_vad",
    "pipeline_for",
    "prewarm",
    "supported",
]
