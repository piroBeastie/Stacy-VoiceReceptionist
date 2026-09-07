"""Vendor selection as data.

The point of leaving a bundled platform is that no vendor gets to be load-bearing.
So every vendor choice is a config record here, and the only code that knows a
vendor's name is :mod:`dana.providers.registry`.

Concretely, this is what makes the answer to "does any client forbid
third-party AI vendors?" a one-line change: point ``TTSConfig.vendor`` at
``cartesia`` instead of ``elevenlabs`` and nothing else in the repo moves.

Audio format is pinned to mu-law 8kHz end to end and is deliberately not a
per-vendor option. Twilio speaks mu-law; every resample on the path costs
20-50ms and some quality, and a transcode that sneaks in makes every latency
measurement downstream a lie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AudioFormat = Literal["ulaw_8000", "pcm_16000"]

TELEPHONY_FORMAT: AudioFormat = "ulaw_8000"
"""The only format that touches the wire. See module docstring."""


@dataclass(frozen=True)
class STTConfig:
    """Speech-to-text.

    `interim_results` is required, not cosmetic: preemptive generation reads
    interim transcripts, and without them the LLM cannot start early.
    """

    vendor: Literal["deepgram", "assemblyai", "cartesia"] = "deepgram"
    model: str = "nova-3"
    language: str = "en-US"
    interim_results: bool = True
    keywords: tuple[str, ...] = ()
    """Boosted terms -- town names, staff names, the agent's own name."""

    audio_format: AudioFormat = TELEPHONY_FORMAT


@dataclass(frozen=True)
class LLMConfig:
    """The conversational model.

    Any OpenAI-compatible endpoint. Host matters more than model here: the same
    weights on Groq or Cerebras versus a general cloud is the difference between
    ~150ms and ~500ms to first token, and first-token time is the only LLM
    number a caller can hear.

    `temperature` defaults to 0.0 to match main. The nanak branch's 0.4 loosens
    grip on scripted lines -- if it is raised, the safety-override eval runs
    first, every time.
    """

    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "qwen-3-235b-a22b-instruct"
    temperature: float = 0.0
    max_tokens: int | None = None
    """None means unbounded, matching main's `max_tokens: -1`."""

    timeout_s: float = 4.0
    """Matches main's `cascade_timeout_seconds`. Past this, fall to the backup."""

    fallback: "LLMConfig | None" = None
    """Second model tried on timeout or error. Set one before taking real traffic."""


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-speech.

    Two vendors are wired because this is the most likely one to be swapped:
    ElevenLabs keeps Casey's validated voice, Cartesia is faster and cheaper and
    requires re-casting the persona.
    """

    vendor: Literal["elevenlabs", "cartesia"] = "elevenlabs"
    model: str = "eleven_flash_v2_5"
    """Flash, not v3: v3's expressiveness costs latency the phone line can hear."""

    voice_id: str = ""
    stability: float = 0.5
    similarity_boost: float = 0.8
    speed: float = 1.0
    audio_format: AudioFormat = TELEPHONY_FORMAT
    streaming_latency: int = 3
    """Mirrors main's `optimize_streaming_latency: 3`."""


@dataclass(frozen=True)
class TurnConfig:
    """Endpointing policy -- the largest single latency lever in the stack.

    A fixed silence timer forces a choice between cutting off slow talkers and
    adding half a second to every turn, and there is no good setting. The
    semantic detector reads the partial transcript and picks the wait per
    utterance instead: short when the sentence is clearly finished, long when it
    obviously is not.
    """

    use_semantic_endpointing: bool = True
    min_endpointing_delay_s: float = 0.12
    """Wait after an utterance the detector judges complete."""

    max_endpointing_delay_s: float = 2.5
    """Ceiling when the detector judges the caller is mid-thought."""

    preemptive_generation: bool = True
    """Start the LLM on interim transcript; cancel if the caller keeps talking.

    Costs some wasted tokens, removes STT finalization from the critical path.
    On a cheap fast model that trade is heavily one-sided.
    """

    interrupt_min_words: int = 3
    """Below this, an utterance must clear the backchannel filter to interrupt."""

    turn_timeout_s: float = 7.0
    silence_end_call_s: float = 30.0


@dataclass(frozen=True)
class PipelineConfig:
    """Everything the runtime needs to build one call's pipeline."""

    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    turn: TurnConfig = field(default_factory=TurnConfig)
    max_call_duration_s: int = 360
    """Matches main. A demo line needs a hard cap; a client line may not."""
