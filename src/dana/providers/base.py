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
measurement downstream a lie. That pin is only real if each vendor is *told*
the format -- the plugins all default to 16k PCM, so the fields below are
passed explicitly rather than left to default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AudioFormat = Literal["ulaw_8000", "pcm_16000"]

TELEPHONY_FORMAT: AudioFormat = "ulaw_8000"
"""The only format that touches the wire. See module docstring."""

TELEPHONY_SAMPLE_RATE = 8000


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
    """Boosted terms -- town names, staff names, the agent's own name.

    Deepgram changed the mechanism at Nova-3: `keywords` is a Nova-2-and-earlier
    parameter and is rejected by Nova-3, which uses keyterm prompting instead.
    :attr:`boost_param` picks the right one so tenant data doesn't have to know.
    """

    audio_format: AudioFormat = TELEPHONY_FORMAT
    sample_rate: int = TELEPHONY_SAMPLE_RATE
    """Passed explicitly. Left to default, the plugin upsamples to 16k PCM and
    the 'mu-law end to end' claim quietly stops being true on the input leg."""

    endpointing_ms: int = 25
    """Vendor-side silence before Deepgram flushes a final. Kept low because the
    semantic turn detector, not the vendor, owns the endpointing decision."""

    utterance_end_ms: int = 1000
    """Vendor-side UtteranceEnd signal. A backstop for a missed final, not the
    primary endpoint."""

    numerals: bool = True
    """Digits as digits. The prompt reads phone numbers back one digit at a
    time; 'two zero three' arriving as words makes that read-back wrong."""

    filler_words: bool = True
    """Keep 'uh' and 'um' in the transcript. The backchannel filter classifies
    them, so stripping them upstream hides the very tokens it exists to catch."""

    @property
    def boost_param(self) -> Literal["keyterm", "keywords", None]:
        """Which Deepgram term-boosting parameter this model accepts."""
        if self.vendor != "deepgram" or not self.keywords:
            return None
        return "keyterm" if self.model.startswith("nova-3") else "keywords"


@dataclass(frozen=True)
class LLMConfig:
    """The conversational model.

    Any OpenAI-compatible endpoint. Host matters more than model here: the same
    weights on Groq or Cerebras versus a general cloud is the difference between
    ~150ms and ~500ms to first token, and first-token time is the only LLM
    number a caller can hear.

    `temperature` stays at 0.0. The prompt forbids repeating a sentence within a
    call, and the usual fix -- raising temperature -- also loosens grip on the
    gas-leak script, which is the one line that cannot slip.
    :attr:`frequency_penalty` buys variety without touching that grip.
    """

    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "qwen-3-235b-a22b-instruct"
    temperature: float = 0.0
    frequency_penalty: float = 0.3
    """Discourages verbatim repetition. Preferred over temperature because it
    penalises reused tokens rather than flattening the whole distribution."""

    max_tokens: int | None = None
    """None means unbounded, matching main's `max_tokens: -1`."""

    timeout_s: float = 4.0
    """Time to first token before falling to :attr:`fallback`. Named after the
    platform's `cascade_timeout_seconds`, which defaults to 4 and clamps to
    2-15."""

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

    first_chunk_chars: int = 50
    """First entry of the streaming websocket's chunk-length schedule.

    The vendor default schedule is [120, 160, 250, 290] *characters*: no audio
    is synthesized until 120 characters of model output have buffered. On a
    two-sentence answer that is most of the reply. Dropping the first entry
    starts audio after the opening clause instead, at some cost in prosody
    across the seam.

    Note there is no `optimize_streaming_latency` here. That parameter is a
    documented no-op on current models and setting it only creates the
    impression the path is tuned.
    """

    chunk_length_schedule: tuple[int, ...] = (50, 160, 250, 290)


@dataclass(frozen=True)
class FillerConfig:
    """Speech that covers a wait, so silence never has to.

    Two different waits need covering, and they are not the same wait:

    * the model is slow to its first token (:attr:`soft_timeout_s`)
    * a tool is out at the network (:attr:`pre_tool_threshold_s`)

    Both are the platform's behaviour rebuilt here, including the detail that
    fillers are suppressed until the caller has actually said something --
    otherwise the agent mutters over its own greeting.
    """

    enabled: bool = True

    soft_timeout_s: float = 0.9
    """Speak a filler if the model has produced nothing by now. -1 disables."""

    soft_timeout_phrases: tuple[str, ...] = ("Let me see.", "One sec.", "Okay, checking.")
    max_soft_timeouts_per_turn: int = 1
    """More than one filler in a single turn reads as a stall, not a person."""

    pre_tool_threshold_s: float = 0.4
    """Speak before a tool call whose recent latency exceeds this. Mirrors the
    platform's `pre_tool_speech: auto` -- decide from observed latency rather
    than annotating every tool by hand."""

    pre_tool_phrases: tuple[str, ...] = ("Let me check that.", "One moment.")

    suppress_until_first_user_message: bool = True
    """No fillers before the caller has spoken."""


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
    min_endpointing_delay_s: float = 0.2
    """Wait after an utterance the detector judges complete.

    Raised from 0.12: below roughly 0.2s the detector is being asked to commit
    on a transcript the vendor has not finished revising, and the failure mode
    is cutting a caller off mid-sentence -- which costs a full turn to repair
    and reads as far less human than the 80ms it saved.
    """

    max_endpointing_delay_s: float = 2.5
    """Ceiling when the detector judges the caller is mid-thought."""

    preemptive_generation: bool = True
    """Start the LLM on interim transcript; cancel if the caller keeps talking.

    Costs some wasted tokens, removes STT finalization from the critical path.
    On a cheap fast model that trade is heavily one-sided.
    """

    interrupt_min_words: int = 3
    """Below this, an utterance must clear the backchannel filter to interrupt.

    Passed to the session as well as to the filter. Held only in the filter, it
    classified correctly and suppressed nothing.
    """

    false_interruption_timeout_s: float = 1.5
    """If a barge-in produces no real speech within this window, treat it as a
    false trigger and resume. Without it, a cough costs the caller the rest of
    the sentence they were being read."""

    first_message_interruptible: bool = True
    """Whether the caller can talk over the greeting.

    True with :attr:`transcribe_when_uninterruptible` is the humane pairing. If
    this is set False for a legal disclosure, the transcription flag matters
    more, not less: a caller talking over a notice they have heard before is
    otherwise both ignored and unrecorded.
    """

    transcribe_when_uninterruptible: bool = True
    """Keep transcribing speech that is not allowed to interrupt, so it carries
    into the next turn instead of being discarded."""

    turn_timeout_s: float = 7.0
    """Silence before the agent re-engages. The prompt tells it to ask 'Still
    there?'; without a timer it never gets the turn in which to do so."""

    silence_end_call_s: float = 30.0
    """Silence before the call is closed out."""


@dataclass(frozen=True)
class PipelineConfig:
    """Everything the runtime needs to build one call's pipeline."""

    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    turn: TurnConfig = field(default_factory=TurnConfig)
    filler: FillerConfig = field(default_factory=FillerConfig)
    max_call_duration_s: int = 360
    """Matches main. A demo line needs a hard cap; a client line may not."""
