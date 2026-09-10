"""Vendor selection as data.

The point of leaving a bundled platform is that no vendor gets to be load-bearing.
So every vendor choice is a config record here, and the only code that knows a
vendor's name is :mod:`dana.providers.registry`.

Concretely, this is what makes the answer to "does any client forbid
third-party AI vendors?" a one-line change: point ``TTSConfig.vendor`` at
``cartesia`` instead of ``elevenlabs`` and nothing else in the repo moves.

Audio is pinned to 8kHz end to end and is deliberately not a per-vendor option.
Twilio speaks 8kHz mu-law; every resample on the path costs
20-50ms and some quality, and a transcode that sneaks in makes every latency
measurement downstream a lie. That pin is only real if each vendor is *told*
the format -- the plugins all default to 16k PCM, so the fields below are
passed explicitly rather than left to default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AudioFormat = Literal["ulaw_8000", "pcm_8000", "pcm_16000"]

TELEPHONY_FORMAT: AudioFormat = "pcm_8000"
"""8kHz linear PCM, not mu-law, and the difference is worth stating.

The ElevenLabs API will return `ulaw_8000`, and this was set to it on the
strength of that. The livekit plugin cannot ask for it: its accepted encodings
are mp3, opus and pcm only, so the value raised `ValueError: Unsupported
encoding` on the first turn that tried to speak.

`pcm_8000` is the honest best available. It matches the telephony sample rate
exactly, so nothing is resampled -- which was the actual point, resampling being
what costs quality. LiveKit converts PCM to mu-law for the SIP leg, and that
conversion is a lossless-at-8kHz companding step, not a resample. The plugin's
own default, `mp3_22050_32`, would be worse on both counts: an MP3 decode plus a
22050 to 8000 resample on every utterance."""
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

    Two things are being chosen here and they are easy to confuse. The *host*
    sets time to first token -- the same open weights on Groq or Cerebras versus
    a general cloud is roughly 150ms against 500ms, and first-token time is the
    only LLM number a caller can hear. The *model* sets whether the answer is
    right.

    The default was Sonnet 5 on the reasoning that a wrong answer costs more
    than a slow one. Measured against this prompt and these tools, Sonnet takes
    about 3.0 seconds to its first token where Haiku 4.5 takes about 0.9 -- and
    three seconds of silence before the agent repeats a caller's own name back
    is not a slower call either, it is the caller deciding nobody is there.

    Haiku is the default now. The line that genuinely cannot slip does not
    depend on it: `dana.agent.safety` intercepts gas and CO before inference.
    LLM_MODEL is one environment variable, so going back is trivial -- and
    `scripts/eval_calls.py` is what should decide it, over archived calls,
    rather than either of these arguments.

    `scripts/eval_calls.py` is what settles this rather than argument -- swap
    the model, take fifty calls, compare the eight criteria.

    `temperature` is sent only where it is accepted -- see the attribute.
    """

    vendor: Literal["openai", "anthropic"] = "anthropic"
    """Which client to build. "openai" covers every OpenAI-compatible host --
    OpenAI itself, Groq, Cerebras, Fireworks, vLLM."""

    base_url: str = ""
    """Empty means the vendor's own default endpoint. Only meaningful for
    "openai", where it is what selects the host."""

    model: str = "claude-haiku-4-5-20251001"
    api_key_env: str = "LLM_API_KEY"
    """*Which environment variable* holds the key, not the key. The primary and
    the fallback are usually different vendors -- that is the whole point of the
    cascade -- and they cannot share one credential."""

    temperature: float | None = None
    """None means "do not send it", which is not the same as sending 0.

    The newest Anthropic models reject `temperature` outright -- a 400 marked
    non-retryable, so every turn fails instantly rather than degrading. It is
    still valid and still worth pinning to 0 on OpenAI-compatible hosts, which
    is why this is per-model rather than a global.

    The original reason for 0 was that the gas-leak script had to come out
    verbatim. That reason is gone: :mod:`dana.agent.safety` intercepts gas and
    CO before inference, so the one line that cannot slip no longer depends on
    the model's sampling at all."""

    frequency_penalty: float = 0.3
    """Discourages verbatim repetition. Preferred over temperature because it
    penalises reused tokens rather than flattening the whole distribution.
    OpenAI-compatible hosts only: the Anthropic API has no such parameter, so on
    that vendor repetition is held down by the prompt alone."""

    caching: bool = True
    """Ask the vendor to cache the prompt prefix. The system prompt is ~5,700
    tokens and 99.2% of it is byte-identical between turns, so this is the
    difference between paying for the tenant's whole knowledge base on every
    turn and paying for it once a call. OpenAI-compatible hosts do it
    automatically and ignore this; Anthropic requires asking."""

    max_tokens: int | None = None
    """None means unbounded, matching main's `max_tokens: -1`."""

    timeout_s: float = 4.0
    """Time to first token before falling to :attr:`fallback`. Named after the
    platform's `cascade_timeout_seconds`, which defaults to 4 and clamps to
    2-15."""

    fallback: "LLMConfig | None" = None
    """Second model tried on timeout or error. Set one before taking real
    traffic. This is where a fast open-weight model earns its place: when the
    smart primary has timed out, the comparison is no longer "smart versus
    fast", it is "fast versus dead air"."""


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

    soft_timeout_s: float = 1.5
    """Speak a filler if the model has produced nothing by now. -1 disables."""

    soft_timeout_phrases: tuple[str, ...] = ("Let me see.", "One sec.", "Okay, checking.")
    max_soft_timeouts_per_turn: int = 1
    """The wait this covers should be rare, and twice it has not been.

    A caller hears "One sec" before an answer that needed no covering, and the
    cover phrase becomes the thing that needs covering. Measured turns put the
    model's first token consistently under this, so what remains above it is a
    genuine stall -- a tool out at the network, or a model having a bad
    minute -- which is the only case worth speaking into."""
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

    noise_cancellation: Literal["bvc_telephony", "bvc", "nc", "off"] = "bvc_telephony"
    """Which of LiveKit's input filters to put in front of the caller's audio.

    "bvc_telephony" is Background Voice Cancellation tuned for 8kHz phone audio.
    It isolates the person actually talking to the agent and suppresses other
    voices and room noise, which is the difference between a receptionist who
    answers you and one who answers your television.

    Plain "nc" removes non-speech noise only, so a second person in the room
    still gets transcribed and still gets answered. "off" is for debugging what
    the filter is doing to a difficult caller.
    """

    vad_sample_rate: int = TELEPHONY_SAMPLE_RATE
    """Silero defaults to 16kHz. Telephony audio is 8kHz, so the default meant
    the detector ran on upsampled narrowband -- inventing no information and
    costing a resample on every frame."""

    vad_activation_threshold: float = 0.6
    """Raised from Silero's 0.5. Background speech sits lower in the mix than
    the caller, so a higher bar is most of what stops the agent replying to a
    television."""

    vad_min_speech_s: float = 0.10
    """Raised from 0.05. Fifty milliseconds is short enough that a door closing
    counts as the caller starting to talk."""

    vad_min_silence_s: float = 0.55
    """Silero's default. Lower cuts people off mid-sentence; the semantic
    detector, not this, owns the endpointing decision."""

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

    turn_timeout_s: float = 9.0
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
