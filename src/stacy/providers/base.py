"""Vendor selection as data.

The point of leaving a bundled platform is that no vendor gets to be load-bearing.
So every vendor choice is a config record here, and the only code that knows a
vendor's name is :mod:`stacy.providers.registry`.

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
    """The format on the wire from Twilio. Not what reaches Deepgram: the plugin
    hard-codes `encoding=linear16` and has no parameter for anything else, so
    LiveKit expands mu-law to 16-bit PCM first. At the same sample rate that
    expansion is lossless and costs no resample, which is why `sample_rate`
    below is the setting that matters."""

    sample_rate: int = TELEPHONY_SAMPLE_RATE
    """Passed explicitly. Left to default the plugin says 16k, and LiveKit
    resamples the 8k line up to meet it -- a transcode that buys nothing."""

    endpointing_ms: int = 25
    """Vendor-side silence before Deepgram flushes a final. Kept low because the
    semantic turn detector, not the vendor, owns the endpointing decision."""

    patient_endpointing_ms: int = 0
    """Vendor-side silence for a question answered in pieces: a name spelled out,
    a number or an address. 0 leaves it alone.

    Deepgram's own guidance is 300-500 ms "for conversations where speakers pause
    mid-thought", against the 10 ms it suggests for short answers. Spelling a
    name is that case exactly: at 25 ms the gap between letters flushes a final,
    and a live call had "N A N A K" arrive as "N a n" then "k" and get recorded
    as Nank. The turns this applies to are the ones :func:`answer_kind` already
    calls patient, and they already wait 0.45s for the caller, so a later final
    costs nothing there.

    Off by default because changing it mid-call reconnects Deepgram's websocket
    (`update_options` sets the reconnect event immediately), and a caller who
    talks over the agent during that reconnect can lose their first word. Turn it
    on once a shadow run shows the longer wait actually hears the letters: see
    :mod:`stacy.bench.shadow_stt`."""

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
    depend on it: `stacy.agent.safety` intercepts gas and CO before inference.
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
    verbatim. That reason is gone: :mod:`stacy.agent.safety` intercepts gas and
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

    strict_tool_schema: bool = False
    """Whether tools are sent with `strict: true`. Off, deliberately, and measured.

    Strict schemas make the API compile a grammar for constrained decoding, and
    it compiles one per request. Same prompt, same tools, same model, interleaved
    over three rounds through the real plugin:

        strict (the plugin default)   median 3762ms to first token, one at 9185ms
        strict off                    median  809ms

    Nothing else the plugin sends -- max_tokens, tool caching, message shape --
    moved the number. Three and a half seconds per generation was the whole of
    the latency on live calls, and the outliers reached the ten-second request
    timeout, which is what turned turns into nineteen-second retries.

    What strict bought, arguments that always match the schema, is already
    covered: an optional field the caller never mentioned arrives null or absent,
    and `_said` turns either into the blank the desk treats as "not mentioned"."""

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

    auto_mode: bool = True
    """Synthesize sentence by sentence, flushing each, instead of on the schedule above.

    Measured on a 29-character reply followed by a tool call: first audio 630 ms
    after the text on the schedule, 300 ms in auto mode with the stall flush in
    :func:`stacy.agent.speech.push_with_flush`. Long replies were the same
    either way, about 170 ms.
    """


@dataclass(frozen=True)
class FillerConfig:
    """When a slow tool gets covered with speech, and in what words.

    Tools only, never the model: see :mod:`stacy.agent.filler` for the measured
    reason a timer on the model put "One sec" after the answer on nearly every
    turn.
    """

    enabled: bool = True

    tool_delay_s: float = 1.0
    """How long the line must be continuously quiet -- agent not speaking or
    thinking, caller not talking -- while a tool is still running before the
    agent speaks into it. A tool that finishes sooner produces no filler at all.
    About a second is where silence on a phone line starts to sound like a
    dropped call rather than someone looking something up."""

    max_per_turn: int = 1
    """More than one filler in a caller's turn reads as a stall, not a person."""

    tool_phrases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "check_availability": ("Let me pull up the schedule.", "Let me see what's open."),
            "lookup_customer": ("Let me look you up.", "Let me check our records."),
            "book_appointment": ("Okay, getting that booked.", "Putting that in now."),
            "take_message": ("Let me write that down.", "Getting that noted."),
        }
    )
    """What to say while each tool runs, rotated so it is never the same line
    twice running. A tool that is absent is never covered: `record_details`
    runs on nearly every caller turn, and `dispatch_emergency` is no moment for
    filler."""

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

    vad_min_silence_s: float = 0.30
    """How long a pause must be before the turn detector is asked at all.

    Down from Silero's 0.55, then from 0.40. On measured calls every reply
    waited this long before anything else began: at 0.40 the end-of-turn gap
    averaged 451 ms, the biggest fixed cost left on a fast-path reply. Pauses
    in the middle of a sentence are the semantic detector's job: it holds the
    turn open up to `max_endpointing_delay_s`. Much below 0.25 the final
    transcript (170-310 ms after speech ends) becomes the wait instead.
    """

    vad_patient_silence_s: float = 0.45
    """The wait after asking for a phone number, a street or a name.

    Callers read those out in groups -- "double nine... double eight..." -- and
    stopping them mid-number is the rudest thing a receptionist can do. Set per
    turn from what the agent just asked; see fastlane.answer_kind.
    """

    vad_eager_silence_s: float = 0.20
    """The wait after a yes-or-no where committing early is cheap: the text consent
    (not straight after digits read back) and "Anything else before I let you go?".
    Never the read-back: a yes there books, and "Yeah, but the street's wrong" has
    to be heard out."""

    use_semantic_endpointing: bool = True
    min_endpointing_delay_s: float = 0.2
    """Wait after an utterance the detector judges complete.

    Raised from 0.12: below roughly 0.2s the detector is being asked to commit
    on a transcript the vendor has not finished revising, and the failure mode
    is cutting a caller off mid-sentence -- which costs a full turn to repair
    and reads as far less human than the 80ms it saved.
    """

    max_endpointing_delay_s: float = 1.2
    """Ceiling when the detector judges the caller is mid-thought.

    Down from 2.0, which was down from 2.5. On the 2026-09-20 call the two
    slowest replies of twelve sat exactly on the two-second ceiling -- "Yes.
    This is this" and "It is", both judged mid-sentence -- and cost 2.0s each
    against 240-500 ms for every other turn. The ceiling only applies when the
    detector thinks more is coming, so the cost of lowering it is cutting off a
    caller who really was still talking; the patient 0.45s wait after a number,
    a name or an address is the separate guard against that.
    """

    preemptive_generation: bool = True
    """Start the LLM on interim transcript; cancel if the caller keeps talking.

    Costs some wasted tokens, removes STT finalization from the critical path.
    On a cheap fast model that trade is heavily one-sided.
    """

    preemptive_tts: bool = True
    """Voice the preemptive reply too, not just write it.

    With only the model running early, audio still waited for the turn to be
    confirmed and then for the voice's first byte. Voiced early, the reply is
    ready to play the moment the turn ends. If the caller keeps talking the
    audio is thrown away with the text: some wasted characters, and never an
    extra word reaching the caller.
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

    noise_commit_after_s: float = 1.2
    """Answer anyway once background noise is what holds the caller's turn open.

    A final transcript ends a turn only when voice detection says the caller has
    stopped, and Silero releases speech only below its deactivation threshold --
    0.45 at an activation of 0.6. Noise above that keeps "speaking" true, so a
    caller who had finished was never answered. Speech produces words and noise
    does not: this long after the last transcribed word, with a transcript still
    waiting, the turn is committed. It sits under the semantic detector's own
    worst-case wait, so a noisy line is never slower than a quiet one would be.

    Down from 2.0, then to 1.2 with `max_endpointing_delay_s`, which it tracks:
    on a live call a noisy line left the agent silent longer than a quiet line
    ever does. Not lower on its own, because a caller pausing to think on a
    noisy line would be cut off instead, which is no more human."""

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
