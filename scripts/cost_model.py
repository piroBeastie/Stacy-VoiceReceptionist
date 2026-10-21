"""What a minute of Stacy costs, and what it would cost built differently.

Run it to regenerate the tables in docs/cost-per-minute.md:

    python scripts/cost_model.py
    python scripts/cost_model.py --inr 94.0 --minutes 3.5

Every price here is a dated quote from a vendor page or a third-party
tracker, not a contract. Change RATES when you get a real one -- the point
of this file is that the assumptions are in one place and visible, rather
than baked into a slide somebody wrote once.

The usage model is measured where it can be: the prompt is rendered from
the real tenant and counted, and the share of turns the desk answers comes
from the live-call logs. The rest is stated and arguable.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

INR_PER_USD = 96.5  # mid-market, Oct 2026; override with --inr


# --------------------------------------------------------------------------
# What one minute of a call actually consumes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    """Consumption per minute of connected call."""

    agent_turns_per_min: float = 4.7
    """A turn every ~13s. A booking call is name, phone, consent, address,
    town, problem, system, fuel, window, read-back, goodbye."""

    desk_share: float = 0.40
    """Share of turns answered in code with no model call. Measured."""

    prompt_tokens: int = 7_486
    """Measured: render_system_prompt(millbrook) counted with cl100k."""

    cached_share: float = 0.992
    """Measured: the stable prefix after the per-call section was moved to
    the end. A test fails if anything per-call is added above it."""

    history_tokens: int = 600
    """Conversation replayed on an average mid-call request."""

    output_tokens: int = 40
    """A receptionist line is short. This is the whole reply."""

    tts_chars_per_min: float = 400.0
    """The agent speaks roughly 40% of the call at ~150 wpm."""

    stt_minutes_per_min: float = 1.0
    """The caller's audio streams for the whole call."""

    @property
    def llm_calls_per_min(self) -> float:
        return self.agent_turns_per_min * (1.0 - self.desk_share)


# --------------------------------------------------------------------------
# Prices. USD. Each carries where it came from and how much to trust it.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Price:
    usd: float
    unit: str
    source: str
    confidence: str  # "vendor" | "third-party" | "estimate"


RATES: dict[str, Price] = {
    # --- telephony -------------------------------------------------------
    "twilio_sip_in": Price(0.0034, "per min", "Telnyx comparison of Twilio list", "third-party"),
    "twilio_number": Price(1.15 / 43_800, "per min (number $1.15/mo over 730h)", "Twilio", "third-party"),
    # --- transport / orchestration ---------------------------------------
    "livekit_agent": Price(0.01, "per min", "third-party trackers", "third-party"),
    "livekit_sip": Price(0.004, "per min", "third-party trackers", "third-party"),
    # --- speech to text ---------------------------------------------------
    "deepgram_nova3": Price(0.0077, "per min", "Deepgram list, streaming mono", "third-party"),
    "deepgram_nova3_promo": Price(0.0048, "per min", "promotional rate", "third-party"),
    "deepgram_keyterm": Price(0.0013, "per min", "keyterm prompting add-on", "third-party"),
    # --- text to speech ---------------------------------------------------
    "eleven_flash": Price(50.0 / 1_000_000, "per char ($50/M)", "third-party, Aug 2026", "third-party"),
    "deepgram_aura2": Price(30.0 / 1_000_000, "per char ($30/M)", "third-party", "third-party"),
    # --- models (per token) ------------------------------------------------
    # Anthropic list. Cache write is 1.25x input, cache read 0.1x input.
    "haiku45_in": Price(1.00 / 1_000_000, "per input token", "Anthropic", "vendor"),
    "haiku45_out": Price(5.00 / 1_000_000, "per output token", "Anthropic", "vendor"),
    "haiku55_in": Price(0.10 / 1_000_000, "per input token", "Anthropic", "vendor"),
    "haiku55_out": Price(0.50 / 1_000_000, "per output token", "Anthropic", "vendor"),
}

CACHE_READ = 0.10  # x input rate
CACHE_WRITE = 1.25  # x input rate


@dataclass
class Stack:
    """One way of building the thing, and what it costs per minute."""

    name: str
    note: str
    llm_in: float
    llm_out: float
    stt_per_min: float
    tts_per_char: float
    transport_per_min: float
    telephony_per_min: float
    desk_share: float | None = None  # override
    lines: dict[str, float] = field(default_factory=dict)

    def cost_per_min(self, u: Usage) -> float:
        share = self.desk_share if self.desk_share is not None else u.desk_share
        calls = u.agent_turns_per_min * (1.0 - share)

        cached = u.prompt_tokens * u.cached_share
        fresh = u.prompt_tokens * (1.0 - u.cached_share) + u.history_tokens

        llm = calls * (
            cached * self.llm_in * CACHE_READ
            + fresh * self.llm_in
            + u.output_tokens * self.llm_out
        )
        stt = u.stt_minutes_per_min * self.stt_per_min
        tts = u.tts_chars_per_min * self.tts_per_char
        self.lines = {
            "LLM": llm,
            "STT": stt,
            "TTS": tts,
            "Transport": self.transport_per_min,
            "Telephony": self.telephony_per_min,
        }
        return sum(self.lines.values())


def build(u: Usage) -> list[Stack]:
    r = RATES
    twilio = r["twilio_sip_in"].usd + r["twilio_number"].usd
    livekit = r["livekit_agent"].usd + r["livekit_sip"].usd
    nova3 = r["deepgram_nova3"].usd + r["deepgram_keyterm"].usd

    return [
        Stack(
            "A. Today",
            "Twilio + LiveKit Cloud + Nova-3 + Haiku 4.5 + ElevenLabs Flash v2.5",
            r["haiku45_in"].usd, r["haiku45_out"].usd, nova3,
            r["eleven_flash"].usd, livekit, twilio,
        ),
        Stack(
            "B. One-line change",
            "A, but Claude Haiku 5.5 instead of 4.5 -- a newer model, 10x cheaper",
            r["haiku55_in"].usd, r["haiku55_out"].usd, nova3,
            r["eleven_flash"].usd, livekit, twilio,
        ),
        Stack(
            "C. Swap the voice too",
            "B, but Deepgram Aura-2 for TTS instead of ElevenLabs",
            r["haiku55_in"].usd, r["haiku55_out"].usd, nova3,
            r["deepgram_aura2"].usd, livekit, twilio,
        ),
        Stack(
            "D. Lean",
            "C, on Deepgram's promotional STT rate, desk answering 60% of turns",
            r["haiku55_in"].usd, r["haiku55_out"].usd,
            r["deepgram_nova3_promo"].usd + r["deepgram_keyterm"].usd,
            r["deepgram_aura2"].usd, livekit, twilio,
            desk_share=0.60,
        ),
        Stack(
            "E. Self-hosted transport",
            "D, with the worker and SIP on our own box instead of LiveKit Cloud",
            r["haiku55_in"].usd, r["haiku55_out"].usd,
            r["deepgram_nova3_promo"].usd + r["deepgram_keyterm"].usd,
            r["deepgram_aura2"].usd, 0.002, twilio,
            desk_share=0.60,
        ),
    ]


def counterfactuals(u: Usage) -> list[tuple[str, float]]:
    """Things already done, priced as if they had not been.

    Worth printing next to the options, because the two largest savings in
    this system were taken before anyone drew up a cost table -- and a
    table that only shows what is left to do makes them invisible.
    """
    r = RATES
    calls = u.agent_turns_per_min * (1.0 - u.desk_share)
    all_turns = u.agent_turns_per_min

    def llm_cost(n_calls: float, cached: bool, rate_in: float, rate_out: float) -> float:
        if cached:
            tokens_in = (
                u.prompt_tokens * u.cached_share * CACHE_READ
                + u.prompt_tokens * (1 - u.cached_share)
                + u.history_tokens
            )
        else:
            tokens_in = u.prompt_tokens + u.history_tokens
        return n_calls * (tokens_in * rate_in + u.output_tokens * rate_out)

    i45, o45 = r["haiku45_in"].usd, r["haiku45_out"].usd
    return [
        ("prompt caching (99.2% prefix)",
         llm_cost(calls, False, i45, o45) - llm_cost(calls, True, i45, o45)),
        ("the desk answering 40% of turns",
         llm_cost(all_turns, True, i45, o45) - llm_cost(calls, True, i45, o45)),
    ]


def money(usd: float, inr_rate: float) -> str:
    return f"${usd:,.4f} / Rs {usd * inr_rate:,.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inr", type=float, default=INR_PER_USD, help="INR per USD")
    ap.add_argument("--minutes", type=float, default=3.0, help="average call length")
    ap.add_argument("--calls", type=int, default=1000, help="calls per month for the volume table")
    args = ap.parse_args()

    u = Usage()
    stacks = build(u)

    print(f"\nAssumptions: {u.agent_turns_per_min} agent turns/min, "
          f"{u.desk_share:.0%} answered by the desk, prompt {u.prompt_tokens:,} tokens "
          f"({u.cached_share:.1%} cached), {u.output_tokens} output tokens/reply, "
          f"{u.tts_chars_per_min:.0f} TTS chars/min.")
    print(f"Call length {args.minutes} min. Rs {args.inr}/USD.\n")

    print(f"{'Stack':<24} {'per minute':<26} {'per call':<26} {'per ' + str(args.calls) + ' calls':<26} {'vs A'}")
    print("-" * 120)
    base = None
    for s in stacks:
        per_min = s.cost_per_min(u)
        if base is None:
            base = per_min
        per_call = per_min * args.minutes
        per_vol = per_call * args.calls
        delta = f"{(per_min / base - 1) * 100:+.0f}%" if base else ""
        print(f"{s.name:<24} {money(per_min, args.inr):<26} {money(per_call, args.inr):<26} "
              f"{money(per_vol, args.inr):<26} {delta}")

    print("\nWhere the money goes, per minute:\n")
    parts = ["LLM", "STT", "TTS", "Transport", "Telephony"]
    print(f"{'Stack':<24}" + "".join(f"{p:>14}" for p in parts) + f"{'total':>14}")
    print("-" * 106)
    for s in stacks:
        total = s.cost_per_min(u)
        print(f"{s.name:<24}" + "".join(f"{s.lines[p]:>14.5f}" for p in parts) + f"{total:>14.5f}")

    print("\nAlready saved, per minute, by work that is done:\n")
    for label, saved in counterfactuals(u):
        print(f"  {label:<34} {money(saved, args.inr)}")

    print("\nPrices used (USD):\n")
    for key, p in RATES.items():
        print(f"  {key:<22} {p.usd:<14.8f} {p.unit:<42} {p.confidence}")
    print()


if __name__ == "__main__":
    main()
