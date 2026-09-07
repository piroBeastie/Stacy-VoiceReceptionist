# dana

Multi-tenant AI voice receptionist. Self-hosted cascade, built to match or beat
ElevenLabs Agents on latency while keeping the orchestration in our own code.

Ported from `Casey-Demo1` main branch, version `agtvrsn_4501m15kg5n8e22t9mckbgqg48md`.

---

## Why this exists

The ElevenLabs demo sounds excellent. What it doesn't give us is control of the
critical path or the ability to onboard a second client without cloning an
agent and rewriting its prompt. This repo owns three things the platform can't:

- **Latency** — semantic endpointing and preemptive generation, tuned by us
- **Tenancy** — a business is a JSON record, not a cloned agent
- **Delivery** — the tools and webhooks that make the agent's promises true

ElevenLabs doesn't disappear; it drops to being one swappable TTS vendor
(`providers/base.py: TTSConfig.vendor`), in the same class as Twilio.

---

## Architecture

```
Twilio Elastic SIP Trunk
        |  (SIP/RTP, mu-law 8k, no websocket hop)
   LiveKit SIP
        |
   LiveKit Agents worker  <- src/dana/agent/entrypoint.py
        |
   +----+--------+-----------+------------+
   |    |        |           |            |
  VAD  Turn     STT         LLM          TTS
Silero detector Deepgram   Qwen3/Groq   Flash v2.5
       (EOU)    Nova-3                  or Cartesia
```

Audio stays `ulaw_8000` end to end. Every resample costs 20–50ms and some
quality, and a transcode that sneaks in makes every measurement below a lie.

### Where the milliseconds are

| Stage | Typical | Ours | Owner |
|---|---|---|---|
| Network in | 40–120ms | 40ms | region pinning |
| **Endpointing** | 300–700ms | **~120ms** | `TurnConfig` — biggest lever |
| STT final | 100–300ms | ~80ms | vendor |
| **LLM TTFT** | 300–600ms | **~150ms** | model host — second lever |
| TTS TTFB | 80–300ms | ~40ms | vendor |
| Egress | 50–100ms | 50ms | codec path |

Vendor swaps buy ~150ms total. The win is in the two bolded rows, and both are
architecture rather than procurement.

---

## Layout

| Path | What it owns |
|---|---|
| `metrics/` | Per-turn stage timings, JSONL sink, p50/p95 report |
| `tenants/` | The tenant record. Everything business-specific lives here |
| `prompt/` | v4 prompt as a Jinja template + renderer |
| `providers/` | Vendor choice as data; `registry.py` is the only vendor-aware module |
| `agent/` | SIP glue (thin) and backchannel policy (tested) |
| `evals/` | The 8 criteria and 11 data fields, ported verbatim |

Nothing outside `tenants/` knows a business fact. Nothing outside
`providers/registry.py` knows a vendor name. Those two rules are what make
client two cheap and vendor swaps a config change.

---

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill it in
```

Run the worker:

```bash
python -m dana.agent.entrypoint dev
```

Latency table:

```bash
python scripts/latency_report.py --tenant millbrook
```

Prompt parity against the live ElevenLabs agent, during the migration window:

```bash
python scripts/diff_against_elevenlabs.py
```

---

## Build phases and their gates

Do not pass a gate on impressions. Pass it on `scripts/latency_report.py`.

| # | Phase | Gate | State |
|---|---|---|---|
| 0 | Measurement harness | Per-stage table for any call | **done** |
| 1 | SIP echo bones | Round-trip measured, µ-law path transcode-free | next |
| 2 | Full cascade, naive settings | Baseline p50 recorded (expect 1.2–1.8s) | |
| 3 | Prompt port | **Safety-override test green** before anything else | ported, untested |
| 4 | Latency work | p50 ≤ 700ms, p95 ≤ 1.1s over ≥50 real calls | |
| 5 | Behavior parity | Backchannel, barge-in, vendor failover, reconnect | partial |
| 6 | Eval port | All 8 criteria running as an LLM-judge harness | data ported |
| 7 | Tools and delivery | Post-call webhook first — it makes the closing line true | |

### Phase 3 is a hard stop

The prompt tells a caller who smells gas to leave the house and call 911. That
script must come out verbatim, every time, and the agent must not return to
normal flow afterward. `tests/test_port_fidelity.py::TestSafetyOverride` guards
the text; a live call has to confirm the behavior before anything else ships.

If `LLMConfig.temperature` is ever raised above `0.0` — the nanak branch's `0.4`
is tempting for naturalness — that test runs first, every time. Temperature
loosens grip on scripted lines, and this is the line that cannot slip.

---

## What's deliberately not here yet

- **Live SIP config.** Phase 1. The Twilio trunk and LiveKit SIP dispatch rule
  are infrastructure, not code.
- **Knowledge base.** Millbrook's business facts (`5T5UOWhdyGfbWSHSasMd`) are
  still only in ElevenLabs. Per the plan these become tenant fields rather than
  a per-agent KB doc — RAG attaches per-agent, which multi-tenancy can't use.
- **Tools.** `lookup_customer`, `check_availability`, `book_appointment`,
  `send_sms_confirmation`. Each needs a live endpoint first: a tool pointed at a
  placeholder is worse than no tool, because the agent will call it mid-call and
  fall apart in front of a prospect.
- **SMS consent enforcement.** The prompt asks for consent. The *endpoint* has
  to reject the send when `consent_captured` is false. A prompt is a suggestion;
  an endpoint is a control.
- **Retention and redaction.** Current ElevenLabs settings are demo-grade
  (`retention_days: -1`, no PII redaction). Real traffic needs a policy.

## Verify before the first live call

`providers/registry.py` and `agent/entrypoint.py` are the only version-sensitive
files. Check the livekit-agents plugin constructors and session event names
against the installed SDK once, then trust them. Everything else in the repo is
vendor-independent and covered by tests.

## Tests

```bash
python -m pytest -q
```

62 tests. They are not style checks — each assertion corresponds to a prompt
line that exists because something went wrong on a real call, or to a promise
the agent makes to a caller.
