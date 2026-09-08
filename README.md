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
   +----+--------+-----------+------------+----------+
   |    |        |           |            |          |
  VAD  Turn     STT         LLM          TTS       Tools
Silero detector Deepgram   Qwen3/Groq   Flash v2.5  desk/
       (EOU)    Nova-3                  or Cartesia
```

Audio stays `ulaw_8000` end to end. Every resample costs 20–50ms and some
quality, and a transcode that sneaks in makes every measurement below a lie.
That pin is only real if each vendor is *told* the format — the plugins all
default to 16k PCM, so `registry.py` passes it explicitly on both legs.

### Where the milliseconds are

| Stage | Typical | Target | Owner |
|---|---|---|---|
| Network in | 40–120ms | 40ms | region pinning |
| **Endpointing** | 300–700ms | **~200ms** | `TurnConfig` — biggest lever |
| STT final | 100–300ms | ~80ms | vendor |
| **LLM TTFT** | 300–600ms | **~150ms** | model host + prompt size |
| TTS TTFB | 80–300ms | ~40ms | vendor + chunk schedule |
| Egress | 50–100ms | 50ms | codec path |

Vendor swaps buy ~150ms total. The win is in the two bolded rows, and both are
architecture rather than procurement.

Three of those numbers moved once the docs were read end to end:

- `optimize_streaming_latency` is a **documented no-op** on current ElevenLabs
  models. It was set to `3` here and doing nothing. The parameter that governs
  time-to-first-audio is the websocket chunk schedule, whose first entry
  defaults to **120 characters** — on a two-sentence answer that is most of the
  reply, buffered before a single frame is synthesized. It is now 50.
- Deepgram **Nova-3 rejects `keywords`** and uses keyterm prompting instead, so
  every boosted town and staff name was silently ignored.
- Prompt size is measurable and large: **~5,460 tokens in demo mode, ~3,360 for
  a real client.** TTFT scales with it, so the demo-only blocks are not free.
- The two per-call values sat ~200 tokens into that prompt, so a provider could
  cache almost none of it — and the timestamp varies by the minute, so even that
  sliver missed across calls. They now live in a `# This call` section at the
  very end: **the stable prefix went from ~1% to 99.2%**, and a test fails if
  anything per-call is added above it.

### What the harness can and cannot see

`egress_ms` is measured inside the worker: vendor's first frame to handing that
frame downstream. It does **not** include the LiveKit → SIP → PSTN leg, which is
not observable from in-process. The 50ms in the table above is an estimate and
should be treated as one until it is measured from the far end.

---

## Layout

| Path | What it owns |
|---|---|
| `metrics/` | Per-turn stage timings, JSONL sink, p50/p95 report |
| `tenants/` | The tenant record, including the knowledge base |
| `prompt/` | v4 prompt as a Jinja template + renderer |
| `providers/` | Vendor choice as data; `registry.py` is the only vendor-aware module |
| `desk/` | Calendar and message pad — capacity, hours, and refusals |
| `delivery/` | Post-call webhook and consent-gated SMS |
| `agent/` | Runtime, filler policy, tools, backchannel, and thin SIP glue |
| `evals/` | The 8 criteria and 11 data fields, plus the judge that runs them |

Nothing outside `tenants/` knows a business fact. Nothing outside
`providers/registry.py` knows a vendor name. Those two rules are what make
client two cheap and vendor swaps a config change.

A third rule now joins them: **nothing outside `agent/nodes.py` and
`agent/entrypoint.py` imports the SDK.** All 219 tests run with no vendor
package installed, which is why the whole policy layer is testable at all.

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

Walk a scripted call through the real calendar, with no SIP trunk and no model
— the fastest way to hear how a refusal reads before a caller does:

```bash
python scripts/dry_run.py
```

Grade archived calls against the 8 criteria — the other half of a model decision:

```bash
EVAL_API_KEY=... python scripts/eval_calls.py --save baselines/qwen.json
EVAL_API_KEY=... python scripts/eval_calls.py --compare baselines/qwen.json
```

---

## Build phases and their gates

Do not pass a gate on impressions. Pass it on `scripts/latency_report.py`.

| # | Phase | Gate | State |
|---|---|---|---|
| 0 | Measurement harness | Per-stage table for any call | **done** |
| 1 | SIP echo bones | Round-trip measured, µ-law path transcode-free | next |
| 2 | Full cascade, naive settings | Baseline p50 recorded | |
| 3 | Prompt port | **Safety-override test green** before anything else | ported + KB, untested live |
| 4 | Latency work | p50 ≤ 700ms, p95 ≤ 1.1s over ≥50 real calls | |
| 5 | Behavior parity | Backchannel, barge-in, vendor failover, reconnect | wired, untested live |
| 6 | Eval port | All 8 criteria running as an LLM-judge harness | **done** |
| 7 | Tools and delivery | Post-call webhook first — it makes the closing line true | code done, needs endpoints |

### Phase 0 was wrong and is now right

The harness shipped marked "done" and could not measure the two stages this
project exists to improve. Spans were derived from `Stage` declaration order
rather than mark timestamps, and the trace was emitted the instant the agent
started speaking — before `metrics_collected` fires. The result:

- `llm_ttft_ms` and `tts_ttfb_ms` were **never recorded at all**
- `stt_ms` was **negative on every turn**
- `egress_ms` silently contained most of the response

Timings are now taken inside the pipeline nodes (`agent/nodes.py`), where first
token and first frame are each observable exactly once, and a turn is closed by
the *next* turn rather than by the agent starting to speak.

### Phase 3 is a hard stop, and the model is no longer trusted with it

The prompt tells a caller who smells gas to leave the house and call 911. Every
other rule in the prompt is an instruction the model is asked to follow. This one
is different in kind: an instruction followed 99.5% of the time is an excellent
instruction and an unacceptable safety control.

So it doesn't reach the model. `agent/safety.py` matches gas-smell and
carbon-monoxide reports, speaks the script verbatim, and **latches** — every
later turn repeats the same words. Because the model is never invoked, no tool
can fire: there is no path from a triggered override to a booking, structurally
rather than by instruction. The most safety-critical path in the product is now
also the fastest, since it costs no inference at all.

Detection is deliberately narrow on the word "gas" alone. The booking flow asks
"gas or electric?" about furnaces on ordinary calls, so a bare mention must not
trigger; 25 cases are pinned in both directions in `tests/test_safety.py`.

The matching eval criterion is graded by string equality rather than by a judge,
because "did it say the script" is decidable and a model's opinion of it is not.
A plausible-sounding paraphrase — *"you should get outside and call 911"* — is
the failure a model judge waves through and equality catches.

`LLMConfig.temperature` stays at `0.0`. Temperature loosens grip on scripted
lines, and this is the line that cannot slip. The prompt's "never say the same
sentence twice" rule is handled with `frequency_penalty` instead, which
penalises reused tokens without flattening the distribution, and
`RepetitionTracker` counts repeats so that penalty can be tuned against a
number rather than a hunch.

---

## What the agent can actually do

Five tools, all backed by real stores rather than placeholders:

| Tool | Backed by | Refuses when |
|---|---|---|
| `lookup_customer` | `desk/calendar.py` | — |
| `check_availability` | `desk/calendar.py` | closed day, elapsed window, full window |
| `book_appointment` | `desk/calendar.py` | no street address, unknown window, at capacity |
| `dispatch_emergency` | `desk/calendar.py` | — (skips capacity by design) |
| `take_message` | `desk/messages.py` | — |

Every refusal returns a sentence the agent can say next, not an error. "That one
just filled. I've still got eleven to two, or two to five" keeps the call
moving; a bare failure makes the agent apologise and stall, which is the moment
a caller decides they are talking to a machine.

Two controls are deliberately **not** in the prompt, because a prompt is a
suggestion to a model and these are not suggestions:

- **SMS consent.** `delivery/sms.py` raises `ConsentMissing` for any booking
  whose `sms_consent` is false. There is no override parameter.
- **Street address.** `desk/calendar.py` refuses the booking outright, so a bad
  turn cannot produce a visit nobody can drive to.

---

## Choosing a model

Two tables decide it, and neither alone is enough. `scripts/latency_report.py`
says how fast a stack is; `scripts/eval_calls.py` says whether it still follows
the rules. Swapping the LLM on one of them is how a faster model quietly stops
honouring the two-sentence ceiling and nobody finds out for three weeks.

```bash
EVAL_API_KEY=... python scripts/eval_calls.py --save baselines/qwen.json
# swap LLM_MODEL, take another 50 calls
EVAL_API_KEY=... python scripts/eval_calls.py --compare baselines/qwen.json --gate
```

Pass rate is computed over the calls a criterion *applies to*. Most criteria are
`unknown` on most calls — nobody asked for the taste, nobody mentioned gas — and
folding those into the denominator would make every number drift with whatever
mix of calls happened to be in the corpus.

Point `EVAL_MODEL` at something other than `LLM_MODEL`. Grading a model's output
with the same model is self-marking, and the judge doesn't need to be fast.

---

## What's deliberately not here yet

- **Live SIP config.** Phase 1. The Twilio trunk and LiveKit SIP dispatch rule
  are infrastructure, not code.
- **A real SMS carrier.** `LoggingProvider` writes the message to the log and
  says so. An unconfigured provider that silently no-ops is indistinguishable
  from a working one until a customer says they never got the text.
- **Cached greeting audio.** The greeting is static per tenant and is currently
  synthesized on every call, which puts TTS connection setup at the most-judged
  moment of the call. Caching it needs the `say(audio=...)` surface confirmed
  against the installed SDK first — writing it blind would be a guess.
- **Retention and redaction.** Traces, transcripts and the calendar are all
  plaintext on disk. Decide the policy before real traffic, not after.
- **Multi-worker booking.** `CalendarStore`'s lock is per process. Two workers
  can double-book the same window. Postgres behind the same interface fixes it.

## Verify before the first live call

`agent/entrypoint.py` and `agent/nodes.py` are the only version-sensitive files
left. `providers/registry.py` used to be a third, and no longer is:
`registry.supported()` inspects each plugin constructor and drops arguments it
doesn't accept, logging what it dropped. A version skew now shows up in the
first line of the worker log rather than as a `TypeError` while a caller is on
the line. The same guard wraps `AgentSession`.

That is a mitigation, not a substitute. **No code in this repo has ever executed
against the SDK** — there is no venv and `livekit` is not installed. The node
method names and their yielded types in `agent/nodes.py` are the highest-risk
surface remaining.

## Tests

```bash
python -m pytest -q
```

219 tests, none of which need a vendor package. They are not style checks — each
assertion corresponds to a prompt line that exists because something went wrong
on a real call, or to a promise the agent makes to a caller.

Two found real bugs while being written: availability offered a two-to-five
window on a Saturday that closes at one, and message ordering broke on Windows'
15ms clock resolution.
