# stacy

Multi-tenant AI voice receptionist. Self-hosted cascade, built to match or beat
ElevenLabs Agents on latency while keeping the orchestration in our own code.

Ported from `Casey-Demo1` main branch, version `agtvrsn_4501m15kg5n8e22t9mckbgqg48md`.

**[docs/how-stacy-works.md](docs/how-stacy-works.md)** is the long form: the call
path end to end, where a second of latency goes, the correctness guards, the
instruments, and the bugs worth remembering. This file is the working reference —
what the pieces are, how to run it, and how to deploy it.

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
   LiveKit Agents worker  <- src/stacy/agent/entrypoint.py
        |
   +----+--------+-----------+------------+----------+
   |    |        |           |            |          |
  VAD  Turn     STT         LLM          TTS       Tools
Silero detector Deepgram   Claude       ElevenLabs  desk/
       (EOU)    Nova-3     Haiku 4.5    Flash v2.5
```

Audio stays at **8kHz** end to end. Every resample costs 20–50ms and some
quality, and one that sneaks in makes every measurement below a lie. The pin is
only real if each vendor is *told* the rate, so `registry.py` passes it
explicitly on both legs rather than letting it default.

It is `pcm_8000` on the TTS leg, not `ulaw_8000`, and the correction is worth
recording. The ElevenLabs *API* will return µ-law and this claimed µ-law for
that reason; the *plugin* cannot ask for it — its accepted encodings are mp3,
opus and pcm only, and the µ-law value raised `ValueError: Unsupported encoding`
on the first turn that tried to speak. `pcm_8000` matches the telephony rate
exactly, so nothing is resampled, which was the actual point. LiveKit companders
it to µ-law for the SIP leg, which is not a resample. The plugin's own default,
`mp3_22050_32`, would have been an MP3 decode *plus* a 22050→8000 resample on
every utterance.

The greeting is the exception to all of it: it is pre-rendered and played from
disk. Measured against the live API, synthesising it cost 334–572ms before the
caller heard a syllable, versus 0.4ms from cache. It is the one stall no filler
can cover, because it happens before the first word.

### Where the milliseconds are

Measured on real calls, caller-stops-talking to agent-starts-speaking. The
per-reply records come from the SDK's own timestamps (`metrics/live.py`), and the
model and transport numbers were taken from inside the us-east worker, because a
laptop in another country measures distance rather than speed.

| | Measured | What owns it |
|---|---|---|
| **Whole reply, median** | **0.95s** | everything below |
| A turn the desk answers | 0.37–0.55s | mostly the pause below |
| Waiting to be sure they finished | ~0.35s | `TurnConfig` — VAD floor plus the detector |
| Deepgram's final transcript | 0.13–0.45s | overlaps the pause, so usually free |
| Claude Haiku's first word | 0.50s | the model, and nothing else |
| Finishing the first sentence | ~0.20s | the model's token rate |
| ElevenLabs' first audio byte | 0.12s | vendor |
| Network to the model host | 0.048s | region pinning, already done |

Two of those rows are the whole story. A turn the desk answers costs about what
the deliberate pause costs, because its audio is built during that pause. A turn
the model answers costs a second, and roughly half of that is the model's first
word. Which is why the lever that paid was moving turns into code, not shopping
for a faster vendor — see "The desk" below.

Three numbers moved once the docs were read end to end:

- `optimize_streaming_latency` is a **documented no-op** on current ElevenLabs
  models. It was set to `3` here and doing nothing. The parameter that governs
  time-to-first-audio is the websocket chunk schedule, whose first entry
  defaults to **120 characters** — on a two-sentence answer that is most of the
  reply, buffered before a single frame is synthesized. It is now 50.
- Deepgram **Nova-3 rejects `keywords`** and uses keyterm prompting instead, so
  every boosted town and staff name was silently ignored.
- Prompt size is measurable and large: **~8,000 tokens in demo mode** (~25,000
  characters) and smaller for a real client. It reads as though it must be the
  problem, and it is not: interleaved against a warm cache, the whole prompt and
  all six tool schemas cost **25ms** of the first token. Trimming it saves money,
  not time. The cache is what makes that true, so the warm-up below earns its keep.
- The two per-call values sat ~200 tokens into that prompt, so a provider could
  cache almost none of it — and the timestamp varies by the minute, so even that
  sliver missed across calls. They now live in a `# This call` section at the
  very end: **the stable prefix went from ~1% to 99.2%**, and a test fails if
  anything per-call is added above it.

### The desk: the turns that never reach a model

Every caller turn goes first to `agent/fastlane.py`, which is ordinary Python. If
it recognises the situation exactly, it answers; if it has any doubt, the model
answers. On the calls we have recorded, that is about **40% of turns at ~0.4s**
instead of ~1.1s.

The turns it takes are the scripted ones: a phone number read out, a yes to the
text-consent script, a name, a street, a town, a window pick, a yes to the
read-back (which books), "go ahead and book it", and any question whose answer is
in the tenant file — price, hours, the service area, the FAQs. Everything else is
the model's: the problem description, an emergency judgment, anything unusual.

It is worth being clear about why this is not just a cache. The turns most
expensive to get wrong are the scripted ones — digits, addresses, whether an
appointment exists — and a deterministic reply cannot invent a price or a phone
number. The speed is a side effect of putting code where correctness mattered.

The cost is that the desk is where bugs hide, because it decides on regular
expressions what a person meant. It carries the most tests in the repo, every
rule names the call that caused it, and a replay harness runs every recorded call
through it before a change ships.

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
| `agent/` | Runtime, the desk, tools, turn-taking, speech handling, thin SIP glue |
| `bench/` | Instruments: model timings from inside the worker, shadow transcribers |
| `evals/` | The 8 criteria and 11 data fields, plus the judge that runs them |

Nothing outside `tenants/` knows a business fact. Nothing outside
`providers/registry.py` knows a vendor name. Those two rules are what make
client two cheap and vendor swaps a config change.

A third rule now joins them: **nothing outside `agent/nodes.py`,
`agent/entrypoint.py` and `bench/` imports the SDK at module scope.** Almost all
of the 637 tests run with no vendor package installed, which is why the whole
policy layer is testable at all.

### The instruments

Built because the arguments they settle cannot be settled by reading code.

| Tool | Answers |
|---|---|
| `metrics/live.py` | What did the caller actually wait, per reply? (the SDK's own timestamps, one JSON line each) |
| `bench/llm_latency.py` + `scripts/bench_models.py` | How fast is each candidate model **from the worker**? A job in a `bench-` room, because measuring from a laptop ranks providers by distance |
| `bench/shadow_stt.py` | Which recogniser settings hear this caller? Extra transcribers on the same live audio, logging only — never speaking, never blocking the call |
| `scripts/latency_report.py` | p50/p90 over a log, split desk against model |
| `scripts/preflight.py` | Would a call fail right now, and why? Cheaper than a call |

Two more live in the scratchpad rather than the repo, because they are for
answering a question rather than for shipping: a call simulator that runs whole
scripted calls through the real desk, prompt and tools against any model with a
model playing the caller, and a replay harness that pushes every recorded call
back through the current desk to catch a regression before it ships.

---

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
python scripts/preflight.py   # tells you what is still missing
cp .env.example .env   # then fill it in
```

Run the worker:

```bash
python -m stacy.agent.entrypoint dev
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

## Deploying

Three stages, each one usable on its own. Do not skip to the third: every
failure that can be found on a laptop costs a real caller's patience once a
trunk is pointed at it.

Before any of them:

```bash
python scripts/preflight.py
```

It checks the SDK and plugins, every credential, that each tenant loads and
renders and has a voice and a greeting, and that the call's write paths exist.
`FAIL` means a call will not work. `WARN` means it will work with something you
probably want switched off — an unconfigured webhook still answers the phone, it
just quietly breaks the promise the agent makes at the end of every call. It
exits non-zero on any `FAIL`, so it gates a deploy; `--strict` makes warnings
count too.

### 1. Your laptop, no phone number

```bash
python -m stacy.agent.entrypoint console
```

Talks to you through your mic. No LiveKit project, no trunk, no charge except
the STT/LLM/TTS keys. `console` and `dev` carry no SIP attributes, so there is
no dialed number to route on — set `STACY_DEFAULT_TENANT=millbrook` in `.env` or
this raises before the agent says a word. On a live trunk the dialed number
always wins and the fallback is logged at warning level, because answering as
the wrong business is worse than a stack trace.

This is where to hear whether the greeting lands, whether the pauses feel
natural, and whether a refusal reads as helpful. `scripts/dry_run.py` is faster
still and needs no keys, but it cannot tell you how the agent *sounds*.

### 2. A real phone number

The number is the routing key. `_route()` looks up the tenant by the dialed
number and refuses to guess, so the number in the tenant file, the number on the
LiveKit trunk and the number on the Twilio trunk have to be the same string.
Typing it three times is how they end up different, and the failure is silent:
the call connects, LiveKit makes a room, and the agent raises `KeyError` while a
customer listens to nothing.

So the tenant file is the source and the rest is generated:

```bash
python scripts/sip_config.py
```

It reads every tenant, writes `deploy/livekit/inbound-trunk.json` and
`deploy/livekit/dispatch-rule.json`, and prints the `lk` commands. `preflight.py`
then compares the two on every run and fails if they have drifted.

**LiveKit.** Create a Cloud project → `LIVEKIT_URL`, `LIVEKIT_API_KEY`,
`LIVEKIT_API_SECRET`. Note the project's SIP host, `<project-id>.sip.livekit.cloud`.
Then push the trunk and rule straight from the tenant files:

```bash
python scripts/sip_config.py --apply
```

It creates what is missing, updates a trunk whose numbers have drifted, and
warns if a dispatch rule names an agent the worker does not register as — the
failure where the call connects, the room is created, and nobody joins. Run it
again any time; it reports what already matched and changes nothing else.

To move a tenant to a different number, change it in one place:

```bash
python scripts/sip_config.py --set-number millbrook=+12138451295 --apply
```

Without `--apply` it writes `deploy/livekit/*.json` and prints the equivalent
`lk` commands instead.

The rule gives every call its own room. The shared-room alternative would have
two customers hearing each other.

**Twilio**, in the console — Elastic SIP Trunking, not a Programmable Voice
webhook:

1. **Elastic SIP Trunking → Trunks → new trunk.**
2. **Origination → Origination URI:** `sip:<project-id>.sip.livekit.cloud;transport=tcp`.
   This is the direction that matters: Twilio *originates* the call to LiveKit.
3. **Numbers → add your number** to the trunk. It has to be the same E.164
   string as the tenant's `phone_number`.
4. **Voice codec: leave PCMU (G.711 μ-law) at the top.** The pipeline is
   `ulaw_8000` end to end; letting Twilio negotiate anything else buys a
   transcode in both directions and no fidelity, on a call that is 8 kHz at the
   caller's microphone anyway.

**Dispatch.** With `STACY_AGENT_NAME` unset the worker joins every room in the
project — fewest moving parts, right for one agent. Set it and dispatch becomes
explicit: the worker joins only rooms whose rule names it. Both the worker and
`sip_config.py` read that one variable, and preflight fails if they disagree,
because a mismatch is a call that connects and then sits in silence.

Then run the worker against it:

```bash
python -m stacy.agent.entrypoint dev     # autoreload, one process
```

Call the number. The first call is the one that turns every latency figure in
this README from a budget into a measurement — the trace lands in
`data/traces/`, and `scripts/latency_report.py` reads it.

### 3. LiveKit Cloud Agents

LiveKit hosts the worker itself, so there is no container to run anywhere else.
It builds from the `Dockerfile` in this repo and reads `livekit.toml`.

```bash
lk agent create --secrets-file .env --ignore-empty-secrets --region us-east .
lk agent deploy --secrets-file .env --ignore-empty-secrets
lk agent status
lk agent logs
```

`--region` is required non-interactively; the choices are `us-east`,
`eu-central`, `ap-south`. Pick it by where the *callers and vendors* are, not
where you are: the worker's per-turn round trips go to Deepgram, Anthropic and
ElevenLabs, and its audio goes to a phone number. A worker near you and far from
both is the wrong optimisation. `--ignore-empty-secrets` matters because
`.env.example` ships keys deliberately blank, and without it the deploy fails
rather than skipping them.

Secrets are **not** in the image: `.dockerignore` excludes `.env` precisely so
an image cannot leak vendor keys to anyone who can pull it. They are injected as
environment variables on the deployment instead, which is what `--secrets-file`
does. To change them later:

```bash
lk agent update-secrets --secrets-file .env
```

`livekit.toml`'s `agent.id` is LiveKit's own agent identifier — `lk` rewrites it
to a `CA_...` value on first create, so do not hand-set it to the agent name.
The *registered* name comes from `STACY_AGENT_NAME`, and it has to match the SIP
dispatch rule's `agent_name`. A mismatch between those two is a call that
connects, creates a room, and is joined by nobody — silence to the caller,
success in every dashboard.

Two deploy-time steps worth running, in this order:

```bash
python scripts/preflight.py --live      # every key proven before anything ships
python scripts/prerender_greetings.py   # so the first caller is not the one who waits
```

The greeting cache lives under `STACY_DATA_DIR`. On hosted infrastructure that is
ephemeral, so the first call in a fresh instance re-renders it once and every
call after is warm — correct either way, just cheaper if you pre-render.

### 3b. Your own container


```bash
docker build -t stacy .
docker run --env-file .env -v stacy-data:/app/data stacy
```

The worker **dials out** to LiveKit and registers. It is not a server, listens
on no port, and needs no inbound firewall rule — anything that runs a container
with egress will host it. Model weights are baked in at build time so a cold
worker isn't fetching them while somebody listens to silence, and the container
healthcheck is `preflight.py`, so bad credentials fail the container instead of
the call.

Two things to get right:

- **The volume is not optional.** Bookings, transcripts and traces are on disk.
  Without a mount, a redeploy loses the calendar.
- **Run one worker per tenant calendar.** `CalendarStore`'s lock is per
  *process*, so a second worker sharing the volume can double-book a window.
  Scaling out is blocked on moving that store behind Postgres — the interface is
  already the seam for it. `start` mode drains gracefully, so redeploys are safe
  at one replica; concurrency is not.

Set `STACY_WEBHOOK_URL` and `STACY_WEBHOOK_SECRET` before taking traffic. The
agent closes every call by telling the caller someone will have their details.
Nothing on the receiving end makes that a lie the agent tells in a confident
voice. Payloads are signed HMAC-SHA256 with a timestamp, and retried on a
backoff ladder; calls are archived locally first, so a webhook outage costs you
delivery, not the record.

## Build phases and their gates

Do not pass a gate on impressions. Pass it on `scripts/latency_report.py`.

| # | Phase | Gate | State |
|---|---|---|---|
| 0 | Measurement harness | Per-stage table for any call | **done** |
| 1 | SIP echo bones | Round-trip measured, 8kHz path resample-free | next |
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

Seven tools, all backed by real stores rather than placeholders. The "refuses
when" column is the interesting one: every entry in it is a sentence the agent
said on a call before the guard existed.

| Tool | Backed by | Refuses when |
|---|---|---|
| `lookup_customer` | `desk/calendar.py` | — (pre-fills the slots on a hit) |
| `check_availability` | `desk/calendar.py` | closed day, elapsed window, full window |
| `record_details` | `agent/slots.py` | town outside the service area; a phone number the caller never said |
| `book_appointment` | `desk/calendar.py` + slots | anything still missing, unknown window, at capacity, the details were never read back, the caller's answer to the read-back was a no or a correction, or this exact visit is already booked |
| `dispatch_emergency` | `desk/calendar.py` + slots | name/number/address/town/problem missing, or a technician is already on the way to that address |
| `take_message` | `desk/messages.py` | — |
| `end_call` | `agent/hangup.py` | details half-collected and nothing booked, unless the caller said they were going; always, while the gas instruction is latched |

### Ending the call

A receptionist says "bye", waits a beat, and puts the phone down. For most of
this project Stacy said "bye" and then sat there, because the prompt said
outright that she had no way to hang up -- the line stayed open until the caller
gave up or the silence watchdog closed it thirty seconds later. On a mobile,
that half-minute of nothing is the part of the call people remember.

There are two ways a call ends now, and the split is deliberate.

**The normal one is decided in code, not by the model.** `agent/hangup.py`
reads the transcript: the caller said they were done, the agent's reply was a
sign-off, and the agent spoke last. All three, or nothing happens. The model is
never asked whether the call is over, because the one question you do not want
answered by a language model is "should I hang up on this person?"

Reading "the caller said they were done" is finer work than it looks. The
sign-off has to be the caller's whole last clause, not merely the end of it:

| They said | Over? | Why |
|---|---|---|
| "No, that's it. Thanks." | yes | the thanks is stripped, the clause is the sign-off |
| "Bye, but first — how much is it?" | no | the last clause is a question, and answering it is the job |
| "My name is Bye" | no | Deepgram hears names as all sorts of things on an 8 kHz line |

That last row is the whole reason the check is written this way. An earlier
version matched the end of the turn and would have hung up on somebody who was
introducing themselves.

**The other way is `end_call`, for a caller who was never going to be a
customer** -- a wrong number, a sales call, somebody abusive who has already
been offered help once. It refuses while a booking is half-collected, because
hanging up on someone mid-address is the worst version of this feature, and it
refuses outright while the gas-leak script is latched, where the only thing
that matters is the caller getting out of the house.

Both paths call `session.drain()` first, so the goodbye finishes playing before
the SIP leg drops -- cutting someone off mid-word is the same bug heard from
the other end. And if the caller speaks again *during* that drain, the hangup
is abandoned: "oh wait, one more thing" is a real thing people say.

### Callers don't answer one question at a time

"Yeah hi, this is Dan over at fourteen Elm in Wallingford and my furnace is
making a noise" is a name, an address, a town and a problem in one breath. The
prompt's "one item per turn" rule governs what the agent *asks*, never what it
can absorb — but read literally it marched through the list regardless, and
being asked for your own name after you have just given it is one of the
sharpest tells there is. The taste section already had the right instruction
("if they've already said either, don't ask again"); the booking flow did not.

`agent/slots.py` is the working memory. `record_details` takes however many
fields arrived, and its reply names only what is still outstanding — so the
model doesn't have to remember, it has to record. `book_appointment` then takes
**no arguments**: it books exactly what was recorded, which is what was read
back, rather than a re-statement the model may have drifted on.

Two more prompt rules became structural on the way:

- **Fuel is only asked for a furnace or a boiler.** The prompt says never to ask
  gas-or-electric about a mini-split or a heat pump; now it can't.
- **The town is checked against the service area when it arrives**, not
  discovered after six more fields have been collected.

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

The default is **Claude Haiku 4.5**. No fallback model is configured: the
cascade exists (`LLM_FALLBACK_*`) and is unset, because every open-weight model
measured against this prompt was either slower or not careful enough. See
`docs/how-stacy-works.md` for the bake-off numbers.

Two decisions get confused here. The *host* sets time to first token — the same
open weights on Groq or Cerebras versus a general cloud is roughly 150ms against
500ms, and first-token time is the only LLM number a caller can hear. The
*model* sets whether the answer is right. Picking the fastest host and taking
whatever open weights it serves optimises the number you can hear at the expense
of the one you can't.

On this workload that trade is the wrong way round. A wrong answer is not a
slower call; it is a technician sent to the wrong address, a price quoted the
business won't honour, or an emergency booked as a Tuesday appointment. The half
second is bought back elsewhere — preemptive generation starts the turn on the
interim transcript, the prompt prefix is cached, a filler covers the gap before
a tool call — and none of those recover a mistake.

The fallback is where a fast open-weight model earns its place. Once the primary
has burned `LLM_TIMEOUT_S`, the comparison is no longer "smart versus fast", it
is "fast versus dead air". Primary and fallback are different vendors on
different accounts, so each names its own key variable (`LLM_API_KEY`,
`LLM_FALLBACK_API_KEY`); they used to share one, which meant the fallback was
handed the primary's credential and would have failed authentication at the one
moment it was needed.

Two tables decide the rest, and neither alone is enough. `scripts/latency_report.py`
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

- **A real SMS carrier.** `LoggingProvider` writes the message to the log and
  says so. An unconfigured provider that silently no-ops is indistinguishable
  from a working one until a customer says they never got the text.
- **Speech recognition on an accented line.** This is the largest remaining
  source of wrong data, not latency. Three calls in a row had a spelled name
  heard wrong — Ananak, Nanat, Nank for Nanak — and one booked "1230 Street" for
  "123 Oak Street". The name is now spelled back to the caller so a dropped
  letter surfaces on the call, and `bench/shadow_stt.py` exists to settle which
  recogniser settings actually help. Until a real call runs with it, the fix is
  reasoned rather than proven.
- **Retention and redaction.** Traces, transcripts and the calendar are all
  plaintext on disk. Decide the policy before real traffic, not after.
- **Multi-worker booking.** `CalendarStore`'s lock is per process. Two workers
  can double-book the same window. Postgres behind the same interface fixes it.
- **More than one live tenant.** A business is a JSON record and the code has no
  per-tenant branches, but only Millbrook has taken real calls. The claim is
  structural, not proven at scale.
- **A named successor to Haiku 4.5.** Anthropic's stated retirement is "not
  sooner than 15 October 2026". `scripts/bench_models.py` is kept ready so the
  replacement is chosen on measurements rather than in a hurry.

## Verify before the first live call

`registry.supported()` inspects each plugin constructor and drops arguments it
doesn't accept, logging what it dropped, so a version skew shows up in the first
line of the worker log rather than as a `TypeError` while a caller is on the
line. The same guard wraps `AgentSession` and `WorkerOptions`.

**The pipeline has now been constructed against the real SDK** (livekit-agents
1.8.0) and builds with nothing dropped. Doing that for the first time found four
things the guard was hiding or could not see:

- `openai.LLM` has no `frequency_penalty` parameter. It was being dropped on
  every call, so the repetition control the config documents as the reason
  temperature can stay at 0 had never once been applied. It goes through
  `extra_body` now, which is the plugin's pass-through to the host.
- The ElevenLabs plugin's own key lookup reads `ELEVEN_API_KEY`, a name used
  nowhere in this repo. Setting the documented `ELEVENLABS_API_KEY` and getting
  "API key is required" is a baffling five minutes. Every vendor key is passed
  explicitly now rather than left to each plugin's env conventions.
- `anthropic` 1.x moved to `httpx2` while `livekit-plugins-anthropic` 1.8 still
  hands it an `httpx.AsyncClient`. An unpinned install raises `TypeError` the
  first time the LLM is constructed — during a call. Pinned to `<1.0`.
- `zoneinfo` has no bundled database, so a clean venv or slim container has no
  `America/New_York`. Every appointment time and hours check goes through it.
  `tzdata` is an explicit dependency now.

The first three were invisible precisely because `supported()` degrades quietly;
that is the right trade for a live call and the wrong one for a first install,
which is why this had to be run once. `agent/nodes.py`'s node method names and
yielded types remain the highest-risk surface, because they need a call rather
than a constructor to exercise.

`scripts/preflight.py` covers the rest of the checklist -- credentials, tenant
loading, prompt rendering, write paths -- and exits non-zero if a call would
fail. Run it first; it is cheaper than a call.

## Tests

```bash
python -m pytest -q
```

637 tests, almost none of which need a vendor package. They are not style checks — each
assertion corresponds to a prompt line that exists because something went wrong
on a real call, or to a promise the agent makes to a caller.

Two found real bugs while being written: availability offered a two-to-five
window on a Saturday that closes at one, and message ordering broke on Windows'
15ms clock resolution.
