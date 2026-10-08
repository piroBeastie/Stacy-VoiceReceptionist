# How stacy works — a study guide

For a flat index of every capability and fix — symptom, file and test on one line — see [features-and-fixes.md](features-and-fixes.md). This document is the walkthrough.

Everything in this document is measured or read out of the code in this repo, as
of 2026-09-20. Numbers without a source are ones we measured ourselves on real
calls; where a number is someone else's claim, it says so.

Read it in order. §1–§4 is the system, §5 is latency, §6 is correctness, §7 is
the tooling, §8 is the war stories (the most useful part in an interview), §9 is
the numbers to quote, §10 is question drill.

---

## 1. What it is, in one paragraph

stacy is a multi-tenant AI phone receptionist for trade businesses. A customer
rings a normal phone number, a voice answers, and by the end of the call a real
appointment exists in a calendar, with the caller's name, number, address and
problem attached. A business is a JSON record, not a cloned agent: the same code
serves any tenant, and the tenant's own file supplies its prompt, its voice, its
service area, its booking windows, its prices and its FAQs.

The product it competes with is ElevenLabs Agents. We give up their turnkey
platform to own three things: the latency path, tenancy, and delivery (the tools
and webhooks that make the agent's promises true).

---

## 2. The call path

```
caller's phone
   │  PSTN
Twilio Elastic SIP Trunk           TK839d2183… ("stacy-livekit")
   │  SIP/RTP, mu-law 8 kHz, no websocket hop
LiveKit SIP                        inbound trunk ST_QdmtGTupQFfv, +1 213 845 1295
   │  dispatch rule SDR_vaKnVD8RcVKK → agent "stacy"
LiveKit Agents worker (us-east)    src/stacy/agent/entrypoint.py
   ├── Silero VAD                  "is someone speaking?"
   ├── turn detector (EnglishModel) "have they finished their sentence?"
   ├── Deepgram Nova-3             speech → text
   ├── the desk (our code)         scripted turns answered without a model
   ├── Claude Haiku 4.5            everything else
   ├── tools                       record_details, book_appointment, …
   └── ElevenLabs Flash v2.5       text → speech
   │
back down the same SIP leg
```

Two things about this diagram matter more than they look:

- **The audio never leaves 8 kHz mu-law telephony quality.** That is the ceiling
  on how well anything can hear the caller. A name that is ambiguous at 8 kHz is
  ambiguous for every vendor.
- **The worker is in us-east, next to the model providers.** Measured from
  inside it: 48 ms round trip to Anthropic, 12 ms to LiveKit's gateway. So when
  a reply is slow, it is the model thinking, not the network. Measuring this
  from a laptop in India would have told us the opposite (544 ms of the 746 ms
  was distance) and sent us down the wrong path. **Always measure from where the
  code runs.**

---

## 3. The repo, module by module

~9,000 lines of source, 657 tests across 23 files.

| Module | What it owns |
|---|---|
| `agent/entrypoint.py` | One call = one job. Builds the session, wires events, paces the VAD, warms the model cache, handles keep-warm and bench rooms. |
| `agent/nodes.py` | The three overridable pipeline nodes — `stt_node`, `llm_node`, `tts_node` — plus `on_user_turn_completed`, the hook where our code reads the caller's turn before the model does. |
| `agent/fastlane.py` | **The desk.** The biggest module (1,200 lines) and the heart of the design: scripted turns answered deterministically, with no model call. |
| `agent/slots.py` | The booking's working memory: what we know, what is still missing, in the prompt's own asking order. Phone/name/address parsing. |
| `agent/tools.py` | The seven tools the model can call, and every guard that stops a tool doing something irreversible and wrong. |
| `desk/calendar.py` | Availability, booking, and the superseding record that makes a correction replace a visit instead of adding one. |
| `agent/hangup.py` | Ending the call: the farewell read from the transcript in code, and the `end_call` tool for a caller who was never a customer. |
| `agent/speech.py` | Text on its way to the voice: audio tags stripped, markdown stripped, sentence flushing. |
| `agent/safety.py` | The gas/CO override. Latches in code, so no sampling step can go wrong. |
| `agent/turns.py` | Noise-held turn guard: answering when background noise, not the caller, is holding the turn open. |
| `agent/greeting.py` | The first sentence, pre-rendered and cached, because a stall before the first word sounds like a dead line. |
| `desk/` | Calendar and message stores — the actual booking records. |
| `prompt/` | The Jinja template and builder. One prompt, rendered per tenant. |
| `providers/` | `base.py` is every tunable with the reasoning in its docstring; `registry.py` builds the vendor objects. |
| `tenants/` | The tenant model and store. A business is a JSON file. |
| `metrics/live.py` | Per-reply latency records built from the SDK's own timestamps. |
| `bench/` | `llm_latency.py` (time models from inside the worker) and `shadow_stt.py` (compare recognisers on real call audio). |

---

## 4. The two-brain design (the thing to lead with)

Most voice agents send every turn to the LLM. We don't. Every caller turn goes
first to **the desk** (`fastlane.desk_reply`), which is ordinary Python:

```
caller turn
    │
    ├─ does the desk recognise this exact situation? ──yes──> deterministic reply (~0.4 s)
    │                                                          record what was heard
    └─ no ──> the model, with the prompt, the history and seven tools (~1.0–1.2 s)
```

The desk answers a turn only when the situation is unambiguous: a phone number
read out, a yes to the text-consent question, a name, a street, a town, a window
pick, a yes to the read-back (which books), a price or hours question (read out
of the tenant file), and "go ahead and book it". Anything else — the problem
description, an emergency judgment, an unusual question — is the model's.

**Why this is the right shape:**

1. **Speed.** A desk turn costs ~0.4 s against ~1.1 s. That is a bigger win than
   any model swap available (see §8.5).
2. **Correctness.** A deterministic reply cannot invent a price or a phone
   number. The turns most likely to be catastrophic when wrong — digits,
   addresses, bookings — are exactly the scripted ones.
3. **It degrades well.** If the desk doesn't recognise something, nothing breaks;
   the model handles it, a bit slower.

The cost is that the desk is the place bugs hide, because it is deciding on
regular expressions what a human meant. That is why `fastlane.py` has the most
tests in the repo and why every rule in it carries the live call that caused it.

---

## 5. Latency: where a second goes

### 5.1 Anatomy of one turn

```
caller stops talking
│
├─ 0.30 s   VAD silence, waiting to be sure they stopped   (0.45 patient / 0.20 eager)
├─ ~0.05 s  turn detector agrees the sentence is finished  (ceiling 1.2 s if it thinks not)
│           ── "end of turn" ≈ 0.35 s ──
│           (Deepgram's final transcript lands ~0.13–0.45 s in, overlapping the above)
├─ ~0.50 s  model's first word            ← only on model turns
├─ ~0.20 s  model finishes the first sentence
├─ ~0.12 s  ElevenLabs' first audio byte
└─ agent starts speaking
```

Desk turns skip the two model lines entirely — **and** their audio is built
during the wait, so what the caller experiences is close to the end-of-turn time
alone.

### 5.2 Measured, on real calls

| | Median | Notes |
|---|---|---|
| All replies | **0.95 s** | best full call; a call with bad transcription ran 1.14 s |
| Desk replies | **0.37–0.55 s** | mostly the deliberate wait |
| Model replies | **1.03–1.15 s** | |
| Model's first word | **0.50–0.56 s** | Haiku 4.5, warm prompt cache |
| Voice first byte | **0.12 s** | ElevenLabs Flash v2.5 |
| End of turn | **0.35 s** | |
| p90 / worst | 1.5 s / 2.0 s | |

History: 1.86 s average when the port was first working → 1.34 s → **0.95 s**.

### 5.3 The three tricks that buy the most

- **Preemptive generation.** The SDK starts the model on the caller's words
  *before* deciding they've finished, and cancels if they keep talking. Measured:
  the spoken model pass starts ~230 ms after the caller goes quiet, not ~400 ms.
- **Early desk replies.** The desk decides its reply during that same early pass,
  so the audio is already rendered when the turn commits. This is why a desk turn
  costs about what the waiting costs.
- **Prompt caching + a warm-up at call start.** The ~8,000-token prompt is read
  from Anthropic's cache every turn; a `max_tokens: 0` warm-up call fires while
  the greeting plays so the first real turn is warm too. First reply went from
  2.4 s to ~1.0 s.

### 5.4 Adaptive waiting

The wait before Stacy answers is not fixed. `fastlane.answer_kind` classifies
the question she just asked:

| Kind | Wait | When |
|---|---|---|
| patient | 0.45 s | after asking for a name, a number, an address — people read those out in groups with pauses |
| eager | 0.20 s | after the text-consent question, and after "Anything else?" — committing early is cheap |
| default | 0.30 s | everything else |

Never eager after a read-back: a yes there books an appointment, and "Yeah, but
the street's wrong" has to be heard out.

---

## 6. Correctness: how a wrong answer is made hard

The owner's rule is that a wrong booking costs far more than a slow reply. The
defences, in order of strength:

1. **Determinism.** Anything the desk answers cannot be hallucinated.
2. **Facts come from the file.** Price, hours, service area and FAQs are read out
   of the tenant record (`_on_question`), never composed.
3. **Code guards on irreversible actions.** In `tools.py`:
   - nothing books until the details have been read back *and* the caller's last
     words were not a "no" or a correction;
   - the same visit cannot be booked twice (name + street + resolved day + window);
   - the on-call technician cannot be dispatched twice;
   - a town outside the service area is refused where the town is recorded, not
     six fields later;
   - a phone number the caller never said is refused, and the model is told to
     ask again.
4. **The safety latch.** "I smell gas" latches in code and speaks a fixed script.
   There is no sampling step between the caller saying it and hearing what to do.
5. **The read-back.** Every booking is repeated to the caller before it is taken,
   and a spelled name is spelled back in the same breath.

---

## 7. The tooling (what an interviewer will actually be impressed by)

Building the agent is ordinary. Building the instruments is what made the work
converge.

| Tool | What it does | Why it exists |
|---|---|---|
| `metrics/live.py` | One JSON line per reply with the true caller-heard wait, from the SDK's own timestamps | Our earlier stage marks misreported every live call |
| `bench/llm_latency.py` + `scripts/bench_models.py` | Times candidate models **from inside the worker** by dispatching a job into a `bench-` room | A laptop in another country ranks providers by distance, not by speed |
| `bench/shadow_stt.py` | Runs extra transcribers on the same live call audio and logs what each heard | Recogniser accuracy can only be judged on the caller's real voice, line and accent |
| `scratchpad/call_sim.py` | Runs whole scripted calls through the real desk, prompt and tools against any model, with a model playing the caller | Compares models on what they *do*, not just speed — no phone line needed |
| `scratchpad/replay_fastlane.py` | Replays every recorded call through the current desk | Catches desk regressions against real transcripts |
| Mutation checks | Re-runs new tests against the code with the fix removed | Proves a test actually catches the bug it claims to |

The call simulator is the one to describe in detail: six scenarios chosen for
the ways a model can be *wrong* — a clean booking, an out-of-area town, price
and warranty questions, availability (Sunday is closed), a phone number
corrected mid-call, and a freezing house with a newborn.

---

## 8. War stories (know these cold)

### 8.1 The voice waited for the tool call

**Symptom:** model replies took ~1.6 s when the parts summed to ~0.9 s.

**Investigation:** for each turn, `model start + whole model pass + voice first
byte` matched the caller's wait to within 6 ms on 10 of 13 turns. So the voice
was starting only when the model's *entire* pass finished — including the
`record_details` call it writes after speaking.

**Root cause:** LiveKit's default `tts_text_transforms` includes a markdown
filter that holds back text after the last split token until more arrives. After
"What town are you", the next thing the model writes is a tool call, so the "in?"
sat in the filter for the whole tool-writing time.

**Fix:** drop that filter for the voice, strip markdown ourselves without
buffering, and flush a finished question immediately. Measured against the real
services: first audio 185 ms earlier on average, up to 414 ms.

**The lesson:** `tts_node_ttfb` is anchored at "text sent to the provider", so
buffering upstream of the provider is invisible in it. The metric looked healthy
while the caller waited.

### 8.2 The flush that ate the rest of the sentence

**Symptom (found in review, before a caller hit it):** "Is it gas? Or electric?"
could be spoken as "Is it gas?".

**Root cause:** `SynthesizeStream.flush()` ends the segment. Every later
`push_text` is dropped with a deprecation warning and never reaches the voice.
Our 120 ms stall-flush had the same flaw; the markdown filter above had been
accidentally hiding it by never letting the flush fire.

**Fix:** put the flush marker on the stream's input channel directly, which is
all the plugin acts on, so the segment stays open. Real ElevenLabs check: that
sentence went from 0.9 s of audio (truncated) to the full 3.3 s.

**The lesson:** the unit tests passed throughout, because the fake stream in the
tests accepted text after a flush and the real one silently dropped it. Test
doubles have to be as strict as the thing they stand in for.

### 8.3 "You're all set" — with nothing booked

**Symptom:** in a simulated call, the caller asked the price between the
read-back and her yes, then said "go ahead and book it". The model recorded the
details, told her she was booked for tomorrow — and never called
`book_appointment`. Nothing in the calendar, and she had no way to know.

**Fix:** the desk books it. When everything is recorded, the details have already
been read back, and the caller says to book, the desk calls the calendar and only
then speaks "You're all set".

**The near-miss:** my first version booked on a "Yes" to "Is that a gas furnace?",
because an earlier sentence in the same agent turn contained "waived if you book
the repair". The check now looks only at the question being answered. A test
written for exactly that case caught it.

### 8.4 The recogniser and the spelled name

Three consecutive calls booked a wrong name: "Nanak", spelled out, came back as
Ananak, then Nanat, then Nank. Deepgram's guidance is 300–500 ms of endpointing
"for conversations where speakers pause mid-thought"; we were set to 25 ms, so
the pause between letters ended the transcript and each fragment was decoded with
less context.

Two responses, deliberately different in kind:

- **A fix that depends on the vendor:** a longer endpointing, but only on name,
  number and address turns. Left **off** by default, because changing it mid-call
  reconnects the websocket and could swallow a barge-in. A shadow run on a real
  call decides it.
- **A fix that doesn't:** the name is spelled back in the same turn — "Thanks,
  Nanak — N, A, N, A, K. And what's the street address there?" — so a dropped
  letter is caught on the call, whatever the recogniser does.

### 8.5 The model bake-off

Measured from the worker, first word / first sentence:

| Model | First word | Verdict |
|---|---|---|
| Gemma 4 31B (LiveKit Inference) | **260 ms** | **Rejected** |
| Claude Haiku 4.5 (current) | 496 ms | Kept |
| Gemini 3 Flash | 793 ms | Slower |
| Grok 4.1 Fast | 4,836 ms | Unusable |

Gemma is twice as fast. On six simulated calls it also read a caller's *street
number* back as their phone number ("nine Birch Road" → "nine-zero-zero,
zero-zero-zero, zero-zero-zero-zero"), lost an address, looped five times
inventing excuses ("my screen is acting up"), and repeated a sentence twice in
one turn. Haiku got all six right.

**The conclusion to state plainly:** the only meaningfully faster model was the
one that makes up phone numbers, so the speed has to come from the desk instead.
This is the answer to "why didn't you just use a faster model?".

### 8.6 The four minutes the line was dead

`lk agent update-secrets --overwrite --secrets "ONE=value"` **replaces every
secret** rather than adding one. The worker was left with one variable and no API
keys. No call came in during the four minutes it took to notice and restore, but
that was luck, not process.

**The lesson, and the honest way to tell it:** I verified after writing instead of
before. The right sequence is list → write → list, and for anything touching a
live line, know what "undo" is before you touch it.

---

### 8.7 "My name is Bye"

Stacy could not hang up. She said goodbye and the line stayed open until the
caller gave up or a watchdog closed it half a minute later, so I gave her a
hangup decided in code from the transcript: the caller says they are done, the
agent answers with a sign-off, done.

The first version read the **end** of the caller's turn, which handles "No,
that's it. Thanks." and correctly ignores "Bye, but first — how much is it?".
Then I ran it over invented turns and found this one:

```
"My name is Bye"   ->  hang up
```

Nobody is called Bye. But Deepgram on an 8 kHz phone line hears names as all
sorts of things — this is the same recogniser that heard a spelled-out name
wrong in §8.4 — and the failure it produces is hanging up on somebody halfway
through introducing themselves. The fix was to require the sign-off to **be**
the caller's whole last clause rather than merely end it, so "My name is Bye"
and "It's Bye, B-Y-E" both stay on the call.

**The lesson:** the question for an irreversible action is never "does my rule
fire when it should?" It is "what is the worst input that fires it?" I found
this one by writing down the inputs I would hate to be wrong about before
writing the regex that judges them — and the test file now carries all 28.

---

### 8.8 "Forty Elm, not fourteen"

Stacy reads the details back and books on a yes, so a correction *before* the
booking has always been safe — the read-back guard refuses while the caller's
last answer was a fix rather than a yes. I was asked how corrections reach the
CRM and went to check the case I had not: a correction that arrives *after*
"you're all set".

It booked a second appointment. The wrong address stayed live, a technician
could drive to it, and it held a slot nobody was coming to.

The fix was a superseding record — append-only, so both rows survive as the
audit trail, with everything that counts visits reading through a filter that
drops the replaced one. But the interesting part was the second bug, which the
first fix created:

```
caller: it's 40 Elm, not 14
agent:  [calls book_appointment]
→ visit moved, with no new confirmation
```

The read-back check passed, because the caller's "yes" to the *original*
read-back was still sitting in the transcript, and that read-back named the
old address. A correction needs its own read-back, carrying the corrected
value, after the booking. Otherwise a van goes to a house the caller never
heard said back to them.

The third problem is the one with no clean answer. "Make it Thursday instead"
and "can you also come Thursday" change the same fields. The only honest
signal is the caller's own words, so a changed detail alone never amends — it
takes a fix word, and "also" or "as well" overrules it. A caller who moves a
day without saying anything corrective still gets two visits. That is the
conservative side on purpose: a duplicate is visible to everyone, and a
silently cancelled visit is visible to no one until a technician doesn't turn
up.

**The lesson:** when you make an operation idempotent, check what the *old*
confirmation now authorises. I would not have found the stale-yes bug by
testing the feature I had just written — it only showed up because the test
for "it waits for the read-back" failed for a reason I had not predicted.

---

## 9. Numbers worth memorising

- **Reply time: 0.95 s median, 1.5 s p90**, measured caller-stops-talking to
  agent-starts-speaking. Scripted turns **~0.4 s**, model turns **~1.0–1.2 s**.
- Roughly **40% of turns** are answered by code.
- Model first word **~0.5 s**; voice first byte **~0.12 s**; deliberate
  end-of-turn wait **~0.35 s**; network to the model **48 ms**.
- Prompt: ~8,000 tokens, ~25,000 characters, cached every turn.
- Cost of the prompt in latency: **25 ms**. Trimming it is not a latency lever.
- Test suite: **657 tests**, ~9,000 lines of source.

Always state the definition with the number. Many vendors quote a figure that
excludes the pause used to be sure the caller has finished, or that measures the
model alone. Ours is the wait a caller actually experiences.

---

## 10. Question drill

**"Why a cascade of vendors rather than one speech-to-speech model?"**
Control of the critical path and of correctness. A booking agent has to be
deterministic where it matters — digits, addresses, whether an appointment
exists. With a cascade we can put ordinary code between the caller and the
answer, which is what gets 40% of turns to 0.4 s and makes a class of mistakes
impossible. A single speech-to-speech model is fluent but every turn is a
sample, and there is nowhere to stand between it and the caller.

**"Why not use a faster LLM?"**
We measured four from inside the worker. The only meaningfully faster one
invented a phone number and lost an address in six test calls. With the network
at 48 ms, the model *is* the wait — but the cheapest safe win is moving another
turn type into code, which is worth ~0.7 s on that turn with no new failure mode.

**"How do you know it's 0.95 s?"**
From the SDK's own timestamps, one JSON record per reply, on real calls:
caller's `stopped_speaking_at` to the agent's `started_speaking_at`. Not a
synthetic benchmark, and not the model's own latency in isolation.

**"How do you stop it saying something wrong?"**
Layered: deterministic replies for scripted turns; facts read out of the tenant
file; code guards on every irreversible action; a safety latch that bypasses the
model entirely; and a read-back before anything is committed. Then six scripted
calls run against the real model and the transcripts are judged for invented
facts and false claims.

**"What was the hardest bug?"**
The voice waiting for the tool call (§8.1). Everything looked healthy — the
model was fast, the voice was fast, the metric said 120 ms — and callers still
waited 1.6 s. The only way in was reconstructing each turn's timeline from logs
and noticing the wait matched the *end* of the model's pass to within 6 ms.

**"What would you do next?"**
Prove the recognition fixes on a real call with the shadow transcriber; move the
goodbye and the system-type question into the desk; and keep the model bench
ready, because Anthropic's stated retirement date for Haiku 4.5 is "not sooner
than 15 October 2026".

**"What's the weakest part?"**
Speech recognition on accented speech over an 8 kHz line, and it is the largest
remaining source of wrong data. Also: one tenant is live, so the multi-tenant
claim is structural rather than proven at scale.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| VAD | Voice activity detection (Silero). Is there speech in this audio frame? |
| Endpointing | Deciding the caller has finished their turn. Ours is semantic (a model judges the sentence), with a VAD silence floor and a 1.2 s ceiling. |
| TTFT / first word | Time to the model's first token. |
| TTFB | Time to the voice's first audio byte, measured from text being sent to the provider. |
| Preemptive generation | Starting the model on the caller's words before deciding they have finished; cancelled if they continue. |
| Barge-in | The caller talking over the agent, which interrupts it. |
| The desk / fast lane | Our deterministic reply layer (`fastlane.py`). |
| Slots | The booking's working memory: what is known and what is still missing. |
| Keyterm prompting | Nova-3's way of boosting expected words (town names, "furnace"). |
| Shadow STT | A second recogniser on the same call audio, logging only. |
| Tenant | A business, as a JSON record. |
