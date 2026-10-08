# Stacy — what it does, and what it had to fix to do it

A reference for someone who wants to find a specific capability or a specific
fix without reading the whole codebase. Everything here is in the repository
and covered by a test; where a row says a thing was wrong, it was wrong on a
real call or in a replay of one, and the fix is a guard in code rather than a
sentence added to a prompt.

**What it is.** A multi-tenant AI phone receptionist for trade businesses
(HVAC first). It answers a real phone number, books appointments against a
real calendar with real capacity and opening hours, handles gas-leak
emergencies, takes messages, answers questions from the business's own file,
and ends the call like a person would.

**Stack.** Twilio Elastic SIP trunk → LiveKit SIP → LiveKit Agents 1.8 worker
→ Silero VAD + semantic turn detector → Deepgram Nova-3 → Claude Haiku 4.5 →
ElevenLabs Flash v2.5.

**Size.** 9,793 lines of source across 43 modules; 5,862 lines of tests
across 23 test modules; **666 tests**, almost none of which need a vendor
package installed.

| | |
|---|---|
| Reply latency, median | **0.95 s** (scripted turns ~0.4 s, p90 1.5 s) |
| Tools the model can call | 7, every one backed by a real store |
| Tests | 666, green |
| Preflight | 28 checks, 0 failing, 4 warnings (all awaiting credentials or a decision) |

---

## 1. Finding things quickly

| Looking for | Go to |
|---|---|
| Latency: what was slow, what fixed it | §4, and `docs/how-stacy-works.md` §5 |
| Correctness guards on booking | §3.1 |
| Corrections that reach the record | §3.2 |
| Gas leak / emergency handling | §3.3 |
| Speech recognition problems | §3.4 |
| How the call ends | §3.5 |
| Error handling and failure modes | §5 |
| Multi-tenancy, what's configurable | §6 |
| How any claim here was measured | §7 |
| Test suite map | §8 |
| Fix history as a timeline | §9 |
| What is **not** done | §10 |

---

## 2. What it does on a call

| Capability | Where | Notes |
|---|---|---|
| Answer an inbound PSTN call | `agent/entrypoint.py` | Twilio trunk → LiveKit SIP → worker job; tenant resolved from the dialled number |
| Pre-rendered greeting | `agent/greeting.py` | The opening line is synthesised ahead of time and cached on disk, so the first thing a caller hears costs no model or TTS time |
| Recognise a returning caller | `desk/calendar.py` `history_for` | Turns "can I get your address?" into "is this about the unit at fourteen Elm?" |
| Collect booking details | `agent/slots.py` | Absorbs several facts from one sentence; asks only for what is missing |
| Read back and confirm | `agent/tools.py` | Nothing is booked until the caller has heard it back and said yes |
| Book against real capacity | `desk/calendar.py` | Opening hours, elapsed windows, per-window capacity, service-area towns |
| Correct a booking already made | `desk/calendar.py` `amend` | Replaces the visit instead of adding one; see §3.2 |
| Dispatch an emergency | `agent/tools.py`, `desk/calendar.py` | Separate path from booking, with its own required fields |
| Gas / CO safety override | `agent/safety.py` | Taken away from the model entirely; see §3.3 |
| Take a message | `desk/messages.py` | The fallback whenever a booking cannot complete |
| Answer questions from the business's file | `agent/fastlane.py` | Hours, prices, services, FAQs — answered in code, not by the model |
| Consent-gated SMS confirmation | `delivery/sms.py` | Will not send without a recorded yes |
| Post-call webhook | `delivery/webhook.py` | Signed, with a retry ladder and auto-disable |
| End the call | `agent/hangup.py` | Two paths, both guarded; see §3.5 |
| Per-turn latency measurement | `metrics/live.py` | From the SDK's own timestamps, one JSON line per reply |
| Scored evaluation | `evals/` | 8 criteria, 11 data fields, LLM judge |
| Retention sweep | `retention.py`, `scripts/retention_sweep.py` | Transcripts and traces do not live forever |

---

## 3. The guards — what was wrong, and what fixed it

Every row is a real failure. The "refuses when" column of the tool table is
the short version; this is the long one.

### 3.1 Booking

| Symptom on a call | Cause | Fix | Test |
|---|---|---|---|
| "You're all set" with nothing in the calendar | The model said it before the tool returned | Only a reply starting `Booked` means booked; the prompt keys on it and the tool is the only thing that can say it | `test_agent.py::TestNothingIsBookedBeforeTheCallerHearsItBack` |
| Booked before the caller agreed | The read-back and the booking happened in one breath | `book_appointment` refuses unless a read-back naming the caller and the window appears before the caller's latest turn | `test_latency_overhaul.py::TestReadBack` |
| Booked on "No, that's all" | A no was treated as a yes | Refuses when the caller's answer to the read-back is a no **or** a correction | `test_agent.py::TestDeskOperations` |
| The same visit booked twice | The model called the tool again | Identical visit (name, street, day, window) returns "booked already — don't book it again" | `test_latency_overhaul.py::TestSecondReview` |
| A second visit on another day refused | The duplicate check was too broad | Day is part of the identity, so Thursday-after-tomorrow is a real second booking | `test_latency_overhaul.py::TestSecondReview` |
| A window offered that the office never runs | Capacity was checked, hours were not | Three ordered filters: open day, window not already elapsed, capacity left | `test_desk.py::TestAvailability` |
| A booking with no street address | Prompt said never to; nothing enforced it | `CalendarStore.book` raises without one — structural, not advisory | `test_desk.py::TestBooking` |
| A town outside the service area booked | — | `record_details` rejects it and names it | `test_slots.py::TestRecordDetails` |
| Two workers took the last slot | The lock was in-process only | Capacity check and append happen under a cross-process lock file | `test_desk.py::TestTwoWorkersOneCalendar` |
| Asked for a name already given | The prompt's "one item per turn" was read as "one item per sentence" | Slots absorb everything heard; the rule governs what is *asked* | `test_slots.py::TestDetailsHeardInSpeech` |
| A phone number the caller never said | The model offered digits from memory | Only digits the caller or `record_details` supplied can be read back | `test_slots.py::TestOnlyNumbersTheCallerSaid` |
| Message ordering broke on Windows | 15 ms clock resolution tied timestamps | Append order breaks ties (deliberate `sorted(reversed(...))`, with the reason in a comment) | `test_desk.py::TestMessages` |

### 3.2 Corrections that reach the record

The case someone asked me about directly, and the one I had not checked.

| Symptom | Cause | Fix |
|---|---|---|
| "Sorry, it's **forty** Elm, not fourteen" after "you're all set" produced a **second** appointment — the wrong one stayed live, a technician could drive to it, and it held a slot nobody was coming to | Any change to the details looked like a new visit | A correction appends a record carrying `supersedes: <old id>`. Both rows stay on disk as the audit trail; `_live()` is what counts capacity, fills the history, and reaches the technician |
| The corrected visit moved with **no new confirmation** | The caller's "yes" to the *original* read-back was still in the transcript, and that read-back named the *old* address | `_correction_confirmed()` requires a read-back that came after the booking **and** contains the changed value, then a plain yes |
| Risk: a second job silently cancelled | "Can you also look at the AC while you're here?" changes the same fields a correction does | A changed detail alone never amends. It takes the caller saying a fix — and "also" / "as well" / "while you're at it" overrules a fix word in the same sentence |

| They said | Read as | Why |
|---|---|---|
| "Sorry, it's 40 Elm, not 14." | correction | a fix, nothing added |
| "Can you also look at the AC while you're here?" | second visit | "also" |
| "Actually, can you *also* look at the AC?" | second visit | "also" outranks "actually" |
| day changed, nothing said | second visit | silence is not a correction |

Tests: `test_corrections.py` (12). **Known limit:** "make it Thursday instead"
and "can you also come Thursday" change the same fields; a caller who moves a
day without saying anything corrective still gets two visits. That is the
deliberate side to fail on — a duplicate is visible to everyone, a silently
cancelled visit is visible to no one until the van doesn't arrive.

### 3.3 Safety

| Symptom | Fix | Test |
|---|---|---|
| A gas-leak mention depended on the model choosing to follow the script | The override is taken away from the model: an interceptor scans the caller's text, latches, and repeats the evacuation instruction in the same words regardless of what is said next | `test_safety.py::TestLatching`, `test_port_fidelity.py::TestSafetyOverride` |
| Something could still act after the override latched | No tool can run once the interceptor bypasses the model; `note_booking` logs loudly if one ever does | `test_agent.py::TestSafetyOverrideAtRuntime` |
| The call could be ended during an evacuation instruction | Both hangup paths refuse while latched — the caller leaving the line is the whole point | `test_hangup.py` |
| Safety graded by vibes | It is graded structurally in the eval harness, not by a judge's opinion | `test_evals.py::TestSafetyIsGradedStructurally` |

### 3.4 Speech and hearing

| Symptom | Cause | Fix |
|---|---|---|
| Agent answered the room, not the caller | Background speech held the turn open | A noise-held-turn guard, plus input filtering; each failure mode leaves a distinct log line (`agent/turns.py`) |
| "Are you still there?" while it was thinking | Waiting-for-you and thinking were the same timer | They are separated, and measured separately (`metrics/turn.py`, `TestEndpointingIsSeparableFromThinking`) |
| Audio tags spoken aloud | TTS read markup | `filter_markdown` / `filter_emoji` transforms, and `[slow]` handled as a control (`agent/speech.py`) |
| The last word of a sentence held back | `filter_markdown` buffers the final token | Known and documented; the flush path drives it out |
| `flush()` dropped the rest of a sentence | The SDK's `flush()` ends the segment, discarding later `push_text` | `flush_voice()` puts a `_FlushSentinel` on the input channel instead. The unit tests had passed because the fake stream hid it, so they were rewritten against a real SDK stream | `test_speech_flush.py` |
| A spelled-out name heard wrong | 25 ms endpointing cut the spelling | Keyterm prompting, `utterance_end_ms`, and a spelled-name read-back (`_said_back`) |
| No way to compare recognisers without risking a call | — | Shadow STT: extra transcribers on the same live audio, logging only, never speaking, never blocking (`bench/shadow_stt.py`) |

### 3.5 Ending the call

Previously Stacy said "bye" and sat there — the prompt admitted she had no way
to hang up, so the line stayed open until the caller gave up or a watchdog
closed it 30 s later.

**The automatic path is decided in code from the transcript, never by the
model.** Four things must hold:

1. The caller said they were done, in their own last turn.
2. The agent's reply was a sign-off, and the agent spoke last.
3. Neither turn left a question hanging.
4. The call finished what it started — booked, dispatched, or a message taken
   — or never started anything.

| Symptom | Fix |
|---|---|
| "My name is Bye" would have hung up on someone introducing themselves | The sign-off must **be** the whole last clause, not merely end the turn. (Deepgram hears names as all sorts of things on an 8 kHz line — see §3.4) |
| "Take care, Dan — did you want the filter on the ticket too?" read as a farewell | No cut while either turn leaves a question open |
| The line could be cut with the caller's details on the pad and nothing booked | `business_unfinished()` — read by **both** ending paths, so they cannot disagree |
| An interrupted goodbye left the intent armed and hung up a turn later | The intent is consumed on the turn it was set, whether or not the goodbye survived |
| The SIP leg dropped mid-word | Both paths `drain()` first; a caller who speaks during the drain keeps their call |

`end_call` exists for a caller who was never going to be a customer (wrong
number, sales call, somebody abusive already offered help once). Tests:
`test_hangup.py` (31).

---

## 4. Latency

Measured caller-stops-talking to agent-starts-speaking, on real calls,
including the deliberate end-of-turn pause.

| | Measured | What owns it |
|---|---|---|
| **Whole reply, median** | **0.95 s** | everything below |
| A turn the desk answers | 0.37–0.55 s | mostly the pause |
| Waiting to be sure they finished | ~0.35 s | VAD floor + semantic detector |
| Deepgram's final transcript | 0.13–0.45 s | overlaps the pause, usually free |
| Claude Haiku's first word | 0.50 s | the model, nothing else |
| Finishing the first sentence | ~0.20 s | token rate |
| ElevenLabs' first audio byte | 0.12 s | vendor |
| Network to the model host | 0.048 s | region pinning |

Progression across real calls: **1,864 → 1,340 → 962 → 952 ms** median.

| Technique | What it bought |
|---|---|
| **The desk / fast lane** (`agent/fastlane.py`) | ~40 % of turns answered in code at ~0.4 s instead of ~1.1 s |
| Tools without strict schemas | 3.5 s a turn |
| SSL imports moved to prewarm | the first turn stopped paying for them |
| Prompt caching + a `max_tokens: 0` warm-up | prompt cost down to ~25 ms |
| Pre-rendered greeting | the opening line costs no model or TTS time |
| Region pinning | 48 ms to the model host |
| Semantic endpointing ceiling at 1.2 s | shorter tail without clipping people |
| Covering slow **tools**, not slow models | no filler where there is nothing to wait for |

---

## 5. Error handling and failure modes

| Failure | Behaviour |
|---|---|
| **A desk tool raises something nobody planned for** (full disk, lock-file permission, a bug) | Previously raised through the SDK and the caller got dead air. Now returns a sentence that cannot be mistaken for success — it says it failed, that nothing was written down, and to offer a callback — with a logged traceback and a turn annotation so it is still a bug someone must fix (`test_tool_failure.py`) |
| Cancellation / shutdown | The catch is `Exception`, not `BaseException`, so a worker can still be stopped — with a test, because getting that wrong gives you a worker you cannot kill |
| Calendar held by another process | `BookingError("the calendar is busy — take a message and call them back")` after a 5 s deadline |
| Booking cannot complete, for any reason | Falls back to taking a message |
| Corrupt JSONL line | Skipped, not fatal |
| Corrupt greeting cache | Treated as a cache miss, not an outage |
| Webhook network error | Retry ladder, then auto-disable (`test_delivery.py::TestRetryLadder`, `TestAutoDisable`) |
| Shadow STT crash | Cannot disturb the call — caught and the shadow is dropped |
| Failed drain on hangup | Logged; the call is still ended rather than left open |
| `delete_room` fails | Falls back to closing the session |
| Bad tenant file | Surfaces rather than crashing the worker |
| Prompt warm-up fails | Costs the old cold start, nothing else |

All 69 exception handlers in `src/` are deliberate; every broad catch carries
a comment saying why. **Open gap:** no LLM fallback is configured — the
cascade machinery exists (`LLM_FALLBACK_MODEL`) but is unset, so a
rate-limited primary is a ~4 s hole mid-call. See §10.

---

## 6. Multi-tenancy and configuration

Two rules make a second client cheap, and a third makes the policy layer
testable:

1. **Nothing outside `tenants/` knows a business fact.**
2. **Nothing outside `providers/registry.py` knows a vendor name.**
3. **Nothing outside `agent/nodes.py`, `agent/entrypoint.py` and `bench/`
   imports the SDK at module scope** — which is why almost all 666 tests run
   with no vendor package installed.

Per tenant, as data (`src/stacy/tenants/data/`, 185 lines for the first
client): phone number, business name, trade, city, timezone, agent name and
persona, service-area towns, staff, booking windows, opening hours per day,
services with price ranges, FAQs, emergency policy, capacity per window,
ASR keyterms, STT language, voice and delivery settings, recording
disclosure, SMS-consent requirement, reseller/escalation contacts.

Infrastructure comes from the environment, so moving every tenant to a new
model host is one environment change.

---

## 7. How the claims here were measured

Built because the arguments they settle cannot be settled by reading code.

| Instrument | Answers |
|---|---|
| `metrics/live.py` | What did the caller actually wait, per reply? |
| `bench/llm_latency.py` + `scripts/bench_models.py` | How fast is each candidate model **from inside the worker**? Measuring from a laptop ranks providers by distance |
| `bench/shadow_stt.py` | Which recogniser settings hear this caller? |
| `scripts/latency_report.py` | p50/p90 over a log, desk split from model |
| `scripts/preflight.py` | Would a call fail right now, and why? Cheaper than a call |
| `scripts/dry_run.py` | A whole call without a phone |
| `evals/` | 8 criteria and 11 data fields, scored by an LLM judge |

The model bake-off is the clearest example: "use a faster LLM" was settled
with evidence — one candidate was twice as fast and invented phone numbers.

---

## 8. Test suite map

666 tests across 23 modules. The large files are the interesting ones: each
test class is a cluster of related failures.

| File | Tests | Covers |
|---|---|---|
| `test_agent.py` | 95 | Runtime, turn lifecycle, desk operations, routing, post-call payload, safety at runtime |
| `test_slots.py` | 88 | Phone normalisation, numbers read aloud and over two turns, spelled names, non-US numbers, details heard in speech |
| `test_latency_overhaul.py` | 85 | Waiting, offers, read-back, early desk replies, warm-up, three review passes |
| `test_fastlane.py` | 70 | The desk: numbers, consent, names, windows, and three live calls replayed |
| `test_port_fidelity.py` | 52 | Promises to callers, conversation discipline, tenant parameterisation, audio path, the two-account cascade |
| `test_safety.py` | 34 | Latching |
| `test_hangup.py` | 31 | Ending itself, `end_call`, politeness, not-before-booked, the wiring |
| `test_desk.py` | 29 | Days, availability, booking, dispatch, returning callers, messages, two workers one calendar |
| `test_evals.py` | 25 | Judge, structural safety grading, aggregation |
| `test_delivery.py` | 22 | Webhook signature, retry ladder, auto-disable, payload |
| `test_backchannel.py` | 20 | Backchannel behaviour |
| `test_greeting.py` | 15 | Cache key, cache on disk, prepare |
| `test_corrections.py` | 12 | §3.2 in full |
| `test_turns.py` / `test_speech.py` / `test_metrics.py` | 12 each | Turn guards, speech transforms, trace plumbing |
| `test_retention.py` | 10 | What goes and what stays, policy from the deployment |
| `test_live_latency.py` | 10 | Real reply times from SDK timestamps |
| `test_tool_failure.py` | 9 | §5, unanticipated tool failures |
| `test_tool_schema.py` | 8 | The tools as the model sees them, against the real SDK |
| `test_speech_flush.py` | 6 | The flush bug, against a real SDK stream |
| `test_turn_handling.py` / `test_llm_config.py` | 5 / 4 | Turn handling, LLM config |

Guards are verified by **mutation**, not assumption: delete the guard, confirm
a test fails. One guard initially survived that check, which is how its test
got written.

---

## 9. Fix history

The substantive commits, newest first. Deploy chores, renames and
documentation passes are left out; `git log` has all 47.

- A tool that fails says so instead of going quiet
- Correct a booking instead of booking it twice
- Don't put the phone down before the job is done
- Let Stacy put the phone down
- Hold the calendar across processes, and stop keeping transcripts forever
- Speak sooner and book only what the caller agreed to
- Measure the models from the worker, not from a laptop in another country
- Answer scripted booking turns in code and cut reply latency
- Answer a caller that background noise is holding on the line
- Send tools without strict schemas: that was three and a half seconds a turn
- Cover slow tools instead of slow models, and stop speaking audio tags
- Separate waiting for you to finish from thinking, and stop covering neither
- Pay for the SSL imports at prewarm, not on the event loop mid-call
- Stop the agent asking if you are there while it is thinking, and cut the wait
- Answer the caller, not the room
- Slow the voice down, widen the accent, and show what was actually heard
- Pre-render the greeting, and add the LiveKit Cloud deployment
- Probe the audio format the pipeline actually asks for
- Make the worker actually start: register plugins on the main thread
- Give the model a way to say "they didn't tell me", and stop sending temperature
- Ask the vendors whether the keys work, instead of whether they are present
- Generate the SIP config from the tenants, so the number can't be typed twice
- Let one sentence fill three slots, and stop asking for what was said
- Run the 8 criteria, so a model swap stops being a matter of opinion
- Take the gas/CO override away from the model, and make the prompt cacheable
- Carry every booking on a call, and add a dry run that found why it mattered
- Give the agent a runtime, five working tools, and a way to cover a pause

---

## 10. What is not done

Stated plainly, because a document like this is worth less if it only lists
wins.

| Open | What it needs |
|---|---|
| **No LLM fallback** | The cascade exists but is unconfigured; a rate-limited primary is a ~4 s hole mid-call. Needs a second provider key, or code for a keyless LiveKit Inference vendor |
| **Post-call webhook unset** | Stacy promises someone gets the caller's details. Calls are archived locally; nothing is delivered. Needs a URL and secret |
| **SMS is a logging stub** | Code is written and consent-gated; confirmations go to the log, not a phone. Needs Twilio credentials |
| **Retention sweep not scheduled** | Policy is set; nothing runs `retention_sweep.py --apply`, so plaintext accumulates |
| **Hangup and corrections unproven live** | Both are fully covered by tests and neither has run on a real call |
| **Media pipeline under real failure untested** | What happens on the line if Deepgram or ElevenLabs drops mid-call needs a live call with a provider deliberately broken, not a unit test |
| **One tenant** | Multi-tenancy is structural and tested, but only one business is configured |
| **No type checker** | ruff is configured and green; mypy is not installed |
| **`fastlane.py` is 1,200 lines** | It works and is heavily tested, but it wants splitting |

---

## 11. Where to look

| Path | What it owns |
|---|---|
| `agent/entrypoint.py` | Job entry, SIP glue, session wiring, event handlers |
| `agent/runtime.py` | One call's state: tenant, transcript, slots, appointments, outcome |
| `agent/tools.py` | The seven tools and every guard that stops one doing something irreversible and wrong |
| `agent/fastlane.py` | The desk: scripted turns answered in code |
| `agent/slots.py` | What the caller has told us, and what is still missing |
| `agent/safety.py` | The gas/CO override, taken away from the model |
| `agent/hangup.py` | Ending the call |
| `agent/turns.py` | Turn-taking and the noise-held-turn guard |
| `agent/speech.py` | Text transforms and the flush path |
| `agent/greeting.py` | Pre-rendered opening line |
| `desk/calendar.py` | Availability, booking, dispatch, and the superseding record |
| `desk/messages.py` | The message pad |
| `providers/registry.py` | The only vendor-aware module |
| `tenants/` | The tenant record, including the knowledge base |
| `prompt/` | The prompt as a Jinja template plus renderer |
| `delivery/` | Webhook and consent-gated SMS |
| `metrics/`, `bench/`, `evals/` | Instruments and scoring |
| `retention.py` | What gets deleted, and when |

Further reading: [`README.md`](../README.md) for the full narrative, and
[`docs/how-stacy-works.md`](how-stacy-works.md) for the design walkthrough and
the war stories in detail.
