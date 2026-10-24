# When self-hosting pays — a cheat sheet

Three different "self-host" decisions get confused because they sound alike.
They differ by a factor of ten in cost and by an order of magnitude in
difficulty.

| | **Transport** | **GPU for the cascade** | **GPU for speech-to-speech** |
|---|---|---|---|
| What it is | The media server and SIP signalling, run by you instead of LiveKit Cloud | Transcription and voice running on your own card — the model stays an API | One model hears and speaks, all on your card |
| Hardware | A normal CPU box | RTX 4090 class | A100 80GB |
| Monthly fixed | **$60–150**<br>₹5,800–14,500 | **$245–497**<br>₹23,600–48,000 | **$857–1,044**<br>₹82,700–1,00,750 |
| Replaces | ₹1.74/min | ₹1.75/min | ₹1.78/min |
| **Starts paying above** | **~5,500 min/mo** | **~13,500 min/mo** | **~46,500 min/mo** |
| Difficulty | A weekend | Weeks — streaming STT is the hard part | Weeks, plus you give up tool calls |

**The headline: a GPU is a better buy for the cascade than for
speech-to-speech.** Same saving per minute, a third of the hardware, and you
keep the text inspection, the reliable tool calls and the safety override.

---

## What the GPU would actually run

| Piece | Share of the bill | Self-hosted with | Worth it? |
|---|---|---|---|
| **Voice (TTS)** | 39% | Kokoro, XTTS, Orpheus | **Yes — biggest line, easiest swap** |
| **Transcription (STT)** | 18% | Whisper turbo, Parakeet | Yes, but the hardest piece |
| The model | 9% | an open model | **No** — ₹0.03/min on Haiku 5.5. Nothing to save |
| Transport | 27% | LiveKit on a CPU box | Yes, and no GPU needed |

---

## The numbers, by volume

Tuned Stacy — Haiku 5.5, Aura-2 voice, promo STT rate, 60% of turns scripted.
Marginal cost ₹2.11/min before any fixed hosting.

| min/month | On LiveKit Cloud | **Own VM** | **Own VM + GPU** |
|---|---|---|---|
| 1,000 | ₹3.84 | ₹11.76 | ₹33.63 |
| 3,500 | ₹3.84 | ₹4.86 | ₹9.87 |
| 12,000 | ₹3.84 | **₹2.91** | ₹3.13 |
| 35,000 | ₹3.84 | **₹2.38** | **₹1.31** |
| 100,000 | ₹3.84 | ₹2.20 | **₹0.69** |

Cloud is flat — no fixed cost, so it never gets cheaper and never gets worse.
Both self-hosted columns are brutal at low volume and win decisively at high.

**Break-evens**

- Own VM beats cloud above **5,556 min/mo** ($100 box); 3,333 on a $60 box, 8,333 on a $150 one.
- GPU for STT+TTS pays above **13,525 min/mo** at $245, **27,448** at $497.
- GPU for speech-to-speech pays above **46,565 min/mo** — and ~57,000 if the cascade is tuned first, because tuning cuts what the GPU is competing against.

You also rent all **720 hours** in a month while a business line uses about
**260**. Every quoted per-minute GPU figure assumes it is busy.

---

## The order to do it in

1. **Tune the cascade.** Free-ish, no infrastructure, and it lowers the bar every later step has to clear.
2. **Own VM for transport** at ~5,500 min/mo. A weekend, then light upkeep.
3. **GPU for the voice only** at ~13,500 min/mo. Biggest line, easiest to swap, easiest to A/B.
4. **GPU for transcription** last. Streaming recognition with good endpointing is weeks of work, not a weekend.

---

## The part that is not about money

Even where a GPU is cheap, **speech-to-speech** gives up three things a
booking agent needs:

1. **Reliable tool calls.** It must call `book_appointment` with the right fields every time. S2S models are worse at structured tool use than text models. This is the real blocker, not cost.
2. **Text you can inspect before it is spoken.** Scripted turns and the deterministic safety override both depend on seeing the words first.
3. **A vendor who fixes it.** When your own GPU serves a bad reply at 2 a.m., that is yours.

A GPU inside the **cascade** gives up none of these — it only swaps who runs
the transcriber and the voice. That is why it is the better GPU bet.

So: **cascade for anything that writes to a calendar or a CRM.** S2S for
qualification and FAQ calls where nothing is recorded.

---

## Quick answers

| If asked | Say |
|---|---|
| "When does our own transport box pay?" | Above about 5,500 minutes a month. ₹100/month replaces ₹1.74 a minute. |
| "When does a GPU pay?" | About 13,500 minutes a month — for the voice and transcription, inside the cascade. For speech-to-speech it is 46,500. |
| "Should we self-host the model?" | Not for booking calls. You lose reliable tool calls and the ability to check the words before they are spoken. |
| "Why is a GPU better in the cascade?" | Same saving, a third of the hardware, and you keep everything that makes the agent safe. |
| "Why not Gemini Live? It's ₹2.86." | Cost is fine. Tool calling, no text to inspect, and audio leaving India are the problems. |
| "Why not OpenAI Realtime?" | ₹5.54–11.33/min against an Enterprise price of ₹7.14. It loses money on the biggest accounts. |
| "What's the single cheapest win?" | The model tier — ten times cheaper per token, same quality class, ten minutes of work. |
| "What saved the most already?" | Prompt caching. Four times more than a model swap would. |

---

Numbers from [`scripts/cost_model.py`](../scripts/cost_model.py) at ₹96.5 to
the dollar. GPU rates: RTX 4090 $0.34–0.69/hr, A100 80GB $1.19–1.45/hr.
Concurrency per box and per card are estimates and have not been load-tested;
self-hosted streaming STT and TTS quality on Indian English are both unproven
here. Only the model prices are vendor-direct — see
[`cost-per-minute.md`](cost-per-minute.md) §7.
