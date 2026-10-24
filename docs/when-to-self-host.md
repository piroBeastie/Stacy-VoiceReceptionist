# When self-hosting pays — a cheat sheet

Two different "self-host" decisions get confused because they sound alike.
They differ by a factor of ten in cost and by an order of magnitude in
difficulty.

| | **Transport** | **The model (GPU)** |
|---|---|---|
| What it is | The media server and SIP signalling — LiveKit, run by you instead of LiveKit Cloud | A speech-to-speech model (Qwen3-Omni Thinker–Talker) running on your own GPU |
| Hardware | A normal CPU box | A100 80GB or better |
| Replaces | LiveKit Cloud's per-minute fee | Transcription + model + voice, all three |
| Monthly fixed cost | **$60–150** (₹5,800–14,500) | **$1,044–1,980** (₹1,00,746–1,91,070) |
| Pays off above | **~3,300–8,300 min/month** | **~31,000 min/month**, or ~57,000 against a tuned cascade |
| Difficulty | A weekend, then light maintenance | You own scaling, failover and the 3 a.m. page |

---

## Transport: self-host above roughly 5,000 minutes a month

LiveKit Cloud costs **₹1.74/min** (agent session + SIP). A box is fixed, so
the per-minute cost falls as volume rises.

| Minutes/month | $60/mo box | $150/mo box | vs Cloud ₹1.74 |
|---|---|---|---|
| 1,000 | ₹5.79 | ₹14.47 | **worse — stay on Cloud** |
| 3,500 | ₹1.65 | ₹4.14 | small box breaks even |
| 12,000 | ₹0.48 | ₹1.21 | **clearly cheaper** |
| 35,000 | ₹0.17 | ₹0.41 | **much cheaper** |
| 100,000 | ₹0.06 | ₹0.14 | negligible |

**Break-even: 3,333 min/mo on a $60 box, 8,333 on a $150 box.**

What you give up: uptime is yours. Cloud absorbs a bad deploy; your own box
does not. Worth it once the saving is more than a few hours of your month.

---

## GPU: don't, until about 31,000 minutes a month

An A100 at $1.45/hr is **₹1,00,746 a month whether the phone rings or not.**

| Minutes/month | GPU cost per minute | Tuned cascade brain |
|---|---|---|
| 1,000 | **₹100.75** | ₹1.78 |
| 3,500 | ₹28.78 | ₹1.78 |
| 12,000 | ₹8.40 | ₹1.78 |
| 35,000 | ₹2.88 | ₹1.78 |
| 100,000 | **₹1.01** | ₹1.78 |

**Break-even: 31,146 min/mo against today's cascade. 56,739 against a tuned
one** — tuning the cascade moves the goalposts, so do that first and the GPU
stops paying until roughly 57,000 minutes.

And you rent all **720 hours** in a month while a business line uses about
**260** of them. The quoted ₹0.82/min assumes the GPU is busy; a phone line
is not busy at 3 a.m.

---

## The part that is not about money

Even where the GPU is cheap, speech-to-speech gives up three things a
booking agent needs:

1. **Reliable tool calls.** It has to call `book_appointment` with the right
   fields every time. S2S models are worse at structured tool use than text
   models. This is the real blocker, not cost.
2. **Text you can inspect before it is spoken.** Scripted turns and the
   deterministic safety override both depend on seeing the words first.
3. **A vendor who fixes it.** When your own GPU serves a bad reply at 2 a.m.,
   that is yours.

So: **cascade for anything that writes to a calendar or a CRM.** S2S is for
qualification and FAQ calls where nothing is recorded.

---

## Quick answers

| If asked | Say |
|---|---|
| "Should we self-host the GPU?" | Not under ~31,000 min/month, and tune the cascade first — that pushes it to ~57,000. |
| "Should we run our own transport?" | Yes above ~5,000 min/month, if someone can own uptime. ₹60–150 a month replaces ₹1.74 a minute. |
| "Why not Gemini Live? It's ₹2.86." | Cost is fine. Tool calling, no text to inspect, and audio leaving India are the problems. |
| "Why not OpenAI Realtime?" | ₹5.54–11.33/min against an Enterprise price of ₹7.14. It loses money on the biggest accounts. |
| "What's the single cheapest win?" | The model tier — ten times cheaper per token, same quality class, ten minutes of work. |
| "What saved the most already?" | Prompt caching. Four times more than a model swap would. |

---

Numbers from [`scripts/cost_model.py`](../scripts/cost_model.py) at ₹96.5 to
the dollar, A100 at $1.45/hr. Concurrency per box and per GPU are estimates
and have not been load-tested. Only the model prices are vendor-direct; the
rest come from public trackers — see [`cost-per-minute.md`](cost-per-minute.md) §7.
