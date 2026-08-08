# Peripheral — Results

Every number here was measured on the machine described below. Nothing is estimated. Where a
number was not measured it says so rather than showing a zero.

**Hardware.** RTX 5070 Ti Laptop, 11.94 GB VRAM, compute capability 12.0 (Blackwell, sm_120),
driver 592.01, **95 W enforced power limit**. Intel Core Ultra 9 275HX, 32 GB RAM. Windows 11
native (10.0.26200), Python 3.12.10, PyTorch 2.11.0+cu128. **All benchmarks ran plugged in** — the
power figures are meaningless on battery.

**A caveat that applies to every latency number.** `t0` is the moment OpenCV returns the frame, not
true photon arrival. Sensor exposure and USB transport add an unmeasured constant offset, so every
photon-to-answer latency reported here is a **lower bound**.

**A second caveat, specific to this camera.** Left on auto-exposure it trades frame rate for
exposure time as the room darkens — measured 30 → 19.9 → 10 FPS across a single evening, silently.
All benchmark runs pin exposure (`configs/capture/webcam.yaml`); runs whose frames are too dark to
be worth interpreting are flagged `too_dark` in their metrics file.

---

## 1. The problem, measured (Phase 1)

A VLM call on every frame it can get, `SmolVLM2-500M-Video-Instruct-Q8_0` on llama.cpp CUDA.
Two 60-second runs. Figure: `results/phase1_naive_baseline.png`.

| | live webcam | recorded clip |
|---|---|---|
| capture FPS | 30.40 | 30.28 |
| VLM calls/s | 10.2 | 10.9 |
| **throughput deficit** | **3.0×** | 2.8× |
| frames dropped (policy `drop_oldest`) | 39.8% | 39.0% |
| photon→first-token p50 / p95 | 70 / 92 ms | 69 / 87 ms |
| photon→answer p50 / p95 / p99 | 115 / 138 / 150 ms | 107 / 129 / 136 ms |
| GPU power mean / peak | 65.6 / 103.6 W | 64.6 / 73.3 W |
| **energy per answer** | **6.47 J** | 6.04 J |
| peak VRAM | 1.47 GB | 1.47 GB |

**Per-frame inference is outside the power envelope, not merely slow.** At 6.47 J per answer, a
30 FPS per-frame oracle needs **197 W sustained against a 95 W cap — 2.1× over budget**. Even given
unlimited time, this laptop cannot answer every frame.

This used the *smallest credible* VLM on purpose. A larger model makes the failure trivially true
and easy to dismiss with "use a smaller model"; if even a 500M model cannot keep up, nothing can.

**Per-stage budget (p50).** `capture_read` 32.09 ms (camera-paced) · `vlm_total` 97.66 ms ·
`vlm_encode_jpeg` 0.84 ms · `fast_tier` and `render` ~0.00 ms. The pipeline is genuinely decoupled:
`capture_q` and `answer_q` sit at mean depth 0.00 while `vlm_q` is pinned at its bound.

**There is a ~15 ms scheduling floor on every llama-server call.** Connection pooling removes it
from a bare `GET /health` (15.08 → 0.46 ms p50) but does nothing for completions (15.14 ms fresh
vs 15.58 ms pooled), so it is server-side task scheduling rather than transport.

---

## 2. The fast tier (Phase 2)

Runs on every frame. Figure: `results/phase2_fast_tier.png`.

### Encoder sweep

Latency includes preprocessing. Quality is scored on how far the embedding moves under three kinds
of change: **lighting** (gamma 0.6/1.6, brightness ±40), **motion** (consecutive frames), and
**semantic** (a synthetic opaque object over ~12% of the frame).

| encoder | p50 ms | p95 ms | sem/light | **sem/motion** |
|---|---|---|---|---|
| downsample32 *(control, no network)* | **0.16** | 0.17 | **19.45** | **0.42** ⚠️ |
| **mobilenetv3_small onnx** ← chosen | **3.32** | 4.08 | 12.48 | **4.61** |
| mobilenetv3_small torch fp16 | 5.62 | 6.17 | 12.45 | 4.59 |
| dinov2_vits14 onnx | 4.20 | 4.52 | 15.84 | 2.99 |
| dinov2_vits14 torch fp16 | 5.21 | 5.56 | 15.81 | 2.99 |
| clip_vitb32 onnx | 4.23 | 5.44 | 4.47 | 2.77 |
| clip_vitb32 torch fp16 | 4.72 | 5.88 | 4.46 | 2.77 |

**Both ratios have to be read together.** The control tops `semantic/lighting` at 19.45 purely
because mean-centring and L2-normalising a grayscale thumbnail makes it brightness-invariant *by
construction*. Its `semantic/motion` of **0.42** exposes what it is: it moves further when
something merely moves than when the scene genuinely changes. A single-ratio table would have
selected a motion detector.

**ONNX Runtime beat PyTorch on every candidate** (1.69× MobileNet, 1.24× DINOv2, 1.12× CLIP) with
quality identical to three decimals, as it should be for the same graph.

**The cheapest learned encoder won outright** — no quality-for-speed trade was required.

*Limitation:* 24 frames from one clip, with a synthetic semantic event. Deliberately easy — failing
it means an encoder certainly fails subtle events; passing it does not prove the converse.

### Sustained run

| | |
|---|---|
| achieved FPS (live, 30 s) | **30.65**, 0 frames dropped |
| fast_tier p50 / p95 / p99 | **4.55 / 6.99 / 8.63 ms** |
| unpaced throughput | **211 FPS — 7.0× the requirement** |
| mean GPU power | **8.17 W** (vs 65.6 W answering every frame) |
| peak VRAM | 0.61 GB |

**Watching every frame costs 8.2 W; answering every frame costs 65.6 W and still cannot keep up.**
That 8× gap is the quantitative case for the two-tier architecture.

---

## 3. The slow tier (Phase 3)

Figure: `results/phase3_slow_tier.png`. 8 configurations, 4 model families × 2 quantization levels,
each on its own `llama-server` instance. 20 frames from the clip, fixed prompt, temperature 0.

**TTFT here is frame-in-hand → first token**: JPEG encode, base64, HTTP, SSE framing, vision
encoding and prefill are all inside it. It excludes camera capture, which §1 measured at 32.09 ms
p50 and which photon-to-first-token adds.

### Model × quantization sweep

| model | quant | file | peak VRAM | load | TTFT p50 | **TTFT p95** | tok/s | fidelity F1 | noise floor | damage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B | Q4_K_M | 1.03 GB | 2.93 GB | 1.6 s | 113 ms | **124 ms** | 135 | 0.712 | 0.943 | 0.230 |
| Qwen3-VL-2B | Q8_0 | 1.71 GB | 3.60 GB | 1.9 s | 118 ms | **125 ms** | 120 | — *(reference)* | 0.955 | — |
| Qwen3-VL-4B | Q8_0 | 3.99 GB | 6.04 GB | 3.0 s | 159 ms | **163 ms** | 64 | — *(reference)* | 0.991 | — |
| Qwen3-VL-4B | Q4_K_M | 2.33 GB | 4.38 GB | 2.1 s | 156 ms | **166 ms** | 80 | 0.849 | 0.992 | 0.143 |
| SmolVLM2-2.2B | Q8_0 | 1.80 GB | 3.78 GB | 1.6 s | 80 ms | **86 ms** | 130 | — *(reference)* | 0.895 | — |
| SmolVLM2-2.2B | Q4_K_M | 1.04 GB | 3.06 GB | 1.3 s | 78 ms | **88 ms** | 144 | 0.736 | 0.944 | 0.208 |
| SmolVLM2-500M | f16 | 0.76 GB | 1.82 GB | 1.1 s | 54 ms | **61 ms** | 222 | — *(reference)* | 1.000 | — |
| SmolVLM2-500M | Q8_0 | 0.41 GB | 1.32 GB | 0.8 s | 52 ms | **61 ms** | 255 | 0.827 | 0.921 | 0.094 |

**All 8 configurations meet the p95 TTFT < 400 ms target**, the slowest by a factor of 2.4. Latency
therefore does not decide the choice — quality does.

### ⚠️ llama-server is not deterministic at temperature 0

This has to be stated before any quality number is read. Asked the **same frame twice, back to
back, with identical cache state**, Qwen3-VL-2B Q4_K_M returned a different string **67% of the
time**. Disabling the prompt cache (`--cache-reuse 0`) changed nothing, and interleaving a
different frame between the two asks changed nothing either — all three arms scored 0.33 exact
match. So this is **floating-point non-determinism in the kernels**, not cache-state sensitivity.

The differences are pure paraphrase — *"looking thoughtfully toward"* vs *"looking thoughtfully
at"* — which is what tiny FP differences flipping a near-tie in greedy argmax look like.

**Consequence: a fidelity score is meaningless without a noise floor.** The *noise floor* column is
each configuration scored against **its own** answers on a second pass over the same frames. The
*damage* column is `noise floor − fidelity`: what quantization actually cost, over and above the
model disagreeing with itself. Q4_K_M damage runs 0.143–0.230 and is smallest on the largest model,
which is the expected pattern.

`exact_match_rate` is recorded but should not be used as a quality metric on this serving path — it
measures FP noise.

### KV-cache reuse and streaming decode

Same model (Qwen3-VL-4B Q8_0), same frames, **every call on a distinct frame** — a real stream never
re-sends a frame, and an earlier version of this benchmark that repeated frames gave every arm a
full-prompt cache hit *including the image*, masking the effect entirely and putting TTFT at 70 ms
instead of the true 160 ms.

| arm | prompt order | streaming | TTFT p50 | TTFT p95 | complete answer p50 | vs arm A |
| --- | --- | --- | --- | --- | --- | --- |
| A image-first, no reuse | image first | yes | 160 ms | 170 ms | 431 ms | — |
| B text-first, slot cache | text first | yes | 154 ms | 166 ms | 443 ms | 1.04× |
| C text-first, `--cache-reuse 256` | text first | yes | **150 ms** | 163 ms | 463 ms | **1.06×** |
| D text-first, non-streaming | text first | no | *n/a — no first token* | *n/a* | 465 ms | — |

**KV-cache reuse buys 6%, and that is a finding rather than a disappointment.** Image tokens differ
on every frame and dominate the prefill, so the cacheable text prefix — even a substantial 330-char
system prompt — is a small fraction of the prompt. Putting the image first makes reuse structurally
impossible; putting the text first makes it possible but not worth much.

**Streaming decode buys 3.1×**: 150 ms to the first token versus 463 ms for the complete answer.

Together these say something that shapes Phase 4: **optimising inside a call is close to pointless
here. The entire win has to come from not making the call.** That is exactly what the scheduler is.

### Chosen configuration — Qwen3-VL-4B-Instruct Q8_0

Defended against the 12 GB constraint and the 400 ms target:

- **Quality decides, because latency does not.** All 8 configurations pass the TTFT target.
  SmolVLM2-500M hallucinates the setting — it described this indoor room as *"a gymnasium"* —
  and SmolVLM2-2.2B is accurate but vague. Both Qwen3-VL models produce specific, correct
  descriptions (*"a white earbud"*, *"a softly blurred indoor setting"*).
- **Q8_0 over Q4_K_M**: Q4_K_M costs 0.143 content-F1 against the Q8_0 reference. That sits well
  below the 0.992 noise floor, so it is real damage, not serving noise.
- **VRAM budget**: 6.04 GB VLM + 0.61 GB fast tier = **6.65 GB of 11.94 GB**, leaving ~5.3 GB for
  the Phase 5 cache, the Phase 7 GUI and the display.
- **Fallback if that headroom is ever needed**: Qwen3-VL-4B Q4_K_M at 4.38 GB for 0.143 F1.

### The target, measured through the full pipeline

The table above measures frame-in-hand. The exit criterion is **photon-to-first-token**, so it is
asserted on a real pipeline run of the chosen config (`results/phase3_chosen_pipeline.json`,
60 s over the clip, 128 VLM calls):

| | |
|---|---|
| **photon→first-token p50 / p95 / p99** | **175 / 195 / 201 ms** ✅ (target < 400 ms) |
| photon→answer p50 / p95 | 487 / 551 ms |
| capture FPS | 30.29, 48.1% of frames dropped by policy |
| VLM calls | 128.7/min = 2.1/s |
| mean GPU power | 92.0 W against a 95 W cap |
| **energy per answer** | **43.8 J** |
| peak VRAM | 5.90 GB |

**p95 photon-to-first-token is 195 ms — the target is met with better than 2× margin.**

Note what the energy column now says. At 43.8 J per answer, this model at 30 FPS would need
**1,315 W**. The Phase 1 figure of 197 W was with the *smallest* VLM; with a model actually worth
deploying, per-frame inference is **14× outside** the power envelope rather than 2×.

### Quality reference: transformers + bitsandbytes

`results/phase3_hf_reference.json`, Qwen3-VL-4B-Instruct via bitsandbytes NF4 4-bit, same 20 frames
and prompt. **This run reports no latency at all** — every latency family in its metrics file is
`null` on purpose.

The GGUF path agreed with it at content-F1 **0.717**, against the GGUF path's own noise floor of
0.991.

**That gap is not attributable to llama.cpp.** bitsandbytes is 4-bit by construction, so the
reference is NF4 while the chosen GGUF config is Q8_0: the comparison varies quantization *and*
serving path together and cannot separate them. Inspecting the pairs, both paths describe the same
scene correctly and differ in wording, which a bag-of-content-words score penalises:

> **HF NF4:** "A man with glasses and a mustache smiles while touching his cheek, with another
> person blurred in the background."
> **GGUF Q8_0:** "A young man with glasses smiles while resting his chin on his hand, with another
> person blurred in the background."

Read it as a sanity check that the llama.cpp path is not degenerate, not as a measurement of its
loss.

### Limitations

- Answer quality is **fidelity, not correctness**. There are no ground-truth annotations for these
  clips until Phase 4, so nothing here says which model is *right* — only how far each drifts from
  its family's highest-precision variant, and how much of that drift is serving noise.
- Cross-family quality ranking rests on the sweep's self-consistency numbers plus qualitative
  inspection of 20 frames from **one clip of one scene**. The "gymnasium" hallucination is
  illustrative, not a metric.
- A GPU shared with another process silently invalidates all of this: an unrelated job holding
  ~9.5 GB produced a complete, plausible-looking sweep in which every configuration reported
  ~11.8 GB peak VRAM and ~3× inflated TTFT. `vlm_bench` now refuses to start when the GPU is not
  idle (`max_foreign_vram_gb`).

---

## 4. The scheduler (Phase 4)

Figure: `results/phase4_pareto.png`. Six 24-second annotated clips (720 frames each), five policy
families across nine operating points each, replayed against a **true per-frame oracle** — 4,320
VLM calls, cached so the ~270-run sweep is an exact lookup rather than days of GPU time.

**The sweep measures accuracy and call rate only. It never measures latency** — that was done end
to end on the real pipeline in §3. Mixing the two would let a cached replay masquerade as a timing
result.

### The clips, and why they are synthesised

| clip | states | semantic events | purpose |
|---|---|---|---|
| `static` | 1 | 0 | control |
| `object_events` | 4 | 3 | object appears, changes, vanishes |
| `lighting_drift` | 1 | 0 | **false-trigger probe** — gamma ramps 1.0 → 0.45 → 1.0 |
| `rapid_motion` | 1 | 0 | **false-trigger probe** — continuous shake and pan |
| `mixed` | 3 | 2 | drift **with** real events — the discriminating case |
| `scene_cuts` | 3 | 2 | hard cuts, an upper bound on detectability |

Events are composited onto real camera footage. That is deliberate: the headline failure mode is a
false trigger where **nothing** semantic happened, and certainty about a negative is exactly what
hand-annotating real footage cannot provide. The cost is that a pasted object is an *easier* event
than a subtle real one — Phase 6 adds real annotated data.

### ⚠️ The obvious accuracy metric measures the wrong thing

The natural metric — text agreement between the answer being held and the oracle's answer for the
current frame — **does not work here**, and finding out why changed the whole analysis.

Because llama-server is not deterministic (§3), the oracle disagrees *with itself*. Measured on
these clips, agreement between the oracle's answers on two **adjacent frames in the same scene
state** — where nothing changed and every difference is serving noise — is **0.783**. That is the
ceiling: a policy calling on every frame but one cannot score higher. And the score keeps decaying
with staleness (0.783 at 1 frame, 0.635 at 15, 0.583 at 30) *even when no event was missed*.

So text agreement largely measures **staleness, not correctness**, and against a ceiling of 0.783
rather than 1.0. Under it, fixed-interval appeared to beat every content-aware policy — an artifact
of frequent calling keeping text fresh, not of better decisions.

**The primary metric is therefore answer validity**: the fraction of frames on which the answer
being held describes the scene state the camera is actually in. It is immune to paraphrase, and
1.0 for the oracle by construction. Text agreement is still reported, as a secondary number
against its measured ceiling.

### Exit criterion — met, on held-out clips

Held out from learned-policy training and from tuning: `lighting_drift` and `mixed` (one probe, one
event clip, so both failure directions are represented).

| policy | operating point | validity | event recall | calls/min | % of oracle |
|---|---|---|---|---|---|
| **embedding_novelty** | 0.12 | **1.000** | **1.00** | **7.5** | **0.42%** |
| motion_threshold | 0.015 | 1.000 | 1.00 | 10.0 | 0.56% |
| learned | 0.95 | 1.000 | 1.00 | 11.2 | 0.62% |
| fixed_interval | 2 s | 1.000 | 1.00 | 30.0 | 1.67% |
| fixed_interval | 10 s | 0.872 | 1.00 | 7.5 | 0.42% |

**Requirement: ≥85% of oracle accuracy at ≤20% of oracle calls. Achieved: 100% answer validity and
100% event recall at 0.42% of oracle calls.**

At a *matched* budget of 7.5 calls/min, embedding novelty holds a valid answer on **100%** of frames
where fixed interval manages **87.2%**. For perfect validity, fixed interval needs 30 calls/min
against novelty's 7.5 — **4× more expensive for the same result**.

### The simple threshold beats the learned policy

`PROMPT.md` asks for this to be said plainly if it happens, and it happened: **embedding novelty
(7.5 calls/min) beats the learned policy (11.2) and the motion threshold (10.0)** for identical
validity and recall.

The learned policy had almost nothing to learn from — **5 positive examples in 2,876 frames**,
because semantic events are rare by construction. Its fitted weights lean hardest on `scene_change`
(251) and `motion` (98), i.e. **it learned to be a motion detector**, which is exactly why it fires
22 times on the probe clips where nothing happens, against novelty's 13.

A one-parameter threshold on a good embedding beat a learned model, on this data. More training
clips with more events might change that; on what exists, the simple thing won.

### False triggers where nothing happens

On `lighting_drift` — a full gamma ramp down and back, zero semantic change, so **every call after
the first is wasted by construction**:

| policy | calls on the probe |
|---|---|
| **embedding_novelty** | **2** |
| motion_threshold | 3 |
| learned | 4 |
| fixed_interval (2 s) | 12 |

The embedding survives a lighting change that a motion detector cannot, which is the Phase 2
`semantic/motion` result (4.61 vs 0.42) showing up where it matters.

### ❗ What does not generalise — read this before quoting the 4×

**Across all six clips the advantage disappears.** For perfect validity, fixed interval at 2 s
(30 calls/min) is *cheaper* than embedding novelty at its best all-clip setting (45 calls/min). At
the 85% bar, embedding novelty is cheapest (5.8 calls/min) but its event recall collapses to 0.67 —
it misses a third of the events.

The reason is structural: **a single global threshold does not transfer across scenes.** The value
that is perfect on the held-out pair misses subtler events elsewhere; the value that catches
everything elsewhere wastes calls on drift. The exit criterion is defined on held-out clips and is
met there by a wide margin, but the honest summary is:

> A scene-aware threshold beats a timer **when its threshold suits the scene**. Making that
> threshold adapt per scene — rather than being tuned once — is the obvious next step, and this
> sweep is the evidence for why it is needed.

`information_gain` underperformed throughout (validity 0.83, recall 0.50–0.67): the staleness
discount made it too conservative, suppressing calls after a change had already been partly paid
for. It is reported as measured rather than tuned until it looked better.

### Limitations

- **Two held-out clips.** `lighting_drift` has a single scene state, so validity there is trivially
  1.0 for any policy that calls at least once — the discriminating clip is `mixed`. The held-out
  numbers rest on a narrow base and the error bars are correspondingly coarse.
- **Call counts are small** (1–12 per probe clip), so false-trigger rates are coarse fractions.
- **Synthetic events are easy.** Passing here does not demonstrate passing on subtle real events.
- Policies were swept, not tuned per clip; no policy saw its held-out clips during fitting.

---

## 5. The semantic cache (Phase 5) — **recommendation: cut it**

`PROMPT.md` marks this phase droppable and asks for a recommendation if it does not earn its place.
It does not, and the reason is more interesting than the feature would have been.

The cache is embedding-keyed: when the scheduler decides a call is warranted, the current fast-tier
embedding is matched against stored answers, and a close-enough, recent-enough match is served for
free. Its only real opportunity is a scene **returning to a state it held before** — an object put
down and picked up, lights dimming and recovering. The clips contain exactly that.

### The measurement

Phase 4's winning policy (`embedding_novelty` @ 0.12) held fixed, so the only variable is the cache.
Held-out clips:

| cache threshold | calls/min | calls avoided | false-hit rate | answer validity |
|---|---|---|---|---|
| **none (baseline)** | 7.50 | — | — | **1.000** |
| 0.999 – 0.70 | 7.50 | **0.0%** | — | 1.000 |
| 0.60 | 5.00 | 25.0% | **1.00** | 0.660 |
| 0.50 | 3.75 | 50.0% | 0.25 | 0.830 |

**There is no threshold that buys anything without costing something.** Above 0.70 the cache never
fires. Below it, it fires and is wrong: at 0.60 *every single hit* served an answer from the wrong
scene state, and answer validity collapsed from 1.000 to 0.660.

### Why — and this is the useful part

It is not the embedding's fault. As a "same scene state" classifier over random frame pairs the key
is decent: **ROC AUC 0.898** (0.873–0.930 per clip).

The problem is *when* the cache gets consulted. It is only ever asked at the moments the scheduler
decides to call — and the scheduler fires precisely when the frame is **unlike** recent scene state.
At those moments, the best available similarity to anything already cached is:

| clip | lookups | best-match similarity (median) | (max) |
|---|---|---|---|
| lighting_drift | 1 | 0.593 | 0.593 |
| mixed | 3 | 0.672 | 0.676 |
| scene_cuts | 3 | 0.615 | 0.718 |
| static | 1 | 0.600 | 0.600 |
| **overall** | 8 | **0.607** | **0.718** |

Same-state pairs have a p10 of 0.754. **The cache is asked at similarities of ~0.61, entirely below
the band where same-state and different-state pairs even begin to separate.**

> **The scheduler and the cache are competing for the same redundancy, and the scheduler already
> took it.** After Phase 4 cuts invocations to 0.42% of the per-frame oracle, the calls that survive
> are — by construction — the moments the scene genuinely changed. Those are exactly the moments a
> cache cannot serve.

### The verdict

A cache would be worth building **before** a good scheduler, not after. Against a fixed-interval
baseline calling 30×/min on a mostly-static scene there is abundant redundancy to reclaim. Against a
novelty-triggered scheduler already at 7.5 calls/min there is none left, and the residual is a
second correctness-critical component whose failure mode — a confidently wrong answer at zero cost,
which the system cannot detect — is worse than the one it replaces.

**Four solid components beat five with one that does not earn its place.** The code remains in the
tree (`src/peripheral/cache/`) with its tests, because the measurement is the deliverable and
someone should be able to re-run it; it is **not wired into the pipeline**.

### Limitations

- Two held-out clips, and only 8 cache lookups across them — the scheduler's efficiency is exactly
  what makes this hard to measure, and the sample is correspondingly tiny.
- A different key (a scene-state embedding trained for the purpose, rather than a generic
  ImageNet-class encoder) might separate states well enough at trigger time. That is a real
  possibility this experiment does not rule out; it rules out *this* key with *this* scheduler.
- The rolling scene-state summary (`SceneStateSummary`) answers queries with no VLM call and is
  tested, but with the cache cut it has no measured benefit to report either.

---

## 6. Evaluation (Phase 6)

### The replay harness, and the bug it caught

`PROMPT.md` calls this *the most likely place the project silently cheats*. So the no-future-frames
invariant is enforced by **three independent gates**, not one convention
(`src/peripheral/eval/replay.py`):

1. **The decoder never runs ahead.** `read()` calls `clock.sleep_until(due)` *before*
   `cap.read()`. Frame *n* is not withheld from the consumer — it is not decoded. There is nothing
   in memory to reach.
2. **Explicit forward access is refused.** `frame_at(i)` raises `FutureFrameError` when frame *i*
   is not yet due. This method exists *only* so the invariant is attackable, because an untested
   invariant is a hope.
3. **Answers are audited against their evidence.** `QueryTimeline.answer()` refuses any answer whose
   evidence timestamp postdates the query.

**Gate 3 fired on the first real benchmark run, and it was right.** StreamingBench sample 41:

```
FutureFrameError: sample_41_1: evidence t=20.020 > query t=20.000
```

The evaluation loop processed the arriving frame — updating the held evidence to t=20.020 — *before*
answering the query due at t=20.000, handing that query 20 ms of its own future. **Our own clips had
hidden this**: at 30 fps a frame lands exactly on every 2-second query boundary, so `evidence_t ==
query_t` and the check passed on arithmetic luck rather than correctness. A real benchmark's frame
rate did not cooperate.

Both runners were restructured so queries due strictly before a frame's arrival are answered from
the previously held frame, and
`test_a_query_between_two_frames_must_use_the_earlier_frame` pins it.

**This is the phase working as intended.** The gate caught a real violation in my own code, in a run
that would otherwise have produced a plausible-looking benchmark number.

**Anti-cheat suite: 20 tests.** They include walking all 59 future indices of a 60-frame clip and
asserting each raises; proving the post-hoc audit has teeth by forging a batch reader's frame count
and asserting the audit flips to `False`; and asserting `sleep_until` actually blocks.

### Wall-clock replay of the annotated clips

Four clips, full pipeline, real VLM, queries every 2 s:

| clip | frames | elapsed | clock allowance | within wall clock | queries | future-evidence violations | mean staleness |
|---|---|---|---|---|---|---|---|
| mixed | 720 | 23.973 s | 720.19 | ✅ | 11/11 | **0** | 1.058 s |
| object_events | 720 | 23.972 s | 720.17 | ✅ | 11/11 | **0** | 2.158 s |
| lighting_drift | 720 | 23.975 s | 720.24 | ✅ | 11/11 | **0** | 2.155 s |
| scene_cuts | 720 | 23.971 s | 720.13 | ✅ | 11/11 | **0** | 0.606 s |

Sitting exactly at the clock allowance is what correct pacing looks like; exceeding it is impossible
by Gate 1.

### Ablations

All six clips. **"Matched call budget" is computed from the full system's measured call rate**, not
guessed — an earlier hardcoded 8 s interval quietly handed the baseline 29% *more* calls than the
system it was being compared against.

| arm | validity | event recall | calls/min | false-trigger rate |
|---|---|---|---|---|
| **full_system** (novelty @ 0.12) | **0.872** | 0.667 | **5.8** | 0.667 |
| fixed_interval_matched (10.34 s) | 0.811 | 0.889 | 7.5 | 0.500 |
| no_fast_tier | 0.811 | 0.889 | 7.5 | 0.500 |
| motion_only (0.015) | 0.957 | 0.889 | 35.8 | 0.750 |
| oracle | 1.000 | 1.000 | 1800 | — |

**`no_fast_tier` and `fixed_interval_matched` are identical by construction.** Remove the fast tier
and the scheduler has no input, so it *is* a timer. That is the cleanest available statement of what
the fast tier buys: it is the difference between having a scheduler and not having one.

**motion_only reaches higher validity (0.957) — at 6× the calls.** Pixel differencing works if you
are willing to pay for it, which is the trade the whole project exists to avoid.

Two ablations `PROMPT.md` lists are not run, with reasons:

- **remove the cache** — there is no cache to remove; Phase 5 measured it and cut it (§5). This
  ablation *is* the shipped configuration.
- **remove KV reuse** — measured in Phase 3 as its own before/after (160 → 150 ms p50 TTFT, 6%). KV
  reuse changes latency, not which answer the model produces, so re-running it against accuracy
  would add noise rather than information.

### ❗ Failure analysis — where the oracle gap actually comes from

Every frame whose held answer describes the wrong scene state is attributed to **exactly one** cause:

| cause | invalid frames | share |
|---|---|---|
| **missed events** | **540** | **100.0%** |
| detection lag | 0 | 0.0% |

**All 540 invalid frames are in one clip.** Five of six clips reach validity 1.000. `object_events`
gets 1 call, misses 3 of 3 events, and scores 0.234:

| missed event | t | novelty at event | peak novelty after |
|---|---|---|---|
| object appears | 6.0 s | 0.0628 | **0.1140** |
| object changes colour | 12.0 s | 0.0425 | 0.0425 |
| object removed | 18.0 s | 0.0755 | 0.0755 |

**The threshold is 0.12. The strongest event peaked at 0.1140.** It missed by 0.006.

This is the mechanism behind the Phase 4 "does not generalise" finding, now quantified. Novelty is
measured against a rolling reference that has already absorbed the scene's baseline variability. On
this clip a person is moving throughout, so the reference is already far from any individual frame,
and the *incremental* novelty contributed by an object appearing is small — 0.04 to 0.11, where the
held-out clips needed 0.12 to suppress lighting drift.

**Detection lag contributes nothing.** When the scheduler fires, it fires promptly; the 0.3 s
minimum gap is never the binding constraint. So the fix is **per-scene threshold adaptation, not
faster reaction** — a conclusion the decomposition supports and prose alone could not.

False triggers are rare and cheap: 1 call each on `lighting_drift`, `mixed`, `scene_cuts` and
`static`, 0 on `rapid_motion` and `object_events`.

### External benchmarks

**StreamingBench, Real-Time Visual Understanding split — subset.**

7 samples, 35 questions, **1,352 s of wall-clock replay at 1.0×** (run took 1,360 s). Each question
answered zero-shot from the single frame the scheduler was holding at that timestamp:

| | |
|---|---|
| **accuracy** | **25/35 = 0.714** (random baseline 0.250) |
| unparsed answers | 0 |
| mean / max staleness of the answering frame | 0.594 s / 5.68 s |
| all within wall clock | ✅ |
| future-evidence violations | **0** |

Per sample: 0.60, 0.40, 0.60, 0.80, **1.00**, 0.80, 0.80. Sample 41 — the one that triggered the
Gate 3 violation before the fix — now runs clean.

By task type:

| task | score |
|---|---|
| Causal Reasoning | 3/3 = 1.00 |
| Clips Summarize | 1/1 = 1.00 |
| Object Perception | 8/10 = 0.80 |
| Text-Rich Understanding | 4/5 = 0.80 |
| Action Perception | 2/3 = 0.67 |
| Attribute Perception | 6/9 = 0.67 |
| Event Understanding | 1/2 = 0.50 |
| Prospective Reasoning | 0/1 = 0.00 |
| Spatial Understanding | 0/1 = 0.00 |

The two zeros are single questions each — noise, not a finding. The pattern that is plausible is
that single-frame questions (Object Perception, Text-Rich) score well and temporally-extended ones
(Event Understanding, Prospective Reasoning) do not, which is what answering from **one** held frame
would predict.

#### ⚠️ The replay ran behind, and that matters

The audit confirms we never ran *ahead* — but it also shows we ran *late*:

| sample | frames | elapsed | frames late (>50 ms) | max lateness |
|---|---|---|---|---|
| 9 | 2,251 | 75.4 s | 75 (3.3%) | 618 ms |
| 1 | 3,201 | 129.2 s | 152 (4.7%) | 1,247 ms |
| 3 | 4,276 | 172.3 s | 161 (3.8%) | 1,188 ms |
| 4 | 5,501 | 221.2 s | 160 (2.9%) | 1,521 ms |
| 41 | 5,852 | 245.3 s | 160 (2.7%) | 1,587 ms |
| 8 | 15,346 | 257.3 s | 829 (5.4%) | 1,203 ms |
| 23 | 6,187 | 259.3 s | 159 (2.6%) | 1,268 ms |

The cause is structural: this runner is **single-threaded on purpose**, so program order is
verifiable line by line — which is what makes the timing auditable. But that means a blocking VLM
call (160–500 ms) delays the next frame read, and the replay falls behind by up to 1.6 s.

**This biases the result downward, not upward.** Running late means the scheduler saw an older frame
than a non-blocking implementation would have provided; it never gave the system information it
should not have had. So 0.714 is a **lower bound** on what the threaded pipeline (Phase 1, where the
capture thread never blocks on inference — invariant 7) would achieve. It is reported rather than
corrected, because correcting it would mean giving up the auditability that is the point of this
harness.

**OVO-Bench: not run, and it is not obtainable here.** The dataset is 199.6 GB published as a single
tar split across 22 parts of 10.74 GB. A split tar cannot be partially extracted — every part is
required — against 130 GB free on this machine. There is no honest partial-subset route, so it is
reported as not done rather than approximated.

For StreamingBench the constraint is time rather than disk. The full RTVU split is ~110 GB across 10
shards; one shard (samples 1–50, 9.1 GB) was fetched. Covering all 250 questions in that shard needs
**377 minutes of wall-clock replay**, because replay runs at 1.0× and the clips are minutes long.
That is the real cost of streaming evaluation and it is not negotiated away — instead a subset is
selected to fit a stated budget, shortest clips first, and reported as a subset.

**The selection bias favours us**: shorter clips give a scheduler less time to drift out of date.

### Limitations of this phase

- StreamingBench is a **subset of a subset**: 7 of 500 samples, from 1 of 10 shards, chosen by clip
  length. It is not a StreamingBench score and must never be quoted as one.
- Our model answers each question zero-shot from **one scheduler-selected frame**. Published
  StreamingBench numbers come from models given the whole clip. The comparison measures our
  streaming system, not the model's ceiling.
- OVO-Bench is absent entirely.
- The annotated clips remain synthetic events on real footage (§4 limitations).
- The single-threaded replay runner falls behind by up to 1.6 s during VLM calls. This depresses the
  StreamingBench number rather than inflating it, and the threaded pipeline does not have the
  problem — but the two are therefore not identical systems, and the benchmark number belongs to the
  auditable one.

---

## 7. What this project established, and what it did not

**Established, with measurements:**

- Per-frame VLM inference on this laptop is outside the **power** envelope, not merely the time
  budget: 43.8 J per answer means a 30 FPS oracle needs 1,315 W against a 95 W cap (§1, §3).
- A two-tier split works: the fast tier watches every frame for **8.2 W** and 4.55 ms, where
  answering every frame costs 65.6 W and still cannot keep up (§2).
- A quality-grade VLM meets an interactive latency target on consumer hardware: **p95
  photon-to-first-token 195 ms** against a 400 ms target, at 6.04 GB of 11.94 GB (§3).
- On held-out clips a novelty-triggered scheduler holds a valid answer on **100%** of frames at
  **0.42%** of the per-frame oracle's calls — 4× cheaper than a timer for the same result (§4).
- A semantic cache does **not** earn its place once a good scheduler exists, and the reason is
  structural rather than incidental (§5).
- The evaluation harness catches its own violations: Gate 3 found a real 20 ms future-evidence bug in
  our code (§6).

**Not established:**

- **The headline claim does not generalise.** "~90% of oracle at ~10% of invocations" holds on
  held-out clips and fails across all six: a single global novelty threshold does not transfer
  between scenes. The failure is quantified — 100% of the oracle gap is missed events, and the
  strongest missed event peaked at 0.1140 against a 0.12 threshold (§6).
- **Per-scene threshold adaptation is the open problem**, and this work is the evidence for why it is
  needed rather than a solution to it.
- **No claim about real semantic events.** Every scheduler number rests on synthetic events
  composited onto one desk scene. The external benchmark result is a 7-sample subset.
- **No claim of architectural novelty.** Dispider already decomposed perception/decision/reaction
  (§ prior art in STATUS.md). The contribution here is the constraint and the measurement.

---

## Reproducing

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Every figure is regenerated from its metrics JSON; no number in this document is typed by hand into
a chart. Run any phase's CLI with `--duration N --headless --metrics-out path.json`.
