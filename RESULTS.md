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
families across nine operating points, replayed against a **true per-frame oracle** — 4,320 VLM
calls, cached so the ~270-run sweep is an exact lookup rather than days of GPU time.

**The sweep measures accuracy and call rate only, never latency** — latency was measured end to end
on the real pipeline in §3.

> ### ⚠️ These numbers were re-derived after a timing bug was found in Phase 7
>
> The fast tier's rolling reference has a half-life **in seconds**, and it was keyed off
> `t_capture` — the moment *we read* the frame. During trace building the oracle's VLM call took
> ~500 ms per frame, so consecutive frames appeared 500 ms apart when they were 33 ms apart in the
> video. The reference therefore tracked ~15× faster than the stream, and **novelty came out 3–4×
> too small** (median 0.016 where it should have been 0.057).
>
> The fix separates `t_presentation` (where the frame sits in the stream) from `t_capture` (when we
> got it), and is pinned by `test_novelty_does_not_depend_on_how_fast_frames_are_read`. Every
> trace-derived number in §4, §5 and §6 was recomputed. **Two conclusions reversed** — see §5 and
> the ranking below. The oracle's answers were unaffected (they depend on the frame, not on timing),
> so only the signals were recomputed rather than re-running 4,320 VLM calls.

### The clips

| clip | states | semantic events | purpose |
|---|---|---|---|
| `static` | 1 | 0 | control |
| `object_events` | 4 | 3 | object appears, changes, vanishes |
| `lighting_drift` | 1 | 0 | **false-trigger probe** — gamma ramps 1.0 → 0.45 → 1.0 |
| `rapid_motion` | 1 | 0 | **false-trigger probe** — continuous shake and pan |
| `mixed` | 3 | 2 | drift **with** real events — the discriminating case |
| `scene_cuts` | 3 | 2 | hard cuts, an upper bound on detectability |

Events are composited onto real camera footage, because the headline failure mode is a false trigger
where **nothing** semantic happened, and certainty about a negative is what hand-annotating real
footage cannot give. The cost: a pasted object is an *easier* event than a subtle real one.

### The metric

The obvious metric — text agreement with the oracle — **does not work here**. Because llama-server
is not deterministic (§3), the oracle disagrees with itself: agreement between its answers on two
adjacent frames in the same scene state is **0.783**, not 1.0. That is the ceiling, and the score
keeps decaying with staleness (0.635 at 15 frames) *even when no event was missed*. It measures
staleness, not correctness.

The primary metric is therefore **answer validity**: the fraction of frames on which the answer
being held describes the scene state the camera is actually in. Paraphrase-immune, and 1.0 for the
oracle by construction.

### Exit criterion — met

Held out from training and tuning: `lighting_drift` and `mixed`. Cheapest operating point reaching
**perfect** validity:

| policy | operating point | validity | event recall | calls/min | % of oracle |
|---|---|---|---|---|---|
| **motion_threshold** | 0.015 | 1.000 | 1.00 | **10.0** | **0.56%** |
| learned | 0.95 | 1.000 | 1.00 | 13.8 | 0.76% |
| fixed_interval | 2 s | 1.000 | 1.00 | 30.0 | 1.67% |
| embedding_novelty | 0.12 | 1.000 | 1.00 | 45.0 | 2.50% |

**Requirement: ≥85% of oracle accuracy at ≤20% of oracle calls. Achieved: 100% validity and 100%
event recall at 0.56% of oracle calls.** The criterion passes with two orders of magnitude to spare.

### ❗ But the ranking is not what the project assumed

With corrected signals, **`embedding_novelty` is the *worst* of the content-aware policies** — it
needs 45 calls/min for perfect validity where a plain motion threshold needs 10. And at the 85% bar,
plain `fixed_interval` at 10 s is cheapest of all (0.872 validity at 7.5 calls/min, 0.42%).

Across all six clips the picture is worse still: for perfect validity `fixed_interval` at 2 s
(30 calls/min) is **cheaper than every content-aware policy**, and the best content result is
`learned` at 0.991 validity for 32.1 calls/min.

**The honest summary: on this data, no content-aware scheduler convincingly beats a timer at a
matched budget.** Different policies win in different regions and no single one dominates. The
embedding-novelty threshold that the earlier (buggy) numbers selected is not the right choice.

The learned policy remains poorly supported — **5 positive examples in 2,876 frames** — and its
weights lean on `scene_change` (263) and `motion` (48), i.e. it learned to be a motion detector,
which is consistent with `motion_threshold` performing well.

---

## 5. The semantic cache (Phase 5) — **recommendation: KEEP**

> **This verdict reversed.** Before the timing fix, the scheduler fired so rarely (an artifact of the
> 3–4× under-scaled novelty) that a cache had nothing left to reclaim, and the recommendation was to
> cut it. With correct signals the scheduler fires ~6× more often, and the redundancy is there.

Phase 4's policy held fixed, so the only variable is the cache. Held-out clips:

| cache threshold | calls/min | calls avoided | false-hit rate | answer validity |
|---|---|---|---|---|
| **none (baseline)** | 45.00 | — | — | **1.000** |
| 0.95 | 42.50 | 5.0% | 0.000 | 1.000 |
| 0.90 | 28.75 | 36.2% | 0.000 | 1.000 |
| **0.85** | **21.25** | **52.5%** | **0.000** | **1.000** |
| 0.80 | 15.00 | 66.2% | 0.036 | 0.994 |
| 0.75 | 11.25 | 74.4% | 0.094 | 0.830 |
| 0.50 | 5.00 | 88.8% | 0.139 | 0.816 |

**At 0.85 the cache removes 52.5% of VLM calls with zero false hits and no loss of validity.** There
is a clean knee: above 0.85 it is free but does less; below 0.80 false hits appear and validity
falls off a cliff.

The key is a decent one — ROC AUC **0.898** as a same-scene-state classifier over random frame
pairs — and now that the scheduler leaves work on the table, that is enough.

**Recommendation: keep, at threshold 0.85 or 0.90.** 0.90 is the conservative choice: 36.2% of calls
removed, still zero false hits, more margin against a scene that merely looks similar.

*The failure mode to respect:* a false hit is a confidently wrong answer at zero cost that the system
cannot detect. That is why the recommendation sits at the zero-false-hit end of the curve rather
than at the maximum-savings end.

---

## 6. Evaluation (Phase 6)

### The replay harness, and the two bugs it caught

The no-future-frames invariant is enforced by **three independent gates**
(`src/peripheral/eval/replay.py`):

1. **The decoder never runs ahead.** `read()` calls `clock.sleep_until(due)` *before* `cap.read()`,
   so a future frame is not withheld — it is not decoded.
2. **Explicit forward access is refused.** `frame_at(i)` raises `FutureFrameError`. It exists so the
   invariant is attackable, because an untested invariant is a hope.
3. **Answers are audited against their evidence.** `QueryTimeline.answer()` refuses any answer whose
   evidence timestamp postdates the query.

**Gate 3 fired on the first real benchmark run.** StreamingBench sample 41:

```
FutureFrameError: sample_41_1: evidence t=20.020 > query t=20.000
```

The loop processed the arriving frame — updating held evidence to 20.020 — *before* answering the
query due at 20.000. Our own clips had hidden it: at 30 fps a frame lands exactly on every 2-second
query, so `evidence_t == query_t` and the check passed on **arithmetic luck, not correctness**.

That is one of two timing bugs this project shipped and then caught. The other is the
`t_presentation` bug in §4. Both were invisible in normal operation and both changed reported
numbers.

**Anti-cheat suite: 20 tests**, including walking all 59 future indices of a 60-frame clip, and
forging a batch reader's frame count to prove the audit flips to `False`.

### Wall-clock replay

| clip | frames | elapsed | clock allowance | within wall clock | queries | violations | mean staleness |
|---|---|---|---|---|---|---|---|
| mixed | 720 | 23.974 s | 720.21 | ✅ | 11/11 | **0** | 1.824 s |
| object_events | 720 | 23.975 s | 720.26 | ✅ | 11/11 | **0** | 3.067 s |
| lighting_drift | 720 | 23.973 s | 720.18 | ✅ | 11/11 | **0** | 2.470 s |
| scene_cuts | 720 | 23.972 s | 720.16 | ✅ | 11/11 | **0** | 1.197 s |

### Ablations

Budget matched by computing the interval from the full system's **measured** call rate.

| arm | validity | event recall | calls/min |
|---|---|---|---|
| full_system (novelty @ 0.12) | 0.941 | 0.889 | 42.5 |
| fixed_interval_matched | **0.955** | **1.000** | 42.5 |
| no_fast_tier | **0.955** | **1.000** | 42.5 |
| motion_only | **0.957** | 0.889 | **35.8** |
| oracle | 1.000 | 1.000 | 1800 |

**`no_fast_tier` and `fixed_interval_matched` are identical by construction** — remove the fast tier
and the scheduler has no input, so it *is* a timer.

**And at a matched budget the timer beats the embedding-novelty scheduler** (0.955 vs 0.941). This
is the same conclusion as §4, reached independently: on this data the fast tier's embedding signal
is not buying what the project assumed it would.

Two ablations are not run, with reasons: **remove the cache** — Phase 5 now recommends keeping it, so
the no-cache column *is* the baseline in §5. **remove KV reuse** — measured in §3 as its own
before/after (6% TTFT); it changes latency, not which answer is produced.

### Failure analysis — where the oracle gap comes from

| cause | invalid frames | share |
|---|---|---|
| **missed events** | **250** | **100.0%** |
| detection lag | 0 | 0.0% |

Five of six clips reach validity 1.000. All 250 invalid frames are in `object_events`, which now
misses **1** of 3 events (it missed all 3 before the timing fix):

| missed event | novelty at event | peak after | threshold |
|---|---|---|---|
| object removed, t=18.0 s | 0.1035 | 0.1035 | 0.12 |

**It misses by 0.017.** Detection lag contributes nothing — when the scheduler fires it fires
promptly, and the 0.3 s minimum gap is never binding. **The fix is per-scene threshold adaptation,
not faster reaction.**

False triggers are now frequent (10–24 per clip) because the corrected signals make the scheduler
fire much more often — which is the same finding as the ablation table from the cost side.

### External benchmarks

**StreamingBench Real-Time Visual Understanding — subset.** 7 samples, 35 questions, **1,352 s of
wall-clock replay at 1.0×**:

| | |
|---|---|
| **accuracy** | **25/35 = 0.714** (random baseline 0.250) |
| unparsed answers | 0 |
| mean / max staleness | 0.594 s / 5.68 s |
| all within wall clock | ✅ |
| future-evidence violations | **0** |

By task: Causal Reasoning 3/3, Clips Summarize 1/1, Object Perception 8/10, Text-Rich 4/5, Action
Perception 2/3, Attribute Perception 6/9, Event Understanding 1/2, Prospective Reasoning 0/1,
Spatial Understanding 0/1. Single-frame tasks score well and temporally-extended ones do not, which
is what answering from **one** held frame predicts.

**The replay ran behind.** The audit confirms it never ran *ahead*, but 2.6–5.4% of frames arrived
late, by up to 1.6 s, because the single-threaded eval runner blocks during VLM calls. That
**depresses** the result — the scheduler saw older frames than a non-blocking implementation would
provide — so 0.714 is a lower bound. Reported rather than corrected, because single-threading is
what makes the timing auditable line by line.

**OVO-Bench: not run, and not obtainable here.** 199.6 GB published as one tar split across 22 parts
of 10.74 GB. A split tar cannot be partially extracted — every part is required — against 130 GB
free. There is no honest partial route, so it is reported as not done rather than approximated.

### Limitations

- StreamingBench is a **subset of a subset**: 7 of 500 samples, from 1 of 10 shards, chosen
  shortest-clip-first to fit a wall-clock budget. **The selection bias favours us.** Never quote it
  as a StreamingBench score.
- Our model answers zero-shot from **one scheduler-selected frame**; published numbers come from
  models given the whole clip.
- OVO-Bench is absent entirely.
- Clips are synthetic events on real footage.

---

## 7. What this project established, and what it did not

**Established, with measurements:**

- Per-frame VLM inference on this laptop is outside the **power** envelope, not merely the time
  budget: 43.8 J per answer means a 30 FPS oracle needs **1,315 W against a 95 W cap** (§1, §3).
- The two-tier split is sound: the fast tier watches every frame for **8.2 W** and 4.55 ms, where
  answering every frame costs 65.6 W and still cannot keep up (§2).
- A quality-grade VLM meets an interactive latency target on consumer hardware: **p95
  photon-to-first-token 195 ms** against a 400 ms target, at 6.04 GB of 11.94 GB (§3).
- Answering rarely is viable: **100% answer validity at 0.56% of the per-frame oracle's calls** on
  held-out clips (§4).
- A semantic cache removes **52.5% of remaining VLM calls with zero false hits** (§5).
- **That a scene-aware scheduler beats a timer — on a static-background camera.** 17–67× fewer VLM
  calls at 100% event recall and ≥0.99 validity, and under lighting drift the learned encoder keeps
  that margin while pixel differencing loses it (§9). Established only in that regime, and on
  synthetic backgrounds — see the five limits listed in §9.
- The evaluation harness catches its own violations — twice (§4, §6).

**Not established:**

- **That a scene-aware scheduler beats a timer on a busy scene.** Where a person is continuously in
  frame, `fixed_interval` is cheaper than every content-aware policy at matched validity — across
  all six Phase 4 clips, and at every sparsity from one event per 19 s to one per 150 s (§8, §9).
  The novelty floor of a moving scene sits above the height of the events. **This is half the
  claim, and it is the half that fails.** §9 has the regime where it holds.
- **That the Phase 8 explanation was right.** It attributed the failure to a timer running at only
  4–6× oversampling and prescribed sparser events. Phase 9 built those clips: oversampling stayed
  pinned at 75× and the timer kept winning. Sparsity was the wrong variable (§9).
- **That the headline claim generalises.** A single global threshold does not transfer between
  scenes; per-scene adaptation is the open problem this work motivates rather than solves.
- **Anything about real semantic events.** Every scheduler number rests on synthetic events
  composited onto one desk scene; the external benchmark is a 7-sample subset.
- **Architectural novelty.** Dispider already decomposed perception/decision/reaction. The
  contribution here is the constraint and the measurement.

**The two bugs are part of the result.** A timing bug in the fast tier under-scaled novelty 3–4×
and inverted the policy ranking; an ordering bug in the eval loop gave one query 20 ms of its own
future. Both survived normal operation and were caught only by tests written specifically to attack
the invariants. That is the argument for building the anti-cheat suite before trusting any number —
including one's own.

---

## 8. Trying to rescue the scheduler thesis (Phase 8) — **the exit criterion failed**

Phase 6 left the project unable to support its own headline: at a matched budget a plain timer
equalled or beat the embedding-novelty scheduler. Phase 8 set out to fix that, with a stated exit
criterion — *a content-aware policy beats `fixed_interval` at a matched budget on **all six**
clips* — and permission to fail.

**It failed.** The gains along the way are real and are reported below, but the headline claim
remains unsupported, and this section exists so that stays visible.

### 8a — Diagnosis: is it the threshold or the signal?

Two explanations were live, needing opposite fixes, so guessing would have been expensive.
`peripheral.cli.signal_diagnosis` measures both, **within each clip**, so per-scene scale cannot act
as a confound:

| clip | AUC (novelty separating event frames from quiet frames) | event median novelty | quiet p95 | separable? |
|---|---|---|---|---|
| scene_cuts | 0.996 | 0.2509 | 0.1465 | ✅ |
| mixed | 0.948 | 0.1642 | 0.1442 | barely |
| **object_events** | **0.696** | **0.0732** | **0.1238** | ❌ **event is below background** |

**Verdict: signal problem.** On `object_events` the events sit *underneath* the scene's own
background novelty, so no threshold exists at any value — confirmed independently by the threshold
sweep, where no point in a 60-value grid reached 0.99 validity on that clip. Adaptive thresholding
would have been built on sand.

The mean AUC of 0.880 hides this completely. Only the per-clip view exposes it.

**Mechanism:** the fast tier pools its embedding over the whole frame. An object appearing in one
corner is a small perturbation next to a person moving through the middle — global pooling averages
the event away against exactly the irrelevant motion the scheduler is supposed to ignore.

### 8b — Spatial (patch) novelty

The fix follows directly from the mechanism: **stop pooling**. Keep the encoder's 7×7 feature map,
give every cell its own rolling reference, and score novelty as the mean over the top-3 most-changed
cells. A corner event then registers at close to full strength.

It costs nothing extra — the pooled vector was already computed *from* this feature map:

| | pooled (Phase 2) | **patch (Phase 8b)** |
|---|---|---|
| latency p50 | 3.32 ms | **2.95 ms** (skips the classifier head) |
| AUC on `object_events` | 0.696 | **0.895** |
| AUC on `mixed` | 0.948 | 0.910 |
| AUC on `scene_cuts` | 0.996 | 0.955 |
| **mean AUC** | 0.880 | **0.920** |

**The signal problem is fixed**, at a small cost on the two clips that were already easy, and the
8a verdict flips to *threshold problem* with 1.68× of adaptation headroom.

### 8b — Adaptive quantile policy

With the verdict now "threshold problem", the threshold was made to adapt: fire when novelty exceeds
a rolling **quantile** of the scene's own recent distribution. That makes it a rate controller —
`q = 0.98` fires on roughly the top 2% of frames whatever the scene's absolute novelty scale is —
which conveniently makes budget-matching automatic, and turns the comparison into exactly the
question the project cares about: *at the same number of calls as a timer, does picking the most
novel frames beat picking evenly spaced ones?*

### 8c — The answer: no

Cheapest operating point reaching validity ≥ 0.99, all six clips:

| policy | signal | calls/min | vs timer |
|---|---|---|---|
| **fixed_interval @ 2 s** | — | **30.0** | — |
| learned @ 0.95 | pooled | 32.1 | 0.94× |
| motion_threshold @ 0.008 | pooled | 55.8 | 0.54× |
| embedding_novelty @ 0.07 | pooled | 103.8 | 0.29× |
| embedding_novelty @ 0.35 | patch | 125.8 | 0.24× |
| adaptive_novelty | either | *never reaches the bar on all six* | — |

**No content-aware policy beats the timer, on either signal.** On the held-out pair several do (up
to 3×), which is the same held-out-vs-all reversal Phase 4 reported. Fixing the signal did not fix
the outcome.

### Why — and this is the useful part

A 2-second timer on these clips is only **4–6× oversampled** relative to the event rate:

| clip | duration | events | seconds/event | timer calls | oversampling |
|---|---|---|---|---|---|
| object_events | 24 s | 3 | 8.0 | 12 | **4.0×** |
| mixed | 24 s | 2 | 12.0 | 12 | 6.0× |
| scene_cuts | 24 s | 2 | 12.0 | 12 | 6.0× |
| static / lighting_drift / rapid_motion | 24 s | 0 | — | 12 | ∞ (all wasted) |

**At 4–6× oversampling a timer physically cannot miss much.** That is the regime where blind
sampling is near-optimal, and no amount of cleverness in frame *selection* can beat it — there is
almost nothing to select. Content-awareness pays when a scene is static for long stretches, and
24-second clips with an event every 8–12 seconds contain almost none of that.

The stasis clips show the effect that *does* exist, on real data. At each policy's cheapest
all-clips setting, calls spent on the three clips where **nothing ever happens**:

| policy | calls on stasis clips | worst event-clip validity |
|---|---|---|
| fixed_interval @ 2 s | 36 | 1.000 |
| **adaptive_novelty q=0.98 (patch)** | **25** | 0.745 |
| motion_threshold @ 0.008 | 95 | 0.984 |
| embedding_novelty @ 0.07 | 115 | 0.967 |

The adaptive patch policy is **the only policy that spends less than a timer on pure stasis** — 31%
fewer calls — and it pays for that by missing events (validity 0.745). Every other content policy
spends *more* than the timer on clips where nothing happens, which is the opposite of the intended
behaviour.

### What Phase 8 actually delivered

1. **A reusable diagnostic** that decides threshold-vs-signal from data instead of intuition, and
   which correctly identified a failure the aggregate AUC hid.
2. **A better fast-tier signal**: patch novelty, +0.20 AUC on the failing clip, at *lower* latency.
3. **An adaptive policy** that is scale-free and is the only one to undercut a timer on stasis.
4. **A quantified reason the thesis cannot be demonstrated on this data** — the 4–6× oversampling
   regime — which is a concrete design requirement for the clips Phase 9 must record.

### What it did not deliver

The claim. `fixed_interval` remains the cheapest way to hold a valid answer across these six clips,
and **the project still cannot show that scene-aware scheduling beats a timer**. Phase 8 narrowed
*why* considerably; it did not change the answer.

### The design requirement this hands to Phase 9

Clips must be **minutes long with sparse events**, not 24 seconds with one every 8–12. At 100×
oversampling a timer must either burn its budget on stasis or miss events, and that is the regime
where frame selection can pay. Until such clips exist, this comparison cannot be settled — and no
amount of policy engineering will settle it.

> **Superseded by Phase 9 (§9).** Those clips were built, and the prescription was wrong. Making
> events sparser does *not* raise the timer's oversampling — the timer's winning interval scales
> with the event rate, so oversampling stayed pinned at 75× across the whole ladder while the
> timer kept winning. The variable that actually decides it is **background stability**: hold the
> background still and content-aware scheduling wins by 17–67× at every sparsity. The diagnosis
> above is measured correctly but reasoned to the wrong cause; §9 has the numbers.

---

## 9. Finding the regime where the scheduler wins (Phase 9) — **the thesis holds, for a reason I had wrong**

Phase 8 failed and offered an explanation: the clips ran a timer at only 4–6× oversampling, and at
that density blind sampling is near-optimal. The fix it prescribed was **longer clips with sparser
events**. Phase 9 built exactly that — an 11-clip ladder, 300 s each, sweeping events per clip from
0 to 16 — and the prescription turned out to be **wrong**.

### The ladder

No VLM was run. `answer_validity`, event recall and false-trigger rate come from scene-state labels
that are exact by construction; only the secondary text-agreement metric needs oracle answers, and
it is reported as `null` rather than fabricated. That is what made an 11-clip sweep affordable at
all — 22 minutes of wall clock instead of hours of VLM decode. Each cell is the **cheapest
operating point of that policy that holds answer validity ≥ 0.99**, swept over its own grid.

Calls per minute, lower is better:

| clip | background | events | s/event | `fixed_interval` | `adaptive` | `embedding` | `motion` |
|---|---|---:|---:|---:|---:|---:|---:|
| `long_static` | live | 0 | — | 1.0 | 11.2 | 0.2 | 0.2 |
| `sparse_2` | live | 2 | 150.0 | **30.0** | none | 48.4 | 186.4 |
| `sparse_4` | live | 4 | 75.0 | **60.0** | none | 101.8 | 186.4 |
| `sparse_8` | live | 8 | 37.5 | 120.0 | none | **100.8** | 186.4 |
| `sparse_16` | live | 16 | 18.8 | 120.0 | none | **102.8** | 186.4 |
| `frozen_long_static` | static | 0 | — | 1.0 | 8.0 | 0.2 | 0.2 |
| `frozen_sparse_2` | static | 2 | 150.0 | 30.0 | 7.4 | 1.6 | **0.6** |
| `frozen_sparse_4` | static | 4 | 75.0 | 60.0 | 7.0 | 3.4 | **1.0** |
| `frozen_sparse_8` | static | 8 | 37.5 | 120.0 | 5.4 | 5.6 | **1.8** |
| `frozen_drift_sparse_4` | static + drift | 4 | 75.0 | 60.0 | 8.2 | **3.4** | 16.6 |
| `frozen_drift_sparse_8` | static + drift | 8 | 37.5 | 120.0 | 14.4 | **5.6** | 16.0 |

Every row holds event recall **1.000**. `none` means no operating point on that policy's grid met
the validity bar at all.

### Sparsity was not the variable

The prescription said: go sparser and the timer must start wasting calls. It does not. On the
live-background rows the timer wins at 150 s/event exactly as it won at 12 s/event — and the
oversampling factor Phase 8 blamed is **75× on nearly every row of this ladder**, not 4–6×. The
timer's winning interval simply scales with the event rate, so oversampling stays pinned while
sparsity sweeps across an order of magnitude. Phase 8's diagnosis was measured on the right data
and drew the wrong variable out of it.

### Background stability was

The same recipes, over a frozen background instead of live footage, reverse the result completely:

| clip | timer | best content policy | speedup |
|---|---:|---|---:|
| `frozen_sparse_2` | 30.0 | `motion` @ 0.6 | **50.0×** |
| `frozen_sparse_4` | 60.0 | `motion` @ 1.0 | **60.0×** |
| `frozen_sparse_8` | 120.0 | `motion` @ 1.8 | **66.7×** |

**The limiting variable is background stability, not event sparsity.** Every clip from Phase 4
onward was composited over desk footage with a person continuously in frame — the worst case for
any novelty signal, because the novelty floor sits above the height of the events. That single
choice, made in Phase 4 for realism, is what suppressed the result for five phases.

This is the regime the claim was always about: a camera that mostly watches an unchanging scene. It
is also the regime the energy argument matters in, and there the scheduler is **50–67× cheaper than
a timer at identical validity and identical (perfect) recall**.

### What the embedding is for

On a frozen background plain pixel differencing is not merely competitive, it is *optimal*:
`motion` holds validity at a **false-trigger rate of 0.000** — every call it makes is a real event.
A 2.95 ms learned encoder cannot beat that, and reporting the ladder without saying so would be
selling the fast tier on a case that does not need it.

So the last two rows add the one thing a frozen scene lacks: a slow gamma drift, two cycles across
the clip. Nothing else changes.

| clip | `motion` | `embedding` |
|---|---:|---:|
| `frozen_sparse_4` | 1.0 | 3.4 |
| `frozen_drift_sparse_4` | **16.6** (16.6× worse) | **3.4** (unchanged) |
| `frozen_sparse_8` | 1.8 | 5.6 |
| `frozen_drift_sparse_8` | **16.0** (8.9× worse) | **5.6** (unchanged) |

The embedding's cost, operating point (0.22708) and false-trigger rate are *identical* with and
without the drift — the signal barely notices. The motion detector's false-trigger rate goes
0.000 → 0.951 and it burns 16× the calls chasing brightness. This is Phase 2's `semantic/motion`
ratio of 0.42 reproduced at clip scale, and it is the first result in the project where the learned
fast tier earns its 2.95 ms.

The honest summary of the trade: **motion is 3.4× cheaper when the light holds still; the embedding
is 4.9× cheaper when it does not, and never falls apart.** For a camera that runs all day, that is
the argument for the encoder.

### Two self-inflicted bugs, both of which produced clean-looking numbers

**Phase-lock.** The first sparse recipes placed events at evenly spaced times. A fixed-interval
timer whose period divides that spacing aligns with every event exactly, and `sparse_4` reported a
timer cost of **1.0 calls/min** — a 60× better number than the honest one. Nothing errored; the
clip was simply built with a symmetry the timer could exploit and a real camera would never offer.
Event times are randomised now, and the same row reports **60.0**.

**A name collision that truncated a run.** `SparsityStudyRunner.setup()` stored the *clip* duration
as `self.duration_s = 300.0`, shadowing `BoundedRunner`'s run deadline. A 3,600 s run ended after
332 s, wrote `status: completed`, and left no note. The deadline now reads from a private
`_run_duration_s` set in `__init__` before any subclass runs, and
`test_a_subclass_cannot_shrink_the_run_deadline_by_shadowing` pins it.

Both belong in the results for the same reason as the `t_presentation` bug in Phase 4: they are the
class of failure this project keeps producing — **wrong numbers that look right**, caught only by
attacking the invariant rather than by reading the output.

### Limits — what this does not show

1. **The static background is synthetic**, a frozen frame plus Gaussian sensor noise. A real fixed
   camera has compression noise, autofocus hunting, and micro-motion this does not model. The
   direction of the result is unlikely to reverse; the 50–67× magnitude is an upper bound.
2. **Events are composited**, not filmed. Recall is 1.000 on every row partly because events are
   crisp by construction.
3. **The zero-event rows (`long_static`) prove nothing.** With no state change, validity is
   satisfied by any policy making a single call, and the timer's 1.0/min is just the coarsest
   interval on its grid (60 s). The 5.0× there is a grid artifact, not a finding — it is in the
   table for completeness and should not be quoted.
4. **A single global threshold still does not transfer between scenes.** The winning operating
   point differs between live and frozen rows. Phase 4's per-scene adaptation problem is untouched;
   `adaptive_novelty` is a rate controller, and it loses to a fixed threshold on every row where
   both run.
5. **No VLM ran.** These are scheduler costs at exact-label validity, not end-to-end accuracy.

### What Phase 9 delivered

The claim, in the regime it was always about: **on a static-background camera, scene-aware
scheduling holds 100% event recall and ≥0.99 answer validity at 17–67× fewer VLM invocations than a
matched timer** — and under lighting drift the learned encoder holds that advantage while pixel
differencing loses it. It also corrects Phase 8's published explanation, which named the wrong
variable.

---

## Reproducing

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Every figure is regenerated from its metrics JSON; no number in this document is typed by hand into
a chart. Run any phase's CLI with `--duration N --headless --metrics-out path.json`.
