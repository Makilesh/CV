# STATUS — where Peripheral is, and where it's going

**Last updated:** 2026-08-08 · **Current phase:** 3 COMPLETE, awaiting confirmation · **Branch:** `phase1`

This file is the single place to look to answer "what is done, what is assumed, what is next."
Update it at every phase boundary. `PROMPT.md` is the plan; `CLAUDE.md` is the operating manual.

---

## 1. What this project is

A system that makes a vision-language model understand a **live video stream at interactive speed
on a single consumer laptop GPU** (RTX 5070 Ti Laptop, 12 GB, Windows 11 native).

**The thesis:** per-frame VLM inference is both impossible (hundreds of ms against a 33 ms budget)
and wasteful (video is overwhelmingly redundant). So the research question is not "make the VLM
fast" — it is **"decide what to send the VLM, when, and reuse everything else."**

```
FAST TIER    every frame, <10 ms, never blocks
             capture → motion/diff → small embedding → novelty + scene-change score
     ↓
SCHEDULER    ← the research contribution
             scene state + novelty + time-since-call + query → cache, or invoke VLM?
     ↓                          ↓
SEMANTIC CACHE            SLOW TIER (VLM)
embedding-keyed answers   quantized small VLM, on demand
+ scene state             KV-cache reuse, streaming decode, TTFT-optimised
+ staleness tracking
```

### The two figures this project exists to produce

1. **Accuracy vs. VLM-calls-per-minute** — our scheduler vs. fixed-interval baselines. *The headline.*
2. **Accuracy vs. p95 photon-to-answer latency** across model × quantization × scheduler.

Everything else is scaffolding for those two plots.

---

## 2. Phase board

| Phase | What it delivers | Exit criterion | State |
|---|---|---|---|
| **0** | Scaffold, Hydra configs, **telemetry**, bounded-runner contract | `pytest tests/test_phase0.py` passes; 5 s run emits valid metrics JSON | **✅ 62 passed · 30.34 FPS · valid JSON** |
| 1 | Async pipeline (threads + bounded queues + backpressure) & naive per-frame VLM baseline | capture ≥25 FPS with VLM stage saturated; `results/phase1_naive_baseline.png` | **✅ 77 passed · 30.4 FPS vs 10.2 calls/s · chart built** |
| 2 | Fast tier: frame diff, small embedding encoder, scene-change + novelty scoring | sustained FPS over 30 s headless, per-stage timings in metrics JSON | **✅ 94 passed · 30.65 FPS live, 211 FPS unpaced · 4.55 ms/frame** |
| 3 | Slow tier: 3–4 small VLMs × GGUF quant levels, KV reuse, streaming decode | comparison table in `RESULTS.md`; **p95 TTFT < 400 ms** test | **✅ 110 passed · 8/8 configs pass · p95 195 ms in-pipeline** |
| 4 | **The scheduler** — pluggable trigger policies, swept against the oracle | `results/phase4_pareto.png`; ≥85% oracle accuracy at ≤20% oracle calls | not started |
| 5 | Semantic cache *(droppable)* | hit rate / staleness / accuracy-cost numbers in `RESULTS.md` | not started |
| 6 | Replay harness + benchmarks + ablations | complete `RESULTS.md`; **no-future-frames test passing** | not started |
| 7 | Ship: GUI demo, CI, README, demo GIF | fresh clone reaches a working live demo | not started |

**Protocol:** phases run strictly in order. Each ends with its test, a reported number, and a full
stop awaiting confirmation. Tag at each boundary (`git tag phase-0-scaffold`).

---

## 3. Verified environment facts

Measured on 2026-08-01 by execution, not by reading docs.

| Item | Value |
|---|---|
| GPU | RTX 5070 Ti Laptop · **11.94 GB** · sm_120 (Blackwell) · 46 SMs · driver 592.01 |
| Sustained fp16 matmul | 61.4 TFLOP/s |
| Power | 4.32 W idle, **95.0 W enforced cap** |
| NVML | power / VRAM / util / temp all return real values → energy-per-query is viable |
| Python / Torch | 3.12.10 · 2.11.0+cu128 (CUDA 12.8) |
| Webcam | 30 FPS ceiling at every mode up to 1080p MJPG — **but only with exposure pinned; see below** |

**Webcam backend choice — `CAP_DSHOW`.** Throughput is identical to MSMF (29.95 vs 29.85 FPS) but
the tail is tighter: p99 **51.1 ms vs 65.1 ms**, max 54 vs 66 ms. Jitter is what hurts a capture
thread. MSMF opens 3.5× faster and reports `CAP_PROP_FPS` correctly (DSHOW returns −1), so FPS comes
from config, not the driver.

**1080p is free.** 1920×1080 MJPG sustains 29.6 FPS — the sensor caps us, not the bus. Resolution
is therefore a quality knob we can spend without a throughput penalty.

### ⚠️ Capture frame rate depends on room lighting — this nearly invalidated every FPS criterion

Found while building Phase 0, not by looking for it. The first webcam smoke run returned **19.2 FPS**
from the same code path that had measured 29.9 FPS ninety minutes earlier. Codec and
`CAP_PROP_FPS` were ruled out by isolation; the cause is **auto-exposure trading frame rate for
exposure time as the room darkens** — silently, with no error and no dropped-frame signal:

| Condition | FPS | Frame brightness |
|---|---|---|
| Auto-exposure, good light (19:40) | 29.9 | 126 / 255 |
| Auto-exposure, dimmer (20:15) | 19.9 | 135 |
| Auto-exposure, dimmer still (20:30) | **10.0** | 162 |
| **Manual, `exposure = -5` (31 ms)** | **30.1, repeatable ±0.1 over 3 trials** | varies with room |
| Manual, `exposure = -4` (62 ms) | 16.0 | — |

`-5` is `log2(seconds)` = 1/32 s = 31 ms — the longest exposure that fits a 33 ms frame budget.
Anything longer cannot sustain 30 FPS, so the driver halves the rate instead.

**Consequences, all load-bearing:**

1. Every FPS-threshold exit criterion (Phase 1 ≥25, Phase 2 ≥30) would otherwise have depended on
   **the time of day**. `configs/capture/webcam.yaml` now pins exposure; a test guards it.
2. **The cost is real:** pinned exposure in a dim room produces near-black frames (measured
   **1.6/255**) while every timing number still looks perfect. `WebcamSource` measures warmup
   brightness, sets `too_dark` in the metrics file, and the runner prints a warning — a dark run
   cannot pass as a result. `CAP_PROP_GAIN` does not compensate; the driver ignores it.
3. **This threatens Phase 4's headline experiment.** The clips with "slow lighting drift and no
   semantic event" are exactly the false-trigger cases. Recorded on auto-exposure, they would have
   *frame-rate drift baked into the clip itself*, confounding the very failure mode being measured.
   **Record all clips with exposure pinned and lighting controlled.**
4. `capture=webcam_demo` exists for the Phase 7 GUI (auto-exposure, usable image, drifting FPS).
   It is not comparable to benchmark runs and says so in its own config.

---

## 4. Prior art and how we differ — the honest version

| Paper | What it does | Hardware |
|---|---|---|
| [Dispider](https://arxiv.org/abs/2501.03218) (CVPR 2025) | Disentangles perception / decision / reaction; decision module proactively triggers responses | n/s |
| [StreamingVLM](https://arxiv.org/abs/2510.09608) | Attention sinks + short vision window + long text window KV cache; training aligned to streaming | **H100, 8 FPS** |
| [LiveVLM](https://arxiv.org/abs/2505.15269) (DAC'26) | Training-free KV compression (Vision Sink Bucketing) + position-agnostic retrieval | n/s |
| [StreamingEval](https://arxiv.org/abs/2603.21493) | Unified evaluation protocol: fixed-capacity memory, encoding efficiency, decode latency, deployability | — |
| [ViCoStream](https://arxiv.org/abs/2606.19849) (Jun 2026) | Stage-wise coordinated inference | **A100, 134 FPS, TTFT <50 ms** |
| [CodecSight](https://arxiv.org/html/2604.06036v3) (Apr 2026) | Codec-guided patch pruning + selective KV refresh; **87% compute cut, 0–8% F1 drop** | n/s |
| [Virtuoso](https://dl.acm.org/doi/full/10.1145/3564289) | Runtime **energy-latency-accuracy Pareto** on SoCs | SoC |

### Where we are genuinely different

1. **Different objective function from Dispider.** Its decision module answers *"should the
   assistant speak now?"* — a helpfulness question. Ours answers *"is spending a VLM invocation
   worth it?"* — a cost question under a hard 12 GB budget. Dispider's perception encoder runs every
   frame regardless; compute reduction is not its target. Invocations/minute as the x-axis is a
   different experiment.
2. **Photon-to-answer latency including capture and pre/post-processing.** None of the seven report
   it. They report model-side TTFT and throughput.
3. **Energy per query on a power-capped device.**
4. **False-trigger rate on lighting-drift / motion-without-semantic-event clips.** I could not find
   this reported anywhere. It is the most defensible novel measurement in the plan and should be
   promoted from a Phase 4 detail to a first-class contribution.

### Where we are weak — state this plainly, do not paper over it

- **Architecturally we are a reimplementation of Dispider.** Our Phase 4 `learned` policy is its
  decision module with less machinery. `PROMPT.md` already concedes this; it is true.
- **"Pareto curves on constrained hardware" is not a new methodology** — Virtuoso did 3D
  energy-latency-accuracy optimization on SoCs. It is only new *for VLMs*.
- **CodecSight already claims 87% compute reduction** — same order as our "~10% of invocations"
  headline, by a different mechanism (intra-call token pruning vs. invocation scheduling). It is
  complementary, not competing, but a reviewer will ask why we didn't compare or stack. It also
  hints at a cheaper fast tier: **motion vectors are already in the compressed bitstream, free.**
- **ViCoStream's <50 ms TTFT on an A100** is 8× better than our Phase 3 target of 400 ms. That gap
  must be presented as *the laptop tax we are measuring*, never as a shortfall.

### The load-bearing risk

**Phase 4 is where this project succeeds or becomes something else.** If `embedding_novelty` beats
`learned` *and* only marginally beats a per-clip-tuned `fixed_interval`, there is no scheduler
result — only a measurement study. That is still a legitimate contribution, but the claim sentence
would need rewriting. Decide that honestly when the numbers land; do not retrofit the claim.

**Recommendation for Phase 6:** adopt StreamingEval's protocol where it fits rather than inventing a
private harness. A self-defined evaluation protocol is exactly what a reviewer discounts.

---

## 5. Windows VLM landscape (Phase 3 input — nothing committed yet)

llama.cpp ships **prebuilt Windows CUDA binaries**; release `b10218` (2026-08-01) includes
`llama-b10218-bin-win-cuda-13.3-x64.zip`. No build toolchain needed.

> **Use the CUDA 13.3 asset, not 12.4.** Our GPU is sm_120; CUDA 12.4 predates Blackwell consumer
> silicon. The 12.4 build might still work via PTX JIT — **verify empirically in Phase 3**, don't assume.

Candidates that fit 12 GB (gating checked live against the HF API — **all ungated**):

| Model | GGUF repo |
|---|---|
| Qwen3-VL 2B / 4B / 8B | `Qwen/Qwen3-VL-*-Instruct-GGUF` |
| Qwen2.5-VL 3B / 7B | `ggml-org/Qwen2.5-VL-*-Instruct-GGUF` |
| Gemma 3 4B · Gemma 4 E2B/E4B | `ggml-org/gemma-*-GGUF` |
| SmolVLM2 2.2B | `ggml-org/SmolVLM2-2.2B-Instruct-GGUF` |
| InternVL3 1B / 2B | `ggml-org/InternVL3-*-GGUF` |
| Moondream2 | `ggml-org/moondream2-20250414-GGUF` |

Excluded as too large for 12 GB: Pixtral 12B, Mistral Small 3.1 24B, Llama 4 Scout, Qwen3-VL-32B.

**Gating:** every GGUF mirror is ungated. Only upstream `google/gemma-3-4b-it` is `gated='manual'`,
which affects **only** Phase 3's transformers + bitsandbytes quality reference → use a Qwen model for
that path and gating never blocks us.

**Integration decision (affects Phase 1):** `llama-cpp-python` publishes no CUDA wheels for Windows
(source build needs MSVC + CUDA toolkit) and its multimodal support lags the C++ side — exactly the
"unofficial thing in the critical path" we're avoiding. **Plan: drive the prebuilt `llama-server`
binary over localhost HTTP**, pinned to a release tag. Gets vision, streaming decode and slot/KV
reuse for free. **Cost: IPC and SSE framing land inside our TTFT budget — measure it in Phase 1,
never assume it's negligible.**

---

## 6. Decisions log

Rejected approaches belong here with their reasons.

| # | Decision | Why | Rejected alternative |
|---|---|---|---|
| D1 | Webcam via `CAP_DSHOW` | Identical FPS to MSMF, but p99 inter-frame 51 ms vs 65 ms. Jitter hurts a capture thread more than mean latency. | `CAP_MSMF` — faster open, correct FPS reporting, worse tail |
| D2 | Drive llama.cpp via prebuilt `llama-server` over HTTP | Official Windows CUDA binaries; vision + streaming + KV slot reuse for free; pinnable to a release tag | `llama-cpp-python` — no CUDA wheels on Windows, lagging mtmd support, source build is a reproducibility hazard |
| D3 | Hydra via `compose()` API behind an argparse front-end | Invariant 4 mandates `--duration/--headless/--metrics-out`; `@hydra.main` hijacks argv and forces `key=value` syntax | `@hydra.main` — would violate the runnable contract |
| D4 | Metrics recorder is event-based, aggregates computed at finalize | Percentiles, rates and energy integration all need the raw series; deriving them live bakes in assumptions we may want to revisit | Live-updating counters — cheaper, but unrecoverable if a definition changes |
| D5 | Unmeasured metrics serialize as `null`, never `0` | A zero must mean "measured, and it was zero". Silent zeros are how a results table lies. | Zero-filling — simpler schema, dishonest output |
| D6 | Phase 0 frame source is deliberately synchronous | The async pipeline is Phase 1's deliverable; Phase 0 only needs *a* source to prove telemetry end-to-end | Building the threaded pipeline now — merges phases, loses the naive baseline |
| D7 | Pin webcam exposure (`auto_exposure=0.25`, `exposure=-5`) for all benchmark runs | Auto-exposure silently trades frame rate for exposure time: measured 30 → 20 → 10 FPS across one evening. Pinned gives 30.1 FPS repeatable ±0.1. | Leaving auto-exposure on — usable image in any light, but FPS depends on the room and no benchmark is reproducible |
| D8 | The measured window closes when the *work* stops, not when cleanup finishes | `cap.release()` on DSHOW costs ~250 ms. Folding it in reported a true 30.2 FPS as 28.7 FPS — a 5% understatement of every rate metric, in our own favour nowhere and against us here, but wrong either way. | Wall-clock from `run()` entry to exit — simpler, and silently wrong |
| D9 | The VLM stage takes the **newest** queued frame and discards the backlog (`drain_newest`) | Answering about a stale frame is strictly worse than answering about the current one, and processing a backlog would hide queue wait inside "processing time" — flattering the latency numbers | FIFO — preserves order, but every answer describes the past and latency becomes unbounded under saturation |
| D10 | Queue sizes stay small (capture 4, VLM 2, answer 8) | A deep queue buys no throughput when the consumer is 3× slower; it only converts *dropped frames* into *stale answers*, moving the damage from a visible metric into an invisible one | Deep queues — smoother-looking drop rate, worse and less honest staleness |
| D11 | **Rejected HTTP connection pooling.** Client keeps plain `urllib`, one connection per call | Measured: pooling fixes a bare `GET /health` (15.08 → 0.46 ms p50) but does **nothing** for completions (15.14 ms fresh vs 15.58 ms pooled). The ~15 ms floor is llama-server's task scheduling, not transport, so pooling adds a moving part for no measured gain | `requests.Session` — better practice in the abstract, zero measured benefit here |
| D12 | Phase 1 baseline uses the **smallest credible** VLM (SmolVLM2-500M), not a representative one | Phase 1 must show per-frame inference cannot keep up. A large model makes that trivially true and easy to dismiss with "use a smaller model". If even the smallest cannot, nothing can. | A 2B–4B model — more representative of final quality, weaker as an argument. Phase 3 does the real sweep |
| D13 | Fast-tier encoder is **MobileNetV3-small via ONNX Runtime CUDA** | Cheapest candidate *and* best on the metric that matters (`semantic/motion` 4.61). 3.32 ms vs DINOv2's 4.20 ms and CLIP's 4.23 ms. The simple thing won outright. | DINOv2-ViT-S/14 — better `semantic/lighting` (15.84 vs 12.48) but 1.5× worse at separating events from movement, and 27% slower |
| D14 | Encoder quality is scored as **lighting vs motion vs semantic embedding displacement**, not ImageNet accuracy or retrieval mAP | A generic benchmark says nothing about our failure mode. Phase 4's headline risk is a false trigger on lighting drift, so the encoder is scored on exactly that discrimination. | Standard embedding benchmarks — comparable to published numbers, irrelevant to the decision being made |
| D15 | **`d_semantic/d_lighting` is never reported alone.** Both ratios always appear together | The mean-centred, L2-normalised control is brightness-invariant *by construction*, so it scores best (19.45) while being a pure motion detector — its `semantic/motion` of 0.42 gives it away. A single-ratio table would have chosen the control. | Reporting the headline ratio only — cleaner table, actively misleading |
| D16 | ONNX Runtime pinned to **1.26.0 (CUDA 12)**, reusing torch's bundled CUDA 12.8 + cuDNN 9 DLLs | `onnxruntime-gpu` 1.28 is built against CUDA 13, whose Windows runtime has no pip route (NVIDIA's `nvidia-*-cu13` wheels are Linux-only). 1.28 fell back to CPU **silently** — sessions succeed and CPU latencies get reported as GPU ones. | Staying on 1.28 with CPU fallback — current version, invalid numbers |
| D17 | `OnnxEncoder` **raises** when a requested GPU provider does not bind | See D16: the failure mode is a plausible-looking wrong number, which is the most dangerous kind. `allow_cpu_fallback=True` is required to measure CPU deliberately. | Warning and continuing — one more silently-wrong benchmark |
| D18 | Slow tier is **Qwen3-VL-4B-Instruct Q8_0** | All 8 configs pass the TTFT target, so quality decides. SmolVLM2-500M hallucinated the setting ("a gymnasium"); SmolVLM2-2.2B is vague. Q4_K_M costs 0.143 F1, well below the 0.992 noise floor, so it is real damage. 6.04 GB + 0.61 GB fast tier = 6.65 of 11.94 GB. | Qwen3-VL-4B Q4_K_M — 1.66 GB cheaper and 25% faster decode, kept as the fallback if VRAM gets tight in Phase 5/7 |
| D19 | **Every fidelity number is reported next to a per-config noise floor.** `exact_match_rate` is never used as a quality metric | llama-server is not deterministic at temperature 0: same frame, back to back, identical cache state, prompt cache disabled → different string 67% of the time. Measured all three arms at 0.33, so it is kernel FP non-determinism, not cache-state sensitivity. A fidelity score without a noise floor cannot separate quantization damage from serving noise. | Reporting fidelity alone — a cleaner table that would have attributed ~0.2 F1 of pure noise to quantization |
| D20 | KV-cache reuse is enabled (text-first + `--cache-reuse 256`) but **reported as a 6% win**, and no further in-call optimisation is pursued | Image tokens differ every frame and dominate prefill, so the cacheable prefix is small: 160 → 150 ms p50. Streaming decode is worth far more (3.1×). Together: optimising inside a call is nearly pointless, and the win must come from not making the call. | Chasing prefill optimisation — Phase 4's scheduler is where the order-of-magnitude is |
| D21 | `vlm_bench` **refuses to start on a GPU that is not idle** | An unrelated process holding ~9.5 GB produced a complete, plausible sweep with every config reporting ~11.8 GB peak VRAM and ~3× inflated TTFT. Peak VRAM is board-wide (that is what the 12 GB limit is), so contention is indistinguishable after the fact and must be caught before the run. | Trusting the operator to check — this already happened once and nothing failed |
| D22 | The transformers + bitsandbytes reference **reports no latency at all**, and its quality gap is **not attributed to llama.cpp** | PROMPT.md forbids latency claims from that path. And bitsandbytes is 4-bit by construction, so the reference is NF4 while our GGUF is Q8_0 — the comparison varies quantization *and* path together and cannot separate them. | Reporting the 0.717 F1 gap as "llama.cpp quality loss" — a claim the experiment does not support |

---

## 7. Open questions — assumptions currently in force

These were raised at the Phase 0 checkpoint and are **being proceeded on as stated**. Overruling any
of them is cheap right now and expensive later.

| # | Question | Assumption in force |
|---|---|---|
| Q1 | Phase 2's exit criterion is "sustained **≥30 FPS**", but the camera's hard ceiling is 29.8–29.9 — the test can never pass on live capture. | Split it: **≥29.5 FPS live** (proves we keep up) **plus ≥30 FPS on a synthetic/file source** (proves headroom, which is the more informative number). |
| Q2 | The claim says **90% of oracle at 10% of calls**; Phase 4's exit test asserts **85% at 20%**. These are different targets. | 85/20 is the **test floor that gates the phase**; 90/10 is the **stated goal**. The README quotes the achieved number, never the aspiration. |
| Q3 | Is "architecture is not novel, measurement is" the accepted positioning? | **Yes**, with false-trigger rate promoted to a headline metric (§4). |
| Q4 | Branch is named `phase1` but we are building Phase 0. | Left as-is — not renaming a branch the user created. Tags carry the real phase boundaries. |

---

## 8. Phase 0 results

`pytest tests/ -q` → **62 passed**, 24 s.

5-second bounded run, real webcam, `results/phase0_smoke.json`:

| | |
|---|---|
| status / exit code | `completed` / 0 |
| duration requested → actual | 5.0 s → **5.0103 s** |
| frames captured / dropped | 152 / 0 (drop rate 0.0) |
| **achieved FPS** | **30.337** |
| inter-frame ms p50 / p95 / p99 / max | 32.03 / 47.84 / 49.05 / 50.57 |
| GPU power mean / peak | 11.65 W / 13.5 W → **60.80 J** over the run |
| peak VRAM (NVML, board-wide) | 0.442 GB |
| frame brightness / `too_dark` | 40.95 / False |
| VLM metric families | all `null` — no VLM ran, and the file says so |

The last row is the point of Phase 0: the smoke runner does not invoke a VLM, so photon-to-answer,
staleness, accuracy and false-trigger all serialize as `null`. Fabricating those events to make the
JSON look complete is exactly the failure this project must not have. Their derivation is proven by
unit tests driving synthetic events with known timestamps.

**What the 62 tests actually assert** — schema enforcement (every metric family in `PROMPT.md` is
required, an `n=0` percentile block is rejected as the shape of a lie); latency measured from the
triggering frame's t0; staleness as the age of *evidence*, not of the answer; false-trigger rate
null without labels; the bounded-runner contract including the crash path; and the two regressions
above (exposure pinned, teardown outside the measured window).

---

## 9. Phase 1 results — the motivating failure, quantified

`pytest tests/ -q` → **77 passed**, 50 s (includes a 20-second real-hardware pipeline run).

Chart: `results/phase1_naive_baseline.png`. Two 60-second runs, `SmolVLM2-500M-Video-Instruct-Q8_0`
on llama.cpp CUDA 13.3, VLM invoked on every frame it can get.

| | live webcam | recorded clip |
|---|---|---|
| capture FPS | **30.40** | 30.28 |
| VLM calls/s | **10.2** | 10.9 |
| **throughput deficit** | **3.0×** | 2.8× |
| frames dropped | 39.8% | 39.0% |
| photon→first-token p50 / p95 | 70 / 92 ms | 69 / 87 ms |
| photon→answer p50 / p95 / p99 | **115 / 138 / 150 ms** | 107 / 129 / 136 ms |
| GPU power mean / peak | 65.6 W / 103.6 W | 64.6 W / 73.3 W |
| energy per answer | **6.47 J** | 6.04 J |
| peak VRAM | 1.47 GB | 1.47 GB |

**The exit criterion holds: capture sustained 30.4 FPS with the VLM stage saturated.** The pipeline
is genuinely decoupled — `capture_q` and `answer_q` sit at mean depth 0.00 while `vlm_q` is pinned
at its bound of 2. The bottleneck is exactly one stage, and the capture thread never felt it.

### Three findings worth more than the chart

1. **Per-frame VLM inference is outside the power envelope, not just the time budget.** At 6.47 J
   per answer, a 30 FPS per-frame oracle would need **197 W sustained** against a **95 W** enforced
   cap — **2.1× over budget**. Even with infinite time, this laptop cannot run per-frame inference.
   That is a stronger motivation than latency alone and it is measured, not argued.
2. **Latency is dominated by the model, not our plumbing.** Per-stage p50: `capture_read` 32.09 ms
   (camera-paced), `vlm_encode_jpeg` 0.84 ms, `fast_tier` and `render` ~0.00 ms, `vlm_total`
   97.66 ms. The scheduler has ~98 ms of VLM cost to avoid and ~1 ms of our own overhead to worry
   about. **The fast tier's entire Phase 2 budget is ~10 ms — it has room.**
3. **There is a ~15 ms scheduling floor on every llama-server call** that no amount of client
   tuning removes (D11). Phase 3's 400 ms p95 TTFT target has to be met *including* it.

### What this does not yet show

The 3.0× deficit is with the **smallest credible** VLM (500M). A model chosen for answer quality
will be far worse — Phase 3 measures how much. The naive baseline also answers a fixed prompt with
no notion of a query, so `accuracy_vs_oracle` and `false_trigger_rate` remain `null`: there is no
oracle to compare against until Phase 4 builds one.

Answer staleness tracks photon-to-answer almost exactly (115.2 vs 115.2 ms) because in the naive
baseline every answer's evidence *is* the frame that triggered it. Staleness only becomes an
independent metric once the cache (Phase 5) starts serving answers from older evidence.

---

## 10. Phase 2 results — the fast tier fits, with 7× room

`pytest tests/ -q` → **94 passed**, 83 s. Chart: `results/phase2_fast_tier.png`.

### Encoder sweep (`results/phase2_encoder_bench.json`)

7 candidates × latency-including-preprocessing × discrimination quality:

| encoder | p50 ms | p95 ms | dim | sem/light | **sem/motion** |
|---|---|---|---|---|---|
| downsample32 *(control, no network)* | **0.16** | 0.17 | 1024 | **19.45** | **0.42** ⚠️ |
| mobilenetv3_small torch fp16 | 5.62 | 6.17 | 1024 | 12.45 | 4.59 |
| **mobilenetv3_small onnx** ← chosen | **3.32** | 4.08 | 1024 | 12.48 | **4.61** |
| dinov2_vits14 torch fp16 | 5.21 | 5.56 | 384 | 15.81 | 2.99 |
| dinov2_vits14 onnx | 4.20 | 4.52 | 384 | 15.84 | 2.99 |
| clip_vitb32 torch fp16 | 4.72 | 5.88 | 768 | 4.46 | 2.77 |
| clip_vitb32 onnx | 4.23 | 5.44 | 768 | 4.47 | 2.77 |

**Read both ratios or you pick the wrong encoder.** The control tops `semantic/lighting` at 19.45 —
purely because mean-centring and L2-normalising a grayscale thumbnail makes it brightness-invariant
by construction. Its `semantic/motion` of **0.42** exposes what it actually is: it moves *more* when
something merely moves than when the scene genuinely changes. That is the motion detector Phase 4
has to beat, and a single-ratio table would have selected it.

**ONNX Runtime beat PyTorch on every candidate** — 1.69× on MobileNet, 1.24× DINOv2, 1.12× CLIP —
with quality identical to 3 decimal places, as it should be for the same graph. `PROMPT.md`'s
preference for ORT is now measured rather than assumed.

**The cheapest learned encoder won outright.** MobileNetV3-small is both the fastest network and the
best at separating events from movement. No quality-for-speed trade had to be made.

### Sustained run (`results/phase2_fasttier_webcam.json`, 30 s live)

| | |
|---|---|
| **achieved FPS** | **30.65**, 0 frames dropped |
| fast_tier p50 / p95 / p99 | **4.55 / 6.99 / 8.63 ms** (budget ~10 ms) |
| **unpaced throughput** | **211 FPS — 7.0× the 30 FPS requirement** |
| mean GPU power | **8.17 W** vs Phase 1's 65.6 W — **8× cheaper** |
| peak VRAM | 0.61 GB |
| novelty p50 / p95 / max | 0.019 / 0.100 / 0.187 |
| motion p50 / p95 / max | 0.0012 / 0.0031 / 0.0087 |

Both readings of the exit criterion hold: **30.65 FPS live** (≥29.5, the camera's ceiling) and
**211 FPS unpaced** (≥30). Per-stage timings are in the metrics JSON as required.

The 8× power gap is the quantitative case for the whole architecture: the fast tier can watch every
frame for 8.2 W, while answering every frame costs 65.6 W and still cannot keep up.

### What this does not yet show

The quality numbers come from **24 frames of one desk clip** with a **synthetic** semantic event
(an opaque textured block over ~12% of the frame). It is a deliberately easy event — an encoder
that fails it certainly fails a subtle one, but passing it does not prove the reverse. Phase 4 needs
real annotated clips with genuine semantic events, and the lighting-drift clips must be recorded
with exposure pinned (§3) or the confound lands inside the very clips meant to expose false triggers.

No threshold has been chosen yet. Phase 2 produces the *signals*; deciding when they mean "invoke
the VLM" is Phase 4, and that is where these numbers get their real test.

---

## 11. Phase 3 results — the target is met with 2× margin

`pytest tests/ -q` → **110 passed**, 121 s. Full write-up in `RESULTS.md` §3; figure
`results/phase3_slow_tier.png`.

**All 8 configurations meet p95 TTFT < 400 ms**, the slowest by 2.4×. In the full pipeline the
chosen config reaches **photon→first-token p95 = 195 ms**.

**Chosen: Qwen3-VL-4B-Instruct Q8_0** — 6.04 GB peak VRAM, 163 ms p95 TTFT, 64 tok/s. See D18.

### The three findings that matter

1. **llama-server is not deterministic at temperature 0** (D19). Same frame, back to back,
   identical cache state, prompt cache disabled → a different string 67% of the time, in paraphrase
   form. Every fidelity number now ships next to the configuration's own noise floor, and the
   *damage* column is the difference. Without it, ~0.2 F1 of pure serving noise would have been
   reported as quantization damage.
2. **In-call optimisation is nearly pointless here** (D20). KV-cache reuse is worth 6%
   (160 → 150 ms) because image tokens change every frame and dominate prefill. Streaming decode is
   worth 3.1×. The order-of-magnitude has to come from **not making the call** — which is Phase 4.
3. **Energy is far worse than Phase 1 suggested.** At 43.8 J per answer for a model actually worth
   deploying, a 30 FPS per-frame oracle would need **1,315 W** against a 95 W cap — **14× outside**
   the envelope, not the 2× measured with the smallest VLM.

### What this does not show

Quality here is **fidelity, not correctness** — no ground-truth annotations exist until Phase 4.
Cross-family ranking rests on self-consistency plus qualitative inspection of 20 frames from one
clip of one scene; the SmolVLM2-500M "gymnasium" hallucination is illustrative, not a metric.

---

## 12. Next action

**Phase 3 is complete and awaiting confirmation.** Do not start Phase 4 until it is given.

Phase 4 is **the core**: pluggable trigger policies behind one interface (`fixed_interval` at
every-frame/1 Hz/0.5 Hz/0.2 Hz, `motion_threshold`, `embedding_novelty`, `information_gain`, and a
small `learned` policy), swept across their operating ranges against a per-frame oracle, producing
`results/phase4_pareto.png` — accuracy vs VLM-calls-per-minute, error bars over repeated runs.

**It needs annotated clips that do not exist yet**, including the lighting-drift and
rapid-motion-without-semantic-event cases where false triggers are the interesting failure. Those
must be recorded with **exposure pinned** (§3) or the confound lands inside the very clips meant to
expose it. That is the first task of Phase 4, and it needs the room set up deliberately.
