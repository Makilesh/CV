# STATUS — where Peripheral is, and where it's going

**Last updated:** 2026-08-03 · **Current phase:** 0 COMPLETE, awaiting confirmation · **Branch:** `phase1`

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
| 1 | Async pipeline (threads + bounded queues + backpressure) & naive per-frame VLM baseline | capture ≥25 FPS with VLM stage saturated; `results/phase1_naive_baseline.png` | not started |
| 2 | Fast tier: frame diff, small embedding encoder, scene-change + novelty scoring | sustained FPS over 30 s headless, per-stage timings in metrics JSON | not started |
| 3 | Slow tier: 3–4 small VLMs × GGUF quant levels, KV reuse, streaming decode | comparison table in `RESULTS.md`; **p95 TTFT < 400 ms** test | not started |
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

## 9. Next action

**Phase 0 is complete and awaiting confirmation.** Do not start Phase 1 until it is given.

Phase 1 is: threaded capture/fast-tier/VLM/render stages joined by bounded queues with a documented
backpressure policy, then the naive per-frame VLM baseline via `llama-server`, producing
`results/phase1_naive_baseline.png`. The first real decision there is measuring the HTTP + SSE
overhead of the `llama-server` path (D2) so it can be stated rather than assumed.
