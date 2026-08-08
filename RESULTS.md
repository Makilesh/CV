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

## Reproducing

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Every figure is regenerated from its metrics JSON; no number in this document is typed by hand into
a chart. Run any phase's CLI with `--duration N --headless --metrics-out path.json`.
