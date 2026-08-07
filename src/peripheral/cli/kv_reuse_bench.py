"""Phase 3: KV-cache reuse and streaming decode, with before/after TTFT numbers.

`PROMPT.md`: *implement KV-cache reuse across calls and streaming decode; show the TTFT improvement
with before/after numbers.*

**Why prompt ordering is the whole game here.** Image tokens differ on every frame, so a prompt
that puts the image first has no shared prefix between calls and nothing can be reused no matter
what flags the server is given. Putting the fixed instruction text first makes that text a
cacheable prefix. This benchmark measures exactly that, plus llama-server's `--cache-reuse`, plus
the separate benefit of streaming decode.

Four arms, all on the same model and the same frames:

  A  image-first, no cache-reuse   — the naive arrangement, nothing is reusable
  B  text-first,  no cache-reuse   — slot prefix caching can now do something
  C  text-first,  --cache-reuse    — plus llama-server's chunked reuse
  D  text-first,  non-streaming    — isolates what streaming decode buys

    python -m peripheral.cli.kv_reuse_bench --duration 1800 --headless \
        --metrics-out results/phase3_kv_reuse.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..capture import FileSource
from ..runtime import BoundedRunner
from ..telemetry.clock import now
from ..vlm.llama_server import LlamaServerClient
from ._args import parse_and_load

# A realistic standing instruction. Kept substantial on purpose: a one-word prompt would have
# almost no prefix to cache, and would understate reuse for anyone whose real system prompt is
# longer than a toy.
SYSTEM = (
    "You are a real-time video understanding assistant monitoring a live camera feed. "
    "Answer questions about the current scene concisely and factually. Do not speculate "
    "about anything you cannot see. Prefer concrete nouns over vague description. "
    "If the scene is unclear, say so plainly rather than guessing."
)


class KvReuseRunner(BoundedRunner):
    name = "phase3_kv_reuse"

    def setup(self) -> None:
        k = self.cfg.kv_reuse
        clip = Path(k.clip)
        src = FileSource(clip, realtime=False)
        src.open()
        frames, i = [], 0
        while len(frames) < int(k.n_frames):
            f = src.read()
            if f is None:
                break
            if i % int(k.frame_stride) == 0:
                frames.append(f.image)
            i += 1
        src.close()
        self.frames = frames
        self.arms = [dict(a) for a in k.arms]
        self.results: list[dict[str, Any]] = []
        self._i = 0
        self.recorder.note(f"{len(frames)} frames from {clip}; system prompt {len(SYSTEM)} chars")

    def step(self) -> bool:
        if self._i >= len(self.arms):
            return False
        arm = self.arms[self._i]
        self._i += 1
        try:
            row = self._run_arm(arm)
        except Exception as exc:  # noqa: BLE001
            self.recorder.note(f"arm {arm['label']!r} failed: {type(exc).__name__}: {exc}")
            self.results.append({"label": arm["label"], "error": f"{type(exc).__name__}: {exc}"})
            return True
        self.results.append(row)
        ttft = row["ttft_p50_ms"]
        print(
            f"{row['label']:38s} TTFT p50 {ttft if ttft else float('nan'):7.1f} ms   "
            f"p95 {row['ttft_p95_ms'] if row['ttft_p95_ms'] else float('nan'):7.1f}   "
            f"answer p50 {row['total_p50_ms']:7.1f} ms",
            file=sys.stderr,
        )
        return True

    def _run_arm(self, arm: dict[str, Any]) -> dict[str, Any]:
        v = self.cfg.vlm
        k = self.cfg.kv_reuse
        client = LlamaServerClient(
            binary=v.binary,
            model=k.model,
            mmproj=k.mmproj,
            host=v.host,
            port=int(v.port),
            n_gpu_layers=int(v.n_gpu_layers),
            ctx_size=int(v.ctx_size),
            jpeg_quality=int(v.jpeg_quality),
            autostart=True,
            startup_timeout_s=float(v.startup_timeout_s),
            extra_args=list(arm.get("server_args", []) or []),
            model_name=arm["label"],
        )
        client.start()
        try:
            kwargs = dict(
                prompt=str(k.prompt),
                max_tokens=int(k.max_tokens),
                image_first=bool(arm.get("image_first", False)),
                system=SYSTEM if arm.get("system", True) else None,
                stream=bool(arm.get("stream", True)),
            )
            for _ in range(int(k.warmup)):
                client.describe(self.frames[0], **kwargs)

            ttft, total = [], []
            for _ in range(int(k.reps)):
                for img in self.frames:
                    t0 = now()
                    r = client.describe(img, **kwargs)
                    total.append((now() - t0) * 1000.0)
                    if r.ttft_ms is not None:
                        ttft.append(r.ttft_ms + r.encode_ms)
        finally:
            client.stop()

        t = np.asarray(ttft) if ttft else None
        tot = np.asarray(total)
        return {
            "label": arm["label"],
            "image_first": bool(arm.get("image_first", False)),
            "stream": bool(arm.get("stream", True)),
            "server_args": list(arm.get("server_args", []) or []),
            "n_calls": len(total),
            "ttft_p50_ms": round(float(np.percentile(t, 50)), 1) if t is not None else None,
            "ttft_p95_ms": round(float(np.percentile(t, 95)), 1) if t is not None else None,
            "total_p50_ms": round(float(np.percentile(tot, 50)), 1),
            "total_p95_ms": round(float(np.percentile(tot, 95)), 1),
        }

    def teardown(self) -> None:
        ok = [r for r in self.results if "error" not in r]
        baseline = next((r for r in ok if r["label"].startswith("A")), None)
        for r in ok:
            if baseline and baseline["ttft_p50_ms"] and r["ttft_p50_ms"]:
                r["ttft_speedup_vs_A"] = round(baseline["ttft_p50_ms"] / r["ttft_p50_ms"], 3)
        self.recorder.record_extra(
            "kv_reuse",
            {
                "arms": self.results,
                "system_prompt_chars": len(SYSTEM),
                "definition": (
                    "TTFT is frame-in-hand to first token, including JPEG encode, HTTP and SSE. "
                    "Arm A is the naive arrangement (image first, nothing cacheable). "
                    "Arm D is non-streaming, where no first token exists before the full answer."
                ),
            },
        )


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.kv_reuse_bench",
        description="Phase 3: KV-cache reuse and streaming decode, before/after.",
        argv=argv,
    )
    runner = KvReuseRunner(
        cfg=cfg,
        duration_s=cfg.run.duration_s,
        metrics_out=cfg.run.metrics_out,
        headless=cfg.run.headless,
        seed=cfg.seed,
    )
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
