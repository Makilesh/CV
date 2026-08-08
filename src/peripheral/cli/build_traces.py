"""Phase 4 step 2: build per-clip traces — fast-tier scores and the per-frame oracle.

**This is the expensive step**: one VLM call per frame per clip. It is cached, so the sweep that
follows is a lookup rather than days of GPU time. Existing traces are skipped unless --rebuild.

    python -m peripheral.cli.build_traces --duration 7200 --headless \
        --metrics-out results/phase4_traces.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..eval.clips import ClipAnnotation
from ..eval.traces import ClipTrace, build_trace
from ..fasttier import FastTier, build_encoder
from ..runtime import BoundedRunner
from ..vlm import build_client
from ._args import parse_and_load


def _progress(line: str) -> None:
    """Progress must never be able to kill a 45-minute run (see hf_reference)."""
    try:
        sys.stderr.write(line.encode("ascii", "replace").decode("ascii") + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


class BuildTracesRunner(BoundedRunner):
    name = "phase4_build_traces"

    def setup(self) -> None:
        c = self.cfg.phase4
        clip_dir = Path(c.clip_dir)
        self.trace_dir = Path(c.trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.annotations = [
            ClipAnnotation.load(p) for p in sorted(clip_dir.glob("*.json"))
        ]
        if not self.annotations:
            raise FileNotFoundError(
                f"no annotated clips in {clip_dir} — run peripheral.cli.make_clips first"
            )
        self.rebuild = bool(c.get("rebuild_traces", False))
        self.prompt = str(c.prompt)
        self.max_tokens = int(c.max_tokens)

        self.encoder = build_encoder(self.cfg)
        self.encoder.warmup(n=20)
        self.fast = FastTier(
            encoder=self.encoder,
            motion_size=self.cfg.fasttier.motion_size,
            half_life_s=self.cfg.fasttier.half_life_s,
        )
        self.client = build_client(self.cfg)
        self.client.start()
        self.recorder.note(f"oracle model: {self.client.model_name}")
        self.built: list[dict] = []
        self._i = 0

    def step(self) -> bool:
        if self._i >= len(self.annotations):
            return False
        ann = self.annotations[self._i]
        self._i += 1
        path = self.trace_dir / ann.name

        if ClipTrace.exists(path) and not self.rebuild:
            _progress(f"  {ann.name}: cached")
            self.built.append({"clip": ann.name, "status": "cached"})
            return True

        _progress(f"  {ann.name}: {ann.n_frames} oracle calls ...")
        trace = build_trace(
            ann, self.fast, self.client, prompt=self.prompt,
            max_tokens=self.max_tokens, progress=_progress,
        )
        trace.save(path)
        self.built.append({"clip": ann.name, "status": "built", "frames": trace.n_frames})
        _progress(f"  {ann.name}: done, {trace.n_frames} frames")
        return True

    def teardown(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            client.stop()
        self.recorder.record_extra("phase4_traces", {
            "trace_dir": str(getattr(self, "trace_dir", "")),
            "clips": self.built,
            "prompt": getattr(self, "prompt", None),
            "note": (
                "A trace holds the per-frame oracle's answers, so any policy that decides to call "
                "on frame i is given the oracle's answer for frame i. The sweep is therefore exact "
                "rather than approximate. It measures accuracy and call rate ONLY — latency comes "
                "from Phase 3, measured end to end on the real pipeline."
            ),
        })


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.build_traces",
        description="Phase 4: build fast-tier + per-frame-oracle traces.",
        argv=argv,
    )
    runner = BuildTracesRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                              metrics_out=cfg.run.metrics_out,
                              headless=cfg.run.headless, seed=cfg.seed)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
