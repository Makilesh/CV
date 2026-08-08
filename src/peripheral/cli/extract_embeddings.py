"""Phase 5 step 1: add per-frame embeddings to existing traces.

The Phase 4 traces stored fast-tier *scores* but not the embedding vectors themselves, which the
cache needs as its keys. Re-running the fast tier is cheap (4.55 ms/frame, no VLM), so this
backfills the traces in seconds rather than repeating the 4,320-call oracle.

    python -m peripheral.cli.extract_embeddings --duration 600 --headless \
        --metrics-out results/phase5_embeddings.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..capture import FileSource
from ..eval.clips import ClipAnnotation
from ..fasttier import FastTier, build_encoder
from ..runtime import BoundedRunner
from ._args import parse_and_load


class ExtractEmbeddingsRunner(BoundedRunner):
    name = "phase5_extract_embeddings"

    def setup(self) -> None:
        c = self.cfg.phase4
        self.clip_dir = Path(c.clip_dir)
        self.trace_dir = Path(c.trace_dir)
        self.annotations = [ClipAnnotation.load(p) for p in sorted(self.clip_dir.glob("*.json"))]
        if not self.annotations:
            raise FileNotFoundError(f"no clips in {self.clip_dir}")

        self.encoder = build_encoder(self.cfg)
        self.encoder.warmup(n=20)
        self.fast = FastTier(
            encoder=self.encoder,
            motion_size=self.cfg.fasttier.motion_size,
            half_life_s=self.cfg.fasttier.half_life_s,
            keep_embedding=True,
        )
        self.done: list[dict] = []
        self._i = 0

    def step(self) -> bool:
        if self._i >= len(self.annotations):
            return False
        ann = self.annotations[self._i]
        self._i += 1

        npz_path = self.trace_dir / f"{ann.name}.npz"
        if not npz_path.exists():
            self.recorder.note(f"{ann.name}: no trace, skipped")
            return True
        existing = dict(np.load(npz_path))
        if "embeddings" in existing and not bool(self.cfg.phase5.get("rebuild", False)):
            self.done.append({"clip": ann.name, "status": "cached"})
            return True

        src = FileSource(ann.path, realtime=False)
        src.open()
        self.fast.reset()
        embs = []
        try:
            while True:
                frame = src.read()
                if frame is None:
                    break
                scores = self.fast.process(frame)
                embs.append(scores.embedding)
        finally:
            src.close()

        arr = np.asarray(embs, dtype=np.float32)
        existing["embeddings"] = arr
        np.savez_compressed(npz_path, **existing)
        self.done.append({"clip": ann.name, "status": "built", "shape": list(arr.shape)})
        print(f"  {ann.name:16s} embeddings {arr.shape}", file=sys.stderr)
        return True

    def teardown(self) -> None:
        self.recorder.record_extra("phase5_embeddings", {
            "clips": self.done,
            "encoder": self.encoder.describe() if hasattr(self, "encoder") else None,
        })


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.extract_embeddings",
        description="Phase 5: backfill per-frame embeddings into existing traces.",
        argv=argv,
    )
    runner = ExtractEmbeddingsRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                                    metrics_out=cfg.run.metrics_out,
                                    headless=cfg.run.headless, seed=cfg.seed)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
