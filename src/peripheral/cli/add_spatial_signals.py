"""Phase 8b: add patch-novelty signals to the existing traces.

Writes `novelty_patch` alongside the existing `novelty`, leaving every other array untouched. Both
signals then exist for the *same frames of the same clips*, so pooled and spatial can be compared
without re-running the oracle and without a paired-run confound.

    python -m peripheral.cli.add_spatial_signals --duration 900 --headless \\
        --metrics-out results/phase8_spatial_signals.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..capture import FileSource
from ..eval.clips import ClipAnnotation
from ..fasttier.spatial import SpatialEncoder, SpatialFastTier, export_spatial_onnx
from ..runtime import BoundedRunner
from ._args import parse_and_load


def _progress(line: str) -> None:
    try:
        sys.stderr.write(line.encode("ascii", "replace").decode("ascii") + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


class AddSpatialSignalsRunner(BoundedRunner):
    name = "phase8_add_spatial_signals"

    def setup(self) -> None:
        p4, p8 = self.cfg.phase4, self.cfg.phase8
        self.clip_dir = Path(p4.clip_dir)
        self.trace_dir = Path(p4.trace_dir)
        self.annotations = [ClipAnnotation.load(p) for p in sorted(self.clip_dir.glob("*.json"))]
        if not self.annotations:
            raise FileNotFoundError(f"no clips in {self.clip_dir}")

        onnx = export_spatial_onnx(
            str(p8.spatial_model), str(p8.spatial_onnx_path), int(p8.spatial_input_size)
        )
        self.encoder = SpatialEncoder(onnx, int(p8.spatial_input_size))
        self.encoder.warmup(n=20)
        self.fast = SpatialFastTier(
            encoder=self.encoder,
            motion_size=int(self.cfg.fasttier.motion_size),
            half_life_s=float(self.cfg.fasttier.half_life_s),
            top_k=int(p8.patch_top_k),
        )
        self.rows: list[dict] = []
        self._i = 0
        self.recorder.note(f"spatial encoder: {self.encoder.describe()}")

    def step(self) -> bool:
        if self._i >= len(self.annotations):
            return False
        ann = self.annotations[self._i]
        self._i += 1

        npz_path = self.trace_dir / f"{ann.name}.npz"
        if not npz_path.exists():
            return True
        existing = dict(np.load(npz_path))

        src = FileSource(ann.path, realtime=False)
        src.open()
        self.fast.reset()
        patch, glob, scene = [], [], []
        try:
            while True:
                frame = src.read()
                if frame is None:
                    break
                s = self.fast.process(frame)
                patch.append(s.novelty)
                glob.append(s.meta.get("global_novelty", 0.0))
                scene.append(s.scene_change)
        finally:
            src.close()

        existing["novelty_patch"] = np.asarray(patch, np.float32)
        existing["novelty_global_check"] = np.asarray(glob, np.float32)
        existing["scene_change_patch"] = np.asarray(scene, np.float32)
        np.savez_compressed(npz_path, **existing)

        pooled = existing.get("novelty")
        row = {
            "clip": ann.name,
            "pooled_p50": round(float(np.percentile(pooled, 50)), 5),
            "pooled_max": round(float(pooled.max()), 5),
            "patch_p50": round(float(np.percentile(patch, 50)), 5),
            "patch_max": round(float(np.max(patch)), 5),
        }
        self.rows.append(row)
        _progress(f"  {ann.name:16s} pooled p50 {row['pooled_p50']:.4f} max {row['pooled_max']:.4f}"
                  f"   ->  patch p50 {row['patch_p50']:.4f} max {row['patch_max']:.4f}")
        return True

    def teardown(self) -> None:
        self.recorder.record_extra("phase8_spatial_signals", {
            "encoder": self.encoder.describe() if hasattr(self, "encoder") else None,
            "top_k": int(self.cfg.phase8.patch_top_k),
            "clips": self.rows,
            "note": (
                "novelty_patch is the mean cosine distance over the top-k most-changed spatial "
                "cells, each with its own rolling reference. Written alongside the pooled novelty "
                "so both exist for identical frames and can be compared without a paired-run "
                "confound. Oracle answers are untouched."
            ),
        })


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.add_spatial_signals",
        description="Phase 8b: add patch-novelty signals to existing traces.",
        argv=argv,
    )
    runner = AddSpatialSignalsRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                                    metrics_out=cfg.run.metrics_out,
                                    headless=cfg.run.headless, seed=cfg.seed)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
