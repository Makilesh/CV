"""Phase 9a: does content-awareness pay when events are sparse?

Phase 8 failed its exit criterion and measured why: on 24-second clips with an event every 8–12 s,
a 2-second timer is only 4–6x oversampled, and at that density blind sampling is near-optimal. The
claim that a scheduler beats a timer was never testable on that data.

This tests it. Clips are built at controlled event sparsity — the same duration, a varying number of
events — so the timer's oversampling factor sweeps from ~10x to ~150x, and the question becomes:
**at what sparsity, if any, does picking frames beat spacing them?**

**No VLM is used, and none is needed.** `answer_validity`, event recall and false-trigger rate are
computed from scene-state labels, which are exact by construction; only the secondary text-agreement
metric needs oracle answers. Sidestepping the oracle turns a ~4-hour run into a few minutes and
costs nothing that this question depends on — so `accuracy` is reported as **null**, not as a
fabricated 1.0.

    python -m peripheral.cli.sparsity_study --duration 3600 --headless \\
        --metrics-out results/phase9_sparsity.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..capture import FileSource
from ..eval.clips import ClipAnnotation, build_clip
from ..eval.simulate import simulate
from ..eval.traces import ClipTrace
from ..fasttier import build_encoder
from ..fasttier.spatial import SpatialEncoder, SpatialFastTier, export_spatial_onnx
from ..runtime import BoundedRunner
from ..scheduler.policies import (
    AdaptiveNoveltyPolicy,
    EmbeddingNoveltyPolicy,
    FixedIntervalPolicy,
    MotionThresholdPolicy,
)
from ._args import parse_and_load

VALIDITY_BAR = 0.99


def _progress(line: str) -> None:
    try:
        sys.stderr.write(line.encode("ascii", "replace").decode("ascii") + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


class SparsityStudyRunner(BoundedRunner):
    name = "phase9_sparsity_study"

    def setup(self) -> None:
        p9 = self.cfg.phase9
        self.clip_duration_s = float(p9.clip_duration_s)
        self.recipes = list(p9.recipes)
        self.clip_dir = Path(p9.clip_dir)
        self.base = Path(self.cfg.phase4.base_footage)
        if not self.base.exists():
            raise FileNotFoundError(f"base footage not found: {self.base}")

        onnx = export_spatial_onnx(
            str(self.cfg.phase8.spatial_model), str(self.cfg.phase8.spatial_onnx_path),
            int(self.cfg.phase8.spatial_input_size),
        )
        self.encoder = SpatialEncoder(onnx, int(self.cfg.phase8.spatial_input_size))
        self.encoder.warmup(n=20)
        self.fast = SpatialFastTier(
            encoder=self.encoder,
            motion_size=int(self.cfg.fasttier.motion_size),
            half_life_s=float(self.cfg.fasttier.half_life_s),
            top_k=int(self.cfg.phase8.patch_top_k),
        )
        self.min_gap_s = float(self.cfg.phase4.min_gap_s)
        self.warmup = int(self.cfg.phase4.warmup_frames)
        self.rows: list[dict[str, Any]] = []
        self._i = 0

    def _build_trace(self, ann: ClipAnnotation) -> ClipTrace:
        """Signals + labels only. No oracle: this question does not need model outputs."""
        src = FileSource(ann.path, realtime=False)
        src.open()
        self.fast.reset()
        motion, patch, pooled, scene = [], [], [], []
        try:
            while True:
                f = src.read()
                if f is None:
                    break
                s = self.fast.process(f)
                motion.append(s.motion)
                patch.append(s.novelty)
                pooled.append(s.meta.get("global_novelty", 0.0))
                scene.append(s.scene_change)
        finally:
            src.close()

        n = len(patch)
        return ClipTrace(
            clip_name=ann.name, fps=ann.fps, n_frames=n,
            motion=np.asarray(motion, np.float32),
            novelty=np.asarray(pooled, np.float32),
            scene_change=np.asarray(scene, np.float32),
            # Sentinel, never scored: `accuracy` is reported as null for this study.
            answers=["__no_oracle__"] * n,
            state_ids=np.asarray(ann.state_ids[:n], np.int32),
            encoder=self.encoder.describe(), model="none (labels only)", prompt="",
            novelty_patch=np.asarray(patch, np.float32),
        )

    def step(self) -> bool:
        if self._i >= len(self.recipes):
            return False
        recipe = str(self.recipes[self._i])
        self._i += 1

        frozen = recipe.startswith("frozen_")
        base_recipe = recipe[len("frozen_"):] if frozen else recipe
        ann = build_clip(
            base_path=self.base, out_dir=self.clip_dir, name=f"p9_{recipe}",
            recipe_name=base_recipe, duration_s=self.clip_duration_s,
            fps=float(self.cfg.phase4.clip_fps), seed=self.seed, freeze_base=frozen,
        )
        trace = self._build_trace(ann)
        n_events = len(ann.semantic_events)
        dur = trace.n_frames / trace.fps

        best: dict[str, Any] = {}
        for kind, grid, sig in (
            ("fixed_interval", [0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60], "pooled"),
            ("adaptive_novelty", [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999], "patch"),
            ("embedding_novelty", list(np.linspace(0.05, 0.9, 25)), "patch"),
            ("motion_threshold", list(np.linspace(0.001, 0.08, 25)), "pooled"),
        ):
            view = trace.with_signal(sig)
            for v in grid:
                if kind == "fixed_interval":
                    pol = FixedIntervalPolicy(float(v))
                elif kind == "adaptive_novelty":
                    pol = AdaptiveNoveltyPolicy(float(v),
                                                window_s=float(self.cfg.phase8.adaptive_window_s),
                                                min_gap_s=self.min_gap_s)
                elif kind == "embedding_novelty":
                    pol = EmbeddingNoveltyPolicy(float(v), min_gap_s=self.min_gap_s)
                else:
                    pol = MotionThresholdPolicy(float(v), min_gap_s=self.min_gap_s)

                r = simulate(view, pol, warmup_frames=self.warmup)
                if r.answer_validity < VALIDITY_BAR:
                    continue
                if kind not in best or r.calls_per_min < best[kind]["calls_per_min"]:
                    best[kind] = {
                        "operating_point": round(float(v), 5),
                        "calls_per_min": r.calls_per_min,
                        "n_calls": r.n_calls,
                        "validity": r.answer_validity,
                        "event_recall": r.event_recall,
                        "false_trigger_rate": r.false_trigger_rate,
                        "signal": sig,
                    }

        timer = best.get("fixed_interval")
        row = {
            "clip": ann.name, "recipe": recipe, "static_background": frozen,
            "duration_s": round(dur, 1), "n_events": n_events,
            "seconds_per_event": round(dur / n_events, 1) if n_events else None,
            # How many calls a timer at the winning interval makes per event — the regime number.
            "timer_oversampling_x": (
                round(timer["n_calls"] / n_events, 2) if timer and n_events else None
            ),
            "best": best,
        }
        for k, v in best.items():
            if timer and k != "fixed_interval" and v["calls_per_min"] > 0:
                v["speedup_vs_timer"] = round(timer["calls_per_min"] / v["calls_per_min"], 3)
        self.rows.append(row)

        parts = " | ".join(
            f"{k.split('_')[0]} {v['calls_per_min']:.1f}" for k, v in sorted(best.items())
        )
        _progress(f"  {recipe:14s} {n_events:3d} events, "
                  f"{row['seconds_per_event'] or float('inf'):6.1f} s/event -> {parts}")
        return True

    def teardown(self) -> None:
        winners = []
        for r in self.rows:
            t = r["best"].get("fixed_interval")
            if not t:
                continue
            for k, v in r["best"].items():
                if k != "fixed_interval" and v["calls_per_min"] < t["calls_per_min"]:
                    winners.append({"clip": r["clip"], "policy": k,
                                    "speedup": v.get("speedup_vs_timer"),
                                    "seconds_per_event": r["seconds_per_event"]})

        self.recorder.record_extra("phase9_sparsity", {
            "rows": self.rows,
            "content_beats_timer": winners,
            "validity_bar": VALIDITY_BAR,
            "no_oracle": (
                "No VLM was run. answer_validity, event recall and false-trigger rate come from "
                "scene-state labels that are exact by construction; only the secondary "
                "text-agreement metric needs oracle answers, and it is reported as null rather "
                "than fabricated. This is what makes sweeping clip length affordable at all."
            ),
            "why": (
                "Phase 8 showed the thesis was untestable at 4-6x timer oversampling. This sweeps "
                "event sparsity to find the regime, if any, where selecting frames beats spacing "
                "them."
            ),
        })


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.sparsity_study",
        description="Phase 9a: does content-awareness pay when events are sparse?",
        argv=argv,
    )
    runner = SparsityStudyRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                                metrics_out=cfg.run.metrics_out,
                                headless=cfg.run.headless, seed=cfg.seed)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
