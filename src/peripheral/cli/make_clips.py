"""Phase 4 step 1: build the annotated evaluation clips.

    python -m peripheral.cli.make_clips --duration 300 --headless \
        --metrics-out results/phase4_clips.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..eval.clips import build_clip
from ..runtime import BoundedRunner
from ._args import parse_and_load


class MakeClipsRunner(BoundedRunner):
    name = "phase4_make_clips"

    def setup(self) -> None:
        c = self.cfg.phase4
        self.base = Path(c.base_footage)
        if not self.base.exists():
            raise FileNotFoundError(
                f"base footage not found: {self.base} — record one with peripheral.cli.record_clip"
            )
        self.out_dir = Path(c.clip_dir)
        self.specs = [dict(s) for s in c.clips]
        self.built = []
        self._i = 0

    def step(self) -> bool:
        if self._i >= len(self.specs):
            return False
        spec = self.specs[self._i]
        self._i += 1
        ann = build_clip(
            base_path=self.base,
            out_dir=self.out_dir,
            name=spec["name"],
            recipe_name=spec["recipe"],
            duration_s=float(self.cfg.phase4.clip_duration_s),
            fps=float(self.cfg.phase4.clip_fps),
            seed=self.seed,
        )
        self.built.append({
            "name": ann.name, "path": ann.path, "frames": ann.n_frames,
            "events": len(ann.events), "semantic_events": len(ann.semantic_events),
            "states": len(set(ann.state_ids)), "purpose": ann.purpose,
        })
        print(f"  {ann.name:16s} {ann.n_frames:4d} frames, "
              f"{len(ann.semantic_events)} semantic events, {len(set(ann.state_ids))} states",
              file=sys.stderr)
        return True

    def teardown(self) -> None:
        self.recorder.record_extra("phase4_clips", {
            "base_footage": str(self.base),
            "clip_dir": str(self.out_dir),
            "clips": self.built,
            "note": (
                "Events are composited onto real camera footage so the labels are exact by "
                "construction. This makes the lighting-drift and motion-without-event probes "
                "possible at all — certainty that NOTHING semantic happened is what hand "
                "annotation cannot provide. The cost is that a pasted object is an easier event "
                "than a subtle real one; Phase 6 adds real annotated data."
            ),
        })


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.make_clips",
        description="Phase 4: build annotated evaluation clips.",
        argv=argv,
    )
    runner = MakeClipsRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                             metrics_out=cfg.run.metrics_out,
                             headless=cfg.run.headless, seed=cfg.seed)
    code = runner.run()
    print(json.dumps(runner.built, indent=2)[:600], file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
