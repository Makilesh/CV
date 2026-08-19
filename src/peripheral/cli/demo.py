"""Phase 7: the live demo. Native Windows, webcam, HUD.

    # live, with a window
    python -m peripheral.cli.demo --duration 120 --metrics-out results/demo.json

    # headless (CI, or recording frames for the GIF) — same code path, no window
    python -m peripheral.cli.demo --duration 30 --headless \
        --metrics-out results/demo.json -o demo.write_frames=results/demo_frames

**Not Docker.** Webcam passthrough into a WSL2-backed container is a pile of pain that would buy
nothing; the eval path is containerised instead and the split is documented in the README.

The demo is a `BoundedRunner` like everything else, so `--duration/--headless/--metrics-out` work and
it produces the same metrics JSON as a benchmark run. A demo that cannot be measured is a
screenshot.
"""

from __future__ import annotations

import json
import sys
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..capture import build_source
from ..fasttier import FastTier, build_encoder
from ..runtime import BoundedRunner
from ..scheduler.policies import FrameContext, build_policy
from ..telemetry.clock import Stopwatch, now
from ..viz.hud import HudState, draw
from ..vlm import build_client
from ._args import parse_and_load

WINDOW = "Peripheral - live"


class DemoRunner(BoundedRunner):
    name = "phase7_demo"

    def setup(self) -> None:
        d = self.cfg.demo
        self.source = build_source(self.cfg)
        self.source.open()
        described = self.source.describe()
        self.recorder.note(f"frame source: {json.dumps(described)}")
        if described.get("too_dark"):
            print("WARNING: frames are near-black — light the room or use capture=webcam_demo",
                  file=sys.stderr)

        self.encoder = build_encoder(self.cfg)
        self.encoder.warmup(n=20)
        self.fast = FastTier(encoder=self.encoder,
                             motion_size=self.cfg.fasttier.motion_size,
                             half_life_s=self.cfg.fasttier.half_life_s)
        self.policy = build_policy(str(d.policy_kind), float(d.policy_value),
                                   min_gap_s=float(self.cfg.phase4.min_gap_s))
        self.policy.reset()

        self.client = build_client(self.cfg)
        self.client.start()
        self.recorder.note(f"vlm: {json.dumps(self.client.describe_config())}")

        self.hud = HudState(
            threshold=float(d.policy_value),
            model=str(self.cfg.vlm.name),
            policy=f"{d.policy_kind} @ {d.policy_value:g}",
        )
        self.write_frames = Path(d.write_frames) if d.get("write_frames") else None
        if self.write_frames:
            self.write_frames.mkdir(parents=True, exist_ok=True)

        self.t_last_call: float | None = None
        self.n_calls = 0
        self.answer: str | None = None
        self.answer_t: float | None = None
        self.ttfts: list[float] = []
        self.t0 = now()
        self._frames = 0
        self._written = 0
        self._window_open = False

        # INVARIANT 7: the capture loop must never block on inference. A synchronous demo drops to
        # 7.6 FPS because a ~500 ms VLM call stalls capture — which both looks wrong and *is* wrong.
        # One worker thread owns the VLM; the loop hands it a frame and moves on. If the worker is
        # busy the trigger is skipped, which is the correct backpressure policy: the next novel
        # frame will trigger again, and answering about a stale frame is worth less anyway.
        self._infer_lock = threading.Lock()
        self._infer_busy = False
        self._pending_skips = 0

    def step(self) -> bool:
        frame = self.source.read()
        if frame is None:
            return False
        self._frames += 1
        t = now() - self.t0
        self.recorder.record_frame_captured(frame.frame_id, frame.t_capture)

        with Stopwatch() as sw:
            scores = self.fast.process(frame)
        self.recorder.record_stage("fast_tier", sw.ms, frame.frame_id)

        ctx = FrameContext(frame.frame_id, t, scores.motion, scores.novelty, scores.scene_change,
                           None, self.t_last_call, self.n_calls)
        decision = self.policy.decide(ctx)
        self.recorder.record_trigger(frame.frame_id, self.policy.name, decision.fire,
                                     decision.score, decision.reason)

        fired = False
        if decision.fire:
            with self._infer_lock:
                busy = self._infer_busy
                if not busy:
                    self._infer_busy = True
            if busy:
                # The slow tier is still working. Skip rather than queue: a backlog would only
                # produce answers about frames that are already stale.
                self._pending_skips += 1
                self.recorder.record_frame_dropped(frame.frame_id, "vlm", "worker_busy")
            else:
                fired = True
                call_id = f"c{self.n_calls}"
                self.n_calls += 1
                self.t_last_call = t
                self.policy.observe_call(ctx)
                threading.Thread(
                    target=self._infer, args=(call_id, frame.image.copy(), frame.frame_id, t),
                    daemon=True, name=f"vlm-{call_id}",
                ).start()

        # --- HUD ------------------------------------------------------------------------
        elapsed = max(1e-6, t)
        self.hud.frame_idx = frame.frame_id
        self.hud.t = t
        self.hud.fps = self._frames / elapsed
        self.hud.novelty = scores.novelty
        self.hud.motion = scores.motion
        self.hud.fired = fired
        self.hud.n_calls = self.n_calls
        self.hud.calls_per_min = self.n_calls / (elapsed / 60.0)
        self.hud.answer = self.answer
        self.hud.answer_age_s = None if self.answer_t is None else t - self.answer_t
        self.hud.p95_ttft_ms = (
            float(np.percentile(self.ttfts, 95)) if len(self.ttfts) >= 3 else None
        )
        self.hud.novelty_history.append(scores.novelty)
        self.hud.fire_history.append(fired)

        with Stopwatch() as sw:
            rendered = draw(frame.image, self.hud)
            if self.write_frames is not None:
                cv2.imwrite(str(self.write_frames / f"{self._written:05d}.jpg"), rendered,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                self._written += 1
            if not self.headless:
                cv2.imshow(WINDOW, rendered)
                self._window_open = True
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    return False
        self.recorder.record_stage("render", sw.ms, frame.frame_id)
        return True

    def _infer(self, call_id: str, image: np.ndarray, frame_id: int, t: float) -> None:
        """Run one VLM call off the capture loop. Owns `_infer_busy` for its lifetime."""
        try:
            self.recorder.vlm_call_start(call_id, frame_id=frame_id, model=self.client.model_name)
            result = self.client.describe(
                image, prompt=str(self.cfg.demo.prompt),
                max_tokens=int(self.cfg.demo.max_tokens),
                on_first_token=lambda: self.recorder.vlm_call_first_token(call_id),
            )
            self.recorder.vlm_call_complete(call_id, gen_tokens=result.n_tokens)
            self.recorder.record_answer(call_id, evidence_frame_id=frame_id, source="vlm",
                                        call_id=call_id, text=result.text)
            self.answer = result.text.strip()
            self.answer_t = t
            if result.ttft_ms is not None:
                ttft = result.ttft_ms + result.encode_ms
                self.ttfts.append(ttft)
                self.hud.last_ttft_ms = ttft
            self.hud.last_answer_ms = result.total_ms
        except Exception as exc:  # noqa: BLE001 - a demo must survive one bad call
            self.recorder.note(f"vlm call {call_id} failed: {type(exc).__name__}: {exc}")
            self.recorder.vlm_call_complete(call_id, gen_tokens=None)
        finally:
            with self._infer_lock:
                self._infer_busy = False

    def teardown(self) -> None:
        if getattr(self, "_window_open", False):
            cv2.destroyAllWindows()
        client = getattr(self, "client", None)
        if client is not None:
            client.stop()
        source = getattr(self, "source", None)
        if source is not None:
            source.close()
        if getattr(self, "write_frames", None) is not None:
            self.recorder.note(f"wrote {self._written} HUD frames to {self.write_frames}")


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.demo",
        description="Phase 7: live webcam demo with a HUD. Native Windows.",
        argv=argv,
    )
    runner = DemoRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                        metrics_out=cfg.run.metrics_out,
                        headless=cfg.run.headless, seed=cfg.seed)
    code = runner.run()
    m = runner.recorder.finalize()["metrics"]
    ttft = m["photon_to_first_token_ms"]
    print(
        f"{m['frames_captured']} frames at {m['achieved_fps']} FPS | "
        f"{m['vlm_calls']} VLM calls ({m['vlm_calls_per_min']}/min) | "
        f"photon->first-token p50 {ttft['p50'] if ttft else None} ms",
        file=sys.stderr,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())

