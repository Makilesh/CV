"""The HUD: what the system is doing, drawn on the frame it is doing it to.

Shared by the live demo (`peripheral.cli.demo`) and the GIF generator, so the GIF shows the real
overlay rather than a mock-up of one.

Design constraints, both from the project's own rules:

* **It must be cheap.** The HUD runs on every displayed frame, so it uses only OpenCV primitives and
  no per-frame allocation beyond the overlay copy. Measured cost is recorded as the `render` stage,
  the same as every other stage.
* **It must show cost, not just output.** A demo that shows only the answer hides the entire point.
  Calls/min, the novelty trace against its threshold, and trigger markers are all on screen, because
  the claim is about how *rarely* the VLM runs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# BGR. Deliberately high-contrast: this gets read off a compressed GIF.
FG = (245, 245, 245)
DIM = (170, 170, 170)
ACCENT = (240, 176, 64)      # novelty trace (blue-ish in BGR)
FIRE = (80, 80, 235)         # trigger / VLM firing (red)
OK = (120, 200, 120)         # within budget (green)
PANEL = (28, 28, 28)


@dataclass
class HudState:
    """Everything the HUD draws. The caller owns the truth; the HUD only renders it."""

    frame_idx: int = 0
    t: float = 0.0
    fps: float = 0.0
    novelty: float = 0.0
    motion: float = 0.0
    threshold: float = 0.12
    fired: bool = False
    n_calls: int = 0
    calls_per_min: float = 0.0
    answer: str | None = None
    answer_age_s: float | None = None
    last_ttft_ms: float | None = None
    last_answer_ms: float | None = None
    p95_ttft_ms: float | None = None
    dropped: int = 0
    model: str = ""
    policy: str = ""
    novelty_history: deque = field(default_factory=lambda: deque(maxlen=180))
    fire_history: deque = field(default_factory=lambda: deque(maxlen=180))


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw(frame: np.ndarray, s: HudState, scale: float = 1.0) -> np.ndarray:
    """Return `frame` with the HUD composited on top. Does not modify the input."""
    img = frame.copy()
    h, w = img.shape[:2]
    f = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.46 * scale
    fs_small = 0.38 * scale
    th = max(1, int(round(scale)))

    # --- top strip: identity and cost ---------------------------------------------------
    strip_h = int(56 * scale)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, strip_h), PANEL, -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    cv2.putText(img, "PERIPHERAL", (int(10 * scale), int(21 * scale)), f, fs * 1.05, FG, th + 1,
                cv2.LINE_AA)
    cv2.putText(img, f"{s.model}  |  {s.policy}", (int(10 * scale), int(40 * scale)), f, fs_small,
                DIM, th, cv2.LINE_AA)

    # Cost is the headline number, so it is the biggest thing after the title.
    calls_txt = f"{s.calls_per_min:.1f} VLM calls/min"
    (tw, _), _ = cv2.getTextSize(calls_txt, f, fs * 1.1, th + 1)
    cv2.putText(img, calls_txt, (w - tw - int(12 * scale), int(23 * scale)), f, fs * 1.1,
                OK if s.calls_per_min < 60 else FIRE, th + 1, cv2.LINE_AA)

    sub = f"{s.n_calls} calls | {s.fps:.1f} FPS capture"
    if s.dropped:
        sub += f" | {s.dropped} dropped"
    (tw2, _), _ = cv2.getTextSize(sub, f, fs_small, th)
    cv2.putText(img, sub, (w - tw2 - int(12 * scale), int(41 * scale)), f, fs_small, DIM, th,
                cv2.LINE_AA)

    # --- novelty trace ------------------------------------------------------------------
    trace_h = int(46 * scale)
    trace_y0 = strip_h + int(6 * scale)
    hist = list(s.novelty_history)
    if len(hist) > 1:
        ov = img.copy()
        cv2.rectangle(ov, (0, trace_y0), (w, trace_y0 + trace_h), PANEL, -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)

        top = max(max(hist), s.threshold) * 1.25 or 1.0
        n = len(hist)
        step = w / max(1, n - 1)

        # threshold line — the decision boundary, which is what makes the trace legible
        ty = trace_y0 + trace_h - int(s.threshold / top * trace_h)
        cv2.line(img, (0, ty), (w, ty), FIRE, max(1, th), cv2.LINE_AA)
        cv2.putText(img, f"novelty threshold {s.threshold:g}", (int(8 * scale), ty - int(4 * scale)),
                    f, fs_small * 0.9, FIRE, th, cv2.LINE_AA)

        pts = [
            (int(i * step), trace_y0 + trace_h - int(v / top * trace_h))
            for i, v in enumerate(hist)
        ]
        cv2.polylines(img, [np.array(pts, np.int32)], False, ACCENT, max(1, th), cv2.LINE_AA)

        for i, fired in enumerate(s.fire_history):
            if fired:
                x = int(i * step)
                cv2.line(img, (x, trace_y0), (x, trace_y0 + trace_h), FIRE, max(1, th))

    # --- answer panel -------------------------------------------------------------------
    lines = _wrap(s.answer or "(waiting for the first VLM call)", int(w / (12 * scale)))[:3]
    panel_h = int(26 * scale) + int(20 * scale) * len(lines)
    y0 = h - panel_h
    ov = img.copy()
    cv2.rectangle(ov, (0, y0), (w, h), PANEL, -1)
    cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)

    age = "" if s.answer_age_s is None else f"  ({s.answer_age_s:.1f}s old)"
    lat = []
    if s.last_ttft_ms is not None:
        lat.append(f"first token {s.last_ttft_ms:.0f} ms")
    if s.p95_ttft_ms is not None:
        lat.append(f"p95 {s.p95_ttft_ms:.0f} ms")
    header = "ANSWER" + age + ("   ·   " + " · ".join(lat) if lat else "")
    cv2.putText(img, header, (int(10 * scale), y0 + int(17 * scale)), f, fs_small, DIM, th,
                cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (int(10 * scale), y0 + int(38 * scale) + i * int(20 * scale)),
                    f, fs, FG, th, cv2.LINE_AA)

    # --- firing flash -------------------------------------------------------------------
    if s.fired:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), FIRE, int(4 * scale))
        badge = "VLM FIRED"
        (bw, bh), _ = cv2.getTextSize(badge, f, fs, th + 1)
        bx, by = w - bw - int(18 * scale), trace_y0 + trace_h + int(28 * scale)
        cv2.rectangle(img, (bx - int(8 * scale), by - bh - int(8 * scale)),
                      (bx + bw + int(8 * scale), by + int(6 * scale)), FIRE, -1)
        cv2.putText(img, badge, (bx, by), f, fs, (255, 255, 255), th + 1, cv2.LINE_AA)

    return img
