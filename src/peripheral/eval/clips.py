"""Evaluation clips with ground-truth annotations.

Phase 4 needs something Phases 0–3 never did: **labels**. A scheduler can only be scored against
"was this call worth making", and that requires knowing when the scene actually changed.

**Why these clips are synthesised on top of real footage rather than staged.** The headline failure
mode is a false trigger on lighting drift or motion carrying no semantic event. To measure that you
need clips where you are *certain* nothing semantic happened — and certainty about a negative is
exactly what hand-annotating real footage cannot give you. Compositing controlled events onto real
camera footage yields exact labels by construction, and lets the lighting-drift and
motion-without-event probes be built deliberately instead of hoped for.

The cost is honesty about difficulty: a pasted object is an *easier* semantic event than a subtle
real one, so passing here does not prove passing on real events. Phase 6 adds real annotated data
(StreamingBench / OVOBench). This is stated in RESULTS.md, not buried.

**Scene state is the label.** Each clip carries a piecewise-constant `state_id` over time. A VLM
call is *justified* when the state differs from the state at the previous call, and a **false
trigger** when it does not. That definition is what makes false-trigger rate first-class.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

SCHEMA_VERSION = 1


@dataclass
class Event:
    """A labelled change. `semantic=True` means the scene content genuinely changed."""

    t_start: float
    kind: str           # object_appear | object_change | object_vanish | scene_cut | lighting | motion
    semantic: bool
    description: str


@dataclass
class ClipAnnotation:
    name: str
    path: str
    fps: float
    n_frames: int
    duration_s: float
    events: list[Event] = field(default_factory=list)
    #: state_id per frame — piecewise constant, changes only on semantic events.
    state_ids: list[int] = field(default_factory=list)
    purpose: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["events"] = [asdict(e) if not isinstance(e, dict) else e for e in self.events]
        return d

    @staticmethod
    def load(path: str | Path) -> "ClipAnnotation":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["events"] = [Event(**e) for e in d["events"]]
        return ClipAnnotation(**d)

    def state_at(self, frame_idx: int) -> int:
        i = min(max(0, frame_idx), len(self.state_ids) - 1)
        return self.state_ids[i]

    @property
    def semantic_events(self) -> list[Event]:
        return [e for e in self.events if e.semantic]


# --- frame manipulations ---------------------------------------------------------------------
def _gamma(img: np.ndarray, g: float) -> np.ndarray:
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _paste(img: np.ndarray, patch: np.ndarray, x: int, y: int) -> np.ndarray:
    out = img.copy()
    h, w = patch.shape[:2]
    y = max(0, min(y, out.shape[0] - h))
    x = max(0, min(x, out.shape[1] - w))
    out[y:y + h, x:x + w] = patch
    return out


def _make_object(rng: np.random.Generator, w: int, h: int, hue: int) -> np.ndarray:
    """A distinctly coloured, textured block — an object that clearly was not there before."""
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = 200
    hsv[..., 2] = rng.integers(120, 245, (h, w), dtype=np.uint8)
    patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    patch = cv2.GaussianBlur(patch, (7, 7), 0)
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), (25, 25, 25), 5)
    return patch


def _shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


# --- clip recipes ----------------------------------------------------------------------------
Recipe = Callable[[np.ndarray, int, float, np.random.Generator], tuple[np.ndarray, int]]
"""(frame, frame_idx, t_seconds, rng) -> (modified_frame, state_id)"""


def recipe_static(purpose: str = "control: nothing happens") -> tuple[Recipe, list[Event], str]:
    def fn(img, i, t, rng):
        return img, 0
    return fn, [], purpose


def recipe_object_events(size: tuple[int, int] = (150, 130)) -> tuple[Recipe, list[Event], str]:
    """Three unmistakable semantic events: an object appears, changes colour, then vanishes."""
    w, h = size
    events = [
        Event(6.0, "object_appear", True, "a coloured object appears on the right"),
        Event(12.0, "object_change", True, "the object changes colour"),
        Event(18.0, "object_vanish", True, "the object is removed"),
    ]

    def fn(img, i, t, rng):
        obj_rng = np.random.default_rng(11)
        if t < 6.0:
            return img, 0
        if t < 12.0:
            return _paste(img, _make_object(obj_rng, w, h, 15), img.shape[1] - w - 20, 60), 1
        if t < 18.0:
            return _paste(img, _make_object(obj_rng, w, h, 110), img.shape[1] - w - 20, 60), 2
        return img, 3

    return fn, events, "semantic events with a static camera and steady light"


def recipe_lighting_drift() -> tuple[Recipe, list[Event], str]:
    """**The false-trigger probe.** The light ramps down and back up. Nothing semantic happens.

    Any call fired on this clip after the first is a false trigger, by construction.
    """
    events = [
        Event(0.0, "lighting", False, "gamma ramps 1.0 -> 0.45 -> 1.0 over the whole clip"),
    ]

    def fn(img, i, t, rng):
        # Smooth ramp down and back, no discontinuity to key off.
        phase = t / 24.0
        g = 1.0 - 0.55 * np.sin(np.pi * phase) ** 2
        return _gamma(img, float(g)), 0

    return fn, events, "FALSE-TRIGGER PROBE: large lighting change, zero semantic change"


def recipe_rapid_motion() -> tuple[Recipe, list[Event], str]:
    """**The second false-trigger probe.** The camera shakes and pans. Nothing semantic happens."""
    events = [
        Event(0.0, "motion", False, "continuous camera shake and pan, same scene throughout"),
    ]

    def fn(img, i, t, rng):
        dx = int(38 * np.sin(t * 2.7) + 16 * np.sin(t * 11.0))
        dy = int(26 * np.cos(t * 3.3) + 10 * np.cos(t * 9.0))
        return _shift(img, dx, dy), 0

    return fn, events, "FALSE-TRIGGER PROBE: large apparent motion, zero semantic change"


def recipe_mixed(size: tuple[int, int] = (150, 130)) -> tuple[Recipe, list[Event], str]:
    """The discriminating case: lighting drift *and* real events. Can the policy tell them apart?"""
    w, h = size
    events = [
        Event(0.0, "lighting", False, "gamma drifts continuously throughout"),
        Event(8.0, "object_appear", True, "an object appears during the drift"),
        Event(16.0, "object_vanish", True, "the object is removed during the drift"),
    ]

    def fn(img, i, t, rng):
        g = 1.0 - 0.5 * np.sin(np.pi * (t / 24.0)) ** 2
        base = _gamma(img, float(g))
        obj_rng = np.random.default_rng(29)
        if 8.0 <= t < 16.0:
            return _paste(base, _make_object(obj_rng, w, h, 75), 40, img.shape[0] - h - 40), 1
        return base, 0 if t < 8.0 else 2

    return fn, events, "lighting drift WITH real events — the discriminating case"


def recipe_scene_cuts() -> tuple[Recipe, list[Event], str]:
    """Hard cuts: the easiest possible semantic events, as an upper bound on detectability."""
    events = [
        Event(8.0, "scene_cut", True, "hard cut to a heavily altered scene"),
        Event(16.0, "scene_cut", True, "hard cut back"),
    ]

    def fn(img, i, t, rng):
        if 8.0 <= t < 16.0:
            flipped = cv2.flip(img, 1)
            return _gamma(cv2.applyColorMap(cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY),
                                            cv2.COLORMAP_OCEAN), 1.1), 1
        return img, 0 if t < 8.0 else 2

    return fn, events, "hard scene cuts — an upper bound on how detectable an event can be"


def recipe_sparse_events(n_events: int, size: tuple[int, int] = (150, 130)):
    """`n_events` object appear/vanish events spread evenly over a long clip.

    Phase 8 ended by measuring *why* the scheduler thesis could not be shown: a 2-second timer on
    24-second clips with an event every 8-12 s is only 4-6x oversampled, and at that density blind
    sampling is near-optimal. This factory produces the regimes that were missing — long clips with
    sparse events, where a timer must either burn its budget on stasis or miss things.

    The object alternates colour between events, so every event is a genuine state change rather
    than a return to a state already seen.
    """
    w, h = size

    def make(duration_s: float = 300.0):
        # EVENT TIMES ARE RANDOMISED, NOT EVENLY SPACED, and that is load-bearing. An earlier
        # version spaced them evenly, which let a fixed-interval timer PHASE-LOCK to the event
        # schedule: at 4 events over 300 s, a 60 s timer fired exactly on each event and reached
        # 99% validity with 5 calls. That is an artifact of the generator, not a property of
        # timers, and it silently handed the baseline the comparison.
        #
        # Seeded, so the clips are reproducible; minimum separation, so events stay resolvable.
        gen = np.random.default_rng(4242 + n_events)
        lo, hi = 0.12 * duration_s, 0.95 * duration_s
        min_gap = 0.6 * (hi - lo) / max(1, n_events)
        times: list[float] = []
        for _ in range(2000):
            cand = sorted(gen.uniform(lo, hi, n_events))
            if n_events < 2 or min(np.diff(cand)) >= min_gap:
                times = [float(x) for x in cand]
                break
        if not times:
            times = [float(x) for x in np.linspace(lo, hi, n_events)]
        events = [
            Event(t, "object_change" if i else "object_appear", True,
                  f"object event {i + 1} of {n_events} at {t:.0f}s")
            for i, t in enumerate(times)
        ]

        def fn(img, i, t, rng):
            state = sum(1 for et in times if t >= et)
            if state == 0:
                return img, 0
            obj_rng = np.random.default_rng(100 + state)
            hue = (15 + 40 * state) % 180
            return _paste(img, _make_object(obj_rng, w, h, hue),
                          img.shape[1] - w - 20, 60), state

        return fn, events, (
            f"SPARSE REGIME: {n_events} events over {duration_s:.0f}s "
            f"({duration_s / max(1, n_events):.0f}s per event)"
        )

    return make


def recipe_long_static(duration_s: float = 300.0):
    """A long clip where nothing ever happens — the pure-stasis regime, at scale."""
    def fn(img, i, t, rng):
        return img, 0
    return fn, [], f"SPARSE REGIME: pure stasis over {duration_s:.0f}s"


RECIPES: dict[str, Callable[[], tuple[Recipe, list[Event], str]]] = {
    "static": recipe_static,
    "object_events": recipe_object_events,
    "lighting_drift": recipe_lighting_drift,
    "rapid_motion": recipe_rapid_motion,
    "mixed": recipe_mixed,
    "scene_cuts": recipe_scene_cuts,
    # Phase 9: long clips at controlled event sparsity, to reach the regimes Phase 8 showed were
    # missing. `duration_s` is supplied by build_clip.
    "long_static": recipe_long_static,
    "sparse_2": recipe_sparse_events(2),
    "sparse_4": recipe_sparse_events(4),
    "sparse_8": recipe_sparse_events(8),
    "sparse_16": recipe_sparse_events(16),
}


def build_clip(
    base_path: str | Path,
    out_dir: str | Path,
    name: str,
    recipe_name: str,
    duration_s: float = 24.0,
    fps: float = 30.0,
    seed: int = 1337,
    fourcc: str = "mp4v",
    freeze_base: bool = False,
) -> ClipAnnotation:
    """Render one annotated clip from base footage.

    `freeze_base` holds a single frame for the whole clip, giving a genuinely static background with
    light synthetic sensor noise. This isolates the variable Phase 9a found to matter: our base
    footage contains a continuously moving person, so "stasis" in those clips is not stasis at all,
    and any threshold low enough to catch a composited event also fires on the person. A frozen
    background answers the question the moving one cannot — does content-awareness pay when the
    background really is still?
    """
    base = cv2.VideoCapture(str(base_path))
    if not base.isOpened():
        raise RuntimeError(f"cannot open base footage: {base_path}")

    factory = RECIPES[recipe_name]
    try:
        recipe, events, purpose = factory(duration_s)
    except TypeError:
        recipe, events, purpose = factory()
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.mp4"

    ok, first = base.read()
    if not ok:
        raise RuntimeError("base footage produced no frames")
    h, w = first.shape[:2]
    base.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {out_path}")

    n_frames = int(round(duration_s * fps))
    state_ids: list[int] = []
    base.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Ping-pong the base footage rather than looping it. A hard loop puts a discontinuity at every
    # wrap, which reads as a novelty spike and would hand every content policy a false trigger it
    # did not earn — fatal for a 300 s clip built from 60 s of footage.
    base_frames: list[np.ndarray] = []
    while True:
        ok, f = base.read()
        if not ok:
            break
        base_frames.append(f)
    if not base_frames:
        raise RuntimeError("base footage produced no frames")
    pingpong = base_frames + base_frames[-2:0:-1] if len(base_frames) > 2 else base_frames

    frozen = base_frames[len(base_frames) // 3]
    noise_rng = np.random.default_rng(seed + 77)

    for i in range(n_frames):
        if freeze_base:
            # Static background is not a noiseless one: real sensors dither, and a perfectly
            # constant frame would hand every content policy a zero-novelty baseline no camera
            # could ever provide.
            noise = noise_rng.normal(0.0, 1.6, frozen.shape).astype(np.float32)
            frame = np.clip(frozen.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            frame = pingpong[i % len(pingpong)]
        t = i / fps
        modified, state = recipe(frame, i, t, rng)
        writer.write(modified)
        state_ids.append(int(state))

    writer.release()
    base.release()

    ann = ClipAnnotation(
        name=name,
        path=str(out_path).replace("\\", "/"),
        fps=fps,
        n_frames=len(state_ids),
        duration_s=len(state_ids) / fps,
        events=events,
        state_ids=state_ids,
        purpose=purpose,
    )
    (out_dir / f"{name}.json").write_text(json.dumps(ann.to_json(), indent=2), encoding="utf-8")
    return ann


def load_all(clip_dir: str | Path) -> list[ClipAnnotation]:
    return [ClipAnnotation.load(p) for p in sorted(Path(clip_dir).glob("*.json"))]
