"""Build the demo GIF from HUD-composited frames.

`PROMPT.md` asks for: *object swap → novelty spike → VLM fires → answer updates, HUD visible.* The
frames come from a real `peripheral.cli.demo` run over the `object_events` clip, so the GIF shows
the actual HUD and the actual decisions — not a reconstruction.

No imageio dependency: Pillow ships with matplotlib and writes animated GIFs fine.

    python -m peripheral.viz.make_gif --frames results/demo_frames \
        --out results/demo.gif --fps 10 --width 720
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def build(frames_dir: Path, out: Path, fps: float, width: int, max_frames: int,
          start: int, stride: int, colors: int = 96) -> Path:
    paths = sorted(frames_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no frames in {frames_dir} — run the demo with demo.write_frames")

    paths = paths[start::stride][:max_frames]
    images: list[Image.Image] = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if width and im.width != width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        # Adaptive palette per frame keeps the HUD text legible; a global palette smears it.
        images.append(im.convert("P", palette=Image.ADAPTIVE, colors=colors))

    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out, save_all=True, append_images=images[1:],
        duration=int(1000 / fps), loop=0, optimize=True, disposal=2,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the demo GIF from HUD frames.")
    p.add_argument("--frames", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--max-frames", type=int, default=240)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=3, help="take every Nth frame (30 fps -> 10 fps)")
    p.add_argument("--colors", type=int, default=96)
    a = p.parse_args(argv)

    out = build(a.frames, a.out, a.fps, a.width, a.max_frames, a.start, a.stride, a.colors)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

