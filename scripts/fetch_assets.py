"""Fetch the pinned llama.cpp build and the chosen GGUF model.

Phase 7's exit criterion is that *a fresh clone plus documented setup reaches a working live demo*.
Prose instructions cannot be checked; this can. Idempotent — existing files are verified and
skipped, so it is safe to re-run.

    .venv/Scripts/python.exe scripts/fetch_assets.py           # everything
    .venv/Scripts/python.exe scripts/fetch_assets.py --check    # verify only, download nothing

**The CUDA build is pinned to `win-cuda-13.3`, not 12.4.** This GPU is sm_120 (Blackwell) and
CUDA 12.4 predates it. The release tag is pinned too: a floating `latest` would silently change the
binary under a set of published measurements.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Pinned. RESULTS.md numbers were produced with this build.
LLAMA_TAG = "b10242"
LLAMA_ASSETS = [
    f"llama-{LLAMA_TAG}-bin-win-cuda-13.3-x64.zip",
    "cudart-llama-bin-win-cuda-13.3-x64.zip",
]
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"

# The Phase 3 chosen configuration (configs/vlm/qwen3vl_4b_q8.yaml).
MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
MODEL_FILES = [
    ("Qwen3VL-4B-Instruct-Q8_0.gguf", 3_900_000_000),
    ("mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf", 400_000_000),
]
HF_URL = "https://huggingface.co/{repo}/resolve/main/{name}"

VENDOR = REPO_ROOT / "vendor" / "llama.cpp"
MODELS = REPO_ROOT / "models" / "Qwen3-VL-4B"


def _human(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    return f"{n / 1e3:.0f} KB"


def download(url: str, dest: Path, min_bytes: int = 0) -> bool:
    """Download to `dest` unless it already looks complete. Returns True if it fetched."""
    if dest.exists() and dest.stat().st_size >= max(min_bytes, 1):
        print(f"  have  {dest.name}  ({_human(dest.stat().st_size)})")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  GET   {dest.name}")
    done = 0
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            while chunk := r.read(1 << 22):
                f.write(chunk)
                done += len(chunk)
                if total and done % (1 << 30) < (1 << 22):
                    print(f"          {_human(done)} / {_human(total)}", flush=True)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"failed to fetch {url}: HTTP {exc.code}") from exc
    tmp.rename(dest)
    print(f"  done  {dest.name}  ({_human(dest.stat().st_size)})")
    return True


def fetch_llama(check_only: bool) -> None:
    print(f"llama.cpp {LLAMA_TAG} (win-cuda-13.3)")
    server = VENDOR / "llama-server.exe"
    if server.exists():
        print(f"  have  llama-server.exe  ({_human(server.stat().st_size)})")
        return
    if check_only:
        print("  MISSING llama-server.exe")
        return

    zips = REPO_ROOT / "vendor"
    for name in LLAMA_ASSETS:
        dest = zips / name
        download(LLAMA_URL.format(tag=LLAMA_TAG, name=name), dest, min_bytes=1_000_000)
        with zipfile.ZipFile(dest) as z:
            z.extractall(VENDOR)
        print(f"  unzip {name} -> {VENDOR.relative_to(REPO_ROOT)}")

    if not server.exists():
        raise SystemExit(f"llama-server.exe not found under {VENDOR} after extraction")


def fetch_model(check_only: bool) -> None:
    print(f"model: {MODEL_REPO}")
    for name, min_bytes in MODEL_FILES:
        dest = MODELS / name
        if check_only:
            state = "have " if dest.exists() and dest.stat().st_size >= min_bytes else "MISSING"
            print(f"  {state} {name}")
            continue
        download(HF_URL.format(repo=MODEL_REPO, name=urllib.parse.quote(name)), dest, min_bytes)


def verify() -> int:
    """Report whether the demo can actually run. Exit code is the answer."""
    print("\nverification")
    problems: list[str] = []

    server = VENDOR / "llama-server.exe"
    if not server.exists():
        problems.append(f"missing {server.relative_to(REPO_ROOT)}")
    for name, min_bytes in MODEL_FILES:
        p = MODELS / name
        if not p.exists():
            problems.append(f"missing {p.relative_to(REPO_ROOT)}")
        elif p.stat().st_size < min_bytes:
            problems.append(f"{p.name} is truncated ({_human(p.stat().st_size)})")

    free = shutil.disk_usage(REPO_ROOT).free
    print(f"  disk free: {_human(free)}")
    if free < 5e9:
        problems.append(f"only {_human(free)} free; the model set needs ~5 GB")

    if problems:
        print("\nNOT READY:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("  all assets present")
    print("\nReady. Run the demo:")
    print("  .venv/Scripts/python.exe -m peripheral.cli.demo --duration 120 "
          "--metrics-out results/demo.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch pinned llama.cpp and model assets.")
    ap.add_argument("--check", action="store_true", help="verify only; download nothing")
    ap.add_argument("--skip-model", action="store_true", help="binaries only (~510 MB)")
    a = ap.parse_args(argv)

    fetch_llama(a.check)
    if not a.skip_model:
        fetch_model(a.check)
    return verify()


if __name__ == "__main__":
    sys.exit(main())
