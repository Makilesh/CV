"""Shared CLI front-end: the mandated flags plus Hydra config composition.

Invariant 4 mandates `--duration N --headless --metrics-out path.json`. Hydra's `@hydra.main`
decorator hijacks `sys.argv` and forces `key=value` syntax, which would break that contract
(STATUS.md decision D3). So we parse the mandated flags with argparse and compose the config
ourselves through Hydra's `compose()` API — full config-tree composition, and the runnable contract
stays exactly as specified.

Hydra overrides are still available, passed through `-o`:

    python -m peripheral.cli.smoke --duration 5 --headless \
        --metrics-out results/smoke.json -o capture=synthetic -o seed=7
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"


def config_dir() -> Path:
    """Config tree location. `PERIPHERAL_CONFIG_DIR` overrides it — no hardcoded paths."""
    return Path(os.environ.get("PERIPHERAL_CONFIG_DIR", DEFAULT_CONFIG_DIR)).resolve()


def build_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Every runnable in this repo gets these flags. Do not add a runnable without them."""
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument(
        "--duration",
        type=float,
        required=True,
        help="seconds to run; the run is bounded by this and always terminates",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="no windows, no interactive output; the only mode CI and self-evaluation use",
    )
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="path to write the metrics JSON; parent directories are created",
    )
    p.add_argument("--seed", type=int, default=None, help="override cfg.seed")
    p.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra override, repeatable (e.g. -o capture=synthetic -o capture.target_fps=15)",
    )
    p.add_argument(
        "--config-name", default="config", help="root config in configs/ (default: config)"
    )
    return p


def load_config(config_name: str, overrides: Sequence[str]) -> Any:
    """Compose the Hydra config tree without `@hydra.main`."""
    from hydra import compose, initialize_config_dir

    cfg_dir = config_dir()
    if not cfg_dir.is_dir():
        raise FileNotFoundError(
            f"config directory not found: {cfg_dir} (set PERIPHERAL_CONFIG_DIR to relocate it)"
        )
    with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.3"):
        return compose(config_name=config_name, overrides=list(overrides))


def parse_and_load(
    prog: str, description: str, argv: Sequence[str] | None = None
) -> tuple[argparse.Namespace, Any]:
    """Parse the mandated flags and return `(args, cfg)` with the flags folded into the config."""
    args = build_parser(prog, description).parse_args(argv)
    cfg = load_config(args.config_name, args.override)

    # Mandated flags win over config file values — the CLI contract is the outer authority.
    cfg.run.duration_s = args.duration
    cfg.run.headless = bool(args.headless)
    if args.metrics_out is not None:
        cfg.run.metrics_out = str(args.metrics_out)
    if args.seed is not None:
        cfg.seed = args.seed
    return args, cfg
