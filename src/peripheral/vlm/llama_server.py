"""llama.cpp integration via the prebuilt `llama-server` binary over localhost HTTP.

Why a subprocess and a socket instead of an in-process binding (STATUS.md decision D2):
`llama-cpp-python` publishes no CUDA wheels for Windows, so a GPU build means compiling from
source with MSVC and the CUDA toolkit, and its multimodal support lags the C++ side. That is
exactly the "unofficial thing in the critical path" this project refuses. The prebuilt
`llama-server` is official, pinnable to a release tag, and gets vision, streaming decode and KV
slot reuse for free.

**The cost is IPC, and it is inside our latency budget, not outside it.** JPEG encoding, base64,
the HTTP round trip and SSE framing all sit between the frame and the first token, so they are all
measured. `LatencyBreakdown` exists to keep that honest — a "model-only" number must never appear
in results.

Verified on this machine: llama.cpp b10242, `win-cuda-13.3` asset, RTX 5070 Ti Laptop (sm_120).
The `cuda-12.4` asset is not used — CUDA 12.4 predates Blackwell consumer silicon.
"""

from __future__ import annotations

import base64
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..telemetry.clock import now


@dataclass
class VlmResult:
    """One VLM answer, with its cost broken out. Every field is measured, none inferred."""

    text: str
    n_tokens: int | None
    encode_ms: float          # JPEG encode + base64 — our cost, not the model's
    ttft_ms: float | None     # request sent -> first token, includes HTTP + SSE
    total_ms: float           # request sent -> final token
    http_status: int = 200
    meta: dict[str, Any] = field(default_factory=dict)


class LlamaServerClient:
    """Talks to a `llama-server` instance, optionally starting and owning one."""

    def __init__(
        self,
        binary: str | Path,
        model: str | Path,
        mmproj: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        n_gpu_layers: int = 99,
        ctx_size: int = 4096,
        jpeg_quality: int = 85,
        autostart: bool = True,
        startup_timeout_s: float = 300.0,
        extra_args: list[str] | None = None,
        model_name: str | None = None,
    ) -> None:
        self.binary = Path(binary)
        self.model = Path(model)
        self.mmproj = Path(mmproj) if mmproj else None
        self.host = host
        self.port = int(port)
        self.n_gpu_layers = int(n_gpu_layers)
        self.ctx_size = int(ctx_size)
        self.jpeg_quality = int(jpeg_quality)
        self.autostart = bool(autostart)
        self.startup_timeout_s = float(startup_timeout_s)
        self.extra_args = list(extra_args or [])
        self.model_name = model_name or self.model.stem

        self.base_url = f"http://{host}:{port}"
        self._proc: subprocess.Popen | None = None
        self._owns_server = False
        self.load_time_s: float | None = None
        self.server_log: Path | None = None

    # -- lifecycle -----------------------------------------------------------------------
    def start(self) -> None:
        """Start the server, or attach to a healthy one already on the port.

        Attaching matters during development: model load is slow enough that restarting it for
        every iteration wastes minutes. A reused server is recorded in the metrics file, because
        a warm server has different first-call behaviour than a cold one.
        """
        if self._is_healthy():
            self._owns_server = False
            self.load_time_s = 0.0
            return

        if not self.autostart:
            raise RuntimeError(
                f"no healthy llama-server at {self.base_url} and autostart is disabled"
            )
        for path, what in ((self.binary, "binary"), (self.model, "model")):
            if not path.exists():
                raise FileNotFoundError(f"llama-server {what} not found: {path}")
        if self.mmproj is not None and not self.mmproj.exists():
            raise FileNotFoundError(f"mmproj not found: {self.mmproj}")

        cmd = [
            str(self.binary),
            "-m", str(self.model),
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.ctx_size),
            "--host", self.host,
            "--port", str(self.port),
            "--no-webui",
        ]
        if self.mmproj is not None:
            cmd += ["--mmproj", str(self.mmproj)]
        cmd += self.extra_args

        self.server_log = Path(f"{self.model.stem}_server.log").resolve()
        t0 = now()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._owns_server = True

        deadline = t0 + self.startup_timeout_s
        while now() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited during startup with code {self._proc.returncode}"
                )
            if self._is_healthy():
                self.load_time_s = now() - t0
                return
            time.sleep(0.25)
        self.stop()
        raise TimeoutError(f"llama-server did not become healthy within {self.startup_timeout_s}s")

    def stop(self) -> None:
        if self._proc is not None and self._owns_server:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=10)
        self._proc = None

    def _is_healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001 - "not up yet" is the expected case
            return False

    def __enter__(self) -> "LlamaServerClient":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- inference -----------------------------------------------------------------------
    def describe(
        self,
        image: np.ndarray,
        prompt: str,
        max_tokens: int = 32,
        on_first_token: Callable[[], None] | None = None,
        temperature: float = 0.0,
    ) -> VlmResult:
        """One streaming vision call.

        `on_first_token` fires the instant the first content token arrives, so the recorder can
        timestamp TTFT at the point it actually happens rather than after the response completes.
        """
        t_encode = now()
        ok, buf = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG encode failed")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        encode_ms = (now() - t_encode) * 1000.0

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": int(max_tokens),
            "temperature": temperature,
            "stream": True,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        t_send = now()
        t_first: float | None = None
        chunks: list[str] = []
        n_tokens = 0
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = resp.status
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if delta:
                    if t_first is None:
                        t_first = now()
                        if on_first_token is not None:
                            on_first_token()
                    chunks.append(delta)
                    n_tokens += 1
        t_end = now()

        return VlmResult(
            text="".join(chunks),
            n_tokens=n_tokens or None,
            encode_ms=round(encode_ms, 3),
            ttft_ms=round((t_first - t_send) * 1000.0, 3) if t_first else None,
            total_ms=round((t_end - t_send) * 1000.0, 3),
            http_status=status,
        )

    def describe_config(self) -> dict[str, Any]:
        return {
            "backend": "llama.cpp/llama-server",
            "binary": str(self.binary),
            "model": str(self.model),
            "model_name": self.model_name,
            "mmproj": str(self.mmproj) if self.mmproj else None,
            "n_gpu_layers": self.n_gpu_layers,
            "ctx_size": self.ctx_size,
            "jpeg_quality": self.jpeg_quality,
            "endpoint": self.base_url,
            "load_time_s": round(self.load_time_s, 3) if self.load_time_s is not None else None,
            "server_owned_by_run": self._owns_server,
        }


def build_client(cfg: Any) -> LlamaServerClient:
    """Construct a client from config. No hardcoded model IDs or paths."""
    v = cfg["vlm"] if not hasattr(cfg, "vlm") else cfg.vlm
    return LlamaServerClient(
        binary=v["binary"],
        model=v["model"],
        mmproj=v.get("mmproj"),
        host=v.get("host", "127.0.0.1"),
        port=v.get("port", 8080),
        n_gpu_layers=v.get("n_gpu_layers", 99),
        ctx_size=v.get("ctx_size", 4096),
        jpeg_quality=v.get("jpeg_quality", 85),
        autostart=v.get("autostart", True),
        startup_timeout_s=v.get("startup_timeout_s", 300.0),
        extra_args=list(v.get("extra_args", []) or []),
        model_name=v.get("name"),
    )
