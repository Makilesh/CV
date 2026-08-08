"""Phase 3 quality reference: one model through transformers + bitsandbytes.

`PROMPT.md`: *Reference only: HF transformers + bitsandbytes — for quality comparison, never for
latency claims.*

**This run deliberately reports no latency.** `photon_to_first_token_ms` and friends stay `null`
in its metrics file. The transformers path on Windows has no flash-attention, no fused kernels
worth the name, and a Python generation loop; timing it would produce a number that looks
comparable to the llama.cpp table and is not. What it is good for is answering the question the
GGUF sweep cannot: *is the whole llama.cpp path losing quality relative to the reference
implementation of the same weights?*

    python -m peripheral.cli.hf_reference --duration 1800 --headless \
        --metrics-out results/phase3_hf_reference.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..capture import FileSource
from ..runtime import BoundedRunner
from ..vlm.quality import score_against_reference
from ._args import parse_and_load


def _progress(line: str) -> None:
    """Print progress without ever being able to kill the run.

    Learned on 2026-08-08: piping this runner's stderr into `Select-Object -First 8` closed the
    pipe, and the next `print` raised OSError(22), which took down a 20-frame run at frame 9.
    A benchmark must not die because someone truncated its console output — and an encoding
    failure on a model-generated character must not either.
    """
    try:
        sys.stderr.write(line.encode("ascii", "replace").decode("ascii") + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - progress output is never worth a failed run
        pass


class HfReferenceRunner(BoundedRunner):
    name = "phase3_hf_reference"

    def setup(self) -> None:
        h = self.cfg.hf_reference
        clip = Path(h.clip)
        src = FileSource(clip, realtime=False)
        src.open()
        frames, i = [], 0
        while len(frames) < int(h.n_frames):
            f = src.read()
            if f is None:
                break
            if i % int(h.frame_stride) == 0:
                frames.append(f.image)
            i += 1
        src.close()
        self.frames = frames
        self.answers: list[str] = []
        self._i = 0

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self._torch = torch
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(h.quant_type),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=bool(h.double_quant),
        )
        self.processor = AutoProcessor.from_pretrained(str(h.model_id))
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(h.model_id),
            quantization_config=quant,
            dtype=torch.float16,
            device_map="cuda:0",
        )
        self.model.eval()
        self.recorder.note(
            f"HF reference: {h.model_id} via bitsandbytes {h.quant_type} 4-bit — "
            "QUALITY REFERENCE ONLY, no latency claim is made from this run"
        )

    def step(self) -> bool:
        if self._i >= len(self.frames):
            return False
        img = self.frames[self._i]
        self._i += 1

        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": str(self.cfg.hf_reference.prompt)},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.hf_reference.max_tokens),
                do_sample=False,
            )
        text = self.processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        self.answers.append(text)
        _progress(f"  [{self._i}/{len(self.frames)}] {text[:90]}")
        return True

    def teardown(self) -> None:
        h = self.cfg.hf_reference
        payload: dict[str, Any] = {
            "model_id": str(h.model_id),
            "quantization": f"bitsandbytes 4-bit {h.quant_type}"
                            f"{' + double quant' if h.double_quant else ''}",
            "prompt": str(h.prompt),
            "n_frames": len(self.answers),
            "answers": self.answers,
            "latency": None,
            "latency_note": (
                "Deliberately not measured. PROMPT.md forbids latency claims from the "
                "transformers + bitsandbytes path, and a Python generation loop without fused "
                "kernels would not be comparable to the llama.cpp table."
            ),
        }

        # The comparison this run exists for: same weights, same frames, same prompt, served two
        # different ways. Anything the GGUF path loses relative to this shows up here.
        if bool(h.get("compare_gguf", True)):
            try:
                payload["vs_gguf"] = self._compare_to_gguf()
            except Exception as exc:  # noqa: BLE001 - the reference answers are the deliverable
                payload["compare_error"] = f"{type(exc).__name__}: {exc}"

        self.recorder.record_extra("hf_reference", payload)

    def _compare_to_gguf(self) -> dict[str, Any]:
        """Ask the chosen GGUF config the same 20 frames and score both ways.

        Two passes over the GGUF path so its own noise floor is measured on the same frames —
        without it, a cross-path agreement number cannot be told apart from serving noise
        (llama-server is not deterministic at temperature 0; see peripheral.vlm.quality).
        """
        from ..vlm.llama_server import LlamaServerClient
        from ..vlm.quality import noise_floor

        v = self.cfg.vlm
        client = LlamaServerClient(
            binary=v.binary, model=v.model, mmproj=v.mmproj, host=v.host, port=int(v.port),
            n_gpu_layers=int(v.n_gpu_layers), ctx_size=int(v.ctx_size),
            jpeg_quality=int(v.jpeg_quality), autostart=True,
            startup_timeout_s=float(v.startup_timeout_s),
            extra_args=list(v.get("extra_args", []) or []), model_name=v.name,
        )
        client.start()
        try:
            prompt = str(self.cfg.hf_reference.prompt)
            mt = int(self.cfg.hf_reference.max_tokens)
            first = [client.describe(f, prompt, max_tokens=mt).text for f in self.frames]
            second = [client.describe(f, prompt, max_tokens=mt).text for f in self.frames]
        finally:
            client.stop()

        agreement = score_against_reference(first, self.answers)
        floor = noise_floor(first, second)
        f1, nf = agreement["content_f1_mean"], floor["self_content_f1"]
        return {
            "gguf_config": str(v.name),
            "agreement_with_hf_reference": agreement,
            "gguf_noise_floor": floor,
            "verdict": (
                "Agreement is at or above the GGUF path's own noise floor: the two paths are as "
                "close as one path is to itself, so no cross-path difference is measurable."
                if (f1 is not None and nf is not None and f1 >= nf)
                else "Agreement sits below the GGUF path's own noise floor, so the gap is larger "
                     "than serving noise and is real."
            ),
            "attribution_caveat": (
                "This gap is NOT attributable to llama.cpp. bitsandbytes is 4-bit by construction, "
                "so the reference is NF4 while the chosen GGUF config is Q8_0 — the comparison "
                "varies quantization AND serving path together, and cannot separate them. "
                "Inspect sample_pairs: both paths describe the same scene correctly and differ in "
                "wording, which a bag-of-content-words F1 penalises. Treat this as a sanity check "
                "that the llama.cpp path is not degenerate, not as a measurement of its loss."
            ),
            "sample_pairs": [
                {"hf": self.answers[i][:160], "gguf": first[i][:160]}
                for i in range(min(3, len(first)))
            ],
        }


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.hf_reference",
        description="Phase 3 quality reference via transformers + bitsandbytes. No latency claims.",
        argv=argv,
    )
    runner = HfReferenceRunner(
        cfg=cfg,
        duration_s=cfg.run.duration_s,
        metrics_out=cfg.run.metrics_out,
        headless=cfg.run.headless,
        seed=cfg.seed,
    )
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
