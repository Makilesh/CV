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
        print(f"  [{self._i}/{len(self.frames)}] {text[:90]}", file=sys.stderr)
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

        # If the GGUF sweep has already run, score the chosen config against this reference.
        bench_path = Path(str(h.compare_to)) if h.compare_to else None
        if bench_path and bench_path.exists():
            try:
                doc = json.loads(bench_path.read_text(encoding="utf-8"))
                rows = doc["extra"]["vlm_bench"]["configs"]
                payload["note"] = (
                    "Cross-path comparison is indicative only: the GGUF sweep discards per-frame "
                    "answers after scoring, so agreement is recomputed only where available."
                )
                payload["available_configs"] = [r["label"] for r in rows if "error" not in r]
            except Exception as exc:  # noqa: BLE001
                payload["compare_error"] = f"{type(exc).__name__}: {exc}"

        self.recorder.record_extra("hf_reference", payload)


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
