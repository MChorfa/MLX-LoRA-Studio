#!/usr/bin/env python3
"""Run the baidu/Unlimited-OCR PyTorch reference on any backend via the device
adapter, so its (CUDA-hardcoded) inference can run on Metal / CPU / NPU for
parity against the MLX port.

    python Backend/ocr_parity.py --image doc.png --device metal
    python Backend/ocr_parity.py --image doc.png            # auto-select backend

Prints the reference OCR text and the backend/dtype it ran on. Compare the text
to the MLX output (Backend/ocr_infer.py) for functional parity.
"""

from __future__ import annotations

import argparse
import sys
import time

from device_adapter import select_adapter


def _log(message: str) -> None:
    print(f"[parity] {message}", file=sys.stderr, flush=True)


def run_reference(image: str, device: str | None, max_length: int) -> str:
    adapter = select_adapter(device)
    if not adapter.is_available():
        raise SystemExit(f"Device backend '{adapter.name}' is not available here.")
    adapter.install()  # reroute the reference's hardcoded .cuda()/bf16 calls
    _log(f"backend={adapter.name} device={adapter.device} dtype={adapter.dtype}")

    import torch
    from transformers import AutoModel, AutoTokenizer

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "baidu/Unlimited-OCR", trust_remote_code=True, torch_dtype=adapter.dtype
    )
    model = model.to(adapter.device).eval()
    _log(f"loaded in {time.monotonic() - started:.0f}s")

    import os
    out_dir = os.path.join(os.path.dirname(image) or ".", "parity_hf_out")
    os.makedirs(out_dir, exist_ok=True)
    started = time.monotonic()
    with torch.no_grad():
        text = model.infer(
            tokenizer,
            prompt="<image>\n<|grounding|>OCR this image.",
            image_file=image,
            output_path=out_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            max_length=max_length,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=False,
        )
    adapter.synchronize()
    _log(f"infer in {time.monotonic() - started:.0f}s on {adapter.name}")
    return str(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="PDF page image / picture to OCR")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "metal", "cpu", "npu"],
        help="Backend to run the reference on (default: best available)",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    text = run_reference(args.image, args.device, args.max_length)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
