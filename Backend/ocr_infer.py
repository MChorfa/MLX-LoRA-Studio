#!/usr/bin/env python3
"""Run Unlimited-OCR (MLX) over PDFs and images — single file or recursive folders.

Renders each PDF page to an image (via PyMuPDF), runs the MLX-VLM model, and writes
one Markdown file per input next to the source (or under --output-dir). An optional
trained LoRA adapter can be applied.

Examples:
    python Backend/ocr_infer.py /path/to/doc.pdf
    python Backend/ocr_infer.py ~/Downloads --recursive --output-dir ./ocr_out
    python Backend/ocr_infer.py a.pdf b.png --adapter run/adapters/adapters.safetensors

Requires the mlx-vlm fork with the unlimited_ocr package, plus pymupdf and pillow.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def _log(message: str) -> None:
    print(f"[ocr] {message}", file=sys.stderr, flush=True)


def collect_inputs(paths: list[str], recursive: bool) -> list[Path]:
    """Expand paths into a sorted list of PDF/image files."""
    wanted = IMAGE_SUFFIXES | {".pdf"}
    found: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_file():
            if path.suffix.lower() in wanted:
                found.append(path)
        elif path.is_dir():
            globber = path.rglob if recursive else path.glob
            found.extend(p for p in globber("*") if p.suffix.lower() in wanted)
        else:
            _log(f"skip (not found): {path}")
    return sorted(set(found))


def render_pdf_pages(pdf_path: Path, dpi: int, max_pages: int | None) -> list:
    """Render PDF pages to PIL images using PyMuPDF (no poppler needed)."""
    import fitz  # PyMuPDF
    from PIL import Image

    images = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(str(pdf_path)) as doc:
        pages = range(len(doc)) if max_pages is None else range(min(max_pages, len(doc)))
        for page_index in pages:
            pix = doc[page_index].get_pixmap(matrix=matrix)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return images


def _ocr_image(model, processor, image, prompt: str, max_tokens: int) -> str:
    from mlx_vlm import generate

    result = generate(
        model=model,
        processor=processor,
        image=image,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    return getattr(result, "text", str(result))


def ocr_file(model, processor, path: Path, args, scratch: Path) -> str:
    """OCR one input file (PDF → per-page; image → single) into Markdown text."""
    from PIL import Image

    if path.suffix.lower() == ".pdf":
        pages = render_pdf_pages(path, args.dpi, args.max_pages)
        _log(f"{path.name}: {len(pages)} page(s)")
        sections = []
        for number, image in enumerate(pages, start=1):
            started = time.monotonic()
            text = _ocr_image(model, processor, image, args.prompt, args.max_tokens)
            _log(f"  page {number}/{len(pages)}: {len(text)} chars "
                 f"({time.monotonic() - started:.1f}s)")
            sections.append(f"<!-- page {number} -->\n\n{text}")
        return "\n\n---\n\n".join(sections)

    image = Image.open(path).convert("RGB")
    return _ocr_image(model, processor, image, args.prompt, args.max_tokens)


def output_path_for(path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return path.with_suffix(".md")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / (path.stem + ".md")


def load_model(model_id: str, adapter: str | None):
    from mlx_vlm import load

    model, processor = load(model_id)
    if adapter:
        from mlx_vlm.utils import apply_lora_layers

        model = apply_lora_layers(model, adapter)
        _log(f"applied LoRA adapter: {adapter}")
    return model, processor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="PDF/image files or folders")
    parser.add_argument("--recursive", action="store_true", help="Recurse into folders")
    parser.add_argument("--output-dir", default=None, help="Write .md here (else beside source)")
    parser.add_argument("--model", default="baidu/Unlimited-OCR", help="HF repo or local path")
    parser.add_argument("--adapter", default=None, help="Optional trained LoRA adapter path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction (keep <image>)")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--dpi", type=int, default=144, help="PDF render DPI")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages per PDF")
    args = parser.parse_args()

    inputs = collect_inputs(args.paths, args.recursive)
    if not inputs:
        _log("no PDF/image inputs found")
        return 1
    _log(f"{len(inputs)} input file(s)")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    scratch = Path(args.output_dir or ".").expanduser()
    model, processor = load_model(args.model, args.adapter)
    _log("model loaded")

    written = 0
    for path in inputs:
        try:
            markdown = ocr_file(model, processor, path, args, scratch)
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            _log(f"FAILED {path}: {type(exc).__name__}: {exc}")
            continue
        destination = output_path_for(path, output_dir)
        destination.write_text(markdown, encoding="utf-8")
        _log(f"wrote {destination} ({len(markdown)} chars)")
        written += 1
    _log(f"done: {written}/{len(inputs)} file(s) OCR'd")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
