# OCR / Multimodal fine-tuning

The **OCR / Multimodal** model family runs a *true* multimodal LoRA fine-tune on
`image → parsed-text` pairs — the target use case being document-OCR models such as
[`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR).

> **Status (read this first).**
> Image-text LoRA fine-tuning is **wired end-to-end and verified**: a short run on
> `baidu/Unlimited-OCR` completes with decreasing loss and live metrics, using the
> MLX-VLM trainer plus the `unlimited_ocr` package in the pinned mlx-vlm fork. It
> requires that fork installed (see `Backend/requirements.txt`); without it the
> backend fails loud. The vision tower is non-trainable in MLX (its custom kernel
> has no backward), so OCR LoRA trains the language model with vision frozen.

## How it differs from the Vision-Language family

The existing **Vision-Language** family is *text-only*: it LoRA-trains the language
backbone and re-grafts those weights into an untouched vision tower. No images flow
through training, so it cannot adapt OCR/vision behavior.

The **OCR / Multimodal** family is different: images and their parsed-text targets both
flow through training, so the model actually learns the document-parsing task.

## Dataset format

Baidu ships no fine-tuning recipe, so the Studio defines its own schema. Provide a
folder containing `train.jsonl` (and optionally `valid.jsonl` / `test.jsonl`), where
each row is:

```json
{ "image": "invoices/0001.png", "prompt": "<image>document parsing.", "target": "..." }
```

- `image` — path relative to the **Image root** folder set in the UI. For multi-page
  documents, an array of paths is also accepted.
- `prompt` — the instruction. The `<image>` token marks where the image is injected;
  the default template is `<image>document parsing.`.
- `target` — the expected parsed text. Loss is masked to the target tokens only; the
  prompt is not trained on.

## Inference modes

Mirrors the model's two upstream profiles:

| Mode | Settings | Use |
| --- | --- | --- |
| **Gundam** | `base_size=1024`, `image_size=640` | Single images |
| **Base** | `image_size=1024` | Multi-page documents / PDFs |

## Memory guidance

A 3B MoE language model plus dual vision towers under LoRA is heavy on unified memory.
Keep **batch size small**, consider **QLoRA** (4/6/8-bit), and keep the vision tower
**frozen** (train language + projector only) unless you have headroom. The Studio's
resource guard will abort a run before it destabilizes macOS.

## Run-spec keys

The UI emits these keys into the run spec (consumed by `Backend/training_runner.py`):

| Key | Meaning |
| --- | --- |
| `model_family` | `"multimodal"` |
| `image_root` | Folder that dataset image paths are relative to |
| `prompt_template` | Instruction template containing `<image>` |
| `freeze_vision_tower` | Train language + projector only when `true` |
| `ocr_inference_mode` | `"gundam"` or `"base"` |

## Implementation status

- **Done & tested:** `ModelFamily.multimodalOCR` in the UI, the run-spec round-trip
  (Swift `TrainingConfig` ⇄ JSON), and `_normalize_spec` handling in the backend.
- **Phase 0 measured (GO verdict).** Running `Backend/convert_unlimited_ocr.py`
  against the real model established:
  - `model_type: unlimited-ocr`, arch `UnlimitedOCRForCausalLM`; 2710 weight tensors.
  - The original **`deepseekocr` (v1)** MLX package (SAM + CLIP DeepEncoder) is the
    correct base: aliasing `model_type → deepseekocr` and calling `load_model`
    returns a `Model` with **no missing/extra parameters** — the full DeepSeek-V2
    MLA+MoE language stack, SAM tower, CLIP tower, and projector all map.
    (`deepseekocr_2` dropped CLIP, so it left a 293-tensor gap; v1 does not.)
- **Phase 1 done (architecture load, verified).** An `unlimited_ocr` package was
  added to a fork of mlx-vlm (branch `feat/unlimited-ocr`): it re-exports the v1
  `deepseekocr` modules, and a `MODEL_REMAPPING` entry resolves the hyphenated
  `model_type`. `load_model` on the real checkpoint (no alias) returns a `Model`
  with a clean weight load. See `Backend/requirements.txt` for the fork pin.
- **Processor + end-to-end OCR done (verified).** Added `UnlimitedOCRProcessor`
  (a `DeepseekOCRProcessor` subclass that forces `trust_remote_code=False`, since
  the raw HF repo's custom processor pulls torch-only remote code MLX never needs),
  registered via the AutoProcessor patch. Full `mlx_vlm.load()` returns
  `Model + UnlimitedOCRProcessor`, and `generate()` on a synthetic document
  recovers the rendered text exactly, with DeepSeek-OCR grounding boxes. This is
  functional OCR validation (not bit-exact logit parity vs the torch reference).
- **Trainer done (verified).** `run_multimodal` in `Backend/training_runner.py`
  loads the dataset, builds the MLX-VLM `TrainingArgs`, runs LoRA training via the
  MLX-VLM `train()` loop, and tees its output into `@@studio_metric` events for
  Live Metrics. Verified: a 2-step run on a synthetic document fixture completed
  with loss 2.410 -> 2.329 and saved adapters.
- **Optional next:** weight conversion (`mlx_vlm.convert`) to publish a quantized
  MLX repo; bit-exact logit parity vs the HF reference; multi-image batch_size > 1
  (currently the multi-crop collation targets batch_size 1).

## In the app: the OCR page

The Studio has a dedicated **OCR** sidebar page (`OCRView`). Pick a PDF/image file
or a folder (with a **Recurse** toggle), choose the model (default
`baidu/Unlimited-OCR`) and an optional trained LoRA adapter, tune options (prompt,
max tokens, PDF DPI, max pages, repetition penalty), and press **Run OCR**. Output
Markdown files are collected in a run folder (or a folder you choose), and progress
streams live in the run console. Under the hood this launches `Backend/ocr_infer.py`
via the same job runner used for training. The OCR model needs the mlx-vlm fork
installed in the selected Python environment (see `Backend/requirements.txt`).

## Running OCR on PDFs and images (CLI)

`Backend/ocr_infer.py` runs Unlimited-OCR over a single file, multiple files, or
folders (optionally recursive). PDFs are rendered to page images with PyMuPDF and
OCR'd page-by-page; output is one Markdown file per input (beside the source or
under `--output-dir`). An optional trained LoRA adapter can be applied.

```bash
# Single PDF
python Backend/ocr_infer.py /path/to/doc.pdf

# A whole tree of PDFs/images, into an output folder
python Backend/ocr_infer.py ~/Downloads --recursive --output-dir ./ocr_out

# With a trained adapter, capping pages for a quick check
python Backend/ocr_infer.py paper.pdf --adapter run/adapters/adapters.safetensors --max-pages 2
```

Flags: `--recursive`, `--output-dir`, `--model`, `--adapter`, `--prompt`
(keep the `<image>` token), `--max-tokens`, `--dpi` (PDF render), `--max-pages`.
Needs the mlx-vlm fork plus `pymupdf` and `pillow`. Verified on a real arXiv PDF:
title, authors, abstract, and headings extracted with grounding boxes.

## Running the Phase 0 diagnostic

The port is driven by measurement, not guesswork. In the Studio's Python env
(`mlx-vlm`, `huggingface_hub`, `safetensors` installed), run:

```bash
# 1. Dump config + weight-key inventory (no MLX needed)
python Backend/convert_unlimited_ocr.py inspect --repo baidu/Unlimited-OCR

# 2. Diff the weight keys against deepseekocr_2's expectations
python Backend/convert_unlimited_ocr.py compare --repo baidu/Unlimited-OCR

# 3. Attempt an MLX-VLM load, aliasing the model_type to a registered arch
python Backend/convert_unlimited_ocr.py probe-load --repo baidu/Unlimited-OCR \
    --alias-model-type deepseek_v2
```

The **unmatched key buckets** from `compare` and the **first failure** from
`probe-load` are the concrete work-list for the MLX `unlimited_ocr` package.
