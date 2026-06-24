# OCR / Multimodal fine-tuning

The **OCR / Multimodal** model family runs a *true* multimodal LoRA fine-tune on
`image → parsed-text` pairs — the target use case being document-OCR models such as
[`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR).

> **Status (read this first).**
> This is **partially implemented**. The Studio UI, run-spec contract, and backend
> spec-parsing are wired and tested. The actual training loop (an MLX-VLM image-text
> trainer plus an MLX port of the `unlimited-ocr` architecture) is **not yet active** —
> selecting this family and starting a run currently aborts with a clear
> `NotImplementedError` rather than silently mis-training. See "Implementation status"
> below.

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
  - Aliasing `model_type → deepseekocr_2` loads **89% (2417/2710)** of tensors:
    the DeepSeek-V2 MLA+MoE language stack, the SAM tower, and the projector all map.
  - **The entire port gap is one subtree:** `model.vision_model.*` — a standard
    **CLIP-L ViT** (293 tensors: embeddings + `pre_layrnorm` + 24 transformer layers).
    `deepseekocr_2` is SAM-only; `unlimited-ocr`'s `deeplip_b_l` adds the CLIP tower.
- **Pending (next, requires Apple-Silicon GPU):**
  1. MLX `unlimited_ocr` package = `deepseekocr_2` + a CLIP-L ViT vision tower
     (reuse an existing MLX-VLM CLIP impl) + the `model.*` weight-key remap.
  2. Weight conversion + a forward-pass parity check vs the HF reference.
  3. The image-text dataset loader/collator and the MLX-VLM LoRA training loop
     (`run_multimodal` in `Backend/training_runner.py`).

Until those land, the backend fails loud when this family is selected.

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
