#!/usr/bin/env python3
"""Phase 0 diagnostic + conversion harness for baidu/Unlimited-OCR on MLX.

This script does NOT yet contain a full MLX port of the ``unlimited-ocr``
architecture. Its job is to turn the port from guesswork into measurement:

  * ``inspect``    — download the repo and dump its config + safetensors weight
                     key inventory (no MLX required).
  * ``compare``    — diff the Unlimited-OCR weight keys / config against what
                     MLX-VLM's ``deepseekocr_2`` loader expects, so we know the
                     exact divergence the port must cover.
  * ``probe-load`` — attempt to load the model through MLX-VLM, optionally
                     aliasing ``model_type`` to a registered architecture, and
                     report which keys are missing / unexpected.

Run ``inspect`` + ``compare`` first and paste the output back: the missing /
unexpected key lists are the work-list for the real ``unlimited_ocr`` MLX model
package (Phase 0 of the support plan). Parity (logits vs the HF reference) is a
later gate, added once the package loads.

Usage:
    python Backend/convert_unlimited_ocr.py inspect --repo baidu/Unlimited-OCR
    python Backend/convert_unlimited_ocr.py compare --repo baidu/Unlimited-OCR
    python Backend/convert_unlimited_ocr.py probe-load --repo baidu/Unlimited-OCR \
        --alias-model-type deepseek_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Weight-key name prefixes MLX-VLM's deepseekocr_2 loader expects. Sourced by
# reading mlx_vlm/models/deepseekocr_2/* — treat as a STARTING reference, not
# ground truth; `compare` reports the actual delta against the live model.
# TODO(ckodex): refresh these against the installed mlx-vlm version before relying on them.
DEEPSEEKOCR2_EXPECTED_PREFIXES = (
    "language_model.",
    "vision_model.",
    "projector.",
)


def _log(message: str) -> None:
    print(f"[convert] {message}", file=sys.stdout, flush=True)


def _download(repo: str, revision: str | None) -> Path:
    """Snapshot-download the repo (config + weights only) and return its path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "huggingface_hub is required. Install it in the Studio env first."
        ) from exc
    local = snapshot_download(
        repo_id=repo,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.py", "tokenizer*"],
    )
    return Path(local)


def _load_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"No config.json found under {model_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _weight_keys(model_dir: Path) -> list[str]:
    """Return every tensor key across all safetensors shards (no tensor load)."""
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("safetensors is required. Install it first.") from exc

    keys: list[str] = []
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No *.safetensors shards under {model_dir}")
    for shard in shards:
        with safe_open(str(shard), framework="np") as handle:
            keys.extend(handle.keys())
    return sorted(set(keys))


def _prefix_histogram(keys: list[str], depth: int = 2) -> dict[str, int]:
    """Bucket keys by their first ``depth`` dotted segments to see structure."""
    hist: dict[str, int] = {}
    for key in keys:
        bucket = ".".join(key.split(".")[:depth])
        hist[bucket] = hist.get(bucket, 0) + 1
    return dict(sorted(hist.items()))


def cmd_inspect(args: argparse.Namespace) -> int:
    model_dir = _download(args.repo, args.revision)
    _log(f"Model snapshot: {model_dir}")
    config = _load_config(model_dir)

    _log(f"model_type: {config.get('model_type')}")
    _log(f"architectures: {config.get('architectures')}")
    for sub in ("language_config", "text_config", "vision_config", "projector_config"):
        if sub in config:
            _log(f"{sub}.keys: {sorted(config[sub])}")
    if "auto_map" in config:
        _log(f"auto_map: {config['auto_map']}")

    keys = _weight_keys(model_dir)
    _log(f"total weight tensors: {len(keys)}")
    _log("top-level key buckets (depth=2):")
    for bucket, count in _prefix_histogram(keys).items():
        print(f"    {bucket:<40} {count}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    model_dir = _download(args.repo, args.revision)
    keys = _weight_keys(model_dir)

    matched = [k for k in keys if k.startswith(DEEPSEEKOCR2_EXPECTED_PREFIXES)]
    unmatched = [k for k in keys if not k.startswith(DEEPSEEKOCR2_EXPECTED_PREFIXES)]

    _log(
        f"keys matching deepseekocr_2 prefixes: {len(matched)} / {len(keys)} "
        f"({100 * len(matched) / max(len(keys), 1):.0f}%)"
    )
    _log("UNMATCHED key buckets (these need port work):")
    for bucket, count in _prefix_histogram(unmatched).items():
        print(f"    {bucket:<40} {count}")
    _log(
        "Paste the unmatched buckets back: each is a sub-module the MLX "
        "`unlimited_ocr` package must implement or remap."
    )
    return 0


def cmd_probe_load(args: argparse.Namespace) -> int:
    model_dir = _download(args.repo, args.revision)
    if args.alias_model_type:
        _alias_model_type(model_dir, args.alias_model_type)

    try:
        from mlx_vlm.utils import load
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "mlx-vlm is required for probe-load. Install it in the Studio env."
        ) from exc

    try:
        model, _processor = load(str(model_dir), strict=False)
    except Exception as exc:  # noqa: BLE001 - we want the raw failure surfaced
        _log(f"LOAD FAILED: {type(exc).__name__}: {exc}")
        _log(
            "A failure here is expected pre-port. The exception names the first "
            "missing module / key — that is the next thing to implement."
        )
        return 1
    _log(f"LOAD OK with alias={args.alias_model_type!r}: {type(model).__name__}")
    _log("Next gate: run a forward pass and compare logits vs the HF reference.")
    return 0


def _alias_model_type(model_dir: Path, alias: str) -> None:
    """Write a sibling config with model_type rewritten, to probe the loader.

    Never overwrites the original config.json — writes config.aliased.json and
    swaps it in place only for the probe, restoring on exit.
    """
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    original = config.get("model_type")
    config["model_type"] = alias
    backup = model_dir / "config.original.json"
    if not backup.exists():
        backup.write_text(json.dumps(config | {"model_type": original}), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _log(f"Aliased model_type {original!r} -> {alias!r} (backup: {backup.name})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="baidu/Unlimited-OCR", help="HF repo id")
    parser.add_argument("--revision", default=None, help="Optional git revision")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="Dump config + weight key inventory")
    sub.add_parser("compare", help="Diff keys vs deepseekocr_2 expectations")
    probe = sub.add_parser("probe-load", help="Attempt an MLX-VLM load")
    probe.add_argument(
        "--alias-model-type",
        default=None,
        help="Rewrite config model_type to this registered arch before loading",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dispatch = {
        "inspect": cmd_inspect,
        "compare": cmd_compare,
        "probe-load": cmd_probe_load,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
