#!/usr/bin/env python3
"""Inspect a saved TextPy/SoA text SSM checkpoint without running JAX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PARAM_PREFIX = "param_"
OPT_PREFIXES = ("opt_m_", "opt_v_")


def load_json_scalar(data: np.lib.npyio.NpzFile, key: str, default: Any) -> Any:
    if key not in data:
        return default
    return json.loads(str(data[key].item()))


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} {unit}"
        amount /= 1024.0
    return f"{value} B"


def array_summary(data: np.lib.npyio.NpzFile, prefix: str) -> dict[str, dict[str, Any]]:
    arrays: dict[str, dict[str, Any]] = {}
    for key in sorted(data.files):
        if not key.startswith(prefix):
            continue
        name = key[len(prefix) :]
        arr = data[key]
        arrays[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "count": int(arr.size),
            "bytes": int(arr.nbytes),
        }
    return arrays


def inspect_checkpoint(checkpoint: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    data = np.load(checkpoint, allow_pickle=False)
    vocab = load_json_scalar(data, "vocab_json", [])
    config = load_json_scalar(data, "config_json", {})
    metrics = load_json_scalar(data, "metrics_json", {})
    if "tokenizer_config_json" in data:
        tokenizer_config = load_json_scalar(data, "tokenizer_config_json", {"type": "word"})
    else:
        tokenizer_config = config.get("tokenizer_config", {"type": "word"})
    params = array_summary(data, PARAM_PREFIX)
    optimizer_arrays: dict[str, dict[str, Any]] = {}
    for prefix in OPT_PREFIXES:
        optimizer_arrays.update({f"{prefix}{name}": value for name, value in array_summary(data, prefix).items()})

    param_count = sum(item["count"] for item in params.values())
    param_bytes = sum(item["bytes"] for item in params.values())
    optimizer_bytes = sum(item["bytes"] for item in optimizer_arrays.values())
    total_npz_arrays = sum(int(data[key].nbytes) for key in data.files if hasattr(data[key], "nbytes"))
    return {
        "checkpoint": str(checkpoint),
        "format_version": int(data["format_version"]) if "format_version" in data else None,
        "config": config,
        "tokenizer": tokenizer_config.get("type", "word"),
        "tokenizer_config": tokenizer_config,
        "metrics": metrics,
        "vocab_size": len(vocab),
        "vocab_preview": vocab[:20],
        "param_count": param_count,
        "param_bytes": param_bytes,
        "param_bytes_human": human_bytes(param_bytes),
        "optimizer_bytes": optimizer_bytes,
        "optimizer_bytes_human": human_bytes(optimizer_bytes),
        "total_array_bytes": total_npz_arrays,
        "total_array_bytes_human": human_bytes(total_npz_arrays),
        "params": params,
        "optimizer_arrays": optimizer_arrays,
        "manifest": manifest,
    }


def load_manifest(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint(args: argparse.Namespace, manifest: dict[str, Any] | None) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    if manifest and manifest.get("promoted_checkpoint"):
        return Path(manifest["promoted_checkpoint"])
    raise SystemExit("Provide --checkpoint or --manifest with promoted_checkpoint.")


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    manifest = report.get("manifest") or {}
    config = report["config"]
    metrics = report["metrics"]
    lines: list[str] = [
        "# TextPy/SoA Text SSM Model Card",
        "",
        "## Artifact",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
    ]
    if manifest:
        lines.extend(
            [
                f"- Source checkpoint: `{manifest.get('source_checkpoint')}`",
                f"- Checkpoint role: `{manifest.get('checkpoint_role')}`",
                f"- Promoted loss: `{manifest.get('promoted_loss')}`",
                f"- Promoted accuracy: `{manifest.get('promoted_accuracy')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            *markdown_table(
                ["Field", "Value"],
                [
                    ["input_dim", config.get("input_dim")],
                    ["state_dim", config.get("state_dim")],
                    ["vocab_size", report["vocab_size"]],
                    ["tokenizer", report["tokenizer"]],
                    ["bpe_merges", len(report["tokenizer_config"].get("merges", []))],
                    ["seq_len", config.get("seq_len")],
                    ["streams", config.get("streams")],
                    ["tokens_per_stream", config.get("tokens_per_stream")],
                ],
            ),
            "",
            "## Metrics",
            "",
            *markdown_table(
                ["Metric", "Value"],
                [
                    ["loss", metrics.get("loss")],
                    ["accuracy", metrics.get("accuracy")],
                    ["mean_decay", metrics.get("mean_decay")],
                    ["updates", metrics.get("updates")],
                ],
            ),
            "",
            "## Size",
            "",
            *markdown_table(
                ["Item", "Value"],
                [
                    ["parameter_count", f"{report['param_count']:,}"],
                    ["parameter_bytes", report["param_bytes_human"]],
                    ["optimizer_bytes", report["optimizer_bytes_human"]],
                    ["total_array_bytes", report["total_array_bytes_human"]],
                ],
            ),
            "",
            "## Parameter Shapes",
            "",
            *markdown_table(
                ["Name", "Shape", "DType", "Count"],
                [
                    [name, tuple(item["shape"]), item["dtype"], f"{item['count']:,}"]
                    for name, item in report["params"].items()
                ],
            ),
            "",
            "## Vocabulary Preview",
            "",
            ", ".join(f"`{token}`" for token in report["vocab_preview"]),
            "",
            "## Run",
            "",
            "```bash",
            f"python3 run_text_checkpoint.py --checkpoint {report['checkpoint']}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def save_markdown(path: str, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")


def print_report(report: dict[str, Any]) -> None:
    print("TextPy/SoA text checkpoint inspector")
    print(f"checkpoint: {report['checkpoint']}")
    if report["manifest"]:
        manifest = report["manifest"]
        print(f"source_checkpoint: {manifest.get('source_checkpoint')}")
        print(f"checkpoint_role: {manifest.get('checkpoint_role')}")
    print("")
    print("Config")
    print(f"  input_dim: {report['config'].get('input_dim')}")
    print(f"  state_dim: {report['config'].get('state_dim')}")
    print(f"  vocab_size: {report['vocab_size']}")
    print(f"  tokenizer: {report['tokenizer']}")
    if report["tokenizer"] == "bpe":
        print(f"  bpe_merges: {len(report['tokenizer_config'].get('merges', []))}")
    print(f"  seq_len: {report['config'].get('seq_len')}")
    print(f"  streams: {report['config'].get('streams')}")
    print("")
    print("Metrics")
    print(f"  loss: {report['metrics'].get('loss')}")
    print(f"  accuracy: {report['metrics'].get('accuracy')}")
    print(f"  mean_decay: {report['metrics'].get('mean_decay')}")
    print(f"  updates: {report['metrics'].get('updates')}")
    print("")
    print("Size")
    print(f"  param_count: {report['param_count']:,}")
    print(f"  param_bytes: {report['param_bytes_human']}")
    print(f"  optimizer_bytes: {report['optimizer_bytes_human']}")
    print(f"  total_array_bytes: {report['total_array_bytes_human']}")
    print("")
    print("Parameter shapes")
    for name, item in report["params"].items():
        print(f"  {name}: {tuple(item['shape'])} {item['dtype']} ({item['count']:,})")
    print("")
    print("Vocab preview")
    print("  " + ", ".join(repr(token) for token in report["vocab_preview"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a TextPy/SoA text checkpoint.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    checkpoint = resolve_checkpoint(args, manifest)
    report = inspect_checkpoint(checkpoint, manifest)
    print_report(report)
    if args.output_json:
        save_json(args.output_json, report)
        print("")
        print(f"Saved JSON: {args.output_json}")
    if args.output_md:
        save_markdown(args.output_md, report)
        print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    run(parse_args())
