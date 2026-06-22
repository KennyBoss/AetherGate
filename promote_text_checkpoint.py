#!/usr/bin/env python3
"""Promote the best checkpoint from text experiment leaderboards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from list_text_experiments import discover_runs, sort_rows


ROOT = Path(__file__).resolve().parent


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.leaderboard:
        with open(args.leaderboard, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload["experiments"]
    else:
        rows = discover_runs(Path(args.root))
    if not rows:
        raise SystemExit("No text experiment rows found to promote.")
    return sort_rows(rows, args.sort)


def choose_checkpoint(row: dict[str, Any], role: str) -> tuple[str, float, float]:
    if role == "best":
        role = "candidate" if row["candidate_loss"] <= row["baseline_loss"] else "baseline"
    if role == "candidate":
        return row["candidate_checkpoint"], row["candidate_loss"], row["candidate_accuracy"]
    if role == "baseline":
        return row["baseline_checkpoint"], row["baseline_loss"], row["baseline_accuracy"]
    raise SystemExit(f"Unknown checkpoint role: {role}")


def run_cmd(args: list[str], timeout: int) -> str:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {proc.returncode}: {' '.join(args)}\n{proc.stdout}"
        )
    return proc.stdout


def smoke_promoted(checkpoint: Path, output_dir: Path, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    smoke_json = output_dir / "smoke_run.json"
    cmd = [
        sys.executable,
        "run_text_checkpoint.py",
        "--checkpoint",
        str(checkpoint),
        "--streams",
        str(row["streams"]),
        "--tokens-per-stream",
        str(row["tokens_per_stream"]),
        "--seq-len",
        str(row["seq_len"]),
        "--sample-steps",
        str(args.sample_steps),
        "--output-json",
        str(smoke_json),
    ]
    if args.text_file:
        cmd.extend(["--text-file", args.text_file])
    stdout = run_cmd(cmd, args.timeout)
    return {
        "command": cmd,
        "stdout_tail": stdout.splitlines()[-14:],
        "output_json": str(smoke_json),
        "result": json.loads(smoke_json.read_text(encoding="utf-8")),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> None:
    rows = load_rows(args)
    selected = rows[0]
    checkpoint_path, promoted_loss, promoted_accuracy = choose_checkpoint(selected, args.checkpoint_role)
    source_checkpoint = Path(checkpoint_path)
    if not source_checkpoint.exists():
        raise SystemExit(f"Selected checkpoint does not exist: {source_checkpoint}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    promoted_checkpoint = output_dir / args.checkpoint_name
    shutil.copy2(source_checkpoint, promoted_checkpoint)
    promoted_sha256 = sha256_file(promoted_checkpoint)
    promoted_bytes = promoted_checkpoint.stat().st_size

    manifest = {
        "created_at": timestamp(),
        "source_checkpoint": str(source_checkpoint),
        "promoted_checkpoint": str(promoted_checkpoint),
        "promoted_checkpoint_sha256": promoted_sha256,
        "promoted_checkpoint_bytes": promoted_bytes,
        "checkpoint_role": args.checkpoint_role,
        "promoted_loss": promoted_loss,
        "promoted_accuracy": promoted_accuracy,
        "selected_run": selected,
        "sort": args.sort,
        "leaderboard": args.leaderboard,
        "root": args.root,
    }
    if args.smoke_run:
        manifest["smoke_run"] = smoke_promoted(promoted_checkpoint, output_dir, selected, args)

    manifest_path = output_dir / args.manifest_name
    write_json(manifest_path, manifest)

    print("TextPy/SoA promoted text checkpoint")
    print(f"source_checkpoint: {source_checkpoint}")
    print(f"promoted_checkpoint: {promoted_checkpoint}")
    print(f"manifest: {manifest_path}")
    print(f"role: {args.checkpoint_role}")
    print(f"loss: {promoted_loss:.4f}")
    print(f"accuracy: {promoted_accuracy:.4f}")
    if args.smoke_run:
        smoke = manifest["smoke_run"]["result"]
        print(f"smoke_loss: {smoke['loss']:.4f}")
        print(f"smoke_sample: {smoke['sample']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a checkpoint from text experiments.")
    parser.add_argument("--leaderboard", default=None)
    parser.add_argument("--root", default="artifacts/text_experiments")
    parser.add_argument("--output-dir", default="artifacts/promoted/text")
    parser.add_argument("--checkpoint-name", default="text_ssm_promoted.npz")
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--checkpoint-role", choices=("best", "candidate", "baseline"), default="best")
    parser.add_argument("--sort", default="candidate_loss")
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
