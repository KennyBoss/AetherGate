#!/usr/bin/env python3
"""Stage 1B trained recall-to-action wrapper for NeoCortex-X.

The inner benchmark trains the existing SSM/KV and Transformer recall models.
This wrapper maps held-out value-token recall to an environment action:
predicting value_j is treated as choosing act(value_j).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_cmd(cmd: list[str], timeout: int) -> tuple[str, int]:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return proc.stdout, proc.returncode


def build_inner_command(args: argparse.Namespace, inner_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        "compare_long_context_recall.py",
        "--task",
        "multi-assignment",
        "--output-json",
        str(inner_json),
        "--train-records",
        str(args.train_records),
        "--eval-records",
        str(args.eval_records),
        "--key-count",
        str(args.key_count),
        "--value-count",
        str(args.value_count),
        "--binding-count",
        str(args.binding_count),
        "--delay",
        str(args.delay),
        "--transformer-context",
        str(args.transformer_context),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--ssm-variant",
        args.ssm_variant,
        "--ssm-input-dim",
        str(args.ssm_input_dim),
        "--ssm-state-dim",
        str(args.ssm_state_dim),
        "--ssm-decay-init",
        str(args.ssm_decay_init),
        "--transformer-layers",
        str(args.transformer_layers),
        "--transformer-heads",
        str(args.transformer_heads),
        "--transformer-ff-mult",
        str(args.transformer_ff_mult),
        "--seed",
        str(args.seed),
        "--backend",
        args.backend,
    ]
    if args.init_assignment_kv_memory:
        cmd.append("--init-assignment-kv-memory")
    optional_flags = [
        ("--kv-read-gate-l1", args.kv_read_gate_l1),
        ("--kv-read-gate-l1-start", args.kv_read_gate_l1_start),
        ("--kv-read-gate-l1-ramp-epochs", args.kv_read_gate_l1_ramp_epochs),
        ("--kv-read-gate-target", args.kv_read_gate_target),
        ("--kv-read-gate-target-weight", args.kv_read_gate_target_weight),
        ("--kv-read-gate-entropy-weight", args.kv_read_gate_entropy_weight),
        ("--kv-read-gate-query-margin", args.kv_read_gate_query_margin),
        ("--kv-read-gate-query-margin-weight", args.kv_read_gate_query_margin_weight),
        ("--kv-read-gate-hard-eval-threshold", args.kv_read_gate_hard_eval_threshold),
    ]
    for flag, value in optional_flags:
        if value is not None and value != 0:
            cmd.extend([flag, str(value)])
    return cmd


def run(args: argparse.Namespace) -> dict[str, Any]:
    inner_json = Path(args.inner_json)
    cmd = build_inner_command(args, inner_json)
    stdout, returncode = run_cmd(cmd, args.timeout)
    if returncode != 0:
        raise SystemExit(
            "Inner trained recall benchmark failed with exit code "
            f"{returncode}:\n{stdout}"
        )
    inner = json.loads(inner_json.read_text(encoding="utf-8"))

    ssm_action_accuracy = float(inner["ssm"]["eval"]["recall_accuracy"])
    transformer_action_accuracy = float(inner["transformer"]["eval"]["recall_accuracy"])
    margin = ssm_action_accuracy - transformer_action_accuracy
    protocol = inner["protocol"]
    protocol_evidence = {
        "valid": bool(
            protocol["task"] == "multi-assignment"
            and protocol["delay"] > protocol["transformer_context"]
            and protocol["binding_count"] >= 2
            and protocol["eval_records"] >= args.eval_records
            and inner["ssm"]["variant"] == args.ssm_variant
        ),
        "inner_artifact": str(inner_json),
        "task": protocol["task"],
        "delay": protocol["delay"],
        "transformer_context": protocol["transformer_context"],
        "delay_beyond_context": protocol["delay"] > protocol["transformer_context"],
        "trained_ssm_variant": inner["ssm"]["variant"],
        "action_mapping": "value-token recall is treated as act(value_j)",
    }
    thresholds = {
        "ssm_action_accuracy_min": args.ssm_action_accuracy_min,
        "ssm_minus_transformer_action_accuracy_min": args.ssm_margin_min,
    }
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "ssm_action_accuracy": ssm_action_accuracy >= thresholds["ssm_action_accuracy_min"],
        "ssm_minus_transformer_action_accuracy": (
            margin >= thresholds["ssm_minus_transformer_action_accuracy_min"]
        ),
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage1b_trained_recall_to_action",
        "config": vars(args),
        "inner_command": cmd,
        "inner_stdout_tail": stdout.splitlines()[-40:],
        "protocol_evidence": protocol_evidence,
        "thresholds": thresholds,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "ssm_action_accuracy": ssm_action_accuracy,
            "transformer_action_accuracy": transformer_action_accuracy,
            "ssm_minus_transformer_action_accuracy": margin,
            "ssm_recall_loss": float(inner["ssm"]["eval"]["recall_loss"]),
            "transformer_recall_loss": float(inner["transformer"]["eval"]["recall_loss"]),
        },
        "claim_guardrail": (
            "Stage 1B maps trained held-out token recall to action selection for "
            "the bounded recall-to-action task. It is not a full RL policy yet."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 1B trained recall-to-action failed: {pass_fail}")
    print("NeoCortex-X Stage 1B trained recall-to-action")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  ssm_action_accuracy: {ssm_action_accuracy:.4f}")
    print(f"  transformer_action_accuracy: {transformer_action_accuracy:.4f}")
    print(f"  ssm_minus_transformer_action_accuracy: {margin:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 1B trained recall-to-action.")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage1b_trained_recall_to_action.json")
    parser.add_argument("--inner-json", default="artifacts/neocortex_x/stage1b_inner_long_context_recall.json")
    parser.add_argument("--train-records", type=int, default=512)
    parser.add_argument("--eval-records", type=int, default=256)
    parser.add_argument("--key-count", type=int, default=16)
    parser.add_argument("--value-count", type=int, default=16)
    parser.add_argument("--binding-count", type=int, default=4)
    parser.add_argument("--delay", type=int, default=32)
    parser.add_argument("--transformer-context", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--ssm-variant", default="sparse-kv-memory")
    parser.add_argument("--ssm-input-dim", type=int, default=48)
    parser.add_argument("--ssm-state-dim", type=int, default=64)
    parser.add_argument("--ssm-decay-init", type=float, default=8.0)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=2)
    parser.add_argument("--transformer-ff-mult", type=int, default=2)
    parser.add_argument("--init-assignment-kv-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kv-read-gate-l1", type=float, default=0.5)
    parser.add_argument("--kv-read-gate-l1-start", type=float, default=0.0)
    parser.add_argument("--kv-read-gate-l1-ramp-epochs", type=int, default=6)
    parser.add_argument("--kv-read-gate-target", type=float, default=0.02)
    parser.add_argument("--kv-read-gate-target-weight", type=float, default=5.0)
    parser.add_argument("--kv-read-gate-entropy-weight", type=float, default=0.01)
    parser.add_argument("--kv-read-gate-query-margin", type=float, default=0.9)
    parser.add_argument("--kv-read-gate-query-margin-weight", type=float, default=2.0)
    parser.add_argument("--kv-read-gate-hard-eval-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--ssm-action-accuracy-min", type=float, default=0.70)
    parser.add_argument("--ssm-margin-min", type=float, default=0.30)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
