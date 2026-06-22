#!/usr/bin/env python3
"""Run repeated-seed long-context recall comparisons."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_csv_ints(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer.")
    return seeds


def run_cmd(args: list[str], timeout: int) -> tuple[str, float]:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_PLATFORMS", "cpu")
    start = time.perf_counter()
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
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with exit code {proc.returncode}: {' '.join(args)}\n{proc.stdout}")
    return proc.stdout, elapsed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def build_inner_command(args: argparse.Namespace, seed: int, output_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        "compare_long_context_recall.py",
        "--task",
        args.task,
        "--output-json",
        str(output_json),
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
        "--query-count",
        str(args.query_count),
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
        "--grad-clip",
        str(args.grad_clip),
        "--kv-read-gate-l1",
        str(args.kv_read_gate_l1),
        "--kv-read-gate-l1-ramp-epochs",
        str(args.kv_read_gate_l1_ramp_epochs),
        "--kv-read-gate-target",
        str(args.kv_read_gate_target),
        "--kv-read-gate-target-weight",
        str(args.kv_read_gate_target_weight),
        "--kv-read-gate-entropy-weight",
        str(args.kv_read_gate_entropy_weight),
        "--kv-read-gate-query-margin",
        str(args.kv_read_gate_query_margin),
        "--kv-read-gate-query-margin-weight",
        str(args.kv_read_gate_query_margin_weight),
        "--ssm-input-dim",
        str(args.ssm_input_dim),
        "--ssm-state-dim",
        str(args.ssm_state_dim),
        "--ssm-variant",
        args.ssm_variant,
        "--ssm-skip-rank",
        str(args.ssm_skip_rank),
        "--ssm-decay-init",
        str(args.ssm_decay_init),
        "--kv-logit-scale",
        str(args.kv_logit_scale),
        "--transformer-max-d-model",
        str(args.transformer_max_d_model),
        "--transformer-layers",
        str(args.transformer_layers),
        "--transformer-heads",
        str(args.transformer_heads),
        "--transformer-ff-mult",
        str(args.transformer_ff_mult),
        "--seed",
        str(seed),
        "--backend",
        args.backend,
    ]
    if args.transformer_d_model is not None:
        cmd.extend(["--transformer-d-model", str(args.transformer_d_model)])
    if args.kv_read_gate_l1_start is not None:
        cmd.extend(["--kv-read-gate-l1-start", str(args.kv_read_gate_l1_start)])
    if args.kv_read_gate_hard_eval_threshold is not None:
        cmd.extend(["--kv-read-gate-hard-eval-threshold", str(args.kv_read_gate_hard_eval_threshold)])
    if args.allow_visible_key:
        cmd.append("--allow-visible-key")
    if args.init_recall_memory:
        cmd.append("--init-recall-memory")
    if args.init_assignment_kv_memory:
        cmd.append("--init-assignment-kv-memory")
    return cmd


def model_summary(runs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    summary = {
        "recall_accuracy": summarize([float(run[name]["eval"]["recall_accuracy"]) for run in runs]),
        "recall_loss": summarize([float(run[name]["eval"]["recall_loss"]) for run in runs]),
        "records_per_s": summarize([float(run[name]["eval"]["records_per_s"]) for run in runs]),
        "train_s": summarize([float(run[name]["train_s"]) for run in runs]),
        "parameter_count": summarize([float(run[name]["parameter_count"]) for run in runs]),
        "parameter_bytes": summarize([float(run[name]["parameter_bytes"]) for run in runs]),
    }
    if name == "ssm":
        summary.update(
            {
                "kv_read_gate_mean": summarize(
                    [float(run[name]["eval"]["kv_read_gate_mean"]) for run in runs]
                ),
                "kv_read_gate_query_mean": summarize(
                    [float(run[name]["eval"]["kv_read_gate_query_mean"]) for run in runs]
                ),
                "kv_read_gate_non_query_mean": summarize(
                    [float(run[name]["eval"]["kv_read_gate_non_query_mean"]) for run in runs]
                ),
                "kv_read_gate_query_gt_0p5_frac": summarize(
                    [float(run[name]["eval"]["kv_read_gate_query_gt_0p5_frac"]) for run in runs]
                ),
                "kv_read_gate_query_gt_0p9_frac": summarize(
                    [float(run[name]["eval"]["kv_read_gate_query_gt_0p9_frac"]) for run in runs]
                ),
                "kv_read_gate_non_query_lt_0p01_frac": summarize(
                    [float(run[name]["eval"]["kv_read_gate_non_query_lt_0p01_frac"]) for run in runs]
                ),
            }
        )
    return summary


def ssm_hard_eval_summary(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    hard_runs = [run["ssm"].get("hard_eval") for run in runs if run["ssm"].get("hard_eval") is not None]
    if not hard_runs:
        return None
    return {
        "recall_accuracy": summarize([float(run["recall_accuracy"]) for run in hard_runs]),
        "recall_loss": summarize([float(run["recall_loss"]) for run in hard_runs]),
        "records_per_s": summarize([float(run["records_per_s"]) for run in hard_runs]),
        "kv_read_gate_mean": summarize([float(run["kv_read_gate_mean"]) for run in hard_runs]),
        "kv_read_gate_query_mean": summarize([float(run["kv_read_gate_query_mean"]) for run in hard_runs]),
        "kv_read_gate_non_query_mean": summarize([float(run["kv_read_gate_non_query_mean"]) for run in hard_runs]),
        "kv_read_gate_query_gt_0p5_frac": summarize(
            [float(run["kv_read_gate_query_gt_0p5_frac"]) for run in hard_runs]
        ),
        "kv_read_gate_query_gt_0p9_frac": summarize(
            [float(run["kv_read_gate_query_gt_0p9_frac"]) for run in hard_runs]
        ),
        "kv_read_gate_non_query_lt_0p01_frac": summarize(
            [float(run["kv_read_gate_non_query_lt_0p01_frac"]) for run in hard_runs]
        ),
    }


def comparison_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    winners: dict[str, int] = {"ssm": 0, "transformer": 0, "tie": 0}
    for run in runs:
        winner = str(run["comparison"]["winner_by_recall_accuracy"])
        winners[winner] = winners.get(winner, 0) + 1
    return {
        "winner_counts": winners,
        "ssm_minus_transformer_recall_accuracy": summarize(
            [float(run["comparison"]["ssm_minus_transformer_recall_accuracy"]) for run in runs]
        ),
        "ssm_minus_transformer_recall_loss": summarize(
            [float(run["comparison"]["ssm_minus_transformer_recall_loss"]) for run in runs]
        ),
        "transformer_to_ssm_param_ratio": summarize(
            [float(run["comparison"]["transformer_to_ssm_param_ratio"]) for run in runs]
        ),
    }


def validate_protocol(args: argparse.Namespace, runs: list[dict[str, Any]]) -> dict[str, Any]:
    query_counts = [int(run["ssm"]["eval"]["query_count"]) for run in runs]
    repeated_seed_count = len(runs)
    evidence = {
        "repeated_seed_count": repeated_seed_count,
        "seed_requirement_met": repeated_seed_count >= args.min_seeds,
        "delay": int(runs[0]["protocol"]["delay"]) if runs else args.delay,
        "transformer_context": int(runs[0]["protocol"]["transformer_context"]) if runs else args.transformer_context,
        "bypass_mode": bool(runs[0]["protocol"]["bypass_mode"]) if runs else False,
        "min_query_positions": min(query_counts) if query_counts else 0,
        "query_position_requirement_met": (min(query_counts) if query_counts else 0) >= args.min_query_positions,
    }
    evidence["delay_beyond_context"] = evidence["delay"] > evidence["transformer_context"]
    evidence["valid"] = all(
        [
            evidence["seed_requirement_met"],
            evidence["query_position_requirement_met"],
            evidence["delay_beyond_context"] or args.allow_visible_key,
        ]
    )
    return evidence


def run(args: argparse.Namespace) -> None:
    seeds = parse_csv_ints(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("TextPy/SoA multi-seed long-context recall comparison", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"task: {args.task}", flush=True)
    print(f"seeds: {', '.join(str(seed) for seed in seeds)}", flush=True)
    print("", flush=True)

    runs: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        seed_json = output_dir / f"seed_{seed}.json"
        cmd = build_inner_command(args, seed, seed_json)
        print(f"seed {index}/{len(seeds)}: {seed}", flush=True)
        print("  " + " ".join(cmd), flush=True)
        stdout, elapsed = run_cmd(cmd, args.timeout)
        payload = json.loads(seed_json.read_text(encoding="utf-8"))
        runs.append(payload)
        executions.append(
            {
                "seed": seed,
                "output_json": str(seed_json),
                "elapsed_s": elapsed,
                "command": cmd,
                "stdout_tail": stdout.splitlines()[-24:],
                "winner_by_recall_accuracy": payload["comparison"]["winner_by_recall_accuracy"],
                "ssm_recall_accuracy": payload["ssm"]["eval"]["recall_accuracy"],
                "ssm_hard_recall_accuracy": (
                    None
                    if payload["ssm"].get("hard_eval") is None
                    else payload["ssm"]["hard_eval"]["recall_accuracy"]
                ),
                "transformer_recall_accuracy": payload["transformer"]["eval"]["recall_accuracy"],
            }
        )
        hard_suffix = ""
        if payload["ssm"].get("hard_eval") is not None:
            hard_suffix = " hard_acc={hard:.4f}".format(
                hard=float(payload["ssm"]["hard_eval"]["recall_accuracy"])
            )
        print(
            "  winner={winner} ssm_acc={ssm:.4f}{hard_suffix} transformer_acc={tr:.4f}".format(
                winner=payload["comparison"]["winner_by_recall_accuracy"],
                ssm=float(payload["ssm"]["eval"]["recall_accuracy"]),
                hard_suffix=hard_suffix,
                tr=float(payload["transformer"]["eval"]["recall_accuracy"]),
            ),
            flush=True,
        )

    protocol_evidence = validate_protocol(args, runs)
    if args.require_valid_protocol and not protocol_evidence["valid"]:
        raise SystemExit(f"Multi-seed long-context protocol did not meet requirements: {protocol_evidence}")

    payload = {
        "created_at": timestamp(),
        "kind": "long_context_recall_multiseed",
        "task": args.task,
        "ssm_variant": args.ssm_variant,
        "runs": executions,
        "seed_count": len(seeds),
        "seeds": seeds,
        "protocol_evidence": protocol_evidence,
        "summary": {
            "ssm": model_summary(runs, "ssm"),
            "ssm_hard_eval": ssm_hard_eval_summary(runs),
            "transformer": model_summary(runs, "transformer"),
            "comparison": comparison_summary(runs),
        },
        "claim_guardrail": (
            "This is repeated-seed evidence for a bounded synthetic long-context/code-like "
            "task. It is stronger than a single demo but not a general natural-language "
            "Transformer superiority claim."
        ),
    }
    output_json = Path(args.output_json) if args.output_json else output_dir / "summary.json"
    write_json(output_json, payload)

    summary = payload["summary"]
    comparison = summary["comparison"]
    print("", flush=True)
    print("Summary", flush=True)
    print(f"  protocol_valid: {protocol_evidence['valid']}", flush=True)
    print(f"  winner_counts: {comparison['winner_counts']}", flush=True)
    print(f"  ssm_recall_mean: {summary['ssm']['recall_accuracy']['mean']:.4f}", flush=True)
    if summary["ssm_hard_eval"] is not None:
        print(f"  ssm_hard_recall_mean: {summary['ssm_hard_eval']['recall_accuracy']['mean']:.4f}", flush=True)
        print(
            "  ssm_hard_gate_query_gt_0p9_mean: "
            f"{summary['ssm_hard_eval']['kv_read_gate_query_gt_0p9_frac']['mean']:.4f}",
            flush=True,
        )
    print(f"  transformer_recall_mean: {summary['transformer']['recall_accuracy']['mean']:.4f}", flush=True)
    print(
        f"  ssm_minus_transformer_recall_mean: {comparison['ssm_minus_transformer_recall_accuracy']['mean']:.4f}",
        flush=True,
    )
    print(f"Saved JSON: {output_json}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compare_long_context_recall.py across repeated seeds and aggregate results."
    )
    parser.add_argument("--output-dir", default="artifacts/long_context_recall_multiseed")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--seeds", default="11,17,23")
    parser.add_argument("--task", default="mixed-code")
    parser.add_argument("--train-records", type=int, default=2048)
    parser.add_argument("--eval-records", type=int, default=1024)
    parser.add_argument("--key-count", type=int, default=16)
    parser.add_argument("--value-count", type=int, default=16)
    parser.add_argument("--binding-count", type=int, default=4)
    parser.add_argument("--query-count", type=int, default=2)
    parser.add_argument("--delay", type=int, default=96)
    parser.add_argument("--transformer-context", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.006)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--kv-read-gate-l1", type=float, default=0.0)
    parser.add_argument("--kv-read-gate-l1-start", type=float, default=None)
    parser.add_argument("--kv-read-gate-l1-ramp-epochs", type=int, default=0)
    parser.add_argument("--kv-read-gate-target", type=float, default=0.0)
    parser.add_argument("--kv-read-gate-target-weight", type=float, default=0.0)
    parser.add_argument("--kv-read-gate-entropy-weight", type=float, default=0.0)
    parser.add_argument("--kv-read-gate-query-margin", type=float, default=0.0)
    parser.add_argument("--kv-read-gate-query-margin-weight", type=float, default=0.0)
    parser.add_argument("--kv-read-gate-hard-eval-threshold", type=float, default=None)
    parser.add_argument("--ssm-input-dim", type=int, default=64)
    parser.add_argument("--ssm-state-dim", type=int, default=96)
    parser.add_argument("--ssm-variant", default="kv-memory")
    parser.add_argument("--ssm-skip-rank", type=int, default=32)
    parser.add_argument("--ssm-decay-init", type=float, default=8.0)
    parser.add_argument("--kv-logit-scale", type=float, default=64.0)
    parser.add_argument("--transformer-d-model", type=int, default=None)
    parser.add_argument("--transformer-max-d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=2)
    parser.add_argument("--transformer-ff-mult", type=int, default=2)
    parser.add_argument("--allow-visible-key", action="store_true")
    parser.add_argument("--init-recall-memory", action="store_true")
    parser.add_argument("--init-assignment-kv-memory", action="store_true")
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--min-query-positions", type=int, default=1024)
    parser.add_argument("--require-valid-protocol", action="store_true")
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default="cpu",
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
