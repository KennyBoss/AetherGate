#!/usr/bin/env python3
"""Run the registered KV Scale Law 1 ladder using existing training harnesses."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SEEDS = "11,17,23"
CLAIM_GUARDRAIL = (
    "KV Scale Law 1 measures bounded MQAR recall and resource trade-offs. "
    "It is not evidence of AGI or general language-model superiority."
)


@dataclass(frozen=True)
class Point:
    name: str
    phase: str
    vocabulary: int
    bindings: int
    queries: int
    delay: int
    input_dim: int
    state_dim: int
    batch_size: int
    target_params: int | None = None

    @property
    def vocab_size(self) -> int:
        return self.vocabulary + 1  # PAD plus the shared MQAR alphabet.

    @property
    def seq_len(self) -> int:
        return self.delay + 2 * self.queries - 1


TASK_POINTS = (
    Point("task-0", "task", 16, 4, 2, 64, 64, 96, 128),
    Point("task-1", "task", 32, 6, 3, 96, 64, 96, 128),
    Point("task-2", "task", 64, 8, 4, 128, 64, 96, 128),
    Point("task-3", "task", 128, 12, 6, 192, 64, 96, 128),
)
CAPACITY_POINTS = (
    Point("cap-10k", "capacity", 128, 12, 6, 192, 32, 36, 128, 10_000),
    Point("cap-100k", "capacity", 128, 12, 6, 192, 128, 320, 32, 100_000),
    Point("cap-1m", "capacity", 128, 12, 6, 192, 256, 2_500, 4, 1_000_000),
    Point("cap-10m", "capacity", 128, 12, 6, 192, 512, 15_450, 1, 10_000_000),
)
POINTS = {point.name: point for point in (*TASK_POINTS, *CAPACITY_POINTS)}


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def kv_parameter_count(point: Point) -> int:
    """Exact count for init_params(..., variant='dynamic-kv-memory')."""
    v, i, s = point.vocab_size, point.input_dim, point.state_dim
    base_kv = i * s + v * i + v * s + 2 * s + 7 * v + 2
    dynamic_role_gates = 3 * s + 3 * i + 4
    return base_kv + dynamic_role_gates


def runtime_state_bytes(point: Point) -> int:
    """Per-sequence recurrent state + key trace + explicit KV matrix."""
    return 4 * (point.state_dim + point.input_dim + point.input_dim**2)


def transformer_cache_bytes(*, layers: int, context: int, d_model: int) -> int:
    return 4 * 2 * layers * context * d_model


def parse_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise SystemExit("At least one point or control is required.")
    return names


def selected_points(args: argparse.Namespace) -> list[Point]:
    if args.points:
        unknown = [name for name in parse_names(args.points) if name not in POINTS]
        if unknown:
            raise SystemExit(f"Unknown points: {', '.join(unknown)}")
        return [POINTS[name] for name in parse_names(args.points)]
    if args.phase == "task":
        return list(TASK_POINTS)
    if args.phase == "capacity":
        return list(CAPACITY_POINTS)
    return [*TASK_POINTS, *CAPACITY_POINTS]


def command_text(command: list[str]) -> str:
    return " ".join(command)


def run_command(command: list[str], *, timeout: int) -> tuple[str, float]:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_PLATFORMS", "cpu")
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode:
        raise RuntimeError(f"Command failed ({proc.returncode}): {command_text(command)}\n{proc.stdout}")
    return proc.stdout, elapsed


def recall_command(
    args: argparse.Namespace,
    point: Point,
    control: str,
    output_dir: Path,
) -> list[str]:
    full_context = control == "full"
    command = [
        sys.executable,
        "run_long_context_recall_multiseed.py",
        "--output-dir",
        str(output_dir),
        "--seeds",
        args.seeds,
        "--task",
        "mqar",
        "--train-records",
        str(args.train_records),
        "--eval-records",
        str(args.eval_records),
        "--key-count",
        str(point.vocabulary),
        "--value-count",
        str(point.vocabulary),
        "--binding-count",
        str(point.bindings),
        "--query-count",
        str(point.queries),
        "--delay",
        str(point.delay),
        "--transformer-context",
        str(point.seq_len if full_context else args.capped_context),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(point.batch_size if not args.smoke else min(point.batch_size, 16)),
        "--learning-rate",
        str(args.learning_rate),
        "--ssm-input-dim",
        str(point.input_dim),
        "--ssm-state-dim",
        str(point.state_dim),
        "--ssm-variant",
        "dynamic-kv-memory",
        "--transformer-max-d-model",
        str(args.transformer_max_d_model),
        "--min-seeds",
        str(1 if args.smoke else 3),
        "--min-query-positions",
        str(1 if args.smoke else args.min_query_positions),
        "--require-valid-protocol",
        "--timeout",
        str(args.inner_timeout),
        "--backend",
        args.backend,
    ]
    if full_context:
        command.append("--allow-visible-key")
    return command


def lm_command(args: argparse.Namespace, point: Point, seed: int, output_json: Path) -> list[str]:
    command = [
        sys.executable,
        "compare_text_architectures.py",
        "--output-json",
        str(output_json),
        "--streams",
        str(args.lm_streams),
        "--tokens-per-stream",
        str(args.lm_tokens_per_stream),
        "--seq-len",
        str(args.lm_seq_len),
        "--ssm-input-dim",
        str(point.input_dim),
        "--ssm-state-dim",
        str(point.state_dim),
        "--ssm-variant",
        "dynamic-kv-memory",
        "--transformer-max-d-model",
        str(args.transformer_max_d_model),
        "--epochs",
        str(args.lm_epochs),
        "--repeat",
        "1",
        "--sample-steps",
        "1",
        "--seed",
        str(seed),
        "--backend",
        args.backend,
    ]
    if args.lm_text_file:
        command.extend(["--text-file", args.lm_text_file])
    return command


def read_seed_payload(summary: dict[str, Any]) -> dict[str, Any]:
    path = Path(summary["runs"][0]["output_json"])
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def point_evidence(
    point: Point,
    full: dict[str, Any] | None,
    capped: dict[str, Any] | None,
    *,
    smoke: bool,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "point": asdict(point),
        "estimated_kv_parameter_count": kv_parameter_count(point),
        "estimated_kv_runtime_state_bytes_per_sequence": runtime_state_bytes(point),
        "full_context": full,
        "capped_context": capped,
        "valid": False,
        "pareto_positive": False,
        "status": "INCOMPLETE",
    }
    if full is None:
        evidence["reason"] = "full-context Transformer control is missing"
        return evidence

    seed_payload = read_seed_payload(full)
    summary = full["summary"]
    ratio = float(summary["comparison"]["transformer_to_ssm_param_ratio"]["mean"])
    recall_margin = float(summary["comparison"]["ssm_minus_transformer_recall_accuracy"]["mean"])
    ssm_throughput = float(summary["ssm"]["records_per_s"]["mean"])
    transformer_throughput = float(summary["transformer"]["records_per_s"]["mean"])
    d_model = int(seed_payload["transformer"]["d_model"])
    cache_bytes = transformer_cache_bytes(layers=1, context=point.seq_len, d_model=d_model)
    state_bytes = runtime_state_bytes(point)
    throughput_ratio = ssm_throughput / max(transformer_throughput, 1e-9)
    state_ratio = state_bytes / max(cache_bytes, 1)
    seeds_valid = int(full["seed_count"]) >= (1 if smoke else 3)
    full_context_valid = int(seed_payload["protocol"]["transformer_context"]) >= int(
        seed_payload["protocol"]["seq_len"]
    )
    target_valid = point.target_params is None or 0.8 <= (
        int(seed_payload["ssm"]["parameter_count"]) / point.target_params
    ) <= 1.25
    evidence.update(
        {
            "measured_kv_parameter_count": int(seed_payload["ssm"]["parameter_count"]),
            "transformer_parameter_count": int(seed_payload["transformer"]["parameter_count"]),
            "transformer_to_kv_param_ratio": ratio,
            "recall_margin": recall_margin,
            "throughput_ratio": throughput_ratio,
            "estimated_transformer_kv_cache_bytes_per_sequence": cache_bytes,
            "runtime_state_ratio": state_ratio,
            "checks": {
                "multiseed_protocol": bool(full["protocol_evidence"]["valid"]),
                "seed_count": seeds_valid,
                "full_context": full_context_valid,
                "capped_control": smoke or capped is not None,
                "parameter_match_0p8_1p25": 0.8 <= ratio <= 1.25,
                "target_match_0p8_1p25": target_valid,
            },
        }
    )
    evidence["valid"] = all(evidence["checks"].values())
    resource_advantage = throughput_ratio >= 1.10 or state_ratio <= 0.90
    evidence["pareto_positive"] = evidence["valid"] and recall_margin >= -0.01 and resource_advantage
    if smoke:
        evidence["reason"] = "smoke runs cannot promote the gate"
    elif not evidence["valid"]:
        evidence["status"] = "INVALID"
        evidence["reason"] = "one or more registered checks failed"
    else:
        evidence["status"] = "POSITIVE" if evidence["pareto_positive"] else "NEGATIVE"
    return evidence


def consecutive(values: list[bool], length: int) -> bool:
    return any(all(values[start : start + length]) for start in range(len(values) - length + 1))


def phase_status(results: list[dict[str, Any]], phase_points: tuple[Point, ...]) -> dict[str, str]:
    ordered = {result["point"]["name"]: result for result in results}
    statuses = [ordered.get(point.name, {}).get("status") for point in phase_points]
    if any(status == "INVALID" for status in statuses):
        return {"status": "INVALID", "reason": "a registered point is invalid"}
    if consecutive([status == "NEGATIVE" for status in statuses], 2):
        return {"status": "FAIL", "reason": "two adjacent valid points are negative"}
    if consecutive([status == "POSITIVE" for status in statuses], 3):
        return {"status": "PASS", "reason": "three adjacent valid points are Pareto-positive"}
    return {"status": "INCOMPLETE", "reason": "three adjacent positive points are required"}


def gate_status(results: list[dict[str, Any]], *, lm_complete: bool, mamba_complete: bool) -> dict[str, Any]:
    task_gate = phase_status(results, TASK_POINTS)
    capacity_gate = phase_status(results, CAPACITY_POINTS)
    if "INVALID" in {task_gate["status"], capacity_gate["status"]}:
        return {"status": "INVALID", "reason": "a phase contains an invalid point"}
    if "FAIL" in {task_gate["status"], capacity_gate["status"]}:
        return {"status": "FAIL", "reason": "a phase reached its stop condition"}
    if task_gate["status"] == capacity_gate["status"] == "PASS" and lm_complete and mamba_complete:
        return {"status": "PASS", "reason": "all registered promotion requirements passed"}
    return {"status": "INCOMPLETE", "reason": "more valid points, LM reports, or Mamba evidence are required"}


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def run_lm(args: argparse.Namespace, point: Point, output_dir: Path) -> dict[str, Any]:
    seeds = [int(value) for value in parse_names(args.seeds)]
    runs = []
    for seed in seeds:
        output_json = output_dir / f"seed_{seed}.json"
        command = lm_command(args, point, seed, output_json)
        reused = args.resume and output_json.exists()
        if reused:
            stdout, elapsed = "", 0.0
        else:
            stdout, elapsed = run_command(command, timeout=args.inner_timeout)
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        runs.append(
            {
                "seed": seed,
                "command": command,
                "elapsed_s": elapsed,
                "reused": reused,
                "stdout_tail": stdout.splitlines()[-16:],
                "budget_match": bool(payload["protocol"]["budget_match_80_125pct"]),
                "ssm_loss": float(payload["ssm"]["eval"]["loss"]),
                "transformer_loss": float(payload["transformer"]["eval"]["loss"]),
            }
        )
    return {
        "status": "COMPLETE" if len(runs) >= 3 and all(run["budget_match"] for run in runs) else "INCOMPLETE",
        "source": args.lm_text_file or "builtin diagnostic corpus",
        "runs": runs,
        "ssm_loss_mean": mean([run["ssm_loss"] for run in runs]),
        "transformer_loss_mean": mean([run["transformer_loss"] for run in runs]),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    points = selected_points(args)
    controls = parse_names(args.controls)
    if any(control not in {"full", "capped"} for control in controls):
        raise SystemExit("--controls accepts only full,capped")
    if args.smoke:
        points, controls = [TASK_POINTS[0]], ["full"]
        args.seeds, args.train_records, args.eval_records, args.epochs = "11", 32, 32, 1

    print("KV Scale Law 1")
    for point in points:
        target = "task-fixed" if point.target_params is None else f"target={point.target_params:,}"
        print(
            f"  {point.name}: params≈{kv_parameter_count(point):,} {target} "
            f"state={runtime_state_bytes(point):,}B seq={point.seq_len}"
        )
        for control in controls:
            command = recall_command(args, point, control, Path(args.output_dir) / point.name / control)
            print(f"    {control}: {command_text(command)}")
    if args.dry_run:
        return

    output_root = Path(args.output_dir).resolve()
    results: list[dict[str, Any]] = []
    lm_reports: dict[str, Any] = {}
    executions: list[dict[str, Any]] = []
    for point in points:
        summaries: dict[str, dict[str, Any]] = {}
        for control in controls:
            control_dir = output_root / point.name / control
            command = recall_command(args, point, control, control_dir)
            summary_path = control_dir / "summary.json"
            reused = args.resume and summary_path.exists()
            if reused:
                stdout, elapsed = "", 0.0
            else:
                stdout, elapsed = run_command(command, timeout=args.outer_timeout)
            summaries[control] = json.loads(summary_path.read_text(encoding="utf-8"))
            executions.append(
                {
                    "point": point.name,
                    "control": control,
                    "command": command,
                    "elapsed_s": elapsed,
                    "reused": reused,
                    "stdout_tail": stdout.splitlines()[-20:],
                }
            )
        results.append(
            point_evidence(point, summaries.get("full"), summaries.get("capped"), smoke=args.smoke)
        )
        if point.phase == "capacity" and not args.skip_lm:
            lm_reports[point.name] = run_lm(args, point, output_root / point.name / "lm")

    lm_required = [point.name for point in points if point.phase == "capacity"]
    lm_complete = bool(lm_required) and all(
        lm_reports.get(name, {}).get("status") == "COMPLETE" for name in lm_required
    )
    mamba = {
        "status": "MISSING",
        "reason": "No validated equivalent-budget Mamba implementation is installed in the JAX/CPU harness.",
    }
    gate = gate_status(results, lm_complete=lm_complete, mamba_complete=False)
    payload = {
        "created_at": timestamp(),
        "kind": "kv_scale_law_1",
        "smoke": args.smoke,
        "registry": [asdict(point) for point in points],
        "executions": executions,
        "points": results,
        "ordinary_lm": lm_reports,
        "mamba": mamba,
        "phase_gates": {
            "task": phase_status(results, TASK_POINTS),
            "capacity": phase_status(results, CAPACITY_POINTS),
        },
        "gate": gate,
        "claim_guardrail": CLAIM_GUARDRAIL,
    }
    summary_path = output_root / "kv_scale_law_1.json"
    write_json(summary_path, payload)
    print(f"gate: {gate['status']} — {gate['reason']}")
    print(f"Saved JSON: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/kv_scale_law_1")
    parser.add_argument("--phase", choices=("task", "capacity", "all"), default="task")
    parser.add_argument("--points", default=None, help="Comma-separated registered point names.")
    parser.add_argument("--controls", default="full,capped")
    parser.add_argument("--seeds", default=SEEDS)
    parser.add_argument("--train-records", type=int, default=2048)
    parser.add_argument("--eval-records", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.006)
    parser.add_argument("--capped-context", type=int, default=32)
    parser.add_argument("--min-query-positions", type=int, default=1024)
    parser.add_argument("--transformer-max-d-model", type=int, default=4096)
    parser.add_argument("--lm-text-file", default=None)
    parser.add_argument("--lm-streams", type=int, default=4)
    parser.add_argument("--lm-tokens-per-stream", type=int, default=96)
    parser.add_argument("--lm-seq-len", type=int, default=16)
    parser.add_argument("--lm-epochs", type=int, default=2)
    parser.add_argument("--skip-lm", action="store_true")
    parser.add_argument("--inner-timeout", type=int, default=900)
    parser.add_argument("--outer-timeout", type=int, default=3600)
    parser.add_argument("--backend", choices=("cpu", "metal", "gpu", "tpu", "auto"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed summaries under --output-dir.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
