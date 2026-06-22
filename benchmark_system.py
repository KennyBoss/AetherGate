#!/usr/bin/env python3
"""Benchmark harness for the SoA/SSM prototype stack.

The verifier answers "does it work?". This benchmark answers "how fast is this
snapshot, and what quality metrics did it produce?" Results are saved as JSON so
future runs can be compared mechanically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCH_DIR = ROOT / "artifacts" / "benchmarks"


PROFILES: dict[str, dict[str, list[str]]] = {
    "quick": {
        "agent": ["soa_ssm_agents.py", "--agents", "16384", "--steps", "64"],
        "text": [
            "train_ssm_text.py",
            "--streams",
            "16",
            "--tokens-per-stream",
            "256",
            "--seq-len",
            "32",
            "--epochs",
            "4",
            "--log-every",
            "2",
            "--sample-steps",
            "8",
        ],
        "evolution": [
            "evolve_ssm_agents.py",
            "--population",
            "24",
            "--elites",
            "4",
            "--agents",
            "1024",
            "--steps",
            "32",
            "--generations",
            "4",
            "--log-every",
            "2",
        ],
        "replay": [
            "run_evolved_agent.py",
            "--agents",
            "1024",
            "--steps",
            "32",
            "--seed",
            "505",
        ],
    },
    "standard": {
        "agent": ["soa_ssm_agents.py", "--agents", "262144", "--steps", "256"],
        "text": [
            "train_ssm_text.py",
            "--streams",
            "64",
            "--tokens-per-stream",
            "1024",
            "--seq-len",
            "32",
            "--epochs",
            "18",
            "--log-every",
            "6",
            "--sample-steps",
            "16",
        ],
        "evolution": [
            "evolve_ssm_agents.py",
            "--population",
            "96",
            "--elites",
            "12",
            "--agents",
            "4096",
            "--steps",
            "96",
            "--generations",
            "18",
            "--log-every",
            "6",
        ],
        "replay": [
            "run_evolved_agent.py",
            "--agents",
            "4096",
            "--steps",
            "96",
            "--seed",
            "505",
        ],
    },
}


class BenchmarkError(RuntimeError):
    pass


def run_cmd(args: list[str], timeout: int) -> tuple[str, float]:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_PLATFORMS", "cpu")
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, *args],
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
        raise BenchmarkError(
            f"Command failed with exit code {proc.returncode}: {' '.join(args)}\n{proc.stdout}"
        )
    return proc.stdout, elapsed


def find_float(pattern: str, text: str, default: float | None = None) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return default
    return float(match.group(1).replace(",", ""))


def find_int(pattern: str, text: str, default: int | None = None) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return default
    return int(match.group(1).replace(",", ""))


def find_distance(label: str, text: str) -> dict[str, float] | None:
    match = re.search(rf"{label}:\s+([0-9.]+)\s+->\s+([0-9.]+)", text)
    if not match:
        return None
    return {"initial": float(match.group(1)), "final": float(match.group(2))}


def parse_agent(out: str) -> dict[str, Any]:
    return {
        "agent_steps": find_int(r"agent_steps:\s+([0-9,]+)", out),
        "cached_run_s": find_float(r"cached_run_s:\s+([0-9.]+)", out),
        "throughput_agent_steps_s": find_float(r"throughput_agent_steps_s:\s+([0-9,]+)", out),
        "mean_energy": find_float(r"mean_energy:\s+([0-9.]+)", out),
        "mean_goal_distance": find_distance("mean_goal_distance", out),
        "mean_threat_distance": find_distance("mean_threat_distance", out),
    }


def parse_text(out: str) -> dict[str, Any]:
    return {
        "trained_tokens": find_int(r"trained_tokens:\s+([0-9,]+)", out),
        "train_s": find_float(r"train_s:\s+([0-9.]+)", out),
        "throughput_tokens_s": find_float(r"throughput_tokens_s:\s+([0-9,]+)", out),
        "initial_loss": find_float(r"initial_loss:\s+([0-9.]+)", out),
        "final_loss": find_float(r"final_loss:\s+([0-9.]+)", out),
        "final_accuracy": find_float(r"final_accuracy:\s+([0-9.]+)", out),
        "mean_decay_A": find_float(r"mean_decay_A:\s+([0-9.]+)", out),
    }


def parse_evolution(out: str) -> dict[str, Any]:
    return {
        "evaluated_agent_steps": find_int(r"evaluated_agent_steps:\s+([0-9,]+)", out),
        "evolution_s": find_float(r"evolution_s:\s+([0-9.]+)", out),
        "throughput_agent_steps_s": find_float(r"throughput_agent_steps_s:\s+([0-9,]+)", out),
        "best_reward_seen": find_float(r"best_reward_seen:\s+([0-9.]+)", out),
        "best_goal_distance": find_float(r"best_goal_distance:\s+([0-9.]+)", out),
        "best_threat_distance": find_float(r"best_threat_distance:\s+([0-9.]+)", out),
        "best_energy": find_float(r"best_energy:\s+([0-9.]+)", out),
    }


def parse_replay(out: str) -> dict[str, Any]:
    return {
        "agent_steps": find_int(r"agent_steps:\s+([0-9,]+)", out),
        "replay_s": find_float(r"replay_s:\s+([0-9.]+)", out),
        "throughput_agent_steps_s": find_float(r"throughput_agent_steps_s:\s+([0-9,]+)", out),
        "reward": find_float(r"reward:\s+([0-9.]+)", out),
        "mean_energy": find_float(r"mean_energy:\s+([0-9.]+)", out),
        "mean_goal_distance": find_distance("mean_goal_distance", out),
        "mean_threat_distance": find_distance("mean_threat_distance", out),
        "policy_entropy": find_float(r"policy_entropy:\s+([0-9.]+)", out),
    }


def snippet(out: str, max_lines: int = 16) -> str:
    lines = [line for line in out.strip().splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def run_one(name: str, command: list[str], timeout: int, parser) -> dict[str, Any]:
    out, wall_s = run_cmd(command, timeout)
    metrics = parser(out)
    print(f"{name}: wall_s={wall_s:.3f} metrics={metrics}")
    return {
        "command": [sys.executable, *command],
        "wall_s": wall_s,
        "metrics": metrics,
        "stdout_tail": snippet(out),
    }


def run(args: argparse.Namespace) -> None:
    profile = PROFILES[args.profile]
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = Path(args.output) if args.output else BENCH_DIR / f"{args.profile}_{timestamp}.json"
    genome_path = BENCH_DIR / f"{args.profile}_{timestamp}_best_genome.npz"

    print(f"Benchmark profile: {args.profile}")
    print(f"Output: {report_path}")

    results: dict[str, Any] = {
        "created_at": timestamp,
        "profile": args.profile,
        "python": sys.version,
        "root": str(ROOT),
        "results": {},
    }

    results["results"]["agent"] = run_one("agent", profile["agent"], args.timeout, parse_agent)
    results["results"]["text"] = run_one("text", profile["text"], args.timeout, parse_text)

    evolution_command = [*profile["evolution"], "--save-best", str(genome_path)]
    results["results"]["evolution"] = run_one(
        "evolution", evolution_command, args.timeout, parse_evolution
    )
    results["results"]["evolution"]["genome_path"] = str(genome_path)

    replay_command = [*profile["replay"], "--genome", str(genome_path)]
    results["results"]["replay"] = run_one("replay", replay_command, args.timeout, parse_replay)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved benchmark report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the SoA/SSM prototype stack.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except BenchmarkError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
