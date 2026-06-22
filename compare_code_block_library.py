#!/usr/bin/env python3
"""Compare CodePy base primitives against a learned macro-block library."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_TARGETS = ("clamp01", "max_abs_x_y")
DEFAULT_PROFILES = ("base", "learned")


def run_command(cmd: list[str], timeout: int) -> tuple[str, float]:
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return completed.stdout, time.perf_counter() - start


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_from_output(pattern: str, out: str) -> float | None:
    match = re.search(pattern, out)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def run_trial(
    *,
    python: str,
    output_dir: Path,
    target: str,
    profile: str,
    seed: int,
    population: int,
    generations: int,
    elites: int,
    program_length: int,
    cases: int,
    timeout: int,
) -> dict[str, Any]:
    output_json = output_dir / f"{target}_{profile}_seed{seed}.json"
    cmd = [
        python,
        "evolve_code_blocks.py",
        "--target",
        target,
        "--block-profile",
        profile,
        "--population",
        str(population),
        "--generations",
        str(generations),
        "--elites",
        str(elites),
        "--program-length",
        str(program_length),
        "--cases",
        str(cases),
        "--log-every",
        str(max(generations, 1)),
        "--seed",
        str(seed),
        "--output-json",
        str(output_json),
    ]
    out, wall_s = run_command(cmd, timeout)
    payload = load_json(output_json)
    minimized = payload["minimized"]
    run = payload["run"]
    success = (
        float(minimized["train_mse"]) <= 1.0e-8
        and float(minimized["holdout_mse"]) <= 1.0e-8
    )
    generation = int(payload["best"]["generation"])
    first_solution = payload.get("first_solution")
    generation_found = (
        int(first_solution["generation"])
        if isinstance(first_solution, dict) and "generation" in first_solution
        else generation
        if success
        else None
    )
    return {
        "target": target,
        "profile": profile,
        "seed": seed,
        "success": success,
        "best_generation": generation,
        "generation_found": generation_found,
        "search_blocks": payload["compression"]["search_blocks"],
        "active_blocks": payload["compression"]["search_blocks"],
        "minimal_blocks": payload["compression"]["minimal_blocks"],
        "compression_ratio": payload["compression"]["compression_ratio"],
        "train_mse": minimized["train_mse"],
        "holdout_mse": minimized["holdout_mse"],
        "compile_s": run["compile_s"],
        "evolution_s": run["evolution_s"],
        "search_time_s": run["evolution_s"],
        "wall_s": wall_s,
        "evaluated_programs": run["evaluated_programs"],
        "throughput_block_ops_s": run["throughput_block_ops_s"],
        "output_json": str(output_json),
        "minimized_blocks": payload["minimized"]["compact_program_blocks"],
        "stdout_tail": "\n".join(out.splitlines()[-12:]),
        "reported_evolution_s": metric_from_output(r"evolution_s:\s+([0-9.]+)", out),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target in sorted({record["target"] for record in records}):
        target_records = [record for record in records if record["target"] == target]
        summary[target] = {}
        for profile in sorted({record["profile"] for record in target_records}):
            group = [record for record in target_records if record["profile"] == profile]
            successes = [record for record in group if record["success"]]
            divisor = max(len(group), 1)
            summary[target][profile] = {
                "trials": len(group),
                "successes": len(successes),
                "success_rate": len(successes) / divisor,
                "mean_best_generation": sum(record["best_generation"] for record in group) / divisor,
                "mean_minimal_blocks": sum(record["minimal_blocks"] for record in group) / divisor,
                "mean_evolution_s": sum(record["evolution_s"] for record in group) / divisor,
                "mean_evaluated_programs": sum(record["evaluated_programs"] for record in group) / divisor,
            }
    for target, profiles in summary.items():
        base = profiles.get("base")
        learned = profiles.get("learned")
        if base and learned:
            profiles["learned_vs_base"] = {
                "minimal_block_delta": learned["mean_minimal_blocks"] - base["mean_minimal_blocks"],
                "evolution_speedup": (
                    base["mean_evolution_s"] / learned["mean_evolution_s"]
                    if learned["mean_evolution_s"] > 0
                    else None
                ),
                "evaluated_program_ratio": (
                    learned["mean_evaluated_programs"] / base["mean_evaluated_programs"]
                    if base["mean_evaluated_programs"] > 0
                    else None
                ),
            }
    return summary


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = parse_csv(args.targets)
    profiles = parse_csv(args.profiles)
    seeds = [int(seed) for seed in parse_csv(args.seeds)]

    records: list[dict[str, Any]] = []
    print("TextPy/SoA CodePy learned-library comparison")
    print(f"targets: {', '.join(targets)}")
    print(f"profiles: {', '.join(profiles)}")
    print(f"seeds: {', '.join(map(str, seeds))}")
    print("")
    for target in targets:
        for profile in profiles:
            for seed in seeds:
                print(f"run target={target} profile={profile} seed={seed}")
                record = run_trial(
                    python=args.python,
                    output_dir=output_dir,
                    target=target,
                    profile=profile,
                    seed=seed,
                    population=args.population,
                    generations=args.generations,
                    elites=args.elites,
                    program_length=args.program_length,
                    cases=args.cases,
                    timeout=args.timeout,
                )
                records.append(record)
                print(
                    f"  success={record['success']} "
                    f"gen={record['best_generation']} "
                    f"minimal_blocks={record['minimal_blocks']} "
                    f"evolution_s={record['evolution_s']:.4f}"
                )

    payload = {
        "format_version": 1,
        "experiment": "CodePy learned macro-block library comparison",
        "guardrail": "Compares fixed safe DSL profiles; no arbitrary generated Python is executed.",
        "targets": targets,
        "profiles": profiles,
        "seeds": seeds,
        "records": records,
        "summary": summarize(records),
        "args": vars(args),
    }
    save_json(Path(args.output_json), payload)
    print("")
    print("Summary")
    for target, profile_summary in payload["summary"].items():
        print(f"  {target}")
        for profile, values in profile_summary.items():
            if profile == "learned_vs_base":
                print(
                    "    learned_vs_base: "
                    f"minimal_block_delta={values['minimal_block_delta']:.3f} "
                    f"evolution_speedup={values['evolution_speedup']:.3f}"
                )
            else:
                print(
                    f"    {profile}: success_rate={values['success_rate']:.2f} "
                    f"mean_minimal_blocks={values['mean_minimal_blocks']:.2f} "
                    f"mean_evolution_s={values['mean_evolution_s']:.4f}"
                )
    print(f"  output_json: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CodePy base vs learned macro-block profiles.")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--seeds", default="31")
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--elites", type=int, default=128)
    parser.add_argument("--program-length", type=int, default=8)
    parser.add_argument("--cases", type=int, default=49)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output-dir", default="artifacts/code_block_library")
    parser.add_argument("--output-json", default="artifacts/code_block_library/comparison.json")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
