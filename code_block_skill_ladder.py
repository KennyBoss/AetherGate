#!/usr/bin/env python3
"""Run a CodePy skill accumulation ladder.

Phase A searches base-DSL skills, Phase B records promoted macro-blocks, and
Phase C compares downstream synthesis with and without the learned library.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from compare_code_block_library import run_trial, save_json, summarize


PHASE_A_SKILLS = {
    "square": "r0 = square(r0)",
    "abs": "r0 = abs(r0)",
    "relu": "r0 = relu(r0)",
    "max_xy": "r0 = max(r0, r1)",
    "sign": "r0 = sign(r0)",
}

PHASE_C_TARGETS = ("clamp01", "max_abs_x_y")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def phase_a_trial(
    *,
    python: str,
    output_dir: Path,
    target: str,
    seed: int,
    population: int,
    generations: int,
    elites: int,
    program_length: int,
    cases: int,
    timeout: int,
) -> dict[str, Any]:
    output_json = output_dir / "phase_a" / f"{target}_seed{seed}.json"
    cmd = [
        python,
        "evolve_code_blocks.py",
        "--target",
        target,
        "--block-profile",
        "base",
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
    output_json.parent.mkdir(parents=True, exist_ok=True)
    out, wall_s = run_command(cmd, timeout)
    payload = load_json(output_json)
    minimized = payload["minimized"]
    success = (
        float(minimized["train_mse"]) <= 1.0e-8
        and float(minimized["holdout_mse"]) <= 1.0e-8
    )
    compression = payload["compression"]
    return {
        "target": target,
        "seed": seed,
        "success": success,
        "promoted_block": PHASE_A_SKILLS[target],
        "best_generation": payload["best"]["generation"],
        "search_blocks": compression["search_blocks"],
        "minimal_blocks": compression["minimal_blocks"],
        "compression_ratio": compression["compression_ratio"],
        "train_mse": minimized["train_mse"],
        "holdout_mse": minimized["holdout_mse"],
        "evolution_s": payload["run"]["evolution_s"],
        "wall_s": wall_s,
        "output_json": str(output_json),
        "minimized_blocks": minimized["compact_program_blocks"],
        "stdout_tail": "\n".join(out.splitlines()[-10:]),
    }


def build_library_manifest(phase_a_records: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = []
    for record in phase_a_records:
        if not record["success"]:
            continue
        promoted.append(
            {
                "name": record["target"],
                "block": record["promoted_block"],
                "source_minimized_blocks": record["minimized_blocks"],
                "compression_ratio": record["compression_ratio"],
                "promotion_reason": {
                    "success": record["success"],
                    "compression_ratio": record["compression_ratio"],
                    "minimal_blocks": record["minimal_blocks"],
                },
            }
        )
    return {
        "format_version": 1,
        "library": "CodePy learned macro-block profile",
        "promotion_policy": "Promote successful Phase A skills into atomic learned-profile blocks.",
        "promoted_blocks": promoted,
    }


def reuse_frequency(records: list[dict[str, Any]], promoted_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    names = [block["block"] for block in promoted_blocks]
    counts = {name: 0 for name in names}
    learned_records = [record for record in records if record["profile"] == "learned"]
    for record in learned_records:
        for block in record["minimized_blocks"]:
            if block in counts:
                counts[block] += 1
    divisor = max(len(learned_records), 1)
    return {
        name: {
            "count": count,
            "per_learned_trial": count / divisor,
        }
        for name, count in counts.items()
    }


def fitness_gain(summary: dict[str, Any]) -> dict[str, Any]:
    gains: dict[str, Any] = {}
    for target, target_summary in summary.items():
        base = target_summary.get("base")
        learned = target_summary.get("learned")
        if not base or not learned:
            continue
        gains[target] = {
            "success_rate_delta": learned["success_rate"] - base["success_rate"],
            "minimal_block_gain": base["mean_minimal_blocks"] - learned["mean_minimal_blocks"],
            "generation_gain": base["mean_best_generation"] - learned["mean_best_generation"],
            "evolution_speedup": (
                base["mean_evolution_s"] / learned["mean_evolution_s"]
                if learned["mean_evolution_s"] > 0
                else None
            ),
            "evaluated_program_reduction": (
                1.0 - learned["mean_evaluated_programs"] / base["mean_evaluated_programs"]
                if base["mean_evaluated_programs"] > 0
                else None
            ),
        }
    return gains


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_a_targets = parse_csv(args.phase_a_targets)
    phase_c_targets = parse_csv(args.phase_c_targets)
    seeds = [int(seed) for seed in parse_csv(args.seeds)]

    print("TextPy/SoA CodePy skill ladder")
    print(f"phase_a_targets: {', '.join(phase_a_targets)}")
    print(f"phase_c_targets: {', '.join(phase_c_targets)}")
    print(f"seeds: {', '.join(map(str, seeds))}")
    print("")

    phase_a_records: list[dict[str, Any]] = []
    for target in phase_a_targets:
        for seed in seeds:
            print(f"Phase A target={target} seed={seed}")
            record = phase_a_trial(
                python=args.python,
                output_dir=output_dir,
                target=target,
                seed=seed,
                population=args.population,
                generations=args.generations,
                elites=args.elites,
                program_length=args.phase_a_program_length,
                cases=args.phase_a_cases,
                timeout=args.timeout,
            )
            phase_a_records.append(record)
            print(
                f"  success={record['success']} "
                f"minimal_blocks={record['minimal_blocks']} "
                f"compression={record['compression_ratio']:.3f}"
            )

    library_manifest = build_library_manifest(phase_a_records)
    manifest_path = output_dir / "library_manifest.json"
    save_json(manifest_path, library_manifest)
    successful_skills = {block["name"] for block in library_manifest["promoted_blocks"]}
    missing_skills = sorted(set(phase_a_targets) - successful_skills)
    if missing_skills:
        raise SystemExit(
            "Phase A did not discover every requested skill; refusing Phase C learned-profile "
            f"comparison. Missing: {', '.join(missing_skills)}"
        )

    phase_c_records: list[dict[str, Any]] = []
    for target in phase_c_targets:
        for profile in ("base", "learned"):
            for seed in seeds:
                print(f"Phase C target={target} profile={profile} seed={seed}")
                record = run_trial(
                    python=args.python,
                    output_dir=output_dir / "phase_c",
                    target=target,
                    profile=profile,
                    seed=seed,
                    population=args.population,
                    generations=args.generations,
                    elites=args.elites,
                    program_length=args.phase_c_program_length,
                    cases=args.phase_c_cases,
                    timeout=args.timeout,
                )
                phase_c_records.append(record)
                print(
                    f"  success={record['success']} "
                    f"gen={record['best_generation']} "
                    f"minimal_blocks={record['minimal_blocks']} "
                    f"evolution_s={record['evolution_s']:.4f}"
                )

    summary = summarize(phase_c_records)
    payload = {
        "format_version": 1,
        "experiment": "CodePy skill accumulation ladder",
        "guardrail": "Safe fixed DSL experiment; learned macros are explicit interpreter primitives.",
        "phase_a": {
            "description": "Search reusable skills from base primitives.",
            "records": phase_a_records,
        },
        "phase_b": {
            "description": "Promote successful minimized skills into a learned macro-block library.",
            "library_manifest": str(manifest_path),
            "promoted_blocks": library_manifest["promoted_blocks"],
        },
        "phase_c": {
            "description": "Compare downstream synthesis with base primitives versus learned macro-blocks.",
            "profile_note": (
                "The learned interpreter profile is a fixed macro set matching the Phase A skills; "
                "this harness only runs Phase C after Phase A discovers every requested skill."
            ),
            "records": phase_c_records,
            "summary": summary,
        },
        "reuse_frequency": reuse_frequency(phase_c_records, library_manifest["promoted_blocks"]),
        "fitness_gain": fitness_gain(summary),
        "args": vars(args),
    }
    save_json(Path(args.output_json), payload)

    print("")
    print("Skill Ladder Summary")
    print(f"  promoted_blocks: {len(library_manifest['promoted_blocks'])}")
    for block in library_manifest["promoted_blocks"]:
        print(f"    {block['name']}: {block['block']}")
    print("  fitness_gain:")
    for target, gain in payload["fitness_gain"].items():
        print(
            f"    {target}: minimal_block_gain={gain['minimal_block_gain']:.3f} "
            f"generation_gain={gain['generation_gain']:.3f} "
            f"speedup={gain['evolution_speedup']:.3f}"
        )
    print(f"  output_json: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodePy find-compress-promote-reuse ladder.")
    parser.add_argument("--phase-a-targets", default="square,abs,relu,max_xy,sign")
    parser.add_argument("--phase-c-targets", default="clamp01,max_abs_x_y")
    parser.add_argument("--seeds", default="31")
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--elites", type=int, default=128)
    parser.add_argument("--phase-a-program-length", type=int, default=10)
    parser.add_argument("--phase-c-program-length", type=int, default=8)
    parser.add_argument("--phase-a-cases", type=int, default=49)
    parser.add_argument("--phase-c-cases", type=int, default=49)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output-dir", default="artifacts/code_block_skill_ladder")
    parser.add_argument("--output-json", default="artifacts/code_block_skill_ladder/skill_ladder.json")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
