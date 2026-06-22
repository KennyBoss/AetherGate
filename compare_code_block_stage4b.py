#!/usr/bin/env python3
"""Measure Stage 4B success-rate scaling over hierarchy depth."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from compare_code_block_library import run_trial, save_json


STAGE4B_TASKS = {
    "max_abs_x_y": {
        "label": "max(abs(x), y)",
        "program_length": 8,
        "depth_profiles": (
            {"depth": 0, "profile": "base", "description": "base primitives"},
            {"depth": 1, "profile": "stage4_after_square", "description": "+ square"},
            {"depth": 2, "profile": "stage4_after_abs", "description": "+ abs"},
            {"depth": 3, "profile": "stage4_after_max", "description": "+ max"},
            {"depth": 4, "profile": "stage4_after_clamp", "description": "+ clamp"},
        ),
    },
    "clamp_abs_01": {
        "label": "clamp(abs(x),0,1)",
        "program_length": 8,
        "depth_profiles": (
            {"depth": 0, "profile": "base", "description": "base primitives"},
            {"depth": 1, "profile": "stage4_after_square", "description": "+ square"},
            {"depth": 2, "profile": "stage4_after_abs", "description": "+ abs"},
            {"depth": 3, "profile": "stage4_after_max", "description": "+ max"},
            {"depth": 4, "profile": "stage4_after_clamp", "description": "+ clamp"},
            {"depth": 5, "profile": "stage4_after_max_abs", "description": "+ max_abs"},
        ),
    },
    "piecewise_max_abs_x_y": {
        "label": "piecewise(max(abs,y))",
        "program_length": 10,
        "depth_profiles": (
            {"depth": 0, "profile": "base", "description": "base primitives"},
            {"depth": 1, "profile": "stage4_after_square", "description": "+ square"},
            {"depth": 2, "profile": "stage4_after_abs", "description": "+ abs"},
            {"depth": 3, "profile": "stage4_after_max", "description": "+ max"},
            {"depth": 4, "profile": "stage4_after_clamp", "description": "+ clamp"},
            {"depth": 5, "profile": "stage4_after_max_abs", "description": "+ max_abs"},
            {"depth": 6, "profile": "stage4_after_clamp_abs", "description": "+ clamp_abs"},
        ),
    },
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(seed) for seed in parse_csv(args.seeds)]
    return list(range(args.seed_start, args.seed_start + args.seed_count))


def selected_tasks(value: str) -> list[str]:
    names = parse_csv(value)
    unknown = sorted(set(names) - set(STAGE4B_TASKS))
    if unknown:
        raise SystemExit(f"Unknown Stage 4B task(s): {', '.join(unknown)}")
    return names


def mean(values: list[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def quantile(values: list[float | int | None], q: float) -> float | None:
    numeric = sorted(float(value) for value in values if value is not None)
    if not numeric:
        return None
    if len(numeric) == 1:
        return numeric[0]
    pos = (len(numeric) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(numeric) - 1)
    frac = pos - lo
    return numeric[lo] * (1.0 - frac) + numeric[hi] * frac


def summarize_depth(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [record for record in records if record["success"]]
    generations = [record["generation_found"] for record in successes]
    divisor = max(len(records), 1)
    return {
        "trials": len(records),
        "successes": len(successes),
        "success_rate": len(successes) / divisor,
        "mean_generation_found": mean(generations),
        "median_generation_found": quantile(generations, 0.5),
        "p90_generation_found": quantile(generations, 0.9),
        "mean_active_blocks": mean([record["active_blocks"] for record in successes]),
        "mean_minimal_blocks": mean([record["minimal_blocks"] for record in successes]),
        "mean_compression_ratio": mean([record["compression_ratio"] for record in successes]),
        "mean_search_time_s": mean([record["search_time_s"] for record in records]),
    }


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for target in sorted({record["target"] for record in records}):
        target_records = [record for record in records if record["target"] == target]
        task = STAGE4B_TASKS[target]
        depth_rows = []
        for depth in sorted({record["hierarchy_depth"] for record in target_records}):
            group = [record for record in target_records if record["hierarchy_depth"] == depth]
            first = group[0]
            depth_rows.append(
                {
                    "depth": depth,
                    "profile": first["profile"],
                    "description": first["depth_description"],
                    **summarize_depth(group),
                }
            )
        base = depth_rows[0] if depth_rows else None
        best = max(depth_rows, key=lambda row: row["success_rate"]) if depth_rows else None
        summary.append(
            {
                "target": target,
                "label": task["label"],
                "depths": depth_rows,
                "best_depth": best["depth"] if best else None,
                "best_success_rate": best["success_rate"] if best else None,
                "success_rate_gain_vs_depth0": (
                    best["success_rate"] - base["success_rate"] if best and base else None
                ),
            }
        )
    return summary


def svg_points(rows: list[dict[str, Any]], width: int, height: int, margin: int) -> str:
    if not rows:
        return ""
    max_depth = max(row["depth"] for row in rows) or 1
    points = []
    for row in rows:
        x = margin + (width - 2 * margin) * row["depth"] / max_depth
        y = height - margin - (height - 2 * margin) * row["success_rate"]
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def save_success_rate_svg(path: Path, summary: list[dict[str, Any]]) -> None:
    width = 920
    height = 520
    margin = 64
    colors = ("#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c")
    max_depth = max((row["depth"] for item in summary for row in item["depths"]), default=1)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="520" viewBox="0 0 920 520">',
        '<rect width="920" height="520" fill="#ffffff"/>',
        '<text x="64" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">CodePy Stage 4B: success_rate vs hierarchy_depth</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#111827" stroke-width="1.5"/>',
    ]
    for tick in range(0, max_depth + 1):
        x = margin + (width - 2 * margin) * tick / max(max_depth, 1)
        lines.append(
            f'<line x1="{x:.1f}" y1="{height - margin}" x2="{x:.1f}" y2="{height - margin + 6}" stroke="#111827"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{height - margin + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{tick}</text>'
        )
    for tick in range(0, 6):
        value = tick / 5
        y = height - margin - (height - 2 * margin) * value
        lines.append(
            f'<line x1="{margin - 6}" y1="{y:.1f}" x2="{margin}" y2="{y:.1f}" stroke="#111827"/>'
        )
        lines.append(
            f'<text x="{margin - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#374151">{value:.1f}</text>'
        )
        if tick not in (0, 5):
            lines.append(
                f'<line x1="{margin}" y1="{y:.1f}" x2="{width - margin}" y2="{y:.1f}" stroke="#e5e7eb"/>'
            )
    lines.append(
        f'<text x="{width / 2:.1f}" y="{height - 14}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">hierarchy_depth</text>'
    )
    lines.append(
        f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {height / 2:.1f})" font-family="Arial, sans-serif" font-size="14" fill="#111827">success_rate</text>'
    )
    for i, item in enumerate(summary):
        color = colors[i % len(colors)]
        points = svg_points(item["depths"], width, height, margin)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in item["depths"]:
            x = margin + (width - 2 * margin) * row["depth"] / max(max_depth, 1)
            y = height - margin - (height - 2 * margin) * row["success_rate"]
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        legend_y = 70 + i * 24
        lines.append(f'<rect x="{width - 300}" y="{legend_y - 11}" width="16" height="4" fill="{color}"/>')
        lines.append(
            f'<text x="{width - 276}" y="{legend_y - 6}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{item["label"]}</text>'
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_one_spec(args: argparse.Namespace, output_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    record_path = (
        output_dir
        / "trials"
        / f"{spec['target']}_{spec['profile']}_seed{spec['seed']}.json"
    )
    if args.resume and record_path.exists():
        with record_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        minimized = payload["minimized"]
        run = payload["run"]
        success = (
            float(minimized["train_mse"]) <= 1.0e-8
            and float(minimized["holdout_mse"]) <= 1.0e-8
        )
        first_solution = payload.get("first_solution")
        generation_found = (
            int(first_solution["generation"])
            if isinstance(first_solution, dict) and "generation" in first_solution
            else int(payload["best"]["generation"])
            if success
            else None
        )
        record = {
            "target": spec["target"],
            "profile": spec["profile"],
            "seed": spec["seed"],
            "success": success,
            "best_generation": int(payload["best"]["generation"]),
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
            "wall_s": None,
            "evaluated_programs": run["evaluated_programs"],
            "throughput_block_ops_s": run["throughput_block_ops_s"],
            "output_json": str(record_path),
            "minimized_blocks": minimized["compact_program_blocks"],
            "stdout_tail": "",
            "reported_evolution_s": None,
            "resumed": True,
        }
        record.update(
            {
                "hierarchy_depth": spec["hierarchy_depth"],
                "depth_description": spec["depth_description"],
                "label": spec["label"],
                "program_length": spec["program_length"],
            }
        )
        return record

    record = run_trial(
        python=args.python,
        output_dir=output_dir / "trials",
        target=spec["target"],
        profile=spec["profile"],
        seed=spec["seed"],
        population=args.population,
        generations=args.generations,
        elites=args.elites,
        program_length=spec["program_length"],
        cases=args.cases,
        timeout=args.timeout,
    )
    record.update(
        {
            "hierarchy_depth": spec["hierarchy_depth"],
            "depth_description": spec["depth_description"],
            "label": spec["label"],
            "program_length": spec["program_length"],
            "resumed": False,
        }
    )
    return record


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args)
    task_names = selected_tasks(args.tasks)

    specs: list[dict[str, Any]] = []
    print("TextPy/SoA CodePy Stage 4B success-rate scaling")
    print(f"tasks: {', '.join(task_names)}")
    print(f"seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")
    print(f"jobs: {args.jobs}")
    print("")

    for target in task_names:
        task = STAGE4B_TASKS[target]
        for depth_profile in task["depth_profiles"]:
            for seed in seeds:
                specs.append(
                    {
                        "target": target,
                        "profile": depth_profile["profile"],
                        "seed": seed,
                        "hierarchy_depth": depth_profile["depth"],
                        "depth_description": depth_profile["description"],
                        "label": task["label"],
                        "program_length": task["program_length"],
                    }
                )

    records: list[dict[str, Any]] = []
    if args.jobs == 1:
        for index, spec in enumerate(specs, 1):
            print(
                f"[{index}/{len(specs)}] target={spec['target']} "
                f"depth={spec['hierarchy_depth']} profile={spec['profile']} seed={spec['seed']}"
            )
            record = run_one_spec(args, output_dir, spec)
            records.append(record)
            if not args.quiet:
                print(
                    f"  success={record['success']} "
                    f"generation_found={record['generation_found']} "
                    f"minimal_blocks={record['minimal_blocks']} "
                    f"search_time_s={record['search_time_s']:.4f} "
                    f"resumed={record['resumed']}"
                )
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_one_spec, args, output_dir, spec): spec for spec in specs
            }
            for index, future in enumerate(as_completed(futures), 1):
                spec = futures[future]
                record = future.result()
                records.append(record)
                if args.quiet:
                    if index == 1 or index == len(specs) or index % args.progress_every == 0:
                        print(f"[{index}/{len(specs)}] completed")
                else:
                    print(
                        f"[{index}/{len(specs)}] target={spec['target']} "
                        f"depth={spec['hierarchy_depth']} profile={spec['profile']} seed={spec['seed']} "
                        f"success={record['success']} generation_found={record['generation_found']} "
                        f"minimal_blocks={record['minimal_blocks']} resumed={record['resumed']}"
                    )

    task_order = {name: i for i, name in enumerate(task_names)}
    records.sort(
        key=lambda record: (
            task_order[record["target"]],
            record["hierarchy_depth"],
            record["seed"],
        )
    )

    summary = summarize(records)
    if args.output_svg:
        save_success_rate_svg(Path(args.output_svg), summary)
    payload = {
        "format_version": 1,
        "experiment": "CodePy Stage 4B hierarchy-depth success-rate scaling",
        "guardrail": "Constrained register DSL search; no arbitrary generated Python is executed.",
        "records": records,
        "summary": summary,
        "output_svg": args.output_svg,
        "args": vars(args),
    }
    save_json(Path(args.output_json), payload)

    print("")
    print("Stage 4B Summary")
    for item in summary:
        print(f"  {item['label']}")
        for row in item["depths"]:
            print(
                f"    depth={row['depth']} profile={row['profile']} "
                f"success_rate={row['success_rate']:.3f} "
                f"mean_gen={row['mean_generation_found']} "
                f"mean_blocks={row['mean_minimal_blocks']}"
            )
    print(f"  output_json: {args.output_json}")
    if args.output_svg:
        print(f"  output_svg: {args.output_svg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 4B success_rate vs hierarchy_depth experiment."
    )
    parser.add_argument(
        "--tasks",
        default="max_abs_x_y,clamp_abs_01,piecewise_max_abs_x_y",
    )
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--elites", type=int, default=128)
    parser.add_argument("--cases", type=int, default=49)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output-dir", default="artifacts/code_block_stage4b")
    parser.add_argument("--output-json", default="artifacts/code_block_stage4b/stage4b.json")
    parser.add_argument("--output-svg", default="artifacts/code_block_stage4b/success_rate_vs_depth.svg")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
