#!/usr/bin/env python3
"""Run a suite of TextPy/SoA hidden-memory probes and pairwise comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from compare_text_memory import compare_rows, cosine_similarity, l2_distance, mean
from probe_text_memory import probe, resolve_args, save_csv as save_probe_csv, save_json as save_probe_json


DEFAULT_PROMPTS = [
    "memory is a river",
    "memory is another world",
    "the future is guessed",
]


def slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:48] or "prompt"


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    for prompt in args.prompt:
        if prompt.strip():
            prompts.append(prompt.strip())
    if args.prompts_file:
        path = Path(args.prompts_file)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    if not prompts:
        prompts = list(DEFAULT_PROMPTS)
    if len(prompts) < 2:
        raise SystemExit("Need at least two prompts for a memory suite.")
    return prompts


def probe_prompt(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    probe_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        release=args.release,
        prompt=prompt,
        include_bos=args.include_bos,
        include_state_vectors=args.include_state_vectors,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        output_json=None,
        output_csv=None,
        backend=args.backend,
    )
    probe_args = resolve_args(probe_args)
    return probe(probe_args)


def compare_variable_length(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    shared_steps = min(len(left["steps"]), len(right["steps"]))
    rows = [
        compare_rows(base, candidate)
        for base, candidate in zip(left["steps"][:shared_steps], right["steps"][:shared_steps])
    ]
    token_match_rate = mean([1.0 if row["same_token"] else 0.0 for row in rows])
    top_match_rate = mean([1.0 if row["same_top_prediction"] else 0.0 for row in rows])
    overlaps = [float(row["top_prediction_overlap"]) for row in rows]
    norm_deltas = [float(row["abs_state_norm_delta"]) for row in rows]
    state_delta_deltas = [float(row["abs_state_delta_norm_delta"]) for row in rows]
    prob_deltas = [float(row["abs_top_probability_delta"]) for row in rows]
    vector_rows = [row for row in rows if "state_l2_distance" in row and "state_cosine_similarity" in row]
    has_state_vectors = len(vector_rows) == len(rows)
    l2_distances = [float(row["state_l2_distance"]) for row in vector_rows]
    cosine_similarities = [float(row["state_cosine_similarity"]) for row in vector_rows]
    left_summary = left["summary"]
    right_summary = right["summary"]
    comparison = {
        "baseline_prompt": left["prompt"],
        "candidate_prompt": right["prompt"],
        "baseline_step_count": len(left["steps"]),
        "candidate_step_count": len(right["steps"]),
        "shared_step_count": shared_steps,
        "same_prompt": left["prompt"] == right["prompt"],
        "same_length": len(left["steps"]) == len(right["steps"]),
        "token_match_rate": token_match_rate,
        "top_match_rate": top_match_rate,
        "mean_top_prediction_overlap": mean(overlaps),
        "min_top_prediction_overlap": min(overlaps),
        "max_abs_state_norm_delta": max(norm_deltas),
        "mean_abs_state_norm_delta": mean(norm_deltas),
        "max_abs_state_delta_norm_delta": max(state_delta_deltas),
        "mean_abs_state_delta_norm_delta": mean(state_delta_deltas),
        "max_abs_top_probability_delta": max(prob_deltas),
        "mean_abs_top_probability_delta": mean(prob_deltas),
        "baseline_final_state_norm": left_summary["final_state_norm"],
        "candidate_final_state_norm": right_summary["final_state_norm"],
        "final_state_norm_delta": right_summary["final_state_norm"] - left_summary["final_state_norm"],
        "has_state_vectors": has_state_vectors,
        "mean_state_l2_distance": mean(l2_distances) if has_state_vectors else None,
        "max_state_l2_distance": max(l2_distances) if has_state_vectors else None,
        "mean_state_cosine_similarity": mean(cosine_similarities) if has_state_vectors else None,
        "min_state_cosine_similarity": min(cosine_similarities) if has_state_vectors else None,
        "final_state_l2_distance": None,
        "final_state_cosine_similarity": None,
        "baseline_final_top_prediction": left_summary["final_top_prediction"],
        "candidate_final_top_prediction": right_summary["final_top_prediction"],
        "same_final_top_prediction": (
            left_summary["final_top_prediction"] == right_summary["final_top_prediction"]
        ),
    }
    if isinstance(left_summary.get("final_state_vector"), list) and isinstance(
        right_summary.get("final_state_vector"), list
    ):
        left_vector = [float(value) for value in left_summary["final_state_vector"]]
        right_vector = [float(value) for value in right_summary["final_state_vector"]]
        comparison["final_state_l2_distance"] = l2_distance(left_vector, right_vector)
        comparison["final_state_cosine_similarity"] = cosine_similarity(left_vector, right_vector)
    return comparison


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probes: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for index, prompt in enumerate(prompts):
        payload = probe_prompt(prompt, args)
        name = f"{index:02d}_{slug(prompt)}"
        while name in used_names:
            name = f"{name}_{index}"
        used_names.add(name)
        json_path = output_dir / f"{name}.json"
        csv_path = output_dir / f"{name}.csv"
        save_probe_json(str(json_path), payload)
        save_probe_csv(str(csv_path), payload)
        payload["_suite_name"] = name
        payload["_suite_json"] = str(json_path)
        payload["_suite_csv"] = str(csv_path)
        probes.append(payload)
        prompt_rows.append(
            {
                "index": index,
                "name": name,
                "prompt": prompt,
                "json": str(json_path),
                "csv": str(csv_path),
                "step_count": payload["summary"]["step_count"],
                "final_state_norm": payload["summary"]["final_state_norm"],
                "final_top_prediction": payload["summary"]["final_top_prediction"],
                "final_top_probability": payload["summary"]["final_top_probability"],
            }
        )

    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(probes):
        for j, right in enumerate(probes):
            if i >= j:
                continue
            comparison = compare_variable_length(left, right)
            comparison["baseline_index"] = i
            comparison["candidate_index"] = j
            comparison["baseline_name"] = left["_suite_name"]
            comparison["candidate_name"] = right["_suite_name"]
            pairs.append(comparison)

    mean_overlap = mean([float(pair["mean_top_prediction_overlap"]) for pair in pairs])
    max_norm_delta = max(float(pair["max_abs_state_norm_delta"]) for pair in pairs)
    min_overlap = min(float(pair["mean_top_prediction_overlap"]) for pair in pairs)
    vector_pairs = [pair for pair in pairs if pair.get("has_state_vectors")]
    has_state_vectors = len(vector_pairs) == len(pairs)
    max_l2_distance = max(float(pair["max_state_l2_distance"]) for pair in vector_pairs) if has_state_vectors else None
    min_cosine_similarity = (
        min(float(pair["min_state_cosine_similarity"]) for pair in vector_pairs) if has_state_vectors else None
    )
    most_similar = max(pairs, key=lambda item: (item["mean_top_prediction_overlap"], -item["max_abs_state_norm_delta"]))
    most_different = max(pairs, key=lambda item: (item["max_abs_state_norm_delta"], -item["mean_top_prediction_overlap"]))
    return {
        "checkpoint": probes[0].get("checkpoint"),
        "manifest": probes[0].get("manifest"),
        "release": probes[0].get("release"),
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "prompt_count": len(prompts),
        "pair_count": len(pairs),
        "prompts": prompt_rows,
        "pairs": pairs,
        "summary": {
            "prompt_count": len(prompts),
            "pair_count": len(pairs),
            "mean_pair_overlap": mean_overlap,
            "min_pair_overlap": min_overlap,
            "max_pair_state_norm_delta": max_norm_delta,
            "has_state_vectors": has_state_vectors,
            "max_pair_state_l2_distance": max_l2_distance,
            "min_pair_state_cosine_similarity": min_cosine_similarity,
            "most_similar_pair": {
                "baseline_name": most_similar["baseline_name"],
                "candidate_name": most_similar["candidate_name"],
                "mean_top_prediction_overlap": most_similar["mean_top_prediction_overlap"],
                "max_abs_state_norm_delta": most_similar["max_abs_state_norm_delta"],
            },
            "most_different_pair": {
                "baseline_name": most_different["baseline_name"],
                "candidate_name": most_different["candidate_name"],
                "mean_top_prediction_overlap": most_different["mean_top_prediction_overlap"],
                "max_abs_state_norm_delta": most_different["max_abs_state_norm_delta"],
            },
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_prompts_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "name",
        "prompt",
        "json",
        "csv",
        "step_count",
        "final_state_norm",
        "final_top_prediction",
        "final_top_probability",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pairs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline_index",
        "candidate_index",
        "baseline_name",
        "candidate_name",
        "baseline_prompt",
        "candidate_prompt",
        "baseline_step_count",
        "candidate_step_count",
        "shared_step_count",
        "same_prompt",
        "same_length",
        "token_match_rate",
        "top_match_rate",
        "mean_top_prediction_overlap",
        "min_top_prediction_overlap",
        "max_abs_state_norm_delta",
        "mean_abs_state_norm_delta",
        "max_abs_state_delta_norm_delta",
        "mean_abs_state_delta_norm_delta",
        "max_abs_top_probability_delta",
        "mean_abs_top_probability_delta",
        "baseline_final_state_norm",
        "candidate_final_state_norm",
        "final_state_norm_delta",
        "has_state_vectors",
        "mean_state_l2_distance",
        "max_state_l2_distance",
        "mean_state_cosine_similarity",
        "min_state_cosine_similarity",
        "final_state_l2_distance",
        "final_state_cosine_similarity",
        "baseline_final_top_prediction",
        "candidate_final_top_prediction",
        "same_final_top_prediction",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_report(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("TextPy/SoA text memory prompt suite")
    print(f"release: {payload['release']}")
    print(f"prompts: {payload['prompt_count']}")
    print(f"pairs: {payload['pair_count']}")
    print("")
    print("Prompt Summary")
    for row in payload["prompts"]:
        print(
            f"  {row['index']:02d} {row['name']} "
            f"steps={row['step_count']} "
            f"final_norm={row['final_state_norm']:.4f} "
            f"top={row['final_top_prediction']!r}"
        )
    print("")
    print("Pair Summary")
    print(f"  mean_pair_overlap: {summary['mean_pair_overlap']:.4f}")
    print(f"  min_pair_overlap: {summary['min_pair_overlap']:.4f}")
    print(f"  max_pair_state_norm_delta: {summary['max_pair_state_norm_delta']:.4f}")
    if summary["has_state_vectors"]:
        print(f"  max_pair_state_l2_distance: {summary['max_pair_state_l2_distance']:.4f}")
        print(f"  min_pair_state_cosine_similarity: {summary['min_pair_state_cosine_similarity']:.4f}")
    print(
        "  most_similar_pair: "
        f"{summary['most_similar_pair']['baseline_name']} -> {summary['most_similar_pair']['candidate_name']}"
    )
    print(
        "  most_different_pair: "
        f"{summary['most_different_pair']['baseline_name']} -> {summary['most_different_pair']['candidate_name']}"
    )


def run(args: argparse.Namespace) -> None:
    payload = run_suite(args)
    output_dir = Path(args.output_dir)
    output_json = Path(args.output_json) if args.output_json else output_dir / "memory_suite.json"
    prompts_csv = Path(args.prompts_csv) if args.prompts_csv else output_dir / "prompts.csv"
    pairs_csv = Path(args.pairs_csv) if args.pairs_csv else output_dir / "pairs.csv"
    write_json(output_json, payload)
    write_prompts_csv(prompts_csv, payload["prompts"])
    write_pairs_csv(pairs_csv, payload["pairs"])
    print_report(payload)
    print("")
    print(f"Saved JSON: {output_json}")
    print(f"Saved prompts CSV: {prompts_csv}")
    print(f"Saved pairs CSV: {pairs_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a suite of text memory probes and pairwise comparisons.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--release", default=None)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--include-bos", action="store_true")
    parser.add_argument("--include-state-vectors", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", default="artifacts/releases/text_memory_suite")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--prompts-csv", default=None)
    parser.add_argument("--pairs-csv", default=None)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default="cpu",
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
