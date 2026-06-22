#!/usr/bin/env python3
"""Analyze late-game CodePy tape near-misses for loop/return ranking."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from code_tape_trace_prior import read_jsonl, safe_float, save_json
from evolve_code_tape import REFERENCE_PROGRAMS, TAPE_BLOCKS


ARGMAX_STAGE_DEFS: list[tuple[str, tuple[int, ...]]] = [
    ("read_max", (3,)),
    ("index_init", (24, 2)),
    ("first_move", (7,)),
    ("compare_current", (20,)),
    ("conditional_value_update", (25,)),
    ("conditional_index_update", (28,)),
    ("second_move", (7,)),
    ("loop_test", (15,)),
    ("loop_jump", (30,)),
    ("return_index", (32,)),
    ("halt", (31,)),
]

PATTERN_DEFS: list[tuple[str, tuple[int, ...]]] = [
    ("read_index_move", (3, 24, 7)),
    ("read_zero_index_move", (3, 2, 7)),
    ("compare_then_value_update", (20, 25)),
    ("compare_then_index_update", (20, 28)),
    ("double_conditional_update", (20, 25, 28)),
    ("post_update_loop", (25, 28, 7, 15, 30)),
    ("loop_return_halt", (15, 30, 32, 31)),
    ("return_halt", (32, 31)),
    ("reference_core", (3, 24, 7, 20, 25, 28, 7, 15, 30, 32, 31)),
]

TERMINAL_OPS = {15, 29, 30, 31, 32}
BAD_OR_WEAK_CONTROL_OPS = {18, 19, 29}


def load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compact_ops(row: dict[str, Any]) -> list[int]:
    ops = row.get("compact_program_ids")
    if not ops:
        ops = [op for op in row.get("program_ids", []) if int(op) != 0]
    return [int(op) for op in ops]


def has_subsequence(ops: list[int], pattern: tuple[int, ...]) -> bool:
    if not pattern:
        return True
    j = 0
    for op in ops:
        if op == pattern[j]:
            j += 1
            if j >= len(pattern):
                return True
    return False


def find_stage_positions(ops: list[int]) -> dict[str, int | None]:
    positions: dict[str, int | None] = {}
    start = 0
    for name, choices in ARGMAX_STAGE_DEFS:
        found = None
        for i in range(start, len(ops)):
            if ops[i] in choices:
                found = i
                start = i + 1
                break
        positions[name] = found
    return positions


def matched_stage_count(positions: dict[str, int | None]) -> int:
    count = 0
    for name, _ in ARGMAX_STAGE_DEFS:
        if positions.get(name) is None:
            break
        count += 1
    return count


def failure_stage(positions: dict[str, int | None]) -> str:
    for name, _ in ARGMAX_STAGE_DEFS:
        if positions.get(name) is None:
            return name
    return "none"


def lcs_len(a: list[int], b: list[int]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        safe_float(row.get("train_mse"), math.inf),
        safe_float(row.get("holdout_mse"), math.inf),
        -safe_float(row.get("output_hit")),
        -safe_float(row.get("paired_running_hit")),
        -safe_float(row.get("raw_score"), -math.inf),
    )


def dedupe_by_program(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for row in sorted(rows, key=rank_key):
        key = tuple(int(op) for op in row.get("program_ids", []) or compact_ops(row))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def mean(rows: list[dict[str, Any]], name: str) -> float:
    if not rows:
        return 0.0
    return sum(safe_float(row.get(name)) for row in rows) / len(rows)


def block_frequency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_counts: Counter[int] = Counter()
    occurrence_counts: Counter[int] = Counter()
    for row in rows:
        ops = compact_ops(row)
        occurrence_counts.update(ops)
        row_counts.update(set(ops))
    result = []
    total = max(len(rows), 1)
    for op_id, row_count in row_counts.most_common():
        result.append(
            {
                "id": int(op_id),
                "block": TAPE_BLOCKS[int(op_id)],
                "rows_with_block": int(row_count),
                "row_fraction": row_count / total,
                "occurrences": int(occurrence_counts[op_id]),
            }
        )
    return result


def pattern_frequency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = max(len(rows), 1)
    result = []
    for name, pattern in PATTERN_DEFS:
        count = sum(1 for row in rows if has_subsequence(compact_ops(row), pattern))
        result.append(
            {
                "name": name,
                "pattern_ids": list(pattern),
                "pattern_blocks": [TAPE_BLOCKS[op] for op in pattern],
                "rows_with_pattern": count,
                "row_fraction": count / total,
            }
        )
    return result


def stage_summary(rows: list[dict[str, Any]], reference: list[int]) -> dict[str, Any]:
    total = max(len(rows), 1)
    stage_counts: Counter[str] = Counter()
    unordered_stage_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    matched_counts: Counter[int] = Counter()
    terminal_counts: Counter[str] = Counter()
    order_issue_counts: Counter[str] = Counter()
    lcs_values: list[int] = []
    for row in rows:
        ops = compact_ops(row)
        positions = find_stage_positions(ops)
        first_positions: dict[int, int] = {}
        for i, op in enumerate(ops):
            first_positions.setdefault(op, i)
        for name, choices in ARGMAX_STAGE_DEFS:
            if any(op in ops for op in choices):
                unordered_stage_counts[name] += 1
        for name, _ in ARGMAX_STAGE_DEFS:
            if positions.get(name) is not None:
                stage_counts[name] += 1
        fail = failure_stage(positions)
        failure_counts[fail] += 1
        matched_counts[matched_stage_count(positions)] += 1
        lcs_values.append(lcs_len(ops, reference))
        if any(op in BAD_OR_WEAK_CONTROL_OPS for op in ops):
            terminal_counts["uses_weak_or_reset_jump"] += 1
        if any(op in TERMINAL_OPS for op in ops):
            terminal_counts["has_terminal_op"] += 1
        if 32 in ops:
            terminal_counts["has_return_index"] += 1
        if 31 in ops:
            terminal_counts["has_halt"] += 1
        if has_subsequence(ops, (30, 32)):
            terminal_counts["loop_jump_before_return"] += 1
        if has_subsequence(ops, (32, 31)):
            terminal_counts["return_before_halt"] += 1
        if has_subsequence(ops, (25, 28, 7, 15, 30, 32, 31)):
            terminal_counts["full_post_compare_continuation"] += 1
        if 25 in first_positions and 28 in first_positions:
            if first_positions[28] < first_positions[25]:
                order_issue_counts["conditional_updates_swapped_28_before_25"] += 1
            else:
                order_issue_counts["conditional_updates_ordered_25_before_28"] += 1
        elif 25 in first_positions:
            order_issue_counts["missing_conditional_index_update_28"] += 1
        elif 28 in first_positions:
            order_issue_counts["missing_conditional_value_update_25"] += 1
        if 30 in first_positions and 32 in first_positions:
            if first_positions[32] < first_positions[30]:
                order_issue_counts["return_before_loop_jump"] += 1
            else:
                order_issue_counts["loop_jump_before_return"] += 1
        elif 32 in first_positions:
            order_issue_counts["return_without_loop_jump"] += 1
        elif 30 in first_positions:
            order_issue_counts["loop_jump_without_return"] += 1
        if 31 not in first_positions:
            order_issue_counts["missing_halt"] += 1
        if 7 in first_positions and 15 in first_positions and first_positions[15] < first_positions[7]:
            order_issue_counts["loop_test_before_first_move"] += 1
    return {
        "stage_presence": [
            {
                "stage": name,
                "rows_with_stage_in_order": int(stage_counts[name]),
                "rows_with_stage_anywhere": int(unordered_stage_counts[name]),
                "row_fraction": stage_counts[name] / total,
                "unordered_row_fraction": unordered_stage_counts[name] / total,
            }
            for name, _ in ARGMAX_STAGE_DEFS
        ],
        "first_missing_stage_counts": dict(failure_counts),
        "matched_stage_count_distribution": {str(k): int(v) for k, v in sorted(matched_counts.items())},
        "terminal_control": [
            {"name": name, "rows": int(count), "row_fraction": count / total}
            for name, count in sorted(terminal_counts.items())
        ],
        "reference_lcs": {
            "mean": sum(lcs_values) / total if rows else 0.0,
            "max": max(lcs_values) if lcs_values else 0,
            "reference_length": len(reference),
        },
        "order_issue_counts": dict(order_issue_counts),
    }


def common_prefix(rows: list[dict[str, Any]]) -> list[int]:
    if not rows:
        return []
    programs = [compact_ops(row) for row in rows]
    prefix: list[int] = []
    for values in zip(*programs):
        first = values[0]
        if any(value != first for value in values):
            break
        prefix.append(first)
    return prefix


def summarize_rows(name: str, rows: list[dict[str, Any]], reference: list[int], top_n: int) -> dict[str, Any]:
    ranked = dedupe_by_program(rows)[:top_n]
    if not ranked:
        return {"name": name, "count": 0, "top_programs": []}
    frequencies = block_frequency(ranked)
    patterns = pattern_frequency(ranked)
    stable_blocks = [row for row in frequencies if row["row_fraction"] >= 0.90]
    prefix = common_prefix(ranked)
    return {
        "name": name,
        "count": len(ranked),
        "source_rows": len(rows),
        "metrics": {
            "best_train_mse": safe_float(ranked[0].get("train_mse"), math.inf),
            "best_holdout_mse": safe_float(ranked[0].get("holdout_mse"), math.inf),
            "mean_train_mse": mean(ranked, "train_mse"),
            "mean_holdout_mse": mean(ranked, "holdout_mse"),
            "mean_output_hit": mean(ranked, "output_hit"),
            "mean_paired_running_hit": mean(ranked, "paired_running_hit"),
            "mean_loop_hit": mean(ranked, "loop_hit"),
            "mean_halt_hit": mean(ranked, "halt_hit"),
            "depth_min": min(int(row.get("depth", 0)) for row in ranked),
            "depth_max": max(int(row.get("depth", 0)) for row in ranked),
        },
        "stable_blocks_90": stable_blocks,
        "block_frequency": frequencies,
        "pattern_frequency": patterns,
        "stage_summary": stage_summary(ranked, reference),
        "common_prefix_ids": prefix,
        "common_prefix_blocks": [TAPE_BLOCKS[op] for op in prefix],
        "top_programs": [
            {
                "rank": i + 1,
                "depth": int(row.get("depth", 0)),
                "train_mse": safe_float(row.get("train_mse"), math.inf),
                "holdout_mse": safe_float(row.get("holdout_mse"), math.inf),
                "raw_score": safe_float(row.get("raw_score"), -math.inf),
                "output_hit": safe_float(row.get("output_hit")),
                "paired_running_hit": safe_float(row.get("paired_running_hit")),
                "loop_hit": safe_float(row.get("loop_hit")),
                "halt_hit": safe_float(row.get("halt_hit")),
                "selected_next_beam": bool(row.get("selected_next_beam")),
                "selection_lane": row.get("selection_lane"),
                "survival_ttl": int(row.get("survival_ttl") or 0),
                "program_ids": [int(op) for op in row.get("program_ids", [])],
                "compact_program_ids": compact_ops(row),
                "compact_program_blocks": [TAPE_BLOCKS[op] for op in compact_ops(row)],
                "first_missing_stage": failure_stage(find_stage_positions(compact_ops(row))),
                "matched_stage_count": matched_stage_count(find_stage_positions(compact_ops(row))),
                "reference_lcs": lcs_len(compact_ops(row), reference),
            }
            for i, row in enumerate(ranked[: min(len(ranked), 12)])
        ],
    }


def filter_rows(records: list[dict[str, Any]], args: argparse.Namespace, program_length: int) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if args.target and record.get("target") not in {None, args.target}:
            continue
        depth = int(record.get("depth", 0))
        if args.final_depth_only and depth < program_length:
            continue
        if depth < args.min_depth:
            continue
        train_mse = safe_float(record.get("train_mse"), math.inf)
        if train_mse > args.max_train_mse:
            continue
        rows.append(record)
    return rows


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# CodePy Late-Game Trace Analysis",
        "",
        "This report analyzes top near-miss tape DSL programs after trajectory/value/survival search.",
        "It is an offline diagnostic pass; no generated Python source is executed.",
        "",
        "## Summary",
        "",
        f"- Target: `{payload['target']}`",
        f"- Candidate records: {payload['records']['loaded']} loaded, {payload['records']['filtered']} filtered",
        f"- Reference program: `{payload['reference']['program_ids']}`",
        "",
    ]
    for cohort in payload["cohorts"]:
        lines.append(f"## {cohort['name']}")
        lines.append("")
        if cohort["count"] <= 0:
            lines.append("- No records.")
            lines.append("")
            continue
        metrics = cohort["metrics"]
        lines.append(
            f"- Count: {cohort['count']} from {cohort['source_rows']} source rows; "
            f"depth {metrics['depth_min']}..{metrics['depth_max']}"
        )
        lines.append(
            f"- Best train/holdout MSE: {metrics['best_train_mse']:.6g} / "
            f"{metrics['best_holdout_mse']:.6g}"
        )
        lines.append(
            f"- Mean output/paired/loop/halt: {metrics['mean_output_hit']:.3f} / "
            f"{metrics['mean_paired_running_hit']:.3f} / {metrics['mean_loop_hit']:.3f} / "
            f"{metrics['mean_halt_hit']:.3f}"
        )
        stable = cohort.get("stable_blocks_90", [])
        if stable:
            stable_text = ", ".join(f"[{row['id']}] {row['block']}" for row in stable)
            lines.append(f"- Blocks present in at least 90%: {stable_text}")
        else:
            lines.append("- Blocks present in at least 90%: none")
        strong_patterns = [
            row for row in cohort["pattern_frequency"] if row["row_fraction"] >= 0.50
        ]
        if strong_patterns:
            pattern_text = ", ".join(
                f"{row['name']}={row['row_fraction']:.2f}" for row in strong_patterns
            )
            lines.append(f"- Common ordered patterns: {pattern_text}")
        stage_summary_data = cohort["stage_summary"]
        missing = stage_summary_data["first_missing_stage_counts"]
        missing_text = ", ".join(f"{key}: {value}" for key, value in sorted(missing.items()))
        lines.append(f"- First missing stage distribution: {missing_text}")
        terminal = stage_summary_data["terminal_control"]
        terminal_text = ", ".join(f"{row['name']}={row['row_fraction']:.2f}" for row in terminal)
        lines.append(f"- Terminal control signals: {terminal_text or 'none'}")
        order_issues = stage_summary_data.get("order_issue_counts", {})
        if order_issues:
            order_text = ", ".join(f"{key}: {value}" for key, value in sorted(order_issues.items()))
            lines.append(f"- Order/control issue counts: {order_text}")
        lines.append("- Top programs:")
        for row in cohort["top_programs"][:5]:
            lines.append(
                f"  - #{row['rank']} mse={row['train_mse']:.6g}/{row['holdout_mse']:.6g} "
                f"missing={row['first_missing_stage']} matched={row['matched_stage_count']} "
                f"ops={row['compact_program_ids']}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    search_payload = load_json(args.search_json)
    search_args = search_payload.get("args", {}) if isinstance(search_payload.get("args"), dict) else {}
    program_length = int(search_args.get("program_length", args.program_length))
    target = args.target or search_payload.get("target", {}).get("name", "argmax_index4")
    reference = [int(op) for op in REFERENCE_PROGRAMS.get(target, [])]
    if not reference:
        raise SystemExit(f"No reference program registered for target {target!r}.")

    records: list[dict[str, Any]] = []
    for path in args.input_jsonl:
        records.extend(read_jsonl(path))
    filtered = filter_rows(records, args, program_length)
    if not filtered:
        raise SystemExit("No candidate records matched the late-game analysis filters.")

    final_rows = [row for row in filtered if int(row.get("depth", 0)) >= program_length]
    selected_rows = [row for row in filtered if row.get("selected_next_beam")]
    survival_rows = [row for row in filtered if int(row.get("survival_ttl") or 0) > 0]
    cohorts = [
        summarize_rows("top_overall", filtered, reference, args.top_n),
        summarize_rows("top_final_depth", final_rows, reference, args.top_n),
        summarize_rows("top_selected", selected_rows, reference, args.top_n),
        summarize_rows("top_survival_protected", survival_rows, reference, args.top_n),
    ]
    payload = {
        "format_version": 1,
        "prototype": "CodePy late-game near-miss trace analysis",
        "guardrail": "Offline analysis of safe DSL candidate summaries; no arbitrary generated source is executed.",
        "target": target,
        "input_jsonl": args.input_jsonl,
        "search_json": args.search_json,
        "records": {
            "loaded": len(records),
            "filtered": len(filtered),
            "program_length": program_length,
            "filters": {
                "top_n": args.top_n,
                "min_depth": args.min_depth,
                "final_depth_only": args.final_depth_only,
                "max_train_mse": args.max_train_mse,
            },
        },
        "reference": {
            "program_ids": reference,
            "program_blocks": [TAPE_BLOCKS[op] for op in reference],
        },
        "cohorts": cohorts,
        "interpretation": {
            "hypothesis": (
                "If top near-misses share compare/update/loop scaffolding but diverge at "
                "loop/return/halt stages, the bottleneck has shifted from exploration to "
                "late-game ranking."
            ),
            "recommended_next_probe": (
                "Use the stage and terminal-control counts to decide whether to add "
                "trace-prior control-flow features or a terminal continuation score."
            ),
        },
        "args": vars(args),
    }
    save_json(args.output_json, payload)
    if args.output_md:
        write_markdown(Path(args.output_md), payload)
    print("CodePy late-game trace analysis")
    print(f"  output_json: {args.output_json}")
    if args.output_md:
        print(f"  output_md: {args.output_md}")
    for cohort in cohorts:
        if cohort["count"] <= 0:
            print(f"  {cohort['name']}: no records")
            continue
        metrics = cohort["metrics"]
        missing = cohort["stage_summary"]["first_missing_stage_counts"]
        print(
            f"  {cohort['name']}: count={cohort['count']} "
            f"best_mse={metrics['best_train_mse']:.6g}/{metrics['best_holdout_mse']:.6g} "
            f"missing={missing}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze late-game CodePy tape near-misses.")
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--search-json", default=None)
    parser.add_argument("--target", default="argmax_index4")
    parser.add_argument("--program-length", type=int, default=11)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--min-depth", type=int, default=0)
    parser.add_argument("--final-depth-only", action="store_true")
    parser.add_argument("--max-train-mse", type=float, default=math.inf)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
