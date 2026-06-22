#!/usr/bin/env python3
"""Read-only diagnostic: trace the canonical argmax prefix through a candidate log.

Does NOT touch the search source or rerun anything. Reads the per-depth candidate
JSONL emitted by search_code_tape_trajectory.py and reports, for each depth, where
the canonical prefix sits: its rank by raw_score, which selection lane (if any)
kept it, its prefix_value_score, and its survival TTL/score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolve_code_tape import REFERENCE_PROGRAMS


def load_by_depth(path: Path) -> dict[int, list[dict]]:
    by_depth: dict[int, list[dict]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_depth.setdefault(int(row.get("depth", -1)), []).append(row)
    return by_depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-jsonl", required=True)
    ap.add_argument("--target", default="argmax_index4")
    args = ap.parse_args()

    canonical = list(REFERENCE_PROGRAMS[args.target])
    by_depth = load_by_depth(Path(args.candidate_jsonl))

    print(f"canonical = {canonical}")
    print(f"depths present: {sorted(by_depth)}")
    header = (
        f"{'d':>2} {'cands':>5} {'canon_prefix':<26} {'present':>7} "
        f"{'rank_raw':>8} {'lane':>12} {'sel':>3} {'pval':>9} {'ttl':>4} {'sscore':>8} {'mse':>9}"
    )
    print(header)
    print("-" * len(header))
    for depth in sorted(by_depth):
        rows = by_depth[depth]
        canon_prefix = canonical[:depth]
        rows_sorted = sorted(rows, key=lambda r: (r.get("raw_score", -1e30), -r.get("train_mse", 1e30)), reverse=True)
        match = None
        rank = None
        for i, r in enumerate(rows_sorted):
            if [int(x) for x in r.get("prefix_ids", [])] == canon_prefix:
                match = r
                rank = i + 1
                break
        if match is None:
            print(f"{depth:>2} {len(rows):>5} {str(canon_prefix):<26} {'NO':>7}  (pruned before this depth)")
            continue
        pval = match.get("prefix_value_score")
        pval_s = f"{pval:.3f}" if isinstance(pval, (int, float)) else "None"
        print(
            f"{depth:>2} {len(rows):>5} {str(canon_prefix):<26} {'yes':>7} "
            f"{rank:>8} {str(match.get('selection_lane')):>12} "
            f"{('Y' if match.get('selected_next_beam') else 'n'):>3} "
            f"{pval_s:>9} {int(match.get('survival_ttl', 0)):>4} "
            f"{float(match.get('survival_score', 0.0)):>8.3f} {match.get('train_mse', float('nan')):>9.4g}"
        )


if __name__ == "__main__":
    main()
