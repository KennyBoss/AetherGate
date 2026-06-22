#!/usr/bin/env python3
"""Perturbation sweep for the delayed-reward valley fix on argmax_index4.

The beam search is deterministic (no RNG, no init seed), so re-running the same
config is a no-op. Instead we perturb things that should NOT matter if the
valley-crossing is fundamental rather than a knife-edge calibration:

  - task instances:   --cases (different argmax datasets)
  - search capacity:  --beam-width
  - calibration:      --survival-min-value, --value-rollout-max-depth

Every config runs to FULL DEPTH (no --stop-on-solution) so we confirm the exact
solution is not only found but persists to program completion. Runs sequentially
(each search already saturates several cores via numpy).
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

SEARCH = "search_code_tape_trajectory.py"
OUTDIR = Path("artifacts/code_tape_prior/sweep")
PRIOR = "artifacts/code_tape_prior/argmax_index4_trace_prior.json"


def base_cmd(out_json: str, *, cases: int, beam: int, min_value: float, vdepth: int) -> list[str]:
    return [
        sys.executable, SEARCH,
        "--target", "argmax_index4",
        "--program-length", "11",
        "--max-steps", "64",
        "--cases", str(cases),
        "--beam-width", str(beam),
        "--signature-cases", "8",
        "--signature-steps", "18",
        "--trace-prior-json", PRIOR,
        "--trace-prior-weight", "0.25",
        "--value-rollouts", "25",
        "--value-rollout-top-k", "128",
        "--value-rollout-role-top-k", "128",
        "--value-rollout-max-depth", str(vdepth),
        "--value-weight", "0.03",
        "--prefix-stage-lane-fraction", "0.125",
        "--prefix-stage-ignore-signature-limit",
        "--survival-lane-fraction", "0.25",
        "--survival-ttl", "8",
        "--survival-min-value", str(min_value),
        "--survival-ignore-signature-limit",
        # NO --stop-on-solution: run full depth on purpose.
        "--output-json", out_json,
    ]


def configs() -> list[dict]:
    # One axis varies at a time off the verified baseline (cases=32, beam=64,
    # min_value=4.0, vdepth=11), plus a couple of combined stress points.
    base = dict(cases=32, beam=64, min_value=4.0, vdepth=11)
    grid: list[dict] = [dict(base)]
    for cases in (24, 48):
        grid.append({**base, "cases": cases})
    for beam in (48, 96):
        grid.append({**base, "beam": beam})
    for mv in (6.0,):
        grid.append({**base, "min_value": mv})
    for vd in (8,):
        grid.append({**base, "vdepth": vd})
    # combined stress: smaller beam + tighter threshold + leaner radar
    grid.append({**base, "beam": 48, "min_value": 6.0, "vdepth": 8})
    # dedup
    seen, out = set(), []
    for c in grid:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    grid = configs()
    print(f"sweep: {len(grid)} configs, full-depth, no stop-on-solution\n")
    results = []
    for i, c in enumerate(grid):
        tag = f"c{c['cases']}_b{c['beam']}_mv{c['min_value']}_vd{c['vdepth']}"
        out_json = str(OUTDIR / f"{tag}.json")
        cmd = base_cmd(out_json, cases=c["cases"], beam=c["beam"],
                       min_value=c["min_value"], vdepth=c["vdepth"])
        print(f"[{i+1}/{len(grid)}] {tag} ...", flush=True)
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        row = {"tag": tag, **c, "elapsed_s": round(dt, 1), "rc": proc.returncode}
        if proc.returncode != 0:
            row["error"] = proc.stderr.strip().splitlines()[-1] if proc.stderr else "nonzero rc"
        else:
            data = json.loads(Path(out_json).read_text())
            fs = data.get("first_solution")
            best = data.get("best", {})
            row["first_solution"] = fs is not None
            row["first_solution_depth"] = fs.get("depth") if fs else None
            row["best_train_mse"] = best.get("train_mse")
            row["best_holdout_mse"] = best.get("holdout_mse")
            row["survival_activations"] = data.get("search", {}).get("survival_activations")
            if fs:
                row["solution_prefix"] = fs.get("prefix_ids")
        results.append(row)
        flag = "OK " if row.get("first_solution") else "FAIL"
        print(f"    {flag} depth={row.get('first_solution_depth')} "
              f"best_mse={row.get('best_train_mse')}/{row.get('best_holdout_mse')} "
              f"surv_act={row.get('survival_activations')} ({dt:.0f}s)", flush=True)

    summary = OUTDIR / "sweep_summary.json"
    summary.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r.get("first_solution"))
    print(f"\n=== {n_ok}/{len(results)} configs found exact solution ===")
    print(f"summary: {summary}")
    if results and n_ok < len(results):
        print("FAILURES:")
        for r in results:
            if not r.get("first_solution"):
                print(f"  {r['tag']}: {r.get('error', 'no first_solution')}")


if __name__ == "__main__":
    main()
