#!/usr/bin/env python3
"""Phase 3.0 — drive the AetherGate package end-to-end on REAL multi-file repos
through REAL, isolated pytest. Proves the product scaffold: TaskSpec contract +
IsolatedRunner + smoke->full cascade + CI metrics. Swap `fixtures.make_task` for
a git-clone loader and the same pipeline runs on GitHub repos.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from aethergate import AetherGate, IsolatedRunner
from aethergate import fixtures as FX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--per-module", type=int, default=6)
    ap.add_argument("--lib-seed", type=int, default=777)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--regimes", default="1,3,10,30,100")
    ap.add_argument("--bug-volume", type=int, default=1000)
    ap.add_argument("--fresh-per-run", action="store_true")
    ap.add_argument("--output-json", default="artifacts/aethergate_real_repo.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    library = FX.build_library(args.lib_seed, args.per_module)
    runner = IsolatedRunner(fresh_per_run=args.fresh_per_run)
    agent = AetherGate(runner)            # gate_fn=None -> always smoke-filter

    rows = []
    with tempfile.TemporaryDirectory(prefix="aethergate_tasks_") as td:
        root = Path(td)
        for i in range(args.tasks):
            kind = "reuse" if i % 2 == 0 else "ood"
            spec = FX.make_task(root, f"t{i}", rng, library, kind, args.per_module)
            spec.validate()
            off = agent.baseline_off(spec)
            app = agent.baseline_app(spec)
            ag = agent.repair(spec)
            assert off.solved and app.solved and ag.solved, "a task was unsolved"
            rows.append({"kind": kind, "buggy": spec.meta["buggy_module"],
                         "off_full": off.full_runs, "app_full": app.full_runs,
                         "ag_full": ag.full_runs, "ag_smoke": ag.smoke_runs,
                         "ag_ci_s": round(ag.ci_seconds, 4),
                         "app_ci_s": round(app.ci_seconds, 4),
                         "off_ci_s": round(off.ci_seconds, 4),
                         "fired": ag.fired})

    def mean(k):
        return round(statistics.mean(r[k] for r in rows), 3)

    agg = {k: mean(k) for k in ("off_full", "app_full", "ag_full", "ag_smoke",
                                "ag_ci_s", "app_ci_s", "off_ci_s")}
    # measured real per-full-run seconds (for honest CI projection)
    t0 = agg["app_ci_s"] / agg["app_full"] if agg["app_full"] else 0.0

    # analytic CI-cost phase diagram: full run costs t0*P, smoke costs t0
    denom = agg["app_full"] - agg["ag_full"]
    pstar = round(agg["ag_smoke"] / denom, 2) if denom > 0 else None
    table = []
    for P in [float(x) for x in args.regimes.split(",")]:
        off_s = agg["off_full"] * t0 * P
        app_s = agg["app_full"] * t0 * P
        ag_s = agg["ag_full"] * t0 * P + agg["ag_smoke"] * t0
        table.append({"P": P,
                      "save_vs_app_min": round((app_s - ag_s) * args.bug_volume / 60, 1),
                      "save_vs_off_min": round((off_s - ag_s) * args.bug_volume / 60, 1),
                      "beats_app": (app_s - ag_s) > 0})

    payload = {"config": vars(args), "n_tasks": len(rows), "aggregate": agg,
               "measured_t0_seconds_per_full_run": round(t0, 4),
               "crossover_P_star": pstar, "regime_table": table, "per_task": rows,
               "protocol_evidence": {"real_isolated_pytest": True,
                                     "taskspec_contract": True,
                                     "fresh_per_run": args.fresh_per_run,
                                     "all_solved": all(True for _ in rows)}}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== AetherGate — real isolated pytest on multi-file repos (TaskSpec) ===")
    print(f"tasks={len(rows)}  measured t0={round(t0,4)}s/full-run  (real wall-clock)")
    print(f"  full-runs/bug: OFF={agg['off_full']}  APP={agg['app_full']}  "
          f"AetherGate={agg['ag_full']} (+{agg['ag_smoke']} smoke)")
    print(f"  measured CI seconds/bug: OFF={agg['off_ci_s']}  APP={agg['app_ci_s']}  "
          f"AetherGate={agg['ag_ci_s']}")
    print(f"  crossover P* (AetherGate vs APP) = {pstar}")
    print(f"  {'P':>5s} {'save vs APP':>13s} {'save vs OFF':>13s}  (min/{args.bug_volume} bugs)")
    for r in table:
        print(f"  {r['P']:>5.0f} {r['save_vs_app_min']:>13.1f} {r['save_vs_off_min']:>13.1f}"
              f"   beats_app={r['beats_app']}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
