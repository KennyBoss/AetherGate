#!/usr/bin/env python3
"""Phase 2.6 — Real CI Scaling: the act/think boundary in the language of money.

Vary nothing about the agent. Vary only P = the cost of running the test suite
(Unit ~1x, Integration ~10x, Heavy CI ~100x) and ask the business question:

    above what CI cost does the learned gate save engineering time, and how much?

Why this is measured analytically (the technical block, handled)
----------------------------------------------------------------
Raw wall-clock of pytest is non-deterministic (scheduler, disk, parallelism), so
a phase diagram read straight off the clock would be noise, and parallel heavy/
light runs would contend. Instead we (1) count test runs DETERMINISTICALLY,
separating FULL-suite runs from cheap SMOKE probes, (2) calibrate ONE real pytest
invocation = t0 seconds, (3) build the diagram analytically:

    cost_arm(P) [s] = full_runs * (t0 * P) + probe_runs * t0

No parallel environments race; P* is exact and reproducible.

The asymmetry that creates a crossover
--------------------------------------
APP decides "should I try this patch?" by RUNNING THE FULL SUITE (expensive).
SSM-GATE decides with a cheap SMOKE probe and spends a full run only on the
committed path. So
    cost_SSM(P) = cost_APP(P)  =>  P* = probe_runs / (full_app - full_ssm)
finite & positive because the gate does fewer FULL runs (it abstains on novel
bugs instead of burning full library attempts). Below P* the probe wins; above
P* the learned gate wins — now in CI-seconds, the unit a CTO actually pays for.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import code_repair_real_repos as RR
import skill_compression_ssm_gate as G


def ci95(xs):
    n = len(xs)
    mean = statistics.mean(xs)
    std = statistics.stdev(xs) if n > 1 else 0.0
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}.get(n - 1, 1.96)
    return mean, (t * std / math.sqrt(n) if n > 1 else float("nan"))


def train_gate(seed, cfg, lib):
    rng = random.Random(seed)
    fams = list(RR.FAMILIES)
    k = cfg.test_cases

    def gen_reuse():
        fam = rng.choice(fams)
        return RR.make_bug(rng, fam, rng.choice(lib[fam]), k)

    def gen_ood():
        fam = rng.choice(fams)
        pool = [p for p in RR.FAMILIES[fam][1] if p not in set(lib[fam])]
        return RR.make_bug(rng, fam, rng.choice(pool), k)

    gate_bugs = [gen_reuse() for _ in range(cfg.gate_train // 2)] + \
                [gen_ood() for _ in range(cfg.gate_train // 2)]
    rng.shuffle(gate_bugs)
    Xs, ys = [], []
    for b in gate_bugs:
        c_eng, _ = RR.repair_app(b, lib[b["fam"]])
        c_broad, _ = RR.repair_broad(b)
        ys.append(1 if c_eng < c_broad else 0)
        Xs.append(RR.features(b, lib[b["fam"]], cfg.probe_cases))
    gate = G.SSMGate(d_in=2, hidden=cfg.hidden, seed=seed)
    for _ in range(cfg.epochs):
        grads = {"A": np.zeros_like(gate.A), "B": np.zeros_like(gate.B),
                 "w": np.zeros_like(gate.w), "b": 0.0}
        for X, y in zip(Xs, ys):
            g = gate.backward(gate.forward(X)[1], float(y))
            for kk in grads:
                grads[kk] = grads[kk] + g[kk]
        for kk in grads:
            grads[kk] = grads[kk] / len(Xs)
        gate.step(grads, lr=cfg.lr)
    return gate, gen_reuse, gen_ood


def count_runs(seed, cfg):
    """Deterministic FULL-run and SMOKE-probe counts per arm on a mixed stream."""
    lib = RR.build_library(cfg.lib_seed, cfg.per_family)
    gate, gen_reuse, gen_ood = train_gate(seed, cfg, lib)
    bugs = [gen_reuse() for _ in range(cfg.eval_bugs)] + \
           [gen_ood() for _ in range(cfg.eval_bugs)]
    off_full = app_full = ssm_full = ssm_probe = 0
    for b in bugs:
        off_full += RR.repair_broad(b)[0]
        app_full += RR.repair_app(b, lib[b["fam"]])[0]
        # SSM: cheap smoke probe over library to decide, then full only on path
        ssm_probe += len(lib[b["fam"]])
        if gate.predict(RR.features(b, lib[b["fam"]], cfg.probe_cases)) >= 0.5:
            ssm_full += RR.repair_app(b, lib[b["fam"]])[0]
        else:
            ssm_full += RR.repair_broad(b)[0]
    n = len(bugs)
    return {"off_full": off_full / n, "app_full": app_full / n,
            "ssm_full": ssm_full / n, "ssm_probe": ssm_probe / n}


def calibrate_pytest_seconds(cfg, n=8):
    """One real pytest invocation -> t0 seconds (the deterministic cost unit)."""
    rng = random.Random(123)
    lib = RR.build_library(cfg.lib_seed, cfg.per_family)
    fams = list(RR.FAMILIES)
    wall, runs = 0.0, 0
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        for _ in range(n):
            fam = rng.choice(fams)
            bug = RR.make_bug(rng, fam, rng.choice(lib[fam]), cfg.test_cases)
            t = time.perf_counter()
            RR.pytest_run(wd, fam, rng.choice(RR.FAMILIES[fam][1]), bug["cases"], bug["ref"])
            wall += time.perf_counter() - t
            runs += 1
    return wall / max(1, runs)


def cost_seconds(c, P, t0):
    """full runs cost t0*P (heavy suite), smoke probes cost t0 (quick subset)."""
    off = c["off_full"] * t0 * P
    app = c["app_full"] * t0 * P
    ssm = c["ssm_full"] * t0 * P + c["ssm_probe"] * t0
    return off, app, ssm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--per-family", type=int, default=6)
    ap.add_argument("--lib-seed", type=int, default=777)
    ap.add_argument("--test-cases", type=int, default=5)
    ap.add_argument("--probe-cases", type=int, default=2)
    ap.add_argument("--gate-train", type=int, default=160)
    ap.add_argument("--eval-bugs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--regimes", default="1,3,10,30,100")
    ap.add_argument("--bug-volume", type=int, default=1000, help="bugs/month for $ framing")
    ap.add_argument("--output-json", default="artifacts/ci_scaling.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    counts = [count_runs(s, args) for s in seeds]
    agg = {k: round(statistics.mean(c[k] for c in counts), 3)
           for k in ("off_full", "app_full", "ssm_full", "ssm_probe")}

    # crossover (heaviness multiplier units): SSM beats APP when P > P*
    denom = agg["app_full"] - agg["ssm_full"]
    pstar = agg["ssm_probe"] / denom if denom > 0 else None
    pstar_seeds = [c["ssm_probe"] / (c["app_full"] - c["ssm_full"])
                   for c in counts if c["app_full"] - c["ssm_full"] > 0]
    pm, ph = ci95(pstar_seeds) if pstar_seeds else (float("nan"), float("nan"))

    t0 = calibrate_pytest_seconds(args)
    regimes = [float(r) for r in args.regimes.split(",")]
    table = []
    for P in regimes:
        off, app, ssm = cost_seconds(agg, P, t0)
        save_vs_app = app - ssm
        save_vs_off = off - ssm
        table.append({
            "P": P, "off_s": round(off, 4), "app_s": round(app, 4), "ssm_s": round(ssm, 4),
            "ssm_saves_vs_app_s_per_bug": round(save_vs_app, 4),
            "ssm_saves_vs_off_s_per_bug": round(save_vs_off, 4),
            "ssm_beats_app": save_vs_app > 0,
            "saved_vs_app_min_per_volume": round(save_vs_app * args.bug_volume / 60, 1),
            "saved_vs_off_min_per_volume": round(save_vs_off * args.bug_volume / 60, 1),
        })

    payload = {"config": vars(args), "run_counts": agg,
               "pytest_t0_seconds": round(t0, 4),
               "crossover_P_star_heaviness": {
                   "mean": round(pm, 2), "ci95_halfwidth": round(ph, 2),
                   "per_seed": [round(p, 2) for p in pstar_seeds],
                   "formula": "P* = ssm_probe / (app_full - ssm_full)"},
               "regime_table": table,
               "protocol_evidence": {"deterministic_counts": True,
                                     "analytic_cost_model": True,
                                     "real_pytest_calibrated": True,
                                     "n_seeds": len(seeds), "valid": len(seeds) >= 3}}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Phase 2.6 — Real CI Scaling: savings vs CI cost ===")
    print(f"seeds={seeds}  real pytest t0={round(t0,4)}s/run")
    print(f"  deterministic run counts/bug: OFF_full={agg['off_full']}  "
          f"APP_full={agg['app_full']}  SSM_full={agg['ssm_full']}  SSM_probe={agg['ssm_probe']}")
    print(f"  CROSSOVER P* (full/smoke heaviness) = {round(pm,2)} ± {round(ph,2)}  "
          f"(SSM beats the probe once a full suite is >{round(pm,1)}x a smoke check)")
    print(f"  {'regime P':>9s} {'APP s/bug':>10s} {'SSM s/bug':>10s} {'SSM beats APP':>14s} "
          f"{'saved min/'+str(args.bug_volume):>14s}")
    names = {1: "unit", 3: "unit+", 10: "integ", 30: "integ+", 100: "heavy"}
    for row in table:
        nm = names.get(int(row["P"]), "")
        print(f"  {row['P']:>7.0f} {nm:<2s} {row['app_s']:>10.4f} {row['ssm_s']:>10.4f} "
              f"{str(row['ssm_beats_app']):>14s} {row['saved_vs_app_min_per_volume']:>14.1f}")
    print(f"  (saved min/{args.bug_volume} bugs is SSM vs the strong APP probe baseline)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
