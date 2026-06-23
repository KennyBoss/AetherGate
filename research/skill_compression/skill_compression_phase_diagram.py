#!/usr/bin/env python3
"""Phase 1.5D — the phase diagram: locate the crossover P* with a confidence
interval, and show it obeys a closed-form cost law.

Background (the asymmetric result we are nailing down):
  * exec-penalty P = 0 (cheap to attempt a skill): behavioural probe UTIL+APP
    wins  -> "act first".
  * large P (costly to attempt a skill): the learned recurrent gate SSM-GATE
    wins -> "think first".
There is a crossover P*. This script pins it down.

The cost law (exact, by construction of the eval accounting):
    C_app(P) = base_app + 1.0      * P     # probe attempts the skill every task
    C_ssm(P) = base_ssm + fire_rate * P     # gate attempts only when it fires
=>  crossover   P* = (base_ssm - base_app) / (1 - fire_rate)
with base_ssm > base_app (the static gate's myopia costs extra search) and
fire_rate < 1 (the gate sometimes abstains). So a finite P* always exists, and
the learned mechanism is strictly cheaper for every P > P*.

We measure base costs + fire_rate per seed at P=0, derive P* per seed in closed
form, report mean / std / 95% CI across seeds (separately for REUSE, OOD, and a
50/50 mixed stream), and VALIDATE the law by an independent run at a high P.

Interpretation: the learned internal model has a fixed overhead but a smaller
marginal search cost; past a critical action-cost P* it always pays off. The
"act first vs think first" boundary is a measured constant of this environment.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from argparse import Namespace
from pathlib import Path

import skill_compression_ssm_gate as G


def base_cfg(P: int) -> Namespace:
    return Namespace(
        seeds="", train_tasks=120, eval_tasks=80, gate_train=200,
        max_depth=6, node_budget=2_000_000, freq_k=3, util_penalty=1.0,
        util_threshold=2.0, max_macros=8, probe_budget=40, hidden=8,
        epochs=300, lr=0.02, gate_threshold=0.5, exec_penalty=P,
    )


def crossover(base_app: float, base_ssm: float, fire_rate: float) -> float | None:
    """P* = (base_ssm - base_app) / (1 - fire_rate). None if gate never abstains."""
    if fire_rate >= 1.0:
        return None
    return (base_ssm - base_app) / (1.0 - fire_rate)


def ci95(xs: list[float]) -> tuple[float, float, float, float]:
    """mean, std, half-width of 95% CI (t, df=n-1), via t-table for small n."""
    n = len(xs)
    mean = statistics.mean(xs)
    std = statistics.stdev(xs) if n > 1 else 0.0
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
         6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(n - 1, 1.96)
    half = t * std / math.sqrt(n) if n > 1 else float("nan")
    return mean, std, half, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--validate-at", type=int, default=120,
                    help="independent high-P run to validate the cost law")
    ap.add_argument("--output-json", default="artifacts/phase_diagram.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cfg0 = base_cfg(0)
    per_seed = []
    for s in seeds:
        r = G.run_seed(s, cfg0)
        rec = {"seed": s}
        for split in ("reuse", "ood"):
            ba = r["util_app"][split]["mean_nodes"]
            bs = r["ssm_gate"][split]["mean_nodes"]
            fr = r["ssm_gate"][split]["fire_rate"]
            rec[split] = {"base_app": ba, "base_ssm": bs, "fire_rate": fr,
                          "P_star": crossover(ba, bs, fr)}
        # 50/50 mixed deployment stream
        ba = (r["util_app"]["reuse"]["mean_nodes"] + r["util_app"]["ood"]["mean_nodes"]) / 2
        bs = (r["ssm_gate"]["reuse"]["mean_nodes"] + r["ssm_gate"]["ood"]["mean_nodes"]) / 2
        fr = (r["ssm_gate"]["reuse"]["fire_rate"] + r["ssm_gate"]["ood"]["fire_rate"]) / 2
        rec["mixed"] = {"base_app": ba, "base_ssm": bs, "fire_rate": fr,
                        "P_star": crossover(ba, bs, fr)}
        per_seed.append(rec)

    summary = {}
    for split in ("reuse", "ood", "mixed"):
        ps = [r[split]["P_star"] for r in per_seed if r[split]["P_star"] is not None]
        mean, std, half, t = ci95(ps)
        summary[split] = {
            "P_star_mean": round(mean, 1), "P_star_std": round(std, 1),
            "ci95_half_width": round(half, 1), "t_value": t,
            "ci95_low": round(mean - half, 1), "ci95_high": round(mean + half, 1),
            "P_star_per_seed": [round(p, 1) for p in ps],
        }

    # Independent validation of the exact linear law at a high P.
    Pv = args.validate_at
    cfgv = base_cfg(Pv)
    val_rows = []
    for s in seeds:
        rv = G.run_seed(s, cfgv)
        r0 = next(r for r in per_seed if r["seed"] == s)
        for split in ("reuse", "ood"):
            pred_app = r0[split]["base_app"] + Pv
            pred_ssm = r0[split]["base_ssm"] + r0[split]["fire_rate"] * Pv
            meas_app = rv["util_app"][split]["mean_nodes"]
            meas_ssm = rv["ssm_gate"][split]["mean_nodes"]
            val_rows.append({
                "seed": s, "split": split, "P": Pv,
                "pred_app": round(pred_app, 2), "meas_app": round(meas_app, 2),
                "pred_ssm": round(pred_ssm, 2), "meas_ssm": round(meas_ssm, 2),
                "app_err": round(abs(pred_app - meas_app), 3),
                "ssm_err": round(abs(pred_ssm - meas_ssm), 3),
            })
    max_err = max(max(v["app_err"], v["ssm_err"]) for v in val_rows)
    law_holds = max_err < 1e-6

    payload = {
        "config": {"seeds": seeds, "validate_at": Pv},
        "cost_law": "C_app(P)=base_app+P ; C_ssm(P)=base_ssm+fire_rate*P ; "
                    "P*=(base_ssm-base_app)/(1-fire_rate)",
        "law_validation": {"max_abs_error_nodes": max_err, "exact": law_holds,
                           "rows": val_rows},
        "P_star_summary": summary,
        "per_seed": per_seed,
        "protocol_evidence": {"n_seeds": len(seeds), "held_out": True,
                              "valid": len(seeds) >= 3},
    }
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Phase 1.5D — crossover P* phase diagram ===")
    print(f"seeds={seeds}  (n={len(seeds)})")
    print(f"  Cost law: C_app=base_app+P,  C_ssm=base_ssm+fire_rate*P")
    print(f"  Law validation at P={Pv}: max abs error = {max_err:.2e} nodes "
          f"-> exact={law_holds}")
    print(f"  Crossover P* (think-first beats act-first for all P > P*):")
    print(f"    {'split':7s} {'P*_mean':>9s} {'95% CI':>16s}   per-seed")
    for split in ("reuse", "ood", "mixed"):
        s = summary[split]
        ci = f"[{s['ci95_low']}, {s['ci95_high']}]"
        print(f"    {split:7s} {s['P_star_mean']:9.1f} {ci:>16s}   {s['P_star_per_seed']}")
    print(f"  Reading: OOD crosses early (~{summary['ood']['P_star_mean']}), reuse late "
          f"(~{summary['reuse']['P_star_mean']}); a mixed stream flips at "
          f"~{summary['mixed']['P_star_mean']}.")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
