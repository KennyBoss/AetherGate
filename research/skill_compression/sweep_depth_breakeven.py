#!/usr/bin/env python3
"""Stage 1a' follow-up — the pre-registered DEPTH BREAK-EVEN test (MODEL_THESIS §6d).

Diagnosis from Stage 1a': skills beat their own misuse/ablation but not their own
absence (dense), because the macro-augmented action set adds search branching that
~cancels its path-shortening AT SHALLOW DEPTH. Falsifiable prediction: as eval
composition depth grows, ECS forward-passes should cross BELOW dense at some d*,
and the gap should widen. If ECS never crosses dense, the apparatus fails here.

Trains ECS (growing M) and dense (shuffled, primitives) once per seed on the SAME
shallow (depth 1-2) stream as Stage 1a', then evaluates BOTH on held-out splits at
composition depths 1..5. Active compute = best-first forward-passes, same metric.
"""
from __future__ import annotations

import argparse
import json
import random as _r
import statistics
from pathlib import Path

import skill_compression_stream as S
from ecs_stage1 import P
from ecs_stage1b_selection import eval_split, train_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73")
    ap.add_argument("--train-tasks", type=int, default=240)
    ap.add_argument("--eval-tasks", type=int, default=80)
    ap.add_argument("--depths", default="1,2,3,4,5")
    ap.add_argument("--node-budget", type=int, default=512)
    ap.add_argument("--bf-depth", type=int, default=24)
    ap.add_argument("--output-json", default="artifacts/sweep_depth_breakeven.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    depths = [int(d) for d in args.depths.split(",")]
    nb, bd = args.node_budget, args.bf_depth

    kw = dict(hidden=32, kmax=8, lr=0.02, steps_per_task=4, batch_size=32,
              max_depth=8, budget=200000, eval_every=10_000,  # no curve needed
              node_budget=nb, bf_depth=bd)

    per_depth = {d: {"ecs": [], "dense": [], "ecs_solve": [], "dense_solve": []} for d in depths}
    for seed in seeds:
        tp = S.train_pool()
        pr = _r.Random(seed)
        stream = [S.gen_task(pr, tp, (1, 2)) for _ in range(args.train_tasks)]
        train_keys = {(x0, t) for x0, t in stream}

        ecs_pol, ecs_lib, _ = train_stream(seed, stream, grow=True, freeze_after=None,
                                           eval_tasks=stream[:1], **kw)
        ecs_macros = ecs_lib.actions[P:]
        shuf = list(stream); _r.Random(seed + 99).shuffle(shuf)
        dense_pol, _, _ = train_stream(seed, shuf, grow=False, freeze_after=None,
                                       eval_tasks=stream[:1], **kw)

        for d in depths:
            ev = S.sample_disjoint(_r.Random(seed + 100 + d), args.eval_tasks, tp, (d, d), train_keys)
            if len(ev) < 5:   # shallow depths are ~fully covered by the train stream
                continue
            e = eval_split(ecs_pol, ev, ecs_macros, "ecs", nb, bd, seed)
            dn = eval_split(dense_pol, ev, [], "dense", nb, bd, seed)
            per_depth[d]["ecs"].append(e["mean_forwards"])
            per_depth[d]["dense"].append(dn["mean_forwards"])
            per_depth[d]["ecs_solve"].append(e["solve_rate"])
            per_depth[d]["dense_solve"].append(dn["solve_rate"])

    rows = []
    for d in depths:
        pd = per_depth[d]
        if not pd["ecs"]:
            continue
        ecs_m = statistics.mean(pd["ecs"]); dn_m = statistics.mean(pd["dense"])
        wins = sum(1 for a, b in zip(pd["ecs"], pd["dense"]) if a < b)
        rows.append({"depth": d, "ecs_fwd": round(ecs_m, 2), "dense_fwd": round(dn_m, 2),
                     "ecs_minus_dense": round(ecs_m - dn_m, 2),
                     "ecs_cheaper": ecs_m < dn_m, "ecs_win_seeds": f"{wins}/{len(seeds)}",
                     "ecs_solve": round(statistics.mean(pd["ecs_solve"]), 3),
                     "dense_solve": round(statistics.mean(pd["dense_solve"]), 3)})

    crossed = [r["depth"] for r in rows if r["ecs_cheaper"]]
    d_star = min(crossed) if crossed else None
    monotone = all(rows[i]["ecs_minus_dense"] >= rows[i + 1]["ecs_minus_dense"]
                   for i in range(len(rows) - 1))
    payload = {"config": vars(args), "rows": rows,
               "d_star": d_star, "gap_widens_with_depth": monotone,
               "n_seeds": len(seeds), "valid": len(seeds) >= 3}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Depth break-even test — ECS vs dense forward-passes by composition depth ===")
    print(f"seeds={seeds}  (train depth 1-2; eval depths {depths})\n")
    print(f"  {'depth':>5} {'ECS':>8} {'dense':>8} {'ECS-dense':>10} {'cheaper':>8} {'win':>6} {'solve(E/D)':>12}")
    for r in rows:
        print(f"  {r['depth']:>5} {r['ecs_fwd']:>8.2f} {r['dense_fwd']:>8.2f} "
              f"{r['ecs_minus_dense']:>+10.2f} {str(r['ecs_cheaper']):>8} {r['ecs_win_seeds']:>6} "
              f"{r['ecs_solve']:>5}/{r['dense_solve']:<5}")
    print(f"\n  d* (first depth ECS<dense in mean): {d_star}   gap widens with depth: {monotone}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
