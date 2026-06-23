#!/usr/bin/env python3
"""ECS Stage 1a' — the NAMED FIX from MODEL_THESIS.md §6c.

Stage 1a (`ecs_stage1.py`) returned PASS=False with a precise diagnosis: the
trained gate selects skills by an *independent per-step argmax* and commits
greedily, so it either ABSTAINS (-> dense behavior, skills add nothing) or
MISFIRES (-> overshoots, solve-rate drops). `random-routing` blowing up to 22.6
proved firing is only safe with the RIGHT skill, and the naive head cannot pick
it. That is the same goal-conditioned-inverse wall the 3.3-3.4 structural arc
isolated.

The named fix: replace greedy single-action commit with Phase-3.4c-style
**best-first ordering over the action choice** -- the policy's action log-probs
become a search priority, so a wrong skill is recoverable by backtracking
instead of fatal. Everything else is held fixed (same stream, same training,
same six arms, same Axis-C and frozen-backbone tests), exactly per G1.

HONESTY CORRECTION ON THE METRIC. Under search, "active compute" can NOT be the
emitted path length -- that would let the selector spend unbounded *unmeasured*
search to find a short path. The faithful conditional-compute metric is the
number of **policy forward-passes** (one per expanded node), counted identically
for every arm (dense expands primitive-only nodes; ECS expands primitive+macro
nodes; cache-M is a 1-forward lookup; random-routing searches with randomized
among-skill ordering at the same gate fire-rate). If macros make the search
*branchier* without shortening it enough, ECS costs MORE forwards -- an honest
FAIL, not a hidden win.

Falsifiable prediction (MODEL_THESIS §6c): with competent selection, ECS matches
dense solve-rate AND beats dense+curriculum / random-routing on active compute
(forwards) AND survives Axis-C. If it still fails to beat curriculum, the law is
not architectural at this scale -- and we say so.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import statistics
from pathlib import Path

import numpy as np

import skill_compression_bench as B
import skill_compression_stream as S
from ecs_stage1 import (D, P, PRIM, ECSPolicy, features, slope, softmax,
                        teacher_actions)

NEG_INF = -1e9


# ---------------------------------------------------------------------------
# Action scoring: turn the policy heads into per-action log-probs at state x.
# ECS: P(action) = gate * P(skill_k) for macros, (1-gate) * P(prim_i) for prims.
# dense / frozen-M (no macros): primitives only, gate ignored.
# random: among-skill ordering randomized at the SAME gate split as ECS (G6 -
#         marginal fire-rate preserved, which-skill randomized).
# Returns list of (logprob, next_x); also reports gate-fire at the query state.
# ---------------------------------------------------------------------------
def action_scores(policy, x, t, macros, mode, rng):
    h, pg, zp, zs = policy.forward(features(x, t))
    sp = softmax(zp)
    lp_prim = np.log(sp + 1e-12)
    out = []
    # cache-M on a held-out MISS has no generalizing abstraction -> primitives only,
    # exactly like dense (the cache only ever shortcuts memorized train pairs).
    if mode in ("dense", "cache") or not macros:
        for i in range(P):
            out.append((float(lp_prim[i]), B.PRIMITIVES[PRIM[i]](x)))
        return out, bool(pg >= 0.5)
    log_pg = math.log(float(pg) + 1e-12)
    log_npg = math.log(1.0 - float(pg) + 1e-12)
    for i in range(P):
        out.append((float(lp_prim[i] + log_npg), B.PRIMITIVES[PRIM[i]](x)))
    n = len(macros)
    if mode == "random":
        ls = np.log(softmax(rng.normal(size=n)) + 1e-12)   # which-skill randomized
    else:
        ls = np.log(softmax(zs[:n]) + 1e-12)
    for k in range(n):
        out.append((float(ls[k] + log_pg), macros[k].apply(x)))
    return out, bool(pg >= 0.5)


# ---------------------------------------------------------------------------
# Policy-guided best-first search to the target. Priority = cumulative negative
# log-prob (most-probable path first). Graph search over the integer state
# (Markov), so visited-pruning keeps it finite. COST = forward-passes (one per
# expanded node). Returns (forwards, solved, path_len, fired_at_start).
# ---------------------------------------------------------------------------
def best_first(policy, x0, t, macros, mode, node_budget, max_depth, rng, cache=None):
    if mode == "cache" and cache is not None and (x0, t) in cache:
        return 1, True, 0, False
    _h, pg0, _zp, _zs = policy.forward(features(x0, t))
    fired = bool(pg0 >= 0.5)
    if x0 == t:
        return 0, True, 0, fired
    pq = [(0.0, 0, x0, 0)]
    best_cost = {x0: 0.0}
    counter = 1
    forwards = 0
    while pq and forwards < node_budget:
        neglp, _, x, depth = heapq.heappop(pq)
        if x == t:
            return forwards, True, depth, fired
        if depth >= max_depth:
            continue
        if neglp > best_cost.get(x, NEG_INF) + 1e-9:
            continue
        forwards += 1
        children, _ = action_scores(policy, x, t, macros, mode, rng)
        for clp, cx in children:
            ncost = neglp - clp
            if cx in best_cost and best_cost[cx] <= ncost + 1e-9:
                continue
            best_cost[cx] = ncost
            heapq.heappush(pq, (ncost, counter, cx, depth + 1))
            counter += 1
    # one final drain for a target sitting at the frontier within budget
    while pq:
        neglp, _, x, depth = heapq.heappop(pq)
        if x == t:
            return forwards, True, depth, fired
    return forwards, False, max_depth, fired


def eval_split(policy, tasks, macros, mode, node_budget, max_depth, seed, cache=None):
    rng = np.random.default_rng(seed * 131 + 7)
    fwds, solved, fires, plens = [], 0, 0, []
    for x0, t in tasks:
        f, ok, pl, fired = best_first(policy, x0, t, macros, mode,
                                      node_budget, max_depth, rng, cache)
        fwds.append(f); solved += int(ok); fires += int(fired)
        if ok:
            plens.append(pl)
    n = len(tasks)
    return {"mean_forwards": statistics.mean(fwds), "solve_rate": solved / n,
            "fire_rate": fires / n,
            "mean_path": (statistics.mean(plens) if plens else 0.0)}


# ---------------------------------------------------------------------------
# Train online over a stream (teacher-forced action labels, identical to Stage
# 1a). The eval curve A(e) uses best-first so the measured compute is the
# selector's actual search cost.
# ---------------------------------------------------------------------------
def train_stream(seed, stream, hidden, kmax, lr, steps_per_task, batch_size,
                 max_depth, budget, grow, eval_every, eval_tasks, node_budget,
                 bf_depth, freeze_after):
    rng = np.random.default_rng(seed * 1009 + 3)
    pol = ECSPolicy(hidden, kmax, seed)
    lib = S.OnlineLibrary(1.0, 2.0, kmax) if grow else None
    buf = []
    curve = []

    def macros_now():
        return lib.actions[P:] if lib else []

    for i, (x0, t) in enumerate(stream):
        actions = (B.primitive_actions() + macros_now()) if lib else B.primitive_actions()
        path = teacher_actions(x0, t, actions, max_depth, budget)
        if path is not None:
            x = x0
            for act in path:
                u = features(x, t)
                if act.name in PRIM:
                    buf.append((u, 0.0, "prim", PRIM.index(act.name)))
                else:
                    buf.append((u, 1.0, "skill", int(act.name[1:])))
                x = act.apply(x)
            if lib:
                prims = [p for a in path for p in a.prims]
                lib.observe(prims, i)
        if buf:
            fb = (freeze_after is not None and i >= freeze_after)
            for _ in range(steps_per_task):
                idx = rng.integers(0, len(buf), size=min(batch_size, len(buf)))
                pol.train_batch([buf[j] for j in idx], lr, freeze_backbone=fb)
        if (i + 1) % eval_every == 0:
            ev = eval_split(pol, eval_tasks, macros_now(), "ecs" if lib else "dense",
                            node_budget, bf_depth, seed)
            curve.append((i + 1, round(ev["mean_forwards"], 3),
                          round(ev["solve_rate"], 3), round(ev["fire_rate"], 3)))
    return pol, lib, curve


def run_seed(seed, cfg):
    import random as _r
    tp = S.train_pool()
    pr = _r.Random(seed)
    stream = [S.gen_task(pr, tp, (1, 2)) for _ in range(cfg.train_tasks)]
    train_keys = {(x0, t) for x0, t in stream}
    held = S.sample_disjoint(_r.Random(seed + 1), cfg.eval_tasks, tp, (1, 2), train_keys)
    axisC = S.sample_disjoint(_r.Random(seed + 2), cfg.eval_tasks, tp, (3, 3), train_keys)

    nb, bd = cfg.node_budget, cfg.bf_depth
    kw = dict(hidden=cfg.hidden, kmax=cfg.max_macros, lr=cfg.lr,
              steps_per_task=cfg.steps_per_task, batch_size=cfg.batch,
              max_depth=cfg.max_depth, budget=cfg.budget, eval_every=cfg.eval_every,
              eval_tasks=held, node_budget=nb, bf_depth=bd)

    ecs_pol, ecs_lib, ecs_curve = train_stream(seed, stream, grow=True, freeze_after=None, **kw)
    ecs_macros = ecs_lib.actions[P:]
    _, _, densecur_curve = train_stream(seed, stream, grow=False, freeze_after=None, **kw)
    shuf = list(stream); _r.Random(seed + 99).shuffle(shuf)
    dense_pol, _, dense_curve = train_stream(seed, shuf, grow=False, freeze_after=None, **kw)
    _, _, fb_curve = train_stream(seed, stream, grow=True,
                                  freeze_after=cfg.train_tasks // 3, **kw)

    ecs_final = eval_split(ecs_pol, held, ecs_macros, "ecs", nb, bd, seed)
    dense_final = eval_split(dense_pol, held, [], "dense", nb, bd, seed)
    frozenM_final = eval_split(ecs_pol, held, [], "dense", nb, bd, seed)
    cacheM_final = eval_split(ecs_pol, held, ecs_macros, "cache", nb, bd, seed,
                              cache=set(train_keys))
    rand_final = eval_split(ecs_pol, held, ecs_macros, "random", nb, bd, seed)
    ecs_axisC = eval_split(ecs_pol, axisC, ecs_macros, "ecs", nb, bd, seed)
    dense_axisC = eval_split(dense_pol, axisC, [], "dense", nb, bd, seed)

    return {
        "seed": seed, "n_macros": len(ecs_macros),
        "curves": {"ecs": ecs_curve, "dense": dense_curve,
                   "dense_curriculum": densecur_curve, "frozen_backbone": fb_curve},
        "slopes": {"ecs": round(slope(ecs_curve), 4),
                   "dense": round(slope(dense_curve), 4),
                   "dense_curriculum": round(slope(densecur_curve), 4),
                   "frozen_backbone": round(slope(fb_curve), 4)},
        "final": {"ecs": ecs_final, "dense": dense_final, "frozenM": frozenM_final,
                  "cacheM": cacheM_final, "random_routing": rand_final},
        "axisC": {"ecs": ecs_axisC, "dense": dense_axisC},
    }


def aggregate(runs):
    def fm(path):
        vals = []
        for r in runs:
            o = r
            for k in path:
                o = o[k]
            vals.append(o)
        return round(statistics.mean(vals), 3)

    out = {}
    out["slope_ecs"] = fm(["slopes", "ecs"])
    out["slope_dense"] = fm(["slopes", "dense"])
    out["slope_dense_curriculum"] = fm(["slopes", "dense_curriculum"])
    out["slope_frozen_backbone"] = fm(["slopes", "frozen_backbone"])
    out["ecs_final_fwds"] = fm(["final", "ecs", "mean_forwards"])
    out["ecs_final_solve"] = fm(["final", "ecs", "solve_rate"])
    out["ecs_final_path"] = fm(["final", "ecs", "mean_path"])
    out["dense_final_fwds"] = fm(["final", "dense", "mean_forwards"])
    out["dense_final_solve"] = fm(["final", "dense", "solve_rate"])
    out["frozenM_fwds"] = fm(["final", "frozenM", "mean_forwards"])
    out["cacheM_fwds"] = fm(["final", "cacheM", "mean_forwards"])
    out["random_routing_fwds"] = fm(["final", "random_routing", "mean_forwards"])
    out["ecs_axisC_fwds"] = fm(["axisC", "ecs", "mean_forwards"])
    out["dense_axisC_fwds"] = fm(["axisC", "dense", "mean_forwards"])
    out["ecs_axisC_solve"] = fm(["axisC", "ecs", "solve_rate"])

    matched = abs(out["ecs_final_solve"] - out["dense_final_solve"]) <= 0.05
    cheaper = (out["ecs_final_fwds"] < out["dense_final_fwds"]
               and out["ecs_final_fwds"] < out["frozenM_fwds"]
               and out["ecs_final_fwds"] < out["cacheM_fwds"]
               and out["ecs_final_fwds"] < out["random_routing_fwds"])
    declines = out["slope_ecs"] < 0 and out["slope_ecs"] < out["slope_dense_curriculum"]
    survives_axisC = out["ecs_axisC_fwds"] < out["dense_axisC_fwds"]
    survives_frozenbb = out["slope_frozen_backbone"] < 0
    out["verdict"] = {
        "matched_solve_rate": matched,
        "ecs_cheaper_than_all_controls": cheaper,
        "ecs_declines_vs_curriculum": declines,
        "survives_axisC": survives_axisC,
        "survives_frozen_backbone": survives_frozenbb,
        "PASS": matched and cheaper and declines and survives_axisC and survives_frozenbb,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--train-tasks", type=int, default=240)
    ap.add_argument("--eval-tasks", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--max-macros", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--steps-per-task", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--budget", type=int, default=200000)
    ap.add_argument("--node-budget", type=int, default=256)
    ap.add_argument("--bf-depth", type=int, default=16)
    ap.add_argument("--output-json", default="artifacts/ecs_stage1b_selection.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    runs = [run_seed(s, args) for s in seeds]
    agg = aggregate(runs)
    protocol = {
        "held_out": True,
        "controls_present": ["dense", "dense_curriculum", "frozen_M", "cache_M",
                             "random_routing", "frozen_backbone"],
        "axis_C_present": True, "matched_loss_guard": True,
        "compute_metric": "policy_forward_passes (nodes expanded in best-first); "
                          "counted identically for every arm so search cost cannot hide",
        "selector": "policy-guided best-first search (Phase-3.4c), replacing Stage-1a "
                    "greedy independent-argmax commit",
        "n_seeds": len(seeds), "valid": len(seeds) >= 3,
    }
    payload = {"config": vars(args), "protocol_evidence": protocol,
               "aggregate": agg, "seed_runs": runs}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    v = agg["verdict"]
    print("=== ECS Stage 1a' — best-first selection, active-compute = forward-passes ===")
    print(f"seeds={seeds}  protocol_valid={protocol['valid']}  "
          f"macros≈{round(statistics.mean(r['n_macros'] for r in runs), 1)}")
    print("\n  active compute (policy forward-passes) at full exposure / solve-rate:")
    print(f"    ECS            : {agg['ecs_final_fwds']:7.2f}  solve {agg['ecs_final_solve']}  "
          f"(path {agg['ecs_final_path']})")
    print(f"    dense (shuf)   : {agg['dense_final_fwds']:7.2f}  solve {agg['dense_final_solve']}")
    print(f"    dense+curric   : slope {agg['slope_dense_curriculum']:+.4f}")
    print(f"    frozen-M       : {agg['frozenM_fwds']:7.2f}")
    print(f"    cache-M        : {agg['cacheM_fwds']:7.2f}")
    print(f"    random-routing : {agg['random_routing_fwds']:7.2f}")
    print(f"  slopes (forwards/task vs exposure):  ECS {agg['slope_ecs']:+.4f}  "
          f"dense {agg['slope_dense']:+.4f}  frozen-bb {agg['slope_frozen_backbone']:+.4f}")
    print(f"  Axis-C (novel composition):  ECS {agg['ecs_axisC_fwds']:.2f} "
          f"(solve {agg['ecs_axisC_solve']}) vs dense {agg['dense_axisC_fwds']:.2f}")
    print(f"\n  VERDICT: matched_solve={v['matched_solve_rate']}  "
          f"cheaper_than_all={v['ecs_cheaper_than_all_controls']}")
    print(f"           declines_vs_curric={v['ecs_declines_vs_curriculum']}  "
          f"axisC={v['survives_axisC']}  frozen_bb={v['survives_frozen_backbone']}")
    print(f"  => STAGE 1a' PASS = {v['PASS']}")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
