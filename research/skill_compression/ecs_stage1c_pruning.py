#!/usr/bin/env python3
"""ECS Stage 1a'' — utility pruning (Path A), and an honest reframing of the other half.

MODEL_THESIS §6d pre-registered a two-part next test: (a) online macro pruning by
marginal forward-pass utility, and (b) promotion that recovers the *generative* hidden
subroutines (match-rate ↑ from ~0). Before building (b), a recoverability probe
(`subroutine_recoverability`, below) shows part (b) is **ill-posed in this DSL**, not
merely unsolved:

    integer state COLLAPSES (many programs reach the same number), so the
    shortest-compute solution is strictly shorter than the generative program ~86% of
    the time and contains a whole hidden subroutine only ~3.8% of the time.

So "recover the generative subroutines" cannot be satisfied under shortest-path
supervision — the optimal solution almost never uses them. We REPORT this as the finding
(not move the goalpost): in a collapsing-state symbolic DSL the only learnable skill is
"recurring optimal-path n-gram", which is exactly why a rich-state environment (real
code / AetherGate, where the generative structure is observable) is the right venue for
the generative-recovery claim. That is the principled bridge to Stage 2, recorded
honestly rather than smuggled.

This script builds the WELL-POSED half — Path A, online utility pruning — and asks the
falsifiable question: does an adaptively-pruned library beat the no-skill dense baseline
on held-out AND survive Axis-C (forward-passes), where the unpruned kmax=8 library did
not? Reuses the canonical best-first selector + metric from `ecs_stage1b_selection.py`
(active compute = policy forward-passes, counted identically per arm). 6 seeds,
deterministic, JSON with protocol_evidence.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np

import skill_compression_bench as B
import skill_compression_stream as S
from ecs_stage1 import D, P, PRIM, ECSPolicy, features, slope, teacher_actions
from ecs_stage1b_selection import action_scores, best_first, eval_split, train_stream

HIDDEN = {"".join(v) for v in B.HIDDEN_SUBROUTINES.values()}


# ---------------------------------------------------------------------------
# Diagnostic: are the generative subroutines recoverable from shortest-path
# supervision at all? (Reported as a finding; it justifies the Stage-2 bridge.)
# ---------------------------------------------------------------------------
def subroutine_recoverability(seed, n, max_depth, budget):
    pr = random.Random(seed)
    tp = S.train_pool()
    prim = B.primitive_actions()
    bypass, contains = 0, 0
    glen, slen = [], []
    for _ in range(n):
        prog = []
        for _ in range(pr.randint(1, 2)):
            prog += pr.choice(tp)
        x0 = pr.randint(0, 5)
        t = B.apply_prims(x0, prog)
        path = teacher_actions(x0, t, prim, max_depth, budget)
        sp = "".join(a.name for a in path) if path else ""
        glen.append(len(prog)); slen.append(len(sp))
        if len(sp) < len(prog):
            bypass += 1
        if any(s in sp for s in HIDDEN):
            contains += 1
    return {"mean_generative_len": round(statistics.mean(glen), 3),
            "mean_shortest_len": round(statistics.mean(slen), 3),
            "shortest_bypasses_generative_rate": round(bypass / n, 4),
            "shortest_contains_hidden_subroutine_rate": round(contains / n, 4)}


# ---------------------------------------------------------------------------
# Path A — online utility pruning. After growth, score each macro by its marginal
# value: forward-passes saved when the library INCLUDES it vs EXCLUDES it, on a
# small *training-side* probe set (NEVER the held-out / Axis-C splits — G: growth &
# pruning stay blind to eval). Drop macros with utility <= 0; keep the top-k useful.
# ---------------------------------------------------------------------------
def macro_marginal_utility(policy, macros, probe_tasks, node_budget, bf_depth, seed):
    """Return per-macro marginal forward-pass saving (positive = worth keeping)."""
    base, _, _, _stats = _mean_forwards(policy, probe_tasks, macros, node_budget, bf_depth, seed)
    utils = []
    for k in range(len(macros)):
        without = macros[:k] + macros[k + 1:]
        f_without, _, _, _ = _mean_forwards(policy, probe_tasks, without, node_budget, bf_depth, seed)
        # marginal utility of macro k = extra forwards WITHOUT it (it saved that many)
        utils.append(f_without - base)
    return utils


def _mean_forwards(policy, tasks, macros, node_budget, bf_depth, seed):
    ev = eval_split(policy, tasks, macros, "ecs" if macros else "dense",
                    node_budget, bf_depth, seed)
    return ev["mean_forwards"], ev["solve_rate"], ev["fire_rate"], ev


def prune_library(policy, macros, probe_tasks, node_budget, bf_depth, seed, keep_positive=True):
    if not macros:
        return [], []
    utils = macro_marginal_utility(policy, macros, probe_tasks, node_budget, bf_depth, seed)
    # keep macros whose removal COSTS forwards (utility > tol); deterministic order
    tol = 1e-6
    kept = [m for m, u in zip(macros, utils) if (u > tol if keep_positive else True)]
    kept_utils = [round(u, 4) for u in utils]
    return kept, kept_utils


def run_seed(seed, cfg):
    tp = S.train_pool()
    pr = random.Random(seed)
    stream = [S.gen_task(pr, tp, (1, 2)) for _ in range(cfg.train_tasks)]
    keys = {(x0, t) for x0, t in stream}
    held = S.sample_disjoint(random.Random(seed + 1), cfg.eval_tasks, tp, (1, 2), keys)
    axisC = S.sample_disjoint(random.Random(seed + 2), cfg.eval_tasks, tp, (3, 3), keys)
    # pruning probe set: training-distribution tasks, disjoint from held/axisC
    probe = S.sample_disjoint(random.Random(seed + 3), cfg.probe_tasks, tp, (1, 2), keys)
    probeC = S.sample_disjoint(random.Random(seed + 4), cfg.probe_tasks, tp, (3, 3), keys)

    nb, bd = cfg.node_budget, cfg.bf_depth
    kw = dict(hidden=cfg.hidden, kmax=cfg.max_macros, lr=cfg.lr,
              steps_per_task=cfg.steps_per_task, batch_size=cfg.batch,
              max_depth=cfg.max_depth, budget=cfg.budget, eval_every=cfg.eval_every,
              eval_tasks=held, node_budget=nb, bf_depth=bd)

    ecs_pol, ecs_lib, _ = train_stream(seed, stream, grow=True, freeze_after=None, **kw)
    macros = ecs_lib.actions[P:]
    shuf = list(stream); random.Random(seed + 99).shuffle(shuf)
    dense_pol, _, _ = train_stream(seed, shuf, grow=False, freeze_after=None, **kw)

    # prune using a DEPTH-3 probe (utility must reflect the regime macros are meant
    # to help — composition; held-out depth-1-2 barely needs them) — still blind to eval.
    pruned, utils = prune_library(ecs_pol, macros, probeC, nb, bd, seed)

    macro_strs = ["".join(m.prims) for m in macros]
    pruned_strs = ["".join(m.prims) for m in pruned]
    match_unpruned = sum(1 for s in macro_strs if s in HIDDEN)

    def ev(pol, tasks, mac, mode="ecs"):
        return eval_split(pol, tasks, mac, mode if mac else "dense", nb, bd, seed)

    out = {
        "seed": seed,
        "n_macros_unpruned": len(macros), "n_macros_pruned": len(pruned),
        "macros_unpruned": macro_strs, "macros_pruned": pruned_strs, "utils": utils,
        "hidden_subroutine_match_unpruned": match_unpruned,
        "held": {
            "ecs_unpruned": ev(ecs_pol, held, macros),
            "ecs_pruned": ev(ecs_pol, held, pruned),
            "dense": ev(dense_pol, held, []),
        },
        "axisC": {
            "ecs_unpruned": ev(ecs_pol, axisC, macros),
            "ecs_pruned": ev(ecs_pol, axisC, pruned),
            "dense": ev(dense_pol, axisC, []),
        },
    }
    return out


def aggregate(runs, recover):
    def M(path):
        vals = []
        for r in runs:
            o = r
            for k in path:
                o = o[k]
            vals.append(o)
        return round(statistics.mean(vals), 3)

    a = {
        "n_unpruned": M(["n_macros_unpruned"]), "n_pruned": M(["n_macros_pruned"]),
        "match_unpruned": M(["hidden_subroutine_match_unpruned"]),
        "held_ecs_unpruned": M(["held", "ecs_unpruned", "mean_forwards"]),
        "held_ecs_pruned": M(["held", "ecs_pruned", "mean_forwards"]),
        "held_dense": M(["held", "dense", "mean_forwards"]),
        "held_ecs_pruned_solve": M(["held", "ecs_pruned", "solve_rate"]),
        "held_dense_solve": M(["held", "dense", "solve_rate"]),
        "axisC_ecs_unpruned": M(["axisC", "ecs_unpruned", "mean_forwards"]),
        "axisC_ecs_pruned": M(["axisC", "ecs_pruned", "mean_forwards"]),
        "axisC_dense": M(["axisC", "dense", "mean_forwards"]),
        "axisC_ecs_pruned_solve": M(["axisC", "ecs_pruned", "solve_rate"]),
    }
    # per-seed win counts (robustness, not just mean)
    held_wins = sum(1 for r in runs
                    if r["held"]["ecs_pruned"]["mean_forwards"] < r["held"]["dense"]["mean_forwards"])
    axisC_wins = sum(1 for r in runs
                     if r["axisC"]["ecs_pruned"]["mean_forwards"] < r["axisC"]["dense"]["mean_forwards"])
    n = len(runs)
    pruning_helps_held = a["held_ecs_pruned"] < a["held_ecs_unpruned"]
    beats_dense_held = a["held_ecs_pruned"] < a["held_dense"]
    beats_dense_axisC = a["axisC_ecs_pruned"] < a["axisC_dense"]
    matched = abs(a["held_ecs_pruned_solve"] - a["held_dense_solve"]) <= 0.05
    a["verdict"] = {
        "recoverability": recover,
        "generative_recovery_is_illposed": recover["shortest_bypasses_generative_rate"] > 0.5
        and recover["shortest_contains_hidden_subroutine_rate"] < 0.2,
        "pruning_reduces_forwards_vs_unpruned": pruning_helps_held,
        "pruned_beats_dense_held": beats_dense_held,
        "pruned_beats_dense_held_seedfrac": f"{held_wins}/{n}",
        "pruned_beats_dense_axisC": beats_dense_axisC,
        "pruned_beats_dense_axisC_seedfrac": f"{axisC_wins}/{n}",
        "matched_solve_rate": matched,
        "PASS_pathA": matched and beats_dense_held and beats_dense_axisC
        and held_wins >= (n // 2 + 1) and axisC_wins >= (n // 2 + 1),
    }
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--train-tasks", type=int, default=240)
    ap.add_argument("--eval-tasks", type=int, default=80)
    ap.add_argument("--probe-tasks", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--max-macros", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--steps-per-task", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--budget", type=int, default=200000)
    ap.add_argument("--node-budget", type=int, default=256)
    ap.add_argument("--bf-depth", type=int, default=16)
    ap.add_argument("--output-json", default="artifacts/ecs_stage1c_pruning.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    recover = subroutine_recoverability(123, 600, args.max_depth, args.budget)
    runs = [run_seed(s, args) for s in seeds]
    agg = aggregate(runs, recover)
    protocol = {
        "held_out": True, "prune_probe_blind_to_eval": True,
        "compute_metric": "policy forward-passes (best-first), identical per arm",
        "controls_present": ["ecs_unpruned", "dense"], "axis_C_present": True,
        "n_seeds": len(seeds), "valid": len(seeds) >= 3,
    }
    payload = {"config": vars(args), "protocol_evidence": protocol,
               "recoverability_diagnostic": recover, "aggregate": agg, "seed_runs": runs}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    v = agg["verdict"]
    print("=== ECS Stage 1a'' — utility pruning (Path A) + recoverability diagnostic ===")
    print(f"seeds={seeds}  protocol_valid={protocol['valid']}")
    print(f"\n  RECOVERABILITY (why generative-subroutine recovery is ill-posed here):")
    print(f"    shortest path bypasses generative program : {recover['shortest_bypasses_generative_rate']:.1%}")
    print(f"    shortest path contains a hidden subroutine : {recover['shortest_contains_hidden_subroutine_rate']:.1%}")
    print(f"    => generative_recovery_is_illposed = {v['generative_recovery_is_illposed']}")
    print(f"\n  PATH A — utility pruning (forward-passes):  unpruned≈{agg['n_unpruned']} -> pruned≈{agg['n_pruned']} macros")
    print(f"    held-out : ecs_unpruned {agg['held_ecs_unpruned']:.2f}  ecs_pruned {agg['held_ecs_pruned']:.2f}  dense {agg['held_dense']:.2f}")
    print(f"    Axis-C   : ecs_unpruned {agg['axisC_ecs_unpruned']:.2f}  ecs_pruned {agg['axisC_ecs_pruned']:.2f}  dense {agg['axisC_dense']:.2f}")
    print(f"  pruning<unpruned={v['pruning_reduces_forwards_vs_unpruned']}  "
          f"pruned<dense held={v['pruned_beats_dense_held']} ({v['pruned_beats_dense_held_seedfrac']})  "
          f"axisC={v['pruned_beats_dense_axisC']} ({v['pruned_beats_dense_axisC_seedfrac']})")
    print(f"  matched_solve={v['matched_solve_rate']}")
    print(f"  => PATH A PASS = {v['PASS_pathA']}")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
