#!/usr/bin/env python3
"""ECS Stage 1b — disentangle SHORTCUT learning from PROGRAM learning.

Stage 1a'' produced the line's deepest result: under shortest-path supervision the
optimal solution BYPASSES the generative program 89% of the time and contains a whole
hidden subroutine only 2.2% — so ECS-as-built is a *shortcut optimizer under state
collapse*, not a program inducer (MODEL_THESIS §6e). The open question that decides
whether the whole compression thesis is salvageable is whether that is (A) a property of
the SUPERVISION or (B) a property of the DOMAIN. These are separable cheaply, in the same
DSL, without moving to AetherGate — and they must be separated BEFORE any domain change,
because a domain swap only helps if the blocker is identifiability, not sufficiency.

Two axes, two independent tests:

(A) IDENTIFIABILITY — does the program become recoverable if supervision is ALIGNED to
    it? Compare promotion driven by two teachers over the same stream:
      * shortest-path teacher (Stage-1a': bypasses the program) ->  match-rate ~0
      * GENERATIVE teacher (the actual subroutine decomposition that generated the task)
    Measure promoted-macro <-> hidden-subroutine match-rate under each. If generative
    supervision lifts match-rate from ~0, the program WAS identifiable — the prior
    failure was the supervision signal, not the domain.

(B) SUFFICIENCY — if the library HELD the true programs, would they pay? Hand the system
    an ORACLE library = exactly the four hidden subroutines, and run the canonical
    best-first selector. Does oracle-ECS beat dense on held-out AND survive Axis-C
    (forward-passes)? This is the ceiling for ANY promotion method; if even the perfect
    library does not beat dense, no domain change rescues the thesis at this scale.

Pre-registered decision table (NOT moved post-hoc):
  A=no                  -> DSL fundamentally collapsed; shortcut-only is intrinsic.
  A=yes, B=no           -> programs identifiable but don't pay -> thesis fails at scale.
  A=yes, B=yes          -> shortcut-dominance was a SUPERVISION artifact; true programs
                           help & transfer -> a program-revealing domain (Stage 2) is
                           justified, and AetherGate is the right venue (earned).

6 seeds, deterministic, CPU/numpy. JSON with protocol_evidence.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

import skill_compression_bench as B
import skill_compression_stream as S
from ecs_stage1 import D, P, PRIM, ECSPolicy, features, slope, teacher_actions
from ecs_stage1b_selection import eval_split, train_stream

HIDDEN_LIST = [list(v) for v in B.HIDDEN_SUBROUTINES.values()]   # [['c','a','c'], ...]
HIDDEN_STR = {"".join(v) for v in B.HIDDEN_SUBROUTINES.values()}


# ---------------------------------------------------------------------------
# Generative task: return BOTH (x0, target) AND the generative subroutine program,
# so a generative teacher can supervise on the true decomposition.
# ---------------------------------------------------------------------------
def gen_task_with_program(rng, pool, depth_range):
    n = rng.randint(*depth_range)
    chosen = [rng.choice(pool) for _ in range(n)]           # list of subroutine prim-lists
    prog = [p for sub in chosen for p in sub]
    x0 = rng.randint(0, 5)
    return x0, B.apply_prims(x0, prog), chosen


# ---------------------------------------------------------------------------
# (A) Identifiability — match-rate of online promotion under each teacher.
# Shortcut teacher: shortest primitive path (Stage-1a'). Generative teacher: the
# true subroutine sequence. Promotion = the bench UTIL gate (via OnlineLibrary).
# ---------------------------------------------------------------------------
def promote_under_teacher(seed, n_tasks, teacher, max_depth, budget, kmax):
    rng = random.Random(seed)
    tp = S.train_pool()
    prim = B.primitive_actions()
    lib = S.OnlineLibrary(1.0, 2.0, kmax)
    for i in range(n_tasks):
        x0, t, chosen = gen_task_with_program(rng, tp, (1, 2))
        if teacher == "shortest":
            path = teacher_actions(x0, t, prim, max_depth, budget)
            prims = [a.name for a in path] if path else []
        else:  # generative
            prims = [p for sub in chosen for p in sub]
        if prims:
            lib.observe(prims, i)
    macros = ["".join(m) for m in lib.macro_prims]
    match = sum(1 for m in macros if m in HIDDEN_STR)
    return {"n_macros": len(macros), "macros": macros,
            "hidden_match": match,
            "match_rate": round(match / max(1, len(macros)), 4)}


# ---------------------------------------------------------------------------
# (B) Sufficiency — hand ECS the ORACLE library (the true subroutines) and run the
# canonical best-first selector. Train the policy on the stream (so action heads are
# competent), but the library is fixed = oracle. Compare to dense.
# ---------------------------------------------------------------------------
def oracle_actions():
    return B.primitive_actions() + [B.Action(f"M{i}", list(s)) for i, s in enumerate(HIDDEN_LIST)]


def run_seed(seed, cfg):
    tp = S.train_pool()
    pr = random.Random(seed)
    stream = [S.gen_task(pr, tp, (1, 2)) for _ in range(cfg.train_tasks)]
    keys = {(x0, t) for x0, t in stream}
    held = S.sample_disjoint(random.Random(seed + 1), cfg.eval_tasks, tp, (1, 2), keys)
    axisC = S.sample_disjoint(random.Random(seed + 2), cfg.eval_tasks, tp, (3, 3), keys)
    nb, bd = cfg.node_budget, cfg.bf_depth

    # (A) identifiability
    idA = {
        "shortest": promote_under_teacher(seed, cfg.train_tasks, "shortest",
                                          cfg.max_depth, cfg.budget, cfg.max_macros),
        "generative": promote_under_teacher(seed, cfg.train_tasks, "generative",
                                           cfg.max_depth, cfg.budget, cfg.max_macros),
    }

    # (B) sufficiency — train a policy on the stream, evaluate with oracle library.
    # We reuse train_stream to fit competent action heads, then swap in oracle macros.
    kw = dict(hidden=cfg.hidden, kmax=cfg.max_macros, lr=cfg.lr,
              steps_per_task=cfg.steps_per_task, batch_size=cfg.batch,
              max_depth=cfg.max_depth, budget=cfg.budget, eval_every=cfg.eval_every,
              eval_tasks=held, node_budget=nb, bf_depth=bd)
    # ECS policy trained WITH growing lib (so its skill head is exercised), but we
    # evaluate it against the ORACLE library to measure the sufficiency ceiling.
    ecs_pol, ecs_lib, _ = train_stream(seed, stream, grow=True, freeze_after=None, **kw)
    oracle_macros = oracle_actions()[P:]
    shuf = list(stream); random.Random(seed + 99).shuffle(shuf)
    dense_pol, _, _ = train_stream(seed, shuf, grow=False, freeze_after=None, **kw)

    def ev(pol, tasks, mac, mode="ecs"):
        return eval_split(pol, tasks, mac, mode if mac else "dense", nb, bd, seed)

    sufficiency = {
        "held": {"oracle_ecs": ev(ecs_pol, held, oracle_macros),
                 "grown_ecs": ev(ecs_pol, held, ecs_lib.actions[P:]),
                 "dense": ev(dense_pol, held, [])},
        "axisC": {"oracle_ecs": ev(ecs_pol, axisC, oracle_macros),
                  "grown_ecs": ev(ecs_pol, axisC, ecs_lib.actions[P:]),
                  "dense": ev(dense_pol, axisC, [])},
    }
    return {"seed": seed, "identifiability": idA, "sufficiency": sufficiency}


def aggregate(runs):
    def M(path):
        vals = []
        for r in runs:
            o = r
            for k in path:
                o = o[k]
            vals.append(o)
        return round(statistics.mean(vals), 3)

    a = {
        "match_rate_shortest": M(["identifiability", "shortest", "match_rate"]),
        "match_rate_generative": M(["identifiability", "generative", "match_rate"]),
        "match_count_generative": M(["identifiability", "generative", "hidden_match"]),
        "held_oracle": M(["sufficiency", "held", "oracle_ecs", "mean_forwards"]),
        "held_grown": M(["sufficiency", "held", "grown_ecs", "mean_forwards"]),
        "held_dense": M(["sufficiency", "held", "dense", "mean_forwards"]),
        "held_oracle_solve": M(["sufficiency", "held", "oracle_ecs", "solve_rate"]),
        "held_dense_solve": M(["sufficiency", "held", "dense", "solve_rate"]),
        "axisC_oracle": M(["sufficiency", "axisC", "oracle_ecs", "mean_forwards"]),
        "axisC_grown": M(["sufficiency", "axisC", "grown_ecs", "mean_forwards"]),
        "axisC_dense": M(["sufficiency", "axisC", "dense", "mean_forwards"]),
    }
    n = len(runs)
    held_wins = sum(1 for r in runs
                    if r["sufficiency"]["held"]["oracle_ecs"]["mean_forwards"]
                    < r["sufficiency"]["held"]["dense"]["mean_forwards"])
    axisC_wins = sum(1 for r in runs
                     if r["sufficiency"]["axisC"]["oracle_ecs"]["mean_forwards"]
                     < r["sufficiency"]["axisC"]["dense"]["mean_forwards"])
    A_yes = a["match_rate_generative"] > 0.3 and a["match_rate_shortest"] < 0.1
    B_held = a["held_oracle"] < a["held_dense"]
    B_axisC = a["axisC_oracle"] < a["axisC_dense"]
    B_yes = B_held and B_axisC and held_wins >= (n // 2 + 1) and axisC_wins >= (n // 2 + 1)
    if not A_yes:
        decision = "A=no: program not identifiable even under generative supervision -> DSL intrinsically collapsed"
    elif not B_yes:
        decision = "A=yes,B=no: programs identifiable but do NOT pay -> compression thesis fails at this scale (any domain)"
    else:
        decision = "A=yes,B=yes: shortcut-dominance was a SUPERVISION artifact; true programs help+transfer -> Stage 2 justified"
    a["verdict"] = {
        "A_identifiable_under_generative_supervision": A_yes,
        "B_oracle_beats_dense_held": B_held, "B_held_seedfrac": f"{held_wins}/{n}",
        "B_oracle_beats_dense_axisC": B_axisC, "B_axisC_seedfrac": f"{axisC_wins}/{n}",
        "B_sufficient": B_yes,
        "matched_solve_rate": abs(a["held_oracle_solve"] - a["held_dense_solve"]) <= 0.05,
        "decision": decision,
    }
    return a


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
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--budget", type=int, default=200000)
    ap.add_argument("--node-budget", type=int, default=256)
    ap.add_argument("--bf-depth", type=int, default=16)
    ap.add_argument("--output-json", default="artifacts/ecs_stage1d_disentangle.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    runs = [run_seed(s, args) for s in seeds]
    agg = aggregate(runs)
    protocol = {"held_out": True,
                "two_teachers": ["shortest_path", "generative_program"],
                "oracle_library": "exact hidden subroutines (sufficiency ceiling)",
                "compute_metric": "policy forward-passes (best-first), identical per arm",
                "n_seeds": len(seeds), "valid": len(seeds) >= 3}
    payload = {"config": vars(args), "protocol_evidence": protocol, "aggregate": agg, "seed_runs": runs}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    v = agg["verdict"]
    print("=== ECS Stage 1b — disentangle shortcut vs program (identifiability x sufficiency) ===")
    print(f"seeds={seeds}  protocol_valid={protocol['valid']}")
    print(f"\n  (A) IDENTIFIABILITY — promoted-macro <-> hidden-subroutine match-rate:")
    print(f"      shortest-path teacher  : {agg['match_rate_shortest']:.3f}")
    print(f"      generative teacher     : {agg['match_rate_generative']:.3f}  "
          f"(≈{agg['match_count_generative']:.1f}/{agg['match_rate_generative'] and 4} true subs)")
    print(f"      => A_identifiable = {v['A_identifiable_under_generative_supervision']}")
    print(f"\n  (B) SUFFICIENCY — oracle library (true subs) vs dense, forward-passes:")
    print(f"      held-out : oracle {agg['held_oracle']:.2f}  grown {agg['held_grown']:.2f}  dense {agg['held_dense']:.2f}")
    print(f"      Axis-C   : oracle {agg['axisC_oracle']:.2f}  grown {agg['axisC_grown']:.2f}  dense {agg['axisC_dense']:.2f}")
    print(f"      oracle<dense: held={v['B_oracle_beats_dense_held']} ({v['B_held_seedfrac']})  "
          f"axisC={v['B_oracle_beats_dense_axisC']} ({v['B_axisC_seedfrac']})")
    print(f"      => B_sufficient = {v['B_sufficient']}  (matched_solve={v['matched_solve_rate']})")
    print(f"\n  DECISION: {v['decision']}")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
