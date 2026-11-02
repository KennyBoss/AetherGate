#!/usr/bin/env python3
"""ECS Stage 2A — composition diagnostic (Axis-C localizer). NOT a new architecture.

Stage 2 (MODEL_THESIS §6g) settled the SUBSTRATE: parametric fast-weight skills beat
clean-dense in-distribution (6.51 vs 8.32, 5/6 seeds), with no branching tax and
negligible interference. The ONE invariant that survived every stage is the failure of
COMPOSITIONAL transfer: Axis-C param 37.35 > clean-dense 32.30. The skills help on
familiar depth and do not compose to novel depth.

Three explanations were fully CONFOUNDED until now:
  H1  skills are not compositional        (each Delta helps only locally; chaining adds nothing)
  H2  composition exists, selector blind  (gate g_k=f(state,target) cannot see the composition
                                           context; depth-3 target is OOD for a gate trained on
                                           single-subroutine targets)
  H3  Delta_k conflict in logit-space     (a correct skill SEQUENCE actively degrades search)

A depth ladder alone (single/double/triple) shows WHERE cost blows up but cannot tell the
three apart — all three break at depth 3. The discriminating instrument is ORACLE ROUTING:
hand the policy perfect knowledge of which subroutine is active at each state (its true
Delta), bypassing the learned gate entirely. This is the ceiling for ANY selector.

  arms (identical 5-primitive action set, cost = best-first forward-passes):
    gate   : the learned param-skill (Stage 2's policy, gate-selected Delta)   = f(state,target)
    oracle : SAME Deltas, but the active skill id is forced from the plan       = perfect routing
    cdense : cleanly-trained dense, skills off (the fair no-skill baseline)
    disc   : true subroutines as discrete search macros (the §6f broken class)

  pre-registered decision table (Axis-C / depth 3; NOT moved post-hoc):
    oracle << gate  AND  oracle <= cdense   -> H2: routing blindness. Deltas DO compose, the
                                               gate cannot pick them on OOD-depth targets.
                                               FIX = composition-aware gate g_k(state,target,
                                               active_skills); no compute growth. (predicted)
    oracle ~= gate  AND  both ~> cdense      -> H1: even PERFECT routing of the true skills does
                                               not beat no-skill on novel depth -> skills not
                                               compositional. Substrate insufficient for transfer.
    oracle  > cdense (perfect routing HURTS) -> H3: chained correct Deltas conflict in logit
                                               space -> needs orthogonalized / state-scaled edits.

6 seeds, deterministic, CPU/numpy. Reuses Stage-2 policy, training, and accounting verbatim.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np

import skill_compression_bench as B
import skill_compression_stream as S
from ecs_stage1 import D, P, PRIM, features, softmax, teacher_actions
from ecs_stage2_fastweight import (
    HIDDEN, NK, NEG_INF, ParamSkillPolicy,
    best_first_discrete, make_dense_training, make_training,
)


# ---------------------------------------------------------------------------
# Composition tasks WITH their plan. Walking the plan from x0 yields, for each
# state on the canonical path, the subroutine that is executing there. That map is
# the oracle: it is exactly the routing signal the learned gate must infer but cannot.
# Held-out by construction (fresh (x0,target), never in any train batch).
# ---------------------------------------------------------------------------
def gen_comp_tasks(seed, n, n_sub):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        plan = [rng.randrange(NK) for _ in range(n_sub)]
        x0 = rng.randint(0, 9)
        x = x0
        oracle = {}                       # state value -> active subroutine id (first occurrence)
        for kid in plan:
            for p in HIDDEN[kid]:
                if x not in oracle:
                    oracle[x] = kid
                x = B.PRIMITIVES[p](x)
        out.append((x0, x, oracle))       # (x0, target, oracle routing map)
    return out


# ---------------------------------------------------------------------------
# Best-first over the 5 primitives with EXTERNALLY-SUPPLIED skill selection.
#   select(x) -> skill id or None.  gate arm: argmax gate>=0.5.  oracle arm: oracle[x].
# Cost accounting (forward-passes per expanded node) identical to Stage 1b / Stage 2.
# ---------------------------------------------------------------------------
def best_first_select(policy, x0, t, select, node_budget, max_depth):
    if x0 == t:
        return 0, True, 0
    pq = [(0.0, 0, x0, 0)]
    best_cost = {x0: 0.0}
    counter, forwards = 1, 0
    while pq and forwards < node_budget:
        neglp, _, x, depth = heapq.heappop(pq)
        if x == t:
            return forwards, True, depth
        if depth >= max_depth:
            continue
        if neglp > best_cost.get(x, NEG_INF) + 1e-9:
            continue
        forwards += 1
        h = policy.base(features(x, t))
        z = policy.Wp @ h + policy.bp
        k = select(x, h)
        if k is not None:
            z = z + policy.Delta[k]
        lp = np.log(softmax(z) + 1e-12)
        for i in range(P):
            cx = B.PRIMITIVES[PRIM[i]](x)
            ncost = neglp - float(lp[i])
            if cx in best_cost and best_cost[cx] <= ncost + 1e-9:
                continue
            best_cost[cx] = ncost
            heapq.heappush(pq, (ncost, counter, cx, depth + 1))
            counter += 1
    while pq:
        neglp, _, x, depth = heapq.heappop(pq)
        if x == t:
            return forwards, True, depth
    return forwards, False, max_depth


def gate_select(policy):
    def sel(x, h):
        pg = policy.gate(h)
        k = int(np.argmax(pg))
        return k if pg[k] >= 0.5 else None
    return sel


def oracle_select(oracle):
    def sel(x, h):
        return oracle.get(x)            # perfect routing on-path; None off-path
    return sel


def no_skill(x, h):
    return None


def eval_select(policy, tasks, select_factory, nb, md):
    fw, solved = [], 0
    for x0, t, oracle in tasks:
        sel = select_factory(oracle) if select_factory.__code__.co_argcount == 1 else select_factory
        f, ok, _ = best_first_select(policy, x0, t, sel, nb, md)
        fw.append(f); solved += int(ok)
    return {"mean_forwards": statistics.mean(fw), "solve_rate": solved / len(tasks)}


def eval_gate(policy, tasks, nb, md):
    sel = gate_select(policy)
    fw, solved = [], 0
    for x0, t, _ in tasks:
        f, ok, _ = best_first_select(policy, x0, t, sel, nb, md)
        fw.append(f); solved += int(ok)
    return {"mean_forwards": statistics.mean(fw), "solve_rate": solved / len(tasks)}


def eval_oracle(policy, tasks, nb, md):
    fw, solved = [], 0
    for x0, t, oracle in tasks:
        f, ok, _ = best_first_select(policy, x0, t, oracle_select(oracle), nb, md)
        fw.append(f); solved += int(ok)
    return {"mean_forwards": statistics.mean(fw), "solve_rate": solved / len(tasks)}


def eval_noskill(policy, tasks, nb, md):
    fw, solved = [], 0
    for x0, t, _ in tasks:
        f, ok, _ = best_first_select(policy, x0, t, no_skill, nb, md)
        fw.append(f); solved += int(ok)
    return {"mean_forwards": statistics.mean(fw), "solve_rate": solved / len(tasks)}


def eval_disc(policy, tasks, nb, md):
    fw, solved = [], 0
    for x0, t, _ in tasks:
        f, ok, _ = best_first_discrete(policy, x0, t, HIDDEN, nb, md)
        fw.append(f); solved += int(ok)
    return {"mean_forwards": statistics.mean(fw), "solve_rate": solved / len(tasks)}


DEPTHS = [1, 2, 3]


def run_seed(seed, cfg):
    # Stage-2 param-skill policy (gate + Delta), trained identically.
    pol = ParamSkillPolicy(cfg.hidden, seed)
    data = make_training(seed, cfg.per_skill)
    rng = np.random.default_rng(seed)
    for _ in range(cfg.epochs):
        idx = rng.integers(0, len(data), size=min(cfg.batch, len(data)))
        pol.train_batch([data[j] for j in idx], cfg.lr)

    # clean dense control (separate policy, shortest-path teacher, never saw skill structure).
    dpol = ParamSkillPolicy(cfg.hidden, seed + 12345)
    ddata = make_dense_training(seed, cfg.per_skill * NK)
    drng = np.random.default_rng(seed + 555)
    for _ in range(cfg.epochs):
        idx = drng.integers(0, len(ddata), size=min(cfg.batch, len(ddata)))
        dpol.train_batch([ddata[j] for j in idx], cfg.lr)

    nb, md = cfg.node_budget, cfg.bf_depth
    ladder = {}
    for d in DEPTHS:
        tasks = gen_comp_tasks(seed * 100 + d, cfg.eval_tasks, d)
        ladder[d] = {
            "gate":   eval_gate(pol, tasks, nb, md),
            "oracle": eval_oracle(pol, tasks, nb, md),
            "off":    eval_noskill(pol, tasks, nb, md),   # same policy, skills off
            "cdense": eval_noskill(dpol, tasks, nb, md),  # fair no-skill baseline
            "disc":   eval_disc(pol, tasks, nb, md),
        }
    return {"seed": seed, "ladder": ladder}


def aggregate(runs):
    out = {}
    for d in DEPTHS:
        row = {}
        for arm in ("gate", "oracle", "off", "cdense", "disc"):
            fws = [r["ladder"][d][arm]["mean_forwards"] for r in runs]
            svs = [r["ladder"][d][arm]["solve_rate"] for r in runs]
            row[arm] = round(statistics.mean(fws), 3)
            row[arm + "_solve"] = round(statistics.mean(svs), 3)
        # seed-majority comparisons at this depth
        n = len(runs)
        row["oracle_beats_gate"] = sum(
            1 for r in runs
            if r["ladder"][d]["oracle"]["mean_forwards"] < r["ladder"][d]["gate"]["mean_forwards"])
        row["oracle_beats_cdense"] = sum(
            1 for r in runs
            if r["ladder"][d]["oracle"]["mean_forwards"] < r["ladder"][d]["cdense"]["mean_forwards"])
        row["gate_beats_cdense"] = sum(
            1 for r in runs
            if r["ladder"][d]["gate"]["mean_forwards"] < r["ladder"][d]["cdense"]["mean_forwards"])
        row["n"] = n
        out[d] = row
    return out


def diagnose(agg, n):
    """Pre-registered decision table on the deepest rung (Axis-C, depth 3).

    Truth table (maj = n//2+1 seeds; 5% slack on ties):
      oracle_win_gate   = oracle_beats_gate >= maj  (perfect routing better than learned gate)
      oracle_win_cdense = oracle_beats_cdense >= maj (perfect routing beats no-skill)
      gate_win_cdense   = gate_beats_cdense >= maj   (learned gate beats no-skill)
      oracle_lose_cdense = oracle > cdense * 1.05    (even perfect routing is clearly worse)

      oracle_win_gate  oracle_win_cdense  gate_win_cdense  oracle_lose_cdense
      ─────────────────────────────────────────────────────────────────────────
      T                T                  -                -            → H2  (routing blindness:
            perfect routing >> learned gate AND oracle ≤ cdense; skills compose, gate blind)
      F                F                  F                T            → H3  (conflict:
            oracle ≈ gate AND oracle >> cdense; correct skills actively hurt)
      F                -                  F                T            → H3  (same)
      -                -                  T                -            → H0  (composition works:
            gate already beats cdense at depth 3 — Axis-C breakdown absent in this config)
      -                -                  F (tied possible) -            → H1  (not compositional:
            even perfect routing doesn't clearly beat no-skill on novel depth)
    """
    d = DEPTHS[-1]
    row = agg[d]
    maj = n // 2 + 1
    oracle, gate, cdense = row["oracle"], row["gate"], row["cdense"]
    oracle_win_gate = row["oracle_beats_gate"] >= maj
    oracle_win_cdense = row["oracle_beats_cdense"] >= maj
    gate_win_cdense = row["gate_beats_cdense"] >= maj
    oracle_lose_cdense = oracle > cdense * 1.05       # perfect routing clearly WORSE than no-skill

    if oracle_win_gate and oracle_win_cdense:
        # oracle >> gate AND oracle ≤ cdense → gate blind, skills compose.
        h = "H2"
        verdict = ("ROUTING BLINDNESS (H2): perfect routing recovers Axis-C (oracle_wins={}/{}) "
                   "while the learned gate does not. The Deltas DO compose at depth 3; the gate "
                   "cannot select them on novel-depth (OOD) targets — it was trained on depth-1 "
                   "subroutines and sees a composition-ignorant feature vector. "
                   "FIX = composition-aware gate g_k(state,target,active_skills) — no compute growth."
                   ).format(row["oracle_beats_cdense"], n)
    elif oracle_lose_cdense:
        # oracle >> cdense → perfect routing HURTS → conflict.
        h = "H3"
        verdict = ("DELTA CONFLICT (H3): even PERFECT routing of the true skills is WORSE than "
                   "no-skill on novel depth (oracle={:.2f} > cdense={:.2f}, {}/{} seeds). Chained "
                   "correct Deltas interfere in logit space. FIX = orthogonalized / state-scaled "
                   "skill edits, not a better selector.").format(
                       oracle, cdense, row["oracle_beats_cdense"], n)
    elif gate_win_cdense:
        # gate already beats cdense → composition is working.
        h = "H0"
        verdict = ("COMPOSITION TRANSFER WORKS at depth 3: gate already beats cdense ({}/{} seeds; "
                   "{:.2f} < {:.2f}). The Axis-C breakdown from Stage 2 does NOT reproduce in this "
                   "diagnostic config — the composition skill transfers. Possible explanations: "
                   "(a) the Stage-2 Axis-C task sampling produced systematically harder tasks for "
                   "cdense in a way this experiment avoids, or (b) the per-subroutine oracle-state "
                   "labeling in gen_comp_tasks provides a better (informational) routing signal than "
                   "the Stage-2 eval tasks did. A positive but needs reconciliation with Stage 2's "
                   "Axis-C param 37.35 > cdense 32.30.").format(
                       row["gate_beats_cdense"], n, gate, cdense)
    else:
        # oracle doesn't beat gate decisively, gate doesn't beat cdense, oracle not clearly worse
        # → skills are neutral-to-negative even with perfect routing → not compositional.
        h = "H1"
        verdict = ("NOT COMPOSITIONAL (H1): perfect routing does NOT outperform the learned gate "
                   "({}/{} seeds), and neither beats no-skill on novel depth (gate={:.2f}, "
                   "cdense={:.2f}, gate_wins={}/{}). The substrate's local Deltas do not chain into "
                   "a novel-depth skill even with perfect knowledge of which skill to use at each "
                   "step. Transfer needs a different skill form, not a better selector.").format(
                       row["oracle_beats_gate"], n, gate, cdense, row["gate_beats_cdense"], n)
    return {"hypothesis": h, "depth": d,
            "oracle": oracle, "gate": gate, "cdense": cdense,
            "oracle_beats_gate_seedfrac": f"{row['oracle_beats_gate']}/{n}",
            "oracle_beats_cdense_seedfrac": f"{row['oracle_beats_cdense']}/{n}",
            "gate_beats_cdense_seedfrac": f"{row['gate_beats_cdense']}/{n}",
            "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--eval-tasks", type=int, default=80)
    ap.add_argument("--per-skill", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--node-budget", type=int, default=256)
    ap.add_argument("--bf-depth", type=int, default=16)
    ap.add_argument("--output-json", default="artifacts/ecs_stage2a_composition.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    runs = [run_seed(s, args) for s in seeds]
    agg = aggregate(runs)
    diag = diagnose(agg, len(seeds))
    protocol = {"held_out": True, "identical_action_set": "5 primitives (gate/oracle/off/cdense)",
                "compute_metric": "best-first forward-passes, identical per arm",
                "oracle_routing": "active skill id forced from the task's composition plan (ceiling)",
                "depth_ladder": DEPTHS, "n_seeds": len(seeds), "valid": len(seeds) >= 3}
    payload = {"config": vars(args), "protocol_evidence": protocol,
               "aggregate": agg, "diagnosis": diag, "seed_runs": runs}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== ECS Stage 2A — composition diagnostic (Axis-C localizer) ===")
    print(f"seeds={seeds}  protocol_valid={protocol['valid']}  (oracle = perfect routing ceiling)\n")
    print(f"  {'depth':>5} | {'gate':>7} {'oracle':>7} {'off':>7} {'cdense':>7} {'disc':>7} | "
          f"{'orc<gate':>9} {'orc<cdns':>9} {'gate<cdns':>9}")
    for d in DEPTHS:
        r = agg[d]
        print(f"  {d:>5} | {r['gate']:>7.2f} {r['oracle']:>7.2f} {r['off']:>7.2f} "
              f"{r['cdense']:>7.2f} {r['disc']:>7.2f} | "
              f"{r['oracle_beats_gate']:>4}/{r['n']:<4} {r['oracle_beats_cdense']:>4}/{r['n']:<4} "
              f"{r['gate_beats_cdense']:>4}/{r['n']:<4}")
    print(f"\n  solve-rate (depth {DEPTHS[-1]}): gate {agg[DEPTHS[-1]]['gate_solve']}  "
          f"oracle {agg[DEPTHS[-1]]['oracle_solve']}  cdense {agg[DEPTHS[-1]]['cdense_solve']}")
    print(f"\n  HYPOTHESIS: {diag['hypothesis']}")
    print(f"  {diag['verdict']}")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
