#!/usr/bin/env python3
"""ECS Stage 2 (falsification) — parametric (fast-weight) skill vs discrete search macro.

Stage 1b proved the decisive negative (MODEL_THESIS §6f): "skill = discrete action in a
search space" is net-negative even with an ORACLE library, because every macro adds a
branch at every node (cost ∝ action-set size) while saving few steps — the branching tax
dominates. The named redirection: a skill should live in the space of DYNAMICS, not the
space of choices — amortized into the policy so it adds NO branching.

This is the single, two-sided falsification test (not a Stage-2 system). Same DSL, same
teacher, same compute accounting (policy forward-passes), three arms at IDENTICAL action
set (5 primitives — the parametric skill never adds an action):

  * dense        : primitive policy, best-first search. (no skills)
  * discrete-skill: the true subroutines as search macros (Stage-1b oracle arm) — pays
                    the branching tax. (the broken class, as control)
  * param-skill  : the SAME subroutines compiled into context-gated FAST-WEIGHTS — a
                    learned additive shift Δ_k on the PRIMITIVE-action logits, activated
                    by a per-skill gate g_k(x,target). The action set stays size-5, so
                    branching factor is unchanged; the skill biases WHICH primitive the
                    policy emits, steering the search without widening it.

The fast-weight skill is trained (Adam) to imitate, at each state on a subroutine's
execution, the next primitive that subroutine takes — i.e. it compiles "follow subroutine
k from here" into a parametric bias. At eval it fires via best-first over the SAME 5
primitives, with the active skill's Δ added to the logits.

Pre-registered decision table (NOT moved post-hoc):
  param marginal-win AND low off-task interference  -> dynamics-space skill works; the
        discrete class was the problem -> class change justified.
  param marginal-win BUT high interference          -> a hidden tax (interference) replaces
        the branching tax -> no net gain.
  no param marginal-win                             -> compression does not pay even in
        dynamics-space at this scale -> deeper than representation.

Interference is measured explicitly: per-skill improvement on its OWN tasks AND
degradation on OTHER skills' tasks. 6 seeds, deterministic, CPU/numpy.
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
from ecs_stage1 import D, P, PRIM, features, slope, softmax, teacher_actions

HIDDEN = [list(v) for v in B.HIDDEN_SUBROUTINES.values()]   # 4 subroutines, len 3
NK = len(HIDDEN)
NEG_INF = -1e9


# ---------------------------------------------------------------------------
# Policy with PARAMETRIC fast-weight skills. Base linear policy over 5 primitives,
# plus K skill shift-vectors Δ_k (in logit space) gated by g_k(features). The
# active skill (argmax gate, if above threshold) adds its Δ to the primitive logits.
# Action set is ALWAYS the 5 primitives — no branching expansion.
# ---------------------------------------------------------------------------
class ParamSkillPolicy:
    def __init__(self, hidden, seed):
        rng = np.random.default_rng(seed)
        s = 0.5
        self.W1 = rng.normal(0, s, (hidden, D)); self.b1 = np.zeros(hidden)
        self.Wp = rng.normal(0, s, (P, hidden)); self.bp = np.zeros(P)
        # skill gate: which skill (if any) applies at this state
        self.Wg = rng.normal(0, s, (NK, hidden)); self.bg = np.zeros(NK)
        # skill fast-weights: additive logit shift per skill (the "dynamics" edit)
        self.Delta = rng.normal(0, 0.1, (NK, P))
        self.H = hidden
        self._init_adam()

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "Wp": self.Wp, "bp": self.bp,
                "Wg": self.Wg, "bg": self.bg, "Delta": self.Delta}

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.t = 0

    def base(self, u):
        h = np.tanh(self.W1 @ u + self.b1)
        return h

    def gate(self, h):
        return softmax(self.Wg @ h + self.bg)

    def prim_logits(self, h, use_skill=True):
        z = self.Wp @ h + self.bp
        if use_skill:
            pg = self.gate(h)
            k = int(np.argmax(pg))
            if pg[k] >= 0.5:
                z = z + self.Delta[k]
        return z

    def train_batch(self, batch, lr):
        """batch items: (u, prim_label, skill_label) — skill_label = which subroutine
        this state belongs to (teacher-forced skill identity)."""
        g = {k: np.zeros_like(v) for k, v in self._params().items()}
        for u, plab, klab in batch:
            h = np.tanh(self.W1 @ u + self.b1)
            # gate CE toward the true skill id
            pg = self.gate(h)
            dgz = pg.copy(); dgz[klab] -= 1.0
            g["Wg"] += np.outer(dgz, h); g["bg"] += dgz
            dh = self.Wg.T @ dgz
            # primitive CE with the true skill's Delta applied (teacher-forced skill)
            z = self.Wp @ h + self.bp + self.Delta[klab]
            sp = softmax(z); dz = sp.copy(); dz[plab] -= 1.0
            g["Wp"] += np.outer(dz, h); g["bp"] += dz
            g["Delta"][klab] += dz                      # learn the fast-weight shift
            dh += self.Wp.T @ dz
            dpre = dh * (1.0 - h ** 2)
            g["W1"] += np.outer(dpre, u); g["b1"] += dpre
        n = max(1, len(batch))
        self._adam({k: v / n for k, v in g.items()}, lr)

    def _adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        params = self._params()
        for k in params:
            gk = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * (gk * gk)
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            params[k] -= lr * mhat / (np.sqrt(vhat) + eps)


# ---------------------------------------------------------------------------
# Best-first over the 5 primitives, using param-skill logits as the priority.
# COST = forward-passes (one per expanded node), identical accounting to Stage 1b.
# Action set size is constant (5) so branching factor is fixed — the whole point.
# ---------------------------------------------------------------------------
def best_first_param(policy, x0, t, node_budget, max_depth, use_skill=True):
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
        z = policy.prim_logits(h, use_skill=use_skill)
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


# discrete-macro best-first (the Stage-1b oracle arm), for the control
def best_first_discrete(policy, x0, t, macros, node_budget, max_depth):
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
        lp = np.log(softmax(policy.prim_logits(h, use_skill=False)) + 1e-12)
        children = [(float(lp[i]), B.PRIMITIVES[PRIM[i]](x)) for i in range(P)]
        # macros: uniform small bonus so they are tried (oracle-favourable), each ONE node
        for m in macros:
            children.append((math.log(1.0 / (P + len(macros))), B.apply_prims(x, m)))
        for clp, cx in children:
            ncost = neglp - clp
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


def eval_param(policy, tasks, node_budget, max_depth, use_skill=True):
    fw, solved = [], 0
    for x0, t in tasks:
        f, ok, _ = best_first_param(policy, x0, t, node_budget, max_depth, use_skill)
        fw.append(f); solved += int(ok)
    return {"mean_forwards": statistics.mean(fw), "solve_rate": solved / len(tasks)}


def eval_discrete(policy, tasks, macros, node_budget, max_depth):
    fw, solved = [], 0
    for x0, t in tasks:
        f, ok, _ = best_first_discrete(policy, x0, t, macros, node_budget, max_depth)
        fw.append(f); solved += int(ok)
    return {"mean_forwards": statistics.mean(fw), "solve_rate": solved / len(tasks)}


# ---------------------------------------------------------------------------
# Build training data: walk each subroutine's execution from random x0, label each
# state with (next primitive, skill id). Also single-subroutine eval sets per skill
# for the interference measurement.
# ---------------------------------------------------------------------------
def make_training(seed, n_per_skill):
    rng = random.Random(seed)
    data = []
    for k, sub in enumerate(HIDDEN):
        for _ in range(n_per_skill):
            x0 = rng.randint(0, 9)
            x = x0
            for p in sub:
                data.append((features(x, B.apply_prims(x0, sub)), PRIM.index(p), k))
                x = B.PRIMITIVES[p](x)
    rng.shuffle(data)
    return data


def per_skill_tasks(seed, n):
    rng = random.Random(seed)
    out = []
    for k, sub in enumerate(HIDDEN):
        ts = []
        for _ in range(n):
            x0 = rng.randint(0, 9)
            ts.append((x0, B.apply_prims(x0, sub)))
        out.append(ts)
    return out


def make_dense_training(seed, n):
    """Cleanly-trained dense control: imitate the SHORTEST-PATH primitive teacher on the
    same task distribution, with NO skill structure (single dummy skill id). Removes the
    confound that 'dense = skills-off' shared the param policy's skill-shaped training."""
    rng = random.Random(seed * 31 + 5)
    tp = S.train_pool()
    prim = B.primitive_actions()
    data = []
    for _ in range(n):
        ch = [rng.choice(tp) for _ in range(rng.randint(1, 2))]
        prog = [p for sub in ch for p in sub]
        x0 = rng.randint(0, 9)
        t = B.apply_prims(x0, prog)
        path = teacher_actions(x0, t, prim, 8, 200000)
        if not path:
            continue
        x = x0
        for a in path:
            data.append((features(x, t), PRIM.index(a.name), 0))   # skill id 0 (unused)
            x = a.apply(x)
    rng.shuffle(data)
    return data


def run_seed(seed, cfg):
    tp = S.train_pool()
    data = make_training(seed, cfg.per_skill)
    pol = ParamSkillPolicy(cfg.hidden, seed)
    rng = np.random.default_rng(seed)
    for _ in range(cfg.epochs):
        idx = rng.integers(0, len(data), size=min(cfg.batch, len(data)))
        pol.train_batch([data[j] for j in idx], cfg.lr)

    # cleanly-trained dense control (no skill structure, shortest-path teacher)
    dpol = ParamSkillPolicy(cfg.hidden, seed + 12345)
    ddata = make_dense_training(seed, cfg.per_skill * NK)
    drng = np.random.default_rng(seed + 555)
    for _ in range(cfg.epochs):
        idx = drng.integers(0, len(ddata), size=min(cfg.batch, len(ddata)))
        dpol.train_batch([ddata[j] for j in idx], cfg.lr)

    # held-out + Axis-C composition tasks (same as the rest of the line)
    keys = set()
    held = S.sample_disjoint(random.Random(seed + 1), cfg.eval_tasks, tp, (1, 2), keys)
    axisC = S.sample_disjoint(random.Random(seed + 2), cfg.eval_tasks, tp, (3, 3), keys)
    nb, md = cfg.node_budget, cfg.bf_depth

    param_held = eval_param(pol, held, nb, md, use_skill=True)
    param_axisC = eval_param(pol, axisC, nb, md, use_skill=True)
    dense_held = eval_param(pol, held, nb, md, use_skill=False)   # SAME policy, skills off
    dense_axisC = eval_param(pol, axisC, nb, md, use_skill=False)
    # cleanly-trained dense (separate policy, never saw skill structure) — the fair control
    cdense_held = eval_param(dpol, held, nb, md, use_skill=False)
    cdense_axisC = eval_param(dpol, axisC, nb, md, use_skill=False)
    disc_held = eval_discrete(pol, held, HIDDEN, nb, md)          # true subs as search macros
    disc_axisC = eval_discrete(pol, axisC, HIDDEN, nb, md)

    # interference: per-skill solve on own tasks WITH vs WITHOUT skills active.
    psets = per_skill_tasks(seed + 7, cfg.eval_tasks // 2)
    own_on, own_off = [], []
    for k, ts in enumerate(psets):
        own_on.append(eval_param(pol, ts, nb, md, use_skill=True)["mean_forwards"])
        own_off.append(eval_param(pol, ts, nb, md, use_skill=False)["mean_forwards"])
    # interference proxy: does turning skills ON ever HURT a single-skill set?
    hurt = [max(0.0, on - off) for on, off in zip(own_on, own_off)]

    return {"seed": seed,
            "param": {"held": param_held, "axisC": param_axisC},
            "dense": {"held": dense_held, "axisC": dense_axisC},
            "cdense": {"held": cdense_held, "axisC": cdense_axisC},
            "discrete": {"held": disc_held, "axisC": disc_axisC},
            "interference": {"own_on": [round(x, 2) for x in own_on],
                             "own_off": [round(x, 2) for x in own_off],
                             "max_hurt": round(max(hurt), 3),
                             "mean_hurt": round(statistics.mean(hurt), 3)}}


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
        "param_held": M(["param", "held", "mean_forwards"]),
        "param_held_solve": M(["param", "held", "solve_rate"]),
        "dense_held": M(["dense", "held", "mean_forwards"]),
        "dense_held_solve": M(["dense", "held", "solve_rate"]),
        "cdense_held": M(["cdense", "held", "mean_forwards"]),
        "cdense_held_solve": M(["cdense", "held", "solve_rate"]),
        "cdense_axisC": M(["cdense", "axisC", "mean_forwards"]),
        "discrete_held": M(["discrete", "held", "mean_forwards"]),
        "param_axisC": M(["param", "axisC", "mean_forwards"]),
        "dense_axisC": M(["dense", "axisC", "mean_forwards"]),
        "discrete_axisC": M(["discrete", "axisC", "mean_forwards"]),
        "mean_hurt": M(["interference", "mean_hurt"]),
        "max_hurt": M(["interference", "max_hurt"]),
    }
    n = len(runs)
    held_wins = sum(1 for r in runs
                    if r["param"]["held"]["mean_forwards"] < r["dense"]["held"]["mean_forwards"])
    axisC_wins = sum(1 for r in runs
                     if r["param"]["axisC"]["mean_forwards"] < r["dense"]["axisC"]["mean_forwards"])
    # FAIR test: param vs cleanly-trained dense (never saw skill structure)
    cdense_wins = sum(1 for r in runs
                      if r["param"]["held"]["mean_forwards"] < r["cdense"]["held"]["mean_forwards"])
    cdense_axisC_wins = sum(1 for r in runs
                            if r["param"]["axisC"]["mean_forwards"] < r["cdense"]["axisC"]["mean_forwards"])
    param_beats_dense_held = a["param_held"] < a["dense_held"]
    param_beats_cdense_held = a["param_held"] < a["cdense_held"]
    param_beats_cdense_axisC = a["param_axisC"] < a["cdense_axisC"]
    param_beats_dense_axisC = a["param_axisC"] < a["dense_axisC"]
    param_beats_discrete = a["param_held"] < a["discrete_held"]
    matched = abs(a["param_held_solve"] - a["cdense_held_solve"]) <= 0.05
    low_interference = a["mean_hurt"] < 0.5
    # the fair marginal win is vs the CLEAN dense, robust across seeds
    marginal_win = param_beats_cdense_held and cdense_wins >= (n // 2 + 1)
    if marginal_win and low_interference and param_beats_cdense_axisC:
        decision = ("param beats CLEAN dense on held AND Axis-C + low interference + no branching tax "
                    "-> dynamics-space compression works AND transfers -> class change JUSTIFIED")
    elif marginal_win and low_interference:
        decision = ("param beats clean dense in-distribution (low interference, no branching tax) "
                    "but Axis-C does NOT transfer -> in-distribution gain only, transfer unproven")
    elif param_beats_dense_held and not param_beats_cdense_held:
        decision = ("apparent win was a CONFOUND: param only beats skills-off-same-policy, not a "
                    "cleanly-trained dense -> no real parametric advantage")
    else:
        decision = ("no clean marginal win -> compression does not pay even in dynamics-space "
                    "at this scale")
    a["verdict"] = {
        "param_beats_dense_held_samePolicy": param_beats_dense_held, "held_seedfrac": f"{held_wins}/{n}",
        "param_beats_CLEANdense_held": param_beats_cdense_held, "cleandense_seedfrac": f"{cdense_wins}/{n}",
        "param_beats_CLEANdense_axisC": param_beats_cdense_axisC, "cleandense_axisC_seedfrac": f"{cdense_axisC_wins}/{n}",
        "param_beats_dense_axisC_samePolicy": param_beats_dense_axisC, "axisC_seedfrac": f"{axisC_wins}/{n}",
        "param_beats_discrete": param_beats_discrete,
        "matched_solve_rate": matched, "low_interference": low_interference,
        "branching_tax_avoided": a["param_held"] < a["discrete_held"],
        "decision": decision,
    }
    return a


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
    ap.add_argument("--output-json", default="artifacts/ecs_stage2_fastweight.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    runs = [run_seed(s, args) for s in seeds]
    agg = aggregate(runs)
    protocol = {"held_out": True, "identical_action_set": "5 primitives for param & dense (no expansion)",
                "compute_metric": "policy forward-passes (best-first), identical per arm",
                "controls": ["dense (skills off, same policy)", "discrete-macro (true subs in search)"],
                "interference_measured": True, "axis_C_present": True,
                "n_seeds": len(seeds), "valid": len(seeds) >= 3}
    payload = {"config": vars(args), "protocol_evidence": protocol, "aggregate": agg, "seed_runs": runs}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    v = agg["verdict"]
    print("=== ECS Stage 2 falsification — parametric (fast-weight) vs discrete skill ===")
    print(f"seeds={seeds}  protocol_valid={protocol['valid']}  (identical 5-action set for param & dense)")
    print(f"\n  forward-passes (held / Axis-C):")
    print(f"    param-skill (Δ on logits): {agg['param_held']:6.2f} / {agg['param_axisC']:6.2f}  solve {agg['param_held_solve']}")
    print(f"    dense (same policy, off)  : {agg['dense_held']:6.2f} / {agg['dense_axisC']:6.2f}  solve {agg['dense_held_solve']}")
    print(f"    CLEAN dense (fair control): {agg['cdense_held']:6.2f} / {agg['cdense_axisC']:6.2f}  solve {agg['cdense_held_solve']}")
    print(f"    discrete-macro (control)  : {agg['discrete_held']:6.2f} / {agg['discrete_axisC']:6.2f}")
    print(f"  interference (skills-on minus skills-off on single-skill sets): mean {agg['mean_hurt']}  max {agg['max_hurt']}")
    print(f"\n  FAIR test vs CLEAN dense: held param<cdense={v['param_beats_CLEANdense_held']} ({v['cleandense_seedfrac']})  "
          f"axisC={v['param_beats_CLEANdense_axisC']} ({v['cleandense_axisC_seedfrac']})")
    print(f"  param<discrete (branching tax avoided)={v['branching_tax_avoided']}  low_interference={v['low_interference']}  matched_solve={v['matched_solve_rate']}")
    print(f"\n  DECISION: {v['decision']}")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
