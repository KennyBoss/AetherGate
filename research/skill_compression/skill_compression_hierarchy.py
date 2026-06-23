#!/usr/bin/env python3
"""Phase 2 — hierarchical skill compression: does the crossover P* shift with
hierarchy depth D? Turns P* into an order parameter P*(D).

Recursive skills without blowing the node budget
------------------------------------------------
A hidden skill hierarchy is built bottom-up, each level a composition of the one
below (a skill that calls skills):
    L1[i] = primitive sequence (len 3)          e.g. [c,a,c]
    L2[i] = concat of two L1                     -> len 6
    L3[i] = concat of two L2                     -> len 12
A macro executes via its primitive expansion (the recursive call).

The budget is tamed by comparing TWO ADJACENT abstraction levels, never
primitives-vs-deep-search:
    engage : full library incl. top-level LD macros -> a depth-D reuse task
             solves in ~1 step.
    abstain: library WITHOUT the top level (up to L(D-1)) -> same task in ~2
             steps (more nodes, bigger branching), but still bounded.
So search stays ~1-2 steps at every depth; only the action-set SIZE grows
(5 + members*D). This is exactly the combinatorial explosion being *tamed* by
hierarchy — each level is searchable only because the level below exists.

OOD at depth D: a same-length but INVALID combination of L(D-1) blocks — the
top-level skill does not fit, so it must be solved one abstraction level down.

Two arms (same library, both abstraction-aware)
-----------------------------------------------
    UTIL+APP : always probe top-level skills, fall back to the lower level.
    SSM-GATE : a learned recurrent gate decides engage-top vs go-low directly.
With exec-penalty P = cost of *attempting* a top-level skill, the crossover is
    P*(D) = (base_ssm - base_app) / (1 - fire_rate)
measured per depth. Hypotheses: B) P*(D) decreases (deeper skills -> internal
model pays off earlier); C) a sharp nonlinear break (a second phase transition).

Methodological note (honest): the depth-D library is the matched skill set (the
promotion target), PROVIDED identically to both arms. This isolates the P*(D)
relationship from promotion noise; both arms see the same library, so the gate-
vs-probe comparison is unbiased. Bottom-up promotion of the same hierarchy is a
separate experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np

import skill_compression_bench as B
import skill_compression_ssm_gate as G


def crossover(base_app, base_ssm, fire_rate):
    """P* = (base_ssm - base_app) / (1 - fire_rate). None if gate never abstains."""
    if fire_rate >= 1.0:
        return None
    return (base_ssm - base_app) / (1.0 - fire_rate)


def ci95(xs):
    n = len(xs)
    mean = statistics.mean(xs)
    std = statistics.stdev(xs) if n > 1 else 0.0
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(n - 1, 1.96)
    half = t * std / math.sqrt(n) if n > 1 else float("nan")
    return mean, std, half, t


# ---------------------------------------------------------------------------
# Build the hidden recursive hierarchy (fixed, independent of experiment seeds)
# ---------------------------------------------------------------------------
def build_hierarchy(max_depth: int, members: int, hseed: int = 12345):
    hier_rng = random.Random(hseed)
    base = [["c", "a", "c"], ["b", "b", "d"], ["e", "d", "d"], ["a", "c", "b"],
            ["c", "c", "a"], ["b", "a", "e"], ["d", "e", "b"], ["a", "a", "d"]]
    hier: dict[int, list[list[str]]] = {1: [list(s) for s in base[:members]]}
    for k in range(2, max_depth + 1):
        seqs, seen = [], set()
        while len(seqs) < members:
            a = hier_rng.choice(hier[k - 1])
            b = hier_rng.choice(hier[k - 1])
            seq = a + b
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                seqs.append(seq)
        hier[k] = seqs
    return hier


def macros_for_level(hier, level: int) -> list[B.Action]:
    return [B.Action(f"L{level}_{i}", list(seq)) for i, seq in enumerate(hier[level])]


def library(hier, up_to: int) -> list[B.Action]:
    acts = B.primitive_actions()
    for k in range(1, up_to + 1):
        acts += macros_for_level(hier, k)
    return acts


# ---------------------------------------------------------------------------
# Task generation per depth
# ---------------------------------------------------------------------------
def make_reuse(rng, D, hier, xmax):
    seq = rng.choice(hier[D])
    x0 = rng.randint(0, xmax)
    return x0, B.apply_prims(x0, seq)


def make_ood(rng, D, hier, xmax):
    if D == 1:
        valid = {tuple(s) for s in hier[1]}
        while True:
            seq = [rng.choice(B.PRIM_NAMES) for _ in range(3)]
            if tuple(seq) not in valid:
                break
    else:
        valid = {tuple(s) for s in hier[D]}
        while True:
            seq = rng.choice(hier[D - 1]) + rng.choice(hier[D - 1])
            if tuple(seq) not in valid:
                break
    x0 = rng.randint(0, xmax)
    return x0, B.apply_prims(x0, seq)


def sample_disjoint(maker, rng, n, taken, *a):
    out = []
    tries = 0
    while len(out) < n and tries < n * 300:
        tries += 1
        t = maker(rng, *a)
        if t not in taken:
            out.append(t)
            taken.add(t)
    return out


# ---------------------------------------------------------------------------
# Per (depth, seed): provide library, train gate, measure base costs + fire_rate
# ---------------------------------------------------------------------------
SCALE = 8.0


def top_features(x0, target, top_macros):
    d0 = abs(x0 - target)
    rows = []
    for m in top_macros:
        dm = abs(m.apply(x0) - target)
        rows.append([np.tanh((d0 - dm) / SCALE), 1.0 if m.apply(x0) == target else 0.0])
    if not rows:
        rows = [[0.0, 0.0]]
    return np.asarray(rows, dtype=np.float64)


def run_depth_seed(D, seed, cfg):
    rng = random.Random(seed * 1000 + D)
    hier = cfg.hier
    full_lib = library(hier, D)
    sub_lib = library(hier, D - 1)
    top_macros = macros_for_level(hier, D)
    md, nb, pb = cfg.max_depth, cfg.node_budget, cfg.probe_budget
    fcost = max(1, len(top_macros))

    # gate-training experience: agent's own mix of familiar + novel at depth D
    taken: set = set()
    g_reuse = sample_disjoint(make_reuse, rng, cfg.gate_train // 2, taken, D, hier, cfg.xmax)
    g_ood = sample_disjoint(make_ood, rng, cfg.gate_train // 2, taken, D, hier, cfg.xmax)
    gate_tasks = g_reuse + g_ood
    rng.shuffle(gate_tasks)

    Xs, ys = [], []
    for x0, target in gate_tasks:
        # label: is ENGAGING the top skill cheaper than going one level down?
        c_eng, _ = B.solve_with_applicability(x0, target, full_lib, sub_lib, md, nb, pb)
        _s, c_abs, _ = B.solve(x0, target, sub_lib, md, nb)
        ys.append(1 if c_eng < c_abs else 0)
        Xs.append(top_features(x0, target, top_macros))

    gate = G.SSMGate(d_in=2, hidden=cfg.hidden, seed=seed)
    for _ in range(cfg.epochs):
        grads = {"A": np.zeros_like(gate.A), "B": np.zeros_like(gate.B),
                 "w": np.zeros_like(gate.w), "b": 0.0}
        for X, y in zip(Xs, ys):
            g = gate.backward(gate.forward(X)[1], float(y))
            for k in grads:
                grads[k] = grads[k] + g[k]
        for k in grads:
            grads[k] = grads[k] / len(Xs)
        gate.step(grads, lr=cfg.lr)
    train_acc = float(np.mean([(gate.predict(X) >= 0.5) == bool(y) for X, y in zip(Xs, ys)]))

    # held-out eval, disjoint from gate-training
    eval_reuse = sample_disjoint(make_reuse, rng, cfg.eval_tasks, taken, D, hier, cfg.xmax)
    eval_ood = sample_disjoint(make_ood, rng, cfg.eval_tasks, taken, D, hier, cfg.xmax)

    def base_app(tasks):  # always engage top skill (probe+fallback)
        return B.evaluate_applicability(tasks, full_lib, sub_lib, md, nb, pb)["mean_nodes"]

    def base_ssm(tasks):  # gate decides engage-top vs go-low
        nodes, fired = [], 0
        for x0, target in tasks:
            if gate.predict(top_features(x0, target, top_macros)) >= 0.5:
                fired += 1
                n, _ = B.solve_with_applicability(x0, target, full_lib, sub_lib, md, nb, pb)
            else:
                _s, n, _ = B.solve(x0, target, sub_lib, md, nb)
            nodes.append(n + fcost)
        return statistics.mean(nodes), fired / len(tasks)

    out = {"seed": seed, "depth": D, "n_top_macros": len(top_macros),
           "gate_train_acc": round(train_acc, 3)}
    for split, tasks in (("reuse", eval_reuse), ("ood", eval_ood)):
        ba = base_app(tasks)
        bs, fr = base_ssm(tasks)
        out[split] = {"base_app": round(ba, 2), "base_ssm": round(bs, 2),
                      "fire_rate": round(fr, 3),
                      "P_star": crossover(ba, bs, fr)}
    # mixed 50/50
    ba = (out["reuse"]["base_app"] + out["ood"]["base_app"]) / 2
    bs = (out["reuse"]["base_ssm"] + out["ood"]["base_ssm"]) / 2
    fr = (out["reuse"]["fire_rate"] + out["ood"]["fire_rate"]) / 2
    out["mixed"] = {"base_app": round(ba, 2), "base_ssm": round(bs, 2),
                    "fire_rate": round(fr, 3), "P_star": crossover(ba, bs, fr)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--xmax", type=int, default=31)
    ap.add_argument("--gate-train", type=int, default=120)
    ap.add_argument("--eval-tasks", type=int, default=60)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--node-budget", type=int, default=500_000)
    ap.add_argument("--probe-budget", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--output-json", default="artifacts/hierarchy_phase_diagram.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    depths = [int(d) for d in args.depths.split(",")]
    args.hier = build_hierarchy(max(depths), args.members)

    per = {D: [run_depth_seed(D, s, args) for s in seeds] for D in depths}

    summary = {}
    for D in depths:
        summary[D] = {}
        for split in ("reuse", "ood", "mixed"):
            ps = [r[split]["P_star"] for r in per[D] if r[split]["P_star"] is not None]
            frs = [r[split]["fire_rate"] for r in per[D]]
            if ps:
                m, sd, half, t = ci95(ps)
            else:
                m = sd = half = float("nan"); t = 0
            summary[D][split] = {
                "P_star_mean": round(m, 1), "P_star_std": round(sd, 1),
                "ci95_low": round(m - half, 1), "ci95_high": round(m + half, 1),
                "fire_rate_mean": round(statistics.mean(frs), 3),
                "P_star_per_seed": [round(p, 1) for p in ps],
            }
        summary[D]["gate_train_acc"] = round(
            statistics.mean(r["gate_train_acc"] for r in per[D]), 3)

    payload = {"config": {k: v for k, v in vars(args).items() if k != "hier"},
               "cost_law": "P*(D) = (base_ssm - base_app) / (1 - fire_rate)",
               "summary": summary, "per_depth_seed": per,
               "protocol_evidence": {"n_seeds": len(seeds), "held_out": True,
                                     "library_provided_matched_to_both_arms": True,
                                     "valid": len(seeds) >= 3}}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Phase 2 — P*(D): does the crossover shift with hierarchy depth? ===")
    print(f"seeds={seeds}  depths={depths}")
    print(f"  {'D':>2s} {'stream':7s} {'P*_mean':>8s} {'95% CI':>16s} {'fire':>6s} {'acc':>5s}")
    for D in depths:
        for split in ("reuse", "ood", "mixed"):
            s = summary[D][split]
            ci = f"[{s['ci95_low']}, {s['ci95_high']}]"
            acc = summary[D]["gate_train_acc"] if split == "reuse" else ""
            print(f"  {D:>2d} {split:7s} {s['P_star_mean']:8.1f} {ci:>16s} "
                  f"{s['fire_rate_mean']:6.3f} {acc:>5}")
    print("  MIXED-stream P*(D) trajectory:")
    traj = [(D, summary[D]["mixed"]["P_star_mean"]) for D in depths]
    print("    " + "  ->  ".join(f"D{D}:{p}" for D, p in traj))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
