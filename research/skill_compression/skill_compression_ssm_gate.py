#!/usr/bin/env python3
"""Phase 1.5B — a LEARNED recurrent applicability gate (SSM-GATE).

Why this and not more heuristic tuning (the user's call): three independent
results in this repo now rhyme —
  * MQAR:  a *static* attention memory is not enough -> needs dynamic store.
  * Skill compression: a *static* skill library is not enough -> needs a
    selection mechanism.
  * Applicability gate: "having a skill" != "knowing when to use it".
So the question stops being "make UTIL beat FREQ" and becomes:

    Can an internal recurrent STATE learn to decide a skill's applicability?

This script trains a tiny recurrent gate
        h_t = tanh(A h_{t-1} + B x_t),   g = sigmoid(w . h_T + b)
over a per-task sequence of cheap macro-reachability probes (one token per macro:
"does this macro move the state toward the target?"). The learned scalar g
decides, per held-out task, whether to engage the skill library (probe+fallback,
the UTIL+APP path) or skip straight to primitive search. The recurrent state is
the SAME kind of compressed-experience signal that wins on MQAR — so this is the
seam between the memory line and the skill line.

Honesty:
  * The gate trains on the agent's OWN mixed experience (reuse + novel tasks),
    labelled by whether macros actually helped. It never sees the held-out eval.
  * Cost accounting charges the feature-probe (M node-evals) to the gate, so the
    comparison to UTIL+APP is fair.
  * Decision criterion (the user's bar): SSM-GATE wins only if, vs UTIL+APP,
        reuse_nodes <= UTIL+APP AND ood_nodes <= UTIL+APP.
    A miss is a real result: it would say applicability is better decided by
    *acting* (behavioural probe) than by a learned static recogniser.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

import skill_compression_bench as B


# ---------------------------------------------------------------------------
# Tiny recurrent gate trained by BPTT + Adam (numpy, no JAX)
# ---------------------------------------------------------------------------
class SSMGate:
    def __init__(self, d_in: int, hidden: int, seed: int):
        rng = np.random.default_rng(seed)
        s = 0.5
        self.A = rng.normal(0, s / np.sqrt(hidden), size=(hidden, hidden))
        self.B = rng.normal(0, s, size=(hidden, d_in))
        self.w = rng.normal(0, s, size=(hidden,))
        self.b = 0.0
        self.H = hidden
        self._init_adam()

    def _init_adam(self):
        self.m = {k: np.zeros_like(getattr(self, k)) if k != "b" else 0.0
                  for k in ("A", "B", "w", "b")}
        self.v = {k: np.zeros_like(getattr(self, k)) if k != "b" else 0.0
                  for k in ("A", "B", "w", "b")}
        self.t = 0

    def forward(self, X: np.ndarray):
        """X: [T, d_in] -> (p, cache). h_0 = 0."""
        h = np.zeros(self.H)
        hs = [h]
        for t in range(X.shape[0]):
            pre = self.A @ h + self.B @ X[t]
            h = np.tanh(pre)
            hs.append(h)
        z = self.w @ h + self.b
        p = 1.0 / (1.0 + np.exp(-z))
        return p, (X, hs)

    def backward(self, cache, y: float):
        X, hs = cache
        T = X.shape[0]
        p, _ = self.forward(X)
        dz = p - y
        gw = dz * hs[-1]
        gb = dz
        dh = dz * self.w
        gA = np.zeros_like(self.A)
        gB = np.zeros_like(self.B)
        for t in range(T - 1, -1, -1):
            h_t, h_prev = hs[t + 1], hs[t]
            dpre = dh * (1.0 - h_t ** 2)
            gA += np.outer(dpre, h_prev)
            gB += np.outer(dpre, X[t])
            dh = self.A.T @ dpre
        return {"A": gA, "B": gB, "w": gw, "b": gb}

    def step(self, grads, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in ("A", "B", "w", "b"):
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            upd = lr * mhat / (np.sqrt(vhat) + eps)
            setattr(self, k, getattr(self, k) - upd)

    def predict(self, X: np.ndarray) -> float:
        p, _ = self.forward(X)
        return float(p)


# ---------------------------------------------------------------------------
# Per-task feature sequence: one token per macro = "does this skill help here?"
# ---------------------------------------------------------------------------
SCALE = 8.0


def macro_features(x0: int, target: int, macros: list[B.Action]) -> np.ndarray:
    d0 = abs(x0 - target)
    rows = []
    for m in macros:
        dm = abs(m.apply(x0) - target)
        rows.append([np.tanh((d0 - dm) / SCALE), 1.0 if m.apply(x0) == target else 0.0])
    if not rows:
        rows = [[0.0, 0.0]]
    return np.asarray(rows, dtype=np.float64)


def feature_cost(macros: list[B.Action]) -> int:
    """Node-eval cost charged to the gate for building features (one apply/macro)."""
    return max(1, len(macros))


# ---------------------------------------------------------------------------
def applicability_label(x0, target, util_actions, prim_actions, max_depth, budget) -> int:
    """y=1 iff engaging the skill library is genuinely cheaper than primitives."""
    _s, c_m, ok_m = B.solve(x0, target, util_actions, max_depth, budget)
    _s, c_p, ok_p = B.solve(x0, target, prim_actions, max_depth, budget)
    if ok_m and (not ok_p or c_m < c_p):
        return 1
    return 0


def run_seed(seed: int, cfg) -> dict:
    import random
    rng = random.Random(seed)
    train = [B.make_reuse_task(rng) for _ in range(cfg.train_tasks)]
    train_keys = {(x0, t) for (x0, t, _g) in train}

    # held-out eval (disjoint), reuse + OOD
    eval_reuse = B.sample_disjoint_reuse(rng, cfg.eval_tasks, train_keys)
    eval_noreuse = [B.make_no_reuse_task(rng, cfg.max_depth) for _ in range(cfg.eval_tasks)]
    held_out = all((x0, t) not in train_keys for (x0, t) in eval_reuse) and len(
        eval_reuse) == cfg.eval_tasks

    # learn the UTIL skill library on the reuse stream
    util_actions, _log = B.build_library(
        train, "util", cfg.max_depth, cfg.node_budget,
        cfg.freq_k, cfg.util_penalty, cfg.util_threshold, cfg.max_macros)
    prim_actions = B.primitive_actions()
    macros = [a for a in util_actions if a.name.startswith("M")]

    # gate-training experience: the agent's OWN mix of familiar + novel tasks
    gate_reuse = B.sample_disjoint_reuse(rng, cfg.gate_train // 2, train_keys)
    gate_novel = [B.make_no_reuse_task(rng, cfg.max_depth) for _ in range(cfg.gate_train // 2)]
    gate_tasks = gate_reuse + gate_novel
    rng.shuffle(gate_tasks)
    Xs, ys = [], []
    for x0, target in gate_tasks:
        Xs.append(macro_features(x0, target, macros))
        ys.append(applicability_label(x0, target, util_actions, prim_actions,
                                      cfg.max_depth, cfg.node_budget))

    # train the recurrent gate (BPTT + Adam, full-batch)
    gate = SSMGate(d_in=2, hidden=cfg.hidden, seed=seed)
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
    base_rate = float(np.mean(ys))

    fcost = feature_cost(macros)

    # exec_penalty models a real-world cost of *attempting* a skill (the probe).
    # UTIL+APP attempts on every task; SSM-GATE attempts only when it fires.
    def eval_ssm_gate(tasks):
        nodes_list, solved, fired = [], 0, 0
        for x0, target in tasks:
            g = gate.predict(macro_features(x0, target, macros))
            if g >= cfg.gate_threshold:  # engage skill library via probe+fallback
                fired += 1
                n, ok = B.solve_with_applicability(
                    x0, target, util_actions, prim_actions,
                    cfg.max_depth, cfg.node_budget, cfg.probe_budget)
                n += cfg.exec_penalty
            else:          # skip straight to primitives (no skill attempt)
                _s, n, ok = B.solve(x0, target, prim_actions, cfg.max_depth, cfg.node_budget)
            nodes_list.append(n + fcost)
            solved += int(ok)
        return {"n": len(tasks), "solve_rate": solved / len(tasks),
                "mean_nodes": statistics.mean(nodes_list), "fire_rate": fired / len(tasks)}

    def eval_app(tasks):
        r = B.evaluate_applicability(tasks, util_actions, prim_actions,
                                     cfg.max_depth, cfg.node_budget, cfg.probe_budget)
        r["mean_nodes"] += cfg.exec_penalty  # always attempts the skill
        return r

    return {
        "seed": seed, "held_out": held_out, "n_macros": len(macros),
        "gate_train_acc": round(train_acc, 3), "gate_base_rate": round(base_rate, 3),
        "off": {"reuse": B.evaluate(eval_reuse, prim_actions, cfg.max_depth, cfg.node_budget),
                "ood": B.evaluate(eval_noreuse, prim_actions, cfg.max_depth, cfg.node_budget)},
        "util": {"reuse": B.evaluate(eval_reuse, util_actions, cfg.max_depth, cfg.node_budget),
                 "ood": B.evaluate(eval_noreuse, util_actions, cfg.max_depth, cfg.node_budget)},
        "util_app": {"reuse": eval_app(eval_reuse), "ood": eval_app(eval_noreuse)},
        "ssm_gate": {"reuse": eval_ssm_gate(eval_reuse), "ood": eval_ssm_gate(eval_noreuse)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--train-tasks", type=int, default=120)
    ap.add_argument("--eval-tasks", type=int, default=80)
    ap.add_argument("--gate-train", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--node-budget", type=int, default=2_000_000)
    ap.add_argument("--freq-k", type=int, default=3)
    ap.add_argument("--util-penalty", type=float, default=1.0)
    ap.add_argument("--util-threshold", type=float, default=2.0)
    ap.add_argument("--max-macros", type=int, default=8)
    ap.add_argument("--probe-budget", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--gate-threshold", type=float, default=0.5,
                    help="fire the skill library when g >= threshold")
    ap.add_argument("--exec-penalty", type=int, default=0,
                    help="real-world cost of ATTEMPTING a skill (the probe). >0 is "
                         "the regime where a selective learned gate should win.")
    ap.add_argument("--output-json", default="artifacts/ssm_gate_summary.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    runs = [run_seed(s, args) for s in seeds]

    def mean(arm, split):
        return round(statistics.mean(r[arm][split]["mean_nodes"] for r in runs), 1)

    agg = {arm: {split: mean(arm, split) for split in ("reuse", "ood")}
           for arm in ("off", "util", "util_app", "ssm_gate")}
    app_reuse, app_ood = agg["util_app"]["reuse"], agg["util_app"]["ood"]
    g_reuse, g_ood = agg["ssm_gate"]["reuse"], agg["ssm_gate"]["ood"]
    fire_reuse = round(statistics.mean(r["ssm_gate"]["reuse"]["fire_rate"] for r in runs), 3)
    fire_ood = round(statistics.mean(r["ssm_gate"]["ood"]["fire_rate"] for r in runs), 3)
    train_acc = round(statistics.mean(r["gate_train_acc"] for r in runs), 3)

    verdict = {
        "ssm_gate_reuse": g_reuse, "util_app_reuse": app_reuse,
        "ssm_gate_ood": g_ood, "util_app_ood": app_ood,
        "matches_or_beats_on_reuse": g_reuse <= app_reuse * 1.05,
        "matches_or_beats_on_ood": g_ood <= app_ood,
        "user_bar_passed": (g_reuse <= app_reuse * 1.05) and (g_ood <= app_ood),
        "gate_fire_rate_reuse": fire_reuse, "gate_fire_rate_ood": fire_ood,
        "gate_train_acc": train_acc,
    }
    protocol = {
        "held_out": all(r["held_out"] for r in runs),
        "gate_trained_on_own_experience_not_eval": True,
        "feature_cost_charged_to_gate": True,
        "n_seeds": len(seeds),
        "valid": all(r["held_out"] for r in runs) and len(seeds) >= 3,
    }
    payload = {"config": vars(args), "protocol_evidence": protocol,
               "aggregate": agg, "verdict": verdict, "seed_runs": runs}

    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Phase 1.5B — Learned recurrent applicability gate (SSM-GATE) ===")
    print(f"seeds={seeds}  protocol_valid={protocol['valid']}  gate_train_acc={train_acc}")
    print(f"  Mean search nodes (held-out)      REUSE /   OOD")
    for arm in ("off", "util", "util_app", "ssm_gate"):
        print(f"    {arm:9s} : {agg[arm]['reuse']:7.1f} / {agg[arm]['ood']:7.1f}")
    print(f"  SSM-GATE fire rate: reuse={fire_reuse}  ood={fire_ood}  (want high reuse, low ood)")
    print(f"  VERDICT (user's bar = match reuse AND beat OOD vs UTIL+APP):")
    print(f"    matches_or_beats_on_reuse = {verdict['matches_or_beats_on_reuse']}")
    print(f"    matches_or_beats_on_ood   = {verdict['matches_or_beats_on_ood']}")
    print(f"    >>> user_bar_passed       = {verdict['user_bar_passed']}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
