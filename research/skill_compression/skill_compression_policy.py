#!/usr/bin/env python3
"""Phase 3.4 / Test A — goal-conditioned slot policy: does the agent FEEL the goal?

Phase 3.3's negative was decisive: flat schema *enumeration* collapses to a random
action set (typed ≈ typed_random; slot-explosion uniformises search). The lesson,
stated sharply: a useful representation is not an expanded action set, it is a
*goal-conditioned constraint* on which instantiation to use. So Phase 3.4 replaces
"enumerate slots, then search" with a one-shot POLICY:

    Skill = (schema, slot_prior, policy)
    policy:  features(x, target) -> a binding for each slot     (NO search, O(1))

`lookahead == implicit search`, and we already proved search-over-instantiations
loses the structure — so the policy must PREDICT the binding, not probe for it.

This script runs **Test A (goal sensitivity)** — the fastest falsification that we are
not back in enumeration, to be cleared BEFORE any search integration:

  * goal_conditioned solve-rate  : π(x,target,schema) -> binding; does apply(binding)
                                    hit the held-out target?
  * goal_BLIND baseline          : ignore (x,target); always emit the schema's modal
                                    binding. (If the policy can't beat this, the slot
                                    choice carries no goal information -> enumeration.)
  * goal_sensitivity             : with schema + x0 fixed, vary the target — does the
                                    predicted binding CHANGE? (>0 means goal-conditioned.)

PASS = goal_conditioned >> goal_blind AND goal_sensitivity > 0 on HELD-OUT goals.
The policy is a tiny MLP + per-slot softmax heads, trained by Adam (numpy, no JAX),
on the agent's own solved (x,target)->binding experience for each promoted schema.
Output is JSON with protocol_evidence (held-out goals, seed count).
"""
from __future__ import annotations

import argparse
import json
import statistics
from itertools import product
from pathlib import Path

import numpy as np

import skill_compression_bench as B
import skill_compression_stream as S
import skill_compression_typed as T

SCALE = 32.0


def features(x0: int, target: int) -> np.ndarray:
    """Generic numeric features of the (state, goal) pair — deliberately NOT a
    closed-form inverse (no mod/divmod leakage); the policy must learn the mapping."""
    dx = target - x0
    return np.array([
        1.0,
        x0 / SCALE,
        target / SCALE,
        dx / SCALE,
        target / (abs(x0) + 1.0) / SCALE,
        (x0 * x0) / (SCALE * SCALE),
        1.0 if dx >= 0 else -1.0,
    ], dtype=np.float64)


N_FEAT = 7


class SlotPolicy:
    """MLP trunk + one softmax head per slot (over that slot's op-class members).
    Predicts a full binding for a fixed schema in a single forward pass."""

    def __init__(self, schema: tuple[str, ...], hidden: int, seed: int):
        self.schema = schema
        self.members = [T.CLASS_MEMBERS[c] for c in schema]
        rng = np.random.default_rng(seed)
        s = 0.5
        self.W1 = rng.normal(0, s, size=(hidden, N_FEAT))
        self.b1 = np.zeros(hidden)
        self.Wh = [rng.normal(0, s, size=(len(m), hidden)) for m in self.members]
        self.bh = [np.zeros(len(m)) for m in self.members]
        self.H = hidden
        self._init_adam()

    def _params(self):
        d = {"W1": self.W1, "b1": self.b1}
        for i in range(len(self.members)):
            d[f"Wh{i}"] = self.Wh[i]
            d[f"bh{i}"] = self.bh[i]
        return d

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.t = 0

    def forward(self, f: np.ndarray):
        h = np.tanh(self.W1 @ f + self.b1)
        probs = []
        for i in range(len(self.members)):
            z = self.Wh[i] @ h + self.bh[i]
            z = z - z.max()
            e = np.exp(z)
            probs.append(e / e.sum())
        return h, probs

    def predict(self, f: np.ndarray) -> list[str]:
        _h, probs = self.forward(f)
        return [self.members[i][int(np.argmax(p))] for i, p in enumerate(probs)]

    def train(self, X: list[np.ndarray], Y: list[list[int]], epochs: int, lr: float):
        for _ in range(epochs):
            grads = {k: np.zeros_like(v) for k, v in self._params().items()}
            for f, labels in zip(X, Y):
                h = np.tanh(self.W1 @ f + self.b1)
                dh = np.zeros(self.H)
                for i, lab in enumerate(labels):
                    z = self.Wh[i] @ h + self.bh[i]
                    z = z - z.max()
                    e = np.exp(z)
                    p = e / e.sum()
                    dz = p.copy()
                    dz[lab] -= 1.0
                    grads[f"Wh{i}"] += np.outer(dz, h)
                    grads[f"bh{i}"] += dz
                    dh += self.Wh[i].T @ dz
                dpre = dh * (1.0 - h ** 2)
                grads["W1"] += np.outer(dpre, f)
                grads["b1"] += dpre
            n = max(1, len(X))
            self._adam({k: g / n for k, g in grads.items()}, lr)

    def _adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        params = self._params()
        for k in params:
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            params[k] -= lr * mhat / (np.sqrt(vhat) + eps)


def build_dataset(schema: tuple[str, ...], x_lo=0, x_hi=5):
    """Agent's solved experience for a schema: every (x0, binding) it could have used,
    keyed by the GOAL it reaches. Each (x0,target) key stores its valid bindings."""
    members = [T.CLASS_MEMBERS[c] for c in schema]
    key_to_bindings: dict[tuple[int, int], list[list[str]]] = {}
    for binding in product(*members):
        b = list(binding)
        for x0 in range(x_lo, x_hi + 1):
            tgt = B.apply_prims(x0, b)
            key_to_bindings.setdefault((x0, tgt), []).append(b)
    return key_to_bindings


def canonical(members, bindings: list[list[str]]) -> list[int]:
    """One canonical label (slot indices) per key: lexicographically-smallest binding."""
    best = min(bindings)
    return [members[i].index(best[i]) for i in range(len(best))]


def run_seed(seed: int, schema: tuple[str, ...], cfg) -> dict:
    members = [T.CLASS_MEMBERS[c] for c in schema]
    keys = sorted(build_dataset(schema).items())  # deterministic order
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(keys))
    n_test = max(1, int(len(keys) * cfg.test_frac))
    test_i, train_i = set(idx[:n_test].tolist()), idx[n_test:].tolist()

    Xtr, Ytr = [], []
    modal_counts = [np.zeros(len(m)) for m in members]
    for j in train_i:
        (x0, tgt), bindings = keys[j]
        lab = canonical(members, bindings)
        Xtr.append(features(x0, tgt))
        Ytr.append(lab)
        for i, li in enumerate(lab):
            modal_counts[i][li] += 1
    modal = [members[i][int(c.argmax())] for i, c in enumerate(modal_counts)]  # goal-blind

    pol = SlotPolicy(schema, cfg.hidden, seed)
    pol.train(Xtr, Ytr, cfg.epochs, cfg.lr)

    # ---- evaluate on HELD-OUT goals ----
    gc_solved = gb_solved = 0
    per_x0_preds: dict[int, set] = {}
    for j in sorted(test_i):
        (x0, tgt), bindings = keys[j]
        pred = pol.predict(features(x0, tgt))
        gc_solved += int(B.apply_prims(x0, pred) == tgt)
        gb_solved += int(B.apply_prims(x0, modal) == tgt)
        per_x0_preds.setdefault(x0, set()).add(tuple(pred))

    n = len(test_i)
    # goal sensitivity: for a fixed x0, do DIFFERENT held-out goals yield DIFFERENT
    # predicted bindings? (>1 distinct prediction => the slot choice tracks the goal)
    sens = [len(v) for v in per_x0_preds.values() if v]
    goal_sensitivity = round(statistics.mean(sens), 3) if sens else 0.0
    return {
        "seed": seed,
        "n_test_goals": n,
        "goal_conditioned_solve": round(gc_solved / n, 4),
        "goal_blind_solve": round(gb_solved / n, 4),
        "goal_sensitivity_distinct_bindings_per_x0": goal_sensitivity,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--schemas", default="MAM,AAA,MAA,AMA",
                    help="op-class schemas to test (the promoted train-subroutine shapes).")
    ap.add_argument("--output-json", default="artifacts/skill_compression_policy.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    schemas = [tuple(s) for s in args.schemas.split(",") if s]

    per_schema = {}
    for sch in schemas:
        runs = [run_seed(s, sch, args) for s in seeds]
        per_schema["".join(sch)] = {
            "seed_runs": runs,
            "goal_conditioned_solve": round(statistics.mean(r["goal_conditioned_solve"] for r in runs), 4),
            "goal_blind_solve": round(statistics.mean(r["goal_blind_solve"] for r in runs), 4),
            "goal_sensitivity": round(statistics.mean(r["goal_sensitivity_distinct_bindings_per_x0"] for r in runs), 3),
        }

    gc = statistics.mean(v["goal_conditioned_solve"] for v in per_schema.values())
    gb = statistics.mean(v["goal_blind_solve"] for v in per_schema.values())
    sens = statistics.mean(v["goal_sensitivity"] for v in per_schema.values())
    summary = {
        "mean_goal_conditioned_solve": round(gc, 4),
        "mean_goal_blind_solve": round(gb, 4),
        "goal_conditioned_minus_blind": round(gc - gb, 4),
        "mean_goal_sensitivity": round(sens, 3),
        # Test A passes if the policy beats goal-blind AND the binding tracks the goal
        "test_A_passes": (gc - gb) > 0.10 and sens > 1.0,
    }
    protocol_evidence = {
        "held_out_goals": True,
        "goal_blind_baseline_present": True,
        "no_search_one_shot_policy": True,
        "n_seeds": len(seeds),
        "valid": len(seeds) >= 3,
    }
    payload = {"config": vars(args), "protocol_evidence": protocol_evidence,
               "summary": summary, "per_schema": per_schema}
    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Skill-Compression Phase 3.4 / Test A — goal-conditioned slot policy ===")
    print(f"seeds={seeds}  protocol_valid={protocol_evidence['valid']}  (held-out goals, one-shot, no search)")
    print(f"\n  {'schema':8s} {'goal-cond':>10s} {'goal-blind':>11s} {'sensitivity':>12s}")
    for name, v in per_schema.items():
        print(f"  {name:8s} {v['goal_conditioned_solve']:10.3f} {v['goal_blind_solve']:11.3f} {v['goal_sensitivity']:12.3f}")
    print(f"\n  MEAN goal-conditioned={summary['mean_goal_conditioned_solve']}  "
          f"goal-blind={summary['mean_goal_blind_solve']}  "
          f"(Δ={summary['goal_conditioned_minus_blind']:+.3f})  sensitivity={summary['mean_goal_sensitivity']}")
    print(f"  => TEST A PASSES = {summary['test_A_passes']}  "
          f"(policy feels the goal, not enumerating)")
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
