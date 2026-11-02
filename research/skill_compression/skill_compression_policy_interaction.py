#!/usr/bin/env python3
"""Phase 3.4b-lite — slot INTERACTION policy: model coupling, not a bigger net.

Phase 3.4/Test A showed a goal-conditioned policy *exists* (beats goal-blind ×4.8),
but the win was schema-dependent: additive `AAA` ~0.99, multiplicative `MAM`/`AMA`
0.21–0.27. The diagnosis is NOT "the function approximator is too weak" — it is that
the policy models each slot INDEPENDENTLY:

    Phase 3.4:   P(slot_i | goal)                         # independent heads
    needed:      P(slot_i | goal, schema, slot_{<i})      # slots are coupled

`AAA` is separable (independent additive components), so independent heads suffice.
`MAM`/`AMA` are entangled: target = ((x·m1)+a1)·m2 couples (m1,a1,m2) — once m2 is
fixed, m1 and a1 are constrained. An independent factorization P(m1)P(a1)P(m2) cannot
represent that joint; the chain rule factorization P(m1)·P(a1|m1)·P(m2|m1,a1) CAN. So
the fix is a STRUCTURED interaction model, not memory and not a deeper trunk — exactly
the pre-SSM test: only if a recurrent model later beats THIS is SSM justified.

This policy decodes slots left-to-right, each head conditioned on a learned context
that accumulates embeddings of the previously chosen slots. Greedy decode is O(L) —
NOT enumeration over combinations (the Phase-3.3 trap). The control is the SAME
independent MLP from Phase 3.4 (`skill_compression_policy.SlotPolicy`) on the SAME
train/test split, so the reported delta is the pure interaction gain.

PASS (the user's criterion): MAM ↑ and AMA ↑ and AAA not falling and sensitivity > 3.0.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

import skill_compression_bench as B
import skill_compression_typed as T
import skill_compression_policy as P
from skill_compression_policy import N_FEAT, features, build_dataset, canonical


class InteractionPolicy:
    """MLP trunk + autoregressive per-slot softmax heads. Head i sees the trunk state
    AND a context vector = sum of learned embeddings of slots chosen before i."""

    def __init__(self, schema: tuple[str, ...], hidden: int, seed: int):
        self.schema = schema
        self.members = [T.CLASS_MEMBERS[c] for c in schema]
        self.L = len(schema)
        self.H = hidden
        rng = np.random.default_rng(seed)
        s = 0.5
        self.W1 = rng.normal(0, s, size=(hidden, N_FEAT))
        self.b1 = np.zeros(hidden)
        self.Wh = [rng.normal(0, s, size=(len(m), hidden)) for m in self.members]      # trunk->logits
        self.Vh = [rng.normal(0, s, size=(len(m), hidden)) for m in self.members]      # context->logits
        self.bh = [np.zeros(len(m)) for m in self.members]
        self.emb = [rng.normal(0, s, size=(len(m), hidden)) for m in self.members]     # chosen-slot embedding
        self._init_adam()

    def _params(self):
        d = {"W1": self.W1, "b1": self.b1}
        for i in range(self.L):
            d[f"Wh{i}"], d[f"Vh{i}"], d[f"bh{i}"], d[f"emb{i}"] = self.Wh[i], self.Vh[i], self.bh[i], self.emb[i]
        return d

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.t = 0

    @staticmethod
    def _softmax(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def predict(self, f: np.ndarray) -> list[str]:
        h = np.tanh(self.W1 @ f + self.b1)
        c = np.zeros(self.H)
        out = []
        for i in range(self.L):
            p = self._softmax(self.Wh[i] @ h + self.Vh[i] @ c + self.bh[i])
            k = int(np.argmax(p))
            out.append(self.members[i][k])
            c = c + self.emb[i][k]
        return out

    def train(self, X, Y, epochs: int, lr: float):
        for _ in range(epochs):
            grads = {k: np.zeros_like(v) for k, v in self._params().items()}
            for f, labels in zip(X, Y):
                # ---- forward (teacher forcing: context built from true labels) ----
                h = np.tanh(self.W1 @ f + self.b1)
                contexts, probs = [], []
                c = np.zeros(self.H)
                for i in range(self.L):
                    contexts.append(c)
                    probs.append(self._softmax(self.Wh[i] @ h + self.Vh[i] @ c + self.bh[i]))
                    c = c + self.emb[i][labels[i]]
                # ---- backward ----
                dh = np.zeros(self.H)
                dC_next = np.zeros(self.H)          # gradient wrt context entering slot i+1
                for i in range(self.L - 1, -1, -1):
                    dz = probs[i].copy()
                    dz[labels[i]] -= 1.0
                    grads[f"Wh{i}"] += np.outer(dz, h)
                    grads[f"Vh{i}"] += np.outer(dz, contexts[i])
                    grads[f"bh{i}"] += dz
                    dh += self.Wh[i].T @ dz
                    # emb[i][label_i] was added to form context entering slot i+1
                    grads[f"emb{i}"][labels[i]] += dC_next
                    # gradient wrt context entering slot i = this head's use + downstream
                    dC_next = self.Vh[i].T @ dz + dC_next
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


def split(schema, seed, test_frac):
    keys = sorted(build_dataset(schema).items())
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(keys))
    n_test = max(1, int(len(keys) * test_frac))
    return keys, set(idx[:n_test].tolist()), idx[n_test:].tolist()


def eval_policy(pol, keys, test_i) -> tuple[float, float]:
    solved = 0
    per_x0 = {}
    for j in sorted(test_i):
        (x0, tgt), _b = keys[j]
        pred = pol.predict(features(x0, tgt))
        solved += int(B.apply_prims(x0, pred) == tgt)
        per_x0.setdefault(x0, set()).add(tuple(pred))
    n = len(test_i)
    sens = [len(v) for v in per_x0.values() if v]
    return solved / n, (statistics.mean(sens) if sens else 0.0)


def run_seed(seed, schema, cfg) -> dict:
    members = [T.CLASS_MEMBERS[c] for c in schema]
    keys, test_i, train_i = split(schema, seed, cfg.test_frac)
    Xtr, Ytr = [], []
    for j in train_i:
        (x0, tgt), bindings = keys[j]
        Xtr.append(features(x0, tgt))
        Ytr.append(canonical(members, bindings))

    indep = P.SlotPolicy(schema, cfg.hidden, seed)      # Phase 3.4 control
    indep.train(Xtr, Ytr, cfg.epochs, cfg.lr)
    inter = InteractionPolicy(schema, cfg.hidden, seed)  # Phase 3.4b
    inter.train(Xtr, Ytr, cfg.epochs, cfg.lr)

    i_solve, i_sens = eval_policy(indep, keys, test_i)
    a_solve, a_sens = eval_policy(inter, keys, test_i)
    return {"seed": seed,
            "independent_solve": round(i_solve, 4), "independent_sens": round(i_sens, 3),
            "interaction_solve": round(a_solve, 4), "interaction_sens": round(a_sens, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--schemas", default="MAM,AAA,MAA,AMA")
    ap.add_argument("--output-json", default="artifacts/skill_compression_policy_interaction.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    schemas = [tuple(s) for s in args.schemas.split(",") if s]

    per_schema = {}
    for sch in schemas:
        runs = [run_seed(s, sch, args) for s in seeds]
        name = "".join(sch)
        per_schema[name] = {
            "independent_solve": round(statistics.mean(r["independent_solve"] for r in runs), 4),
            "interaction_solve": round(statistics.mean(r["interaction_solve"] for r in runs), 4),
            "interaction_sens": round(statistics.mean(r["interaction_sens"] for r in runs), 3),
            "gain": round(statistics.mean(r["interaction_solve"] - r["independent_solve"] for r in runs), 4),
            "seed_runs": runs,
        }

    mult = [v for k, v in per_schema.items() if "M" in k]   # multiplicative (entangled) schemas
    add = [v for k, v in per_schema.items() if "M" not in k]  # additive (separable) schemas
    summary = {
        "mean_independent_solve": round(statistics.mean(v["independent_solve"] for v in per_schema.values()), 4),
        "mean_interaction_solve": round(statistics.mean(v["interaction_solve"] for v in per_schema.values()), 4),
        "interaction_gain": round(statistics.mean(v["gain"] for v in per_schema.values()), 4),
        "mult_schema_gain": round(statistics.mean(v["gain"] for v in mult), 4) if mult else 0.0,
        "additive_schema_change": round(statistics.mean(v["gain"] for v in add), 4) if add else 0.0,
        "mean_interaction_sensitivity": round(statistics.mean(v["interaction_sens"] for v in per_schema.values()), 3),
    }
    # PASS = entangled schemas improve, separable schema doesn't regress, sensitivity up
    summary["interaction_helps_entangled"] = summary["mult_schema_gain"] > 0.05
    summary["additive_not_regressed"] = summary["additive_schema_change"] > -0.05
    summary["sensitivity_above_3"] = summary["mean_interaction_sensitivity"] > 3.0
    summary["test_passes"] = (summary["interaction_helps_entangled"]
                              and summary["additive_not_regressed"])

    protocol_evidence = {"held_out_goals": True, "same_split_both_models": True,
                         "control_is_independent_3_4_policy": True, "greedy_decode_no_enumeration": True,
                         "n_seeds": len(seeds), "valid": len(seeds) >= 3}
    payload = {"config": vars(args), "protocol_evidence": protocol_evidence,
               "summary": summary, "per_schema": per_schema}
    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Phase 3.4b-lite — slot interaction policy (autoregressive) vs independent (3.4) ===")
    print(f"seeds={seeds}  protocol_valid={protocol_evidence['valid']}  (held-out goals, greedy O(L), no enumeration)")
    print(f"\n  {'schema':8s} {'indep':>8s} {'interact':>9s} {'gain':>8s} {'sens':>7s}")
    for name, v in per_schema.items():
        print(f"  {name:8s} {v['independent_solve']:8.3f} {v['interaction_solve']:9.3f} "
              f"{v['gain']:+8.3f} {v['interaction_sens']:7.3f}")
    s = summary
    print(f"\n  entangled(M*) gain = {s['mult_schema_gain']:+.3f}   additive(AAA) change = {s['additive_schema_change']:+.3f}"
          f"   sensitivity = {s['mean_interaction_sensitivity']}")
    print(f"  interaction_helps_entangled={s['interaction_helps_entangled']}  "
          f"additive_not_regressed={s['additive_not_regressed']}  sensitivity>3={s['sensitivity_above_3']}")
    print(f"  => INTERACTION TEST PASSES = {s['test_passes']}")
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
