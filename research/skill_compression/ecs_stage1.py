#!/usr/bin/env python3
"""ECS Stage 1a — does a TRAINED gate implicitly reduce active inference compute?

The contract is `research/MODEL_THESIS.md`. Every prior `skill_compression` result used
a HAND-BUILT gate; Stage 1 makes the gate (and the action policy) *trained by gradient
descent over a task stream* and asks the learning-dynamics question:

    Does minimizing ordinary task loss implicitly lower marginal active-compute on
    repeated structure — beating dense / curriculum / cache / random-routing /
    frozen-memory controls, surviving a novel-composition (Axis-C) test and a
    frozen-backbone ablation, at matched solve-rate?

Design note (honest scope). The state here is the integer `x` itself (a Markov
decode), so the "backbone" is a per-step MLP encoder of `(x, target)` rather than a
cross-step recurrence — recurrence earns its keep in Stage 2 (real token streams). The
metric is **decode steps to solve** (active actions emitted), a forward-compute proxy
in this symbolic setting; Stage 2 moves to FLOPs over tokens. Teacher-forcing the action
labels partly bakes the raw drop, so the emergence claim rests on **held-out
generalization beating the controls + surviving Axis-C**, never on the drop alone.

Arms (MODEL_THESIS §3): ECS, dense, dense+curriculum, frozen-M, cache-M, random-routing;
ablation frozen-backbone. CPU/numpy, deterministic, >=6 seeds, JSON with protocol_evidence.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

import skill_compression_bench as B
import skill_compression_stream as S

PRIM = B.PRIM_NAMES                      # ['a','b','c','d','e']
P = len(PRIM)
SCALE = 32.0


def features(x: int, target: int) -> np.ndarray:
    d = target - x
    return np.array([
        1.0, x / SCALE, target / SCALE, d / SCALE,
        1.0 if d >= 0 else -1.0, np.tanh(d / SCALE),
    ], dtype=np.float64)


D = 6  # feature dim


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Teacher: shortest, lexicographically-first ACTION path under the current action
# set (primitives + macros). Mirrors B.solve's determinism (reversed(actions)).
# ---------------------------------------------------------------------------
def teacher_actions(x0, target, actions, max_depth, budget):
    total = 0
    for dl in range(max_depth + 1):
        stack = [(x0, 0, [])]
        while stack:
            x, d, path = stack.pop()
            if x == target:
                return path
            if d >= dl:
                continue
            total += 1
            if total > budget:
                return None
            for act in reversed(actions):
                stack.append((act.apply(x), d + 1, path + [act]))
    return None


# ---------------------------------------------------------------------------
# The trained policy: MLP backbone + gate / primitive / skill heads, Adam.
# ---------------------------------------------------------------------------
class ECSPolicy:
    def __init__(self, hidden, kmax, seed):
        rng = np.random.default_rng(seed)
        s = 0.5
        self.W1 = rng.normal(0, s, (hidden, D)); self.b1 = np.zeros(hidden)
        self.wg = rng.normal(0, s, hidden); self.bg = 0.0
        self.Wp = rng.normal(0, s, (P, hidden)); self.bp = np.zeros(P)
        self.Ws = rng.normal(0, s, (kmax, hidden)); self.bs = np.zeros(kmax)
        self.H = hidden; self.K = kmax
        self._init_adam()

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "wg": self.wg, "bg": self.bg,
                "Wp": self.Wp, "bp": self.bp, "Ws": self.Ws, "bs": self.bs}

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.t = 0

    def forward(self, u):
        h = np.tanh(self.W1 @ u + self.b1)
        pg = 1.0 / (1.0 + np.exp(-(self.wg @ h + self.bg)))
        return h, pg, self.Wp @ h + self.bp, self.Ws @ h + self.bs

    def train_batch(self, batch, lr, freeze_backbone=False):
        g = {k: np.zeros_like(v) for k, v in self._params().items()}
        for u, glab, kind, lab in batch:
            h, pg, zp, zs = self.forward(u)
            dzg = pg - glab
            g["wg"] += dzg * h; g["bg"] += dzg
            dh = dzg * self.wg
            if kind == "prim":
                dp = softmax(zp); dp[lab] -= 1.0
                g["Wp"] += np.outer(dp, h); g["bp"] += dp
                dh += self.Wp.T @ dp
            else:  # skill
                ds = softmax(zs); ds[lab] -= 1.0
                g["Ws"] += np.outer(ds, h); g["bs"] += ds
                dh += self.Ws.T @ ds
            dpre = dh * (1.0 - h ** 2)
            g["W1"] += np.outer(dpre, u); g["b1"] += dpre
        n = max(1, len(batch))
        self._adam({k: v / n for k, v in g.items()}, lr, freeze_backbone)

    def _adam(self, grads, lr, freeze_backbone, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        params = self._params()
        for k in params:
            if freeze_backbone and k in ("W1", "b1"):
                continue
            gk = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * (gk * gk)
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            params[k] -= lr * mhat / (np.sqrt(vhat) + eps)


# ---------------------------------------------------------------------------
# Greedy rollout (eval) — active compute = number of actions emitted to reach target.
# ---------------------------------------------------------------------------
def rollout(policy, x0, target, macros, cap, mode, rng=None, fire_rate=0.0, cache=None):
    if mode == "cache" and cache is not None and (x0, target) in cache:
        return 1, True
    x, steps = x0, 0
    while x != target and steps < cap:
        h, pg, zp, zs = policy.forward(features(x, target))
        use_skill = False
        if mode == "ecs" and macros:
            use_skill = pg >= 0.5
        elif mode == "random" and macros:
            use_skill = rng.random() < fire_rate
        # dense / cache / frozenM(empty macros): primitives only
        if use_skill:
            if mode == "random":
                k = int(rng.integers(len(macros)))
            else:
                k = int(np.argmax(zs[:len(macros)]))
            x = macros[k].apply(x)
        else:
            x = B.PRIMITIVES[PRIM[int(np.argmax(zp))]](x)
        steps += 1
    return steps, (x == target)


def eval_split(policy, tasks, macros, cap, mode, seed=0, fire_rate=0.0, cache=None):
    rng = np.random.default_rng(seed * 131 + 7)
    steps_list, solved, fires = [], 0, 0
    for x0, t in tasks:
        # count gate fire-rate on solved-path proxy: measure at start state only
        _h, pg, _zp, _zs = policy.forward(features(x0, t))
        fires += int(pg >= 0.5)
        st, ok = rollout(policy, x0, t, macros, cap, mode, rng, fire_rate, cache)
        steps_list.append(st); solved += int(ok)
    n = len(tasks)
    return {"mean_steps": statistics.mean(steps_list), "solve_rate": solved / n,
            "fire_rate": fires / n}


# ---------------------------------------------------------------------------
# Train one policy online over a stream; record A(e) on a held-out split.
# ---------------------------------------------------------------------------
def train_stream(seed, stream, hidden, kmax, lr, steps_per_task, batch_size,
                 max_depth, budget, grow, eval_every, eval_tasks, cap, freeze_after):
    rng = np.random.default_rng(seed * 1009 + 3)
    pol = ECSPolicy(hidden, kmax, seed)
    lib = S.OnlineLibrary(1.0, 2.0, kmax) if grow else None
    buf = []
    curve = []  # (exposure, mean_steps, solve_rate) on eval_tasks with current macros

    def macros_now():
        return lib.actions[P:] if lib else []  # macro Action objects (skip primitives)

    for i, (x0, t) in enumerate(stream):
        actions = (B.primitive_actions() + macros_now()) if lib else B.primitive_actions()
        path = teacher_actions(x0, t, actions, max_depth, budget)
        if path is not None:
            x = x0
            for act in path:
                u = features(x, t)
                if act.name in PRIM:                      # primitive label
                    buf.append((u, 0.0, "prim", PRIM.index(act.name)))
                else:                                     # macro label
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
            ev = eval_split(pol, eval_tasks, macros_now(), cap, "ecs" if lib else "dense", seed)
            curve.append((i + 1, round(ev["mean_steps"], 3), round(ev["solve_rate"], 3),
                          round(ev["fire_rate"], 3)))
    return pol, lib, curve


def oracle_steps(tasks, actions, max_depth, budget):
    """Ceiling diagnostic: ACTIVE COMPUTE of the SHORTEST action path (perfect
    selection) under `actions`. Pure search, no learning — upper bound on what any
    selector could achieve. Distinguishes 'library lacks abstractions' (C) from
    'learned policy cannot realize them' (A)."""
    vals = []
    for x0, t in tasks:
        p = teacher_actions(x0, t, actions, max_depth, budget)
        if p is not None:
            vals.append(len(p))
    return (statistics.mean(vals) if vals else float("nan"), len(vals) / len(tasks))


def slope(curve, idx=1):
    xs = np.array([c[0] for c in curve], float)
    ys = np.array([c[idx] for c in curve], float)
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])  # per-task slope of mean_steps


def run_seed(seed, cfg):
    import random as _r
    tp = S.train_pool()
    # structured stream: repeated subroutine structure (depth 1-2), random.Random for determinism
    pr = _r.Random(seed)
    stream = [S.gen_task(pr, tp, (1, 2)) for _ in range(cfg.train_tasks)]
    train_keys = {(x0, t) for x0, t in stream}

    # eval splits (held out from stream on (x0,target))
    held = S.sample_disjoint(_r.Random(seed + 1), cfg.eval_tasks, tp, (1, 2), train_keys)
    # Axis-C: same subroutines, NOVEL deeper compositions (depth 3)
    axisC = S.sample_disjoint(_r.Random(seed + 2), cfg.eval_tasks, tp, (3, 3), train_keys)

    kw = dict(hidden=cfg.hidden, kmax=cfg.max_macros, lr=cfg.lr,
              steps_per_task=cfg.steps_per_task, batch_size=cfg.batch,
              max_depth=cfg.max_depth, budget=cfg.budget, eval_every=cfg.eval_every,
              eval_tasks=held, cap=cfg.cap)

    # --- ECS: growing memory + learned gate ---
    ecs_pol, ecs_lib, ecs_curve = train_stream(seed, stream, grow=True, freeze_after=None, **kw)
    ecs_macros = ecs_lib.actions[P:]

    # --- dense+curriculum: primitives only, same (structured) order ---
    _, _, densecur_curve = train_stream(seed, stream, grow=False, freeze_after=None, **kw)
    # --- dense: primitives only, SHUFFLED order (curriculum control) ---
    shuf = list(stream); _r.Random(seed + 99).shuffle(shuf)
    dense_pol, _, dense_curve = train_stream(seed, shuf, grow=False, freeze_after=None, **kw)
    # --- frozen-backbone ablation (ECS, freeze W1,b1 after warmup) ---
    _, fb_lib, fb_curve = train_stream(seed, stream, grow=True,
                                       freeze_after=cfg.train_tasks // 3, **kw)

    # --- final-state evals at full exposure for the control arms that reuse ECS ---
    cap = cfg.cap
    ecs_final = eval_split(ecs_pol, held, ecs_macros, cap, "ecs", seed)
    dense_final = eval_split(dense_pol, held, [], cap, "dense", seed)
    frozenM_final = eval_split(ecs_pol, held, [], cap, "dense", seed)         # ECS policy, no skills available
    cache = set(train_keys)
    cacheM_final = eval_split(ecs_pol, held, ecs_macros, cap, "cache", seed, cache=cache)
    rand_final = eval_split(ecs_pol, held, ecs_macros, cap, "random", seed,
                            fire_rate=ecs_final["fire_rate"])
    # Axis-C (novel composition) for ECS vs dense
    ecs_axisC = eval_split(ecs_pol, axisC, ecs_macros, cap, "ecs", seed)
    dense_axisC = eval_split(dense_pol, axisC, [], cap, "dense", seed)

    # --- ORACLE CEILING (perfect selection; pure search) — distinguishes A vs C ---
    prim_actions = B.primitive_actions()
    skill_actions = prim_actions + ecs_macros
    held_prim = oracle_steps(held, prim_actions, cfg.max_depth, cfg.budget)
    held_skill = oracle_steps(held, skill_actions, cfg.max_depth, cfg.budget)
    axisC_prim = oracle_steps(axisC, prim_actions, cfg.max_depth, cfg.budget)
    axisC_skill = oracle_steps(axisC, skill_actions, cfg.max_depth, cfg.budget)
    oracle = {"held_prim": round(held_prim[0], 3), "held_skill": round(held_skill[0], 3),
              "axisC_prim": round(axisC_prim[0], 3), "axisC_skill": round(axisC_skill[0], 3)}

    return {
        "seed": seed,
        "n_macros": len(ecs_macros),
        "curves": {"ecs": ecs_curve, "dense": dense_curve,
                   "dense_curriculum": densecur_curve, "frozen_backbone": fb_curve},
        "slopes": {"ecs": round(slope(ecs_curve), 4),
                   "dense": round(slope(dense_curve), 4),
                   "dense_curriculum": round(slope(densecur_curve), 4),
                   "frozen_backbone": round(slope(fb_curve), 4)},
        "final": {"ecs": ecs_final, "dense": dense_final, "frozenM": frozenM_final,
                  "cacheM": cacheM_final, "random_routing": rand_final},
        "axisC": {"ecs": ecs_axisC, "dense": dense_axisC},
        "oracle": oracle,
    }


def aggregate(runs):
    def fm(path):
        vals = []
        for r in runs:
            o = r
            for k in path:
                o = o[k]
            vals.append(o)
        return round(statistics.mean(vals), 3), vals

    out = {}
    out["slope_ecs"], se = fm(["slopes", "ecs"])
    out["slope_dense"], _ = fm(["slopes", "dense"])
    out["slope_dense_curriculum"], sdc = fm(["slopes", "dense_curriculum"])
    out["slope_frozen_backbone"], _ = fm(["slopes", "frozen_backbone"])
    out["ecs_final_steps"], _ = fm(["final", "ecs", "mean_steps"])
    out["ecs_final_solve"], _ = fm(["final", "ecs", "solve_rate"])
    out["dense_final_steps"], _ = fm(["final", "dense", "mean_steps"])
    out["dense_final_solve"], _ = fm(["final", "dense", "solve_rate"])
    out["frozenM_steps"], _ = fm(["final", "frozenM", "mean_steps"])
    out["cacheM_steps"], _ = fm(["final", "cacheM", "mean_steps"])
    out["random_routing_steps"], _ = fm(["final", "random_routing", "mean_steps"])
    out["ecs_axisC_steps"], _ = fm(["axisC", "ecs", "mean_steps"])
    out["dense_axisC_steps"], _ = fm(["axisC", "dense", "mean_steps"])
    out["ecs_axisC_solve"], _ = fm(["axisC", "ecs", "solve_rate"])

    out["oracle_held_prim"], _ = fm(["oracle", "held_prim"])
    out["oracle_held_skill"], _ = fm(["oracle", "held_skill"])
    out["oracle_axisC_prim"], _ = fm(["oracle", "axisC_prim"])
    out["oracle_axisC_skill"], _ = fm(["oracle", "axisC_skill"])
    # A-vs-C: does a PERFECT selector have anything to win? (ceiling diagnostic)
    held_ceiling_gain = round(out["oracle_held_prim"] - out["oracle_held_skill"], 3)
    axisC_ceiling_gain = round(out["oracle_axisC_prim"] - out["oracle_axisC_skill"], 3)
    out["ceiling"] = {
        "held_prim": out["oracle_held_prim"], "held_skill": out["oracle_held_skill"],
        "held_gain": held_ceiling_gain,
        "axisC_prim": out["oracle_axisC_prim"], "axisC_skill": out["oracle_axisC_skill"],
        "axisC_gain": axisC_ceiling_gain,
        # pre-registered: gain>~0.5 step => abstractions exist => bottleneck is the
        # learned policy (A/B), build a selector. gain~0 => library inadequate (C).
        "library_has_abstractions": held_ceiling_gain > 0.5,
        "abstractions_compose_to_depth": axisC_ceiling_gain > 0.5,
        "diagnosis": ("A/B: selector is the bottleneck (build 1a')" if held_ceiling_gain > 0.5
                      else "C: library inadequate (fix representation/promotion, not selector)"),
    }
    # matched-loss guard: ECS solve-rate must match dense (within 0.05)
    matched = abs(out["ecs_final_solve"] - out["dense_final_solve"]) <= 0.05
    # core: ECS cheaper than every control at full exposure (active compute)
    cheaper_than_controls = (
        out["ecs_final_steps"] < out["dense_final_steps"]
        and out["ecs_final_steps"] < out["frozenM_steps"]
        and out["ecs_final_steps"] < out["cacheM_steps"]
        and out["ecs_final_steps"] < out["random_routing_steps"])
    declines = out["slope_ecs"] < 0 and out["slope_ecs"] < out["slope_dense_curriculum"]
    survives_axisC = out["ecs_axisC_steps"] < out["dense_axisC_steps"]
    survives_frozenbb = out["slope_frozen_backbone"] < 0
    out["verdict"] = {
        "matched_solve_rate": matched,
        "ecs_cheaper_than_all_controls": cheaper_than_controls,
        "ecs_declines_vs_curriculum": declines,
        "survives_axisC": survives_axisC,
        "survives_frozen_backbone": survives_frozenbb,
        "PASS": matched and cheaper_than_controls and declines and survives_axisC and survives_frozenbb,
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
    ap.add_argument("--cap", type=int, default=24)
    ap.add_argument("--output-json", default="artifacts/ecs_stage1.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    runs = [run_seed(s, args) for s in seeds]
    agg = aggregate(runs)
    protocol = {
        "held_out": True, "controls_present": ["dense", "dense_curriculum", "frozen_M",
                                               "cache_M", "random_routing", "frozen_backbone"],
        "axis_C_present": True, "matched_loss_guard": True,
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
    print("=== ECS Stage 1a — trained gate, active-compute over a stream ===")
    print(f"seeds={seeds}  protocol_valid={protocol['valid']}  macros≈{round(statistics.mean(r['n_macros'] for r in runs),1)}")
    print(f"\n  active compute (decode steps) at full exposure / solve-rate:")
    print(f"    ECS            : {agg['ecs_final_steps']:6.2f}  solve {agg['ecs_final_solve']}")
    print(f"    dense (shuf)   : {agg['dense_final_steps']:6.2f}  solve {agg['dense_final_solve']}")
    print(f"    dense+curric   : slope {agg['slope_dense_curriculum']:+.4f}")
    print(f"    frozen-M       : {agg['frozenM_steps']:6.2f}")
    print(f"    cache-M        : {agg['cacheM_steps']:6.2f}")
    print(f"    random-routing : {agg['random_routing_steps']:6.2f}")
    print(f"  slopes (steps/task vs exposure):  ECS {agg['slope_ecs']:+.4f}  "
          f"dense {agg['slope_dense']:+.4f}  frozen-bb {agg['slope_frozen_backbone']:+.4f}")
    print(f"  Axis-C (novel composition):  ECS {agg['ecs_axisC_steps']:.2f} (solve {agg['ecs_axisC_solve']}) "
          f"vs dense {agg['dense_axisC_steps']:.2f}")
    c = agg["ceiling"]
    print(f"\n  ORACLE CEILING (perfect selection, pure search) — distinguishes A vs C:")
    print(f"    held-out : prim-optimal {c['held_prim']:.2f}  skill-optimal {c['held_skill']:.2f}  "
          f"gain {c['held_gain']:+.2f}")
    print(f"    Axis-C   : prim-optimal {c['axisC_prim']:.2f}  skill-optimal {c['axisC_skill']:.2f}  "
          f"gain {c['axisC_gain']:+.2f}")
    print(f"    => {c['diagnosis']}")
    print(f"\n  VERDICT: matched_solve={v['matched_solve_rate']}  cheaper_than_all={v['ecs_cheaper_than_all_controls']}")
    print(f"           declines_vs_curric={v['ecs_declines_vs_curriculum']}  axisC={v['survives_axisC']}  frozen_bb={v['survives_frozen_backbone']}")
    print(f"  => STAGE 1a PASS = {v['PASS']}")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
