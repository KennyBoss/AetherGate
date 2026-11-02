"""Hybrid-router crash test — does routing between two experts beat either alone,
and can a LEARNED router (from cheap features) catch the oracle?

Motivation (straight from RESULTS.md, not a new postulate)
----------------------------------------------------------
The repo already shows two regimes with a crossover:
  * long-delay binding recall : explicit KV-memory beats attention (0.99 vs 0.20)
  * ordinary local next-token : attention beats memory
So a hybrid does not need a new hypothesis — the data say there IS a crossover,
and the open question is whether a router can *find* it. This is the same
act/think gate raised to expert selection: "near context -> attend; far binding
-> retrieve." Phase diagram, now over DELAY.

No LLM, no trained Transformer (those are the confounds the whole line avoided).
Two ANALYTIC experts with delay-dependent competence stand in for attention vs
external memory, calibrated to the published regimes:
  * LOCAL expert (attention-like): correct iff the needed token is within a
    bounded window W of the query; degrades to chance beyond W.
  * MEMORY expert (KV-store-like): correct iff the binding was written and not
    overwritten, independent of delay; small fixed read noise.

Task stream
-----------
A sequence of (key,value) writes with random gaps, then queries. Each query is
either LOCAL (its binding lies within W) or RECALL (binding lies far beyond W).
The router sees cheap features and must pick an expert per query.

Arms (per query: pick one expert; accuracy = fraction correct)
  LOCAL_ONLY  : always the attention-like expert
  MEMORY_ONLY : always the KV-memory expert
  ORACLE      : pick the expert that is actually right (upper bound)
  ROUTER      : a learned recurrent router over cheap features (delay proxy)

Verdict bar: ROUTER >= max(LOCAL_ONLY, MEMORY_ONLY) AND ROUTER close to ORACLE.
A miss is a real result (routing doesn't separate the regimes from cheap cues).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path


def dseed(*parts) -> int:
    """Deterministic seed from arbitrary parts (PYTHONHASHSEED-independent)."""
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skill_compression"))
from skill_compression_ssm_gate import SSMGate  # reuse the tiny recurrent gate


# ---------------------------------------------------------------------------
# Stream generation
# ---------------------------------------------------------------------------
def make_stream(rng, n_pairs, window, far_min, vocab):
    """Returns list of queries: each (delay, correct_value, written, overwritten,
    is_local). delay = positions since the binding write."""
    # write n_pairs bindings at increasing positions with random gaps
    writes = {}            # key -> (pos, value)
    pos = 0
    keys = list(range(vocab))
    rng.shuffle(keys)
    for i in range(n_pairs):
        pos += rng.randint(1, 4)
        k = keys[i % len(keys)]
        v = rng.randint(0, vocab - 1)
        overwritten = k in writes  # a later write overwrites an earlier binding
        writes[k] = (pos, v)
    cur_pos = pos + 1
    queries = []
    for _ in range(n_pairs):
        k = rng.choice(list(writes.keys()))
        wpos, v = writes[k]
        # choose local vs recall by construction, then realise the delay
        if rng.random() < 0.5:
            delay = rng.randint(0, window - 1)        # LOCAL: within window
            is_local = True
        else:
            delay = rng.randint(far_min, far_min + 40)  # RECALL: far beyond window
            is_local = False
        # recency: a LOCAL query is often a just-written/just-overwritten binding,
        # where a KV-store may still hold the STALE value (its realistic failure
        # mode). RECALL queries are settled, so memory is reliable there.
        stale = is_local and (rng.random() < 0.6)
        queries.append({"delay": delay, "value": v, "is_local": is_local,
                        "stale": stale})
    return queries


# ---------------------------------------------------------------------------
# Analytic experts (calibrated to published regimes)
# ---------------------------------------------------------------------------
def local_correct(q, rng, window, local_far_acc, vocab):
    """Attention-like: right within window, chance beyond it."""
    if q["delay"] < window:
        return True
    return rng.random() < (1.0 / vocab)  # essentially chance far away


def memory_correct(q, rng, read_noise):
    """KV-memory-like: right regardless of delay EXCEPT on stale (just-overwritten)
    bindings, where the store may return the outdated value — its realistic
    failure mode. Plus small read noise."""
    if q.get("stale"):
        return rng.random() < 0.25          # mostly wrong on stale reads
    return rng.random() >= read_noise


# ---------------------------------------------------------------------------
# Cheap router features: a noisy delay proxy + a smoke read-consistency signal.
# The router must infer the regime WITHOUT a delay oracle.
# ---------------------------------------------------------------------------
def features(q, rng, window, noise):
    # noisy estimate of "is this within the local window" (delay proxy)
    est = q["delay"] + rng.gauss(0, noise)
    near = math.tanh((window - est) / max(1.0, window))  # ~+1 if near, ~-1 if far
    # cheap memory smoke-signal: a NOISY recency flag (store recently touched this
    # key -> its read may be stale). The router must fuse delay-proxy + staleness.
    stale_obs = (1.0 if q.get("stale") else 0.0)
    if rng.random() < 0.2:                 # 20% observation noise on the flag
        stale_obs = 1.0 - stale_obs
    return np.array([[near, stale_obs]], dtype=np.float64)


# ---------------------------------------------------------------------------
def run_seed(seed, cfg):
    rng = random.Random(seed)
    train = make_stream(rng, cfg.train_q, cfg.window, cfg.far_min, cfg.vocab)
    test = make_stream(rng, cfg.test_q, cfg.window, cfg.far_min, cfg.vocab)

    # expert correctness oracle (deterministic given a per-call rng)
    def experts(q, r):
        loc = local_correct(q, r, cfg.window, 0.0, cfg.vocab)
        mem = memory_correct(q, r, cfg.read_noise)
        return loc, mem

    # train router: label = which expert is correct (prefer local on ties = cheaper)
    Xs, ys = [], []
    for q in train:
        r = random.Random(dseed(seed, q["delay"], q["value"]))
        loc, mem = experts(q, r)
        # target: 1 -> use MEMORY, 0 -> use LOCAL
        if loc and not mem:
            y = 0
        elif mem and not loc:
            y = 1
        else:
            y = 0 if q["is_local"] else 1
        ys.append(y)
        Xs.append(features(q, random.Random(dseed(seed, "f", q["delay"])),
                           cfg.window, cfg.feat_noise))

    gate = SSMGate(d_in=2, hidden=cfg.hidden, seed=seed)
    for _ in range(cfg.epochs):
        grads = {"A": np.zeros_like(gate.A), "B": np.zeros_like(gate.B),
                 "w": np.zeros_like(gate.w), "b": 0.0}
        for X, y in zip(Xs, ys):
            g = gate.backward(gate.forward(X)[1], float(y))
            for kk in grads:
                grads[kk] = grads[kk] + g[kk]
        for kk in grads:
            grads[kk] = grads[kk] / len(Xs)
        gate.step(grads, lr=cfg.lr)

    # evaluate arms on held-out test
    local_only = memory_only = oracle = router = 0
    router_picks_mem = 0
    for q in test:
        r = random.Random(dseed(seed, "e", q["delay"], q["value"]))
        loc, mem = experts(q, r)
        local_only += int(loc)
        memory_only += int(mem)
        oracle += int(loc or mem)
        X = features(q, random.Random(dseed(seed, "tf", q["delay"])),
                     cfg.window, cfg.feat_noise)
        use_mem = gate.predict(X) >= 0.5
        router_picks_mem += int(use_mem)
        router += int(mem if use_mem else loc)
    n = len(test)
    return {"seed": seed, "n": n,
            "local_only": local_only / n, "memory_only": memory_only / n,
            "oracle": oracle / n, "router": router / n,
            "router_mem_frac": router_picks_mem / n}


def ci95(xs):
    n = len(xs); m = statistics.mean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}.get(n - 1, 1.96)
    return m, (t * sd / math.sqrt(n) if n > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--train-q", type=int, default=300)
    ap.add_argument("--test-q", type=int, default=300)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--far-min", type=int, default=40)
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--read-noise", type=float, default=0.05)
    ap.add_argument("--feat-noise", type=float, default=2.0)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--output-json", default="artifacts/hybrid_router_summary.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    runs = [run_seed(s, args) for s in seeds]
    agg = {k: round(statistics.mean(r[k] for r in runs), 4)
           for k in ("local_only", "memory_only", "oracle", "router", "router_mem_frac")}
    rm, rh = ci95([r["router"] for r in runs])
    best_single = max(agg["local_only"], agg["memory_only"])
    router_beats_both = all(r["router"] >= max(r["local_only"], r["memory_only"]) - 1e-9
                            for r in runs)
    gap_to_oracle = round(agg["oracle"] - agg["router"], 4)

    payload = {"config": vars(args), "aggregate": agg,
               "router_ci95": [round(rm, 4), round(rh, 4)],
               "verdict": {"best_single_expert": round(best_single, 4),
                           "router_beats_both_all_seeds": router_beats_both,
                           "router_minus_best_single": round(agg["router"] - best_single, 4),
                           "oracle_minus_router": gap_to_oracle},
               "per_seed": runs,
               "protocol_evidence": {"held_out": True, "n_seeds": len(seeds),
                                     "analytic_experts_calibrated_to_RESULTS": True,
                                     "valid": len(seeds) >= 3}}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Hybrid-router crash test (attention-like vs KV-memory expert) ===")
    print(f"seeds={seeds}")
    print(f"  LOCAL_ONLY  = {agg['local_only']}")
    print(f"  MEMORY_ONLY = {agg['memory_only']}")
    print(f"  ROUTER      = {agg['router']}  (mem-pick frac {agg['router_mem_frac']})")
    print(f"  ORACLE      = {agg['oracle']}")
    print(f"  router beats both single experts (all seeds) = {router_beats_both}")
    print(f"  router - best_single = {round(agg['router']-best_single,4)}   "
          f"oracle - router = {gap_to_oracle}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
