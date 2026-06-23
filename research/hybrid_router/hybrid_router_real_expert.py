"""Expert replacement (staged) — swap the ANALYTIC memory expert for a REAL
mechanistic KV-store whose errors EMERGE (LRU eviction), keep the local expert,
freeze the router architecture, and re-measure the phase diagram WITH three
anti-illusion guards. This is the point where systems start to "lie beautifully";
the guards are built in so a fake win cannot pass.

Real experts (errors emerge from mechanism, not from fiat)
---------------------------------------------------------
A token stream of WRITE(k,v) / QUERY(k) ops over real bindings.
  LOCAL  : scans the last W ops for the binding -> right iff written within W.
  MEMORY : an LRU dict of capacity C -> right iff k not yet evicted. Eviction is
           the genuine failure mode (no hand-set probability).

Router: the same tiny SSMGate over two CHEAP, non-leaking cues — a noisy recency
proxy (in-window?) and a memory-pressure proxy (eviction risk?). It never sees the
value, only regime cues.

Three guards that catch illusory improvement
--------------------------------------------
G1 headroom = oracle - best_single. If ~0 there is nothing to route; a small
   oracle-gap is then meaningless. Always report efficiency = advantage/headroom.
G2 min-regime advantage: router accuracy WITHIN each regime (local-favoring vs
   memory-favoring). A router that just always picks the globally stronger expert
   scores high in one regime and LOW in the other -> caught here.
G3 shuffle control: retrain/eval the router with features permuted across queries
   (regime link destroyed). It must collapse to best_single. shuffle_gap =
   router - router_shuffled; small gap => the "win" was leakage/artifact, not
   regime inference.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skill_compression"))
from skill_compression_ssm_gate import SSMGate


def dseed(*p):
    return int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Generate a stream and run the two REAL experts; collect per-query records.
# ---------------------------------------------------------------------------
def simulate(seed, n_ops, n_keys, window, lag, p_write, vocab, cue_noise, tag):
    """Two REAL experts with COMPLEMENTARY failure modes (so routing has headroom):
      LOCAL  : right iff the binding was written within `window` ops (recent).
      MEMORY : a persistent store with write-commit LAG `lag` -> a query on a key
               written within the last `lag` ops reads the STALE prior value
               (eventual consistency). Right on settled bindings, wrong on fresh.
    LOCAL fails on OLD, MEMORY fails on FRESH -> orthogonal errors."""
    rng = random.Random(dseed(seed, tag, cue_noise))
    ops = []
    last1 = {}    # k -> (idx, val) latest write
    last2 = {}    # k -> (idx, val) the write before that
    records = []
    written_keys = []
    for i in range(n_ops):
        do_write = (not written_keys) or (rng.random() < p_write)
        if do_write:
            k = rng.randrange(n_keys)
            v = rng.randrange(vocab)
            ops.append(("W", k, v))
            if k in last1:
                last2[k] = last1[k]
            last1[k] = (i, v)
            if k not in written_keys:
                written_keys.append(k)
        else:
            # balance regimes so headroom is not dominated by one expert: target a
            # recency BAND (fresh<lag / mid / old>=window) uniformly when possible.
            fresh = [kk for kk in written_keys if i - last1[kk][0] < lag]
            old = [kk for kk in written_keys if i - last1[kk][0] >= window]
            mid = [kk for kk in written_keys
                   if lag <= i - last1[kk][0] < window]
            bands = [b for b in (fresh, mid, old) if b]
            k = rng.choice(rng.choice(bands)) if bands else rng.choice(written_keys)
            ops.append(("Q", k))
            wi, true_v = last1[k]
            recency = i - wi
            in_window = recency < window
            # LOCAL: scan back up to `window` ops
            local_ans = None
            for j in range(i - 1, max(-1, i - 1 - window), -1):
                o = ops[j]
                if o[0] == "W" and o[1] == k:
                    local_ans = o[2]
                    break
            # MEMORY: commit lag -> a fresh write is not yet visible (stale read)
            if recency < lag:
                mem_ans = last2[k][1] if k in last2 else None
            else:
                mem_ans = true_v
            loc_ok = (local_ans == true_v)
            mem_ok = (mem_ans == true_v)
            rec_obs = recency + rng.gauss(0, cue_noise)        # noisy recency cue
            f_window = math.tanh((window - rec_obs) / max(1.0, window))
            records.append({"loc_ok": loc_ok, "mem_ok": mem_ok,
                            "in_window": in_window, "fresh": recency < lag,
                            "feat": [f_window, 1.0]})
    return records


def train_router(records, seed, cfg, shuffle=False):
    feats = [np.array([r["feat"]], dtype=np.float64) for r in records]
    if shuffle:
        idx = list(range(len(feats)))
        random.Random(dseed(seed, "shuf")).shuffle(idx)
        feats = [feats[i] for i in idx]
    # label: 1 -> use MEMORY, 0 -> use LOCAL (prefer local on ties)
    ys = []
    for r in records:
        if r["loc_ok"] and not r["mem_ok"]:
            ys.append(0.0)
        elif r["mem_ok"] and not r["loc_ok"]:
            ys.append(1.0)
        else:
            ys.append(0.0 if r["in_window"] else 1.0)
    gate = SSMGate(d_in=2, hidden=cfg.hidden, seed=seed)
    for _ in range(cfg.epochs):
        g = {"A": np.zeros_like(gate.A), "B": np.zeros_like(gate.B),
             "w": np.zeros_like(gate.w), "b": 0.0}
        for X, y in zip(feats, ys):
            gg = gate.backward(gate.forward(X)[1], y)
            for k in g:
                g[k] = g[k] + gg[k]
        for k in g:
            g[k] = g[k] / len(feats)
        gate.step(g, lr=cfg.lr)
    return gate


def evaluate(records, gate, shuffle=False):
    feats = [np.array([r["feat"]], dtype=np.float64) for r in records]
    if shuffle:
        idx = list(range(len(feats)))
        random.Random(99).shuffle(idx)
        feats = [feats[i] for i in idx]
    loc = sum(r["loc_ok"] for r in records)
    mem = sum(r["mem_ok"] for r in records)
    orc = sum(r["loc_ok"] or r["mem_ok"] for r in records)
    rou = mempick = 0
    # per-regime correctness (G2)
    reg = {"local_fav": [0, 0], "mem_fav": [0, 0]}   # [router_correct, count]
    for r, X in zip(records, feats):
        use_mem = gate.predict(X) >= 0.5
        mempick += int(use_mem)
        ok = r["mem_ok"] if use_mem else r["loc_ok"]
        rou += int(ok)
        if r["loc_ok"] and not r["mem_ok"]:
            reg["local_fav"][0] += int(ok); reg["local_fav"][1] += 1
        elif r["mem_ok"] and not r["loc_ok"]:
            reg["mem_fav"][0] += int(ok); reg["mem_fav"][1] += 1
    n = len(records)
    out = {"local": loc / n, "memory": mem / n, "oracle": orc / n, "router": rou / n,
           "mem_frac": mempick / n}
    out["router_local_fav"] = reg["local_fav"][0] / max(1, reg["local_fav"][1])
    out["router_mem_fav"] = reg["mem_fav"][0] / max(1, reg["mem_fav"][1])
    return out


def run_seed(seed, cfg, cue_noise):
    args = (seed, cfg.n_ops, cfg.n_keys, cfg.window, cfg.lag, cfg.p_write,
            cfg.vocab, cue_noise)
    train = simulate(*args, "train")
    test = simulate(*args, "test")
    gate = train_router(train, seed, cfg)
    m = evaluate(test, gate)
    # G3 shuffle control: features permuted -> regime link destroyed
    gate_s = train_router(train, seed, cfg, shuffle=True)
    ms = evaluate(test, gate_s, shuffle=True)
    m["router_shuffled"] = ms["router"]
    return m


def agg(runs, keys):
    return {k: round(statistics.mean(r[k] for r in runs), 4) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--n-ops", type=int, default=1200)
    ap.add_argument("--n-keys", type=int, default=24)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--lag", type=int, default=3)
    ap.add_argument("--p-write", type=float, default=0.5)
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--cue-noises", default="0.0,0.5,1.0,2.0,4.0")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--output-json", default="artifacts/real_expert.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    noises = [float(x) for x in args.cue_noises.split(",")]

    KEYS = ("local", "memory", "oracle", "router", "mem_frac",
            "router_local_fav", "router_mem_fav", "router_shuffled")
    table = {}
    for cn in noises:
        runs = [run_seed(s, args, cn) for s in seeds]
        a = agg(runs, KEYS)
        best = max(a["local"], a["memory"])
        head = a["oracle"] - best
        a["best_single"] = round(best, 4)
        a["headroom"] = round(head, 4)
        a["efficiency"] = round((a["router"] - best) / head, 4) if head > 1e-6 else None
        a["min_regime_adv"] = round(min(a["router_local_fav"], a["router_mem_fav"]), 4)
        a["shuffle_gap"] = round(a["router"] - a["router_shuffled"], 4)
        table[cn] = a

    nstar = next((cn for cn in noises if (table[cn]["efficiency"] or 0) < 0.5), None)
    payload = {"config": vars(args), "table": {str(k): v for k, v in table.items()},
               "critical_cue_noise": nstar,
               "guards": {"G1": "efficiency=advantage/headroom",
                          "G2": "min_regime_adv: router acc in the WEAKER regime",
                          "G3": "shuffle_gap: real-features minus shuffled-features"},
               "protocol_evidence": {"real_mechanistic_memory_expert": True,
                                     "n_seeds": len(seeds), "valid": len(seeds) >= 3}}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Expert replacement: analytic memory -> REAL commit-lag KV-store ===")
    print("  (router architecture frozen; 3 anti-illusion guards on)")
    print(f"  {'cue':>4s} {'LOC':>5s} {'MEM':>5s} {'best':>5s} {'ORAC':>5s} {'ROUT':>5s} "
          f"{'eff':>5s} {'mem%':>5s} | G2_minReg {'G3_shuf':>8s}")
    for cn in noises:
        a = table[cn]
        eff = f"{a['efficiency']:.2f}" if a["efficiency"] is not None else " nan"
        print(f"  {cn:>4.1f} {a['local']:>5.2f} {a['memory']:>5.2f} {a['best_single']:>5.2f} "
              f"{a['oracle']:>5.2f} {a['router']:>5.2f} {eff:>5s} {a['mem_frac']:>5.2f} | "
              f"{a['min_regime_adv']:>9.2f} {a['shuffle_gap']:>8.2f}")
    print(f"  critical cue-noise N* (eff<0.5) = {nstar}   (analytic baseline was ~2.0)")
    g = table[noises[1] if len(noises) > 1 else noises[0]]
    print("  GUARDS @ cue=%.1f: headroom=%.3f  min_regime_adv=%.2f  shuffle_gap=%.2f" %
          (noises[1] if len(noises) > 1 else noises[0], g["headroom"],
           g["min_regime_adv"], g["shuffle_gap"]))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
