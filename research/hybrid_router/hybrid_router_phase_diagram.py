"""Phase diagram of computation regimes — when does routing stay oracle-like, and
when does it collapse to a single expert?

The bench (`hybrid_router_bench.py`) showed a learned router recovers near-oracle
mixture selection from cheap cues. This asks the deeper question the result
raised: *under what conditions* does that hold? We decompose routing value into
two independent knobs and sweep them:

  complementarity (1 - overlap): how ORTHOGONAL the experts' errors are.
      -> sets the HEADROOM available to routing:  headroom = oracle - best_single.
  identifiability (1 / cue_noise): how separable the latent regime is from the
      cheap signal the router sees.
      -> sets how much of that headroom the router CAPTURES.

Order parameter:  routing_efficiency = (router - best_single) / headroom in [0,1].
Hypothesis: efficiency ~ 1 (oracle-like) at low cue noise, collapses to ~0 at
high cue noise, and the critical noise is ~independent of complementarity
(because efficiency is normalised by headroom). If so: "computational regime is
more predictable than the task" holds precisely while cue_noise < a critical N*.

Latent-regime model (controllable abstraction of attention-vs-memory):
  regime r in {local_good, memory_good}, p=0.5.
  LOCAL correct  : True in local_good ; with prob `overlap` in memory_good.
  MEMORY correct : True in memory_good; with prob `overlap` in local_good.
  router sees cue = sign(r) + N(0, cue_noise) and picks an expert per query.
No LLM; deterministic; reuses the tiny recurrent SSMGate as the router.
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


def gen(n, overlap, cue_noise, seed, tag):
    rng = random.Random(dseed(seed, tag, overlap, cue_noise))
    qs = []
    for i in range(n):
        r = 1 if rng.random() < 0.5 else 0          # 1=memory_good, 0=local_good
        loc = True if r == 0 else (rng.random() < overlap)
        mem = True if r == 1 else (rng.random() < overlap)
        cue = (1.0 if r == 1 else -1.0) + rng.gauss(0, cue_noise)
        qs.append((r, loc, mem, cue))
    return qs


def feats(cue):
    return np.array([[math.tanh(cue), 1.0]], dtype=np.float64)


def cell(overlap, cue_noise, seed, cfg):
    train = gen(cfg.train_q, overlap, cue_noise, seed, "tr")
    test = gen(cfg.test_q, overlap, cue_noise, seed, "te")
    Xs = [feats(c) for (_r, _l, _m, c) in train]
    ys = [1.0 if m and not l else (0.0 if l and not m else float(r))
          for (r, l, m, _c) in train]
    gate = SSMGate(d_in=2, hidden=cfg.hidden, seed=seed)
    for _ in range(cfg.epochs):
        g = {"A": np.zeros_like(gate.A), "B": np.zeros_like(gate.B),
             "w": np.zeros_like(gate.w), "b": 0.0}
        for X, y in zip(Xs, ys):
            gg = gate.backward(gate.forward(X)[1], y)
            for k in g:
                g[k] = g[k] + gg[k]
        for k in g:
            g[k] = g[k] / len(Xs)
        gate.step(g, lr=cfg.lr)

    loc = mem = orc = rou = mempick = 0
    for (r, l, m, c) in test:
        loc += int(l); mem += int(m); orc += int(l or m)
        use_mem = gate.predict(feats(c)) >= 0.5
        mempick += int(use_mem)
        rou += int(m if use_mem else l)
    n = len(test)
    best = max(loc, mem) / n
    oracle = orc / n
    router = rou / n
    head = oracle - best
    eff = (router - best) / head if head > 1e-6 else float("nan")
    return {"best": best, "oracle": oracle, "router": router,
            "headroom": head, "efficiency": eff, "mem_frac": mempick / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23")
    ap.add_argument("--overlaps", default="0.0,0.25,0.5,0.75")
    ap.add_argument("--cue-noises", default="0.0,0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--train-q", type=int, default=400)
    ap.add_argument("--test-q", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--eff-collapse", type=float, default=0.5,
                    help="efficiency below this = routing collapsed")
    ap.add_argument("--output-json", default="artifacts/phase_diagram.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    overlaps = [float(x) for x in args.overlaps.split(",")]
    noises = [float(x) for x in args.cue_noises.split(",")]

    grid = {}
    for ov in overlaps:
        for cn in noises:
            cells = [cell(ov, cn, s, args) for s in seeds]
            grid[(ov, cn)] = {k: round(statistics.mean(c[k] for c in cells), 4)
                              for k in ("best", "oracle", "router", "headroom",
                                        "efficiency", "mem_frac")}

    # critical cue noise N* per overlap: first noise where efficiency < threshold
    nstar = {}
    for ov in overlaps:
        crit = None
        for cn in noises:
            if grid[(ov, cn)]["efficiency"] < args.eff_collapse:
                crit = cn
                break
        nstar[ov] = crit

    payload = {"config": vars(args),
               "grid": {f"ov={k[0]},cue={k[1]}": v for k, v in grid.items()},
               "critical_cue_noise_per_overlap": {str(k): v for k, v in nstar.items()},
               "protocol_evidence": {"order_param": "efficiency=(router-best)/headroom",
                                     "n_seeds": len(seeds), "valid": len(seeds) >= 3}}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Phase diagram of computation regimes (routing efficiency) ===")
    print("efficiency = captured / available headroom; 1.0=oracle-like, 0=collapsed")
    print(f"  rows=complementarity(1-overlap), cols=cue_noise; entries=efficiency")
    print("  overlap \\ cue_noise   " + "  ".join(f"{cn:>5.2f}" for cn in noises))
    for ov in overlaps:
        row = "  ".join(
            (f"{grid[(ov,cn)]['efficiency']:>5.2f}"
             if grid[(ov, cn)]['efficiency'] == grid[(ov, cn)]['efficiency'] else "  nan")
            for cn in noises)
        print(f"  ov={ov:<4} (1-ov={1-ov:.2f})   {row}")
    print("  --- headroom (oracle - best_single) per overlap (cue=0) ---")
    for ov in overlaps:
        print(f"    ov={ov}: headroom={grid[(ov, noises[0])]['headroom']}")
    print("  --- critical cue_noise N* (efficiency drops below "
          f"{args.eff_collapse}) ---")
    for ov in overlaps:
        print(f"    ov={ov}: N* = {nstar[ov]}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
