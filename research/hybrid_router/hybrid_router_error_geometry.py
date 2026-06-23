"""STEP 2 (sharpened) — measure the emergent law: routing exists iff errors are
orthogonal. Both experts are already mechanistic (window-scan vs commit-lag), so
the real frontier is not "replace an expert" but the ERROR-GEOMETRY axis.

We add a controllable error-correlation knob `corr`: a fraction of queries are
"hard" — BOTH experts fail together (correlated failure). corr=0 keeps the
orthogonal structure (one expert always right); corr=1 makes failures fully
correlated. We sweep corr x cue_noise and predict the factorisation found at
v1.0, now with mechanistic experts and the error axis made explicit:

  headroom (= oracle - best_single)  depends on corr ONLY   -> error geometry
  efficiency (= advantage/headroom)  depends on cue_noise ONLY -> identifiability
  routing_value (absolute advantage) = headroom x efficiency  -> collapses with corr

So routing is a property of ERROR GEOMETRY, not model capacity: it vanishes as
errors correlate even when the regime is perfectly identifiable. Anti-illusion
guards (efficiency/headroom, min-regime, shuffle) ride along at every cell.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_router_real_expert import (dseed, simulate, train_router, evaluate)


def corrupt(records, corr, seed, tag):
    """Inject correlated failure: with prob corr, force BOTH experts wrong on a
    query (a 'hard' regime shared by both) -> errors become correlated."""
    rng = random.Random(dseed(seed, "corr", corr, tag))
    out = []
    for r in records:
        rr = dict(r)
        if rng.random() < corr:
            rr["loc_ok"] = False
            rr["mem_ok"] = False
        out.append(rr)
    return out


def run_cell(seed, cfg, corr, cue_noise):
    a = (seed, cfg.n_ops, cfg.n_keys, cfg.window, cfg.lag, cfg.p_write,
         cfg.vocab, cue_noise)
    train = corrupt(simulate(*a, "train"), corr, seed, "train")
    test = corrupt(simulate(*a, "test"), corr, seed, "test")
    gate = train_router(train, seed, cfg)
    m = evaluate(test, gate)
    gate_s = train_router(train, seed, cfg, shuffle=True)
    ms = evaluate(test, gate_s, shuffle=True)
    best = max(m["local"], m["memory"])
    head = m["oracle"] - best
    return {"best": best, "oracle": m["oracle"], "router": m["router"],
            "headroom": head,
            "advantage": m["router"] - best,
            "efficiency": (m["router"] - best) / head if head > 1e-6 else None,
            "min_regime": min(m["router_local_fav"], m["router_mem_fav"]),
            "shuffle_gap": m["router"] - ms["router"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--corrs", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--cue-noises", default="0.0,2.0,8.0")
    ap.add_argument("--n-ops", type=int, default=1200)
    ap.add_argument("--n-keys", type=int, default=24)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--lag", type=int, default=3)
    ap.add_argument("--p-write", type=float, default=0.5)
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--output-json", default="artifacts/error_geometry.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    corrs = [float(x) for x in args.corrs.split(",")]
    noises = [float(x) for x in args.cue_noises.split(",")]

    grid = {}
    for corr in corrs:
        for cn in noises:
            cells = [run_cell(s, args, corr, cn) for s in seeds]
            grid[(corr, cn)] = {
                k: (round(statistics.mean(c[k] for c in cells), 4)
                    if all(c[k] is not None for c in cells) else None)
                for k in ("headroom", "advantage", "efficiency",
                          "min_regime", "shuffle_gap", "best", "oracle", "router")}

    payload = {"config": vars(args),
               "grid": {f"corr={k[0]},cue={k[1]}": v for k, v in grid.items()},
               "law": "headroom~f(corr); efficiency~g(cue_noise); "
                      "routing_advantage=headroom*efficiency",
               "protocol_evidence": {"both_experts_mechanistic": True,
                                     "n_seeds": len(seeds), "valid": len(seeds) >= 3}}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Error-geometry law: routing_value vs error correlation x cue noise ===")
    print("  HEADROOM (oracle-best) — expect: depends on corr, flat across cue:")
    print("   corr\\cue   " + "  ".join(f"{cn:>5.1f}" for cn in noises))
    for corr in corrs:
        print(f"   {corr:>4.2f}      " +
              "  ".join(f"{grid[(corr,cn)]['headroom']:>5.2f}" for cn in noises))
    print("  ADVANTAGE (router-best) — absolute routing value (collapses with corr):")
    for corr in corrs:
        print(f"   {corr:>4.2f}      " +
              "  ".join(f"{grid[(corr,cn)]['advantage']:>5.2f}" for cn in noises))
    print("  EFFICIENCY (adv/headroom) — expect: depends on cue, flat across corr:")
    for corr in corrs:
        row = "  ".join((f"{grid[(corr,cn)]['efficiency']:>5.2f}"
                         if grid[(corr,cn)]['efficiency'] is not None else "  nan")
                        for cn in noises)
        print(f"   {corr:>4.2f}      {row}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
