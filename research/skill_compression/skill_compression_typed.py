#!/usr/bin/env python3
"""Phase 3.3 — typed skill schemas: a controlled REPRESENTATION intervention.

Phase 3.1 found a grammar-specific transfer ceiling: the margin of learned skills
over a matched RANDOM library decays as the grammar diverges (R0 17.8 -> R1 12.8 ->
R2_disjoint 10.4). Phase 3.2 proved the R2_depth half of the story was a *selector*
(routing) limit and fixed it with a depth-aware probe — WITHOUT touching the skills.
So the residual ceiling is now cleanly isolated to the third multiplier: the skill
*representation* itself. A skill here is a CONCRETE primitive sequence, so it can
only fire when that exact byte-sequence helps; a brand-new grammar uses different
constants and never matches.

This script tests whether *typing* the skill lifts that ceiling. A concrete macro is
abstracted into an operation-CLASS schema plus constant slots:

    primitives split into classes:  M = {c(*2), e(*3)},  A = {a(+1), b(+3), d(-1)}
    [c,a,c]  ->  schema (M,A,M) + observed bindings (2,+1,2)

A *typed* skill expands its schema into concrete instantiations (bounded), so it can
match a novel grammar by SHAPE and fill in that grammar's constants. The structural
bet is explicit: novel subroutine N1=[d,e,a] has schema (A,M,A) = the schema of train
subroutine S4=[a,c,b] — so a typed S4 skill should fire on N1-composed tasks that the
concrete S4 macro cannot touch.

Everything else is held fixed: same promoted skill set (grown by the Phase-3.1 online
loop), same solver, same DEPTH-AWARE gate from Phase 3.2. Only the action-set
representation changes. The invariance checks (the user's success criteria) are the
whole point — a representation fix must look DIFFERENT from a routing fix:

  * R2_disjoint  : should RISE (the target)
  * R2_depth     : should stay ~STABLE (if it jumps too, we're back in routing)
  * budget-sensitivity : should DROP (structural matches are found shallow; if more
                          probe budget again "saves everything", it's still routing)
  * R0 / R1      : must NOT degrade (else slot-explosion / macro-overfit)

Controls (the RANDMACRO discipline, repeated): `typed_random` adds the SAME number of
extra actions as `typed` but with random contents — so a win must be SCHEMA-specific,
not just "more / longer jumps". Output is JSON with protocol_evidence.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import skill_compression_bench as B
import skill_compression_stream as S

# operation classes (the "type mask")
CLASS_OF = {"c": "M", "e": "M", "a": "A", "b": "A", "d": "A"}
CLASS_MEMBERS = {"M": ["c", "e"], "A": ["a", "b", "d"]}  # sorted -> deterministic


def schema(prims: list[str]) -> tuple[str, ...]:
    return tuple(CLASS_OF[p] for p in prims)


def instantiations(mp: list[str]) -> list[list[str]]:
    """All concrete sequences sharing mp's schema, ordered by Hamming distance to the
    OBSERVED macro (so a bounded slot budget keeps the observed binding + its nearest
    structural variants first — the 'max bindings' constraint as a single knob)."""
    sch = schema(mp)
    combos = [list(c) for c in product(*[CLASS_MEMBERS[k] for k in sch])]
    combos.sort(key=lambda s: (sum(a != b for a, b in zip(s, mp)), s))
    return combos


def grow_library(seed: int, cfg) -> tuple[S.OnlineLibrary, set]:
    """Phase-3.1 online promotion over a train stream (cheap gate) -> frozen lib."""
    rng = random.Random(seed)
    tp = S.train_pool()
    train = [S.gen_task(rng, tp, (cfg.depth_min, cfg.depth_max)) for _ in range(cfg.train_tasks)]
    lib = S.OnlineLibrary(cfg.util_penalty, cfg.util_threshold, cfg.max_macros)
    for step, (x0, target) in enumerate(train):
        sol, _, ok = B.solve(x0, target, lib.actions, cfg.max_depth, cfg.node_budget)
        if ok and sol is not None:
            lib.observe(sol, step)
    lib.freeze()
    return lib, {(x0, t) for (x0, t) in train}, rng


def build_action_sets(grown: S.OnlineLibrary, cfg, rng_seed: int):
    """Return (concrete, typed, typed_random) action sets over the SAME promoted
    skills. typed = bounded schema expansion; typed_random = matched-count random."""
    prim = B.primitive_actions()
    concrete_macros = [list(mp) for mp in grown.macro_prims]
    concrete = prim + [B.Action(f"M{i}", mp) for i, mp in enumerate(concrete_macros)]

    # typed: bounded schema instantiations, deduped, globally capped
    extra: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for mp in concrete_macros:
        for inst in instantiations(mp)[: cfg.max_inst_per_skill]:
            t = tuple(inst)
            if t not in seen:
                seen.add(t)
                extra.append(inst)
    extra = extra[: cfg.max_typed_actions]
    typed = prim + [B.Action(f"T{i}", e) for i, e in enumerate(extra)]

    # typed_random control: same count & lengths as typed's extras, random contents
    rr = random.Random(rng_seed * 7919 + 3)
    rand_extra = [[rr.choice(B.PRIM_NAMES) for _ in range(len(e))] for e in extra]
    typed_random = prim + [B.Action(f"Z{i}", e) for i, e in enumerate(rand_extra)]

    return prim, concrete, typed, typed_random, len(extra)


def eval_regime(test, actions, prim, cfg) -> dict:
    costs, solved = S.eval_stream(test, actions, "depth", prim, cfg)
    return {
        "mean_nodes": round(statistics.mean(costs), 1),
        "median_nodes": round(statistics.median(costs), 1),
        "solve_rate": round(statistics.mean(solved), 4),
    }


def run_seed(seed: int, cfg) -> dict:
    grown, train_keys, rng = grow_library(seed, cfg)
    prim, concrete, typed, typed_random, n_extra = build_action_sets(grown, cfg, seed)

    arms = {"concrete": concrete, "typed": typed, "typed_random": typed_random}
    regimes_out: dict[str, dict] = {}
    held_out_all = True
    for name in cfg.regimes:
        pool, depth = S.regime_grammar(name, cfg)
        test = S.sample_disjoint(rng, cfg.test_tasks, pool, depth, train_keys)
        held_out_all &= all((x0, t) not in train_keys for (x0, t) in test) and len(test) == cfg.test_tasks
        regimes_out[name] = {arm: eval_regime(test, acts, prim, cfg) for arm, acts in arms.items()}

    return {
        "seed": seed,
        "held_out": held_out_all,
        "n_macros": len(grown.macro_prims),
        "n_typed_extra": n_extra,
        "schemas": sorted({"".join(schema(mp)) for mp in grown.macro_prims}),
        "regimes": regimes_out,
    }


def aggregate(seed_runs, cfg) -> dict:
    arms = ("concrete", "typed", "typed_random")

    def amean(name, arm, field="mean_nodes"):
        return round(statistics.mean([r["regimes"][name][arm][field] for r in seed_runs]), 1)

    out = {}
    for name in cfg.regimes:
        ra = {arm: {"mean_nodes": amean(name, arm),
                    "solve_rate": round(statistics.mean([r["regimes"][name][arm]["solve_rate"] for r in seed_runs]), 4)}
              for arm in arms}
        c, t, z = ra["concrete"]["mean_nodes"], ra["typed"]["mean_nodes"], ra["typed_random"]["mean_nodes"]
        ra["verdict"] = {
            "typed_vs_concrete": round(c - t, 1),          # >0 => typing helps here
            "typed_vs_random": round(z - t, 1),            # >0 => schema-specific (not just more actions)
            "typed_helps": t < c,
            "schema_specific": t < z,
        }
        out[name] = ra

    # Headline invariance verdict (the user's success criteria): a REPRESENTATION
    # fix must (1) lift R2_disjoint by a SCHEMA-specific margin (beat typed_random,
    # not just concrete), (2) leave R2_depth ~stable (else it's a routing/action-
    # count effect), and the gain must not be the generic-more-actions artifact.
    rd, dp = out.get("R2_disjoint", {}), out.get("R2_depth", {})
    rd_schema_margin = rd.get("verdict", {}).get("typed_vs_random", 0.0)
    dp_c = dp.get("concrete", {}).get("mean_nodes", 1.0) or 1.0
    dp_move = abs(dp.get("verdict", {}).get("typed_vs_concrete", 0.0)) / dp_c
    out["summary"] = {
        "r2_disjoint_schema_specific_margin": rd_schema_margin,
        "r2_depth_relative_move_vs_concrete": round(dp_move, 3),
        "r2_depth_stable": dp_move < 0.15,
        # all regimes: is the typed gain schema-specific (beats matched random)?
        "schema_specific_all_regimes": all(out[n]["verdict"]["schema_specific"] for n in cfg.regimes),
        # the claim under test:
        "representation_lifts_ceiling": rd_schema_margin > 2.0 and dp_move < 0.15
        and all(out[n]["verdict"]["schema_specific"] for n in cfg.regimes),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--train-tasks", type=int, default=120)
    ap.add_argument("--test-tasks", type=int, default=120)
    ap.add_argument("--depth-min", type=int, default=1)
    ap.add_argument("--depth-max", type=int, default=2)
    ap.add_argument("--depth-shift-min", type=int, default=3)
    ap.add_argument("--depth-shift-max", type=int, default=4)
    ap.add_argument("--r1-overlap", type=float, default=0.5)
    ap.add_argument("--max-depth", type=int, default=14)
    ap.add_argument("--node-budget", type=int, default=2_000_000)
    ap.add_argument("--util-penalty", type=float, default=1.0)
    ap.add_argument("--util-threshold", type=float, default=2.0)
    ap.add_argument("--max-macros", type=int, default=8)
    # depth-aware gate (Phase 3.2) — reused, held fixed
    ap.add_argument("--deep-probe-budget", type=int, default=4000)
    ap.add_argument("--checkpoints", default="40,200,700")
    ap.add_argument("--probe-min-improve", type=int, default=1)
    ap.add_argument("--probe-patience", type=int, default=2)
    # typed-schema knobs (the intervention)
    ap.add_argument("--max-inst-per-skill", type=int, default=6,
                    help="bounded slot budget: instantiations kept per schema (nearest-first). "
                         "1 ≈ concrete; large = full schema expansion (slot-explosion risk).")
    ap.add_argument("--max-typed-actions", type=int, default=40,
                    help="global cap on extra typed actions (controls branching / slot explosion).")
    ap.add_argument("--regimes", default="R0_identical,R1_partial,R2_disjoint,R2_depth")
    ap.add_argument("--output-json", default="artifacts/skill_compression_typed.json")
    args = ap.parse_args()
    args.regimes = [r for r in args.regimes.split(",") if r]
    args.checkpoints = tuple(int(c) for c in str(args.checkpoints).split(",") if c)

    seeds = [int(s) for s in args.seeds.split(",")]
    seed_runs = [run_seed(s, args) for s in seeds]
    agg = aggregate(seed_runs, args)

    held_out = all(r["held_out"] for r in seed_runs)
    protocol_evidence = {
        "held_out": held_out,
        "same_promoted_skills_across_arms": True,
        "depth_aware_gate_held_fixed": True,
        "typed_random_control_present": True,
        "n_seeds": len(seeds),
        "parameter_matched": "identical DSL + solver + depth-aware gate; only action-set representation differs",
        "valid": held_out and len(seeds) >= 3,
    }
    payload = {"config": vars(args), "protocol_evidence": protocol_evidence,
               "aggregate": agg, "seed_runs": seed_runs}
    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Skill-Compression Phase 3.3 — typed skill schemas (representation) ===")
    print(f"seeds={seeds}  protocol_valid={protocol_evidence['valid']}  "
          f"typed_extra≈{round(statistics.mean([r['n_typed_extra'] for r in seed_runs]),1)}  "
          f"max_inst={args.max_inst_per_skill} cap={args.max_typed_actions}")
    print("\nTEST mean nodes/task (concrete / typed / typed_random):")
    for name in args.regimes:
        ra = agg[name]
        v = ra["verdict"]
        print(f"  {name:13s}: {ra['concrete']['mean_nodes']:8.1f} / {ra['typed']['mean_nodes']:8.1f} / "
              f"{ra['typed_random']['mean_nodes']:8.1f}   "
              f"typed_helps={str(v['typed_helps']):5s} schema_specific={str(v['schema_specific']):5s} "
              f"(Δvs_concrete={v['typed_vs_concrete']:+.1f}, Δvs_random={v['typed_vs_random']:+.1f})")
    s = agg["summary"]
    print("\nINVARIANCE VERDICT (representation vs generic-more-actions artifact):")
    print(f"  R2_disjoint schema-specific margin (typed vs RANDOM) = {s['r2_disjoint_schema_specific_margin']:+.1f}  "
          f"(must be >0 and large)")
    print(f"  R2_depth relative move vs concrete = {s['r2_depth_relative_move_vs_concrete']}  "
          f"stable={s['r2_depth_stable']}  (must be stable, else it's routing)")
    print(f"  schema_specific_all_regimes = {s['schema_specific_all_regimes']}")
    print(f"  => representation_lifts_ceiling = {s['representation_lifts_ceiling']}")
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
