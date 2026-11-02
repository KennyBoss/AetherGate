#!/usr/bin/env python3
"""Phase 3.1 — the self-improvement stream: skills BORN from solved tasks.

Every prior experiment in this directory either promoted the skill library in a
single batch (`skill_compression_bench.build_library`) and then froze it for eval,
or — in Phase 2 — *provided the library identically to both arms* to isolate the
applicability gate. None of them ran the closed loop the roadmap flags as THE open
term `g_t` (`../ROADMAP_SKILL_COMPRESSION.md` §1):

    Task -> Solve (with current skills + applicability gate) -> extract skill from
    the solution -> library grows -> the NEXT task is cheaper.

This script runs that loop online over a task stream and asks the user's question:

    Does cost-per-task FALL as experience accumulates, and — after we FREEZE the
    library and SHIFT the task grammar ("a new repository") — does the saving
    TRANSFER, or does the curve rebound to baseline?

It reuses the proven, parameter-matched machinery of `skill_compression_bench`
(identical DSL, identical deterministic iterative-deepening solver, identical UTIL
promotion criterion, identical applicability gate) so the only new thing measured
is the *temporal* behaviour of an auto-growing library.

Three arms, streamed over the SAME tasks (parameter-matched)
-----------------------------------------------------------
* growing_gate  : the protagonist. Online UTIL promotion; each task solved behind
                  the applicability gate (cheap macro-probe + primitive fallback).
* frozen_empty  : control. Primitives only, never promotes. The "no compression"
                  baseline; its curve should be flat.
* random_growing: control. Grows a library COUNT/LENGTH-matched to growing_gate's
                  promotion schedule, but RANDOM contents. Isolates "real reuse"
                  from "just having more / longer jumps" (the Phase-1 RANDMACRO
                  control, now online).

The freeze -> grammar-shift axis (the "new repo")
-------------------------------------------------
After `--train-tasks` tasks the library is FROZEN; the test stream is drawn from a
shifted grammar:
* R0_identical : same subroutine pool, same depth.            (sanity)
* R1_partial   : 50% train + 50% novel subroutines.           (transfer ∝ overlap)
* R2_disjoint  : 100% NOVEL subroutines, same depth.          (HEADLINE transfer)
* R2_depth     : train pool, DEEPER compositions.             (compositional reach)

Honest prediction (the result is meant to be falsifiable, not flattering):
R0/R1 transfer; R2_disjoint does NOT (concrete macros cannot match a brand-new
grammar) but the gate prevents a *blow-up* (no benefit, no harm); R2_depth shows
the library reaching tasks too deep for primitive brute force — while the gate's
CHEAP probe is too myopic to verify those deep compositions (the Phase-1.5B myopia
re-appearing on its own). Output is JSON with a `protocol_evidence` block.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

import skill_compression_bench as B

# ---------------------------------------------------------------------------
# Grammars: a "repository" is a hidden subroutine pool + a composition depth.
# Train pool = the bench's hidden subroutines. The novel pool is disjoint from it
# (no shared subroutine), same length (3) and count (4) so difficulty is matched.
# ---------------------------------------------------------------------------
NOVEL_SUBROUTINES: dict[str, list[str]] = {
    "N1": ["d", "e", "a"],
    "N2": ["c", "c", "d"],
    "N3": ["a", "a", "e"],
    "N4": ["b", "d", "c"],
}
REGIMES = ("R0_identical", "R1_partial", "R2_disjoint", "R2_depth")


def train_pool() -> list[list[str]]:
    return [list(v) for v in B.HIDDEN_SUBROUTINES.values()]


def novel_pool() -> list[list[str]]:
    return [list(v) for v in NOVEL_SUBROUTINES.values()]


def regime_grammar(name: str, cfg: argparse.Namespace) -> tuple[list[list[str]], tuple[int, int]]:
    """Return (subroutine pool, depth range) for a test regime."""
    tp, npool = train_pool(), novel_pool()
    shallow = (cfg.depth_min, cfg.depth_max)
    if name == "R0_identical":
        return tp, shallow
    if name == "R1_partial":
        k = round(len(tp) * cfg.r1_overlap)
        return tp[:k] + npool[: len(tp) - k], shallow
    if name == "R2_disjoint":
        return npool, shallow
    if name == "R2_depth":
        return tp, (cfg.depth_shift_min, cfg.depth_shift_max)
    raise ValueError(name)


def gen_task(rng: random.Random, pool: list[list[str]], depth_range: tuple[int, int]) -> tuple[int, int]:
    """Compose randint(depth_range) subroutines from the pool -> (x0, target)."""
    n_sub = rng.randint(*depth_range)
    prog: list[str] = []
    for _ in range(n_sub):
        prog += rng.choice(pool)
    x0 = rng.randint(0, 5)
    return x0, B.apply_prims(x0, prog)


def sample_disjoint(rng, n, pool, depth_range, train_keys) -> list[tuple[int, int]]:
    """Test tasks whose (x0,target) is held out from the train stream."""
    out: list[tuple[int, int]] = []
    attempts = 0
    while len(out) < n and attempts < n * 400:
        attempts += 1
        x0, t = gen_task(rng, pool, depth_range)
        if (x0, t) not in train_keys:
            out.append((x0, t))
    return out


# ---------------------------------------------------------------------------
# The online skill library. `observe` mirrors build_library's UTIL gate exactly
# (one promotion step per solved task; sorted tie-break for determinism), but is
# driven incrementally so the library is BORN during the stream and used at once.
# ---------------------------------------------------------------------------
class OnlineLibrary:
    def __init__(self, util_penalty: float, util_threshold: float, max_macros: int):
        self.macro_prims: list[list[str]] = []
        self.counts: Counter[tuple[str, ...]] = Counter()
        self.promoted: set[tuple[str, ...]] = set()
        self.frozen = False
        self.util_penalty = util_penalty
        self.util_threshold = util_threshold
        self.max_macros = max_macros
        self.schedule: dict[int, int] = {}  # step -> length of macro promoted
        self._actions = B.primitive_actions()

    @property
    def actions(self) -> list[B.Action]:
        return self._actions

    def _rebuild(self) -> None:
        self._actions = B.primitive_actions() + [
            B.Action(f"M{i}", mp) for i, mp in enumerate(self.macro_prims)
        ]

    def observe(self, solution_prims: list[str], step: int) -> None:
        if self.frozen:
            return
        for sub in B.candidate_subseqs(solution_prims):
            self.counts[sub] += 1
        if len(self.macro_prims) >= self.max_macros:
            return
        best, best_score = None, 0.0
        for sub, cnt in sorted(self.counts.items()):
            if sub in self.promoted or cnt < 2:
                continue
            score = cnt * (len(sub) - 1) - self.util_penalty * len(self.macro_prims)
            if score > best_score:
                best, best_score = sub, score
        if best is not None and best_score > self.util_threshold:
            self.promoted.add(best)
            self.macro_prims.append(list(best))
            self.schedule[step] = len(best)
            self._rebuild()

    def freeze(self) -> None:
        self.frozen = True


class RandomLibrary:
    """RANDMACRO control, online: adds a random macro of the SAME length whenever
    growing_gate promoted one (replays its schedule), so the two libraries are
    count/length-matched step for step — only the contents differ."""

    def __init__(self, schedule: dict[int, int], rng: random.Random):
        self.schedule = schedule
        self.rng = rng
        self.macro_prims: list[list[str]] = []
        self._actions = B.primitive_actions()

    @property
    def actions(self) -> list[B.Action]:
        return self._actions

    def maybe_add(self, step: int) -> None:
        if step in self.schedule:
            L = self.schedule[step]
            self.macro_prims.append([self.rng.choice(B.PRIM_NAMES) for _ in range(L)])
            self._actions = B.primitive_actions() + [
                B.Action(f"R{i}", mp) for i, mp in enumerate(self.macro_prims)
            ]


# ---------------------------------------------------------------------------
# Cost of solving one task under an arm.
#   gate : applicability gate (cheap probe + primitive fallback) — the agent.
#   full : plain solve with the given action set (used for frozen_empty primitives
#          and for the R2_depth "library reach" measurement).
# ---------------------------------------------------------------------------
def probe_macros_depth_aware(
    x0, target, actions, max_depth, max_budget, checkpoints, min_improve, patience=1
) -> tuple[list[str] | None, int, bool, int]:
    """Adaptive-horizon macro probe (Phase 3.2). Same deterministic iterative-
    deepening DFS as `B.solve`, but two things change — and ONLY these two, so the
    solver/library stay parameter-matched and the gate's *horizon control* is the
    isolated variable:

      (a) it escalates up to `max_budget` instead of a tiny fixed cap, so a deep
          multi-macro composition can actually be reached;
      (b) it watches the COST GRADIENT, not just cost: it tracks the best
          distance-to-target seen, and at each checkpoint, if `best` has not
          improved by >= `min_improve` since the previous checkpoint, it concludes
          the library is STALLED on this task and bails (not-solved) — pursuing
          depth where the skills are paying off, abandoning a foreign grammar early.

    Returns (solution_prims, nodes_spent, solved, best_distance).
    """
    total = 0
    best = abs(x0 - target)
    last_ckpt_best = best
    ci = 0
    stalls = 0  # consecutive stalled checkpoint windows
    for depth_limit in range(0, max_depth + 1):
        stack: list[tuple[int, int, list[B.Action]]] = [(x0, 0, [])]
        while stack:
            x, depth, path = stack.pop()
            if x == target:
                return [p for act in path for p in act.prims], total, True, 0
            d = abs(x - target)
            if d < best:
                best = d
            if depth >= depth_limit:
                continue
            total += 1
            if total > max_budget:
                return None, total, False, best
            # cost-gradient stall check at each crossed checkpoint. Bail only after
            # `patience` CONSECUTIVE stalled windows, so a deep solution that
            # transiently overshoots (|x-target| rises under *2 / *3) survives.
            while ci < len(checkpoints) and total >= checkpoints[ci]:
                if last_ckpt_best - best < min_improve:
                    stalls += 1
                    if stalls >= patience:
                        return None, total, False, best
                else:
                    stalls = 0
                last_ckpt_best = best
                ci += 1
            for act in reversed(actions):
                stack.append((act.apply(x), depth + 1, path + [act]))
    return None, total, False, best


def cost_one(x0, target, actions, mode, prim, cfg) -> tuple[int, int]:
    if mode == "gate":
        nodes, ok = B.solve_with_applicability(
            x0, target, actions, prim, cfg.max_depth, cfg.node_budget, cfg.probe_budget
        )
    elif mode == "depth":
        # depth-aware probe; bail -> primitive fallback (probe nodes are charged)
        _sol, pnodes, ok, _best = probe_macros_depth_aware(
            x0, target, actions, cfg.max_depth, cfg.deep_probe_budget,
            cfg.checkpoints, cfg.probe_min_improve, cfg.probe_patience,
        )
        if ok:
            nodes = pnodes
        else:
            _s, fnodes, ok = B.solve(x0, target, prim, cfg.max_depth, cfg.node_budget)
            nodes = pnodes + fnodes
    else:  # "full"
        _sol, nodes, ok = B.solve(x0, target, actions, cfg.max_depth, cfg.node_budget)
    return nodes, int(ok)


def eval_stream(tasks, actions, mode, prim, cfg) -> tuple[list[int], list[int]]:
    costs, solved = [], []
    for x0, target in tasks:
        nodes, ok = cost_one(x0, target, actions, mode, prim, cfg)
        costs.append(nodes)
        solved.append(ok)
    return costs, solved


# --- summary helpers --------------------------------------------------------
def seg_mean(xs: list[int], lo_frac: float, hi_frac: float) -> float:
    n = len(xs)
    lo, hi = int(n * lo_frac), int(n * hi_frac)
    seg = xs[lo:hi] or xs
    return statistics.mean(seg)


def binned(xs: list[int], bins: int) -> list[float]:
    n = len(xs)
    if n == 0:
        return []
    out = []
    for i in range(bins):
        lo, hi = i * n // bins, (i + 1) * n // bins
        seg = xs[lo:hi] or xs[lo : lo + 1]
        out.append(round(statistics.mean(seg), 1))
    return out


def summary(costs: list[int], solved: list[int]) -> dict:
    return {
        "mean_nodes": round(statistics.mean(costs), 1),
        "median_nodes": round(statistics.median(costs), 1),
        "solve_rate": round(statistics.mean(solved), 4),
    }


def sparkline(vals: list[float]) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    return "".join(blocks[min(8, int((v - lo) / rng * 8))] for v in vals)


# ---------------------------------------------------------------------------
def run_seed(seed: int, cfg: argparse.Namespace) -> dict:
    rng = random.Random(seed)
    tp = train_pool()
    shallow = (cfg.depth_min, cfg.depth_max)
    prim = B.primitive_actions()

    # ---- train stream (shared across all regimes; the library at freeze is the
    #      same no matter which grammar comes next) ----
    train = [gen_task(rng, tp, shallow) for _ in range(cfg.train_tasks)]
    train_keys = {(x0, t) for (x0, t) in train}

    grown = OnlineLibrary(cfg.util_penalty, cfg.util_threshold, cfg.max_macros)
    g_costs, g_solved = [], []
    for step, (x0, target) in enumerate(train):
        nodes, ok = cost_one(x0, target, grown.actions, "gate", prim, cfg)
        g_costs.append(nodes)
        g_solved.append(ok)
        # learn: extract a skill from the solution actually found (full budget,
        # current action set) — exactly build_library's promotion behaviour.
        sol, _, ok2 = B.solve(x0, target, grown.actions, cfg.max_depth, cfg.node_budget)
        if ok2 and sol is not None:
            grown.observe(sol, step)
    grown.freeze()

    # random control replays growing's promotion schedule over the same stream
    rlib = RandomLibrary(dict(grown.schedule), random.Random(seed * 9973 + 1))
    r_costs, r_solved = [], []
    for step, (x0, target) in enumerate(train):
        nodes, ok = cost_one(x0, target, rlib.actions, "gate", prim, cfg)
        r_costs.append(nodes)
        r_solved.append(ok)
        rlib.maybe_add(step)

    # frozen_empty: primitives only over the same stream
    f_costs, f_solved = eval_stream(train, prim, "full", prim, cfg)

    train_arms = {
        "growing_gate": {"costs": g_costs, "solved": g_solved},
        "random_growing": {"costs": r_costs, "solved": r_solved},
        "frozen_empty": {"costs": f_costs, "solved": f_solved},
    }
    train_summary = {
        arm: {
            "early": round(seg_mean(d["costs"], 0.0, 0.25), 1),
            "late": round(seg_mean(d["costs"], 0.75, 1.0), 1),
            **summary(d["costs"], d["solved"]),
            "curve": binned(d["costs"], cfg.bins),
        }
        for arm, d in train_arms.items()
    }

    # ---- test streams (one frozen library, four shifted grammars) ----
    n_macros = len(grown.macro_prims)
    regimes_out: dict[str, dict] = {}
    held_out_all = True
    for name in cfg.regimes:
        pool, depth = regime_grammar(name, cfg)
        test = sample_disjoint(rng, cfg.test_tasks, pool, depth, train_keys)
        held_out_all &= all((x0, t) not in train_keys for (x0, t) in test) and len(test) == cfg.test_tasks

        gc, gs = eval_stream(test, grown.actions, "gate", prim, cfg)
        rc, rs = eval_stream(test, rlib.actions, "gate", prim, cfg)
        fc, fs = eval_stream(test, prim, "full", prim, cfg)
        # Phase 3.2: SAME frozen library, depth-aware gate (only the horizon
        # control differs from growing_gate) — isolates "fix the depth sensor".
        dc, ds = eval_stream(test, grown.actions, "depth", prim, cfg)
        arms = {
            "growing_gate": {"costs": gc, "solved": gs},
            "growing_depth": {"costs": dc, "solved": ds},
            "random_growing": {"costs": rc, "solved": rs},
            "frozen_empty": {"costs": fc, "solved": fs},
        }
        reg = {
            arm: {**summary(d["costs"], d["solved"]), "curve": binned(d["costs"], cfg.bins)}
            for arm, d in arms.items()
        }
        # R2_depth: the reach question is about the LIBRARY's action set, not the
        # cheap probe — so also measure full macro search (the gate's myopic probe
        # cannot verify deep compositions; that gap is itself a finding).
        if name == "R2_depth":
            lc, ls = eval_stream(test, grown.actions, "full", prim, cfg)
            reg["growing_library_full"] = {**summary(lc, ls), "curve": binned(lc, cfg.bins)}
        regimes_out[name] = reg

    return {
        "seed": seed,
        "held_out": held_out_all,
        "n_macros": n_macros,
        "macros": ["".join(mp) for mp in grown.macro_prims],
        "promotion_steps": sorted(grown.schedule),
        "train": train_summary,
        "regimes": regimes_out,
    }


def aggregate(seed_runs: list[dict], cfg: argparse.Namespace) -> dict:
    arms = ("growing_gate", "random_growing", "frozen_empty")

    def amean(vals):
        return round(statistics.mean(vals), 1)

    train_agg = {}
    for arm in arms:
        train_agg[arm] = {
            "early": amean([r["train"][arm]["early"] for r in seed_runs]),
            "late": amean([r["train"][arm]["late"] for r in seed_runs]),
            "mean_nodes": amean([r["train"][arm]["mean_nodes"] for r in seed_runs]),
            "solve_rate": round(statistics.mean([r["train"][arm]["solve_rate"] for r in seed_runs]), 4),
            "curve": [
                round(statistics.mean([r["train"][arm]["curve"][i] for r in seed_runs]), 1)
                for i in range(cfg.bins)
            ],
        }
    train_verdict = {
        # cost falls with experience vs the no-compression baseline
        "compression_pays_in_stream": train_agg["growing_gate"]["late"] < train_agg["frozen_empty"]["late"],
        # and the fall is reuse-specific, not an artifact of a bigger action set
        "fall_is_reuse_specific": train_agg["growing_gate"]["late"] < train_agg["random_growing"]["late"],
        # the curve itself declines (late < early) for the protagonist
        "curve_declines": train_agg["growing_gate"]["late"] < train_agg["growing_gate"]["early"],
    }

    regimes_agg = {}
    for name in cfg.regimes:
        ra = {}
        keys = list(seed_runs[0]["regimes"][name].keys())
        for arm in keys:
            ra[arm] = {
                "mean_nodes": amean([r["regimes"][name][arm]["mean_nodes"] for r in seed_runs]),
                "median_nodes": amean([r["regimes"][name][arm]["median_nodes"] for r in seed_runs]),
                "solve_rate": round(statistics.mean([r["regimes"][name][arm]["solve_rate"] for r in seed_runs]), 4),
            }
        g, f, rnd = ra["growing_gate"], ra["frozen_empty"], ra["random_growing"]
        # Two distinct claims, kept separate on purpose:
        #  - beats_no_skill: cheaper than the no-compression baseline. In an
        #    arithmetic DSL this is TRUE even for a disjoint grammar, because a
        #    macro is a generically useful multi-step JUMP (the RANDMACRO effect).
        #    So this alone does NOT prove grammar transfer.
        #  - grammar_specific_gain: cheaper than a count/length-matched RANDOM
        #    library. THIS is the transfer signal: it isolates "learned structure
        #    still fits the shifted grammar" from "just having longer jumps".
        #    Its MARGIN (random - growing) decays as the grammar diverges.
        margin = round(rnd["mean_nodes"] - g["mean_nodes"], 1)
        verdict = {
            "beats_no_skill": g["mean_nodes"] < f["mean_nodes"],
            "grammar_specific_gain": g["mean_nodes"] < rnd["mean_nodes"],
            "grammar_specific_margin": margin,
            # even when skills don't fit, the gate prevents a blow-up (no harm)
            "gate_prevents_collapse": g["mean_nodes"] <= f["mean_nodes"] * (1.0 + cfg.collapse_eps),
        }
        # Phase 3.2 — does fixing the gate's depth sensor unlock the library?
        if "growing_depth" in ra:
            dep = ra["growing_depth"]
            verdict["depth_aware_mean"] = dep["mean_nodes"]
            # helps where the cheap probe was myopic (esp. R2_depth)
            verdict["depth_aware_helps"] = dep["mean_nodes"] < g["mean_nodes"]
            verdict["depth_aware_speedup_vs_cheap"] = (
                round(g["mean_nodes"] / dep["mean_nodes"], 2) if dep["mean_nodes"] else None
            )
            # must not regress shallow regimes nor break collapse-prevention
            verdict["depth_aware_no_regression"] = dep["mean_nodes"] <= g["mean_nodes"] * (1.0 + cfg.collapse_eps)
            verdict["depth_aware_no_collapse"] = dep["mean_nodes"] <= f["mean_nodes"] * (1.0 + cfg.collapse_eps)
        if name == "R2_depth" and "growing_library_full" in ra:
            # honesty: with --max-depth high enough, primitives solve deep tasks
            # too (expensively), so a solve-RATE reach gap need not appear; the
            # informative quantity is the cost gap of the full library vs prims.
            lib = ra["growing_library_full"]
            verdict["reach_solverate"] = lib["solve_rate"] > f["solve_rate"]
            verdict["library_full_cost_vs_prim"] = round(lib["mean_nodes"] - f["mean_nodes"], 1)
        regimes_agg[name] = {"arms": ra, "verdict": verdict}

    return {"train": {"arms": train_agg, "verdict": train_verdict}, "regimes": regimes_agg}


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
    ap.add_argument("--max-depth", type=int, default=14,
                    help="action-step depth cap; raised so deep R2_depth tasks are "
                         "solvable-in-principle by primitives (expensive), not walled off.")
    ap.add_argument("--node-budget", type=int, default=2_000_000)
    ap.add_argument("--probe-budget", type=int, default=40,
                    help="cheap applicability gate: fixed macro-probe cap before fallback.")
    ap.add_argument("--deep-probe-budget", type=int, default=4000,
                    help="Phase 3.2 depth-aware gate: max macro-probe budget (adaptive horizon). "
                         "Only consumed where the cost gradient stays positive (deep reuse); "
                         "foreign/shallow tasks bail at the checkpoints regardless of this cap.")
    ap.add_argument("--checkpoints", default="40,200,700",
                    help="Phase 3.2: node counts at which the depth-aware probe checks the cost gradient.")
    ap.add_argument("--probe-min-improve", type=int, default=1,
                    help="Phase 3.2: min best-distance improvement per checkpoint window, else stall.")
    ap.add_argument("--probe-patience", type=int, default=2,
                    help="Phase 3.2: bail only after this many CONSECUTIVE stalled checkpoint windows.")
    ap.add_argument("--util-penalty", type=float, default=1.0)
    ap.add_argument("--util-threshold", type=float, default=2.0)
    ap.add_argument("--max-macros", type=int, default=8)
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--collapse-eps", type=float, default=0.10,
                    help="tolerance for 'gate_prevents_collapse' (no OOD blow-up).")
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--output-json", default="artifacts/skill_compression_stream.json")
    args = ap.parse_args()
    args.regimes = [r for r in args.regimes.split(",") if r]
    args.checkpoints = tuple(int(c) for c in str(args.checkpoints).split(",") if c)

    seeds = [int(s) for s in args.seeds.split(",")]
    seed_runs = [run_seed(s, args) for s in seeds]
    agg = aggregate(seed_runs, args)

    held_out = all(r["held_out"] for r in seed_runs)
    protocol_evidence = {
        "held_out": held_out,
        "online_promotion": True,
        "applicability_gate_present": True,
        "frozen_empty_control_present": True,
        "random_growing_control_present": True,
        "grammar_shift_axis": args.regimes,
        "n_seeds": len(seeds),
        "parameter_matched": "identical DSL + solver + UTIL gate across arms",
        "valid": held_out and len(seeds) >= 3,
    }
    payload = {
        "config": vars(args),
        "protocol_evidence": protocol_evidence,
        "aggregate": agg,
        "seed_runs": seed_runs,
    }

    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ta = agg["train"]["arms"]
    tv = agg["train"]["verdict"]
    print("=== Skill-Compression Phase 3.1 — self-improvement stream ===")
    print(f"seeds={seeds}  protocol_valid={protocol_evidence['valid']}  "
          f"macros_grown≈{round(statistics.mean([r['n_macros'] for r in seed_runs]),1)}")
    print("\nTRAIN stream — mean nodes/task (early 25% -> late 25%):")
    for arm in ("growing_gate", "random_growing", "frozen_empty"):
        a = ta[arm]
        print(f"  {arm:15s}: {a['early']:7.1f} -> {a['late']:7.1f}   curve {sparkline(a['curve'])}")
    print(f"  VERDICT: compression_pays_in_stream={tv['compression_pays_in_stream']}  "
          f"fall_is_reuse_specific={tv['fall_is_reuse_specific']}  "
          f"curve_declines={tv['curve_declines']}")

    print("\nFREEZE -> grammar shift — TEST mean nodes/task "
          "(cheap-gate / DEPTH-gate / random / frozen):")
    for name in args.regimes:
        r = agg["regimes"][name]
        v = r["verdict"]
        g = r["arms"]["growing_gate"]["mean_nodes"]
        dep = r["arms"]["growing_depth"]["mean_nodes"]
        rnd = r["arms"]["random_growing"]["mean_nodes"]
        f = r["arms"]["frozen_empty"]["mean_nodes"]
        extra = ""
        if "library_full_cost_vs_prim" in v:
            extra = f"  [lib_full={r['arms']['growing_library_full']['mean_nodes']:.0f}]"
        print(f"  {name:13s}: {g:8.1f} / {dep:8.1f} / {rnd:8.1f} / {f:8.1f}  "
              f"depth_speedup×{v.get('depth_aware_speedup_vs_cheap')}  "
              f"no_regress={str(v.get('depth_aware_no_regression')):5s} "
              f"grammar_margin={v['grammar_specific_margin']:6.1f}{extra}")
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
