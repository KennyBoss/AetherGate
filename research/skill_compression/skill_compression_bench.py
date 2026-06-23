#!/usr/bin/env python3
"""Phase 1 crash test — does compressing experience into skills lower cost?

Self-contained, CPU-only, no JAX. Implements the one make-or-break A/B from
research/ROADMAP_SKILL_COMPRESSION.md:

    Promotion ON vs OFF, parameter-matched (same DSL, same search), held-out
    eval, multi-seed. Cost metric = search nodes expanded to solve a task.

Benchmark — program synthesis with reusable subroutines
-------------------------------------------------------
* State is a single integer. A DSL of `P` primitive int->int ops.
* A *hidden* library of `S` subroutines (short op sequences). Each task is built
  by composing subroutines, so task solutions share recurring sub-sequences.
* A task = (x0, target). Solve = find ANY op sequence (<= max_depth action-steps)
  taking x0 to target, via deterministic iterative-deepening DFS that returns the
  shortest, lexicographically-first solution. Cost = nodes expanded.

The three arms (parameter-matched: identical solver, identical DSL)
------------------------------------------------------------------
* OFF  : action set = primitives only, frozen. (no-compression baseline)
* FREQ : promote a sub-sequence once it recurs >= K times in solved solutions
         (pure frequency = the "success_rate" heuristic, our own baseline gate).
* UTIL : promote by scored importance g_t:
            utility = reuse_count * (len-1)  -  penalty * library_size
         i.e. (steps saved per use x reuse) minus library cost. Beating FREQ is
         "the first bar" from the roadmap.

Honest controls
---------------
* held_out: eval tasks are explicitly RESAMPLED to be disjoint from train on
  (x0,target), same hidden subroutines (reuse exists) -> compression *can* pay.
* RANDMACRO control (the clean isolation): same NUMBER and same LENGTHS of macros
  as UTIL, but contents are RANDOM primitive sequences. A macro that is merely a
  multi-step jump shortens any depth-limited search regardless of reuse. If UTIL
  beats RANDMACRO on reuse-eval, the *content* of compression matters (genuine
  reuse capture), not just "having longer jumps." This is the decisive control.
* no_reuse eval: targets that are random reachable values NOT composed from the
  subroutines, to observe how much speedup is generic depth-reduction vs reuse.

A negative result (OFF matches the gates, or gates don't beat FREQ) is a
SUCCESSFUL crash test, recorded honestly. Output is JSON with protocol_evidence.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# DSL: primitive integer operations (the atoms)
# ---------------------------------------------------------------------------
PRIMITIVES: dict[str, Callable[[int], int]] = {
    "a": lambda x: x + 1,
    "b": lambda x: x + 3,
    "c": lambda x: x * 2,
    "d": lambda x: x - 1,
    "e": lambda x: x * 3,
}
PRIM_NAMES = sorted(PRIMITIVES)  # deterministic action order

# Hidden subroutines used to GENERATE tasks (the search never sees these).
HIDDEN_SUBROUTINES: dict[str, list[str]] = {
    "S1": ["c", "a", "c"],
    "S2": ["b", "b", "d"],
    "S3": ["e", "d", "d"],
    "S4": ["a", "c", "b"],
}


def apply_prims(x: int, prims: list[str]) -> int:
    for p in prims:
        x = PRIMITIVES[p](x)
    return x


# ---------------------------------------------------------------------------
# Action = a named callable over the integer state, with its primitive
# expansion (length = how many primitive steps it covers). Macros are actions
# whose expansion is longer than 1.
# ---------------------------------------------------------------------------
class Action:
    __slots__ = ("name", "prims")

    def __init__(self, name: str, prims: list[str]):
        self.name = name
        self.prims = prims  # primitive-grounded expansion

    def apply(self, x: int) -> int:
        return apply_prims(x, self.prims)


def primitive_actions() -> list[Action]:
    return [Action(p, [p]) for p in PRIM_NAMES]


# ---------------------------------------------------------------------------
# Deterministic iterative-deepening DFS. Returns (solution_prims, nodes, solved).
# Action order is fixed -> shortest, lexicographically-first solution. Cost is
# total node expansions across deepening rounds (the budget the agent spends).
# ---------------------------------------------------------------------------
def solve(
    x0: int,
    target: int,
    actions: list[Action],
    max_depth: int,
    node_budget: int,
) -> tuple[list[str] | None, int, bool]:
    total_nodes = 0
    for depth_limit in range(0, max_depth + 1):
        stack: list[tuple[int, int, list[Action]]] = [(x0, 0, [])]
        while stack:
            x, depth, path = stack.pop()
            if x == target:
                prims = [p for act in path for p in act.prims]
                return prims, total_nodes, True
            if depth >= depth_limit:
                continue
            total_nodes += 1
            if total_nodes > node_budget:
                return None, total_nodes, False
            # push in reverse so PRIM_NAMES order is explored first (determinism)
            for act in reversed(actions):
                stack.append((act.apply(x), depth + 1, path + [act]))
    return None, total_nodes, False


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------
import random


def make_reuse_task(rng: random.Random) -> tuple[int, int, list[str]]:
    """Task whose solution is a composition of hidden subroutines."""
    n_sub = rng.randint(1, 2)
    prog: list[str] = []
    for _ in range(n_sub):
        prog += HIDDEN_SUBROUTINES[rng.choice(list(HIDDEN_SUBROUTINES))]
    x0 = rng.randint(0, 5)
    return x0, apply_prims(x0, prog), prog


def make_no_reuse_task(rng: random.Random, max_depth: int) -> tuple[int, int]:
    """Control task: target = random reachable value via random PRIMITIVES,
    NOT aligned to the subroutine library."""
    length = rng.randint(3, max_depth)
    prog = [rng.choice(PRIM_NAMES) for _ in range(length)]
    x0 = rng.randint(0, 5)
    return x0, apply_prims(x0, prog)


def sample_disjoint_reuse(
    rng: random.Random, n: int, train_keys: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Held-out eval: reuse-structured tasks whose (x0,target) is NOT in train."""
    out: list[tuple[int, int]] = []
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        x0, target, _g = make_reuse_task(rng)
        if (x0, target) not in train_keys:
            out.append((x0, target))
    return out


def random_macros_like(
    rng: random.Random, learned: list[Action]
) -> list[Action]:
    """RANDMACRO control: same count & lengths as the learned macros, random
    contents. Isolates 'content of compression' from 'having longer jumps'."""
    macros = [a for a in learned if a.name.startswith("M")]
    out: list[Action] = []
    for i, m in enumerate(macros):
        prims = [rng.choice(PRIM_NAMES) for _ in range(len(m.prims))]
        out.append(Action(f"R{i}", prims))
    return primitive_actions() + out


# ---------------------------------------------------------------------------
# Promotion: extract recurring primitive sub-sequences from solved solutions.
# ---------------------------------------------------------------------------
def candidate_subseqs(prims: list[str], lo: int = 2, hi: int = 4) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    for L in range(lo, hi + 1):
        for i in range(0, len(prims) - L + 1):
            out.add(tuple(prims[i : i + L]))
    return out


def build_library(
    train_tasks: list[tuple[int, int, list[str]]],
    gate: str,
    max_depth: int,
    node_budget: int,
    freq_k: int,
    util_penalty: float,
    util_threshold: float,
    max_macros: int,
) -> tuple[list[Action], dict]:
    """Process the train stream, solve each task, and promote macros per the
    chosen gate. Returns the frozen action set + promotion log."""
    actions = primitive_actions()
    macro_prims: list[list[str]] = []
    counts: Counter[tuple[str, ...]] = Counter()
    promoted: set[tuple[str, ...]] = set()
    log = {"promotions": [], "train_solved": 0, "train_nodes": 0}

    for x0, target, _gen in train_tasks:
        sol, nodes, solved = solve(x0, target, actions, max_depth, node_budget)
        log["train_nodes"] += nodes
        if not solved or sol is None:
            continue
        log["train_solved"] += 1
        for sub in candidate_subseqs(sol):
            counts[sub] += 1

        if gate == "off":
            continue
        if len(macro_prims) >= max_macros:
            continue

        # Decide a single best promotion candidate this step (the gate g_t).
        # Iterate in sorted order so ties break deterministically regardless of
        # PYTHONHASHSEED (set/dict string-hash randomization) -> reproducible.
        best, best_score = None, 0.0
        for sub, cnt in sorted(counts.items()):
            if sub in promoted or cnt < 2:
                continue
            if gate == "freq":
                score = float(cnt) if cnt >= freq_k else 0.0
            elif gate == "util":
                # utility = reuse * steps_saved_per_use - penalty * lib_size
                score = cnt * (len(sub) - 1) - util_penalty * len(macro_prims)
            else:
                raise ValueError(gate)
            if score > best_score:
                best, best_score = sub, score

        thr = 0.0 if gate == "freq" else util_threshold
        if best is not None and best_score > thr:
            promoted.add(best)
            macro_prims.append(list(best))
            actions = primitive_actions() + [
                Action(f"M{i}", mp) for i, mp in enumerate(macro_prims)
            ]
            log["promotions"].append(
                {"macro": "".join(best), "count": counts[best], "score": round(best_score, 2)}
            )
    return actions, log


# ---------------------------------------------------------------------------
# Evaluation on a frozen action set (held-out; no promotion during eval)
# ---------------------------------------------------------------------------
def evaluate(
    tasks: list[tuple[int, int]],
    actions: list[Action],
    max_depth: int,
    node_budget: int,
) -> dict:
    nodes_list, solved = [], 0
    for x0, target in tasks:
        _sol, nodes, ok = solve(x0, target, actions, max_depth, node_budget)
        nodes_list.append(nodes)
        solved += int(ok)
    n = len(tasks)
    return {
        "n": n,
        "solve_rate": solved / n,
        "mean_nodes": statistics.mean(nodes_list),
        "median_nodes": statistics.median(nodes_list),
    }


def solve_with_applicability(
    x0: int,
    target: int,
    macro_actions: list[Action],
    prim_actions: list[Action],
    max_depth: int,
    node_budget: int,
    probe_budget: int,
) -> tuple[int, bool]:
    """Applicability gate (Phase 1.5). Probe the skill library under a SMALL
    budget; if the macros pay off the task is solved cheaply (reuse case). If the
    probe fails, the library is judged inapplicable to this task and we fall back
    to primitives only — paying the probe overhead but NOT the inflated-branching
    penalty of carrying foreign macros through a deep search.

    No oracle label: the decision is made purely from search behaviour, so it
    works on a mixed stream of in-distribution and out-of-distribution tasks.
    """
    _sol, probe_nodes, ok = solve(x0, target, macro_actions, max_depth, probe_budget)
    if ok:
        return probe_nodes, True
    _sol, fb_nodes, ok2 = solve(x0, target, prim_actions, max_depth, node_budget)
    return probe_nodes + fb_nodes, ok2


def evaluate_applicability(
    tasks: list[tuple[int, int]],
    macro_actions: list[Action],
    prim_actions: list[Action],
    max_depth: int,
    node_budget: int,
    probe_budget: int,
) -> dict:
    nodes_list, solved = [], 0
    for x0, target in tasks:
        nodes, ok = solve_with_applicability(
            x0, target, macro_actions, prim_actions, max_depth, node_budget, probe_budget
        )
        nodes_list.append(nodes)
        solved += int(ok)
    n = len(tasks)
    return {
        "n": n,
        "solve_rate": solved / n,
        "mean_nodes": statistics.mean(nodes_list),
        "median_nodes": statistics.median(nodes_list),
    }


# ---------------------------------------------------------------------------
GATES = ("off", "freq", "util", "randmacro")
ALL_ARMS = GATES + ("util_app",)


def run_seed(seed: int, cfg: argparse.Namespace) -> dict:
    rng = random.Random(seed)
    train = [make_reuse_task(rng) for _ in range(cfg.train_tasks)]
    train_keys = {(x0, t) for (x0, t, _g) in train}

    # held-out eval explicitly disjoint from train on (x0,target)
    eval_reuse = sample_disjoint_reuse(rng, cfg.eval_tasks, train_keys)
    eval_noreuse = [make_no_reuse_task(rng, cfg.max_depth) for _ in range(cfg.eval_tasks)]
    held_out = all((x0, t) not in train_keys for (x0, t) in eval_reuse) and len(
        eval_reuse
    ) == cfg.eval_tasks

    results = {}
    # learn the UTIL library first so RANDMACRO can match its count/lengths
    util_actions, util_log = build_library(
        train, "util", cfg.max_depth, cfg.node_budget,
        cfg.freq_k, cfg.util_penalty, cfg.util_threshold, cfg.max_macros,
    )
    rand_actions = random_macros_like(rng, util_actions)

    for gate in GATES:
        if gate == "util":
            actions, log = util_actions, util_log
        elif gate == "randmacro":
            actions, log = rand_actions, {"promotions": []}
        else:
            actions, log = build_library(
                train, gate, cfg.max_depth, cfg.node_budget,
                cfg.freq_k, cfg.util_penalty, cfg.util_threshold, cfg.max_macros,
            )
        results[gate] = {
            "n_macros": sum(1 for a in actions if a.name[0] in ("M", "R")),
            "macros": [a.name + "=" + "".join(a.prims) for a in actions if a.name[0] in ("M", "R")],
            "eval_reuse": evaluate(eval_reuse, actions, cfg.max_depth, cfg.node_budget),
            "eval_no_reuse": evaluate(eval_noreuse, actions, cfg.max_depth, cfg.node_budget),
            "promotions": log["promotions"],
        }

    # Phase 1.5: applicability-gated UTIL — same learned library, but the gate
    # probes cheaply and falls back to primitives when the skills don't fit.
    prim_actions = primitive_actions()
    results["util_app"] = {
        "n_macros": results["util"]["n_macros"],
        "macros": results["util"]["macros"],
        "probe_budget": cfg.probe_budget,
        "eval_reuse": evaluate_applicability(
            eval_reuse, util_actions, prim_actions, cfg.max_depth, cfg.node_budget, cfg.probe_budget
        ),
        "eval_no_reuse": evaluate_applicability(
            eval_noreuse, util_actions, prim_actions, cfg.max_depth, cfg.node_budget, cfg.probe_budget
        ),
        "promotions": util_log["promotions"],
    }
    return {"seed": seed, "held_out": held_out, "gates": results}


def aggregate(seed_runs: list[dict]) -> dict:
    out = {}
    for gate in ALL_ARMS:
        for split in ("eval_reuse", "eval_no_reuse"):
            nodes = [r["gates"][gate][split]["mean_nodes"] for r in seed_runs]
            solve = [r["gates"][gate][split]["solve_rate"] for r in seed_runs]
            out[f"{gate}.{split}.mean_nodes"] = round(statistics.mean(nodes), 1)
            out[f"{gate}.{split}.solve_rate"] = round(statistics.mean(solve), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23")
    ap.add_argument("--train-tasks", type=int, default=120)
    ap.add_argument("--eval-tasks", type=int, default=80)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--node-budget", type=int, default=2_000_000)
    ap.add_argument("--freq-k", type=int, default=3)
    ap.add_argument("--util-penalty", type=float, default=1.0)
    ap.add_argument("--util-threshold", type=float, default=2.0)
    ap.add_argument("--max-macros", type=int, default=8)
    ap.add_argument("--probe-budget", type=int, default=40,
                    help="Phase 1.5 applicability gate: macro-probe node cap before fallback.")
    ap.add_argument("--output-json", default="artifacts/skill_compression_summary.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    seed_runs = [run_seed(s, args) for s in seeds]
    agg = aggregate(seed_runs)

    off_reuse = agg["off.eval_reuse.mean_nodes"]
    off_noreuse = agg["off.eval_no_reuse.mean_nodes"]
    util_reuse = agg["util.eval_reuse.mean_nodes"]
    util_noreuse = agg["util.eval_no_reuse.mean_nodes"]
    freq_reuse = agg["freq.eval_reuse.mean_nodes"]
    rand_reuse = agg["randmacro.eval_reuse.mean_nodes"]
    app_reuse = agg["util_app.eval_reuse.mean_nodes"]
    app_noreuse = agg["util_app.eval_no_reuse.mean_nodes"]
    util_speedup = round(off_reuse / util_reuse, 2) if util_reuse else None
    app_speedup = round(off_reuse / app_reuse, 2) if app_reuse else None
    util_beats_freq = util_reuse < freq_reuse
    util_beats_randmacro = util_reuse < rand_reuse
    rand_speedup = round(off_reuse / rand_reuse, 2) if rand_reuse else None

    protocol_evidence = {
        "held_out": all(r["held_out"] for r in seed_runs),
        "subroutine_reuse_present": True,
        "randmacro_control_present": True,
        "no_reuse_eval_present": True,
        "applicability_gate_present": True,
        "n_seeds": len(seeds),
        "parameter_matched": "identical DSL + solver across arms",
        "valid": all(r["held_out"] for r in seed_runs) and len(seeds) >= 3,
    }
    # Applicability gate target: keep the reuse speedup, kill the OOD penalty
    # (bring no-reuse cost back toward the OFF baseline instead of UTIL's blowup).
    app_keeps_reuse_gain = app_reuse < off_reuse
    app_cuts_ood_penalty = app_noreuse < util_noreuse
    verdict = {
        "off_reuse_mean_nodes": off_reuse,
        "util_reuse_mean_nodes": util_reuse,
        "randmacro_reuse_mean_nodes": rand_reuse,
        "util_app_reuse_mean_nodes": app_reuse,
        "util_speedup_vs_off": util_speedup,
        "util_app_speedup_vs_off": app_speedup,
        "randmacro_speedup_vs_off": rand_speedup,
        "util_beats_freq": util_beats_freq,
        # decisive: learned content beats random macros of matched count/length
        "util_beats_randmacro": util_beats_randmacro,
        "compression_pays_on_reuse": util_reuse < off_reuse,
        "gain_is_reuse_specific_not_just_longer_jumps": util_beats_randmacro,
        # Phase 1.5 — applicability gate
        "ood_penalty_off_noreuse": off_noreuse,
        "ood_penalty_util_noreuse": util_noreuse,
        "ood_penalty_util_app_noreuse": app_noreuse,
        "app_keeps_reuse_gain": app_keeps_reuse_gain,
        "app_cuts_ood_penalty": app_cuts_ood_penalty,
    }
    payload = {
        "config": vars(args),
        "protocol_evidence": protocol_evidence,
        "aggregate": agg,
        "verdict": verdict,
        "seed_runs": seed_runs,
    }

    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Skill-Compression Phase 1 / 1.5 crash test ===")
    print(f"seeds={seeds}  protocol_valid={protocol_evidence['valid']}")
    print(f"  Mean search nodes to solve (held-out) — REUSE / NO-REUSE(OOD):")
    print(f"    OFF      (no compression)      : {off_reuse:7.1f} / {off_noreuse:7.1f}")
    print(f"    FREQ     (frequency heuristic) : {freq_reuse:7.1f} / {agg['freq.eval_no_reuse.mean_nodes']:7.1f}")
    print(f"    RANDMACRO(matched random jumps): {rand_reuse:7.1f} / {agg['randmacro.eval_no_reuse.mean_nodes']:7.1f}")
    print(f"    UTIL     (scored gate g_t)     : {util_reuse:7.1f} / {util_noreuse:7.1f}   (OOD penalty!)")
    print(f"    UTIL+APP (applicability gate)  : {app_reuse:7.1f} / {app_noreuse:7.1f}   speedup vs OFF = {app_speedup}x")
    print(f"  VERDICT:")
    print(f"    compression_pays_on_reuse                 = {verdict['compression_pays_on_reuse']}")
    print(f"    util_beats_randmacro (reuse-specific gain)= {util_beats_randmacro}")
    print(f"    util_beats_freq (beat own heuristic)      = {util_beats_freq}   (open)")
    print(f"    [P1.5] app_keeps_reuse_gain               = {app_keeps_reuse_gain}")
    print(f"    [P1.5] app_cuts_ood_penalty               = {app_cuts_ood_penalty}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
