#!/usr/bin/env python3
"""ECS Stage 3: observable recurrent state-porting falsification.

The protocol is frozen in:
docs/superpowers/specs/2026-07-13-ecs-stage3-state-porting-design.md
"""
from __future__ import annotations

import argparse
import heapq
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import skill_compression_bench as B
from ecs_stage1 import D, P, PRIM, features, softmax
from ecs_stage2_fastweight import HIDDEN, NK, ParamSkillPolicy, best_first_param, make_training


@dataclass(frozen=True)
class Task:
    task_id: str
    x0: int
    target: int
    plan: tuple[int, ...]

    @property
    def key(self) -> tuple[int, int, tuple[int, ...]]:
        return self.x0, self.target, self.plan


def make_tasks(seed: int, n: int, depth: int, starts: range) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    for index in range(n):
        plan = tuple(rng.randrange(NK) for _ in range(depth))
        x0 = rng.choice(starts)
        x = x0
        for skill_id in plan:
            for primitive in HIDDEN[skill_id]:
                x = B.PRIMITIVES[primitive](x)
        tasks.append(Task(f"s{seed}-d{depth}-n{index}", x0, x, plan))
    return tasks


def sequence(task: Task) -> list[tuple[int, int]]:
    return [
        (skill_id, PRIM.index(primitive))
        for skill_id in task.plan
        for primitive in HIDDEN[skill_id]
    ]


class StatePortPolicy:
    """Matched policy for ported, reset, and dense-context arms.

    The recurrent transition is a frozen reservoir. Training learns how to use
    its observable path summary, without BPTT or any hidden plan signal.
    """

    def __init__(self, hidden: int, context: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        scale = 0.35
        self.W1 = rng.normal(0, scale, (hidden, D))
        self.b1 = np.zeros(hidden)
        self.Wp = rng.normal(0, scale, (P, hidden))
        self.bp = np.zeros(P)
        self.Wg = rng.normal(0, scale, (NK, hidden))
        self.bg = np.zeros(NK)
        self.Delta = rng.normal(0, 0.05, (NK, P))
        self.U0 = rng.normal(0, 0.05, (P, context))
        self.U = rng.normal(0, 0.05, (NK, P, context))
        self.A = rng.normal(0, 0.15, (context, context)).astype(np.float32)
        self.Bc = rng.normal(0, 0.15, (context, hidden)).astype(np.float32)
        self.E = rng.normal(0, 0.15, (context, P)).astype(np.float32)
        self.bc = np.zeros(context, dtype=np.float32)
        self.hidden = hidden
        self.context = context
        self._init_adam()

    def _trainable(self) -> dict[str, np.ndarray]:
        return {
            "W1": self.W1,
            "b1": self.b1,
            "Wp": self.Wp,
            "bp": self.bp,
            "Wg": self.Wg,
            "bg": self.bg,
            "Delta": self.Delta,
            "U0": self.U0,
            "U": self.U,
        }

    def _init_adam(self) -> None:
        self.m = {name: np.zeros_like(value) for name, value in self._trainable().items()}
        self.v = {name: np.zeros_like(value) for name, value in self._trainable().items()}
        self.t = 0

    def zero_context(self) -> np.ndarray:
        return np.zeros(self.context, dtype=np.float32)

    def base(self, inputs: np.ndarray) -> np.ndarray:
        return np.tanh(self.W1 @ inputs + self.b1)

    def gate(self, hidden: np.ndarray) -> np.ndarray:
        return softmax(self.Wg @ hidden + self.bg)

    def logits(
        self,
        hidden: np.ndarray,
        context: np.ndarray,
        use_skill: bool,
    ) -> tuple[np.ndarray, int]:
        gate = self.gate(hidden)
        skill_id = int(np.argmax(gate))
        shared = self.U0 @ context
        skill = self.Delta[skill_id] + self.U[skill_id] @ context
        # Compute both paths in every arm; masking changes information, not cost.
        return self.Wp @ hidden + self.bp + shared + float(use_skill) * skill, skill_id

    def next_context(
        self,
        context: np.ndarray,
        hidden: np.ndarray,
        primitive_index: int,
    ) -> np.ndarray:
        action = np.zeros(P, dtype=np.float32)
        action[primitive_index] = 1.0
        value = self.A @ context + self.Bc @ hidden.astype(np.float32) + self.E @ action + self.bc
        return np.tanh(value).astype(np.float32)

    def train_batch(self, tasks: list[Task], lr: float, use_skill: bool) -> None:
        gradients = {name: np.zeros_like(value) for name, value in self._trainable().items()}
        examples = 0
        for task in tasks:
            x = task.x0
            context = self.zero_context()
            for skill_id, primitive_index in sequence(task):
                inputs = features(x, task.target)
                hidden = self.base(inputs)
                gate = self.gate(hidden)
                logits = self.Wp @ hidden + self.bp + self.U0 @ context
                if use_skill:
                    logits = logits + self.Delta[skill_id] + self.U[skill_id] @ context

                dz = softmax(logits)
                dz[primitive_index] -= 1.0
                gradients["Wp"] += np.outer(dz, hidden)
                gradients["bp"] += dz
                gradients["U0"] += np.outer(dz, context)
                dh = self.Wp.T @ dz

                if use_skill:
                    gradients["Delta"][skill_id] += dz
                    gradients["U"][skill_id] += np.outer(dz, context)
                    dg = gate.copy()
                    dg[skill_id] -= 1.0
                    gradients["Wg"] += np.outer(dg, hidden)
                    gradients["bg"] += dg
                    dh += self.Wg.T @ dg

                dpre = dh * (1.0 - hidden * hidden)
                gradients["W1"] += np.outer(dpre, inputs)
                gradients["b1"] += dpre
                context = self.next_context(context, hidden, primitive_index)
                x = B.PRIMITIVES[PRIM[primitive_index]](x)
                examples += 1

        self._adam({name: value / max(1, examples) for name, value in gradients.items()}, lr)

    def _adam(
        self,
        gradients: dict[str, np.ndarray],
        lr: float,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.t += 1
        params = self._trainable()
        for name, parameter in params.items():
            gradient = gradients[name]
            self.m[name] = b1 * self.m[name] + (1.0 - b1) * gradient
            self.v[name] = b2 * self.v[name] + (1.0 - b2) * gradient * gradient
            mhat = self.m[name] / (1.0 - b1**self.t)
            vhat = self.v[name] / (1.0 - b2**self.t)
            parameter -= lr * mhat / (np.sqrt(vhat) + eps)

    @property
    def parameter_count(self) -> int:
        arrays = list(self._trainable().values()) + [self.A, self.Bc, self.E, self.bc]
        return sum(array.size for array in arrays)

    @property
    def forward_macs(self) -> int:
        # Includes all five child-context updates and the masked skill path.
        return (
            self.hidden * D
            + NK * self.hidden
            + P * self.hidden
            + 2 * P * self.context
            + P * self.context * (self.context + self.hidden + P)
        )

    def finite(self) -> bool:
        arrays = list(self._trainable().values()) + [self.A, self.Bc, self.E, self.bc]
        return all(np.isfinite(array).all() for array in arrays)


def context_key(state: int, context: np.ndarray) -> tuple[int, bytes]:
    return state, np.asarray(context, dtype=np.float32).tobytes()


def best_first(
    policy: StatePortPolicy,
    task: Task,
    node_budget: int,
    max_depth: int,
    *,
    use_skill: bool,
    reset_context: bool,
) -> tuple[int, bool, bool]:
    if task.x0 == task.target:
        return 0, True, False
    zero = policy.zero_context()
    queue: list[tuple[float, int, int, int, np.ndarray]] = [(0.0, 0, task.x0, 0, zero)]
    best_cost = {context_key(task.x0, zero): 0.0}
    counter = 1
    forwards = 0
    context_changed = False

    while queue and forwards < node_budget:
        negative_logp, _, state, depth, carried = heapq.heappop(queue)
        if state == task.target:
            return forwards, True, context_changed
        if depth >= max_depth:
            continue
        key = context_key(state, carried)
        if negative_logp > best_cost.get(key, float("inf")) + 1e-9:
            continue

        decision_context = zero if reset_context else carried
        hidden = policy.base(features(state, task.target))
        logits, _ = policy.logits(hidden, decision_context, use_skill=use_skill)
        log_probabilities = np.log(softmax(logits) + 1e-12)
        forwards += 1

        for primitive_index, primitive in enumerate(PRIM):
            child = B.PRIMITIVES[primitive](state)
            child_context = (
                zero
                if reset_context
                else policy.next_context(decision_context, hidden, primitive_index)
            )
            context_changed = context_changed or bool(np.any(child_context != 0.0))
            child_key = context_key(child, child_context)
            child_cost = negative_logp - float(log_probabilities[primitive_index])
            if best_cost.get(child_key, float("inf")) <= child_cost + 1e-9:
                continue
            best_cost[child_key] = child_cost
            heapq.heappush(
                queue,
                (child_cost, counter, child, depth + 1, child_context.copy()),
            )
            counter += 1

    if any(state == task.target for _, _, state, _, _ in queue):
        return forwards, True, context_changed
    return forwards, False, context_changed


def evaluate(
    policy: StatePortPolicy,
    tasks: list[Task],
    node_budget: int,
    max_depth: int,
    *,
    use_skill: bool,
    reset_context: bool,
) -> dict[str, object]:
    forwards: list[int] = []
    solved = 0
    changed = False
    for task in tasks:
        cost, ok, task_changed = best_first(
            policy,
            task,
            node_budget,
            max_depth,
            use_skill=use_skill,
            reset_context=reset_context,
        )
        forwards.append(cost)
        solved += int(ok)
        changed = changed or task_changed
    return {
        "mean_forwards": statistics.mean(forwards),
        "solve_rate": solved / len(tasks),
        "context_changed": changed,
    }


def evaluate_local(
    policy: ParamSkillPolicy,
    tasks: list[Task],
    node_budget: int,
    max_depth: int,
) -> dict[str, object]:
    forwards: list[int] = []
    solved = 0
    for task in tasks:
        cost, ok, _ = best_first_param(
            policy,
            task.x0,
            task.target,
            node_budget,
            max_depth,
            use_skill=True,
        )
        forwards.append(cost)
        solved += int(ok)
    return {
        "mean_forwards": statistics.mean(forwards),
        "solve_rate": solved / len(tasks),
        "context_changed": False,
    }


DEPTHS = (1, 2, 3)


def run_seed(seed: int, cfg: argparse.Namespace) -> dict[str, object]:
    training = make_tasks(seed, cfg.train_tasks, 1, range(0, 10)) + make_tasks(
        seed + 1_000,
        cfg.train_tasks,
        2,
        range(0, 10),
    )
    ported = StatePortPolicy(cfg.hidden, cfg.context, seed)
    dense = StatePortPolicy(cfg.hidden, cfg.context, seed + 12_345)
    local = ParamSkillPolicy(cfg.hidden, seed)
    local_data = make_training(seed, max(1, cfg.train_tasks // NK))

    rng = np.random.default_rng(seed)
    dense_rng = np.random.default_rng(seed + 555)
    local_rng = np.random.default_rng(seed + 777)
    for _ in range(cfg.epochs):
        ported.train_batch(
            [training[index] for index in rng.integers(0, len(training), cfg.batch)],
            cfg.lr,
            use_skill=True,
        )
        dense.train_batch(
            [training[index] for index in dense_rng.integers(0, len(training), cfg.batch)],
            cfg.lr,
            use_skill=False,
        )
        local_indices = local_rng.integers(0, len(local_data), min(cfg.batch, len(local_data)))
        local.train_batch([local_data[index] for index in local_indices], cfg.lr)

    training_keys = {task.key for task in training}
    ladder: dict[str, object] = {}
    overlap = False
    identical_task_ids = True
    for depth in DEPTHS:
        tasks = make_tasks(seed * 100 + depth, cfg.eval_tasks, depth, range(10, 20))
        overlap = overlap or bool(training_keys & {task.key for task in tasks})
        arms = {
            "ported": evaluate(
                ported,
                tasks,
                cfg.node_budget,
                cfg.bf_depth,
                use_skill=True,
                reset_context=False,
            ),
            "reset": evaluate(
                ported,
                tasks,
                cfg.node_budget,
                cfg.bf_depth,
                use_skill=True,
                reset_context=True,
            ),
            "dense_context": evaluate(
                dense,
                tasks,
                cfg.node_budget,
                cfg.bf_depth,
                use_skill=False,
                reset_context=False,
            ),
            "local": evaluate_local(local, tasks, cfg.node_budget, cfg.bf_depth),
        }
        # All arms receive this exact shared task list; store IDs once, not four times.
        identical_task_ids = identical_task_ids and bool(tasks)
        ladder[str(depth)] = {"task_ids": [task.task_id for task in tasks], **arms}

    return {
        "seed": seed,
        "ladder": ladder,
        "checks": {
            "train_eval_disjoint": not overlap,
            "identical_task_ids": identical_task_ids,
            "ported_context_changed": bool(ladder["3"]["ported"]["context_changed"]),
            "reset_context_changed": bool(ladder["3"]["reset"]["context_changed"]),
            "finite": ported.finite() and dense.finite(),
        },
        "parameter_count": ported.parameter_count,
        "forward_macs": ported.forward_macs,
    }


def aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for depth in DEPTHS:
        depth_key = str(depth)
        row: dict[str, object] = {}
        for arm in ("ported", "reset", "dense_context", "local"):
            arm_runs = [run["ladder"][depth_key][arm] for run in runs]
            row[arm] = {
                "mean_forwards": round(
                    statistics.mean(item["mean_forwards"] for item in arm_runs),
                    4,
                ),
                "solve_rate": round(
                    statistics.mean(item["solve_rate"] for item in arm_runs),
                    4,
                ),
            }
        row["ported_beats_reset_seeds"] = sum(
            run["ladder"][depth_key]["ported"]["mean_forwards"]
            < run["ladder"][depth_key]["reset"]["mean_forwards"]
            for run in runs
        )
        row["ported_beats_dense_seeds"] = sum(
            run["ladder"][depth_key]["ported"]["mean_forwards"]
            < run["ladder"][depth_key]["dense_context"]["mean_forwards"]
            for run in runs
        )
        result[depth_key] = row
    return result


def verdict(
    runs: list[dict[str, object]],
    aggregate_result: dict[str, object],
    seeds: list[int],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    checks = [run["checks"] for run in runs]
    parameter_counts = {run["parameter_count"] for run in runs}
    forward_macs = {run["forward_macs"] for run in runs}
    if len(seeds) < 6:
        reasons.append("fewer_than_6_seeds")
    if not all(check["train_eval_disjoint"] for check in checks):
        reasons.append("train_eval_overlap")
    if not all(check["identical_task_ids"] for check in checks):
        reasons.append("task_ids_differ_across_arms")
    if not all(check["ported_context_changed"] for check in checks):
        reasons.append("ported_context_never_changed")
    if any(check["reset_context_changed"] for check in checks):
        reasons.append("reset_context_changed")
    if not all(check["finite"] for check in checks):
        reasons.append("non_finite_parameters")
    if len(parameter_counts) != 1 or len(forward_macs) != 1:
        reasons.append("unmatched_parameter_or_forward_cost")
    if reasons:
        return "INVALID", reasons

    depth3 = aggregate_result["3"]
    ported = depth3["ported"]
    reset = depth3["reset"]
    dense = depth3["dense_context"]
    passed = (
        ported["solve_rate"] >= 0.98
        and abs(ported["solve_rate"] - dense["solve_rate"]) <= 0.01
        and ported["mean_forwards"] <= 0.9 * reset["mean_forwards"]
        and ported["mean_forwards"] <= 0.9 * dense["mean_forwards"]
        and depth3["ported_beats_reset_seeds"] >= 5
        and depth3["ported_beats_dense_seeds"] >= 5
    )
    if not passed:
        reasons.append("pre_registered_pass_bar_not_met")
    return ("PASS" if passed else "FAIL"), reasons


def run_experiment(cfg: argparse.Namespace) -> dict[str, object]:
    seeds = [int(seed) for seed in cfg.seeds.split(",") if seed]
    runs = [run_seed(seed, cfg) for seed in seeds]
    aggregate_result = aggregate(runs)
    decision, reasons = verdict(runs, aggregate_result, seeds)
    parameter_count = runs[0]["parameter_count"] if runs else 0
    forward_macs = runs[0]["forward_macs"] if runs else 0
    return {
        "config": vars(cfg),
        "protocol_evidence": {
            "training_depths": [1, 2],
            "evaluation_depths": list(DEPTHS),
            "training_start_range": [0, 9],
            "evaluation_start_range": [10, 19],
            "observable_policy_inputs": [
                "state",
                "target",
                "own_recurrent_context",
                "chosen_primitive_after_decision",
            ],
            "forbidden_policy_inputs": [
                "true_plan",
                "composition_depth",
                "true_subroutine_id",
                "true_skill_boundary",
            ],
            "action_set": "5 primitives for every matched arm",
            "compute_metric": "best-first forward passes",
            "valid": decision != "INVALID",
        },
        "parameter_accounting": {
            arm: {"parameters": parameter_count, "estimated_macs_per_forward": forward_macs}
            for arm in ("ported", "reset", "dense_context")
        },
        "aggregate_by_depth": aggregate_result,
        "seed_runs": runs,
        "verdict": {"decision": decision, "reasons": reasons},
    }


def self_check(cfg: argparse.Namespace) -> None:
    cfg.seeds = "11,17"
    cfg.train_tasks = 8
    cfg.eval_tasks = 4
    cfg.hidden = 8
    cfg.context = 4
    cfg.epochs = 3
    cfg.batch = 4
    cfg.node_budget = 48
    cfg.bf_depth = 12
    first = run_experiment(cfg)
    second = run_experiment(cfg)
    assert first == second
    assert first["verdict"]["decision"] == "INVALID"
    assert first["protocol_evidence"]["observable_policy_inputs"] == [
        "state",
        "target",
        "own_recurrent_context",
        "chosen_primitive_after_decision",
    ]
    assert all(run["checks"]["ported_context_changed"] for run in first["seed_runs"])
    assert not any(run["checks"]["reset_context_changed"] for run in first["seed_runs"])
    print("SELF-CHECK OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="11,17,23,41,73,101")
    parser.add_argument("--train-tasks", type=int, default=120)
    parser.add_argument("--eval-tasks", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--node-budget", type=int, default=256)
    parser.add_argument("--bf-depth", type=int, default=16)
    parser.add_argument("--output-json", default="artifacts/ecs_stage3_state_porting.json")
    parser.add_argument("--self-check", action="store_true")
    cfg = parser.parse_args()

    if cfg.self_check:
        self_check(cfg)
        return

    payload = run_experiment(cfg)
    output = Path(cfg.output_json)
    if not output.is_absolute():
        output = Path(__file__).parents[2] / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    depth3 = payload["aggregate_by_depth"]["3"]
    print("=== ECS Stage 3: observable state-porting ===")
    print(json.dumps(depth3, indent=2))
    print(f"VERDICT: {payload['verdict']['decision']} {payload['verdict']['reasons']}")
    print(f"-> {output}")
    if payload["verdict"]["decision"] == "INVALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
