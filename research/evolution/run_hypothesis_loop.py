#!/usr/bin/env python3
"""Evaluate LLM-style architecture hypotheses in a safe loop.

The loop accepts JSON/JSONL proposals that look like what an external LLM could
produce, validates them against the constrained formula DSL from
`evolve_ssm_formulas.py`, evaluates accepted hypotheses in the existing JAX
agent world, and writes feedback for the next round. It never patches
`agent_ssm_core.py` or executes arbitrary generated Python.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def requested_backend(argv: list[str]) -> str:
    backend = os.environ.get("JAX_PLATFORM_NAME", "cpu")
    for i, arg in enumerate(argv):
        if arg == "--backend" and i + 1 < len(argv):
            backend = argv[i + 1]
        elif arg.startswith("--backend="):
            backend = arg.split("=", 1)[1]
    return backend


REQUESTED_BACKEND = requested_backend(sys.argv[1:])
if REQUESTED_BACKEND == "auto":
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ.pop("JAX_PLATFORMS", None)
else:
    os.environ["JAX_PLATFORM_NAME"] = REQUESTED_BACKEND
    os.environ["JAX_PLATFORMS"] = REQUESTED_BACKEND

import jax

from agent_ssm_core import init_positions_energy
from evolve_ssm_formulas import (
    BASELINE_FORMULA,
    COMBINE_OPS,
    FINAL_OPS,
    INPUT_OPS,
    MEMORY_OPS,
    FormulaSpec,
    evaluate_formula_architecture,
    formula_id,
    formula_payload,
)


FALLBACK_HYPOTHESES = (
    {
        "id": "baseline_reference",
        "source": "fallback",
        "rationale": "Known SSM baseline for calibration.",
        "formula": {
            "memory_op": "identity",
            "input_op": "identity",
            "combine_op": "add",
            "final_op": "tanh",
        },
    },
    {
        "id": "relu_max_gate_probe",
        "source": "fallback",
        "rationale": "Test whether rectified memory plus max input selection creates stable obstacle-avoiding state.",
        "formula": {
            "memory_op": "relu",
            "input_op": "identity",
            "combine_op": "max",
            "final_op": "tanh",
        },
    },
    {
        "id": "softsign_gated_mix_probe",
        "source": "fallback",
        "rationale": "Bound both memory and input terms, then let each gate the other.",
        "formula": {
            "memory_op": "softsign",
            "input_op": "tanh",
            "combine_op": "gated_mix",
            "final_op": "softsign",
        },
    },
)


def load_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON proposal file must contain a list of proposals.")
        return payload
    proposals: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            proposal = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {lineno}: {exc}") from exc
        if not isinstance(proposal, dict):
            raise ValueError(f"Proposal at line {lineno} is not an object.")
        proposals.append(proposal)
    return proposals


def parse_proposals_text(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("Generator JSON must contain a list of proposals.")
        return payload
    proposals: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        proposal = json.loads(line)
        if not isinstance(proposal, dict):
            raise ValueError(f"Generated proposal at line {lineno} is not an object.")
        proposals.append(proposal)
    return proposals


def run_generator_command(command: str, prompt_context: dict[str, object], timeout: int) -> tuple[list[dict[str, Any]], str]:
    proc = subprocess.run(
        command,
        input=json.dumps(prompt_context, ensure_ascii=False),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Generator command failed with exit code {proc.returncode}: {command}\n{proc.stderr}"
        )
    return parse_proposals_text(proc.stdout), proc.stderr


def write_json(path: str, payload: dict[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, records: list[dict[str, object]]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def proposal_formula_payload(proposal: dict[str, Any]) -> dict[str, Any]:
    if "formula" in proposal and isinstance(proposal["formula"], dict):
        return proposal["formula"]
    return proposal


def validate_proposal(proposal: dict[str, Any], index: int) -> tuple[FormulaSpec | None, list[str]]:
    formula_payload_obj = proposal_formula_payload(proposal)
    errors: list[str] = []
    memory_op = formula_payload_obj.get("memory_op")
    input_op = formula_payload_obj.get("input_op")
    combine_op = formula_payload_obj.get("combine_op")
    final_op = formula_payload_obj.get("final_op")

    if memory_op not in MEMORY_OPS:
        errors.append(f"memory_op must be one of {MEMORY_OPS}, got {memory_op!r}")
    if input_op not in INPUT_OPS:
        errors.append(f"input_op must be one of {INPUT_OPS}, got {input_op!r}")
    if combine_op not in COMBINE_OPS:
        errors.append(f"combine_op must be one of {COMBINE_OPS}, got {combine_op!r}")
    if final_op not in FINAL_OPS:
        errors.append(f"final_op must be one of {FINAL_OPS}, got {final_op!r}")
    if errors:
        return None, errors
    assert isinstance(memory_op, str)
    assert isinstance(input_op, str)
    assert isinstance(combine_op, str)
    assert isinstance(final_op, str)
    return FormulaSpec(memory_op, input_op, combine_op, final_op), []


def proposal_id(proposal: dict[str, Any], index: int, formula: FormulaSpec | None = None) -> str:
    raw_id = proposal.get("id") or proposal.get("name")
    if isinstance(raw_id, str) and raw_id.strip():
        return raw_id.strip()
    if formula is not None:
        return formula_id(formula)
    return f"proposal_{index:03d}"


def make_prompt_context() -> dict[str, object]:
    return {
        "task": "Propose SSM hidden-state update formulas that may improve agent reward.",
        "schema": {
            "id": "short_hypothesis_id",
            "rationale": "why this may work",
            "formula": {
                "memory_op": list(MEMORY_OPS),
                "input_op": list(INPUT_OPS),
                "combine_op": list(COMBINE_OPS),
                "final_op": list(FINAL_OPS),
            },
        },
        "baseline_formula": formula_payload(FormulaSpec(*BASELINE_FORMULA)),
        "guardrail": "Return JSON objects only. No arbitrary Python will be executed.",
    }


def run(args: argparse.Namespace) -> None:
    prompt_context = make_prompt_context()
    generator_stderr = ""
    if args.generator_command:
        proposals, generator_stderr = run_generator_command(
            args.generator_command,
            prompt_context,
            args.generator_timeout,
        )
        proposal_source = f"generator_command:{args.generator_command}"
    elif args.proposals:
        proposals = load_json_or_jsonl(args.proposals)
        proposal_source = args.proposals
    else:
        proposals = [dict(item) for item in FALLBACK_HYPOTHESES]
        proposal_source = "fallback_builtin"
    proposals = proposals[: args.max_proposals]
    if not proposals:
        raise SystemExit("No hypotheses to evaluate.")

    init_positions_xy, init_energy = init_positions_energy(args.agents, args.seed)
    baseline_formula = FormulaSpec(*BASELINE_FORMULA)

    print("TextPy/SoA hypothesis feedback loop")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print("")
    print("Hypothesis source")
    print(f"  source: {proposal_source}")
    print(f"  proposals: {len(proposals)}")
    print("")
    print("Safe formula DSL")
    print(f"  memory_ops: {', '.join(MEMORY_OPS)}")
    print(f"  input_ops: {', '.join(INPUT_OPS)}")
    print(f"  combine_ops: {', '.join(COMBINE_OPS)}")
    print(f"  final_ops: {', '.join(FINAL_OPS)}")
    print("")

    start = time.perf_counter()
    baseline_record = evaluate_formula_architecture(
        baseline_formula,
        architecture_generation=0,
        architecture_index=0,
        init_positions_xy=init_positions_xy,
        init_energy=init_energy,
        args=args,
    )
    baseline_reward = float(baseline_record["best"]["reward"])
    print(f"Baseline reward: {baseline_reward:.4f} ({baseline_record['formula']['id']})")

    feedback: list[dict[str, object]] = []
    accepted_count = 0
    rejected_count = 0
    seen: set[str] = set()
    for index, proposal in enumerate(proposals):
        formula, errors = validate_proposal(proposal, index)
        hyp_id = proposal_id(proposal, index, formula)
        rationale = proposal.get("rationale") if isinstance(proposal.get("rationale"), str) else ""
        if formula is None:
            rejected_count += 1
            record = {
                "id": hyp_id,
                "status": "rejected",
                "errors": errors,
                "rationale": rationale,
                "raw_proposal": proposal,
            }
            feedback.append(record)
            print(f"Rejected {hyp_id}: {'; '.join(errors)}")
            continue
        fid = formula_id(formula)
        if fid in seen:
            rejected_count += 1
            record = {
                "id": hyp_id,
                "status": "rejected",
                "errors": [f"duplicate formula: {fid}"],
                "formula": formula_payload(formula),
                "rationale": rationale,
            }
            feedback.append(record)
            print(f"Rejected {hyp_id}: duplicate formula")
            continue
        seen.add(fid)
        accepted_count += 1
        record = evaluate_formula_architecture(
            formula,
            architecture_generation=0,
            architecture_index=index + 1,
            init_positions_xy=init_positions_xy,
            init_energy=init_energy,
            args=args,
        )
        reward = float(record["best"]["reward"])
        delta = reward - baseline_reward
        feedback_record = {
            "id": hyp_id,
            "status": "accepted",
            "rationale": rationale,
            "formula": record["formula"],
            "best": record["best"],
            "compile_s": record["compile_s"],
            "weight_evolution_s": record["weight_evolution_s"],
            "throughput_agent_steps_s": record["throughput_agent_steps_s"],
            "baseline_reward": baseline_reward,
            "reward_delta_vs_baseline": delta,
            "feedback_to_generator": (
                "promote_or_mutate" if delta > 0 else "revise: did not beat baseline in this smoke protocol"
            ),
        }
        feedback.append(feedback_record)
        print(f"Accepted {hyp_id}: reward={reward:.4f}, delta={delta:+.4f}, formula={fid}")

    accepted = [record for record in feedback if record["status"] == "accepted"]
    best = max(accepted, key=lambda item: float(item["best"]["reward"])) if accepted else None
    elapsed = time.perf_counter() - start
    payload = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "proposal_source": proposal_source,
        "generator_stderr": generator_stderr,
        "prompt_context": prompt_context,
        "protocol": {
            "max_proposals": args.max_proposals,
            "accepted": accepted_count,
            "rejected": rejected_count,
            "agents": args.agents,
            "steps": args.steps,
            "weight_population": args.weight_population,
            "weight_elites": args.weight_elites,
            "weight_generations": args.weight_generations,
            "weight_mutation_sigma": args.weight_mutation_sigma,
            "speed": args.speed,
            "seed": args.seed,
        },
        "baseline_record": baseline_record,
        "best_accepted": best,
        "feedback": feedback,
        "elapsed_s": elapsed,
        "claim_guardrail": (
            "This is an LLM-compatible hypothesis feedback loop over a constrained DSL. "
            "It validates and benchmarks proposals; it does not execute arbitrary code or prove AGI."
        ),
    }
    if args.output_json:
        write_json(args.output_json, payload)
    if args.feedback_jsonl:
        write_jsonl(args.feedback_jsonl, feedback)
    if args.prompt_json:
        write_json(args.prompt_json, prompt_context)

    print("")
    print("Hypothesis loop result")
    print(f"  accepted: {accepted_count}")
    print(f"  rejected: {rejected_count}")
    if best is not None:
        print(f"  best_id: {best['id']}")
        print(f"  best_reward: {best['best']['reward']:.4f}")
        print(f"  best_delta_vs_baseline: {best['reward_delta_vs_baseline']:+.4f}")
        print(f"  best_formula: {best['formula']['id']}")
    print(f"  elapsed_s: {elapsed:.4f}")
    print("  guardrail: validates DSL hypotheses, does not execute arbitrary code")
    if args.output_json:
        print(f"  output_json: {args.output_json}")
    if args.feedback_jsonl:
        print(f"  feedback_jsonl: {args.feedback_jsonl}")
    if args.prompt_json:
        print(f"  prompt_json: {args.prompt_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM-style SSM formula hypotheses safely.")
    parser.add_argument("--proposals", default=None, help="JSON list or JSONL hypothesis proposals.")
    parser.add_argument(
        "--generator-command",
        default=None,
        help="Optional external generator command. Receives prompt JSON on stdin and must print JSON/JSONL proposals.",
    )
    parser.add_argument("--generator-timeout", type=int, default=20)
    parser.add_argument("--max-proposals", type=int, default=5)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--feedback-jsonl", default=None)
    parser.add_argument("--prompt-json", default=None)
    parser.add_argument("--weight-population", type=int, default=8)
    parser.add_argument("--weight-elites", type=int, default=2)
    parser.add_argument("--weight-generations", type=int, default=1)
    parser.add_argument("--weight-mutation-sigma", type=float, default=0.08)
    parser.add_argument("--agents", type=int, default=256)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--speed", type=float, default=0.028)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
