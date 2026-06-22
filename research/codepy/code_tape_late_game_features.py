#!/usr/bin/env python3
"""Structural opcode-sequence features for the CodePy late-game ranker.

These features are deliberately the *opposite* of the trace-only prior in
``code_tape_trace_prior.py``: that prior estimates ``P(trace | task)`` and avoids
opcode identity, while this module looks **only** at the opcode sequence and its
control-flow grammar (canonical ordering of read/compare/update/loop/return/halt
stages).  The H4 late-game analysis showed that survival-protected near-misses
already collect the right argmax blocks but assemble them in an invalid order;
this feature space is built to score that ordering.

Guardrail: all features are derived generically from the *registered* safe-DSL
reference program (the task specification).  Nothing here hardcodes an accept
rule -- the late-game ranker learns a logistic score over these features.  No
generated Python is executed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analyze_code_tape_late_game import (
    ARGMAX_STAGE_DEFS,
    PATTERN_DEFS,
    compact_ops,
    find_stage_positions,
    has_subsequence,
    lcs_len,
    matched_stage_count,
)
from code_tape_trace_prior import safe_float

# Terminal / control opcodes referenced by name in the structural features.
OP_FIRST_MOVE = 7      # ptr += 1
OP_LOOP_TEST = 15      # p = ptr < input_len
OP_VALUE_UPDATE = 25   # r0 = where(p, tape[ptr], r0)
OP_INDEX_UPDATE = 28   # r1 = where(p, ptr, r1)
OP_LOOP_JUMP = 30      # pc = where(p, 3, pc + 1)
OP_HALT = 31           # halt
OP_RETURN_INDEX = 32   # r0 = r1


def first_positions(ops: list[int]) -> dict[int, int]:
    """Map each opcode to the index of its first occurrence in ``ops``."""
    positions: dict[int, int] = {}
    for i, op in enumerate(ops):
        positions.setdefault(int(op), i)
    return positions


def reference_transition_pairs(reference: list[int]) -> list[tuple[int, int]]:
    """Consecutive distinct-opcode transitions in first-appearance order.

    This turns the registered reference program into a canonical bigram grammar
    (e.g. read -> index-init -> move -> compare -> ...), generically and without
    naming argmax stages explicitly.
    """
    order: list[int] = []
    seen: set[int] = set()
    for op in reference:
        op = int(op)
        if op not in seen:
            seen.add(op)
            order.append(op)
    return [(order[i], order[i + 1]) for i in range(len(order) - 1)]


def _ordered_relation(positions: dict[int, int], a: int, b: int) -> float:
    """Signed indicator: +1 if first(a) < first(b), -1 if wrong order, 0 if absent."""
    if a in positions and b in positions:
        return 1.0 if positions[a] < positions[b] else -1.0
    return 0.0


def structural_feature_names(reference: list[int]) -> list[str]:
    names: list[str] = []
    for a, b in reference_transition_pairs(reference):
        names.append(f"order_{a}_before_{b}")
    names.extend(
        [
            "has_halt",
            "ends_with_halt",
            "single_halt",
            "has_return_index",
            "return_before_halt",
            "loop_jump_before_return",
            "first_move_before_loop_test",
            "value_update_before_index_update",
        ]
    )
    names.append("stage_completeness")
    names.extend(f"ngram_{name}" for name, _ in PATTERN_DEFS)
    names.extend(["reference_lcs_ratio", "length_ratio", "length_abs_gap"])
    return names


def structural_feature_vector(ops: list[int], reference: list[int]) -> np.ndarray:
    """Vectorize a compact opcode sequence into structural control-flow features."""
    ops = [int(op) for op in ops]
    reference = [int(op) for op in reference]
    positions = first_positions(ops)
    ref_len = max(len(reference), 1)

    values: list[float] = []
    for a, b in reference_transition_pairs(reference):
        values.append(_ordered_relation(positions, a, b))

    halt_count = ops.count(OP_HALT)
    values.append(1.0 if OP_HALT in positions else 0.0)
    values.append(1.0 if ops and ops[-1] == OP_HALT else 0.0)
    values.append(1.0 if halt_count == 1 else 0.0)
    values.append(1.0 if OP_RETURN_INDEX in positions else 0.0)
    values.append(1.0 if has_subsequence(ops, (OP_RETURN_INDEX, OP_HALT)) else 0.0)
    values.append(1.0 if has_subsequence(ops, (OP_LOOP_JUMP, OP_RETURN_INDEX)) else 0.0)
    values.append(_ordered_relation(positions, OP_FIRST_MOVE, OP_LOOP_TEST))
    values.append(_ordered_relation(positions, OP_VALUE_UPDATE, OP_INDEX_UPDATE))

    matched = matched_stage_count(find_stage_positions(ops))
    values.append(matched / max(len(ARGMAX_STAGE_DEFS), 1))

    for _, pattern in PATTERN_DEFS:
        values.append(1.0 if has_subsequence(ops, pattern) else 0.0)

    values.append(lcs_len(ops, reference) / ref_len)
    values.append(len(ops) / ref_len)
    values.append(abs(len(ops) - len(reference)) / ref_len)

    return np.asarray(values, dtype=np.float64)


def ops_from_record(record: dict[str, Any]) -> list[int]:
    """Compact opcode list for a candidate record (drops padding noops)."""
    return compact_ops(record)


def predict_late_game_score(model_payload: dict[str, Any], ops: list[int]) -> float:
    """Logistic score that a compact opcode sequence is structurally canonical."""
    model = model_payload["model"]
    reference = [int(op) for op in model.get("reference", [])]
    features = structural_feature_vector(ops, reference)
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = float(model.get("bias", 0.0))
    logit = float(features @ weights + bias)
    if logit >= 0:
        return float(1.0 / (1.0 + np.exp(-logit)))
    exp_logit = float(np.exp(logit))
    return exp_logit / (1.0 + exp_logit)


def reference_corruptions(reference: list[int]) -> list[dict[str, Any]]:
    """Synthetic structural negatives derived from the reference per H4 families.

    These are *known-bad* control-flow corruptions (clearly labeled negatives),
    not data leakage: they teach the ranker the decision boundary that the H4
    analysis localized -- missing ``halt``, swapped value/index updates, and a
    loop test placed before the first pointer move.
    """
    reference = [int(op) for op in reference]
    corruptions: list[dict[str, Any]] = []

    # 1. Drop the terminal halt (missing_halt family, 49/50 of survivors).
    if OP_HALT in reference:
        dropped = [op for op in reference if op != OP_HALT]
        corruptions.append({"name": "drop_halt", "program_ids": dropped})

    # 2. Swap the first conditional value/index updates (28-before-25 family).
    swapped = list(reference)
    try:
        i_val = swapped.index(OP_VALUE_UPDATE)
        i_idx = swapped.index(OP_INDEX_UPDATE)
        swapped[i_val], swapped[i_idx] = swapped[i_idx], swapped[i_val]
        corruptions.append({"name": "swap_value_index_update", "program_ids": swapped})
    except ValueError:
        pass

    # 3. Move the loop test ahead of the first pointer move
    #    (loop_test_before_first_move family, 50/50 of survivors).
    if OP_LOOP_TEST in reference and OP_FIRST_MOVE in reference:
        rest = [op for op in reference if op != OP_LOOP_TEST]
        move_at = rest.index(OP_FIRST_MOVE)
        moved = rest[:move_at] + [OP_LOOP_TEST] + rest[move_at:]
        corruptions.append({"name": "loop_test_before_first_move", "program_ids": moved})

    return corruptions
