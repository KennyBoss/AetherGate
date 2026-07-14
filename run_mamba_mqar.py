#!/usr/bin/env python3
"""Run the official state-spaces/mamba block on the registered MQAR protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_REPOSITORY = "https://github.com/state-spaces/mamba"
IMPLEMENTATION_CLASS = "mamba_ssm.Mamba"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_mqar_recall(
    *,
    records: int,
    delay: int,
    key_count: int,
    binding_count: int,
    query_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if binding_count < 1 or query_count < 1 or query_count > binding_count:
        raise ValueError("Invalid MQAR binding/query counts.")
    if binding_count > key_count or delay < 4 * binding_count:
        raise ValueError("Invalid MQAR vocabulary or delay.")
    rng = np.random.default_rng(seed)
    total_len = delay + 2 * query_count
    sequence = np.zeros((records, total_len), dtype=np.int32)
    candidate_starts = np.arange(0, delay // 2, 2)
    if len(candidate_starts) < binding_count:
        raise ValueError("Delay cannot hold the requested MQAR pairs.")
    for row in range(records):
        keys = rng.choice(key_count, size=binding_count, replace=False) + 1
        values = rng.integers(0, key_count, size=binding_count, dtype=np.int32) + 1
        starts = np.sort(rng.choice(candidate_starts, size=binding_count, replace=False))
        for key, value, start in zip(keys, values, starts):
            sequence[row, start] = int(key)
            sequence[row, start + 1] = int(value)
        slots = rng.choice(binding_count, size=query_count, replace=False)
        for query, slot in enumerate(slots):
            base = delay + 2 * query
            sequence[row, base] = int(keys[slot])
            sequence[row, base + 1] = int(values[slot])
    inputs = sequence[:, :-1]
    targets = sequence[:, 1:]
    mask = np.zeros_like(inputs, dtype=np.float32)
    for query in range(query_count):
        mask[:, delay + 2 * query] = 1.0
    return inputs, targets, mask


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def estimated_parameter_count(
    *,
    d_model: int,
    vocab_size: int,
    d_state: int,
    d_conv: int,
    expand: int,
) -> int:
    """Exact count for Embedding + one official Mamba-1 block + LayerNorm + head."""
    d_inner = expand * d_model
    dt_rank = math.ceil(d_model / 16)
    embedding_and_head = 2 * vocab_size * d_model
    layer_norm = 2 * d_model
    in_proj = 2 * d_inner * d_model
    conv1d = d_inner * d_conv + d_inner
    x_proj = (dt_rank + 2 * d_state) * d_inner
    dt_proj = dt_rank * d_inner + d_inner
    a_log = d_inner * d_state
    d_skip = d_inner
    out_proj = d_model * d_inner
    return embedding_and_head + layer_norm + in_proj + conv1d + x_proj + dt_proj + a_log + d_skip + out_proj


def choose_d_model(
    *,
    target_params: int,
    vocab_size: int,
    d_state: int,
    d_conv: int,
    expand: int,
) -> tuple[int, int]:
    choices = (
        (
            abs(estimated_parameter_count(
                d_model=d_model,
                vocab_size=vocab_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) - target_params),
            d_model,
        )
        for d_model in range(8, 4097)
    )
    _, d_model = min(choices)
    count = estimated_parameter_count(
        d_model=d_model,
        vocab_size=vocab_size,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )
    return d_model, count


def train_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from mamba_ssm import Mamba
        from importlib.metadata import version
    except ImportError as error:
        raise SystemExit(
            "Official Mamba dependencies are missing. Install CUDA PyTorch, then "
            "`pip install mamba-ssm[causal-conv1d] --no-build-isolation`."
        ) from error
    if not torch.cuda.is_available():
        raise SystemExit("Official Mamba evidence requires a CUDA-visible NVIDIA GPU.")

    vocab_size = args.key_count + 1
    d_model, estimated = choose_d_model(
        target_params=args.target_params,
        vocab_size=vocab_size,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
    )

    class MambaRecall(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.mamba = Mamba(
                d_model=d_model,
                d_state=args.d_state,
                d_conv=args.d_conv,
                expand=args.expand,
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, tokens: Any) -> Any:
            return self.head(self.norm(self.mamba(self.embedding(tokens))))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model = MambaRecall().cuda()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != estimated:
        raise RuntimeError(f"Mamba parameter formula drifted: estimated={estimated}, measured={parameter_count}.")
    ratio = parameter_count / args.target_params
    if not 0.8 <= ratio <= 1.25:
        raise RuntimeError(f"Mamba parameter budget mismatch: ratio={ratio:.4f}.")

    train_arrays = make_mqar_recall(
        records=args.train_records,
        delay=args.delay,
        key_count=args.key_count,
        binding_count=args.binding_count,
        query_count=args.query_count,
        seed=seed,
    )
    eval_arrays = make_mqar_recall(
        records=args.eval_records,
        delay=args.delay,
        key_count=args.key_count,
        binding_count=args.binding_count,
        query_count=args.query_count,
        seed=seed + 1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    order_rng = np.random.default_rng(seed + 10_000)
    train_start = time.perf_counter()
    model.train()
    for _ in range(args.epochs):
        order = order_rng.permutation(args.train_records)
        for start in range(0, args.train_records, args.batch_size):
            rows = order[start : start + args.batch_size]
            inputs = torch.as_tensor(train_arrays[0][rows], dtype=torch.long, device="cuda")
            targets = torch.as_tensor(train_arrays[1][rows], dtype=torch.long, device="cuda")
            mask = torch.as_tensor(train_arrays[2][rows], dtype=torch.float32, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            losses = functional.cross_entropy(
                logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none"
            ).reshape_as(mask)
            loss = (losses * mask).sum() / mask.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
    torch.cuda.synchronize()
    train_s = time.perf_counter() - train_start

    model.eval()
    loss_sum = correct = query_count = 0.0
    eval_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, args.eval_records, args.batch_size):
            end = min(start + args.batch_size, args.eval_records)
            inputs = torch.as_tensor(eval_arrays[0][start:end], dtype=torch.long, device="cuda")
            targets = torch.as_tensor(eval_arrays[1][start:end], dtype=torch.long, device="cuda")
            mask = torch.as_tensor(eval_arrays[2][start:end], dtype=torch.float32, device="cuda")
            logits = model(inputs)
            losses = functional.cross_entropy(
                logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none"
            ).reshape_as(mask)
            predictions = logits.argmax(dim=-1)
            loss_sum += float((losses * mask).sum().item())
            correct += float(((predictions == targets).float() * mask).sum().item())
            query_count += float(mask.sum().item())
    torch.cuda.synchronize()
    eval_s = time.perf_counter() - eval_start
    return {
        "seed": seed,
        "d_model": d_model,
        "parameter_count": parameter_count,
        "target_parameter_count": args.target_params,
        "parameter_ratio": ratio,
        "train_s": train_s,
        "eval_s": eval_s,
        "records_per_s": round(args.eval_records / eval_s),
        "recall_accuracy": correct / query_count,
        "recall_loss": loss_sum / query_count,
        "query_count": int(query_count),
        "train_data_sha256": array_sha256(*train_arrays),
        "eval_data_sha256": array_sha256(*eval_arrays),
        "torch_version": torch.__version__,
        "mamba_ssm_version": version("mamba-ssm"),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--point", default="cap-100k")
    parser.add_argument("--seeds", default="11,17,23")
    parser.add_argument("--target-params", type=int, default=100_000)
    parser.add_argument("--train-records", type=int, default=2048)
    parser.add_argument("--eval-records", type=int, default=1024)
    parser.add_argument("--key-count", type=int, default=128)
    parser.add_argument("--binding-count", type=int, default=12)
    parser.add_argument("--query-count", type=int, default=6)
    parser.add_argument("--delay", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.006)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    runs = [train_seed(args, seed) for seed in seeds]
    payload = {
        "created_at": timestamp(),
        "kind": "kv_scale_law_mamba_mqar",
        "status": "COMPLETE",
        "implementation": {
            "official_repository": OFFICIAL_REPOSITORY,
            "class": IMPLEMENTATION_CLASS,
            "block": "Mamba-1",
            "layers": 1,
            "d_state": args.d_state,
            "d_conv": args.d_conv,
            "expand": args.expand,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cuda_required": True,
        },
        "protocol": {
            "point": args.point,
            "seeds": seeds,
            "train_records": args.train_records,
            "eval_records": args.eval_records,
            "key_count": args.key_count,
            "binding_count": args.binding_count,
            "query_count": args.query_count,
            "delay": args.delay,
            "seq_len": args.delay + 2 * args.query_count - 1,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "grad_clip": args.grad_clip,
            "query_only_loss": True,
        },
        "runs": runs,
        "claim_guardrail": (
            "This is an equivalent-budget official Mamba block on bounded MQAR, "
            "not evidence of language-model or AGI superiority."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
