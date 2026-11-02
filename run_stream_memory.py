#!/usr/bin/env python3
"""Train and measure the product core on chunked, revised fact streams.

Targets/masks are used only by the loss, never by the neural forward pass.
No pretrained weights or hand-initialized gates are used.
"""
import argparse
from functools import partial
import hashlib
import json
import platform
from pathlib import Path
import time

import train_ssm_text as core
import jax
import jax.numpy as jnp
import numpy as np
from compare_text_architectures import init_transformer_params
from compare_long_context_recall import transformer_forward_limited, recall_loss_and_accuracy


def make_data(records, delay, seed):
    rng = np.random.default_rng(seed)
    sequence = np.zeros((records, 24 + delay + 16), dtype=np.int32)
    mask = np.zeros_like(sequence, dtype=np.float32)
    for row in range(records):
        keys = rng.choice(8, 4, replace=False) + 1
        values = rng.integers(9, 17, 4)
        for slot in range(4):
            sequence[row, 4*slot:4*slot+4] = [keys[slot], 17, values[slot], 18]
        for update, slot in enumerate(rng.choice(4, 2, replace=False)):
            values[slot] = 9 + ((values[slot] - 9 + rng.integers(1, 8)) % 8)
            at = 16 + 4*update
            sequence[row, at:at+4] = [keys[slot], 17, values[slot], 18]
        for query, slot in enumerate(rng.permutation(4)):
            at = 24 + delay + 4*query
            sequence[row, at:at+4] = [keys[slot], 19, values[slot], 18]
            mask[row, at+1] = 1.
    return sequence[:, :-1], sequence[:, 1:], mask[:, :-1]


def predict(params, inputs, mode, chunk_size):
    if mode == "transformer":
        return transformer_forward_limited(params, inputs, layers=1, heads=2, context=inputs.shape[1])
    state = core.init_stream_state(params, inputs.shape[0])
    if mode == "whole":
        return core.forward(params, inputs, state)[0]
    if mode == "stream":
        return core.forward_chunks(params, inputs, state, chunk_size=chunk_size)[0]
    outputs = []
    for start in range(0, inputs.shape[1], chunk_size):
        # Exact old API: keep h, reset every auxiliary rail at each portion.
        logits, h = core.forward(params, inputs[:, start:start+chunk_size], state.h)
        state = state._replace(h=h)
        outputs.append(logits)
    return jnp.concatenate(outputs, axis=1)


def objective(params, x, y, mask, mode, chunk_size):
    logits = predict(params, x, mode, chunk_size)
    return recall_loss_and_accuracy(logits, x, y, mask)[0]


@partial(jax.jit, static_argnames=("mode", "chunk_size"))
def update(params, opt, x, y, mask, *, mode, chunk_size):
    loss, grads = jax.value_and_grad(objective)(params, x, y, mask, mode, chunk_size)
    grads, norm = core.clip_grads(grads, 1.)
    params, opt = core.adam_update(params, grads, opt, .006)
    return params, opt, loss, norm


@partial(jax.jit, static_argnames=("mode", "chunk_size"))
def evaluate(params, x, y, mask, *, mode, chunk_size):
    logits = predict(params, x, mode, chunk_size)
    loss, accuracy, _, _ = recall_loss_and_accuracy(logits, x, y, mask)
    return loss, accuracy


def score(params, data, mode, chunk_size):
    values = []
    for start in range(0, len(data[0]), 64):
        result = evaluate(params, *(jnp.asarray(a[start:start+64]) for a in data), mode=mode, chunk_size=chunk_size)
        values.append([float(a) for a in result])
    return {"loss": float(np.mean(values, axis=0)[0]), "accuracy": float(np.mean(values, axis=0)[1])}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"version": "stream-memory-v1", "seeds": [3] if args.pilot else [11, 17, 23],
                "seconds_per_model": 10 if args.pilot else 30, "maximum_updates": 16384,
                "batch": 64, "train_records": 2048, "test_records": 128 if args.pilot else 1024,
                "chunk_size": 8, "train_delay": 32, "ood_delay": 96,
                "models": ["stream", "legacy-reset", "transformer"], "learning_rate": .006,
                "equal_experience_updates": 256,
                "ssm_input_dim": 16, "ssm_state_dim": 32, "transformer_d_model": 12,
                "claim": "causal benefit of complete carry; no general language superiority claim",
                "selection": "fixed final checkpoint, no dev/test selection",
                "cost": "same elapsed training ceiling excluding separately reported compilation; all tokens and updates reported",
                "criterion": "stream >= .90 and stream-minus-same-weights-reset >= .10 on all three seeds"}
    protocol["environment"] = {"python":platform.python_version(), "platform":platform.platform(),
                               "jax":jax.__version__, "numpy":np.__version__, "backend":jax.default_backend()}
    write(args.output / "protocol.json", protocol)
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
              ["train_ssm_text.py", "run_stream_memory.py", "compare_long_context_recall.py",
               "compare_text_architectures.py", "jax_backend.py", "text_tokenization.py"]}
    results = []
    # Train and freeze every candidate before creating any held-out examples.
    for seed_index, seed in enumerate(protocol["seeds"]):
        train = make_data(protocol["train_records"], protocol["train_delay"], seed)
        arrays = tuple(jnp.asarray(a) for a in train)
        batch = tuple(a[:64] for a in arrays)
        initial = core.init_params(jax.random.PRNGKey(seed), 20, 16, 32, variant="kv-memory")
        modes = protocol["models"][seed_index:] + protocol["models"][:seed_index]
        for mode in modes:
            params = initial if mode != "transformer" else init_transformer_params(
                jax.random.PRNGKey(seed), vocab_size=20, seq_len=24+protocol["ood_delay"]+16,
                d_model=12, layers=1, ff_mult=2)
            opt = core.init_adam(params)
            before = score(params, tuple(a[:128] for a in train), mode, 8)
            start = time.perf_counter()
            warm = update(params, opt, *batch, mode=mode, chunk_size=8)
            jax.block_until_ready(warm)
            compile_seconds = time.perf_counter() - start
            del warm
            updates = 0
            equal_params = None
            start = time.perf_counter()
            while updates < protocol["maximum_updates"] and time.perf_counter()-start < protocol["seconds_per_model"]:
                offset = (updates % 32) * 64
                params, opt, loss, norm = update(params, opt, *(a[offset:offset+64] for a in arrays), mode=mode, chunk_size=8)
                jax.block_until_ready(loss)
                updates += 1
                if updates == protocol["equal_experience_updates"]:
                    equal_params = params
            elapsed = time.perf_counter() - start
            if not np.isfinite(float(loss)):
                raise RuntimeError("Nonfinite training loss")
            name = f"seed{seed}-{mode}"
            payload = {"seed": seed, "mode": mode, "updates": updates,
                       "tokens": updates*64*train[0].shape[1], "train_seconds": elapsed,
                       "compile_seconds": compile_seconds,
                       "parameters": sum(int(a.size) for a in jax.tree_util.tree_leaves(params)),
                       "train_before": before,
                       "parameter_delta_l2": float(core.global_norm(jax.tree_util.tree_map(lambda a,b:a-b, params, initial))) if mode != "transformer" else None}
            checkpoint = args.output / f"{name}.npz"
            if mode != "transformer":
                state = core.init_stream_state(params, 1)
                metrics = core.TrainMetrics(loss, jnp.asarray(0.), norm, jnp.mean(core.decay_from_raw(params["decay_raw"])))
                core.save_checkpoint(str(checkpoint), params, opt, [str(i) for i in range(20)],
                                     argparse.Namespace(seq_len=71, streams=1, tokens_per_stream=72), metrics, updates,
                                     stream_state=state)
            else:
                np.savez_compressed(checkpoint, **{f"param_{k}": np.asarray(v) for k,v in params.items()},
                                    **{f"opt_m_{k}":np.asarray(v) for k,v in opt["m"].items()},
                                    **{f"opt_v_{k}":np.asarray(v) for k,v in opt["v"].items()}, opt_step=np.asarray(opt["step"]))
            payload["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if equal_params is None:
                raise RuntimeError("Time budget did not reach the registered equal-experience checkpoint")
            equal_path = args.output / f"{name}-equal-updates.npz"
            np.savez_compressed(equal_path, **{f"param_{k}":np.asarray(v) for k,v in equal_params.items()})
            payload["equal_experience_checkpoint_sha256"] = hashlib.sha256(equal_path.read_bytes()).hexdigest()
            results.append(payload)
            write(args.output / f"{name}-train.json", payload)
            print(f"FROZEN {name} updates={updates} seconds={elapsed:.2f}", flush=True)
    write(args.output / "FROZEN.json", {"sources":hashes,"models":results})
    for result in results:
        seed, mode = result["seed"], result["mode"]
        checkpoint = args.output / f"seed{seed}-{mode}.npz"
        if mode != "transformer":
            params, _, _, _, _ = core.load_stream_checkpoint(str(checkpoint))
        else:
            with np.load(checkpoint, allow_pickle=False) as data:
                params = {k[6:]:jnp.asarray(data[k]) for k in data.files if k.startswith("param_")}
        train = make_data(2048,32,seed)
        seen = {row.tobytes() for row in train[0]}
        for label,delay,test_seed in [("heldout",32,seed+10000),("long_delay",96,seed+20000)]:
            data = make_data(protocol["test_records"],delay,test_seed)
            assert not any(row.tobytes() in seen for row in data[0]), "Train/test overlap"
            result[label] = score(params,data,mode,8)
            with np.load(args.output / f"seed{seed}-{mode}-equal-updates.npz", allow_pickle=False) as saved:
                equal_params = {k[6:]:jnp.asarray(saved[k]) for k in saved.files}
            result[label+"_equal_experience"] = score(equal_params,data,mode,8)
            if mode == "stream":
                result[label+"_same_weights_reset"] = score(params,data,"legacy-reset",8)
                result[label+"_whole"] = score(params,data,"whole",8)
                result[label+"_alternate_chunks"] = score(params,data,"stream",7)
                state = core.init_stream_state(params,64)
                result["stream_state_bytes_per_row"] = sum(int(x.size*x.dtype.itemsize) for x in jax.tree_util.tree_leaves(state))//64
        print(f"EVAL seed{seed}-{mode}: {result['heldout']} long={result['long_delay']}",flush=True)
    stream = [r for r in results if r["mode"] == "stream"]
    summary = {"protocol":protocol,"sources":hashes,"models":results,
               "criterion_passed":len(stream)==3 and all(r["heldout"]["accuracy"]>=.90 and
                r["heldout"]["accuracy"]-r["heldout_same_weights_reset"]["accuracy"]>=.10 for r in stream),
               "means":{m:{label:float(np.mean([r[label]["accuracy"] for r in results if r["mode"]==m]))
                            for label in ["heldout","long_delay"]} for m in protocol["models"]}}
    write(args.output/"results.json",summary)
    print(json.dumps({"means":summary["means"],"criterion_passed":summary["criterion_passed"]}),flush=True)


if __name__ == "__main__":
    main()
