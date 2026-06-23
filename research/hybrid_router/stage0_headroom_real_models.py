"""Stage 0 — first contact of the Law of Routing Existence with REAL trained
weights. No router yet: train the harness's real Transformer (context-capped) and
real KV-memory SSM on delayed-binding recall, then measure per-query

    headroom = oracle - best_single   (oracle = either expert correct).

Law prediction: on delayed recall the Transformer (binding beyond its window) and
the KV-memory SSM fail on ORTHOGONAL regions -> headroom > 0 (routing could pay).
Kill-condition: headroom ~ 0 here too -> real components are not orthogonal and
the whole hybrid premise is false.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # METAL (Apple GPU) backend broken here

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import compare_long_context_recall as H


def build_args(seed, context, epochs):
    sys.argv = [
        "x", "--task", "assignment", "--delay", "96",
        "--transformer-context", str(context),
        "--train-records", "1024", "--eval-records", "512",
        "--epochs", str(epochs), "--batch-size", "128", "--learning-rate", "0.006",
        "--key-count", "16", "--value-count", "16",
        "--ssm-variant", "kv-memory", "--kv-logit-scale", "64", "--ssm-decay-init", "8.0",
        "--seed", str(seed),
    ]
    if context >= 96:
        sys.argv.append("--allow-visible-key")
    return H.parse_args()


def train_both(args):
    vocab = H.vocab_size_for_task(args.task, args.key_count, args.value_count)
    tr_in, tr_tg, _ = H.make_task_data(
        task=args.task, records=args.train_records, delay=args.delay,
        key_count=args.key_count, value_count=args.value_count,
        binding_count=args.binding_count, query_count=args.query_count, seed=args.seed)
    ev_in, ev_tg, _ = H.make_task_data(
        task=args.task, records=args.eval_records, delay=args.delay,
        key_count=args.key_count, value_count=args.value_count,
        binding_count=args.binding_count, query_count=args.query_count, seed=args.seed + 1)
    tr_in = jnp.asarray(tr_in); tr_tg = jnp.asarray(tr_tg)
    ev_in = jnp.asarray(ev_in); ev_tg = jnp.asarray(ev_tg)
    seq_len = int(tr_in.shape[1])

    key = jax.random.PRNGKey(args.seed)
    ssm_key, tf_key = jax.random.split(key)
    ssm_params = H.init_ssm_params(ssm_key, vocab, args.ssm_input_dim, args.ssm_state_dim,
                                   variant=args.ssm_variant, skip_rank=args.ssm_skip_rank)
    if args.ssm_decay_init is not None:
        ssm_params["decay_raw"] = jnp.full(ssm_params["decay_raw"].shape,
                                           args.ssm_decay_init, dtype=jnp.float32)
    ssm_params, _, _, _ = H.train_model(params=ssm_params, train_inputs=tr_in,
                                        train_targets=tr_tg, args=args, model="ssm")

    ssm_count = H.count_params(ssm_params)
    d_model = H.choose_transformer_d_model(
        target_params=ssm_count, vocab_size=vocab, seq_len=seq_len,
        layers=args.transformer_layers, heads=args.transformer_heads,
        ff_mult=args.transformer_ff_mult, max_d_model=args.transformer_max_d_model)
    tf_params = H.init_transformer_params(tf_key, vocab_size=vocab, seq_len=seq_len,
                                          d_model=d_model, layers=args.transformer_layers,
                                          ff_mult=args.transformer_ff_mult)
    tf_params, _, _, _ = H.train_model(params=tf_params, train_inputs=tr_in,
                                       train_targets=tr_tg, args=args, model="transformer")

    # per-query correctness of BOTH on the SAME eval set
    ssm_pred, _, mask = H.ssm_confusion(ssm_params, ev_in, ev_tg)
    tf_logits = H.transformer_forward_limited(tf_params, ev_in, layers=args.transformer_layers,
                                              heads=args.transformer_heads, context=args.transformer_context)
    tf_pred = jnp.argmax(tf_logits, axis=-1).astype(ev_tg.dtype)
    mask = np.asarray(mask).astype(bool)
    ssm_ok = (np.asarray(ssm_pred) == np.asarray(ev_tg))[mask]
    tf_ok = (np.asarray(tf_pred) == np.asarray(ev_tg))[mask]
    return ssm_ok, tf_ok


def measure(seed, context, epochs):
    args = build_args(seed, context, epochs)
    ssm_ok, tf_ok = train_both(args)
    ssm_acc = float(ssm_ok.mean()); tf_acc = float(tf_ok.mean())
    oracle = float((ssm_ok | tf_ok).mean())
    best = max(ssm_acc, tf_acc)
    both_wrong = float((~ssm_ok & ~tf_ok).mean())
    return {"ssm": round(ssm_acc, 4), "tf": round(tf_acc, 4),
            "oracle": round(oracle, 4), "best_single": round(best, 4),
            "headroom": round(oracle - best, 4), "both_wrong": round(both_wrong, 4),
            "n_queries": int(ssm_ok.size)}


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    print("=== Stage 0: headroom on REAL trained Transformer + KV-memory SSM ===")
    print("REGIME A — delayed recall, bypass (context=32 < delay=96): expect headroom>0")
    a = measure(seed=11, context=32, epochs=epochs)
    print(f"  ssm={a['ssm']}  transformer={a['tf']}  oracle={a['oracle']}  "
          f"best={a['best_single']}  HEADROOM={a['headroom']}  both_wrong={a['both_wrong']}")
    # distinguish the TWO ways headroom can be zero (the key honesty)
    if a["headroom"] > 0.05:
        verdict = "ORTHOGONAL errors -> routing could pay"
    elif a["best_single"] > 0.95:
        verdict = ("SINGLE-EXPERT DOMINANCE (one expert ~perfect) -> headroom 0, "
                   "but NOT because errors correlate. The Law says: don't route, "
                   "deploy the dominant expert.")
    else:
        verdict = "CORRELATED errors (both fail together) -> headroom 0, routing useless"
    print(f"  -> {verdict}")
