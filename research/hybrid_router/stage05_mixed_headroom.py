"""Stage 0.5 — the headroom hunt on REAL models. Build a MIXED task whose stream
interleaves each architecture's exclusive strength, train the harness's real
context-capped Transformer + real KV-memory SSM on it, and measure whether
NEITHER dominates (headroom > 0).

Two query types in one stream (explicit query mask, harness format):
  RECALL : a key appears late; its binding was written far back (> transformer
           context) -> memory retrieves it, the capped Transformer is blind.
  LOCAL-COPY : copy the token `offset` positions back (in the Transformer window)
           -> attention does positional copy cleanly; a retrieval-routed memory
           has no slot for it.
Prediction: SSM wins RECALL, Transformer wins LOCAL -> complementary -> headroom>0,
best_single < 1. If instead one model aces both, these architectures are NOT
complementary here (honest negative -> pivot to efficiency frontier).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import compare_long_context_recall as H


def generate_mixed(records, seq_len, n_keys, n_locals, n_queries, offset, ctx, seed):
    rng = np.random.default_rng(seed)
    FILL = 0
    KEY0 = 1
    VAL0 = KEY0 + n_keys
    LQ_MARK = VAL0 + n_keys
    LOC0 = LQ_MARK + 1
    vocab = LOC0 + n_locals
    X = np.zeros((records, seq_len), dtype=np.int32)
    Y = np.zeros((records, seq_len), dtype=np.int32)
    M = np.zeros((records, seq_len), dtype=np.float32)
    is_recall = np.zeros((records, n_queries), dtype=bool)
    qpos_all = np.zeros((records, n_queries), dtype=np.int32)
    for r in range(records):
        # bindings written early (far before the query region)
        val_of = {k: VAL0 + int(rng.integers(n_keys)) for k in range(n_keys)}
        pos = 0
        bpos = {}
        order = rng.permutation(n_keys)
        for k in order:
            X[r, pos] = KEY0 + k
            X[r, pos + 1] = val_of[k]
            bpos[k] = pos
            pos += 2 + int(rng.integers(0, 2))
        # query region: last positions, each binding is now > ctx away
        qstart = seq_len - 2 * n_queries
        for j in range(n_queries):
            qpos = qstart + 2 * j + 1
            qpos_all[r, j] = qpos
            if rng.random() < 0.5:  # RECALL
                k = int(rng.integers(n_keys))
                assert qpos - bpos[k] > ctx
                X[r, qpos] = KEY0 + k
                Y[r, qpos] = val_of[k]
                is_recall[r, j] = True
            else:                   # LOCAL-COPY (token `offset` back, in window)
                src = LOC0 + int(rng.integers(n_locals))
                X[r, qpos - offset] = src
                X[r, qpos] = LQ_MARK
                Y[r, qpos] = src
            M[r, qpos] = 1.0
    return X, Y, M, is_recall, qpos_all, vocab


def build_args(seed, ctx, epochs):
    sys.argv = ["x", "--task", "assignment", "--transformer-context", str(ctx),
                "--epochs", str(epochs), "--batch-size", "128", "--learning-rate", "0.006",
                "--ssm-variant", "kv-memory", "--kv-logit-scale", "64", "--ssm-decay-init", "8.0",
                "--seed", str(seed)]
    return H.parse_args()


def per_query_correct(pred, Y, M):
    pred = np.asarray(pred); Y = np.asarray(Y); M = np.asarray(M).astype(bool)
    return (pred == Y), M


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    ctx = 32
    seq_len = 96
    args = build_args(11, ctx, epochs)
    trX, trY, trM, _, _, vocab = generate_mixed(2048, seq_len, 8, 8, 4, 2, ctx, seed=11)
    evX, evY, evM, ev_recall, ev_qpos, _ = generate_mixed(1024, seq_len, 8, 8, 4, 2, ctx, seed=12)
    trX_j, trY_j, trM_j = jnp.asarray(trX), jnp.asarray(trY), jnp.asarray(trM)
    evX_j, evY_j, evM_j = jnp.asarray(evX), jnp.asarray(evY), jnp.asarray(evM)

    key = jax.random.PRNGKey(11)
    ssm_key, tf_key = jax.random.split(key)
    ssm = H.init_ssm_params(ssm_key, vocab, args.ssm_input_dim, args.ssm_state_dim,
                            variant=args.ssm_variant, skip_rank=args.ssm_skip_rank)
    ssm["decay_raw"] = jnp.full(ssm["decay_raw"].shape, args.ssm_decay_init, dtype=jnp.float32)
    ssm, _, _, _ = H.train_model(params=ssm, train_inputs=trX_j, train_targets=trY_j,
                                 args=args, model="ssm", train_query_mask=trM_j)

    d_model = H.choose_transformer_d_model(
        target_params=H.count_params(ssm), vocab_size=vocab, seq_len=seq_len,
        layers=args.transformer_layers, heads=args.transformer_heads,
        ff_mult=args.transformer_ff_mult, max_d_model=args.transformer_max_d_model)
    tf = H.init_transformer_params(tf_key, vocab_size=vocab, seq_len=seq_len, d_model=d_model,
                                   layers=args.transformer_layers, ff_mult=args.transformer_ff_mult)
    tf, _, _, _ = H.train_model(params=tf, train_inputs=trX_j, train_targets=trY_j,
                                args=args, model="transformer", train_query_mask=trM_j)

    ssm_pred, _, _ = H.ssm_confusion(ssm, evX_j, evY_j, evM_j)
    tf_logits = H.transformer_forward_limited(tf, evX_j, layers=args.transformer_layers,
                                              heads=args.transformer_heads, context=ctx)
    tf_pred = jnp.argmax(tf_logits, axis=-1).astype(evY_j.dtype)

    ssm_ok = (np.asarray(ssm_pred) == evY)
    tf_ok = (np.asarray(tf_pred) == evY)
    mask = evM.astype(bool)

    def split(ok_arr):
        recall_mask = mask & np.zeros_like(mask)
        loc_mask = mask & np.zeros_like(mask)
        for r in range(evX.shape[0]):
            for j in range(ev_qpos.shape[1]):
                q = ev_qpos[r, j]
                (recall_mask if ev_recall[r, j] else loc_mask)[r, q] = True
        return ok_arr[recall_mask].mean(), ok_arr[loc_mask].mean()

    ssm_recall, ssm_local = split(ssm_ok)
    tf_recall, tf_local = split(tf_ok)
    ssm_all = ssm_ok[mask].mean(); tf_all = tf_ok[mask].mean()
    oracle = (ssm_ok | tf_ok)[mask].mean(); best = max(ssm_all, tf_all)

    print("=== Stage 0.5: mixed-task headroom on REAL models ===")
    print(f"            {'RECALL':>8s} {'LOCAL':>8s} {'ALL':>8s}")
    print(f"  SSM       {ssm_recall:8.3f} {ssm_local:8.3f} {ssm_all:8.3f}")
    print(f"  Transform {tf_recall:8.3f} {tf_local:8.3f} {tf_all:8.3f}")
    print(f"  oracle={oracle:.3f}  best_single={best:.3f}  HEADROOM={oracle-best:.3f}")
    if oracle - best > 0.05 and best < 0.95:
        print("  -> NEITHER DOMINATES + headroom>0: real complementary regime FOUND. Stage 1 unlocked.")
    elif best > 0.95:
        print("  -> one model dominates -> no headroom (architectures not complementary here).")
    else:
        print("  -> low headroom; complementarity weak.")


if __name__ == "__main__":
    main()
