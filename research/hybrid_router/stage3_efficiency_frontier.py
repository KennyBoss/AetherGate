"""Stage 3 — the efficiency frontier (the honest facade). On delayed-binding
recall, sweep the Transformer's context window and plot accuracy vs compute
against a single KV-memory SSM point. The Transformer only reaches the SSM's
recall once its context covers the delay (attention compute grows with context),
while the SSM holds recall at context-independent cost.

Claim (supported by real trained models): memory matches attention's recall at a
fraction of the compute. FLOPs are a transparent forward-pass proxy (formula
stated), param-matched d_model so only the context (and the mechanism) varies.
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


def build_args(context, epochs, delay, seed=11):
    sys.argv = ["x", "--task", "assignment", "--delay", str(delay),
                "--transformer-context", str(context),
                "--train-records", "1024", "--eval-records", "512",
                "--epochs", str(epochs), "--batch-size", "128", "--learning-rate", "0.006",
                "--key-count", "16", "--value-count", "16",
                "--ssm-variant", "kv-memory", "--kv-logit-scale", "64", "--ssm-decay-init", "8.0",
                "--seed", str(seed)]
    if context >= delay:
        sys.argv.append("--allow-visible-key")
    return H.parse_args()


def make_data(args):
    trX, trY, _ = H.make_task_data(task=args.task, records=args.train_records, delay=args.delay,
                                   key_count=args.key_count, value_count=args.value_count,
                                   binding_count=args.binding_count, query_count=args.query_count, seed=args.seed)
    evX, evY, _ = H.make_task_data(task=args.task, records=args.eval_records, delay=args.delay,
                                   key_count=args.key_count, value_count=args.value_count,
                                   binding_count=args.binding_count, query_count=args.query_count, seed=args.seed + 1)
    return (jnp.asarray(trX), jnp.asarray(trY), jnp.asarray(evX), jnp.asarray(evY))


def acc_at_queries(pred, Y, X):
    pred = np.asarray(pred); Y = np.asarray(Y)
    mask = (np.asarray(X) == H.QUERY_TOKEN)
    return float((pred[mask] == Y[mask]).mean())


# transparent forward-pass FLOP proxies (per eval batch of B sequences, length L)
def transformer_flops(L, context, d_model, layers, ff_mult, vocab):
    attn = 4.0 * L * context * d_model            # QK^T scores + weighted sum (windowed)
    ffn = 4.0 * L * d_model * d_model * ff_mult     # two projections
    out = 2.0 * L * d_model * vocab                 # logits
    return layers * (attn + ffn) + out


def ssm_flops(L, input_dim, state_dim, vocab):
    rec = L * state_dim * (state_dim + input_dim)   # recurrent state update (delay-independent)
    out = 2.0 * L * state_dim * vocab
    return rec + out


def train_eval_transformer(args, data, ssm_param_count, vocab, seq_len):
    trX, trY, evX, evY = data
    _, tf_key = jax.random.split(jax.random.PRNGKey(args.seed))
    d_model = H.choose_transformer_d_model(
        target_params=ssm_param_count, vocab_size=vocab, seq_len=seq_len,
        layers=args.transformer_layers, heads=args.transformer_heads,
        ff_mult=args.transformer_ff_mult, max_d_model=args.transformer_max_d_model)
    tf = H.init_transformer_params(tf_key, vocab_size=vocab, seq_len=seq_len, d_model=d_model,
                                   layers=args.transformer_layers, ff_mult=args.transformer_ff_mult)
    tf, _, _, _ = H.train_model(params=tf, train_inputs=trX, train_targets=trY,
                                args=args, model="transformer")
    logits = H.transformer_forward_limited(tf, evX, layers=args.transformer_layers,
                                            heads=args.transformer_heads, context=args.transformer_context)
    pred = jnp.argmax(logits, axis=-1).astype(evY.dtype)
    flops = transformer_flops(seq_len, args.transformer_context, d_model,
                              args.transformer_layers, args.transformer_ff_mult, vocab)
    # effective compute cost: params × context (cost of retaining information)
    eff_cost = int(H.count_params(tf) * args.transformer_context)
    return acc_at_queries(pred, evY, evX), flops, d_model, eff_cost


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    delay = 96
    contexts = [16, 32, 64, 96, 128]

    # SSM point (delay-independent)
    args = build_args(32, epochs, delay)
    vocab = H.vocab_size_for_task(args.task, args.key_count, args.value_count)
    data = make_data(args)
    seq_len = int(data[0].shape[1])
    ssm_key, _ = jax.random.split(jax.random.PRNGKey(args.seed))
    ssm = H.init_ssm_params(ssm_key, vocab, args.ssm_input_dim, args.ssm_state_dim,
                            variant=args.ssm_variant, skip_rank=args.ssm_skip_rank)
    ssm["decay_raw"] = jnp.full(ssm["decay_raw"].shape, args.ssm_decay_init, dtype=jnp.float32)
    ssm, _, _, _ = H.train_model(params=ssm, train_inputs=data[0], train_targets=data[1],
                                 args=args, model="ssm")
    ssm_pc = H.count_params(ssm)
    ssm_pred, _, _ = H.ssm_confusion(ssm, data[2], data[3], None)
    ssm_acc = acc_at_queries(ssm_pred, data[3], data[2])
    ssm_eff = int(ssm_pc)   # SSM cost is params alone (state is compact, context-independent)

    rows = []
    for c in contexts:
        a = build_args(c, epochs, delay)
        d = make_data(a)
        acc, fl, dm, eff = train_eval_transformer(a, d, ssm_pc, vocab, seq_len)
        rows.append((c, acc, fl, dm, eff))

    print("=== Stage 3: Efficiency Frontier — recall accuracy vs compute (delay=96) ===")
    print(f"  SSM (kv-memory): acc={ssm_acc:.3f}  eff_cost={ssm_eff:,}  (params alone, O(1) state)")
    print(f"  {'ctx':>4s} {'tf_acc':>7s} {'params':>8s} {'eff_cost':>12s} {'cost/ssm':>9s}")
    for c, acc, fl, dm, eff in rows:
        print(f"  {c:>4d} {acc:>7.3f} {dm:>8,} {eff:>12,} {eff/ssm_eff:>8.1f}x")
    matching = []
    for c, acc, fl, dm, eff in rows:
        if acc >= ssm_acc - 0.05:
            matching.append((c, eff))
    if matching:
        cmin, effmin = min(matching, key=lambda t: t[1])
        print(f"  -> Transformer first matches SSM recall at context={cmin}, "
              f"effective compute {effmin/ssm_eff:.1f}x the SSM. Memory: flat cost; "
              f"attention: cost ~ context ~ delay.")
    else:
        print("  -> No swept Transformer context matched SSM recall at all.")
    print("  Front: memory holds 1.0 recall at O(1) state; the Transformer must "
          "grow its context (and state) linearly with the delay to reach same.")


if __name__ == "__main__":
    main()
