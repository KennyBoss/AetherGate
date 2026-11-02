"""Bounded causal attention for the experimental conv-local-attention SSM.

The window includes the current input token. Projected keys/values persist
across chunks; padding and all future tokens are masked. Training uses dense
masked chunk matrices, not a specialized sparse attention kernel.
"""
import jax
import jax.numpy as jnp


def local_attention(params, embeddings, cached_keys, cached_values, count):
    window = params["local_position_bias"].shape[0]
    history = window - 1
    length = embeddings.shape[1]
    normalized = embeddings * jax.lax.rsqrt(jnp.mean(embeddings ** 2, axis=-1, keepdims=True) + 1e-5)
    queries = normalized @ params["local_q"]
    keys = jnp.concatenate((cached_keys, normalized @ params["local_k"]), axis=1)
    values = jnp.concatenate((cached_values, normalized @ params["local_v"]), axis=1)
    # Source positions -history .. length-1; target positions 0 .. length-1.
    source = jnp.arange(history + length) - history
    distance = jnp.arange(length)[:, None] - source[None, :]
    valid = ((distance >= 0) & (distance < window))[None, :, :]
    valid = valid & (source[None, None, :] >= -count[:, None, None])
    scores = jnp.einsum("btd,bsd->bts", queries, keys) / jnp.sqrt(float(queries.shape[-1]))
    scores = scores + params["local_position_bias"][jnp.clip(distance, 0, history)][None, :, :]
    weights = jax.nn.softmax(jnp.where(valid, scores, -jnp.inf), axis=-1)
    result = jnp.einsum("bts,bsd->btd", weights, values)
    # Explicit slice handles window=1: -0 would incorrectly keep the whole chunk.
    return (result, keys[:, keys.shape[1] - history:], values[:, values.shape[1] - history:],
            jnp.minimum(count + length, history))
