"""Causal sparse transition evidence within a fixed output group.

Completed input transitions are the only observations. Group probability mass
and every other class probability are preserved; an existing common winner is
protected against being overtaken by a sharpened rare class.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp


class RareTransitionState(NamedTuple):
    keys: jax.Array
    values: jax.Array
    cursor: jax.Array
    previous: jax.Array


def initial_rare_transition_state(batch_size, capacity=256):
    if batch_size < 1 or capacity < 1:
        raise ValueError("Transition cache needs positive batch size and capacity")
    return RareTransitionState(jnp.full((batch_size, capacity), -1, dtype=jnp.int32),
        jnp.zeros((batch_size, capacity), dtype=jnp.int32),jnp.zeros((batch_size,),dtype=jnp.int32),
        jnp.full((batch_size,),-1,dtype=jnp.int32))


def rare_transition_logits(logits, ids, state, target_mask, *, prior_strength=8.0):
    """Return corrected logits and state; no next-target argument is accepted."""
    vocab=logits.shape[-1];batch=ids.shape[0];capacity=state.keys.shape[1];rows=jnp.arange(batch)
    def step(carry, pair):
        current,z=pair
        write=target_mask[current] & (carry.previous>=0)
        slot=carry.cursor
        keys=carry.keys.at[rows,slot].set(jnp.where(write,carry.previous,carry.keys[rows,slot]))
        values=carry.values.at[rows,slot].set(jnp.where(write,current,carry.values[rows,slot]))
        next_state=RareTransitionState(keys,values,(slot+write.astype(jnp.int32))%capacity,current)
        matches=(keys==current[:,None]).astype(jnp.float32)
        counts=jax.vmap(lambda v,w:jnp.bincount(v,weights=w,length=vocab))(values,matches)
        n=jnp.sum(counts,axis=-1)
        masked=jnp.where(target_mask,z,-jnp.inf)
        mass=jax.scipy.special.logsumexp(masked,axis=-1)
        mass=jnp.where(jnp.any(target_mask),mass,0.)
        old_q=jnp.exp(jnp.where(target_mask,z-mass[:,None],-jnp.inf))
        cache_q=counts/jnp.maximum(n[:,None],1.)
        weight=n/(prior_strength+n)
        proposed=(1-weight[:,None])*old_q+weight[:,None]*cache_q
        # Protect the old common argmax while retaining the same rare-group mass.
        common_max=jnp.max(jnp.where(target_mask,-jnp.inf,z),axis=-1)
        cap=jnp.exp(jnp.minimum(common_max-mass,0.))*(1.-1e-6)
        delta=proposed-old_q
        limits=jnp.where(target_mask & (delta>0),(cap[:,None]-old_q)/jnp.maximum(delta,1e-30),1.)
        fraction=jnp.clip(jnp.min(limits,axis=-1),0.,1.)
        old_winner_common=~target_mask[jnp.argmax(z,axis=-1)]
        fraction=jnp.where(old_winner_common,fraction,1.)
        effective=weight*fraction
        corrected=jnp.logaddexp(jnp.log1p(-effective[:,None])+z,
            jnp.log(effective[:,None])+jnp.log(cache_q)+mass[:,None])
        result=jnp.where(target_mask & (effective[:,None]>0),corrected,z)
        return next_state,result
    final,adjusted=jax.lax.scan(step,state,(jnp.swapaxes(ids,0,1),jnp.swapaxes(logits,0,1)))
    return jnp.swapaxes(adjusted,0,1),final
