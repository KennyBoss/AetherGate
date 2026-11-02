"""Causal Bernoulli evidence for occurrence of a fixed rare-token group."""
from typing import NamedTuple
import jax
import jax.numpy as jnp

class RareContextState(NamedTuple):
    keys: jax.Array
    events: jax.Array
    cursor: jax.Array
    previous: jax.Array

def initial_rare_context_state(batch_size,capacity=256):
    if batch_size<1 or capacity<1:
        raise ValueError("Context cache needs positive batch size and capacity")
    return RareContextState(jnp.full((batch_size,capacity),-1,dtype=jnp.int32),
        jnp.zeros((batch_size,capacity),dtype=jnp.bool_),jnp.zeros((batch_size,),dtype=jnp.int32),
        jnp.full((batch_size,),-1,dtype=jnp.int32))

def rare_context_logits(logits,ids,state,target_mask,*,prior_strength=8.0):
    """Update with observed INPUT transitions and then forecast the successor."""
    rows=jnp.arange(ids.shape[0]);capacity=state.keys.shape[1]
    def step(carry,pair):
        current,z=pair;slot=carry.cursor;write=carry.previous>=0
        keys=carry.keys.at[rows,slot].set(jnp.where(write,carry.previous,carry.keys[rows,slot]))
        events=carry.events.at[rows,slot].set(jnp.where(write,target_mask[current],carry.events[rows,slot]))
        next_state=RareContextState(keys,events,(slot+write.astype(jnp.int32))%capacity,current)
        matches=keys==current[:,None]
        n=jnp.sum(matches,axis=-1).astype(jnp.float32)
        positive=jnp.sum(matches & events,axis=-1).astype(jnp.float32)
        rare_mass=jax.scipy.special.logsumexp(jnp.where(target_mask,z,-jnp.inf),axis=-1)
        other_mass=jax.scipy.special.logsumexp(jnp.where(target_mask,-jnp.inf,z),axis=-1)
        total=jnp.logaddexp(rare_mass,other_mass)
        old_rare=rare_mass-total;old_other=other_mass-total
        new_rare=jnp.logaddexp(jnp.log(prior_strength)+old_rare,jnp.log(positive))-jnp.log(prior_strength+n)
        new_other=jnp.logaddexp(jnp.log(prior_strength)+old_other,jnp.log(n-positive))-jnp.log(prior_strength+n)
        # Preserve any existing winner outside the target group, including common classes.
        rare_max=jnp.max(jnp.where(target_mask,z,-jnp.inf),axis=-1)-rare_mass
        other_max=jnp.max(jnp.where(target_mask,-jnp.inf,z),axis=-1)-other_mass
        threshold=other_max-jnp.logaddexp(other_max,rare_max)
        cap=threshold+jnp.log1p(-1e-6)
        allowed=jnp.maximum(old_rare,cap)
        outside_winner=~target_mask[jnp.argmax(z,axis=-1)]
        capped=outside_winner & (new_rare>allowed)
        cap_other=jnp.where(allowed==old_rare,old_other,jnp.log1p(-jnp.exp(allowed)))
        new_rare=jnp.where(capped,allowed,new_rare);new_other=jnp.where(capped,cap_other,new_other)
        # Also retain an existing target-group winner: quality gains are probabilistic.
        floor=jnp.minimum(old_rare,threshold+jnp.log1p(1e-6))
        floored=(~outside_winner) & (new_rare<floor)
        floor_other=jnp.where(floor==old_rare,old_other,jnp.log1p(-jnp.exp(floor)))
        new_rare=jnp.where(floored,floor,new_rare);new_other=jnp.where(floored,floor_other,new_other)
        corrected=z+jnp.where(target_mask,new_rare[:,None]-old_rare[:,None],new_other[:,None]-old_other[:,None])
        return next_state,jnp.where(n[:,None]>0,corrected,z)
    final,output=jax.lax.scan(step,state,(jnp.swapaxes(ids,0,1),jnp.swapaxes(logits,0,1)))
    return jnp.swapaxes(output,0,1),final

"""Causal log-loss budget for a group-mass proposal against its base predictor.

Each protected cohort has its own per-stream credit, initially zero. Completed
forecasts earn/spend credit only when their target becomes the CURRENT input.
The next forecast cannot risk more than the least available cohort credit.
"""

class GroupBudgetState(NamedTuple):
    credit: jax.Array
    previous_other_gain: jax.Array

def initial_group_budget(batch_size, cohorts=2):
    return GroupBudgetState(jnp.zeros((batch_size,cohorts),jnp.float32),jnp.zeros((batch_size,),jnp.float32))

def protected_group_logits(base,proposal,ids,state,target_mask,protected_cohort):
    """No target argument: ids are already observed inputs, never successors."""
    rows=jnp.arange(ids.shape[0])
    def step(carry,item):
        old,new,current=item
        cohort=protected_cohort[current];valid=cohort>=0;index=jnp.maximum(cohort,0)
        credit=carry.credit.at[rows,index].add(jnp.where(valid,carry.previous_other_gain,0.))
        credit=jnp.maximum(credit,0.)
        def masses(z):
            a=jax.scipy.special.logsumexp(jnp.where(target_mask,z,-jnp.inf),-1)
            b=jax.scipy.special.logsumexp(jnp.where(target_mask,-jnp.inf,z),-1)
            return jax.nn.log_sigmoid(a-b),jax.nn.log_sigmoid(b-a)
        old_group,old_other=masses(old);new_group,new_other=masses(new)
        allowance=jnp.maximum(jnp.min(credit,axis=-1)-2e-7,0.)
        bound=old_other-allowance;limited=new_other<bound
        other=jnp.where(limited,bound,new_other)
        group=jnp.where(limited,jnp.log(-jnp.expm1(other)),new_group)
        output=old+jnp.where(target_mask,group[:,None]-old_group[:,None],other[:,None]-old_other[:,None])
        # With zero credit and an upward proposal, keep base bits exactly.
        blocked=limited & (allowance==0)
        output=jnp.where(blocked[:,None],old,output)
        changed_common=(~target_mask[jnp.argmax(old,-1)]) & (jnp.argmax(output,-1)!=jnp.argmax(old,-1))
        output=jnp.where(changed_common[:,None],old,output)
        gain=jnp.where(blocked | changed_common,0.,other-old_other)
        return GroupBudgetState(credit,gain),output
    final,z=jax.lax.scan(step,state,(jnp.swapaxes(base,0,1),jnp.swapaxes(proposal,0,1),jnp.swapaxes(ids,0,1)))
    return jnp.swapaxes(z,0,1),final


class RareOccurrenceState(NamedTuple):
    context: RareContextState
    budget: GroupBudgetState


def initial_rare_occurrence_state(batch_size,capacity=256):
    return RareOccurrenceState(initial_rare_context_state(batch_size,capacity),initial_group_budget(batch_size))


def rare_occurrence_logits(logits,ids,state,target_mask,protected_cohort,*,prior_strength=8.0):
    proposal,context=rare_context_logits(logits,ids,state.context,target_mask,prior_strength=prior_strength)
    corrected,budget=protected_group_logits(logits,proposal,ids,state.budget,target_mask,protected_cohort)
    return corrected,RareOccurrenceState(context,budget)
