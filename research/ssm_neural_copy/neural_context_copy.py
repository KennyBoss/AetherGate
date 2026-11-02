"""Jointly trained causal pointer output, shared by SSM and Transformer.

The cache holds observations, never evaluation labels or fitted probabilities.
A context key from t-1 is paired with the observed input at t before predicting t+1.
"""
from typing import NamedTuple
import math
import jax
import jax.numpy as jnp

class CopyState(NamedTuple):
    keys: jax.Array
    ids: jax.Array
    previous: jax.Array
    last_key: jax.Array
    seen: jax.Array

class ModelState(NamedTuple):
    base: object
    pointer: CopyState

def init_params(params,key,dimension=12,capacity=64):
    p=dict(params);width=p['token_embed'].shape[1];a,b=jax.random.split(key)
    p['pointer_current_w']=jax.random.normal(a,(width,dimension),dtype=jnp.float32)/math.sqrt(width)
    p['pointer_previous_w']=jax.random.normal(b,(width,dimension),dtype=jnp.float32)/math.sqrt(width)
    p['pointer_temperature_raw']=jnp.asarray(math.log(math.expm1(12.)),jnp.float32)
    p['pointer_null_bias']=jnp.asarray(math.log(capacity)+10.,jnp.float32)
    return p

def initial(params,batch,capacity=64):
    width,dimension=params['pointer_current_w'].shape
    return CopyState(jnp.zeros((batch,capacity,dimension),jnp.float32),jnp.zeros((batch,capacity),jnp.int32),jnp.zeros((batch,width),jnp.float32),jnp.zeros((batch,dimension),jnp.float32),jnp.zeros((batch,),jnp.int32))

def rms(x):
    return x*jax.lax.rsqrt(jnp.mean(x*x,-1,keepdims=True)+1e-5)

def apply(params,input_ids,base_logits,state):
    batch,capacity,_=state.keys.shape;vocab=base_logits.shape[-1];rows=jnp.arange(batch)
    embeddings=params['token_embed'][input_ids]
    temperature=jax.nn.softplus(params['pointer_temperature_raw'])
    def step(st,pair):
        ids,x,z=pair;slot=jnp.maximum(st.seen-1,0)%capacity;can_write=st.seen>0
        keys=st.keys.at[rows,slot].set(jnp.where(can_write[:,None],st.last_key,st.keys[rows,slot]))
        values=st.ids.at[rows,slot].set(jnp.where(can_write,ids,st.ids[rows,slot]))
        query=rms(x)@params['pointer_current_w']+rms(st.previous)@params['pointer_previous_w']
        query=query*jax.lax.rsqrt(jnp.sum(query*query,-1,keepdims=True)+1e-5)
        scores=jnp.einsum('bkd,bd->bk',keys,query)*temperature
        valid=jnp.arange(capacity)[None,:]<jnp.minimum(st.seen,capacity)[:,None]
        scores=jnp.where(valid,scores,-jnp.inf)
        all_scores=jnp.concatenate([scores,jnp.broadcast_to(params['pointer_null_bias'],(batch,1))],-1)
        lp=jax.nn.log_softmax(all_scores,-1)
        copy=jnp.zeros((batch,vocab),jnp.float32).at[rows[:,None],values].add(jnp.exp(lp[:,:capacity]))
        copy_log=jnp.where(copy>0,jnp.log(jnp.maximum(copy,1e-30)),-jnp.inf)
        output=jnp.logaddexp(jax.nn.log_softmax(z,-1)+lp[:,-1:],copy_log)
        return CopyState(keys,values,x,query,st.seen+1),output
    st,z=jax.lax.scan(step,state,(input_ids.T,jnp.swapaxes(embeddings,0,1),jnp.swapaxes(base_logits,0,1)))
    return jnp.swapaxes(z,0,1),st

def initial_model(params,batch,model,capacity=64):
    import train_ssm_text as c
    return ModelState(c.init_stream_state(params,batch) if model=='ssm' else None,initial(params,batch,capacity))

def forward_parts(params,input_ids,state,model):
    import train_ssm_text as c
    import compare_text_architectures as b
    if model=='ssm':z,base=c.forward(params,input_ids,state.base)
    elif model=='transformer':z=b.transformer_forward(params,input_ids,layers=1,heads=2);base=None
    else:raise ValueError('Unknown model')
    output,pointer=apply(params,input_ids,z,state.pointer)
    return output,ModelState(base,pointer),z

def forward(params,input_ids,state,model):
    z,st,_=forward_parts(params,input_ids,state,model)
    return z,st
