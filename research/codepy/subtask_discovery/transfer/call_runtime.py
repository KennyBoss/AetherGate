"""Original isolated pair ABI generalized only in the number of caller cells."""
from transfer_env import execute_local as _float_local,numpy_pair
from research.codepy.subtask_discovery.transfer.float32_vm import execute_local as _bits_local
from functools import partial
import itertools
import numpy as np
import jax,jax.numpy as jnp

def pairs(n):return np.asarray(list(itertools.permutations(range(n),2)),np.int32)

@partial(jax.jit,static_argnames=('local_executor',))
def _compose(genomes,x,codes,layouts,returns,local_executor):
    pair_table=jnp.asarray(pairs(x.shape[1]));pair_count=len(pair_table);tape=jnp.broadcast_to(x,(genomes.shape[0],*x.shape));rows=jnp.arange(genomes.shape[0])[:,None];cases=jnp.arange(x.shape[0])[None,:]
    def call(tape,tokens):
        index=jnp.maximum(tokens-1,0);block=index//pair_count;pair=pair_table[index%pair_count];layout=layouts[block];gathered=jnp.take_along_axis(tape,jnp.broadcast_to(pair[:,None,:],(len(tokens),len(x),2)),2);local=jnp.zeros((len(tokens),len(x),3),jnp.float32)
        for k in range(2):local=local.at[rows,cases,layout[:,k,None]].set(gathered[:,:,k])
        bank=local_executor(codes[block],local);values=jnp.take_along_axis(bank,jnp.broadcast_to(returns[block,None,:],(len(tokens),len(x),2)),2)
        for k in range(2):
            old=jnp.take_along_axis(tape,jnp.broadcast_to(pair[:,None,k:k+1],(len(tokens),len(x),1)),2)[...,0];tape=tape.at[rows,cases,pair[:,k,None]].set(jnp.where(tokens[:,None]>0,values[:,:,k],old))
        return tape,None
    return jax.lax.scan(call,tape,genomes.T)[0]

def float_path_safe(x,codes):
    # All DSL arithmetic is addition of binary32 values or small integers.
    # exponent>=24 gives an input grid >=2**-126; addition/rounding/clipping
    # cannot create subnormals. No negative zero is then generated either.
    words=jax.lax.bitcast_convert_type(x,jnp.uint32);magnitude=words&jnp.uint32(0x7fffffff)
    values=jnp.all((words==0)|((magnitude>=jnp.uint32(24<<23))&(magnitude<jnp.uint32(0x7f800000))))
    allowed=jnp.asarray([i in (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,28,32,33,34) for i in range(35)])
    ops=jnp.all((codes>=0)&(codes<35)&allowed[jnp.clip(codes,0,34)])
    return values&ops

@jax.jit
def compose(genomes,x,codes,layouts,returns):
    return jax.lax.cond(float_path_safe(x,codes),
        lambda _: _compose(genomes,x,codes,layouts,returns,_float_local),
        lambda _: _compose(genomes,x,codes,layouts,returns,_bits_local),None)

def numpy_compose(program,x,blocks):
    out=x.copy();addresses=pairs(x.shape[1])
    for token in program:
        if not token:continue
        block,pair_index=divmod(int(token)-1,len(addresses));b=blocks[block];p=addresses[pair_index];out[:,p]=numpy_pair(b['code'],b['layout'],b.get('outputs',b['layout']),out[:,p])
    return out
