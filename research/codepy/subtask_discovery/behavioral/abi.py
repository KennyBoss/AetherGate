"""Generic isolated pair ABI. Every executable operation comes from DSL bytecode."""
import os,sys,importlib.util,itertools
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_PLATFORM_NAME']='cpu'
from pathlib import Path
from functools import partial
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'research/codepy/autonomous_blocks'))
import run_experiment as vm
import jax,jax.numpy as jnp,numpy as np
MAX_BODY=12
PAIRS=np.asarray(list(itertools.permutations(range(4),2)),np.int32)
LAYOUTS=np.asarray(list(itertools.combinations(range(3),2)),np.int32)

def tables(blocks):
    codes=np.zeros((max(1,len(blocks)),MAX_BODY),np.int32);layouts=np.zeros((max(1,len(blocks)),2),np.int32)
    layouts[:]=[0,1]
    for i,b in enumerate(blocks):
        if not 1<=len(b['code'])<=MAX_BODY:raise ValueError('Block length outside budget')
        codes[i,:len(b['code'])]=b['code'];layouts[i]=b['layout']
    return jnp.asarray(codes),jnp.asarray(layouts)

@jax.jit
def pair_outputs(codes,layouts,x):
    local=jnp.zeros((codes.shape[0],x.shape[0],3),jnp.float32)
    rows=jnp.arange(codes.shape[0])[:,None];cases=jnp.arange(x.shape[0])[None,:]
    for k in range(2):local=local.at[rows,cases,layouts[:,k,None]].set(x[None,:,k])
    result=jax.vmap(lambda code,inputs:vm.execute(code[None],inputs,3)[0])(codes,local)
    return jnp.take_along_axis(result,jnp.broadcast_to(layouts[:,None,:],(codes.shape[0],x.shape[0],2)),2)

@jax.jit
def compose(genomes,x,codes,layouts):
    tapes=jnp.broadcast_to(x,(genomes.shape[0],*x.shape));rows=jnp.arange(genomes.shape[0])[:,None];cases=jnp.arange(x.shape[0])[None,:]
    def call(tape,tokens):
        index=jnp.maximum(tokens-1,0);block=index//12;pair=jnp.asarray(PAIRS)[index%12]
        gathered=jnp.take_along_axis(tape,jnp.broadcast_to(pair[:,None,:],(len(tokens),x.shape[0],2)),2)
        local=jnp.zeros((len(tokens),x.shape[0],3),jnp.float32);layout=layouts[block]
        for k in range(2):local=local.at[rows,cases,layout[:,k,None]].set(gathered[:,:,k])
        output=jax.vmap(lambda code,inputs:vm.execute(code[None],inputs,3)[0])(codes[block],local)
        for k in range(2):
            value=jnp.take_along_axis(output,jnp.broadcast_to(layout[:,None,k:k+1],(len(tokens),x.shape[0],1)),2)[...,0]
            old=jnp.take_along_axis(tape,jnp.broadcast_to(pair[:,None,k:k+1],(len(tokens),x.shape[0],1)),2)[...,0]
            tape=tape.at[rows,cases,pair[:,k,None]].set(jnp.where(tokens[:,None]>0,value,old))
        return tape,None
    return jax.lax.scan(call,tapes,genomes.T)[0]

def numpy_compose(genome,x,blocks):
    result=x.copy()
    for token in genome:
        if not token:continue
        index=int(token)-1;block=blocks[index//12];pair=PAIRS[index%12];local=np.zeros((len(x),3),np.float32);local[:,block['layout']]=result[:,pair]
        # Independent VM resets all registers, pointer and scratch on each call.
        out=vm.numpy_execute(block['code'],local);result[:,pair]=out[:,block['layout']]
    return result
