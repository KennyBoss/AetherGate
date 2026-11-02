"""Same isolated pair calls; explicit results may be local cells or computed registers."""
from common import previous,ROOT
import jax,jax.numpy as jnp,numpy as np
c=previous.baseline.calls.c
from reference import numpy_state
PAIRS=previous.PAIRS if hasattr(previous,'PAIRS') else __import__('itertools').permutations(range(4),2)
PAIRS=np.asarray(list(PAIRS),np.int32)
LAYOUTS=np.asarray([[0,1],[0,2],[1,2]],np.int32)

def tables(blocks):
    code=np.zeros((max(1,len(blocks)),12),np.int32);layout=np.tile([0,1],(max(1,len(blocks)),1)).astype(np.int32);returns=layout.copy()
    for i,b in enumerate(blocks):
        if not 1<=len(b['code'])<=12:raise ValueError('Invalid body length')
        code[i,:len(b['code'])]=b['code'];layout[i]=b['layout'];returns[i]=b.get('outputs',b['layout'])
    return jnp.asarray(code),jnp.asarray(layout),jnp.asarray(returns)

@jax.jit
def execute_local(codes,inputs):
    state=c.zeros(codes.shape[0],inputs.shape[1]);state=state._replace(tape=state.tape.at[:,:,:3].set(inputs))
    state=jax.lax.scan(lambda st,op:(c.step(st,op[:,None],3),None),state,codes.T)[0]
    return jnp.concatenate([state.tape[:,:,:3],state.r0[...,None],state.r1[...,None]],-1)

@jax.jit
def pair_outputs(codes,layouts,returns,x):
    local=jnp.zeros((len(codes),len(x),3),jnp.float32);rows=jnp.arange(len(codes))[:,None];cases=jnp.arange(len(x))[None,:]
    for k in range(2):local=local.at[rows,cases,layouts[:,k,None]].set(x[None,:,k])
    bank=execute_local(codes,local)
    return jnp.take_along_axis(bank,jnp.broadcast_to(returns[:,None,:],(len(codes),len(x),2)),2)

@jax.jit
def compose(genomes,x,codes,layouts,returns):
    tape=jnp.broadcast_to(x,(genomes.shape[0],*x.shape));rows=jnp.arange(genomes.shape[0])[:,None];cases=jnp.arange(x.shape[0])[None,:]
    def call(tape,tokens):
        index=jnp.maximum(tokens-1,0);block=index//12;pair=jnp.asarray(PAIRS)[index%12];layout=layouts[block]
        gathered=jnp.take_along_axis(tape,jnp.broadcast_to(pair[:,None,:],(len(tokens),len(x),2)),2);local=jnp.zeros((len(tokens),len(x),3),jnp.float32)
        for k in range(2):local=local.at[rows,cases,layout[:,k,None]].set(gathered[:,:,k])
        bank=execute_local(codes[block],local);values=jnp.take_along_axis(bank,jnp.broadcast_to(returns[block,None,:],(len(tokens),len(x),2)),2)
        for k in range(2):
            old=jnp.take_along_axis(tape,jnp.broadcast_to(pair[:,None,k:k+1],(len(tokens),len(x),1)),2)[...,0]
            tape=tape.at[rows,cases,pair[:,k,None]].set(jnp.where(tokens[:,None]>0,values[:,:,k],old))
        return tape,None
    return jax.lax.scan(call,tape,genomes.T)[0]

def numpy_pair(code,layout,returns,x):
    tape=np.zeros((len(x),8),np.float32);tape[:,layout]=x;z=np.zeros(len(x),np.float32);state=numpy_state(code,[z,z,np.zeros(len(x),np.int32),np.zeros(len(x),bool),tape],3)
    bank=np.concatenate([state[4][:,:3],state[0][:,None],state[1][:,None]],-1);return bank[:,returns]

def numpy_compose(genome,x,blocks):
    result=x.copy()
    for token in genome:
        if not token:continue
        i=int(token)-1;b=blocks[i//12];pair=PAIRS[i%12];result[:,pair]=numpy_pair(b['code'],b['layout'],b.get('outputs',b['layout']),result[:,pair])
    return result
