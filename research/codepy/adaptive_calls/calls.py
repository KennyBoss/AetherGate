"""Evolvable argument binding and gates for inferred CodePy contracts.

Fields: action, r0 source, r1 source, local pointer, predicate source,
permutation of four data cells, call gate. Identity fields are all zero.
Only library calls use these fields; primitive instructions remain unchanged.
"""
import os
os.environ['JAX_PLATFORM_NAME']='cpu';os.environ['JAX_PLATFORMS']='cpu'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'block_contracts'))
import contracts as c
from reference import numpy_state
import itertools
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

PERMUTATIONS=np.asarray([list(p)+[4,5,6,7] for p in itertools.permutations(range(4))],np.int32)
BOUNDS=(10,10,5,6,24,8)


@jax.jit
def metadata_at_steps(genomes,lengths):
    actions=genomes[:,:,0];sizes=lengths[actions]
    starts=jnp.cumsum(sizes,axis=1)-sizes
    positions=starts[:,:,None]+jnp.arange(8)[None,None,:]
    positions=jnp.where((jnp.arange(8)[None,None,:]<sizes[:,:,None])&(positions<72),positions,72)
    ids=jnp.broadcast_to(jnp.arange(72)[None,:,None],positions.shape)
    call_ids=jnp.zeros((genomes.shape[0],72),jnp.int32).at[jnp.arange(genomes.shape[0])[:,None,None],positions].set(ids,mode='drop')
    return jnp.take_along_axis(genomes,call_ids[:,:,None],axis=1)


def binding(g,fields):
    a0,a1,ptr_src,pred_src,perm_id,gate_id=fields
    mapping=jnp.broadcast_to(jnp.asarray(PERMUTATIONS)[perm_id][:,None,:],g.tape.shape)
    inverse=jnp.argsort(mapping,axis=-1)
    tape=jnp.take_along_axis(g.tape,mapping,axis=-1)
    original_local_ptr=jnp.take_along_axis(inverse,g.ptr[...,None],axis=-1)[...,0]
    pointer=jnp.where(ptr_src[:,None]==0,original_local_ptr,jnp.broadcast_to((ptr_src-1)[:,None],g.ptr.shape))
    cell=jnp.take_along_axis(g.tape,g.ptr[...,None],axis=-1)[...,0]
    values=jnp.stack((g.r0,g.r1,*[g.tape[:,:,i] for i in range(4)],cell,
                       jnp.zeros_like(g.r0),jnp.ones_like(g.r0),-jnp.ones_like(g.r0)),axis=-1)
    r0=jnp.take_along_axis(values,jnp.broadcast_to(a0[:,None,None],g.r0.shape+(1,)),axis=-1)[...,0]
    r1_values=values.at[:,:,0].set(g.r1).at[:,:,1].set(g.r0)
    r1=jnp.take_along_axis(r1_values,jnp.broadcast_to(a1[:,None,None],g.r0.shape+(1,)),axis=-1)[...,0]
    predicates=jnp.stack((g.predicate,jnp.zeros_like(g.predicate),jnp.ones_like(g.predicate),r0>r1,r0>0,cell>r0),axis=-1)
    predicate=jnp.take_along_axis(predicates,jnp.broadcast_to(pred_src[:,None,None],g.r0.shape+(1,)),axis=-1)[...,0]
    gates=jnp.stack((jnp.ones_like(g.predicate),g.predicate,~g.predicate,g.r0>g.r1,g.r0<g.r1,
                     g.tape[:,:,0]>g.tape[:,:,1],g.tape[:,:,2]>g.tape[:,:,3],g.r0>0),axis=-1)
    gate=jnp.take_along_axis(gates,jnp.broadcast_to(gate_id[:,None,None],g.r0.shape+(1,)),axis=-1)[...,0]
    return c.State(r0,r1,pointer,predicate,tape),mapping,inverse,gate


@partial(jax.jit,static_argnames=('n','primitive_count'))
def execute(genomes,x,table,lengths,tables,n,primitive_count):
    codes,labels,prefix,begin,end=c.decode(genomes[:,:,0],table,lengths)
    metadata=metadata_at_steps(genomes,lengths)
    g=c.zeros(genomes.shape[0],x.shape[0]);g=g._replace(tape=g.tape.at[:,:,:n].set(x))
    local=c.zeros(genomes.shape[0],x.shape[0]);rin,rout,tin,tout=tables
    def body(carry,token):
        g,l=carry;op,label,length,first,last,meta=token
        bound,mapping,inverse,gate=binding(g,tuple(meta[:,i] for i in range(1,7)))
        macro=(label>=primitive_count)[:,None]
        ri=jnp.broadcast_to(rin[label,length][:,None,:],g.ptr.shape+(4,));ro=jnp.broadcast_to(rout[label,length][:,None,:],ri.shape)
        ti=tin[label[:,None],length[:,None],bound.ptr];to=tout[label[:,None],length[:,None],bound.ptr]
        l=c.where_rows(macro & first[:,None],c.capsule_input(bound,ri,ti),l)
        l=c.step(l,op[:,None],n)
        physical=c.State(l.r0,l.r1,jnp.take_along_axis(mapping,l.ptr[...,None],axis=-1)[...,0],
                         l.predicate,jnp.take_along_axis(l.tape,inverse,axis=-1))
        physical_mask=jnp.take_along_axis(to,inverse,axis=-1)
        committed=c.capsule_output(g,physical,ro,physical_mask)
        g=c.where_rows(macro & last[:,None] & gate,committed,c.where_rows(macro,g,c.step(g,op[:,None],n)))
        return (g,l),None
    data=tuple(v.T for v in (codes,labels,prefix,begin,end))+(jnp.swapaxes(metadata,0,1),)
    return jax.lax.scan(body,(g,local),data)[0][0].tape[:,:,:n]


def numpy_execute(genome,x,entries,primitive_count):
    size=len(x);state=c.State(np.zeros(size,np.float32),np.zeros(size,np.float32),np.zeros(size,np.int32),np.zeros(size,bool),np.pad(x,((0,0),(0,4))))
    remaining=72
    for gene in genome:
        if remaining<=0:break
        token,a0,a1,ptr_src,pred_src,perm_id,gate_id=map(int,gene)
        code=entries[token][:remaining];remaining-=len(code)
        if token<primitive_count:state=c.State(*numpy_state(code,state,4));continue
        mapping=PERMUTATIONS[perm_id];inverse=np.argsort(mapping)
        tape=state.tape[:,mapping];pointer=inverse[state.ptr] if ptr_src==0 else np.full(size,ptr_src-1,np.int32)
        cell=state.tape[np.arange(size),state.ptr]
        vals=[state.r0,state.r1,*[state.tape[:,i] for i in range(4)],cell,np.zeros(size,np.float32),np.ones(size,np.float32),-np.ones(size,np.float32)]
        r0=vals[a0];vals[0],vals[1]=vals[1],vals[0];r1=vals[a1]
        predicate=[state.predicate,np.zeros(size,bool),np.ones(size,bool),r0>r1,r0>0,cell>r0][pred_src]
        gate=[np.ones(size,bool),state.predicate,~state.predicate,state.r0>state.r1,state.r0<state.r1,
              state.tape[:,0]>state.tape[:,1],state.tape[:,2]>state.tape[:,3],state.r0>0][gate_id]
        bound=c.State(r0,r1,pointer,predicate,tape);contract=c.infer(code)
        local=[np.zeros_like(v) for v in bound]
        for name in contract['input_registers']:local[c.REGS.index(name)]=bound[c.REGS.index(name)].copy()
        ti=np.zeros_like(tape,bool);to=np.zeros_like(tape,bool)
        for i,p in enumerate(pointer):
            footprint=contract['memory_by_initial_pointer'][int(p)]
            ti[i,footprint['input_cells']]=True;to[i,footprint['output_cells']]=True
        local[4]=np.where(ti,tape,0);local=list(numpy_state(code,local,4))
        local[2]=mapping[local[2]];local[4]=local[4][:,inverse];to=to[:,inverse]
        committed=[v.copy() for v in state]
        for name in contract['output_registers']:
            index=c.REGS.index(name);committed[index]=np.where(gate,local[index],committed[index])
        committed[4]=np.where(to & gate[:,None],local[4],state.tape)
        state=c.State(*committed)
    return state.tape[:,:4]
