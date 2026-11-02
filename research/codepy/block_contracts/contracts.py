"""Conservative inferred contracts and isolated execution for straight-line DSL.

All read-before-write registers are explicit arguments. Tape footprints are
computed exactly for each of eight initial pointers, including clipping.
All written state is returned (a conservative, context-independent live-out).
Valid states: finite float32 registers/tape in [-1e6,1e6], pointer0..7.
"""
import os
os.environ['JAX_PLATFORM_NAME']='cpu';os.environ['JAX_PLATFORMS']='cpu'
from typing import NamedTuple
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

REGS=('r0','r1','ptr','predicate')
READS={0:(),1:(),2:(),3:(2,),4:(2,),5:(0,2),6:(1,2),7:(2,),8:(2,),9:(),
 10:(0,1),11:(0,),12:(0,),13:(0,),14:(1,),15:(2,),16:(0,3),17:(0,1,3),
 20:(0,2),21:(1,2),22:(2,),23:(2,),24:(2,),25:(0,2,3),26:(1,2,3),
 27:(0,2,3),28:(1,2,3),32:(1,),33:(0,),34:(0,1,3)}
WRITES={1:(0,),2:(1,),3:(0,),4:(1,),7:(2,),8:(2,),9:(2,),10:(0,),11:(0,),12:(0,),
 13:(3,),14:(3,),15:(3,),16:(0,),17:(0,),20:(3,),21:(3,),22:(3,),23:(0,),24:(1,),
 25:(0,),26:(1,),27:(0,),28:(1,),32:(0,),33:(1,),34:(1,)}
TAPE_READ={3,4,20,21,25,26}


def infer(code):
    inputs=set();outputs=set()
    for op in code:
        if op not in READS:raise ValueError('Only straight-line primitive subset is supported')
        inputs.update(set(READS[op])-outputs);outputs.update(WRITES.get(op,()))
    memory=[]
    for initial in range(8):
        ptr=initial;reads=set();writes=set()
        for op in code:
            if op in TAPE_READ and ptr not in writes:reads.add(ptr)
            if op in (5,6):writes.add(ptr)
            if op==7:ptr=min(ptr+1,7)
            if op==8:ptr=max(ptr-1,0)
            if op==9:ptr=0
        memory.append({'initial_ptr':initial,'input_cells':sorted(reads),'output_cells':sorted(writes)})
    return {'code':list(code),'input_registers':[REGS[i] for i in sorted(inputs)],
        'output_registers':[REGS[i] for i in sorted(outputs)],'input_length_required':any(o in (15,22) for o in code),
        'memory_by_initial_pointer':memory,'output_policy':'all written state, conservative live-out'}


def contract_tables(entries):
    rin=np.zeros((len(entries),9,4),bool);rout=np.zeros_like(rin)
    tin=np.zeros((len(entries),9,8,8),bool);tout=np.zeros_like(tin)
    for i,code in enumerate(entries):
        for length in range(1,len(code)+1):
            c=infer(code[:length])
            for name in c['input_registers']:rin[i,length,REGS.index(name)]=True
            for name in c['output_registers']:rout[i,length,REGS.index(name)]=True
            for row in c['memory_by_initial_pointer']:
                tin[i,length,row['initial_ptr'],row['input_cells']]=True
                tout[i,length,row['initial_ptr'],row['output_cells']]=True
    return tuple(jnp.asarray(a) for a in (rin,rout,tin,tout))


class State(NamedTuple):
    r0:object;r1:object;ptr:object;predicate:object;tape:object


def zeros(population,cases):
    shape=(population,cases)
    return State(jnp.zeros(shape),jnp.zeros(shape),jnp.zeros(shape,jnp.int32),
        jnp.zeros(shape,bool),jnp.zeros(shape+(8,)))


def step(s,op,n):
    a,b,p,q,t=s;cell=jnp.take_along_axis(t,p[...,None],axis=2)[...,0]
    choose=lambda i,v,old:jnp.where(op==i,v,old)
    r0=choose(1,0.,a);r1=choose(2,0.,b)
    r0=choose(3,cell,r0);r1=choose(4,cell,r1)
    one=jax.nn.one_hot(p,8)
    tape=jnp.where((op==5)[...,None],t*(1-one)+a[...,None]*one,t)
    tape=jnp.where((op==6)[...,None],t*(1-one)+b[...,None]*one,tape)
    ptr=jnp.clip(choose(9,0,choose(8,p-1,choose(7,p+1,p))),0,7)
    for code,value in ((10,a+b),(11,a+1.),(12,a-1.),(23,p.astype(jnp.float32)),(32,b),
        (16,a+q.astype(jnp.float32)),(17,jnp.where(q,b,a)),(25,jnp.where(q,cell,a)),(27,jnp.where(q,p,a))):
        r0=choose(code,value,r0)
    for code,value in ((24,p.astype(jnp.float32)),(33,a),(26,jnp.where(q,cell,b)),
        (28,jnp.where(q,p,b)),(34,jnp.where(q,a,b))):r1=choose(code,value,r1)
    predicate=q
    for code,value in ((13,a>0),(14,b>0),(15,p<n),(20,cell>a),(21,cell>b),(22,p<n-1)):
        predicate=choose(code,value,predicate)
    return State(jnp.clip(r0,-1e6,1e6),jnp.clip(r1,-1e6,1e6),ptr,predicate,jnp.clip(tape,-1e6,1e6))


def where_rows(mask,a,b):
    return State(*(jnp.where(mask[...,None] if x.ndim==3 else mask,x,y) for x,y in zip(a,b)))


def capsule_input(s,rin,tin):
    return State(*(jnp.where(rin[...,i],x,jnp.zeros_like(x)) for i,x in enumerate(s[:4])),jnp.where(tin,s.tape,0))


def capsule_output(global_state,local,rout,tout):
    return State(*(jnp.where(rout[...,i],x,y) for i,(x,y) in enumerate(zip(local[:4],global_state[:4]))),
        jnp.where(tout,local.tape,global_state.tape))


@partial(jax.jit,static_argnames=('n',))
def run_block(state,code,tables,n):
    rin,rout,tin,tout=tables;length=len(code)
    ri=jnp.broadcast_to(rin[0,length],state.ptr.shape+(4,));ro=jnp.broadcast_to(rout[0,length],ri.shape)
    ti=tin[0,length,state.ptr];to=tout[0,length,state.ptr]
    local=capsule_input(state,ri,ti)
    local=jax.lax.scan(lambda s,o:(step(s,jnp.broadcast_to(o,s.ptr.shape),n),None),local,code)[0]
    return capsule_output(state,local,ro,to)


@jax.jit
def decode(genomes,table,lengths):
    sizes=lengths[genomes];starts=jnp.cumsum(sizes,axis=1)-sizes
    positions=starts[:,:,None]+jnp.arange(8)[None,None,:]
    valid=(jnp.arange(8)[None,None,:]<sizes[:,:,None])&(positions<72)
    scatter=jnp.where(valid,positions,72);batch=jnp.arange(genomes.shape[0])[:,None,None]
    clipped=jnp.minimum(sizes,72-starts)
    fields=(table[genomes],jnp.broadcast_to(genomes[:,:,None],positions.shape),
        jnp.broadcast_to(clipped[:,:,None],positions.shape),
        jnp.broadcast_to((jnp.arange(8)[None,None,:]==0),positions.shape),
        jnp.arange(8)[None,None,:]==clipped[:,:,None]-1)
    return tuple(jnp.zeros((genomes.shape[0],72),v.dtype).at[batch,scatter].set(v,mode='drop') for v in fields)


@partial(jax.jit,static_argnames=('n','primitive_count'))
def execute(genomes,x,table,lengths,tables,n,primitive_count):
    codes,labels,prefix,begin,end=decode(genomes,table,lengths)
    global_state=zeros(genomes.shape[0],x.shape[0]);global_state=global_state._replace(tape=global_state.tape.at[:,:,:n].set(x))
    local=zeros(genomes.shape[0],x.shape[0]);rin,rout,tin,tout=tables
    def body(carry,token):
        g,l=carry;op,label,length,first,last=token
        macro=(label>=primitive_count)[:,None]
        ri=jnp.broadcast_to(rin[label,length][:,None,:],g.ptr.shape+(4,))
        ro=jnp.broadcast_to(rout[label,length][:,None,:],ri.shape)
        ti=tin[label[:,None],length[:,None],g.ptr];to=tout[label[:,None],length[:,None],g.ptr]
        l=where_rows(macro & first[:,None],capsule_input(g,ri,ti),l)
        l=step(l,op[:,None],n)
        g=where_rows(macro & last[:,None],capsule_output(g,l,ro,to),
            where_rows(macro,g,step(g,op[:,None],n)))
        return (g,l),None
    return jax.lax.scan(body,(global_state,local),tuple(v.T for v in (codes,labels,prefix,begin,end)))[0][0].tape[:,:,:n]
