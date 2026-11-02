"""Bit-preserving JAX execution of the existing finite-float32 instruction ABI.

Copy/select/compare operate on words; addition uses guard/round/sticky bits.
The bounded mode exists only for historical instruction-reference tests. Product
entry points always use the full finite binary32 range without saturation.
No floating-point operation observes a subnormal operand on the CPU.
"""
from functools import partial
import itertools
import numpy as np
import jax
import jax.numpy as jnp

SIGN=np.uint32(0x80000000);MAG=np.uint32(0x7fffffff);ONE=np.uint32(0x3f800000);CLIP=np.uint32(0x49742400)

def needed(x):
    x=np.asarray(x)
    if x.dtype!=np.float32:return False
    # Tiny normal operands can also subtract to a subnormal result.
    words=x.view(np.uint32);mag=words&MAG
    return bool(np.any((mag>0)&(mag<0x0c800000)) or np.any(words==SIGN))

def key(a):
    a=jnp.where((a&MAG)==0,jnp.uint32(0),a)
    return jnp.where((a&SIGN)!=0,~a,a^SIGN)

def greater(a,b):return (key(a)>key(b)) & ((a&MAG)<=0x7f800000) & ((b&MAG)<=0x7f800000)
def minimum(a,b):
    return jnp.where((a&MAG)>0x7f800000,a,jnp.where((b&MAG)>0x7f800000,b,jnp.where(((a|b)&MAG)==0,a|b,jnp.where(greater(b,a),a,b))))
def maximum(a,b):
    return jnp.where((a&MAG)>0x7f800000,a,jnp.where((b&MAG)>0x7f800000,b,jnp.where(((a|b)&MAG)==0,a&b,jnp.where(greater(a,b),a,b))))
def clip(a):return jnp.where((a&MAG)>CLIP,(a&SIGN)|CLIP,a)
def jam(a,shift):
    s=jnp.minimum(shift,31).astype(jnp.uint32)
    mask=(jnp.uint32(1)<<s)-jnp.uint32(1)
    return jnp.where(shift>=32,(a!=0).astype(jnp.uint32),(a>>s)|((a&mask)!=0).astype(jnp.uint32))

def add(a,b):
    """IEEE round-to-nearest-even addition for finite words (including signed zero)."""
    aa=a&MAG;bb=b&MAG;swap=aa<bb
    a,b=jnp.where(swap,b,a),jnp.where(swap,a,b)
    ea=(a>>23)&255;eb=(b>>23)&255
    ma=((a&0x7fffff)|jnp.where(ea!=0,jnp.uint32(0x800000),jnp.uint32(0)))<<3
    mb=((b&0x7fffff)|jnp.where(eb!=0,jnp.uint32(0x800000),jnp.uint32(0)))<<3
    e=jnp.maximum(ea,1).astype(jnp.int32);delta=e-jnp.maximum(eb,1).astype(jnp.int32)
    mb=jam(mb,delta);same=((a^b)&SIGN)==0
    m=jnp.where(same,ma+mb,ma-mb)
    carry=m>=0x8000000
    m=jnp.where(carry,jam(m,jnp.ones_like(e)),m);e=e+carry.astype(jnp.int32)
    shift=jnp.minimum(jnp.maximum(jax.lax.clz(m).astype(jnp.int32)-5,0),e-1)
    m=m<<shift.astype(jnp.uint32);e=e-shift
    rounded=(m>>3)+(((m&7)>4)|(((m&7)==4)&((m&8)!=0))).astype(jnp.uint32)
    carry=rounded>=0x1000000
    rounded=jnp.where(carry,rounded>>1,rounded);e=e+carry.astype(jnp.int32)
    exponent=jnp.where(rounded<0x800000,0,e).astype(jnp.uint32)
    result=jnp.where(e>=255,jnp.uint32(0x7f800000),(exponent<<23)|(rounded&0x7fffff))
    sign=jnp.where((result==0)&~same,jnp.uint32(0),a&SIGN)
    result=result|sign
    # Overflow may create infinities inside a program; propagate IEEE specials
    # until the public finite-output check (never saturate a finite operand).
    ia=(a&MAG)==0x7f800000;ib=(b&MAG)==0x7f800000
    result=jnp.where(ia,a,jnp.where(ib,b,result))
    result=jnp.where(ia&ib&~same,jnp.uint32(0xffc00000),result)
    result=jnp.where((b&MAG)>0x7f800000,b|jnp.uint32(0x00400000),result)
    return jnp.where((a&MAG)>0x7f800000,a|jnp.uint32(0x00400000),result)

def zeros(population,cases):
    s=(population,cases)
    return (jnp.zeros(s,jnp.uint32),jnp.zeros(s,jnp.uint32),jnp.zeros(s,jnp.int32),jnp.zeros(s,bool),jnp.zeros(s+(8,),jnp.uint32))

def step(state,op,n,*,bounded=True,native_add=False):
    # All finite inputs on the 2**-126 grid remain on that grid under sums
    # and integer constants. Without initial -0, no signed-zero/FTZ hazard arises.
    plus=(lambda a,b:jax.lax.bitcast_convert_type(
        jax.lax.bitcast_convert_type(a,jnp.float32)+jax.lax.bitcast_convert_type(b,jnp.float32),jnp.uint32)) if native_add else add
    op=jnp.asarray(op,jnp.int32)
    a,b,p,q,t=state;cell=jnp.take_along_axis(t,p[...,None],2)[...,0]
    choose=lambda code,value,old:jnp.where(op==code,value,old)
    pf=jax.lax.bitcast_convert_type(p.astype(jnp.float32),jnp.uint32)
    r0=choose(1,jnp.uint32(0),a);r1=choose(2,jnp.uint32(0),b)
    for code,value in ((3,cell),(10,plus(a,b)),(11,plus(a,ONE)),(12,plus(a,ONE|SIGN)),(16,plus(a,jnp.where(q,ONE,jnp.uint32(0)))),(17,jnp.where(q,b,a)),(23,pf),(25,jnp.where(q,cell,a)),(27,jnp.where(q,pf,a)),(32,b),(35,minimum(a,b))):r0=choose(code,value,r0)
    for code,value in ((4,cell),(24,pf),(26,jnp.where(q,cell,b)),(28,jnp.where(q,pf,b)),(33,a),(34,jnp.where(q,a,b)),(35,maximum(a,b))):r1=choose(code,value,r1)
    # Address masks select integer words, never multiply the floating tape by zero/one.
    address=jnp.arange(8)==p[...,None]
    tape=jnp.where(((op==5)[...,None]&address),a[...,None],t)
    tape=jnp.where(((op==6)[...,None]&address),b[...,None],tape)
    ptr=jnp.clip(choose(9,0,choose(8,p-1,choose(7,p+1,p))),0,7)
    pred=q
    for code,value in ((13,greater(a,jnp.uint32(0))),(14,greater(b,jnp.uint32(0))),(15,p<n),(20,greater(cell,a)),(21,greater(cell,b)),(22,p<n-1)):pred=choose(code,value,pred)
    if bounded:return clip(r0),clip(r1),ptr,pred,clip(tape)
    return r0,r1,ptr,pred,tape

@partial(jax.jit,static_argnames=('bounded','native_add'))
def execute_local(codes,inputs,*,bounded=True,native_add=False):
    state=zeros(len(codes),inputs.shape[1]);state=(*state[:4],state[4].at[:,:,:3].set(inputs))
    pc=jnp.zeros(state[0].shape,jnp.int32);rows=jnp.arange(len(codes))[:,None]
    def tick(carry,_):
        state,pc=carry;op=codes[rows,pc];newpc=jnp.minimum(pc+1,codes.shape[1]-1)
        newpc=jnp.where((op==18)|((op==19)&state[3]),0,newpc)
        newpc=jnp.where((op==29)|((op==30)&state[3]),min(3,codes.shape[1]-1),newpc)
        newpc=jnp.where(op==31,pc,newpc)
        return (step(state,op,3,bounded=bounded,native_add=native_add),newpc),None
    (state,_),_=jax.lax.scan(tick,(state,pc),None,length=codes.shape[1])
    return jnp.concatenate((state[4][:,:,:3],state[0][...,None],state[1][...,None]),-1)

@partial(jax.jit,static_argnames=('bounded','native_add'))
def compose(genomes,words,codes,layouts,returns,*,bounded=True,native_add=False):
    addresses=jnp.asarray(list(itertools.permutations(range(words.shape[1]),2)),jnp.int32)
    tape=jnp.broadcast_to(words,(len(genomes),*words.shape));rows=jnp.arange(len(genomes))[:,None];cases=jnp.arange(len(words))[None,:]
    def call(tape,tokens):
        index=jnp.maximum(tokens-1,0);block=index//len(addresses);pair=addresses[index%len(addresses)];layout=layouts[block]
        gathered=jnp.take_along_axis(tape,jnp.broadcast_to(pair[:,None,:],(len(tokens),len(words),2)),2)
        local=jnp.zeros((len(tokens),len(words),3),jnp.uint32)
        for k in range(2):local=local.at[rows,cases,layout[:,k,None]].set(gathered[:,:,k])
        bank=execute_local(codes[block],local,bounded=bounded,native_add=native_add);values=jnp.take_along_axis(bank,jnp.broadcast_to(returns[block,None,:],(len(tokens),len(words),2)),2)
        for k in range(2):
            old=jnp.take_along_axis(tape,jnp.broadcast_to(pair[:,None,k:k+1],(len(tokens),len(words),1)),2)[...,0]
            tape=tape.at[rows,cases,pair[:,k,None]].set(jnp.where(tokens[:,None]>0,values[:,:,k],old))
        return tape,None
    return jax.lax.scan(call,tape,genomes.T)[0]

def execute_pairs(program,x,blocks,*,bounded=True,native_add=False):
    from research.codepy.subtask_discovery.branch_guided.register_abi import tables
    words=jnp.asarray(np.ascontiguousarray(x).view(np.uint32))
    result=compose(jnp.asarray([program],jnp.int32),words,*tables(blocks),bounded=bounded,native_add=native_add)[0]
    return np.asarray(result).view(np.float32)

def execute(verifier,candidate,x,*,independent=False):
    # Validation and certification remain with the original independent verifier.
    from research.codepy.ssm_dispatch.programs import vector_arguments,CertificationError
    verifier.validate(candidate);x=vector_arguments(x)
    if x.shape[1]!=candidate.arity:raise CertificationError('Program/argument arity mismatch')
    program=verifier.flatten(candidate) if hasattr(verifier,'flatten') else candidate.program
    if independent:
        from research.codepy.ssm_dispatch.numpy_runtime import execute_pairs as reference
        return reference(program,x,verifier.blocks)
    return execute_pairs(program,x,verifier.blocks,bounded=False,native_add=not needed(x))

def binding(g,fields):
    a,b,p,q,t=g;a0,a1,ps,qs,permutation,gate_id=fields
    permutations=jnp.asarray([list(v)+[4,5,6,7] for v in itertools.permutations(range(4))],jnp.int32)
    mapping=jnp.broadcast_to(permutations[permutation][:,None,:],t.shape);inverse=jnp.argsort(mapping,axis=-1)
    tape=jnp.take_along_axis(t,mapping,-1);pointer=jnp.take_along_axis(inverse,p[...,None],-1)[...,0]
    pointer=jnp.where(ps[:,None]==0,pointer,jnp.broadcast_to((ps-1)[:,None],p.shape))
    cell=jnp.take_along_axis(t,p[...,None],-1)[...,0]
    values=jnp.stack((a,b,*[t[:,:,i] for i in range(4)],cell,jnp.zeros_like(a),jnp.full_like(a,ONE),jnp.full_like(a,ONE|SIGN)),axis=-1)
    take=lambda values,index:jnp.take_along_axis(values,jnp.broadcast_to(index[:,None,None],a.shape+(1,)),-1)[...,0]
    r0=take(values,a0);r1=take(values.at[:,:,0].set(b).at[:,:,1].set(a),a1)
    pred=take(jnp.stack((q,jnp.zeros_like(q),jnp.ones_like(q),greater(r0,r1),greater(r0,jnp.uint32(0)),greater(cell,r0)),-1),qs)
    gate=take(jnp.stack((jnp.ones_like(q),q,~q,greater(a,b),greater(b,a),greater(t[:,:,0],t[:,:,1]),greater(t[:,:,2],t[:,:,3]),greater(a,jnp.uint32(0))),-1),gate_id)
    return (r0,r1,pointer,pred,tape),mapping,inverse,gate

@partial(jax.jit,static_argnames=('n','primitive_count','bounded','native_add'))
def arguments_compose(genomes,words,table,lengths,tables,*,n,primitive_count,bounded=True,native_add=False):
    from research.codepy.block_contracts.contracts import decode
    from research.codepy.adaptive_calls.calls import metadata_at_steps
    codes,labels,prefix,begin,end=decode(genomes[:,:,0],table,lengths)
    metadata=metadata_at_steps(genomes,lengths)
    global_state=zeros(len(genomes),len(words));global_state=(*global_state[:4],global_state[4].at[:,:,:n].set(words))
    local=zeros(len(genomes),len(words));rin,rout,tin,tout=tables
    def select(mask,a,b):return tuple(jnp.where(mask[...,None] if x.ndim==3 else mask,x,y) for x,y in zip(a,b))
    def input_state(s,ri,ti):return tuple(jnp.where(ri[...,i],v,jnp.zeros_like(v)) for i,v in enumerate(s[:4]))+(jnp.where(ti,s[4],jnp.uint32(0)),)
    def output_state(g,l,ro,to):return tuple(jnp.where(ro[...,i],x,y) for i,(x,y) in enumerate(zip(l[:4],g[:4])))+(jnp.where(to,l[4],g[4]),)
    def body(carry,token):
        g,l=carry;op,label,length,first,last,meta=token;macro=(label>=primitive_count)[:,None]
        bound,mapping,inverse,gate=binding(g,tuple(meta[:,i] for i in range(1,7)))
        ri=jnp.broadcast_to(rin[label,length][:,None,:],g[2].shape+(4,));ro=jnp.broadcast_to(rout[label,length][:,None,:],ri.shape)
        ti=tin[label[:,None],length[:,None],bound[2]];to=tout[label[:,None],length[:,None],bound[2]]
        l=select(macro&first[:,None],input_state(bound,ri,ti),l);l=step(l,op[:,None],n,bounded=bounded,native_add=native_add)
        physical=(l[0],l[1],jnp.take_along_axis(mapping,l[2][...,None],-1)[...,0],l[3],jnp.take_along_axis(l[4],inverse,-1))
        physical_mask=jnp.take_along_axis(to,inverse,-1)
        g=select(macro&last[:,None]&gate,output_state(g,physical,ro,physical_mask),select(macro,g,step(g,op[:,None],n,bounded=bounded,native_add=native_add)))
        return (g,l),None
    return jax.lax.scan(body,(global_state,local),tuple(v.T for v in (codes,labels,prefix,begin,end))+(jnp.swapaxes(metadata,0,1),))[0][0][4][:,:,:n]

def execute_catalog(catalog,command,x,*,independent=False):
    from research.codepy.ssm_dispatch.catalog import arguments
    x=arguments(x)
    task,pi=divmod(command,24)
    from research.codepy.ssm_dispatch.catalog import PERMS
    p=PERMS[pi];words=np.ascontiguousarray(x[:,p]).view(np.uint32)
    row=catalog.selected[task];q=row['branches'][row['chosen_by_TRAIN']]
    program=jnp.asarray([q['program']],jnp.int32)
    if independent:
        from research.codepy.ssm_dispatch import numpy_runtime
        if q['search_kind']=='arguments':
            result=numpy_runtime.numpy_execute(q['program'],x[:,p],catalog.entries,len(catalog.g.s.ALPHABET))
        else:result=numpy_runtime.execute_pairs(q['program'],x[:,p],q['blocks'])
        return result[:,np.argsort(p)]
    if q['search_kind']=='arguments':
        result=arguments_compose(program,jnp.asarray(words),catalog.table,catalog.lengths,catalog.contracts,n=4,primitive_count=len(catalog.g.s.ALPHABET),bounded=False,native_add=not needed(x))
    else:
        return execute_pairs(tuple(q['program']),np.ascontiguousarray(x[:,p]),q['blocks'],bounded=False,native_add=not needed(x))[:,np.argsort(p)]
    return np.asarray(result)[0][:,np.argsort(p)].copy().view(np.float32)
