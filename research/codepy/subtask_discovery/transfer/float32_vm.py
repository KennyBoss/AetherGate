"""Finite binary32 VM: integer storage and round-to-nearest-even arithmetic.

The DSL clips each step to +/-1e6. No floating arithmetic touches tape values,
so CPU DAZ/FTZ cannot erase subnormals or the sign of zero. NumPy stays separate.
"""
import jax
import jax.numpy as jnp

SIGN=jnp.uint32(0x80000000)
MAG=jnp.uint32(0x7fffffff)
FRAC=jnp.uint32(0x007fffff)
LIMIT=jnp.uint32(0x49742400)  # exact float32(1e6)

def _jam(value,shift):
    shift=jnp.minimum(shift,31).astype(jnp.uint32)
    lost=(value&((jnp.uint32(1)<<shift)-jnp.uint32(1)))!=0
    return (value>>shift)|lost.astype(jnp.uint32)

def add32(a,b):
    """IEEE binary32 addition for finite operands, including gradual underflow."""
    aa=a&MAG;bb=b&MAG;swap=aa<bb
    large=jnp.where(swap,b,a);small=jnp.where(swap,a,b)
    le=jnp.maximum((large>>23)&255,1);se=jnp.maximum((small>>23)&255,1)
    lm=(large&FRAC)|jnp.where((large&jnp.uint32(0x7f800000))!=0,jnp.uint32(0x800000),jnp.uint32(0))
    sm=(small&FRAC)|jnp.where((small&jnp.uint32(0x7f800000))!=0,jnp.uint32(0x800000),jnp.uint32(0))
    x=lm<<3;y=_jam(sm<<3,le-se)
    total=jnp.where(((a^b)&SIGN)==0,x+y,x-y)
    carry=total>=jnp.uint32(1<<27)
    total=jnp.where(carry,_jam(total,jnp.uint32(1)),total)
    exponent=le+carry.astype(jnp.uint32)
    shift=jnp.minimum(jnp.maximum(jax.lax.clz(total).astype(jnp.int32)-5,0),exponent.astype(jnp.int32)-1).astype(jnp.uint32)
    total=total<<shift;exponent=exponent-shift
    mantissa=total>>3;tail=total&7
    mantissa=mantissa+((tail>4)|((tail==4)&((mantissa&1)!=0))).astype(jnp.uint32)
    round_carry=mantissa>=jnp.uint32(1<<24)
    mantissa=jnp.where(round_carry,mantissa>>1,mantissa);exponent=exponent+round_carry.astype(jnp.uint32)
    exp_field=jnp.where(mantissa<jnp.uint32(1<<23),0,exponent)
    magnitude=jnp.where(exp_field>=255,jnp.uint32(0x7f800000),(exp_field<<23)|(mantissa&FRAC))
    sign=large&SIGN
    sign=jnp.where(total==0,(a&b)&SIGN,sign)
    return magnitude|sign

def greater(a,b):
    am=a&MAG;bm=b&MAG;an=(a&SIGN)!=0;bn=(b&SIGN)!=0
    return ~((am==0)&(bm==0))&jnp.where(an!=bn,~an,jnp.where(an,am<bm,am>bm))

def clip32(a):return jnp.where((a&MAG)>LIMIT,(a&SIGN)|LIMIT,a)

def _step(state,op,n):
    a,b,p,q,t=state;cell=jnp.take_along_axis(t,p[...,None],2)[...,0]
    choose=lambda code,value,old:jnp.where(op==code,value,old)
    r0=choose(1,jnp.uint32(0),a);r1=choose(2,jnp.uint32(0),b)
    r0=choose(3,cell,r0);r1=choose(4,cell,r1)
    selected=jnp.arange(8)==p[...,None]
    tape=jnp.where(((op==5)[...,None]&selected),a[...,None],t)
    tape=jnp.where(((op==6)[...,None]&selected),b[...,None],tape)
    ptr=jnp.clip(choose(9,0,choose(8,p-1,choose(7,p+1,p))),0,7)
    pointer=jax.lax.bitcast_convert_type(p.astype(jnp.float32),jnp.uint32)
    for code,value in ((10,add32(a,b)),(11,add32(a,jnp.uint32(0x3f800000))),
                       (12,add32(a,jnp.uint32(0xbf800000))),(23,pointer),(32,b),
                       (16,add32(a,jnp.where(q,jnp.uint32(0x3f800000),jnp.uint32(0)))),
                       (17,jnp.where(q,b,a)),(25,jnp.where(q,cell,a)),(27,jnp.where(q,pointer,a))):
        r0=choose(code,value,r0)
    for code,value in ((24,pointer),(33,a),(26,jnp.where(q,cell,b)),(28,jnp.where(q,pointer,b)),(34,jnp.where(q,a,b))):r1=choose(code,value,r1)
    predicate=q
    for code,value in ((13,greater(a,jnp.uint32(0))),(14,greater(b,jnp.uint32(0))),
                       (15,p<n),(20,greater(cell,a)),(21,greater(cell,b)),(22,p<n-1)):
        predicate=choose(code,value,predicate)
    return clip32(r0),clip32(r1),ptr,predicate,clip32(tape)

@jax.jit
def execute_local(codes,inputs):
    shape=(codes.shape[0],inputs.shape[1]);zero=jnp.zeros(shape,jnp.uint32)
    tape=jnp.zeros((*shape,8),jnp.uint32).at[:,:,:3].set(jax.lax.bitcast_convert_type(inputs,jnp.uint32))
    state=(zero,zero,jnp.zeros(shape,jnp.int32),jnp.zeros(shape,bool),tape)
    state=jax.lax.scan(lambda s,op:(_step(s,op[:,None],3),None),state,codes.T)[0]
    words=jnp.concatenate((state[4][:,:,:3],state[0][...,None],state[1][...,None]),axis=-1)
    return jax.lax.bitcast_convert_type(words,jnp.float32)
