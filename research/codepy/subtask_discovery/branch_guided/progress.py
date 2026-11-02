"""Partial progress per input order, behavioral diversity, protected coherent mutation."""
from common import legacy
import numpy as np
import jax,jax.numpy as jnp
from branch_slicing import condition_mask
X=legacy.PROBE

def metrics(out):
    x=jnp.asarray(X);a=x[:,0];b=x[:,1];lt=a<b;gt=a>b;eq=a==b;scale=jnp.maximum(jnp.abs(a-b),1.)**2
    ascending=jnp.where(lt[:,None],x,x[:,::-1]);descending=ascending[:,::-1]
    err_a=jnp.mean((out-ascending[None])**2,-1)/scale;err_d=jnp.mean((out-descending[None])**2,-1)/scale
    mean=lambda z,m:jnp.sum(z*m[None],-1)/jnp.sum(m)
    al,ag=mean(err_a,lt),mean(err_a,gt);dl,dg=mean(err_d,lt),mean(err_d,gt)
    preserve=jnp.mean(jnp.mean((jnp.sort(out,-1)-jnp.sort(x,-1)[None])**2,-1)/scale,-1);ties=mean(err_a,eq)
    asc=jnp.maximum(al,ag)+.125*(al+ag)+2*preserve+.25*ties
    desc=jnp.maximum(dl,dg)+.125*(dl+dg)+2*preserve+.25*ties
    return jnp.stack([al,ag,dl,dg,preserve,ties,asc,desc],-1)

def behavioral_elites(values,outputs,codes,layouts,count=16):
    values=np.asarray(values);outputs=np.asarray(outputs);codes=np.asarray(codes);layouts=np.asarray(layouts)
    masks=np.asarray([condition_mask(c.tolist(),l.tolist()) for c,l in zip(codes,layouts)],bool);live=masks.any(-1);best=np.minimum(values[:,6],values[:,7]);signatures=[o.tobytes() for o in outputs];chosen=[];seen=set()
    def take(score,n,eligible=None):
        eligible=np.ones(len(values),bool) if eligible is None else eligible
        order=sorted(np.flatnonzero(eligible),key=lambda i:(float(score[i]),not live[i],int(np.count_nonzero(codes[i])),int(i)))
        for i in order:
            if signatures[i] in seen:continue
            chosen.append(int(i));seen.add(signatures[i]);n-=1
            if n==0:return
    take(values[:,6],4);take(values[:,7],4);take(best,4,live)
    for col,other in [(0,1),(1,0),(2,3),(3,2)]:take(values[:,col]+.5*values[:,other]+2*values[:,4],1,live)
    if len(chosen)<count:take(best,count-len(chosen))
    if len(chosen)<count:
        for i in np.argsort(best,kind='stable'):
            if int(i) not in chosen:chosen.append(int(i))
            if len(chosen)==count:break
    ids=np.asarray(chosen[:count],np.int32)
    # Protect only live conditions with real correct cases on BOTH input orders
    # and at least75% exact value preservation on distinct pairs.
    protected=[]
    for i in ids:
        o=outputs[i];x=X;lt=x[:,0]<x[:,1];gt=x[:,0]>x[:,1];good_a=(o==np.sort(x,-1)).all(-1);good_d=(o==np.sort(x,-1)[:,::-1]).all(-1);pres=(np.sort(o,-1)==np.sort(x,-1)).all(-1)
        useful=((good_a[lt].any() and good_a[gt].any()) or (good_d[lt].any() and good_d[gt].any())) and pres[lt|gt].mean()>=.75
        protected.append(masks[i] if useful else np.zeros(12,bool))
    return ids,np.asarray(protected,bool),{'unique_elite_behaviors':len({signatures[i] for i in ids}),'live_elites':int(live[ids].sum()),'protected_elites':sum(bool(m.any()) for m in protected)}

@jax.jit
def mutate(key,elites,layouts,critical):
    keys=jax.random.split(key,10);n=len(elites);size=256;half=jnp.arange(size)<128
    a=jax.random.randint(keys[0],(size,),0,n);b=jax.random.randint(keys[1],(size,),0,n)
    # Keep metadata tied to the same parent as its protected instruction skeleton.
    mask=critical.any(-1);weights=jnp.where(mask,1.,0.);weights=jnp.where(jnp.sum(weights)>0,weights,jnp.ones(n));protected_a=jax.random.categorical(keys[2],jnp.log(weights),shape=(size,));a=jnp.where(half,protected_a,a)
    left=elites[a];right=elites[b];cut=jax.random.randint(keys[3],(size,1),1,12);children=jnp.where((jnp.arange(12)[None]<cut)&~half[:,None],left,jnp.where(half[:,None],left,right))
    changes=jax.random.bernoulli(keys[4],.16,children.shape)&~(half[:,None]&critical[a]);new=jax.random.randint(keys[5],children.shape,0,len(legacy.ALPHABET));children=jnp.where(changes,new,children)
    local=layouts[a];change_layout=jax.random.bernoulli(keys[6],.16,(size,))&~half;local=jnp.where(change_layout,jax.random.randint(keys[7],(size,),0,3),local)
    children=children.at[-12:].set(jax.random.randint(keys[8],(12,12),0,len(legacy.ALPHABET)));local=local.at[-12:].set(jax.random.randint(keys[9],(12,),0,3))
    return children.at[:n].set(elites),local.at[:n].set(layouts)
