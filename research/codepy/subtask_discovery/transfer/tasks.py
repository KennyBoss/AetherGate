"""New declarative tasks and independent checks; no executable solution templates."""
from transfer_env import vm
import jax.numpy as jnp,numpy as np
TASKS={
 'rotate_right4':{'arity':4,'contract':'Output x3,x0,x1,x2; preserve all values.'},
 'minmax4':{'arity':4,'contract':'Cell0 minimum,cell3 maximum; middle cells may be in either order; preserve all values.'},
 'median3':{'arity':3,'contract':'Cell1 median; other cells may be in either order; preserve all values.'},
 'partition4':{'arity':4,'contract':'Every value in cells0,1 <= every value in cells2,3; order within halves free; preserve all values.'},
 'pairwise_order4':{'arity':4,'contract':'Cells0,1 are the sorted original first pair; cells2,3 sorted original second pair; no cross-pair exchange.'},
 'reverse5':{'arity':5,'contract':'Output x4,x3,x2,x1,x0; preserve all values.'}}

def losses(output,x,task):
    preserved=jnp.mean((jnp.sort(output,-1)-jnp.sort(x,-1)[None])**2,axis=(1,2))
    if task=='rotate_right4':wanted=jnp.roll(x,1,axis=-1);err=jnp.mean((output-wanted[None])**2,axis=(1,2))
    elif task=='reverse5':err=jnp.mean((output-x[None,:,::-1])**2,axis=(1,2))
    elif task=='minmax4':err=jnp.mean((output[:,:,[0,3]]-jnp.stack([jnp.min(x,-1),jnp.max(x,-1)],-1)[None])**2,axis=(1,2))
    elif task=='median3':err=jnp.mean((output[:,:,1]-jnp.sort(x,-1)[None,:,1])**2,axis=1)
    elif task=='partition4':err=jnp.mean(jnp.maximum(output[:,:,:2,None]-output[:,:,None,2:],0.)**2,axis=(1,2,3))
    elif task=='pairwise_order4':wanted=jnp.concatenate([jnp.sort(x[:,:2],-1),jnp.sort(x[:,2:],-1)],-1);err=jnp.mean((output-wanted[None])**2,axis=(1,2))
    else:raise ValueError(task)
    return err+preserved,err

def numpy_accept(output,x,task):
    preserved=np.all(np.sort(output,-1)==np.sort(x,-1),-1)
    if task=='rotate_right4':ok=(output==np.concatenate([x[:,-1:],x[:,:-1]],-1)).all(-1)
    elif task=='reverse5':ok=(output==x[:,::-1]).all(-1)
    elif task=='minmax4':ok=(output[:,0]==np.min(x,-1))&(output[:,-1]==np.max(x,-1))
    elif task=='median3':ok=output[:,1]==np.sort(x,-1)[:,1]
    elif task=='partition4':ok=np.max(output[:,:2],-1)<=np.min(output[:,2:],-1)
    elif task=='pairwise_order4':ok=(output[:,:2]==np.sort(x[:,:2],-1)).all(-1)&(output[:,2:]==np.sort(x[:,2:],-1)).all(-1)
    else:raise ValueError(task)
    return ok&preserved,preserved

def train_cases(task_index,seed,n):
    x=np.random.default_rng(310000+100*task_index+seed).uniform(-3,3,(64,n)).astype(np.float32)
    x[:8,1]=x[:8,0];x[8:12]=x[8:12,:1]
    return x

def test_cases(task_index,n):return vm.cases(n,910000+task_index)
