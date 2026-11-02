from common import ROOT,legacy
import sys,unittest
import numpy as np
import jax,jax.numpy as jnp
from progress import metrics,behavioral_elites,mutate
from branch_slicing import condition_mask,register_slice
from register_abi import tables,compose,numpy_compose,numpy_pair
from reference import numpy_state
class GuidedTests(unittest.TestCase):
 def test_two_order_progress_and_constant_penalty(self):
  x=legacy.PROBE;identity=np.asarray(metrics(jnp.asarray(x[None])))[0];zero=np.asarray(metrics(jnp.zeros((1,len(x),2))))[0]
  self.assertGreater(min(zero[6:]),min(identity[6:]))
  partial=x.copy();gt=np.flatnonzero(x[:,0]>x[:,1]);partial[gt[:3]]=partial[gt[:3],::-1]
  p=np.asarray(metrics(jnp.asarray(partial[None])))[0];self.assertLess(p[6],identity[6]);self.assertEqual(float(p[0]),0.);self.assertLess(p[1],1.)
 def test_functional_diversity_not_sixteen_clones(self):
  x=legacy.PROBE;outputs=np.concatenate([np.broadcast_to(x,(128,*x.shape)),np.broadcast_to(x[:,::-1],(128,*x.shape))]);values=np.asarray(metrics(jnp.asarray(outputs)));codes=np.zeros((256,12),np.int32);layouts=np.tile([0,1],(256,1));ids,_,info=behavioral_elites(values,outputs,codes,layouts)
  self.assertEqual(len(ids),16);self.assertEqual(info['unique_elite_behaviors'],2)
 def test_protected_body_and_layout_travel_together(self):
  elite=np.tile(np.arange(12),(4,1));elite[:,3]=np.arange(4)+10;elite[:,7]=np.arange(4)+15;layout=jnp.array([0,1,2,0]);critical=np.zeros((4,12),bool);critical[:,[3,7]]=True
  got,l=mutate(jax.random.PRNGKey(71),jnp.asarray(elite),layout,jnp.asarray(critical));got=np.asarray(got);l=np.asarray(l)
  for i in range(4,128):
   parent=int(got[i,3]-10);self.assertTrue(0<=parent<4);self.assertEqual(int(got[i,7]),int(elite[parent,7]));self.assertEqual(int(l[i]),int(layout[parent]))
 def test_only_live_input_dependent_conditions_protected(self):
  code=[3,7,20,25];mask=condition_mask(code,[0,1]);self.assertTrue(mask[2] and mask[3]);self.assertFalse(any(condition_mask(code+[1],[0,1])))
  self.assertFalse(any(condition_mask([1,13,25],[0,1])))
 def test_register_slices_preserve_register_outputs(self):
  rng=np.random.default_rng(14);x=rng.uniform(-1000,1000,(29,3)).astype(np.float32)
  def execute(code):
   z=np.zeros(len(x),np.float32);return numpy_state(code,[z,z,np.zeros(len(x),np.int32),np.zeros(len(x),bool),np.pad(x,((0,0),(0,5)))],3)
  for _ in range(120):
   code=rng.choice(legacy.ALPHABET,size=int(rng.integers(2,37))).tolist();sl,_=register_slice(code);a=execute(code);b=execute(sl or [0])
   for k in [0,1]:np.testing.assert_array_equal(a[k],b[k])
 def test_both_return_conventions_match_independent_vm(self):
  rng=np.random.default_rng(23);blocks=[{'code':rng.choice(legacy.ALPHABET,size=n).tolist(),'layout':layout,'outputs':ret} for n,layout,ret in [(7,[0,2],[0,2]),(12,[0,1],[3,4]),(8,[1,2],[3,4])]];codes,layouts,returns=tables(blocks);x=rng.uniform(-20,20,(31,4)).astype(np.float32);g=rng.integers(0,37,(13,6),dtype=np.int32);native=np.asarray(compose(jnp.asarray(g),jnp.asarray(x),codes,layouts,returns))
  for i,row in enumerate(g):np.testing.assert_array_equal(native[i],numpy_compose(row,x,blocks))
if __name__=='__main__':unittest.main()
