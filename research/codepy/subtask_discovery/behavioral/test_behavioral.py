import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_PLATFORM_NAME']='cpu'
import sys,itertools,unittest
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
sys.path.insert(0,str(Path(__file__).resolve().parent))
from abi import vm,tables,compose,numpy_compose,pair_outputs
from slicing import slice_outputs
from mining import ALPHABET,classify,Budget
class BehaviorTests(unittest.TestCase):
 def test_slicing_preserves_selected_outputs(self):
  rng=np.random.default_rng(741);x=rng.uniform(-1000,1000,(47,3)).astype(np.float32)
  for i in range(120):
   program=rng.choice(ALPHABET,size=rng.integers(2,37)).tolist();pair=list(itertools.combinations(range(3),2))[i%3];sliced,indices=slice_outputs(program,pair)
   if not sliced:sliced=[0]
   np.testing.assert_array_equal(vm.numpy_execute(program,x)[:,pair],vm.numpy_execute(sliced,x)[:,pair])
   self.assertTrue(all(o in ALPHABET for o in sliced))
 def test_isolated_calls_match_independent_vm(self):
  rng=np.random.default_rng(13);blocks=[{'code':rng.choice(ALPHABET,size=n).tolist(),'layout':list(p)} for n,p in [(4,(0,1)),(9,(0,2)),(12,(1,2))]];code,layout=tables(blocks);g=rng.integers(0,37,(13,6),dtype=np.int32);x=rng.uniform(-10,10,(31,4)).astype(np.float32)
  got=np.asarray(compose(jnp.asarray(g),jnp.asarray(x),code,layout))
  for i,row in enumerate(g):np.testing.assert_array_equal(got[i],numpy_compose(row,x,blocks))
 def test_noop_and_unselected_cells_preserved(self):
  block={'code':[3,7,5],'layout':[0,1]};code,layout=tables([block]);x=np.array([[4.,7.,9.,3.]],np.float32)
  np.testing.assert_array_equal(compose(jnp.zeros((1,6),jnp.int32),jnp.asarray(x),code,layout)[0],x)
  g=jnp.array([[1,0,0,0,0,0]],jnp.int32);z=np.asarray(compose(g,jnp.asarray(x),code,layout))[0];np.testing.assert_array_equal(z[:,2:],x[:,2:])
  # A body reading an uninitialized register always sees zero, never caller junk.
  code,layout=tables([{'code':[5],'layout':[0,1]}]);z=np.asarray(compose(g,jnp.asarray(x),code,layout))[0];self.assertEqual(float(z[0,0]),0.)
 def test_behavior_and_budget_reject_invalid(self):
  x=np.array([[-2.,1.],[3.,-1.],[2.,2.]],np.float32)
  self.assertIsNone(classify(x,x));self.assertIsNone(classify(np.zeros_like(x),x));self.assertEqual(classify(x[:,::-1],x),'swap');self.assertEqual(classify(np.sort(x,axis=-1),x),'order_ascending')
  b=Budget(100);self.assertTrue(b.spend(90,'a'));self.assertFalse(b.spend(11,'b'));self.assertEqual(b.used,90)
if __name__=='__main__':unittest.main()
