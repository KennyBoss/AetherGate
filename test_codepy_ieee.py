"""Independent NumPy comparisons for bit-exact CPU float32 instruction execution."""
import unittest,json
from pathlib import Path
import numpy as np,jax,jax.numpy as jnp
from research.codepy.ssm_dispatch import ieee_runtime as vm
from research.codepy.ssm_dispatch.pipeline import Pipeline
from research.codepy.ssm_dispatch.programs import blocks_for
from test_codepy_development import WRITE
from test_codepy_pipeline import QUESTION

BITS=np.asarray([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f7fffff,0xff7fffff,0x3f800000,0xbf800000,2,0x80000002,0x3f000000,0xbf000000],np.uint32)

class IEEEExecutionTests(unittest.TestCase):
 def test_integer_add_rounds_like_numpy(self):
  self.assertTrue(vm.needed(np.asarray([[0x00800001,0x80800000]],np.uint32).view(np.float32)))
  rng=np.random.default_rng(99101)
  for bound in (0x00800001,2**32):
   a=rng.integers(0,bound,100000,dtype=np.uint32);b=rng.integers(0,bound,100000,dtype=np.uint32)
   if bound<2**32:
    a|=rng.integers(0,2,len(a),dtype=np.uint32)<<31;b|=rng.integers(0,2,len(b),dtype=np.uint32)<<31
   a[(a&0x7f800000)==0x7f800000]=0;b[(b&0x7f800000)==0x7f800000]=0
   a=np.concatenate((a,np.repeat(BITS,len(BITS))));b=np.concatenate((b,np.tile(BITS,len(BITS))))
   with np.errstate(over='ignore'):reference=a.view(np.float32)+b.view(np.float32)
   actual=np.asarray(jax.jit(vm.add)(jnp.asarray(a),jnp.asarray(b)))
   np.testing.assert_array_equal(actual,reference.view(np.uint32))

 def test_all_instructions_and_random_bodies_against_independent_numpy(self):
  # Import the same independent oracle used by frozen block certification.
  p=Pipeline(0);blocks_for(p.catalog)
  from reference import numpy_state
  rng=np.random.default_rng(1167);words=np.tile(BITS,(3,1)).T.copy()
  words=np.concatenate((words,rng.integers(0,2**32,(47,3),dtype=np.uint32)))
  words[(words&0x7f800000)==0x7f800000]=0
  codes=[[op]+[0]*11 for op in range(36)]+rng.integers(0,36,(37,12)).tolist()
  native=np.asarray(vm.execute_local(jnp.asarray(codes,jnp.int32),jnp.broadcast_to(jnp.asarray(words),(len(codes),*words.shape))))
  for i,code in enumerate(codes):
   x=words.view(np.float32);tape=np.zeros((len(x),8),np.float32);tape[:,:3]=x;z=np.zeros(len(x),np.float32)
   with np.errstate(over='ignore',invalid='ignore'):
    result=numpy_state(code,[z,z,np.zeros(len(x),np.int32),np.zeros(len(x),bool),tape],3)
   bank=np.concatenate((result[4][:,:3],result[0][:,None],result[1][:,None]),axis=-1)
   np.testing.assert_array_equal(native[i],bank.view(np.uint32),err_msg=str(code))

 def test_base_twenty_and_argument_permutations_exact_on_edge_words(self):
  x=np.stack([np.roll(BITS,i)[:4] for i in range(16)]).copy().view(np.float32)
  for seed in range(5):
   p=Pipeline(seed)
   for command in range(96):
    with self.subTest(seed=seed,command=command):
     a=p._execute_command(command,x);b=p._execute_command(command,x,independent=True)
     np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))

 def test_semantic_sort8_reverse16_edge_words_all_seeds(self):
  root=Path(__file__).resolve().parent
  for seed in range(5):
   for task,width in [('sort8',8),('reverse16',16)]:
    with self.subTest(seed=seed,task=task):
     p=Pipeline(seed);entry=json.loads((root/f'artifacts/codepy_performance_20261101/entries/hierarchical_{seed}_{task}.json').read_text());p.programs[512]=entry
     x=np.stack([np.roll(BITS,i)[:width] for i in range(16)]).copy().view(np.float32)
     for bad in (x.astype(np.float64),x[:,:2],np.full_like(x,np.nan)):
      with self.assertRaises(ValueError):p._execute_command(512,bad)
     p.remember(command=512,values=x,**WRITE);result=p.execute(QUESTION,verify=True)
     reference=p._execute_command(512,x,independent=True)
     np.testing.assert_array_equal(result.arguments.view(np.uint32),x.view(np.uint32))
     np.testing.assert_array_equal(result.output.view(np.uint32),reference.view(np.uint32))

if __name__=='__main__':unittest.main()
