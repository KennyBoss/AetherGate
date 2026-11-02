"""Full finite binary32 range, independent instructions and accepted algorithms."""
import json
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from research.codepy.ssm_dispatch import ieee_runtime as vm, numpy_runtime as oracle
from research.codepy.ssm_dispatch.pipeline import Pipeline

ROOT=Path(__file__).resolve().parent

def full_range_words(seed,width,rows=8192):
    rng=np.random.default_rng(971600+seed+width)
    words=rng.integers(0,2**32,(rows,width),dtype=np.uint32)
    words[(words&0x7f800000)==0x7f800000]&=np.uint32(0xff7fffff)
    # Every exponent, both signs, and boundary mantissas, plus signed zeros.
    edges=np.asarray([(sign<<31)|(exponent<<23)|mantissa for sign in (0,1)
                      for exponent in range(255) for mantissa in (0,1,0x7fffff)],np.uint32)
    words.flat[:min(words.size,len(edges))]=edges[:min(words.size,len(edges))]
    words[:,:min(width,4)]=np.asarray([0x7f7fffff,0xff7fffff,1,0x80000000],np.uint32)[:min(width,4)]
    # Reserve complete rows for coverage rather than overwriting every boundary.
    count=min(rows//2,len(edges)//width)
    words[-count:]=edges[:count*width].reshape(count,width)
    return words

class FullRangeTests(unittest.TestCase):
    def test_all_base_commands_preserve_extremes(self):
        x=full_range_words(0,4,128).view(np.float32)
        for seed in range(5):
            p=Pipeline(seed)
            for command in range(96):
                with self.subTest(seed=seed,command=command):
                    a=p._execute_command(command,x);b=p._execute_command(command,x,independent=True)
                    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
                    np.testing.assert_array_equal(np.sort(a,axis=1),np.sort(x,axis=1))

    def test_sort8_reverse16_entire_exponent_range(self):
        for seed in range(5):
            for task,width in [('sort8',8),('reverse16',16)]:
                with self.subTest(seed=seed,task=task):
                    p=Pipeline(seed)
                    p.programs[512]=json.loads((ROOT/f'artifacts/codepy_performance_20261101/entries/hierarchical_{seed}_{task}.json').read_text())
                    x=full_range_words(seed,width,1024).view(np.float32)
                    a=p._execute_command(512,x);b=p._execute_command(512,x,independent=True)
                    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
                    np.testing.assert_array_equal(np.sort(a,axis=1),np.sort(x,axis=1))
                    if task=='reverse16':np.testing.assert_array_equal(a,x[:,::-1])
                    else:np.testing.assert_array_equal(a,np.sort(x,axis=1))

    def test_instructions_no_saturation_and_ieee_overflow(self):
        # Read two cells, add, store; no silent saturation on a finite result.
        codes=[[3,7,4,10,5],[3,7,4,10,5]]
        x=np.asarray([[2e6,3e6,0],[np.finfo(np.float32).max,np.finfo(np.float32).max,0]],np.float32)
        native=np.asarray(vm.execute_local(jnp.asarray(codes),jnp.broadcast_to(jnp.asarray(x.view(np.uint32)),(2,*x.shape)),bounded=False))
        with np.errstate(over='ignore'):
            z=np.zeros(len(x),np.float32);t=np.pad(x,((0,0),(0,5)))
            state=oracle.numpy_state(codes[0],[z,z,np.zeros(2,np.int32),np.zeros(2,bool),t],3)
        bank=np.concatenate((state[4][:,:3],state[0][:,None],state[1][:,None]),axis=1)
        np.testing.assert_array_equal(native[0],bank.view(np.uint32))
        self.assertEqual(bank[0,1],np.float32(5e6));self.assertTrue(np.isposinf(bank[1,1]))

    def test_full_range_instruction_reference(self):
        rng=np.random.default_rng(717)
        words=full_range_words(1,3,1024)
        # All opcodes, branches, and longer sequences including overflow/cancellation.
        codes=[[op]+[0]*15 for op in range(36)]+rng.integers(0,36,(64,16)).tolist()
        codes += [[3,7,4,10,10,35,5]+[0]*9]
        native=np.asarray(vm.execute_local(jnp.asarray(codes,jnp.int32),jnp.broadcast_to(jnp.asarray(words),(len(codes),*words.shape)),bounded=False))
        x=words.view(np.float32);z=np.zeros(len(x),np.float32);t=np.pad(x,((0,0),(0,5)))
        for i,code in enumerate(codes):
            with np.errstate(over='ignore',invalid='ignore'):
                st=oracle.numpy_state(code,[z,z,np.zeros(len(x),np.int32),np.zeros(len(x),bool),t],3)
            bank=np.concatenate((st[4][:,:3],st[0][:,None],st[1][:,None]),axis=1)
            np.testing.assert_array_equal(native[i],bank.view(np.uint32),err_msg=str(code))

    def test_proven_safe_addition_fast_path(self):
        # Every exponent on the proven-safe grid, both signs; tiny/negative-zero
        # inputs must never select hardware addition. Constants/pointer values
        # also lie on the same grid, so closure holds across arbitrary chains.
        rng=np.random.default_rng(81991)
        words=rng.integers(0,2**32,(4096,3),dtype=np.uint32)
        words=((words&0x807fffff)|rng.integers(25,255,words.shape,dtype=np.uint32)<<23)
        words[0]=[0,0x0c800000,0x8c800000]
        x=words.view(np.float32);self.assertFalse(vm.needed(x))
        self.assertTrue(vm.needed(np.array([1,0x80000000],np.uint32).view(np.float32)))
        codes=rng.integers(0,36,(64,16),dtype=np.int32)
        inputs=jnp.broadcast_to(jnp.asarray(words),(len(codes),*words.shape))
        exact=np.asarray(vm.execute_local(jnp.asarray(codes),inputs,bounded=False))
        fast=np.asarray(vm.execute_local(jnp.asarray(codes),inputs,bounded=False,native_add=True))
        finite=(exact&0x7fffffff)<0x7f800000
        np.testing.assert_array_equal(fast[finite],exact[finite])
        np.testing.assert_array_equal(np.isfinite(fast.view(np.float32)),np.isfinite(exact.view(np.float32)))

    def test_corrupt_reference_and_nonfinite_arguments_rejected(self):
        p=Pipeline(0)
        for x in (np.full((1,4),np.inf,np.float32),np.full((1,4),np.nan,np.float32)):
            with self.assertRaises(ValueError):p._execute_command(0,x)
        with patch.object(vm,'execute_catalog',return_value=np.full((1,4),np.inf,np.float32)):
            with self.assertRaisesRegex(ValueError,'Nonfinite'):p._execute_command(0,np.ones((1,4),np.float32))
        p.version['numpy_runtime']='0'*64
        with self.assertRaisesRegex(ValueError,'integrity'):p._execute_command(0,np.ones((1,4),np.float32))

if __name__=='__main__':unittest.main()
