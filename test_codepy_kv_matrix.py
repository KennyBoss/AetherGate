"""Matrix reader agrees with the full neural oracle and keeps conservative fallbacks."""
import json,unittest
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
import train_ssm_text as core
from research.codepy.ssm_dispatch import pipeline as module
from research.codepy.ssm_scaling import runtime as codec
from test_codepy_development import WRITE
from test_codepy_pipeline import QUESTION

@jax.jit
def oracle_queries(params,state):
    def field(st,i):
        n=st.h.shape[0]
        key=jnp.full((n,1),codec.KEY+i,jnp.int32)
        _,key_state=core.forward(params,key,st)
        value=jnp.einsum('bi,bij->bj',key_state.key_trace,key_state.kv)
        logits,query_state=core.forward(params,jnp.ones((n,1),jnp.int32),key_state)
        _,st=core.forward(params,jnp.tile(jnp.asarray([[0,3]],jnp.int32),(n,1)),query_state)
        return st,(query_state.h,value,logits[:,0,codec.VALUE:codec.VALUE+96])
    return jax.lax.scan(field,state,jnp.arange(10))[1]

class MatrixReadTests(unittest.TestCase):
    def test_transition_operators_and_logits_match_oracle_all_seeds(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                params=codec.load_params(seed);rng=np.random.default_rng(320500+seed)
                st=core.init_stream_state(params,6)
                st=st._replace(h=jnp.asarray(rng.normal(size=st.h.shape),jnp.float32),
                               kv=jnp.asarray(rng.normal(size=st.kv.shape),jnp.float32),
                               key_trace=jnp.asarray(rng.normal(size=st.key_trace.shape),jnp.float32),prev_ids=jnp.full_like(st.prev_ids,3))
                hidden,values,logits=oracle_queries(params,st)
                tables=jax.jit(module._numeric_plan)(params);self.assertTrue(bool(tables['reset']))
                actual=jnp.einsum('qi,bij->qbj',tables['query'],st.kv)+tables['bias'][:,None,:]
                np.testing.assert_allclose(actual,values,rtol=2e-5,atol=2e-5)
                projected=hidden@tables['c']+tables['out_bias']+tables['scale']*(actual@tables['embed'])
                np.testing.assert_allclose(projected,logits,rtol=2e-5,atol=2e-5)
                np.testing.assert_array_equal(jnp.argmax(projected,-1),jnp.argmax(logits,-1))
                _,digits=jax.jit(lambda p,s:module._numeric_matrix_fields(p,s.h,s.kv,module._numeric_plan(p),0,10))(params,st)
                np.testing.assert_array_equal(digits,jnp.argmax(logits,-1))

    def test_fallbacks_and_current_parameter_values_are_honored(self):
        params=codec.load_params(0);rng=np.random.default_rng(9977)
        words=rng.integers(0,2**32,(module.BATCH,5),dtype=np.uint32)
        st,_=module.numeric_step(params,core.init_stream_state(params,module.BATCH*3),jnp.asarray(words),jnp.asarray(True))
        cases=[]
        cases.append((params,st._replace(prev_ids=st.prev_ids.at[0].set(2))))
        altered=dict(params);altered['kv_key_token']=altered['kv_key_token'].at[codec.KEY].set(0.)
        cases.append((altered,st))
        added=dict(params);added['skip_d']=jnp.zeros_like(params['output_c'])
        cases.append((added,st))
        cases.append((params,st._replace(kv=st.kv.at[0,0,0].set(jnp.nan))))
        changed=dict(params);changed['output_bias']=changed['output_bias'].at[codec.VALUE+95].set(1e6)
        cases.append((changed,st))
        for index,(p,s) in enumerate(cases):
            with self.subTest(case=index):np.testing.assert_array_equal(module.numeric_read(p,s),module._numeric_read_reference(p,s))
        np.testing.assert_array_equal(module.numeric_read(params,st),words)
        with self.assertRaises(ValueError):module.numeric_read(params,st._replace(h=st.h.astype(jnp.float16)))

    def test_page_release_and_subnormal_execution_is_exact(self):
        root=Path(__file__).resolve().parent
        sources=list((root/'artifacts/codepy_kv_kernel_20261101/external_sessions').glob('v7_seed*.session.npz'))
        self.assertTrue(sources)
        for path in sources:
            with np.load(path,allow_pickle=False) as z:m=json.loads(str(z['metadata_json'].item()))
            self.assertEqual(m['version']['implementation'],module.V7_PAGE_IMPLEMENTATION)
            p=module.Pipeline.load(path,seed=m['version']['seed']);self.assertEqual(p._verifier_cache.stats['size'],0)
        patterns=np.asarray([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
                             0x7f7fffff,0xff7fffff,0x3f800000,0xbf800000,2,0x80000002,0x7f000001,0xff000001],np.uint32)
        for seed in range(5):
            with self.subTest(seed=seed):
                p=module.Pipeline(seed)
                entry=json.loads((root/f'artifacts/codepy_performance_20261101/entries/hierarchical_{seed}_reverse16.json').read_text())
                p.programs[512]=entry
                x=np.tile(np.roll(patterns,seed),(257,1)).copy().view(np.float32)
                p.remember(command=512,values=x,**WRITE)
                command,recovered=p.recall(WRITE['handle'])
                np.testing.assert_array_equal(recovered.view(np.uint32),x.view(np.uint32))
                # Both direct execution and the semantic path preserve subnormal words.
                native=p._execute_command(command,x)
                reference=p._execute_command(command,x,independent=True)
                np.testing.assert_array_equal(native.view(np.uint32),reference.view(np.uint32))
                result=p.execute(QUESTION,verify=True)
                np.testing.assert_array_equal(result.output.view(np.uint32),reference.view(np.uint32))

if __name__=='__main__':unittest.main()
