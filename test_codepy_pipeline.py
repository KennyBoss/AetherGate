"""Product-level tests: memory is the only source of commands and operands."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from research.codepy.ssm_dispatch.pipeline import Pipeline, UnknownKey, text_step
from research.codepy.ssm_dispatch.catalog import ROOT, catalog

X=np.asarray([[3.,-1.,2.,0.],[-0.,1.25,-6.5,1e5]],np.float32)
QUESTION='Что именно относится к Кларе: главное издание? Ответ:'

class PipelineTests(unittest.TestCase):
    def test_semantic_execution_uses_neural_payload_and_latest_overwrite(self):
        p=Pipeline(0)
        p.remember('клары','основная книга','морской атлас',0,X)
        p.observe('На берегу реки стоял старый дом. '*100)
        result=p.execute(QUESTION)
        np.testing.assert_array_equal(result.arguments.view(np.uint32),X.view(np.uint32))
        np.testing.assert_array_equal(result.output,np.sort(X,axis=1))
        self.assertTrue(result.independent_agrees)
        self.assertGreater(int(p.text_carry.seen[0]),100)
        changed=X[:,::-1].copy()
        p.remember('клары','основная книга','морской атлас',24,changed)
        result=p.execute('Уточни, что у клары указано как главное издание? Ответ:')
        self.assertEqual(result.command,24)
        np.testing.assert_array_equal(result.arguments.view(np.uint32),changed.view(np.uint32))
        np.testing.assert_array_equal(result.output,changed[:,::-1])

    def test_unknown_invalid_inputs_and_payload_corruption_fail_closed(self):
        p=Pipeline(1)
        with self.assertRaises(UnknownKey):p.recall('нет записи')
        with self.assertRaises(UnknownKey):p.execute(QUESTION)
        for command,values in [(-1,X),(96,X),(0,X.astype(np.float64)),(0,np.zeros((1,5),np.float32)),(0,np.full((1,4),np.nan,np.float32))]:
            with self.assertRaises(ValueError):p.remember('клары','основная книга','морской атлас',command,values)
        p.remember('клары','основная книга','морской атлас',0,X)
        record=p.records['морской атлас'];old=record['chunks'][0]
        record['chunks'][0]=old._replace(kv=jnp.zeros_like(old.kv))
        with self.assertRaisesRegex(ValueError,'integrity'):p.execute(QUESTION)

    def test_session_restart_and_versions(self):
        p=Pipeline(2);p.remember('клары','основная книга','морской атлас',48,X)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'session.npz';p.save(path)
            q=Pipeline.load(path,seed=2)
            result=q.execute(QUESTION)
            self.assertEqual(result.command,48)
            np.testing.assert_array_equal(result.output,np.roll(X,1,axis=1))
            with self.assertRaisesRegex(ValueError,'Incompatible'):Pipeline.load(path,seed=3)
            with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
            metadata=json.loads(str(arrays['metadata_json'].item()));metadata['version']['schema']='v0'
            arrays['metadata_json']=np.asarray(json.dumps(metadata));np.savez(path,**arrays)
            with self.assertRaisesRegex(ValueError,'Incompatible'):Pipeline.load(path,seed=2)
        with self.assertRaises(ValueError):p.save(ROOT/'artifacts/forbidden-session.npz')

    def test_all_seeds_four_algorithms_through_memory(self):
        for seed in range(5):
            p=Pipeline(seed)
            for command,target in [(0,np.sort(X,axis=1)),(24,X[:,::-1]),(48,np.roll(X,1,axis=1)),(72,np.sort(X,axis=1)[:,::-1])]:
                p.remember('клары','основная книга','морской атлас',command,X)
                result=p.execute(QUESTION)
                self.assertEqual(result.command,command)
                np.testing.assert_array_equal(result.output,target)

    def test_pointer_same_weights_match_frozen_delivered_kernel(self):
        p=Pipeline(0);prompt='memory is'
        ids=jnp.asarray([p.text_encoder.encode(prompt)],jnp.int32)
        logits,_,_=text_step(p.text_params,ids,p.text_base,p.text_carry)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'logits.npy'
            code='''import sys,numpy as np,jax.numpy as jnp
from runtime import load
p,cfg,v,n=load(sys.argv[1])
from text_tokenization import tokenize_with_config,encode_tokens
x=jnp.asarray([encode_tokens(tokenize_with_config('memory is',cfg['tokenizer_config']),{s:i for i,s in enumerate(v)}).tolist()],jnp.int32)
z,_=n.forward(p,x,n.initial_model(p,1,'ssm',64),'ssm')
np.save(sys.argv[2],np.asarray(z[:,-1]))
'''
            checkpoint=ROOT/'artifacts/ssm_neural_copy_20261031/exports/ssm_23.npz'
            subprocess.run([sys.executable,'-c',code,str(checkpoint),str(path)],cwd=ROOT/'research/ssm_neural_copy',check=True,timeout=120)
            np.testing.assert_allclose(logits,np.load(path),rtol=1e-6,atol=1e-6)

    def test_default_dispatcher_uses_certified_catalog(self):
        from research.codepy.ssm_dispatch import experiment
        selected,_,_,_,execute,_,_=experiment.prepare(4)
        self.assertTrue(all(q['certificate']['passed'] for q in selected))
        np.testing.assert_array_equal(execute(jnp.asarray(0),jnp.asarray(X)),np.sort(X,axis=1))

if __name__=='__main__':unittest.main()
