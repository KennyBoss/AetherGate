"""Cache hits must preserve independent execution and fail closed on any changed trust input."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import jax
from research.codepy.ssm_dispatch import examples as flat,hierarchy as hier
from research.codepy.ssm_dispatch.pipeline import Pipeline,VerifierCache
from research.codepy.ssm_dispatch.programs import Candidate,CertificationError,json_hash
from test_codepy_pipeline import QUESTION
from test_codepy_development import WRITE

class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries={};cls.pairs={}
        for seed in range(5):
            cat=Pipeline(seed).catalog;rng=np.random.default_rng(94000+seed)
            x=rng.uniform(-3,3,(17,4)).astype(np.float32);z=rng.uniform(-10,10,(19,4)).astype(np.float32)
            train=flat.ExamplePairs(x,x[:,::-1].copy());held=flat.ExamplePairs(z,z[:,::-1].copy());c=Candidate(seed,'inner',(3,5))
            entry={'mode':flat.MODE,'candidate':c.payload(),'certificate':flat.Verifier(cat).certify(c,train,held),'verification':{'train':train.payload(),'holdout':held.payload()}}
            language={'arity':4,'max_pair_ops':2,'dependencies':[{'command':96,'sha256':json_hash(entry),'entry':entry}],
                      'actions':[{'kind':'noop'},{'kind':'macro','dependency':0,'indices':[0,1,2,3]}]}
            c=Candidate(seed,'outer',(1,));verifier=hier.Verifier(cat,language)
            outer={'mode':hier.MODE,'candidate':c.payload(),'language':language,'certificate':verifier.certify(c,train,held),'verification':{'train':train.payload(),'holdout':held.payload()}}
            cls.entries[seed]={96:entry,97:outer};cls.pairs[seed]=z[:3].copy()

    def pipeline(self,seed=0):
        p=Pipeline(seed);p.programs=copy.deepcopy(self.entries[seed]);p.task_bindings={'inner':96,'outer':97}
        p.remember(command=97,values=self.pairs[seed],**WRITE)
        return p

    def test_one_build_shared_by_jax_numpy_and_repeated_execute(self):
        p=self.pipeline()
        from research.codepy.ssm_dispatch import programs
        with patch.object(programs,'verifier_for',wraps=programs.verifier_for) as factory:
            result=p.execute(QUESTION);self.assertTrue(result.independent_agrees)
            builds=factory.call_count;self.assertGreater(builds,0)
            v,c=p._verifier_cache._items[97]
            from research.codepy.ssm_dispatch import ieee_runtime
            with patch.object(ieee_runtime,'execute',wraps=ieee_runtime.execute) as execute:
                result=p.execute(QUESTION)
                self.assertEqual([call.kwargs['independent'] for call in execute.call_args_list],[False,True])
            self.assertEqual(factory.call_count,builds)
        self.assertEqual(p._verifier_cache.stats,{'hits':1,'misses':1,'invalidations':0,'size':1})
        np.testing.assert_array_equal(result.output,self.pairs[0][:,::-1])

    def test_remember_observe_clone_and_load_invalidate(self):
        p=self.pipeline();p.execute(QUESTION);p.remember(command=97,values=self.pairs[0],**WRITE)
        self.assertEqual(p._verifier_cache.stats['size'],0);p.execute(QUESTION)
        p.observe('Пётр читает журнал.');self.assertEqual(p._verifier_cache.stats['size'],0);p.execute(QUESTION)
        clone=copy.copy(p);self.assertIsNot(clone._verifier_cache,p._verifier_cache);self.assertEqual(clone._verifier_cache.stats['size'],0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'session.npz';p.save(path)
            with np.load(path,allow_pickle=False) as z:m=json.loads(str(z['metadata_json'].item()))
            self.assertNotIn('_verifier_cache',m)
            q=Pipeline.load(path);self.assertEqual(q._verifier_cache.stats['size'],0)
            q.execute(QUESTION);self.assertEqual(q._verifier_cache.stats['misses'],1)

    def test_v6_migration_v7_roundtrip_and_unknown_version_rejected(self):
        from research.codepy.ssm_dispatch.pipeline import V6_IMPLEMENTATION
        p=self.pipeline()
        with tempfile.TemporaryDirectory() as directory:
            original=Path(directory)/'original.npz';p.save(original)
            with np.load(original,allow_pickle=False) as z:
                arrays={k:z[k].copy() for k in z.files}
            metadata=json.loads(str(arrays['metadata_json'].item()))
            for schema,implementation in [('semantic-codepy-session-v6',V6_IMPLEMENTATION),
                                          ('semantic-codepy-session-v7',p.version['implementation'])]:
                with self.subTest(schema=schema):
                    m=copy.deepcopy(metadata);m['version'].update(schema=schema,implementation=implementation)
                    if schema=='semantic-codepy-session-v6':
                        m['version'].pop('ieee',None);m['version'].pop('numpy_runtime',None)
                    arrays['metadata_json']=np.asarray(json.dumps(m))
                    path=Path(directory)/(schema+'.npz');np.savez_compressed(path,**arrays)
                    q=Pipeline.load(path)
                    self.assertEqual(q.version,p.version);self.assertEqual(q._verifier_cache.stats['size'],0)
                    result=q.execute(QUESTION);self.assertTrue(result.independent_agrees)
                    np.testing.assert_array_equal(result.output,self.pairs[0][:,::-1])
                    migrated=Path(directory)/'migrated.npz';q.save(migrated)
                    self.assertEqual(Pipeline.load(migrated).version,p.version)
            m=copy.deepcopy(metadata);m['version']['implementation']='0'*64
            arrays['metadata_json']=np.asarray(json.dumps(m));bad=Path(directory)/'unknown.npz'
            np.savez_compressed(bad,**arrays)
            with self.assertRaises(ValueError):Pipeline.load(bad)

    def test_catalog_binding_and_semantic_changes_rebuild(self):
        p=self.pipeline();p.execute(QUESTION);old=p._verifier_cache._items[97][0]
        p.task_bindings['another-name']=97;p.execute(QUESTION)
        self.assertIsNot(old,p._verifier_cache._items[97][0]);misses=p._verifier_cache.misses
        # Semantic arrays are immutable JAX values; replacing one changes the session digest.
        leaves,tree=jax.tree_util.tree_flatten(p.semantic)
        array=leaves[0];leaves[0]=array.at[(0,)*array.ndim].add(1)
        p.semantic=jax.tree_util.tree_unflatten(tree,leaves)
        p._execute_command(97,self.pairs[0]);self.assertEqual(p._verifier_cache.misses,misses+1)
        p.programs[96]=copy.deepcopy(p.programs[96]);p.programs[96]['search']={'audit':'changed metadata'}
        p._execute_command(97,self.pairs[0]);self.assertEqual(p._verifier_cache.misses,misses+2)

    def test_root_forgery_is_recertified_and_rejected_all_five_seeds(self):
        for seed in range(5):
            p=self.pipeline(seed);p.execute(QUESTION)
            entry=p.programs[97];entry['candidate']['program']=[0]
            entry['certificate']['candidate_sha256']=json_hash(entry['candidate'])
            with self.assertRaises(CertificationError):p.execute(QUESTION)
            self.assertEqual(p._verifier_cache.stats['size'],0)

    def test_dependency_and_budget_forgery_cannot_use_warm_object(self):
        for change in ('body','budget','indices'):
            p=self.pipeline();p.execute(QUESTION);entry=p.programs[97]
            if change=='body':
                dep=entry['language']['dependencies'][0];dep['entry']['candidate']['program']=[0]
                dep['entry']['certificate']['candidate_sha256']=json_hash(dep['entry']['candidate']);dep['sha256']=json_hash(dep['entry'])
            elif change=='budget':entry['language']['max_pair_ops']=1
            else:entry['language']['actions'][1]['indices']=[0,0,2,3]
            entry['certificate']['language_sha256']=json_hash(entry['language'])
            with self.assertRaises(CertificationError):p.execute(QUESTION)
            self.assertEqual(p._verifier_cache.stats['size'],0)

    def test_numpy_bit_disagreement_rejected_on_hit(self):
        p=self.pipeline();x=self.pairs[0].copy();x[0,-1]=0.;p.remember(command=97,values=x,**WRITE);p.execute(QUESTION)
        from research.codepy.ssm_dispatch import ieee_runtime
        original=ieee_runtime.execute;calls=[]
        def altered(verifier,candidate,values,*,independent=False):
            calls.append(independent);out=original(verifier,candidate,values,independent=independent)
            if independent:out=out.copy();out[0,0]=np.float32(-0.)
            return out
        with patch.object(ieee_runtime,'execute',side_effect=altered):
            with self.assertRaises(AssertionError):p.execute(QUESTION)
        self.assertEqual(calls,[False,True]);self.assertEqual(p._verifier_cache.stats['size'],0)

    def test_source_version_and_payload_corruption_clear_cache(self):
        from research.codepy.ssm_dispatch import pipeline as module
        p=self.pipeline();p.execute(QUESTION);p.version['schema']='tampered'
        with self.assertRaisesRegex(ValueError,'version'):p.execute(QUESTION)
        self.assertEqual(p._verifier_cache.stats['size'],0)
        p=self.pipeline();p.execute(QUESTION);original=module.digest
        def altered(path):return '0'*64 if Path(path).name=='hierarchy_v1.py' else original(path)
        with patch.object(module,'digest',side_effect=altered):
            with self.assertRaisesRegex(CertificationError,'source'):p.execute(QUESTION)
        self.assertEqual(p._verifier_cache.stats['size'],0)
        p=self.pipeline();p.execute(QUESTION);record=p.records[WRITE['handle']];state=record['chunks'][0]
        record['chunks'][0]=state._replace(kv=state.kv.at[0].set(0.))
        with self.assertRaisesRegex(ValueError,'integrity'):p.execute(QUESTION)
        self.assertEqual(p._verifier_cache.stats['size'],0)

    def test_failed_build_is_not_cached_and_mutation_during_build_rejected(self):
        p=self.pipeline();from research.codepy.ssm_dispatch import programs
        original=programs.verifier_for
        def altered(cat,entry):
            result=original(cat,entry);p.task_bindings['changed-during-build']=97;return result
        with patch.object(programs,'verifier_for',side_effect=altered):
            with self.assertRaisesRegex(CertificationError,'changed during'):p.execute(QUESTION)
        self.assertEqual(p._verifier_cache.stats['size'],0)
        p.execute(QUESTION);self.assertEqual(p._verifier_cache.stats['size'],1)

    def test_fast_hash_rejects_custom_objects_without_executing_them(self):
        class Untrusted:
            def __reduce__(self):raise AssertionError('must not run')
            def __deepcopy__(self,memo):raise AssertionError('must not run')
        p=self.pipeline();p.execute(QUESTION);p.programs[96]['extra']=Untrusted()
        with self.assertRaises(ValueError):p.execute(QUESTION)
        self.assertEqual(p._verifier_cache.stats['size'],0)
        p=self.pipeline();p.programs[96]['extra']=float('nan')
        with self.assertRaises(ValueError):p.execute(QUESTION)

if __name__=='__main__':unittest.main()
