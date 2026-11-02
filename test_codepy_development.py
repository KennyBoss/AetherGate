"""Closed-loop product tests: real synthesis, independent rejection and atomic publish."""
from pathlib import Path
from unittest.mock import patch
import hashlib,json,tempfile,unittest
import numpy as np
import jax
from research.codepy.ssm_dispatch.pipeline import Pipeline
from research.codepy.ssm_dispatch.development import Candidate,BlockSearch,Verifier,CertificationError,json_hash,contracts
from test_codepy_pipeline import X,QUESTION

WRITE=dict(entity='клары',property='основная книга',handle='морской атлас')

def training():
    return np.random.default_rng(67111).uniform(-3,3,(64,4)).astype(np.float32)

def fingerprint(p):
    h=hashlib.sha256()
    metadata={'programs':p.programs,'tasks':p.task_bindings,'last':p.last_text_token,'records':{k:{a:b for a,b in r.items() if a!='chunks'} for k,r in p.records.items()}}
    h.update(json.dumps(metadata,sort_keys=True).encode())
    states=(p.semantic,p.text_base,p.text_carry,p.params,p.text_params,p.numeric_params,[r['chunks'] for r in p.records.values()])
    for array in jax.tree_util.tree_leaves(states):h.update(np.asarray(array).tobytes())
    return h.hexdigest()

class DevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate,cls.search=BlockSearch(Pipeline(0).catalog).synthesize('pairwise_order4',training())

    def test_closed_loop_real_search_memory_restart_and_execution(self):
        p=Pipeline(0)
        self.assertNotIn('pairwise_order4',p.task_bindings)
        # This public call really searches; the separate class candidate is not supplied.
        result=p.develop('pairwise_order4',training(),values=X,**WRITE)
        self.assertEqual(result.status,'accepted');self.assertEqual(result.command,96)
        self.assertEqual(result.evidence['events'],['gap_detected','synthesized','candidate_frozen','independently_certified','catalog_registered','kv_and_v5_bound'])
        self.assertEqual(result.evidence['certificate']['cases'],7041)
        self.assertFalse(result.evidence['search']['heldout_available_to_search'])
        p.observe('На берегу реки стоял старый дом. '*100)
        answer=p.execute(QUESTION)
        self.assertEqual(answer.command,96)
        self.assertTrue(contracts.numpy_accept(answer.output,X,'pairwise_order4')[0].all())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'session.npz';p.save(path);q=Pipeline.load(path)
            self.assertTrue(contracts.numpy_accept(q.execute(QUESTION).output,X,'pairwise_order4')[0].all())
            with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
            metadata=json.loads(str(arrays['metadata_json'].item()))
            entry=metadata['programs']['96'];entry['candidate']['program']=[0]*6
            entry['certificate']['candidate_sha256']=json_hash(entry['candidate'])
            arrays['metadata_json']=np.asarray(json.dumps(metadata));np.savez_compressed(path,**arrays)
            with self.assertRaises(CertificationError):Pipeline.load(path)
        with patch.object(BlockSearch,'synthesize',side_effect=AssertionError('No search for a registered working task')):
            repeated=p.develop('pairwise_order4',training(),values=X,**WRITE)
        self.assertEqual(repeated.status,'already_registered')

    def test_train_perfect_but_heldout_wrong_candidate_cannot_write_memory(self):
        p=Pipeline(0);p.remember(command=24,values=X,**WRITE);before=fingerprint(p)
        x=training();x[:,:2]=np.sort(x[:,:2],axis=1);x[:,2:]=np.sort(x[:,2:],axis=1)
        self.assertTrue(contracts.numpy_accept(x,x,'pairwise_order4')[0].all())
        bad=Candidate(0,'pairwise_order4',(0,)*6)
        with patch.object(BlockSearch,'synthesize',return_value=(bad,{'training_objective':0.,'candidate_sha256':json_hash(bad.payload())})):
            result=p.develop('pairwise_order4',x,values=X,**WRITE)
        self.assertEqual(result.status,'rejected');self.assertEqual(fingerprint(p),before)
        self.assertFalse(p.programs);self.assertEqual(p.recall(WRITE['handle'])[0],24)
        self.assertEqual(result.evidence['events'][-1],'rejected')

    def test_neural_write_failure_does_not_publish_certified_candidate(self):
        p=Pipeline(0);before=fingerprint(p)
        with patch.object(BlockSearch,'synthesize',return_value=(self.candidate,self.search)),patch.object(Pipeline,'observe',side_effect=RuntimeError('injected write failure')):
            with self.assertRaises(RuntimeError):p.develop('pairwise_order4',training(),values=X,**WRITE)
        self.assertEqual(fingerprint(p),before)

    def test_incomplete_binding_triggers_repair_and_base_is_preserved(self):
        p=Pipeline(0);p.task_bindings['pairwise_order4']=24
        before=p.catalog.version
        with patch.object(BlockSearch,'synthesize',return_value=(self.candidate,self.search)):
            result=p.develop('pairwise_order4',training(),values=X,**WRITE)
        self.assertEqual(result.evidence['gap_kind'],'incomplete_solution')
        self.assertEqual(result.status,'accepted');self.assertEqual(p.catalog.version,before)
        for command,target in [(0,np.sort(X,axis=1)),(24,X[:,::-1]),(48,np.roll(X,1,axis=1)),(72,np.sort(X,axis=1)[:,::-1])]:
            np.testing.assert_array_equal(p._execute_command(command,X),target)

    def test_verifier_rejects_overlap_bad_bytecode_and_unknown_contract(self):
        p=Pipeline(0);v=Verifier(p.catalog)
        with self.assertRaises(CertificationError):v.certify(Candidate(0,'pairwise_order4',(999,)*6))
        with self.assertRaises(CertificationError):v.certify(Candidate(0,'pairwise_order4',(True,)*6))
        with self.assertRaisesRegex(CertificationError,'overlap'):v.certify(self.candidate,training_examples=np.zeros((64,4),np.float32))
        before=fingerprint(p)
        with self.assertRaises(ValueError):p.develop('unknown-contract',training(),values=X,**WRITE)
        self.assertEqual(fingerprint(p),before)

    def test_known_v1_session_is_upgraded_without_changing_memories(self):
        p=Pipeline(0);p.remember(command=24,values=X,**WRITE)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.npz';p.save(path)
            with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
            metadata=json.loads(str(arrays['metadata_json'].item()))
            from research.codepy.ssm_dispatch.pipeline import V1_IMPLEMENTATION
            [metadata['version'].pop(k) for k in ('development','examples','programs','hierarchy','ieee','numpy_runtime')];metadata['version'].update(schema='semantic-codepy-session-v1',implementation=V1_IMPLEMENTATION)
            metadata.pop('programs');metadata.pop('task_bindings')
            arrays['metadata_json']=np.asarray(json.dumps(metadata));np.savez_compressed(path,**arrays)
            q=Pipeline.load(path)
            self.assertEqual(fingerprint(p),fingerprint(q))
            self.assertEqual(q.execute(QUESTION).command,24)

if __name__=='__main__':unittest.main()
