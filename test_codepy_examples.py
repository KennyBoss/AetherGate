"""Example-only path: no task oracle, unseen labels, holdout isolation and persistence."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.codepy.ssm_dispatch.pipeline import Pipeline
from research.codepy.ssm_dispatch.examples import ExamplePairs,BlockSearch,Verifier,CertificationError
from research.codepy.ssm_dispatch.programs import Candidate,json_hash
from test_codepy_development import fingerprint,WRITE,training,contracts
from test_codepy_pipeline import QUESTION


def pairs(x):
    return np.concatenate((np.sort(x[:,:2],axis=1),np.sort(x[:,2:],axis=1)),axis=1)

def data():
    x=training();h=np.random.default_rng(99112).uniform(-100,100,(151,4)).astype(np.float32)
    return x,pairs(x),h,pairs(h)

class ExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x,y,h,hy=data();cls.train=ExamplePairs(x,y);cls.holdout=ExamplePairs(h,hy)
        cls.candidate,cls.search=BlockSearch(Pipeline(0).catalog).synthesize(cls.train,label='opaque label')

    def test_full_example_only_cycle_does_not_consult_named_oracles(self):
        x,y,h,hy=data();p=Pipeline(0)
        with patch.object(contracts,'losses',side_effect=AssertionError('Named oracle used')),patch.object(contracts,'numpy_accept',side_effect=AssertionError('Named verifier used')):
            result=p.develop(task='opaque label',X_train=x,Y_train=y,X_holdout=h,Y_holdout=hy,values=h,**WRITE)
            self.assertEqual(result.status,'accepted');self.assertEqual(result.evidence['base_commands_matching_training'],0)
            p.observe('На берегу реки стоял старый дом. '*100)
            np.testing.assert_array_equal(p.execute(QUESTION).output,hy)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'session.npz';p.save(path);q=Pipeline.load(path)
                np.testing.assert_array_equal(q.execute(QUESTION).output,hy)
                with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
                metadata=json.loads(str(arrays['metadata_json'].item()));entry=metadata['programs']['96']
                entry['candidate']['program']=[0]*6;entry['certificate']['candidate_sha256']=json_hash(entry['candidate'])
                arrays['metadata_json']=np.asarray(json.dumps(metadata));np.savez_compressed(path,**arrays)
                with self.assertRaises(CertificationError):Pipeline.load(path)
        with patch.object(BlockSearch,'synthesize',side_effect=AssertionError('Unnecessary search')):
            reused=p.develop(task='opaque label',X_train=x,Y_train=y,X_holdout=h,Y_holdout=hy,values=h,**WRITE)
        self.assertEqual(reused.status,'already_registered')

    def test_holdout_never_passed_to_search_and_bad_candidate_changes_nothing(self):
        x,y,h,hy=data();x=pairs(x);p=Pipeline(0);p.remember(command=24,values=h,**WRITE);before=fingerprint(p)
        def bad_search(search,train,*,label=None,populations=95):
            self.assertEqual(set(train.hashes()),{'inputs','targets','rows'})
            np.testing.assert_array_equal(train.inputs,x);np.testing.assert_array_equal(train.targets,x)
            self.assertFalse(hasattr(search,'holdout'))
            return Candidate(0,label,(0,)*6),{'training_objective':0.}
        with patch.object(BlockSearch,'synthesize',new=bad_search):
            result=p.develop(task='never listed anywhere',X_train=x,Y_train=x,X_holdout=h,Y_holdout=hy,values=h,**WRITE)
        self.assertEqual(result.status,'rejected');self.assertEqual(fingerprint(p),before)
        self.assertIn('holdout',result.evidence['reason'])

    def test_invalid_pairs_and_overlap_are_rejected_before_search(self):
        x,y,h,hy=data();p=Pipeline(0);before=fingerprint(p)
        with patch.object(BlockSearch,'synthesize',side_effect=AssertionError('Validation was too late')):
            for kwargs in [dict(X_train=x,Y_train=y,X_holdout=x,Y_holdout=y),dict(X_train=x,Y_train=y+1,X_holdout=h,Y_holdout=hy),
                           dict(X_train=x,Y_train=y[:1],X_holdout=h,Y_holdout=hy),dict(X_train=x,Y_train=y,X_holdout=h,Y_holdout=np.full_like(h,np.nan))]:
                with self.assertRaises(ValueError):p.develop(values=h,**WRITE,**kwargs)
            with self.assertRaises(ValueError):p.develop(X_train=x,values=h,**WRITE)
        self.assertEqual(fingerprint(p),before)
        a=np.asarray([[0.,1,2,3]],np.float32);b=a.copy();b[0,0]=-0.
        from research.codepy.ssm_dispatch.examples import disjoint
        with self.assertRaises(CertificationError):disjoint(ExamplePairs(a,a),ExamplePairs(b,b))
        xx=np.repeat(x[:1],2,axis=0);yy=xx.copy();yy[1]=yy[1,::-1]
        with self.assertRaisesRegex(ValueError,'Contradictory'):ExamplePairs(xx,yy)

    def test_arbitrary_labels_and_non64_training_size(self):
        p=Pipeline(0)
        with patch.object(contracts,'losses',side_effect=AssertionError('Named lookup')):
            a,_=BlockSearch(p.catalog).synthesize(ExamplePairs(self.train.inputs[:23],self.train.targets[:23]),label='minmax4',populations=2)
            b,_=BlockSearch(p.catalog).synthesize(ExamplePairs(self.train.inputs[:23],self.train.targets[:23]),label='unlisted-string-93',populations=2)
        self.assertEqual(a.program,b.program)
        self.assertNotEqual(a.task,b.task)
        self.assertFalse(self.train.inputs.flags.writeable)

    def test_failed_write_is_atomic_in_example_mode(self):
        x,y,h,hy=data();p=Pipeline(0);before=fingerprint(p)
        with patch.object(BlockSearch,'synthesize',return_value=(self.candidate,self.search)),patch.object(Pipeline,'observe',side_effect=RuntimeError('write failed')):
            with self.assertRaises(RuntimeError):p.develop(task='opaque label',X_train=x,Y_train=y,X_holdout=h,Y_holdout=hy,values=h,**WRITE)
        self.assertEqual(fingerprint(p),before)

    def test_registered_program_is_not_tuned_to_a_changed_holdout(self):
        x,y,h,hy=data();p=Pipeline(0)
        with patch.object(BlockSearch,'synthesize',return_value=(self.candidate,self.search)):
            self.assertEqual(p.develop(task='opaque label',X_train=x,Y_train=y,X_holdout=h,Y_holdout=hy,values=h,**WRITE).status,'accepted')
        before=fingerprint(p)
        with patch.object(BlockSearch,'synthesize',side_effect=AssertionError('Heldout-driven retry')):
            result=p.develop(task='opaque label',X_train=x,Y_train=y,X_holdout=h,Y_holdout=h[:,::-1].copy(),values=h,**WRITE)
        self.assertEqual(result.status,'rejected');self.assertEqual(fingerprint(p),before)

if __name__=='__main__':unittest.main()
