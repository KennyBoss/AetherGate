"""Variable-arity page storage, longer synthesized chains and old-session compatibility."""
import itertools,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import jax.numpy as jnp
from research.codepy.ssm_dispatch.pipeline import Pipeline
from research.codepy.ssm_dispatch.examples import ExamplePairs,BlockSearch,Verifier
from research.codepy.ssm_dispatch.programs import Candidate,CertificationError
from test_codepy_development import fingerprint,WRITE
from test_codepy_pipeline import QUESTION,X

class VariableTests(unittest.TestCase):
    def test_chunked_neural_roundtrip_five_eight_and_cross_chunk_nine(self):
        p=Pipeline(0)
        for n in (5,8,9,16):
            x=np.random.default_rng(800+n).normal(size=(129,n)).astype(np.float32)
            x[0,:4]=[0.,-0.,np.finfo(np.float32).max,np.nextafter(np.float32(0),np.float32(1))]
            train=np.random.default_rng(900+n).normal(size=(17,n)).astype(np.float32)
            result=p.develop(X_train=train,Y_train=train,X_holdout=x,Y_holdout=x,values=x,max_calls=16,**WRITE)
            self.assertEqual(result.status,'accepted')
            command,recovered=p.recall(WRITE['handle']);self.assertEqual(command,result.command)
            np.testing.assert_array_equal(recovered.view(np.uint32),x.view(np.uint32))
            record=p.records[WRITE['handle']]
            self.assertEqual(record['pages'],(n+3)//4)
            self.assertEqual(len(record['chunks']),(129*record['pages']+127)//128)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'session.npz';p.save(path);q=Pipeline.load(path)
                np.testing.assert_array_equal(q.execute(QUESTION).output.view(np.uint32),x.view(np.uint32))
            state=record['chunks'][0];record['chunks'][0]=state._replace(kv=state.kv.at[3].set(0.))
            with self.assertRaisesRegex(ValueError,'integrity'):p.recall(WRITE['handle'])
            record['chunks'][0]=state
        with self.assertRaisesRegex(ValueError,'arity'):p.remember(command=0,values=np.zeros((1,5),np.float32),**WRITE)

    def test_sort5_is_really_synthesized_with_more_than_six_calls(self):
        p=Pipeline(0)
        x=np.asarray(list(itertools.permutations(np.asarray([-.7,.15,1.2,2.8,4.9],np.float32))),np.float32)
        h=np.random.default_rng(480).uniform(-100,100,(257,5)).astype(np.float32)
        result=p.develop(task='opaque-long',X_train=x,Y_train=np.sort(x,axis=1),X_holdout=h,Y_holdout=np.sort(h,axis=1),values=h,max_calls=16,**WRITE)
        self.assertEqual(result.status,'accepted')
        candidate=p.programs[result.command]['candidate']
        self.assertGreater(len(candidate['program']),6);self.assertEqual(candidate['arity'],5)
        np.testing.assert_array_equal(p.execute(QUESTION).output,np.sort(h,axis=1))
        before=fingerprint(p)
        with patch.object(BlockSearch,'synthesize',return_value=(Candidate(0,'bad-long',(0,)*12,5),{})):
            rejected=p.develop(task='bad-long',X_train=x,Y_train=x,X_holdout=h,Y_holdout=np.sort(h,axis=1),values=h,max_calls=16,**WRITE)
        self.assertEqual(rejected.status,'rejected');self.assertEqual(fingerprint(p),before)

    def test_wrong_arities_and_out_of_range_addresses_fail_closed(self):
        p=Pipeline(0);v=Verifier(p.catalog)
        with self.assertRaises(CertificationError):v.execute(Candidate(0,'x',(0,),5),X)
        with self.assertRaises(CertificationError):v.validate(Candidate(0,'x',(41,),5))
        with self.assertRaises(CertificationError):v.validate(Candidate(0,'x',(0,),1))
        with self.assertRaises(ValueError):BlockSearch(p.catalog).synthesize(ExamplePairs(X,X),max_calls=0)
        with self.assertRaises(ValueError):p.develop(X_train=X,Y_train=X,X_holdout=np.ones((2,5),np.float32),Y_holdout=np.ones((2,5),np.float32),values=X,**WRITE)

if __name__=='__main__':unittest.main()
