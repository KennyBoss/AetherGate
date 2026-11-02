"""Certified macros, independent index interpretation, atomic rejection and persistence."""
import copy,itertools,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.codepy.ssm_dispatch.pipeline import Pipeline
from research.codepy.ssm_dispatch import hierarchy as h
from research.codepy.ssm_dispatch.programs import Candidate,CertificationError,json_hash
from research.codepy.ssm_dispatch.examples_exam import fingerprint
from research.codepy.ssm_dispatch.exam import RECORDS

class HierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=Pipeline(0);rng=np.random.default_rng(901)
        x=np.asarray(list(itertools.permutations(np.asarray([-.7,.1,1.5,3.9],np.float32))),np.float32)
        holdout=rng.normal(size=(129,4)).astype(np.float32)
        for i,fn in enumerate((lambda a:np.sort(a,axis=1),lambda a:a[:,::-1].copy(),lambda a:np.concatenate([np.sort(a[:,:2],axis=1),np.sort(a[:,2:],axis=1)],axis=1))):
            entity,prop,handle,_=RECORDS[i]
            result=cls.p.develop(task=f'skill-{i}',X_train=x,Y_train=fn(x),X_holdout=holdout,Y_holdout=fn(holdout),values=holdout[:1],entity=entity,property=prop,handle=handle)
            assert result.status=='accepted'
        cls.macros=sorted(cls.p.programs);cls.results=[]
        for i,n in enumerate((8,16)):
            x=rng.normal(size=(64,n)).astype(np.float32)
            if n==8:x=np.concatenate([np.asarray(list(itertools.product([.137,.913],repeat=8)),np.float32),np.random.default_rng(921900).uniform(-3,3,(64,8)).astype(np.float32)])
            fn=(lambda a:np.sort(a,axis=1)) if n==8 else lambda a:a[:,::-1].copy()
            holdout=rng.normal(size=(129,n)).astype(np.float32)
            entity,prop,handle,question=RECORDS[i]
            result=cls.p.develop(task=f'wide-skill-{i}',X_train=x,Y_train=fn(x),X_holdout=holdout,Y_holdout=fn(holdout),values=holdout,
                                 use_macros=True,macro_commands=cls.macros,max_calls=24,max_pair_ops=21,entity=entity,property=prop,handle=handle)
            assert result.status=='accepted',result.evidence.get('reason')
            cls.results.append((result,holdout,fn(holdout),question,handle))

    def test_full_loop_and_v5_session_reload(self):
        p=self.p;p.observe('Пётр читает журнал. Антон пьёт воду. Ветер шумит за окном. '*30)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'hierarchy.npz';p.save(path);q=Pipeline.load(path)
            for result,x,y,question,handle in self.results:
                for session in (p,q):
                    out=session.execute(question)
                    np.testing.assert_array_equal(out.arguments.view(np.uint32),x.view(np.uint32))
                    np.testing.assert_array_equal(out.output,y)
                    self.assertTrue(out.independent_agrees)
                    self.assertEqual(session.records[handle]['pages'],x.shape[1]//4)
                self.assertGreater(result.evidence['certificate']['macro_calls'],0)
        self.assertLess(self.results[1][0].evidence['certificate']['calls'],8)
        self.assertLess(self.results[0][0].evidence['certificate']['calls'],21)
        self.assertLessEqual(self.results[0][0].evidence['certificate']['unrolled_pair_ops'],21)

    def test_bad_candidate_rejected_without_mutation(self):
        p=self.p;_,x,y,_,_=self.results[1];train=x[:4]+np.float32(.01)
        search=h.BlockSearch(p,self.macros)
        good,evidence=search.synthesize(h.flat.ExamplePairs(train,train),label='bad')
        before=fingerprint(p)
        with patch.object(h.BlockSearch,'synthesize',return_value=(good,evidence)):
            result=p.develop(task='bad',X_train=train,Y_train=train,X_holdout=x,Y_holdout=y,values=x,use_macros=True,macro_commands=self.macros,
                             entity='клары',property='основная книга',handle='морской атлас')
        self.assertEqual(result.status,'rejected');self.assertEqual(fingerprint(p),before)

    def test_invalid_indices_forged_dependencies_and_tampering(self):
        p=self.p;entry=p.programs[self.results[0][0].command]
        language=copy.deepcopy(entry['language']);index=next(i for i,a in enumerate(language['actions']) if a['kind']=='macro')
        language['actions'][index]['indices']=[0,0,1,2]
        with self.assertRaisesRegex(CertificationError,'distinct'):h.Verifier(p.catalog,language)
        language=copy.deepcopy(entry['language']);language['dependencies'][0]['entry']['candidate']['program']=[0]
        with self.assertRaisesRegex(CertificationError,'dependency'):h.Verifier(p.catalog,language)
        language=copy.deepcopy(entry['language']);language['actions'][index]['indices']=[4,3,2,1]
        with self.assertRaisesRegex(CertificationError,'integrity'):h.Verifier(p.catalog,language,entry['certificate'])
        with self.assertRaises(CertificationError):h.BlockSearch(p,[999])
        with self.assertRaises(CertificationError):h.Verifier(p.catalog,entry['language'],depth=8)

    def test_cost_budget_and_previous_macro_certificates(self):
        p=self.p;source=p.programs[self.macros[0]]
        language={'arity':8,'dependencies':[{'command':self.macros[0],'sha256':json_hash(source),'entry':source}],
                  'actions':[{'kind':'macro','dependency':0,'indices':[0,1,2,3]}],'max_pair_ops':4}
        verifier=h.Verifier(p.catalog,language)
        with self.assertRaisesRegex(CertificationError,'budget'):verifier.execute(Candidate(0,'too-costly',(0,),8),self.results[0][1])
        with self.assertRaises(ValueError):h.BlockSearch(p).synthesize(h.flat.ExamplePairs(self.results[0][1],self.results[0][1]),max_pair_ops=0)
        # Keep the original v1 verifier byte-identical and accept its certified body as a dependency.
        from research.codepy.ssm_dispatch.hierarchy_v1 import Verifier as OldVerifier
        from research.codepy.ssm_dispatch.programs import verifier_for
        old_language=copy.deepcopy(language);old_language.pop('max_pair_ops')
        old=OldVerifier(p.catalog,old_language);c=Candidate(0,'old-skill',(0,),8)
        x=np.random.default_rng(998).normal(size=(17,8)).astype(np.float32);z=np.random.default_rng(999).normal(size=(19,8)).astype(np.float32)
        train=h.flat.ExamplePairs(x,old.execute(c,x));held=h.flat.ExamplePairs(z,old.execute(c,z))
        entry={'mode':'hierarchical-v1','language':old_language,'candidate':c.payload(),'certificate':old.certify(c,train,held),'verification':{'train':train.payload(),'holdout':held.payload()}}
        verifier_for(p.catalog,entry).restore(entry)
        outer={'arity':8,'dependencies':[{'command':999,'sha256':json_hash(entry),'entry':entry}],
               'actions':[{'kind':'macro','dependency':0,'indices':list(range(8))}],'max_pair_ops':5}
        new=h.Verifier(p.catalog,outer);candidate=Candidate(0,'new-skill',(0,),8)
        self.assertEqual(new.unrolled_pair_ops(candidate),5)
        np.testing.assert_array_equal(new.execute(candidate,z),old.execute(c,z,independent=True))

    def test_recursive_macro_and_noncontiguous_ordered_indices(self):
        p=self.p;entry=p.programs[self.results[1][0].command]
        # Reuse the newly accepted 16-cell program itself; exercise nesting and reversed addressing.
        language={'arity':16,'dependencies':[{'command':self.results[1][0].command,'sha256':json_hash(entry),'entry':entry}],
                  'actions':[{'kind':'macro','dependency':0,'indices':list(reversed(range(16)))}]}
        verifier=h.Verifier(p.catalog,language);candidate=Candidate(0,'nested',(0,),16)
        x=np.random.default_rng(991).normal(size=(17,16)).astype(np.float32)
        native=verifier.execute(candidate,x);reference=verifier.execute(candidate,x,independent=True)
        np.testing.assert_array_equal(native,x[:,::-1]);np.testing.assert_array_equal(native,reference)
        # A genuine four-cell skill on a non-contiguous ordered quadruple.
        source=p.programs[self.macros[1]]
        language={'arity':8,'dependencies':[{'command':self.macros[1],'sha256':json_hash(source),'entry':source}],
                  'actions':[{'kind':'macro','dependency':0,'indices':[7,1,5,0]}]}
        v=h.Verifier(p.catalog,language);c=Candidate(0,'indices',(0,),8);a=x[:,:8].copy();expected=a.copy();expected[:,[7,1,5,0]]=a[:,[0,5,1,7]]
        np.testing.assert_array_equal(v.execute(c,a),expected);np.testing.assert_array_equal(v.execute(c,a,independent=True),expected)

if __name__=='__main__':unittest.main()
