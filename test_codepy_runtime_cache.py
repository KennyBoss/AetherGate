"""Cached preparation must never bypass mutation, provenance or NumPy checks."""
import copy,json,unittest
from unittest.mock import patch
import numpy as np
from research.codepy.ssm_dispatch.pipeline import Pipeline
from research.codepy.ssm_dispatch.catalog import ROOT,catalog
from research.codepy.ssm_dispatch import hierarchy as h,examples,programs

class RuntimeCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry=json.loads((ROOT/'artifacts/codepy_performance_20261101/entries/hierarchical_0_sort8.json').read_text())
        cls.x=np.random.default_rng(319).normal(size=(17,8)).astype(np.float32)
    def setUp(self):
        self.p=Pipeline(0);self.p.programs={96:copy.deepcopy(self.entry)}
    def test_reuses_verified_snapshot_across_engines_and_requests(self):
        with patch.object(programs,'verifier_for',wraps=programs.verifier_for) as build:
            for _ in range(3):
                a=self.p._execute_command(96,self.x);b=self.p._execute_command(96,self.x,independent=True);np.testing.assert_array_equal(a,b)
            self.assertEqual(build.call_count,1)
    def test_nested_candidate_mutation_cannot_use_warm_cache(self):
        self.p._execute_command(96,self.x)
        self.p.programs[96]['language']['dependencies'][0]['entry']['candidate']['program']=[0]
        with self.assertRaises(ValueError):self.p._execute_command(96,self.x)
    def test_top_level_verification_packet_is_checked(self):
        self.p._execute_command(96,self.x)
        target=next(t for t in self.p.programs[96]['verification']['holdout']['targets'] if len(set(t))>1);target.reverse()
        with self.assertRaises(ValueError):self.p._execute_command(96,self.x)
    def test_provenance_is_checked_even_on_cache_hit(self):
        self.p._execute_command(96,self.x)
        with patch.object(programs,'blocks_for',side_effect=ValueError('changed provenance')):
            with self.assertRaisesRegex(ValueError,'provenance'):self.p._execute_command(96,self.x)
    def test_verifier_source_change_invalidates_cached_proof(self):
        self.p._execute_command(96,self.x)
        from research.codepy.ssm_dispatch import pipeline
        original=pipeline.digest
        def changed(path):return 'changed' if str(path).endswith('/examples.py') else original(path)
        with patch.object(pipeline,'digest',side_effect=changed),patch.object(examples,'digest',side_effect=changed):
            with self.assertRaises(ValueError):self.p._execute_command(96,self.x)
    def test_cache_is_bounded(self):
        for command in range(96,105):self.p.programs[command]=self.p.programs[96]
        for command in range(96,105):self.p._execute_command(command,self.x)
        self.assertEqual(len(self.p._execution_cache),8)
    def test_execute_shares_preparation_but_still_runs_independent_numpy(self):
        self.p._text=lambda _:None;self.p.params=self.p.encoder=self.p.semantic=None;self.p.last_text_token=''
        self.p.recall=lambda _: (96,self.x)
        self.p._execute_command(96,self.x)
        from research.codepy.ssm_dispatch import ieee_runtime
        original=ieee_runtime.execute
        def wrong(self,candidate,x,*,independent=False):
            y=original(self,candidate,x,independent=independent)
            return y+np.float32(1) if independent else y
        with patch('research.codepy.ssm_dispatch.pipeline.ru.neural_reply',return_value=('handle',None)),patch.object(ieee_runtime,'execute',wrong),patch.object(self.p,'_command_executor',wraps=self.p._command_executor) as prepare:
            with self.assertRaises(AssertionError):self.p.execute('question')
            self.assertEqual(prepare.call_count,1)
if __name__=='__main__':unittest.main()
