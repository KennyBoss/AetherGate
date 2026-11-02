"""Batched reads preserve the trained codec, live KV checks and session compatibility."""
import hashlib,json,unittest
from pathlib import Path
from unittest.mock import patch
import jax,jax.numpy as jnp,numpy as np
import train_ssm_text as core
from research.codepy.ssm_dispatch import pipeline as module
from research.codepy.ssm_dispatch.pipeline import Pipeline,BATCH,numeric_step,numeric_read
from research.codepy.ssm_scaling import runtime as codec
from test_codepy_development import WRITE

def state_hash(chunks):
    h=hashlib.sha256()
    for a in jax.tree_util.tree_leaves(chunks):h.update(np.asarray(a).tobytes())
    return h.hexdigest()

class KVReadTests(unittest.TestCase):
    def test_same_neural_decoder_all_seeds_all_uint32_bits_and_banks(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                params=codec.load_params(seed);rng=np.random.default_rng(180000+seed)
                words=rng.integers(0,2**32,(2*BATCH,5),dtype=np.uint32)
                words[:3]=np.asarray([[0,95,96,2**32-1,96**4],[1,96**2,96**3,2**31,2**31-1],[0x80000000,1,0x7f7fffff,0x00000001,0xffffffff]],np.uint32)
                chunks=[];old=[]
                for batch in np.split(words,2):
                    state=core.init_stream_state(params,BATCH*3)
                    state,_=numeric_step(params,state,jnp.asarray(batch),jnp.asarray(True));chunks.append(state)
                    _,decoded=numeric_step(params,state,jnp.zeros((BATCH,5),jnp.uint32),jnp.asarray(False));old.append(np.asarray(decoded))
                before=state_hash(chunks)
                joined=jax.tree_util.tree_map(lambda *a:jnp.concatenate(a,axis=0),*chunks)
                decoded=np.asarray(numeric_read(params,joined))
                np.testing.assert_array_equal(decoded,np.concatenate(old));np.testing.assert_array_equal(decoded,words)
                self.assertEqual(state_hash(chunks),before)

    def test_partial_group_padding_repeated_reads_and_live_corruption(self):
        p=Pipeline(0);x=np.random.default_rng(4100).normal(size=(1025,4)).astype(np.float32)
        x[0]=[0.,-0.,np.finfo(np.float32).max,np.nextafter(np.float32(0),np.float32(1))]
        p.remember(command=0,values=x,**WRITE);record=p.records[WRITE['handle']]
        self.assertEqual(len(record['chunks']),9);before=state_hash(record['chunks'])
        with patch.object(module,'numeric_read',wraps=module.numeric_read) as read:
            for _ in range(2):
                command,y=p.recall(WRITE['handle']);self.assertEqual(command,0)
                np.testing.assert_array_equal(y.view(np.uint32),x.view(np.uint32))
            self.assertEqual(read.call_count,4)
        self.assertEqual(state_hash(record['chunks']),before)
        state=record['chunks'][-1];record['chunks'][-1]=state._replace(kv=jnp.zeros_like(state.kv))
        with self.assertRaisesRegex(ValueError,'integrity'):p.recall(WRITE['handle'])

    def test_word_checksum_command_arity_and_finite_checks_remain(self):
        p=Pipeline(0);x=np.asarray([[3.,2.,1.,0.]],np.float32);p.remember(command=0,values=x,**WRITE)
        record=p.records[WRITE['handle']]
        raw=np.asarray(module.numeric_read(p.numeric_params,record['chunks'][0])).copy()
        old_sha=record['sha256']
        for kind in ('checksum','command','arity','finite'):
            with self.subTest(kind=kind):
                words=raw.copy()
                if kind=='checksum':words[0,1]^=np.uint32(1)
                if kind=='command':words[0,0]=np.uint32(2**32-1)
                if kind=='arity':record['width']=5
                if kind=='finite':words[0,1]=np.uint32(0x7fc00000)
                if kind in ('command','finite'):record['sha256']=hashlib.sha256(words[:1].tobytes()).hexdigest()
                with patch.object(module,'numeric_read',return_value=jnp.asarray(words)):
                    with self.assertRaises(ValueError):p.recall(WRITE['handle'])
                record['width']=4;record['sha256']=old_sha

    def test_previous_v7_checkpoint_loads_with_cold_cache_and_bitwise_arrays(self):
        root=Path(__file__).resolve().parent
        source=root/'artifacts/codepy_cache_merge_20261101/v7_seed0.session.npz'
        p=Pipeline.load(source)
        self.assertEqual(p._verifier_cache.stats['size'],0)
        with np.load(source,allow_pickle=False) as z:
            metadata=json.loads(str(z['metadata_json'].item()))
            self.assertEqual(metadata['version']['implementation'],module.V7_IMPLEMENTATION)
            np.testing.assert_array_equal(np.asarray(p.semantic.h).view(np.uint8),z['semantic/h'].view(np.uint8))
        for handle in p.records:
            command,x=p.recall(handle)
            a=p._execute_command(command,x);b=p._execute_command(command,x,independent=True)
            np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))

    def test_previous_batched_v7_checkpoints_all_seeds(self):
        root=Path(__file__).resolve().parent
        from research.codepy.ssm_dispatch.exam import RECORDS
        for seed in range(5):
            with self.subTest(seed=seed):
                source=root/f'artifacts/codepy_kv_read_20261101/v7_seed{seed}.session.npz'
                p=Pipeline.load(source,seed=seed)
                self.assertEqual(p._verifier_cache.stats['size'],0)
                with np.load(source,allow_pickle=False) as z:
                    metadata=json.loads(str(z['metadata_json'].item()))
                    self.assertEqual(metadata['version']['implementation'],module.V7_READ_IMPLEMENTATION)
                    np.testing.assert_array_equal(np.asarray(p.semantic.h).view(np.uint8),z['semantic/h'].view(np.uint8))
                result=p.execute(RECORDS[0][3]);self.assertTrue(result.independent_agrees)
                np.testing.assert_array_equal(result.output,np.sort(result.arguments,axis=1))

    def test_multipage_bits_and_page_checksums_all_seeds(self):
        from tempfile import TemporaryDirectory
        patterns=np.asarray([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
                             0x7f7fffff,0xff7fffff,0x3f800000,0xbf800000,2,0x80000002,0x7f000001,0xff000001],np.uint32)
        for seed in range(5):
            for width,task in ((8,'sort8'),(16,'reverse16')):
                with self.subTest(seed=seed,width=width):
                    p=Pipeline(seed)
                    entry=json.loads((Path(__file__).resolve().parent/f'artifacts/codepy_performance_20261101/entries/hierarchical_{seed}_{task}.json').read_text())
                    p.programs[512]=entry
                    x=np.roll(patterns,seed)[:width].copy().view(np.float32).reshape(1,width)
                    p.remember(command=512,values=x,**WRITE)
                    record=p.records[WRITE['handle']]
                    self.assertEqual(len(record['page_sha256']),width//4)
                    command,y=p.recall(WRITE['handle']);self.assertEqual(command,512)
                    np.testing.assert_array_equal(y.view(np.uint32),x.view(np.uint32))
                    # Re-signing the record cannot conceal a corrupted page.
                    raw=np.asarray(module.numeric_read(p.numeric_params,record['chunks'][0])).copy()
                    raw[width//4-1,4]^=np.uint32(1)
                    old_sha=record['sha256'];record['sha256']=hashlib.sha256(raw[:width//4].tobytes()).hexdigest()
                    with patch.object(module,'numeric_read',return_value=jnp.asarray(raw)):
                        with self.assertRaisesRegex(ValueError,'page.*integrity'):p.recall(WRITE['handle'])
                    record['sha256']=old_sha
                    with TemporaryDirectory() as directory:
                        path=Path(directory)/'session.npz';p.save(path);loaded=Pipeline.load(path,seed=seed)
                        self.assertEqual(loaded.records[WRITE['handle']]['page_sha256'],record['page_sha256'])
                        _,y=loaded.recall(WRITE['handle'])
                        np.testing.assert_array_equal(y.view(np.uint32),x.view(np.uint32))

    def test_previous_fused_v7_checkpoint(self):
        source=Path(__file__).resolve().parent/'artifacts/codepy_kv_fused_20261101/v7_seed0.session.npz'
        p=Pipeline.load(source)
        self.assertEqual(p._verifier_cache.stats['size'],0)
        for handle in p.records:
            command,x=p.recall(handle)
            a=p._execute_command(command,x);b=p._execute_command(command,x,independent=True)
            np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))

if __name__=='__main__':unittest.main()
