import argparse,tempfile,unittest
from pathlib import Path
import train_ssm_text as core
import russian_memory as ru
import jax,jax.numpy as jnp,numpy as np

class DurableMemoryTests(unittest.TestCase):
 def params(self):
  p=core.init_params(jax.random.PRNGKey(73),20,8,12,variant='context-kv-memory')
  p.update(kv_durable_marker=jnp.array(1.),kv_commit_w=jnp.zeros((12,1)),kv_commit_b=jnp.array([-20.]))
  return p
 def test_closed_commit_keeps_real_neural_memory_exactly(self):
  p=self.params();state=core.init_stream_state(p,2)._replace(kv=jax.random.normal(jax.random.PRNGKey(74),(2,8,8)))
  ids=jnp.tile(jnp.arange(20,dtype=jnp.int32),(2,8))
  _,after=core.forward_chunks(p,ids,state,chunk_size=8)
  np.testing.assert_array_equal(after.kv,state.kv)
  leaking=dict(p);del leaking['kv_durable_marker']
  _,before=core.forward_chunks(leaking,ids,state,chunk_size=8)
  self.assertGreater(float(jnp.linalg.norm(before.kv-state.kv)),.01)
 def test_commit_gradient_and_active_write(self):
  p=self.params();p['kv_commit_b']=jnp.array([2.])
  ids=jnp.array([[3,4,5,6]],dtype=jnp.int32)
  def loss(z):return jnp.sum(core.forward(z,ids,core.init_stream_state(z,1))[1].kv**2)
  self.assertGreater(float(loss(p)),0.)
  grad=jax.grad(loss)(p)
  self.assertGreater(float(jnp.linalg.norm(grad['kv_commit_w'])),0.)
 def test_new_weights_and_carry_survive_checkpoint(self):
  p=self.params();opt=core.init_adam(p)
  state=core.init_stream_state(p,1)._replace(kv=jnp.ones((1,8,8)))
  metrics=core.TrainMetrics(*[jnp.array(0.) for _ in range(4)])
  with tempfile.TemporaryDirectory() as d:
   path=str(Path(d)/'model.npz')
   core.save_checkpoint(path,p,opt,[str(i) for i in range(20)],argparse.Namespace(seq_len=4,streams=1,tokens_per_stream=5),metrics,0,stream_state=state)
   q,_,_,_,loaded=core.load_stream_checkpoint(path)
   for name in ['kv_durable_marker','kv_commit_w','kv_commit_b']:np.testing.assert_array_equal(q[name],p[name])
   np.testing.assert_array_equal(loaded.kv,state.kv)
 def window_params(self):
  p=self.params();p.pop('kv_commit_w')
  p.update(kv_commit_embed=.05*jax.random.normal(jax.random.PRNGKey(75),(20,8)),
           kv_commit_window_w=.1*jax.random.normal(jax.random.PRNGKey(76),(96,32)),
           kv_commit_window_b=jnp.zeros((32,)),kv_commit_out=jnp.ones((32,1))*.1)
  return p
 def test_learned_window_is_causal_and_crosses_chunks(self):
  p=self.window_params();p['kv_commit_b']=jnp.array([2.])
  ids=jnp.array([[3,4,5,6,7,8,9,10,11,12,13,14,15]],dtype=jnp.int32)
  state=core.init_stream_state(p,1)
  whole,finish=core.forward(p,ids,state)
  chunks,carried=core.forward_chunks(p,ids,state,chunk_size=4)
  np.testing.assert_allclose(whole,chunks,rtol=1e-5,atol=1e-6)
  for a,b in zip(jax.tree_util.tree_leaves(finish),jax.tree_util.tree_leaves(carried)):
   np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
  z,_=core.commit_logits(p,ids,state.commit_context)
  prefix,_=core.commit_logits(p,ids[:,:6],state.commit_context)
  np.testing.assert_allclose(z[:,:6],prefix,rtol=1e-6,atol=1e-6)
 def test_window_context_checkpoint(self):
  p=self.window_params();state=core.init_stream_state(p,1)
  _,state=core.forward(p,jnp.array([[3,4,5]],dtype=jnp.int32),state)
  class Enc:
   vocab=[str(i) for i in range(20)]
   config={'kind':'word'}
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'window.npz';ru.save(path,p,core.init_adam(p),Enc(),0,state)
   q,_,_,_,loaded=core.load_stream_checkpoint(str(path))
   np.testing.assert_array_equal(loaded.commit_context,state.commit_context)
   ids=jnp.array([[6,7]],dtype=jnp.int32)
   a,aa=core.forward(p,ids,state);b,bb=core.forward(q,ids,loaded)
   np.testing.assert_array_equal(a,b)
 def test_delta_correction_replaces_projection_at_nonunit_key(self):
  from unittest.mock import patch
  p=self.window_params();p['kv_window_roles_marker']=jnp.array(1.)
  p['kv_commit_out']=jnp.zeros((32,2));p['kv_commit_b']=jnp.zeros((2,))
  p['token_embed']=p['token_embed'].at[3].set(jnp.array([2.,0.,0.,0.,0.,0.,0.,0.]))
  p['kv_deref_token']=jnp.full_like(p['kv_deref_token'],-100.)
  ids=jnp.array([[3,4,5]],dtype=jnp.int32)
  def roles(params,tokens,context):
   value=jnp.where((tokens==4)|(tokens==5),100.,-100.)
   key=jnp.where(tokens==3,100.,-100.)
   return jnp.stack([value,key],axis=-1),context
  with patch.object(core,'commit_logits',roles):
   _,state=core.forward(p,ids,core.init_stream_state(p,1))
  np.testing.assert_allclose(np.asarray(p['token_embed'][3])@np.asarray(state.kv[0]),np.asarray(p['token_embed'][5]),rtol=1e-5,atol=1e-6)
if __name__=='__main__':unittest.main()
