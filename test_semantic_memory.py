import unittest
from collections import namedtuple
from unittest.mock import patch
import jax,jax.numpy as jnp,numpy as np
import open_neural_memory as m
import semantic_keys as sem
State=namedtuple('State',m.FIELDS+('open_sem_entities','open_sem_relations'))
class SemanticMemoryTests(unittest.TestCase):
 def setUp(self):
  self.p={'token_embed':jnp.eye(20),'open_embed':jnp.zeros((20,1)),'open_slot_bias':jnp.zeros((4,)),'open_read_scale':jnp.ones((4,))*64,'open_terminal_id':jnp.array(5.),'open_mode_threshold':jnp.array([.999,.95]),'open_match_threshold':jnp.array(.9995),'open_write_after_delimiter':jnp.array(1.),'sem_relation_embed':jnp.zeros((20,20)),'sem_match_threshold':jnp.array(.9),'sem_match_margin':jnp.array(.08),'sem_query_threshold':jnp.array(.95)}
  self.s=State(*m.initial(self.p,1),*sem.initial(self.p,1))
 def control(self,p,past,content_mask=None):
  last=past[:,-1];label=jnp.where(last==4,1,jnp.where(last==9,2,0));mode=jax.nn.one_hot(label,3)*40
  positions=jnp.where((last==4)[:,None],jnp.array([45,45,46,46]),jnp.array([44,44,45,45]));pointer=jax.nn.one_hot(positions,48)*40
  return mode,pointer
 def encode(self,p,past,pointer):
  positions=pointer.argmax(-1);e=jnp.take_along_axis(past,positions[:,0:1],axis=-1)[:,0];r=jnp.take_along_axis(past,positions[:,2:3],axis=-1)[:,0]
  e=jnp.where((e==3)|(e==11),1,e);r=jnp.where(r==12,2,r)
  return jax.nn.one_hot(e,20),jax.nn.one_hot(r,20)
 def run_tokens(self,state,tokens,write=True):
  ids=jnp.array([tokens],jnp.int32)
  with patch.object(m,'control_windows',self.control),patch.object(sem,'query_control',self.control),patch.object(sem,'encode',self.encode):
   return m.apply(self.p,ids,jnp.zeros((1,len(tokens),20)),state,jnp.array([float(write)]))
 def fact(self,state,e,r,v):return self.run_tokens(state,[e,r,4,v,5])[1]
 def answer(self,state,e,r):return int(np.asarray(self.run_tokens(state,[e,r,10,9],False)[0])[0,-1].argmax())
 def test_different_surface_keys_read_same_semantic_binding(self):
  state=self.fact(self.s,1,2,6)
  self.assertEqual(self.answer(state,3,12),6)
  self.assertEqual(self.answer(state,1,2),6)
 def test_semantic_query_cannot_write_any_persistent_bank(self):
  s=self.fact(self.s,1,2,6);_,after=self.run_tokens(s,[3,12,10,9],False)
  for key in ['open_values','open_entities','open_relations','open_used','open_clock','open_sem_entities','open_sem_relations']:np.testing.assert_array_equal(getattr(s,key),getattr(after,key))
 def test_ambiguous_alias_is_rejected_but_exact_old_key_still_reads(self):
  s=self.fact(self.s,1,2,6);s=self.fact(s,11,2,7)
  self.assertEqual(self.answer(s,3,12),0)
  self.assertEqual(self.answer(s,1,2),6);self.assertEqual(self.answer(s,11,2),7)
 def test_distinct_properties_never_merge_on_write(self):
  s=self.fact(self.s,1,2,6);s=self.fact(s,1,12,7)
  self.assertEqual(int(s.open_clock[0]),2)
  self.assertEqual(self.answer(s,1,2),6);self.assertEqual(self.answer(s,1,12),7)
 def test_literal_correction_updates_semantic_answer_without_new_slot(self):
  s=self.fact(self.s,1,2,6);s=self.fact(s,1,2,8)
  self.assertEqual(int(s.open_clock[0]),1);self.assertEqual(self.answer(s,3,12),8)
 def test_character_packing_is_independent_of_bpe_segmentation(self):
  table=jnp.array([[0,0,0,0,0],[1,0,0,0,0],[2,3,0,0,0],[4,1,0,0,0],[1,2,3,4,1]],jnp.float32)
  chars,valid=sem.characters({'sem_char_table':table},jnp.array([[1,2,3],[0,0,4]]),jnp.array([0,2]),jnp.array([2,2]))
  np.testing.assert_array_equal(chars[0],chars[1]);self.assertTrue(np.asarray(valid).all())
 def test_compaction_handles_internal_and_final_deletion(self):
  chars=jnp.array([[1,2,3,4,5]+[0]*27,[1,2,3,5,2]+[0]*27]);keep=jnp.ones_like(chars).at[0,3].set(0).at[1,4].set(0)
  codes=jax.random.normal(jax.random.PRNGKey(71),(32,6,20));p={'sem_name_out_b':jnp.zeros((2,)),'sem_char_codes':codes}
  with patch.object(sem,'canonical_logits',lambda p,x:jax.nn.one_hot(keep,2)*50):e=sem.entity_key(p,chars)
  np.testing.assert_allclose(e[0],e[1],atol=1e-6)
if __name__=='__main__':unittest.main()
