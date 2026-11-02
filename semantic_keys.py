"""Learned character normalization and continuous compositional property keys."""
import jax,jax.numpy as jnp
CHAR_WIDTH=32
QUERY_WEIGHTS=('embed','window_w','window_b','mode_w','mode_b','token_w','position','query_w')

def initial(params,batch):
 if 'sem_relation_embed' not in params:return (None,None)
 shape=(batch,params['open_slot_bias'].shape[0],params['token_embed'].shape[1])
 return jnp.zeros(shape),jnp.zeros(shape)

def query_control(params,past):
 local=dict(params)
 for name in QUERY_WEIGHTS:local['open_'+name]=params['sem_query_'+name]
 from open_neural_memory import control_windows
 return control_windows(local,past)

def characters(params,past,start,end):
 positions=jnp.arange(past.shape[1])[None,:];selected=(positions>=start[:,None])&(positions<=end[:,None])
 chars=params['sem_char_table'][past].astype(jnp.int32)*selected[...,None]
 chars=chars.reshape(len(past),-1);ranks=jnp.cumsum(chars>0,axis=-1)-1
 result=jnp.max(jnp.where((ranks[:,:,None]==jnp.arange(CHAR_WIDTH)[None,None,:])&(chars[:,:,None]>0),chars[:,:,None],0),axis=1)
 valid=jnp.sum(chars>0,axis=-1)<=CHAR_WIDTH
 return result,valid

def canonical_logits(params,chars):
 feature=params['sem_name_embed'][chars];padded=jnp.pad(feature,((0,0),(3,3),(0,0)))
 local=jnp.concatenate([padded[:,i:i+CHAR_WIDTH] for i in range(7)],axis=-1)
 length=jnp.sum(chars>0,axis=-1);distance=jnp.clip(length[:,None]-1-jnp.arange(CHAR_WIDTH)[None,:],0,5)
 local=jnp.concatenate([local,jax.nn.one_hot(distance,6)],axis=-1)
 h=jnp.tanh(local@params['sem_name_w']+params['sem_name_b'])
 logits=h@params['sem_name_out']+params['sem_name_out_b']
 return logits+jnp.array([0.,3.]) if params['sem_name_out_b'].shape[0]==2 else logits+jax.nn.one_hot(chars,params['sem_name_out_b'].shape[0])*3.

def entity_key(params,chars):
 symbols=jnp.argmax(canonical_logits(params,chars),axis=-1)
 if params['sem_name_out_b'].shape[0]==2:symbols=jnp.where(symbols>0,chars,0)
 symbols=jnp.where(chars>0,symbols,0)
 positions=jnp.maximum(jnp.cumsum(symbols>0,axis=-1)-1,0)
 codes=params['sem_char_codes'][positions,symbols]
 vector=jnp.sum(codes*(symbols>0)[...,None],axis=1)
 vector=vector/jnp.maximum(jnp.linalg.norm(vector,axis=-1,keepdims=True),1e-6)
 return vector

def property_key(params,past,start,end):
 positions=jnp.arange(past.shape[1])[None,:];selected=(positions>=start[:,None])&(positions<=end[:,None])
 vector=jnp.sum(params['sem_relation_embed'][past]*selected[...,None],axis=1)
 return vector/jnp.maximum(jnp.linalg.norm(vector,axis=-1,keepdims=True),1e-6)

def encode(params,past,pointers):
 from open_neural_memory import encode_keys
 indices=jnp.argmax(pointers,axis=-1);chars,valid=characters(params,past,indices[:,0],indices[:,1])
 raw_e,raw_r=encode_keys(params,past,pointers)
 valid=valid&(jnp.linalg.norm(raw_e,axis=-1)>.5)&(jnp.linalg.norm(raw_r,axis=-1)>.5)
 entity=entity_key(params,chars)*valid[:,None]
 relation=property_key(params,past,indices[:,2],indices[:,3])*valid[:,None]
 return entity,relation
