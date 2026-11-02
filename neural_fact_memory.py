"""Neural typed-address memory. All language decisions come from trained tensors."""
import jax
import jax.numpy as jnp

FIELDS=('fact_memory','fact_context','fact_write_key','fact_read_key','fact_write_pos','fact_read_pos','fact_write_active','fact_read_active','fact_entity_trace','fact_relation_trace')


def initial(params,batch):
    if 'fact_embed' not in params:return (None,)*len(FIELDS)
    entities=params['fact_entity_w'].shape[1];relations=params['fact_relation_w'].shape[1]
    slots=entities*relations;length=params['fact_read_scale'].shape[0];dim=params['token_embed'].shape[1]
    return (jnp.zeros((batch,slots,length,dim),jnp.float32),
            jnp.zeros((batch,31,params['fact_embed'].shape[1]),jnp.float32),
            jnp.zeros((batch,slots),jnp.float32),jnp.zeros((batch,slots),jnp.float32),
            jnp.zeros((batch,),jnp.int32),jnp.zeros((batch,),jnp.int32),
            jnp.zeros((batch,),jnp.float32),jnp.zeros((batch,),jnp.float32),
            jnp.zeros((batch,entities),jnp.float32) if 'fact_ner_w' in params else None,
            jnp.zeros((batch,relations),jnp.float32) if 'fact_ner_w' in params else None)


def control(params,ids,context):
    embedded=params['fact_embed'][ids];joined=jnp.concatenate([context,embedded],axis=1)
    windows=jax.vmap(lambda t:jax.lax.dynamic_slice_in_dim(joined,t,32,axis=1))(jnp.arange(ids.shape[1]))
    windows=jnp.swapaxes(windows,0,1).reshape(ids.shape[0],ids.shape[1],-1)
    h=jnp.tanh(windows@params['fact_window_w']+params['fact_window_b'])
    return (h@params['fact_mode_w']+params['fact_mode_b'],
            h@params['fact_entity_w']+params['fact_entity_b'],
            h@params['fact_relation_w']+params['fact_relation_b']),joined[:,-31:]


def typed_events(params,ids,context):
    joined=jnp.concatenate([context,params['fact_embed'][ids]],axis=1)
    windows=jax.vmap(lambda t:jax.lax.dynamic_slice_in_dim(joined,t+27,5,axis=1))(jnp.arange(ids.shape[1]))
    windows=jnp.swapaxes(windows,0,1).reshape(ids.shape[0],ids.shape[1],-1)
    h=jnp.tanh(windows@params['fact_ner_w']+params['fact_ner_b'])
    return h@params['fact_ner_entity_w']+params['fact_ner_entity_b'],h@params['fact_ner_relation_w']+params['fact_ner_relation_b']


def hard_distribution(logits):
    soft=jax.nn.softmax(logits,axis=-1);hard=jax.nn.one_hot(jnp.argmax(soft,axis=-1),soft.shape[-1])
    return soft+jax.lax.stop_gradient(hard-soft)


def apply(params,ids,base_logits,state,write_mask):
    (modes,entities,relations),context=control(params,ids,state.fact_context)
    e=hard_distribution(entities);r=hard_distribution(relations)
    addresses=(e[..., :,None]*r[...,None,:]).reshape(*ids.shape,-1)
    if 'fact_ner_w' in params:
        ne,nr=typed_events(params,ids,state.fact_context)
        ne=jax.nn.softmax(ne,axis=-1);nr=jax.nn.softmax(nr,axis=-1)
    else:
        ne=jnp.zeros((*ids.shape,params['fact_entity_w'].shape[1]+1));nr=jnp.zeros((*ids.shape,params['fact_relation_w'].shape[1]+1))
    probability=jax.nn.softmax(modes,axis=-1)
    starts=(probability[...,1]>=params['fact_mode_threshold'][0])
    queries=(probability[...,2]>=params['fact_mode_threshold'][1])
    embed=params['token_embed'];normalized=embed/jnp.maximum(jnp.linalg.norm(embed,axis=-1,keepdims=True),1e-6)
    length=params['fact_read_scale'].shape[0]
    unknown=params['fact_unknown_values'];unknown=unknown/jnp.maximum(jnp.linalg.norm(unknown,axis=-1,keepdims=True),1e-6)
    carry=tuple(getattr(state,n) for n in FIELDS if n!='fact_context')
    def step(carry,items):
        memory,wkey,rkey,wpos,rpos,wactive,ractive,etrace,rtrace=carry
        token,base,start,query,address,entity_event,relation_event=items
        if 'fact_ner_w' in params:
            ec=jnp.argmax(entity_event,axis=-1);rc=jnp.argmax(relation_event,axis=-1)
            et=jax.nn.one_hot(jnp.maximum(ec-1,0),etrace.shape[-1]);rt=jax.nn.one_hot(jnp.maximum(rc-1,0),rtrace.shape[-1])
            etrace=jnp.where(((ec>0)&(jnp.max(entity_event,axis=-1)>=params['fact_ner_threshold'][0]))[:,None],et,etrace)
            rtrace=jnp.where(((rc>0)&(jnp.max(relation_event,axis=-1)>=params['fact_ner_threshold'][1]))[:,None],rt,rtrace)
            address=(etrace[:,:,None]*rtrace[:,None,:]).reshape(len(token),-1)
        writing=(start|(wactive>.5))&(write_mask>0)
        wkey=jnp.where(start[:,None],address,wkey)
        wpos=jnp.where(start,0,jnp.minimum(wpos+1,length-1))
        weights=wkey[:,:,None]*jax.nn.one_hot(wpos,length)[:,None,:]*writing[:,None,None]
        value=normalized[token]
        memory=jax.lax.cond(jnp.any(writing),lambda m:m+weights[...,None]*(value[:,None,None,:]-m),lambda m:m,memory)
        terminal=token==params['fact_terminal_id']
        wactive=(writing&~terminal&(wpos<length-1)).astype(jnp.float32)
        reading=query|((ractive>.5)&~terminal)
        rkey=jnp.where(query[:,None],address,rkey)
        rpos=jnp.where(query,0,jnp.minimum(rpos+1,length-1))
        def decode(_):
            selected=jnp.einsum('bs,bstd->btd',rkey,memory)
            known=jnp.clip(jnp.sum(selected*selected,axis=(-1,-2)),0.,1.)
            vector=jnp.take_along_axis(selected,rpos[:,None,None],axis=1)[:,0,:]
            vector=known[:,None]*vector+(1-known[:,None])*unknown[rpos]
            scores=(vector@normalized.T)*params['fact_read_scale'][rpos,None]
            return jnp.where(reading[:,None],scores,base)
        logits=jax.lax.cond(jnp.any(reading),decode,lambda _:base,None)
        ractive=(reading&(rpos<length-1)).astype(jnp.float32)
        if 'open_embed' in params:
            clear=(terminal&(write_mask>0))[:,None]
            etrace=jnp.where(clear,0.,etrace);rtrace=jnp.where(clear,0.,rtrace)
        return (memory,wkey,rkey,wpos,rpos,wactive,ractive,etrace,rtrace),logits
    final,logits=jax.lax.scan(step,carry,tuple(jnp.swapaxes(x,0,1) for x in [ids,base_logits,starts,queries,addresses,ne,nr]))
    memory,wkey,rkey,wpos,rpos,wactive,ractive,etrace,rtrace=final
    state=state._replace(fact_memory=memory,fact_context=context,fact_write_key=wkey,fact_read_key=rkey,
                         fact_write_pos=wpos,fact_read_pos=rpos,fact_write_active=wactive,fact_read_active=ractive,fact_entity_trace=etrace,fact_relation_trace=rtrace)
    return jnp.swapaxes(logits,0,1),state
