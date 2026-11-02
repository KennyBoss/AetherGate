"""Open-vocabulary neural span addressing; no entity/relation dictionaries at inference."""
import jax
import jax.numpy as jnp

WINDOW=48
FIELDS=('open_values','open_entities','open_relations','open_used','open_clock','open_write_slot','open_read_slot','open_write_pos','open_read_pos','open_write_active','open_read_active','open_context')


def initial(params,batch):
    if 'open_embed' not in params:return (None,)*len(FIELDS)
    slots=params['open_slot_bias'].shape[0];length=params['open_read_scale'].shape[0];dim=params['token_embed'].shape[1]
    z=lambda *shape:jnp.zeros(shape,jnp.float32)
    i=lambda:jnp.zeros((batch,),jnp.int32)
    return (z(batch,slots,length,dim),z(batch,slots,dim),z(batch,slots,dim),z(batch,slots),i(),i(),i(),i(),i(),z(batch),z(batch),jnp.zeros((batch,WINDOW-1),jnp.int32))


def windows(ids,context):
    joined=jnp.concatenate([context,ids],axis=1)
    past=jax.vmap(lambda t:jax.lax.dynamic_slice_in_dim(joined,t,WINDOW,axis=1))(jnp.arange(ids.shape[1]))
    return jnp.swapaxes(past,0,1),joined[:,-WINDOW+1:]


def control_windows(params,past,content_mask=None):
    features=params['open_embed'][past]
    if content_mask is not None:features=features*content_mask[...,None]
    end=params['open_word_end'][past];begin=jnp.concatenate([jnp.ones((len(past),1)),end[:,:-1]],axis=1)
    features=jnp.concatenate([features,end[...,None],begin[...,None]],axis=-1)
    h=jnp.tanh(features.reshape(len(past),-1)@params['open_window_w']+params['open_window_b'])
    mode=h@params['open_mode_w']+params['open_mode_b']
    padded=jnp.pad(features,((0,0),(2,2),(0,0)))
    local=jnp.concatenate([padded[:,i:i+WINDOW] for i in range(5)],axis=-1)
    keys=jnp.tanh(local@params['open_token_w']+params['open_position'])
    query=(h@params['open_query_w']).reshape(len(past),4,32)
    pointers=jnp.einsum('bqd,btd->bqt',query,keys)
    valid=jnp.stack([begin,end,begin,end],axis=1)>0
    pointers=jnp.where(valid,pointers,-1e4)
    return mode,pointers


def encode_keys(params,past,pointers):
    # Relative-span recurrence gives identical keys for new token sequences in different contexts.
    embed=params['token_embed'];embed=embed/jnp.maximum(jnp.linalg.norm(embed,axis=-1,keepdims=True),1e-6)
    positions=jnp.arange(WINDOW)[None,:]
    indices=jnp.argmax(pointers,axis=-1)
    def encode(start,end):
        selected=(positions>=start[:,None])&(positions<=end[:,None])
        def step(acc,pair):
            token,use=pair;candidate=.8*acc+embed[token]
            return jnp.where(use[:,None],candidate,acc),None
        value,_=jax.lax.scan(step,jnp.zeros((len(past),embed.shape[1]),jnp.float32),(past.T,selected.T))
        return value/jnp.maximum(jnp.linalg.norm(value,axis=-1,keepdims=True),1e-6)
    entity,relation=encode(indices[:,0],indices[:,1]),encode(indices[:,2],indices[:,3])
    if 'open_schema_guard' in params:
        es,ee,rs,re=[indices[:,i] for i in range(4)]
        em=(positions>=es[:,None])&(positions<=ee[:,None]);rm=(positions>=rs[:,None])&(positions<=re[:,None])
        ends=params['open_word_end'][past];words=params['open_token_is_word'][past]>0
        # A predicted address must contain a whole single-word name and disjoint word-only relation span.
        valid=(es<=ee)&(rs<=re)&((ee<rs)|(re<es))
        valid=valid&(jnp.sum(ends*em,axis=-1)==1)&(jnp.sum(ends*rm,axis=-1)>=1)
        valid=valid&jnp.all(~(em|rm)|words,axis=-1)
        entity=entity*valid[:,None];relation=relation*valid[:,None]
    return entity,relation


def apply(params,ids,base_logits,state,write_mask):
    past,context=windows(ids,state.open_context)
    modes,pointers=control_windows(params,past.reshape(-1,WINDOW))
    modes=modes.reshape(*ids.shape,3);pointers=pointers.reshape(*ids.shape,4,WINDOW)
    probability=jax.nn.softmax(modes,axis=-1)
    starts=probability[...,1]>=params['open_mode_threshold'][0]
    queries=probability[...,2]>=params['open_mode_threshold'][1]
    semantic='sem_relation_embed' in params
    if semantic:
        import semantic_keys as sem
        def query_head(_):
            mode,span=sem.query_control(params,past.reshape(-1,WINDOW))
            enabled=(write_mask<=0)[:,None]
            return (jax.nn.softmax(mode,axis=-1)[:,2].reshape(ids.shape)>=params['sem_query_threshold'])&enabled,span.reshape(*ids.shape,4,WINDOW)
        sem_queries,sem_pointers=jax.lax.cond(jnp.any(write_mask<=0),query_head,lambda _:(jnp.zeros(ids.shape,bool),pointers),None)
    else:sem_queries=jnp.zeros(ids.shape,bool);sem_pointers=pointers
    external='v6_projection' in params
    if external:
        import general_language as general
        def extra_head(_):
            mode,span=general.query_control(params,past.reshape(-1,WINDOW))
            return (jax.nn.softmax(mode,axis=-1)[:,2].reshape(ids.shape)>=params['v6_query_threshold'])&(write_mask<=0)[:,None],span.reshape(*ids.shape,4,WINDOW)
        extra_queries,extra_pointers=jax.lax.cond(jnp.any(write_mask<=0),extra_head,lambda _:(jnp.zeros(ids.shape,bool),pointers),None)
    else:extra_queries=jnp.zeros(ids.shape,bool);extra_pointers=pointers
    embed=params['token_embed'];embed=embed/jnp.maximum(jnp.linalg.norm(embed,axis=-1,keepdims=True),1e-6)
    length=params['open_read_scale'].shape[0];slots=params['open_slot_bias'].shape[0]
    carry=tuple(getattr(state,name) for name in FIELDS if name!='open_context')+(getattr(state,'open_sem_entities',None),getattr(state,'open_sem_relations',None),getattr(state,'open_external_relations',None))
    def step(carry,items):
        values,entities,relations,used,clock,wslot,rslot,wpos,rpos,wactive,ractive,semantic_entities,semantic_relations,external_relations=carry
        token,base,start,query,word_window,pointer,sem_query,sem_pointer,extra_query,extra_pointer=items
        event=(start&(write_mask>0))|query|sem_query|extra_query
        def find(_):
            ek,rk=encode_keys(params,word_window,pointer)
            scores=jnp.minimum(jnp.einsum('bsd,bd->bs',entities,ek),jnp.einsum('bsd,bd->bs',relations,rk))+(used-1.)*4.+params['open_slot_bias'][None,:]
            rawbest=jnp.argmax(scores,axis=-1);rawmatched=jnp.max(scores,axis=-1)>=params['open_match_threshold']
            rawvalid=(jnp.linalg.norm(ek,axis=-1)>.5)&(jnp.linalg.norm(rk,axis=-1)>.5)
            readbest=rawbest;readmatched=query&rawmatched&rawvalid
            se=jnp.zeros_like(ek);sr=jnp.zeros_like(rk)
            if semantic:
                chosen=jnp.where(sem_query[:,None,None],sem_pointer,pointer)
                se,sr=sem.encode(params,word_window,chosen)
                qe,qr=encode_keys(params,word_window,chosen)
                qs=jnp.minimum(jnp.einsum('bsd,bd->bs',entities,qe),jnp.einsum('bsd,bd->bs',relations,qr))+(used-1.)*4.
                qbest=jnp.argmax(qs,axis=-1);qmatched=(jnp.max(qs,axis=-1)>=params['open_match_threshold'])&sem_query
                eligible=(jnp.einsum('bsd,bd->bs',semantic_entities,se)>=params['open_match_threshold'])&(used>.5)
                ss=jnp.where(eligible,jnp.einsum('bsd,bd->bs',semantic_relations,sr),-4.)
                sbest=jnp.argmax(ss,axis=-1);top=jnp.max(ss,axis=-1)
                runner=jnp.max(jnp.where(jnp.arange(slots)[None,:]!=sbest[:,None],ss,-4.),axis=-1)
                smatched=(top>=params['sem_match_threshold'])&((top-runner)>=params['sem_match_margin'])&sem_query
                use_raw=qmatched&~readmatched;readbest=jnp.where(use_raw,qbest,readbest);readmatched=readmatched|qmatched
                readbest=jnp.where(smatched&~readmatched,sbest,readbest);readmatched=readmatched|smatched
            xr=jnp.zeros_like(ek)
            if external:
                ge,gr,gx=general.encode(params,word_window,extra_pointer)
                entity_match=(jnp.einsum('bsd,bd->bs',semantic_entities,ge)>=params['open_match_threshold'])&(used>.5)
                learned=jnp.where(entity_match,jnp.einsum('bsd,bd->bs',semantic_relations,gr),-4.)
                ext=jnp.where(entity_match,jnp.einsum('bsd,bd->bs',external_relations,gx),-4.)
                def choose(scores,threshold,margin):
                    i=jnp.argmax(scores,axis=-1);top=jnp.max(scores,axis=-1);runner=jnp.max(jnp.where(jnp.arange(slots)[None,:]!=i[:,None],scores,-4.),axis=-1)
                    return i,(top>=threshold)&(top-runner>=margin)&extra_query
                gi,gm=choose(learned,params['sem_match_threshold'],params['sem_match_margin'])
                xi,xm=choose(ext,params['v6_match_threshold'],params['v6_match_margin'])
                new=gm|xm;best_extra=jnp.where(gm,gi,xi)
                readbest=jnp.where(new&~readmatched,best_extra,readbest);readmatched=readmatched|new
                write_indices=jnp.argmax(pointer,axis=-1)
                xr=general.external_key(params,word_window,write_indices[:,2],write_indices[:,3])
            return ek,rk,rawbest,rawmatched,readbest,readmatched,se,sr,xr
        empty=lambda _:(jnp.zeros((len(ids),embed.shape[1])),jnp.zeros((len(ids),embed.shape[1])),jnp.zeros((len(ids),),jnp.int32),jnp.zeros((len(ids),),bool),jnp.zeros((len(ids),),jnp.int32),jnp.zeros((len(ids),),bool),jnp.zeros((len(ids),embed.shape[1])),jnp.zeros((len(ids),embed.shape[1])),jnp.zeros((len(ids),embed.shape[1])))
        ek,rk,best,matched,readbest,readmatched,se,sr,xr=jax.lax.cond(jnp.any(event),find,empty,None)
        valid=(jnp.linalg.norm(ek,axis=-1)>.5)&(jnp.linalg.norm(rk,axis=-1)>.5)
        start=start&valid&(write_mask>0)
        target=jnp.where(matched,best,clock%slots);batch=jnp.arange(len(ids))
        def allocate(data):
            entities,relations,used,semantic_entities,semantic_relations,external_relations=data
            entities=entities.at[batch,target].set(jnp.where(start[:,None],ek,entities[batch,target]))
            relations=relations.at[batch,target].set(jnp.where(start[:,None],rk,relations[batch,target]))
            used=used.at[batch,target].set(jnp.where(start,1.,used[batch,target]))
            if semantic:
                semantic_entities=semantic_entities.at[batch,target].set(jnp.where(start[:,None],se,semantic_entities[batch,target]))
                semantic_relations=semantic_relations.at[batch,target].set(jnp.where(start[:,None],sr,semantic_relations[batch,target]))
            if external:
                external_relations=external_relations.at[batch,target].set(jnp.where(start[:,None],xr,external_relations[batch,target]))
            return entities,relations,used,semantic_entities,semantic_relations,external_relations
        entities,relations,used,semantic_entities,semantic_relations,external_relations=jax.lax.cond(jnp.any(start),allocate,lambda x:x,(entities,relations,used,semantic_entities,semantic_relations,external_relations))
        clock=clock+(start&~matched).astype(jnp.int32)
        delayed='open_write_after_delimiter' in params
        wslot=jnp.where(start,target,wslot);wpos=jnp.where(start,-1 if delayed else 0,jnp.minimum(wpos+1,length-1))
        writing=((wactive>.5)&~start if delayed else (start|(wactive>.5)))&(write_mask>0)
        def write(v):
            current=v[batch,wslot,wpos]
            return v.at[batch,wslot,wpos].set(jnp.where(writing[:,None],embed[token],current))
        values=jax.lax.cond(jnp.any(writing),write,lambda v:v,values)
        terminal=token==params['open_terminal_id']
        wactive=((writing|start)&~terminal&(wpos<length-1)).astype(jnp.float32)
        query=query|sem_query|extra_query
        reading=jnp.where(query,readmatched,(ractive>.5)&~terminal)
        rslot=jnp.where(query&readmatched,readbest,rslot);rpos=jnp.where(query,0,jnp.minimum(rpos+1,length-1))
        def decode(_):
            vector=values[jnp.arange(len(ids)),rslot,rpos]
            output=(vector@embed.T)*params['open_read_scale'][rpos,None]
            return jnp.where(reading[:,None],output,base)
        logits=jax.lax.cond(jnp.any(reading),decode,lambda _:base,None)
        ractive=(reading&(rpos<length-1)).astype(jnp.float32)
        return (values,entities,relations,used,clock,wslot,rslot,wpos,rpos,wactive,ractive,semantic_entities,semantic_relations,external_relations),logits
    final,logits=jax.lax.scan(step,carry,tuple(jnp.swapaxes(v,0,1) for v in [ids,base_logits,starts,queries,past,pointers,sem_queries,sem_pointers,extra_queries,extra_pointers]))
    updates=dict(zip([n for n in FIELDS if n!='open_context'],final[:11]));updates['open_context']=context
    if semantic:updates.update(open_sem_entities=final[11],open_sem_relations=final[12])
    if external:updates['open_external_relations']=final[13]
    return jnp.swapaxes(logits,0,1),state._replace(**updates)
