#!/usr/bin/env python3
"""Russian text learning and fact QA using the existing KV-Memory SSM core."""
import argparse
from functools import lru_cache, partial
import hashlib
import json
import shutil
from pathlib import Path
import time

import train_ssm_text as core
import jax
import jax.numpy as jnp
import numpy as np
from text_tokenization import (build_tokenizer_config, build_vocab_for_tokenizer,
                              tokenize_with_config, detokenize_with_config, word_tokenize, UNK, BOS)
from run_stream_memory import make_data as old_data

NAMES='анны марии ирины елены ольги дарьи веры зои нины софьи тамары инны аллы галины лидии юлии полины ксении евгении валерии любови людмилы светланы надежды марины оксаны натальи татьяны ларисы александры виктории екатерины'.split()
VALUES='янтарь гранит коралл алмаз сапфир изумруд рубин опал топаз агат малахит кварц нефрит жемчуг аметист оникс'.split()
FACTS=['Кодовое слово {name} — {value}.', 'Для {name} кодовое слово — {value}.',
       'Для {name} выбрали слово {value}.']
QUESTIONS=['Какое кодовое слово у {name}? Ответ:', 'Назови кодовое слово для {name}. Ответ:',
           'Для входа назови кодовое слово {name}. Ответ:']


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))


class Encoder:
    def __init__(self, vocab, config):
        self.vocab=vocab;self.mapping={v:i for i,v in enumerate(vocab)};self.config=config
    @lru_cache(maxsize=32000)
    def encode(self,text):
        return tuple(self.mapping.get(t,self.mapping[UNK]) for t in tokenize_with_config(text,self.config))
    def decode(self,ids):
        return detokenize_with_config([self.vocab[int(i)] for i in ids],self.config)


def qa_records(count, split, seed, *, filler_ids=None, filler_count=24, enc=None, templates_heldout=False):
    """Values are generated labels only; model input is literal Russian text."""
    rng=np.random.default_rng(seed);records=[]
    for _ in range(count):
        people=rng.choice(len(NAMES),4,replace=False)
        def allowed(person):
            options=[v for v in range(len(VALUES)) if
                     ((person+3*v)%10<6 if split=='train' else (person+3*v)%10 in (6,7) if split=='dev' else (person+3*v)%10>=8)]
            return options
        chosen=[int(rng.choice(allowed(int(p)))) for p in people]
        initial=chosen.copy();lines=[]
        for person,value in zip(people,chosen):
            fmt='Для {name} выбрали кодовое слово {value}.' if templates_heldout else FACTS[int(rng.integers(3))]
            lines.append(fmt.format(name=NAMES[person],value=VALUES[value]))
        revised=rng.choice(4,2,replace=False)
        for slot in revised:
            choices=[v for v in allowed(int(people[slot])) if v!=chosen[slot]]
            chosen[slot]=int(rng.choice(choices))
            fmt='Теперь кодовое слово {name} — {value}.' if templates_heldout else 'Исправление: кодовое слово {name} теперь {value}.'
            lines.append(fmt.format(name=NAMES[people[slot]],value=VALUES[chosen[slot]]))
        target=int(rng.integers(4));name=NAMES[people[target]]
        q=('У {name} какое кодовое слово? Ответ:' if templates_heldout else QUESTIONS[int(rng.integers(3))]).format(name=name)
        answer=VALUES[chosen[target]]
        raw=' '.join(lines)
        record={'facts':raw,'question':q,'answer':answer,'name':name,'updated':target in revised,
                'pair':[int(people[target]),chosen[target]],'initial_values':initial}
        if enc is not None:
            if filler_ids is not None and filler_count:
                at=int(rng.integers(max(1,len(filler_ids)-filler_count)))
                filler=list(map(int,filler_ids[at:at+filler_count]))
            else:filler=[]
            prefix=[enc.mapping[BOS]]+list(enc.encode(raw))+filler+list(enc.encode(q))
            answer_ids=list(enc.encode(answer+'.'))
            record['prefix']=prefix;record['answer_ids']=answer_ids
        records.append(record)
    return records


def lexical_words(enc, split):
    words=[]
    excluded=set(NAMES+VALUES+['кодовое','слово','записи','ответ','теперь','исправление'])
    for token in enc.vocab:
        if token.endswith('</w>'):
            word=token[:-4]
            fold=int(hashlib.sha256(word.encode()).hexdigest()[:8],16)%10
            ok=fold<6 if split=='train' else fold in (6,7) if split=='dev' else fold>=8
            if ok and word in enc.actual_words and word.isalpha() and 3<=len(word)<=12 and word not in excluded:words.append(word)
    return sorted(set(words))


def lexical_records(count,split,seed,enc,filler_ids,heldout=False):
    rng=np.random.default_rng(seed);pool=lexical_words(enc,split)
    if len(pool)<16:raise ValueError('Insufficient disjoint lexical pool')
    records=[]
    for _ in range(count):
        keys=list(rng.choice(pool,4,replace=False));values=[str(rng.choice([w for w in pool if w!=key])) for key in keys]
        statements=[]
        for key,value in zip(keys,values):
            fmt=['Кодовое слово записи «{key}» — «{value}».',
                 'Для записи «{key}» выбрали слово «{value}».',
                 'Для записи «{key}» кодовое слово — «{value}».'][int(rng.integers(3))]
            statements.append(fmt.format(key=key,value=value))
        revised=rng.choice(4,2,replace=False)
        for slot in revised:
            values[slot]=str(rng.choice([w for w in pool if w!=values[slot] and w!=keys[slot]]))
            fmt='Теперь для записи «{key}» кодовое слово — «{value}».' if heldout else 'Исправление: кодовое слово записи «{key}» теперь «{value}».'
            statements.append(fmt.format(key=keys[slot],value=values[slot]))
        slot=int(rng.integers(4))
        fmt=('У записи «{key}» какое кодовое слово? Ответ:' if heldout else
             ['Какое кодовое слово у записи «{key}»? Ответ:','Назови кодовое слово записи «{key}». Ответ:'][int(rng.integers(2))])
        question=fmt.format(key=keys[slot]);facts=' '.join(statements)
        at=int(rng.integers(0,len(filler_ids)-24))
        prefix=[enc.mapping[BOS]]+list(enc.encode(facts))+list(map(int,filler_ids[at:at+24]))+list(enc.encode(question))
        records.append({'facts':facts,'question':question,'answer':values[slot],'prefix':prefix,
                        'answer_ids':list(enc.encode(values[slot]+'.')),'updated':slot in revised,'name':keys[slot]})
    return records


def prepare_lexical(root):
    root=Path(root);out=root/'data_lexical';out.mkdir(exist_ok=False);enc=load_encoder(root)
    ids=np.load(root/'data/train.npy')
    records=lexical_records(3072,'train',771,enc,ids)
    old=qa_records(1024,'train',772,filler_ids=ids,enc=enc)
    records+=old
    np.random.default_rng(773).shuffle(records)
    arrays=pack_records(records,width=128)
    np.savez_compressed(out/'qa_train.npz',**arrays)
    write(out/'protocol.json',{'word_pools':{s:lexical_words(enc,s) for s in ['train','dev','test']},
                             'split':'SHA256(word)%10:0..5 train,6..7 dev,8..9 test; key and value use the SAME word pool',
                             'supervision':'Literal Russian quotations, no labels or parsing in neural forward;25% old Russian QA replay',
                             'horizon':128,'examples':[{k:r[k] for k in ['facts','question','answer']} for r in records[:8]]})


def examine_lexical(root,candidates,output):
    root=Path(root);output=Path(output);output.mkdir(exist_ok=False);enc=load_encoder(root)
    for directory in candidates:
        if not (Path(directory)/'FROZEN.json').exists():raise ValueError('Freeze every lexical candidate before testing')
    ids=np.load(root/'data/test.npy')
    records=lexical_records(512,'test',99701,enc,ids)
    changed=lexical_records(512,'test',99702,enc,ids,heldout=True)
    older=qa_records(256,'test',99703,filler_ids=ids,enc=enc)
    rows=[]
    for directory in candidates:
        p,_,_,_,_=core.load_stream_checkpoint(str(Path(directory)/'knowledge.npz'))
        row={'candidate':str(directory),'parameters':sum(int(a.size) for a in jax.tree_util.tree_leaves(p)),
             'new_words':score_qa(p,records,enc),'memory_reset':score_qa(p,records,enc,reset=True),
             'new_word_order':score_qa(p,changed,enc),'old_russian_qa':score_qa(p,older,enc),
             'old_numerical_skill':score_old(p,seed=99704,count=512),'russian_nll':score_lm(p,ids)}
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    write(output/'results.json',{'models':rows,'limits':'QA words are disjoint across splits, but all are single-piece words from the pretraining vocabulary. Multiword answers, arbitrary new character strings and broad semantic reasoning are not established.'})


def prepare_data(root):
    root=Path(root);target=root/'data';target.mkdir(exist_ok=False)
    corpus=root/'corpus'
    natural={s:(corpus/f'{s}.txt').read_text() for s in ['train','dev','test']}
    # Tokenizer uses train prose and TRAIN dialogues only, never held-out pairs or wording.
    raw=qa_records(1024,'train',701)
    dialogues='\n'.join(r['facts']+' '+r['question']+' '+r['answer']+'.' for r in raw)
    fitting=natural['train'][:200000]+'\n'+dialogues*2
    config=build_tokenizer_config(fitting,tokenizer='bpe',bpe_vocab_size=1024,
                                 bpe_min_frequency=2,bpe_train_chars=None)
    tokens=tokenize_with_config(natural['train']+'\n'+dialogues,config)
    _,new_vocab=build_vocab_for_tokenizer(tokens,config)
    vocab=[f'<legacy:{i}>' for i in range(20)]+new_vocab
    enc=Encoder(vocab,config)
    write(target/'tokenizer.json',config);write(target/'vocab.json',vocab)
    metadata={'legacy_reserved_ids':20,'vocab_size':len(vocab),'natural':{},
              'pair_split':'(name_index+3*value_index)%10: 0..5 train,6..7 dev,8..9 test',
              'qa_semantics':'Four fictional codeword assignments, two corrections, prose distraction, one literal question; labels never passed to neural forward',
              'corpus_manifest_sha256':hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest()}
    for split,text in natural.items():
        ids=np.asarray(enc.encode(text),dtype=np.int32);np.save(target/f'{split}.npy',ids)
        metadata['natural'][split]={'tokens':len(ids),'unknown_fraction':float(np.mean(ids==enc.mapping[UNK])),
                                    'sha256':hashlib.sha256(ids.tobytes()).hexdigest()}
    train_ids=np.load(target/'train.npy')
    records=qa_records(4096,'train',702,filler_ids=train_ids,enc=enc)
    arrays=pack_records(records)
    np.savez_compressed(target/'qa_train.npz',**arrays)
    write(target/'qa_train_examples.json', [{k:r[k] for k in ['facts','question','answer','pair','updated']} for r in records[:16]])
    write(target/'metadata.json',metadata)
    print(json.dumps(metadata,ensure_ascii=False),flush=True)


def load_encoder(root):
    root=Path(root)/'data'
    enc=Encoder(json.loads((root/'vocab.json').read_text()),json.loads((root/'tokenizer.json').read_text()))
    enc.actual_words=set(word_tokenize((root.parent/'corpus/train.txt').read_text()))
    return enc


def pack_records(records,width=96):
    x=np.zeros((len(records),width),np.int32);y=x.copy();mask=np.zeros_like(x,dtype=np.float32)
    prompts=x.copy();lengths=np.zeros(len(records),np.int32)
    for row,r in enumerate(records):
        prefix=r['prefix'];answer=r['answer_ids'];seq=prefix+answer
        if len(seq)>width:raise ValueError(f'QA exceeds registered width: {len(seq)}')
        x[row,:len(seq)-1]=seq[:-1];y[row,:len(seq)-1]=seq[1:]
        mask[row,len(prefix)-1:len(seq)-1]=1.
        prompts[row,:len(prefix)]=prefix;lengths[row]=len(prefix)
    return {'x':x,'y':y,'mask':mask,'prompts':prompts,'lengths':lengths}


def expand_parent(parent,vocab,input_dim,state_dim,seed):
    old,_,_,_,_=core.load_stream_checkpoint(str(parent))
    if input_dim<old['token_embed'].shape[1] or state_dim<old['decay_raw'].shape[0]:
        raise ValueError('Expansion cannot discard existing dimensions')
    p=core.init_params(jax.random.PRNGKey(seed),len(vocab),input_dim,state_dim,variant='kv-memory')
    d,h=old['input_b'].shape
    p['token_embed']=p['token_embed'].at[:20,:].set(0)
    p['input_b']=p['input_b'].at[:d,:].set(0)
    p['output_c']=p['output_c'].at[h:,:20].set(0)
    for name,value in old.items():
        if value.ndim==0:p[name]=value
        else:p[name]=p[name].at[tuple(slice(0,n) for n in value.shape)].set(value)
    # Adam is restarted for the expanded parameter shapes; the parent is never overwritten.
    return p,core.init_adam(p)


def loss(params,x,y,mask,kind):
    state=core.init_stream_state(params,len(x))
    logits,_=core.forward_chunks(params,x,state,chunk_size=32)
    nll=-jnp.take_along_axis(jax.nn.log_softmax(logits),y[...,None],axis=-1)[...,0]
    return jnp.sum(nll*mask)/jnp.maximum(1.,jnp.sum(mask))


@partial(jax.jit,static_argnames=('kind','learning_rate'))
def train_update(params,opt,x,y,mask,*,kind,learning_rate):
    value,grads=jax.value_and_grad(loss)(params,x,y,mask,kind)
    grads,norm=core.clip_grads(grads,1.)
    params,opt=core.adam_update(params,grads,opt,learning_rate)
    return params,opt,value,norm


@jax.jit
def old_eval(params,x,y,mask):
    logits,_=core.forward_chunks(params,x,core.init_stream_state(params,len(x)),chunk_size=8)
    correct=(jnp.argmax(logits,axis=-1)==y).astype(jnp.float32)
    return jnp.sum(correct*mask)/jnp.sum(mask)


def score_old(params,seed=81001,count=256):
    data=old_data(count,32,seed)
    return float(np.mean([float(old_eval(params,*(jnp.asarray(a[at:at+32]) for a in data))) for at in range(0,count,32)]))


@jax.jit
def lm_eval(params,x,y):
    logits,_=core.forward_chunks(params,x,core.init_stream_state(params,len(x)),chunk_size=32)
    nll=-jnp.take_along_axis(jax.nn.log_softmax(logits),y[...,None],axis=-1)[...,0]
    return jnp.mean(nll)


def text_batch(ids,seed,step,batch=16,width=128):
    rng=np.random.default_rng(seed*100003+step)
    starts=rng.integers(0,len(ids)-width-1,size=batch)
    seq=np.stack([ids[at:at+width+1] for at in starts])
    return jnp.asarray(seq[:,:-1]),jnp.asarray(seq[:,1:]),jnp.ones((batch,width),jnp.float32)


def score_lm(params,ids):
    values=[]
    for step in range(4):
        x,y,_=text_batch(ids,777,step,batch=8)
        values.append(float(lm_eval(params,x,y)))
    return float(np.mean(values))


def choose_rows(active,new,old):
    return jax.tree_util.tree_map(lambda a,b:jnp.where(active.reshape((-1,)+(1,)*(a.ndim-1)),a,b),new,old)


@partial(jax.jit,static_argnames=('reset','max_tokens'))
def greedy(params,prompts,lengths,*,reset=False,max_tokens=6,question_starts=None):
    state=core.init_stream_state(params,len(prompts));last=jnp.zeros((len(prompts),params['token_embed'].shape[0]))
    def step(carry,pair):
        state,last=carry;t,ids=pair;prior=state
        if reset:
            fresh=core.init_stream_state(params,len(prompts))._replace(h=state.h)
            state=jax.lax.cond(t%8==0,lambda _:fresh,lambda _:state,None)
        logits,new=core.forward(params,ids[:,None],state,memory_write_mask=(t<question_starts).astype(jnp.float32) if question_starts is not None else None)
        active=t<lengths
        return (choose_rows(active,new,prior),jnp.where(active[:,None],logits[:,0],last)),None
    (state,last),_=jax.lax.scan(step,(state,last),(jnp.arange(prompts.shape[1]),prompts.T))
    outputs=[]
    for t in range(max_tokens):
        ids=jnp.argmax(last,axis=-1).astype(jnp.int32);outputs.append(ids)
        if reset:
            fresh=core.init_stream_state(params,len(prompts))._replace(h=state.h)
            state=choose_rows((lengths+t)%8==0,fresh,state)
        logits,state=core.forward(params,ids[:,None],state,memory_write_mask=jnp.zeros_like(lengths,dtype=jnp.float32) if question_starts is not None else None);last=logits[:,0]
    return jnp.stack(outputs,axis=1)


def score_qa(params,records,enc,reset=False):
    width=max(96,((max(len(r['prefix'])+len(r['answer_ids']) for r in records)+31)//32)*32)
    arrays=pack_records(records,width=width);hits=[];details=[];dot=enc.mapping['.']
    for at in range(0,len(records),16):
        prompts=jnp.asarray(arrays['prompts'][at:at+16]);lengths=jnp.asarray(arrays['lengths'][at:at+16])
        outputs=np.asarray(greedy(params,prompts,lengths,reset=reset))
        for r,tokens in zip(records[at:at+16],outputs):
            ids=list(map(int,tokens));ids=ids[:ids.index(dot)] if dot in ids else ids
            answer=enc.decode(ids).strip();correct=answer==r['answer'];hits.append(correct)
            details.append({'question':r['question'],'expected':r['answer'],'answer':answer,'correct':correct,'updated':bool(r['updated'])})
    return {'exact':float(np.mean(hits)),'count':len(hits),'updated_exact':float(np.mean([d['correct'] for d in details if d['updated']])),
            'old_facts_exact':float(np.mean([d['correct'] for d in details if not d['updated']])),'examples':details[:12]}


def save(path,params,opt,enc,updates,state=None):
    path=Path(path);temp=path.with_name(path.stem+'.tmp.npz')
    metrics=core.TrainMetrics(jnp.asarray(0.),jnp.asarray(0.),jnp.asarray(0.),jnp.mean(core.decay_from_raw(params['decay_raw'])))
    core.save_checkpoint(str(temp),params,opt,enc.vocab,argparse.Namespace(seq_len=256,streams=1,tokens_per_stream=257),
                         metrics,updates,tokenizer_config=enc.config,stream_state=state if state is not None else core.init_stream_state(params,1))
    temp.replace(path)


def train(root,output,*,seed=11,input_dim=64,state_dim=128,lm_steps=512,qa_steps=1200,replay=True,resume=None,lexical=False,context_roles=False):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    enc=load_encoder(root);train_ids=np.load(root/'data/train.npy');dev_ids=np.load(root/'data/dev.npy')
    horizon=128 if lexical else 96
    qa_folder='data_lexical' if lexical else 'data'
    with np.load(root/qa_folder/'qa_train.npz') as z:
        if np.any(z['mask'][:,horizon:]):raise ValueError('Registered QA horizon would truncate targets')
        qa={k:z[k][:,:horizon] for k in ['x','y','mask']}
    protocol={'seed':seed,'input_dim':input_dim,'state_dim':state_dim,'lm_steps':lm_steps,'qa_steps':qa_steps,
              'qa_horizon':horizon,'lexical':lexical,'context_roles':context_roles,'old_replay':replay,'resume':str(resume) if resume else None,'parent':'seed11-stream.npz','QA_train_records':len(qa['x']),
              'learning_rates':{'lm':.002,'qa':.003,'old':.001},'evaluation':'Dev only; final examination must be separate after freezing all candidates'}
    snapshot=output/'source';snapshot.mkdir()
    for filename in ['russian_memory.py','train_ssm_text.py','text_tokenization.py','jax_backend.py',
                     'local_attention_ssm.py','run_stream_memory.py','compare_long_context_recall.py',
                     'compare_text_architectures.py','run_mamba_mqar.py','run_kv_scale_law.py']:
        shutil.copy2(Path(__file__).resolve().parent/filename,snapshot/filename)
    protocol['sources']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.iterdir()}
    write(output/'protocol.json',protocol)
    parent=root.parent/'stream_memory_v1_confirmatory/seed11-stream.npz'
    if resume:
        params,opt,vocab,config,_=core.load_stream_checkpoint(str(resume))
        if vocab!=enc.vocab or config.get("tokenizer_config")!=enc.config:raise ValueError('Resume tokenizer/vocabulary mismatch')
    else:
        params,opt=expand_parent(parent,enc.vocab,input_dim,state_dim,seed)
    if context_roles and 'kv_context_w' not in params:
        params['kv_context_w']=jnp.zeros((state_dim,5),dtype=jnp.float32)
        params['kv_context_b']=jnp.zeros((5,),dtype=jnp.float32)
    if lexical or context_roles:opt=core.init_adam(params)
    dev=(lexical_records(128,'dev',99001,enc,dev_ids) if lexical else
         qa_records(128,'dev',90001,filler_ids=dev_ids,enc=enc))
    baseline={'qa':score_qa(params,dev,enc),'old_skill':score_old(params),'lm_nll':score_lm(params,dev_ids)}
    write(output/'baseline.json',baseline);save(output/'expanded_parent.npz',params,opt,enc,0)
    updates=int(opt['step']);events=[];start=time.perf_counter()
    def perform(kind,step):
        nonlocal params,opt,updates
        if kind=='lm':x,y,mask=text_batch(train_ids,seed,step)
        elif kind=='qa':
            at=(step*16)%len(qa['x']);x,y,mask=(jnp.asarray(qa[k][at:at+16]) for k in ['x','y','mask'])
        else:
            data=old_data(32,32,seed*10000+step);x,y,mask=(jnp.asarray(a) for a in data)
        t=time.perf_counter();params,opt,value,norm=train_update(params,opt,x,y,mask,kind=kind,learning_rate={'lm':.002,'qa':.003,'old':.001}[kind])
        jax.block_until_ready(value);elapsed=time.perf_counter()-t;updates+=1
        if not np.isfinite(float(value)):raise RuntimeError(f'Nonfinite {kind} training')
        events.append({'kind':kind,'step':updates,'loss':float(value),'seconds':elapsed,'tokens':int(x.size),'targets':int(np.asarray(mask).sum())})
    for phase,count in [('language',lm_steps),('knowledge',qa_steps)]:
        qa_index=0
        for step in range(count):
            if phase=='language' or step%4==3:perform('lm',step+lm_steps)
            else:
                perform('qa',qa_index);qa_index+=1
            primary=events[-1]
            if replay and step%8==7:perform('old',step+(0 if phase=='language' else lm_steps))
            if (step+1)%100==0 or step+1==count:
                save(output/'latest.npz',params,opt,enc,updates)
                if phase=='knowledge' and step+1==1200:
                    save(output/'knowledge_equal1200.npz',params,opt,enc,updates)
                write(output/'progress.json',{'phase':phase,'step_in_phase':step+1,'updates':updates,'elapsed':time.perf_counter()-start,'last_kind':primary['kind'],'last_loss':primary['loss']})
                print(json.dumps({'phase':phase,'step':step+1,'updates':updates,'kind':primary['kind'],'loss':primary['loss'],'elapsed':round(time.perf_counter()-start,2)}),flush=True)
        metrics={'old_skill':score_old(params),'lm_nll':score_lm(params,dev_ids),'qa':score_qa(params,dev,enc)}
        save(output/f'{phase}.npz',params,opt,enc,updates);write(output/f'{phase}_dev.json',metrics)
        print(phase,json.dumps(metrics,ensure_ascii=False),flush=True)
    write(output/'events.json',events)
    write(output/'FROZEN.json',{'checkpoint_sha256':hashlib.sha256((output/'knowledge.npz').read_bytes()).hexdigest(),
                              'parameters':sum(int(a.size) for a in jax.tree_util.tree_leaves(params)),'updates':updates,
                              'train_seconds':time.perf_counter()-start,'protocol':protocol})


def examine(root,candidates,output):
    root=Path(root);output=Path(output);output.mkdir(exist_ok=False)
    enc=load_encoder(root);ids=np.load(root/'data/test.npy')
    for c in candidates:
        if not (Path(c)/'FROZEN.json').exists():raise ValueError('All models must be frozen before exam')
    records=qa_records(512,'test',98001,filler_ids=ids,enc=enc)
    paraphrases=qa_records(512,'test',98002,filler_ids=ids,enc=enc,templates_heldout=True)
    rows=[]
    for candidate in candidates:
        candidate=Path(candidate);p,_,_,_,_=core.load_stream_checkpoint(str(candidate/'knowledge.npz'))
        result={'candidate':str(candidate),'parameters':sum(int(a.size) for a in jax.tree_util.tree_leaves(p)),
                'qa':score_qa(p,records,enc),'qa_memory_reset':score_qa(p,records,enc,reset=True),
                'unseen_word_order':score_qa(p,paraphrases,enc),'old_skill':score_old(p,seed=98003,count=512),
                'russian_nll':score_lm(p,ids)}
        rows.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
    write(output/'results.json',{'models':rows,'protocol':'russian-memory-v1','test_pair_rules':'8..9 only; train 0..5',
                                 'limits':'Finite fictional codeword relation and question families; natural prose corpus is one public-domain novel; not universal Russian question answering'})


@jax.jit
def neural_portion(params,state,ids):
    return core.forward(params,ids,state)


@jax.jit
def read_portion(params,state,ids):
    return core.forward(params,ids,state,memory_write_mask=jnp.zeros((len(ids),),dtype=jnp.float32))


@jax.jit
def ingest_portion(params,state,ids):
    # Fact ingestion needs the carry, not predictions for every prefix token.
    return core.forward(params,ids,state)[1]


def text_portions(ids):
    # Two compiled shapes keep interactive JAX caches bounded; every token still advances the state.
    end=len(ids)//32*32
    for at in range(0,end,32):yield ids[at:at+32]
    for token in ids[end:]:yield [token]


def ingest_text(params,enc,state,text):
    ids=list(enc.encode(text))
    for portion in text_portions(ids):
        state=ingest_portion(params,state,jnp.asarray([portion],dtype=jnp.int32))
    return state


def consume_text(params,enc,state,text,*,read_only=False):
    ids=list(enc.encode(text))
    if not ids:raise ValueError('Empty input')
    for portion in text_portions(ids):
        logits,state=(read_portion if read_only else neural_portion)(params,state,jnp.asarray([portion],dtype=jnp.int32))
    return logits,state


def neural_reply(params,enc,state,text,*,max_tokens=20):
    # The explicit question field starts its own local language window; durable keys/values stay intact.
    if 'open_embed' in params:
        state=state._replace(open_context=jnp.zeros_like(state.open_context))
    logits,state=consume_text(params,enc,state,text,read_only=True)
    answer=[];dot=enc.mapping['.']
    for _ in range(max_tokens):
        token=int(jnp.argmax(logits[0,-1]));answer.append(token)
        logits,state=read_portion(params,state,jnp.asarray([[token]],dtype=jnp.int32))
        if token==dot:break
    return enc.decode(answer),state


def interact(checkpoint,session=None):
    import fcntl
    checkpoint=Path(checkpoint)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    lock=None
    if session:
        session=Path(session)
        if session.resolve()==checkpoint.resolve():raise ValueError('Session must not overwrite trained weights')
        session.parent.mkdir(parents=True,exist_ok=True)
        lock=session.with_suffix('.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if session.exists():
            metadata=json.loads(session.with_suffix('.json').read_text())
            if metadata['base_checkpoint_sha256']!=digest:raise ValueError('Session belongs to a different model version; original session preserved')
    params,opt,vocab,config,state=core.load_stream_checkpoint(str(session if session and session.exists() else checkpoint))
    enc=Encoder(vocab,config['tokenizer_config'])
    if not (session and session.exists()):
        _,state=neural_portion(params,state,jnp.asarray([[enc.mapping[BOS]]],dtype=jnp.int32))
    def persist():
        if session:
            save(session,params,opt,enc,int(opt['step']),state=state)
            temporary=session.with_suffix('.json.tmp')
            write(temporary,{'base_checkpoint_sha256':digest,'memory':'Complete neural carry; no input text or answer database'})
            temporary.replace(session.with_suffix('.json'))
    print('000cor — экспериментальная русская сеть с памятью. Пустые сведения и вопрос — выход.')
    if 'sem_relation_embed' in params:
        print('Пример сведений: «У Алисы любимый напиток — облепиховый чай.»; вопрос: «Алиса: назови предпочитаемое питьё.»')
        print('000cor v6: падежи, новые вопросы и русские векторы Navec. Покрытие слов ограничено; память — 512 записей.')
    elif 'open_embed' in params:
        print('Пример сведений: «У Алисы любимый напиток — облепиховый чай.»; вопрос: «Назови: любимый напиток у Алисы.»')
        print('Для нового свойства повторяйте его название в вопросе. Память содержит 512 записей; новые имена не требуют обучения.')
    elif 'fact_embed' in params:
        print('Пример: «Анна живёт в Казани. Анна работает врачом.»; вопрос: «Где живёт Анна?»')
    else:
        print('Пример сведений: «Кодовое слово Анны — янтарь.»; вопрос: «Какое кодовое слово у Анны?»')
    try:
        while True:
            facts=input('Новые сведения (Enter, если их нет): ').strip()
            if facts:state=ingest_text(params,enc,state,facts)
            question=input('Вопрос: ').strip()
            if not question:
                persist()
                if not facts:break
                print('Текст обработан, состояние памяти обновлено.');continue
            # Explicit question channel reads a fork; only new facts commit durable state.
            answer,_=neural_reply(params,enc,state,question+' Ответ:')
            persist();print('Сеть:',answer,flush=True)
    except (EOFError,KeyboardInterrupt):
        persist()
    finally:
        if lock:lock.close()


def migrate_session(checkpoint, source, destination):
    """Carry old neural state into an extension whose parent weights are identical."""
    checkpoint,source,destination=map(Path,(checkpoint,source,destination))
    if destination.exists() or destination.with_suffix('.json').exists():
        raise ValueError('Migration destination already exists; original state preserved')
    params,_,vocab,config,_=core.load_stream_checkpoint(str(checkpoint))
    parent,_,old_vocab,old_config,previous=core.load_stream_checkpoint(str(source))
    if old_vocab!=vocab or old_config.get('tokenizer_config')!=config.get('tokenizer_config'):
        raise ValueError('Migration requires the same tokenizer')
    if not all(k in params and np.array_equal(v,params[k]) for k,v in parent.items()):
        raise ValueError('Migration requires unchanged parent weights')
    fresh=core.init_stream_state(params,previous.h.shape[0]);fields={}
    for name in core.SSMState._fields:
        prior=getattr(previous,name);target=getattr(fresh,name)
        if prior is not None:
            if target is None or prior.shape!=target.shape or prior.dtype!=target.dtype:
                raise ValueError('Incompatible neural state field: '+name)
            fields[name]=prior
    if 'open_embed' in params and 'open_embed' not in parent:
        # An upgrade starts a fresh sentence; old per-sentence routing traces are not stored facts.
        fields.pop('fact_entity_trace',None);fields.pop('fact_relation_trace',None)
    state=fresh._replace(**fields);enc=Encoder(vocab,config['tokenizer_config'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    save(destination,params,core.init_adam(params),enc,0,state=state)
    write(destination.with_suffix('.json'),{'base_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'memory':'Migrated complete neural carry; no input text or answer database','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    print('Состояние памяти перенесено:',destination)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare-lexical');a.add_argument('--root',required=True)
    a=sub.add_parser('examine-lexical');a.add_argument('--root',required=True);a.add_argument('--output',required=True);a.add_argument('candidates',nargs='+')
    a=sub.add_parser('prepare');a.add_argument('--root',required=True)
    a=sub.add_parser('train');a.add_argument('--root',required=True);a.add_argument('--output',required=True)
    a.add_argument('--seed',type=int,default=11);a.add_argument('--input-dim',type=int,default=64);a.add_argument('--state-dim',type=int,default=128)
    a.add_argument('--lm-steps',type=int,default=512);a.add_argument('--qa-steps',type=int,default=1200);a.add_argument('--no-replay',action='store_true');a.add_argument('--resume');a.add_argument('--lexical',action='store_true');a.add_argument('--context-roles',action='store_true')
    a=sub.add_parser('interact');a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--session',type=Path)
    a=sub.add_parser('migrate-session');a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--source',type=Path,required=True);a.add_argument('--destination',type=Path,required=True)
    a=sub.add_parser('answer');a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--text',required=True)
    a=sub.add_parser('examine');a.add_argument('--root',required=True);a.add_argument('--output',required=True);a.add_argument('candidates',nargs='+')
    args=p.parse_args()
    if args.command=='prepare-lexical':prepare_lexical(args.root)
    elif args.command=='examine-lexical':examine_lexical(args.root,args.candidates,args.output)
    elif args.command=='prepare':prepare_data(args.root)
    elif args.command=='train':train(args.root,args.output,seed=args.seed,input_dim=args.input_dim,state_dim=args.state_dim,lm_steps=args.lm_steps,qa_steps=args.qa_steps,replay=not args.no_replay,resume=args.resume,lexical=args.lexical,context_roles=args.context_roles)
    elif args.command=='interact':interact(args.checkpoint,args.session)
    elif args.command=='migrate-session':migrate_session(args.checkpoint,args.source,args.destination)
    elif args.command=='answer':
        params,_,vocab,config,state=core.load_stream_checkpoint(str(args.checkpoint))
        enc=Encoder(vocab,config['tokenizer_config'])
        answer,_=neural_reply(params,enc,state,args.text)
        print(answer)
    else:examine(args.root,args.candidates,args.output)


if __name__=='__main__':main()
