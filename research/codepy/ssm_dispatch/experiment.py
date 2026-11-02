"""Registered mixed-stream learned KV dispatcher for frozen CodePy chains."""
import os
os.environ['JAX_PLATFORM_NAME']='cpu'; os.environ['JAX_PLATFORMS']='cpu'
import sys, json, time, hashlib, itertools, argparse, importlib.util
from pathlib import Path
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(HERE.parent/'adaptive_calls'))
spec=importlib.util.spec_from_file_location('adaptive_trial',HERE.parent/'adaptive_calls/run_experiment.py')
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
from train_ssm_text import init_params, init_stream_state, forward, init_adam, adam_update, clip_grads
from compare_long_context_recall import init_assignment_kv_memory_params
TASKS=('sort4','reverse4','rotate4','descending4'); PHASES=('familiar','delay1024','updates','new_aliases','composition')
PERMS=np.asarray(list(itertools.permutations(range(4))),np.int32)
KEY0=4; VALUE0=KEY0+12; VALUES=96; VOCAB=VALUE0+VALUES; CHUNK=128

def save(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tree_bytes(tree):return sum(np.asarray(v).nbytes for v in jax.tree_util.tree_leaves(tree))
def wanted(x,task,perm):
    y=x[:,PERMS[perm]]
    if task==0:y=np.sort(y,axis=1)
    elif task==1:y=y[:,::-1]
    elif task==2:y=np.roll(y,1,axis=1)
    else:y=np.sort(y,axis=1)[:,::-1]
    return y[:,np.argsort(PERMS[perm])].copy()

def prepare_legacy(seed):
    old=json.loads((ROOT/'artifacts/codepy_adaptive_calls_20261030/results.json').read_text())
    priority=('arguments','combined','gates','fixed'); selected=[]
    for task in TASKS:
        rows=[r for r in old['runs'] if r['seed']==seed and r['task']==task]
        selected.append(min(rows,key=lambda r:(r['training_objective'],priority.index(r['arm']))))
    lib=json.loads((ROOT/f'artifacts/codepy_subtask_discovery_20261030/library_seed{seed}.json').read_text())
    entries=[[v] for v in a.s.ALPHABET]+lib['blocks']; table,lengths=a.s.table_for(lib['blocks'])
    tables=a.calls.c.contract_tables(entries); genomes=jnp.asarray([r['program'] for r in selected])
    ids=np.repeat(np.arange(4),24); ps=np.tile(np.arange(24),4)
    @jax.jit
    def candidates(x):
        def one(g,p):
            out=a.calls.execute(g[None],x[:,p],table,lengths,tables,n=4,primitive_count=len(a.s.ALPHABET))[0]
            return out[:,jnp.argsort(p)]
        return jax.vmap(one)(genomes[ids],jnp.asarray(PERMS[ps]))
    x=np.random.default_rng(50000+seed).integers(-2,3,(64,4)).astype(np.float32)
    t=time.perf_counter();jax.block_until_ready(candidates(jnp.asarray(x)));compile_s=time.perf_counter()-t
    @jax.jit
    def resolve_kernel(x,y):
        out=candidates(x)
        score=jnp.mean((out-y[None])**2,axis=(1,2))+jnp.mean((jnp.sort(out,axis=-1)-jnp.sort(x,axis=-1)[None])**2,axis=(1,2))
        return jnp.argmin(score),jnp.min(score)
    t=time.perf_counter();jax.block_until_ready(resolve_kernel(jnp.asarray(x),jnp.asarray(x)));resolver_compile=time.perf_counter()-t
    @jax.jit
    def execute_route(command,x):
        p=jnp.asarray(PERMS)[command%24]
        out=a.calls.execute(genomes[command//24][None],x[:,p],table,lengths,tables,n=4,primitive_count=len(a.s.ALPHABET))[0]
        return out[:,jnp.argsort(p)]
    return selected,entries,candidates,resolve_kernel,execute_route,x,compile_s+resolver_compile

def prepare(seed):
    """Default dispatcher uses the accepted 20/20 library; old runs remain explicit."""
    from research.codepy.ssm_dispatch.catalog import catalog
    current = catalog(seed)
    x = np.random.default_rng(50000+seed).integers(-2,3,(64,4)).astype(np.float32)
    def candidates(values):
        return jnp.asarray(current.candidates(np.asarray(values)))
    def resolve(values, expected):
        out = current.candidates(np.asarray(values))
        score = np.mean((out-np.asarray(expected)[None])**2,axis=(1,2))
        at = int(np.argmin(score))
        return jnp.asarray(at), jnp.asarray(score[at])
    def execute(command, values):
        return jnp.asarray(current.execute(int(command),np.asarray(values)))
    selected = [{**r['branches'][r['chosen_by_TRAIN']], 'task':r['task'], 'arm':r['chosen_by_TRAIN']} for r in current.selected]
    return selected,current.entries,candidates,resolve,execute,x,0.

def training_batch(rng,batch=32):
    x=np.zeros((batch,64),np.int32);labels=np.full_like(x,-1)
    for row in range(batch):
        mapping={}; cursor=0
        aliases=rng.choice(8,4,replace=False)
        for alias in aliases:
            v=int(rng.integers(VALUES));mapping[int(alias)]=v
            x[row,cursor:cursor+4]=[KEY0+alias,2,VALUE0+v,3];cursor+=4
        for q in range(4):
            if q in (1,3):
                alias=int(rng.choice(aliases));v=int(rng.integers(VALUES));mapping[alias]=v
                x[row,cursor:cursor+4]=[KEY0+alias,2,VALUE0+v,3];cursor+=4
            cursor+=int(rng.integers(2,7)); alias=int(rng.choice(aliases))
            x[row,cursor:cursor+2]=[KEY0+alias,1];labels[row,cursor+1]=mapping[alias];cursor+=2
    return x,labels

@jax.jit
def train_step(params,opt,x,labels):
    def objective(p):
        logits,_=forward(p,x,init_stream_state(p,len(x)))
        logp=jax.nn.log_softmax(logits[:,:,VALUE0:VALUE0+VALUES],axis=-1)
        mask=labels>=0; index=jnp.maximum(labels,0)
        ce=-jnp.take_along_axis(logp,index[:,:,None],axis=-1)[:,:,0]
        return jnp.sum(ce*mask)/jnp.sum(mask)
    loss,grads=jax.value_and_grad(objective)(params);grads,_=clip_grads(grads,1.)
    params,opt=adam_update(params,grads,opt,.01)
    return params,opt,loss

def train(seed,out):
    params=init_params(jax.random.PRNGKey(seed),VOCAB,32,32,variant='kv-memory')
    params=init_assignment_kv_memory_params(params,key_count=12,value_count=96,key_logit=4.,value_logit=4.,erase_logit=4.,query_logit=4.,closed_logit=-4.,kv_decay_raw=8.,kv_logit_scale=10.)
    opt=init_adam(params);rng=np.random.default_rng(60000+seed)
    xb,yb=training_batch(np.random.default_rng(777))
    t=time.perf_counter();jax.block_until_ready(train_step(params,opt,jnp.asarray(xb),jnp.asarray(yb)));compile_s=time.perf_counter()-t
    t=time.perf_counter();losses=[]
    for step in range(800):
        xb,yb=training_batch(rng);params,opt,loss=train_step(params,opt,jnp.asarray(xb),jnp.asarray(yb))
        if step%100==0 or step==799:
            losses.append([step,float(loss)]);print('train',seed,step,round(float(loss),4),flush=True)
    jax.block_until_ready(params);elapsed=time.perf_counter()-t
    np.savez_compressed(out/f'weights_seed{seed}.npz',**{k:np.asarray(v) for k,v in params.items()})
    flat={f'{rail}__{k}':np.asarray(v) for rail in ('m','v') for k,v in opt[rail].items()};flat['step']=np.asarray(opt['step'])
    np.savez_compressed(out/f'optimizer_seed{seed}.npz',**flat)
    return params,{'updates':800,'examples':25600,'query_labels':102400,'elapsed_s':elapsed,'compile_s':compile_s,'trace':losses,'parameter_bytes':tree_bytes(params),'carry_bytes':tree_bytes(init_stream_state(params,1))}

@jax.jit
def infer_chunk(params,x,state):return forward(params,x,state)

def infer(params,tokens,ablate=False):
    if ablate:params={**params,'kv_logit_scale':jnp.asarray(0.,jnp.float32)}
    state=init_stream_state(params,1);data=np.pad(np.asarray(tokens,np.int32),(0,(-len(tokens))%CHUNK)).reshape(-1,CHUNK)
    t=time.perf_counter();jax.block_until_ready(infer_chunk(params,jnp.asarray(data[:1]),state));compile_s=time.perf_counter()-t
    t=time.perf_counter();all_logits=[]
    for chunk in data:
        logits,state=infer_chunk(params,jnp.asarray(chunk[None]),state)
        all_logits.append(np.asarray(logits)[0,:,VALUE0:VALUE0+VALUES])
    logits=np.concatenate(all_logits)[:len(tokens)]
    return logits.argmax(axis=-1),{'elapsed_s':time.perf_counter()-t,'compile_and_warmup_s':compile_s,'tokens':len(tokens),'processed_tokens':data.size}

def build_stream(seed,resolver,x):
    rng=np.random.default_rng(70000+seed);tokens=[];events=[];definitions={};registry={};definition_time=0.
    def define(alias):
        nonlocal definition_time
        task=int(rng.integers(4));perm=int(rng.integers(24));y=wanted(x,task,perm)
        t=time.perf_counter();route,loss=resolver(jnp.asarray(x),jnp.asarray(y));route=int(route);loss=float(loss);definition_time+=time.perf_counter()-t
        ident=len(definitions);definitions[ident]={'alias':alias,'task':task,'perm':perm,'route':route,'loss':loss}
        registry[alias]=ident;tokens.extend([KEY0+alias,2,VALUE0+route,3]);events.append({'kind':'define','definition':ident,'end':len(tokens)-1})
    for alias in range(8):define(alias)
    for phase in PHASES:
        if phase=='delay1024':tokens.extend([0]*1024)
        if phase=='new_aliases':
            for alias in range(8,12):define(alias)
        for qi in range(64):
            if phase=='updates' and qi%4==0:define(int(rng.integers(8)))
            aliases=rng.integers(8,12,size=1) if phase=='new_aliases' else rng.integers(0,8,size=2 if phase=='composition' else 1)
            request=[]
            for alias in aliases:
                d=registry[int(alias)];tokens.extend([KEY0+int(alias),1]);request.append({'alias':int(alias),'definition':d,'position':len(tokens)-1})
                tokens.extend([0]*int(rng.integers(1,5)))
            events.append({'kind':'query','phase':phase,'calls':request,'sample_seed':80000+seed*1000+len(events)})
    return tokens,events,definitions,definition_time

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__).resolve(),HERE/'README.md',ROOT/'train_ssm_text.py',ROOT/'compare_long_context_recall.py',HERE.parent/'adaptive_calls/calls.py',ROOT/'artifacts/codepy_adaptive_calls_20261030/results.json']
    protocol={'version':'codepy-ssm-dispatch-v1','seeds':list(range(5)),'tasks':TASKS,'phases':PHASES,'commands':96,'queries_per_phase':64,'training_updates':800,'training_batch':32,'source_hashes':{str(p):sha(p) for p in paths},'library_hashes':{str(ROOT/f'artifacts/codepy_subtask_discovery_20261030/library_seed{i}.json'):sha(ROOT/f'artifacts/codepy_subtask_discovery_20261030/library_seed{i}.json') for i in range(5)}}
    protocol=json.loads(json.dumps(protocol))
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:save(out/'protocol.json',protocol)
    runs=[]
    for seed in range(5):
        path=out/f'seed{seed}.json'
        if path.exists():runs.append(json.loads(path.read_text()));continue
        selected,entries,candidates,resolver,execute_route,dx,compile_s=prepare_legacy(seed)
        params,training=train(seed,out)
        tokens,events,definitions,definition_s=build_stream(seed,resolver,dx)
        pred,inference=infer(params,tokens);no_kv,ablation_time=infer(params,tokens,True)
        save(out/f'stream_seed{seed}.json',{'tokens':tokens,'events':events,'definitions':definitions,'ssm_predictions':pred.tolist(),'ablated_predictions':no_kv.tolist()})
        # Strong exact-key dictionary; no query data used to select routes.
        dictionary={};dict_pred=np.zeros(len(tokens),np.int32);search_pred=np.zeros(len(tokens),np.int32);search_s=0.;search_count=0;dict_s=0.
        for e in events:
            if e['kind']=='define':
                d=definitions[e['definition']];t=time.perf_counter();dictionary[d['alias']]=d['route'];dict_s+=time.perf_counter()-t
            else:
                for call in e['calls']:
                    t=time.perf_counter();dict_pred[call['position']]=dictionary[call['alias']];dict_s+=time.perf_counter()-t
                    d=definitions[call['definition']];y=wanted(dx,d['task'],d['perm'])
                    t=time.perf_counter();route,_=resolver(jnp.asarray(dx),jnp.asarray(y));search_pred[call['position']]=int(route);search_s+=time.perf_counter()-t;search_count+=1
        assert all(dict_pred[c['position']]==search_pred[c['position']] for e in events if e['kind']=='query' for c in e['calls'])
        groups=a.s.old.cases(4,16000);ex=np.concatenate(list(groups.values()))
        t=time.perf_counter();eval_outputs=np.asarray(candidates(jnp.asarray(ex)));evaluation_s=time.perf_counter()-t
        # Independent interpreter verifies every frozen chain and binding on64 unseen numeric cases.
        for task in range(4):
            for pi,p in enumerate(PERMS):
                ref=a.calls.numpy_execute(selected[task]['program'],ex[:64,p],entries,len(a.s.ALPHABET))[:,np.argsort(p)]
                assert np.array_equal(ref,eval_outputs[task*24+pi,:64])
        t=time.perf_counter();jax.block_until_ready(execute_route(jnp.asarray(0),jnp.asarray(ex[:16])));execution_compile_s=time.perf_counter()-t
        arms={'ssm':pred,'dictionary':dict_pred,'search':search_pred,'no_kv':no_kv};metrics={}
        t=time.perf_counter()
        for arm,predictions in arms.items():
            rows=[];execution_s=0.
            for e in events:
                if e['kind']!='query':continue
                idx=np.random.default_rng(e['sample_seed']).choice(len(ex),16,replace=False);actual=ex[idx].copy();ideal=actual.copy();oracle=actual.copy();routes=[];truth=[]
                for ci,call in enumerate(e['calls']):
                    d=definitions[call['definition']];route=int(predictions[call['position']]);routes.append(route);truth.append(d['route'])
                    ideal=wanted(ideal,d['task'],d['perm'])
                    t_exec=time.perf_counter()
                    actual=np.asarray(execute_route(jnp.asarray(route),jnp.asarray(actual)))
                    execution_s+=time.perf_counter()-t_exec
                    oracle=np.asarray(execute_route(jnp.asarray(d['route']),jnp.asarray(oracle)))
                exact=np.all(actual==ideal,axis=1);close=np.all(np.isclose(actual,ideal,rtol=1e-6,atol=1e-6),axis=1)
                rows.append({'phase':e['phase'],'routes':routes,'expected_routes':truth,'route_correct':routes==truth,'calls_correct':sum(x==y for x,y in zip(routes,truth)),'calls':len(routes),'cases':16,'exact_cases':int(exact.sum()),'close_cases':int(close.sum()),'request_exact':bool(exact.all()),'oracle_exact_cases':int(np.all(oracle==ideal,axis=1).sum())})
            metrics[arm]={phase:{'requests':sum(r['phase']==phase for r in rows),'route_accuracy':np.mean([r['route_correct'] for r in rows if r['phase']==phase]),'call_accuracy':sum(r['calls_correct'] for r in rows if r['phase']==phase)/sum(r['calls'] for r in rows if r['phase']==phase),'exact_output_accuracy':sum(r['exact_cases'] for r in rows if r['phase']==phase)/1024,'tolerance_accuracy':sum(r['close_cases'] for r in rows if r['phase']==phase)/1024,'strict_requests':sum(r['request_exact'] for r in rows if r['phase']==phase),'oracle_output_accuracy':sum(r['oracle_exact_cases'] for r in rows if r['phase']==phase)/1024} for phase in PHASES}
            metrics[arm]['execution_s']=execution_s
            save(out/f'{arm}_requests_seed{seed}.json',rows)
        scoring_s=time.perf_counter()-t
        result={'seed':seed,'selected_chains':[{'task':r['task'],'arm':r['arm'],'training_objective':r['training_objective'],'prior_strict_certificate':r['certificate']['passed'],'program':r['program']} for r in selected],'training':training,'inference':inference,'ablation_inference':ablation_time,'metrics':metrics,'costs':{'resolver_compile_s':compile_s,'execution_compile_s':execution_compile_s,'definition_search_s':definition_s,'definition_searches':len(definitions),'definition_candidates':96*len(definitions),'search_query_s':search_s,'search_query_searches':search_count,'search_query_candidates':96*search_count,'dictionary_lookup_update_s':dict_s,'dictionary_entries':len(dictionary),'dictionary_payload_bytes':len(dictionary)*8,'dictionary_python_bytes':sys.getsizeof(dictionary)+sum(sys.getsizeof(k)+sys.getsizeof(v) for k,v in dictionary.items()),'evaluation_only_cache_and_compile_s':evaluation_s,'evaluation_only_scoring_s':scoring_s},'independent_checks':96*64,'heldout_pool_cases':len(ex)}
        save(path,result);runs.append(result);print('finished',seed,json.dumps(metrics['ssm']),flush=True)
    save(out/'results.json',{'protocol':protocol,'runs':runs})
    print('ALL COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/codepy_ssm_dispatch_20261031');main(p.parse_args().output_dir)
