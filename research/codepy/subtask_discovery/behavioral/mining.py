"""Behavioral pair discovery with an explicit old-ablation case-step budget.

Desired behavior families are a human-supplied criterion. No bytecode body is
supplied for swapping or ordering; source slices and bounded mutations supply it.
"""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_PLATFORM_NAME']='cpu'
import time,json,hashlib,itertools
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from abi import vm,pair_outputs,MAX_BODY,LAYOUTS
from slicing import candidate_pool
ALPHABET=[i for i in range(36) if i not in (18,19,29,30,31,35)]
PROBE=np.asarray(list(itertools.product(range(-2,3),repeat=2)),np.float32)

class Budget:
    def __init__(self,cap):self.cap=int(cap);self.used=0;self.charges=[]
    def spend(self,n,reason):
        n=int(n)
        if self.used+n>self.cap:return False
        self.used+=n;self.charges.append({'kind':reason,'units':n});return True

def classify(out,x):
    if not np.array_equal(np.sort(out,axis=-1),np.sort(x,axis=-1)):return None
    if np.array_equal(out,x):return None
    if np.array_equal(out,x[:,::-1]):return 'swap'
    if np.all(out[:,0]<=out[:,1]):return 'order_ascending'
    if np.all(out[:,0]>=out[:,1]):return 'order_descending'
    return None

def family(kind):return 'order' if kind and kind.startswith('order_') else kind

def encode(rows):
    code=np.zeros((len(rows),MAX_BODY),np.int32)
    for i,r in enumerate(rows):code[i,:len(r['code'])]=r['code']
    return jnp.asarray(code),jnp.asarray([r['layout'] for r in rows],jnp.int32)

def scores(outputs,x):
    preserve=jnp.mean((jnp.sort(outputs,axis=-1)-jnp.sort(x,axis=-1)[None])**2,axis=(1,2))
    swap=jnp.mean((outputs-x[None,:,::-1])**2,axis=(1,2))+preserve
    diff=outputs[:,:,0]-outputs[:,:,1]
    order=jnp.minimum(jnp.mean(jnp.maximum(diff,0)**2,1),jnp.mean(jnp.maximum(-diff,0)**2,1))+preserve
    return jnp.stack([swap,order],-1)

@jax.jit
def evaluate(codes,layouts):
    out=pair_outputs(codes,layouts,jnp.asarray(PROBE));return scores(out,jnp.asarray(PROBE)),out

def qualify(row,kind,seed,budget):
    rng=np.random.default_rng(24601+seed);challenge=np.concatenate([rng.uniform(-100000,100000,(32,2)),rng.uniform(-1,1,(32,2))]).astype(np.float32)
    datasets=[('challenge64',challenge),('promotion6429',np.concatenate(list(vm.cases(2,27000+seed).values())))]
    proofs=[]
    code,layout=encode([row])
    for name,x in datasets:
        if not budget.spend(len(x)*MAX_BODY*2,name):return None
        native=np.asarray(pair_outputs(code,layout,jnp.asarray(x)))[0]
        local=np.zeros((len(x),3),np.float32);local[:,row['layout']]=x;independent=vm.numpy_execute(row['code'],local)[:,row['layout']]
        np.testing.assert_array_equal(native,independent)
        found=classify(native,x)
        if found!=kind:return None
        proofs.append({'group':name,'cases':len(x),'native_numpy_exact':True,'behavior':found,'inputs_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
    return proofs

def mine(library,seed):
    start=time.perf_counter();cap=library['mining']['ablation_evaluations']*64*36;budget=Budget(cap);pool=candidate_pool(library['sources']);found={};probes=[];attempted=set();best_source=[]
    diagnostic_rows=[{'code':b,'layout':list(layout)} for b in library['blocks'] for layout in LAYOUTS];assert budget.spend(len(diagnostic_rows)*len(PROBE)*MAX_BODY,'old_selected_diagnosis');dc,dl=encode(diagnostic_rows);dv,do=evaluate(dc,dl);old_behavior=[classify(o,PROBE) for o in np.asarray(do)]
    limit=min(len(pool),int(cap*.25)//(len(PROBE)*MAX_BODY));repair_log=[]
    def promote(row,out):
        kind=classify(out,PROBE);which=family(kind)
        if not kind or which in found:return
        key=(tuple(row['code']),tuple(row['layout']))
        if key in attempted:return
        attempted.add(key);proof=qualify(row,kind,seed,budget)
        if proof:found[which]={**row,'behavior':kind,'qualification':proof}
    for start_index in range(0,limit,128):
        rows=pool[start_index:min(limit,start_index+128)]
        if not budget.spend(len(rows)*len(PROBE)*MAX_BODY,'source_probe'):break
        code,layout=encode(rows);values,outputs=evaluate(code,layout);values=np.asarray(values);outputs=np.asarray(outputs)
        for row,value,out in zip(rows,values,outputs):
            probes.append({'code_sha256':hashlib.sha256(np.asarray(row['code'],np.int32).tobytes()).hexdigest(),'layout':row['layout'],'scores':value.tolist()});best_source.append((row,value.copy()));promote(row,out)
        if len(found)==2:break
    # Recombine/mutate source-initialized programs only for missing behaviors.
    # Every evaluation, including qualification and padding, is charged to cap.
    states={};population=256;lookup={v:i for i,v in enumerate(ALPHABET)}
    for fi,name in enumerate(['swap','order']):
        if name in found:continue
        key,init=jax.random.split(jax.random.PRNGKey(31000+seed*2+fi));genomes=np.asarray(vm.ect.init_programs(init,population,MAX_BODY,len(ALPHABET))).copy();layouts=np.asarray(jax.random.randint(jax.random.fold_in(init,4),(population,),0,3)).copy()
        ordered=sorted(best_source,key=lambda z:(float(z[1][fi]),len(z[0]['code']),z[0]['code']))
        for i,(row,value) in enumerate(ordered[:population]):
            genomes[i]=0;genomes[i,:len(row['code'])]=[lookup[v] for v in row['code']];layouts[i]=list(map(tuple,LAYOUTS)).index(tuple(row['layout']))
        states[name]=[key,jnp.asarray(genomes),jnp.asarray(layouts),0]
    mutate=jax.jit(lambda key,g:vm.ect.mutate_elites(key,g,population=population,mutation_rate=.16,random_reset_count=12,vocab_size=len(ALPHABET)))
    while states and budget.cap-budget.used>=population*len(PROBE)*MAX_BODY:
        for fi,name in enumerate(['swap','order']):
            if name not in states or name in found:continue
            if not budget.spend(population*len(PROBE)*MAX_BODY,'repair_'+name):break
            key,genomes,layout_ids,generation=states[name];codes=jnp.asarray(ALPHABET)[genomes];layouts=jnp.asarray(LAYOUTS)[layout_ids];value,out=evaluate(codes,layouts);value=np.asarray(value);out=np.asarray(out);index=0 if name=='swap' else 1;ids=np.argsort(value[:,index],kind='stable')[:16].copy()
            repair_log.append({'family':name,'generation':generation,'best_probe_objective':float(value[ids[0],index]),'evaluations':population})
            for idx in np.flatnonzero(value[:,index]==0):
                code=np.asarray(codes[idx]).tolist()
                while code and code[-1]==0:code.pop()
                row={'code':code or [0],'layout':np.asarray(layouts[idx]).tolist(),'provenance':{'method':'source_initialized_evolution','family':name,'generation':generation,'candidate_index':int(idx)},'frequency':0}
                promote(row,out[idx])
                if name in found:break
            if name in found:del states[name];continue
            key,k1,k2,k3=jax.random.split(key,4);new=mutate(k1,genomes[jnp.asarray(ids)]);parents=jax.random.randint(k2,(population,),0,len(ids));new_layout=layout_ids[jnp.asarray(ids)][parents];change=jax.random.bernoulli(k3,.16,(population,));random_layout=jax.random.randint(jax.random.fold_in(k3,8),(population,),0,3);new_layout=jnp.where(change,random_layout,new_layout).at[:16].set(layout_ids[jnp.asarray(ids)])
            states[name]=[key,new,new_layout,generation+1]
        states={k:v for k,v in states.items() if k not in found}
    return {'blocks':[found[k] for k in ['swap','order'] if k in found],'old_selected_pair_probe_behaviors':old_behavior,'source_pool_count':len(pool),'source_probe_count':len(probes),'source_probe_limit':limit,'probe_records':probes,'repair_log':repair_log,'promotion_attempts':len(attempted),'preparation_case_step_budget':cap,'preparation_case_steps_used':budget.used,'budget_charges':budget.charges,'preparation_s_including_jit':time.perf_counter()-start,'missing_families':[k for k in ['swap','order'] if k not in found]}
