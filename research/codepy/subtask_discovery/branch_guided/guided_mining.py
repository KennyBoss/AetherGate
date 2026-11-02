"""Order-only local repair under the old case-step cap, using branch-stratified progress."""
import time,hashlib
from common import legacy
from register_abi import tables,pair_outputs,numpy_pair,LAYOUTS
from branch_slicing import candidate_pool
from progress import metrics,behavioral_elites,mutate
import jax,jax.numpy as jnp,numpy as np

@jax.jit
def evaluate(codes,layouts):
    returns=jnp.tile(jnp.array([[3,4]],jnp.int32),(len(codes),1));out=pair_outputs(codes,layouts,returns,jnp.asarray(legacy.PROBE))
    return metrics(out),out

def encode(rows):
    codes=np.zeros((len(rows),12),np.int32)
    for i,row in enumerate(rows):codes[i,:len(row['code'])]=row['code']
    return jnp.asarray(codes),jnp.asarray([r['layout'] for r in rows],jnp.int32)

def qualify(row,kind,seed,budget):
    rng=np.random.default_rng(24601+seed);challenge=np.concatenate([rng.uniform(-100000,100000,(32,2)),rng.uniform(-1,1,(32,2))]).astype(np.float32);code,layout=encode([row]);returns=jnp.array([[3,4]],jnp.int32);proof=[]
    for name,x in [('challenge64',challenge),('promotion6429',np.concatenate(list(legacy.vm.cases(2,27000+seed).values())))]:
        if not budget.spend(2*12*len(x),name):return None
        native=np.asarray(pair_outputs(code,layout,returns,jnp.asarray(x)))[0];independent=numpy_pair(row['code'],row['layout'],[3,4],x);np.testing.assert_array_equal(native,independent)
        if legacy.classify(native,x)!=kind:return None
        proof.append({'group':name,'cases':len(x),'behavior':kind,'native_numpy_exact':True,'inputs_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
    return proof

def mine(source,incumbent,seed):
    if any(b['behavior'].startswith('order_') for b in incumbent['blocks']):
        return {'seed':seed,'status':'incumbent_order_retained','blocks':[],'preparation_case_steps_used':0,'preparation_case_step_budget':source['mining']['ablation_evaluations']*64*36,'trace':[],'source_probe_count':0}
    begin=time.perf_counter();budget=legacy.Budget(source['mining']['ablation_evaluations']*64*36);pool=candidate_pool(source['sources']);limit=min(len(pool),int(.25*budget.cap)//(25*12));values_all=[];outputs_all=[];rows_all=[];found=None;attempted=set();trace=[]
    def promote(row,output):
        nonlocal found
        kind=legacy.classify(output,legacy.PROBE)
        if not kind or not kind.startswith('order_'):return
        key=(tuple(row['code']),tuple(row['layout']))
        if key in attempted:return
        attempted.add(key);proof=qualify(row,kind,seed,budget)
        if proof:found={**row,'outputs':[3,4],'behavior':kind,'qualification':proof}
    for first in range(0,limit,128):
        rows=pool[first:min(limit,first+128)]
        if not budget.spend(len(rows)*25*12,'register_source_probe'):break
        code,layout=encode(rows);value,out=evaluate(code,layout);value=np.asarray(value);out=np.asarray(out);rows_all.extend(rows);values_all.append(value);outputs_all.append(out)
        for row,z in zip(rows,out):
            promote(row,z)
            if found:break
        if found:break
    population=256;key,init=jax.random.split(jax.random.PRNGKey(41000+seed));genomes=np.asarray(legacy.vm.ect.init_programs(init,population,12,len(legacy.ALPHABET))).copy();layout_ids=np.asarray(jax.random.randint(jax.random.fold_in(init,5),(population,),0,3)).copy();lookup={v:i for i,v in enumerate(legacy.ALPHABET)}
    if rows_all and not found:
        vals=np.concatenate(values_all);outs=np.concatenate(outputs_all);codes,layouts=encode(rows_all);ids,_,_=behavioral_elites(vals,outs,np.asarray(codes),np.asarray(layouts),count=min(128,len(rows_all)))
        for i,idx in enumerate(ids):
            row=rows_all[int(idx)];genomes[i]=0;genomes[i,:len(row['code'])]=[lookup[v] for v in row['code']];layout_ids[i]=list(map(tuple,LAYOUTS)).index(tuple(row['layout']))
    genomes=jnp.asarray(genomes);layout_ids=jnp.asarray(layout_ids);generation=0
    while not found and budget.spend(population*25*12,'order_repair'):
        codes=jnp.asarray(legacy.ALPHABET)[genomes];layouts=jnp.asarray(LAYOUTS)[layout_ids];value,out=evaluate(codes,layouts);value=np.asarray(value);out=np.asarray(out);ids,critical,diversity=behavioral_elites(value,out,np.asarray(codes),np.asarray(layouts));i=int(ids[np.argmin(np.minimum(value[ids,6],value[ids,7]))]);trace.append({'generation':generation,'best_score':float(min(value[i,6],value[i,7])),'best_progress':value[i].tolist(),'best_code':np.asarray(codes[i]).tolist(),'best_layout':np.asarray(layouts[i]).tolist(),'evaluations':population,**diversity})
        for idx in np.flatnonzero(np.minimum(value[:,6],value[:,7])==0):
            code=np.asarray(codes[idx]).tolist()
            while code and code[-1]==0:code.pop()
            row={'code':code or [0],'layout':np.asarray(layouts[idx]).tolist(),'outputs':[3,4],'provenance':{'method':'branch_guided_source_repair','generation':generation,'candidate_index':int(idx)}};promote(row,out[idx])
            if found:break
        if found:break
        key,mutation=jax.random.split(key);genomes,layout_ids=mutate(mutation,genomes[jnp.asarray(ids)],layout_ids[jnp.asarray(ids)],jnp.asarray(critical));generation+=1
    return {'seed':seed,'status':'qualified' if found else 'not_found','blocks':[found] if found else [],'source_pool_count':len(pool),'source_probe_count':len(rows_all),'source_probe_limit':limit,'trace':trace,'promotion_attempts':len(attempted),'preparation_case_step_budget':budget.cap,'preparation_case_steps_used':budget.used,'budget_charges':budget.charges,'preparation_s_including_jit':time.perf_counter()-begin}
