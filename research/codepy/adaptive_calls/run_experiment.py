#!/usr/bin/env python3
"""Frozen-library search: fixed vs arguments vs gates vs both, paired controls."""
import calls
import importlib.util
from pathlib import Path
import json
import hashlib
import argparse
import time
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('contract_trial',HERE.parent/'block_contracts/run_experiment.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
s=previous.s
ARMS=('fixed','arguments','gates','combined')


@partial(jax.jit,static_argnames=('actions','arm'))
def mutate(key,elites,actions,arm):
    population=256;k=jax.random.split(key,6)
    a=jax.random.randint(k[0],(population,),0,len(elites));b=jax.random.randint(k[1],(population,),0,len(elites))
    cuts=jax.random.randint(k[2],(population,1,1),1,72)
    children=jnp.where(jnp.arange(72)[None,:,None]<cuts,elites[a],elites[b])
    # Preserve exactly the previous mutation RNG for the action column.
    action_mask=jax.random.bernoulli(k[3],.16,(population,72))
    action_values=jax.random.randint(k[4],(population,72),0,actions)
    children=children.at[:,:,0].set(jnp.where(action_mask,action_values,children[:,:,0]))
    for field,bound in enumerate(calls.BOUNDS,1):
        enabled=(field<=5 and arm in ('arguments','combined')) or (field==6 and arm in ('gates','combined'))
        if enabled:
            fk=jax.random.fold_in(key,field+100)
            mk,vk=jax.random.split(fk)
            values=jax.random.randint(vk,(population,72),0,bound)
            mask=jax.random.bernoulli(mk,.16,(population,72))
            children=children.at[:,:,field].set(jnp.where(mask,values,children[:,:,field]))
        else:children=children.at[:,:,field].set(0)
    resets=jnp.zeros((12,72,7),jnp.int32)
    resets=resets.at[:,:,0].set(s.old.ect.init_programs(k[5],12,72,actions))
    children=children.at[-12:].set(resets)
    return children.at[:len(elites)].set(elites)


def search(seed,x,arm,table,lengths,tables,task):
    key,init=jax.random.split(jax.random.PRNGKey(seed));actions=len(table)
    genomes=jnp.zeros((256,72,7),jnp.int32).at[:,:,0].set(s.old.ect.init_programs(init,256,72,actions))
    # Every arm begins with the fixed interfaces; adaptive alleles arise by mutation.
    @jax.jit
    def scorer(g):return s.losses(calls.execute(g,jnp.asarray(x),table,lengths,tables,n=4,primitive_count=len(s.ALPHABET)),jnp.asarray(x),task)
    start=time.perf_counter();jax.block_until_ready(scorer(genomes));jax.block_until_ready(mutate(key,genomes[:16],actions=actions,arm=arm))
    compilation_s=time.perf_counter()-start
    start=time.perf_counter();best=float('inf');champion=None;trace=[];first=None
    for generation in range(101):
        objective,mse=scorer(genomes);values=np.asarray(objective);errors=np.asarray(mse)
        index=int(np.argmin(values))
        if float(values[index])<best:best=float(values[index]);champion=np.asarray(genomes[index]).tolist()
        if first is None and np.any(values==0):first={'generation':generation,'evaluations':256*(generation+1)}
        trace.append({'generation':generation,'best_objective':best,'min_mse':float(errors.min())})
        if generation<100:
            key,mk=jax.random.split(key);ids=np.argsort(values,kind='stable')[:16].copy()
            genomes=mutate(mk,genomes[jnp.asarray(ids)],actions=actions,arm=arm)
    return {'program':champion,'training_objective':best,'first_perfect':first,'trace':trace,
            'evaluations':25856,'elapsed_s':time.perf_counter()-start,'compilation_and_warmup_s':compilation_s,
            'training_sha256':hashlib.sha256(x.tobytes()).hexdigest()}


def certify(result,xgroups,entries,table,lengths,tables,task):
    rows={}
    for name,x in xgroups.items():
        native=np.asarray(calls.execute(jnp.asarray([result['program']]),jnp.asarray(x),table,lengths,tables,n=4,primitive_count=len(s.ALPHABET)))[0]
        independent=calls.numpy_execute(result['program'],x,entries,len(s.ALPHABET))
        assert np.array_equal(native,independent)
        wanted=np.asarray(s.target(jnp.asarray(x),task));exact=np.all(native==wanted,axis=1)
        rows[name]={'cases':len(x),'exact':int(exact.sum()),'mse':float(np.mean((native.astype(np.float64)-wanted)**2)),
            'max_abs_error':float(np.max(np.abs(native.astype(np.float64)-wanted))),
            'all_close':bool(np.allclose(native,wanted,rtol=1e-6,atol=1e-6)),
            'preserved':int(np.all(np.sort(native,axis=1)==np.sort(x,axis=1),axis=1).sum())}
    return {'passed':all(v['exact']==v['cases'] for v in rows.values()),
        'float32_diagnostic_passed':all(v['all_close'] for v in rows.values()),'groups':rows,
        'cases':sum(v['cases'] for v in rows.values()),'independent_agrees':True}


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    old_dir=ROOT/'artifacts/codepy_block_contracts_20261030'
    sources=[HERE/'calls.py',Path(__file__).resolve(),HERE.parent/'block_contracts/contracts.py',
        HERE.parent/'block_contracts/reference.py',HERE.parent/'subtask_discovery/run_experiment.py',
        HERE.parent/'autonomous_blocks/run_experiment.py']
    protocol={'seeds':list(range(5)),'tasks':s.TARGETS,'arms':ARMS,'population':256,'generations':100,'max_primitive_steps':72,
        'fields':['action','r0_source','r1_source','local_pointer','predicate_source','cell_permutation','gate'],
        'field_bounds':calls.BOUNDS,'initialization':'identity bindings and unconditional gates in all arms',
        'binding_sources':['caller registers','four data cells','current pointer cell','zero/one/minusone'],
        'memory_binding':'24permutations of four cells; scratch addresses4..7fixed',
        'gate_features':['always','predicate','not predicate','r0>r1','r0<r1','cell0>cell1','cell2>cell3','r0>0'],
        'budget':'25856candidates per arm;72primitive steps,skipped calls still charged; extra interface cost measured separately',
        'validation':'same heldout groups; strict primary; previously used float32 tolerance secondary; no candidate selection',
        'source_hashes':{str(p):previous.file_hash(p) for p in sources},
        'library_hashes':{str(ROOT/f'artifacts/codepy_subtask_discovery_20261030/library_seed{i}.json'):previous.file_hash(ROOT/f'artifacts/codepy_subtask_discovery_20261030/library_seed{i}.json') for i in range(5)},
        'fixed_baseline_sha256':previous.file_hash(old_dir/'results.json')}
    normalized=json.loads(json.dumps(protocol))
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==normalized
    else:s.old.save(out/'protocol.json',protocol)
    baseline=json.loads((old_dir/'results.json').read_text());runs=[]
    for seed in range(5):
        lib=json.loads((ROOT/f'artifacts/codepy_subtask_discovery_20261030/library_seed{seed}.json').read_text())
        entries=[[v] for v in s.ALPHABET]+lib['blocks'];table,lengths=s.table_for(lib['blocks']);tables=calls.c.contract_tables(entries)
        for ti,task in enumerate(s.TARGETS):
            for ai in range(4):
                arm=ARMS[(ai+seed)%4];path=out/f'{arm}_seed{seed}_{task}.json'
                if path.exists():runs.append(json.loads(path.read_text()));continue
                print('Run',seed,task,arm,flush=True)
                x=np.random.default_rng(11000+seed*10+ti).integers(-2,3,(64,4)).astype(np.float32)
                result=search(seed,x,arm,table,lengths,tables,task)
                result.update(seed=seed,task=task,arm=arm)
                result['certificate']=certify(result,s.old.cases(4,16000+ti),entries,table,lengths,tables,task)
                if arm=='fixed':
                    prior=next(r for r in baseline['runs'] if r['seed']==seed)['tasks'][task]['isolated']
                    assert [g[0] for g in result['program']]==prior['program']
                    assert result['trace']==prior['trace']
                s.old.save(path,result);runs.append(result)
                print('  exact',result['certificate']['passed'],'close',result['certificate']['float32_diagnostic_passed'],
                      'objective',result['training_objective'],'search_s',round(result['elapsed_s'],2),flush=True)
    summary={task:{arm:sum(r['certificate']['passed'] for r in runs if r['task']==task and r['arm']==arm) for arm in ARMS} for task in s.TARGETS}
    s.old.save(out/'results.json',{'protocol':protocol,'runs':runs,'summary':summary})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/codepy_adaptive_calls_20261030'))
    run(parser.parse_args().output_dir)
