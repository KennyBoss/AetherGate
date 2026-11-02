#!/usr/bin/env python3
"""Select reusable subprograms by source-task ablation, freeze, test new tasks.

No two-cell objective or pair interface is used by the automatic method.
Subtasks here mean concrete fragments whose removal harms a source solution.
This is fragment/subgoal selection, not autonomous invention of final goals.
"""
import os
os.environ['JAX_PLATFORM_NAME']='cpu'
os.environ['JAX_PLATFORMS']='cpu'
import sys
from pathlib import Path
import argparse
import itertools
import hashlib
import json
import math
from functools import partial
from collections import defaultdict

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'autonomous_blocks'))
import run_experiment as old
import jax
import jax.numpy as jnp
import numpy as np

ROOT=HERE.parents[2]
ALPHABET=[i for i in range(36) if i not in (18,19,29,30,31,35)]
SOURCES=('sort3','reverse3')
TARGETS=('sort4','reverse4','rotate4','descending4')


def target(x,task):
    if task.startswith('sort'):return jnp.sort(x,axis=-1)
    if task.startswith('descending'):return jnp.sort(x,axis=-1)[:,::-1]
    if task.startswith('reverse'):return x[:,::-1]
    if task.startswith('rotate'):return jnp.roll(x,1,axis=-1)
    raise ValueError(task)


def losses(output,x,task):
    y=target(x,task)[None]
    mse=jnp.mean((output-y)**2,axis=(1,2))
    preservation=jnp.mean((jnp.sort(output,axis=-1)-jnp.sort(x,axis=-1)[None])**2,axis=(1,2))
    return mse+preservation,mse


@partial(jax.jit,static_argnames=('task',))
def score(programs,x,task):
    return losses(old.execute(programs,x,x.shape[1]),x,task)


def table_for(blocks):
    entries=[[i] for i in ALPHABET]+[list(b) for b in blocks]
    table=np.zeros((len(entries),8),np.int32)
    for i,e in enumerate(entries):table[i,:len(e)]=e
    return jnp.asarray(table),jnp.asarray([len(e) for e in entries],jnp.int32)


@jax.jit
def expand(genomes,table,lengths):
    sizes=lengths[genomes]
    starts=jnp.cumsum(sizes,axis=1)-sizes
    positions=starts[:,:,None]+jnp.arange(8)[None,None,:]
    positions=jnp.where((jnp.arange(8)[None,None,:]<sizes[:,:,None]) & (positions<72),positions,72)
    batch=jnp.arange(genomes.shape[0])[:,None,None]
    return jnp.zeros((genomes.shape[0],72),jnp.int32).at[batch,positions].set(table[genomes],mode='drop')


def source_search(seed,task):
    x=np.random.default_rng(3000+seed+SOURCES.index(task)).integers(-2,3,(64,3)).astype(np.float32)
    key,init=jax.random.split(jax.random.PRNGKey(seed))
    genomes=old.ect.init_programs(init,256,36,len(ALPHABET))
    mutate=jax.jit(lambda key,elites:old.ect.mutate_elites(key,elites,population=256,
        mutation_rate=.16,random_reset_count=12,vocab_size=len(ALPHABET)))
    archive={}
    for generation in range(201):
        programs=jnp.asarray(ALPHABET)[genomes]
        objective,_=score(programs,jnp.asarray(x),task)
        values=np.asarray(objective);ids=np.argsort(values,kind='stable')[:16].copy()
        if generation%20==0 or generation==200:
            for p in np.asarray(programs)[ids]:archive[tuple(p.tolist())]=float(values[np.where(np.all(np.asarray(programs)==p,axis=1))[0][0]])
        key,sub=jax.random.split(key)
        if generation<200:genomes=mutate(sub,genomes[jnp.asarray(ids)])
    best=min(archive,key=archive.get)
    return {'task':task,'x':x.tolist(),'archive':[list(p) for p in archive],
            'best_objective':archive[best],'evaluations':256*201}


def mine(sources):
    occurrences=defaultdict(list)
    for si,source in enumerate(sources):
        for pi,program in enumerate(source['archive']):
            for width in (2,3,4,6,8):
                for start in range(37-width):
                    fragment=tuple(program[start:start+width])
                    if 0 not in fragment and len(set(fragment))>1:
                        occurrences[fragment].append((si,pi,start))
    ranked=sorted(occurrences,key=lambda f:(-(len(set((s,p) for s,p,_ in occurrences[f]))*(len(f)-1)-len(f)),f))[:128]
    candidates=[]; ablation_evaluations=0
    for fragment in ranked:
        contexts=[];seen=set()
        for si,pi,start in occurrences[fragment]:
            if (si,pi) not in seen:
                contexts.append((si,pi,start));seen.add((si,pi))
            if len(contexts)==8:break
        gains=[]
        for si,pi,start in contexts:
            source=sources[si];p=source['archive'][pi];ablated=p.copy()
            ablated[start:start+len(fragment)]=[0]*len(fragment)
            values=np.asarray(score(jnp.asarray([p,ablated]),jnp.asarray(source['x'],dtype=jnp.float32),source['task'])[0])
            gains.append(float(values[1]-values[0]));ablation_evaluations+=2
        usefulness=float(np.mean(np.maximum(gains,0)))
        compression=len(set((s,p) for s,p,_ in occurrences[fragment]))*(len(fragment)-1)-len(fragment)
        candidates.append({'code':list(fragment),'contexts':len(contexts),'ablation_gains':gains,
            'utility':usefulness,'compression_proxy':compression,'selection_score':usefulness*max(compression,0)})
    ordered=sorted(candidates,key=lambda c:-c['selection_score'])
    selected=[]
    for c in ordered:
        if c['selection_score']<=0:continue
        # Avoid promoting only overlapping prefixes/suffixes of a single chunk.
        grams=set(zip(c['code'],c['code'][1:]))
        if any(len(grams & set(zip(b['code'],b['code'][1:]))) / len(grams | set(zip(b['code'],b['code'][1:])))>.6 for b in selected):continue
        selected.append(c)
        if len(selected)==8:break
    return {'candidates':candidates,'selected':selected,'ablation_evaluations':ablation_evaluations}


def certify(program,task,table=None,lengths=None,legacy=None):
    groups=old.cases(4,16000+TARGETS.index(task)); rows={}
    expanded=np.asarray(expand(jnp.asarray([program]),table,lengths))[0] if table is not None else program
    for name,x in groups.items():
        if legacy is None:
            independent=old.numpy_execute(expanded,x)
            native=np.asarray(old.execute(jnp.asarray([expanded]),jnp.asarray(x),4))[0]
        else:
            independent=old.independent_composition(program,x,legacy)
            native=np.asarray(old.composition_outputs(jnp.asarray([program]),jnp.asarray(x),jnp.asarray(legacy),4,6))[0]
        assert np.array_equal(independent,native)
        y=np.asarray(target(jnp.asarray(x),task));exact=np.all(independent==y,axis=1)
        rows[name]={'cases':len(x),'exact':int(exact.sum()),'mse':float(np.mean((independent.astype(np.float64)-y)**2)),
            'preserved':int(np.all(np.sort(independent,axis=1)==np.sort(x,axis=1),axis=1).sum())}
    return {'passed':all(v['cases']==v['exact'] for v in rows.values()),'groups':rows,
            'cases':sum(v['cases'] for v in rows.values()),'exact':sum(v['exact'] for v in rows.values()),
            'independent_agrees':True}


def run(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    legacy_path=ROOT/'artifacts/codepy_autonomous_blocks_20261030/results.json'
    legacy_data=json.loads(legacy_path.read_text())
    protocol={'seeds':args.seeds,'sources':SOURCES,'heldout_tasks':TARGETS,'source_population':256,
        'source_generations':200,'source_length':36,'max_blocks':8,'downstream_population':256,
        'downstream_generations':100,'max_steps':72,'alphabet':ALPHABET,
        'selection':'frequency/compression shortlist128; source-context ablation gains; overlap-diverse top8',
        'legacy_sha256':hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__).resolve(),HERE.parent/'autonomous_blocks/run_experiment.py',HERE.parent/'autonomous_blocks/independent_executor.py',HERE.parent/'evolve_code_tape.py')},
        'limits':['Human supplied source whole tasks and mining rule; no explicit intermediate targets.',
            'Automatic arm uses inline state-sharing blocks; legacy arm retains its pair-call ABI.',
            'Source search and ablation costs charged to raw control; ABI/JIT/certification uncharged.']}
    normalized=json.loads(json.dumps(protocol))
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==normalized
    else:old.save(out/'protocol.json',protocol)
    rows=[]
    for seed in args.seeds:
        path=out/f'seed{seed}.json'
        if path.exists():rows.append(json.loads(path.read_text()));continue
        print('Source searches',seed,flush=True)
        sources=[source_search(seed,t) for t in SOURCES]
        mining=mine(sources);blocks=[c['code'] for c in mining['selected']]
        rng=np.random.default_rng(8000+seed)
        pool=[c['code'] for c in mining['candidates']]
        random_blocks=[pool[i] for i in rng.choice(len(pool),len(blocks),replace=False)]
        frozen={'seed':seed,'sources':sources,'mining':mining,'blocks':blocks,'random_blocks':random_blocks}
        old.save(out/f'library_seed{seed}.json',frozen)
        library_hash=hashlib.sha256((out/f'library_seed{seed}.json').read_bytes()).hexdigest()
        print('  selected',len(blocks),'blocks',blocks,flush=True)
        # Charge all source evaluations and ablation evaluations, amortized across4 targets.
        overhead=(2*256*201+mining['ablation_evaluations'])*36
        control_evals=math.ceil((256*101+overhead/(4*72))/256)*256
        legacy_record=next(r for r in legacy_data['runs'] if r['mode']=='without_sort2_opcode' and r['seed']==seed)
        legacy=legacy_record['discovery']['program'] if legacy_record['discovery']['certificate']['passed'] else None
        row={'seed':seed,'library_hash':library_hash,'block_count':len(blocks),'source_candidate_steps':overhead,
             'cost_control_evaluations':control_evals,'tasks':{}}
        for ti,task in enumerate(TARGETS):
            x=np.random.default_rng(11000+seed*10+ti).integers(-2,3,(64,4)).astype(np.float32)
            task_results={}
            for arm,lib in (('raw',[]),('automatic',blocks),('random',random_blocks),('charged_raw',[])):
                table,lengths=table_for(lib)
                scorer=lambda genomes,x: score(expand(genomes,table,lengths),x,task)
                result=old.search(seed,x,alphabet=list(range(len(ALPHABET)+len(lib))),length=72,population=256,
                    generations=(control_evals//256-1 if arm=='charged_raw' else 100),scorer=scorer)
                result['certificate']=certify(result['program'],task,table,lengths)
                task_results[arm]=result
                print(' ',task,arm,result['certificate']['passed'],result['certificate']['groups']['small_exhaustive']['mse'],flush=True)
            if legacy is not None:
                scorer=lambda genomes,x:losses(old.composition_outputs(genomes,x,jnp.asarray(legacy),4,6),x,task)
                result=old.search(seed,x,alphabet=list(range(7)),length=6,population=256,generations=100,scorer=scorer)
                result['certificate']=certify(result['program'],task,legacy=legacy)
                task_results['current_curriculum']=result
            else:task_results['current_curriculum']=None
            row['tasks'][task]=task_results
            assert hashlib.sha256((out/f'library_seed{seed}.json').read_bytes()).hexdigest()==library_hash
        old.save(path,row);rows.append(row)
    summary={t:{a:sum(bool(r['tasks'][t][a]) and r['tasks'][t][a]['certificate']['passed'] for r in rows)
        for a in ('raw','automatic','random','charged_raw','current_curriculum')} for t in TARGETS}
    old.save(out/'results.json',{'protocol':protocol,'summary':summary,'runs':rows})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/codepy_subtask_discovery_20261030'))
    run(parser.parse_args())
