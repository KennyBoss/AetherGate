#!/usr/bin/env python3
"""Paired search with inferred capsules vs unchanged inline fragments."""
import contracts as c
from reference import numpy_state
import importlib.util
from pathlib import Path
import argparse
import hashlib
import json
import time
import math
import numpy as np
import jax
import jax.numpy as jnp

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('fragment_experiment',HERE.parent/'subtask_discovery/run_experiment.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def file_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def numpy_capsule_program(genome,x,entries,contracts,primitive_count):
    state=c.State(np.zeros(len(x),np.float32),np.zeros(len(x),np.float32),np.zeros(len(x),np.int32),
        np.zeros(len(x),bool),np.pad(x,((0,0),(0,4))))
    remaining=72
    for token in genome:
        if remaining<=0:break
        code=entries[token][:remaining];remaining-=len(code)
        if token<primitive_count:
            state=c.State(*numpy_state(code,state,4));continue
        contract=c.infer(code)
        local=[np.zeros_like(v) for v in state]
        for name in contract['input_registers']:local[c.REGS.index(name)]=state[c.REGS.index(name)].copy()
        input_mask=np.zeros_like(state.tape,dtype=bool);output_mask=np.zeros_like(input_mask)
        for i,ptr in enumerate(state.ptr):
            footprint=contract['memory_by_initial_pointer'][int(ptr)]
            input_mask[i,footprint['input_cells']]=True;output_mask[i,footprint['output_cells']]=True
        local[4]=np.where(input_mask,state.tape,0)
        local=numpy_state(code,local,4)
        committed=[v.copy() for v in state]
        for name in contract['output_registers']:committed[c.REGS.index(name)]=local[c.REGS.index(name)]
        committed[4]=np.where(output_mask,local[4],state.tape)
        state=c.State(*committed)
    return state.tape[:,:4]


def certify(result,task,entries,table,lengths,tables):
    groups=s.old.cases(4,16000+s.TARGETS.index(task));rows={}
    for name,x in groups.items():
        native=np.asarray(c.execute(jnp.asarray([result['program']]),jnp.asarray(x),table,lengths,tables,n=4,primitive_count=len(s.ALPHABET)))[0]
        independent=numpy_capsule_program(result['program'],x,entries,None,len(s.ALPHABET))
        assert np.array_equal(native,independent)
        wanted=np.asarray(s.target(jnp.asarray(x),task));exact=np.all(native==wanted,axis=1)
        rows[name]={'cases':len(x),'exact':int(exact.sum()),'mse':float(np.mean((native.astype(np.float64)-wanted)**2)),
            'max_abs_error':float(np.max(np.abs(native.astype(np.float64)-wanted))),
            'preserved':int(np.all(np.sort(native,axis=1)==np.sort(x,axis=1),axis=1).sum()),
            'all_close':bool(np.allclose(native,wanted,rtol=1e-6,atol=1e-6))}
    return {'passed':all(v['cases']==v['exact'] for v in rows.values()),
        'float32_diagnostic_passed':all(v['all_close'] for v in rows.values()),
        'groups':rows,'cases':sum(v['cases'] for v in rows.values()),'independent_agrees':True}


def prepare(out):
    previous=ROOT/'artifacts/codepy_subtask_discovery_20261030'
    files=[HERE/'contracts.py',HERE/'reference.py',Path(__file__).resolve(),
        HERE.parent/'subtask_discovery/run_experiment.py',HERE.parent/'autonomous_blocks/run_experiment.py',
        HERE.parent/'evolve_code_tape.py']
    protocol={'seeds':list(range(5)),'tasks':s.TARGETS,'population':256,'generations':100,'max_steps':72,
        'inputs':'read-before-write registers, exact conservative tape reads by initial pointer, input length when used',
        'outputs':'all written registers/tape, conservative; no source-dependent dead-output removal',
        'isolation':'zero nonarguments, execute on local state, commit only declared writes; preserve required input values',
        'contracts_semantics':'includes clips and partial macro at72-step boundary; complete-state calls should preserve inline behavior',
        'primary':'exact certification; unchanged posthoc float32 tolerance also reported',
        'previous_results_sha256':file_hash(previous/'results.json'),
        'sources':{str(p):file_hash(p) for p in files},
        'libraries':{str(previous/f'library_seed{i}.json'):file_hash(previous/f'library_seed{i}.json') for i in range(5)}}
    normalized=json.loads(json.dumps(protocol))
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==normalized
    else:s.old.save(out/'protocol.json',protocol)
    return previous,protocol


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    previous,protocol=prepare(out)
    baseline=json.loads((previous/'results.json').read_text())
    runs=[]
    for seed in range(5):
        record=out/f'seed{seed}.json'
        if record.exists():runs.append(json.loads(record.read_text()));continue
        start=time.perf_counter()
        library=json.loads((previous/f'library_seed{seed}.json').read_text());blocks=library['blocks']
        entries=[[v] for v in s.ALPHABET]+blocks
        contracts=[c.infer(block) for block in blocks]
        table,lengths=s.table_for(blocks);tables=c.contract_tables(entries)
        s.old.save(out/f'contracts_seed{seed}.json',{'contracts':contracts,'source_library_sha256':file_hash(previous/f'library_seed{seed}.json')})
        preparation_s=time.perf_counter()-start
        row={'seed':seed,'preparation_s':preparation_s,'contracts_sha256':file_hash(out/f'contracts_seed{seed}.json'),'tasks':{}}
        print('Seed',seed,'contracts',len(contracts),flush=True)
        for ti,task in enumerate(s.TARGETS):
            x=np.random.default_rng(11000+seed*10+ti).integers(-2,3,(64,4)).astype(np.float32)
            results={}
            # Same seeds/data/budgets; order alternates to reduce order bias in timing.
            for arm in (('inline','isolated') if seed%2==0 else ('isolated','inline')):
                def scorer(genomes,inputs):
                    if arm=='inline':return s.score(s.expand(genomes,table,lengths),inputs,task)
                    return s.losses(c.execute(genomes,inputs,table,lengths,tables,n=4,primitive_count=len(s.ALPHABET)),inputs,task)
                result=s.old.search(seed,x,alphabet=list(range(len(entries))),length=72,population=256,generations=100,scorer=scorer)
                if arm=='isolated':result['certificate']=certify(result,task,entries,table,lengths,tables)
                else:result['certificate']=s.certify(result['program'],task,table,lengths)
                results[arm]=result
                print(' ',task,arm,result['certificate']['passed'],result['training_objective'],flush=True)
            results['same_champion']=results['inline']['program']==results['isolated']['program']
            results['same_training_trace']=results['inline']['trace']==results['isolated']['trace']
            old_record=next(v for v in baseline['runs'] if v['seed']==seed)
            assert results['inline']['program']==old_record['tasks'][task]['automatic']['program']
            assert results['inline']['training_sha256']==results['isolated']['training_sha256']
            row['tasks'][task]=results
        s.old.save(record,row);runs.append(row)
    summary={task:{arm:sum(r['tasks'][task][arm]['certificate']['passed'] for r in runs) for arm in ('inline','isolated')} for task in s.TARGETS}
    s.old.save(out/'results.json',{'protocol':protocol,'runs':runs,'summary':summary})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/codepy_block_contracts_20261030'))
    run(parser.parse_args().output_dir)
