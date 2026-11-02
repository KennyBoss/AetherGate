#!/usr/bin/env python3
"""Discover a two-cell program, freeze it, then search sort3 and transfer to sort4.

No comparator implementation or sorting network is seeded. The human prior is
the two-cell training curriculum and a generic gather/run/scatter call ABI.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
from functools import partial

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_PLATFORMS'] = 'cpu'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import evolve_code_tape as ect
import jax
import jax.numpy as jnp
import numpy as np
from independent_executor import numpy_execute


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


@partial(jax.jit, static_argnames=('n',))
def execute(programs, vectors, n):
    return ect.execute_programs(programs, vectors, input_len=n, tape_len=8,
                                max_steps=programs.shape[1]).final_tape[:, :, :n]


@jax.jit
def primitive_score(programs, x):
    output = execute(programs, x, x.shape[1])
    wanted = jnp.sort(x, axis=1)[None]
    mse = jnp.mean((output-wanted)**2, axis=(1, 2))
    preservation = jnp.mean((jnp.sort(output, axis=-1)-wanted)**2, axis=(1, 2))
    return mse + preservation, mse


@partial(jax.jit, static_argnames=('n', 'slots'))
def composition_outputs(genomes, x, block, n, slots):
    pairs = jnp.asarray(list(itertools.combinations(range(n), 2)), jnp.int32)
    table = jnp.concatenate((jnp.zeros((1, 2), jnp.int32), pairs))
    tapes = jnp.broadcast_to(x, (genomes.shape[0], *x.shape))
    def step(tapes, tokens):
        addresses = table[tokens]
        local = jnp.take_along_axis(tapes, addresses[:, None, :], axis=2)
        # Each call uses a fresh local workspace/registers. No sorting code in ABI.
        flat = local.reshape(-1, 2)
        result = execute(block[None], flat, 2)[0].reshape(local.shape)
        batch = jnp.arange(tapes.shape[0])[:, None]
        cases = jnp.arange(x.shape[0])[None, :]
        for k in (0, 1):
            target = addresses[:, k, None]
            old = tapes[batch, cases, target]
            tapes = tapes.at[batch, cases, target].set(jnp.where(tokens[:, None] == 0, old, result[:, :, k]))
        return tapes, None
    return jax.lax.scan(step, tapes, genomes.T)[0]


@partial(jax.jit, static_argnames=('n', 'slots'))
def macro_score(genomes, x, block, n, slots):
    output = composition_outputs(genomes, x, block, n, slots)
    wanted = jnp.sort(x, axis=1)[None]
    mse = jnp.mean((output-wanted)**2, axis=(1, 2))
    preserve = jnp.mean((jnp.sort(output, axis=-1)-wanted)**2, axis=(1, 2))
    return mse + preserve, mse


def search(seed, x, *, alphabet, length, population, generations, scorer):
    key, init = jax.random.split(jax.random.PRNGKey(seed))
    programs = ect.init_programs(init, population, length, len(alphabet))
    alphabet = jnp.asarray(alphabet)
    mutate = jax.jit(lambda key, elites: ect.mutate_elites(key, elites, population=population,
        mutation_rate=.16, random_reset_count=population//20, vocab_size=len(alphabet)))
    jax.block_until_ready(scorer(alphabet[programs], jnp.asarray(x)))
    jax.block_until_ready(mutate(key, programs[:max(16, population//16)]))
    start = time.perf_counter()
    best, champion, first = float('inf'), None, None
    trace = []
    for generation in range(generations+1):
        losses, mse = scorer(alphabet[programs], jnp.asarray(x))
        scores, errors = np.asarray(losses), np.asarray(mse)
        index = int(np.argmin(scores))
        if float(scores[index]) < best:
            best, champion = float(scores[index]), np.asarray(alphabet[programs[index]]).tolist()
        if first is None and np.any(scores == 0):
            first = {'generation':generation, 'evaluations':(generation+1)*population}
        trace.append({'generation':generation, 'best_objective':best, 'min_mse':float(errors.min())})
        if generation < generations:
            key, mutation = jax.random.split(key)
            elite_ids = np.argsort(scores, kind='stable')[:max(16, population//16)].copy()
            programs = mutate(mutation, programs[jnp.asarray(elite_ids)])
    return {'program':champion, 'training_objective':best, 'first_perfect':first,
            'evaluations':population*(generations+1), 'elapsed_s':time.perf_counter()-start,
            'trace':trace, 'training_sha256':hashlib.sha256(x.tobytes()).hexdigest()}


def cases(n, seed):
    rng = np.random.default_rng(seed)
    return {
        'binary': np.asarray(list(itertools.product([0.,1.], repeat=n)), np.float32),
        'small_exhaustive': np.asarray(list(itertools.product(range(-2,3), repeat=n)), np.float32),
        'integer': rng.integers(-50,51,(2048,n)).astype(np.float32),
        'fractional': rng.uniform(-10,10,(2048,n)).astype(np.float32),
        'wide': rng.uniform(-100000,100000,(2048,n)).astype(np.float32),
        'ties': np.repeat(rng.uniform(-50,50,(256,1)).astype(np.float32),n,axis=1)}


def independent_composition(genome, x, block):
    result=x.copy(); pairs=list(itertools.combinations(range(x.shape[1]),2))
    for token in genome:
        if token:
            addresses=list(pairs[token-1])
            result[:,addresses]=numpy_execute(block,result[:,addresses])
    return result


def certify(program, groups, block=None):
    n=next(iter(groups.values())).shape[1]
    rows={}
    for name,x in groups.items():
        out=numpy_execute(program,x) if block is None else independent_composition(program,x,block)
        native=np.asarray(execute(jnp.asarray([program]),jnp.asarray(x),n))[0] if block is None else np.asarray(
            composition_outputs(jnp.asarray([program]),jnp.asarray(x),jnp.asarray(block),n,len(program)))[0]
        if not np.array_equal(native,out):
            raise AssertionError('Independent interpreter disagrees')
        wanted=np.sort(x,axis=1)
        rows[name]={'cases':len(x),'exact':int(np.all(out==wanted,axis=1).sum()),
            'preserved':int(np.all(np.sort(out,axis=1)==wanted,axis=1).sum()),
            'max_abs_error':float(np.max(np.abs(out-wanted)))}
    return {'passed':all(v['exact']==v['cases'] and v['preserved']==v['cases'] for v in rows.values()),
            'cases':sum(v['cases'] for v in rows.values()),'groups':rows,'independent_agrees':True}


def run(args):
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    protocol={'seeds':args.seeds,'discovery_population':args.discovery_population,
        'discovery_generations':args.discovery_generations,'discovery_length':12,
        'composition_population':256,'composition_generations':100,'slots':{'3':3,'4':6},
        'cases':64,'manual_priors':['two-cell sorting curriculum','gather/run/scatter pair ABI',
            'existing DSL primitives; two alphabets, with and without opcode35'],
        'selection':'training objective only; separate promotion validation; final certificates never select candidates',
        'transfer':'identical frozen block, no sort4 library learning, search new call sequence',
        'budget':'equal downstream evaluations and maximum block-instruction steps; discovery cost additional and disclosed',
        'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__).resolve(),HERE/'independent_executor.py',HERE.parent/'evolve_code_tape.py')}}
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=protocol: raise ValueError('Protocol mismatch')
    else: save(out/'protocol.json',protocol)
    results=[]
    for mode in ('existing_dsl','without_sort2_opcode'):
        alphabet=[i for i in range(len(ect.TAPE_BLOCKS)) if mode=='existing_dsl' or i!=35]
        for seed in args.seeds:
            record=out/f'{mode}_seed{seed}.json'
            if record.exists(): results.append(json.loads(record.read_text())); continue
            print('Discover',mode,seed,flush=True)
            x=np.random.default_rng(1000+seed).integers(-2,3,(64,2)).astype(np.float32)
            discovery=search(seed,x,alphabet=alphabet,length=12,population=args.discovery_population,
                generations=args.discovery_generations,scorer=primitive_score)
            discovery['certificate']=certify(discovery['program'],cases(2,7000+seed))
            row={'mode':mode,'seed':seed,'discovery':discovery,'tasks':{}}
            block=discovery['program'] if discovery['certificate']['passed'] else None
            print('  discovered',bool(block),'objective',discovery['training_objective'],flush=True)
            # No failed-discovery fallback to the manual library.
            for n,slots in ((3,3),(4,6)):
                train=np.random.default_rng(2000+seed+n).integers(-2,3,(64,n)).astype(np.float32)
                raw=search(seed,train,alphabet=alphabet,length=slots*12,population=256,generations=100,
                    scorer=primitive_score)
                raw['certificate']=certify(raw['program'],cases(n,9000+n))
                task={'primitive':raw,'composition':None}
                if block is not None:
                    scorer=lambda genomes,x: macro_score(genomes,x,jnp.asarray(block),n,slots)
                    composed=search(seed,train,alphabet=list(range(1+n*(n-1)//2)),length=slots,
                        population=256,generations=100,scorer=scorer)
                    composed['certificate']=certify(composed['program'],cases(n,9000+n),block)
                    task['composition']=composed
                row['tasks'][str(n)]=task
                print('  sort',n,'raw',raw['certificate']['passed'],'composition',
                      task['composition']['certificate']['passed'] if block else 'no block',flush=True)
            save(record,row); results.append(row)
    summary={}
    for mode in ('existing_dsl','without_sort2_opcode'):
        rows=[r for r in results if r['mode']==mode]
        summary[mode]={'discovered':sum(r['discovery']['certificate']['passed'] for r in rows),'seeds':len(rows)}
        for n in ('3','4'):
            summary[mode]['sort'+n]={arm:sum(bool(r['tasks'][n][arm]) and r['tasks'][n][arm]['certificate']['passed'] for r in rows)
                                     for arm in ('primitive','composition')}
    save(out/'results.json',{'protocol':protocol,'summary':summary,'runs':results})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/codepy_autonomous_blocks_20261030'))
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    parser.add_argument('--discovery-population',type=int,default=1024)
    parser.add_argument('--discovery-generations',type=int,default=1000)
    run(parser.parse_args())
