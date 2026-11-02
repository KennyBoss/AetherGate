"""Single registered fixed-budget comparison against the existing argument-search arm."""
import os,sys,time,json,hashlib,importlib.util,argparse
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_PLATFORM_NAME']='cpu'
from pathlib import Path
import numpy as np
from abi import vm,tables,compose,numpy_compose
from mining import mine
import jax,jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
OLD=ROOT/'artifacts/codepy_subtask_discovery_20261030';BASE=ROOT/'artifacts/codepy_adaptive_calls_20261030'
sys.path.insert(0,str(ROOT/'research/codepy/adaptive_calls'))
spec=importlib.util.spec_from_file_location('adaptive_baseline',ROOT/'research/codepy/adaptive_calls/run_experiment.py');baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
s=baseline.s

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def prepare(out):
    assert not (out/'protocol.json').exists()
    source_files=list(HERE.glob('*.py'))+[ROOT/'research/codepy/subtask_discovery/run_experiment.py',ROOT/'research/codepy/autonomous_blocks/run_experiment.py',ROOT/'research/codepy/autonomous_blocks/independent_executor.py',ROOT/'research/codepy/evolve_code_tape.py',ROOT/'research/codepy/block_contracts/contracts.py',ROOT/'research/codepy/block_contracts/reference.py',ROOT/'research/codepy/adaptive_calls/run_experiment.py',ROOT/'research/codepy/adaptive_calls/calls.py']
    p={'version':'codepy-behavioral-pair-budget-v1','seeds':list(range(5)),'tasks':list(s.TARGETS),'population':256,'evaluated_generations':101,'evaluations_per_target':25856,'train_cases_per_target':64,'max_primitive_steps':72,'composition_slots':6,'block_step_cap':12,'local_input_length':3,'local_argument_layouts':[[0,1],[0,2],[1,2]],'caller_pairs':'all12 ordered distinct pairs of4 cells; token0 noop','source_archives':{str(OLD/f'library_seed{i}.json'):sha(OLD/f'library_seed{i}.json') for i in range(5)},'baseline_results_sha256':sha(BASE/'results.json'),'source_sha256':{str(p):sha(p) for p in source_files},'candidate_generation':'source nonnoopwindows2..12 and generic backward tape-output slices with pointer relocation. Atmost25%ofoldablation budget probes these; remaining budget may mutate/recombine source-initialized12command candidates for missingbehaviorfamilies. No supplied program bodies or networks.','library_discovery':{'probe_pairs':'Cartesian{-2,-1,0,1,2}^2','source_probe_budget_fraction':.25,'repair_population':256,'mutation_rate':.16,'repair_order':'alternate missing swap/order niches, stop promoted family','challenge_cases':64,'promotion_cases':6429,'qualification_data':'seeds24601+seed and27000+seed, arity2 only'},'behavioral_prior':'Human specified exact two-value preservation, nonidentity, either unconditional exchange or consistent ascending/descending output. It is a supplied pair criterion, not autonomous invention ofgoals.','preparation_budget':'Perseed cap=old ablation_evaluations*64*36 primitivecase steps. All gridprobes, boundedrepair evaluations, challenge64, nativeandNumPy promotion6429 charged at padded12steps. Staticanalysis, data movement andJIT measuredwalltime separately; no FLOP/time equality claimed. Originalsourcearchives shared unchanged.','selection':'One representative per swap/order family; orderorientation learned frombehavior, bothdirections supported by orderedcallerarguments. Frozen5libraries before any downstreamsearch. No manual or historicalprogramfallback.','composition':'6 random/mutated slots withnoops,allowing shortchains up to6calls. Fresh localzero registers,predicate,pointer,scratch everycall; two arguments gathered, onlythoseoutputs scattered. All6slots charged72steps evenifnoop.','fitness':'Same original targetMSE+multisetpreservation on same64TRAINvectors; first stricttrainingimprovement retained. Final7041inputs neverselect programs/libraries.','baseline':'Fresh existing arguments-search for all20seedtask pairs, same25856evaluations/64TRAIN/72stepcap; must reproduce archivedchampion andtrainingtrace. Historical preparation inputs unchanged.','acceptance':'Atleastone strictsort4 andonedescending4 success; retain every previouslysuccessful arguments reverse4/rotate4 seed. Allseeds included evenmissinglibrary.','fairness_limit':'Compare whole approaches; interfaces, candidate extraction and pair behavioral prior change jointly. No isolated attribution to only one component. All newfunctions are learned bytecode, no sort2 opcode35, hand-coded swap/comparator, supplied sortingnetwork or reuseof downstreamchampions in newsearch.'}
    save(out/'protocol.json',p)
    previous={}
    for folder in (ROOT/'artifacts').glob('codepy_*'):
        if folder==out:continue
        for f in folder.rglob('*'):
            if f.is_file() and f.suffix!='.pyc' and f.name!='.DS_Store' and not any('cache' in part.lower() for part in f.parts):previous[str(f)]=sha(f)
    save(out/'old_artifact_hashes.json',previous);return p

def certify(genome,task,blocks):
    codes,layouts=tables(blocks);rows={}
    for name,x in vm.cases(4,16000+s.TARGETS.index(task)).items():
        native=np.asarray(compose(jnp.asarray([genome]),jnp.asarray(x),codes,layouts))[0];independent=numpy_compose(genome,x,blocks);np.testing.assert_array_equal(native,independent)
        wanted=np.asarray(s.target(jnp.asarray(x),task));exact=np.all(native==wanted,1);preserved=np.all(np.sort(native,axis=1)==np.sort(x,axis=1),1)
        rows[name]={'cases':len(x),'exact':int(exact.sum()),'preserved':int(preserved.sum()),'mse':float(np.mean((native.astype(np.float64)-wanted)**2)),'max_abs_error':float(np.max(np.abs(native.astype(np.float64)-wanted)))}
    return {'passed':all(r['cases']==r['exact']==r['preserved'] for r in rows.values()),'cases':sum(r['cases'] for r in rows.values()),'groups':rows,'independent_agrees':True}

def run(out):
    out.mkdir(parents=True,exist_ok=True);p=json.loads((out/'protocol.json').read_text()) if (out/'protocol.json').exists() else prepare(out)
    for f,h in p['source_sha256'].items():assert sha(f)==h,('Sourcechanged',f)
    for f,h in p['source_archives'].items():assert sha(f)==h
    prior=json.loads((BASE/'results.json').read_text());assert sha(BASE/'results.json')==p['baseline_results_sha256'];libraries={}
    for seed in p['seeds']:
        path=out/f'library_seed{seed}.json'
        if path.exists():r=json.loads(path.read_text())
        else:
            r=mine(json.loads((OLD/f'library_seed{seed}.json').read_text()),seed);r.update(seed=seed,source_archive_sha256=p['source_archives'][str(OLD/f'library_seed{seed}.json')]);save(path,r)
        assert r['preparation_case_steps_used']<=r['preparation_case_step_budget'];libraries[seed]=r
        print('LIBRARY',seed,[b['behavior'] for b in r['blocks']],'prep',r['preparation_case_steps_used'],'/',r['preparation_case_step_budget'],flush=True)
    manifest={str(out/f'library_seed{i}.json'):sha(out/f'library_seed{i}.json') for i in p['seeds']};save(out/'library_manifest.json',manifest)
    results=[]
    for seed in p['seeds']:
        oldlib=json.loads((OLD/f'library_seed{seed}.json').read_text());oldtable,oldlength=s.table_for(oldlib['blocks']);oldtables=baseline.calls.c.contract_tables([[v] for v in s.ALPHABET]+oldlib['blocks']);blocks=libraries[seed]['blocks'];codes,layouts=tables(blocks)
        for ti,task in enumerate(p['tasks']):
            record=out/f'seed{seed}_{task}.json'
            if record.exists():results.append(json.loads(record.read_text()));continue
            x=np.random.default_rng(11000+seed*10+ti).integers(-2,3,(64,4)).astype(np.float32);r={'seed':seed,'task':task,'arms':{}}
            for arm in (['baseline','behavioral'] if seed%2==0 else ['behavioral','baseline']):
                print('SEARCH',seed,task,arm,flush=True)
                if arm=='baseline':
                    q=baseline.search(seed,x,'arguments',oldtable,oldlength,oldtables,task);q['certificate']=baseline.certify(q,vm.cases(4,16000+ti),[[v] for v in s.ALPHABET]+oldlib['blocks'],oldtable,oldlength,oldtables,task)
                    old=next(z for z in prior['runs'] if z['seed']==seed and z['task']==task and z['arm']=='arguments');assert q['program']==old['program'] and q['trace']==old['trace'];q['archived_search_reproduced']=True
                else:
                    @jax.jit
                    def scorer(g,inputs):return s.losses(compose(g,inputs,codes,layouts),inputs,task)
                    start=time.perf_counter();q=vm.search(seed,x,alphabet=list(range(1+12*len(blocks))),length=6,population=256,generations=100,scorer=scorer);q['compilation_and_warmup_s']=max(0.,time.perf_counter()-start-q['elapsed_s']);q['certificate']=certify(q['program'],task,blocks);q['nonnoop_calls']=sum(v!=0 for v in q['program'])
                assert q['evaluations']==25856 and q['certificate']['cases']==7041;r['arms'][arm]=q;print('RESULT',seed,task,arm,q['certificate']['passed'],q['training_objective'],flush=True)
            assert r['arms']['baseline']['training_sha256']==r['arms']['behavioral']['training_sha256']
            for f,h in manifest.items():assert sha(f)==h
            save(record,r);results.append(r)
    summary={task:{arm:sum(r['arms'][arm]['certificate']['passed'] for r in results if r['task']==task) for arm in ['baseline','behavioral']} for task in p['tasks']}
    guards=[{'seed':r['seed'],'task':r['task'],'baseline_passed':r['arms']['baseline']['certificate']['passed'],'new_passed':r['arms']['behavioral']['certificate']['passed'],'preserved':not r['arms']['baseline']['certificate']['passed'] or r['arms']['behavioral']['certificate']['passed']} for r in results if r['task'] in ['reverse4','rotate4']]
    success=summary['sort4']['behavioral']>0 and summary['descending4']['behavioral']>0 and all(r['preserved'] for r in guards)
    save(out/'results.json',{'protocol':p,'summary':summary,'runs':results,'preservation_guards':guards,'success':success,'candidate_status':'accepted' if success else 'rejected'});print(json.dumps({'summary':summary,'preservation':guards,'success':success},indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/codepy_behavioral_blocks_20261101');run(parser.parse_args().output_dir.resolve())
