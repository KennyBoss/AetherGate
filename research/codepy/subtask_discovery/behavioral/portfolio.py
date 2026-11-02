"""TRAIN-selected portfolio paid from the original complete-pipeline step budget."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_PLATFORM_NAME']='cpu'
import json,time,hashlib
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from run_behavioral import ROOT,HERE,OLD,BASE,sha,save,baseline,s,vm,tables,compose,certify
FIRST=ROOT/'artifacts/codepy_behavioral_blocks_20261101'
OUT=ROOT/'artifacts/codepy_behavioral_portfolio_20261101'

def choose_training_program(branches):
    return min(branches,key=lambda name:branches[name]['training_objective'])

def budget_ledger(source,selection,qualification,new_preparation):
    old_target=4*25856*64*72;new_target=4*25600*64*72
    return {'shared_source_units':int(source),'old_selection_units':int(selection),'shared_contract_qualification_units':int(qualification),'extra_behavior_preparation_units':int(new_preparation),'old_target_units':old_target,'new_target_units':new_target,'saved_target_units':old_target-new_target,'baseline_total_units':int(source+selection+qualification+old_target),'new_total_units':int(source+selection+qualification+new_preparation+new_target),'within_original_total_budget':new_preparation<=old_target-new_target}

def arguments_prefix(seed,x,table,lengths,tables_,task,generations):
    key,init=jax.random.split(jax.random.PRNGKey(seed));actions=len(table);genomes=jnp.zeros((256,72,7),jnp.int32).at[:,:,0].set(vm.ect.init_programs(init,256,72,actions))
    @jax.jit
    def scorer(g):return s.losses(baseline.calls.execute(g,jnp.asarray(x),table,lengths,tables_,n=4,primitive_count=len(s.ALPHABET)),jnp.asarray(x),task)
    begin=time.perf_counter();jax.block_until_ready(scorer(genomes));jax.block_until_ready(baseline.mutate(key,genomes[:16],actions=actions,arm='arguments'));warm=time.perf_counter()-begin
    begin=time.perf_counter();best=float('inf');champion=None;trace=[];first=None
    for generation in range(generations):
        objective,mse=scorer(genomes);value=np.asarray(objective);errors=np.asarray(mse);i=int(np.argmin(value))
        if float(value[i])<best:best=float(value[i]);champion=np.asarray(genomes[i]).tolist()
        if first is None and np.any(value==0):first={'generation':generation,'evaluations':256*(generation+1)}
        trace.append({'generation':generation,'best_objective':best,'min_mse':float(errors.min())})
        if generation+1<generations:
            key,mk=jax.random.split(key);ids=np.argsort(value,kind='stable')[:16].copy();genomes=baseline.mutate(mk,genomes[jnp.asarray(ids)],actions=actions,arm='arguments')
    return {'program':champion,'training_objective':best,'first_perfect':first,'trace':trace,'evaluations':256*generations,'elapsed_s':time.perf_counter()-begin,'compilation_and_warmup_s':warm,'training_sha256':hashlib.sha256(x.tobytes()).hexdigest()}

def prepare():
    OUT.mkdir(exist_ok=False);first_protocol=json.loads((FIRST/'protocol.json').read_text());files={**first_protocol['source_sha256'],str(Path(__file__).resolve()):sha(__file__),str(HERE/'test_portfolio.py'):sha(HERE/'test_portfolio.py')}
    for name,h in files.items():assert sha(name)==h
    libraries={str(FIRST/f'library_seed{i}.json'):sha(FIRST/f'library_seed{i}.json') for i in range(5)}
    p={'version':'codepy-behavioral-portfolio-totalbudget-v1','seeds':list(range(5)),'tasks':list(s.TARGETS),'baseline_evaluations_per_task':25856,'portfolio_evaluations_per_task':25600,'population':256,'evaluated_generations_by_branch_when_library_exists':{'arguments':50,'behavioral':50},'when_library_empty':'100generations of original arguments search; no invented fallbackblock','selection':'min TRAINobjective only, ties prefer arguments because inserted first; choose before any certificates','source_sha256':files,'libraries':libraries,'first_series_results_sha256':sha(FIRST/'results.json'),'first_series_library_manifest_sha256':sha(FIRST/'library_manifest.json'),'budget':'One256candidate batch saved on each of4targets:4*256*64*72=4718592 units, covers all extra behavioralpreparation. Old source,oldmining andoldcontractqualification charged tobothmethods; extra newpreparation charged onlytoportfolio. Staticanalysis/JIT/ABI cost notFLOPmatched; walltime recorded separately.','acceptance':'Positive sort4 anddescending4, preserve every old arguments success forreverse4/rotate4; finalall7041strictcerts; allseedsreported. No manual body/network/downstreamchampion seeding.'}
    save(OUT/'protocol.json',p);previous=json.loads((FIRST/'old_artifact_hashes.json').read_text())
    for file in FIRST.rglob('*'):
        if file.is_file() and file.suffix!='.pyc' and file.name!='.DS_Store' and not any('cache' in part.lower() for part in file.parts):previous[str(file)]=sha(file)
    save(OUT/'old_artifact_hashes.json',previous);return p

def run():
    p=json.loads((OUT/'protocol.json').read_text()) if (OUT/'protocol.json').exists() else prepare()
    for file,h in p['source_sha256'].items():assert sha(file)==h
    for file,h in p['libraries'].items():assert sha(file)==h
    assert sha(FIRST/'results.json')==p['first_series_results_sha256'];reference=json.loads((FIRST/'results.json').read_text());qualification=json.loads((ROOT/'artifacts/codepy_block_contracts_20261030/preparation_validation.json').read_text());rows=[];ledgers={}
    for seed in range(5):
        oldlib=json.loads((OLD/f'library_seed{seed}.json').read_text());newlib=json.loads((FIRST/f'library_seed{seed}.json').read_text());blocks=newlib['blocks'];codes,layouts=tables(blocks);table,lengths=s.table_for(oldlib['blocks']);entries=[[v] for v in s.ALPHABET]+oldlib['blocks'];contract_tables=baseline.calls.c.contract_tables(entries)
        common=next(q['primitive_case_steps'] for q in qualification['rows'] if q['seed']==seed);ledger=budget_ledger(sum(a['evaluations'] for a in oldlib['sources'])*64*36,oldlib['mining']['ablation_evaluations']*64*36,common,newlib['preparation_case_steps_used']);assert ledger['within_original_total_budget'];ledgers[str(seed)]=ledger
        for ti,task in enumerate(p['tasks']):
            path=OUT/f'seed{seed}_{task}.json'
            if path.exists():rows.append(json.loads(path.read_text()));continue
            print('PORTFOLIO',seed,task,flush=True);x=np.random.default_rng(11000+seed*10+ti).integers(-2,3,(64,4)).astype(np.float32);branches={};ref=next(r for r in reference['runs'] if r['seed']==seed and r['task']==task)
            count=50 if blocks else 100;q=arguments_prefix(seed,x,table,lengths,contract_tables,task,count);assert q['trace']==ref['arms']['baseline']['trace'][:count];q['search_kind']='arguments';branches['arguments']=q
            if blocks:
                @jax.jit
                def scorer(g,inputs):return s.losses(compose(g,inputs,codes,layouts),inputs,task)
                start=time.perf_counter();q=vm.search(seed,x,alphabet=list(range(1+12*len(blocks))),length=6,population=256,generations=49,scorer=scorer);q['compilation_and_warmup_s']=max(0.,time.perf_counter()-start-q['elapsed_s']);q['search_kind']='behavioral';assert q['trace']==ref['arms']['behavioral']['trace'][:50];branches['behavioral']=q
            chosen=choose_training_program(branches)
            # Certificates are computed only after the TRAIN-only choice.
            for name,q in branches.items():
                q['certificate']=baseline.certify(q,vm.cases(4,16000+ti),entries,table,lengths,contract_tables,task) if name=='arguments' else certify(q['program'],task,blocks)
            assert sum(q['evaluations'] for q in branches.values())==25600
            assert len({q['training_sha256'] for q in branches.values()})==1
            row={'seed':seed,'task':task,'chosen_by_TRAIN':chosen,'branches':branches,'evaluations':25600,'certificate':branches[chosen]['certificate'],'baseline_certificate':ref['arms']['baseline']['certificate'],'training_sha256':branches[chosen]['training_sha256']};save(path,row);rows.append(row)
            print('RESULT',seed,task,chosen,row['certificate']['passed'],flush=True)
            for file,h in p['libraries'].items():assert sha(file)==h
    summary={task:{'baseline':sum(r['baseline_certificate']['passed'] for r in rows if r['task']==task),'portfolio':sum(r['certificate']['passed'] for r in rows if r['task']==task)} for task in p['tasks']}
    guards=[{'seed':r['seed'],'task':r['task'],'baseline_passed':r['baseline_certificate']['passed'],'new_passed':r['certificate']['passed'],'preserved':not r['baseline_certificate']['passed'] or r['certificate']['passed']} for r in rows if r['task'] in ['reverse4','rotate4']]
    success=summary['sort4']['portfolio']>0 and summary['descending4']['portfolio']>0 and all(g['preserved'] for g in guards)
    save(OUT/'results.json',{'protocol':p,'runs':rows,'summary':summary,'preservation_guards':guards,'budget_ledgers':ledgers,'success':success,'candidate_status':'accepted' if success else 'rejected'});print(json.dumps({'summary':summary,'success':success},indent=2),flush=True)
if __name__=='__main__':run()
