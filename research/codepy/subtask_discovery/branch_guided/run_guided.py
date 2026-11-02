"""Registered train-only selection with an extra mining pass paid by target savings."""
from common import ROOT,OUT,previous,legacy
from guided_mining import mine
from register_abi import tables,compose,numpy_compose
import json,time
import numpy as np
import jax,jax.numpy as jnp
sha,save,s,vm,baseline=previous.sha,previous.save,previous.s,previous.vm,previous.baseline
HERE=__import__('pathlib').Path(__file__).resolve().parent

def certify(program,task,blocks):
    code,layout,returns=tables(blocks);groups={}
    for name,x in vm.cases(4,16000+s.TARGETS.index(task)).items():
        native=np.asarray(compose(jnp.asarray([program]),jnp.asarray(x),code,layout,returns))[0];independent=numpy_compose(program,x,blocks);np.testing.assert_array_equal(native,independent);target=np.asarray(s.target(jnp.asarray(x),task));exact=(native==target).all(-1);pres=(np.sort(native,-1)==np.sort(x,-1)).all(-1)
        groups[name]={'cases':len(x),'exact':int(exact.sum()),'preserved':int(pres.sum()),'mse':float(np.mean((native.astype(np.float64)-target)**2)),'max_abs_error':float(np.max(np.abs(native.astype(np.float64)-target)))}
    return {'passed':all(q['cases']==q['exact']==q['preserved'] for q in groups.values()),'cases':sum(q['cases'] for q in groups.values()),'groups':groups,'independent_agrees':True}

def prepare():
    OUT.mkdir(exist_ok=True);old=json.loads((previous.OUT/'protocol.json').read_text());files={**old['source_sha256'],**{str(f):sha(f) for f in HERE.glob('*.py')}}
    for f,h in files.items():assert sha(f)==h
    p={'version':'codepy-branch-guided-totalbudget-v1','seeds':list(range(5)),'tasks':list(s.TARGETS),'source_sha256':files,'baseline_results_sha256':sha(previous.OUT/'results.json'),'budget':'Current accepted pipeline is baseline. Extra mining <= old ablation cap <= 4718592 case steps, paid by reducing each of four target searches from 25600 to 25344 candidates, 64 cases, 72 steps. All prior source and library mining costs remain charged in both totals. Static analysis/JIT/independent final certification excluded as before; no FLOP or walltime parity claim.','allocation':'99 evaluated generations: arguments49 + incumbent25 + neworder25 when both exist; arguments49 + solepair50 when one exists; arguments99 if neither. Ready incumbent order counts as sole incumbent. No task-specific allocation.','selection':'Minimum TRAIN objective; ties prefer arguments then incumbent then neworder; selection before certification. Fresh populations, no target champion reuse.','mining':'Same-seed source register-output backward slices and live-branch windows <=12 commands, 25% max source probes. Two-order normalized losses and preservation. Functional diversity quotas, protect useful source-dependent predicate instructions in half offspring; metadata inherited coherently. Register outputs allowed, all state zeroed. No opcode35 or supplied comparator body.','qualification':'25 discovery pairs, 64 challenge and6429 promotion pairs native and NumPy, all mining VM evaluations charged. Final7041 cases per target never guide search.','acceptance':'sort4>1/5 and descending4>1/5; preserve ALL previously successful seed/task results including reverse5/5 rotate4/5.'}
    save(OUT/'protocol.json',p);hashes=json.loads((previous.OUT/'old_artifact_hashes.json').read_text())
    for folder in [previous.OUT,previous.FIRST]:
        for f in folder.rglob('*'):
            if f.is_file() and f.suffix!='.pyc' and f.name!='.DS_Store' and not any('cache' in part.lower() for part in f.parts):hashes[str(f)]=sha(f)
    save(OUT/'old_artifact_hashes.json',hashes);return p

def run():
    p=json.loads((OUT/'protocol.json').read_text()) if (OUT/'protocol.json').exists() else prepare()
    for f,h in p['source_sha256'].items():assert sha(f)==h,f
    assert sha(previous.OUT/'results.json')==p['baseline_results_sha256'];old=json.loads((previous.OUT/'results.json').read_text());libraries={};ledgers={}
    # Freeze every library before any target search.
    for seed in p['seeds']:
        source=json.loads((previous.OLD/f'library_seed{seed}.json').read_text());inc=json.loads((previous.FIRST/f'library_seed{seed}.json').read_text());path=OUT/f'library_seed{seed}.json'
        if path.exists():r=json.loads(path.read_text())
        else:r=mine(source,inc,seed);r['incumbent_blocks']=inc['blocks'];save(path,r)
        libraries[seed]=r;before=old['budget_ledgers'][str(seed)]['new_total_units'];used=r['preparation_case_steps_used'];saved=4*256*64*72;assert used<=r['preparation_case_step_budget']<=saved
        ledgers[str(seed)]={'baseline_total_units':before,'saved_target_units':saved,'extra_mining_units':used,'new_total_units':before-saved+used,'within_baseline':used<=saved};print('LIBRARY',seed,r['status'],used,flush=True)
    manifest={str(OUT/f'library_seed{i}.json'):sha(OUT/f'library_seed{i}.json') for i in p['seeds']};save(OUT/'library_manifest.json',manifest);rows=[]
    for seed in p['seeds']:
        source=json.loads((previous.OLD/f'library_seed{seed}.json').read_text());inc=libraries[seed]['incumbent_blocks'];new=libraries[seed]['blocks'];table,lengths=s.table_for(source['blocks']);entries=[[v] for v in s.ALPHABET]+source['blocks'];contract=baseline.calls.c.contract_tables(entries)
        for ti,task in enumerate(p['tasks']):
            path=OUT/f'seed{seed}_{task}.json'
            if path.exists():rows.append(json.loads(path.read_text()));continue
            ref=next(r for r in old['runs'] if r['seed']==seed and r['task']==task);x=np.random.default_rng(11000+seed*10+ti).integers(-2,3,(64,4)).astype(np.float32);branches={};count=49 if inc or new else 99;print('SEARCH',seed,task,flush=True)
            q=previous.arguments_prefix(seed,x,table,lengths,contract,task,count);assert q['trace']==ref['branches']['arguments']['trace'][:count];q.update(search_kind='arguments');branches['arguments']=q
            specs=([('incumbent',inc,25 if new else 50)] if inc else [])+([('neworder',new,25 if inc else 50)] if new else [])
            for name,blocks,count in specs:
                codes,layouts,returns=tables(blocks)
                @jax.jit
                def scorer(g,inputs):return s.losses(compose(g,inputs,codes,layouts,returns),inputs,task)
                start=time.perf_counter();q=vm.search(seed,x,alphabet=list(range(1+12*len(blocks))),length=6,population=256,generations=count-1,scorer=scorer);q['compilation_and_warmup_s']=max(0.,time.perf_counter()-start-q['elapsed_s']);q.update(search_kind='pair',blocks=blocks)
                if name=='incumbent':assert q['trace']==ref['branches']['behavioral']['trace'][:count],(seed,task,'incumbent ABI mismatch')
                branches[name]=q
            chosen=previous.choose_training_program(branches)
            for name,q in branches.items():q['certificate']=baseline.certify(q,vm.cases(4,16000+ti),entries,table,lengths,contract,task) if name=='arguments' else certify(q['program'],task,q['blocks'])
            assert sum(q['evaluations'] for q in branches.values())==25344;assert len({q['training_sha256'] for q in branches.values()})==1
            row={'seed':seed,'task':task,'chosen_by_TRAIN':chosen,'branches':branches,'evaluations':25344,'certificate':branches[chosen]['certificate'],'baseline_certificate':ref['certificate'],'training_sha256':branches[chosen]['training_sha256']};save(path,row);rows.append(row);print('RESULT',seed,task,chosen,row['certificate']['passed'],flush=True)
    summary={task:{'baseline':sum(r['baseline_certificate']['passed'] for r in rows if r['task']==task),'guided':sum(r['certificate']['passed'] for r in rows if r['task']==task)} for task in p['tasks']};guards=[{'seed':r['seed'],'task':r['task'],'preserved':not r['baseline_certificate']['passed'] or r['certificate']['passed']} for r in rows];success=summary['sort4']['guided']>1 and summary['descending4']['guided']>1 and all(g['preserved'] for g in guards)
    for f,h in manifest.items():assert sha(f)==h
    save(OUT/'results.json',{'protocol':p,'summary':summary,'runs':rows,'budget_ledgers':ledgers,'preservation_guards':guards,'success':success,'candidate_status':'accepted' if success else 'rejected'});print(json.dumps({'summary':summary,'success':success},indent=2),flush=True)
if __name__=='__main__':run()
