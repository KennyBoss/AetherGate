"""One registered completion experiment, paid from the existing full-pipeline cap."""
from context import ROOT,BASE,OUT,prior,previous,legacy
from completion import complete
from register_abi import tables,compose
import json,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
save,sha,s,vm,baseline=prior.save,prior.sha,prior.s,prior.vm,prior.baseline
HERE=Path(__file__).resolve().parent

def prepare():
    old=json.loads((BASE/'results.json').read_text());files={**old['protocol']['source_sha256'],**json.loads((BASE/'final_source_manifest.json').read_text()),**{str(f):sha(f) for f in HERE.glob('*.py')}}
    for f,h in files.items():assert sha(f)==h,f
    p={'version':'codepy-missing-family-local-completion-v1','seeds':list(range(5)),'tasks':list(s.TARGETS),'source_sha256':files,'baseline_results_sha256':sha(BASE/'results.json'),'target_evaluations':95*256,'source_completion_cap':4*4*256*64*72,'allocation':'95evaluated populations per target:45arguments+50split evenly between existing library groups and missing-family completion. If no pair groups,95arguments. Generic inventory rule, no target-specific allocation.','completion':'Inspect own-seed existing swap/order inventory. Missing swap probes unfiltered original-source nonnoop windows sorted by length; missing order initializes from own prior partial source-repair trajectory and256source candidates. Then bounded16beam substitutions/deletion/insertion/layout edits, half behavioral and half syntactic diversity. All candidate and challenge/promotion evaluations charged. No target champion or other seed body used.','selection':'Minimum TRAIN objective before certification, stable tie order arguments then existing groups then completion. Fresh target populations and same64TRAIN cases.','budget':'Compared to branch_guided accepted actual per-seed total; all historical source/preparation costs retained. Four populations saved on each of four targets=18874368primitive-case steps pays all extra source completion. Final certification, diagnostics, static analysis/JIT excluded as in historical metric; no walltime/FLOP equality claim.','acceptance':'More than17strict seed/task successes, preserve every prior successful seed/task; all20selected programs7041native/independent certificates; makeci.'}
    save(OUT/'protocol.json',p);hashes=json.loads((BASE/'old_artifact_hashes.json').read_text())
    for f in BASE.rglob('*'):
        if f.is_file() and f.suffix!='.pyc' and f.name!='.DS_Store' and not any('cache' in part.lower() for part in f.parts):hashes[str(f)]=sha(f)
    save(OUT/'old_artifact_hashes.json',hashes);return p

def run():
    p=json.loads((OUT/'protocol.json').read_text()) if (OUT/'protocol.json').exists() else prepare()
    for f,h in p['source_sha256'].items():assert sha(f)==h,f
    assert sha(BASE/'results.json')==p['baseline_results_sha256'];old=json.loads((BASE/'results.json').read_text());libraries={};ledgers={}
    for seed in p['seeds']:
        path=OUT/f'library_seed{seed}.json';source=json.loads((previous.OLD/f'library_seed{seed}.json').read_text());inc=json.loads((BASE/f'library_seed{seed}.json').read_text())
        if path.exists():q=json.loads(path.read_text())
        else:q=complete(source,inc,seed,p['source_completion_cap']);save(path,q)
        libraries[seed]=q;before=old['budget_ledgers'][str(seed)]['new_total_units'];assert q['preparation_case_steps_used']<=p['source_completion_cap'];ledgers[str(seed)]={'baseline_total_units':before,'saved_target_units':p['source_completion_cap'],'extra_completion_units':q['preparation_case_steps_used'],'new_total_units':before-p['source_completion_cap']+q['preparation_case_steps_used']};print('LIBRARY',seed,q['status'],q['missing_before'],[b['behavior'] for b in q['blocks']],q['preparation_case_steps_used'],flush=True)
    manifest={str(OUT/f'library_seed{seed}.json'):sha(OUT/f'library_seed{seed}.json') for seed in p['seeds']};save(OUT/'library_manifest.json',manifest);rows=[]
    for seed in p['seeds']:
        source=json.loads((previous.OLD/f'library_seed{seed}.json').read_text());inc=json.loads((BASE/f'library_seed{seed}.json').read_text());groups={}
        if inc['incumbent_blocks']:groups['incumbent']=inc['incumbent_blocks']
        if inc['blocks']:groups['neworder']=inc['blocks']
        if libraries[seed]['blocks']:groups['completion']=libraries[seed]['blocks']
        entries=[[v] for v in s.ALPHABET]+source['blocks'];table,lengths=s.table_for(source['blocks']);contract=baseline.calls.c.contract_tables(entries)
        for ti,task in enumerate(p['tasks']):
            path=OUT/f'seed{seed}_{task}.json'
            if path.exists():rows.append(json.loads(path.read_text()));continue
            ref=next(q for q in old['runs'] if q['seed']==seed and q['task']==task);x=np.random.default_rng(11000+seed*10+ti).integers(-2,3,(64,4)).astype(np.float32);branches={};print('SEARCH',seed,task,flush=True)
            count=45 if groups else 95
            q=previous.arguments_prefix(seed,x,table,lengths,contract,task,count);assert q['trace']==ref['branches']['arguments']['trace'][:count];q['search_kind']='arguments';branches['arguments']=q
            for i,(name,blocks) in enumerate(groups.items()):
                count=50//len(groups)+(i<50%len(groups));codes,layouts,returns=tables(blocks)
                @jax.jit
                def scorer(g,inputs):return s.losses(compose(g,inputs,codes,layouts,returns),inputs,task)
                start=time.perf_counter();q=vm.search(seed,x,alphabet=list(range(1+12*len(blocks))),length=6,population=256,generations=count-1,scorer=scorer);q.update(search_kind='pair',blocks=blocks,compilation_and_warmup_s=max(0.,time.perf_counter()-start-q['elapsed_s']))
                if name!='completion':assert q['trace']==ref['branches'][name]['trace'][:count],(seed,task,name)
                branches[name]=q
            chosen=previous.choose_training_program(branches)
            for name,q in branches.items():q['certificate']=baseline.certify(q,vm.cases(4,16000+ti),entries,table,lengths,contract,task) if name=='arguments' else prior.certify(q['program'],task,q['blocks'])
            assert sum(q['evaluations'] for q in branches.values())==p['target_evaluations'];assert len({q['training_sha256'] for q in branches.values()})==1
            row={'seed':seed,'task':task,'chosen_by_TRAIN':chosen,'branches':branches,'evaluations':p['target_evaluations'],'certificate':branches[chosen]['certificate'],'baseline_certificate':ref['certificate'],'training_sha256':branches[chosen]['training_sha256']};save(path,row);rows.append(row);print('RESULT',seed,task,chosen,row['certificate']['passed'],flush=True)
    summary={t:{'baseline':sum(q['baseline_certificate']['passed'] for q in rows if q['task']==t),'completion':sum(q['certificate']['passed'] for q in rows if q['task']==t)} for t in p['tasks']};guards=[{'seed':q['seed'],'task':q['task'],'preserved':not q['baseline_certificate']['passed'] or q['certificate']['passed']} for q in rows];wins=sum(q['certificate']['passed'] for q in rows);success=wins>17 and all(q['preserved'] for q in guards)
    for f,h in manifest.items():assert sha(f)==h
    save(OUT/'results.json',{'protocol':p,'summary':summary,'runs':rows,'budget_ledgers':ledgers,'preservation_guards':guards,'wins':wins,'success':success,'candidate_status':'accepted' if success else 'rejected'});print(json.dumps({'summary':summary,'wins':wins,'success':success},indent=2),flush=True)
if __name__=='__main__':run()
