"""Replay saved new and baseline winners, requalify blocks and check frozen ledgers."""
from common import ROOT,OUT,previous,legacy
from run_guided import certify,sha,save,s,vm,baseline
from guided_mining import qualify
import json,hashlib
from pathlib import Path
import numpy as np

def replay(row,old=False):
    seed,task=row['seed'],row['task'];name=row['chosen_by_TRAIN'];q=row['branches'][name]
    if name=='arguments':
        lib=json.loads((previous.OLD/f'library_seed{seed}.json').read_text());entries=[[v] for v in s.ALPHABET]+lib['blocks'];table,lengths=s.table_for(lib['blocks']);contract=baseline.calls.c.contract_tables(entries);fresh=baseline.certify(q,vm.cases(4,16000+s.TARGETS.index(task)),entries,table,lengths,contract,task)
    elif old:fresh=previous.certify(q['program'],task,json.loads((previous.FIRST/f'library_seed{seed}.json').read_text())['blocks'])
    else:fresh=certify(q['program'],task,q['blocks'])
    assert fresh==row['certificate']==q['certificate'];assert fresh['cases']==7041 and fresh['independent_agrees'];return fresh

def main():
    r=json.loads((OUT/'results.json').read_text());old=json.loads((previous.OUT/'results.json').read_text());p=r['protocol'];programs=[];blocks=[];exports=[]
    for f,h in p['source_sha256'].items():assert sha(f)==h,f
    assert sha(previous.OUT/'results.json')==p['baseline_results_sha256']
    for f,h in json.loads((OUT/'library_manifest.json').read_text()).items():assert sha(f)==h
    for seed in range(5):
        lib=json.loads((OUT/f'library_seed{seed}.json').read_text());assert sum(q['units'] for q in lib.get('budget_charges',[]))==lib['preparation_case_steps_used']<=lib['preparation_case_step_budget']
        for b in lib['blocks']:
            assert len(b['code'])<=12 and set(b['code'])<=set(legacy.ALPHABET) and 35 not in b['code'];assert b['outputs']==[3,4];assert b['layout'] in [[0,1],[0,2],[1,2]]
            proof=qualify(b,b['behavior'],seed,legacy.Budget(2*12*(64+6429)));assert proof==b['qualification'];blocks.append({'seed':seed,'behavior':b['behavior'],'code':b['code'],'layout':b['layout'],'outputs':b['outputs'],'qualification_replayed':True})
        ledger=r['budget_ledgers'][str(seed)];before=old['budget_ledgers'][str(seed)]['new_total_units'];assert ledger['baseline_total_units']==before and ledger['saved_target_units']==4718592;assert ledger['extra_mining_units']==lib['preparation_case_steps_used'];assert ledger['new_total_units']==before-4718592+lib['preparation_case_steps_used']<=before
    for row in r['runs']:
        seed,task=row['seed'],row['task'];ref=next(q for q in old['runs'] if q['seed']==seed and q['task']==task);assert row['chosen_by_TRAIN']==previous.choose_training_program(row['branches']);assert sum(q['evaluations'] for q in row['branches'].values())==row['evaluations']==25344
        x=np.random.default_rng(11000+seed*10+s.TARGETS.index(task)).integers(-2,3,(64,4)).astype(np.float32);digest=hashlib.sha256(x.tobytes()).hexdigest();assert digest==row['training_sha256']==ref['training_sha256']
        for name,q in row['branches'].items():
            assert q['training_sha256']==digest
            if name!='neworder':assert q['trace']==ref['branches']['arguments' if name=='arguments' else 'behavioral']['trace'][:q['evaluations']//256]
        fresh=replay(row);previous_fresh=replay(ref,True);assert row['baseline_certificate']==previous_fresh;programs.append({'seed':seed,'task':task,'new_passed':fresh['passed'],'baseline_passed':previous_fresh['passed'],'cases_each':7041,'native_numpy_exact':True})
        chosen=row['branches'][row['chosen_by_TRAIN']];exports.append({'seed':seed,'task':task,'kind':chosen['search_kind'],'program':chosen['program'],'blocks':chosen.get('blocks'),'argument_library':str(previous.OLD/f'library_seed{seed}.json') if chosen['search_kind']=='arguments' else None,'certificate':fresh})
    hashes=json.loads((OUT/'old_artifact_hashes.json').read_text());changed=[f for f,h in hashes.items() if not Path(f).is_file() or sha(f)!=h];assert not changed,changed
    result={'passed':True,'programs':programs,'blocks':blocks,'new_and_old_programs_replayed':40,'target_inputs_each':7041,'all_vm_case_steps_within_baseline':True,'all_old_successes_preserved':all(g['preserved'] for g in r['preservation_guards']),'old_files_checked':len(hashes),'old_files_changed':changed,'source_hashes_verified':True,'finite_float32_test_not_universal_proof':True}
    save(OUT/'verification.json',result);save(OUT/'exported_programs.json',exports);print(json.dumps({'passed':True,'old_files':len(hashes),'programs_replayed':40,'summary':r['summary']},indent=2))
if __name__=='__main__':main()
