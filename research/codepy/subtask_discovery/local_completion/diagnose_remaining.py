"""Missing family and full one-command neighborhood, no supplied replacement body."""
from context import ROOT,BASE,OUT,prior,legacy,previous
from branch_slicing import candidate_pool,condition_mask
from guided_mining import evaluate,encode
import json,numpy as np

def neighbors(code,layout):
    rows=[]
    for pos in range(12):
        for op in legacy.ALPHABET:
            if op==code[pos]:continue
            c=code.copy();c[pos]=op;rows.append({'code':c,'layout':layout,'edit':[pos,op]})
    return rows

def main():
    seed=4;source=json.loads((previous.OLD/f'library_seed{seed}.json').read_text());pool=candidate_pool(source['sources']);r=json.loads((BASE/f'library_seed{seed}.json').read_text());best=r['trace'][-1];rows=neighbors(best['best_code'],best['best_layout']);code,layout=encode(rows);values,out=evaluate(code,layout);values=np.asarray(values);out=np.asarray(out);scores=values[:,6:].min(-1);i=int(scores.argmin())
    result={'seed4':{'source_candidates':len(pool),'live_input_condition_candidates':sum(q['live_condition'] for q in pool),'relative_input_conditions':sum(q['branch_strength']==2 for q in pool),'best_old_score':best['best_score'],'best_old_code':best['best_code'],'best_layout':best['best_layout'],'best_live_condition_mask':list(condition_mask(best['best_code'],best['best_layout'])),'single_edit_candidates':len(rows),'best_single_edit_score':float(scores[i]),'best_single_edit':rows[i],'exact_single_edit_blocks':int((scores==0).sum())},'rotate_seed1':{'pair_behaviors':[b['behavior'] for b in json.loads((BASE/'library_seed1.json').read_text())['incumbent_blocks']+json.loads((BASE/'library_seed1.json').read_text())['blocks']],'reason':'Every available nonidentity pair block sorts two values, collapsing the two input orders. Any nonempty chain is noninjective; rotation must be injective. Thus pair branch cannot solve arbitrary rotation, regardless of more generations. No unconditional permutation block in this seed.'},'scope':'Diagnostic one-edit probe is not a qualified library, never seeded as a target champion. Missing-family completion will be registered and charged independently.'}
    prior.save(OUT/'remaining_diagnosis.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
