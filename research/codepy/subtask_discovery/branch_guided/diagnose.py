"""Replay prior mining unchanged; record diversity and live branches of old elites."""
import json,hashlib,inspect
import numpy as np
from common import ROOT,OUT,legacy
from branch_slicing import condition_mask
original=legacy.evaluate
rows=[];current_seed=None

def evaluated(code,layout):
    value,out=original(code,layout);scores=np.asarray(value);outputs=np.asarray(out);frame=inspect.currentframe().f_back.f_locals
    if frame.get('name')=='order' and 'generation' in frame and scores.shape[0]==256:
        ids=np.argsort(scores[:,1],kind='stable')[:16];codes=np.asarray(code);layouts=np.asarray(layout);mask=np.asarray(legacy.PROBE[:,0]<legacy.PROBE[:,1]);rev=np.asarray(legacy.PROBE[:,0]>legacy.PROBE[:,1]);idx=int(ids[0]);z=outputs[idx];pres=(np.sort(z,-1)==np.sort(legacy.PROBE,-1)).all(-1)
        rows.append({'seed':current_seed,'generation':int(frame['generation']),'best_score':float(scores[idx,1]),'best_code':codes[idx].tolist(),'best_layout':layouts[idx].tolist(),'elite_unique_output_behaviors':len({outputs[i].tobytes() for i in ids}),'elite_live_conditions':sum(any(condition_mask(codes[i].tolist(),layouts[i].tolist(),layouts[i].tolist())) for i in ids),'best_preserved_a_lt_b':int(pres[mask].sum()),'best_preserved_a_gt_b':int(pres[rev].sum())})
    return value,out
legacy.evaluate=evaluated
summary=[]
for seed in range(5):
    current_seed=seed;old=json.loads((ROOT/f'artifacts/codepy_behavioral_blocks_20261101/library_seed{seed}.json').read_text());source=json.loads((ROOT/f'artifacts/codepy_subtask_discovery_20261030/library_seed{seed}.json').read_text());replay=legacy.mine(source,seed)
    assert replay['repair_log']==old['repair_log'];assert [(b['code'],b['layout'],b['behavior']) for b in replay['blocks']]==[(b['code'],b['layout'],b['behavior']) for b in old['blocks']]
    curve=[r for r in rows if r['seed']==seed];summary.append({'seed':seed,'old_source_pool':old['source_pool_count'],'source_best_order_score':min((p['scores'][1] for p in old['probe_records']),default=None),'order_found':any(b['behavior'].startswith('order') for b in old['blocks']),'order_rounds':len(curve),'first':curve[0] if curve else None,'last':curve[-1] if curve else None,'trace_reproduced':True});print('OLD_REPLAY',seed,summary[-1],flush=True)
# Explicitly show the old scalar's tied degenerate behaviors on the pair grid.
x=legacy.PROBE;examples={'identity':x,'unconditional_swap':x[:,::-1],'constant_zero':np.zeros_like(x)}
plateau={name:float(np.asarray(legacy.scores(z[None],x))[0,1]) for name,z in examples.items()}
(OUT/'old_trajectory_diagnosis.json').write_text(json.dumps({'seeds':summary,'old_scalar_degenerate_scores':plateau,'generations':rows,'scope':'Unmodified oldminer replay,diagnostic only; no model from this replay is passed to newmining.'},indent=2)+'\n')
