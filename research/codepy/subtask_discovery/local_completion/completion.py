"""Complete missing behavioral families with source windows and bounded edit beams."""
from context import legacy,previous
from guided_mining import encode,qualify
from register_abi import pair_outputs
from branch_slicing import candidate_pool
from progress import metrics
import itertools,time,hashlib
import numpy as np
import jax,jax.numpy as jnp
ALPHABET=tuple(legacy.ALPHABET)

def key(row):return tuple(row['code']),tuple(row['layout'])

def source_windows(sources):
    rows={}
    for si,source in enumerate(sources):
        for pi,p in enumerate(source['archive']):
            p=[(i,op) for i,op in enumerate(p) if op]
            for n in range(2,13):
                for start in range(len(p)-n+1):
                    part=p[start:start+n];code=[op for _,op in part]
                    for layout in ([0,1],[0,2],[1,2]):
                        r={'code':code,'layout':layout,'outputs':[3,4],'provenance':{'method':'unfiltered_source_window','source':si,'program':pi,'indices':[i for i,_ in part]}};rows.setdefault(key(r),r)
    return sorted(rows.values(),key=lambda r:(len(r['code']),r['code'],r['layout']))

def edits(row):
    code=(row['code']+[0]*12)[:12];layout=row['layout'];seen=set()
    def emit(c,l):
        k=(tuple(c),tuple(l))
        if k in seen or k==(tuple(code),tuple(layout)):return None
        seen.add(k);return {'code':list(c),'layout':list(l),'outputs':[3,4],'provenance':{'method':'bounded_edit_beam'}}
    for i in range(12):
        for op in ALPHABET:
            r=emit(code[:i]+[op]+code[i+1:],layout)
            if r:yield r
        # Deletion and insertion move an existing useful branch skeleton intact.
        r=emit(code[:i]+code[i+1:]+[0],layout)
        if r:yield r
        if code[-1]==0:
            for op in ALPHABET:
                r=emit(code[:i]+[op]+code[i:-1],layout)
                if r:yield r
    for l in ([0,1],[0,2],[1,2]):
        r=emit(code,l)
        if r:yield r

@jax.jit
def evaluate(codes,layouts):
    returns=jnp.tile(jnp.array([[3,4]],jnp.int32),(len(codes),1));out=pair_outputs(codes,layouts,returns,jnp.asarray(legacy.PROBE));m=metrics(out);scale=jnp.maximum(jnp.abs(jnp.asarray(legacy.PROBE)[:,0]-jnp.asarray(legacy.PROBE)[:,1]),1.)**2
    swap=jnp.mean(jnp.mean((out-jnp.asarray(legacy.PROBE)[None,:,::-1])**2,-1)/scale,-1)+2*m[:,4]
    return jnp.concatenate([m,swap[:,None]],-1),out

def score(values,family):return values[:,8] if family=='swap' else values[:,6:8].min(-1)

def beam_select(rows,values,outputs,family,width=16):
    # Half distinct behavior, half distinct syntax, allowing neutral structural bridges.
    order=np.argsort(score(values,family),kind='stable');chosen=[];seen=set()
    for i in order:
        signature=np.asarray(outputs[i]).tobytes()
        if signature not in seen:chosen.append(int(i));seen.add(signature)
        if len(chosen)>=width//2:break
    for i in order:
        if int(i) not in chosen:chosen.append(int(i))
        if len(chosen)==min(width,len(rows)):break
    return [rows[i] for i in chosen]

def complete(source,prior_library,seed,cap):
    start=time.perf_counter();inc=prior_library['incumbent_blocks']+prior_library['blocks'];missing=[f for f in ['swap','order'] if not any(legacy.family(b['behavior'])==f for b in inc)];budget=legacy.Budget(cap);found=[];logs=[];attempted=set()
    def inspect(rows,family,reason):
        if not rows:return None
        assert budget.spend(len(rows)*25*12,reason)
        code,layout=encode(rows);values,outputs=evaluate(code,layout);values=np.asarray(values);outputs=np.asarray(outputs)
        for i in np.flatnonzero(score(values,family)==0):
            row=rows[int(i)];kind=legacy.classify(outputs[i],legacy.PROBE)
            if legacy.family(kind)!=family or key(row) in attempted:continue
            attempted.add(key(row));proof=qualify(row,kind,seed,budget)
            if proof:
                block={**row,'behavior':kind,'qualification':proof};found.append(block);return block
        return values,outputs
    # The same missing-family rule applies to every seed; no target task is inspected.
    for family in missing:
        reserved=2*12*(64+6429);evaluated=set();rows=[]
        if family=='swap':rows=source_windows(source['sources'])[:2048]
        else:
            rows=[{'code':q['best_code'],'layout':q['best_layout'],'outputs':[3,4],'provenance':{'method':'prior_source_repair_partial','generation':q['generation']}} for q in prior_library['trace']]
            rows+=candidate_pool(source['sources'])[:256]
        rows=list({key(r):r for r in rows}.values());initial=[];vv=[];oo=[];promoted=False
        for first in range(0,len(rows),128):
            batch=rows[first:first+128]
            if budget.used+len(batch)*300+reserved>budget.cap:break
            result=inspect(batch,family,'missing_family_source_probes');evaluated.update(map(key,batch))
            if isinstance(result,dict):promoted=True;break
            v,o=result;initial.extend(batch);vv.append(v);oo.append(o)
        if promoted:continue
        if not initial:break
        beam=beam_select(initial,np.concatenate(vv),np.concatenate(oo),family);round_=0
        while budget.cap-budget.used>=300+reserved:
            candidates=[]
            for parent in beam:
                for row in edits(parent):
                    k=key(row)
                    if k not in evaluated:evaluated.add(k);candidates.append(row)
            if not candidates:break
            limit=(budget.cap-budget.used-reserved)//300;candidates=candidates[:limit];next_rows=[];vv=[];oo=[]
            for first in range(0,len(candidates),256):
                batch=candidates[first:first+256];result=inspect(batch,family,'local_edit_beam')
                if isinstance(result,dict):promoted=True;break
                v,o=result;next_rows.extend(batch);vv.append(v);oo.append(o)
            if promoted:logs.append({'family':family,'round':round_,'promoted':True,'budget_used':budget.used});break
            if not next_rows:break
            values=np.concatenate(vv);outputs=np.concatenate(oo);beam=beam_select(next_rows,values,outputs,family);logs.append({'family':family,'round':round_,'candidates':len(next_rows),'best_score':float(score(values,family).min()),'budget_used':budget.used});round_+=1
    return {'seed':seed,'missing_before':missing,'blocks':found,'status':'complete' if len(found)==len(missing) else 'incomplete','preparation_case_step_budget':budget.cap,'preparation_case_steps_used':budget.used,'budget_charges':budget.charges,'trace':logs,'elapsed_s':time.perf_counter()-start,'source_only_no_target_inputs':True}
