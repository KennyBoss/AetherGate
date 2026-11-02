"""Fixtures become frozen paired arrays before any task-free synthesis begins."""
import argparse,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import jax
from research.codepy.ssm_dispatch.catalog import ROOT,TASKS
from research.codepy.ssm_dispatch.pipeline import Pipeline
from research.codepy.ssm_dispatch.examples import ExamplePairs,BlockSearch
from research.codepy.ssm_dispatch.programs import Candidate
from research.codepy.ssm_dispatch.exam import RECORDS,target

# Evaluation-only data generators; neither these functions nor these labels enter the search.
PATTERNS=(((0,1,False),(2,3,False)),((0,1,True),(2,3,True)),((0,2,False),(1,3,False)),((0,3,False),(1,2,True)))

def fixture_targets(index,x):
    y=x.copy()
    for a,b,descending in PATTERNS[index]:
        swap=(x[:,a]<x[:,b]) if descending else (x[:,a]>x[:,b])
        y[:,a]=np.where(swap,x[:,b],x[:,a]);y[:,b]=np.where(swap,x[:,a],x[:,b])
    return y

def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def fingerprint(p,weights=False):
    h=hashlib.sha256()
    if weights:state=(p.params,p.text_params,p.numeric_params)
    else:
        h.update(json.dumps({'programs':p.programs,'tasks':p.task_bindings,'last':p.last_text_token,'records':{k:{a:b for a,b in r.items() if a!='chunks'} for k,r in p.records.items()}},sort_keys=True).encode())
        state=(p.semantic,p.text_base,p.text_carry,[r['chunks'] for r in p.records.values()])
    for array in jax.tree_util.tree_leaves(state):h.update(np.asarray(array).tobytes())
    return h.hexdigest()

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Use a fresh directory')
    base=Pipeline(0).catalog;datasets={}
    for index in range(4):
        h=np.concatenate(list(base.g.vm.cases(4,11012026+index).values()))
        values={'X_holdout':h,'Y_holdout':fixture_targets(index,h)}
        for seed in range(5):
            x=np.random.default_rng(710000+100*index+seed).uniform(-3,3,(64,4)).astype(np.float32)
            values[f'X_train_{seed}']=x;values[f'Y_train_{seed}']=fixture_targets(index,x)
        path=out/f'pairs_{index}.npz';np.savez_compressed(path,**values)
        datasets[str(index)]={'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    protocol={'version':'universal-example-loop-v1','seeds':list(range(5)),'opaque_labels':[f'packet-{i}' for i in range(4)],'datasets':datasets,'populations_per_restart':95,'max_train_only_restarts':4,
              'selection':'One candidate reaches heldout; restarts depend only on TRAIN fit. No retry after failed certification. All20reported.',
              'fixtures_are_not_runtime_oracles':True,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('examples.py'),Path(__file__).with_name('pipeline.py'),Path(__file__).with_name('programs.py')]}}
    save(out/'protocol.json',protocol)
    rows=[];baseline=[];rejects=[];sessions=[];noise=(ROOT/'artifacts/russian_memory_v5/unique_corpus/text.txt').read_text()[:20000]
    for seed in range(5):
        p=Pipeline(seed);weights=fingerprint(p,True);packs=[]
        for index in range(4):
            with np.load(datasets[str(index)]['file']) as z:
                x=z[f'X_train_{seed}'];y=z[f'Y_train_{seed}'];h=z['X_holdout'];hy=z['Y_holdout']
            witnesses=[]
            for c in range(96):
                failed=np.flatnonzero(np.any(p.catalog.execute(c,x)!=y,axis=1))
                assert len(failed),(seed,index,c,'Fixture already solved by a base command')
                witnesses.append({'command':c,'counterexample_row':int(failed[0])})
            entity,prop,handle,question=RECORDS[index]
            result=p.develop(task=f'packet-{index}',X_train=x,Y_train=y,X_holdout=h,Y_holdout=hy,entity=entity,property=prop,handle=handle,values=h)
            row={'seed':seed,'packet':index,'status':result.status,'command':result.command,'evidence':result.evidence,'base_counterexamples':witnesses,'question':question}
            rows.append(row);packs.append((x,y,h,hy));save(out/'search_progress.json',rows)
            print('DEVELOP',seed,index,result.status,'base_matches',result.evidence['base_commands_matching_training'],flush=True)
        if any(r['status']!='accepted' for r in rows if r['seed']==seed):continue
        p.observe(noise)
        for index,(_,_,h,hy) in enumerate(packs):
            result=p.execute(RECORDS[index][3]);np.testing.assert_array_equal(result.output,hy)
            np.testing.assert_array_equal(result.arguments.view(np.uint32),h.view(np.uint32))
            row=next(r for r in rows if r['seed']==seed and r['packet']==index)
            row.update(executed=True,execution_cases=len(h),exact=len(h),jax_numpy_exact=result.independent_agrees,noise_v5_tokens=len(p.encoder.encode(noise)))
            print('EXECUTE',seed,index,len(h),'/',len(h),flush=True)
        bad_x=packs[0][1];bad_h=packs[0][2];bad_hy=packs[0][3];before=fingerprint(p)
        with patch.object(BlockSearch,'synthesize',return_value=(Candidate(seed,'negative-probe',(0,)*6),{'training_objective':0.})):
            rejected=p.develop(task='negative-probe',X_train=bad_x,Y_train=bad_x,X_holdout=bad_h,Y_holdout=bad_hy,entity='клары',property='основная книга',handle='морской атлас',values=bad_x)
        assert rejected.status=='rejected' and fingerprint(p)==before
        rejects.append({'seed':seed,'status':rejected.status,'session_unchanged':True,'train_perfect':True,'reason':rejected.evidence['reason']})
        for i,task in enumerate(TASKS):
            x=np.concatenate(list(p.catalog.g.vm.cases(4,16000+i).values()))
            a=p._execute_command(i*24,x);b=p._execute_command(i*24,x,independent=True)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,target(x,task))
            baseline.append({'seed':seed,'task':task,'cases':len(x),'passed':True})
        session=ROOT/f'.ci-output/example_development_seed{seed}.session.npz';p.save(session);q=Pipeline.load(session,seed=seed)
        for index,(_,_,h,hy) in enumerate(packs):np.testing.assert_array_equal(q.execute(RECORDS[index][3]).output,hy)
        assert fingerprint(p,True)==weights==fingerprint(q,True)
        assert 'research.codepy.subtask_discovery.transfer.tasks' not in sys.modules
        assert 'tasks' not in sys.modules
        sessions.append({'seed':seed,'file':str(session),'after_reload_executions':4,'weights_unchanged':True,'named_tasks_module_loaded':False})
        save(out/'progress.json',{'new':rows,'baseline':baseline,'rejections':rejects,'sessions':sessions})
    result={'passed':len(rows)==20 and all(r['status']=='accepted' and r.get('executed') and r['evidence']['base_commands_matching_training']==0 for r in rows) and len(baseline)==20,
            'new_programs':rows,'baseline':baseline,'rejections':rejects,'sessions':sessions,'accepted':sum(r['status']=='accepted' for r in rows),'new_execution_cases':sum(r.get('execution_cases',0) for r in rows),
            'baseline_successes':len(baseline),'named_tasks_module_loaded':False,'new_neural_training_updates':0,'certification_failure_retries':0}
    save(out/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['new_programs','baseline','rejections','sessions']},indent=2),flush=True)
    assert result['passed']

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args().output_dir)
