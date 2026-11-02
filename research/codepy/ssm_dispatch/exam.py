"""Frozen-model end-to-end exam; never calls learning or synthesis routines."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from research.codepy.ssm_dispatch.pipeline import Pipeline, V5, ROOT, UnknownKey
from research.codepy.ssm_dispatch.catalog import TASKS, catalog

# Literal records. Task meanings are assigned only by the remembered numeric ID.
RECORDS = (
 ('клары','основная книга','морской атлас','Что именно относится к Кларе: главное издание? Ответ:'),
 ('алисы','любимый напиток','облепиховый чай','Алиса: назови предпочитаемое питьё. Ответ:'),
 ('тимура','рабочий язык','испанский','Что у тимура указано как рабочий язык? Ответ:'),
 ('анны','кодовое слово','янтарь','Какое кодовое слово у Анны? Ответ:'),
)

def target(x, task):
    if task=='sort4':return np.sort(x,axis=1)
    if task=='reverse4':return x[:,::-1]
    if task=='rotate4':return np.roll(x,1,axis=1)
    return np.sort(x,axis=1)[:,::-1]

def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def paired_comparison(out):
    """Same 1600 requests, routes, permutations and 25600 numbers as 19.5625%."""
    base=ROOT/'artifacts/codepy_ssm_diagnosis_20261031/guarded'
    old=json.loads((base/'repair_results.json').read_text())
    runs=[]
    for seed in range(5):
        c=catalog(seed)
        stream=json.loads((base/f'fresh_stream_seed{seed}.json').read_text())
        x=np.concatenate(list(c.g.vm.cases(4,16000).values()))
        queries=[ev for ev in stream['events'] if ev['kind']=='query']
        previous=next(r for r in old['runs'] if r['seed']==seed)['metrics']['repaired']
        assert len(queries)==len(previous)==320
        before=after=cases=0
        for ev,ref in zip(queries,previous):
            indices=np.random.default_rng(ev['sample_seed']).choice(len(x),16,replace=False)
            actual=x[indices].copy();ideal=actual.copy()
            for call in ev['calls']:
                # Reuse the historical neural route, not the expected task label.
                route=int(stream['repaired_predictions'][call['position']])
                df=stream['definitions'][str(call['definition'])]
                assert route==ref['routes'][ev['calls'].index(call)]
                p=c.g.previous.baseline.calls.PERMUTATIONS[df['perm'],:4]
                wanted=target(ideal[:,p],TASKS[df['task']]);ideal=wanted[:,np.argsort(p)]
                independent=c.execute(route,actual,independent=True)
                actual=c.execute(route,actual)
                np.testing.assert_array_equal(actual,independent)
            before+=ref['exact'];after+=int(np.all(actual==ideal,axis=1).sum());cases+=len(actual)
        runs.append({'seed':seed,'before_exact':before,'after_exact':after,'cases':cases,'requests':320})
        print('PAIRED',runs[-1],flush=True)
    result={'runs':runs,'before_exact':sum(r['before_exact'] for r in runs),'after_exact':sum(r['after_exact'] for r in runs),'cases':sum(r['cases'] for r in runs),
            'same_saved_routes_and_inputs':True,'historical_controller_not_retrained':True,'baseline_outputs_from_frozen_results':True,'new_outputs_fresh_jax_and_numpy':True}
    assert result['before_exact']==5008 and result['cases']==25600
    save(out/'paired_library_comparison.json',result)
    return result

def main(out,noise_tokens):
    out.mkdir(parents=True,exist_ok=True)
    protocol={'version':'semantic-codepy-end-to-end-v1','seeds':list(range(5)),'tasks':TASKS,'records':RECORDS,'noise_tokens_requested':noise_tokens,'new_trainings':0,
              'selection':'All five seeds, all four target-specific 7041 pools; no heldout selection. Fixed semantic requests, exact bitwise argument roundtrip.'}
    if (out/'protocol.json').exists():raise ValueError('Use a fresh output directory')
    save(out/'protocol.json',protocol)
    ids=np.load(V5/'unique_corpus/test.npy')[:noise_tokens]
    enc=Pipeline(0).encoder
    noise=enc.decode(ids)
    rows=[]
    for seed in range(5):
        p=Pipeline(seed)
        pools=[]
        for task in TASKS:
            pool=np.concatenate(list(p.catalog.g.vm.cases(4,16000+TASKS.index(task)).values()))
            assert len(pool)==7041
            pools.append(pool)
        for i,(entity,prop,handle,question) in enumerate(RECORDS):
            p.remember(entity,prop,handle,i*24,pools[i])
        allocated_before=int(p.semantic.open_clock[0])
        p.observe(noise)
        text_seen=int(p.text_carry.seen[0])
        for i,(entity,prop,handle,question) in enumerate(RECORDS):
            result=p.execute(question)
            assert result.handle==handle and result.command==i*24
            np.testing.assert_array_equal(result.arguments.view(np.uint32),pools[i].view(np.uint32))
            exact=int(np.all(result.output==target(pools[i],TASKS[i]),axis=1).sum())
            row={'seed':seed,'task':TASKS[i],'handle':result.handle,'recovered_command':result.command,'cases':len(pools[i]),'exact':exact,'jax_numpy_agree':result.independent_agrees,
                 'argument_bits_preserved':True,'noise_v5_tokens':len(p.encoder.encode(noise)),'text_pointer_tokens_seen':text_seen,'memory_allocated_before':allocated_before,'memory_allocated_after':int(p.semantic.open_clock[0]),'version':p.version}
            rows.append(row);save(out/'progress.json',rows);print('SEMANTIC',seed,TASKS[i],exact,'/',len(pools[i]),flush=True)
            assert exact==7041
        try:p.recall('несуществующая запись')
        except UnknownKey:pass
        else:raise AssertionError('Unknown key did not abstain')
        entity,prop,handle,question=RECORDS[0]
        changed=pools[0][:16,::-1].copy();p.remember(entity,prop,handle,24,changed)
        result=p.execute(question)
        assert result.command==24
        np.testing.assert_array_equal(result.output,changed[:,::-1])
    paired=paired_comparison(out)
    result={'passed':True,'runs':rows,'seeds':5,'programs':20,'numeric_cases':sum(r['cases'] for r in rows),'exact':sum(r['exact'] for r in rows),'noise_text_sha256':hashlib.sha256(noise.encode()).hexdigest(),
            'all_arguments_decoded_from_neural_kv':True,'semantic_handle_decoded_by_v5':True,'pointer_model_streamed':True,'unknown_and_overwrite_controls_passed':True,'research_training_updates':0,'paired_comparison':paired,
            'limitations':['Frozen models orchestrated; weights not fused.','Literal typed records and short v5 handles; no unrestricted language-to-program induction.','Text SSM contributes streaming language context, not learned numerical task selection.','Finite float32 length-four cases, not a universal algorithm proof.']}
    save(out/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','paired_comparison']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--noise-tokens',type=int,default=1000000)
    args=parser.parse_args()
    if not 0<=args.noise_tokens<=1000000:parser.error('Noise token count must be 0..1000000')
    main(args.output_dir,args.noise_tokens)
