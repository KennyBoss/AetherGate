"""Read source archives and diagnose old selection without modifying artifacts."""
import sys,json,itertools,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'research/codepy/block_contracts'));import contracts as c
OLD=ROOT/'artifacts/codepy_subtask_discovery_20261030'
rows=[]
for seed in range(5):
 d=json.loads((OLD/f'library_seed{seed}.json').read_text());selected=d['blocks'];naked=sum(not c.infer(b)['memory_by_initial_pointer'][0]['output_cells'] for b in selected);pool=set();windows=set()
 for source in d['sources']:
  for program in source['archive']:
   code=[x for x in program if x!=0]
   for width in range(2,13):
    for start in range(len(code)-width+1):windows.add(tuple(code[start:start+width]))
 for code in windows:
  writes=c.infer(code)['memory_by_initial_pointer'][0]['output_cells']
  for pair in itertools.combinations(range(3),2):
   if set(pair)<=set(writes):pool.add((code,pair))
 old_units=d['mining']['ablation_evaluations']*64*36
 rows.append({'seed':seed,'selected':len(selected),'selected_without_tape_writes_at_entry0':naked,'selected_max_length':max(map(len,selected)),'source_programs':sum(len(s['archive']) for s in d['sources']),'unique_nonnoop_windows_2to12':len(windows),'two_output_layout_candidates':len(pool),'old_ablation_evaluations':d['mining']['ablation_evaluations'],'old_ablation_primitive_case_budget':old_units})
out=ROOT/'artifacts/codepy_behavioral_blocks_20261101';(out/'static_diagnosis.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
