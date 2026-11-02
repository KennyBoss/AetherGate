"""Task-independent backward slicing of two tape outputs in straight-line programs."""
import sys,itertools,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'block_contracts'))
import contracts
POINTER_USES={3,4,5,6,15,20,21,22,23,24,25,26,27,28}
POINTER_MOVES={7,8,9}

def slice_outputs(program,outputs):
    ptr=0;trace=[]
    for i,op in enumerate(program):
        if op not in contracts.READS:raise ValueError('Unsupported opcode')
        reads={('r',r) for r in contracts.READS[op] if r!=2};writes={('r',r) for r in contracts.WRITES.get(op,()) if r!=2}
        if op in contracts.TAPE_READ:reads.add(('t',ptr))
        if op in (5,6):writes.add(('t',ptr))
        trace.append((i,op,ptr,reads,writes))
        if op==7:ptr=min(7,ptr+1)
        elif op==8:ptr=max(0,ptr-1)
        elif op==9:ptr=0
    live={('t',k) for k in outputs};kept=[]
    for i,op,ptr,reads,writes in reversed(trace):
        if writes&live:
            kept.append((i,op,ptr));live=(live-writes)|reads
    code=[];indices=[];pointer=0
    for i,op,ptr in reversed(kept):
        if op in POINTER_USES and ptr!=pointer:
            direct=[7 if ptr>pointer else 8]*abs(ptr-pointer);reset=[9]+[7]*ptr
            moves=direct if len(direct)<=len(reset) else reset
            code.extend(moves);indices.extend([-1]*len(moves));pointer=ptr
        code.append(op);indices.append(i)
    return code,indices

def candidate_pool(sources):
    records={}
    def add(code,layout,provenance):
        if not 2<=len(code)<=12:return
        writes=contracts.infer(code)['memory_by_initial_pointer'][0]['output_cells']
        if not set(layout)<=set(writes):return
        key=(tuple(code),tuple(layout))
        if key not in records:records[key]={'code':list(code),'layout':list(layout),'provenance':provenance,'source_programs':set()}
        records[key]['source_programs'].add((provenance['source'],provenance['program']))
    for si,source in enumerate(sources):
        for pi,program in enumerate(source['archive']):
            for end in range(2,len(program)+1):
                for layout in itertools.combinations(range(3),2):
                    code,indices=slice_outputs(program[:end],layout)
                    add(code,layout,{'method':'backward_slice','source':si,'program':pi,'prefix':end,'source_indices_or_pointer_relocation':indices})
            nonzero=[(i,v) for i,v in enumerate(program) if v!=0]
            for width in range(2,13):
                for start in range(len(nonzero)-width+1):
                    fragment=nonzero[start:start+width];code=[v for i,v in fragment]
                    writes=contracts.infer(code)['memory_by_initial_pointer'][0]['output_cells']
                    for layout in itertools.combinations(range(3),2):
                        if set(layout)<=set(writes):add(code,layout,{'method':'nonnoop_window','source':si,'program':pi,'source_indices':[i for i,v in fragment]})
    rows=[]
    for r in records.values():r['frequency']=len(r.pop('source_programs'));rows.append(r)
    return sorted(rows,key=lambda r:(len(r['code']),-r['frequency'],r['code'],r['layout']))
