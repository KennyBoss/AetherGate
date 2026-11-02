"""Extract register-return slices; retain live input-dependent predicate skeletons."""
import itertools
from functools import lru_cache
from common import old_slicing
c=old_slicing.contracts
CONDITIONAL={17,25,26,27,28,34}
COMPARISONS={13,14,20,21}

def trace(program):
    ptr=0;rows=[]
    for i,op in enumerate(program):
        if op not in c.READS:raise ValueError('Unsupported opcode')
        reads={('r',r) for r in c.READS[op] if r!=2};writes={('r',r) for r in c.WRITES.get(op,()) if r!=2}
        if op in c.TAPE_READ:reads.add(('t',ptr))
        if op in (5,6):writes.add(('t',ptr))
        rows.append((i,op,ptr,reads,writes))
        ptr=min(7,ptr+1) if op==7 else max(0,ptr-1) if op==8 else 0 if op==9 else ptr
    return rows

def live_indices(program,roots):
    live=set(roots);kept=[]
    for i,op,ptr,reads,writes in reversed(trace(program)):
        if writes&live:kept.append(i);live=(live-writes)|reads
    return set(kept)

@lru_cache(maxsize=131072)
def _condition_info(program,layout,outputs):
    roots=[('r',k-3) if k>=3 else ('t',k) for k in outputs];live=live_indices(program,roots)
    tags={('t',layout[0]):1,('t',layout[1]):2};producer=None;mask=[False]*len(program);strength=0
    for i,op,ptr,reads,writes in trace(program):
        predicate=tags.get(('r',3),0)
        if i in live and op in CONDITIONAL and predicate:
            mask[i]=True;strength=max(strength,int(predicate).bit_count())
            if producer is not None and producer in live:mask[producer]=True
        tag=0
        for field in reads:tag|=tags.get(field,0)
        for field in writes:tags[field]=tag
        if ('r',3) in writes:producer=i if op in COMPARISONS else None
    return tuple(mask),strength

def condition_mask(program,layout,outputs=(3,4)):
    return _condition_info(tuple(program),tuple(layout),tuple(outputs))[0]

def branch_strength(program,layout,outputs=(3,4)):
    return _condition_info(tuple(program),tuple(layout),tuple(outputs))[1]

def register_slice(program):
    kept=live_indices(program,[('r',0),('r',1)]);code=[];indices=[];pointer=0
    for i,op,ptr,_,_ in trace(program):
        if i not in kept:continue
        if op in old_slicing.POINTER_USES and ptr!=pointer:
            direct=[7 if ptr>pointer else 8]*abs(ptr-pointer);reset=[9]+[7]*ptr
            moves=direct if len(direct)<=len(reset) else reset
            code.extend(moves);indices.extend([-1]*len(moves));pointer=ptr
        code.append(op);indices.append(i)
    return code,indices

def candidate_pool(sources):
    rows={}
    def add(code,origin):
        if not 2<=len(code)<=12:return
        for layout in itertools.combinations(range(3),2):
            key=(tuple(code),layout)
            if key not in rows:rows[key]={'code':list(code),'layout':list(layout),'outputs':[3,4],'provenance':origin,'frequency':0}
            rows[key]['frequency']+=1
    for si,source in enumerate(sources):
        for pi,program in enumerate(source['archive']):
            for end in range(2,len(program)+1):
                code,indices=register_slice(program[:end]);add(code,{'method':'register_dependency_slice','source':si,'program':pi,'prefix':end,'source_indices_or_pointer_moves':indices})
            nonzero=[(i,op) for i,op in enumerate(program) if op]
            for n in range(2,13):
                for start in range(len(nonzero)-n+1):
                    fragment=nonzero[start:start+n];code=[op for i,op in fragment]
                    # Only live variable-dependent branches get extra window candidates.
                    if any(condition_mask(code,[0,1])) or any(condition_mask(code,[0,2])) or any(condition_mask(code,[1,2])):
                        add(code,{'method':'live_branch_window','source':si,'program':pi,'source_indices':[i for i,op in fragment]})
    result=list(rows.values())
    for row in result:row['live_condition']=any(condition_mask(row['code'],row['layout']));row['branch_strength']=branch_strength(row['code'],row['layout'])
    return sorted(result,key=lambda r:(-r['branch_strength'],not r['live_condition'],len(r['code']),-r['frequency'],r['code'],r['layout']))
