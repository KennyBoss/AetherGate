"""Independent full-range NumPy VM; historical clipped certificates stay frozen."""
import numpy as np
def numpy_state(program, state, n):
    """Independent NumPy interpreter of all current DSL operations, no JAX."""
    r0,r1,ptr,pred,tape = [np.array(x,copy=True) for x in state]
    count = len(r0)
    rows = np.arange(count)
    pc = np.zeros(count,np.int32)
    for _ in range(len(program)):
        ops = np.asarray(program)[pc]
        cell = tape[rows, ptr].copy()
        old0, old1, oldp, oldptr = r0.copy(), r1.copy(), pred.copy(), ptr.copy()
        newpc = np.minimum(pc + 1, len(program) - 1)
        for op in np.unique(ops):
            m = ops == op
            if op == 1: r0[m] = 0
            elif op == 2: r1[m] = 0
            elif op == 3: r0[m] = cell[m]
            elif op == 4: r1[m] = cell[m]
            elif op == 5: tape[rows[m], oldptr[m]] = old0[m]
            elif op == 6: tape[rows[m], oldptr[m]] = old1[m]
            elif op == 7: ptr[m] = oldptr[m] + 1
            elif op == 8: ptr[m] = oldptr[m] - 1
            elif op == 9: ptr[m] = 0
            elif op == 10: r0[m] = old0[m] + old1[m]
            elif op == 11: r0[m] = old0[m] + 1
            elif op == 12: r0[m] = old0[m] - 1
            elif op == 13: pred[m] = old0[m] > 0
            elif op == 14: pred[m] = old1[m] > 0
            elif op == 15: pred[m] = oldptr[m] < n
            elif op == 16: r0[m] = old0[m] + oldp[m].astype(np.float32)
            elif op == 17: r0[m] = np.where(oldp[m], old1[m], old0[m])
            elif op == 18: newpc[m] = 0
            elif op == 19: newpc[m] = np.where(oldp[m], 0, newpc[m])
            elif op == 20: pred[m] = cell[m] > old0[m]
            elif op == 21: pred[m] = cell[m] > old1[m]
            elif op == 22: pred[m] = oldptr[m] < n - 1
            elif op == 23: r0[m] = oldptr[m]
            elif op == 24: r1[m] = oldptr[m]
            elif op == 25: r0[m] = np.where(oldp[m], cell[m], old0[m])
            elif op == 26: r1[m] = np.where(oldp[m], cell[m], old1[m])
            elif op == 27: r0[m] = np.where(oldp[m], oldptr[m], old0[m])
            elif op == 28: r1[m] = np.where(oldp[m], oldptr[m], old1[m])
            elif op == 29: newpc[m] = min(3, len(program) - 1)
            elif op == 30: newpc[m] = np.where(oldp[m], min(3, len(program) - 1), newpc[m])
            elif op == 31: newpc[m] = pc[m]
            elif op == 32: r0[m] = old1[m]
            elif op == 33: r1[m] = old0[m]
            elif op == 34: r1[m] = np.where(oldp[m], old0[m], old1[m])
            elif op == 35:
                a, b = old0[m], old1[m]
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                # NumPy SIMD tie selection differs across CPUs. The ABI fixes
                # min(-0,+0)=-0 and max(-0,+0)=+0 in either operand order.
                zeros = (a == 0) & (b == 0)
                lo_zero = np.copysign(np.zeros_like(a), np.where(np.signbit(a) | np.signbit(b), np.float32(-1), np.float32(1)))
                hi_zero = np.copysign(np.zeros_like(a), np.where(np.signbit(a) & np.signbit(b), np.float32(-1), np.float32(1)))
                r0[m], r1[m] = np.where(zeros, lo_zero, lo), np.where(zeros, hi_zero, hi)
        ptr = np.clip(ptr, 0, 7)
        pc = newpc
    return r0,r1,ptr,pred,tape


def execute_pairs(program,x,blocks):
    import itertools
    addresses=list(itertools.permutations(range(x.shape[1]),2))
    out=x.copy()
    for token in program:
        if not token:continue
        block,pair=divmod(int(token)-1,len(addresses));block=blocks[block];indices=list(addresses[pair])
        tape=np.zeros((len(x),8),np.float32);tape[:,block['layout']]=out[:,indices]
        z=np.zeros(len(x),np.float32)
        state=numpy_state(block['code'],[z,z,np.zeros(len(x),np.int32),np.zeros(len(x),bool),tape],3)
        bank=np.concatenate((state[4][:,:3],state[0][:,None],state[1][:,None]),axis=1)
        out[:,indices]=bank[:,block.get('outputs',block['layout'])]
    return out

def numpy_execute(genome,x,entries,primitive_count):
    import itertools
    from research.codepy.block_contracts import contracts as c
    PERMUTATIONS=np.asarray([list(p)+[4,5,6,7] for p in itertools.permutations(range(4))],np.int32)
    size=len(x);state=c.State(np.zeros(size,np.float32),np.zeros(size,np.float32),np.zeros(size,np.int32),np.zeros(size,bool),np.pad(x,((0,0),(0,4))))
    remaining=72
    for gene in genome:
        if remaining<=0:break
        token,a0,a1,ptr_src,pred_src,perm_id,gate_id=map(int,gene)
        code=entries[token][:remaining];remaining-=len(code)
        if token<primitive_count:state=c.State(*numpy_state(code,state,4));continue
        mapping=PERMUTATIONS[perm_id];inverse=np.argsort(mapping)
        tape=state.tape[:,mapping];pointer=inverse[state.ptr] if ptr_src==0 else np.full(size,ptr_src-1,np.int32)
        cell=state.tape[np.arange(size),state.ptr]
        vals=[state.r0,state.r1,*[state.tape[:,i] for i in range(4)],cell,np.zeros(size,np.float32),np.ones(size,np.float32),-np.ones(size,np.float32)]
        r0=vals[a0];vals[0],vals[1]=vals[1],vals[0];r1=vals[a1]
        predicate=[state.predicate,np.zeros(size,bool),np.ones(size,bool),r0>r1,r0>0,cell>r0][pred_src]
        gate=[np.ones(size,bool),state.predicate,~state.predicate,state.r0>state.r1,state.r0<state.r1,
              state.tape[:,0]>state.tape[:,1],state.tape[:,2]>state.tape[:,3],state.r0>0][gate_id]
        bound=c.State(r0,r1,pointer,predicate,tape);contract=c.infer(code)
        local=[np.zeros_like(v) for v in bound]
        for name in contract['input_registers']:local[c.REGS.index(name)]=bound[c.REGS.index(name)].copy()
        ti=np.zeros_like(tape,bool);to=np.zeros_like(tape,bool)
        for i,p in enumerate(pointer):
            footprint=contract['memory_by_initial_pointer'][int(p)]
            ti[i,footprint['input_cells']]=True;to[i,footprint['output_cells']]=True
        local[4]=np.where(ti,tape,0);local=list(numpy_state(code,local,4))
        local[2]=mapping[local[2]];local[4]=local[4][:,inverse];to=to[:,inverse]
        committed=[v.copy() for v in state]
        for name in contract['output_registers']:
            index=c.REGS.index(name);committed[index]=np.where(gate,local[index],committed[index])
        committed[4]=np.where(to & gate[:,None],local[4],state.tape)
        state=c.State(*committed)
    return state.tape[:,:4]
