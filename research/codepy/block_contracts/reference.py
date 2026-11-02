"""NumPy state reference adapted from the existing independent executor."""
import numpy as np
CLIP_VALUE=1e6
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
                r0[m], r1[m] = np.minimum(old0[m], old1[m]), np.maximum(old0[m], old1[m])
        ptr = np.clip(ptr, 0, 7)
        pc = newpc
        r0 = np.clip(r0, -CLIP_VALUE, CLIP_VALUE)
        r1 = np.clip(r1, -CLIP_VALUE, CLIP_VALUE)
        tape = np.clip(tape, -CLIP_VALUE, CLIP_VALUE)
    return r0,r1,ptr,pred,tape

