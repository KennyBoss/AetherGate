"""Open-key adapters around the frozen root KV-Memory SSM.

Commands use a fixed uint32/base96 codec. The neural decoder still has96
symbols. PagedSSM uses an explicit key->address index, never a payload cache.
"""
import os
os.environ['JAX_PLATFORM_NAME']='cpu';os.environ['JAX_PLATFORMS']='cpu'
import sys,time,hashlib
from pathlib import Path
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'ssm_dispatch'))
import experiment as old
KEY=old.KEY0;VALUE=old.VALUE0;BASE=96;DIGITS=5;CHUNK=128;PER_BANK=12//DIGITS
POWERS=BASE**np.arange(DIGITS,dtype=np.int64)
KERNELS={}

class Event(NamedTuple):
    write: bool
    key: str
    value: int=0


def validate_value(value):
    if not isinstance(value,(int,np.integer)) or not 0<=int(value)<2**32:raise ValueError('Command ID must be uint32')
    return int(value)

def encode(value):
    return ((validate_value(value)//POWERS)%BASE).astype(np.int32)


def hashed_key(seed,key,field):
    material=str(seed).encode()+b'\0'+key.encode('utf-8')+bytes([field])
    vector=np.frombuffer(hashlib.sha256(material).digest(),np.uint8).astype(np.float32)-127.5
    return vector/np.linalg.norm(vector)


def load_params(seed):
    path=ROOT/f'artifacts/codepy_ssm_diagnosis_20261031/guarded/repaired_weights_seed{seed}.npz'
    with np.load(path,allow_pickle=False) as z:return {k:jnp.asarray(z[k]) for k in z.files}


@jax.jit
def run_chunk(params,states,pages,features,digits,writes,present):
    def command(states,data):
        page,feature,value,write,known=data
        def apply(states):
            selected=jax.tree_util.tree_map(lambda rail:rail[page][None],states)
            def field(state,args):
                keyvec,digit=args
                p={**params,'token_embed':params['token_embed'].at[KEY].set(keyvec)}
                tokens=jnp.asarray([KEY,0,0,3],jnp.int32).at[1].set(jnp.where(write,2,1)).at[2].set(jnp.where(write,VALUE+digit,0))
                logits,state=old.forward(p,tokens[None],state)
                prediction=jnp.argmax(logits[0,1,VALUE:VALUE+BASE]).astype(jnp.int32)
                return state,prediction
            selected,result=jax.lax.scan(field,selected,(feature,value))
            updated=jax.tree_util.tree_map(lambda rail,new:rail.at[page].set(new[0]),states,selected)
            return updated,result
        return jax.lax.cond(known,apply,lambda st:(st,jnp.full((DIGITS,),-1,jnp.int32)),states)
    return jax.lax.scan(command,states,(pages,features,digits,writes,present))


def deep_size(obj,seen=None):
    if seen is None:seen=set()
    if id(obj) in seen:return 0
    seen.add(id(obj));size=sys.getsizeof(obj)
    if isinstance(obj,dict):size+=sum(deep_size(k,seen)+deep_size(v,seen) for k,v in obj.items())
    elif isinstance(obj,(tuple,list)):size+=sum(deep_size(v,seen) for v in obj)
    return size


class Dictionary:
    def __init__(self):self.data={};self.catalog_limit=96;self.elapsed_s=0.;self.cpu_s=0.
    def process(self,events):
        t=time.perf_counter();cpu=time.process_time();result=[]
        for event in events:
            if not isinstance(event.key,str):raise TypeError('Keys must be strings')
            if event.write:
                validate_value(event.value);self.data[event.key]=int(event.value);self.catalog_limit=max(self.catalog_limit,int(event.value)+1)
            else:
                if event.value!=0:raise ValueError('A query cannot contain its answer')
                result.append(self.data.get(event.key,-1))
        self.elapsed_s+=time.perf_counter()-t;self.cpu_s+=time.process_time()-cpu
        return result
    def storage(self):return {'keys':len(self.data),'payload_bytes':len(self.data)*4,'python_index_or_dictionary_bytes':deep_size(self.data),'neural_state_bytes':0,'parameter_bytes':0}


class OpenSSM:
    def __init__(self,params,seed,mode):
        if mode not in ('hash','paged'):raise ValueError(mode)
        self.params=params;self.seed=seed;self.mode=mode;self.states=old.init_stream_state(params,1)
        self.capacity=1;self.index={} if mode=='paged' else None;self.seen_count=0;self.catalog_limit=96
        self.elapsed_s=0.;self.cpu_s=0.;self.compile_s=0.;self.kernel_s=0.;self.growths=[]
        self.basis=np.asarray(params['token_embed'])[KEY:VALUE]
        np.testing.assert_allclose(self.basis@self.basis.T,np.eye(12),atol=2e-7)
    def _grow(self,needed):
        if needed<=self.capacity:return
        before=self.capacity
        while self.capacity<needed:self.capacity*=2
        t=time.perf_counter();new=old.init_stream_state(self.params,self.capacity)
        self.states=jax.tree_util.tree_map(lambda dst,src:dst.at[:before].set(src),new,self.states)
        jax.block_until_ready(self.states);self.growths.append({'before':before,'after':self.capacity,'elapsed_s':time.perf_counter()-t})
    def process(self,events):
        if len(events)>CHUNK:raise ValueError('Stream chunks must contain at most128 events')
        t=time.perf_counter();cpu=time.process_time();n=len(events)
        pages=np.zeros(CHUNK,np.int32);features=np.zeros((CHUNK,DIGITS,32),np.float32);digits=np.zeros((CHUNK,DIGITS),np.int32)
        writes=np.zeros(CHUNK,bool);present=np.zeros(CHUNK,bool);limits=np.zeros(CHUNK,np.int64)
        for i,event in enumerate(events):
            if not isinstance(event.key,str):raise TypeError('Keys must be strings')
            writes[i]=event.write
            if event.write:
                digits[i]=encode(event.value);self.catalog_limit=max(self.catalog_limit,int(event.value)+1)
            elif event.value!=0:raise ValueError('A query cannot contain its answer')
            limits[i]=self.catalog_limit
            if self.mode=='paged':
                if event.write and event.key not in self.index:
                    number=len(self.index);self.index[event.key]=(number//PER_BANK,(number%PER_BANK)*DIGITS)
                address=self.index.get(event.key)
                if address is None:continue
                page,base=address;pages[i]=page;features[i]=self.basis[base:base+DIGITS];present[i]=True
            else:
                features[i]=np.stack([hashed_key(self.seed,event.key,f) for f in range(DIGITS)]);present[i]=True
        if self.mode=='paged':self._grow(max(1,(len(self.index)+PER_BANK-1)//PER_BANK))
        args=(self.params,self.states,jnp.asarray(pages),jnp.asarray(features),jnp.asarray(digits),jnp.asarray(writes),jnp.asarray(present))
        if self.capacity not in KERNELS:
            start=time.perf_counter();KERNELS[self.capacity]=run_chunk.lower(*args).compile();self.compile_s+=time.perf_counter()-start
        start=time.perf_counter();self.states,pred=KERNELS[self.capacity](*args);jax.block_until_ready((self.states,pred));self.kernel_s+=time.perf_counter()-start
        values=np.asarray(pred,dtype=np.int64);valid=np.all(values>=0,axis=1);decoded=np.maximum(values,0)@POWERS
        valid&=(decoded<limits)&(decoded<2**32);decoded=np.where(valid,decoded,-1)
        result=[int(decoded[i]) for i in range(n) if not events[i].write]
        self.elapsed_s+=time.perf_counter()-t;self.cpu_s+=time.process_time()-cpu
        return result
    def storage(self):
        return {'keys':len(self.index) if self.index is not None else None,'bank_capacity':self.capacity,'active_banks':max(1,(len(self.index)+PER_BANK-1)//PER_BANK) if self.index is not None else 1,
                'neural_state_bytes':old.tree_bytes(self.states),'parameter_bytes':old.tree_bytes(self.params),'python_index_or_dictionary_bytes':deep_size(self.index) if self.index is not None else 0,
                'payload_bytes':None,'batch_feature_buffer_bytes':CHUNK*DIGITS*32*4,'growths':self.growths}
