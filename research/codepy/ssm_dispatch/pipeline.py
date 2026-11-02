"""Versioned semantic requests with command AND float32 operands in neural KV state.

The text pointer and v5 keep distinct frozen weights and carries. Language reads
only select an opaque handle; executable payloads are decoded from KV, not labels.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache, partial
import hashlib
import copy
import marshal
import json
from pathlib import Path
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import train_ssm_text as core
import russian_memory as ru
from research.ssm_neural_copy import neural_context_copy as pointer
from research.codepy.ssm_scaling import runtime as codec
from research.codepy.ssm_dispatch.catalog import ROOT, TASKS, arguments, catalog, digest, release_path
from research.codepy.ssm_dispatch.programs import vector_arguments

SCHEMA = 'semantic-codepy-session-v7'
V3_IMPLEMENTATION = 'a1e54f1297ec6e2cde13fec56ca4ee9d33e7f5bf7d8da25a590e354909c3a923'
V6_IMPLEMENTATION = '2454aa7d40aaa56e9295f8b82bcb1237e10d96cee35b96a45b2165f177f370e3'
V7_IMPLEMENTATION = 'a4b341e2c4b68243625be0cb3511d865fa387618609c2f0ed7a58b875a658298'
V7_FUSED_IMPLEMENTATION = '5117b6afbe1de0e7eb063f01436aebc73cdcf914bae43daa2025f71eabaa8c5a'
V7_READ_IMPLEMENTATION = '2c4ad6865d7f257f4cb32ecee13db40e0d6f5ec2db4da239595526007ad81eb4'
V7_MATRIX_IMPLEMENTATION = '212a4e81adeaa41a743cffcc571080c68884a1a1ed29015fe03fcee9f393ef00'
V7_PAGE_IMPLEMENTATION = 'd24d0263a64e3b02f8b35949448901dd0f1678573f41debea52faa26394b1a14'
V3_EXAMPLES = '6c4462a1abec7fd0079c8ed3f523356225a4ea4e70e9f67a224387e68bdcfba6'
V3_PROGRAMS = '0128c81d1b14c79f2ec5db13e1ce42c8c3b5a9dc23101fa4b958a5ecaca25c65'
V2_IMPLEMENTATION = '43602a21032fb24d5289e4d90fd1d60e1e47fe204064989d32c08e8d25486d25'
V2_DEVELOPMENT = '06312f02bda658704d51381d8dc45abc4ee2a082dd2f13e8364a135e85cae4f3'
V1_IMPLEMENTATION = '2da43a49714d321d43cfe3440e6c69b4f6a63b1dbaa6f215aa413537355e8080'
V5 = ROOT / 'artifacts/russian_memory_v5'
POINTER = ROOT / 'artifacts/ssm_neural_copy_20261031'
BATCH = 128
READ_CHUNKS = 8

class UnknownKey(KeyError):
    """The semantic answer has no registered executable payload."""

@dataclass(frozen=True)
class Execution:
    handle: str
    command: int
    arguments: np.ndarray
    output: np.ndarray
    independent_agrees: bool
    next_text_token: str

@lru_cache(maxsize=1)
def models():
    selection = json.loads((V5 / 'FINAL_SELECTION.json').read_text())
    if digest(V5 / selection['checkpoint']) != selection['sha256']:
        raise ValueError('v5 checkpoint hash mismatch')
    v5, _, vocab, config, _ = core.load_stream_checkpoint(str(V5 / selection['checkpoint']))
    enc = ru.Encoder(vocab, config['tokenizer_config'])
    item = next(i for i in json.loads((POINTER / 'exports/exports.json').read_text()) if i['model']=='ssm' and i['seed']==23)
    if digest(item['path']) != item['sha256']:
        raise ValueError('Pointer checkpoint hash mismatch')
    with np.load(release_path(item['path']), allow_pickle=False) as data:
        text = {k[6:]: jnp.asarray(data[k]) for k in data.files if k.startswith('param_')}
        cfg = json.loads(str(data['config_json'].item()))
        vocab = json.loads(str(data['vocab_json'].item()))
    if cfg['inference_recipe'] != 'joint_neural_context_copy':
        raise ValueError('Full pointer model required')
    return v5, enc, text, ru.Encoder(vocab, cfg['tokenizer_config']), selection['sha256'], item['sha256']

@jax.jit
def text_step(params, ids, base, carry):
    logits, base = core.forward(params, ids, base)
    logits, carry = pointer.apply(params, ids, logits, carry)
    return logits[:, -1], base, carry

@jax.jit
def numeric_step(params, state, words, write):
    """Each record has three banks: 25 base96 fields for five uint32 values."""
    count = words.shape[0]
    digits = ((words.astype(jnp.uint32)[..., None] // jnp.asarray(codec.POWERS, jnp.uint32)) % 96).astype(jnp.int32).reshape(count, 25)
    def field(state, i):
        bank = jnp.arange(count)*3 + i//10
        selected = jax.tree_util.tree_map(lambda a: a[bank], state)
        tokens = jnp.zeros((count, 4), jnp.int32).at[:,0].set(codec.KEY+i%10)
        tokens = tokens.at[:,1].set(jnp.where(write,2,1)).at[:,2].set(jnp.where(write,codec.VALUE+digits[:,i],0)).at[:,3].set(3)
        logits, selected = core.forward(params, tokens, selected)
        state = jax.tree_util.tree_map(lambda a,b: a.at[bank].set(b), state, selected)
        return state, jnp.argmax(logits[:,1,codec.VALUE:codec.VALUE+96], -1).astype(jnp.uint32)
    state, predictions = jax.lax.scan(field, state, jnp.arange(25))
    predictions = predictions.T.reshape(count,5,5)
    decoded = jnp.sum(predictions*jnp.asarray(codec.POWERS, jnp.uint32), -1, dtype=jnp.uint32)
    return state, decoded

@jax.jit
def _numeric_read_reference(params, state):
    """Read exactly 10/10/5 fields through the unchanged trained decoder."""
    def field(state, i):
        count = state.h.shape[0]
        tokens = jnp.zeros((count,4),jnp.int32).at[:,0].set(codec.KEY+i).at[:,1].set(1).at[:,3].set(3)
        logits, state = core.forward(params,tokens,state)
        return state, jnp.argmax(logits[:,1,codec.VALUE:codec.VALUE+96],-1).astype(jnp.uint32)
    state, first = jax.lax.scan(field,state,jnp.arange(5),unroll=5)
    first = first.T.reshape(-1,3,5)
    # Bank 2 is complete; keep only banks 0/1 for their remaining five fields.
    indices = (jnp.arange(first.shape[0])[:,None]*3+jnp.arange(2)).reshape(-1)
    state = jax.tree_util.tree_map(lambda a:a[indices],state)
    _, last = jax.lax.scan(field,state,jnp.arange(5,10),unroll=5)
    last = last.T.reshape(-1,2,5)
    digits = jnp.concatenate((first[:,0],last[:,0],first[:,1],last[:,1],first[:,2]),axis=-1).reshape(-1,5,5)
    return jnp.sum(digits*jnp.asarray(codec.POWERS,jnp.uint32),-1,dtype=jnp.uint32)

_NUMERIC_FAST_KEYS=frozenset(('decay_raw','hidden_bias','input_b','kv_decay_raw','kv_deref_token','kv_erase_token','kv_key_token','kv_logit_scale','kv_query_token','kv_value_token','kv_write_context_token','output_bias','output_c','token_embed','kv_durable_marker'))

def _numeric_plan(params):
    """For each query compose M_t = P_t @ M_0 + C_t, including tiny gates."""
    ids=jnp.stack((codec.KEY+jnp.arange(10),jnp.ones(10,jnp.int32),jnp.zeros(10,jnp.int32),jnp.full(10,3,jnp.int32)),axis=1)
    x=params['token_embed'][ids]
    previous=jnp.concatenate((jnp.asarray([3],jnp.int32),ids.reshape(-1)[:-1])).reshape(10,4)
    ctx=jax.nn.sigmoid(params['kv_write_context_token'][previous])
    kg=jax.nn.sigmoid(params['kv_key_token'][ids])*(1.0-ctx)
    dg=jax.nn.sigmoid(params['kv_deref_token'][ids])*ctx
    vg=jnp.maximum(jax.nn.sigmoid(params['kv_value_token'][ids])*ctx,dg)
    eg=jnp.maximum(jax.nn.sigmoid(params['kv_erase_token'][ids]),dg)
    d=x.shape[-1]
    def field(carry,i):
        p,c,key=carry
        for j in range(4):
            xx=x[i,j]
            if j==1:
                query=key@p;bias=key@c
            key=(1.0-kg[i,j])*key+kg[i,j]*xx
            # Same rank-one erase/write algebra, with the affine input in C only.
            pv=dg[i,j]*(xx@p)
            cv=(1.0-dg[i,j])*xx+dg[i,j]*(xx@c)
            p=p-vg[i,j]*eg[i,j]*jnp.outer(key,key@p)
            p=p+vg[i,j]*jnp.outer(key,pv)
            c=c-vg[i,j]*eg[i,j]*jnp.outer(key,key@c)
            c=c+vg[i,j]*jnp.outer(key,cv)
        return (p,c,key),(query,bias)
    _,(query,bias)=jax.lax.scan(field,(jnp.eye(d,dtype=jnp.float32),jnp.zeros((d,d),jnp.float32),jnp.zeros(d,jnp.float32)),jnp.arange(10))
    return {'query':query,'bias':bias,'input':x@params['input_b'],'decay':core.decay_from_raw(params['decay_raw']),
            'reset':jnp.all(kg[:,0]==1.0),'c':params['output_c'][:,codec.VALUE:codec.VALUE+96],
            'out_bias':params['output_bias'][codec.VALUE:codec.VALUE+96],
            'embed':params['token_embed'][codec.VALUE:codec.VALUE+96].T,
            'scale':jax.nn.sigmoid(params['kv_query_token'][1])*params['kv_logit_scale']}

def _numeric_matrix_fields(params,h,kv,tables,start,stop):
    def field(h,i):
        for j in range(4):
            h=jnp.tanh(h*tables['decay']+tables['input'][i,j]+params['hidden_bias'])
            if j==1:query_h=h
        return h,query_h
    h,hidden=jax.lax.scan(field,h,jnp.arange(start,stop),unroll=stop-start)
    values=jnp.einsum('qi,bij->qbj',tables['query'][start:stop],kv)+tables['bias'][start:stop,None,:]
    logits=hidden@tables['c']+tables['out_bias']+tables['scale']*(values@tables['embed'])
    return h,jnp.argmax(logits,axis=-1).astype(jnp.uint32)

def _numeric_matrix_read(params,state):
    tables=_numeric_plan(params)
    h,first=_numeric_matrix_fields(params,state.h,state.kv,tables,0,5)
    first=first.T.reshape(-1,3,5)
    indices=(jnp.arange(first.shape[0])[:,None]*3+jnp.arange(2)).reshape(-1)
    _,last=_numeric_matrix_fields(params,h[indices],state.kv[indices],tables,5,10)
    last=last.T.reshape(-1,2,5)
    digits=jnp.concatenate((first[:,0],last[:,0],first[:,1],last[:,1],first[:,2]),axis=-1).reshape(-1,5,5)
    return jnp.sum(digits*jnp.asarray(codec.POWERS,jnp.uint32),-1,dtype=jnp.uint32)

@jax.jit
def numeric_read(params,state):
    """Compose fixed KV transitions, then decode the current live bank matrices."""
    expected=core.init_stream_state(params,state.h.shape[0])
    for name,value,reference in zip(core.SSMState._fields,state,expected):
        if (value is None)!=(reference is None) or (value is not None and (value.shape!=reference.shape or value.dtype!=reference.dtype)):
            raise ValueError('Invalid stream state field: '+name)
    if frozenset(params)!=_NUMERIC_FAST_KEYS:
        return _numeric_read_reference(params,state)
    # Only exact key resets remove the initial trace. Never threshold tiny gates.
    reset=jax.nn.sigmoid(params['kv_key_token'][codec.KEY:codec.KEY+10])*(1.0-jax.nn.sigmoid(params['kv_write_context_token'][3]))
    valid=jnp.all(reset==1.0)&jnp.all(state.prev_ids==3)
    for array in (state.h,state.kv,state.key_trace,*params.values()):
        valid=valid&jnp.all(jnp.isfinite(array))
    return jax.lax.cond(valid,lambda st:_numeric_matrix_read(params,st),lambda st:_numeric_read_reference(params,st),state)

class VerifierCache:
    """Process-local certified snapshots. No results, weights or KV reads are cached."""
    def __init__(self):
        self._scope=None;self._items={}
        self.hits=0;self.misses=0;self.invalidations=0
        paths=[Path(__file__).with_name(n+'.py') for n in ('hierarchy_v1','examples_v1')]
        paths.extend(ROOT/'research/codepy/subtask_discovery/transfer'/n for n in ('call_runtime.py','tasks.py','float32_vm.py'))
        self._runtime_sources={str(p):digest(p) for p in paths}

    def clear(self):
        if self._scope is not None or self._items:self.invalidations+=1
        self._scope=None;self._items.clear()

    @property
    def stats(self):
        return {'hits':self.hits,'misses':self.misses,'invalidations':self.invalidations,'size':len(self._items)}

    @staticmethod
    def _hash(value):
        # marshal v2 has no object-reference/refcount flags. It hashes primitive
        # structures quickly, never calls __reduce__, and is never deserialized.
        return hashlib.sha256(marshal.dumps(value,2)).hexdigest()

    def _signature(self,pipeline):
        from research.codepy.ssm_dispatch.programs import blocks_for,CertificationError,json_hash
        if pipeline.version!=pipeline._trusted_version:raise CertificationError('Session version integrity mismatch')
        cat=pipeline.catalog;version=pipeline.version
        if cat.seed!=version['seed'] or cat.version!=version['catalog'] or cat.argument_library_sha256!=version['argument_library']:
            raise CertificationError('Catalog version integrity mismatch')
        sources={'implementation':'pipeline','catalog_implementation':'catalog','development':'development','hierarchy':'hierarchy','examples':'examples','programs':'programs','ieee':'ieee_runtime','numpy_runtime':'numpy_runtime'}
        for key,name in sources.items():
            if digest(Path(__file__).with_name(name+'.py'))!=version[key]:raise CertificationError('Verifier source integrity mismatch: '+name)
        for path,expected in self._runtime_sources.items():
            if digest(path)!=expected:raise CertificationError('Verifier runtime source integrity mismatch')
        # Source blocks and their immutable provenance must still match on hits.
        source_blocks=json_hash({'blocks':blocks_for(cat),
                                 'file':digest(ROOT/f'artifacts/codepy_transfer_20261101/frozen_library_seed{cat.seed}.json'),
                                 'protocol':digest(ROOT/'artifacts/codepy_transfer_20261101/protocol.json')})
        catalog_sha=self._hash({'owner':id(cat),'seed':cat.seed,'base':cat.version,'argument_library':cat.argument_library_sha256,'programs':pipeline.programs})
        records={k:{'rows':v['rows'],'width':v.get('width',4),'pages':v.get('pages',1),'sha256':v['sha256'],**({'page_sha256':v['page_sha256']} if 'page_sha256' in v else {}),'chunks':len(v['chunks'])} for k,v in pipeline.records.items()}
        # Read-side text carries evolve on every execute; they do not certify
        # bytecode. KV content is independently decoded and checksummed per call.
        semantic=hashlib.sha256()
        for array in jax.tree_util.tree_leaves(pipeline.semantic):
            value=np.asarray(array);semantic.update(str((value.shape,value.dtype.str)).encode());semantic.update(value.tobytes())
        session_sha=self._hash({'owner':id(pipeline),'version':version,'bindings':pipeline.task_bindings,'records':records,'semantic':semantic.hexdigest()})
        return catalog_sha,session_sha,source_blocks

    def get(self,pipeline,command):
        from research.codepy.ssm_dispatch.programs import verifier_for,json_hash,CertificationError
        try:
            scope=self._signature(pipeline)
            if scope!=self._scope:self.clear();self._scope=scope
            if command in self._items:
                self.hits+=1;return self._items[command]
            self.misses+=1
            # The fast binary hash is only a lookup guard. A miss validates JSON
            # types/finiteness and fully certifies a private immutable snapshot.
            json_hash(pipeline.programs)
            entry=copy.deepcopy(pipeline.programs[command])
            verifier=verifier_for(pipeline.catalog,entry);candidate=verifier.restore(entry)
            if self._signature(pipeline)!=scope:raise CertificationError('Session/catalog changed during certification')
            if len(self._items)>=8:self._items.pop(next(iter(self._items)))
            self._items[command]=(verifier,candidate)
            return verifier,candidate
        except Exception:
            self.clear();raise

class Pipeline:
    """remember → observe → execute, with explicit schema/checkpoint compatibility."""
    def __init__(self, seed: int = 0):
        self.catalog = catalog(seed)
        self.params, self.encoder, self.text_params, self.text_encoder, v5hash, texthash = models()
        self.numeric_params = codec.load_params(seed)
        self.version = {'schema':SCHEMA, 'seed':seed, 'catalog':self.catalog.version,
                        'argument_library':self.catalog.argument_library_sha256,
                        'catalog_implementation':digest(ROOT/'research/codepy/ssm_dispatch/catalog.py'),
                        'v5':v5hash, 'text':texthash,
                        'implementation':digest(Path(__file__)),
                        'development':digest(Path(__file__).with_name('development.py')),
                        'hierarchy':digest(Path(__file__).with_name('hierarchy.py')),
                        'examples':digest(Path(__file__).with_name('examples.py')),
                        'programs':digest(Path(__file__).with_name('programs.py')),
                        'ieee':digest(Path(__file__).with_name('ieee_runtime.py')),
                        'numpy_runtime':digest(Path(__file__).with_name('numpy_runtime.py')),
                        'numeric':digest(ROOT/f'artifacts/codepy_ssm_diagnosis_20261031/guarded/repaired_weights_seed{seed}.npz')}
        self.semantic = ru.ingest_portion(self.params,core.init_stream_state(self.params,1),jnp.asarray([[self.encoder.mapping["<bos>"]]],jnp.int32))
        self.text_base = core.init_stream_state(self.text_params,1)
        self.text_carry = pointer.initial(self.text_params,1)
        _,self.text_base,self.text_carry = text_step(self.text_params,jnp.asarray([[self.text_encoder.mapping['<bos>']]],jnp.int32),self.text_base,self.text_carry)
        self.records = {}
        self.programs = {}
        self.task_bindings = {}
        self.last_text_token = ''
        self._trusted_version=dict(self.version)
        self._verifier_cache=VerifierCache()
        self._execution_cache=self._verifier_cache._items

    def __copy__(self):
        result=object.__new__(type(self));result.__dict__=self.__dict__.copy()
        result.version=dict(self.version);result._trusted_version=dict(self._trusted_version)
        result._verifier_cache=VerifierCache()
        result._execution_cache=result._verifier_cache._items
        return result

    def _text(self, text):
        ids = self.text_encoder.encode(text)
        for at in range(0,len(ids),512):
            logits, self.text_base, self.text_carry = text_step(self.text_params,jnp.asarray([ids[at:at+512]],jnp.int32),self.text_base,self.text_carry)
            self.last_text_token = self.text_encoder.vocab[int(jnp.argmax(logits[0]))]

    def observe(self, text: str):
        if not isinstance(text,str):
            raise TypeError('Text must be a string')
        self._verifier_cache.clear()
        self._text(text)
        ids = self.encoder.encode(text)
        for at in range(0,len(ids),1024):
            self.semantic = ru.ingest_portion(self.params,self.semantic,jnp.asarray([ids[at:at+1024]],jnp.int32))

    def remember(self, entity: str, property: str, handle: str, command: int, values):
        """Entity is the literal genitive form; property is the written fact wording.

        Handles are short literal values v5 can copy. Their meaning is declared by
        this write, not inferred from a task-name lookup table.
        """
        x = vector_arguments(values)
        if isinstance(handle,str):handle=handle.strip().lower()
        if not self.has_command(command):
            raise ValueError('Unknown command ID')
        if x.shape[1]!=self._command_arity(command):raise ValueError('Command/argument arity mismatch')
        for value in (entity,property,handle):
            if not isinstance(value,str) or not value.strip() or any(c in value for c in '.—\n\r'):
                raise ValueError('One literal fact field required')
        if len(self.encoder.encode(handle))>16 or self.encoder.mapping['<unk>'] in self.encoder.encode(handle):
            raise ValueError('Handle must fit the v5 value vocabulary and 16-token limit')
        width=x.shape[1];pages=(width+3)//4
        padded=np.zeros((len(x),pages*4),np.uint32);padded[:,:width]=x.copy().view(np.uint32)
        words=np.column_stack((np.full(len(x)*pages,command,np.uint32),padded.reshape(-1,4)))
        chunks = []
        for at in range(0,len(words),BATCH):
            batch = np.zeros((BATCH,5),np.uint32)
            batch[:len(words[at:at+BATCH])] = words[at:at+BATCH]
            state = core.init_stream_state(self.numeric_params,BATCH*3)
            state, _ = numeric_step(self.numeric_params,state,jnp.asarray(batch),jnp.asarray(True))
            chunks.append(state)
        record = {'rows':len(x),'width':width,'pages':pages,'sha256':hashlib.sha256(words.tobytes()).hexdigest(),
                  'page_sha256':[hashlib.sha256(memoryview(words).cast('B')[at:at+20]).hexdigest() for at in range(0,words.nbytes,20)], 'chunks':chunks}
        # Finish fallible model work before replacing the executable payload.
        self.observe(f'У {entity} {property} — {handle}.')
        self.records[handle] = record

    def recall(self, handle: str):
        if handle not in self.records:
            raise UnknownKey(handle)
        record = self.records[handle]
        recovered = []
        for at in range(0,len(record['chunks']),READ_CHUNKS):
            chunks = record['chunks'][at:at+READ_CHUNKS]
            state = chunks[0] if len(chunks)==1 else jax.tree_util.tree_map(lambda *a:jnp.concatenate(a,axis=0),*chunks)
            recovered.append(np.asarray(numeric_read(self.numeric_params,state)))
        words = np.concatenate(recovered)[:record['rows']*record.get('pages',1)]
        if hashlib.sha256(words.tobytes()).hexdigest()!=record['sha256']:
            raise ValueError('Neural command/argument payload failed integrity check')
        page_hashes=record.get('page_sha256')
        if page_hashes is not None:
            data=memoryview(words).cast('B')
            if len(page_hashes)!=len(words) or any(hashlib.sha256(data[20*i:20*(i+1)]).hexdigest()!=expected for i,expected in enumerate(page_hashes)):
                raise ValueError('Neural page failed integrity check')
        commands = words[:,0]
        if not np.all(commands==commands[0]) or not self.has_command(int(commands[0])):
            raise ValueError('Invalid recovered command ID')
        width=record.get('width',4)
        if width!=self._command_arity(int(commands[0])):raise ValueError('Stored command/argument arity mismatch')
        values=words[:,1:].copy().reshape(record['rows'],-1)[:,:width].copy().view(np.float32)
        return int(commands[0]),vector_arguments(values)

    def execute(self, question: str, *, verify: bool = True) -> Execution:
        if not isinstance(question,str) or not question.strip():
            raise ValueError('A semantic question is required')
        try:
            self._text(question)
            answer, _ = ru.neural_reply(self.params,self.encoder,self.semantic,question,max_tokens=20)
            handle = answer.strip().removesuffix('.').strip()
            command, values = self.recall(handle)
            execute=self._command_executor(command)
            output=execute(values,independent=False)
            if verify:
                reference=execute(values,independent=True)
                np.testing.assert_array_equal(output.view(np.uint32),reference.view(np.uint32))
            return Execution(handle,command,values,output,verify,self.last_text_token)
        except Exception:
            self._verifier_cache.clear();raise

    def has_command(self, command):
        return type(command) is int and (0<=command<96 or command in self.programs)

    def _command_arity(self, command):
        return 4 if command<96 else self.programs[command]['candidate'].get('arity',4)

    def _executor(self, command):
        if self.version!=self._trusted_version:
            self._verifier_cache.clear();raise ValueError('Session version integrity mismatch')
        if type(command) is int and 0<=command<96:
            def execute(values,independent=False):
                from research.codepy.ssm_dispatch import ieee_runtime, numpy_runtime
                for module,key in ((ieee_runtime,'ieee'),(numpy_runtime,'numpy_runtime')):
                    if digest(module.__file__)!=self.version[key]:raise ValueError('Numeric executor source integrity mismatch')
                with np.errstate(over='ignore',invalid='ignore'):
                    result=ieee_runtime.execute_catalog(self.catalog,command,values,independent=independent)
                if not np.isfinite(result).all():raise ValueError('Nonfinite execution result')
                return result
            return execute
        if not self.has_command(command):
            self._verifier_cache.clear();raise ValueError('Unknown command ID')
        verifier,candidate=self._verifier_cache.get(self,command)
        def execute(values,independent=False):
            from research.codepy.ssm_dispatch import ieee_runtime
            with np.errstate(over='ignore',invalid='ignore'):
                result=ieee_runtime.execute(verifier,candidate,values,independent=independent)
            if not np.isfinite(result).all():raise ValueError('Nonfinite execution result')
            return result
        return execute

    def _command_executor(self, command):
        return self._executor(command)

    def _execute_command(self, command, values, *, independent=False):
        return self._command_executor(command)(values,independent=independent)

    def develop(self, task=None, examples=None, *, X_train=None, Y_train=None, X_holdout=None, Y_holdout=None,
                entity, property, handle, values, populations=95, max_calls=None, use_macros=False, macro_commands=None, max_pair_ops=None):
        pairs=(X_train,Y_train,X_holdout,Y_holdout)
        if any(v is not None for v in pairs):
            if not all(v is not None for v in pairs) or examples is not None:
                raise ValueError('Provide all four paired arrays, without legacy examples')
            if use_macros:
                from research.codepy.ssm_dispatch.hierarchy import develop
                return develop(self,*pairs,label=task,entity=entity,property=property,handle=handle,values=values,populations=populations,max_calls=max_calls,macro_commands=macro_commands,max_pair_ops=max_pair_ops)
            if macro_commands is not None or max_pair_ops is not None:raise ValueError('macro_commands/max_pair_ops requires use_macros')
            from research.codepy.ssm_dispatch.examples import develop
            return develop(self,*pairs,label=task,entity=entity,property=property,handle=handle,values=values,populations=populations,max_calls=max_calls)
        if use_macros or macro_commands is not None or max_pair_ops is not None:raise ValueError('Macros require paired arrays')
        from research.codepy.ssm_dispatch.development import develop
        return develop(self,task,examples,entity=entity,property=property,handle=handle,values=values,populations=populations)

    def save(self, path):
        """Atomic state and certified extensions; base programs/weights stay external."""
        path = Path(path)
        if path.resolve().is_relative_to((ROOT/'artifacts').resolve()):
            raise ValueError('Store writable sessions outside immutable artifacts/')
        data = {}
        def state(prefix,value):
            for name,array in zip(value._fields,value):
                if array is not None:data[prefix+name]=np.asarray(array)
        state('semantic/',self.semantic);state('text/',self.text_base);state('pointer/',self.text_carry)
        metadata = {'programs':self.programs,'task_bindings':self.task_bindings,'version':self.version,'last_text_token':self.last_text_token,'records':{k:{'rows':v['rows'],'width':v.get('width',4),'pages':v.get('pages',1),'sha256':v['sha256'],**({'page_sha256':v['page_sha256']} if 'page_sha256' in v else {}),'chunks':len(v['chunks'])} for k,v in self.records.items()}}
        for i,(handle,r) in enumerate(self.records.items()):
            for j,chunk in enumerate(r['chunks']):state(f'record/{i}/{j}/',chunk)
        data['metadata_json']=np.asarray(json.dumps(metadata,ensure_ascii=False))
        path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.npz',delete=False) as f:
            tmp=Path(f.name)
        try:
            np.savez_compressed(tmp,**data);tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls,path, *, seed=0):
        current=cls(seed)
        with np.load(path,allow_pickle=False) as data:
            metadata=json.loads(str(data['metadata_json'].item()))
            legacy={k:v for k,v in current.version.items() if k not in ('development','examples','programs','hierarchy')}
            legacy.update(schema='semantic-codepy-session-v1',implementation=V1_IMPLEMENTATION)
            v2={k:v for k,v in current.version.items() if k not in ('examples','programs','hierarchy')}
            v2.update(schema='semantic-codepy-session-v2',implementation=V2_IMPLEMENTATION,development=V2_DEVELOPMENT)
            v3={k:v for k,v in current.version.items() if k!='hierarchy'}
            v3.update(schema='semantic-codepy-session-v3',implementation=V3_IMPLEMENTATION,examples=V3_EXAMPLES,programs=V3_PROGRAMS)
            v4={k:v for k,v in current.version.items() if k!='hierarchy'}
            v4.update(schema='semantic-codepy-session-v4',implementation='0d8f9e8dccb32c1146d851114be9bf79bf52c57ea0e705a73f4ecb6a25e48f1e',programs='5df3347630e28e1b2972d0820498513c13cbab37a7025f7f678f29e945851df3')
            v5=dict(current.version)
            v5.update(schema='semantic-codepy-session-v5',implementation='581e8592cdfa17f23dabcdba7f0707673233820bf27a3b22676fd2619bd838ca',programs='74d44469a2134d0c9568b5b92ffdb6aa8b56ce8dbafaecd591083da0205cc83d',hierarchy='a56a7173d7f7b37ae654ab0e2cca756517b716c11d125744c7f250e5417664e2')
            v6=dict(current.version);v6.update(schema='semantic-codepy-session-v6',implementation=V6_IMPLEMENTATION)
            v7=dict(current.version);v7.update(schema='semantic-codepy-session-v7',implementation=V7_IMPLEMENTATION)
            v7_read=dict(v7);v7_read.update(implementation=V7_READ_IMPLEMENTATION)
            v7_fused=dict(v7);v7_fused.update(implementation=V7_FUSED_IMPLEMENTATION)
            v7_page=dict(v7);v7_page.update(implementation=V7_PAGE_IMPLEMENTATION)
            v7_matrix=dict(v7);v7_matrix.update(implementation=V7_MATRIX_IMPLEMENTATION)
            previous=[{k:v for k,v in version.items() if k not in ('ieee','numpy_runtime')} for version in (legacy,v2,v3,v4,v5,v6,v7,v7_read,v7_fused,v7_page,v7_matrix)]
            subnormal={k:v for k,v in current.version.items() if k!='numpy_runtime'}
            subnormal.update(implementation='00d0d3349c049661fc146111eb8eb07b0180f8b2bd795cbbb921a8da166e3408',
                             ieee='0f0def9d56282dd02e3c856867125e5745703227a06739c69305d5e917462a1b')
            # Only the audited pre-release path relocation is compatible. Numeric
            # executors, model hashes, seeds and library versions must still match.
            original = dict(current.version, implementation='9860cf53e5963d18e11cabe98d6e18f02977a0db5b24ef155f7ed0d8acd12290')
            supported = [current.version, subnormal, *previous, original]
            for version in list(supported):
                historical = dict(version, catalog_implementation='0176d202e5d3e611b1af0533fb9131fbbc1508189d28cb69b6a25b7d17946130')
                if historical.get('programs') == current.version['programs']:
                    historical['programs'] = '137b856aee54d4e306c2536c677d884eb10878068c5116b066341a1605418ae7'
                supported.append(historical)
            if metadata['version'] not in supported:
                raise ValueError('Incompatible session schema, seed, library or checkpoint version')
            from research.codepy.ssm_dispatch.programs import verifier_for
            for key,entry in metadata.get('programs',{}).items():
                command=int(key)
                if command<96 or command>=2**32 or str(command)!=key:raise ValueError('Invalid program ID')
                verifier_for(current.catalog,entry).restore(entry)
                current.programs[command]=entry
            for task,command in metadata.get('task_bindings',{}).items():
                if type(command) is not int or command not in current.programs or current.programs[command]['candidate']['task']!=task:
                    raise ValueError('Invalid registered task binding')
                current.task_bindings[task]=command
            def state(prefix,prototype):
                arrays=[]
                for name,old in zip(prototype._fields,prototype):
                    new=data[prefix+name] if old is not None else None
                    if old is not None and (new.shape!=old.shape or new.dtype!=np.asarray(old).dtype or not np.isfinite(new).all()):
                        raise ValueError('Invalid session state array: '+prefix+name)
                    arrays.append(jnp.asarray(new) if new is not None else None)
                return type(prototype)(*arrays)
            current.semantic=state('semantic/',current.semantic)
            current.text_base=state('text/',current.text_base)
            current.text_carry=state('pointer/',current.text_carry)
            current.last_text_token=metadata['last_text_token']
            for i,(handle,r) in enumerate(metadata['records'].items()):
                width=r.get('width',4);pages=r.get('pages',1)
                if type(width) is not int or width<2 or type(pages) is not int or pages!=(width+3)//4:
                    raise ValueError('Invalid record pages/width')
                if type(r['rows']) is not int or r['rows']<1 or r['chunks']!=(r['rows']*pages+BATCH-1)//BATCH:
                    raise ValueError('Invalid record shape')
                if 'page_sha256' in r and (not isinstance(r['page_sha256'],list) or len(r['page_sha256'])!=r['rows']*pages or any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in r['page_sha256'])):
                    raise ValueError('Invalid page checksums')
                chunks=[state(f'record/{i}/{j}/',core.init_stream_state(current.numeric_params,BATCH*3)) for j in range(r['chunks'])]
                current.records[handle]={**r,'chunks':chunks}
        return current
