"""Task-free synthesis from paired arrays; heldout labels never reach search."""
from __future__ import annotations
import copy
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from research.codepy.ssm_dispatch.catalog import ROOT,digest
from research.codepy.ssm_dispatch.programs import vector_arguments as arguments
from research.codepy.ssm_dispatch.programs import Candidate,DevelopmentResult,CertificationError,blocks_for,input_hash,json_hash

sys.path.insert(0,str(ROOT/'research/codepy/subtask_discovery/transfer'))
from research.codepy.subtask_discovery.transfer.call_runtime import compose,numpy_compose

MODE='examples-v2'

def row_keys(x):
    # Numerically equal +0/-0 inputs must not evade the split check.
    canonical=np.array(x,copy=True);canonical[canonical==0]=0
    return [row.tobytes() for row in np.ascontiguousarray(canonical)]

def preserves(x,y):
    return np.all(np.sort(x.copy().view(np.uint32),axis=1)==np.sort(y.copy().view(np.uint32),axis=1),axis=1)

@dataclass(frozen=True)
class ExamplePairs:
    inputs: np.ndarray
    targets: np.ndarray
    def __post_init__(self):
        x=arguments(self.inputs).copy();y=arguments(self.targets).copy()
        if x.shape!=y.shape:raise ValueError('Input and target shapes must match')
        if not preserves(x,y).all():raise ValueError('Targets must preserve the input multiset')
        seen={}
        for key,target in zip(row_keys(x),row_keys(y)):
            if key in seen and seen[key]!=target:raise ValueError('Contradictory targets for one input')
            seen[key]=target
        x.flags.writeable=False;y.flags.writeable=False
        object.__setattr__(self,'inputs',x);object.__setattr__(self,'targets',y)
    def hashes(self):return {'inputs':input_hash(self.inputs),'targets':input_hash(self.targets),'rows':len(self.inputs)}
    def payload(self):return {'inputs':self.inputs.tolist(),'targets':self.targets.tolist()}
    @classmethod
    def restore(cls,value):return cls(np.asarray(value['inputs'],np.float32),np.asarray(value['targets'],np.float32))

def disjoint(train,holdout):
    if train.inputs.shape[1]!=holdout.inputs.shape[1]:raise CertificationError('TRAIN/HOLDOUT arity mismatch')
    if set(row_keys(train.inputs)) & set(row_keys(holdout.inputs)):
        raise CertificationError('Training/heldout input overlap')

def label_for(label,train):
    if label is None:return 'examples:'+json_hash(train.hashes())
    if not isinstance(label,str) or not label.strip():raise ValueError('A nonempty opaque label is required')
    return label

class BlockSearch:
    """Only TRAIN pairs enter this object/method, never an oracle or heldout set."""
    def __init__(self,catalog):
        self.catalog=catalog;self.blocks=blocks_for(catalog);self.tables=catalog.g.tables(self.blocks)

    def synthesize(self,train,*,label=None,populations=95,max_calls=None):
        if not isinstance(train,ExamplePairs):raise TypeError('Training pairs required')
        if type(populations) is not int or not 1<=populations<=95:raise ValueError('Budget must be 1..95 frontiers')
        n=train.inputs.shape[1];label=label_for(label,train)
        limit=max(6,n*(n-1)//2) if max_calls is None else max_calls
        if type(limit) is not int or not 1<=limit<=64:raise ValueError('Call limit must be 1..64')
        codes,layouts,returns=self.tables;targets=train.targets
        actions=np.arange(1+n*(n-1)*len(self.blocks),dtype=np.int32)
        @jax.jit
        def expand(states):
            return jax.vmap(lambda state:compose(jnp.asarray(actions[:,None]),state,codes,layouts,returns))(states)
        def score(outputs):
            scale=np.maximum(1.,np.max(np.abs(train.inputs),axis=1))[None,:,None]
            return np.mean(outputs!=targets[None],axis=(1,2))+np.mean((outputs/scale-targets[None]/scale)**2,axis=(1,2))
        states=train.inputs[None].copy();programs=[()];trace=[];evaluations=0;best_program=(0,);best=float(score(states)[0])
        if best==0:limit=0
        for depth in range(min(limit,populations)):
            outputs=np.asarray(expand(jnp.asarray(states)))
            flat=outputs.reshape(-1,*train.inputs.shape);scores=score(flat);evaluations+=len(flat)
            ordered=np.argsort(scores,kind='stable');winner=int(ordered[0]);parent,token=divmod(winner,len(actions))
            if float(scores[winner])<best:
                best=float(scores[winner]);best_program=programs[parent]+(int(actions[token]),)
            trace.append({'depth':depth+1,'candidates':len(flat),'best_objective':best})
            if best==0:break
            selected=[];seen=set()
            for index in ordered:
                signature=flat[index].tobytes()
                if signature in seen:continue
                seen.add(signature);selected.append(int(index))
                if len(selected)==32:break
            states=flat[selected].copy()
            programs=[programs[index//len(actions)]+(int(actions[index%len(actions)]),) for index in selected]
        candidate=Candidate(self.catalog.seed,label,best_program,n)
        exact=bool(np.array_equal(self._execute(candidate,train.inputs),targets))
        attempt={'evaluations':evaluations,'training_objective':best,'train_exact':exact,'trace':trace}
        return candidate,{'algorithm':'behavioral_beam','beam_width':32,'max_calls':max(1,limit),'arity':n,'evaluations':evaluations,'attempts':[attempt],
                          'training_pairs':train.hashes(),'candidate_sha256':json_hash(candidate.payload()),'source_blocks_sha256':json_hash(self.blocks),
                          'heldout_available_to_search':False,'task_oracle_used':False,'previous_target_champions_used':False,'new_gradient_updates':0}

    def _execute(self,candidate,x):
        return np.asarray(compose(jnp.asarray([candidate.program]),jnp.asarray(x),*self.tables))[0]

class Verifier:
    def __init__(self,catalog):
        self.catalog=catalog;self.blocks=blocks_for(catalog);self.tables=catalog.g.tables(self.blocks)

    def validate(self,candidate):
        if not isinstance(candidate,Candidate) or type(candidate.seed) is not int or candidate.seed!=self.catalog.seed:raise CertificationError('Candidate seed/type mismatch')
        if not isinstance(candidate.task,str) or not candidate.task.strip():raise CertificationError('Invalid opaque label')
        if type(candidate.arity) is not int or candidate.arity<2:raise CertificationError('Invalid candidate arity')
        if type(candidate.program) is not tuple or not 1<=len(candidate.program)<=64 or any(type(v) is not int or not 0<=v<=candidate.arity*(candidate.arity-1)*len(self.blocks) for v in candidate.program):raise CertificationError('Invalid bounded pair-call program')

    def execute(self,candidate,x,*,independent=False):
        self.validate(candidate);x=arguments(x)
        if x.shape[1]!=candidate.arity:raise CertificationError('Program/argument arity mismatch')
        if independent:return numpy_compose(candidate.program,x,self.blocks)
        return np.asarray(compose(jnp.asarray([candidate.program]),jnp.asarray(x),*self.tables))[0]

    def certify(self,candidate,train,holdout):
        self.validate(candidate);disjoint(train,holdout)
        groups={}
        for name,pairs in [('train',train),('holdout',holdout)]:
            native=self.execute(candidate,pairs.inputs);reference=self.execute(candidate,pairs.inputs,independent=True)
            if not np.array_equal(native.view(np.uint32),reference.view(np.uint32)):raise CertificationError('JAX/NumPy disagreement')
            accepted=np.all(native==pairs.targets,axis=1);kept=preserves(pairs.inputs,native)
            groups[name]={'cases':len(native),'exact':int(accepted.sum()),'multisets_preserved':int(kept.sum()),**pairs.hashes()}
            if not accepted.all() or not kept.all():raise CertificationError('Candidate failed supplied '+name+' pairs')
        return {'mode':MODE,'passed':True,'candidate_sha256':json_hash(candidate.payload()),'source_blocks_sha256':json_hash(self.blocks),
                'verifier_source_sha256':digest(__file__),'arity':candidate.arity,'calls':len(candidate.program),'groups':groups,'cases':len(holdout.inputs),'train_heldout_overlap':0,'jax_numpy_exact':True,
                'task_oracle_used':False,'finite_examples_not_universal_proof':True}

    def restore(self,entry):
        try:
            value=entry['candidate'];candidate=Candidate(value['seed'],value['task'],tuple(value['program']),value.get('arity',4))
            train=ExamplePairs.restore(entry['verification']['train']);holdout=ExamplePairs.restore(entry['verification']['holdout'])
            certificate=self.certify(candidate,train,holdout)
            if entry.get('mode')!=MODE or certificate!=entry['certificate']:raise CertificationError('Stored example certificate mismatch')
            return candidate
        except (KeyError,TypeError) as exc:raise CertificationError('Invalid example program') from exc


def develop(pipeline,X_train,Y_train,X_holdout,Y_holdout,*,label=None,entity,property,handle,values,populations=95,max_calls=None):
    train=ExamplePairs(X_train,Y_train);holdout=ExamplePairs(X_holdout,Y_holdout);disjoint(train,holdout)
    label=label_for(label,train);execution=arguments(values)
    if execution.shape[1]!=train.inputs.shape[1]:raise ValueError('Execution/TRAIN arity mismatch')
    matches=[c for c in range(96) if np.array_equal(pipeline.catalog.execute(c,train.inputs),train.targets)] if train.inputs.shape[1]==4 else []
    existing=pipeline.task_bindings.get(label);candidate=None;search=None
    if existing in pipeline.programs and pipeline.programs[existing].get('mode')==MODE:
        v=pipeline.programs[existing]['candidate'];previous=Candidate(v['seed'],v['task'],tuple(v['program']),v.get('arity',4))
        if previous.arity==train.inputs.shape[1] and np.array_equal(Verifier(pipeline.catalog).execute(previous,train.inputs),train.targets):candidate=previous
    evidence={'mode':MODE,'gap':candidate is None,'gap_kind':'unknown_label' if existing is None else 'incomplete_solution','base_commands_matching_training':len(matches),'matching_base_commands':matches,
              'events':['gap_detected'] if candidate is None else ['registered_program_found'],'label_used_as_oracle':False}
    if candidate is None:
        options={} if max_calls is None else {'max_calls':max_calls}
        candidate,search=BlockSearch(pipeline.catalog).synthesize(train,label=label,populations=populations,**options)
        evidence['search']=search;evidence['events']+=['synthesized','candidate_frozen']
    try:
        if candidate.task!=label:raise CertificationError('Synthesized label mismatch')
        certificate=Verifier(pipeline.catalog).certify(candidate,train,holdout)
    except CertificationError as exc:
        evidence['events'].append('rejected');evidence['reason']=str(exc)
        return DevelopmentResult('rejected',None,evidence)
    evidence['events'].append('independently_certified')
    staged=copy.copy(pipeline);staged.records=dict(pipeline.records);staged.programs=dict(pipeline.programs);staged.task_bindings=dict(pipeline.task_bindings)
    command=existing if search is None else max([95,*staged.programs])+1
    if search is not None:
        staged.programs[command]={'mode':MODE,'candidate':candidate.payload(),'certificate':certificate,'search':search,'verification':{'train':train.payload(),'holdout':holdout.payload()}}
        staged.task_bindings[label]=command
    staged.remember(entity,property,handle,command,values)
    pipeline.__dict__.update(staged.__dict__)
    evidence['events']+=['catalog_registered','kv_and_v5_bound'];evidence['certificate']=certificate
    return DevelopmentResult('accepted' if search is not None else 'already_registered',command,evidence)
