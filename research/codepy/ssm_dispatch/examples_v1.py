"""Task-free synthesis from paired arrays; heldout labels never reach search."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from research.codepy.ssm_dispatch.catalog import arguments,digest
from research.codepy.ssm_dispatch.programs import Candidate,DevelopmentResult,CertificationError,blocks_for,input_hash,json_hash

MODE='examples-v1'

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

    def synthesize(self,train,*,label=None,populations=95):
        if not isinstance(train,ExamplePairs):raise TypeError('Training pairs required')
        if type(populations) is not int or not 1<=populations<=95:raise ValueError('Budget must be 1..95 populations')
        label=label_for(label,train);codes,layouts,returns=self.tables;targets=jnp.asarray(train.targets)
        @jax.jit
        def scorer(programs,inputs):
            out=self.catalog.g.compose(programs,inputs,codes,layouts,returns)
            scale=jnp.maximum(1.,jnp.max(jnp.abs(inputs),axis=1))[None,:,None]
            error=jnp.mean((out/scale-targets[None]/scale)**2,axis=(1,2))
            loss=jnp.mean(out!=targets[None],axis=(1,2))+error
            return loss,error
        attempts=[];best=None;candidate=None
        # Restarts depend ONLY on exact TRAIN fit. No candidate is sent to the verifier in this loop.
        for restart in range(4):
            found=self.catalog.g.vm.search(self.catalog.seed+100003*restart,train.inputs,alphabet=list(range(1+12*len(self.blocks))),length=6,population=256,generations=populations-1,scorer=scorer)
            if found['program'] is None:continue
            proposed=Candidate(self.catalog.seed,label,tuple(found['program']))
            output=np.asarray(self.catalog.g.compose(jnp.asarray([proposed.program]),jnp.asarray(train.inputs),*self.tables))[0]
            exact=bool(np.array_equal(output,train.targets))
            attempts.append({'restart':restart,'evaluations':found['evaluations'],'training_objective':found['training_objective'],'train_exact':exact,'trace':found['trace']})
            if best is None or found['training_objective']<best:
                best=found['training_objective'];candidate=proposed
            if exact:break
        if candidate is None:raise CertificationError('Search produced no finite candidate')
        return candidate,{'evaluations':sum(a['evaluations'] for a in attempts),'attempts':attempts,'training_pairs':train.hashes(),'candidate_sha256':json_hash(candidate.payload()),
                          'source_blocks_sha256':json_hash(self.blocks),'heldout_available_to_search':False,'task_oracle_used':False,'previous_target_champions_used':False,'new_gradient_updates':0}

class Verifier:
    def __init__(self,catalog):
        self.catalog=catalog;self.blocks=blocks_for(catalog);self.tables=catalog.g.tables(self.blocks)

    def validate(self,candidate):
        if not isinstance(candidate,Candidate) or type(candidate.seed) is not int or candidate.seed!=self.catalog.seed:raise CertificationError('Candidate seed/type mismatch')
        if not isinstance(candidate.task,str) or not candidate.task.strip():raise CertificationError('Invalid opaque label')
        if type(candidate.program) is not tuple or len(candidate.program)!=6 or any(type(v) is not int or not 0<=v<=12*len(self.blocks) for v in candidate.program):raise CertificationError('Invalid six-call program')

    def execute(self,candidate,x,*,independent=False):
        self.validate(candidate);x=arguments(x)
        if independent:return self.catalog.g.numpy_compose(candidate.program,x,self.blocks)
        return np.asarray(self.catalog.g.compose(jnp.asarray([candidate.program]),jnp.asarray(x),*self.tables))[0]

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
                'verifier_source_sha256':digest(__file__),'groups':groups,'cases':len(holdout.inputs),'train_heldout_overlap':0,'jax_numpy_exact':True,
                'task_oracle_used':False,'finite_examples_not_universal_proof':True}

    def restore(self,entry):
        try:
            value=entry['candidate'];candidate=Candidate(value['seed'],value['task'],tuple(value['program']))
            train=ExamplePairs.restore(entry['verification']['train']);holdout=ExamplePairs.restore(entry['verification']['holdout'])
            certificate=self.certify(candidate,train,holdout)
            if entry.get('mode')!=MODE or certificate!=entry['certificate']:raise CertificationError('Stored example certificate mismatch')
            return candidate
        except (KeyError,TypeError) as exc:raise CertificationError('Invalid example program') from exc


def develop(pipeline,X_train,Y_train,X_holdout,Y_holdout,*,label=None,entity,property,handle,values,populations=95):
    train=ExamplePairs(X_train,Y_train);holdout=ExamplePairs(X_holdout,Y_holdout);disjoint(train,holdout)
    label=label_for(label,train);arguments(values)
    matches=[c for c in range(96) if np.array_equal(pipeline.catalog.execute(c,train.inputs),train.targets)]
    existing=pipeline.task_bindings.get(label);candidate=None;search=None
    if existing in pipeline.programs and pipeline.programs[existing].get('mode')==MODE:
        v=pipeline.programs[existing]['candidate'];previous=Candidate(v['seed'],v['task'],tuple(v['program']))
        if np.array_equal(Verifier(pipeline.catalog).execute(previous,train.inputs),train.targets):candidate=previous
    evidence={'mode':MODE,'gap':candidate is None,'gap_kind':'unknown_label' if existing is None else 'incomplete_solution','base_commands_matching_training':len(matches),'matching_base_commands':matches,
              'events':['gap_detected'] if candidate is None else ['registered_program_found'],'label_used_as_oracle':False}
    if candidate is None:
        candidate,search=BlockSearch(pipeline.catalog).synthesize(train,label=label,populations=populations)
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
