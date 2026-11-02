"""TRAIN-only composition of immutable certified skills on distinct tape indices."""
from __future__ import annotations
import copy
import itertools
import numpy as np
import jax.numpy as jnp
from research.codepy.ssm_dispatch import examples as flat
from research.codepy.ssm_dispatch.catalog import digest
from research.codepy.ssm_dispatch.programs import Candidate, CertificationError, DevelopmentResult, json_hash, verifier_for

MODE='hierarchical-v2'

def candidate_of(entry):
    v=entry['candidate']
    return Candidate(v['seed'],v['task'],tuple(v['program']),v.get('arity',4))

def dependency_mode(verifier):
    return verifier.__class__.__module__ in ('research.codepy.ssm_dispatch.examples','research.codepy.ssm_dispatch.examples_v1','research.codepy.ssm_dispatch.development')

def mapped(program, inner, indices, outer):
    pairs=list(itertools.permutations(range(inner),2));addresses={p:i for i,p in enumerate(itertools.permutations(range(outer),2))}
    result=[]
    for token in program:
        if token:
            block,pair=divmod(token-1,len(pairs));a,b=pairs[pair]
            result.append(1+block*outer*(outer-1)+addresses[indices[a],indices[b]])
    return result

class Verifier(flat.Verifier):
    def __init__(self,catalog,language,expected_certificate=None,depth=0):
        super().__init__(catalog)
        self.language=copy.deepcopy(language);self.dependencies=[]
        if depth>=8:raise CertificationError('Macro nesting limit exceeded')
        try:
            n=language['arity'];actions=language['actions'];deps=language['dependencies']
            if type(n) is not int or not 2<=n<=64 or not actions or len(actions)>8192:raise CertificationError('Invalid macro language bounds')
            if expected_certificate is not None and expected_certificate['language_sha256']!=json_hash(language):raise CertificationError('Macro language integrity mismatch')
            for dependency in deps:
                entry=dependency['entry']
                if type(dependency['command']) is not int or dependency['command']<96 or dependency['sha256']!=json_hash(entry):raise CertificationError('Invalid certified dependency')
                if entry.get('mode')==MODE:
                    verifier=Verifier(catalog,entry['language'],entry['certificate'],depth+1)
                elif entry.get('mode')=='hierarchical-v1':
                    from research.codepy.ssm_dispatch.hierarchy_v1 import Verifier as PreviousVerifier
                    verifier=PreviousVerifier(catalog,entry['language'],entry['certificate'],depth+1)
                else:verifier=verifier_for(catalog,entry)
                skill=verifier.restore(entry)
                if skill.seed!=catalog.seed:raise CertificationError('Cross-seed macro dependency')
                self.dependencies.append((skill,verifier))
            self.expansions=[]
            for action in actions:
                kind=action['kind'];indices=action.get('indices',[])
                if kind=='noop':
                    if action!={'kind':'noop'}:raise CertificationError('Malformed no-op')
                    self.expansions.append([]);continue
                if not isinstance(indices,list) or any(type(i) is not int or not 0<=i<n for i in indices) or len(indices)!=len(set(indices)):raise CertificationError('Macro indices must be distinct and in range')
                if kind=='pair':
                    if len(indices)!=2 or type(action['block']) is not int or not 0<=action['block']<len(self.blocks):raise CertificationError('Invalid pair block')
                    self.expansions.append(mapped([1+action['block']*2],2,indices,n))
                elif kind=='macro':
                    d=action['dependency']
                    if type(d) is not int or not 0<=d<len(self.dependencies):raise CertificationError('Invalid macro dependency index')
                    skill,verifier=self.dependencies[d]
                    if len(indices)!=skill.arity:raise CertificationError('Macro/index arity mismatch')
                    if deps[d]['entry'].get('mode') in (MODE,'hierarchical-v1'):program=verifier.flatten(skill)
                    elif dependency_mode(self.dependencies[d][1]):program=skill.program
                    else:raise CertificationError('Macro must use the portable pair-call ABI')
                    self.expansions.append(mapped(program,skill.arity,indices,n))
                else:raise CertificationError('Unknown macro action')
        except (KeyError,IndexError,TypeError) as exc:raise CertificationError('Malformed macro language') from exc

    def validate(self,candidate):
        if not isinstance(candidate,Candidate) or type(candidate.seed) is not int or candidate.seed!=self.catalog.seed or not isinstance(candidate.task,str) or not candidate.task.strip():raise CertificationError('Invalid hierarchical candidate')
        if candidate.arity!=self.language['arity'] or type(candidate.program) is not tuple or not 1<=len(candidate.program)<=64 or any(type(t) is not int or not 0<=t<len(self.expansions) for t in candidate.program):raise CertificationError('Invalid hierarchical calls')
        bound=self.language.get('max_pair_ops',4096)
        if type(bound) is not int or not 1<=bound<=4096 or self.unrolled_pair_ops(candidate)>bound:raise CertificationError('Expanded pair operation budget exceeded')

    def unrolled_pair_ops(self,candidate):
        return sum(len(self.expansions[t]) for t in candidate.program)

    def flatten(self,candidate):
        self.validate(candidate)
        return tuple(v for t in candidate.program for v in self.expansions[t]) or (0,)

    def execute(self,candidate,x,*,independent=False):
        self.validate(candidate);x=flat.arguments(x)
        if x.shape[1]!=candidate.arity:raise CertificationError('Program/argument arity mismatch')
        if not independent:
            return np.asarray(flat.compose(jnp.asarray([self.flatten(candidate)]),jnp.asarray(x),*self.tables))[0]
        # This interpreter does not use the JAX flattening/remapping path.
        result=x.copy()
        for token in candidate.program:
            action=self.language['actions'][token];kind=action['kind']
            if kind=='noop':continue
            indices=action['indices'];local=result[:,indices].copy()
            if kind=='pair':
                local=flat.numpy_compose((1+action['block']*2,),local,self.blocks)
            else:
                skill,verifier=self.dependencies[action['dependency']]
                local=verifier.execute(skill,local,independent=True)
            result[:,indices]=local
        return result

    def certify(self,candidate,train,holdout):
        certificate=super().certify(candidate,train,holdout)
        certificate.update(mode=MODE,verifier_source_sha256=digest(__file__),language_sha256=json_hash(self.language),
                           expanded_pair_calls=self.unrolled_pair_ops(candidate),unrolled_pair_ops=self.unrolled_pair_ops(candidate),
                           macro_calls=sum(self.language['actions'][t]['kind']=='macro' for t in candidate.program))
        return certificate

    def restore(self,entry):
        candidate=candidate_of(entry)
        certificate=self.certify(candidate,flat.ExamplePairs.restore(entry['verification']['train']),flat.ExamplePairs.restore(entry['verification']['holdout']))
        if entry.get('mode')!=MODE or certificate!=entry['certificate']:raise CertificationError('Stored macro certificate mismatch')
        return candidate

class BlockSearch:
    def __init__(self,pipeline,macro_commands=None):
        self.catalog=pipeline.catalog;self.dependencies=[]
        commands=sorted(pipeline.programs) if macro_commands is None else list(macro_commands)
        if len(commands)!=len(set(commands)):raise ValueError('Duplicate macro commands')
        for command in commands:
            if type(command) is not int or command not in pipeline.programs:raise CertificationError('Macro is not a certified session command')
            entry=copy.deepcopy(pipeline.programs[command]);v=verifier_for(self.catalog,entry);v.restore(entry)
            if entry.get('mode') not in (None,'examples-v1','examples-v2','hierarchical-v1',MODE):raise CertificationError('Unsupported macro ABI')
            self.dependencies.append({'command':command,'sha256':json_hash(entry),'entry':entry})

    def synthesize(self,train,*,label=None,populations=95,max_calls=None,max_pair_ops=None):
        n=train.inputs.shape[1];label=flat.label_for(label,train);limit=32 if max_calls is None else max_calls
        budget=max(6,n*(n-1)//2) if max_pair_ops is None else max_pair_ops
        if type(limit) is not int or not 1<=limit<=64 or type(populations) is not int or not 1<=populations<=95:raise ValueError('Invalid search budget')
        if type(budget) is not int or not 1<=budget<=4096:raise ValueError('Pair operation budget must be 1..4096')
        actions=[{'kind':'noop'}]+[{'kind':'pair','block':b,'indices':list(p)} for b in range(2) for p in itertools.permutations(range(n),2)]
        macro_start=len(actions)
        for d,dependency in enumerate(self.dependencies):
            k=candidate_of(dependency['entry']).arity
            if k>n:continue
            # shortcut: bounded placement enumeration; widen this budget for larger tapes.
            placements=list(itertools.islice(itertools.combinations(range(n),k),4096))
            placements=sorted(set(placements)|{tuple(range(i,i+k)) for i in range(n-k+1)})
            actions.extend({'kind':'macro','dependency':d,'indices':list(p)} for p in placements)
        language={'arity':n,'dependencies':self.dependencies,'actions':actions,'max_pair_ops':budget}
        verifier=Verifier(self.catalog,language)
        scale=np.maximum(1.,np.max(np.abs(train.inputs),axis=1))[None,:,None]
        def score(outputs):return np.mean(outputs!=train.targets[None],axis=(1,2))+np.mean((outputs/scale-train.targets[None]/scale)**2,axis=(1,2))
        def expand(states,expansions):
            results=[];width=max(1,max(map(len,expansions)))
            for start in range(0,len(expansions),64):
                group=expansions[start:start+64];programs=np.zeros((64,width),np.int32)
                for i,program in enumerate(group):programs[i,:len(program)]=program
                outputs=[]
                for state in states:
                    outputs.append(np.asarray(flat.compose(jnp.asarray(programs),jnp.asarray(state),*verifier.tables))[:len(group)])
                results.append(np.stack(outputs,axis=0))
            return np.concatenate(results,axis=1)
        # A separate placement quota for each source prevents costly skills from
        # displacing lightweight skills before cost-aware search even starts.
        prescreen=0;retained=list(range(macro_start))
        for d in range(len(self.dependencies)):
            indices=[i for i in range(macro_start,len(actions)) if actions[i]['dependency']==d]
            if len(indices)>128:
                output=expand(train.inputs[None],[verifier.expansions[i] for i in indices])[0]
                rank=np.argsort(score(output),kind='stable')[:128];prescreen+=len(output)
                indices=[indices[int(i)] for i in rank]
            retained.extend(indices)
        actions=[actions[i] for i in retained];language={**language,'actions':actions};verifier=Verifier(self.catalog,language)
        costs=np.asarray([len(e) for e in verifier.expansions],np.int32)
        groups={int(cost):np.flatnonzero(costs==cost) for cost in sorted(set(costs)) if cost>0}
        frontiers={0:(train.inputs[None].copy(),[()])};best=float(score(train.inputs[None])[0]);winner=(0,);trace=[];evaluations=prescreen
        # Frontiers are indexed by expanded cost, so a macro pays for all its
        # inner operations instead of jumping ahead at the price of one call.
        seen={train.inputs.tobytes():(0,0)}
        for cost in range(1,min(budget,populations)+1) if best else []:
            batches=[];programs=[]
            for action_cost,tokens in groups.items():
                if cost-action_cost not in frontiers:continue
                states,prefixes=frontiers[cost-action_cost]
                selected=[i for i,p in enumerate(prefixes) if len(p)<limit]
                if not selected:continue
                states=states[selected];prefixes=[prefixes[i] for i in selected]
                output=expand(states,[verifier.expansions[int(t)] for t in tokens])
                batches.append(output.reshape(-1,*train.inputs.shape))
                programs.extend(prefix+(int(token),) for prefix in prefixes for token in tokens)
            if not batches:continue
            output=np.concatenate(batches);scores=score(output);evaluations+=len(output)
            lengths=np.asarray([len(p) for p in programs]);rank=np.lexsort((np.arange(len(scores)),lengths,scores))
            chosen=[];local=set()
            for i in rank:
                i=int(i);signature=output[i].tobytes();old=seen.get(signature)
                if signature in local or (old is not None and old[0]<=cost and old[1]<=lengths[i]):continue
                local.add(signature);chosen.append(i)
                if len(chosen)==64:break
            if not chosen:continue
            index=chosen[0]
            if float(scores[index])<best:
                best=float(scores[index]);winner=programs[index]
            trace.append({'unrolled_pair_ops':cost,'best_objective':best,'candidates':len(output),'best_top_level_calls':len(winner)})
            if best==0:break
            states=output[chosen].copy();prefixes=[programs[i] for i in chosen]
            frontiers[cost]=(states,prefixes)
            for i in chosen:seen[output[i].tobytes()]=(cost,len(programs[i]))
        candidate=Candidate(self.catalog.seed,label,winner,n)
        return candidate,{'algorithm':'cost_layered_behavioral_beam','language':language,'beam_width':64,'max_calls':limit,'max_pair_ops':budget,
                          'unrolled_pair_ops':verifier.unrolled_pair_ops(candidate),'evaluations':evaluations,'trace':trace,
                          'training_pairs':train.hashes(),'train_exact':best==0,'heldout_available_to_search':False,'task_oracle_used':False,'new_gradient_updates':0,
                          'macro_placements_before_prescreen':sum(a['kind']=='macro' for a in verifier.language['actions']) if not prescreen else prescreen,
                          'macro_placements_retained':len(actions)-macro_start}

def develop(pipeline,X_train,Y_train,X_holdout,Y_holdout,*,label=None,entity,property,handle,values,populations=95,max_calls=None,macro_commands=None,max_pair_ops=None):
    train=flat.ExamplePairs(X_train,Y_train);holdout=flat.ExamplePairs(X_holdout,Y_holdout);flat.disjoint(train,holdout)
    label=flat.label_for(label,train);values=flat.arguments(values)
    if max_pair_ops is not None and (type(max_pair_ops) is not int or not 1<=max_pair_ops<=4096):raise ValueError('Pair operation budget must be 1..4096')
    if values.shape[1]!=train.inputs.shape[1]:raise ValueError('Execution/TRAIN arity mismatch')
    existing=pipeline.task_bindings.get(label);entry=pipeline.programs.get(existing)
    reuse=entry is not None and entry.get('mode')==MODE
    if reuse:
        candidate=candidate_of(entry);verifier=verifier_for(pipeline.catalog,entry)
        reuse=candidate.arity==train.inputs.shape[1] and (max_pair_ops is None or verifier.unrolled_pair_ops(candidate)<=max_pair_ops) and np.array_equal(verifier.execute(candidate,train.inputs),train.targets)
    if reuse:search=None;language=entry['language']
    else:
        candidate,search=BlockSearch(pipeline,macro_commands).synthesize(train,label=label,populations=populations,max_calls=max_calls,max_pair_ops=max_pair_ops)
        language=search['language'];verifier=Verifier(pipeline.catalog,language)
    evidence={'mode':MODE,'gap':not reuse,'base_commands_matching_training':0 if candidate.arity!=4 else sum(np.array_equal(pipeline.catalog.execute(i,train.inputs),train.targets) for i in range(96)),
              'events':['registered_program_found'] if reuse else ['gap_detected','synthesized','candidate_frozen'],'search':search,'label_used_as_oracle':False}
    try:
        if candidate.task!=label:raise CertificationError('Synthesized label mismatch')
        certificate=verifier.certify(candidate,train,holdout)
    except CertificationError as exc:
        evidence['events'].append('rejected');evidence['reason']=str(exc)
        return DevelopmentResult('rejected',None,evidence)
    staged=copy.copy(pipeline);staged.records=dict(pipeline.records);staged.programs=dict(pipeline.programs);staged.task_bindings=dict(pipeline.task_bindings)
    command=existing if reuse else max([95,*staged.programs])+1
    if not reuse:
        staged.programs[command]={'mode':MODE,'language':language,'candidate':candidate.payload(),'certificate':certificate,'search':{k:v for k,v in search.items() if k!='language'},
                                  'verification':{'train':train.payload(),'holdout':holdout.payload()}}
        staged.task_bindings[label]=command
    staged.remember(entity,property,handle,command,values);pipeline.__dict__.update(staged.__dict__)
    evidence['events']+=['independently_certified','catalog_registered','kv_and_v5_bound'];evidence['certificate']=certificate
    return DevelopmentResult('already_registered' if reuse else 'accepted',command,evidence)
