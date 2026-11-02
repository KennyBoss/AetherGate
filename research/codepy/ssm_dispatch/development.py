"""Bounded TRAIN-only block synthesis and a separate, post-freeze verifier.

No target champions, gradients, heldout-driven retries or executable templates.
Task names select declarative contracts, not instruction sequences.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import sys
import numpy as np
import jax
import jax.numpy as jnp
from research.codepy.ssm_dispatch.catalog import ROOT, arguments, digest

sys.path.insert(0,str(ROOT/'research/codepy/subtask_discovery/transfer'))
from research.codepy.subtask_discovery.transfer import tasks as contracts
SUPPORTED = ('minmax4','rotate_right4','partition4','pairwise_order4')

from research.codepy.ssm_dispatch.programs import (CertificationError, json_hash, input_hash, blocks_for, Candidate, DevelopmentResult)

def validate_task(task):
    if task not in SUPPORTED:
        raise ValueError('Unknown declarative contract: '+str(task))

class BlockSearch:
    """Receives only the training examples and contract; never the heldout pack."""
    def __init__(self,catalog):
        self.catalog=catalog
        self.blocks=blocks_for(catalog)
        self.tables=catalog.g.tables(self.blocks)

    def synthesize(self,task,examples,*,populations=95):
        validate_task(task);x=arguments(examples)
        if len(x)!=64:raise ValueError('Exactly 64 training examples required')
        if type(populations) is not int or not 1<=populations<=95:
            raise ValueError('Search budget must be 1..95 populations')
        codes,layouts,returns=self.tables
        @jax.jit
        def scorer(genomes,inputs):
            outputs=self.catalog.g.compose(genomes,inputs,codes,layouts,returns)
            return contracts.losses(outputs,inputs,task)
        found=self.catalog.g.vm.search(self.catalog.seed,x,alphabet=list(range(1+12*len(self.blocks))),length=6,population=256,generations=populations-1,scorer=scorer)
        candidate=Candidate(self.catalog.seed,task,tuple(found['program']))
        evidence={'evaluations':found['evaluations'],'populations':populations,'training_objective':found['training_objective'],'training_sha256':input_hash(x),
                  'first_perfect':found['first_perfect'],'candidate_sha256':json_hash(candidate.payload()),'source_blocks_sha256':json_hash(self.blocks),
                  'heldout_available_to_search':False,'previous_target_champions_used':False,'new_gradient_updates':0,'trace':found['trace']}
        return candidate,evidence

class Verifier:
    """Certification starts only after a concrete candidate is frozen."""
    def __init__(self,catalog):
        self.catalog=catalog
        self.blocks=blocks_for(catalog)
        self.tables=catalog.g.tables(self.blocks)

    def validate(self,candidate):
        if not isinstance(candidate,Candidate) or type(candidate.seed) is not int or candidate.seed!=self.catalog.seed:
            raise CertificationError('Candidate seed/type mismatch')
        validate_task(candidate.task)
        if type(candidate.program) is not tuple or len(candidate.program)!=6 or any(type(v) is not int or not 0<=v<=12*len(self.blocks) for v in candidate.program):
            raise CertificationError('Invalid six-call candidate')

    def execute(self,candidate,x,*,independent=False):
        self.validate(candidate);x=arguments(x)
        if independent:return self.catalog.g.numpy_compose(candidate.program,x,self.blocks)
        return np.asarray(self.catalog.g.compose(jnp.asarray([candidate.program]),jnp.asarray(x),*self.tables))[0]

    def certify(self,candidate,*,training_examples=None):
        self.validate(candidate)
        # Full package is created here, after synthesis has returned its one frozen winner.
        groups=self.catalog.g.vm.cases(4,20261101+SUPPORTED.index(candidate.task))
        training_rows=set() if training_examples is None else {row.tobytes() for row in np.ascontiguousarray(arguments(training_examples))}
        rows={};passed=True
        for name,x in groups.items():
            if training_rows & {row.tobytes() for row in np.ascontiguousarray(x)}:
                raise CertificationError('Training/heldout input overlap')
            native=self.execute(candidate,x);independent=self.execute(candidate,x,independent=True)
            if not np.array_equal(native,independent):raise CertificationError('JAX/NumPy disagreement')
            accepted,preserved=contracts.numpy_accept(independent,x,candidate.task)
            rows[name]={'cases':len(x),'accepted':int(accepted.sum()),'values_preserved':int(preserved.sum()),'input_sha256':input_hash(x)}
            passed &= bool(accepted.all())
        certificate={'passed':passed,'cases':sum(r['cases'] for r in rows.values()),'groups':rows,'candidate_sha256':json_hash(candidate.payload()),
                     'source_blocks_sha256':json_hash(self.blocks),'contract_source_sha256':digest(contracts.__file__),'jax_numpy_exact':True,'train_heldout_overlap':0,
                     'finite_float32_contract_not_universal_proof':True}
        if not passed:raise CertificationError('Candidate failed independent full-package certification')
        if certificate['cases']!=7041:raise CertificationError('Incomplete checking package')
        return certificate

    def restore(self,entry):
        try:
            value=entry['candidate']
            candidate=Candidate(value['seed'],value['task'],tuple(value['program']))
            certificate=self.certify(candidate)
            if certificate!=entry['certificate']:raise CertificationError('Stored certificate mismatch')
            return candidate
        except (KeyError,TypeError) as exc:
            raise CertificationError('Invalid stored program') from exc


def develop(pipeline,task,examples,*,entity,property,handle,values,populations=95):
    """One attempt, one certificate, atomic publish of code + both memory bindings."""
    import copy
    validate_task(task);x=arguments(examples);arguments(values)
    if len(x)!=64:raise ValueError('Exactly 64 training examples required')
    existing=pipeline.task_bindings.get(task)
    matches=[]
    for command in range(96):
        accepted,_=contracts.numpy_accept(pipeline.catalog.execute(command,x),x,task)
        if accepted.all():matches.append(command)
    if existing is not None:
        accepted,_=contracts.numpy_accept(pipeline._execute_command(existing,x),x,task)
        if accepted.all():
            staged=copy.copy(pipeline);staged.records=dict(pipeline.records)
            staged.remember(entity,property,handle,existing,values)
            pipeline.__dict__.update(staged.__dict__)
            return DevelopmentResult('already_registered',existing,{'gap':False,'training_sha256':input_hash(x)})
    evidence={'gap':True,'gap_kind':'unknown_task' if existing is None else 'incomplete_solution','base_commands_matching_training':matches,
              'semantic_name_new':existing is None,'algorithmic_novelty_claimed':False,'events':['gap_detected']}
    candidate,search=BlockSearch(pipeline.catalog).synthesize(task,x,populations=populations)
    evidence['search']=search;evidence['events']+=['synthesized','candidate_frozen']
    try:
        if candidate.task!=task:raise CertificationError('Synthesized contract mismatch')
        certificate=Verifier(pipeline.catalog).certify(candidate,training_examples=x)
    except CertificationError as exc:
        evidence['events'].append('rejected');evidence['reason']=str(exc)
        return DevelopmentResult('rejected',None,evidence)
    evidence['events'].append('independently_certified')
    staged=copy.copy(pipeline)
    staged.records=dict(pipeline.records);staged.programs=dict(pipeline.programs);staged.task_bindings=dict(pipeline.task_bindings)
    command=max([95,*staged.programs])+1
    staged.programs[command]={'candidate':candidate.payload(),'certificate':certificate,'search':search}
    staged.task_bindings[task]=command
    # A failed neural write or invalid fact cannot leave a registered program behind.
    staged.remember(entity,property,handle,command,values)
    pipeline.__dict__.update(staged.__dict__)
    evidence['events']+=['catalog_registered','kv_and_v5_bound'];evidence['certificate']=certificate
    return DevelopmentResult('accepted',command,evidence)
