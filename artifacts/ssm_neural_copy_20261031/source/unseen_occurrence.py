"""Causal TRAIN-zero group occurrence with four-cohort protection."""
from rare_occurrence import (RareOccurrenceState,initial_rare_context_state,
    initial_group_budget,rare_context_logits,protected_group_logits)

def initial_unseen_occurrence_state(batch_size,capacity=256):
    return RareOccurrenceState(initial_rare_context_state(batch_size,capacity),
                               initial_group_budget(batch_size,cohorts=4))

def unseen_occurrence_logits(logits,ids,state,target_mask,protected_cohort,*,prior_strength=8.0):
    proposal,context=rare_context_logits(logits,ids,state.context,target_mask,prior_strength=prior_strength)
    corrected,budget=protected_group_logits(logits,proposal,ids,state.budget,target_mask,protected_cohort)
    return corrected,RareOccurrenceState(context,budget)
