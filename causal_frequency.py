"""Causal input-frequency correction for frozen language-model logits.

No target or future-token argument is accepted. Training frequencies initialize
per-stream counts; the current input is observed before predicting its successor.
"""
import numpy as np
import jax.numpy as jnp


def frequency_kwargs(config, vocab_size):
    """Validate the optional inference recipe saved in a checkpoint."""
    spec = config.get("causal_frequency")
    if spec is None:
        if (config.get("rare_transition") is not None or config.get("rare_occurrence") is not None
                or config.get("unseen_occurrence") is not None):
            raise ValueError("Rare transition recipe requires causal frequency metadata")
        return {}
    if not isinstance(spec, dict) or spec.get("kind") != "causal_label_prior_v1":
        raise ValueError("Unsupported causal frequency checkpoint recipe")
    frequency = np.asarray(spec.get("train_frequency"), dtype=np.float32)
    if (frequency.shape != (vocab_size,) or not np.isfinite(frequency).all()
            or np.any(frequency < 0) or abs(float(frequency.sum()) - 1.0) > 1e-5):
        raise ValueError("Checkpoint frequency prior must be a finite vocabulary distribution")
    prior_tokens = float(spec.get("prior_tokens", 1024.0))
    epsilon = float(spec.get("epsilon", 0.05))
    if not np.isfinite(prior_tokens) or prior_tokens <= 0 or not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("Checkpoint frequency recipe needs positive prior_tokens and epsilon in (0,1)")
    kwargs = {"frequency_prior": jnp.asarray(frequency), "frequency_prior_tokens": prior_tokens,
              "frequency_epsilon": epsilon}
    cache = config.get("rare_transition")
    if cache is not None:
        if not isinstance(cache, dict) or cache.get("kind") != "conditional_rare_transition_v1":
            raise ValueError("Unsupported rare transition checkpoint recipe")
        mask = np.asarray(cache.get("target_mask"))
        if mask.shape != (vocab_size,) or mask.dtype != np.bool_ or not mask.any() or mask.all():
            raise ValueError("Transition target mask must define a nonempty proper vocabulary group")
        capacity = cache.get("capacity", 256)
        strength = float(cache.get("prior_strength", 8.0))
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Transition capacity must be a positive integer")
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("Transition prior strength must be finite and positive")
        kwargs.update(transition_mask=jnp.asarray(mask), transition_capacity=capacity, transition_strength=strength)
    occurrence = config.get("rare_occurrence")
    if occurrence is not None:
        if (not isinstance(occurrence, dict) or occurrence.get("kind") != "protected_group_occurrence_v1"
                or cache is None):
            raise ValueError("Protected occurrence recipe requires a supported transition recipe")
        cohorts = np.asarray(occurrence.get("protected_cohort"))
        if (cohorts.shape != (vocab_size,) or cohorts.dtype.kind not in "iu"
                or not np.isin(cohorts, [-1, 0, 1]).all() or not (cohorts == 0).any()
                or not (cohorts == 1).any() or np.any(mask & (cohorts >= 0))):
            raise ValueError("Occurrence cohorts must define two disjoint groups outside the rare mask")
        capacity = occurrence.get("capacity", 256)
        strength = float(occurrence.get("prior_strength", 8.0))
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Occurrence capacity must be a positive integer")
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("Occurrence prior strength must be finite and positive")
        kwargs.update(occurrence_cohort=jnp.asarray(cohorts, dtype=jnp.int32),
                      occurrence_capacity=capacity, occurrence_strength=strength)
    unseen = config.get("unseen_occurrence")
    if unseen is not None:
        if not isinstance(unseen, dict) or unseen.get("kind") != "protected_unseen_occurrence_v1":
            raise ValueError("Unsupported unseen occurrence recipe")
        mask = frequency == 0
        cohorts = np.asarray(unseen.get("protected_cohort"))
        if (not mask.any() or mask.all() or cohorts.shape != (vocab_size,)
                or cohorts.dtype.kind not in "iu" or not np.isin(cohorts, [-1, 0, 1, 2, 3]).all()
                or not np.array_equal(cohorts == -1, mask)
                or not all((cohorts == group).any() for group in range(4))):
            raise ValueError("Unseen recipe needs the TRAIN-zero mask and four protected seen cohorts")
        capacity = unseen.get("capacity", 256)
        strength = float(unseen.get("prior_strength", 8.0))
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Unseen occurrence capacity must be a positive integer")
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("Unseen occurrence prior strength must be finite and positive")
        kwargs.update(unseen_mask=jnp.asarray(mask), unseen_cohort=jnp.asarray(cohorts, dtype=jnp.int32),
                      unseen_capacity=capacity, unseen_strength=strength)
    return kwargs


def initial_frequency_state(train_frequency, batch_size, prior_tokens=1024.0):
    return (jnp.broadcast_to(prior_tokens * train_frequency, (batch_size, train_frequency.shape[0])),
            jnp.asarray(0.0, dtype=jnp.float32))


def causal_frequency_logits(logits, ids, history, seen, train_frequency, *, prior_tokens=1024., epsilon=.05):
    import jax
    import jax.numpy as jnp
    vocab=logits.shape[-1]
    prefix=history[:,None,:]+jnp.cumsum(jax.nn.one_hot(ids,vocab,dtype=jnp.float32),axis=1)
    denominator=prior_tokens+seen+jnp.arange(1,ids.shape[1]+1,dtype=jnp.float32)
    new_prior=(1-epsilon)*prefix/denominator[None,:,None]+epsilon/vocab
    old_prior=(1-epsilon)*train_frequency+epsilon/vocab
    return logits+jnp.log(new_prior)-jnp.log(old_prior),prefix[:,-1,:],seen+ids.shape[1]
