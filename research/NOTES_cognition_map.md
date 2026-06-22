# Cognition Map — concepts → mechanism → measurable → next experiment

**Status: research notes, not a results claim.** This file is a strategic
compass. Its job is to keep speculative cognition/AGI ideas from floating free by
forcing each one onto our actual JAX/SoA pipeline: every row must name a
*measurable* quantity we can already snap (or a concrete experiment that would
produce one). Metaphors that are not yet falsifiable are marked as such — they
stay, but flagged, so we never confuse a picture with a mechanism.

Ground truth lives in `../RESULTS.md` and `artifacts/`. Numbers below reference
committed evidence; speculative rows carry no numbers by design.

## Discipline (read first)

- **Mechanism over scale.** Our headline win comes from a **~7,837-parameter**
  model (see `text-inspect`) beating a parameter-matched Transformer on
  delayed-binding recall (`0.9932` vs `0.2035`). The lever was an *explicit
  memory mechanism*, not neuron count. Treat "more neurons → mind" as
  not-yet-falsifiable.
- **A phase transition needs a metric.** "Wait for the phase transition / AGI to
  emerge" is only science once we name the quantity that changes discontinuously.
  Our proto-example: the **soft→hard gate threshold**, where behavior changes
  qualitatively at a cutoff while recall holds at `1.0000`.
- **Falsifiable or flagged.** Each row is either measurable today or carries an
  explicit experiment that would make it measurable.

## The map

| Concept | Metaphor (intuition) | What is actually measurable (in our pipeline) | Next experiment |
|---|---|---|---|
| **Memory vs prediction** | "LLM predicts but does not remember; no experience" | At delay=96 the Transformer loses the binding (`recall 0.20`); the KV-Memory SSM holds (`0.99`). Recall CE: SSM `0.027` vs Tr `3.843`. | Sweep delay ∈ {32,64,96,160,256}: chart the delay at which Tr recall collapses vs SSM — locate the forgetting cliff. |
| **Time: past → present → future** | experience → action → prediction | Recurrent state = compressed past; one recurrence step = present (read/write action); output logits = prediction. | Probe the recurrent state at the query step: can a linear readout recover *which* binding was written N steps ago? Plot recoverability vs N. |
| **Action (the thing LLMs lack)** | the model should *do*, not only emit | The read/write **gate** is the action: write a new binding vs read a bound value. Margin `~0.94` on query tokens, `~0.0002` elsewhere. | Log gate trajectory per token; measure write-precision / read-precision separately against ground-truth bind/query positions. |
| **Synapse = bit (1/0 or other?)** | binary contact point | Hard-eval discretizes the gate (`gate>0.5`) and recall stays `1.0000` on 6/6 seeds — a binary synapse works here. | Push discretization further: ternary / k-bit gate. At what bit-width does recall start to drop? That width is the synapse's information floor. |
| **Weights = synapses** | static structure stores knowledge | Our knowledge is stored in **gate firing patterns over time**, not only in static weights — the architectural shift from spatial window to recurrent timing. | Ablate: freeze weights, perturb only the recurrent state; measure recall loss. Separates "knowledge in weights" from "knowledge in dynamics". |
| **Fast vs slow thinking** | System-1 / System-2 | The sparse-KV read gate is literally a "System-1/System-2" gate; sparse activation routes most tokens cheaply, few through the read path. | Measure fraction of tokens taking the expensive read path; relate sparsity to recall — find the cheapest gate that still recalls. |
| **Phase transition for AGI** | "wait for emergence" | Proto-transition: soft→hard gate cutoff — qualitative behavior change at a threshold with recall preserved. | Define an order parameter (e.g. gate bimodality / margin gap); track it across training; look for a discontinuity, not a wish. |
| **Scale (86B neurons)** | size → mind | Counter-evidence: `7,837` params win the targeted capability. Size is *not* the lever we have evidence for. | None planned — flagged not-yet-falsifiable. Revisit only with a scale→capability curve, not an assertion. |
| **World Model** | "what should the world be?" | Closest concrete handle: `research/codepy/` stateful synthesis — a code "tape" the agent reads/writes. A minimal environment with consequences. | Define env metrics: state-consistency over steps, write→read fidelity, trajectory length before divergence. See codepy plan. |
| **Awareness / self-monitoring** | noticing here-and-now | Operationalize as the model's ability to report its own gate state (read vs write) — not consciousness, just introspective readout. | Aux head predicting "am I reading or writing now?"; measure its accuracy. A model that can't report its own action has no self-monitor. |
| **Quantum / virtual world bridge** | "quantum sandbox bridged to the computer" | No measurable hook for our recall/synthesis task; classical recurrence already beats the Transformer. | Deliberately **out of scope** — flagged. Adds no measurable lever; revisit only if a classical baseline provably caps out. |
| **Language (incl. an AI's own language)** | model could invent its own encoding | Adjacent, measurable: the gate margin is already a learned compact code (`1k2f`-style internal token). NL-LM loss still favors Transformer (negative control). | Inspect learned key embeddings: do bindings cluster into a discrete codebook? Measure codebook size / reuse — emergent "vocabulary" of bindings. |

## The bridge worth pursuing (World Model ↔ codepy)

The strategically richest row is **World Model → `research/codepy/`**. The shift
from a Transformer's *spatial window* to a recurrent *read/write gate* means
information is held in the **patterns and timing of gate firings over time**,
not in a static architectural structure. Stateful synthesis — agents that
continuously read, act, and update their state on a tape — is therefore the
minimal World Model: an environment with consequences and memory.

When we extend that line, the metrics to capture on the new environment:

1. **State-consistency over steps** — does the tape's state stay coherent as the
   trajectory lengthens?
2. **Write→read fidelity** — a value written at step *t* recovered correctly at
   step *t+k*, as a function of *k*.
3. **Trajectory length before divergence** — the honest horizon of autonomous
   stateful synthesis (where `research/codepy/` currently stalls).

This file is the compass; `../RESULTS.md` is the evidence; `research/codepy/` is
the documented frontier. Keep the three in sync — every speculative row should
either earn a number and migrate toward RESULTS, or stay flagged here.
