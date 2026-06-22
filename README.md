# KV-Memory SSM — Long-Delay Recall Benchmark

A compact, reproducible benchmark showing that a small **state-space model (SSM)
with an explicit recurrent key→value memory** solves long-delay associative
recall where a **parameter-matched, context-capped Transformer** cannot — on
synthetic bindings, on generated code, and on bindings extracted from real
Python source via static AST analysis.

Everything runs on CPU with JAX. No corpus download is required for the headline
benchmarks (records are generated/extracted deterministically per seed).

## Headline result

Recurrent KV memory routes around a short attention window and generalizes the
binding rule on held-out records. Numbers below are means over repeated seeds
from the committed evidence in `artifacts/long_context_recall_multiseed/`
(delay `96`, Transformer context cap `32`, parameter budgets matched within ~5%):

| Benchmark | Seeds | SSM recall | Transformer recall | SSM wins | Protocol valid |
|---|---|---|---|---|---|
| `real-python-code` (static AST of this repo) | 3 | **0.9932** | 0.2035 | 3/3 | ✓ |
| `generated-python-code` | 3 | **0.9998** | 0.0700 | 3/3 | ✓ |
| `mixed-code` | 3 | **0.9998** | 0.0822 | 3/3 | ✓ |
| `sparse-kv` margin gate, hard (binary) eval | 6 | **1.0000** | 0.0592 | 6/6 | ✓ |

A **full-context control** (Transformer context `128`, covering the whole
sequence) does *not* close the gap: the matched Transformer memorizes the train
fold but held-out recall stays near `0.58` on real code while the SSM stays at
`0.99`. So this is not only an attention-window bypass — under a tiny matched
budget, explicit recurrent KV memory generalizes the binding rule better. See
`RESULTS.md` for the full protocol, every metric, and the honest scope limits.

## Quick start

```bash
python3 -m pip install -r requirements.txt   # jax, jaxlib, numpy
make verify        # end-to-end smoke verifier for the product surface
make help          # list operator commands
```

Reproduce the headline evidence (CPU, a few minutes per benchmark):

```bash
./reproduce_kv_memory_benchmark.sh
# or individually:
make compare-real-python-code-recall-multiseed
make compare-generated-python-code-recall-multiseed
make compare-mixed-code-recall-multiseed
make compare-sparse-kv-routing-margin-multiseed
```

Each writes per-seed JSON plus an aggregate `summary.json` with a
`protocol_evidence.valid` flag that fails loudly if the delay/context/seed
protocol is not actually a long-delay bypass.

## What the model is

The recurrent core is a state-space update; the trainable text model is:

```text
token_id -> embedding x_t
h_t      = tanh(A * h_(t-1) + x_t @ B + bias)
logits_t = h_t @ C + out_bias
```

The **KV-memory variant** adds an explicit associative memory rail: write gates
store `key -> value` bindings into a recurrent matrix, and a read gate retrieves
the bound value at a later query token. Variants add learned context/dereference/
erase gates (for distractors, aliases, last-write-wins updates) and a sparse
"System-1/System-2" read gate that can be driven near-binary with a query margin
and still pass deterministic hard-threshold inference. All of this lives in
`compare_long_context_recall.py` and `compare_text_architectures.py`.

## Repository map

Root is the product surface (23 scripts). Run `make help` for commands.

- **Core runtime** — `agent_ssm_core.py`, `soa_ssm_agents.py`, `train_ssm_text.py`,
  `text_tokenization.py`.
- **The benchmark** — `compare_long_context_recall.py` (all recall tasks +
  KV/sparse-KV variants), `run_long_context_recall_multiseed.py` (repeated-seed
  aggregation with protocol gating).
- **SSM vs Transformer** — `compare_text_architectures.py`,
  `run_text_architecture_multiseed.py` (million-token natural-language control;
  see limits below).
- **Data + text tooling** — `ingest_text_data.py`, `profile_text_corpus.py`,
  `run_text_checkpoint.py`, `compare_text_checkpoints.py`, `run_text_experiment.py`,
  `sweep_text_experiments.py`, `promote_text_checkpoint.py`, `inspect_text_checkpoint.py`,
  plus sweep planning/materialization helpers.
- **Quality** — `verify_system.py`, `benchmark_system.py`, `compare_benchmarks.py`.
- **`research/`** — documented experiments kept for the record, *not* part of the
  product: `research/codepy/` (safe program synthesis; honest negative on
  autonomous stateful synthesis) and `research/evolution/` (agent/formula
  evolution, self-play). See `research/README.md`.

## Honest scope limits

- The recall wins are on **synthetic and code-extracted binding tasks**, not on
  open natural-language modeling.
- On ordinary natural-language LM loss, a parameter-matched Transformer still
  **wins** (`make compare-text-architectures-million`): the SSM keeps higher
  throughput, far smaller recurrent-state memory than attention-score memory, and
  a longer effective context, but loses held-out LM loss. A measured trade-off,
  not an architecture-superiority claim.
- `real-python-code` uses bucketized variable environments over static AST, not a
  full Python language model.

## History

The repo was deliberately converged from a ~92-script research sandbox (which had
grown a large release/evidence/audit ceremony layer) down to this focused
benchmark. The complete prior state is preserved in the git freeze commit.
