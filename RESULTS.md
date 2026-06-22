# Explicit Sparse KV-Memory in State-Space Models Solves Long-Delay Associative Recall

**Technical report.** All numbers are from the committed evidence under
`artifacts/long_context_recall_multiseed/` and reproduce on CPU via
`./reproduce_kv_memory_benchmark.sh`.

## Abstract

A bounded attention window forces causal Transformers into catastrophic
forgetting on long-delay associative recall: when a fact is bound early and
queried after a filler span longer than the window, the binding is gone. We
present a state-space model (SSM) with an **explicit recurrent, sparse key→value
memory** that retains fact bindings across long distances and **generalizes the
binding rule to held-out records**. Under a parameter-matched budget (within
~5%) and a protocol that places the binding beyond the Transformer's context cap,
the KV-Memory SSM reaches **0.9932** held-out recall on real Python AST traces
versus **0.2035** for the Transformer, with consistent margins across synthetic,
generated, and code-extracted tasks over repeated seeds. The advantage **survives
external validation**: on canonical, marker-free, shared-vocabulary MQAR — a
benchmark we did not design — a context-routed variant holds **0.9309** recall
versus **0.0630** (§2.4), confirming the result is not an artifact of our task
design. We make a deliberately
*bounded* claim: this is a delayed-binding recall result, **not** a
natural-language-modeling superiority claim — on ordinary next-token LM loss the
matched Transformer still wins (see Negative Controls).

## 1. Methodology

**Mechanism — Sparse-KV routing.** The SSM carries an explicit recurrent memory
of key→value slots. At each step a read/write gate decides whether the token
*writes* a new binding into memory or *reads* a previously bound value. A
squared-hinge query margin drives the gate near-binary — `~0.94` on query tokens,
`~0.0002` elsewhere — so the learned soft gate supports deterministic hard
inference (`gate = (gate > 0.5)`) without loss of recall.

**Runtime.** The training/eval stack is a JAX, structure-of-arrays (SoA) runtime;
the recurrent state is a compact per-slot memory rather than an
attention-score matrix, giving a far smaller state footprint than Transformer
attention (~`341x` ratio at matched budget) and a longer effective context
(`8192` vs `128`).

**Protocol (long-delay bypass).**

- **Delay** `96` filler tokens between the binding and the query.
- **Transformer context cap** `32` — the binding sits *beyond* the window
  (`bypass_mode`). A separate full-context control uses `128`.
- **Parameter budget** matched between SSM and Transformer within ~5%.
- **Held-out evaluation** — train and eval use disjoint folds (stable hash-parity
  splits); recall accuracy is measured over every query-token position on the
  eval fold.
- **Repeated seeds** with an aggregate `summary.json`. Each run records
  `protocol_evidence` (delay-beyond-context, seed count, minimum query-position
  count) and sets `valid=false` if the protocol is not actually a long-delay
  bypass. **Every result below has `protocol_evidence.valid = true`.**

## 2. Empirical Results

### 2.1 Headline (repeated-seed means)

| Benchmark | Seeds | SSM recall | Transformer recall | SSM − Tr | SSM wins | Valid |
|---|---|---|---|---|---|---|
| `real-python-code` (static AST) | 3 | **0.9932** | 0.2035 | +0.7897 | 3/3 | ✓ |
| `generated-python-code` | 3 | 0.9998 | 0.0700 | +0.9299 | 3/3 | ✓ |
| `mixed-code` | 3 | 0.9998 | 0.0822 | +0.9176 | 3/3 | ✓ |
| `sparse-kv` margin, soft eval | 6 | 1.0000 | 0.0592 | +0.9408 | 6/6 | ✓ |
| `sparse-kv` margin, **hard** eval | 6 | 1.0000 | 0.0592 | +0.9408 | 6/6 | ✓ |

Recall loss tells the same story (e.g. `real-python-code`: SSM `0.027` vs
Transformer `3.843` cross-entropy).

### 2.2 Task descriptions

- **`real-python-code`** — parses Python files in this repository with `ast`
  (never executes them), extracts real assignment/alias/update sequences from
  functions, and statically simulates a compact bucketized variable environment.
  Train/eval use opposite stable hash-parity folds. The strongest move from
  synthetic toward actual project code.
- **`generated-python-code`** — interpreter-checked synthetic programs with
  `def`/assignments/aliases/updates/`print`/`return` and two later reads; targets
  computed from the simulated environment.
- **`mixed-code`** — compact mini-executor records (literal/alias assignments,
  updates, a non-assignment literal, two reads).
- **`sparse-kv` margin** — `multi-assignment` with a sparse "System-1/System-2"
  read gate. The **hard eval** re-runs inference with a discrete threshold; recall
  stays `1.0000` on all six seeds, i.e. soft training reliably learns a margin
  that supports deterministic hard inference.

### 2.3 Full-context control — is it just a short window?

Give the Transformer context `128`, covering the whole sequence, with
`--allow-visible-key`. On `real-python-code`, 3 seeds:

- Transformer **train** recall: `0.9922 / 1.0000 / 0.9961` (it can memorize).
- Transformer **held-out** recall mean: `0.5830`.
- SSM held-out recall mean: `0.9932` (still wins 3/3).

So under a tiny matched budget the gap is **not only** an attention-window
artifact — explicit recurrent KV memory generalizes the binding rule better than
a full-context Transformer that has merely memorized the train fold.

### 2.4 External validation — canonical MQAR (the crash test)

Every result above is on a benchmark we designed, which invites the obvious
objection: *did we build a task our architecture is uniquely good at?* To answer
it we ran **canonical Multi-Query Associative Recall (MQAR)** — the standard
associative-recall benchmark from the SSM/attention literature — with none of our
task's structural aids: a **single shared vocabulary** for keys and values (the
same token is a key in one pair and a value/query elsewhere), **no
query-marker token** (a query is simply a repeated key), and key→value pairs
**scattered with random gaps**. Scoring rides an explicit position mask (standard
teacher-forcing bookkeeping; the model sees no marker).

Meeting this formulation required extending the architecture: the
`dynamic-kv-memory` SSM infers a token's key/value/query *role* from recurrent
context (`sigmoid(state·W + token·W + b)`) instead of from token id, and reads
memory with the current token to score at the query position. **No privileged
initialization.** Parameter-matched (Transformer/SSM `0.96`; SSM `9,661` params,
Transformer `9,273`), `delay 96`, Transformer context cap `32` (bindings beyond
the window), 3 seeds, `protocol_evidence.valid = true`.

| Metric | KV-Memory SSM | Transformer | SSM − Tr |
|---|---|---|---|
| Held-out recall (mean, 3 seeds) | **0.9309** | 0.0630 | +0.8678 |
| Recall cross-entropy (mean) | **0.3600** | 3.0286 | −2.67 |
| Seeds won | 3/3 | 0/3 | — |

The Transformer sits at chance (`1/16 = 0.0625`): context-capped, it cannot see
the bindings. The result **holds** — the binding-recall advantage is **not** an
artifact of our task design; it survives the standard, marker-free, shared-vocab
formulation. (Recall is `0.93`, not the `1.0` of our marked tasks: shared-vocab
collisions make canonical MQAR strictly harder, and this run used no gate-shaping
curriculum.)

## 3. Limitations & Negative Controls

We state the boundaries of the claim directly; the win does **not** generalize to
open natural language.

**Negative control — natural-language LM loss.** On ordinary next-token LM loss
over a million-token Gutenberg/BPE protocol (`make
compare-text-architectures-million`, 3 seeds, parameter-matched), the
**Transformer wins held-out loss**. The best SSM variant found (`conv-skip`)
improved mean SSM loss to `5.5403`, but the matched Transformer reached `5.4885`
and won all three seeds. The SSM's advantages there are orthogonal — higher eval
throughput, far smaller recurrent-state memory (~`341x` ratio), longer effective
context (`8192` vs `128`). Treat this as a measured trade-off, not an
architecture-superiority claim.

**Scope limits (read before citing).**

1. Wins are on **delayed-binding recall**, synthetic or code-extracted — not open
   natural-language modeling.
2. `real-python-code` uses **bucketized** variable environments over static AST,
   not a full Python language model.
3. The natural-language LM objective still favors the Transformer (above).
4. Some single-seed capacity probes (e.g. 8 bindings) are promising but not yet
   repeated; they are labeled as such in their artifacts.
5. The MQAR validation (§2.4) is one configuration (`16`-symbol vocab, `6`
   bindings, `3` queries, `delay 96`) at CPU scale (~`9.7k` params). Scaling the
   vocabulary, binding count, delay, and parameter budget — and a full-context
   Transformer control on MQAR — remain open follow-ups.

**Documented limit of autonomous stateful synthesis.** The honest upper bound of
this line of work — pushing the KV-Memory mechanism toward general, *stateful*
program synthesis (a code "tape" that reads and writes its own intermediate
state) — is recorded in `research/codepy/`, including
`research/codepy/CODEPY_TURING_COMPLETENESS_PLAN.md`. That directory documents
where autonomous stateful synthesis currently stalls; it is provided as a
transparent negative/boundary record, not as a positive result.

## 4. Reproduction

```bash
python3 -m pip install -r requirements.txt
./reproduce_kv_memory_benchmark.sh
```

The script regenerates all per-seed JSON and aggregate summaries and prints the
SSM/Transformer recall plus the `protocol_valid` flag for each benchmark. A live
single-seed `real-python-code` run on the development machine reproduced
`winner = ssm`, `ssm − transformer recall = +0.82`. The full integration suite
(`make verify`, 22 checks) passes end-to-end with `SYSTEM OK`.
