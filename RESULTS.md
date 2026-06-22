# Results — KV-Memory SSM vs Parameter-Matched Transformer

Technical report for the long-delay associative-recall benchmark. All numbers
are from the committed evidence under `artifacts/long_context_recall_multiseed/`
and reproduce on CPU via `./reproduce_kv_memory_benchmark.sh`.

## Claim

A small state-space model with an **explicit recurrent key→value memory** solves
long-delay associative recall — emit a value bound earlier, after a filler span
longer than the comparison Transformer's attention window — and **generalizes the
binding rule to held-out records**, where a parameter-matched causal Transformer
does not.

This is a bounded claim about delayed-binding recall on synthetic and
code-extracted tasks. It is **not** a natural-language-modeling superiority claim
(see "Negative control" below).

## Protocol

- **Delay** `96` filler tokens between the binding and the query.
- **Transformer context cap** `32` (the binding sits beyond the window) — the
  `bypass_mode` setting. A separate full-context control uses `128`.
- **Parameter budget** matched between SSM and Transformer within ~5%.
- **Held-out evaluation**: train and eval use disjoint folds; recall accuracy is
  measured over every query-token position on the eval fold.
- **Repeated seeds** with an aggregate `summary.json`. Each run records
  `protocol_evidence` — delay-beyond-context, seed count, and minimum
  query-position count — and sets `valid=false` if the protocol is not actually a
  long-delay bypass. Every result below has `protocol_evidence.valid = true`.

## Headline (repeated-seed means)

| Benchmark | Seeds | SSM recall | Transformer recall | SSM − Tr | SSM wins | Valid |
|---|---|---|---|---|---|---|
| `real-python-code` (static AST) | 3 | 0.9932 | 0.2035 | +0.7897 | 3/3 | ✓ |
| `generated-python-code` | 3 | 0.9998 | 0.0700 | +0.9299 | 3/3 | ✓ |
| `mixed-code` | 3 | 0.9998 | 0.0822 | +0.9176 | 3/3 | ✓ |
| `sparse-kv` margin, soft eval | 6 | 1.0000 | 0.0592 | +0.9408 | 6/6 | ✓ |
| `sparse-kv` margin, **hard** eval | 6 | 1.0000 | 0.0592 | +0.9408 | 6/6 | ✓ |

Recall loss tells the same story (e.g. real-python-code: SSM `0.027` vs
Transformer `3.843` cross-entropy).

### Task descriptions

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
  read gate. A squared-hinge query margin drives the read gate near-binary
  (`~0.94` on query tokens, `~0.0002` elsewhere). The **hard eval** re-runs
  inference with a discrete `gate = (gate > 0.5)` threshold: recall stays
  `1.0000` on all six seeds, i.e. soft training reliably learns a margin that
  supports deterministic hard inference.

## Full-context control (is it just a short window?)

Give the Transformer context `128`, covering the whole sequence, with
`--allow-visible-key`. On `real-python-code`, 3 seeds:

- Transformer **train** recall: `0.9922 / 1.0000 / 0.9961` (it can memorize).
- Transformer **held-out** recall mean: `0.5830`.
- SSM held-out recall mean: `0.9932` (still wins 3/3).

So under a tiny matched budget the gap is **not only** an attention-window
artifact — explicit recurrent KV memory generalizes the binding rule better than
a full-context Transformer that has memorized the train fold.

## Negative control — natural-language LM loss

On ordinary next-token LM loss over a million-token Gutenberg/BPE protocol
(`make compare-text-architectures-million`, 3 seeds, parameter-matched), the
**Transformer wins held-out loss**. The best SSM variant found (`conv-skip`)
improved mean SSM loss to `5.5403`, but the matched Transformer reached `5.4885`
and won all three seeds. The SSM's advantages there are orthogonal: higher eval
throughput, far smaller recurrent-state memory than Transformer attention-score
memory (~`341x` ratio), and a longer effective context (`8192` vs `128`). Treat
this as a measured trade-off, not an architecture-superiority claim.

## How to reproduce

```bash
python3 -m pip install -r requirements.txt
./reproduce_kv_memory_benchmark.sh
```

The script regenerates all per-seed JSON and aggregate summaries and prints the
SSM/Transformer recall plus the `protocol_valid` flag for each benchmark. A live
single-seed `real-python-code` run on the development machine reproduced
`winner = ssm`, `ssm − transformer recall = +0.82`.

## Scope limits (read before citing)

1. Wins are on **delayed-binding recall**, synthetic or code-extracted — not open
   natural-language modeling.
2. `real-python-code` uses **bucketized** variable environments over static AST,
   not a full Python language model.
3. The natural-language LM objective still favors the Transformer (above).
4. Some single-seed capacity probes (e.g. 8 bindings) are promising but not yet
   repeated; they are labeled as such in their artifacts.
