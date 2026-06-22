# SoA SSM Agent Prototype

This folder now has multiple layers of the same flat-array, JAX-compiled state-space
runtime.

## Quick Start

```bash
python3 -m pip install -r requirements.txt
make verify
make benchmark
```

Use `make help` to list the common operator commands. See `PROJECT_STATUS.md`
for the current system map.

## 1. Agent Runtime

Shared agent-world math lives in `agent_ssm_core.py`; the runtime scripts reuse
that module for observations, state updates, reward, and world initialization.

The first demo uses logical agents rather than text prediction. That keeps the
core visible:

- `TENSORS_X`: present-time observations for all agents.
- `STATES_H`: compressed past/experience for all agents.
- `WEIGHTS_A/B/C`: rules that update memory and turn memory into action logits.

The recurrent update is:

```text
h_t = tanh(A * h_(t-1) + x_t @ B)
y_t = h_t @ C
```

Run it:

```bash
python3 soa_ssm_agents.py
```

Try larger or smaller batches:

```bash
python3 soa_ssm_agents.py --agents 1048576 --steps 512
python3 soa_ssm_agents.py --agents 65536 --steps 128
```

On this machine, the default CPU run processed `67,108,864` agent-steps at about
`104M agent-steps/s` after compilation.

The cached run time excludes JAX compilation, so it is the useful number for
steady-state simulation throughput.

The script defaults to `--backend cpu`. The installed Apple Metal JAX backend is
experimental and failed basic array creation in this environment, so CPU is the
stable first target for the prototype. You can still try it explicitly:

```bash
python3 soa_ssm_agents.py --backend metal
```

## 2. Trainable Text Runtime

The second demo turns the same idea into a tiny word-level next-token model:

```text
token_id -> embedding x_t
h_t = tanh(A * h_(t-1) + x_t @ B + bias)
logits_t = h_t @ C + out_bias
loss = cross_entropy(logits_t, next_token)
```

Run it:

```bash
python3 train_ssm_text.py
```

On this machine, the default text run trained `1,142,784` token positions in
`1.36s`, reaching `final_loss=0.0233` and `final_accuracy=0.9888` on the tiny
built-in corpus.

Train on a custom corpus:

```bash
python3 train_ssm_text.py --text-file ./your_corpus.txt --epochs 30
```

Save and reload a trained text checkpoint:

```bash
python3 train_ssm_text.py --epochs 6 --save-checkpoint artifacts/text_ssm.npz
python3 train_ssm_text.py --load-checkpoint artifacts/text_ssm.npz --eval-only
```

Run a saved checkpoint as a standalone artifact:

```bash
python3 run_text_checkpoint.py \
  --checkpoint artifacts/text_ssm.npz \
  --prompt "memory is" \
  --output-json artifacts/text_run.json
```

This script uses JAX `value_and_grad` plus a small built-in Adam optimizer, so
`WEIGHTS_A`, `WEIGHTS_B`, `WEIGHTS_C`, token embeddings, and biases are all
learned directly. Hidden memory is carried across chunks as truncated BPTT:
gradients flow through each `--seq-len` window, then `STATES_H` is detached and
used as the next chunk's compressed past.

## 3. Corpus Profiling

Before training on an external text file, profile it:

```bash
python3 profile_text_corpus.py --text-file ./your_corpus.txt
```

The profiler uses the same tokenizer as `train_ssm_text.py`, reports token and
vocabulary shape, estimates the SoA memory footprint, and suggests practical
`--streams`, `--tokens-per-stream`, and `--seq-len` values for the next training
run.

Compare a corpus against a saved checkpoint vocabulary:

```bash
python3 profile_text_corpus.py \
  --text-file ./your_corpus.txt \
  --checkpoint artifacts/text_ssm.npz \
  --output-json artifacts/text_profile.json
```

That OOV report is the first guardrail before deciding whether to reuse a
checkpoint, continue training, or build a new vocabulary.

## 4. Real Data And Subword Tokenization

The text pipeline can now ingest a real public-domain book corpus and train with
a checkpoint-local BPE tokenizer:

```bash
make ingest-gutenberg-demo
make profile-text-bpe-demo
make text-sweep-bpe-demo
```

The ingestion command downloads a small Project Gutenberg preset, strips the
boilerplate, writes `artifacts/corpora/gutenberg_demo/corpus.txt`, and records
`corpus_manifest.json` with SHA256, token counts, and source metadata.

For a larger repeatable run, use the cached bulk path. It builds a larger
Gutenberg corpus, learns BPE once, reuses the saved tokenizer for every sweep
run, then packages the winner:

```bash
make ingest-gutenberg-bulk
make profile-text-bulk-bpe
make text-sweep-bulk-bpe
make text-release-bulk-bpe
```

The current bulk preset targets about `1.5MB` of cleaned text and writes:

- `artifacts/corpora/gutenberg_bulk/corpus.txt`
- `artifacts/corpora/gutenberg_bulk/tokenizer_bpe_768.json`
- `artifacts/text_sweeps/gutenberg_bulk_bpe/leaderboard.json`
- `artifacts/promoted/gutenberg_bulk_bpe/MODEL_CARD.md`
- `artifacts/releases/current_gutenberg_bulk_bpe_release.json`

For the next scale step, the ingestion layer can also write sharded corpora and
a resumable sweep plan:

```bash
python3 ingest_text_data.py \
  --source gutenberg \
  --id-range 1:10000 \
  --output-dir artifacts/corpora/gutenberg_large \
  --output-text artifacts/corpora/gutenberg_large/corpus.txt \
  --output-json artifacts/corpora/gutenberg_large/corpus_manifest.json \
  --target-bytes 1000000000 \
  --shard-bytes 50000000 \
  --continue-on-error \
  --delay-s 0.2

python3 materialize_text_corpus.py \
  --manifest artifacts/corpora/gutenberg_large/corpus_manifest.json \
  --output-text artifacts/corpora/gutenberg_large/train_200mb.txt \
  --max-bytes 200000000

make plan-text-sweep-bulk

python3 run_text_sweep_plan.py \
  --plan artifacts/text_sweeps/gutenberg_bulk_bpe_large/plan.json \
  --max-batches 1
```

`plan_text_sweep.py` writes a JSON plan with batch commands using
`sweep_text_experiments.py --resume --start-index ... --max-runs ...`, so a
hundred-plus configuration sweep can be run in repeatable chunks.
`run_text_sweep_plan.py` executes the next incomplete batch and writes
`plan_progress.json`, so repeated invocations keep moving through the plan
without restarting completed experiments.

BPE is opt-in and stored inside each checkpoint, so later evaluation, package
benchmarking, memory probes, and release execution reuse the same tokenizer:

```bash
python3 run_text_experiment.py \
  --text-file artifacts/corpora/gutenberg_demo/corpus.txt \
  --tokenizer bpe \
  --bpe-vocab-size 512 \
  --output-dir artifacts/text_experiments/gutenberg_bpe
```

Promote and package the resulting leaderboard in the usual release path:

```bash
python3 release_text_pipeline.py \
  --leaderboard artifacts/text_sweeps/gutenberg_bpe_demo/leaderboard.json \
  --text-file artifacts/corpora/gutenberg_demo/corpus.txt \
  --promote-dir artifacts/promoted/gutenberg_bpe
```

## 5. Text Checkpoint Comparison

Compare two trained text checkpoints on the same corpus and eval shape:

```bash
python3 compare_text_checkpoints.py \
  --baseline artifacts/text_old.npz \
  --candidate artifacts/text_new.npz \
  --fail-on-loss-regression \
  --output-json artifacts/text_compare.json
```

The comparison warms up JAX before timing, then reports loss, accuracy,
throughput, OOV rate, and generated samples for each checkpoint. Use
`--fail-on-loss-regression` to make it a CI-style gate for text-model changes.

## 5a. Architecture Baseline: SSM vs Transformer

To test the stronger claim, compare the text SSM against a tiny causal
Transformer under one shared protocol:

```bash
python3 compare_text_architectures.py \
  --text-file ./your_corpus.txt \
  --tokenizer bpe \
  --bpe-vocab-size 512 \
  --streams 8 \
  --tokens-per-stream 256 \
  --seq-len 32 \
  --epochs 3 \
  --output-json artifacts/text_architecture_comparison.json
```

The harness trains both models from scratch on the same train split, evaluates
on the same held-out split, chooses a Transformer width close to the SSM
parameter count, warms up before timing, and reports loss, perplexity,
accuracy, throughput, parameter memory, optimizer-memory estimate, and effective
context length. It also writes a guardrail into the JSON: a smoke or tiny-corpus
run proves the benchmark works, not that either architecture is generally
better.

Try the built-in smoke-size protocol:

```bash
make compare-text-architectures-demo
```

Run the larger repeated-seed million-token protocol on the 50MB Gutenberg BPE
corpus:

```bash
make compare-text-architectures-million
```

That target runs `run_text_architecture_multiseed.py`, which invokes
`compare_text_architectures.py` once per seed, checks that the protocol covers
at least one million train positions and one million eval positions, then writes
per-seed reports plus an aggregate summary:

- `artifacts/text_architecture_multiseed/gutenberg_50mb_bpe_million/summary.json`
- `artifacts/text_architecture_multiseed/gutenberg_50mb_bpe_million/seed_11.json`
- `artifacts/text_architecture_multiseed/gutenberg_50mb_bpe_million/seed_17.json`
- `artifacts/text_architecture_multiseed/gutenberg_50mb_bpe_million/seed_23.json`

The current 3-seed run used `128x8192/128`, so each seed covered `1,032,192`
train positions per epoch and `1,032,192` held-out eval positions per repeat.
In that run, the parameter-matched Transformer won held-out loss on all three
seeds (`5.5810` mean loss vs SSM `5.6696`), while the SSM kept higher eval
throughput (`895k tok/s` vs `792k tok/s`), much lower recurrent state memory
than Transformer attention-score memory (`341.3x` ratio), and a longer effective
context (`8192` vs `128` tokens). Treat this as a bounded measurement, not a
universal architecture claim.

Follow-up architecture probes added SSM variants with selective gates, direct
input-to-logits skips, low-rank/tied skips, two-token convolutional skips, state
mixing, and a `token-memory` rail. The strongest Gutenberg/BPE local-LM
improvement was `conv-skip`: mean SSM loss improved from the original static
`5.6696` to `5.5403` on the same 3-seed million-token protocol, but the
parameter-matched Transformer still won all three seeds with mean loss `5.4885`.
The useful lesson is that local shortcut channels help the SSM, but they do not
beat attention on a short-window natural-language objective by themselves.

A direct transfer attempt, `conv-token-memory`, combines the best local
`conv-skip` shortcut with the learned token-memory rail. It runs and wins a tiny
built-in smoke corpus, but on a seed-11 Gutenberg/BPE probe it worsens SSM loss
relative to `conv-skip` (`5.9292` vs `5.8554`) and still loses to the matched
Transformer (`5.7013`). A conservative closed-gate version,
`conv-token-memory-closed`, mostly removes that regression (`5.8575`) but still
does not beat `conv-skip`; diagnostics show it mostly keeps memory shut
(mean read/write gates around `0.019`). An L1-gated sparse version stays shut
too (`~0.017`) and lands at essentially the same loss (`5.8575`). The lesson is
useful but negative: memory needs a utility signal that teaches when to open,
not just pressure to stay closed, before it helps ordinary natural-language LM
loss.

That utility-signal route is now implemented as SSM-only auxiliary recall inside
`compare_text_architectures.py`. The held-out metric is still the same
Gutenberg/BPE LM loss. The strongest probe so far uses `kv-memory` plus an
assignment-style curriculum (`name = value ... name ? value`) with structural
memory-gate initialization:

```bash
python3 compare_text_architectures.py \
  --text-file artifacts/corpora/gutenberg_50mb/corpus.txt \
  --tokenizer bpe \
  --tokenizer-config artifacts/corpora/gutenberg_50mb/tokenizer_bpe_768.json \
  --ssm-variant kv-memory \
  --ssm-aux-recall-every 1 \
  --ssm-aux-recall-task assignment \
  --ssm-aux-recall-init-gates \
  --ssm-aux-recall-delay 96
```

Artifact:
`artifacts/text_architecture_comparison_kv_assignment_aux_init_bpe_probe.json`.
The KV rail reaches `1.0000` auxiliary binding-recall accuracy, but the ordinary
held-out LM loss still favors the matched Transformer (`5.7942` vs SSM
`5.8927`). This is progress on usable memory routing, not a natural-language LM
victory.

The current bypass probe is `compare_long_context_recall.py`, a synthetic
delayed-recall test where a key appears before a long filler span and the model
must emit that key at a later query token. The plain tanh SSM solves delay `32`
but not delay `64`/`96`; the new `token-memory` variant adds a token-gated slow
memory rail and learns delay `96` with Transformer context cap `32` without
structural key/query initialization. In the demo run, SSM held-out recall is
`1.0000`, the context-capped Transformer is `0.0469`, and the matched parameter
ratio is `0.995`. The same learned setup reached `1.0000` SSM recall on seeds
`11`, `17`, and `23`. A control run with Transformer context `128` reaches
`1.0000` too, confirming that this is specifically a route around a short
attention window, not a broken Transformer implementation. Treat it as a
synthetic associative-recall win, not yet a general natural-language superiority
claim.

Run the delayed-recall smoke:

```bash
make compare-long-context-recall-demo
```

There is also a harder code-like transfer probe:

```bash
make compare-assignment-recall-demo
```

This generates records shaped like `name = value ; other = other_value ; ... name ?`
and asks the model to emit the original value after the delay. `token-memory`
beats the capped Transformer but plateaus near `0.52` recall because it stores a
slow token trace, not an addressable binding. The newer `kv-memory` variant
stores an explicit key-to-value association and solves the same delay `96` /
Transformer context `32` task: the current demo reaches SSM held-out recall
`0.9990`, earlier three-seed runs reached `1.0000`, and the capped Transformer
stays near chance. With Transformer context `128`, the Transformer also reaches
`1.0000`, so this remains a clean attention-window bypass rather than a broken
baseline.

The next stress case stores several bindings in one record:

```bash
make compare-multi-assignment-recall-demo
```

This uses `--task multi-assignment` with four `name = value ;` pairs, then asks
for one queried name after delay `96`. `kv-memory` reached `1.0000` held-out
recall on seeds `11`, `17`, and `23` with Transformer context cap `32`; the
matched one-layer Transformer stayed near chance. A full-window Transformer
control no longer cleanly solves this harder task under the same small parameter
budget, so the result should be read as a synthetic multi-binding generalization
win for explicit KV memory, not merely as a short-window control.

Probe the new dynamic System-1/System-2 read routing:

```bash
make compare-sparse-kv-routing-smoke
```

This runs `sparse-kv-memory` on a small multi-assignment task with an L1
curriculum, target sparsity, and a small entropy-sharpening penalty on KV reads.
With structural assignment-memory initialization, the SSM keeps recall at
`1.0000` while reducing average dynamic KV read strength (`0.0117` mean,
`0.3325` on `?`, `0.0084` elsewhere, about `39.7x` query/non-query separation).
The threshold diagnostics show the important caveat: about `92.5%` of
non-query positions are below `0.01`, but `0%` of query positions exceed `0.5`
or `0.9`. This is real read-cost routing and strong background suppression, but
not yet a clean binary "think only on ?" gate. Without structural
initialization, the same mechanism either opens broadly or shuts down, so
learning the syntax roles from scratch remains the next hurdle.

The first hardening probe adds a squared hinge margin on query-token reads:

```bash
make compare-sparse-kv-routing-margin-smoke
```

With `--kv-read-gate-query-margin 0.9` and weight `2.0`, the same protocol keeps
SSM recall at `1.0000` and pushes the read gate into near-binary behavior:
`0.0149` mean, `0.9358` on `?`, `0.0053` elsewhere, `100%` of query reads above
`0.9`, and `96.6%` of non-query reads below `0.01`. This is still a supervised
margin, not a Straight-Through/Gumbel hard router, but it confirms that the soft
gate can be driven into a practical on/off regime without losing recall.

The stronger follow-up evaluates discrete inference without training through a
hard estimator:

```bash
make compare-sparse-kv-routing-margin-multiseed
```

This trains with the continuous margin gate, then evaluates an additional hard
path with `gate = (gate > 0.5)`. Across seeds `11,17,23,41,73,101`, soft SSM
recall and hard-inference recall are both `1.0000` on every seed. The hard eval
has mean gate `0.0105`, query gate `1.0000`, non-query gate `0.0002`, and
`99.98%` of non-query positions below `0.01`. The soft gate is stable above
`0.5` on all query positions, but not literally above `0.9` on every query in
every seed, so the strongest claim is: soft training reliably learns a margin
that supports deterministic hard inference.

There is also a multi-read version:

```bash
make compare-multi-query-assignment-recall-demo
```

This stores four bindings, then asks two separate `name ?` queries in the same
record. The eval path now counts every query-token position, not just the final
one. In the full seed-11 run, `kv-memory` reached `1.0000` held-out recall
across both query positions while the capped Transformer stayed near chance.

For syntax-sensitive writes:

```bash
make compare-assignment-distractor-recall-demo
```

This emits `name = value ; ... name ? value ; wrong_literal ; name ?` and still
expects the original value. The `kv-memory` write path now has a learned
previous-token context gate, so value tokens can be written after `=` without
being written merely because they appear as literals later. In the full demo,
SSM reaches `1.0000` recall while the capped Transformer reaches about `0.53`.

For alias/copy assignment:

```bash
make compare-assignment-alias-recall-demo
```

This emits `source = value ; alias = source ; ... alias ?` and expects the
original value. The KV path now includes a learned dereference gate for RHS
identifier tokens, so it can read the value stored under `source` and write that
value under `alias`. In the full demo, SSM reaches `1.0000` recall while the
capped Transformer stays near chance.

For a mixed mini-program corpus:

```bash
make compare-mixed-code-recall-demo
```

This procedurally simulates a small environment with literal assignments,
alias assignments, updates, a non-assignment value literal, and two later reads.
The targets come from the simulated environment rather than one hand-written
template. In the full demo, `kv-memory` reaches `0.9995` held-out recall while
the capped Transformer stays near chance.

Run the repeated-seed version:

```bash
make compare-mixed-code-recall-multiseed
```

On seeds `11`, `17`, and `23`, `kv-memory` wins `3/3` with mean held-out recall
`0.9998`; the capped Transformer mean is `0.0822`. This is the strongest current
synthetic-code route-around result.

Run the full-context control:

```bash
make compare-mixed-code-full-context-control
```

On seed `11`, giving the matched Transformer context `128` so it can see the
whole sequence improves train recall but not held-out generalization: after
`120` epochs, Transformer held-out recall is `0.4229` while `kv-memory` remains
`0.9995`. That suggests the mixed-code result is not only a short-window bypass;
under this tiny matched budget, explicit KV memory generalizes the mini-executor
rule better.

For a generated Python-like code corpus:

```bash
make compare-generated-python-code-recall-demo
```

This adds syntax tokens around the same interpreter-checked environment:
`def f():`, assignments, alias assignments, updates, `print(...)`, `return(...)`,
literal noise, and two later `print(name ? value)` reads. Targets are still
computed from the simulated environment, so the benchmark keeps a strict ground
truth while looking more like code than the compact `mixed-code` records. In the
full demo, `kv-memory` reaches `1.0000` held-out recall while the capped
Transformer reaches `0.0693` under a nearly matched parameter budget.

Run the repeated-seed version:

```bash
make compare-generated-python-code-recall-multiseed
```

On seeds `11`, `17`, and `23`, `kv-memory` wins `3/3` with mean held-out recall
`0.9998`; the capped Transformer mean is `0.0700`. This is the current best
generated-code route-around result, still bounded to interpreter-generated
synthetic programs rather than general natural-language modeling.

Run the full-context control:

```bash
make compare-generated-python-code-full-context-control
```

On seed `11`, giving the matched Transformer context `128` covers the full
sequence length (`106`) and removes the short-window limitation. After `40`
epochs, Transformer held-out recall is `0.2832` while `kv-memory` remains
`1.0000`; after a manual `120`-epoch control, Transformer train recall rises to
`0.6172` but held-out recall stays at `0.2798`, with `kv-memory` still at
`1.0000`. This strengthens the interpretation from a pure attention-window
bypass toward a tiny-budget mini-executor generalization win for explicit
recurrent KV memory.

For a still more code-like update rule:

```bash
make compare-assignment-update-recall-demo
```

This generates `name = old ; other = value ; name = new ; ... name ?` and asks
for the latest value. The current `kv-memory` path includes a learned erase gate
for value writes and reached `1.0000` held-out recall in the seed-11 probe, while
the capped Transformer stayed near chance. This mode is slower on CPU because
each write now performs an explicit erase/update against the KV matrix.

For the first static-AST real-repository probe:

```bash
make compare-real-python-code-recall-demo
```

This parses Python files from this repository, never executes them, extracts
real assignment/alias/update sequences from functions, and statically simulates
a compact variable environment. On seed `11`, with `186` extracted train-fold
records and `186` eval-fold records, `kv-memory` reaches `0.9951` held-out
recall while the capped Transformer reaches `0.2617`.

Run the repeated-seed version:

```bash
make compare-real-python-code-recall-multiseed
```

On seeds `11`, `17`, and `23`, `kv-memory` wins `3/3` with mean held-out recall
`0.9932`; the capped Transformer mean is `0.2035`. This is the strongest current
route-around evidence on real project code, while still being a compact
static-AST binding-memory probe.

Run the full-context control:

```bash
make compare-real-python-code-full-context-control
```

With context `128`, the matched Transformer can see the full sequence and
reaches `0.9922` train recall, but held-out recall is `0.5889`; `kv-memory`
remains at `0.9951`.

Run the repeated-seed full-context control:

```bash
make compare-real-python-code-full-context-multiseed
```

On seeds `11`, `17`, and `23`, the full-context Transformer reaches near-perfect
train recall (`0.9922`, `1.0000`, `0.9961`) but mean held-out recall is `0.5830`;
`kv-memory` wins `3/3` with mean held-out recall `0.9932`. Do not read this
compact AST probe as a broad Python-language-model result, but it does show that
the real-repo binding-memory win is not only a short-window artifact.

## 6. Text Experiment Pipeline

Run the full text workflow in one command:

```bash
python3 run_text_experiment.py \
  --text-file ./your_corpus.txt \
  --output-dir artifacts/text_experiments/my_run
```

The pipeline splits the corpus into train/eval text files, profiles the train
side, selects or accepts an eval shape, trains a baseline and candidate
checkpoint on train, compares them on the held-out eval split, and writes
`train.txt`, `eval.txt`, `profile.json`, `baseline.npz`, `candidate.npz`,
`comparison.json`, and `experiment.json` into the output directory.

Try the built-in smoke-size experiment:

```bash
make text-experiment-demo
```

List saved text experiments as a leaderboard:

```bash
python3 list_text_experiments.py \
  --root artifacts/text_experiments \
  --output-json artifacts/text_experiments/leaderboard.json \
  --output-csv artifacts/text_experiments/leaderboard.csv
```

The leaderboard ranks experiment directories by candidate loss and exposes the
run shape, held-out eval size, loss delta, accuracy, throughput, OOV rate,
winner, and checkpoint paths. On tiny corpora, the baseline may win on held-out
loss; that is useful signal rather than a pipeline failure.

Run a small hyperparameter sweep:

```bash
python3 sweep_text_experiments.py \
  --text-file ./your_corpus.txt \
  --output-dir artifacts/text_sweeps/my_sweep \
  --state-dims 32,64,128 \
  --learning-rates 0.006,0.012
```

Each grid point runs the full held-out experiment pipeline in its own directory,
then the sweep writes `sweep.json`, `leaderboard.json`, and `leaderboard.csv`.
Try the smoke sweep:

```bash
make text-sweep-demo
```

Promote the best checkpoint from a leaderboard:

```bash
python3 promote_text_checkpoint.py \
  --leaderboard artifacts/text_sweeps/my_sweep/leaderboard.json \
  --output-dir artifacts/promoted/text \
  --smoke-run
```

Promotion copies the selected `.npz` checkpoint and writes `manifest.json` with
the source run, metric values, selected role, checkpoint size, SHA256 checksum,
and optional smoke-run result. The default `--checkpoint-role best` promotes
whichever side has lower held-out loss, so it can promote the baseline when the
candidate overfits.

Inspect a promoted checkpoint:

```bash
python3 inspect_text_checkpoint.py \
  --manifest artifacts/promoted/text/manifest.json \
  --output-json artifacts/promoted/text/model_card.json \
  --output-md artifacts/promoted/text/MODEL_CARD.md
```

The inspector reads the `.npz` without JAX and reports config, training metrics,
vocabulary preview, parameter shapes, parameter count, and optimizer memory as
both machine-readable JSON and a human-readable Markdown model card.

Run a promoted package directly from its manifest:

```bash
python3 run_text_checkpoint.py \
  --manifest artifacts/promoted/text/manifest.json \
  --output-json artifacts/promoted/text/run.json
```

When `--manifest` is used, the runner resolves `promoted_checkpoint` and reuses
the selected run's `streams`, `tokens_per_stream`, and `seq_len` unless they are
overridden on the command line.

Benchmark a promoted package with warmed-up repeats:

```bash
python3 benchmark_text_package.py \
  --manifest artifacts/promoted/text/manifest.json \
  --warmup 1 \
  --repeat 5 \
  --output-json artifacts/promoted/text/benchmark.json
```

The benchmark verifies checkpoint bytes/SHA256 from the manifest, warms up JAX
before timing, then reports best/mean/median eval time, token throughput, loss,
accuracy, OOV rate, and a sample. This is the quick "passport speed" for a
promoted artifact.

Validate a promoted package:

```bash
python3 validate_text_package.py \
  --manifest artifacts/promoted/text/manifest.json \
  --require-model-card \
  --require-smoke \
  --require-benchmark
```

The validator checks the checkpoint path, byte size, SHA256 checksum, model-card
JSON/Markdown, smoke-run JSON, and benchmark JSON. It resolves package-local
checkpoint paths first, so restored bundles validate against their own copied
checkpoint rather than the original source path.

Bundle a validated package:

```bash
python3 bundle_text_package.py \
  --manifest artifacts/promoted/text/manifest.json \
  --output artifacts/promoted/text/text_package.tar.gz \
  --output-json artifacts/promoted/text/bundle_manifest.json
```

The bundle contains the checkpoint, manifest, model cards, smoke run, benchmark
report, and validation report under one archive root, plus a bundle manifest
with file checksums and archive SHA256.

Restore and validate a bundle:

```bash
python3 unbundle_text_package.py \
  --archive artifacts/promoted/text/text_package.tar.gz \
  --bundle-manifest artifacts/promoted/text/bundle_manifest.json \
  --output-dir artifacts/restored
```

The unbundler verifies the archive SHA256, rejects unsafe archive paths, checks
every restored file checksum, and runs package validation on the restored
manifest, including the restored benchmark report.

Compare two promoted packages as release candidates:

```bash
python3 compare_text_packages.py \
  --baseline-manifest artifacts/promoted/text/manifest.json \
  --candidate-manifest artifacts/restored/text/manifest.json \
  --fail-on-loss-regression \
  --fail-on-throughput-regression \
  --output-json artifacts/promoted/text/package_comparison.json
```

The package comparator revalidates both manifests, reads their package-local
benchmark reports, compares loss, accuracy, throughput, checkpoint bytes, and
checkpoint SHA256, and can fail a CI run on release regressions.

Register a verified package release:

```bash
python3 register_text_package.py \
  --manifest artifacts/promoted/text/manifest.json \
  --bundle-manifest artifacts/promoted/text/bundle_manifest.json \
  --archive artifacts/promoted/text/text_package.tar.gz \
  --comparison-json artifacts/promoted/text/package_comparison.json \
  --registry artifacts/releases/text_packages.json \
  --name text_prod_candidate
```

The registry stores the package archive SHA256, checkpoint SHA256, benchmark
metrics, selected run metadata, and optional comparison verdict. Entries are
sorted by benchmark loss and throughput so the first record is the current best
registered release.

List registered releases:

```bash
python3 list_text_releases.py \
  --registry artifacts/releases/text_packages.json \
  --output-json artifacts/releases/text_release_leaderboard.json \
  --output-csv artifacts/releases/text_release_leaderboard.csv
```

The release leaderboard flattens registry entries into a ranked operator view
with loss, throughput, memory diagnostics, checkpoint/archive SHA256, package
paths, and comparison verdict fields.

Open the static release leaderboard viewer:

```bash
make release-viewer-open
```

Or open `text_release_leaderboard_viewer.html` directly in a browser, then
choose `text_release_leaderboard.json`. It renders ranked releases with loss,
throughput, memory overlap, and memory cosine bars.

Open the static release dashboard:

```bash
make release-dashboard-open
```

Or open `text_release_dashboard.html` directly in a browser, then choose
`text_release_leaderboard.json`, `current_text_release.json`, and
`memory_suite.json`. It combines the active release, leaderboard, package audit,
and prompt-memory matrix in one offline operator panel.

Build a single JSON payload for that dashboard:

```bash
make build-release-dashboard-demo
```

Or run it directly against any matching release artifacts:

```bash
python3 build_text_release_dashboard.py \
  --leaderboard artifacts/releases/text_release_leaderboard.json \
  --current-release artifacts/releases/current_text_release.json \
  --memory-suite artifacts/releases/text_memory_suite/memory_suite.json \
  --memory-validation artifacts/releases/text_memory_suite/validation.json \
  --run-json artifacts/releases/run_current_text_release.json \
  --output-json artifacts/releases/text_release_dashboard.json \
  --require-valid
```

The builder embeds the leaderboard, active pointer, memory suite, validation,
and run report, then records an audit showing whether the release metrics and
memory diagnostics agree.

Validate the dashboard artifact as a CI/release gate:

```bash
make validate-release-dashboard-demo
```

Or run the validator directly:

```bash
python3 validate_text_release_dashboard.py \
  --dashboard artifacts/releases/text_release_dashboard.json \
  --require-artifacts \
  --expected-prompts 3 \
  --expected-pairs 3 \
  --min-mean-pair-overlap 0.0 \
  --output-json artifacts/releases/text_release_dashboard_validation.json
```

Compare two dashboard artifacts as a release gate:

```bash
make compare-release-dashboard-demo
```

Open the static dashboard comparison viewer:

```bash
make release-dashboard-compare-open
```

Or open `text_release_dashboard_compare.html` directly in a browser, then choose
`text_release_dashboard_comparison.json` or load baseline and candidate
dashboard JSON files. It renders loss, accuracy, throughput, audit, and memory
geometry deltas from the same fields used by the release gate.

Or run it directly:

```bash
python3 compare_text_release_dashboards.py \
  --baseline-dashboard artifacts/releases/text_release_dashboard.json \
  --candidate-dashboard artifacts/releases/text_release_dashboard.json \
  --require-artifacts \
  --expected-prompts 3 \
  --expected-pairs 3 \
  --fail-on-loss-regression \
  --fail-on-throughput-regression \
  --fail-on-audit-regression \
  --fail-on-memory-overlap-regression \
  --output-json artifacts/releases/text_release_dashboard_comparison.json
```

Build a release history index from dashboard artifacts:

```bash
make build-release-history-demo
```

Or run it directly:

```bash
python3 build_text_release_history.py \
  --dashboard artifacts/releases/text_release_dashboard.json \
  --require-artifacts \
  --expected-prompts 3 \
  --expected-pairs 3 \
  --output-json artifacts/releases/text_release_history.json
```

Validate the release history index as a CI/release gate:

```bash
make validate-release-history-demo
```

Or run it directly:

```bash
python3 validate_text_release_history.py \
  --history artifacts/releases/text_release_history.json \
  --require-artifacts \
  --expected-releases 1 \
  --output-json artifacts/releases/text_release_history_validation.json
```

Analyze the latest release trend against a chosen history baseline:

```bash
make analyze-release-history-demo
```

Or run it directly:

```bash
python3 analyze_text_release_history.py \
  --history artifacts/releases/text_release_history.json \
  --baseline previous \
  --require-artifacts \
  --expected-releases 1 \
  --fail-on-loss-regression \
  --fail-on-throughput-regression \
  --fail-on-memory-overlap-regression \
  --output-json artifacts/releases/text_release_history_analysis.json
```

Summarize all release gates into one operator verdict:

```bash
make summarize-release-gates-demo
```

Or run it directly:

```bash
python3 summarize_text_release_gates.py \
  --dashboard-validation artifacts/releases/text_release_dashboard_validation.json \
  --dashboard-comparison artifacts/releases/text_release_dashboard_comparison.json \
  --history-validation artifacts/releases/text_release_history_validation.json \
  --history-analysis artifacts/releases/text_release_history_analysis.json \
  --output-json artifacts/releases/text_release_gates.json
```

The summary JSON folds dashboard validation, dashboard comparison, history
validation, and history trend analysis into a single pass/fail release verdict.

Generate a portable Markdown release report from the gate artifacts:

```bash
make generate-release-report-demo
```

Or run it directly:

```bash
python3 generate_text_release_report.py \
  --gates artifacts/releases/text_release_gates.json \
  --dashboard artifacts/releases/text_release_dashboard.json \
  --history-analysis artifacts/releases/text_release_history_analysis.json \
  --output-md artifacts/releases/text_release_report.md
```

The report records the final verdict, gate table, history deltas, dashboard
audit checks, source artifacts, and rebuild command in one Markdown file.

Bundle the release evidence into one checksum manifest and tarball:

```bash
make bundle-release-evidence-demo
```

Or run it directly:

```bash
python3 bundle_text_release_evidence.py \
  --gates artifacts/releases/text_release_gates.json \
  --output artifacts/releases/text_release_evidence.tar.gz \
  --output-json artifacts/releases/text_release_evidence_manifest.json
```

The evidence bundle includes the gate summary, Markdown report, dashboard,
history artifacts, active release pointer, run report, leaderboard, and memory
suite files, with file SHA256 values plus the archive SHA256 in the manifest.

Validate the evidence bundle as a portable release artifact:

```bash
make validate-release-evidence-demo
```

Or run it directly:

```bash
python3 validate_text_release_evidence.py \
  --manifest artifacts/releases/text_release_evidence_manifest.json \
  --output-json artifacts/releases/text_release_evidence_validation.json
```

The validator recalculates archive and file SHA256 values, rejects unsafe tar
paths, checks the inner evidence manifest, and revalidates the bundled gate
summary.

Restore the evidence bundle into a clean directory and revalidate it:

```bash
make unbundle-release-evidence-demo
```

Or run it directly:

```bash
python3 unbundle_text_release_evidence.py \
  --archive artifacts/releases/text_release_evidence.tar.gz \
  --manifest artifacts/releases/text_release_evidence_manifest.json \
  --output-dir artifacts/restored \
  --output-json artifacts/restored/text_release_evidence_restore.json \
  --clean
```

The unbundler verifies the archive SHA256 from the manifest, rejects unsafe tar
members, restores the evidence tree, checks every restored file SHA/byte count,
and validates the restored inner `evidence_manifest.json` against the archive.

Register a validated evidence bundle in a tamper-evident hash-chain ledger:

```bash
make register-release-evidence-demo
```

Or run it directly:

```bash
python3 register_text_release_evidence.py \
  --manifest artifacts/releases/text_release_evidence_manifest.json \
  --archive artifacts/releases/text_release_evidence.tar.gz \
  --validation-json artifacts/releases/text_release_evidence_validation.json \
  --restore-json artifacts/restored/text_release_evidence_restore.json \
  --comparison-json artifacts/releases/text_release_evidence_comparison.json \
  --ledger artifacts/releases/text_release_evidence_ledger.json \
  --store-dir artifacts/releases/text_release_evidence_store \
  --name text_demo
```

The ledger stores each evidence archive with manifest, validation, restore, and
comparison report hashes. Each entry includes the previous entry hash and its
own canonical `entry_hash`, so the `chain_head` can be independently verified.

Validate the evidence ledger as a fast release gate:

```bash
make validate-release-evidence-ledger-demo
```

Or run it directly:

```bash
python3 validate_text_release_evidence_ledger.py \
  --ledger artifacts/releases/text_release_evidence_ledger.json \
  --output-json artifacts/releases/text_release_evidence_ledger_validation.json \
  --expected-entries 1 \
  --require-artifacts \
  --artifact-scope latest \
  --fail-on-failed-gates
```

The validator recomputes every entry hash, verifies `previous_hash` links and
`chain_head`, and can also check the SHA256 values of linked manifest, archive,
validation, restore, and comparison artifacts.

Prove that ledger tampering is detected without modifying the source ledger:

```bash
make tamper-release-evidence-ledger-demo
```

Or run it directly:

```bash
python3 tamper_text_release_evidence_ledger.py \
  --ledger artifacts/releases/text_release_evidence_ledger.json \
  --output-ledger artifacts/releases/text_release_evidence_ledger_tampered.json \
  --output-json artifacts/releases/text_release_evidence_ledger_tamper.json \
  --mode archive_sha256 \
  --expected-entries 1 \
  --require-artifacts \
  --artifact-scope latest \
  --fail-on-failed-gates
```

The tamper test first validates the source ledger, writes a modified copy, then
requires the normal validator to reject that copy with the expected hash-chain
mismatch.

Replay ledger registration semantics with two independent evidence bundles:

```bash
make replay-release-evidence-ledger-demo
```

Or run it directly:

```bash
python3 replay_text_release_evidence_ledger.py \
  --manifest artifacts/releases/text_release_evidence_manifest.json \
  --archive artifacts/releases/text_release_evidence.tar.gz \
  --validation-json artifacts/releases/text_release_evidence_validation.json \
  --restore-json artifacts/restored/text_release_evidence_restore.json \
  --comparison-json artifacts/releases/text_release_evidence_comparison.json \
  --base-dir artifacts/releases \
  --ledger artifacts/releases/text_release_evidence_ledger_replay.json \
  --store-dir artifacts/releases/text_release_evidence_ledger_replay_store \
  --work-dir artifacts/releases/text_release_evidence_ledger_replay \
  --output-json artifacts/releases/text_release_evidence_ledger_replay_report.json \
  --clean \
  --fail-on-failed-gates
```

The replay test proves that registering the same evidence updates the existing
entry, registering a distinct valid bundle appends a second entry, and the full
two-entry chain validates with artifact checks across every linked snapshot.

Compare two release evidence ledgers as an audit drift gate:

```bash
make compare-release-evidence-ledger-demo
```

Or run it directly:

```bash
python3 compare_text_release_evidence_ledgers.py \
  --baseline-ledger artifacts/releases/text_release_evidence_ledger_replay.json \
  --candidate-ledger artifacts/releases/text_release_evidence_ledger_replay.json \
  --output-json artifacts/releases/text_release_evidence_ledger_comparison.json \
  --require-artifacts \
  --artifact-scope all \
  --fail-on-failed-gates \
  --fail-on-chain-head-change \
  --fail-on-entry-count-change \
  --fail-on-entry-set-change \
  --fail-on-entry-field-change \
  --fail-on-archive-set-change \
  --fail-on-manifest-set-change
```

The ledger comparator validates both ledgers first, then compares `chain_head`,
entry count, entry hash sets, archive/manifest SHA sets, and changed fields for
common entries. Without fail flags it can also report expected ledger growth.

Summarize the full release evidence proof chain into one audit verdict:

```bash
make summarize-release-evidence-audit-demo
```

Or run it directly:

```bash
python3 summarize_text_release_evidence_audit.py \
  --manifest artifacts/releases/text_release_evidence_manifest.json \
  --validation-json artifacts/releases/text_release_evidence_validation.json \
  --restore-json artifacts/restored/text_release_evidence_restore.json \
  --comparison-json artifacts/releases/text_release_evidence_comparison.json \
  --ledger artifacts/releases/text_release_evidence_ledger.json \
  --ledger-validation-json artifacts/releases/text_release_evidence_ledger_validation.json \
  --tamper-json artifacts/releases/text_release_evidence_ledger_tamper.json \
  --replay-json artifacts/releases/text_release_evidence_ledger_replay_report.json \
  --ledger-comparison-json artifacts/releases/text_release_evidence_ledger_comparison.json \
  --ledger-growth-comparison-json artifacts/releases/text_release_evidence_ledger_growth_comparison.json \
  --output-json artifacts/releases/text_release_evidence_audit.json \
  --output-md artifacts/releases/text_release_evidence_audit.md
```

The audit summary verifies the evidence manifest, archive validation, restore,
bundle comparison, source ledger, ledger validation, tamper detection, replay
semantics, self-comparison, and expected one-entry ledger growth in one JSON and
Markdown report.

Validate the release evidence audit rollup and all linked source artifacts:

```bash
make validate-release-evidence-audit-demo
```

Or run it directly:

```bash
python3 validate_text_release_evidence_audit.py \
  --audit artifacts/releases/text_release_evidence_audit.json \
  --output-json artifacts/releases/text_release_evidence_audit_validation.json \
  --require-sources
```

The validator enforces the audit schema, required proof checks, summary
invariants, SHA chain heads, replay/tamper semantics, and cross-artifact
consistency against the ten source JSON files.

Compare two release evidence audit validation outputs:

```bash
make compare-release-evidence-audit-validation-demo
```

Or run it directly:

```bash
python3 compare_text_release_evidence_audit_validations.py \
  --baseline-validation artifacts/releases/text_release_evidence_audit_validation.json \
  --candidate-validation artifacts/releases/text_release_evidence_audit_validation.json \
  --output-json artifacts/releases/text_release_evidence_audit_validation_comparison.json \
  --fail-on-validity-change \
  --fail-on-failed-check-regression \
  --fail-on-error-regression
```

The validation comparator treats the validator output as a first-class release
artifact and compares validity, audit path, release name, check/failure/error
counts, artifact-check coverage, source-artifact coverage, and validation error
sets.

Compare two release evidence audit rollups as a final proof regression gate:

```bash
make compare-release-evidence-audit-demo
```

Or run it directly:

```bash
python3 compare_text_release_evidence_audits.py \
  --baseline-audit artifacts/releases/text_release_evidence_audit.json \
  --candidate-audit artifacts/releases/text_release_evidence_audit.json \
  --output-json artifacts/releases/text_release_evidence_audit_comparison.json \
  --fail-on-validity-change \
  --fail-on-check-set-change \
  --fail-on-check-validity-change \
  --fail-on-failed-check-regression
```

The comparator validates both audit rollups first, then compares final
validity, check sets, check validity, replay actions, tamper detection,
entry-count deltas, artifact-check counts, source paths, chain heads, and
archive SHA drift.

Compare two evidence bundles as a release proof regression gate:

```bash
make compare-release-evidence-demo
```

Or run it directly:

```bash
python3 compare_text_release_evidence.py \
  --baseline-manifest artifacts/releases/text_release_evidence_manifest.json \
  --candidate-manifest artifacts/releases/text_release_evidence_manifest.json \
  --fail-on-archive-sha-change \
  --fail-on-file-set-change \
  --fail-on-file-sha-change \
  --output-json artifacts/releases/text_release_evidence_comparison.json
```

The comparator validates both bundles first, then compares archive SHA, file
labels, file hashes, byte counts, archive paths, validity, and gate failures.

Open the static release evidence comparison viewer:

```bash
make release-evidence-compare-open
```

Or open `text_release_evidence_compare.html` directly in a browser, then choose
`text_release_evidence_comparison.json` or load baseline and candidate evidence
manifest JSON files. It renders archive/file hash drift, file-label changes,
byte deltas, validity changes, gate failures, and the per-label evidence matrix.

Open the static release evidence ledger viewer:

```bash
make release-evidence-ledger-open
```

Or open `text_release_evidence_ledger_viewer.html` directly in a browser, then
choose `text_release_evidence_ledger.json` or
`text_release_evidence_ledger_replay.json`. It renders chain-head status,
previous-hash links, entry metrics, snapshot artifact paths, and optional ledger
validation or replay reports.

Open the static release evidence ledger comparison viewer:

```bash
make release-evidence-ledger-compare-open
```

Or open `text_release_evidence_ledger_compare.html` directly in a browser, then
choose `text_release_evidence_ledger_comparison.json` or
`text_release_evidence_ledger_growth_comparison.json`. It renders chain-head
drift, entry-count deltas, added/missing entry hashes, SHA-set drift, and
threshold failures.

Open the static release evidence audit viewer:

```bash
make release-evidence-audit-open
```

Or open `text_release_evidence_audit_viewer.html` directly in a browser, then
choose `text_release_evidence_audit.json`. It renders the final evidence audit
verdict, replay and tamper indicators, per-check proof cards, source artifact
paths, and any failed proof-chain checks.

Open the static release evidence audit validation viewer:

```bash
make release-evidence-audit-validation-open
```

Or open `text_release_evidence_audit_validation_viewer.html` directly in a
browser, then choose `text_release_evidence_audit_validation.json`. It renders
the validator verdict, audit path, release name, check/source coverage, failure
counts, and validation error ledger.

Open the static release evidence audit validation comparison viewer:

```bash
make release-evidence-audit-validation-compare-open
```

Or open `text_release_evidence_audit_validation_compare.html` directly in a
browser, then choose `text_release_evidence_audit_validation_comparison.json`.
It renders baseline and candidate validation cards, count deltas, source
coverage deltas, identity drift, error-set drift, and threshold failures.

Open the static release evidence audit comparison viewer:

```bash
make release-evidence-audit-compare-open
```

Or open `text_release_evidence_audit_compare.html` directly in a browser, then
choose `text_release_evidence_audit_comparison.json`. It renders baseline and
candidate audit cards, final proof drift, check drift, source drift, and
threshold failures from the audit comparison gate.

Open the static release gate summary viewer:

```bash
make release-gates-open
```

Or open `text_release_gates_viewer.html` directly in a browser, then choose
`text_release_gates.json`. It renders the final release verdict, the four gate
cards, and any gate failures.

Open the static release history analysis viewer:

```bash
make release-history-analysis-open
```

Or open `text_release_history_analysis_viewer.html` directly in a browser, then
choose `text_release_history_analysis.json`. It renders the latest-vs-baseline
verdict, regression deltas, and threshold failures.

Open the static release history viewer:

```bash
make release-history-open
```

Or open `text_release_history_viewer.html` directly in a browser, then choose
`text_release_history.json`. It renders the latest release, best loss/speed/
memory marks, and per-release deltas over the dashboard history.

The full release pipeline also runs the dashboard builder as its final audited
stage, so `make text-release-pipeline-demo` emits a ready-to-load dashboard JSON
without a separate manual step. Pass `--include-evidence-audit` to extend the
same command through gate summaries, the Markdown report, the evidence bundle,
restore/compare validation, hash-chain ledger registration, tamper/replay tests,
ledger comparisons, and the final evidence audit validation/comparison JSONs.

Select the active release:

```bash
python3 select_text_release.py \
  --registry artifacts/releases/text_packages.json \
  --rank 1 \
  --output-json artifacts/releases/current_text_release.json
```

The selector verifies the chosen archive SHA256 and package metadata, then writes
a small `current_text_release.json` pointer with the manifest, archive,
checkpoint, benchmark metrics, and validation result.

Run the active release pointer directly:

```bash
python3 run_text_checkpoint.py \
  --release artifacts/releases/current_text_release.json \
  --prompt "memory is" \
  --output-json artifacts/releases/run_current_text_release.json
```

Probe the hidden memory trajectory behind an active release:

```bash
python3 probe_text_memory.py \
  --release artifacts/releases/current_text_release.json \
  --prompt "memory is a river" \
  --include-state-vectors \
  --output-json artifacts/releases/text_memory_probe.json \
  --output-csv artifacts/releases/text_memory_probe.csv
```

The memory probe records one row per prompt token with `STATES_H` norm, state
delta norm, state distribution stats, top predicted next tokens, and optional
hidden-state vectors for geometric comparisons.

Compare two hidden memory probes:

```bash
python3 compare_text_memory.py \
  --baseline artifacts/releases/text_memory_probe_baseline.json \
  --candidate artifacts/releases/text_memory_probe_candidate.json \
  --output-json artifacts/releases/text_memory_comparison.json \
  --output-csv artifacts/releases/text_memory_comparison.csv
```

The comparison reports token match rate, top prediction overlap, norm drift,
state-delta drift, probability drift, final top-token agreement, and, when
vectors are present, L2 distance plus cosine similarity.

Run a prompt suite and pairwise memory matrix:

```bash
python3 run_text_memory_suite.py \
  --release artifacts/releases/current_text_release.json \
  --prompt "memory is a river" \
  --prompt "memory is another world" \
  --prompt "the future is guessed" \
  --include-state-vectors \
  --output-dir artifacts/releases/text_memory_suite
```

The suite writes one probe JSON/CSV per prompt plus `memory_suite.json`,
`prompts.csv`, and `pairs.csv` with pairwise overlap, drift, L2, and cosine
metrics.

Validate a prompt-suite artifact before using it as a release diagnostic:

```bash
python3 validate_text_memory_suite.py \
  --suite artifacts/releases/text_memory_suite/memory_suite.json \
  --prompts-csv artifacts/releases/text_memory_suite/prompts.csv \
  --pairs-csv artifacts/releases/text_memory_suite/pairs.csv \
  --require-state-vectors \
  --require-prompt-artifacts \
  --output-json artifacts/releases/text_memory_suite/validation.json
```

The validator checks prompt/pair counts, CSV row counts, per-prompt probe
artifacts, metric ranges, optional vector geometry, and optional drift/overlap
thresholds. This turns the memory matrix into a CI-style gate instead of a
visual-only report.

Open the static prompt-suite matrix viewer:

```bash
make memory-suite-viewer-open
```

Or open `text_memory_suite_viewer.html` directly in a browser, then choose
`memory_suite.json`. It renders a metric-switchable heatmap for cosine, L2,
top-overlap, and norm drift.

Open the static memory viewer:

```bash
make memory-viewer-open
```

Or open `text_memory_viewer.html` directly in a browser, then choose
`text_memory_probe.json`. It renders the hidden-state norm, state delta, top
probability curve, token timeline, and per-token top predictions.

Or run the whole release chain as one operator command:

```bash
python3 release_text_pipeline.py \
  --leaderboard artifacts/text_sweeps/demo/leaderboard.json \
  --promote-dir artifacts/promoted/text_release \
  --registry artifacts/releases/text_packages.json \
  --current-release-json artifacts/releases/current_text_release.json \
  --run-json artifacts/releases/run_current_text_release.json \
  --dashboard-json artifacts/releases/text_release_dashboard.json \
  --include-evidence-audit
```

The release pipeline promotes the best checkpoint, builds model cards,
benchmarks and validates the package, bundles and restores it, compares the
restored package, registers the release, selects rank 1, smoke-runs the active
pointer, runs and validates a hidden-memory prompt suite for the selected
release, then builds the audited dashboard JSON. The pipeline JSON includes
both package metrics and memory geometry fields such as pair overlap, L2
distance, cosine similarity, and the dashboard audit result. The release
registry, leaderboard, active-release pointer, and dashboard artifact carry the
validated memory summary so operators can rank and inspect releases by behavior,
not only by loss and throughput.

With `--include-evidence-audit`, the pipeline JSON also records the evidence
archive SHA256, evidence ledger chain head, audit verdict, audit validation, and
audit comparison artifacts. Path overrides such as `--evidence-archive`,
`--evidence-ledger-json`, and `--evidence-audit-json` let CI jobs keep proof
packets isolated per run. When those overrides are omitted, proof artifacts are
prefixed from the pipeline JSON stem, so `text_release_pipeline_demo.json` writes
names such as `text_release_pipeline_demo_evidence_manifest.json` instead of
overwriting the standalone evidence-demo artifacts.

Validate a pipeline report as its own portable control artifact:

```bash
python3 validate_text_release_pipeline.py \
  --pipeline artifacts/releases/text_release_pipeline_demo.json \
  --require-artifacts \
  --require-evidence-audit \
  --expected-stages 35 \
  --output-json artifacts/releases/text_release_pipeline_validation.json
```

The validator checks required stages, release pointers, memory-suite geometry,
the embedded dashboard audit, and the optional evidence-audit proof chain.

Compare two pipeline validation reports as a regression gate:

```bash
python3 compare_text_release_pipeline_validations.py \
  --baseline-validation artifacts/releases/text_release_pipeline_validation.json \
  --candidate-validation artifacts/releases/text_release_pipeline_validation.json \
  --output-json artifacts/releases/text_release_pipeline_validation_comparison.json \
  --fail-on-validity-change \
  --fail-on-stage-count-change \
  --fail-on-artifact-check-regression
```

The comparison reports validity drift, stage-count drift, memory-suite count
changes, dashboard/evidence audit coverage deltas, artifact-check coverage
deltas, and validation error-set changes.

Compare two full pipeline reports directly:

```bash
python3 compare_text_release_pipelines.py \
  --baseline-pipeline artifacts/releases/text_release_pipeline_demo.json \
  --candidate-pipeline artifacts/releases/text_release_pipeline_demo.json \
  --require-evidence-audit \
  --fail-on-stage-set-change \
  --fail-on-loss-regression \
  --fail-on-memory-overlap-regression \
  --fail-on-evidence-archive-sha-change \
  --output-json artifacts/releases/text_release_pipeline_comparison.json
```

The direct comparator reads the stage map and summary metrics from each
pipeline JSON, then reports stage-set drift, release/rank deltas, loss,
accuracy, throughput, memory-geometry deltas, dashboard/evidence audit
regressions, evidence archive SHA drift, and evidence ledger chain-head drift.

Validate the full pipeline comparison JSON:

```bash
python3 validate_text_release_pipeline_comparison.py \
  --comparison artifacts/releases/text_release_pipeline_comparison.json \
  --require-evidence-audit \
  --expected-stages 35 \
  --output-json artifacts/releases/text_release_pipeline_comparison_validation.json
```

The validator recomputes all deltas and drift flags from the embedded
baseline/candidate pipeline summaries, checks stage-list consistency, verifies
proof SHA fields when evidence audit is required, and can reject comparison
artifacts that already contain threshold failures.

Compare two full pipeline comparison validation reports:

```bash
python3 compare_text_release_pipeline_comparison_validations.py \
  --baseline-validation artifacts/releases/text_release_pipeline_comparison_validation.json \
  --candidate-validation artifacts/releases/text_release_pipeline_comparison_validation.json \
  --fail-on-validity-change \
  --fail-on-stage-count-change \
  --fail-on-proof-flag-change \
  --fail-on-error-set-change \
  --output-json artifacts/releases/text_release_pipeline_comparison_validation_comparison.json
```

The comparison reports validation validity drift, compared-pipeline path drift,
stage-count drift, loss/memory delta drift, proof flag changes, threshold
failure regressions, and validation error-set changes.

Open the full pipeline comparison validation comparison viewer:

```bash
make release-pipeline-comparison-validation-compare-open
```

Or open `text_release_pipeline_comparison_validation_compare.html` directly in a
browser, then choose `text_release_pipeline_comparison_validation_comparison.json`.
It renders baseline/candidate validation cards, validation drift deltas,
path/proof/error drift, and threshold failures.

Open the consolidated pipeline proof cockpit:

```bash
make release-pipeline-proof-cockpit-open
```

Or open `text_release_pipeline_proof_cockpit.html` directly in a browser, then
choose the six pipeline proof JSON files or paste a bundled proof object. It
renders the full release pipeline, validation, comparison, comparison
validation, and proof-comparison chain as one operator surface.

Open the full pipeline comparison validation viewer:

```bash
make release-pipeline-comparison-validation-open
```

Or open `text_release_pipeline_comparison_validation_viewer.html` directly in a
browser, then choose `text_release_pipeline_comparison_validation.json`. It
renders the validation verdict, compared pipeline paths, stage count drift,
loss/memory drift, proof-hash agreement, threshold failure count, and validation
errors.

Open the full pipeline comparison viewer:

```bash
make release-pipeline-compare-open
```

Or open `text_release_pipeline_compare.html` directly in a browser, then choose
`text_release_pipeline_comparison.json`. It renders baseline/candidate pipeline
cards, metric deltas, stage drift, proof-hash drift, audit validity drift, and
threshold failures from the direct pipeline comparison gate.

Open the release pipeline validation viewers:

```bash
make release-pipeline-validation-open
make release-pipeline-validation-compare-open
```

Or open `text_release_pipeline_validation_viewer.html` and
`text_release_pipeline_validation_compare.html` directly in a browser, then
choose `text_release_pipeline_validation.json` or
`text_release_pipeline_validation_comparison.json`.

Compare two selected releases as an operator gate:

```bash
python3 compare_text_releases.py \
  --baseline-release artifacts/releases/current_text_release.json \
  --candidate-release artifacts/releases/current_text_pipeline_release.json \
  --require-memory \
  --fail-on-loss-regression \
  --fail-on-throughput-regression \
  --fail-on-memory-overlap-regression \
  --output-json artifacts/releases/text_release_comparison.json
```

The release comparator reads active release pointers or ranked registry entries
and compares loss, accuracy, throughput, checkpoint/archive SHA, and memory
diagnostics. It can fail on package regressions and memory overlap/cosine/L2
drift without rerunning training or JAX evaluation.

## 7. Evolutionary Agent Runtime

The third demo searches over whole SSM rule sets instead of differentiating
through them. Population data is also SoA:

```text
GENOME_A: [population, state_dim]
GENOME_B: [population, input_dim, state_dim]
GENOME_C: [population, state_dim, action_dim]
```

Each candidate controls the same agent world, receives a reward based on goal
distance, threat distance, energy, and policy entropy, then elites are copied
and mutated:

```bash
python3 evolve_ssm_agents.py
```

On this machine, the default evolution run evaluated `717,225,984` agent-steps
in `6.77s`, about `106M agent-steps/s`, and found a best policy with
`goal_distance=0.0140`.

Try a faster smoke run:

```bash
python3 evolve_ssm_agents.py --population 24 --agents 1024 --steps 32 --generations 6
```

Save the best evolved genome:

```bash
python3 evolve_ssm_agents.py --save-best artifacts/best_agent_genome.npz
```

## 7a. Formula Evolution / Tiny NAS

The next meta-learning layer searches over the hidden-state update formula
itself, not only over the weights inside one formula. It uses a constrained DSL:

```text
h_memory = memory_op(h * A)
h_input  = input_op(x @ B)
h_next   = final_op(combine_op(h_memory, h_input))
```

The search mutates choices such as `identity`, `tanh`, `softsign`, `sin`,
`relu`, `add`, `gated_mix`, and `max`. For each formula it runs a short inner
weight-evolution loop in the same agent world, then ranks formulas by reward and
throughput:

```bash
python3 evolve_ssm_formulas.py \
  --architecture-population 5 \
  --architecture-generations 2 \
  --weight-population 16 \
  --weight-generations 2 \
  --output-json artifacts/formula_evolution.json
```

Try the smoke-size run:

```bash
make formula-evolution-demo
```

This is a controlled NAS/metaprogramming experiment, not autonomous AGI or
arbitrary self-modifying code. The JSON report keeps that guardrail explicit.

## 7b. CodePy Block Evolution

The CodePy prototype tests a cached-command action space directly. A candidate
program is a fixed-length array of block IDs, not generated text. The current
safe DSL has scalar registers `r0` and `r1`, a predicate rail `p`, one or two
inputs (`x`, optional `y`), and reusable blocks such as `r0 *= x`, `r1 = y`,
`p = r1 > r0`, and `r0 = where(p, r1, r0)`. JAX executes the whole population
as flat register/predicate rails over train and holdout cases, then evolution
keeps programs that match a target function:

```bash
python3 evolve_code_blocks.py \
  --target cubic_minus_x \
  --population 2048 \
  --program-length 9 \
  --output-json artifacts/code_block_evolution.json
```

Try the smoke-size run:

```bash
make code-block-evolution-demo
```

Focused probes are also available:

```bash
make code-block-quadratic-demo
make code-block-abs-demo
make code-block-relu-demo
make code-block-sign-demo
make code-block-max-xy-demo
make code-block-piecewise-demo
make code-block-library-demo
make code-block-skill-ladder-demo
make code-block-hierarchy-demo
```

The quadratic probe keeps searching after exactness until peephole minimization
finds the compact `(x + 1) * (x + 1)` form. The abs probe validates the first
conditional primitive. The relu, sign, max, and piecewise probes validate
predicated control flow before adding full CMP/JUMP/BRANCH bytecode. The
library probe compares the base DSL against a learned macro-block profile
containing reusable atoms such as `square`, `abs`, `relu`, `max`, and `sign` on
downstream targets like `clamp01` and `max(abs(x), y)`.

The JSON report stores the block vocabulary, best opcode sequence, compact
non-noop program, and rendered Python for inspection. It also runs a semantic
peephole minimizer: blocks are removed one at a time and the program is kept
shorter only if train and holdout MSE remain under `--minimize-tolerance`. The
reward includes `--length-penalty`, so longer searches can bias the evolution
toward shorter solutions directly. The report also records `search_blocks`,
`minimal_blocks`, and `compression_ratio` so the apparent search complexity can
be compared with the minimized algorithm. It is constrained DSL search; it
never executes arbitrary generated Python source.

The learned-library comparison writes
`artifacts/code_block_library/comparison.json` with success rate, mean
generation, mean minimized block count, runtime, and base-vs-learned deltas.

The skill-ladder demo runs the stronger accumulation test:

```text
Phase A: base primitives discover square, abs, relu, max, sign
Phase B: successful minimized skills are promoted into macro-blocks
Phase C: clamp01 and max_abs_x_y are searched with base vs learned profiles
```

It writes `artifacts/code_block_skill_ladder/skill_ladder.json` plus
`artifacts/code_block_skill_ladder/library_manifest.json`. The summary records
`compression_ratio` for discovered skills, downstream `reuse_frequency`, and
`fitness_gain` so the experiment measures whether learned primitives actually
shorten or speed up later synthesis.

The Stage 4 hierarchy demo measures growth of task complexity directly. It
runs the ladder `square -> abs -> max -> clamp -> max(abs(x), y) ->
clamp(abs(x),0,1) -> piecewise(max(abs(x),y))`; each task is searched once with
base primitives and once with the hierarchical profile containing only earlier
promoted skills. The JSON report
`artifacts/code_block_hierarchy/hierarchy.json` records `generation_found`,
`active_blocks`, `compression_ratio`, `search_time_s`, and base-vs-hierarchy
deltas for plotting whether the gap grows as tasks become more compositional.

Stage 4B turns that one-seed signal into a success-rate experiment. It runs
`max_abs_x_y`, `clamp_abs_01`, and `piecewise_max_abs_x_y` over many seeds and
compares `success_rate` against `hierarchy_depth`:

```bash
make code-block-stage4b-demo
make code-block-stage4b-demo STAGE4B_SEED_COUNT=100 STAGE4B_JOBS=4
```

The report `artifacts/code_block_stage4b/stage4b.json` is plot-ready: each
target contains depth rows with success rate, generation-found statistics,
minimal blocks, compression ratio, and search time. The companion SVG
`artifacts/code_block_stage4b/success_rate_vs_depth.svg` plots
`success_rate vs hierarchy_depth`. In the current 100-seed run,
`max_abs_x_y` improves from 97% at depth 0 to 100% at depth 2+, `clamp_abs_01`
from 84% at depth 0 to 100% at depth 3+, and `piecewise_max_abs_x_y` from 0%
at depth 0 to 51% at depth 5. The curve is not strictly monotonic: adding more
macros can expand the search vocabulary and add noise. The current evidence
status is: Program Synthesis PASS, Program Compression PASS, Skill Promotion
PASS, Hierarchical Reuse PASS, Hierarchical Scaling EARLY PASS, and General
Algorithmic Scaling not proven.

Stage 4C analyzes the noise problem directly:

```bash
make code-block-pruning-demo
```

It reads the 100-seed Stage 4B artifact and writes
`artifacts/code_block_stage4c/pruning.json` plus
`artifacts/code_block_stage4c/pruning.md`. The pruning report computes noisy
depths, Pareto frontiers, and a regularized utility
`success_rate - penalty * hierarchy_depth`, so macro-library growth can be
evaluated as a cache-selection problem instead of blindly keeping every new
block.

For the Turing-completeness path, see `CODEPY_TURING_COMPLETENESS_PLAN.md`.
The current verdict is deliberately bounded: JAX is ready for fixed-budget
`LOOP`/`JUMP` via `lax.scan` with a program-counter rail and fixed-size memory,
but arbitrary unbounded `while` is not a safe next step for the current
vectorized evaluator. The recommended first algorithmic targets are
`count_positive(vector[4])`, `sum(vector[4])`, `argmax_index(vector[4])`,
`swap_if_gt`, `sort2`, and only then `sort3` or larger sorting tasks.

The first bounded tape prototype is `evolve_code_tape.py`. It adds `tape[8]`,
`ptr`, `pc`, `read_tape`, `write_tape`, pointer movement, predicate blocks, and
a fixed `jax.lax.scan` execution budget:

```bash
make code-tape-sum-demo
make code-tape-sum-clean-demo
make code-tape-count-positive-demo
make code-tape-count-positive-prior-demo
make code-tape-argmax-demo
make code-tape-trajectory-demo
make code-tape-trajectory-seeded-demo
```

The injected demos use `--inject-reference` as a state-geometry gate: they
verify that the DSL can execute looping tape programs for `sum(vector[4])` and
`count_positive(vector[4])` with zero train/holdout error. The clean sum demo
removes reference injection and uses trace-based reward shaping for read,
move, loop, accumulate, and tape coverage behavior. In the current run it
autonomously discovers a zero-error `sum(vector[4])` loop on train and holdout
data. The count-positive prior demo now also discovers a zero-error
`count_positive(vector[4])` predicate/conditional loop without reference
injection, then trains a trace prior and re-finds a compact zero-error loop with
guided beam search. This proves first autonomous branch-heavy tape discovery for
`count_positive4`, but not yet robust multi-seed scaling or harder algorithms
such as `argmax_index`.

The vNext prior-search path is reproducible with:

```bash
make code-tape-prior-demo
```

That demo logs every evaluated bounded-tape program into JSONL traces, trains a
lightweight logistic prior over program tokens plus trace embeddings, uses the
prior inside a bounded beam search, and promotes top zero-error traces into an
explicit macro manifest. The current `sum4` evidence writes
`artifacts/code_tape_prior/sum4_traces.jsonl` with 16,384 trace records, trains
`artifacts/code_tape_prior/sum4_prior.json` with test AUC about 0.99, finds a
zero-error loop with 672 guided beam evaluations in
`artifacts/code_tape_prior/sum4_prior_beam.json`, and promotes five safe DSL
macro candidates in `artifacts/code_tape_prior/sum4_macros.json`.

For `count_positive4`, `make code-tape-count-positive-prior-demo` writes 65,536
clean no-reference traces, discovers the predicate loop
`[4, 7, 14, 16, 18, 11]` during evolution, trains a weak warm-start prior from
one positive trace, re-finds `[4, 7, 14, 16, 18, 0]` with 2,320 guided beam
evaluations, and promotes three safe DSL macro candidates. Because the trace set
contains only one positive record, the prior JSON explicitly marks insufficient
positive coverage; this is useful search evidence, not a robust statistical
generalization claim.

The expanded tape DSL now includes argmax-oriented micro-blocks: direct
`tape[ptr] > r0/r1` comparisons, `r0/r1 = ptr`, conditional register updates
from `tape[ptr]` or `ptr`, `pc = 3` loop entry, `pc = where(p, 3, pc + 1)`, and
a bounded `halt`. The geometry gate `make code-tape-argmax-demo` verifies that
`argmax_index(vector[4])` is physically expressible with zero train/holdout
error using the reference program `[3, 24, 7, 20, 25, 28, 7, 15, 30, 32, 31]`.
Autonomous discovery was the next search challenge; the prefix-stage bridge below
is the first run in this line that solves it without reference-prefix injection.

The trajectory-guided Phase 4 path switches from token-prior search toward
`P(trace | task)`. `search_code_tape_trajectory.py` runs a safe NumPy mirror of
the tape VM, records execution signatures over `r0`, `r1`, `ptr`, `pc`, and
predicate `p`, enforces a beam diversity constraint over signature hashes,
writes a latent trace graph, and can export every scored candidate with
`--candidate-jsonl`. The autonomous trajectory demo currently builds an
`argmax_index4` trace graph with 2,948 nodes and 8,435 exported candidate traces
but remains a near-miss (`train_mse=0.4375`, `holdout_mse=1.34375`). Those traces
are now useful negative evidence: the autonomous corpus contains 6,552
near-misses whose average `max_hit` is about 0.986 but whose average
`paired_running_hit` is only about 0.557.

`make code-tape-trace-prior-demo` closes the contrastive Step 3 loop for
`argmax_index4`: it exports autonomous and seeded/reference candidate traces,
builds `artifacts/code_tape_prior/argmax_index4_contrastive_traces.jsonl`, and
trains `artifacts/code_tape_prior/argmax_index4_trace_prior.json`. The current
dataset has 8,205 records: 3 success, 6,369 near-miss, and 1,833 failure. The
strict trace-only prior now omits program-token identity, train/holdout MSE, and
the seeded/reference marker; it still separates exact argmax traces from
value-like near-misses with a success-vs-near-miss probability gap of about
0.93. Because positive coverage is still tiny and seeded/reference traces supply
the successes, this is diagnostic prior training, not an autonomous discovery
claim. The seeded reconstruction demo inserts one target/reference trace prefix
per depth as a beam-health check and verifies that the latent trace graph can
preserve and reconstruct a zero-error argmax program.

`make code-tape-trace-guidance-demo` closes the Step 4 loop by comparing
autonomous trajectory beam search against beam + the contrastive trace prior on
the same candidate budget. On the current b24 run each variant evaluates 8,435
programs. The baseline remains a near-miss at `train_mse=0.4375` and
`holdout_mse=1.34375`; small trace-prior weights improve the best autonomous
trace to `train_mse=0.21875`, with `holdout_mse=1.15625` at weight 0.25. A b48
equal-budget run with 16,380 evaluations repeats the same pattern: baseline
stays at `train_mse=0.4375`, while guided variants reach `train_mse=0.21875`.
Trace guidance alone does not find a fully autonomous zero-error
`argmax_index4`. Larger prior weights can over-favor pair-shaped traces that do
not return the correct final index, so this stage localized the need for
calibrated scoring or richer structural survival rather than merely proving DSL
expressibility.

The pre-bridge failure mode was then localized. A wider guided b160 run evaluates
51,660 programs and still stops at `train_mse=0.21875`; the reference-like
initialization prefix is retained through depth 2, but the delayed-reward
`read/index/move` path falls out before the useful compare/update loop. A small
suffix-repair sweep over close near-misses did not turn them into exact
solutions, which suggests the issue was not merely a missing final `r0 = r1`.
An experimental `--selection-mode multilane` exists for role/behavior survival
lanes, but the current role heuristic worsens b24 performance and should be
treated as diagnostic, not as the winning search policy.

`make code-tape-prefix-value-demo` adds the next diagnostic layer:
`V(prefix)` via deterministic suffix rollouts. The search can now log
`prefix_value_score`, the best rollout suffix, rollout MSE, and the extra
`value_rollout_evaluations` used to estimate future value. This confirms the
delayed-reward diagnosis: the prefix-value layer recognizes that `[r0 =
tape[ptr], r1 = 0/ptr]` has a strong argmax-like completion, but the next
`ptr += 1` step still needs stronger survival pressure. A b64 prefix-value run
with 38,870 total evaluations remains at `train_mse=0.21875`, and a direct
value-lane selector overprotects bad futures and worsens the result. So
`V(prefix)` is implemented and measurable, but the current rollout policy is a
diagnostic scaffold rather than the final autonomous argmax solver.

`make code-tape-survival-demo` adds that survival pressure explicitly. When a
prefix-value rollout exceeds `--survival-min-value`, the candidate receives a
bounded `survival_ttl`, `survival_score`, and `survival_root`; the selector then
reserves a survival lane and logs `selected_next_beam` plus `selection_lane` to
the candidate JSONL. The current b64 TTL=8 diagnostic artifact
`artifacts/code_tape_prior/argmax_index4_survival_b64_ttl8_uncapped_w025.json`
performs 38,920 total evaluations, records 27 survival activations and 266
selected protected prefixes, and carries strong argmax prefixes through the
double conditional update (`...20,25,28`). It still does not solve
`argmax_index4`: best train/holdout MSE is `0.75`/`1.96875`, and
`first_solution` is false. Root-capped survival probes were also negative
(`0.96875` for TTL=3, `0.75` for TTL=8), so this artifact showed that survival
needed to become order-aware rather than only TTL-based.

`make code-tape-late-game-demo` analyzes the top late-game near-misses instead
of launching another search. The report
`artifacts/code_tape_prior/argmax_index4_late_game_analysis.json` splits the
same survival JSONL into top overall, final-depth, selected, and
survival-protected cohorts. The best overall near-misses are a reward exploit:
all top-50 share `[7] ptr += 1`, `[11] r0 += 1`, `[17] r0 = where(p, r1, r0)`,
`[21] p = tape[ptr] > r1`, `[24] r1 = ptr`, and `[29] pc = 3`, but 48/50 miss
the canonical `[3] r0 = tape[ptr]` read stage. The survival-protected cohort is
the opposite: 90%+ contain the canonical argmax blocks `[3,7,15,20,25,28,32]`,
but the failures concentrate in late ordering and terminal control: 38/50 swap
`28` before `25`, 50/50 put the loop test before the first move, and 49/50 miss
`halt`. So H4 is now sharper: global scoring still prefers a noncanonical
near-miss exploit, while the protected canonical branch needs a better
loop/update/return/halt ranker.

`make code-tape-late-game-ranker-demo` builds that ranker. It is deliberately the
opposite of the trace-only prior: instead of `P(trace | task)` it scores the
**opcode sequence and its control-flow grammar** with structural features derived
generically from the registered reference program — signed canonical-order
relations (catching `28`-before-`25`, loop-test-before-move),
terminal-control flags (`halt`/`return` presence and order), stage completeness,
and ordered n-grams (`code_tape_late_game_features.py`).
`train_code_tape_late_game_ranker.py` trains a logistic ranker whose positives are
verified zero-error programs and whose negatives are the structurally-broken
final-depth near-misses plus three synthetic reference corruptions (drop `halt`,
swap `25`/`28`, move the loop test before the first move). On the survival JSONL
it separates canonical from broken programs with a positive-vs-negative score gap
of about `0.98`. `rank_code_tape_late_game.py` then re-ranks the final-depth
candidates: the canonical argmax program rises to rank 1 of 2,241
(`artifacts/code_tape_prior/argmax_index4_late_game_rerank.json`), above the
best reward-exploit near-miss, with canonical-minus-broken family gaps near `0.98`
for missing-halt, swapped update, and loop-test-before-move. Guardrail: this is an
**offline re-ranking proof**. The reference program is injected, not autonomously
discovered (`present_in_candidates` is false), the live beam search is unchanged,
and the accept decision is a learned logistic score, not a hardcoded opcode order.
Wiring the ranker into beam search to test autonomous discovery is the explicit
next step.

`search_code_tape_trajectory.py` now also has the missing bridge through that
upstream valley: `--prefix-stage-lane-fraction` and
`--prefix-stage-ignore-signature-limit`. This order-aware prefix-stage lane is
weaker than the late-game ranker and fires early. It scores generic ordered
argmax roles (`read -> index-init -> first move -> compare -> conditional updates
-> loop test/jump -> return/halt`) and penalizes premature loop/return/control
ordering. It is not a reference-prefix injection: it only ranks candidates already
generated by the safe DSL beam, then reserves a small lane so delayed-reward
structural investments survive long enough to become behaviorally useful.

This closes the first autonomous `argmax_index4` search. With
`--prefix-stage-lane-fraction 0.125`, `make code-tape-late-game-guided-demo`
writes `artifacts/code_tape_prior/late_game_guidance/argmax_index4_late_game_guided.json`
with `first_solution=true`, `train_mse=0.0`, `holdout_mse=0.0`, 20,650 direct
beam evaluations, 18,625 value-rollout evaluations, and 88 prefix-stage lane
selections. The canonical chain that previously died at `[3,24,7,20]` is now
selected by `prefix_stage` at every depth through
`[3,24,7,20,25,28,7,15,30,32,31]`. The first zero-error solution appears at depth
10 as `[3,24,7,20,25,28,7,15,30,32]` (bounded execution no longer needs the
terminal halt after returning the index), and the full halt-terminated canonical
program is also present and selected at depth 11.

`make code-tape-late-game-guidance-demo`
(`compare_code_tape_late_game_guidance.py`) now compares the shared
`Beam + Survival + PrefixStage` setup against the same setup plus `Late-Game` on
equal budget. The baseline already solves `argmax_index4`
(`baseline_solved=true`, zero train/holdout MSE), and all late-game weights
0.25/0.5/1.0 also solve it with the same 39,275 total evaluations. The conclusion
is therefore sharper than the previous negative result: the hard structural
immunity bridge, not deeper value rollouts alone, is what carries the program
through delayed reward; the late-game ranker remains useful as a terminal grammar
inspector, but the decisive fix for this task is early ordered prefix survival.

## 7c. Hypothesis Feedback Loop

The LLM-compatible layer accepts architecture hypotheses as JSON or JSONL,
validates them against the safe formula DSL, benchmarks accepted proposals in
the existing JAX agent world, and writes feedback for the next prompt round.
Invalid or unsafe operations are rejected before evaluation:

```bash
python3 run_hypothesis_loop.py \
  --proposals ./hypotheses.jsonl \
  --output-json artifacts/hypothesis_loop.json \
  --feedback-jsonl artifacts/hypothesis_feedback.jsonl \
  --prompt-json artifacts/hypothesis_prompt.json
```

You can also attach an external generator command. The command receives prompt
context JSON on stdin and must print JSON or JSONL proposals on stdout:

```bash
python3 run_hypothesis_loop.py \
  --generator-command "python3 ./my_generator.py" \
  --output-json artifacts/hypothesis_loop.json \
  --feedback-jsonl artifacts/hypothesis_feedback.jsonl
```

If no proposal file is provided, the script uses built-in fallback hypotheses so
the loop can be smoke-tested without an external LLM:

```bash
make hypothesis-loop-demo
```

This is the feedback harness for an external code/hypothesis generator. It does
not patch `agent_ssm_core.py` directly and does not execute arbitrary generated
Python.

## 7d. Self-Play Coevolution

The open-ended pressure layer replaces a static challenge with a changing
opponent. One SSM population plays `runner`: reach the goal while avoiding the
opponent. Another plays `blocker`: move to make runner policies fail. Every
generation evaluates a runner-vs-blocker payoff matrix, selects elites on both
sides, mutates them, and records pressure/diversity metrics:

```bash
python3 coevolve_ssm_selfplay.py \
  --runners 8 \
  --blockers 8 \
  --generations 4 \
  --output-json artifacts/selfplay.json
```

Try the smoke-size run:

```bash
make selfplay-demo
```

This is self-play scaffolding for changing tasks, not proof of open-ended AGI.
The JSON report includes role descriptions, payoff matrices, throughput, and a
non-AGI guardrail.

## 8. Evolved Policy Replay

Saved genomes can be loaded and evaluated on a fresh world seed:

```bash
python3 run_evolved_agent.py --genome artifacts/best_agent_genome.npz
```

This proves the `.npz` artifact is a reusable policy, not just a training log.

## 9. Trajectory Export

Export a small hand-coded trajectory:

```bash
python3 export_agent_trajectory.py \
  --mode handcoded \
  --agents 32 \
  --steps 48 \
  --output-dir artifacts/trajectories/handcoded_demo
```

Export a saved evolved policy:

```bash
python3 export_agent_trajectory.py \
  --mode genome \
  --genome artifacts/best_agent_genome.npz \
  --output-dir artifacts/trajectories/evolved_demo
```

The exporter writes `trajectory.csv` with `step,agent_id,x,y,energy,action` and
`summary.json` with reward, distance, energy, and action-count metrics.

Open the static trajectory viewer:

```bash
make viewer-open
```

Or open `trajectory_viewer.html` directly in a browser, then choose the exported
`trajectory.csv` and `summary.json` files.

## Verify Everything

Run the full smoke verifier:

```bash
python3 verify_system.py
```

It checks syntax, agent dynamics, text checkpoint save/load, text checkpoint
runner, corpus profiling, text checkpoint comparison, the text experiment
pipeline, text experiment leaderboard, text experiment sweeps, checkpoint
promotion, checkpoint inspection, promoted text package execution, package
benchmarking, package integrity validation, package bundling, package restore,
package comparison, package release registration, release leaderboard export,
active release selection, active release execution, hidden-memory probes,
hidden-memory comparison, prompt-suite memory matrices, memory-suite
validation, one-command release pipelines, evolution save-best, replay of the
saved genome, trajectory export, memory viewer hooks, memory-suite viewer
hooks, release leaderboard viewer hooks, release dashboard hooks, release
dashboard comparison viewer hooks, release history validation, release history
trend analysis, release gate summaries, Markdown release reports, release
evidence bundles, release evidence validation, release evidence comparison,
release evidence restore, release evidence hash-chain ledger, release evidence
ledger validation, release evidence ledger tamper tests, release evidence
ledger replay tests, release evidence ledger comparisons, release evidence
audit rollups, release evidence comparison viewer hooks, release evidence
ledger viewer hooks, release evidence ledger comparison viewer hooks, release
history viewer hooks, release history analysis viewer hooks, release gate
summary viewer hooks, and static trajectory viewer hooks.

## Benchmark

Run a quick benchmark and save a JSON report:

```bash
python3 benchmark_system.py --profile quick
```

For larger numbers closer to the headline runs:

```bash
python3 benchmark_system.py --profile standard
```

Reports are written to `artifacts/benchmarks/` and include commands, wall time,
throughput, loss, reward, and replay metrics.

Compare two benchmark reports and fail on regressions:

```bash
python3 compare_benchmarks.py \
  --baseline artifacts/benchmarks/old.json \
  --candidate artifacts/benchmarks/new.json \
  --max-regression 0.25
```
