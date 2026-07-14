# Official Mamba CUDA Run

This is the external NVIDIA-only baseline required by KV Scale Law 1. It uses
the official `state-spaces/mamba` `Mamba` block and the byte-identical registered
MQAR generator. A local JAX selective SSM must not be substituted.

## Environment

The official project currently requires Linux, an NVIDIA GPU, CUDA, and a
CUDA-enabled PyTorch installation. Install PyTorch for the machine's CUDA
version first, then follow the official package command:

```bash
python -m pip install 'mamba-ssm[causal-conv1d]' --no-build-isolation
```

Official source and installation contract:
<https://github.com/state-spaces/mamba/blob/main/README.md>

## Registered cap-100k command

From a clean checkout of this repository:

```bash
python run_mamba_mqar.py \
  --output-json artifacts/kv_scale_law_1/registered/cap-100k/mamba.json \
  --point cap-100k \
  --seeds 11,17,23 \
  --target-params 100000 \
  --train-records 2048 \
  --eval-records 1024 \
  --key-count 128 \
  --binding-count 12 \
  --query-count 6 \
  --delay 192 \
  --epochs 40 \
  --batch-size 32 \
  --learning-rate 0.006
```

The runner refuses non-CUDA execution, chooses the closest one-block Mamba-1
dimension, requires a 0.8–1.25 parameter ratio, trains all three registered
seeds with query-only MQAR loss, and records CUDA/PyTorch/Mamba versions, GPU
identity, data hashes, timings, recall loss, and recall accuracy.

## Import into the gate

Copy the JSON back without editing it, then validate it while rebuilding the
aggregate from completed artifacts:

```bash
python run_kv_scale_law.py \
  --points cap-100k \
  --controls full,capped \
  --output-dir artifacts/kv_scale_law_1/registered \
  --backend cpu \
  --resume \
  --skip-lm \
  --allow-expensive \
  --mamba-report artifacts/kv_scale_law_1/registered/cap-100k/mamba.json
```

The importer recomputes every train/eval MQAR data hash and rejects protocol,
seed, parameter-budget, implementation, package, CUDA, metric, or query-count
drift. Supplying a report marks only that registered point complete; later
capacity points require their own equivalent-budget reports.
