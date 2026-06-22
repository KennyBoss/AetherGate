#!/usr/bin/env bash
# Reproduce the KV-Memory SSM headline evidence from scratch on CPU.
# Each target writes per-seed JSON + an aggregate summary.json with a
# protocol_evidence.valid gate under artifacts/long_context_recall_multiseed/.
set -euo pipefail

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

echo "==> [1/5] real-python-code (static AST of this repo), 3 seeds"
make compare-real-python-code-recall-multiseed

echo "==> [2/5] generated-python-code, 3 seeds"
make compare-generated-python-code-recall-multiseed

echo "==> [3/5] mixed-code, 3 seeds"
make compare-mixed-code-recall-multiseed

echo "==> [4/5] sparse-kv query-margin gate, hard (binary) eval, 6 seeds"
make compare-sparse-kv-routing-margin-multiseed

echo "==> [5/5] full-context control (Transformer sees whole sequence), 3 seeds"
make compare-real-python-code-full-context-multiseed

echo
echo "Done. Aggregate verdicts:"
for s in artifacts/long_context_recall_multiseed/*/summary.json; do
  python3 - "$s" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
try:
    ssm = d["summary"]["ssm"]["recall_accuracy"]["mean"]
    tr  = d["summary"]["transformer"]["recall_accuracy"]["mean"]
    win = d["summary"]["comparison"]["winner_counts"]["ssm"]
    valid = d["protocol_evidence"]["valid"]
    print(f"  {sys.argv[1]}: SSM={ssm:.4f} Transformer={tr:.4f} ssm_wins={win} protocol_valid={valid}")
except Exception:
    print(f"  {sys.argv[1]}: (unrecognized schema)")
PY
done
