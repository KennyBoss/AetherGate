# NeoCortex-X Stage 22 Contextual Runtime Loader Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage21-contextual-policy-promotion-plan.md`
- `research/neocortex_x/stage21_contextual_policy_promotion.py`

## Цель

Stage 22 загружает promoted contextual manifest как runtime snapshot:

```text
contextual manifest -> runtime contextual policy -> smoke decisions
```

Это контекстный аналог Stage 14. Важное отличие: runtime policy не имеет одного
глобального default action. Она хранит bucketed action order:

```text
context_decision -> ordered actions
```

## Runtime Contract

Runtime snapshot содержит:

- stable `runtime_policy_id`;
- source `manifest_id`;
- `policy_id`;
- per-decision probabilities;
- per-decision action order;
- per-decision default action;
- rollback metadata;
- replay hash.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 21 promotion artifact valid and passed;
- every context decision bucket sums to `1.0`;
- all required context decisions available;
- top action for each context decision matches manifest top action;
- rollback metadata present;
- smoke decisions cover all context decisions;
- unknown context decision falls back to `ask_context_question`;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage22_contextual_runtime_loader.py`
- `Makefile`
  - add `neocortex-x-stage22-contextual-runtime-loader-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage22_contextual_runtime_loader.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage22_contextual_runtime_loader.py
make neocortex-x-stage22-contextual-runtime-loader-smoke
```
