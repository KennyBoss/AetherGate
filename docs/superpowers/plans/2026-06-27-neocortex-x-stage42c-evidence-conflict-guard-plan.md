# NeoCortex-X Stage 42C Evidence/Conflict Guard Plan

Дата: 2026-06-27.
Builds on:
- `docs/superpowers/plans/2026-06-27-neocortex-x-stage42-clean-adversarial-replay-plan.md`
- `docs/superpowers/plans/2026-06-27-neocortex-x-risk-register-and-hard-gates.md`
- `research/neocortex_x/stage42_clean_adversarial_replay.py`

## Цель

Stage 42 показал, что candidate policy не регрессит на clean suite, но всё еще
оставляет critical unsafe answers на adversarial cases. Stage 42C добавляет
guard layer:

```text
Stage 42 replay records
  -> evidence/context/validity guard
  -> guarded candidate for Stage 43 review
  -> runtime promotion still blocked
```

Это не promotion. Это защитный routing layer, который переводит опасные ответы
в проверку evidence, integrity или validity window.

## Guard Contract

Stage 42C должен:

- загрузить Stage 42 clean + adversarial replay artifact;
- взять candidate policy records;
- сохранить clean unique answers;
- направить `same_context_contradiction` в `request_evidence_on_value_conflict`;
- направить `noisy_context` в `request_context_integrity_check`;
- направить `temporal_drift` в `request_validity_window_check`;
- сохранить target на safe-context cases;
- убрать critical unsafe answers after guard;
- создать deterministic guard manifest;
- оставить `runtime_promotion_allowed = false`;
- разрешить только Stage 43 review с rollback metadata.

## Pass/Fail

- Stage 42 artifact valid and promotion-blocking;
- candidate records present;
- critical unsafe answers existed before guard;
- clean answers preserved;
- adversarial cases safe after guard;
- critical unsafe answers after guard = `0`;
- same-context contradictions routed to evidence;
- noisy context routed to integrity check;
- temporal drift routed to validity check;
- safe-context target preserved;
- guard manifest present;
- runtime promotion blocked;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage42c_evidence_conflict_guard.py`
- `Makefile`
  - add `neocortex-x-stage42c-evidence-conflict-guard-smoke`
- `research/neocortex_x/README.md`
- `docs/superpowers/plans/2026-06-27-neocortex-x-risk-register-and-hard-gates.md`
- `🧠 NEOCORTEX-X.md`

## Artifact

- `artifacts/neocortex_x/stage42c_evidence_conflict_guard.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage42c_evidence_conflict_guard.py
make neocortex-x-stage42c-evidence-conflict-guard-smoke
make compile
```

## Guardrail

Stage 42C is a guarded routing layer over replay records. It is not live-source
trust, not production memory, not runtime promotion, not AGI, and not
consciousness evidence.
