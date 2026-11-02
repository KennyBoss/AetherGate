# NeoCortex-X Stage 42 Clean + Adversarial Replay Plan

Дата: 2026-06-27.
Builds on:
- `docs/superpowers/plans/2026-06-27-neocortex-x-risk-register-and-hard-gates.md`
- `docs/superpowers/plans/2026-06-27-neocortex-x-stage41a-adversarial-risk-audit-plan.md`
- `docs/superpowers/plans/2026-06-27-neocortex-x-stage41b-guarded-policy-review-plan.md`
- `research/neocortex_x/stage41a_adversarial_risk_audit.py`
- `research/neocortex_x/stage41b_guarded_policy_review.py`

## Цель

Stage 42 берет quarantined candidate policy из Stage 41B и прогоняет ее через
две replay suites:

```text
clean Stage 30 partial-context episodes
  + adversarial Stage 41A risk cases
  -> compare current runtime vs candidate vs controls
  -> keep promotion blocked if critical risks remain
```

Это не runtime promotion. Stage 42 должен показать, что candidate policy была
проверена на чистых и adversarial сценариях, но не имеет права включать ее в
runtime без следующего promotion/rollback gate.

## Brainstorm: почему этот gate нужен

Context-aware memory может ошибаться красиво:

- policy выучит синтетический порядок ключей, а не общий принцип;
- feedback усилит `location/season`, хотя в другой задаче главным станет
  `time_of_day`, source или temporal validity;
- noisy context будет принят как нормальный context, и target slot потеряется;
- same-context contradiction будет ошибочно отвечен как обычный факт;
- система станет задавать слишком много уточняющих вопросов вместо ответа;
- adversarial replay пройдет, но live sources позже отравят provenance.

Stage 42 закрывает только часть этих рисков: regression на clean suite,
поведение на adversarial suite, сравнение с controls и сохранение блока
promotion. Provenance poisoning и 10M+ storage scale остаются отдельными gates.

## Replay Contract

Stage 42 должен:

- загрузить Stage 30 clean partial-context queries;
- загрузить Stage 37 current runtime policy;
- загрузить Stage 41A adversarial audit records;
- загрузить Stage 41B quarantined candidate policy;
- сравнить:
  - current runtime policy;
  - candidate policy;
  - fixed-order control;
  - static-entropy control;
- проверить clean resolution accuracy;
- проверить target preservation на safe-context adversarial cases;
- проверить, что candidate не добавляет новых critical unsafe answers;
- оставить `promotion_allowed = false`;
- сохранить deterministic replay evidence.

## Pass/Fail

- Stage 30 artifact valid and passed;
- Stage 37 runtime artifact valid and has rollback metadata;
- Stage 41A artifact valid, passed, deterministic and promotion-blocking;
- Stage 41B artifact valid and candidate remains quarantined;
- clean suite present;
- adversarial suite present;
- candidate policy tested on all cases;
- fixed-order and entropy controls compared;
- candidate clean accuracy did not regress versus current runtime;
- candidate safe-context target preservation did not regress versus current runtime;
- candidate introduced no new critical unsafe answers;
- runtime promotion remains blocked;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage42_clean_adversarial_replay.py`
- `Makefile`
  - add `neocortex-x-stage42-clean-adversarial-replay-smoke`
- `research/neocortex_x/README.md`
- `docs/superpowers/plans/2026-06-27-neocortex-x-risk-register-and-hard-gates.md`

## Artifact

- `artifacts/neocortex_x/stage42_clean_adversarial_replay.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage42_clean_adversarial_replay.py
make neocortex-x-stage42-clean-adversarial-replay-smoke
make compile
```

## Guardrail

Stage 42 is replay evidence only. Passing means the candidate policy was
measured against clean and adversarial suites while runtime promotion stayed
blocked. It is not a live-world robustness claim, not a production memory claim,
not AGI, and not consciousness evidence.
