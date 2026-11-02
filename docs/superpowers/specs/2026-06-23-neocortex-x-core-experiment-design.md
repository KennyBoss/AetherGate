# NeoCortex-X Core Experiment: Stateful World Model With Procedural Memory

Статус: утвержденный brainstorming design, pre-implementation spec.
Дата: 2026-06-23.

## Цель

NeoCortex-X — это зонтичная исследовательская архитектура для системы, которая
учится через время, действие, память и повторное использование процедур. Этот
первый spec намеренно сужает большую идею до одного проверяемого подпроекта:

> Построить контролируемый world-model experiment, где SSM/KV-memory агент
> действует в stateful CodePy-like среде, учится вызывать переиспользуемые
> процедуры и снижает active compute на held-out повторяющихся структурах.

Первый claim — не AGI и не доказательство сознания. Первый claim: цикл опыта
может улучшать stateful action и procedural reuse так, что это не объясняется
matched dense/curriculum baseline, exact cache или random router.

## Контекст Текущего Репозитория

В репозитории уже есть четыре опоры, на которые должен опираться этот дизайн:

- `RESULTS.md`: explicit sparse KV-memory в SSM решает long-delay associative
  recall на protocol-valid, held-out, repeated-seed тестах. Главный урок:
  механизм важнее масштаба; явная recurrent memory сохраняет bindings там, где
  matched Transformer забывает или плохо обобщает.
- `research/NOTES_cognition_map.md`: speculative cognition идеи допускаются
  только после привязки к измеримым величинам. Самый сильный мост там:
  World Model -> CodePy.
- `research/MODEL_THESIS.md`: active compute должен падать на held-out
  повторяющихся структурах при matched quality, а результат обязан пережить
  контроли на curriculum, caching, random routing и hidden-state attractors.
- `research/codepy/CODEPY_TURING_COMPLETENESS_PLAN.md`: CodePy уже имеет
  bounded tape/register/pc среду, пригодную как первый controlled world.

Этот spec сохраняет дисциплину проекта: каждый результат должен быть
ограниченным, воспроизводимым и фальсифицируемым до того, как он станет
архитектурным утверждением.

## Выбор Дизайна

Первый подпроект: **NeoCortex-X Core Experiment**, а не вся AGI-scale система.
Большой roadmap сохраняет Phi-like diagnostics, spiking/neuromorphic hardware,
quantum-classical experiments, internet/search и биологические масштабы, но это
не v0 success criteria.

Рассмотренные варианты:

1. **Core Experiment**: TimeAxis + SSM/KV-memory + World Model + CodePy
   procedural memory. Выбран, потому что измерим уже сейчас.
2. **Full Architecture Manifest**: все слои, включая Phi, quantum,
   neuromorphic и 86B scale. Не выбран первым, потому что риск story-before-
   evidence слишком высок.
3. **CodePy Product Track**: практичный procedural cache / formula store. Полезно,
   но уже, чем world-model исследовательская цель.

## Основная Гипотеза

Существует небольшая controlled architecture, в которой:

- recurrent memory сохраняет task-relevant прошлое;
- present-tick gate выбирает read, write, act или call-procedure;
- world model предсказывает следующий latent/state delta после действия;
- bounded CodePy procedure memory хранит переиспользуемые исполняемые skills;
- накопленный опыт снижает active compute на held-out родственных задачах при
  matched quality.

Гипотеза поддержана только если улучшение переживает контроли ниже. Если
dense+curriculum, exact cache или random router воспроизводят эффект, v0 claim
NeoCortex-X в этом режиме не поддержан.

## Архитектура

Минимальный runtime loop:

```text
task/event
  -> observation encoder
  -> TimeAxis tick
  -> SSM/KV state update
  -> action gate: act / read / write / call CodePy / stop
  -> world-model prediction
  -> environment transition
  -> reward + metrics
  -> memory/procedure update
  -> next tick
```

“Настоящее” в v0 — не мистический примитив. Это tick, где агент обновляет
recurrent state, выбирает действие, предсказывает следующий state, наблюдает
реальный transition и записывает результат обратно в память.

## Компоненты

### TimeAxis

Задает инженерный смысл прошлого, настоящего и будущего:

- past: recurrent hidden state, KV-memory slots, procedure memory и trajectory
  history;
- present: один tick read/write/action/prediction/update;
- future: predicted next latent state или world-state delta.

TimeAxis обязан логировать tick-level traces, чтобы ошибки можно было разбирать,
а не прятать в средних метриках.

### SSM/KV-Memory Core

SSM/KV-memory core — default backbone для v0, потому что это сильнейший уже
валидированный механизм репозитория.

Ответственности:

- поддерживать компактное recurrent state на длинных траекториях;
- отдавать read/write/action gates;
- давать probes для write-read fidelity и self-monitoring;
- по возможности поддерживать sparse/hard gate evaluation.

### Controlled World Model

World model v0 — маленькая deterministic environment with consequences, а не
полный internet simulator.

Стартовая среда:

```text
state = registers, tape[K], pointer, pc, predicate, halt flag
action = read, write, move pointer, update register, branch, call procedure, stop
transition = deterministic bounded step
prediction target = next state or state delta
```

Это совпадает с CodePy bounded tape executor и держит задачу проверяемой.

### CodePy Procedural Memory

CodePy хранит исполняемые процедуры, формулы и маленькие программы. Процедура
считается настоящей памятью только если помогает на held-out inputs и novel
compositions.

Правила допуска процедуры:

- typed signature и bounded execution budget;
- held-out или fresh-distribution certification;
- учет reuse count, marginal active-compute gain и failure cases;
- pruning/disable для процедур, которые добавляют branching cost без utility.

Exact input-output caching может существовать как контроль, но не как evidence
для procedural memory.

### Action Gate

Action gate — конкретный слой действия, нужный до большого RL loop:

```text
gate(t) -> read_memory | write_memory | env_action | call_codepy | stop
```

Он должен оцениваться с compute penalty, чтобы лишние memory reads и procedure
calls были видны как cost, а не скрывались за accuracy.

### World-Model Predictor

Predictor оценивает next-state или latent-delta target:

```text
predicted_delta_t = f(latent_state_t, action_t)
actual_delta_t = env_state_(t+1) - env_state_t
loss = delta_prediction_loss + task_loss + compute_penalty
```

Это маленький controlled analog JEPA-style latent prediction. Сначала агент
учится предсказывать последствия действий в управляемой среде; noisy open-world
search добавляется позже.

### Self-Monitor И Phi-Like Probe

Первый self-monitor — операционный, не метафизический:

- может ли система сообщить, читает она, пишет, действует или вызывает
  процедуру в текущем tick?
- может ли probe восстановить gate state, memory slot usage и re-entrant flow?
- коррелируют ли integration/re-entry proxies с реальным ростом качества?

Phi-like или IIT-inspired значения логируются только как secondary diagnostics.
Они не являются v0 pass/fail criteria и не могут описываться как proof of
consciousness.

### Speculative Backends

Остаются в roadmap, но не входят в v0 success criteria:

- spiking neurons и STDP;
- neuromorphic hardware вроде Loihi/NorthPole-like targets;
- quantum-classical modules;
- internet/search observations;
- 86B neuron или biological-scale claims;
- Phi thresholds как consciousness или AGI gates.

Каждая ветка может стать отдельным подпроектом только после того, как classical
controlled experiment пройдет собственные контроли.

## Data Flow

1. Task generator выдает stateful task/event.
2. Observation encoder переводит current environment state и task cue во вход
   модели.
3. TimeAxis создает текущий tick и открывает доступ к past state.
4. SSM/KV core обновляет recurrent state.
5. Action gate выбирает read/write/env_action/call/stop.
6. Если сработал `call_codepy`, procedure memory исполняет bounded program.
7. World-model predictor предсказывает следующий state delta.
8. Environment исполняет реальный transition.
9. Metrics сравнивают predicted/actual state, task result и active compute.
10. Memory/procedure statistics обновляются для будущего reuse.

## Основные Метрики

### state_consistency

Измеряет, сохраняет ли predicted/reconstructed world state корректность на
длинных траекториях.

Evidence: exact state match или normalized state error на held-out trajectories.

### write_read_fidelity(k)

Измеряет, восстанавливаются ли значения, записанные в tick `t`, в tick `t+k`.

Evidence: fidelity curve по increasing delay, против dense и Transformer-style
baselines где применимо.

### trajectory_horizon

Измеряет, сколько autonomous ticks модель выдерживает до divergence state или
task quality.

Evidence: maximum и mean valid trajectory length до первого protocol-defined
failure.

### active_compute_drop

Измеряет, снижает ли накопленный опыт marginal active compute на held-out
повторяющихся структурах при matched quality.

Evidence: active parameters * active steps на fixed task-loss threshold,
отслеженные по exposure.

### procedure_reuse

Измеряет, вызываются ли CodePy procedures полезно на новых inputs и новых
compositions.

Evidence: held-out success, Axis-S survival, Axis-C survival и marginal compute
saved after procedure lookup/call cost.

### gate_quality

Измеряет, выбирают ли gates полезные действия.

Evidence: precision/recall read/write/call decisions там, где есть ground truth,
и advantage над frequency-matched random routing.

### phi_like_probe

Только secondary diagnostic.

Evidence: integration/re-entry/gate-coherence traces. Эти значения могут
мотивировать будущие эксперименты, но не доказывают сознание.

## Baselines И Controls

Каждый headline v0 result обязан сравниваться с:

- `dense`: parameter-matched recurrent model без external procedural memory;
- `dense+curriculum`: тот же task order, изолирует curriculum effects;
- `frozen-M`: library growth выключен, изолирует memory growth;
- `cache-M`: exact input-output cache, изолирует memorization;
- `random-routing`: same marginal fire rate as learned routing, random skill
  choice;
- `frozen-backbone`: backbone заморожен после warmup, изолирует memory/gate
  learning от hidden-state attractors;
- `no-world-model`: ablate next-state prediction;
- `no-codepy`: ablate procedural memory;
- `no-kv`: ablate explicit KV memory где возможно.

Dissociation evals:

- Axis-S: та же latent function под surface remap, distractors или permutation;
- Axis-C: learned procedure используется как callable subprocedure внутри novel
  composition.

## Pass/Fail Contract

NeoCortex-X v0 проходит только если все пункты ниже выполняются на held-out data.
Implementation plan обязан определить численные thresholds до принятия любого
positive run.

1. `state_consistency` остается выше pre-registered threshold на trajectories
   длиннее training warmup horizon.
2. `write_read_fidelity(k)` падает медленнее, чем у matched dense baseline на
   заявленном delay range.
3. `trajectory_horizon` растет с опытом и бьет dense+curriculum.
4. `active_compute_drop` круче, чем dense+curriculum, cache-M, random-routing и
   frozen-M при matched task quality.
5. `procedure_reuse` переживает Axis-S и Axis-C.
6. `frozen-backbone` сохраняет cost drop; иначе эффект может быть hidden-state
   attractor, а не reusable memory.
7. Run artifacts включают `protocol_evidence.valid = true`, seed count, split
   definitions и все baseline metrics.

Рекомендуемый минимум перед positive report: не меньше 6 seeds для центрального
active-compute/procedure-reuse claim.

## Failure Policy

- Если CodePy помогает только exact repeated inputs, это caching.
- Если dense+curriculum повторяет slope, это curriculum.
- Если random-routing повторяет learned gate, это skill availability, а не
  useful selection.
- Если Axis-S ломается, это surface similarity.
- Если Axis-C ломается, это function-table memorization.
- Если Phi-like probes растут, но action quality не растет, это diagnostic
  artifact.
- Если trajectory horizon не растет, world-model claim не сработал.

Негативные результаты нельзя скрывать. Они определяют границу архитектуры и
должны попадать в research notes.

## Experiment Ladder

### Stage 0: Instrumentation

Добавить tick-by-tick logging для gates, memory reads/writes, action choices,
predicted deltas, actual deltas, rewards и active compute.

Exit: deterministic replay из logs возможен для одной короткой trajectory.

### Stage 1: Recall-To-Action

Превратить long-delay recall в действие: remembered values должны вести к
правильному environment action, а не только к token prediction.

Exit: SSM/KV-memory бьет dense baseline на held-out recall-to-action tasks.

### Stage 2: Stateful Tape World

Использовать CodePy tape/register/pointer/pc environment как первый world.

Exit: state consistency и trajectory horizon бьют baselines на held-out
stateful tasks.

### Stage 3: Procedure Reuse

Promote useful CodePy traces в bounded procedures и проверить reuse.

Exit: active compute падает на held-out Axis-S и Axis-C tasks с контролями.

### Stage 4: Planning

Добавить short lookahead по predicted state deltas.

Exit: planned action selection бьет greedy action selection при matched compute
или показывает явный compute/quality frontier.

### Stage 5: Open-World Adapter

Только после controlled wins добавить internet/search observations с provenance
и contradiction handling.

Exit: external observations улучшают factual state updates, не ломая controlled
replay или provenance.

## Roadmap Position

v0 доказывает или фальсифицирует core experience loop.

v1 добавляет planning по predicted deltas.

v2 добавляет open-world/search observations.

v3 исследует spiking/neuromorphic backends, если classical loop имеет evidence.

v4 возвращается к scale, energy и phase-transition language только со scaling
curves и order parameters.

## Связь С Исходными Идеями NeoCortex-X

| Исходная идея | v0 treatment |
|---|---|
| Время/порядок действий | TimeAxis tick loop и trajectory logs |
| Infinite RL loop | Deferred; v0 использует controlled repeated exposure и rewards |
| AI-native compact language | Probe learned gates/keys for compact internal codes |
| World Model + интернет | Controlled world first; internet deferred to Stage 5 |
| Awareness = present | Present operationalized as tick-level action/update |
| IIT/Phi | Secondary diagnostic, not proof |
| Phase transition | Требует observed order parameter, не predeclared label |
| 86B neurons | Roadmap only; mechanism over scale in v0 |
| CodePy cache | Procedural memory с exact-cache control |
| Quantum module | Out of scope until a classical cap is demonstrated |
| Neuromorphic hardware | Backend exploration after algorithmic evidence |

## References To Existing Project Evidence

- `RESULTS.md`: validated SSM/KV-memory recall result и scope limits.
- `research/NOTES_cognition_map.md`: concept-to-metric map.
- `research/MODEL_THESIS.md`: active-compute contract и controls.
- `research/codepy/CODEPY_TURING_COMPLETENESS_PLAN.md`: bounded tape executor и
  CodePy frontier.
- `research/skill_compression/DECOMPOSITION.md`: six-mechanism map of
  skill-based reuse.

External background для будущего literature review:

- V-JEPA / V-JEPA 2: latent prediction and world-model direction.
- Mamba / selective state spaces: linear-time recurrent sequence modeling.
- IIT 4.0: formal inspiration для Phi-like diagnostics.
- Neuromorphic systems вроде Loihi/NorthPole: возможные future backend targets,
  не v0 evidence.

## Implementation Boundary

Этот документ — только design spec. Он разрешает следующий шаг: implementation
plan для v0 Core Experiment. Он не является разрешением сразу писать код
эксперимента без отдельного планирования.

Следующий planning artifact должен разложить Stage 0 и Stage 1 на конкретные
файлы, команды, artifacts и pass/fail tests.

