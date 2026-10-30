# Запуски эволюции и гипотез

Установка из корня проекта (Python 3.12 или 3.13; Hydra 1.3 несовместима с 3.14):

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_evolution.py
.venv/bin/python run_evolution.py experiment=formulas
.venv/bin/python run_evolution.py experiment=selfplay experiment.parameters.seed=17
```

Параметры находятся в `configs/evolution/experiment/`. Hydra объединяет выбранный
эксперимент с общей конфигурацией; значения можно переопределять в командной
строке. Например, `experiment.parameters.proposals=/path/to/proposals.jsonl`.
Текущая рабочая папка сохраняется, поэтому относительные пути считаются от неё.
Результаты, обратная связь и `.hydra/` с настройками запуска сохраняются в
отдельной папке `artifacts/evolution/`. Для просмотра настроек без вычислений:
`run_evolution.py --cfg job`. Старые команды запуска скриптов продолжают работать.

Входные JSON-массивы и JSONL используют строгие модели Pydantic. Все четыре
операции формулы обязательны; допустимы только операции DSL, неизвестные поля
и неверные типы отклоняются. Метаданные `id`, `name`, `source`, `rationale`
необязательны и должны быть строками. Старый плоский формат формулы поддерживается.
Канонический формат:

```json
{"id":"probe","rationale":"Проверка базовой формулы","formula":{"memory_op":"identity","input_op":"identity","combine_op":"add","final_op":"tanh"}}
```

Формальная JSON Schema включена в `prompt.json`, передаваемый генератору.
Структурно повреждённый файл прекращает запуск; отдельная некорректная гипотеза
попадает в обратную связь как `rejected` и не передаётся на оценку.

Проверки из корня проекта:

```sh
make test-production PYTHON=.venv/bin/python
make ci PYTHON=.venv/bin/python
```

GitHub Actions запускает эти проверки на CPU при каждом pull request.
Для параллельной локальной проверки задайте отдельную папку результатов:
`VERIFY_ARTIFACT_DIR=artifacts/verify-local make ci PYTHON=.venv/bin/python`.
Это начальный инфраструктурный этап. Запуски ещё не изолированы контейнером;
`generator_command` — доверенная команда оператора, выполняемая на хосте.
Конфигурации и seed не гарантируют побитовую воспроизводимость на другом железе.
Распределённые вычисления, удалённое хранилище, мониторинг и полный lock-файл
окружения требуют отдельных изменений.
