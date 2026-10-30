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
При прямом запуске `run_evolution.py` изоляции контейнером нет:
команда `generator_command` выполняется на хосте. Для изоляции используйте Docker,
как описано ниже.
Конфигурации и seed не гарантируют побитовую воспроизводимость на другом железе.
Распределённые вычисления, удалённое хранилище, мониторинг и полный lock-файл
окружения требуют отдельных изменений.

## Изолированный запуск в Docker

Нужен работающий Docker Engine. Из корня проекта:

```sh
make docker-build
python3 run_evolution_docker.py --experiment hypothesis
python3 run_evolution_docker.py --experiment formulas experiment.parameters.seed=17
python3 run_evolution_docker.py --experiment selfplay experiment.parameters.generations=2
python3 run_evolution_docker.py --proposals /path/to/proposals.jsonl
```

Образ CPU используется локально и не публикуется в реестр. В контейнер копируются
только код эволюции, ядро, конфигурации и зависимости. Рабочая папка проекта,
Git, переменные окружения хоста, ключи и Docker socket внутрь не передаются.
Образ запускается непривилегированным пользователем, без сети и Linux capabilities,
с `no-new-privileges` и корневой файловой системой только для чтения.
Входной файл гипотез монтируется только для чтения. Команда генератора, если
задана через конфигурацию, также выполняется внутри этого офлайн-контейнера.

Каждый запуск получает новую папку `artifacts/evolution-docker/`; её можно выбрать
через `--output-dir` (папка должна ещё не существовать). В ней остаются JSON-результаты
и `.hydra/` с настройками. Лимиты: 2 CPU, 2 ГиБ RAM без swap, 256 процессов и
256 МиБ временной памяти. `--timeout` ограничивает длительность (по умолчанию
600 секунд); при завершении и тайм-ауте контейнер удаляется. Лаунчер запускайте
от обычного пользователя. Docker bind mount не поддерживает здесь запятые в пути.

```sh
make docker-test PYTHON=python3
```

CI собирает этот же образ, запускает все три режима с малым бюджетом и проверяет
ограничения внутри настоящего контейнера, после чего выполняет обычный системный
прогон. При локальном запуске тестов без `RUN_DOCKER_TESTS=1` выполняются только
проверки команды и обработки тайм-аута. Параметры изоляции описаны в
[документации Docker](https://docs.docker.com/reference/cli/docker/container/run/).

Контейнеры используют общее ядро с Docker Engine; это не отдельная виртуальная
машина. Для исполнения враждебного произвольного кода нужна более строгая граница,
например выделенный worker с gVisor или microVM. Этот образ поддерживает только CPU.
