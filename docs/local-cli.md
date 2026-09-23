# Локальный CLI

[README](../README.md) · [Данные](data.md) · [Конвейер и контракты](pipeline.md) · [Docker](docker.md)

Нужны Python 3.13+ и [uv](https://docs.astral.sh/uv/). Из корня репозитория:

```bash
uv sync --frozen
uv run buy_today --help
```

Примеры ниже — для Bash. Относительные пути разрешаются от рабочего каталога
команды. Подробная справка: `uv run buy_today <команда> --help`.
Коды завершения: `0` — успех, `2` — неверные аргументы, `1` — ошибка исполнения.
У `run`, `inference`, `summary` ошибки исполнения выходят с traceback.

## Подготовка

Заранее распакуйте [исходные CSV](data.md#исходные-файлы) в `dataset/`:

```bash
uv run buy_today prepare --raw-dir dataset --output-dir data/olist-stream \
  --batch-size 5000 --min-category-count 1000
```

Оба пути обязательны. Размер последующих батчей по умолчанию — 5000,
минимальная частота категории — 1000. [Правила и метаданные](data.md#правила-подготовки).

## Полный прогон

Один вызов `run` обрабатывает один следующий батч; `--all` — весь остаток потока.
При первом вызове обязательны подготовленный `--data-dir` и **новый** `--run-dir`:

```bash
uv run buy_today run --data-dir data/olist-stream \
  --run-dir models/olist-stream/runs/example --model svd
uv run buy_today run --run-dir models/olist-stream/runs/example
uv run buy_today run --run-dir models/olist-stream/runs/example --all
uv run buy_today summary --run-dir models/olist-stream/runs/example
```

Выберите отсутствующий каталог для нового эксперимента. Один прогон использует
один ранжировщик; для Random создайте отдельный прогон с `--model random`.
На исчерпанном потоке `run` успешно сообщает, что новых батчей нет.
Сводка локально вызывается отдельно, а в Docker строится автоматически.

### Параметры прогона

Настройки сохраняются в `config.json` при создании прогона. Эта таблица также
относится к контейнерному `init`.

| Параметр | По умолчанию |
| --- | --- |
| `--model` | `svd` |
| `--random-state` | `42` |
| `--temperature` | `86400` секунд для TimestampDistance |
| `--initial-users` / `--additional-users` | `2000` / `250` новых пользователей |
| `--split-sizes TRAIN VALIDATION TEST` | `70 15 15` событий пользователя |
| `--k` | `10` |
| `--svd-n-components` / `--svd-n-iter` | `32` / `7` |
| `--temporal-n-clusters` | Определяет временная модель: `20` |
| `--max-evaluation-rows` | `1000` |
| `--price-threshold`, `--category-threshold`, `--state-threshold` | `0.10` каждый |

При продолжении явно переданные параметры должны совпадать с сохранёнными,
остальные наследуются. SVD-флаги требуют `--model svd`. K и размерность SVD
проверяются по фактическим данным, автоматического уменьшения нет.
Обучение ранжировщика и обе оценки выполняются с одним потоком BLAS/OpenMP;
отдельное ядро оценки кластеризации тоже получает лимит `1`.
[Порядок этапов и состояние](pipeline.md#полный-цикл).

## Inference и summary

```bash
uv run buy_today inference \
  --model-dir models/olist-stream/runs/example/steps/step_000/ranking \
  --user-id user_000000_000000 --k 10 \
  --output models/olist-stream/runs/example/recommendations.csv

uv run buy_today summary --run-dir models/olist-stream/runs/example
```

Для локального inference `--model-dir`, `--user-id`, `--output` обязательны;
K по умолчанию 10. Модель выбирается явно, пользователь должен быть ей известен.
Входные истории не требуются. CSV содержит `user_id,rank,product_id`, ранг от 1.
Путь результата должен находиться вне каталога модели.

`summary` собирает завершённые шаги в `summary/summary.{html,json,csv}` внутри
прогона; повторный вызов обновляет файлы. [Состав и проверки](pipeline.md#сводка).

## EDA и эталон

Отдельный EDA и создание эталона из первого батча:

```bash
uv run buy_today eda --dataset data/olist-stream/batches/batch_000.csv \
  --output-dir data/manual-reports/eda
uv run buy_today init --batch data/olist-stream/batches/batch_000.csv \
  --reference data/olist-stream/reference.csv \
  --output-dir data/olist-stream/reports/step_000
```

Сравнение со следующим батчем и обновление эталона:

```bash
uv run buy_today deda --reference data/olist-stream/reference.csv \
  --batch data/olist-stream/batches/batch_001.csv \
  --output-dir data/manual-reports/deda
uv run buy_today update --reference data/olist-stream/reference.csv \
  --batch data/olist-stream/batches/batch_001.csv \
  --output-dir data/olist-stream/reports/step_001
```

Все показанные пути обязательны. `eda` и `deda` создают `report.ipynb`,
`report.html`, `metrics.json` непосредственно в `--output-dir`; `init` — в `eda/`,
`update` — в `deda/` и `eda/` внутри каталога шага. Для истории задавайте отдельный
каталог на шаг: одноимённые отчёты перезаписываются.

`init` заменяет эталон. `update` требует существующий эталон и выполняет DEDA
перед добавлением батча, затем EDA обновлённого CSV. `deda` входы не изменяет.
Манифест подготовки и `state.json` этим командам не нужны.
Это операции с данными и отчётами; полный модельный цикл выполняет `run`.

У `deda` и `update` доступны `--price-threshold`, `--category-threshold`,
`--state-threshold` (все 0.10). Допустимы конечные числа в `[0, 1]`; дрейф при
достижении порога не блокирует обновление. [Проверки и поведение при ошибках](pipeline.md#deda-и-обновление).

## Кластеризация

Для отдельного эксперимента достаточно первого батча, предварительный `init` не нужен:

```bash
uv run buy_today cluster --batch data/olist-stream/batches/batch_000.csv \
  --output-dir models/olist-stream/clustering/example/step_000 \
  --model temporal --distance timestamp --temporal-n-clusters 20
uv run buy_today evaluate --dataset data/olist-stream/batches/batch_000.csv \
  --new-batch data/olist-stream/batches/batch_000.csv \
  --model-dir models/olist-stream/clustering/example/step_000 \
  --max-evaluation-rows 1000 --random-state 42
```

`cluster` требует `--batch` и `--output-dir`; доступны только `temporal` и
`timestamp`, они же значения по умолчанию. Число кластеров — положительное целое,
не больше числа строк; если не задано, модель выбирает 20.

`evaluate` требует все три пути. `--dataset` — накопленный оцениваемый датасет,
`--new-batch` — последний **уже включённый** в него батч. На первом шаге они
совпадают. Лимит оценки по умолчанию 1000, seed 42.

На следующем шаге обучайте кластеризацию на обновлённом `reference.csv`,
используйте его как `--dataset`, а последний батч — как `--new-batch`.
Задавайте отдельный каталог модели для каждого шага. Обучение сохраняет
`model.joblib`, `distance.joblib`, `labels.csv`; оценка добавляет notebook, HTML
и метрики в `--model-dir`, не меняя модель и метки.
[Алгоритм, выборка и контракты](pipeline.md#временная-кластеризация).

## Генерация историй

После сохранения расстояния командой `cluster`:

```bash
uv run buy_today generate --batch data/olist-stream/batches/batch_000.csv \
  --distance models/olist-stream/clustering/example/step_000/distance.joblib \
  --output-dir data/olist-stream/histories/example/step_000 --temperature 86400
uv run buy_today generate --batch data/olist-stream/batches/batch_001.csv \
  --distance models/olist-stream/clustering/example/step_000/distance.joblib \
  --output-dir data/olist-stream/histories/example/step_001 \
  --previous-dir data/olist-stream/histories/example/step_000 --temperature 86400
```

`--batch`, `--distance`, `--output-dir`, `--temperature` обязательны. Температура —
конечное положительное число **в единицах расстояния**, здесь секунды. Для другого
расстояния требуется собственная калибровка. TimestampDistance не имеет обученного
состояния, поэтому его артефакт можно использовать на обоих шагах.

- `--previous-dir` — предыдущий накопленный снимок; без него создаётся первый шаг.
- `--n-users` — новые пользователи: сначала 2000, затем 250 по умолчанию.
- `--split-sizes TRAIN VALIDATION TEST` — положительные длины частей: сначала
  `70 15 15`, затем наследуются; изменение при продолжении — ошибка.
- `--random-state` — целое в `[0, 2**32 - 1]`, по умолчанию 42.

Индекс шага определяется предыдущим снимком, а не именем CSV. Выходной каталог
должен быть **новым и вне предыдущего снимка**. Результат — накопленные
`train.csv`, `validation.csv`, `test.csv`, `catalog.csv` и `generator/` с
происхождением событий. [Правила генерации и формат снимка](pipeline.md#модельные-истории).

## Качество историй

```bash
uv run buy_today history-quality \
  --dataset-dir data/olist-stream/histories/example/step_000 \
  --reference data/olist-stream/batches/batch_000.csv \
  --output-dir data/olist-stream/history-quality/example/step_000 \
  --iid-repeats 100 --random-state 42
```

Три пути обязательны. `--reference` — один рабочий CSV с точной совокупностью
исходных батчей этого снимка; пропущенные и лишние ключи — ошибка. Для накопленного
снимка заранее объедините его батчи. Выходной каталог должен быть новым и вне
snapshot. `--iid-repeats` — положительное целое (100), seed — `[0, 2**32 - 1]` (42).

В `metrics.json` сохраняются общие метрики по **train + validation + test**:
TVD категорий, концентрация главной категории, inverse Simpson, привязка к anchor
и сравнение с IID. Это report-only, без критериев допуска генерации.
[Формулы, валидация, JSON и API](history-quality.md).

## Ранжирование

Обучение и независимая оценка на снимке из предыдущего раздела:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run buy_today rank --dataset-dir data/olist-stream/histories/example/step_000 \
  --model svd --svd-n-components 32 --svd-n-iter 7 --random-state 42 \
  --output-dir models/olist-stream/ranking/example/step_000
uv run buy_today evaluate-ranking --dataset-dir data/olist-stream/histories/example/step_000 \
  --model-dir models/olist-stream/ranking/example/step_000 \
  --split validation --k 10 --output-dir models/olist-stream/ranking/example/step_000/validation
# Test запускайте после фиксации настроек по validation.
uv run buy_today evaluate-ranking --dataset-dir data/olist-stream/histories/example/step_000 \
  --model-dir models/olist-stream/ranking/example/step_000 \
  --split test --k 10 --output-dir models/olist-stream/ranking/example/step_000/test
```

`rank` требует пути датасета и результата. В отличие от `run`, его модель
по умолчанию — **random**, seed 42. SVD-флаги допустимы только с `--model svd`,
значения по умолчанию — 32 компоненты и 7 итераций. Для Random уберите SVD-флаги.
Число компонент должно быть не больше `min(число пользователей, размер каталога)`;
каталог SVD требует минимум два товара. Автоматического уменьшения нет.

У `evaluate-ranking` обязательны три пути и `--split validation|test`; K по
умолчанию 10. Для продолжения повторите команды с накопленным снимком следующего
шага и новыми каталогами результатов. Каждая модель обучается на всём train
своего снимка. Для сравнения моделей используйте одинаковые истории, каталог и K.

Обучение сохраняет `model.joblib` и `manifest.json`; оценка — `metrics.json` и
`per_user.csv`. Каждый результат требует нового каталога вне снимка историй;
оценку можно разместить внутри каталога модели. [Контракты и метрики](pipeline.md#ранжирование-и-оценка).

## Сравнение моделей

`benchmark-ranking` выполняет свежее обучение и test-оценку для каждой пары
«конфигурация × готовый датасет». Генерация историй и validation в него не входят;
это не k-fold cross-validation. Передавайте сами папки с `train.csv`, `test.csv`
и `catalog.csv`. Внутренние пути pipeline автоматически не разыскиваются.

```bash
uv run buy_today benchmark-ranking --output models/ranking-benchmark \
  --dataset data/olist-stream/histories/example/step_000 \
  --model random --model "svd --n-components 16 --n-iter 7" --k 10
uv run buy_today benchmark-ranking-report models/ranking-benchmark \
  --output models/ranking-benchmark-report/comparison.html
```

Для сравнения на нескольких датасетах повторите `--dataset` с другими готовыми
снимками; удобно называть их по расстояниям генерации. Basename должны различаться.
Корень результата должен быть отдельным от входов: не внутри них и не их родителем.

- `--dataset` и `--model` обязательны и повторяемы. Порядок исполнения:
  модели по порядку аргументов, внутри каждой — датасеты.
- Каждая спецификация `--model` — одна строка. `random` принимает `--random-state`
  (42); `svd` — `--n-components` (32), `--n-iter` (7), `--random-state` (42).
  Здесь SVD-флаги имеют имена **без префикса `svd-`**.
- Каноническое имя включает только отличия от дефолтов: `svd`,
  `svd_n-components-16`, `random_random-state-73`. Порядок флагов и явные дефолты
  не меняют имя; повтор одной конфигурации в вызове — ошибка.
- K по умолчанию 10, от 1 до размера каталога; train/test должны содержать
  одинаковый полный набор пользователей. Ограничения SVD те же, что у `rank`.
- Каждый вызов создаёт новый timestamped run на конфигурацию. Можно запускать
  модели отдельными командами в один корень. `--no-save-model` отключает только
  сериализацию модели, сохраняя параметры, метрики и происхождение данных.
- При первой ошибке выполнение прекращается.

### Результаты benchmark

```text
<output>/
├── .completed/<invocation-id>.json
└── svd_n-components-16/runs/<timestamp>/
    ├── manifest.json
    ├── results.csv
    └── datasets/<dataset-name>/
        ├── model/              # model.joblib, manifest.json; если сохранение включено
        └── test/               # metrics.json, per_user.csv
```

Манифест содержит параметры, seed, версии NumPy/sklearn, пути и SHA-256 входов,
размеры данных и хеш пользовательской когорты. `results.csv` — строка на датасет.
`fit_seconds` и `evaluation_seconds` измеряют только fit и evaluate, без чтения
и записи файлов. Сохранённые модели совместимы с `inference` и `evaluate-ranking`.

Вызов рассчитывается в staging; после переноса runs атомарно публикуется общий
маркер `.completed/<invocation-id>.json`. Только он делает вызов видимым отчёту.
Обрабатываемая ошибка удаляет новые неопубликованные runs; после аварии они могут
остаться, но отчёт их игнорирует. При переносе сохраняйте весь корень с `.completed/`.

### Сравнительный отчёт

Для каждой конфигурации выбирается последний завершённый run по лексикографическому
имени, не по mtime. Последние runs должны иметь одинаковые наборы датасетов и K;
по каждому basename проверяются SHA-256 трёх CSV, размеры и пользовательская
когорта. Несовместимость — ошибка, без отката к старому run или пересечения наборов.

Рейтинг: **`model_score = max(NDCG@K по всем датасетам)`**, по убыванию;
при равенстве — каноническое имя. Самодостаточный HTML содержит тепловые карты
Recall/NDCG со шкалами 0–1, сравнения датасетов, точные значения, рейтинг,
длительности, параметры и выбранные runs. Рядом создаётся CSV полной матрицы
с происхождением результатов. Повторная сборка заменяет HTML и CSV; исходные
датасеты и модели не загружаются, обучение не запускается.

```python
from buy_today.ranking import run_ranking_benchmark, report_ranking_benchmark

runs = run_ranking_benchmark(
    ["data/olist-stream/histories/example/step_000"], "models/ranking-benchmark",
    models=["random", "svd --n-components 16"], k=10, save_model=False,
)
paths = report_ranking_benchmark(
    "models/ranking-benchmark", "models/ranking-benchmark-report/comparison.html",
)
```

## Проверка реализации

```bash
uv run --frozen pytest
```

Тесты используют небольшие искусственные CSV и временные каталоги; локальный
Olist не требуется. [Контейнерные тесты и smoke-сценарий](docker.md#проверки-контейнера).
