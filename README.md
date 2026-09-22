# Рекомендательная система списка покупок

Учебный Python-конвейер: подготовка потока данных Olist, контроль качества,
обучение моделей по батчам, рекомендации и история метрик. Этапы реализованы
на pandas, NumPy и scikit-learn, без специализированных MLOps-платформ.

Поток состоит из реальных позиций заказов [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
Для оценки рекомендаций из них генерируются **синтетические пользовательские истории**.
Временная кластеризация обслуживает модельный этап; ранжировщик — Random или SVD,
один на прогон. Метрики описывают модельные истории, а не полезность для реальных покупателей.

Основной сценарий сдачи — **MVP + локальный Docker**. Он позволяет обрабатывать
батчи отдельными запусками и открывать результаты из примонтированной папки.
Локальный Docker предусмотрен как альтернатива GitHub Actions в задании 2.

## Docker: сборка и запуск

Нужен Docker Desktop с запущенным **Linux engine**. Команды ниже — для
Windows PowerShell, из корня репозитория. Python и uv на Windows не требуются.
Для сборки нужен Интернет: исходные CSV Olist скачиваются в образ автоматически.

```powershell
docker version
docker build -t prak:local .
if ($LASTEXITCODE -ne 0) { throw 'Image build failed' }
docker run --rm --network none prak:local --help
```

Выберите папку результатов и подключайте её к `/workspace` при каждом вызове:

```powershell
$WorkDir = Join-Path $env:USERPROFILE 'prak-workspace'
New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null

docker run --rm --network none --mount "type=bind,source=$WorkDir,target=/workspace" `
    prak:local init --verbose
if ($LASTEXITCODE -ne 0) { throw 'init failed; see workspace logs' }
```

`init` выгружает исходные CSV, подготавливает весь поток, обрабатывает **первый
батч** и строит сводку. **Повторный `init` сбрасывает эксперимент:** удаляет
`data/`, `run/`, `recommendations/` в рабочей папке и перезаписывает исходные CSV;
старые `logs/` сохраняются. Для продолжения используйте `update`:

```powershell
docker run --rm --network none --mount "type=bind,source=$WorkDir,target=/workspace" `
    prak:local update --verbose
if ($LASTEXITCODE -ne 0) { throw 'update failed; see workspace logs' }
```

Каждый `update` обрабатывает **один следующий батч** и обновляет сводку.
Повторяйте его до сообщения об исчерпании потока; число батчей зависит от данных
и параметров подготовки. Настройки модели задаются при `init` и затем наследуются.

Рекомендации из явно выбранной модели первого шага:

```powershell
docker run --rm --network none --mount "type=bind,source=$WorkDir,target=/workspace" `
    prak:local inference --model-dir run/steps/step_000/ranking `
    --user-id user_000000_000000 --k 10 --verbose
if ($LASTEXITCODE -ne 0) { throw 'inference failed; see workspace logs' }
```

`user_id` — известный модели синтетический пользователь, не `customer_unique_id`
из Olist. Для другого шага укажите его каталог `ranking/` явно.
Все три команды после сборки работают без сети.
Параметры, логи и поведение при ошибках: [справка Docker](docs/docker.md).

## Минимальный локальный запуск

Нужны Python **3.13+**, [uv](https://docs.astral.sh/uv/) и исходные CSV Olist,
распакованные в `dataset/`. `uv sync` устанавливает окружение, но не скачивает
данные. [Нужные файлы и правила подготовки](docs/data.md#исходные-файлы).
Команды ниже — для Bash из корня репозитория; каталог нового прогона должен отсутствовать.

```bash
uv sync --frozen
uv run prak prepare --raw-dir dataset --output-dir data/olist-stream

# Новый прогон: первый батч, ранжировщик SVD
uv run prak run --data-dir data/olist-stream \
  --run-dir models/olist-stream/runs/example --model svd

# Продолжение: все оставшиеся батчи (без --all — только один)
uv run prak run --run-dir models/olist-stream/runs/example --all
uv run prak summary --run-dir models/olist-stream/runs/example

uv run prak inference \
  --model-dir models/olist-stream/runs/example/steps/step_000/ranking \
  --user-id user_000000_000000 --k 10 \
  --output models/olist-stream/runs/example/recommendations.csv
```

Для повторного эксперимента выберите новый `--run-dir`.
Локальная `run` выполняет полный цикл батча, а `summary` вызывается отдельно.
Локальные `init` / `update` управляют только эталоном и EDA/DEDA; контейнерные
одноимённые команды выполняют полный цикл. [Справка локального CLI](docs/local-cli.md).

## Где появятся результаты

Пути ниже создаются после выполнения команд. Для Docker база — `$WorkDir`,
для локального прогона — указанный `--run-dir`.

| Результат | Docker | Локальный прогон |
| --- | --- | --- |
| Главная сводка | `run/summary/summary.html` | `summary/summary.html` |
| Сводные данные | `run/summary/summary.{csv,json}` | `summary/summary.{csv,json}` |
| Отчёты EDA, DEDA, кластеризации | `run/steps/step_NNN/` | `steps/step_NNN/` |
| Модель и метрики validation/test | `run/steps/step_NNN/ranking/` | `steps/step_NNN/ranking/` |
| Рекомендации | `recommendations/*.csv` | Путь из `inference --output` |
| Текстовые логи вызовов | `logs/*.log` | Вывод CLI; длительности в манифестах шагов |

Начните просмотр с `summary.html`: там динамика качества и ссылки на отчёты шагов.
EDA/DEDA и кластеризация сохраняют выполненный `report.ipynb`, самодостаточный
`report.html` и `metrics.json`. Рекомендации — CSV `user_id,rank,product_id`.
Сводка Docker строится автоматически после `init` / `update`.

## Остальные команды

Синтаксис: `uv run prak <команда> --help`.

| Команды | Назначение и подробности |
| --- | --- |
| `eda`, `deda` | [Отдельные отчёты качества и дрейфа](docs/local-cli.md#eda-и-эталон) |
| `init`, `update` | [Создание и пополнение эталона](docs/local-cli.md#eda-и-эталон) |
| `cluster`, `evaluate` | [Временная кластеризация и независимая оценка](docs/local-cli.md#кластеризация) |
| `generate` | [Накопление модельных историй](docs/local-cli.md#генерация-историй) |
| `rank`, `evaluate-ranking` | [Обучение и оценка ранжировщика](docs/local-cli.md#ранжирование) |
| `benchmark-ranking`, `benchmark-ranking-report` | [Сравнение моделей на готовых датасетах](docs/local-cli.md#сравнение-моделей) |

## Задания и состояние реализации

Исходные требования: [задание 1 — MVP](docs/assignments/320_MLOps_task_1.pdf),
[задание 2 — развёртывание](docs/assignments/320_MLOps_task_2.pdf).
Краткий чеклист отражает реализованный объём; это не оценка по всем балльным критериям PDF.

- [x] Сбор и очистка Olist, временные батчи, файловое хранилище и метаданные.
- [x] Автоматические EDA/DEDA, проверки качества, метрики и отчёты о дрейфе.
- [x] Обучение Random/SVD, train/validation/test, версии моделей и метрики качества.
- [x] CLI полного цикла, рекомендации известному пользователю и сводка по шагам.
- [x] Docker: установка окружения, загрузка Olist, обучение, модели, отчёты и логи в mount.
- [ ] `requirements.txt` из `uv.lock` — требуется форматом сдачи обоих заданий.

**Отличия от задания 1:** вместо перечисленных LR/kNN/дерева используются
рекомендательные Random/SVD; обновление — полное переобучение на накопленном train.
Автовыбор лучшей модели/гиперпараметров и inference внешнего файла с колонкой
`predict` не реализованы: каталог модели и известный `user_id` задаются явно.

**Исключено из объёма:** GitHub Actions, CI/CD workflow, CRON, публикация
образов/артефактов через GitHub; MiniBatchKMeans.

## Документация

- [Docker](docs/docker.md) — образ, рабочая папка, параметры, логи и проверки.
- [Локальный CLI](docs/local-cli.md) — команды, настройки и сравнение моделей.
- [Данные](docs/data.md) — источник, подготовка, схема и все 35 признаков.
- [Конвейер](docs/pipeline.md) — этапы, контракты, метрики и состояние прогона.
