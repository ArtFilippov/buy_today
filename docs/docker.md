# Docker

[README](../README.md) · [Локальный CLI](local-cli.md) · [Конвейер](pipeline.md)

## Образ и сборка

Основной сценарий — Windows PowerShell и Docker Desktop с Linux engine.
Python и uv на хосте не нужны. Рабочая папка результатов может находиться отдельно.
Сборка выполняется одной командой из корня репозитория; нужен Интернет для загрузки Olist:

```powershell
docker build -t buy_today:local .
```

[Dockerfile](../Dockerfile) использует Python 3.13 slim, uv 0.12.6 и
`uv sync --frozen`. Пакет устанавливается не в editable-режиме. Зависимости
notebook входят в runtime; служебные файлы ядра и кеши находятся в `/tmp`.
Точка входа — `python -m buy_today.container_cli`, рабочий каталог — `/workspace`.

Стадия `olist-data` скачивает ZIP [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
через публичный Kaggle API. Проверяются ZIP/CRC и точный набор девяти непустых
исходных CSV; файлы помещаются в `/opt/buy_today/olist`. Локальный `dataset/` для
сборки не нужен. Версия датасета и ожидаемые контрольные суммы не закреплены;
исследовательский `olist_prepared_dataset.csv` в образ не входит.

После сборки все команды не требуют сети. Образ содержит код, окружение и
исходные CSV; подключённая папка — состояние эксперимента и его результаты.

## Рабочая папка и команды

```powershell
$WorkDir = Join-Path $env:USERPROFILE 'buy_today-workspace'
New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null

docker run --rm --mount "type=bind,source=$WorkDir,target=/workspace" `
    buy_today:local init --verbose
if ($LASTEXITCODE -ne 0) { throw 'init failed; see workspace logs' }

docker run --rm --mount "type=bind,source=$WorkDir,target=/workspace" `
    buy_today:local update --verbose
if ($LASTEXITCODE -ne 0) { throw 'update failed; see workspace logs' }

docker run --rm --mount "type=bind,source=$WorkDir,target=/workspace" `
    buy_today:local inference --model-dir run/steps/step_000/ranking `
    --user-id user_000000_000000 --k 10 --verbose
if ($LASTEXITCODE -ne 0) { throw 'inference failed; see workspace logs' }
```

| Команда | Действие |
| --- | --- |
| `init` | Проверка настроек и встроенных CSV → сброс эксперимента → выгрузка CSV → подготовка потока → полный первый батч → сводка |
| `update` | Загрузка состояния → полный следующий батч → обновлённая сводка |
| `inference` | Загрузка явно выбранной модели → рекомендации известному модельному пользователю → CSV |

Для потока из N батчей нужны один `init` и N−1 вызовов `update`. Число батчей
определяется подготовкой; оно не зафиксировано в реализации. На исчерпанном
потоке `update` успешно пересобирает сводку без обучения. `update` требует
существующий прогон; `inference` может читать отдельный сохранённый каталог модели.

```text
<WorkDir>/
├── dataset/          # девять CSV из образа
├── data/             # working_dataset.csv, manifest.json, state.json, batches/
├── run/              # config/state, reference.csv, steps/, summary/
├── recommendations/  # CSV результатов inference
└── logs/             # текстовый файл на каждый вызов
```

**Повторный `init` сбрасывает эксперимент:** удаляет `data/`, `run/`,
`recommendations/`, перезаписывает девять исходных CSV. Старые логи и прочие файлы
вне перечисленных результатов сохраняются. `update` и `inference` не выгружают
CSV заново. Сброс и полный шаг не являются транзакциями: после ошибки могут
остаться частичные результаты; автоматического восстановления нет.

Настройки содержат абсолютные пути **внутри контейнера**. Для продолжения
сохраняйте подключение к `/workspace`. HTML открывается непосредственно на хосте,
начиная с `<WorkDir>\run\summary\summary.html`.

## Параметры

Справка: `docker run --rm buy_today:local init --help` и аналогично для других команд.
Параметры передаются после имени команды.

- `init`: `--batch-size` (5000), `--min-category-count` (1000) и
  [параметры полного прогона](local-cli.md#параметры-прогона): модель, seed,
  температура, число пользователей, split, K, SVD, кластеризация и пороги дрейфа.
  По умолчанию — SVD 32/7, 2000 пользователей первого батча и 250 следующих,
  split 70/15/15. Настройки фиксируются при `init`.
- `update`: только `--verbose`; настройки загружаются из прогона.
- `inference`: обязательны `--model-dir` и `--user-id`; K по умолчанию 10.
  Относительный путь модели считается от `/workspace`. Последняя модель
  автоматически не выбирается. `user_id` берётся из синтетических историй;
  неизвестный пользователь — ошибка.
- `inference --output example.csv` задаёт имя относительно `recommendations/`.
  Абсолютный путь тоже должен находиться внутри `/workspace/recommendations`.
  Без `--output` создаётся CSV с уникальным именем.

Контейнерные `init`/`update` выполняют полный цикл. Одноимённые локальные команды
`uv run buy_today init/update` работают только с эталоном и EDA/DEDA.

## Логи и ошибки

`--verbose` выводит начало стадии до её исполнения, завершение, длительность и
номер батча. Видны подготовка, EDA/DEDA, кластеризация, notebook, генерация историй,
обучение, validation/test и сводка. Вывод ячеек и графики остаются в notebook/HTML.
Без verbose в консоли остаются итог, пути и ошибки.

В любом режиме создаётся UTF-8 лог `logs/<время-UTC>_<id>_<команда>.log` со стадиями,
длительностями и traceback. Неверные аргументы тоже логируются; `--help` не создаёт
файлов. Коды: `0` — успех, `2` — ошибка аргументов, `1` — ошибка исполнения.

`batch_committed` означает фиксацию батча, `command_succeeded` — успех всей команды.
Сводка строится **после** фиксации: её ошибка не отменяет батч. Следующий `update`
возьмёт следующий батч и заново соберёт сводку завершённых шагов; на исчерпанном
потоке он только пересоберёт сводку. Подробнее: [состояние прогона](pipeline.md#состояние-прогона).

## Проверки контейнера

Стадия `tests` устанавливает dev-зависимости и тесты, не скачивая Olist:

```powershell
docker build --target tests -t buy_today-tests:local .
if ($LASTEXITCODE -ne 0) { throw 'Tests image build failed' }
docker run --rm --network none buy_today-tests:local
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }

$CheckDir = Join-Path $env:TEMP ('buy_today-smoke-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\docker-smoke.ps1 -Root $CheckDir
if ($LASTEXITCODE -ne 0) { throw 'Docker smoke check failed' }
```

[Скрипт](../scripts/docker-smoke.ps1) требует новый корневой каталог с существующим
родителем. Он подставляет искусственные исходные таблицы и выполняет offline-цикл
`init → update → inference → исчерпанный update → повторный init` с настоящими
notebook, SVD и отчётами. Проверяются сохранность моделей/историй, сводка, сброс и
сохранение логов. JSON появятся в `<CheckDir>\checks\`, результаты — в
`<CheckDir>\workspace with spaces\`. Выделяйте больше пяти минут.
`ExecutionPolicy Bypass` действует только на запускающий скрипт процесс PowerShell.
