"""Self-contained Russian HTML for validated benchmark comparisons."""

from collections.abc import Sequence
from html import escape
import json
from pathlib import Path

from buy_today.ranking.publication import Metadata, SelectedRuns
from buy_today.ranking.report_plots import dataset_plot, heatmap, image
from buy_today.ranking.report_validation import METRICS, TIMINGS

_HEADER = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Сравнение моделей ранжирования</title><style>
body{font:16px system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#202530}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:.55rem;border:1px solid #d5dbe3;text-align:left}
th{background:#edf1f6}tbody tr:nth-child(even){background:#f8fafc}.table{overflow-x:auto}
td.best{background:#e3f2d4}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:1rem}
figure{margin:1rem 0;overflow-x:auto}img{display:block;width:100%;min-width:700px}
h2{margin-top:2rem}</style></head><body><h1>Сравнение моделей ранжирования</h1>"""


def _cells(row: Sequence[object], index: int, best: set[tuple[int, int]]) -> str:
    cells: list[str] = []
    for column, value in enumerate(row):
        text = escape(str(value))
        cells.append(
            f'<td class="best"><strong>{text}</strong></td>'
            if (index, column) in best
            else f"<td>{text}</td>"
        )
    return "<tr>" + "".join(cells) + "</tr>"


def _table(
    headers: list[str],
    rows: Sequence[Sequence[object]],
    *,
    best: set[tuple[int, int]] | None = None,
) -> str:
    heading = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(_cells(row, index, best or set()) for index, row in enumerate(rows))
    return (
        '<div class="table"><table><thead><tr>'
        + heading
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def _dataset_section(name: str, rows: list[Metadata]) -> str:
    best = {
        (i, j + 1)
        for j, metric in enumerate(METRICS)
        for i, row in enumerate(rows)
        if row[metric] == max(item[metric] for item in rows)
    }
    return "\n".join(
        [
            f"<h3>{escape(name)}</h3>",
            image(dataset_plot(rows), f"Сравнение Recall@K и NDCG@K: {name}; шкалы от 0 до 1"),
            _table(
                [
                    "Модель",
                    *METRICS.values(),
                    "Пользователей",
                    "Товаров",
                    "Событий train",
                    "Событий test",
                ],
                [
                    [
                        row[field]
                        for field in (
                            "model",
                            *METRICS,
                            "n_users",
                            "n_catalog",
                            "n_train_events",
                            "n_test_events",
                        )
                    ]
                    for row in rows
                ],
                best=best,
            ),
        ]
    )


def _provenance(selected: SelectedRuns) -> str:
    parts = [
        """<h2>Происхождение данных и метаданные экспериментов</h2><p>Совпадение train,
test и catalog между моделями проверено по SHA256 и числу строк для каждого basename.
Пути могут различаться. Отчёт построен по сохранённым метаданным, без чтения исходных
данных и файлов моделей.</p>"""
    ]
    for run_path, manifest in selected:
        identity = escape(manifest["model"]["name"]) + " · " + escape(manifest["run_id"])
        metadata = json.dumps(
            {"run_path": str(run_path), **manifest},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        parts.append(
            f"<details><summary>{identity}</summary><pre>{escape(metadata)}</pre></details>"
        )
    return "\n".join(parts)


def comparison_html(root: Path, selected: SelectedRuns, rows: list[Metadata]) -> str:
    representatives = {row["model"]: row for row in rows}
    models = list(representatives)
    parts = [
        _HEADER,
        f"<p>Корень результатов: <code>{escape(str(root))}</code>. K = {rows[0]['k']}.</p>",
        """<p>Обучение на train, оценка на test того же датасета. Recall@K и бинарный NDCG@K —
средние по пользователям; больше — лучше. Каждая ячейка — один запуск, без оценки
статистической неопределённости.</p>
<h2>Общий рейтинг: максимум NDCG@K по датасетам</h2>
<p><code>model_score = max(ndcg_at_k по всем датасетам)</code>. При равных score порядок
определяется каноническим именем модели. Место — позиция в этом порядке.
Максимум отражает лучший результат модели на одном из датасетов.</p>""",
        _table(
            ["Место", "Конфигурация модели", "Score: максимум NDCG@K"],
            [
                [representatives[name][field] for field in ("rank", "model", "score")]
                for name in models
            ],
        ),
        """<h2>Тепловые карты</h2><p>Обе цветовые шкалы фиксированы: 0–1. Звёздочка (*)
отмечает все лучшие значения по датасету и метрике, включая равенства. Подписи графиков
округлены до трёх знаков; точные сохранённые значения приведены в таблицах и CSV.</p>""",
    ]
    for metric, title in METRICS.items():
        parts.append(image(heatmap(rows, metric), f"Тепловая карта {title}; шкала от 0 до 1"))
    parts.append("<h2>Сравнение по датасетам и точные значения</h2>")
    for name in sorted({row["dataset"] for row in rows}):
        parts.append(_dataset_section(name, [row for row in rows if row["dataset"] == name]))
    parts.extend(
        [
            """<h2>Время выполнения, секунды</h2><p>Обучение (fit) и оценка измерены отдельно
для каждой пары. Запуски могли выполняться в разных условиях; время не участвует в рейтинге.</p>""",
            _table(
                ["Модель", "Датасет", "Обучение (fit), с", "Оценка, с"],
                [[row[field] for field in ("model", "dataset", *TIMINGS)] for row in rows],
            ),
            """<h2>Параметры, seeds и выбранные runs</h2><p>Для каждой модели выбран последний
завершённый run по лексикографическому имени папки, независимо от mtime и created_at.</p>""",
            _table(
                [
                    "Модель",
                    "Алгоритм",
                    "Все параметры, включая seed",
                    "Run ID",
                    "Путь run",
                    "Создан",
                ],
                [
                    [
                        representatives[name][field]
                        for field in (
                            "model",
                            "algorithm",
                            "parameters",
                            "run_id",
                            "run_path",
                            "created_at",
                        )
                    ]
                    for name in models
                ],
            ),
            _provenance(selected),
            "</body></html>\n",
        ]
    )
    return "\n".join(parts)
