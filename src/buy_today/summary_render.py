"""Tabular and HTML presentation of verified summary metadata."""

from collections.abc import Iterable, Sequence
import csv
from html import escape
from io import StringIO
import json

from buy_today.pipeline_state import Metadata

type Cell = str | int | float | bool | None
type Row = dict[str, Cell]

TIMINGS = (
    "reference",
    "clustering_train",
    "clustering_evaluation",
    "generation",
    "ranking_train",
    "validation",
    "test",
    "total",
)
_DRIFT_FEATURES = {
    "price_ks": "price",
    "category_tvd": "product_category_name",
    "state_tvd": "customer_state",
}
_OK = "OK"
_COLUMNS = (
    "step_index",
    "batch_id",
    "batch_rows",
    "source_sha256",
    "reference_sha256",
    "reference_rows",
    "n_orders",
    "n_customers",
    "n_products",
    "quality_checks",
    "quality_checks_failed",
    "n_users",
    "n_catalog",
    "drift_detected",
    "price_ks",
    "category_tvd",
    "state_tvd",
    "price_drift",
    "category_drift",
    "state_drift",
    "silhouette",
    "silhouette_reason",
    "k",
    "validation_n_events",
    "validation_recall",
    "validation_ndcg",
    "test_n_events",
    "test_recall",
    "test_ndcg",
    *[f"{name}_seconds" for name in TIMINGS],
)


def _drift_cells(drift: Metadata | None) -> Row:
    features = {row["feature"]: row for row in drift["metrics"]} if drift else {}
    cells: Row = {}
    for name, feature in _DRIFT_FEATURES.items():
        metric = features[feature] if drift else None
        cells[name] = metric["value"] if metric else None
        cells[name.split("_", maxsplit=1)[0] + "_drift"] = metric["drift"] if metric else None
    return cells


def _ranking_cells(ranking: Metadata) -> Row:
    cells: Row = {}
    for split in ("validation", "test"):
        report = ranking[split]
        cells[f"{split}_n_events"] = report["n_events"]
        cells[f"{split}_recall"] = report["metrics"]["recall_at_k"]
        cells[f"{split}_ndcg"] = report["metrics"]["ndcg_at_k"]
    return cells


def flat_row(step: Metadata) -> Row:
    quality, clustering = step["data_quality"], step["clustering"]
    validation, drift = step["ranking"]["validation"], step["drift"]
    return {
        "step_index": step["step_index"],
        "batch_id": step["batch"]["id"],
        "batch_rows": step["batch"]["rows"],
        "source_sha256": step["batch"]["sha256"],
        "reference_sha256": step["reference"]["sha256"],
        "reference_rows": quality["rows"],
        **{key: quality[key] for key in ("n_orders", "n_customers", "n_products")},
        "quality_checks": len(quality["checks"]),
        "quality_checks_failed": sum(check["Результат"] != _OK for check in quality["checks"]),
        "n_users": validation["n_users"],
        "n_catalog": validation["n_catalog"],
        "drift_detected": drift["drift_detected"] if drift else None,
        "silhouette": clustering["silhouette"],
        "silhouette_reason": clustering["silhouette_reason"],
        "k": validation["k"],
        **_drift_cells(drift),
        **_ranking_cells(step["ranking"]),
        **{f"{name}_seconds": step["durations_seconds"].get(name) for name in TIMINGS},
    }


def csv_text(rows: list[Row]) -> str:
    text = StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=_COLUMNS)
    _ = writer.writeheader()
    writer.writerows(rows)
    return text.getvalue()


def _text(value: Cell) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    return format(value, ".6g") if isinstance(value, float) else str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Cell]]) -> str:
    heading = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_text(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table"><table><thead><tr>{heading}</tr></thead>'
        + f"<tbody>{body}</tbody></table></div>"
    )


def _quality_table(rows: list[Row]) -> str:
    headers = [
        "Шаг",
        "Батч",
        "Строк эталона",
        "Заказов",
        "Покупателей",
        "Товаров",
        "Проверок",
        "Нарушений",
        "Пользователей",
        "Каталог",
    ]
    keys = (
        "step_index",
        "batch_id",
        "reference_rows",
        "n_orders",
        "n_customers",
        "n_products",
        "quality_checks",
        "quality_checks_failed",
        "n_users",
        "n_catalog",
    )
    return _table(headers, [[row[key] for key in keys] for row in rows])


def _drift_table(rows: list[Row]) -> str:
    headers = [
        "Шаг",
        "Дрейф",
        "Цена KS D",
        "Дрейф цены",
        "Категории TVD",
        "Дрейф категорий",
        "Штаты TVD",
        "Дрейф штатов",
        "Силуэт / причина",
    ]
    keys = (
        "step_index",
        "drift_detected",
        "price_ks",
        "price_drift",
        "category_tvd",
        "category_drift",
        "state_tvd",
        "state_drift",
    )
    return _table(
        headers,
        [
            [
                *[row[key] for key in keys],
                row["silhouette"] if row["silhouette"] is not None else row["silhouette_reason"],
            ]
            for row in rows
        ],
    )


def _ranking_table(rows: list[Row]) -> str:
    headers = [
        "Шаг",
        "K",
        "Validation: событий",
        "Validation Recall@K",
        "Validation NDCG@K",
        "Test: событий",
        "Test Recall@K",
        "Test NDCG@K",
    ]
    keys = (
        "step_index",
        "k",
        "validation_n_events",
        "validation_recall",
        "validation_ndcg",
        "test_n_events",
        "test_recall",
        "test_ndcg",
    )
    return _table(headers, [[row[key] for key in keys] for row in rows])


def _timing_table(rows: list[Row]) -> str:
    headers = [
        "Шаг",
        "Эталон",
        "Кластеры: обучение",
        "Кластеры: оценка",
        "Истории",
        "Ранкер: обучение",
        "Validation",
        "Test",
        "Всего",
    ]
    return _table(
        headers,
        [[row["step_index"], *[row[f"{name}_seconds"] for name in TIMINGS]] for row in rows],
    )


def _details(step: Metadata) -> str:
    details = {
        "ranking": step["ranking"]["training"]["model"],
        "clustering": step["clustering"]["model"],
        "distance": step["clustering"]["distance"],
        "checks": step["data_quality"]["checks"],
    }
    return (
        f"<details><summary>Шаг {step['step_index']}</summary><pre>"
        + escape(json.dumps(details, ensure_ascii=False, indent=2))
        + "</pre></details>"
    )


def _links(step: Metadata) -> str:
    links = " · ".join(
        f'<a href="{escape(step["links"][key], quote=True)}">{label}</a>'
        for key, label in (
            ("eda", "EDA"),
            ("deda", "DEDA"),
            ("clustering", "Кластеризация"),
            ("validation", "Validation (JSON)"),
            ("test", "Test (JSON)"),
        )
        if key in step["links"]
    )
    return f"<li>Шаг {step['step_index']}: {links}</li>"


def html_text(summary: Metadata, rows: list[Row]) -> str:
    parts = [
        '<!doctype html><html lang="ru"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Сводка запуска</title><style>",
        "body{font:16px system-ui,sans-serif;max-width:1400px;margin:2rem auto;"
        + "padding:0 1rem;color:#202530}",
        "table{border-collapse:collapse;width:100%;font-size:.9rem}"
        + "th,td{padding:.55rem;border:1px solid #d5dbe3;text-align:left}",
        "th{background:#edf1f6}tbody tr:nth-child(even){background:#f8fafc}.table{overflow-x:auto}",
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;"
        + "padding:1rem}h2{margin-top:2rem}",
        "a{color:#155da8}li{margin:.5rem 0}</style><body><h1>Сводка запуска</h1>",
        f"<p>Модель: <strong>{escape(summary['model'])}</strong>. "
        + f"Завершено шагов: {len(rows)}.</p>",
        "<h2>Параметры запуска</h2><pre>",
        escape(
            json.dumps(
                {"parameters": summary["parameters"], "stream": summary["stream"]},
                ensure_ascii=False,
                indent=2,
            )
        ),
        "</pre>",
        "<h2>Динамика данных и качество</h2>",
        _quality_table(rows),
        "<h2>Дрейф и силуэт</h2><p>Дрейф сравнивает батч с эталоном до добавления; "
        + "на первом шаге не оценивается.</p>",
        _drift_table(rows),
        "<h2>Ранжирование: validation и test</h2><p>Обе оценки используют одну обученную модель "
        + "шага, полный каталог и одинаковое число пользователей.</p>",
        _ranking_table(rows),
        "<h2>Время выполнения, секунды</h2>",
        _timing_table(rows),
        "<h2>Гиперпараметры моделей и проверки</h2>",
        *[_details(step) for step in summary["steps"]],
        "<h2>Отчёты шагов</h2><ul>",
        *[_links(step) for step in summary["steps"]],
        "</ul></body></html>\n",
    ]
    return "\n".join(parts)
