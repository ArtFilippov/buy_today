"""Small portable report contracts for summary integration fixtures."""

import hashlib
import json
from pathlib import Path
from typing import Any

type Metadata = dict[str, Any]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Metadata) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return digest(path.read_bytes())


def prepared_stream(source: Path) -> tuple[list[Metadata], Metadata]:
    source.mkdir()
    batches: list[Metadata] = []
    for index in range(2):
        path = source / f"batch_{index:03d}.csv"
        path.write_text("product_id\n" + f"p{index}\n" * 12, encoding="utf-8")
        batches.append(
            {"id": path.stem, "path": str(path), "rows": 12, "sha256": digest(path.read_bytes())}
        )
    manifest = source / "manifest.json"
    checksum = write_json(manifest, {"format_version": 1, "batches": batches})
    return batches, {"path": str(manifest), "sha256": checksum}


def _identity(histories: Path, index: int, name: str, rows: int) -> Metadata:
    return {
        "path": str(histories / f"{name}.csv"),
        "rows": rows,
        "sha256": digest(f"{index}:{name}:{rows}".encode()),
    }


def _training(output: Path, index: int, model: str) -> Metadata:
    users = 2 + index
    model_class = "RandomRanker" if model == "random" else "SVDRanker"
    training: Metadata = {
        "format_version": 1,
        "inputs": {
            "train": _identity(output / "histories", index, "train", users * 3),
            "catalog": _identity(output / "histories", index, "catalog", 4 + index),
        },
        "n_users": users,
        "user_ids_sha256": digest(json.dumps([f"u{i}" for i in range(users)]).encode()),
        "versions": {"numpy": "2.1.0", "scikit_learn": "1.6.0"},
        "model": {
            "file": "model.joblib",
            "sha256": digest(f"model:{index}".encode()),
            "class": f"buy_today.ranking.{model}.{model_class}",
            "parameters": {"random_state": 42},
        },
    }
    if model == "svd":
        training["model"]["parameters"].update(n_components=2, n_iter=7)
    return training


def _eda(index: int, reference_input: Metadata, rows: int) -> Metadata:
    return {
        "format_version": 1,
        "kind": "eda",
        "input": reference_input,
        "rows": rows,
        "n_columns": 35,
        "n_orders": 6 * (index + 1),
        "n_customers": 3 + index,
        "n_products": 4 + index,
        "period_start": "2018-01-01 00:00:00",
        "period_end": f"2018-01-0{index + 2} 00:00:00",
        "checks": [
            {"Проверка": name, "Результат": "OK", "Фактически": actual, "Ожидается": expected}
            for name, actual, expected in (
                (
                    "Схема и порядок колонок",
                    "35 колонок, порядок совпадает",
                    "35 колонок в порядке schema.COLUMNS",
                ),
                ("Типы и даты", "35 типов совпадают", "типы schema.DTYPES"),
                ("Непустой датасет", str(rows), "число строк > 0"),
                ("Пропуски", "0", "число пропущенных значений = 0"),
                (
                    "Повторы ключа (order_id, order_item_id)",
                    "0",
                    "число повторений сверх первого = 0",
                ),
                ("Хронологический порядок покупок", "0", "число переходов назад во времени = 0"),
                ("Конечная положительная цена", "0", "число цен, нарушающих 0 < price < inf, = 0"),
            )
        ],
    }


def _clustering(
    output: Path, index: int, batch: Metadata, reference: Metadata, reference_input: Metadata
) -> Metadata:
    return {
        "format_version": 1,
        "kind": "clustering",
        "inputs": {
            "dataset": reference_input,
            "new_batch": {key: batch[key] for key in ("path", "sha256")},
            "labels": {
                "path": str(output / "clustering/labels.csv"),
                "sha256": digest(f"labels:{index}".encode()),
            },
        },
        "silhouette": 0.45 if index else None,
        "silhouette_reason": None if index else 'Один кластер <script>alert("x")</script>',
        "sample_rows": reference["rows"],
        "new_count": 12,
        "previous_count": index * 12,
        "max_evaluation_rows": 1000,
        "random_state": 42,
        "model": {"class": "TemporalClusterer", "parameters": {"n_clusters": 1 + index}},
        "distance": {"class": "TimestampDistance", "parameters": {}},
    }


def _evaluations(output: Path, index: int, training: Metadata) -> dict[str, tuple[str, Metadata]]:
    reports: dict[str, tuple[str, Metadata]] = {}
    users = 2 + index
    for split, events_per_user, recall, ndcg in (
        ("validation", 2, 0.5, 0.6),
        ("test", 1, 0.25, 0.3),
    ):
        reports[split] = (
            f"ranking/{split}/metrics.json",
            {
                "format_version": 1,
                "split": split,
                "k": 2,
                "n_users": users,
                "n_catalog": 4 + index,
                "n_events": users * events_per_user,
                "metrics": {"recall_at_k": recall + index * 0.1, "ndcg_at_k": ndcg + index * 0.1},
                "inputs": {
                    split: _identity(output / "histories", index, split, users * events_per_user),
                    "catalog": training["inputs"]["catalog"],
                },
                "model": {
                    "path": str(output / "ranking/model.joblib"),
                    "sha256": training["model"]["sha256"],
                },
            },
        )
    return reports


def step_reports(
    run: Path, prefix: str, index: int, model: str, batch: Metadata, reference: Metadata
) -> dict[str, tuple[str, Metadata]]:
    output = run / prefix
    reference_input = {"path": str(run / "reference.csv"), "sha256": reference["sha256"]}
    training = _training(output, index, model)
    return {
        "eda_metrics": ("eda/metrics.json", _eda(index, reference_input, reference["rows"])),
        "clustering_metrics": (
            "clustering/metrics.json",
            _clustering(output, index, batch, reference, reference_input),
        ),
        "ranking_manifest": ("ranking/manifest.json", training),
        **_evaluations(output, index, training),
    }


def drift_report(
    previous: Metadata, reference_path: Path, batch: Metadata, parameters: Metadata
) -> Metadata:
    return {
        "format_version": 1,
        "kind": "deda",
        "drift_detected": True,
        "inputs": {
            "reference": {**previous, "path": str(reference_path)},
            "batch": {key: batch[key] for key in ("path", "sha256", "rows")},
        },
        "thresholds": parameters["thresholds"],
        "metrics": [
            {
                "feature": feature,
                "measure": measure,
                "value": value,
                "threshold": 0.1,
                "drift": value >= 0.1,
            }
            for feature, measure, value in (
                ("price", "KS D", 0.2),
                ("product_category_name", "TVD", 0.05),
                ("customer_state", "TVD", 0.15),
            )
        ],
    }


def save_reports(
    run: Path, prefix: str, reports: dict[str, tuple[str, Metadata]], artifacts: dict[str, str]
) -> dict[str, str]:
    files: dict[str, str] = {}
    for name, (relative, report) in reports.items():
        artifacts[name] = f"{prefix}/{relative}"
        files[artifacts[name]] = write_json(run / artifacts[name], report)
    return files
