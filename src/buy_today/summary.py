"""Portable summaries of completed steps in a single-ranker pipeline run."""

import csv
from dataclasses import dataclass
from html import escape
from io import StringIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

from buy_today.pipeline import read_completed_steps


@dataclass(frozen=True)
class SummaryPaths:
    json_path: Path
    csv_path: Path
    html_path: Path


_TIMINGS = (
    "reference", "clustering_train", "clustering_evaluation", "generation",
    "ranking_train", "validation", "test", "total",
)
_DRIFT_FEATURES = {
    "price_ks": "price", "category_tvd": "product_category_name",
    "state_tvd": "customer_state",
}
_COLUMNS = (
    "step_index", "batch_id", "batch_rows", "source_sha256", "reference_sha256",
    "reference_rows", "n_orders", "n_customers", "n_products", "quality_checks",
    "quality_checks_failed", "n_users", "n_catalog", "drift_detected",
    "price_ks", "category_tvd", "state_tvd", "price_drift", "category_drift",
    "state_drift", "silhouette", "silhouette_reason", "k", "validation_n_events",
    "validation_recall", "validation_ndcg", "test_n_events", "test_recall",
    "test_ndcg", *[f"{name}_seconds" for name in _TIMINGS],
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_path(run_dir, relative):
    path = Path(relative)
    resolved = (run_dir / path).resolve()
    if (path.is_absolute() or ".." in path.parts or run_dir not in resolved.parents
            or resolved == run_dir / "summary" or run_dir / "summary" in resolved.parents):
        raise ValueError(f"Expected an input path within the run, outside summary: {relative}")
    return path


def _report(run_dir, step, name, *, kind=None):
    relative = step["artifacts"][name]
    path = _relative_path(run_dir, relative)
    if relative not in step["files"] or path.suffix != ".json":
        raise ValueError(f"Step {step['step_index']}: unhashed JSON artifact {name}")
    value = _read_json(run_dir / path)
    if value.get("format_version") != 1 or (kind and value.get("kind") != kind):
        raise ValueError(f"Step {step['step_index']}: unexpected report format for {name}")
    return value


def _check_ranking(training, evaluations, parameters):
    """Compare recorded identities, without opening models or history snapshots."""
    model = training["model"]
    selected = parameters["model"]
    expected_class = {
        "random": "buy_today.ranking.random.RandomRanker", "svd": "buy_today.ranking.svd.SVDRanker",
    }[selected]
    expected_parameters = {"random_state": parameters["random_state"]}
    if selected == "svd":
        expected_parameters.update(
            n_components=parameters["svd_n_components"], n_iter=parameters["svd_n_iter"],
        )
    if model["class"] != expected_class or model["parameters"] != expected_parameters:
        raise ValueError("Ranking model parameters differ from frozen run parameters")
    train = training["inputs"]["train"]
    catalog = training["inputs"]["catalog"]
    if Path(catalog["path"]) != Path(train["path"]).with_name("catalog.csv"):
        raise ValueError("Ranking catalog and train snapshot paths differ")
    for split, report in evaluations.items():
        if report["split"] != split or set(report["inputs"]) != {split, "catalog"}:
            raise ValueError(f"Ranking {split} split identity mismatch")
        if report["k"] != parameters["k"] or not 1 <= report["k"] <= report["n_catalog"]:
            raise ValueError(f"Ranking {split} k differs from frozen parameters or catalog size")
        if report["n_users"] != training["n_users"] or report["n_users"] <= 0:
            raise ValueError(f"Ranking {split} user count differs from training")
        if (report["inputs"]["catalog"] != catalog
                or report["n_catalog"] != catalog["rows"]):
            raise ValueError(f"Ranking {split} catalog identity mismatch")
        if (report["model"]["sha256"] != model["sha256"]
                or Path(report["model"]["path"]).name != model["file"]):
            raise ValueError(f"Ranking {split} trained model identity mismatch")
        source = report["inputs"][split]
        if (Path(source["path"]) != Path(train["path"]).with_name(f"{split}.csv")
                or source["rows"] != report["n_events"]
                or report["n_events"] < report["n_users"]):
            raise ValueError(f"Ranking {split} input identity or event count mismatch")
        for metric in ("recall_at_k", "ndcg_at_k"):
            if not 0 <= report["metrics"][metric] <= 1:
                raise ValueError(f"Ranking {split} {metric} must be in [0, 1]")
    if evaluations["validation"]["model"] != evaluations["test"]["model"]:
        raise ValueError("Ranking validation/test trained model identities differ")


def _step_record(run_dir, step, parameters, previous):
    eda = _report(run_dir, step, "eda_metrics", kind="eda")
    clustering = _report(run_dir, step, "clustering_metrics", kind="clustering")
    training = _report(run_dir, step, "ranking_manifest")
    evaluations = {split: _report(run_dir, step, split) for split in ("validation", "test")}
    _check_ranking(training, evaluations, parameters)
    reference, batch = step["reference"], step["batch"]
    expected_rows = batch["rows"] + (previous["reference"]["rows"] if previous else 0)
    if (eda["rows"] != reference["rows"] or reference["rows"] != expected_rows
            or eda["input"]["sha256"] != reference["sha256"]
            or clustering["inputs"]["dataset"] != eda["input"]
            or clustering["inputs"]["new_batch"]["sha256"] != batch["sha256"]):
        raise ValueError(f"Step {step['step_index']}: reference/batch report identity mismatch")
    drift = None
    if "deda_metrics" in step["artifacts"]:
        drift = _report(run_dir, step, "deda_metrics", kind="deda")
        if (not previous or drift["inputs"]["reference"]["sha256"] != previous["reference"]["sha256"]
                or drift["inputs"]["reference"]["rows"] != previous["reference"]["rows"]
                or drift["inputs"]["batch"] != {key: batch[key] for key in ("path", "sha256", "rows")}
                or drift["thresholds"] != parameters["thresholds"]):
            raise ValueError(f"Step {step['step_index']}: drift input identity or thresholds mismatch")
    elif previous:
        raise ValueError(f"Step {step['step_index']}: missing DEDA metrics")
    links = {
        name: "../" + quote(_relative_path(run_dir, relative).as_posix(), safe="/")
        for name, relative in step["artifacts"].items()
    }
    return {
        "step_index": step["step_index"], "model": step["model"],
        "config_sha256": step["config_sha256"], "batch": batch, "reference": reference,
        "data_quality": eda, "drift": drift, "clustering": clustering,
        "ranking": {"training": training, **evaluations},
        "durations_seconds": step["durations_seconds"], "links": links,
    }


def _flat_row(step):
    quality, clustering, ranking = step["data_quality"], step["clustering"], step["ranking"]
    validation, test = ranking["validation"], ranking["test"]
    drift = step["drift"]
    features = {row["feature"]: row for row in drift["metrics"]} if drift else {}
    row = {
        "step_index": step["step_index"], "batch_id": step["batch"]["id"],
        "batch_rows": step["batch"]["rows"], "source_sha256": step["batch"]["sha256"],
        "reference_sha256": step["reference"]["sha256"], "reference_rows": quality["rows"],
        **{key: quality[key] for key in ("n_orders", "n_customers", "n_products")},
        "quality_checks": len(quality["checks"]),
        "quality_checks_failed": sum(check["Результат"] != "OK" for check in quality["checks"]),
        "n_users": validation["n_users"], "n_catalog": validation["n_catalog"],
        "drift_detected": drift["drift_detected"] if drift else None,
        "silhouette": clustering["silhouette"], "silhouette_reason": clustering["silhouette_reason"],
        "k": validation["k"],
    }
    for name, feature in _DRIFT_FEATURES.items():
        metric = features[feature] if drift else None
        row[name] = metric["value"] if metric else None
        row[name.split("_")[0] + "_drift"] = metric["drift"] if metric else None
    for split, report in (("validation", validation), ("test", test)):
        row[f"{split}_n_events"] = report["n_events"]
        row[f"{split}_recall"] = report["metrics"]["recall_at_k"]
        row[f"{split}_ndcg"] = report["metrics"]["ndcg_at_k"]
    row.update({f"{name}_seconds": step["durations_seconds"].get(name) for name in _TIMINGS})
    return row


def _text(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    return format(value, ".6g") if isinstance(value, float) else str(value)


def _table(headers, rows):
    heading = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_text(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table"><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>'


def _html(summary, rows):
    parts = [
        '<!doctype html><html lang="ru"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Сводка запуска</title><style>',
        'body{font:16px system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#202530}',
        'table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{padding:.55rem;border:1px solid #d5dbe3;text-align:left}',
        'th{background:#edf1f6}tbody tr:nth-child(even){background:#f8fafc}.table{overflow-x:auto}',
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:1rem}h2{margin-top:2rem}',
        'a{color:#155da8}li{margin:.5rem 0}</style><body><h1>Сводка запуска</h1>',
        f'<p>Модель: <strong>{escape(summary["model"])}</strong>. Завершено шагов: {len(rows)}.</p>',
        '<h2>Параметры запуска</h2><pre>',
        escape(json.dumps({"parameters": summary["parameters"], "stream": summary["stream"]},
                          ensure_ascii=False, indent=2)), '</pre>',
        '<h2>Динамика данных и качество</h2>',
        _table(
            ["Шаг", "Батч", "Строк эталона", "Заказов", "Покупателей", "Товаров", "Проверок", "Нарушений", "Пользователей", "Каталог"],
            [[row[key] for key in ("step_index", "batch_id", "reference_rows", "n_orders", "n_customers",
                                  "n_products", "quality_checks", "quality_checks_failed", "n_users", "n_catalog")]
             for row in rows],
        ),
        '<h2>Дрейф и силуэт</h2><p>Дрейф сравнивает батч с эталоном до добавления; на первом шаге не оценивается.</p>',
        _table(
            ["Шаг", "Дрейф", "Цена KS D", "Дрейф цены", "Категории TVD", "Дрейф категорий", "Штаты TVD", "Дрейф штатов", "Силуэт / причина"],
            [[row["step_index"], row["drift_detected"], row["price_ks"], row["price_drift"],
              row["category_tvd"], row["category_drift"], row["state_tvd"], row["state_drift"],
              row["silhouette"] if row["silhouette"] is not None else row["silhouette_reason"]]
             for row in rows],
        ),
        '<h2>Ранжирование: validation и test</h2><p>Обе оценки используют одну обученную модель шага, полный каталог и одинаковое число пользователей.</p>',
        _table(
            ["Шаг", "K", "Validation: событий", "Validation Recall@K", "Validation NDCG@K", "Test: событий", "Test Recall@K", "Test NDCG@K"],
            [[row[key] for key in ("step_index", "k", "validation_n_events", "validation_recall", "validation_ndcg",
                                  "test_n_events", "test_recall", "test_ndcg")] for row in rows],
        ),
        '<h2>Время выполнения, секунды</h2>',
        _table(
            ["Шаг", "Эталон", "Кластеры: обучение", "Кластеры: оценка", "Истории", "Ранкер: обучение", "Validation", "Test", "Всего"],
            [[row["step_index"], *[row[f"{name}_seconds"] for name in _TIMINGS]] for row in rows],
        ),
        '<h2>Гиперпараметры моделей и проверки</h2>',
    ]
    for step in summary["steps"]:
        details = {
            "ranking": step["ranking"]["training"]["model"],
            "clustering": step["clustering"]["model"], "distance": step["clustering"]["distance"],
            "checks": step["data_quality"]["checks"],
        }
        parts.append(f'<details><summary>Шаг {step["step_index"]}</summary><pre>'
                     + escape(json.dumps(details, ensure_ascii=False, indent=2)) + '</pre></details>')
    parts.append('<h2>Отчёты шагов</h2><ul>')
    for step in summary["steps"]:
        links = " · ".join(
            f'<a href="{escape(step["links"][key], quote=True)}">{label}</a>'
            for key, label in (("eda", "EDA"), ("deda", "DEDA"), ("clustering", "Кластеризация"),
                               ("validation", "Validation (JSON)"), ("test", "Test (JSON)"))
            if key in step["links"]
        )
        parts.append(f'<li>Шаг {step["step_index"]}: {links}</li>')
    parts.append('</ul></body></html>\n')
    return "\n".join(parts)


def _output_paths(run_dir):
    output = run_dir / "summary"
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Summary output directory must be a real directory, not a symlink")
    paths = SummaryPaths(*(output / f"summary.{suffix}" for suffix in ("json", "csv", "html")))
    for path in vars(paths).values():
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink > 1)):
            raise ValueError(f"Summary output must not be a symlink or file alias: {path}")
    return paths


def _write_output(path, text):
    # Replace rather than truncate, so even an existing output inode is not modified.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(text)
            temporary.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def report_summary(run_dir: Path | str) -> SummaryPaths:
    """Write JSON, CSV and self-contained Russian HTML from verified completed steps.

    Original CSVs, notebooks, HTML reports and model binaries are never opened.
    Repeating the call replaces only the three summary outputs. No completed
    steps, inconsistent report identities and unsafe output aliases are errors.
    """
    run_dir = Path(run_dir).resolve()
    completed = read_completed_steps(run_dir)
    if not completed:
        raise ValueError("No completed steps in this run; complete a pipeline step before summarizing")
    config = _read_json(run_dir / "config.json")
    steps = []
    for step in completed:
        try:
            steps.append(_step_record(run_dir, step, config["parameters"], steps[-1] if steps else None))
        except (KeyError, TypeError) as error:
            raise ValueError(f"Step {step['step_index']}: invalid summary report metadata: {error}") from error
    summary = {
        "format_version": 1, "model": config["parameters"]["model"],
        "parameters": config["parameters"], "stream": config["stream"],
        "config_sha256": completed[0]["config_sha256"], "steps": steps,
    }
    rows = [_flat_row(step) for step in steps]
    csv_text = StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    texts = (
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        csv_text.getvalue(), _html(summary, rows),
    )
    paths = _output_paths(run_dir)
    paths.json_path.parent.mkdir(exist_ok=True)
    for path, text in zip(vars(paths).values(), texts, strict=True):
        _write_output(path, text)
    return paths
