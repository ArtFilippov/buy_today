"""Portable comparisons of the latest committed ranking benchmark runs."""

import base64
import csv
from dataclasses import dataclass
from datetime import datetime
from html import escape
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import textwrap

import numpy as np

from prak.ranking.benchmark import read_latest_benchmark_runs


@dataclass(frozen=True)
class BenchmarkReportPaths:
    html_path: Path
    csv_path: Path


_METRICS = {"recall_at_k": "Recall@K", "ndcg_at_k": "NDCG@K"}
_COUNTS = ("k", "n_users", "n_catalog", "n_train_events", "n_test_events")
_TIMINGS = ("fit_seconds", "evaluation_seconds")
_RESULT_COLUMNS = ("model", "dataset", *_COUNTS, *_METRICS, *_TIMINGS)
_CSV_COLUMNS = (
    "rank", "score", "model", "algorithm", "parameters", "run_id", "run_path",
    "invocation_id", "created_at", "save_model", "versions", "dataset", "dataset_path",
    *_COUNTS, *_METRICS, *_TIMINGS, "user_ids_sha256",
    *[f"{split}_{field}" for split in ("train", "test", "catalog")
      for field in ("path", "sha256", "rows")],
)
_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 120, "text.parse_math": False,
}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, maximum=None):
    # DCG dot-product and IDCG cumulative sum can round differently at 1.
    # Accept floating-point noise without changing the saved metric or score.
    if (type(value) not in (float, int) or not math.isfinite(value) or value < 0
            or (maximum is not None and value > maximum + 1e-12)):
        interval = "[0, 1]" if maximum == 1 else "[0, infinity)"
        raise ValueError(f"{name} must be finite and in {interval}")
    return value


def _sha256(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a SHA256 hex digest")


def _absolute_path(value, name):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return Path(value)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _check_model(model):
    algorithm = model["algorithm"]
    classes = {"random": "prak.ranking.random.RandomRanker", "svd": "prak.ranking.svd.SVDRanker"}
    if algorithm not in classes or model["class"] != classes[algorithm]:
        raise ValueError("Model algorithm/class mismatch")
    parameters = model["parameters"]
    expected = {"random_state"} | ({"n_components", "n_iter"} if algorithm == "svd" else set())
    if not isinstance(parameters, dict) or set(parameters) != expected:
        raise ValueError("Model parameters must include all effective parameters and seed")
    if _integer(parameters["random_state"], "random_state", 0) >= 2**32:
        raise ValueError("random_state must be less than 2**32")
    for name in expected - {"random_state"}:
        _integer(parameters[name], name)


def _check_dataset(name, dataset, model, k):
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
        raise ValueError("Dataset key must be a basename")
    directory = _absolute_path(dataset["path"], "dataset path")
    if directory.name != name or set(dataset["inputs"]) != {"train", "test", "catalog"}:
        raise ValueError(f"Dataset {name}: input identity mismatch")
    for split, source in dataset["inputs"].items():
        if _absolute_path(source["path"], f"{split} path") != directory / f"{split}.csv":
            raise ValueError(f"Dataset {name}: {split} input path mismatch")
        _sha256(source["sha256"], f"{split} SHA256")
        _integer(source["rows"], f"{split} rows")
    users = _integer(dataset["n_users"], "n_users")
    _sha256(dataset["user_ids_sha256"], "user_ids_sha256")
    catalog = dataset["inputs"]["catalog"]["rows"]
    if k > catalog or any(dataset["inputs"][split]["rows"] < users for split in ("train", "test")):
        raise ValueError(f"Dataset {name}: inconsistent counts or K exceeds catalog size")
    if model["algorithm"] == "svd" and (
        catalog < 2 or model["parameters"]["n_components"] > min(users, catalog)
    ):
        raise ValueError(f"Dataset {name}: SVD dimensions exceed user/catalog counts")
    for field in _TIMINGS:
        _number(dataset[field], field)


def _results(run_path):
    with (run_path / "results.csv").open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not set(_RESULT_COLUMNS).issubset(reader.fieldnames)):
            raise ValueError("results.csv has missing or duplicate columns")
        rows = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("results.csv has an invalid row")
            name = row["dataset"]
            if name in rows:
                raise ValueError(f"results.csv has duplicate dataset {name}")
            for field in _COUNTS:
                row[field] = _integer(int(row[field]), field)
            for field in (*_METRICS, *_TIMINGS):
                row[field] = _number(float(row[field]), field, 1 if field in _METRICS else None)
            rows[name] = row
    return rows


def _check_evaluation(run_path, manifest, name, dataset, row):
    path = run_path / "datasets" / name / "test" / "metrics.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if type(report["format_version"]) is not int or report["format_version"] != 1 or report["split"] != "test":
        raise ValueError(f"Dataset {name}: expected test metrics format_version=1")
    expected_counts = {
        "k": manifest["k"], "n_users": dataset["n_users"],
        "n_catalog": dataset["inputs"]["catalog"]["rows"],
        "n_events": dataset["inputs"]["test"]["rows"],
    }
    for field, expected in expected_counts.items():
        if _integer(report[field], field) != expected:
            raise ValueError(f"Dataset {name}: evaluation {field} count/K mismatch")
    if _json(report["inputs"]) != _json({split: dataset["inputs"][split] for split in ("test", "catalog")}):
        raise ValueError(f"Dataset {name}: evaluation input provenance mismatch")
    identity = report["model"]
    if (identity["name"] != manifest["model"]["name"]
            or _json(identity["parameters"]) != _json(manifest["model"]["parameters"])):
        raise ValueError(f"Dataset {name}: evaluation model parameters/identity mismatch")
    if manifest["save_model"]:
        model_path = _absolute_path(identity["path"], "model path")
        # The absolute provenance may refer to the original location of a copied run.
        expected_suffix = (manifest["model"]["name"], "runs", manifest["run_id"],
                           "datasets", name, "model", "model.joblib")
        if model_path.parts[-len(expected_suffix):] != expected_suffix:
            raise ValueError(f"Dataset {name}: saved model path identity mismatch")
        _sha256(identity["sha256"], "model SHA256")
    elif identity["path"] is not None or identity["sha256"] is not None:
        raise ValueError(f"Dataset {name}: unsaved model must have null path and SHA256")
    for field in _METRICS:
        if _number(report["metrics"][field], field, 1) != row[field]:
            raise ValueError(f"Dataset {name}: {field} differs between results.csv and metrics.json")


def _run_rows(run_path, manifest):
    model = manifest["model"]
    _check_model(model)
    k = _integer(manifest["k"], "K")
    if type(manifest["save_model"]) is not bool:
        raise ValueError("save_model must be a boolean")
    datetime.fromisoformat(manifest["created_at"])
    for package in ("numpy", "scikit_learn"):
        if not isinstance(manifest["versions"][package], str) or not manifest["versions"][package]:
            raise ValueError("Package versions must be nonempty strings")
    datasets = manifest["datasets"]
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("A benchmark run must contain datasets")
    results = _results(run_path)
    if set(results) != set(datasets):
        raise ValueError("results.csv and manifest must contain the same dataset set")
    rows = []
    for name, dataset in sorted(datasets.items()):
        _check_dataset(name, dataset, model, k)
        row = results[name]
        expected = {
            "model": model["name"], "k": k, "n_users": dataset["n_users"],
            "n_catalog": dataset["inputs"]["catalog"]["rows"],
            "n_train_events": dataset["inputs"]["train"]["rows"],
            "n_test_events": dataset["inputs"]["test"]["rows"],
            **{field: dataset[field] for field in _TIMINGS},
        }
        if any(row[field] != value for field, value in expected.items()):
            raise ValueError(f"Dataset {name}: results.csv model/counts/K/timings mismatch")
        _check_evaluation(run_path, manifest, name, dataset, row)
        rows.append({
            **{field: row[field] for field in _RESULT_COLUMNS},
            "algorithm": model["algorithm"], "parameters": _json(model["parameters"]),
            "run_id": manifest["run_id"], "run_path": str(run_path),
            "invocation_id": manifest["invocation_id"], "created_at": manifest["created_at"],
            "save_model": manifest["save_model"], "versions": _json(manifest["versions"]),
            "dataset_path": dataset["path"], "user_ids_sha256": dataset["user_ids_sha256"],
            **{f"{split}_{field}": source[field] for split, source in dataset["inputs"].items()
               for field in ("path", "sha256", "rows")},
        })
    return rows


def _comparison_rows(selected):
    rows, reference = [], None
    for run_path, manifest in selected:
        try:
            current = _run_rows(run_path, manifest)
        except (KeyError, TypeError, AttributeError, OverflowError, OSError, ValueError) as error:
            raise ValueError(f"Run {run_path}: invalid benchmark report metadata: {error}") from error
        identities = {
            row["dataset"]: {field: row[field] for field in (
                "k", "n_users", "user_ids_sha256", "train_sha256", "test_sha256",
                "catalog_sha256", "train_rows", "test_rows", "catalog_rows",
            )} for row in current
        }
        if reference is None:
            reference = identities
        elif set(identities) != set(reference):
            raise ValueError("Latest model runs must contain the same dataset set")
        else:
            for name, identity in identities.items():
                if identity["k"] != reference[name]["k"]:
                    raise ValueError("Latest model runs must use the same K")
                if identity != reference[name]:
                    raise ValueError(f"Dataset {name}: input SHA256/counts/user identity differs across models")
        rows.extend(current)
    scores = {manifest["model"]["name"]: max(
        row["ndcg_at_k"] for row in rows if row["model"] == manifest["model"]["name"]
    ) for _, manifest in selected}
    order = sorted(scores, key=lambda name: (-scores[name], name))
    ranks = {name: index + 1 for index, name in enumerate(order)}
    for row in rows:
        row.update(score=scores[row["model"]], rank=ranks[row["model"]])
    return sorted(rows, key=lambda row: (row["rank"], row["dataset"]))


def _figure(width, height):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _labels(names, width=34):
    return [textwrap.fill(name, width) for name in names]


def _heatmap(rows, metric):
    import matplotlib as mpl

    models = list(dict.fromkeys(row["model"] for row in rows))
    datasets = sorted({row["dataset"] for row in rows})
    lookup = {(row["model"], row["dataset"]): row[metric] for row in rows}
    values = np.array([[lookup[model, dataset] for dataset in datasets] for model in models])
    labels = _labels(models)
    with mpl.rc_context(_STYLE):
        figure = _figure(max(8, 4 + 1.5 * len(datasets)), max(3.5, 2.2 + .35 * sum(
            label.count("\n") + 1 for label in labels)))
        ax = figure.subplots()
        image = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_yticks(range(len(models)), labels)
        ax.set_xticks(range(len(datasets)), _labels(datasets, 18))
        ax.set(title=f"{_METRICS[metric]} · K = {rows[0]['k']} · больше — лучше",
               xlabel="Датасет", ylabel="Конфигурация модели")
        for i, j in np.ndindex(values.shape):
            best = values[i, j] == values[:, j].max()
            ax.text(j, i, f"{values[i, j]:.3f}" + (" *" if best else ""),
                    ha="center", va="center", color="white" if values[i, j] < .45 else "black",
                    fontweight="bold" if best else "normal")
        figure.colorbar(image, ax=ax, label="Среднее по пользователям", ticks=np.linspace(0, 1, 6))
        return figure


def _dataset_plot(rows):
    import matplotlib as mpl

    labels = _labels([row["model"] for row in rows])
    with mpl.rc_context(_STYLE):
        figure = _figure(13, max(3.5, 2 + .42 * sum(label.count("\n") + 1 for label in labels)))
        axes = figure.subplots(1, 2, sharey=True)
        for panel, (metric, title), ax in zip("ab", _METRICS.items(), axes, strict=True):
            values = [row[metric] for row in rows]
            bars = ax.barh(range(len(rows)), values, color="#287a9f", height=.65)
            ax.set_yticks(range(len(rows)), labels)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set(title=f"({panel}) {title}", xlabel="Среднее по пользователям · больше — лучше")
            ax.grid(axis="x", alpha=.2)
            ax.set_axisbelow(True)
            for bar, value in zip(bars, values, strict=True):
                best = value == max(values)
                ax.text(value - .015 if value > .8 else value + .015,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.3f}" + (" *" if best else ""), va="center",
                        ha="right" if value > .8 else "left",
                        color="white" if value > .8 else "#202530",
                        fontweight="bold" if best else "normal")
        axes[0].invert_yaxis()
        figure.suptitle(textwrap.fill(rows[0]["dataset"], 90)
                       + f" · K = {rows[0]['k']} · пользователей: {rows[0]['n_users']}")
        return figure


def _image(figure, title):
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    figure.clear()
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<figure><img alt="{escape(title, quote=True)}" src="data:image/png;base64,{encoded}"></figure>'


def _table(headers, rows, *, best=None):
    heading = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = []
    for index, row in enumerate(rows):
        cells = []
        for column, value in enumerate(row):
            text = escape(str(value))
            if best and (index, column) in best:
                cells.append(f'<td class="best"><strong>{text}</strong></td>')
            else:
                cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return '<div class="table"><table><thead><tr>' + heading + '</tr></thead><tbody>' + "".join(body) + '</tbody></table></div>'


def _html(root, selected, rows):
    models = list(dict.fromkeys(row["model"] for row in rows))
    representatives = {row["model"]: row for row in rows}
    parts = [
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Сравнение моделей ранжирования</title><style>',
        'body{font:16px system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#202530}',
        'table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{padding:.55rem;border:1px solid #d5dbe3;text-align:left}',
        'th{background:#edf1f6}tbody tr:nth-child(even){background:#f8fafc}.table{overflow-x:auto}',
        'td.best{background:#e3f2d4}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:1rem}',
        'figure{margin:1rem 0;overflow-x:auto}img{display:block;width:100%;min-width:700px}h2{margin-top:2rem}',
        '</style></head><body><h1>Сравнение моделей ранжирования</h1>',
        f'<p>Корень результатов: <code>{escape(str(root))}</code>. K = {rows[0]["k"]}.</p>',
        '<p>Обучение на train, оценка на test того же датасета. Recall@K и бинарный NDCG@K — средние по пользователям; больше — лучше. '
        'Каждая ячейка — один запуск, без оценки статистической неопределённости.</p>',
        '<h2>Общий рейтинг: максимум NDCG@K по датасетам</h2>',
        '<p><code>model_score = max(ndcg_at_k по всем датасетам)</code>. '
        'При равных score порядок определяется каноническим именем модели. Место — позиция в этом порядке. '
        'Максимум отражает лучший результат модели на одном из датасетов.</p>',
        _table(["Место", "Конфигурация модели", "Score: максимум NDCG@K"],
               [[representatives[name][field] for field in ("rank", "model", "score")] for name in models]),
        '<h2>Тепловые карты</h2><p>Обе цветовые шкалы фиксированы: 0–1. '
        'Звёздочка (*) отмечает все лучшие значения по датасету и метрике, включая равенства. '
        'Подписи графиков округлены до трёх знаков; точные сохранённые значения приведены в таблицах и CSV.</p>',
    ]
    for metric, title in _METRICS.items():
        parts.append(_image(_heatmap(rows, metric), f"Тепловая карта {title}; шкала от 0 до 1"))
    parts.append('<h2>Сравнение по датасетам и точные значения</h2>')
    for name in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == name]
        best = {(i, j + 1) for j, metric in enumerate(_METRICS) for i, row in enumerate(subset)
                if row[metric] == max(item[metric] for item in subset)}
        parts.extend([
            f'<h3>{escape(name)}</h3>',
            _image(_dataset_plot(subset), f"Сравнение Recall@K и NDCG@K: {name}; шкалы от 0 до 1"),
            _table(["Модель", *_METRICS.values(), "Пользователей", "Товаров", "Событий train", "Событий test"],
                   [[row[field] for field in ("model", *_METRICS, "n_users", "n_catalog", "n_train_events", "n_test_events")]
                    for row in subset], best=best),
        ])
    parts.extend([
        '<h2>Время выполнения, секунды</h2><p>Обучение (fit) и оценка измерены отдельно для каждой пары. '
        'Запуски могли выполняться в разных условиях; время не участвует в рейтинге.</p>',
        _table(["Модель", "Датасет", "Обучение (fit), с", "Оценка, с"],
               [[row[field] for field in ("model", "dataset", *_TIMINGS)] for row in rows]),
        '<h2>Параметры, seeds и выбранные runs</h2><p>Для каждой модели выбран последний завершённый run '
        'по лексикографическому имени папки, независимо от mtime и created_at.</p>',
        _table(["Модель", "Алгоритм", "Все параметры, включая seed", "Run ID", "Путь run", "Создан"],
               [[representatives[name][field] for field in ("model", "algorithm", "parameters", "run_id", "run_path", "created_at")]
                for name in models]),
        '<h2>Происхождение данных и метаданные экспериментов</h2><p>Совпадение train, test и catalog '
        'между моделями проверено по SHA256 и числу строк для каждого basename. Пути могут различаться. '
        'Отчёт построен по сохранённым метаданным, без чтения исходных данных и файлов моделей.</p>',
    ])
    for run_path, manifest in selected:
        parts.append(f'<details><summary>{escape(manifest["model"]["name"])} · {escape(manifest["run_id"])}</summary><pre>'
                     + escape(json.dumps({"run_path": str(run_path), **manifest}, ensure_ascii=False,
                                         sort_keys=True, indent=2, allow_nan=False)) + '</pre></details>')
    parts.append('</body></html>\n')
    return "\n".join(parts)


def _output_paths(root, output, rows):
    output = Path(output).absolute()
    if output.suffix.lower() not in (".html", ".htm"):
        raise ValueError("Report output must be an HTML file (.html or .htm)")
    paths = BenchmarkReportPaths(output, output.with_suffix(".csv"))
    sources = {Path(row["dataset_path"]).resolve() for row in rows}
    for path in vars(paths).values():
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink > 1)):
            raise ValueError(f"Report output must not be a symlink, directory or file alias: {path}")
        resolved = path.resolve()
        if resolved.is_relative_to(root):
            relative = resolved.relative_to(root).parts
            if relative and (relative[0].startswith(".") or (len(relative) > 1 and relative[1] == "runs")):
                raise ValueError("Report output collides with benchmark artifacts")
        if any(resolved == source or source in resolved.parents for source in sources):
            raise ValueError("Report output collides with source dataset paths")
    return BenchmarkReportPaths(paths.html_path.resolve(), paths.csv_path.resolve())


def _write_output(path, text):
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(text)
            temporary.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def report_ranking_benchmark(root: Path | str, output: Path | str) -> BenchmarkReportPaths:
    """Write self-contained Russian HTML and an adjacent long-matrix CSV.

    Only committed run metadata, results.csv and test/metrics.json are read.
    Every model must have the same datasets, K and recorded input identities.
    The publication reader owns run selection and commit validation. Outputs
    are replaced atomically per file after the entire comparison is validated.
    """
    root = Path(root).resolve()
    selected = read_latest_benchmark_runs(root)
    if not selected:
        raise ValueError("No completed benchmark runs to compare")
    rows = _comparison_rows(selected)
    paths = _output_paths(root, output, rows)
    csv_text = StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=_CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    html = _html(root, selected, rows)
    paths.html_path.parent.mkdir(parents=True, exist_ok=True)
    _write_output(paths.csv_path, csv_text.getvalue())
    _write_output(paths.html_path, html)
    return paths
