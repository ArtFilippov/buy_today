from functools import partial
from contextlib import contextmanager
from collections.abc import Generator
import hashlib
from html.parser import HTMLParser
import json
import sys
from typing import Any, override

import joblib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import nbformat
from nbclient.exceptions import CellExecutionError
import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

import fixture_types as ft

from buy_today.clustering.distances import TimestampDistance
from buy_today.clustering.evaluation import evaluate_clustering
from buy_today.clustering.models.temporal import train_temporal
from buy_today.clustering.plots import plot_tsne, project_tsne
from buy_today.clustering.report import report_clustering
from buy_today.clustering.training import train_clustering

matplotlib.use("Agg")


@contextmanager
def _figure_axis(figure: Figure) -> Generator[Axes]:
    try:
        assert len(figure.axes) == 1
        yield figure.axes[0]
    finally:
        plt.close(figure)


class Assets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []
        self.external: list[str] = []

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "img":
            self.images.append(attributes.get("src") or "")
        if tag in {"img", "script", "iframe", "link"}:
            source = attributes.get("src") or attributes.get("href", "")
            if source and not source.startswith("data:"):
                self.external.append(source)


def test_silhouette_and_tsne_share_sample_and_distances(
    working_frame: ft.DataFrame, monkeypatch: ft.MonkeyPatch
) -> None:
    captured: dict[str, NDArray[Any]] = {}

    def score(matrix: NDArray[Any], labels: NDArray[Any], *, metric: str) -> float:
        captured["matrix"] = matrix.copy()
        captured["labels"] = labels.copy()
        return float(silhouette_score(matrix, labels, metric=metric))

    class ProjectionSpy:
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.parameters = kwargs

        def get_params(self) -> dict[str, object]:
            return self.parameters

        def fit_transform(self, matrix: NDArray[Any]) -> NDArray[np.int64]:
            np.testing.assert_array_equal(matrix, captured["matrix"])
            # Even an in-place distance transformation must preserve score input.
            matrix[:] = 123
            return np.arange(len(matrix) * 2).reshape(-1, 2)

    monkeypatch.setattr("buy_today.clustering.evaluation.silhouette_score", score)
    monkeypatch.setattr("buy_today.clustering.plots.TSNE", ProjectionSpy)
    result = evaluate_clustering(
        working_frame,
        np.arange(12) // 4,
        distance=TimestampDistance(),
        new_rows=np.arange(12) >= 6,
        max_evaluation_rows=8,
    )
    projection = project_tsne(result, random_state=17)
    np.testing.assert_array_equal(result.distances, captured["matrix"])
    np.testing.assert_array_equal(result.labels, captured["labels"])
    assert projection.parameters["perplexity"] == 7
    assert projection.parameters["metric"] == "precomputed"
    assert projection.parameters["init"] == "random"
    assert projection.parameters["random_state"] == 17
    with _figure_axis(plot_tsne(result, projection)) as axis:
        assert axis.get_xlabel() == "t-SNE 1"
        assert axis.get_ylabel() == "t-SNE 2"
        assert axis.get_aspect() == 1
        legend = axis.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["0", "1", "2"]
        assert sum(len(np.asarray(points.get_offsets())) for points in axis.collections) == 8


@pytest.mark.parametrize("size", [1, 2, 3])
def test_tiny_projection_boundary(working_frame: ft.DataFrame, size: int) -> None:
    result = evaluate_clustering(
        working_frame.iloc[:size],
        np.zeros(size, dtype=int),
        distance=TimestampDistance(),
        new_rows=[True] * size,
    )
    projection = project_tsne(result)
    if size < 3:
        assert projection.coordinates is None
        assert projection.reason is not None
        assert "хотя бы 3" in projection.reason
    else:
        assert projection.coordinates is not None
        assert projection.coordinates.shape == (3, 2)
        assert np.isfinite(projection.coordinates).all()
        assert projection.parameters["perplexity"] == 2


def _executed_report_text(path: ft.Path) -> str:
    notebook: nbformat.NotebookNode = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    outputs = [output for cell in code for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    assert sum("image/png" in output.get("data", {}) for output in outputs) == 1
    assert "evaluate_clustering(" in "\n".join(cell.source for cell in code)
    return "\n".join(output.get("text", "") for output in outputs)


def _check_report_html(path: ft.Path) -> None:
    html = path.read_text(encoding="utf-8")
    parser = Assets()
    parser.feed(html)
    assert len(parser.images) == 1
    assert parser.images[0].startswith("data:image/png;base64,")
    assert not parser.external
    for content in (
        "Размеры и период",
        "24",
        "2018-01-02 05:00:00",
        "Обязательные проверки",
        "Параметры",
        "SHA-256",
    ):
        assert content in html


def test_executed_report_uses_saved_labels_exports_html_and_leaves_artifacts_unchanged(
    working_frame: ft.DataFrame,
    new_batch: ft.DataFrame,
    write_dataset: ft.DatasetWriter,
    tmp_path: ft.Path,
) -> None:
    frame = pd.concat([working_frame, new_batch], ignore_index=True)
    dataset = write_dataset(frame, 'dataset " whole.csv')
    batch = write_dataset(new_batch, "new ' batch.csv")
    model_dir = tmp_path / "model ' dir"
    paths = train_clustering(dataset, model_dir, strategy=partial(train_temporal, n_clusters=3))
    # A model without fit/predict/labels_ suffices: saved labels are authoritative.
    joblib.dump(BaseEstimator(), paths.model_path)
    assignments = pd.read_csv(paths.labels_path).sample(frac=1, random_state=2)
    assignments.to_csv(paths.labels_path, index=False)
    inputs = [dataset, batch, paths.model_path, paths.distance_path, paths.labels_path]
    before = {path: path.read_bytes() for path in inputs}
    reports = report_clustering(
        dataset_path=dataset,
        new_batch_path=batch,
        model_dir=model_dir,
        max_evaluation_rows=10,
        random_state=17,
    )
    text = _executed_report_text(reports.notebook_path)
    assert sys.executable in text
    assert "Выбрано строк: 10; новых: 5; прежних: 5" in text
    assert "Силуэт:" in text
    assert "'perplexity': 9.0" in text
    assert "'random_state': 17" in text
    for path in (dataset, batch, paths.labels_path):
        assert hashlib.sha256(before[path]).hexdigest() in text
    assert {path: path.read_bytes() for path in inputs} == before
    assert {path.name for path in model_dir.iterdir()} == {
        "model.joblib",
        "distance.joblib",
        "labels.csv",
        "report.ipynb",
        "report.html",
        "metrics.json",
    }
    metrics_text = reports.metrics_path.read_text(encoding="utf-8")
    metrics = json.loads(metrics_text)
    expected: dict[str, Any] = {
        "format_version": 1,
        "kind": "clustering",
        "inputs": {
            role: {"path": str(path), "sha256": hashlib.sha256(before[path]).hexdigest()}
            for role, path in (
                ("dataset", dataset),
                ("new_batch", batch),
                ("labels", paths.labels_path),
            )
        },
        "silhouette": pytest.approx(0.5479603297766322),
        "silhouette_reason": None,
        "sample_rows": 10,
        "new_count": 5,
        "previous_count": 5,
        "max_evaluation_rows": 10,
        "random_state": 17,
        "model": {"class": "BaseEstimator", "parameters": {}},
        "distance": {"class": "TimestampDistance", "parameters": {}},
    }
    assert metrics == expected
    assert isinstance(metrics["silhouette"], float)
    for key in (
        "format_version",
        "sample_rows",
        "new_count",
        "previous_count",
        "max_evaluation_rows",
        "random_state",
    ):
        assert isinstance(metrics[key], int) and not isinstance(metrics[key], bool)
    assert metrics["sample_rows"] == metrics["new_count"] + metrics["previous_count"]
    assert f"Силуэт: {metrics['silhouette']:.6f}" in text
    assert metrics_text == json.dumps(metrics, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    _check_report_html(reports.html_path)

    # Re-evaluation overwrites reports and can succeed with no defined silhouette.
    report_clustering(dataset, batch, model_dir, max_evaluation_rows=1)
    html = reports.html_path.read_text(encoding="utf-8")
    assert "Силуэт не определён" in html
    assert "t-SNE не построена" in html
    assert "data:image/png" not in html
    assert {path: path.read_bytes() for path in inputs} == before
    metrics_text = reports.metrics_path.read_text(encoding="utf-8")
    metrics = json.loads(metrics_text)
    assert metrics == {
        **expected,
        "silhouette": None,
        "silhouette_reason": (
            "Силуэт не определён: требуется 2 <= число меток <= число строк - 1; "
            + "меток: 1, строк: 1."
        ),
        "sample_rows": 1,
        "new_count": 1,
        "previous_count": 0,
        "max_evaluation_rows": 1,
        "random_state": 42,
    }
    assert metrics["silhouette"] is None
    assert metrics_text == json.dumps(metrics, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def test_bad_saved_keys_stop_execution_before_projection(
    working_frame: ft.DataFrame, write_dataset: ft.DatasetWriter, tmp_path: ft.Path
) -> None:
    dataset = write_dataset(working_frame)
    model_dir = tmp_path / "model"
    paths = train_clustering(dataset, model_dir, strategy=partial(train_temporal, n_clusters=2))
    table = pd.read_csv(paths.labels_path).iloc[1:]
    table.to_csv(paths.labels_path, index=False)
    (model_dir / "metrics.json").write_text('{"kind": "clustering"}\n', encoding="utf-8")
    with pytest.raises(CellExecutionError, match="missing=1"):
        report_clustering(dataset, dataset, model_dir)
    notebook: nbformat.NotebookNode = nbformat.read(model_dir / "report.ipynb", as_version=4)
    outputs = [
        output for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
    ]
    assert any(output.output_type == "error" for output in outputs)
    assert not any("image/png" in output.get("data", {}) for output in outputs)
    assert not (model_dir / "report.html").exists()
    assert not (model_dir / "metrics.json").exists()


@pytest.mark.parametrize(
    "options", [{"max_evaluation_rows": 0}, {"random_state": -1}, {"random_state": np.bool_(True)}]
)
def test_invalid_report_parameters_fail_before_io(
    tmp_path: ft.Path, options: dict[str, Any]
) -> None:
    model_dir = tmp_path / "model"
    with pytest.raises(ValueError):
        report_clustering(tmp_path / "dataset.csv", tmp_path / "batch.csv", model_dir, **options)
    assert not model_dir.exists()


def test_unknown_report_options_fail_before_io(tmp_path: ft.Path) -> None:
    options: dict[str, Any] = {"random_seed": 42}
    model_dir = tmp_path / "model"
    with pytest.raises(TypeError, match="random_seed"):
        report_clustering(tmp_path / "dataset.csv", tmp_path / "batch.csv", model_dir, **options)
    assert not model_dir.exists()


def test_report_serializes_nested_estimator_parameters(
    working_frame: ft.DataFrame, write_dataset: ft.DatasetWriter, tmp_path: ft.Path
) -> None:
    dataset = write_dataset(working_frame)
    model_dir = tmp_path / "nested estimator"
    paths = train_clustering(dataset, model_dir, strategy=partial(train_temporal, n_clusters=3))
    # Saved labels, not the model's inference, are evaluated. Metadata may have
    # nested estimators and RNG state, both legitimate sklearn parameters.
    model = Pipeline([("model", DummyClassifier(random_state=RandomState(42)))])
    joblib.dump(model, paths.model_path)
    report = report_clustering(dataset, dataset, model_dir, max_evaluation_rows=6, thread_limit=1)
    metrics = json.loads(report.metrics_path.read_text(encoding="utf-8"))
    parameters = metrics["model"]["parameters"]
    assert parameters["model"]["class"] == "DummyClassifier"
    assert parameters["model__random_state"]["class"] == "RandomState"
    assert parameters["model__random_state"]["state"][0] == "MT19937"
    assert report.html_path.is_file()
