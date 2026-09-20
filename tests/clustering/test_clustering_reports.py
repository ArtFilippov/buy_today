from functools import partial
import hashlib
from html.parser import HTMLParser
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbclient.exceptions import CellExecutionError
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.metrics import silhouette_score

from prak.clustering.distances import TimestampDistance
from prak.clustering.evaluation import evaluate_clustering
from prak.clustering.models.temporal import train_temporal
from prak.clustering.plots import plot_tsne, project_tsne
from prak.clustering.report import report_clustering
from prak.clustering.training import train_clustering


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.external = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            self.images.append(attrs.get("src", ""))
        if tag in {"img", "script", "iframe", "link"}:
            source = attrs.get("src") or attrs.get("href", "")
            if source and not source.startswith("data:"):
                self.external.append(source)


def test_silhouette_and_tsne_share_sample_and_distances(working_frame, monkeypatch):
    captured = {}

    def score(matrix, labels, *, metric):
        captured["matrix"] = matrix.copy()
        captured["labels"] = labels.copy()
        return silhouette_score(matrix, labels, metric=metric)

    class ProjectionSpy:
        def __init__(self, **kwargs):
            self.parameters = kwargs

        def get_params(self):
            return self.parameters

        def fit_transform(self, matrix):
            np.testing.assert_array_equal(matrix, captured["matrix"])
            # Even an in-place distance transformation must preserve score input.
            matrix[:] = 123
            return np.arange(len(matrix) * 2).reshape(-1, 2)

    monkeypatch.setattr("prak.clustering.evaluation.silhouette_score", score)
    monkeypatch.setattr("prak.clustering.plots.TSNE", ProjectionSpy)
    result = evaluate_clustering(
        working_frame, np.arange(12) // 4, distance=TimestampDistance(),
        new_rows=np.arange(12) >= 6, max_evaluation_rows=8,
    )
    projection = project_tsne(result, random_state=17)
    np.testing.assert_array_equal(result.distances, captured["matrix"])
    np.testing.assert_array_equal(result.labels, captured["labels"])
    assert projection.parameters["perplexity"] == 7
    assert projection.parameters["metric"] == "precomputed"
    assert projection.parameters["init"] == "random"
    assert projection.parameters["random_state"] == 17
    figure = plot_tsne(result, projection)
    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert axis.get_xlabel() == "t-SNE 1"
        assert axis.get_ylabel() == "t-SNE 2"
        assert axis.get_aspect() == 1
        assert [t.get_text() for t in axis.get_legend().get_texts()] == ["0", "1", "2"]
        assert sum(len(points.get_offsets()) for points in axis.collections) == 8
    finally:
        plt.close(figure)


@pytest.mark.parametrize("size", [1, 2, 3])
def test_tiny_projection_boundary(working_frame, size):
    result = evaluate_clustering(
        working_frame.iloc[:size], np.zeros(size, dtype=int), distance=TimestampDistance(), new_rows=[True] * size,
    )
    projection = project_tsne(result)
    if size < 3:
        assert projection.coordinates is None
        assert "хотя бы 3" in projection.reason
    else:
        assert projection.coordinates.shape == (3, 2)
        assert np.isfinite(projection.coordinates).all()
        assert projection.parameters["perplexity"] == 2


def test_executed_report_uses_saved_labels_exports_html_and_leaves_artifacts_unchanged(
    working_frame, new_batch, write_dataset, tmp_path,
):
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
    reports = report_clustering(dataset, batch, model_dir, max_evaluation_rows=10, random_state=17)
    notebook = nbformat.read(reports.notebook_path, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    outputs = [output for cell in code for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    assert sum("image/png" in output.get("data", {}) for output in outputs) == 1
    text = "\n".join(output.get("text", "") for output in outputs)
    assert sys.executable in text
    assert "Выбрано строк: 10; новых: 5; прежних: 5" in text
    assert "Силуэт:" in text
    assert "'perplexity': 9.0" in text
    assert "'random_state': 17" in text
    for path in (dataset, batch, paths.labels_path):
        assert hashlib.sha256(before[path]).hexdigest() in text
    assert "evaluate_clustering(" in "\n".join(cell.source for cell in code)
    assert {path: path.read_bytes() for path in inputs} == before
    assert {path.name for path in model_dir.iterdir()} == {
        "model.joblib", "distance.joblib", "labels.csv", "report.ipynb", "report.html",
    }
    html = reports.html_path.read_text(encoding="utf-8")
    parser = Assets()
    parser.feed(html)
    assert len(parser.images) == 1
    assert parser.images[0].startswith("data:image/png;base64,")
    assert parser.external == []
    for content in ("Размеры и период", "24", "2018-01-02 05:00:00", "Обязательные проверки", "Параметры", "SHA-256"):
        assert content in html

    # Re-evaluation overwrites reports and can succeed with no defined silhouette.
    report_clustering(dataset, batch, model_dir, max_evaluation_rows=1)
    html = reports.html_path.read_text(encoding="utf-8")
    assert "Силуэт не определён" in html
    assert "t-SNE не построена" in html
    assert "data:image/png" not in html
    assert {path: path.read_bytes() for path in inputs} == before


def test_bad_saved_keys_stop_execution_before_projection(working_frame, write_dataset, tmp_path):
    dataset = write_dataset(working_frame)
    model_dir = tmp_path / "model"
    paths = train_clustering(dataset, model_dir, strategy=partial(train_temporal, n_clusters=2))
    table = pd.read_csv(paths.labels_path).iloc[1:]
    table.to_csv(paths.labels_path, index=False)
    with pytest.raises(CellExecutionError, match="missing=1"):
        report_clustering(dataset, dataset, model_dir)
    notebook = nbformat.read(model_dir / "report.ipynb", as_version=4)
    outputs = [output for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs]
    assert any(output.output_type == "error" for output in outputs)
    assert not any("image/png" in output.get("data", {}) for output in outputs)
    assert not (model_dir / "report.html").exists()
