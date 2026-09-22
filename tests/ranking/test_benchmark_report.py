"""Comparative reports over committed artifacts and actual benchmark runs."""

import base64
import csv
from dataclasses import FrozenInstanceError
import hashlib
from html.parser import HTMLParser
from io import BytesIO
import json
import os
from pathlib import Path
import shutil

import joblib
import matplotlib.image as mpimg
import pandas as pd
import pytest

from prak.ranking.benchmark import run_ranking_benchmark
from prak.ranking.benchmark_report import BenchmarkReportPaths, report_ranking_benchmark
from prak.ranking.random import RandomRanker
from prak.ranking.svd import SVDRanker


_FIRST = "20260922T100000000000Z"
_LAST = "20260922T110000000000Z"
_RESULT_COLUMNS = (
    "model", "dataset", "k", "n_users", "n_catalog", "n_train_events", "n_test_events",
    "recall_at_k", "ndcg_at_k", "fit_seconds", "evaluation_seconds",
)
_MODELS = {
    "random": ("random", "RandomRanker", {"random_state": 42}),
    "random_random-state-73": ("random", "RandomRanker", {"random_state": 73}),
    "svd_n-components-2": ("svd", "SVDRanker", {"n_components": 2, "n_iter": 7, "random_state": 42}),
}


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _committed_run(root, model, scores, *, run_id=_FIRST, k=2, saved=False,
                   source_root=None, changed_input=None):
    """Small metadata artifacts matching the runner, with no source/model files.

    scores maps dataset basenames to (Recall@K, NDCG@K). Both users have the
    aggregate values in per_user.csv; deliberate scores make max != mean.
    """
    run = root / model / "runs" / run_id
    algorithm, class_name, parameters = _MODELS[model]
    invocation = _digest(model + run_id)[:32]
    source_root = source_root or root.parent / "removed sources"
    manifest = {
        "format_version": 1, "kind": "ranking_benchmark", "run_id": run_id,
        "invocation_id": invocation, "created_at": "2026-09-22T10:00:00+00:00",
        "model": {"name": model, "algorithm": algorithm,
                  "class": f"prak.ranking.{algorithm}.{class_name}", "parameters": parameters},
        "k": k, "save_model": saved, "versions": {"numpy": "2.1.0", "scikit_learn": "1.6.0"},
        "datasets": {},
    }
    rows = []
    for name, (recall, ndcg) in scores.items():
        inputs = {
            split: {"path": str(source_root / name / f"{split}.csv"), "rows": count,
                    "sha256": _digest(f"{name}:{split}" + ("changed" if split == changed_input else ""))}
            for split, count in (("train", 6), ("test", 4), ("catalog", 12))
        }
        manifest["datasets"][name] = {
            "path": str(source_root / name), "inputs": inputs, "n_users": 2,
            "user_ids_sha256": _digest('["001", "NA"]'), "fit_seconds": .125,
            "evaluation_seconds": .03125,
        }
        _write_json(run / "datasets" / name / "test" / "metrics.json", {
            "format_version": 1, "split": "test", "k": k, "n_users": 2,
            "n_catalog": 12, "n_events": 4,
            "metrics": {"recall_at_k": recall, "ndcg_at_k": ndcg},
            "inputs": {split: inputs[split] for split in ("test", "catalog")},
            "model": {"name": model, "parameters": parameters,
                      "path": str(run / "datasets" / name / "model" / "model.joblib") if saved else None,
                      "sha256": _digest(model + name) if saved else None},
        })
        (run / "datasets" / name / "test" / "per_user.csv").write_text(
            f"user_id,recall_at_k,ndcg_at_k\n001,{recall},{ndcg}\nNA,{recall},{ndcg}\n", encoding="utf-8",
        )
        rows.append({
            "model": model, "dataset": name, "k": k, "n_users": 2, "n_catalog": 12,
            "n_train_events": 6, "n_test_events": 4, "recall_at_k": recall,
            "ndcg_at_k": ndcg, "fit_seconds": .125, "evaluation_seconds": .03125,
        })
    _write_json(run / "manifest.json", manifest)
    _write_csv(run / "results.csv", rows)
    _write_json(root / ".completed" / f"{invocation}.json", {
        "format_version": 1, "kind": "ranking_benchmark_commit",
        "invocation_id": invocation, "runs": {model: run_id},
    })
    return run


class _Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images, self.tags, self.best_values = [], [], []
        self._best = False

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attrs = dict(attrs)
        if tag == "img":
            self.images.append(attrs)
        if tag == "td":
            self._best = attrs.get("class") == "best"

    def handle_endtag(self, tag):
        if tag == "td":
            self._best = False

    def handle_data(self, data):
        if self._best:
            self.best_values.append(data)


def test_full_matrix_max_ranking_ties_provenance_and_portable_html(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    dataset_a, dataset_b = "metric-a", 'метрика & <b>"'
    selections = [
        _committed_run(root, "svd_n-components-2", {dataset_a: (.6, .9), dataset_b: (.6, .6)}),
        _committed_run(root, "random_random-state-73", {dataset_a: (.4, .8), dataset_b: (.7, .8)},
                       source_root=tmp_path / "different source location"),
        _committed_run(root, "random", {dataset_a: (.6, .9), dataset_b: (.1, .1)}),
    ]
    monkeypatch.chdir(tmp_path)
    paths = report_ranking_benchmark("benchmark", "reports/comparison.html")
    assert paths == BenchmarkReportPaths(tmp_path / "reports/comparison.html", tmp_path / "reports/comparison.csv")
    with pytest.raises(FrozenInstanceError):
        paths.html_path = Path("other.html")
    rows = _read_csv(paths.csv_path)
    assert len(rows) == 6
    assert [(row["model"], row["dataset"]) for row in rows] == [
        (model, dataset) for model in ("random", "svd_n-components-2", "random_random-state-73")
        for dataset in sorted((dataset_a, dataset_b))
    ]
    # Mean would put random last; max and canonical tie-breaking put it first.
    assert [row["rank"] for row in rows] == ["1", "1", "2", "2", "3", "3"]
    assert [float(row["score"]) for row in rows] == [.9, .9, .9, .9, .8, .8]
    for row in rows:
        algorithm, _, parameters = _MODELS[row["model"]]
        run = root / row["model"] / "runs" / _FIRST
        dataset = _read_json(run / "manifest.json")["datasets"][row["dataset"]]
        assert row["algorithm"] == algorithm
        assert json.loads(row["parameters"]) == parameters
        assert row["run_id"] == _FIRST and row["run_path"] == str(run)
        assert row["k"] == "2" and row["n_users"] == "2"
        assert float(row["fit_seconds"]) == .125
        assert float(row["evaluation_seconds"]) == .03125
        for split, source in dataset["inputs"].items():
            for field in ("path", "rows", "sha256"):
                assert row[f"{split}_{field}"] == str(source[field])
    html = paths.html_path.read_text(encoding="utf-8")
    page = _Page()
    page.feed(html)
    assert '<html lang="ru">' in html
    assert "максимум NDCG@K по датасетам" in html
    assert "random_state" in html and "n_components" in html and "73" in html
    assert "mtime" in html and "created_at" in html and "0–1" in html
    assert "метрика &amp; &lt;b&gt;&quot;" in html
    assert "script" not in page.tags and "link" not in page.tags
    assert len(page.images) == 4  # Two heatmaps and one comparison per dataset.
    assert sorted(map(float, page.best_values)) == [.6, .6, .7, .8, .9, .9]
    for image in page.images:
        assert image["src"].startswith("data:image/png;base64,") and image["alt"]
        pixels = mpimg.imread(BytesIO(base64.b64decode(image["src"].split(",", 1)[1])), format="png")
        assert pixels.shape[0] > 300 and pixels.shape[1] > 700
    for run in selections:
        assert str(run) in html and _read_json(run / "manifest.json")["invocation_id"] in html
    before = (paths.html_path.read_bytes(), paths.csv_path.read_bytes())
    assert report_ranking_benchmark(root, paths.html_path) == paths
    assert (paths.html_path.read_bytes(), paths.csv_path.read_bytes()) == before


def test_latest_is_lexicographic_not_mtime_or_created_at(tmp_path):
    root = tmp_path / "benchmark"
    older = _committed_run(root, "random", {"metric-a": (.1, .1)}, run_id=_FIRST)
    latest = _committed_run(root, "random", {"metric-a": (.9, .9)}, run_id=_LAST)
    manifest = _read_json(older / "manifest.json")
    manifest["created_at"] = "2099-01-01T00:00:00+00:00"
    _write_json(older / "manifest.json", manifest)
    os.utime(older, (2_000_000_000, 2_000_000_000))
    os.utime(latest, (1, 1))
    # A newer, unpublished directory must not influence report selection.
    unfinished = root / "random" / "runs" / "20260922T120000000000Z"
    unfinished.mkdir()
    (unfinished / "manifest.json").write_text("incomplete", encoding="utf-8")
    rows = _read_csv(report_ranking_benchmark(root, tmp_path / "report.html").csv_path)
    assert len(rows) == 1 and rows[0]["run_id"] == _LAST and float(rows[0]["ndcg_at_k"]) == .9


@pytest.mark.parametrize("mismatch,message", [
    ("datasets", "same dataset"), ("k", "same K"),
    ("train", "SHA256"), ("test", "SHA256"), ("catalog", "SHA256"),
])
def test_latest_runs_must_be_comparable(tmp_path, mismatch, message):
    root = tmp_path / "benchmark"
    _committed_run(root, "random", {"metric-a": (.5, .5), "metric-b": (.6, .6)})
    _committed_run(root, "random_random-state-73", {"metric-a": (.5, .5), "metric-b": (.6, .6)})
    # The reader must not fall back to the older compatible run.
    scores = {"metric-a": (.5, .5)} if mismatch == "datasets" else {"metric-a": (.5, .5), "metric-b": (.6, .6)}
    _committed_run(root, "random_random-state-73", scores, run_id=_LAST,
                   k=3 if mismatch == "k" else 2,
                   changed_input=mismatch if mismatch in ("train", "test", "catalog") else None)
    output = tmp_path / "reports" / "report.html"
    with pytest.raises(ValueError, match=message):
        report_ranking_benchmark(root, output)
    assert not output.parent.exists()


@pytest.mark.parametrize("save_model", [False, True])
def test_actual_runner_report_needs_no_sources_models_or_training(tmp_path, ranking_snapshot, monkeypatch, save_model):
    second = tmp_path / "second snapshot"
    shutil.copytree(ranking_snapshot, second)
    root = tmp_path / "benchmark"
    runs = run_ranking_benchmark([ranking_snapshot, second], root,
                                 models=["random", "svd --n-components 2"], k=2, save_model=save_model)
    shutil.rmtree(ranking_snapshot)
    shutil.rmtree(second)
    for run in runs:
        for path in run.glob("datasets/*/model"):
            shutil.rmtree(path)

    def forbidden(*args, **kwargs):
        pytest.fail("A report must not load or fit models or read original data")

    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(RandomRanker, "fit", forbidden)
    monkeypatch.setattr(SVDRanker, "fit", forbidden)
    original_open = Path.open
    allowed = {root / ".completed" / f"{_read_json(run / 'manifest.json')['invocation_id']}.json" for run in runs}
    for run in runs:
        allowed.update((run / "manifest.json", run / "results.csv"))
        allowed.update(run.glob("datasets/*/test/metrics.json"))

    def only_metadata(path, mode="r", *args, **kwargs):
        if "r" in mode:
            assert path in allowed, f"Unexpected input read: {path}"
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", only_metadata)
        paths = report_ranking_benchmark(root, tmp_path / "report.html")
    assert len(_read_csv(paths.csv_path)) == 4
    assert all(row["save_model"] == str(save_model) for row in _read_csv(paths.csv_path))


@pytest.mark.parametrize("metric", ["recall_at_k", "ndcg_at_k"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.01, 1.01, True, "0.5", None])
def test_corrupted_metrics_fail_before_output(tmp_path, metric, value):
    root = tmp_path / "benchmark"
    run = _committed_run(root, "random", {"metric-a": (.5, .5)})
    path = run / "datasets/metric-a/test/metrics.json"
    report = _read_json(path)
    report["metrics"][metric] = value
    _write_json(path, report)
    with pytest.raises(ValueError, match=metric):
        report_ranking_benchmark(root, tmp_path / "report.html")
    assert not (tmp_path / "report.html").exists()
    assert not (tmp_path / "report.csv").exists()


@pytest.mark.parametrize("problem,message", [
    ("metric-disagreement", "differs"), ("negative-time", "finite"),
    ("infinite-time", "finite"), ("zero-users", "n_users"),
    ("incorrect-count", "counts"), ("evaluation-count", "count"),
    ("parameters", "parameters"), ("provenance", "provenance"),
    ("missing-row", "dataset set"), ("duplicate-row", "duplicate dataset"),
    ("csv-nan", "finite"),
])
def test_inconsistent_metadata_rejected_and_existing_outputs_preserved(tmp_path, problem, message):
    root = tmp_path / "benchmark"
    run = _committed_run(root, "random", {"metric-a": (.5, .5)})
    output = tmp_path / "report.html"
    output.write_text("previous HTML", encoding="utf-8")
    output.with_suffix(".csv").write_text("previous CSV", encoding="utf-8")
    manifest = _read_json(run / "manifest.json")
    metric_path = run / "datasets/metric-a/test/metrics.json"
    metrics = _read_json(metric_path)
    rows = _read_csv(run / "results.csv")
    if problem == "metric-disagreement":
        metrics["metrics"]["ndcg_at_k"] = .75
    elif problem in ("negative-time", "infinite-time"):
        manifest["datasets"]["metric-a"]["fit_seconds"] = -1 if problem == "negative-time" else float("inf")
    elif problem == "zero-users":
        manifest["datasets"]["metric-a"]["n_users"] = 0
    elif problem == "incorrect-count":
        rows[0]["n_train_events"] = 9
    elif problem == "evaluation-count":
        metrics["n_users"] = 1
    elif problem == "parameters":
        metrics["model"]["parameters"]["random_state"] = 73
    elif problem == "provenance":
        metrics["inputs"]["test"]["sha256"] = "0" * 64
    elif problem == "missing-row":
        rows = []
    elif problem == "duplicate-row":
        rows.append(dict(rows[0]))
    else:
        rows[0]["recall_at_k"] = "nan"
    _write_json(run / "manifest.json", manifest)
    _write_json(metric_path, metrics)
    _write_csv(run / "results.csv", rows)
    with pytest.raises(ValueError, match=message):
        report_ranking_benchmark(root, output)
    assert output.read_text(encoding="utf-8") == "previous HTML"
    assert output.with_suffix(".csv").read_text(encoding="utf-8") == "previous CSV"


def test_empty_root_has_no_report(tmp_path):
    root = tmp_path / "benchmark"
    root.mkdir()
    with pytest.raises(ValueError, match="[Nn]o completed"):
        report_ranking_benchmark(root, tmp_path / "report.html")
    assert not (tmp_path / "report.html").exists()


def test_perfect_ranking_roundoff_from_real_evaluator_is_reportable(tmp_path):
    dataset = tmp_path / "perfect"
    dataset.mkdir()
    products = [f"p{index:02}" for index in range(32)]
    pd.DataFrame({"product_id": products}).to_csv(dataset / "catalog.csv", index=False)
    for split in ("train", "test"):
        pd.DataFrame([
            {"event_id": f"{split}-{index}", "user_id": "user", "event_index": index,
             "split": split, "product_id": product}
            for index, product in enumerate(products)
        ]).to_csv(dataset / f"{split}.csv", index=False)
    root = tmp_path / "benchmark"
    run = run_ranking_benchmark([dataset], root, models=["random"], k=32, save_model=False)[0]
    recorded = _read_json(run / "datasets/perfect/test/metrics.json")["metrics"]["ndcg_at_k"]
    assert recorded == pytest.approx(1)
    paths = report_ranking_benchmark(root, tmp_path / "report.html")
    assert float(_read_csv(paths.csv_path)[0]["ndcg_at_k"]) == recorded
    assert paths.html_path.is_file()


@pytest.mark.parametrize("kind", ["run", "csv-collision", "unselected-run", "commit-dir", "symlink", "hardlink", "source", "parent-symlink"])
def test_output_cannot_replace_benchmark_artifacts_or_sources(tmp_path, kind):
    root = tmp_path / "benchmark"
    run = _committed_run(root, "random", {"metric-a": (.5, .5)})
    before = (run / "results.csv").read_bytes()
    output = tmp_path / "report.html"
    if kind == "run":
        output = run / "report.html"
    elif kind == "csv-collision":
        output = run / "results.html"
    elif kind == "unselected-run":
        output = root / "another-model/runs/unfinished/report.html"
    elif kind == "commit-dir":
        output = root / ".completed/report.html"
    elif kind == "symlink":
        output.with_suffix(".csv").symlink_to(run / "results.csv")
    elif kind == "hardlink":
        output.with_suffix(".csv").hardlink_to(run / "results.csv")
    elif kind == "source":
        output = tmp_path / "removed sources/metric-a/train.html"
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(run, target_is_directory=True)
        output = alias / "report.html"
    with pytest.raises(ValueError, match="output"):
        report_ranking_benchmark(root, output)
    assert (run / "results.csv").read_bytes() == before
    assert not output.exists()
