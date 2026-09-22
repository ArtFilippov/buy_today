import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from buy_today.ranking import (
    RandomRanker, SVDRanker, parse_model_spec, read_latest_benchmark_runs,
    read_ranking_data, report_ranking, run_ranking_benchmark,
)
from buy_today.ranking import benchmark


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


@pytest.mark.parametrize("spec,name,parameters", [
    ("random", "random", {"random_state": 42}),
    ("random --random-state 42", "random", {"random_state": 42}),
    ('random --random-state "73"', "random_random-state-73", {"random_state": 73}),
    ("svd", "svd", {"n_components": 32, "n_iter": 7, "random_state": 42}),
    ("svd --n-components 32 --n-iter 7", "svd", {"n_components": 32, "n_iter": 7, "random_state": 42}),
    ("svd --random-state 73 --n-iter 5 --n-components 16", "svd_n-components-16_n-iter-5_random-state-73",
     {"n_components": 16, "n_iter": 5, "random_state": 73}),
    ("svd --n-components=16 --n-iter=5 --random-state=73", "svd_n-components-16_n-iter-5_random-state-73",
     {"n_components": 16, "n_iter": 5, "random_state": 73}),
])
def test_model_spec_canonical_identity(spec, name, parameters):
    config = parse_model_spec(spec)
    assert config.name == name
    assert config.parameters == parameters
    assert config.create_ranker().get_params() == parameters


@pytest.mark.parametrize("spec", [
    "", "unknown", "random --n-components 2", "svd --wat 2", "svd --n-components 0",
    "svd --n-iter -1", "random --random-state -1", "random --random-state 4294967296",
    "random --random-state 2.5", "svd --n-components", 'svd "', "random --help",
    "random --random 2",
])
def test_bad_specs_rejected_before_any_training(ranking_snapshot, tmp_path, monkeypatch, spec):
    monkeypatch.setattr(RandomRanker, "fit", lambda *a: pytest.fail("must validate before fit"))
    with pytest.raises(ValueError):
        run_ranking_benchmark([ranking_snapshot], tmp_path / "out", models=["random", spec])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("problem", [
    "models", "datasets", "no-models", "no-datasets", "missing-directory", "missing-test",
    "bad-k", "k-exceeds-catalog", "svd-dimensions", "users", "output-inside", "output-ancestor",
])
def test_preflight_rejects_invalid_matrix(ranking_snapshot, tmp_path, monkeypatch, problem):
    models, datasets, output, k = ["random"], [ranking_snapshot], tmp_path / "out", 10
    if problem == "models":
        models += ["random --random-state 42"]
    elif problem == "datasets":
        second = tmp_path / "another" / ranking_snapshot.name
        shutil.copytree(ranking_snapshot, second)
        datasets += [second]
    elif problem == "no-models":
        models = []
    elif problem == "no-datasets":
        datasets = []
    elif problem == "missing-directory":
        datasets += [tmp_path / "missing"]
    elif problem == "missing-test":
        (ranking_snapshot / "test.csv").unlink()
    elif problem == "bad-k":
        k = True
    elif problem == "k-exceeds-catalog":
        k = 13
    elif problem == "svd-dimensions":
        models += ["svd"]
    elif problem == "users":
        path = ranking_snapshot / "test.csv"
        path.write_text(path.read_text().replace("NA", "other"))
    elif problem == "output-inside":
        output = ranking_snapshot / "out"
    else:
        output = tmp_path
    monkeypatch.setattr(RandomRanker, "fit", lambda *a: pytest.fail("must validate before fit"))
    with pytest.raises(ValueError):
        run_ranking_benchmark(datasets, output, models=models, k=k)
    assert not list(tmp_path.rglob("model.joblib"))
    assert not list(tmp_path.rglob(".completed/*.json"))


def test_entire_matrix_one_fresh_fit_train_only_test_only(ranking_snapshot, tmp_path, monkeypatch):
    second = tmp_path / "second"
    shutil.copytree(ranking_snapshot, second)
    path = second / "train.csv"
    path.write_text(path.read_text().replace("p00", "p04"))
    datasets = [ranking_snapshot, second]
    expected = [read_ranking_data(directory) for directory in datasets]
    before = {path: path.read_bytes() for directory in datasets for path in directory.iterdir()}
    fitted, evaluated = [], []

    def spy_fit(original):
        def fit(model, data, y=None):
            pd.testing.assert_frame_equal(data.interactions, expected[len(fitted) % 2].interactions)
            assert all(model is not previous for previous in fitted)
            fitted.append(model)
            return original(model, data, y)
        return fit

    monkeypatch.setattr(RandomRanker, "fit", spy_fit(RandomRanker.fit))
    monkeypatch.setattr(SVDRanker, "fit", spy_fit(SVDRanker.fit))
    original_evaluate = benchmark.evaluate_ranker

    def evaluate(model, data, *, k):
        assert model is fitted[-1]
        pd.testing.assert_frame_equal(
            data.interactions, read_ranking_data(datasets[len(evaluated) % 2], split="test").interactions,
        )
        evaluated.append(model)
        return original_evaluate(model, data, k=k)

    monkeypatch.setattr(benchmark, "evaluate_ranker", evaluate)
    original_reader = benchmark.read_ranking_data

    def reader(directory, *, split="train"):
        assert split in ("train", "test")
        return original_reader(directory, split=split)

    monkeypatch.setattr(benchmark, "read_ranking_data", reader)
    root = tmp_path / "out"
    runs = run_ranking_benchmark(datasets, root, models=["random", "svd --n-components 2"], k=3)
    assert len(fitted) == len(evaluated) == 4
    assert [type(model) for model in fitted] == [RandomRanker, RandomRanker, SVDRanker, SVDRanker]
    assert {path: path.read_bytes() for path in before} == before
    for run, model in zip(runs, ("random", "svd_n-components-2"), strict=True):
        manifest = read_json(run / "manifest.json")
        assert manifest["model"]["name"] == model
        assert manifest["model"]["parameters"]["random_state"] == 42
        assert manifest["k"] == 3 and manifest["save_model"]
        rows = read_csv(run / "results.csv")
        assert [row["dataset"] for row in rows] == [directory.name for directory in datasets]
        for index, (directory, row) in enumerate(zip(datasets, rows, strict=True)):
            metadata = manifest["datasets"][directory.name]
            for split, source in metadata["inputs"].items():
                assert source["path"] == str(directory / f"{split}.csv")
                assert source["sha256"] == hashlib.sha256(before[directory / f"{split}.csv"]).hexdigest()
            assert metadata["fit_seconds"] >= 0 and metadata["evaluation_seconds"] >= 0
            saved = run / "datasets" / directory.name
            metrics = read_json(saved / "test" / "metrics.json")
            model_path = saved / "model" / "model.joblib"
            assert metrics["model"]["path"] == str(model_path)
            assert ".benchmark-" not in (saved / "test" / "metrics.json").read_text()
            assert metrics["split"] == "test" and metrics["k"] == 3
            loaded = joblib.load(model_path)
            np.testing.assert_array_equal(loaded.predict("001", 3), fitted[index + (2 if model != "random" else 0)].predict("001", 3))
            per_user = pd.read_csv(saved / "test" / "per_user.csv", keep_default_na=False)
            assert list(per_user.user_id) == ["001", "NA"]
            for metric in ("recall_at_k", "ndcg_at_k"):
                assert metrics["metrics"][metric] == float(row[metric])
                assert per_user[metric].mean() == pytest.approx(float(row[metric]))
            # The standard storage API can evaluate benchmark-saved models.
            regular = report_ranking(directory, saved / "model", tmp_path / f"check-{model}-{index}", split="test", k=3)
            assert read_json(regular.metrics_path)["metrics"] == metrics["metrics"]


def test_no_save_model_never_serializes(ranking_snapshot, tmp_path, monkeypatch):
    monkeypatch.setattr(joblib, "dump", lambda *a: pytest.fail("no-save-model must not serialize"))
    runs = run_ranking_benchmark([ranking_snapshot], tmp_path / "out", models=["random"], save_model=False)
    run = runs[0]
    assert not list(run.rglob("*.joblib"))
    assert not (run / "datasets" / ranking_snapshot.name / "model").exists()
    metrics = read_json(run / "datasets" / ranking_snapshot.name / "test" / "metrics.json")
    assert metrics["model"]["path"] is None and metrics["model"]["sha256"] is None
    assert metrics["model"]["parameters"] == {"random_state": 42}
    assert read_csv(run / "results.csv")


def test_dataset_name_uses_supplied_folder_basename(ranking_snapshot, tmp_path):
    directory = tmp_path / "metric-a"
    directory.symlink_to(ranking_snapshot, target_is_directory=True)
    run = run_ranking_benchmark([directory], tmp_path / "out", models=["random"])[0]
    manifest = read_json(run / "manifest.json")
    assert list(manifest["datasets"]) == ["metric-a"]
    assert manifest["datasets"]["metric-a"]["path"] == str(directory)
    assert (run / "datasets/metric-a/test/per_user.csv").is_file()


def test_repeated_timestamp_creates_new_sorted_run_without_overwrite(ranking_snapshot, tmp_path, monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 22, tzinfo=timezone.utc)

    monkeypatch.setattr(benchmark, "datetime", FrozenDateTime)
    root = tmp_path / "out"
    first = run_ranking_benchmark([ranking_snapshot], root, models=["random"])[0]
    before = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    second = run_ranking_benchmark([ranking_snapshot], root, models=["random"])[0]
    assert second.name > first.name
    assert {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()} == before
    assert [path for path, _ in read_latest_benchmark_runs(root)] == [second]


def test_later_fit_failure_publishes_none_and_preserves_previous(ranking_snapshot, tmp_path, monkeypatch):
    root = tmp_path / "out"
    old = run_ranking_benchmark([ranking_snapshot], root, models=["random"])[0]
    original_fit = RandomRanker.fit
    calls = []

    def fit(model, data, y=None):
        calls.append(model.random_state)
        return original_fit(model, data, y)

    def failure(*args, **kwargs):
        raise RuntimeError("deliberate later fit failure")

    monkeypatch.setattr(RandomRanker, "fit", fit)
    monkeypatch.setattr(SVDRanker, "fit", failure)
    with pytest.raises(RuntimeError, match="later fit"):
        run_ranking_benchmark([ranking_snapshot], root,
                              models=["random", "svd --n-components 2", "random --random-state 73"])
    assert calls == [42]  # Stop at the first failure, do not fit the third model.
    assert [path for path, _ in read_latest_benchmark_runs(root)] == [old]
    assert list(root.glob("*/runs/*")) == [old]
    assert not list(root.glob(".benchmark-*"))


@pytest.mark.parametrize("fail_at", ["second-run", "commit", None])
def test_invocation_becomes_visible_only_after_shared_commit(ranking_snapshot, tmp_path, monkeypatch, fail_at):
    root = tmp_path / "out"
    old = run_ranking_benchmark([ranking_snapshot], root, models=["random"])[0]
    original_rename = Path.rename
    observed = []

    def rename(source, destination):
        if destination.parent.name == "runs" or source.name == "commit.json":
            visible = [path for path, _ in read_latest_benchmark_runs(root)]
            observed.append(visible)
            assert visible == [old]
            if ((fail_at == "second-run" and destination.parent.parent.name == "svd_n-components-2")
                    or (fail_at == "commit" and source.name == "commit.json")):
                raise OSError("deliberate publish failure")
        return original_rename(source, destination)

    monkeypatch.setattr(Path, "rename", rename)
    if fail_at:
        with pytest.raises(OSError, match="publish failure"):
            run_ranking_benchmark([ranking_snapshot], root, models=["random", "svd --n-components 2"])
        assert list(root.glob("*/runs/*")) == [old]
    else:
        new = run_ranking_benchmark([ranking_snapshot], root, models=["random", "svd --n-components 2"])
        assert [path for path, _ in read_latest_benchmark_runs(root)] == new
    assert len(observed) == (2 if fail_at == "second-run" else 3)


def test_input_mutation_during_training_prevents_publication(ranking_snapshot, tmp_path, monkeypatch):
    original_fit = RandomRanker.fit

    def mutate(model, data, y=None):
        path = ranking_snapshot / "test.csv"
        path.write_text(path.read_text().replace("p03", "p04"))
        return original_fit(model, data, y)

    monkeypatch.setattr(RandomRanker, "fit", mutate)
    root = tmp_path / "out"
    with pytest.raises(ValueError, match="input changed"):
        run_ranking_benchmark([ranking_snapshot], root, models=["random"])
    assert read_latest_benchmark_runs(root) == []
    assert not list(root.glob("*/runs/*"))
