import csv
from collections.abc import Callable
from datetime import datetime, timezone, tzinfo
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Self, override

import joblib
import numpy as np
import pandas as pd
import pytest
import fixture_types as ft

from buy_today.ranking import (
    RandomRanker,
    SVDRanker,
    parse_model_spec,
    read_latest_benchmark_runs,
    read_ranking_data,
    report_ranking,
    run_ranking_benchmark,
)
from buy_today.ranking import benchmark
from buy_today.ranking.evaluation import RankingEvaluation
from buy_today.ranking.domain import Predictor


type Ranker = RandomRanker | SVDRanker


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


@pytest.mark.parametrize(
    "spec,name,parameters",
    [
        ("random", "random", {"random_state": 42}),
        ("random --random-state 42", "random", {"random_state": 42}),
        ('random --random-state "73"', "random_random-state-73", {"random_state": 73}),
        ("svd", "svd", {"n_components": 32, "n_iter": 7, "random_state": 42}),
        (
            "svd --n-components 32 --n-iter 7",
            "svd",
            {"n_components": 32, "n_iter": 7, "random_state": 42},
        ),
        (
            "svd --random-state 73 --n-iter 5 --n-components 16",
            "svd_n-components-16_n-iter-5_random-state-73",
            {"n_components": 16, "n_iter": 5, "random_state": 73},
        ),
        (
            "svd --n-components=16 --n-iter=5 --random-state=73",
            "svd_n-components-16_n-iter-5_random-state-73",
            {"n_components": 16, "n_iter": 5, "random_state": 73},
        ),
    ],
)
def test_model_spec_canonical_identity(spec: str, name: str, parameters: dict[str, int]):
    config = parse_model_spec(spec)
    assert config.name == name
    assert config.parameters == parameters
    assert config.create_ranker().get_params() == parameters


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "unknown",
        "random --n-components 2",
        "svd --wat 2",
        "svd --n-components 0",
        "svd --n-iter -1",
        "random --random-state -1",
        "random --random-state 4294967296",
        "random --random-state 2.5",
        "svd --n-components",
        'svd "',
        "random --help",
        "random --random 2",
    ],
)
def test_bad_specs_rejected_before_any_training(
    ranking_snapshot: Path, tmp_path: Path, monkeypatch: ft.MonkeyPatch, spec: str
):
    monkeypatch.setattr(RandomRanker, "fit", _forbidden_fit)
    with pytest.raises(ValueError):
        run_ranking_benchmark([ranking_snapshot], tmp_path / "out", models=["random", spec])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "problem",
    [
        "models",
        "datasets",
        "no-models",
        "no-datasets",
        "missing-directory",
        "missing-test",
        "bad-k",
        "k-exceeds-catalog",
        "svd-dimensions",
        "users",
        "output-inside",
        "output-ancestor",
    ],
)
def test_preflight_rejects_invalid_matrix(
    ranking_snapshot: Path, tmp_path: Path, monkeypatch: ft.MonkeyPatch, problem: str
):
    models = {
        "models": ["random", "random --random-state 42"],
        "no-models": [],
        "svd-dimensions": ["random", "svd"],
    }.get(problem, ["random"])
    datasets = _invalid_datasets(problem, ranking_snapshot, tmp_path)
    output = {"output-inside": ranking_snapshot / "out", "output-ancestor": tmp_path}.get(
        problem, tmp_path / "out"
    )
    k = {"bad-k": True, "k-exceeds-catalog": 13}.get(problem, 10)
    monkeypatch.setattr(RandomRanker, "fit", _forbidden_fit)
    with pytest.raises(ValueError):
        run_ranking_benchmark(datasets, output, models=models, k=k)
    assert not list(tmp_path.rglob("model.joblib"))
    assert not list(tmp_path.rglob(".completed/*.json"))


def _invalid_datasets(problem: str, ranking_snapshot: Path, tmp_path: Path) -> list[Path]:
    datasets = [ranking_snapshot]
    if problem == "datasets":
        second = tmp_path / "another" / ranking_snapshot.name
        shutil.copytree(ranking_snapshot, second)
        datasets += [second]
    elif problem == "no-datasets":
        datasets = []
    elif problem == "missing-directory":
        datasets += [tmp_path / "missing"]
    elif problem == "missing-test":
        (ranking_snapshot / "test.csv").unlink()
    elif problem == "users":
        path = ranking_snapshot / "test.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("NA", "other"), encoding="utf-8")
    return datasets


def _forbidden_fit(*args: object, **kwargs: object) -> None:
    pytest.fail("must validate before fit")


def test_entire_matrix_one_fresh_fit_train_only_test_only(
    ranking_snapshot: Path, tmp_path: Path, monkeypatch: ft.MonkeyPatch
):
    second = tmp_path / "second"
    shutil.copytree(ranking_snapshot, second)
    path = second / "train.csv"
    path.write_text(path.read_text().replace("p00", "p04"))
    datasets = [ranking_snapshot, second]
    expected = [read_ranking_data(directory) for directory in datasets]
    before = {path: path.read_bytes() for directory in datasets for path in directory.iterdir()}
    fitted: list[Ranker] = []
    evaluated: list[Predictor[object]] = []

    def spy_fit[Model: Ranker](
        original: Callable[[Model, ft.RankingData, object], Model],
    ) -> Callable[[Model, ft.RankingData, object], Model]:
        def fit(model: Model, data: ft.RankingData, y: object = None) -> Model:
            pd.testing.assert_frame_equal(data.interactions, expected[len(fitted) % 2].interactions)
            assert all(model is not previous for previous in fitted)
            fitted.append(model)
            return original(model, data, y)

        return fit

    monkeypatch.setattr(RandomRanker, "fit", spy_fit(RandomRanker.fit))
    monkeypatch.setattr(SVDRanker, "fit", spy_fit(SVDRanker.fit))
    original_evaluate = benchmark.evaluate_ranker

    def evaluate(model: Predictor[object], data: ft.RankingData, *, k: int) -> RankingEvaluation:
        assert model is fitted[-1]
        pd.testing.assert_frame_equal(
            data.interactions,
            read_ranking_data(datasets[len(evaluated) % 2], split="test").interactions,
        )
        evaluated.append(model)
        return original_evaluate(model, data, k=k)

    monkeypatch.setattr(benchmark, "evaluate_ranker", evaluate)
    original_reader = benchmark.read_ranking_data

    def reader(directory: Path | str, *, split: str = "train") -> ft.RankingData:
        assert split in {"train", "test"}
        return original_reader(directory, split=split)

    monkeypatch.setattr(benchmark, "read_ranking_data", reader)
    root = tmp_path / "out"
    runs = run_ranking_benchmark(datasets, root, models=["random", "svd --n-components 2"], k=3)
    assert len(fitted) == len(evaluated) == 4
    assert [type(model) for model in fitted] == [RandomRanker, RandomRanker, SVDRanker, SVDRanker]
    assert {path: path.read_bytes() for path in before} == before
    for run, model in zip(runs, ("random", "svd_n-components-2"), strict=True):
        _assert_run(run, model, datasets, before, fitted, tmp_path)


def _assert_run(
    run: Path,
    model: str,
    datasets: list[Path],
    before: dict[Path, bytes],
    fitted: list[Ranker],
    tmp_path: Path,
) -> None:
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
            assert (
                source["sha256"] == hashlib.sha256(before[directory / f"{split}.csv"]).hexdigest()
            )
        assert metadata["fit_seconds"] >= 0 and metadata["evaluation_seconds"] >= 0
        _assert_pair(
            run / "datasets" / directory.name,
            row,
            directory,
            fitted[index + (2 if model != "random" else 0)],
            tmp_path / f"check-{model}-{index}",
        )


def _assert_pair(
    saved: Path, row: dict[str, str], directory: Path, fitted: Ranker, output: Path
) -> None:
    metrics = read_json(saved / "test" / "metrics.json")
    model_path = saved / "model" / "model.joblib"
    assert metrics["model"]["path"] == str(model_path)
    assert ".benchmark-" not in (saved / "test" / "metrics.json").read_text(encoding="utf-8")
    assert metrics["split"] == "test" and metrics["k"] == 3
    loaded = joblib.load(model_path)
    np.testing.assert_array_equal(loaded.predict("001", 3), fitted.predict("001", 3))
    per_user = pd.read_csv(saved / "test" / "per_user.csv", keep_default_na=False)
    assert list(per_user.user_id) == ["001", "NA"]
    for metric in ("recall_at_k", "ndcg_at_k"):
        assert metrics["metrics"][metric] == float(row[metric])
        assert per_user[metric].mean() == pytest.approx(float(row[metric]))
    regular = report_ranking(directory, saved / "model", output, split="test", k=3)
    assert read_json(regular.metrics_path)["metrics"] == metrics["metrics"]


def test_no_save_model_never_serializes(
    ranking_snapshot: Path, tmp_path: Path, monkeypatch: ft.MonkeyPatch
):
    def forbidden_dump(*args: object, **kwargs: object) -> None:
        pytest.fail("no-save-model must not serialize")

    monkeypatch.setattr(joblib, "dump", forbidden_dump)
    runs = run_ranking_benchmark(
        [ranking_snapshot], tmp_path / "out", models=["random"], save_model=False
    )
    run = runs[0]
    assert not list(run.rglob("*.joblib"))
    assert not (run / "datasets" / ranking_snapshot.name / "model").exists()
    metrics = read_json(run / "datasets" / ranking_snapshot.name / "test" / "metrics.json")
    assert metrics["model"]["path"] is None and metrics["model"]["sha256"] is None
    assert metrics["model"]["parameters"] == {"random_state": 42}
    assert read_csv(run / "results.csv")


def test_dataset_name_uses_supplied_folder_basename(ranking_snapshot: Path, tmp_path: Path):
    directory = tmp_path / "metric-a"
    directory.symlink_to(ranking_snapshot, target_is_directory=True)
    run = run_ranking_benchmark([directory], tmp_path / "out", models=["random"])[0]
    manifest = read_json(run / "manifest.json")
    assert list(manifest["datasets"]) == ["metric-a"]
    assert manifest["datasets"]["metric-a"]["path"] == str(directory)
    assert (run / "datasets/metric-a/test/per_user.csv").is_file()


def test_repeated_timestamp_creates_new_sorted_run_without_overwrite(
    ranking_snapshot: Path, tmp_path: Path, monkeypatch: ft.MonkeyPatch
):
    class FrozenDateTime(datetime):
        @classmethod
        @override
        def now(cls, tz: tzinfo | None = None) -> Self:
            return cls(2026, 9, 22, tzinfo=timezone.utc)

    monkeypatch.setattr(benchmark, "datetime", FrozenDateTime)
    root = tmp_path / "out"
    first = run_ranking_benchmark([ranking_snapshot], root, models=["random"])[0]
    before = {
        path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()
    }
    second = run_ranking_benchmark([ranking_snapshot], root, models=["random"])[0]
    assert second.name > first.name
    assert {
        path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()
    } == before
    assert [path for path, _ in read_latest_benchmark_runs(root)] == [second]


def test_later_fit_failure_publishes_none_and_preserves_previous(
    ranking_snapshot: Path, tmp_path: Path, monkeypatch: ft.MonkeyPatch
):
    root = tmp_path / "out"
    old = run_ranking_benchmark([ranking_snapshot], root, models=["random"])[0]
    original_fit = RandomRanker.fit
    calls: list[object] = []

    def fit(model: RandomRanker, data: ft.RankingData, y: object = None) -> RandomRanker:
        calls.append(model.random_state)
        return original_fit(model, data, y)

    def failure(*args: object, **kwargs: object) -> None:
        raise RuntimeError("deliberate later fit failure")

    monkeypatch.setattr(RandomRanker, "fit", fit)
    monkeypatch.setattr(SVDRanker, "fit", failure)
    with pytest.raises(RuntimeError, match="later fit"):
        run_ranking_benchmark(
            [ranking_snapshot],
            root,
            models=["random", "svd --n-components 2", "random --random-state 73"],
        )
    assert calls == [42]  # Stop at the first failure, do not fit the third model.
    assert [path for path, _ in read_latest_benchmark_runs(root)] == [old]
    assert list(root.glob("*/runs/*")) == [old]
    assert not list(root.glob(".benchmark-*"))


@pytest.mark.parametrize("fail_at", ["second-run", "commit", None])
def test_invocation_becomes_visible_only_after_shared_commit(
    ranking_snapshot: Path, tmp_path: Path, monkeypatch: ft.MonkeyPatch, fail_at: str | None
):
    root = tmp_path / "out"
    old = run_ranking_benchmark([ranking_snapshot], root, models=["random"])[0]
    original_rename = Path.rename
    observed: list[list[Path]] = []

    def rename(source: Path, destination: Path) -> Path:
        if destination.parent.name == "runs" or source.name == "commit.json":
            visible = [path for path, _ in read_latest_benchmark_runs(root)]
            observed.append(visible)
            assert visible == [old]
            if (
                fail_at == "second-run" and destination.parent.parent.name == "svd_n-components-2"
            ) or (fail_at == "commit" and source.name == "commit.json"):
                raise OSError("deliberate publish failure")
        return original_rename(source, destination)

    monkeypatch.setattr(Path, "rename", rename)
    if fail_at:
        with pytest.raises(OSError, match="publish failure"):
            run_ranking_benchmark(
                [ranking_snapshot], root, models=["random", "svd --n-components 2"]
            )
        assert list(root.glob("*/runs/*")) == [old]
    else:
        new = run_ranking_benchmark(
            [ranking_snapshot], root, models=["random", "svd --n-components 2"]
        )
        assert [path for path, _ in read_latest_benchmark_runs(root)] == new
    assert len(observed) == (2 if fail_at == "second-run" else 3)


def test_input_mutation_during_training_prevents_publication(
    ranking_snapshot: Path, tmp_path: Path, monkeypatch: ft.MonkeyPatch
):
    original_fit = RandomRanker.fit

    def mutate(model: RandomRanker, data: ft.RankingData, y: object = None) -> RandomRanker:
        path = ranking_snapshot / "test.csv"
        path.write_text(path.read_text().replace("p03", "p04"))
        return original_fit(model, data, y)

    monkeypatch.setattr(RandomRanker, "fit", mutate)
    root = tmp_path / "out"
    with pytest.raises(ValueError, match="input changed"):
        run_ranking_benchmark([ranking_snapshot], root, models=["random"])
    assert not read_latest_benchmark_runs(root)
    assert not list(root.glob("*/runs/*"))
