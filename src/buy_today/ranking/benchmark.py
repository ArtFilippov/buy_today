"""Fresh train/test experiments, published as complete multi-model invocations."""

from collections.abc import Iterable
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn

from buy_today.ranking.data import RankingData, check_integer, read_ranking_data
from buy_today.ranking.domain import Dataset, DatasetRun, Estimator, Timings
from buy_today.ranking.evaluation import RankingEvaluation, evaluate_ranker
from buy_today.ranking.model_spec import ModelConfiguration, SVD, parse_model_spec
from buy_today.ranking.publication import (
    COMMIT_KIND,
    RUN_ID,
    RUN_KIND,
    TIMESTAMP,
    Metadata,
    read_latest_benchmark_runs,
)
from buy_today.ranking.snapshots import (
    digest,
    source_record,
    users_digest,
    write_fitted_ranker,
    write_json,
)

_MIN_PRODUCTS = 2
__all__ = [
    "ModelConfiguration",
    "parse_model_spec",
    "read_latest_benchmark_runs",
    "run_ranking_benchmark",
]

_COLUMNS: tuple[str, ...] = (
    "model",
    "dataset",
    "k",
    "n_users",
    "n_catalog",
    "n_train_events",
    "n_test_events",
    "recall_at_k",
    "ndcg_at_k",
    "fit_seconds",
    "evaluation_seconds",
)


@dataclass(frozen=True)
class _Invocation:
    run_id: str
    invocation_id: str
    k: int
    save_model: bool


@dataclass(frozen=True)
class _Outcome:
    model: Estimator[RankingData[pd.DataFrame], object]
    train: RankingData[pd.DataFrame]
    test: RankingData[pd.DataFrame]
    evaluation: RankingEvaluation[pd.DataFrame]


def _check_sources(dataset: Dataset) -> None:
    for source in dataset["inputs"].values():
        if digest(source["path"]) != source["sha256"]:
            raise ValueError(f"Benchmark input changed: {source['path']}")


def _load_pair(
    directory: Path | str, k: int
) -> tuple[RankingData[pd.DataFrame], RankingData[pd.DataFrame]]:
    train = read_ranking_data(directory, split="train")
    test = read_ranking_data(directory, split="test")
    _ = check_integer(k, "k", 1, len(train.catalog))
    if not train.catalog.equals(test.catalog):
        raise ValueError("Evaluation catalog differs from training catalog")
    if users_digest(train.interactions) != users_digest(test.interactions):
        raise ValueError("Evaluation users differ from training users")
    return train, test


def _check_directory(directory: Path, output: Path) -> None:
    if not directory.is_dir():
        raise ValueError(f"Dataset directory does not exist: {directory}")
    resolved = directory.resolve()
    if output.is_relative_to(resolved) or output in resolved.parents or output == resolved:
        raise ValueError("Benchmark output must be outside and separate from dataset snapshots")
    for split in ("train", "test", "catalog"):
        if not (directory / f"{split}.csv").is_file():
            raise ValueError(f"Missing benchmark input: {directory / f'{split}.csv'}")


def _check_dimensions(
    configurations: list[ModelConfiguration], data: RankingData[pd.DataFrame]
) -> None:
    for configuration in configurations:
        if configuration.algorithm == SVD:
            if len(data.catalog) < _MIN_PRODUCTS:
                raise ValueError("SVD requires at least two products in catalog")
            _ = check_integer(
                configuration.parameters["n_components"],
                "n_components",
                1,
                min(len(set(data.interactions["user_id"])), len(data.catalog)),
            )


def _dataset(directory: Path, configurations: list[ModelConfiguration], k: int) -> Dataset:
    before = {split: digest(directory / f"{split}.csv") for split in ("train", "test", "catalog")}
    train, test = _load_pair(directory, k)
    _check_dimensions(configurations, train)
    inputs = {
        split: source_record(directory / f"{split}.csv", rows)
        for split, rows in (
            ("train", len(train.interactions)),
            ("test", len(test.interactions)),
            ("catalog", len(train.catalog)),
        )
    }
    if any(inputs[split]["sha256"] != checksum for split, checksum in before.items()):
        raise ValueError(f"Benchmark input changed while reading: {directory}")
    return {
        "path": str(directory),
        "inputs": inputs,
        "n_users": len(set(train.interactions["user_id"])),
        "user_ids_sha256": users_digest(train.interactions),
    }


def _preflight(
    dataset_dirs: Iterable[Path | str],
    output: Path,
    configurations: list[ModelConfiguration],
    k: int,
) -> dict[str, Dataset]:
    # Preserve the supplied basename, including a directory symlink.
    if not (directories := [Path(os.path.abspath(directory)) for directory in dataset_dirs]):
        raise ValueError("At least one dataset is required")
    if len({path.name for path in directories}) != len(directories):
        raise ValueError("Duplicate dataset basename")
    for directory in directories:
        _check_directory(directory, output)
    return {directory.name: _dataset(directory, configurations, k) for directory in directories}


def _new_run_id(output: Path, configurations: list[ModelConfiguration]) -> str:
    # A microsecond increment handles collisions and clocks moving backwards.
    now = datetime.now(timezone.utc)
    for configuration in configurations:
        runs = output / configuration.name / "runs"
        if runs.exists():
            for path in runs.iterdir():
                if RUN_ID.fullmatch(path.name):
                    previous = datetime.strptime(path.name, TIMESTAMP).replace(tzinfo=timezone.utc)
                    now = max(now, previous + timedelta(microseconds=1))
    return datetime.strftime(now, TIMESTAMP)


def _metrics(
    data: RankingData[pd.DataFrame], result: RankingEvaluation[pd.DataFrame], k: int
) -> Metadata:
    return {
        "format_version": 1,
        "split": "test",
        "k": k,
        "n_users": len(result.per_user),
        "n_catalog": len(data.catalog),
        "n_events": len(data.interactions),
        "metrics": result.metrics,
    }


def _run_dataset(
    paths: tuple[Path, Path],
    configuration: ModelConfiguration,
    dataset: Dataset,
    invocation: _Invocation,
) -> tuple[Metadata, DatasetRun]:
    _check_sources(dataset)
    train, test = _load_pair(dataset["path"], invocation.k)
    _check_sources(dataset)
    model: Estimator[RankingData[pd.DataFrame], object] = configuration.create_ranker()
    started = perf_counter()
    _ = model.fit(train)
    timings: Timings = {"fit_seconds": perf_counter() - started, "evaluation_seconds": 0}
    started = perf_counter()
    evaluation = evaluate_ranker(model, test, k=invocation.k)
    timings["evaluation_seconds"] = perf_counter() - started
    _save_evaluation(paths, dataset, _Outcome(model, train, test, evaluation), invocation)
    return (
        {
            "model": configuration.name,
            "dataset": paths[0].name,
            "k": invocation.k,
            "n_users": len(evaluation.per_user),
            "n_catalog": len(test.catalog),
            "n_train_events": len(train.interactions),
            "n_test_events": len(test.interactions),
            **evaluation.metrics,
            **timings,
        },
        {**dataset, **timings},
    )


def _save_evaluation(
    paths: tuple[Path, Path], dataset: Dataset, result: _Outcome, invocation: _Invocation
) -> None:
    directory, final = paths
    (directory / "test").mkdir(parents=True)
    metrics = _metrics(result.test, result.evaluation, invocation.k)
    metrics["inputs"] = {split: dataset["inputs"][split] for split in ("test", "catalog")}
    metrics["model"] = {
        "name": final.parent.parent.parent.parent.name,
        "parameters": _model_parameters(result.model),
        "path": None,
        "sha256": None,
    }
    if invocation.save_model:
        (directory / "model").mkdir()
        saved = write_fitted_ranker(
            result.model,
            directory / "model",
            interactions=result.train.interactions,
            inputs={split: dataset["inputs"][split] for split in ("train", "catalog")},
        )
        metrics["model"].update(
            path=str(final / "model" / "model.joblib"), sha256=saved["model"]["sha256"]
        )
    pd.DataFrame.to_csv(
        result.evaluation.per_user, directory / "test" / "per_user.csv",
        index=False, encoding="utf-8",
    )
    write_json(directory / "test" / "metrics.json", metrics)


def _manifest(configuration: ModelConfiguration, invocation: _Invocation) -> Metadata:
    prototype = configuration.create_ranker()
    return {
        "format_version": 1,
        "kind": RUN_KIND,
        "run_id": invocation.run_id,
        "invocation_id": invocation.invocation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": configuration.name,
            "algorithm": configuration.algorithm,
            "class": f"{type(prototype).__module__}.{type(prototype).__qualname__}",
            "parameters": _model_parameters(prototype),
        },
        "k": invocation.k,
        "save_model": invocation.save_model,
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "datasets": {},
    }


def _model_parameters(model: Estimator[RankingData[pd.DataFrame], object]) -> dict[str, object]:
    return model.get_params(deep=False)


def _write_results(staging: Path, rows: list[Metadata]) -> None:
    with (staging / "results.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=_COLUMNS)
        _ = writer.writeheader()
        writer.writerows(rows)


def _run_model(
    paths: tuple[Path, Path],
    configuration: ModelConfiguration,
    datasets: dict[str, Dataset],
    invocation: _Invocation,
) -> None:
    paths[0].mkdir()
    manifest = _manifest(configuration, invocation)
    rows: list[Metadata] = []
    for name, dataset in datasets.items():
        row, manifest["datasets"][name] = _run_dataset(
            (paths[0] / "datasets" / name, paths[1] / "datasets" / name),
            configuration,
            dataset,
            invocation,
        )
        rows.append(row)
    _write_results(paths[0], rows)
    write_json(paths[0] / "manifest.json", manifest)


def _publish(
    staging: Path,
    destinations: list[Path],
    configurations: list[ModelConfiguration],
    work: tuple[Path, dict[str, Dataset], _Invocation],
) -> None:
    output, datasets, invocation = work
    for configuration in configurations:
        destination = output / configuration.name / "runs" / invocation.run_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()  # Exclusive reservation, invisible until the shared commit.
        destinations.append(destination)
    for configuration, destination in zip(configurations, destinations, strict=True):
        _run_model((staging / configuration.name, destination), configuration, datasets, invocation)
    for dataset in datasets.values():
        _check_sources(dataset)
    write_json(
        staging / "commit.json",
        {
            "format_version": 1,
            "kind": COMMIT_KIND,
            "invocation_id": invocation.invocation_id,
            "runs": {config.name: invocation.run_id for config in configurations},
        },
    )
    for configuration, destination in zip(configurations, destinations, strict=True):
        _ = (staging / configuration.name).rename(destination)
    _ = (staging / "commit.json").rename(output / ".completed" / f"{invocation.invocation_id}.json")


def _execute(
    output: Path,
    configurations: list[ModelConfiguration],
    datasets: dict[str, Dataset],
    invocation: _Invocation,
) -> list[Path]:
    destinations: list[Path] = []
    with TemporaryDirectory(prefix=".benchmark-", dir=output) as temporary:
        try:
            _publish(Path(temporary), destinations, configurations, (output, datasets, invocation))
        except BaseException:
            if not (output / ".completed" / f"{invocation.invocation_id}.json").exists():
                for destination in destinations:
                    shutil.rmtree(destination)
            raise
    return destinations


def run_ranking_benchmark(
    dataset_dirs: Iterable[Path | str],
    output_dir: Path | str,
    *,
    models: Iterable[str],
    k: object = 10,
    save_model: object = True,
) -> list[Path]:
    """Fit each model/dataset pair on train, evaluate test and publish all runs.

    The shared commit marker is the visibility boundary. Uncommitted directories
    remain invisible; failures roll back only this invocation's reservations.

    Args:
        dataset_dirs (Iterable[Path | str]): Immutable dataset snapshots.
        output_dir (Path | str): Benchmark result root.
        models (Iterable[str]): Shell-quoted algorithm specifications.
        k (object, default=10): Positive integer recommendation cutoff.
        save_model (object, default=True): Whether to serialize each fitted estimator.

    Returns:
        list[Path]: Committed model run directories in specification order.

    Raises:
        ValueError: Options or the dataset matrix are invalid.
    """
    size = check_integer(k, "k", 1, np.inf)
    if not isinstance(save_model, bool):
        raise ValueError("save_model must be a boolean")
    if not (configurations := [parse_model_spec(spec) for spec in models]):
        raise ValueError("At least one model is required")
    if len({config.name for config in configurations}) != len(configurations):
        raise ValueError("Duplicate canonical model name")
    output = Path(output_dir).resolve()
    datasets = _preflight(dataset_dirs, output, configurations, size)
    (output / ".completed").mkdir(parents=True, exist_ok=True)
    invocation = _Invocation(_new_run_id(output, configurations), uuid4().hex, size, save_model)
    return _execute(output, configurations, datasets, invocation)
