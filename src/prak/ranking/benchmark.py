"""Fresh train/test experiments, published as complete multi-model invocations."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import numpy as np
import sklearn

from prak.ranking.data import _check_integer, read_ranking_data
from prak.ranking.evaluation import evaluate_ranker
from prak.ranking.random import RandomRanker
from prak.ranking.storage import _digest, _source, _users_digest, _write_fitted_ranker, _write_json
from prak.ranking.svd import SVDRanker


_RANKERS = {"random": RandomRanker, "svd": SVDRanker}
_TIMESTAMP = "%Y%m%dT%H%M%S%fZ"
_RUN_ID = re.compile(r"[0-9]{8}T[0-9]{12}Z")
_INVOCATION_ID = re.compile(r"[0-9a-f]{32}")
_COLUMNS = (
    "model", "dataset", "k", "n_users", "n_catalog", "n_train_events", "n_test_events",
    "recall_at_k", "ndcg_at_k", "fit_seconds", "evaluation_seconds",
)


class _ModelParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(f"{self.prog}: {message}")


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise ValueError("expected a positive integer")
    return number


def _seed(value):
    number = int(value)
    if not 0 <= number < 2**32:
        raise ValueError("expected an integer in [0, 2**32 - 1]")
    return number


@dataclass(frozen=True)
class ModelConfiguration:
    algorithm: str
    parameters: dict[str, int]

    @property
    def name(self):
        defaults = _RANKERS[self.algorithm]().get_params(deep=False)
        return self.algorithm + "".join(
            f"_{key.replace('_', '-')}-{value}" for key, value in sorted(self.parameters.items())
            if value != defaults[key]
        )

    def create_ranker(self):
        return _RANKERS[self.algorithm](**self.parameters)


def parse_model_spec(spec: str) -> ModelConfiguration:
    """Parse a shell-quoted model and its own options, without exiting the process."""
    tokens = shlex.split(spec)
    if not tokens or tokens[0] not in _RANKERS:
        raise ValueError("Expected model random or svd")
    algorithm = tokens[0]
    parser = _ModelParser(prog=algorithm, add_help=False, allow_abbrev=False)
    for name, default in _RANKERS[algorithm]().get_params(deep=False).items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=default,
                            type=_seed if name == "random_state" else _positive_int)
    return ModelConfiguration(algorithm, vars(parser.parse_args(tokens[1:])))


def _check_sources(dataset):
    for source in dataset["inputs"].values():
        if _digest(source["path"]) != source["sha256"]:
            raise ValueError(f"Benchmark input changed: {source['path']}")


def _load_pair(directory, k):
    train = read_ranking_data(directory, split="train")
    test = read_ranking_data(directory, split="test")
    _check_integer(k, "k", 1, len(train.catalog))
    if not train.catalog.equals(test.catalog):
        raise ValueError("Evaluation catalog differs from training catalog")
    if _users_digest(train.interactions) != _users_digest(test.interactions):
        raise ValueError("Evaluation users differ from training users")
    return train, test


def _preflight(dataset_dirs, output, configurations, k):
    # Preserve the basename supplied by the caller, including a directory symlink.
    directories = [Path(os.path.abspath(directory)) for directory in dataset_dirs]
    if not directories:
        raise ValueError("At least one dataset is required")
    if len({path.name for path in directories}) != len(directories):
        raise ValueError("Duplicate dataset basename")
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"Dataset directory does not exist: {directory}")
        resolved = directory.resolve()
        if output.is_relative_to(resolved) or resolved.is_relative_to(output):
            raise ValueError("Benchmark output must be outside and separate from dataset snapshots")
        for split in ("train", "test", "catalog"):
            if not (directory / f"{split}.csv").is_file():
                raise ValueError(f"Missing benchmark input: {directory / f'{split}.csv'}")
    datasets = {}
    for directory in directories:
        before = {split: _digest(directory / f"{split}.csv") for split in ("train", "test", "catalog")}
        train, test = _load_pair(directory, k)
        n_users = int(train.interactions.user_id.nunique())
        for configuration in configurations:
            if configuration.algorithm == "svd":
                if len(train.catalog) < 2:
                    raise ValueError("SVD requires at least two products in catalog")
                _check_integer(configuration.parameters["n_components"], "n_components",
                               1, min(n_users, len(train.catalog)))
        inputs = {
            split: _source(directory / f"{split}.csv", rows)
            for split, rows in (("train", len(train.interactions)), ("test", len(test.interactions)),
                                ("catalog", len(train.catalog)))
        }
        if any(inputs[split]["sha256"] != digest for split, digest in before.items()):
            raise ValueError(f"Benchmark input changed while reading: {directory}")
        datasets[directory.name] = {
            "path": str(directory), "inputs": inputs, "n_users": n_users,
            "user_ids_sha256": _users_digest(train.interactions),
        }
    return datasets


def _new_run_id(output, configurations):
    # A microsecond increment handles collisions and clocks moving backwards.
    now = datetime.now(timezone.utc)
    for configuration in configurations:
        runs = output / configuration.name / "runs"
        if runs.exists():
            for path in runs.iterdir():
                if _RUN_ID.fullmatch(path.name):
                    previous = datetime.strptime(path.name, _TIMESTAMP).replace(tzinfo=timezone.utc)
                    now = max(now, previous + timedelta(microseconds=1))
    return now.strftime(_TIMESTAMP)


def _run_model(staging, final, configuration, datasets, *, run_id, invocation_id, k, save_model):
    staging.mkdir()
    versions = {"numpy": np.__version__, "scikit_learn": sklearn.__version__}
    prototype = configuration.create_ranker()
    manifest = {
        "format_version": 1, "kind": "ranking_benchmark", "run_id": run_id,
        "invocation_id": invocation_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": configuration.name, "algorithm": configuration.algorithm,
            "class": f"{type(prototype).__module__}.{type(prototype).__qualname__}",
            "parameters": prototype.get_params(deep=False),
        },
        "k": int(k), "save_model": save_model, "versions": versions, "datasets": {},
    }
    rows = []
    for name, dataset in datasets.items():
        _check_sources(dataset)
        train, test = _load_pair(dataset["path"], k)
        _check_sources(dataset)
        model = configuration.create_ranker()
        started = perf_counter()
        model.fit(train)
        fit_seconds = perf_counter() - started
        started = perf_counter()
        evaluation = evaluate_ranker(model, test, k=k)
        evaluation_seconds = perf_counter() - started
        directory = staging / "datasets" / name
        (directory / "test").mkdir(parents=True)
        model_hash = None
        if save_model:
            (directory / "model").mkdir()
            saved = _write_fitted_ranker(
                model, directory / "model", interactions=train.interactions,
                inputs={split: dataset["inputs"][split] for split in ("train", "catalog")},
            )
            model_hash = saved["model"]["sha256"]
        evaluation.per_user.to_csv(directory / "test" / "per_user.csv", index=False, encoding="utf-8")
        _write_json(directory / "test" / "metrics.json", {
            "format_version": 1, "split": "test", "k": int(k), "n_users": len(evaluation.per_user),
            "n_catalog": len(test.catalog), "n_events": len(test.interactions),
            "metrics": evaluation.metrics,
            "inputs": {split: dataset["inputs"][split] for split in ("test", "catalog")},
            "model": {
                "name": configuration.name, "parameters": model.get_params(deep=False),
                "path": str(final / "datasets" / name / "model" / "model.joblib") if save_model else None,
                "sha256": model_hash,
            },
        })
        timings = {"fit_seconds": fit_seconds, "evaluation_seconds": evaluation_seconds}
        manifest["datasets"][name] = {**dataset, **timings}
        rows.append({
            "model": configuration.name, "dataset": name, "k": int(k),
            "n_users": len(evaluation.per_user), "n_catalog": len(test.catalog),
            "n_train_events": len(train.interactions), "n_test_events": len(test.interactions),
            **evaluation.metrics, **timings,
        })
    with (staging / "results.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(staging / "manifest.json", manifest)


def run_ranking_benchmark(
    dataset_dirs, output_dir: Path | str, *, models, k=10, save_model=True,
) -> list[Path]:
    """Fit once per model × dataset on train, evaluate on test, then publish all runs.

    A shared commit marker is the visibility boundary for the entire invocation.
    Renamed directories without that marker (e.g. after a process crash) are
    unpublished and ignored by readers. No model serialization occurs when
    save_model=False. Inputs and parameters are checked before any fitting.
    """
    _check_integer(k, "k", 1, np.inf)
    if not isinstance(save_model, bool):
        raise ValueError("save_model must be a boolean")
    configurations = [parse_model_spec(spec) for spec in models]
    if not configurations:
        raise ValueError("At least one model is required")
    if len({config.name for config in configurations}) != len(configurations):
        raise ValueError("Duplicate canonical model name")
    output = Path(output_dir).resolve()
    datasets = _preflight(dataset_dirs, output, configurations, k)
    output.mkdir(parents=True, exist_ok=True)
    completed = output / ".completed"
    completed.mkdir(exist_ok=True)
    # Reserving the final directories with mkdir prevents overwrite even if two
    # invocations pick the same timestamp. They remain invisible until commit.
    destinations = []
    invocation_id = uuid4().hex
    with TemporaryDirectory(prefix=".benchmark-", dir=output) as temporary:
        staging = Path(temporary)
        try:
            run_id = _new_run_id(output, configurations)
            for configuration in configurations:
                parent = output / configuration.name / "runs"
                parent.mkdir(parents=True, exist_ok=True)
                destination = parent / run_id
                destination.mkdir()  # Exclusive reservation; never reuse an existing run.
                destinations.append(destination)
            for configuration, destination in zip(configurations, destinations, strict=True):
                _run_model(staging / configuration.name, destination, configuration, datasets,
                           run_id=run_id, invocation_id=invocation_id, k=k, save_model=save_model)
            for dataset in datasets.values():
                _check_sources(dataset)
            commit = {
                "format_version": 1, "kind": "ranking_benchmark_commit", "invocation_id": invocation_id,
                "runs": {config.name: run_id for config in configurations},
            }
            _write_json(staging / "commit.json", commit)
            for configuration, destination in zip(configurations, destinations, strict=True):
                # os.rename replaces our empty reservation on the same filesystem.
                (staging / configuration.name).rename(destination)
            (staging / "commit.json").rename(completed / f"{invocation_id}.json")
        except BaseException:
            # If interrupted immediately after committing, preserve the whole
            # visible invocation. Otherwise roll back only our reservations/runs.
            if not (completed / f"{invocation_id}.json").exists():
                for destination in destinations:
                    shutil.rmtree(destination)
            raise
    return destinations


def read_latest_benchmark_runs(output_dir: Path | str) -> list[tuple[Path, dict]]:
    """Select latest committed runs by folder name from one commit-list snapshot.

    Enumerating markers rather than run folders excludes staging and orphaned
    reservations without ever parsing their possibly incomplete manifests.
    """
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise ValueError(f"Benchmark root does not exist: {output}")
    latest = {}
    ownership = {}
    for path in sorted((output / ".completed").glob("*.json")):
        try:
            commit = json.loads(path.read_text(encoding="utf-8"))
            invocation_id = commit["invocation_id"]
            if (commit["format_version"] != 1 or commit["kind"] != "ranking_benchmark_commit"
                    or not _INVOCATION_ID.fullmatch(invocation_id) or path.stem != invocation_id
                    or not isinstance(commit["runs"], dict) or not commit["runs"]):
                raise ValueError("Invalid benchmark commit identity")
            for model, run_id in commit["runs"].items():
                if (not re.fullmatch(r"[a-z][a-z0-9_-]*", model) or not _RUN_ID.fullmatch(run_id)):
                    raise ValueError("Invalid model/run name in benchmark commit")
                run = output / model / "runs" / run_id
                if not run.is_dir() or (model, run_id) in ownership:
                    raise ValueError(f"Missing or multiply committed benchmark run: {run}")
                ownership[model, run_id] = invocation_id
                if model not in latest or run_id > latest[model]:
                    latest[model] = run_id
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid benchmark commit {path}: {error}") from error
    selected = []
    for model, run_id in sorted(latest.items()):
        run = output / model / "runs" / run_id
        try:
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            if (manifest["format_version"] != 1 or manifest["kind"] != "ranking_benchmark"
                    or manifest["run_id"] != run_id or manifest["model"]["name"] != model
                    or manifest["invocation_id"] != ownership[model, run_id]):
                raise ValueError("Benchmark run identity differs from commit")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid benchmark run {run}: {error}") from error
        selected.append((run, manifest))
    return selected
