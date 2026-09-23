"""Validate the JSON/CSV boundary of portable benchmark comparisons."""

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import TypeIs

from buy_today.ranking.domain import SVD
from buy_today.ranking.publication import Metadata, SelectedRuns

METRICS = {"recall_at_k": "Recall@K", "ndcg_at_k": "NDCG@K"}
COUNTS = ("k", "n_users", "n_catalog", "n_train_events", "n_test_events")
TIMINGS = ("fit_seconds", "evaluation_seconds")
RESULT_COLUMNS = ("model", "dataset", *COUNTS, *METRICS, *TIMINGS)
CSV_COLUMNS = (
    "rank",
    "score",
    "model",
    "algorithm",
    "parameters",
    "run_id",
    "run_path",
    "invocation_id",
    "created_at",
    "save_model",
    "versions",
    "dataset",
    "dataset_path",
    *COUNTS,
    *METRICS,
    *TIMINGS,
    "user_ids_sha256",
    *[
        f"{split}_{field}"
        for split in ("train", "test", "catalog")
        for field in ("path", "sha256", "rows")
    ],
)
_TEST = "test"
_MIN_PRODUCTS = 2


def _json_object(value: object) -> TypeIs[Metadata]:
    # JSON decoding guarantees string keys; values are validated by their consumers.
    return isinstance(value, dict)


def _integer(value: object, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, maximum: int | None = None) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise ValueError(f"{name} must be finite and numeric")
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum + 1e-12):
        interval = "[0, 1]" if maximum == 1 else "[0, infinity)"
        raise ValueError(f"{name} must be finite and in {interval}")
    return value


def _sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a SHA256 hex digest")


def _absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return Path(value)


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _check_model(model: Metadata) -> None:
    algorithm = model["algorithm"]
    classes = {
        "random": "buy_today.ranking.random.RandomRanker",
        "svd": "buy_today.ranking.svd.SVDRanker",
    }
    if algorithm not in classes or model["class"] != classes[algorithm]:
        raise ValueError("Model algorithm/class mismatch")
    parameters = model["parameters"]
    expected: set[str] = {"random_state"} | (
        {"n_components", "n_iter"} if algorithm == SVD else set()
    )
    if not _json_object(parameters) or parameters.keys() != expected:
        raise ValueError("Model parameters must include all effective parameters and seed")
    if _integer(parameters["random_state"], "random_state", 0) >= 2**32:
        raise ValueError("random_state must be less than 2**32")
    for name in expected - {"random_state"}:
        _ = _integer(parameters[name], name)


def _check_dataset(name: str, dataset: Metadata, model: Metadata, k: int) -> None:
    if name in {"", ".", ".."} or Path(name).name != name:
        raise ValueError("Dataset key must be a basename")
    directory = _absolute_path(dataset["path"], "dataset path")
    if directory.name != name or set(dataset["inputs"]) != {"train", "test", "catalog"}:
        raise ValueError(f"Dataset {name}: input identity mismatch")
    for split, source in dataset["inputs"].items():
        if _absolute_path(source["path"], f"{split} path") != directory / f"{split}.csv":
            raise ValueError(f"Dataset {name}: {split} input path mismatch")
        _sha256(source["sha256"], f"{split} SHA256")
        _ = _integer(source["rows"], f"{split} rows")
    users = _integer(dataset["n_users"], "n_users")
    _sha256(dataset["user_ids_sha256"], "user_ids_sha256")
    catalog = dataset["inputs"]["catalog"]["rows"]
    if k > catalog or any(dataset["inputs"][split]["rows"] < users for split in ("train", "test")):
        raise ValueError(f"Dataset {name}: inconsistent counts or K exceeds catalog size")
    if model["algorithm"] == SVD and (
        catalog < _MIN_PRODUCTS or model["parameters"]["n_components"] > min(users, catalog)
    ):
        raise ValueError(f"Dataset {name}: SVD dimensions exceed user/catalog counts")
    for field in TIMINGS:
        _ = _number(dataset[field], field)


def _results(run_path: Path) -> dict[str, Metadata]:
    with (run_path / "results.csv").open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if (
            reader.fieldnames is None
            or len(set(reader.fieldnames)) != len(reader.fieldnames)
            or not set(RESULT_COLUMNS).issubset(reader.fieldnames)
        ):
            raise ValueError("results.csv has missing or duplicate columns")
        rows: dict[str, Metadata] = {}
        for record in reader:
            if None in record or any(value is None for value in record.values()):
                raise ValueError("results.csv has an invalid row")
            row: Metadata = dict(record)
            if (name := row["dataset"]) in rows:
                raise ValueError(f"results.csv has duplicate dataset {name}")
            for field in COUNTS:
                row[field] = _integer(int(row[field]), field)
            for field in (*METRICS, *TIMINGS):
                row[field] = _number(float(row[field]), field, 1 if field in METRICS else None)
            rows[name] = row
    return rows


def _check_saved_model(identity: Metadata, manifest: Metadata, name: str) -> None:
    if manifest["save_model"]:
        model_path = _absolute_path(identity["path"], "model path")
        expected_suffix = (
            manifest["model"]["name"],
            "runs",
            manifest["run_id"],
            "datasets",
            name,
            "model",
            "model.joblib",
        )
        if model_path.parts[-len(expected_suffix) :] != expected_suffix:
            raise ValueError(f"Dataset {name}: saved model path identity mismatch")
        _sha256(identity["sha256"], "model SHA256")
    elif identity["path"] is not None or identity["sha256"] is not None:
        raise ValueError(f"Dataset {name}: unsaved model must have null path and SHA256")


def _check_evaluation(run_path: Path, manifest: Metadata, name: str, *, row: Metadata) -> None:
    dataset = manifest["datasets"][name]
    report: Metadata = json.loads(
        (run_path / "datasets" / name / "test" / "metrics.json").read_text(encoding="utf-8")
    )
    if (
        isinstance(report["format_version"], bool)
        or not isinstance(report["format_version"], int)
        or report["format_version"] != 1
        or report["split"] != _TEST
    ):
        raise ValueError(f"Dataset {name}: expected test metrics format_version=1")
    expected_counts = {
        "k": manifest["k"],
        "n_users": dataset["n_users"],
        "n_catalog": dataset["inputs"]["catalog"]["rows"],
        "n_events": dataset["inputs"]["test"]["rows"],
    }
    for field, expected in expected_counts.items():
        if _integer(report[field], field) != expected:
            raise ValueError(f"Dataset {name}: evaluation {field} count/K mismatch")
    if json_text(report["inputs"]) != json_text(
        {split: dataset["inputs"][split] for split in ("test", "catalog")}
    ):
        raise ValueError(f"Dataset {name}: evaluation input provenance mismatch")
    identity = report["model"]
    if identity["name"] != manifest["model"]["name"] or json_text(
        identity["parameters"]
    ) != json_text(manifest["model"]["parameters"]):
        raise ValueError(f"Dataset {name}: evaluation model parameters/identity mismatch")
    _check_saved_model(identity, manifest, name)
    for field in METRICS:
        if _number(report["metrics"][field], field, 1) != row[field]:
            raise ValueError(
                f"Dataset {name}: {field} differs between results.csv and metrics.json"
            )


def _manifest_datasets(manifest: Metadata) -> dict[str, Metadata]:
    _check_model(manifest["model"])
    if not isinstance(manifest["save_model"], bool):
        raise ValueError("save_model must be a boolean")
    _ = datetime.fromisoformat(manifest["created_at"])
    for package in ("numpy", "scikit_learn"):
        if not isinstance(manifest["versions"][package], str) or not manifest["versions"][package]:
            raise ValueError("Package versions must be nonempty strings")
    if not _json_object(manifest["datasets"]) or not manifest["datasets"]:
        raise ValueError("A benchmark run must contain datasets")
    datasets: dict[str, Metadata] = manifest["datasets"]
    return datasets


def _enriched_row(run_path: Path, manifest: Metadata, name: str, row: Metadata) -> Metadata:
    dataset = manifest["datasets"][name]
    expected = {
        "model": manifest["model"]["name"],
        "k": manifest["k"],
        "n_users": dataset["n_users"],
        "n_catalog": dataset["inputs"]["catalog"]["rows"],
        "n_train_events": dataset["inputs"]["train"]["rows"],
        "n_test_events": dataset["inputs"]["test"]["rows"],
        **{field: dataset[field] for field in TIMINGS},
    }
    if any(row[field] != value for field, value in expected.items()):
        raise ValueError(f"Dataset {name}: results.csv model/counts/K/timings mismatch")
    _check_evaluation(run_path, manifest, name, row=row)
    return {
        **{field: row[field] for field in RESULT_COLUMNS},
        "algorithm": manifest["model"]["algorithm"],
        "parameters": json_text(manifest["model"]["parameters"]),
        "run_path": str(run_path),
        **{
            field: manifest[field]
            for field in ("run_id", "invocation_id", "created_at", "save_model")
        },
        "versions": json_text(manifest["versions"]),
        "dataset_path": dataset["path"],
        "user_ids_sha256": dataset["user_ids_sha256"],
        **{
            f"{split}_{field}": source[field]
            for split, source in dataset["inputs"].items()
            for field in ("path", "sha256", "rows")
        },
    }


def _run_rows(run_path: Path, manifest: Metadata) -> list[Metadata]:
    datasets = _manifest_datasets(manifest)
    k = _integer(manifest["k"], "K")
    results = _results(run_path)
    if set(results) != set(datasets):
        raise ValueError("results.csv and manifest must contain the same dataset set")
    rows: list[Metadata] = []
    for name, dataset in sorted(datasets.items()):
        _check_dataset(name, dataset, manifest["model"], k)
        rows.append(_enriched_row(run_path, manifest, name, results[name]))
    return rows


def _identities(rows: list[Metadata]) -> dict[str, Metadata]:
    return {
        row["dataset"]: {
            field: row[field]
            for field in (
                "k",
                "n_users",
                "user_ids_sha256",
                "train_sha256",
                "test_sha256",
                "catalog_sha256",
                "train_rows",
                "test_rows",
                "catalog_rows",
            )
        }
        for row in rows
    }


def _check_comparable(current: dict[str, Metadata], reference: dict[str, Metadata]) -> None:
    if set(current) != set(reference):
        raise ValueError("Latest model runs must contain the same dataset set")
    for name, identity in current.items():
        if identity["k"] != reference[name]["k"]:
            raise ValueError("Latest model runs must use the same K")
        if identity != reference[name]:
            raise ValueError(
                f"Dataset {name}: input SHA256/counts/user identity differs across models"
            )


def comparison_rows(selected: SelectedRuns) -> list[Metadata]:
    rows: list[Metadata] = []
    reference: dict[str, Metadata] | None = None
    for run_path, manifest in selected:
        try:
            current = _run_rows(run_path, manifest)
        except (KeyError, TypeError, AttributeError, OverflowError, OSError, ValueError) as error:
            raise ValueError(
                f"Run {run_path}: invalid benchmark report metadata: {error}"
            ) from error
        identities = _identities(current)
        if reference is None:
            reference = identities
        else:
            _check_comparable(identities, reference)
        rows.extend(current)
    return _rank_rows(selected, rows)


def _rank_rows(selected: SelectedRuns, rows: list[Metadata]) -> list[Metadata]:
    scores = {
        manifest["model"]["name"]: max(
            row["ndcg_at_k"] for row in rows if row["model"] == manifest["model"]["name"]
        )
        for _, manifest in selected
    }
    order = sorted(scores, key=lambda name: (-scores[name], name))
    ranks = {name: index + 1 for index, name in enumerate(order)}
    for row in rows:
        row.update(score=scores[row["model"]], rank=ranks[row["model"]])
    return sorted(rows, key=lambda row: (row["rank"], row["dataset"]))
