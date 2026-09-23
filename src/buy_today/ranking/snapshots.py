"""Independent training and evaluation over explicit, immutable file snapshots."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Generator
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeIs

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from buy_today.ranking.data import RankingData, read_ranking_data
from buy_today.ranking.domain import (
    Estimator, Source, TrainingPaths, EvaluationPaths, EvaluationSource,
)
from buy_today.ranking.evaluation import evaluate_ranker


_EvaluationSource = EvaluationSource[RankingData[pd.DataFrame]]


def digest(path: Path | str) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def users_digest(interactions: pd.DataFrame) -> str:
    encoded = json.dumps(sorted(set(interactions["user_id"])), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_record(path: Path, rows: int) -> Source:
    return {"path": str(path), "sha256": digest(path), "rows": rows}


def _is_scalar(value: object) -> TypeIs[np.generic[object]]:
    return isinstance(value, np.generic)


def _json_default(value: object) -> object:
    if _is_scalar(value):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: object) -> None:
    _ = path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _output_directory(output_dir: Path | str, dataset_dir: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    if dataset_dir in output_dir.parents:
        raise ValueError("Output directory must be outside the history snapshot")
    return output_dir


@contextmanager
def _publish(output_dir: Path) -> Generator[Path]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ranking-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        yield staging
        _ = staging.rename(output_dir)


def write_fitted_ranker(
    model: Estimator[RankingData[pd.DataFrame], object],
    directory: Path,
    *,
    inputs: dict[str, Source],
    interactions: pd.DataFrame,
) -> dict[str, Any]:
    # The manifest is the extensible JSON serialization boundary.
    manifest: dict[str, Any] = {
        "format_version": 1,
        "inputs": inputs,
        "n_users": len(set(interactions["user_id"])),
        "user_ids_sha256": users_digest(interactions),
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "model": {
            "file": "model.joblib",
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "parameters": model.get_params(deep=False),
        },
    }
    _ = joblib.dump(model, directory / "model.joblib")
    manifest["model"]["sha256"] = digest(directory / "model.joblib")
    write_json(directory / "manifest.json", manifest)
    return manifest


def train_snapshot(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    ranker: Estimator[RankingData[pd.DataFrame], object],
) -> TrainingPaths:
    """Fit a fresh estimator on accumulated train + catalog and save it.

    The supplied sklearn-style estimator is cloned, so caller state is not
    modified. Validation/test and generator state
    are never opened. The destination must be new and outside the dataset.

    Args:
        dataset_dir (Path | str): Immutable history snapshot.
        output_dir (Path | str): New model destination.
        ranker (Estimator[RankingData[pd.DataFrame], object]): Cloneable estimator.

    Returns:
        TrainingPaths: Published model and manifest paths.
    """
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = _output_directory(output_dir, dataset_dir)
    data = read_ranking_data(dataset_dir)
    inputs = {
        "train": source_record(dataset_dir / "train.csv", len(data.interactions)),
        "catalog": source_record(dataset_dir / "catalog.csv", len(data.catalog)),
    }
    model: Estimator[RankingData[pd.DataFrame], object] = clone(ranker)
    _ = model.fit(data)
    with _publish(output_dir) as staging:
        _ = write_fitted_ranker(model, staging, inputs=inputs, interactions=data.interactions)
    return TrainingPaths(output_dir / "model.joblib", output_dir / "manifest.json")


def _evaluation_source(dataset_dir: Path, model_dir: Path, split: str) -> _EvaluationSource:
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Expected ranking model format_version=1")
    data = read_ranking_data(dataset_dir, split=split)
    inputs = {
        split: source_record(dataset_dir / f"{split}.csv", len(data.interactions)),
        "catalog": source_record(dataset_dir / "catalog.csv", len(data.catalog)),
    }
    if inputs["catalog"]["sha256"] != manifest["inputs"]["catalog"]["sha256"]:
        raise ValueError("Evaluation catalog differs from training catalog")
    if users_digest(data.interactions) != manifest["user_ids_sha256"]:
        raise ValueError("Evaluation users differ from training users")
    model_path = model_dir / "model.joblib"
    if (model_hash := digest(model_path)) != manifest["model"]["sha256"]:
        raise ValueError("Ranking model SHA-256 mismatch")
    return _EvaluationSource(data, inputs, model_path, model_hash)


def _evaluate_source(
    source: _EvaluationSource, split: str, k: object
) -> tuple[dict[str, object], pd.DataFrame]:
    result = evaluate_ranker(joblib.load(source.model_path), source.data, k=k)
    return (
        {
            "format_version": 1,
            "split": split,
            "k": k,
            "n_users": len(result.per_user),
            "n_catalog": len(source.data.catalog),
            "n_events": len(source.data.interactions),
            "metrics": result.metrics,
            "inputs": source.inputs,
            "model": {"path": str(source.model_path), "sha256": source.model_hash},
        },
        result.per_user,
    )


def report_ranking(
    dataset_dir: Path | str,
    model_dir: Path | str,
    output_dir: Path | str,
    *,
    split: str,
    k: object = 10,
) -> EvaluationPaths:
    """Score one held-out split with a saved model, without reading train or fitting.

    Catalog bytes and the full user cohort must match the training snapshot.
    Only the selected held-out split and catalog are read from the dataset.
    The model checksum is verified before loading. Reports are published to a
    new directory; a directory under model_dir is allowed.

    Args:
        dataset_dir (Path | str): History snapshot to evaluate.
        model_dir (Path | str): Saved model snapshot.
        output_dir (Path | str): New report destination.
        split (str): Validation or test split.
        k (object, default=10): Positive integer recommendation cutoff.

    Returns:
        EvaluationPaths: Published aggregate and per-user metrics.

    Raises:
        ValueError: Split, snapshot identity or model checksum is invalid.
    """
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    dataset_dir, model_dir = Path(dataset_dir).resolve(), Path(model_dir).resolve()
    output_dir = _output_directory(output_dir, dataset_dir)
    source = _evaluation_source(dataset_dir, model_dir, split)
    metrics, per_user = _evaluate_source(source, split, k)
    with _publish(output_dir) as staging:
        per_user.to_csv(staging / "per_user.csv", index=False, encoding="utf-8")
        write_json(staging / "metrics.json", metrics)
    return EvaluationPaths(output_dir / "metrics.json", output_dir / "per_user.csv")
