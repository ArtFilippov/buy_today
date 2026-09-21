"""Independent training and evaluation over explicit, immutable file snapshots."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import sklearn
from sklearn.base import clone

from prak.ranking.data import read_ranking_data
from prak.ranking.evaluation import evaluate_ranker
from prak.ranking.random import RandomRanker


@dataclass(frozen=True)
class TrainingPaths:
    model_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class EvaluationPaths:
    metrics_path: Path
    per_user_path: Path


def _digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _users_digest(interactions):
    encoded = json.dumps(sorted(interactions.user_id.unique()), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source(path, rows):
    return {"path": str(path), "sha256": _digest(path), "rows": rows}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _output_directory(output_dir, dataset_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    if dataset_dir in output_dir.parents:
        raise ValueError("Output directory must be outside the history snapshot")
    return output_dir


@contextmanager
def _publish(output_dir):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ranking-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        yield staging
        staging.rename(output_dir)


def train_ranker(
    dataset_dir: Path | str, output_dir: Path | str, *, ranker=None,
) -> TrainingPaths:
    """Fit a fresh estimator on accumulated train + catalog and save it.

    ranker defaults to RandomRanker(); a supplied sklearn-style estimator is
    cloned, so caller state is not modified. Validation/test and generator state
    are never opened. The destination must be new and outside the dataset.
    """
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = _output_directory(output_dir, dataset_dir)
    data = read_ranking_data(dataset_dir)
    inputs = {
        "train": _source(dataset_dir / "train.csv", len(data.interactions)),
        "catalog": _source(dataset_dir / "catalog.csv", len(data.catalog)),
    }
    model = clone(RandomRanker() if ranker is None else ranker).fit(data)
    manifest = {
        "format_version": 1, "inputs": inputs,
        "n_users": int(data.interactions.user_id.nunique()),
        "user_ids_sha256": _users_digest(data.interactions),
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "model": {
            "file": "model.joblib",
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "parameters": model.get_params(deep=False),
        },
    }
    with _publish(output_dir) as staging:
        joblib.dump(model, staging / "model.joblib")
        manifest["model"]["sha256"] = _digest(staging / "model.joblib")
        _write_json(staging / "manifest.json", manifest)
    return TrainingPaths(output_dir / "model.joblib", output_dir / "manifest.json")


def report_ranking(
    dataset_dir: Path | str, model_dir: Path | str, output_dir: Path | str, *,
    split, k=10,
) -> EvaluationPaths:
    """Score one held-out split with a saved model, without reading train or fitting.

    Catalog bytes and the full user cohort must match the training snapshot.
    Only the selected held-out split and catalog are read from the dataset.
    The model checksum is verified before loading. Reports are published to a
    new directory; a directory under model_dir is allowed.
    """
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    dataset_dir, model_dir = Path(dataset_dir).resolve(), Path(model_dir).resolve()
    output_dir = _output_directory(output_dir, dataset_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Expected ranking model format_version=1")
    data = read_ranking_data(dataset_dir, split=split)
    inputs = {
        split: _source(dataset_dir / f"{split}.csv", len(data.interactions)),
        "catalog": _source(dataset_dir / "catalog.csv", len(data.catalog)),
    }
    if inputs["catalog"]["sha256"] != manifest["inputs"]["catalog"]["sha256"]:
        raise ValueError("Evaluation catalog differs from training catalog")
    if _users_digest(data.interactions) != manifest["user_ids_sha256"]:
        raise ValueError("Evaluation users differ from training users")
    model_path = model_dir / "model.joblib"
    model_hash = _digest(model_path)
    if model_hash != manifest["model"]["sha256"]:
        raise ValueError("Ranking model SHA-256 mismatch")
    model = joblib.load(model_path)
    result = evaluate_ranker(model, data, k=k)
    metrics = {
        "format_version": 1, "split": split, "k": int(k),
        "n_users": len(result.per_user), "n_catalog": len(data.catalog),
        "n_events": len(data.interactions), "metrics": result.metrics,
        "inputs": inputs, "model": {"path": str(model_path), "sha256": model_hash},
    }
    with _publish(output_dir) as staging:
        result.per_user.to_csv(staging / "per_user.csv", index=False, encoding="utf-8")
        _write_json(staging / "metrics.json", metrics)
    return EvaluationPaths(output_dir / "metrics.json", output_dir / "per_user.csv")
