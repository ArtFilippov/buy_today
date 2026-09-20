"""Delegate model policy and persist its snapshot, without creating a report."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from prak.auto_eda.checks import check_dataset
from prak.clustering.models import ModelSnapshot
from prak.clustering.models.temporal import train_temporal
from prak.schema import ROW_KEY, read_dataset


@dataclass(frozen=True)
class TrainingPaths:
    model_path: Path
    distance_path: Path
    labels_path: Path


def check_labels(labels: pd.DataFrame) -> None:
    """Validate the portable keyed assignment table, including integer IDs."""
    if tuple(labels.columns) != (*ROW_KEY, "cluster_id"):
        raise ValueError(f"labels must contain exactly {(*ROW_KEY, 'cluster_id')}")
    if labels.empty or labels.isna().any().any():
        raise ValueError("labels must be nonempty and contain no missing values")
    if labels.duplicated(list(ROW_KEY)).any():
        raise ValueError("Duplicate row keys in labels")
    for column in ("order_item_id", "cluster_id"):
        if not pd.api.types.is_integer_dtype(labels[column].dtype):
            raise ValueError(f"{column} in labels must contain integers")
    if not labels["order_id"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("order_id in labels must contain strings")


def train_clustering(
    batch_path: Path | str,
    output_dir: Path | str,
    *,
    strategy: Callable[[pd.DataFrame], ModelSnapshot] = train_temporal,
) -> TrainingPaths:
    """Run a strategy on the explicit input and save its complete snapshot.

    A callable/closure can carry previous state for incremental strategies.
    Only the strategy decides what to fit and which old assignments to replace.
    """
    frame = read_dataset(batch_path)
    check_dataset(frame)
    snapshot = strategy(frame)
    check_labels(snapshot.labels)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = TrainingPaths(
        output_dir / "model.joblib", output_dir / "distance.joblib",
        output_dir / "labels.csv",
    )
    joblib.dump(snapshot.model, paths.model_path)
    joblib.dump(snapshot.distance, paths.distance_path)
    snapshot.labels.to_csv(paths.labels_path, index=False, encoding="utf-8")
    return paths
