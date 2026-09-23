"""Delegate model policy and persist its snapshot, without creating a report."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from buy_today.auto_eda.checks import check_dataset
from buy_today.clustering.labels import check_labels
from buy_today.clustering.models import ModelSnapshot
from buy_today.clustering.models.temporal import train_temporal
from buy_today.schema import read_dataset


@dataclass(frozen=True)
class TrainingPaths:
    model_path: Path
    distance_path: Path
    labels_path: Path


def train_clustering(
    batch_path: Path | str,
    output_dir: Path | str,
    *,
    strategy: Callable[[pd.DataFrame], ModelSnapshot] = train_temporal,
) -> TrainingPaths:
    """Run a strategy on the explicit input and save its complete snapshot.

    A callable/closure can carry previous state for incremental strategies.
    Only the strategy decides what to fit and which old assignments to replace.

    Args:
        batch_path (Path | str): Explicit input CSV for the strategy.
        output_dir (Path | str): Directory for the complete saved snapshot.
        strategy (Callable[[pd.DataFrame], ModelSnapshot], default=train_temporal):
            Model-specific training policy.

    Returns:
        TrainingPaths: Paths of the model, distance and keyed assignments.
    """
    frame = read_dataset(batch_path)
    _checks = check_dataset(frame)
    snapshot = strategy(frame)
    check_labels(snapshot.labels)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = TrainingPaths(
        output_dir / "model.joblib",
        output_dir / "distance.joblib",
        output_dir / "labels.csv",
    )
    _ = joblib.dump(snapshot.model, paths.model_path)
    _ = joblib.dump(snapshot.distance, paths.distance_path)
    pd.DataFrame.to_csv(snapshot.labels, paths.labels_path, index=False, encoding="utf-8")
    return paths
