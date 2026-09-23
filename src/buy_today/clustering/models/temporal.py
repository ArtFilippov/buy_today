"""Compose the temporal estimator, a ready distance and keyed assignments."""

import pandas as pd

from buy_today.clustering.distances.temporal_distances import TimestampDistance
from buy_today.clustering.contracts import TrainableDistance
from buy_today.clustering.models import ModelSnapshot
from buy_today.clustering.models.temporal_estimator import TemporalClustering
from buy_today.schema import ROW_KEY

__all__ = ["TemporalClustering", "train_temporal"]


def train_temporal(
    X: pd.DataFrame, *, n_clusters: int = 20, distance: TrainableDistance | None = None
) -> ModelSnapshot:
    """Refit on the supplied accumulated dataset and replace all assignments.

    This is the temporal baseline's policy, not a requirement of other models.
    A learned distance can also be prepared here, before independent evaluation.

    Args:
        X (pd.DataFrame): Accumulated positions to assign.
        n_clusters (int, default=20): Number of consecutive temporal groups.
        distance (TrainableDistance | None, default=None): Optional learned distance to fit.

    Returns:
        ModelSnapshot: Fitted model, ready distance and keyed assignments.
    """
    model = TemporalClustering(n_clusters=n_clusters).fit(X)
    distance = TimestampDistance() if distance is None else distance
    _ = distance.fit(X)
    labels = X[list(ROW_KEY)].copy()
    labels["cluster_id"] = model.labels_
    return ModelSnapshot(model=model, distance=distance, labels=labels)
