"""Default ranking-model composition and compatibility persistence exports."""

from pathlib import Path

import pandas as pd

from buy_today.ranking.domain import Estimator, RankingData, TrainingPaths
from buy_today.ranking.random import RandomRanker
from buy_today.ranking.snapshots import (
    EvaluationPaths,
    digest,
    report_ranking,
    source_record,
    train_snapshot,
    users_digest,
    write_fitted_ranker,
    write_json,
)

__all__ = [
    "EvaluationPaths", "TrainingPaths", "digest", "report_ranking", "source_record",
    "train_ranker", "users_digest", "write_fitted_ranker", "write_json",
]


def train_ranker(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    ranker: Estimator[RankingData[pd.DataFrame], object] | None = None,
) -> TrainingPaths:
    """Select the random default or supplied estimator and publish a fresh clone.

    Args:
        dataset_dir (Path | str): Immutable history snapshot.
        output_dir (Path | str): New model destination outside the dataset.
        ranker (Estimator[RankingData[pd.DataFrame], object] | None, default=None):
            Cloneable estimator, defaulting to RandomRanker.

    Returns:
        TrainingPaths: Published model and manifest paths.
    """
    selected = RandomRanker() if ranker is None else ranker
    return train_snapshot(dataset_dir, output_dir, ranker=selected)
