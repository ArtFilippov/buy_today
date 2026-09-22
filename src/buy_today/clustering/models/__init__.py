"""Model strategies return a complete snapshot; their update policy is local."""

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class ModelSnapshot:
    """Ready model/distance and keyed assignments for the evaluated population.

    A strategy may refit or update existing state, and may retain old labels.
    The persistence coordinator does not choose that policy.
    """

    model: BaseEstimator
    distance: BaseEstimator
    labels: pd.DataFrame
