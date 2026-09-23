"""Structural contracts for independent clustering evaluation and training."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

from buy_today.clustering import domain
from buy_today.clustering.domain import Parameterized

__all__ = [
    "FeatureMatrix", "IntegerArray", "FloatArray", "BooleanArray",
    "PairwiseDistance", "TrainableDistance", "Parameterized",
]

type FeatureMatrix = pd.DataFrame | NDArray[Any]
type IntegerArray = NDArray[np.integer[Any]]
type FloatArray = NDArray[np.float64]
type BooleanArray = NDArray[np.bool_]


PairwiseDistance = domain.PairwiseDistance[FeatureMatrix, ArrayLike]
TrainableDistance = domain.TrainableDistance[pd.DataFrame, FeatureMatrix, ArrayLike]
