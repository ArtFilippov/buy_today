"""Dataframe specializations and compatibility exports of history records."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from buy_today.generation import domain
from buy_today.generation.domain import (
    BatchRecord,
    DatasetOptions,
    DatasetSettings,
    DistanceRecord,
    GeneratedHistories,
    GenerationPaths,
    HistoryManifest,
    SamplingOptions,
    SamplingParameters,
    SnapshotInputs,
    SourceRecord,
)
from buy_today.generation.parameters import DEFAULT_SPLIT_SIZES, SPLITS

type Integer = int | np.integer[Any]
type SplitSizes = Sequence[Integer]

HistoryDataset = domain.HistoryDataset[pd.DataFrame]
PairwiseDistance = domain.PairwiseDistance[pd.DataFrame, ArrayLike]

__all__ = [
    "BatchRecord", "DatasetOptions", "DatasetSettings", "DEFAULT_SPLIT_SIZES",
    "DistanceRecord", "GeneratedHistories", "GenerationPaths", "HistoryDataset",
    "HistoryManifest", "Integer", "PairwiseDistance", "SamplingOptions", "SamplingParameters",
    "SnapshotInputs", "SourceRecord", "SPLITS", "SplitSizes",
]
