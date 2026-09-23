"""Distance-weighted sampling; no clustering or fitting is involved."""

from typing import Unpack

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from buy_today.auto_eda.checks import check_dataset
from buy_today.generation.parameters import check_integer, check_parameters
from buy_today.generation.records import (
    DEFAULT_SPLIT_SIZES,
    SPLITS,
    GeneratedHistories,
    Integer,
    PairwiseDistance,
    SamplingOptions,
    SamplingParameters,
)
from buy_today.schema import ROW_KEY


REAL_DTYPE_KINDS = "iuf"
__all__ = [
    "ANCHOR_DTYPES", "DEFAULT_SPLIT_SIZES", "EVENT_DTYPES", "GeneratedHistories",
    "SOURCE_DTYPES", "SPLITS", "check_integer", "check_parameters", "generate_histories",
    "validate_parameters",
]
SOURCE_DTYPES = {
    "batch_index": "int64",
    "source_position": "int64",
    "order_id": "string",
    "order_item_id": "int64",
    "product_id": "string",
    "order_purchase_timestamp": "datetime64[ns]",
}
ANCHOR_DTYPES = {
    "user_id": "string",
    "batch_index": "int64",
    "source_position": "int64",
    "order_id": "string",
    "order_item_id": "int64",
}
EVENT_DTYPES = {
    "event_id": "string",
    "user_id": "string",
    "event_index": "int64",
    "split": "string",
    "batch_index": "int64",
    "product_id": "string",
    "order_id": "string",
    "order_item_id": "int64",
    "order_purchase_timestamp": "datetime64[ns]",
}


def validate_parameters(parameters: SamplingParameters) -> None:
    check_parameters(
        n_users=parameters.n_users, batch_index=parameters.batch_index,
        temperature=parameters.temperature, random_state=parameters.random_state,
        split_sizes=parameters.split_sizes,
    )


def _probabilities(
    distance: PairwiseDistance, anchor: pd.DataFrame, batch: pd.DataFrame, temperature: float
) -> NDArray[np.float64]:
    values = np.asarray(distance.pairwise(anchor, batch))
    if values.shape != (1, len(batch)) or values.dtype.kind not in REAL_DTYPE_KINDS:
        raise ValueError("pairwise(anchor, batch) must return a real matrix of shape (1, n_rows)")
    distances: NDArray[np.float64] = values[0].astype(np.float64, copy=False)
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    # Subtracting the minimum leaves the normalized probabilities unchanged.
    # At least one weight is 1 even for tiny temperatures or large distances.
    with np.errstate(over="ignore", under="ignore"):
        weights: NDArray[np.float64] = np.exp(-(distances - np.min(distances)) / temperature)
    return weights / np.sum(weights)


def generate_histories(
    batch: pd.DataFrame,
    *,
    distance: PairwiseDistance,
    batch_index: Integer,
    n_users: Integer,
    temperature: float,
    **options: Unpack[SamplingOptions],
) -> GeneratedHistories[pd.DataFrame]:
    """Sample new users from this batch only, preserving sampling order.

    ``event_index`` is zero-based synthetic time within a user. Source timestamps
    are provenance, never the sorting/splitting clock. The ready distance needs
    only ``pairwise(X, Y)``; neither its fit nor any cluster model is consulted.
    Memory for distances is O(batch rows), not O(batch rows squared).

    Args:
        batch (pd.DataFrame): Validated source purchase rows for the new batch.
        distance (PairwiseDistance): Ready rectangular pairwise distance.
        batch_index (Integer): Nonnegative batch number used in IDs and RNG seeding.
        n_users (Integer): Positive number of synthetic users to create.
        temperature (float): Positive finite sampling temperature in distance units.
        **options (Unpack[SamplingOptions]): Split sizes (default 70/15/15) and
            random seed (default 42).

    Returns:
        GeneratedHistories[pd.DataFrame]: Events in sampling order and their user anchors.
    """
    parameters = SamplingParameters(
        n_users=n_users,
        batch_index=batch_index,
        temperature=temperature,
        **options,
    )
    validate_parameters(parameters)
    _ = check_dataset(batch)
    return _sample(batch, distance, parameters)


def _sample(
    batch: pd.DataFrame, distance: PairwiseDistance, parameters: SamplingParameters
) -> GeneratedHistories[pd.DataFrame]:
    rng: np.random.Generator = np.random.default_rng(
        np.random.SeedSequence([int(parameters.random_state), int(parameters.batch_index)])
    )
    positions = rng.integers(len(batch), size=parameters.n_users)
    anchors = _anchors(batch, positions, int(parameters.batch_index))
    selected = np.empty(
        (int(parameters.n_users), sum(int(size) for size in parameters.split_sizes)), dtype=np.int64
    )
    for index, position in enumerate(positions):
        probabilities = _probabilities(
            distance, batch.iloc[[position]], batch, parameters.temperature,
        )
        selected[index] = rng.choice(
            len(batch), size=selected.shape[1], replace=True, p=probabilities,
        )
    return GeneratedHistories(_events(batch, selected, anchors, parameters), anchors)


def _anchors(
    batch: pd.DataFrame, positions: NDArray[np.int64], batch_index: int
) -> pd.DataFrame:
    anchors: pd.DataFrame = batch.iloc[positions][list(ROW_KEY)].reset_index(drop=True)
    anchors["user_id"] = [f"user_{batch_index:06d}_{index:06d}" for index in range(len(positions))]
    anchors["batch_index"] = batch_index
    anchors["source_position"] = positions
    return anchors[list(ANCHOR_DTYPES)].astype(ANCHOR_DTYPES)


def _events(
    batch: pd.DataFrame, selected: NDArray[np.int64], anchors: pd.DataFrame,
    parameters: SamplingParameters,
) -> pd.DataFrame:
    events: pd.DataFrame = batch.iloc[selected.ravel()][
        ["product_id", *ROW_KEY, "order_purchase_timestamp"]
    ].reset_index(drop=True)
    length = selected.shape[1]
    user_ids = anchors["user_id"].tolist()
    events["event_id"] = [f"{user}:{index:06d}" for user in user_ids for index in range(length)]
    events["user_id"] = np.repeat(user_ids, length)
    events["event_index"] = np.tile(np.arange(length, dtype=np.int64), int(parameters.n_users))
    events["split"] = np.tile(
        np.repeat(SPLITS, [int(size) for size in parameters.split_sizes]), int(parameters.n_users),
    )
    events["batch_index"] = int(parameters.batch_index)
    return events[list(EVENT_DTYPES)].astype(EVENT_DTYPES)
