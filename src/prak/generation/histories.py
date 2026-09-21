"""Distance-weighted sampling; no clustering or fitting is involved."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from prak.auto_eda.checks import check_dataset
from prak.schema import ROW_KEY


SPLITS = ("train", "validation", "test")
DEFAULT_SPLIT_SIZES = (70, 15, 15)
SOURCE_DTYPES = {
    "batch_index": "int64", "source_position": "int64",
    "order_id": "string", "order_item_id": "int64", "product_id": "string",
    "order_purchase_timestamp": "datetime64[ns]",
}
ANCHOR_DTYPES = {
    "user_id": "string", "batch_index": "int64", "source_position": "int64",
    "order_id": "string", "order_item_id": "int64",
}
EVENT_DTYPES = {
    "event_id": "string", "user_id": "string", "event_index": "int64",
    "split": "string", "batch_index": "int64", "product_id": "string",
    "order_id": "string", "order_item_id": "int64",
    "order_purchase_timestamp": "datetime64[ns]",
}


@dataclass(frozen=True)
class GeneratedHistories:
    events: pd.DataFrame
    anchors: pd.DataFrame


def check_integer(value, name, minimum=0, maximum=None):
    if (
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        or value < minimum or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum or 'inf'}]")


def check_parameters(*, n_users, batch_index, temperature, random_state, split_sizes):
    check_integer(n_users, "n_users", 1)
    check_integer(batch_index, "batch_index")
    check_integer(random_state, "random_state", maximum=2**32 - 1)
    if (
        isinstance(temperature, (bool, np.bool_)) or not isinstance(temperature, Real)
        or not np.isfinite(temperature) or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive, in distance units")
    if not isinstance(split_sizes, (tuple, list)) or len(split_sizes) != 3:
        raise ValueError("split_sizes must contain train, validation and test sizes")
    for split, size in zip(SPLITS, split_sizes):
        check_integer(size, f"{split} size", 1)


def _probabilities(distance, anchor, batch, temperature):
    values = np.asarray(distance.pairwise(anchor, batch))
    if values.shape != (1, len(batch)) or values.dtype.kind not in "iuf":
        raise ValueError("pairwise(anchor, batch) must return a real matrix of shape (1, n_rows)")
    distances = values[0].astype(np.float64, copy=False)
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    # Subtracting the minimum leaves the normalized probabilities unchanged.
    # At least one weight is 1 even for tiny temperatures or large distances.
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-(distances - distances.min()) / temperature)
    return weights / weights.sum()


def generate_histories(
    batch: pd.DataFrame, *, distance, batch_index, n_users, temperature,
    split_sizes=DEFAULT_SPLIT_SIZES, random_state=42,
) -> GeneratedHistories:
    """Sample new users from this batch only, preserving sampling order.

    ``event_index`` is zero-based synthetic time within a user. Source timestamps
    are provenance, never the sorting/splitting clock. The ready distance needs
    only ``pairwise(X, Y)``; neither its fit nor any cluster model is consulted.
    Memory for distances is O(batch rows), not O(batch rows squared).
    """
    check_parameters(
        n_users=n_users, batch_index=batch_index, temperature=temperature,
        random_state=random_state, split_sizes=split_sizes,
    )
    check_dataset(batch)
    rng = np.random.default_rng(np.random.SeedSequence([random_state, batch_index]))
    anchor_positions = rng.integers(len(batch), size=n_users)
    user_ids = [f"user_{batch_index:06d}_{index:06d}" for index in range(n_users)]
    anchors = batch.iloc[anchor_positions][list(ROW_KEY)].reset_index(drop=True)
    anchors = anchors.assign(
        user_id=user_ids, batch_index=batch_index, source_position=anchor_positions,
    )[list(ANCHOR_DTYPES)].astype(ANCHOR_DTYPES)

    length = sum(split_sizes)
    selected = np.empty((n_users, length), dtype=np.int64)
    for index, position in enumerate(anchor_positions):
        probabilities = _probabilities(distance, batch.iloc[[position]], batch, temperature)
        selected[index] = rng.choice(len(batch), size=length, replace=True, p=probabilities)
    events = batch.iloc[selected.ravel()][
        ["product_id", *ROW_KEY, "order_purchase_timestamp"]
    ].reset_index(drop=True)
    events = events.assign(
        event_id=[f"{user}:{index:06d}" for user in user_ids for index in range(length)],
        user_id=np.repeat(user_ids, length), event_index=np.tile(np.arange(length), n_users),
        split=np.tile(np.repeat(SPLITS, split_sizes), n_users), batch_index=batch_index,
    )[list(EVENT_DTYPES)].astype(EVENT_DTYPES)
    return GeneratedHistories(events, anchors)
