"""Event-weighted frequencies and equally weighted user concentration diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from buy_today.auto_eda.drift import total_variation_distance
from buy_today.generation.parameters import check_integer
from buy_today.history_quality.domain import HistoryQuality, IIDBaseline
from buy_today.history_quality.inputs import CATEGORY, categorized_events


IID_CHUNK_USERS = 1024


def _user_statistics(events: pd.DataFrame) -> pd.DataFrame:
    counts: pd.Series[int] = events.groupby(["user_id", CATEGORY], sort=True, observed=True).size()
    lengths = events.groupby("user_id", sort=True).size()
    shares = counts / counts.groupby(level=0).transform("sum")
    matches = events.groupby("user_id", sort=True)["anchor_match"].sum()
    return pd.DataFrame({
        "length": lengths,
        "top": shares.groupby(level=0).max(),
        "effective": 1.0 / shares.pow(2).groupby(level=0).sum(),
        "anchor": matches / lengths,
    })


def _distribution(
    values: NDArray[np.float64], prefix: str, quantiles: tuple[int, ...],
) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        **{
            f"{prefix}_p{quantile}": float(np.quantile(values, quantile / 100, method="linear"))
            for quantile in quantiles
        },
    }


def _threshold_shares(
    values: NDArray[np.float64], prefix: str, thresholds: tuple[int, ...],
) -> dict[str, float]:
    return {
        f"{prefix}_ge_0_{threshold:02d}": float(np.mean(values >= threshold / 100))
        for threshold in thresholds
    }


def _concentration_metrics(users: pd.DataFrame) -> dict[str, float]:
    top = users["top"].to_numpy(dtype=np.float64)
    effective = users["effective"].to_numpy(dtype=np.float64)
    anchor = users["anchor"].to_numpy(dtype=np.float64)
    return {
        **_distribution(top, "user_top_category_share", (50, 90, 95, 99)),
        **_threshold_shares(top, "users_top_category_share", (80, 90, 95, 99)),
        **_distribution(effective, "user_effective_category_count", (10, 50)),
        **_distribution(anchor, "user_anchor_category_share", (95,)),
        **_threshold_shares(anchor, "users_anchor_category_share", (95, 99)),
    }


def _iid_means(
    lengths: NDArray[np.int64], probabilities: NDArray[np.float64], repeats: int, seed: int,
) -> tuple[float, float]:
    rng: np.random.Generator = np.random.Generator(np.random.PCG64(seed))
    totals = np.zeros(2, dtype=np.float64)
    # Multinomial counts have exactly the law of IID categorical events. Chunking
    # bounds memory independently of history length and the number of repeats.
    for _ in range(repeats):
        for start in range(0, len(lengths), IID_CHUNK_USERS):
            sizes = lengths[start : start + IID_CHUNK_USERS]
            counts = rng.multinomial(sizes, probabilities)
            shares = counts / sizes[:, None]
            totals[0] += np.max(shares, axis=1).sum()
            totals[1] += (1.0 / np.square(shares).sum(axis=1)).sum()
    totals /= len(lengths) * repeats
    return float(totals[0]), float(totals[1])


def _baseline(
    users: pd.DataFrame, reference: pd.DataFrame, repeats: int, seed: int,
) -> IIDBaseline:
    frequencies = reference[CATEGORY].value_counts(sort=False).sort_index()
    top, effective = _iid_means(
        users["length"].to_numpy(dtype=np.int64),
        np.asarray(frequencies / len(reference), dtype=np.float64), repeats, seed,
    )
    return {
        "repeats": repeats,
        "random_state": seed,
        "rng": "numpy.random.Generator(PCG64)",
        "sampling": (
            "multinomial; categories and user IDs sorted ascending; repeats then users; "
            f"chunks of {IID_CHUNK_USERS} users"
        ),
        "aggregation": "synthetic user mean / IID mean over all users and repeats",
        "user_top_category_share_mean": top,
        "user_effective_category_count_mean": effective,
    }


def evaluate_history_quality(
    events: pd.DataFrame, anchors: pd.DataFrame, reference: pd.DataFrame,
    *, iid_repeats: int = 100, random_state: int = 42,
) -> HistoryQuality:
    """Measure all supplied events, counting repeats and weighting users equally.

    Categories are joined by source row key, never by product. Event/anchor
    tables require user_id and both source-key columns. Reference requires both
    key columns and product_category_name. Inputs are not modified. This core
    supports unequal positive history lengths; the file adapter additionally
    validates the complete snapshot and the exact reference source-key set.

    Args:
        events (pd.DataFrame): One row per synthetic event, including repeats.
        anchors (pd.DataFrame): Exactly one source anchor per event user.
        reference (pd.DataFrame): Unique source rows with nonblank categories.
        iid_repeats (int, default=100): Positive number of independent IID cohorts.
        random_state (int, default=42): PCG64 seed in [0, 2**32 - 1].

    Returns:
        HistoryQuality: Counts, descriptive metrics and reproducible IID settings/means.
    """
    check_integer(iid_repeats, "iid_repeats", 1)
    check_integer(random_state, "random_state", maximum=2**32 - 1)
    categorized = categorized_events(events, anchors, reference)
    users = _user_statistics(categorized)
    baseline = _baseline(users, reference, int(iid_repeats), int(random_state))
    metrics = _concentration_metrics(users)
    metrics.update({
        "category_frequency_tvd": total_variation_distance(
            reference[CATEGORY], categorized[CATEGORY],
        ),
        "anchor_category_event_share": float(categorized["anchor_match"].mean()),
        "top_category_share_lift_vs_iid": (
            metrics["user_top_category_share_mean"] / baseline["user_top_category_share_mean"]
        ),
        "effective_category_count_ratio_vs_iid": (
            metrics["user_effective_category_count_mean"]
            / baseline["user_effective_category_count_mean"]
        ),
    })
    return {
        "n_users": len(users), "n_events": len(events), "n_reference_rows": len(reference),
        "metrics": metrics, "iid": baseline,
    }
