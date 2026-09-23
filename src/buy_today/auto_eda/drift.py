"""Deterministic distribution distances and threshold-based drift decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


from buy_today.auto_eda.domain import DriftResult, DriftThresholds


def ks_distance(reference: pd.Series[Any], batch: pd.Series[Any]) -> float:
    """Return KS D without computing a p-value.

    Args:
        reference (pd.Series[Any]): Finite reference sample.
        batch (pd.Series[Any]): Finite comparison sample.

    Returns:
        float: Maximum absolute difference between empirical CDFs.

    Raises:
        ValueError: Either sample is empty.
    """
    if reference.empty or batch.empty:
        raise ValueError("KS D: ожидаются две непустые выборки")
    reference_sorted = np.sort(reference)
    batch_sorted = np.sort(batch)
    values = np.union1d(reference_sorted, batch_sorted)
    reference_counts = np.searchsorted(reference_sorted, values, side="right")
    batch_counts = np.searchsorted(batch_sorted, values, side="right")
    # Subtract integer counts before division to avoid cancellation at thresholds.
    differences = reference_counts * len(batch) - batch_counts * len(reference)
    return float(np.abs(differences).max()) / (len(reference) * len(batch))


def total_variation_distance(reference: pd.Series[Any], batch: pd.Series[Any]) -> float:
    """Return TVD over every category in either sample.

    Args:
        reference (pd.Series[Any]): Reference categories without missing values.
        batch (pd.Series[Any]): Comparison categories without missing values.

    Returns:
        float: Half the absolute difference between category proportions.

    Raises:
        ValueError: Either sample is empty.
    """
    if reference.empty or batch.empty:
        raise ValueError("TVD: ожидаются две непустые выборки")
    reference_counts, batch_counts = reference.value_counts(sort=False).align(
        batch.value_counts(sort=False), join="outer", fill_value=0
    )
    differences = reference_counts * len(batch) - batch_counts * len(reference)
    return float(differences.abs().sum()) / (2 * len(reference) * len(batch))


DEFAULT_THRESHOLDS = DriftThresholds()


def evaluate_drift(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    *,
    thresholds: DriftThresholds = DEFAULT_THRESHOLDS,
) -> DriftResult[pd.DataFrame]:
    """Compare validated datasets; detected drift is a normal result.

    Args:
        reference (pd.DataFrame): Validated accumulated positions.
        batch (pd.DataFrame): Validated incoming positions.
        thresholds (DriftThresholds, default=DEFAULT_THRESHOLDS): Inclusive detection limits.

    Returns:
        DriftResult[pd.DataFrame]: Distances, limits and detection decisions for all features.
    """
    records: list[dict[str, str | float | bool]] = []
    for feature, measure, distance, threshold in (
        ("price", "KS D", ks_distance, thresholds.price),
        ("product_category_name", "TVD", total_variation_distance, thresholds.category),
        ("customer_state", "TVD", total_variation_distance, thresholds.state),
    ):
        value = distance(reference[feature], batch[feature])
        records.append(
            {
                "Признак": feature,
                "Мера": measure,
                "Значение": value,
                "Порог": threshold,
                "Дрейф": bool(value >= threshold),
            }
        )
    return DriftResult(metrics=pd.DataFrame(records))
