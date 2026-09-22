"""Deterministic distribution distances and threshold-based drift decisions."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftThresholds:
    price: float = 0.10
    category: float = 0.10
    state: float = 0.10

    def __post_init__(self) -> None:
        for name in ("price", "category", "state"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not 0 <= value <= 1 or not isfinite(value):
                raise ValueError(
                    f"Порог {name}: фактически {value!r}; "
                    "ожидается конечное число в [0, 1]"
                )


@dataclass(frozen=True)
class DriftResult:
    metrics: pd.DataFrame

    @property
    def drift_detected(self) -> bool:
        return bool(self.metrics["Дрейф"].any())


def ks_distance(reference: pd.Series, batch: pd.Series) -> float:
    """Return KS D for nonempty, finite samples, without computing a p-value."""
    if len(reference) == 0 or len(batch) == 0:
        raise ValueError("KS D: ожидаются две непустые выборки")
    reference_sorted = np.sort(reference)
    batch_sorted = np.sort(batch)
    values = np.union1d(reference_sorted, batch_sorted)
    reference_counts = np.searchsorted(reference_sorted, values, side="right")
    batch_counts = np.searchsorted(batch_sorted, values, side="right")
    # Subtract integer counts before division to avoid cancellation at thresholds.
    differences = reference_counts * len(batch) - batch_counts * len(reference)
    return float(np.abs(differences).max()) / (len(reference) * len(batch))


def total_variation_distance(reference: pd.Series, batch: pd.Series) -> float:
    """Return TVD over all values in two nonempty samples without missing values."""
    if len(reference) == 0 or len(batch) == 0:
        raise ValueError("TVD: ожидаются две непустые выборки")
    reference_counts, batch_counts = reference.value_counts(sort=False).align(
        batch.value_counts(sort=False), join="outer", fill_value=0
    )
    differences = reference_counts * len(batch) - batch_counts * len(reference)
    return float(differences.abs().sum()) / (2 * len(reference) * len(batch))


def evaluate_drift(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    *,
    thresholds: DriftThresholds = DriftThresholds(),
) -> DriftResult:
    """Compare datasets after check_dataset; detected drift is a normal result."""
    records = []
    for feature, measure, distance, threshold in (
        ("price", "KS D", ks_distance, thresholds.price),
        ("product_category_name", "TVD", total_variation_distance, thresholds.category),
        ("customer_state", "TVD", total_variation_distance, thresholds.state),
    ):
        value = distance(reference[feature], batch[feature])
        records.append({
            "Признак": feature, "Мера": measure, "Значение": value,
            "Порог": threshold, "Дрейф": bool(value >= threshold),
        })
    return DriftResult(metrics=pd.DataFrame(records))
