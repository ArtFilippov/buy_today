"""Schema-independent evaluation of existing labels with a new/old row quota."""

from typing import Any, Unpack, cast

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from sklearn.metrics import silhouette_score

from buy_today.clustering.contracts import BooleanArray, FloatArray, PairwiseDistance
from buy_today.clustering import domain
from buy_today.clustering.parameters import (
    EvaluationOptions,
    EvaluationParameters,
    validate_evaluation_parameters,
)

__all__ = ["EvaluationResult", "evaluate_clustering", "validate_evaluation_parameters"]

_MATRIX_DIMENSIONS = 2
_MIN_SILHOUETTE_LABELS = 2
_REAL_KINDS = frozenset("iuf")
_INTEGER_KINDS = frozenset("iu")
_BOOLEAN_KIND = "b"


EvaluationResult = domain.EvaluationResult[NDArray[np.intp], NDArray[Any], BooleanArray, FloatArray]


def _sample_positions(new_rows: BooleanArray, parameters: EvaluationParameters) -> NDArray[np.intp]:
    limit = parameters.max_evaluation_rows
    if len(new_rows) <= limit:
        return np.arange(len(new_rows))
    new = np.flatnonzero(new_rows)
    old = np.flatnonzero(~new_rows)
    n_new = min(len(new), (limit + 1) // 2)
    n_old = min(len(old), limit // 2)
    n_new += min(len(new) - n_new, limit - n_new - n_old)
    n_old += min(len(old) - n_old, limit - n_new - n_old)
    rng = np.random.default_rng(parameters.random_state)
    return np.sort(
        np.concatenate(
            [
                Generator.choice(rng, new, size=n_new, replace=False),
                Generator.choice(rng, old, size=n_old, replace=False),
            ]
        )
    )


def _check_distances(values: ArrayLike, size: int) -> FloatArray:
    values = np.asarray(values)
    if values.shape != (size, size) or values.dtype.kind not in _REAL_KINDS:
        raise ValueError("pairwise must return a real numeric square distance matrix")
    matrix = np.asarray(values, dtype=np.float64)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Distance matrix must be symmetric")
    if np.any(np.diag(matrix)):
        raise ValueError("Distance matrix must have a zero diagonal")
    return matrix


def _checked_features(values: object) -> pd.DataFrame | NDArray[Any]:
    matrix: pd.DataFrame | NDArray[Any]
    if isinstance(values, pd.DataFrame):
        matrix = values
    elif isinstance(values, np.ndarray):
        matrix = cast(NDArray[Any], values)
    else:
        raise ValueError("X must be a nonempty two-dimensional DataFrame or ndarray")
    if matrix.ndim != _MATRIX_DIMENSIONS or matrix.shape[0] < 1:
        raise ValueError("X must be a nonempty two-dimensional DataFrame or ndarray")
    return matrix


def _checked_assignments(
    labels: ArrayLike, new_rows: ArrayLike, size: int
) -> tuple[NDArray[Any], BooleanArray]:
    labels = np.asarray(labels)
    new_rows = np.asarray(new_rows)
    if labels.shape != (size,) or labels.dtype.kind not in _INTEGER_KINDS:
        raise ValueError("labels must be an integer array of shape (len(X),)")
    if new_rows.shape != (size,) or new_rows.dtype.kind != _BOOLEAN_KIND:
        raise ValueError("new_rows must be a boolean array of shape (len(X),)")
    return labels, new_rows


def _silhouette(matrix: FloatArray, labels: NDArray[Any]) -> tuple[float | None, str | None]:
    n_labels = len(np.unique(labels))
    if not _MIN_SILHOUETTE_LABELS <= n_labels < len(labels):
        return None, (
            "Силуэт не определён: требуется 2 <= число меток <= число строк - 1; "
            + f"меток: {n_labels}, строк: {len(labels)}."
        )
    return float(silhouette_score(matrix, labels, metric="precomputed")), None


def evaluate_clustering(
    X: object,
    labels: ArrayLike,
    *,
    distance: PairwiseDistance,
    new_rows: ArrayLike,
    **options: Unpack[EvaluationOptions],
) -> EvaluationResult:
    """Sample once, compute distances once, and score saved integer labels.

    X may be a DataFrame or a 2D ndarray. No Olist schema or model is required;
    distance is already ready for pairwise. Invalid inputs/errors propagate.

    Args:
        X (object): Nonempty DataFrame or two-dimensional ndarray.
        labels (ArrayLike): Saved integer labels in input order.
        distance (PairwiseDistance): Ready pairwise distance implementation.
        new_rows (ArrayLike): Boolean membership mask in input order.
        **options (Unpack[EvaluationOptions]): Maximum sample size via
            ``max_evaluation_rows`` (default 1000) and reproducible sampling seed
            via ``random_state`` (default 42).

    Returns:
        EvaluationResult: Sample, pairwise distances and silhouette or its unavailable reason.
    """
    parameters = EvaluationParameters(**options)
    features = _checked_features(X)
    labels, new_rows = _checked_assignments(labels, new_rows, len(features))
    positions = _sample_positions(new_rows, parameters)
    sample = features.iloc[positions] if isinstance(features, pd.DataFrame) else features[positions]
    labels = labels[positions]
    matrix = _check_distances(distance.pairwise(sample), len(positions))
    return EvaluationResult(
        *_silhouette(matrix, labels),
        sample_positions=positions,
        labels=labels,
        new_rows=new_rows[positions],
        distances=matrix,
    )
