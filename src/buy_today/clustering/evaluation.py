"""Schema-independent evaluation of existing labels with a new/old row quota."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class EvaluationResult:
    silhouette: float | None
    silhouette_reason: str | None
    sample_positions: np.ndarray
    labels: np.ndarray
    new_rows: np.ndarray
    distances: np.ndarray

    @property
    def new_count(self) -> int:
        return int(self.new_rows.sum())

    @property
    def previous_count(self) -> int:
        return len(self.new_rows) - self.new_count


def validate_evaluation_parameters(max_evaluation_rows, random_state) -> None:
    if (
        isinstance(max_evaluation_rows, (bool, np.bool_))
        or not isinstance(max_evaluation_rows, Integral)
        or max_evaluation_rows <= 0
    ):
        raise ValueError("max_evaluation_rows must be a positive integer")
    if (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, Integral)
        or not 0 <= random_state <= 2**32 - 1
    ):
        raise ValueError("random_state must be an integer in [0, 2**32 - 1]")


def _sample_positions(new_rows, limit, random_state):
    if len(new_rows) <= limit:
        return np.arange(len(new_rows))
    new = np.flatnonzero(new_rows)
    old = np.flatnonzero(~new_rows)
    n_new = min(len(new), (limit + 1) // 2)
    n_old = min(len(old), limit // 2)
    n_new += min(len(new) - n_new, limit - n_new - n_old)
    n_old += min(len(old) - n_old, limit - n_new - n_old)
    rng = np.random.default_rng(random_state)
    return np.sort(np.concatenate([
        rng.choice(new, size=n_new, replace=False),
        rng.choice(old, size=n_old, replace=False),
    ]))


def _check_distances(values, size):
    values = np.asarray(values)
    if values.shape != (size, size) or values.dtype.kind not in "iuf":
        raise ValueError("pairwise must return a real numeric square distance matrix")
    matrix = values.astype(np.float64, copy=False)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Distance matrix must be symmetric")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("Distance matrix must have a zero diagonal")
    return matrix


def evaluate_clustering(
    X, labels, *, distance, new_rows, max_evaluation_rows=1000, random_state=42,
) -> EvaluationResult:
    """Sample once, compute distances once, and score saved integer labels.

    X may be a DataFrame or a 2D ndarray. No Olist schema or model is required;
    distance is already ready for pairwise. Invalid inputs/errors propagate.
    """
    validate_evaluation_parameters(max_evaluation_rows, random_state)
    if not isinstance(X, (pd.DataFrame, np.ndarray)) or X.ndim != 2 or len(X) == 0:
        raise ValueError("X must be a nonempty two-dimensional DataFrame or ndarray")
    labels = np.asarray(labels)
    new_rows = np.asarray(new_rows)
    if labels.shape != (len(X),) or labels.dtype.kind not in "iu":
        raise ValueError("labels must be an integer array of shape (len(X),)")
    if new_rows.shape != (len(X),) or new_rows.dtype.kind != "b":
        raise ValueError("new_rows must be a boolean array of shape (len(X),)")
    positions = _sample_positions(new_rows, max_evaluation_rows, random_state)
    sample = X.iloc[positions] if isinstance(X, pd.DataFrame) else X[positions]
    sampled_labels = labels[positions]
    matrix = _check_distances(distance.pairwise(sample), len(positions))
    n_labels = len(np.unique(sampled_labels))
    reason = None
    score = None
    if not 2 <= n_labels < len(positions):
        reason = (
            "Силуэт не определён: требуется 2 <= число меток <= число строк - 1; "
            f"меток: {n_labels}, строк: {len(positions)}."
        )
    else:
        score = float(silhouette_score(matrix, sampled_labels, metric="precomputed"))
    return EvaluationResult(
        score, reason, positions, sampled_labels, new_rows[positions], matrix,
    )
