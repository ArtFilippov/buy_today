"""Pairwise metric callbacks receive rows and optional metric-specific keywords."""

from collections.abc import Callable
from typing import Any
import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray
from sklearn._typing import Int, MatrixLike

type _Metric = str | Callable[..., float | np.floating[Any]]

def check_number_of_labels(n_labels: Int, n_samples: Int) -> None: ...
def silhouette_score(X: MatrixLike | ArrayLike, labels: ArrayLike, *, metric: _Metric = "euclidean",
                     sample_size: Int | None = None, random_state: RandomState | Int | None = None,
                     **kwds: Any) -> float: ...
def silhouette_samples(X: MatrixLike | ArrayLike, labels: ArrayLike, *, metric: _Metric = "euclidean",
                       **kwds: Any) -> NDArray[np.floating[Any]]: ...
def calinski_harabasz_score(X: MatrixLike | ArrayLike, labels: ArrayLike) -> float: ...
def davies_bouldin_score(X: MatrixLike | ArrayLike, labels: ArrayLike) -> float: ...
