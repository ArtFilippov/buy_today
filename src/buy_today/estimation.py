"""Library-independent estimator lifecycle shared by numerical adapters."""

from typing import Any, Protocol, Self


class Fittable[Data](Protocol):
    """Fluent fitting delegates state updates to the estimator's training hook."""

    def fit(self, X: Data, y: object = None) -> Self:
        self._fit(X, y)
        return self

    def _fit(self, X: Data, y: object = None) -> None: ...

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...

    def set_params(self, **params: Any) -> Self: ...


class Distance[Data, Matrix](Fittable[Data], Protocol):
    @staticmethod
    def pairwise(X: Data, Y: Data | None = None) -> Matrix: ...


class Clusterable[Data, Labels](Fittable[Data], Protocol):
    def fit_predict(self, X: Data, y: object = None, **kwargs: Any) -> Labels: ...


class Ranker[Data, Prediction](Fittable[Data], Protocol):
    def predict(self, user_id: object, k: object) -> Prediction: ...
