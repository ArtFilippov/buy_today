"""Generic distance contracts and clustering values without numerical imports."""

from dataclasses import dataclass
from typing import Any, Protocol, Self, SupportsInt


class PairwiseDistance[Features, Matrix](Protocol):
    def pairwise(self, X: Features, /) -> Matrix: ...


class TrainableDistance[Training, Features, Matrix](PairwiseDistance[Features, Matrix], Protocol):
    def fit(self, X: Training, /) -> Self: ...


class Parameterized(Protocol):
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ModelSnapshot[Model, Distance, Table]:
    model: Model
    distance: Distance
    labels: Table


@dataclass(frozen=True)
class EvaluationResult[Positions, Labels, Mask: CountableMask, Matrix]:
    silhouette: float | None
    silhouette_reason: str | None
    sample_positions: Positions
    labels: Labels
    new_rows: Mask
    distances: Matrix

    @property
    def new_count(self) -> int:
        return int(self.new_rows.sum())

    @property
    def previous_count(self) -> int:
        return len(self.new_rows) - self.new_count


class CountableMask(Protocol):
    def sum(self) -> SupportsInt: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class ProjectionResult[Matrix]:
    coordinates: Matrix | None
    parameters: dict[str, Any]
    reason: str | None
