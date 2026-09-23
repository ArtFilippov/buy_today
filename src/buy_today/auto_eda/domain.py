"""Library-independent drift decision parameters."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Protocol


@dataclass(frozen=True)
class DriftThresholds:
    price: float = 0.10
    category: float = 0.10
    state: float = 0.10

    def __post_init__(self) -> None:
        for name in ("price", "category", "state"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not 0 <= float(value) <= 1 or not isfinite(value):
                raise ValueError(
                    f"Порог {name}: фактически {value!r}; ожидается конечное число в [0, 1]"
                )


class Decisions(Protocol):
    def any(self) -> object: ...


class DriftMetrics(Protocol):
    def __getitem__(self, key: str, /) -> Decisions: ...


@dataclass(frozen=True)
class DriftResult[Table: DriftMetrics = Any]:
    metrics: Table

    @property
    def drift_detected(self) -> bool:
        return bool(self.metrics["Дрейф"].any())
