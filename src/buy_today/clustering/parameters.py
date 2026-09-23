"""Validated sampling parameters and the keyword options accepted by public APIs."""

from dataclasses import dataclass
from numbers import Integral
from typing import TypedDict


class EvaluationOptions(TypedDict, total=False):
    max_evaluation_rows: int
    random_state: int


class ReportOptions(EvaluationOptions, total=False):
    thread_limit: int | None


def validate_evaluation_parameters(max_evaluation_rows: object, random_state: object) -> None:
    if (
        isinstance(max_evaluation_rows, bool)
        or not isinstance(max_evaluation_rows, Integral)
        or max_evaluation_rows <= 0
    ):
        raise ValueError("max_evaluation_rows must be a positive integer")
    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, Integral)
        or not 0 <= int(random_state) <= 2**32 - 1
    ):
        raise ValueError("random_state must be an integer in [0, 2**32 - 1]")


@dataclass(frozen=True)
class EvaluationParameters:
    max_evaluation_rows: int = 1000
    random_state: int = 42

    def __post_init__(self) -> None:
        validate_evaluation_parameters(self.max_evaluation_rows, self.random_state)


@dataclass(frozen=True)
class ReportParameters(EvaluationParameters):
    thread_limit: int | None = None
