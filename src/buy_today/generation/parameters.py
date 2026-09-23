"""Sampling invariants expressed without a numerical-library dependency."""

from math import isfinite
from numbers import Integral, Real
from collections.abc import Sequence

SPLITS = ("train", "validation", "test")
DEFAULT_SPLIT_SIZES = (70, 15, 15)


def check_integer(
    value: object, name: str, minimum: int = 0, maximum: int | None = None
) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum or 'inf'}]")
    integer = int(value)
    if integer < minimum or (maximum is not None and integer > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum or 'inf'}]")


def check_parameters(
    *, n_users: object, batch_index: object, temperature: object,
    random_state: object, split_sizes: Sequence[object]
) -> None:
    check_integer(n_users, "n_users", 1)
    check_integer(batch_index, "batch_index")
    check_integer(random_state, "random_state", maximum=2**32 - 1)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive, in distance units")
    if not isinstance(split_sizes, (tuple, list)) or len(split_sizes) != len(SPLITS):
        raise ValueError("split_sizes must contain train, validation and test sizes")
    for split, size in zip(SPLITS, split_sizes):
        check_integer(size, f"{split} size", 1)
