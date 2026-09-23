"""Canonical benchmark configurations and their shell-quoted syntax."""

import argparse
from dataclasses import dataclass
import shlex
from typing import NoReturn, override

from buy_today.ranking.random import RandomRanker
from buy_today.ranking.svd import SVDRanker
from buy_today.ranking.domain import SVD

RANKERS = {"random": RandomRanker, "svd": SVDRanker}
__all__ = ["ModelConfiguration", "SVD", "parse_model_spec"]
SEED = "random_state"


class _ModelParser(argparse.ArgumentParser):
    @override
    def error(self, message: str) -> NoReturn:
        raise ValueError(f"{self.prog}: {message}")


def _positive_int(value: str) -> int:
    if (number := int(value)) < 1:
        raise ValueError("expected a positive integer")
    return number


def _seed(value: str) -> int:
    number = int(value)
    if not 0 <= number < 2**32:
        raise ValueError("expected an integer in [0, 2**32 - 1]")
    return number


@dataclass(frozen=True)
class ModelConfiguration:
    algorithm: str
    parameters: dict[str, int]

    @property
    def name(self) -> str:
        defaults = RANKERS[self.algorithm]().get_params(deep=False)
        return self.algorithm + "".join(
            f"_{key.replace('_', '-')}-{value}"
            for key, value in sorted(self.parameters.items())
            if value != defaults[key]
        )

    def create_ranker(self) -> RandomRanker | SVDRanker:
        return RANKERS[self.algorithm](**self.parameters)


def parse_model_spec(spec: str) -> ModelConfiguration:
    """Parse a shell-quoted model and its own options without exiting the process.

    Args:
        spec (str): Algorithm name followed by its command-line options.

    Returns:
        ModelConfiguration: Effective parameters and canonical model identity.

    Raises:
        ValueError: The algorithm or its options are invalid.
    """
    tokens = shlex.split(spec)
    if not tokens or tokens[0] not in RANKERS:
        raise ValueError("Expected model random or svd")
    algorithm = tokens[0]
    parser = _ModelParser(prog=algorithm, add_help=False, allow_abbrev=False)
    for name, default in RANKERS[algorithm]().get_params(deep=False).items():
        _ = parser.add_argument(
            f"--{name.replace('_', '-')}",
            default=default,
            type=_seed if name == SEED else _positive_int,
        )
    return ModelConfiguration(algorithm, vars(parser.parse_args(tokens[1:])))
