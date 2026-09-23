"""Argument converters and shared pipeline overrides for both CLI adapters."""

import argparse
from dataclasses import fields
from enum import StrEnum
from math import isfinite
from typing import Any

from buy_today.pipeline_config import PipelineConfig
from buy_today.ranking.model_spec import parse_model_spec


class Command(StrEnum):
    RUN = "run"
    INFERENCE = "inference"
    SUMMARY = "summary"
    PREPARE = "prepare"
    EDA = "eda"
    INIT = "init"
    DEDA = "deda"
    UPDATE = "update"
    CLUSTER = "cluster"
    GENERATE = "generate"
    RANK = "rank"
    EVALUATE_RANKING = "evaluate-ranking"
    BENCHMARK = "benchmark-ranking"
    BENCHMARK_REPORT = "benchmark-ranking-report"
    EVALUATE = "evaluate"


SVD = "svd"
SVD_OPTIONS_ERROR = "--svd-n-components и --svd-n-iter требуют --model svd"
THRESHOLDS = "thresholds"
SPLIT_SIZES = "split_sizes"
MAX_RANDOM_STATE = 2**32 - 1


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ожидается положительное целое число") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("ожидается положительное целое число")
    return number


def drift_threshold(value: str) -> float:
    message = "ожидается конечное число в [0, 1]"
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(message) from exc
    if not isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError(message)
    return number


def random_state(value: str) -> int:
    message = "ожидается целое число в [0, 2**32 - 1]"
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(message) from exc
    if not 0 <= number <= MAX_RANDOM_STATE:
        raise argparse.ArgumentTypeError(message)
    return number


def positive_float(value: str) -> float:
    message = "ожидается конечное положительное число"
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(message) from exc
    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(message)
    return number


def benchmark_model(value: str) -> str:
    try:
        _ = parse_model_spec(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def pipeline_options(args: argparse.Namespace) -> dict[str, Any]:
    # argparse owns this heterogeneous boundary; dataclass construction validates it.
    options = {
        field.name: getattr(args, field.name)
        for field in fields(PipelineConfig)
        if field.name != THRESHOLDS and getattr(args, field.name) is not None
    }
    if SPLIT_SIZES in options:
        options[SPLIT_SIZES] = tuple(options[SPLIT_SIZES])
    return options


def threshold_options(args: argparse.Namespace) -> dict[str, float]:
    return {
        name: getattr(args, f"{name}_threshold")
        for name in ("price", "category", "state")
        if getattr(args, f"{name}_threshold") is not None
    }


def svd_options_requested(args: argparse.Namespace) -> bool:
    return args.svd_n_components is not None or args.svd_n_iter is not None
