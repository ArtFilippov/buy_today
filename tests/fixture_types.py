"""Shared contracts for pytest's injected fixtures and fixture factories."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pandas import DataFrame
from pytest import CaptureFixture, LogCaptureFixture, MonkeyPatch

from buy_today.ranking.data import RankingData

__all__ = [
    "Path",
    "DataFrame",
    "CaptureFixture",
    "LogCaptureFixture",
    "MonkeyPatch",
    "RankingData",
    "RawTables",
    "CategoryTables",
    "RawWriter",
    "DatasetWriter",
]

type RawTables = dict[str, DataFrame]
type CategoryTables = Callable[[dict[str, int]], RawTables]
type RawWriter = Callable[[RawTables], Path]


class DatasetWriter(Protocol):
    def __call__(self, frame: DataFrame, name: str = "input ' batch.csv") -> Path: ...
