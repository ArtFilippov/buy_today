"""Library-independent history records and distance-sampling contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, SupportsIndex, TypedDict

from buy_today.generation.parameters import DEFAULT_SPLIT_SIZES

type Integer = SupportsIndex
type SplitSizes = Sequence[Integer]


class PairwiseDistance[Table, Matrix](Protocol):
    def pairwise(self, left: Table, right: Table, /) -> Matrix: ...


class SamplingOptions(TypedDict, total=False):
    split_sizes: SplitSizes
    random_state: Integer


class DatasetOptions(TypedDict, total=False):
    n_users: Integer | None
    split_sizes: SplitSizes | None
    random_state: Integer


@dataclass(frozen=True, kw_only=True)
class DatasetSettings:
    n_users: Integer | None = None
    split_sizes: SplitSizes | None = None
    random_state: Integer = 42


@dataclass(frozen=True, kw_only=True)
class SamplingParameters:
    n_users: Integer
    batch_index: Integer
    temperature: float
    random_state: Integer = 42
    split_sizes: SplitSizes = DEFAULT_SPLIT_SIZES


@dataclass(frozen=True)
class GeneratedHistories[Table = Any]:
    events: Table
    anchors: Table


@dataclass(frozen=True)
class GenerationPaths:
    output_dir: Path
    manifest_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    catalog_path: Path


class SourceRecord(TypedDict):
    path: str
    sha256: str
    rows: int


# Functional syntax retains the persisted JSON key named after Python's keyword.
DistanceRecord = TypedDict(
    "DistanceRecord",
    {"path": str, "sha256": str, "snapshot": str, "class": str, "parameters": str | None},
)


class BatchRecord(TypedDict):
    batch_index: int
    n_users: int
    temperature: float
    random_state: int
    source: SourceRecord
    distance: DistanceRecord


class HistoryManifest(TypedDict):
    format_version: int
    split_sizes: list[int]
    synthetic_time: str
    rng: str
    numpy_version: str
    batches: list[BatchRecord]
    files: dict[str, str]


@dataclass(frozen=True)
class HistoryDataset[Table]:
    events: Table
    anchors: Table
    sources: Table
    catalog: Table
    manifest: HistoryManifest


@dataclass(frozen=True)
class SnapshotInputs:
    batch_path: Path
    distance_path: Path
    output_dir: Path
    previous_dir: Path | None
