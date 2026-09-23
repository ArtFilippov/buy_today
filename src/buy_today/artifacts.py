"""Immutable artifact locations and preparation statistics; no file operations."""

from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict


@dataclass(frozen=True)
class ReportPaths:
    notebook_path: Path
    html_path: Path

    @property
    def metrics_path(self) -> Path:
        return self.notebook_path.with_name("metrics.json")


@dataclass(frozen=True)
class UpdateReports:
    deda: ReportPaths
    eda: ReportPaths


@dataclass(frozen=True)
class SummaryPaths:
    json_path: Path
    csv_path: Path
    html_path: Path


@dataclass(frozen=True)
class _DatasetResult:
    working_dataset_path: Path
    manifest_path: Path
    rows_before_cleaning: int
    rows_after_cleaning: int
    rows_after_category_filtering: int
    categories_before_filtering: int
    categories_after_filtering: int


@dataclass(frozen=True)
class PreparationResult(_DatasetResult):
    batch_sizes: tuple[int, ...]
    dropped_tail_rows: int


class PreparationStatistics(TypedDict):
    stages: dict[str, dict[str, int]]
    excluded_non_delivered_orders: int
    multi_seller_products: int
    missing_before_cleaning: dict[str, int]
    rows_dropped_missing: NotRequired[int]
    rows_dropped_rare_categories: NotRequired[int]
    categories_before_filtering: NotRequired[int]
    categories_after_filtering: NotRequired[int]
    dropped_tail_rows: NotRequired[int]
