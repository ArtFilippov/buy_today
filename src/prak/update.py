"""Coordinate the physical reference CSV and its reports."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from prak.auto_eda import DriftThresholds, report_dataset, report_drift
from prak.auto_eda.checks import check_dataset
from prak.auto_eda.notebook import ReportPaths
from prak.schema import DATE_FORMAT, SORT_KEY, read_dataset


@dataclass(frozen=True)
class UpdateReports:
    deda: ReportPaths
    eda: ReportPaths


def initialize_reference(
    batch_path: Path | str, reference_path: Path | str, output_dir: Path | str
) -> ReportPaths:
    """Validate a batch, create/replace the reference, then report the saved CSV.

    Invalid input leaves an existing reference untouched. If EDA fails after
    writing, the new reference remains on disk and the error propagates.
    """
    batch = read_dataset(batch_path)
    check_dataset(batch)
    reference_path = Path(reference_path).resolve()
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    batch.sort_values(list(SORT_KEY)).to_csv(
        reference_path, index=False, encoding="utf-8", date_format=DATE_FORMAT,
    )
    return report_dataset(reference_path, Path(output_dir) / "eda")


def update_reference(
    batch_path: Path | str,
    reference_path: Path | str,
    output_dir: Path | str,
    *,
    thresholds: DriftThresholds = DriftThresholds(),
) -> UpdateReports:
    """Report drift, append the batch to an existing reference, then report it.

    DEDA must finish, including HTML export, before the reference is written.
    Detected drift permits the update. If the final EDA fails, the updated
    reference remains on disk and the error propagates.
    """
    reference_path = Path(reference_path).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Эталон CSV не найден: {reference_path}")
    output_dir = Path(output_dir)
    deda = report_drift(
        batch_path, reference_path, output_dir / "deda", thresholds=thresholds,
    )
    reference = read_dataset(reference_path)
    batch = read_dataset(batch_path)
    pd.concat([reference, batch], ignore_index=True).sort_values(list(SORT_KEY)).to_csv(
        reference_path, index=False, encoding="utf-8", date_format=DATE_FORMAT,
    )
    eda = report_dataset(reference_path, output_dir / "eda")
    return UpdateReports(deda=deda, eda=eda)
