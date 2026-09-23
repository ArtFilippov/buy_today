"""Build the working Olist dataset and its chronological batch stream."""

import json
from pathlib import Path
from typing import TypedDict

import pandas as pd

from buy_today.preparation_sources import read_sources, source_metadata
from buy_today.preparation_tables import PreparationStatistics, assemble, clean
from buy_today.progress import stage
from buy_today.schema import COLUMNS, DATE_FORMAT, DTYPES, SORT_KEY
from buy_today.artifacts import PreparationResult


class _FileMetadata(TypedDict):
    path: str
    rows: int
    period_start: str
    period_end: str


class _BatchMetadata(_FileMetadata):
    id: str


def _file_metadata(frame: pd.DataFrame, path: str) -> _FileMetadata:
    timestamps = frame["order_purchase_timestamp"]
    return {
        "path": path,
        "rows": len(frame),
        "period_start": timestamps.iloc[0].isoformat(),
        "period_end": timestamps.iloc[-1].isoformat(),
    }


def _batch_spans(rows: int, batch_size: int) -> list[tuple[int, int]]:
    first_size = rows // 2
    return [(0, first_size)] + [
        (start, start + batch_size)
        for start in range(first_size, rows - batch_size + 1, batch_size)
    ]


def _write_batches(
    working: pd.DataFrame, output_dir: Path, batch_size: int
) -> list[_BatchMetadata]:
    (output_dir / "batches").mkdir(parents=True, exist_ok=True)
    working.to_csv(
        output_dir / "working_dataset.csv", index=False, encoding="utf-8", date_format=DATE_FORMAT
    )
    batches: list[_BatchMetadata] = []
    for index, (start, stop) in enumerate(_batch_spans(len(working), batch_size)):
        batch = working.iloc[start:stop]
        batch_id = f"batch_{index:03d}"
        relative = f"batches/{batch_id}.csv"
        batch.to_csv(output_dir / relative, index=False, encoding="utf-8", date_format=DATE_FORMAT)
        batches.append({"id": batch_id, **_file_metadata(batch, relative)})
    return batches


def _parameters(batch_size: int, minimum: int) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "first_batch_rule": "N // 2",
        "drop_last": True,
        "sort_key": list(SORT_KEY),
        "drop_missing": "any_of_selected_columns",
        "min_category_count": minimum,
    }


def _write_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    _ = (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _ = (output_dir / "state.json").write_text(
        json.dumps({"next_batch_index": 0}, indent=2) + "\n",
        encoding="utf-8",
    )


def _result(
    output_dir: Path, statistics: PreparationStatistics, batches: list[_BatchMetadata]
) -> PreparationResult:
    return PreparationResult(
        working_dataset_path=output_dir / "working_dataset.csv",
        manifest_path=output_dir / "manifest.json",
        rows_before_cleaning=statistics["stages"]["assembled"]["rows"],
        rows_after_cleaning=statistics["stages"]["cleaned"]["rows"],
        rows_after_category_filtering=statistics["stages"]["category_filtered"]["rows"],
        categories_before_filtering=statistics.get("categories_before_filtering", 0),
        categories_after_filtering=statistics.get("categories_after_filtering", 0),
        batch_sizes=tuple(batch["rows"] for batch in batches),
        dropped_tail_rows=statistics.get("dropped_tail_rows", 0),
    )


def _check_positive(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def prepare_data(
    raw_dir: Path | str,
    output_dir: Path | str,
    batch_size: int = 5000,
    *,
    min_category_count: int = 1000,
) -> PreparationResult:
    """Prepare seven raw CSVs and overwrite the dataset, batches and manifest.

    Retain complete rows from categories with at least min_category_count
    positions. The first batch contains N // 2 retained rows; subsequent batches
    contain batch_size rows. An incomplete tail stays in working_dataset.csv
    but is omitted from the stream. state.json resets to index 0. At least two
    rows must survive both filters. Invalid sources and I/O errors propagate.

    Args:
        raw_dir (Path | str): Directory with the seven source CSVs.
        output_dir (Path | str): Destination prepared-stream directory.
        batch_size (int, default=5000): Positive size of every batch after the first.
        min_category_count (int, default=1000): Positive category frequency cutoff.

    Returns:
        PreparationResult: Dataset paths and counts from each preparation stage.
    """
    _check_positive("batch_size", batch_size)
    _check_positive("min_category_count", min_category_count)
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    with stage("preparation.read", raw_dir=raw_dir):
        tables = read_sources(raw_dir)
    with stage("preparation.assemble"):
        assembled, statistics = assemble(tables)
    with stage("preparation.clean", min_category_count=min_category_count):
        working = clean(assembled, statistics, min_category_count)
    with stage("preparation.write", output_dir=output_dir):
        batches = _write_batches(working, output_dir, batch_size)
        statistics["dropped_tail_rows"] = len(working) - sum(batch["rows"] for batch in batches)
        _write_manifest(
            output_dir,
            {
                "format_version": 1,
                "sources": source_metadata(raw_dir, tables),
                "schema": {"columns": list(COLUMNS), "dtypes": DTYPES},
                "parameters": _parameters(batch_size, min_category_count),
                "statistics": statistics,
                "working_dataset": _file_metadata(working, "working_dataset.csv"),
                "batches": batches,
            },
        )
    return _result(output_dir, statistics, batches)
