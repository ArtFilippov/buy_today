"""Portable comparisons of the latest committed ranking benchmark runs."""

import csv
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

from buy_today.ranking.publication import Metadata, read_latest_benchmark_runs
from buy_today.ranking.report_html import comparison_html
from buy_today.ranking.report_validation import CSV_COLUMNS, comparison_rows
from buy_today.ranking.domain import BenchmarkReportPaths

_RUNS = "runs"


class _TemporaryText(Protocol):
    name: str

    def write(self, text: str, /) -> int: ...

    def close(self) -> None: ...


def _check_output(path: Path, root: Path, sources: set[Path]) -> None:
    if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink > 1)):
        raise ValueError(f"Report output must not be a symlink, directory or file alias: {path}")
    resolved = path.resolve()
    if root in resolved.parents or root == resolved:
        relative = Path.relative_to(resolved, root).parts
        if relative and (
            relative[0].startswith(".") or (len(relative) > 1 and relative[1] == _RUNS)
        ):
            raise ValueError("Report output collides with benchmark artifacts")
    if any(resolved == source or source in resolved.parents for source in sources):
        raise ValueError("Report output collides with source dataset paths")


def _output_paths(root: Path, output: Path | str, rows: list[Metadata]) -> BenchmarkReportPaths:
    output = Path(output).absolute()
    if output.suffix.lower() not in (".html", ".htm"):
        raise ValueError("Report output must be an HTML file (.html or .htm)")
    paths = BenchmarkReportPaths(output, Path.with_suffix(output, ".csv"))
    sources = {Path(row["dataset_path"]).resolve() for row in rows}
    for path in (paths.html_path, paths.csv_path):
        _check_output(path, root, sources)
    return BenchmarkReportPaths(paths.html_path.resolve(), paths.csv_path.resolve())


def _replace(temporary: _TemporaryText, path: Path, text: str) -> None:
    _ = temporary.write(text)
    temporary.close()
    _ = Path(temporary.name).replace(path)


def _write_output(path: Path, text: str) -> None:
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            _replace(temporary, path, text)
        finally:
            temporary_path.unlink(missing_ok=True)


def report_ranking_benchmark(root: Path | str, output: Path | str) -> BenchmarkReportPaths:
    """Write self-contained Russian HTML and an adjacent long-matrix CSV.

    Only committed metadata and saved metrics are read. Every model must have
    the same datasets, K and input identities. Outputs are replaced atomically
    per file after the complete comparison is validated.

    Args:
        root (Path | str): Benchmark root with completed invocations.
        output (Path | str): HTML destination with an adjacent CSV destination.

    Returns:
        BenchmarkReportPaths: Published HTML and CSV paths.

    Raises:
        ValueError: No completed runs exist or comparison metadata is invalid.
    """
    root = Path(root).resolve()
    if not (selected := read_latest_benchmark_runs(root)):
        raise ValueError("No completed benchmark runs to compare")
    rows = comparison_rows(selected)
    paths = _output_paths(root, output, rows)
    csv_text = StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=CSV_COLUMNS)
    _ = writer.writeheader()
    writer.writerows(rows)
    html = comparison_html(root, selected, rows)
    paths.html_path.parent.mkdir(parents=True, exist_ok=True)
    _write_output(paths.csv_path, csv_text.getvalue())
    _write_output(paths.html_path, html)
    return paths
