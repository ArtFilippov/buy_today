"""Portable summaries of completed steps in a single-ranker pipeline run."""

import json
from pathlib import Path

from buy_today.pipeline_state import Metadata, read_completed_steps, read_json, write_output
from buy_today.summary_render import csv_text, flat_row, html_text
from buy_today.summary_reports import step_record
from buy_today.artifacts import SummaryPaths


def _output_paths(run_dir: Path) -> SummaryPaths:
    output = run_dir / "summary"
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Summary output directory must be a real directory, not a symlink")
    paths = SummaryPaths(*(output / f"summary.{suffix}" for suffix in ("json", "csv", "html")))
    for path in (paths.json_path, paths.csv_path, paths.html_path):
        if path.is_symlink() or (
            path.exists() and (not path.is_file() or path.stat().st_nlink > 1)
        ):
            raise ValueError(f"Summary output must not be a symlink or file alias: {path}")
    return paths


def _summary(run_dir: Path) -> Metadata:
    if not (completed := read_completed_steps(run_dir)):
        raise ValueError(
            "No completed steps in this run; complete a pipeline step before summarizing"
        )
    config = read_json(run_dir / "config.json")
    steps: list[Metadata] = []
    for step in completed:
        try:
            steps.append(
                step_record(run_dir, step, config["parameters"], steps[-1] if steps else None)
            )
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Step {step['step_index']}: invalid summary report metadata: {error}"
            ) from error
    return {
        "format_version": 1,
        "model": config["parameters"]["model"],
        "parameters": config["parameters"],
        "stream": config["stream"],
        "config_sha256": completed[0]["config_sha256"],
        "steps": steps,
    }


def report_summary(run_dir: Path | str) -> SummaryPaths:
    """Write JSON, CSV and self-contained Russian HTML from verified steps.

    Original CSVs, notebooks, HTML reports and model binaries are never opened.
    Repeating the call replaces only the three summary outputs. No completed
    steps, inconsistent report identities and unsafe output aliases are errors.

    Args:
        run_dir (Path | str): Existing pipeline run to summarize.

    Returns:
        SummaryPaths: Absolute paths of the three generated reports.
    """
    run_dir = Path(run_dir).resolve()
    summary = _summary(run_dir)
    rows = [flat_row(step) for step in summary["steps"]]
    texts = (
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        csv_text(rows),
        html_text(summary, rows),
    )
    paths = _output_paths(run_dir)
    paths.json_path.parent.mkdir(exist_ok=True)
    for path, text in zip((paths.json_path, paths.csv_path, paths.html_path), texts, strict=True):
        write_output(path, text)
    return paths
