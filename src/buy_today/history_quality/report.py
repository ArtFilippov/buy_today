"""Read-only snapshot evaluation and atomic publication of metrics.json."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from buy_today.auto_eda.checks import check_dataset
from buy_today.generation import read_history_dataset
from buy_today.history_quality.evaluation import evaluate_history_quality
from buy_today.history_quality.inputs import check_reference_sources
from buy_today.schema import read_dataset


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _output_directory(output_dir: Path | str, dataset_dir: Path) -> Path:
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    if dataset_dir in output.parents:
        raise ValueError("Output directory must be outside the history snapshot")
    return output


def _publish(output: Path, report: dict[str, object]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".history-quality-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        _ = (staging / "metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise ValueError(f"Output directory already exists: {output}")
        _ = staging.rename(output)
    return output / "metrics.json"


def _report(dataset: Path, reference: Path, repeats: int, seed: int) -> dict[str, object]:
    manifest_hash = _digest(dataset / "generator/manifest.json")
    reference_hash = _digest(reference)
    data = read_history_dataset(dataset)
    working: pd.DataFrame = read_dataset(reference)
    check_reference_sources(working, data.sources)
    # Source order has no statistical meaning, including for cumulative CSVs.
    _ = check_dataset(working.sort_values("order_purchase_timestamp"))
    result = evaluate_history_quality(
        data.events, data.anchors, working, iid_repeats=repeats, random_state=seed,
    )
    files = {**data.manifest["files"], "generator/manifest.json": manifest_hash}
    if _digest(reference) != reference_hash or any(
        _digest(dataset / name) != checksum for name, checksum in files.items()
    ):
        raise ValueError("History-quality inputs changed during evaluation")
    return {
        "format_version": 1,
        "scope": "train+validation+test",
        "quantile_method": "linear",
        **result,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "inputs": {
            "dataset": {"path": str(dataset), "files": files},
            "reference": {"path": str(reference), "sha256": reference_hash},
        },
    }


def report_history_quality(
    dataset_dir: Path | str, reference: Path | str, output_dir: Path | str,
    *, iid_repeats: int = 100, random_state: int = 42,
) -> Path:
    """Publish descriptive metrics for a complete immutable history snapshot.

    The reference CSV must have the working schema and exactly the snapshot's
    source keys, with matching products and purchase timestamps. No original
    source paths are opened and no estimators are loaded or trained.

    Args:
        dataset_dir (Path | str): Complete generation snapshot to evaluate.
        reference (Path | str): One CSV containing exactly its source batches.
        output_dir (Path | str): New report directory outside the snapshot.
        iid_repeats (int, default=100): Positive number of IID cohorts.
        random_state (int, default=42): IID seed in [0, 2**32 - 1].

    Returns:
        Path: Absolute path to the atomically published metrics.json.
    """
    dataset = Path(dataset_dir).resolve()
    output = _output_directory(output_dir, dataset)
    report = _report(dataset, Path(reference).resolve(), iid_repeats, random_state)
    return _publish(output, report)
