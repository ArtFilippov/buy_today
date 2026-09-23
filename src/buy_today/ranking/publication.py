"""Read the committed visibility boundary of ranking benchmark invocations."""

import json
from pathlib import Path
import re
from typing import Any

TIMESTAMP = "%Y%m%dT%H%M%S%fZ"
RUN_ID = re.compile(r"[0-9]{8}T[0-9]{12}Z")
INVOCATION_ID = re.compile(r"[0-9a-f]{32}")
COMMIT_KIND = "ranking_benchmark_commit"
RUN_KIND = "ranking_benchmark"

type Metadata = dict[str, Any]
type SelectedRuns = list[tuple[Path, Metadata]]


def _read_commit(path: Path) -> Metadata:
    commit: Metadata = json.loads(path.read_text(encoding="utf-8"))
    identity = commit["invocation_id"]
    if (commit["format_version"], commit["kind"], path.stem) != (1, COMMIT_KIND, identity):
        raise ValueError("Invalid benchmark commit identity")
    if (
        not INVOCATION_ID.fullmatch(identity)
        or not isinstance(commit["runs"], dict)
        or not commit["runs"]
    ):
        raise ValueError("Invalid benchmark commit identity")
    return commit


def _register_commit(
    path: Path, latest: dict[str, str], ownership: dict[tuple[str, str], str]
) -> None:
    commit = _read_commit(path)
    for model, run_id in commit["runs"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", model) or not RUN_ID.fullmatch(run_id):
            raise ValueError("Invalid model/run name in benchmark commit")
        run = path.parent.parent / model / "runs" / run_id
        if not run.is_dir() or (model, run_id) in ownership:
            raise ValueError(f"Missing or multiply committed benchmark run: {run}")
        ownership[model, run_id] = commit["invocation_id"]
        if model not in latest or run_id > latest[model]:
            latest[model] = run_id


def _read_run(run: Path, invocation_id: str) -> Metadata:
    manifest: Metadata = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected = (1, RUN_KIND, run.name, run.parent.parent.name, invocation_id)
    actual = (
        manifest["format_version"],
        manifest["kind"],
        manifest["run_id"],
        manifest["model"]["name"],
        manifest["invocation_id"],
    )
    if actual != expected:
        raise ValueError("Benchmark run identity differs from commit")
    return manifest


def _selected_runs(
    output: Path, latest: dict[str, str], ownership: dict[tuple[str, str], str]
) -> SelectedRuns:
    selected: SelectedRuns = []
    for model, run_id in sorted(latest.items()):
        run = output / model / "runs" / run_id
        try:
            manifest = _read_run(run, ownership[model, run_id])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid benchmark run {run}: {error}") from error
        selected.append((run, manifest))
    return selected


def read_latest_benchmark_runs(output_dir: Path | str) -> SelectedRuns:
    """Select latest committed runs from one snapshot of the commit list.

    Enumerating markers excludes staging and orphaned reservations without
    parsing their possibly incomplete manifests.

    Args:
        output_dir (Path | str): Benchmark root containing commit markers.

    Returns:
        SelectedRuns: Run paths and decoded manifests in canonical model order.

    Raises:
        ValueError: The root, a commit or a selected run is invalid.
    """
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise ValueError(f"Benchmark root does not exist: {output}")
    latest: dict[str, str] = {}
    ownership: dict[tuple[str, str], str] = {}
    for path in sorted((output / ".completed").glob("*.json")):
        try:
            _register_commit(path, latest, ownership)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid benchmark commit {path}: {error}") from error
    return _selected_runs(output, latest, ownership)
