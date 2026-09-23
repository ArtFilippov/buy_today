"""Integrity-checked metadata storage, independent of pipeline execution."""

from contextlib import contextmanager
from collections.abc import Generator
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Any

from buy_today.pipeline_config import PipelineConfig, config_dict, parse_config

# JSON report versions have heterogeneous nested values. Keep this dynamic
# boundary explicit; execution parameters are parsed into PipelineConfig.
type Metadata = dict[str, Any]


def digest(path: Path | str) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path: Path | str) -> Metadata:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def _replacement(path: Path) -> Generator[tuple[IO[str], Path]]:
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            yield temporary.file, temporary_path
        finally:
            temporary_path.unlink(missing_ok=True)


def write_output(path: Path, text: str) -> None:
    # Replace rather than truncate: existing output inodes remain untouched.
    with _replacement(path) as (temporary, temporary_path):
        _ = temporary.write(text)
        temporary.close()
        _ = temporary_path.replace(path)


def write_json(path: Path, value: Metadata) -> None:
    """Replace one metadata file, without making a whole step transactional.

    Args:
        path (Path): Destination metadata file.
        value (Metadata): JSON-compatible metadata to serialize.
    """
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_output(path, text)


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root not in path.parents:
        raise ValueError(f"Expected a relative path within {root}: {relative}")
    return path


def _stream_batches(data_dir: Path, manifest: Metadata) -> list[Metadata]:
    if manifest.get("format_version") != 1 or not manifest.get("batches"):
        raise ValueError("Expected a preparation manifest with format_version=1 and batches")
    batches: list[Metadata] = []
    for index, batch in enumerate(manifest["batches"]):
        if batch["id"] != f"batch_{index:03d}":
            raise ValueError("Stream batch IDs must be consecutive, starting at batch_000")
        path = contained(data_dir, batch["path"])
        if (
            isinstance(batch["rows"], bool)
            or not isinstance(batch["rows"], int)
            or batch["rows"] <= 0
        ):
            raise ValueError("Batch rows must be a positive integer")
        batches.append(
            {"id": batch["id"], "path": str(path), "rows": batch["rows"], "sha256": digest(path)}
        )
    if len({batch["path"] for batch in batches}) != len(batches):
        raise ValueError("Stream batch paths must be unique")
    return batches


def initialize(run_dir: Path, data_dir: Path | str, config: PipelineConfig) -> None:
    data_dir = Path(data_dir).resolve()
    manifest_path = data_dir / "manifest.json"
    batches = _stream_batches(data_dir, read_json(manifest_path))
    if run_dir == data_dir or run_dir in data_dir.parents:
        raise ValueError("Run directory must not contain the prepared stream")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "config.json",
        {
            "format_version": 1,
            "stream": {"path": str(manifest_path), "sha256": digest(manifest_path)},
            "batches": batches,
            "parameters": config_dict(config),
            "thread_limit": 1,
        },
    )
    write_json(
        run_dir / "state.json",
        {
            "format_version": 1,
            "config_sha256": digest(run_dir / "config.json"),
            "next_batch_index": 0,
            "reference_sha256": None,
            "steps": [],
        },
    )


def load_run(run_dir: Path) -> tuple[Metadata, Metadata]:
    record = read_json(run_dir / "config.json")
    state = read_json(run_dir / "state.json")
    if record.get("format_version") != 1 or state.get("format_version") != 1:
        raise ValueError("Expected run format_version=1")
    if digest(run_dir / "config.json") != state["config_sha256"]:
        raise ValueError("Run configuration SHA-256 mismatch")
    index = state["next_batch_index"]
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index <= len(record["batches"])
        or index != len(state["steps"])
    ):
        raise ValueError("Inconsistent run progress")
    return record, state


def read_pipeline_config(run_dir: Path | str) -> PipelineConfig:
    """Return the frozen, integrity-checked parameters of an existing run.

    Args:
        run_dir (Path | str): Existing run directory.

    Returns:
        PipelineConfig: Validated saved settings.
    """
    record, _ = load_run(Path(run_dir).resolve())
    return parse_config(record["parameters"])


def _completed_step(run_dir: Path, saved: Metadata, index: int) -> Metadata:
    relative = f"steps/step_{index:03d}/manifest.json"
    if saved["path"] != relative:
        raise ValueError("Completed step paths must be consecutive")
    path = contained(run_dir, relative)
    if digest(path) != saved["sha256"]:
        raise ValueError(f"Step manifest SHA-256 mismatch: {relative}")
    return read_json(path)


def _check_step(step: Metadata, record: Metadata, state: Metadata, index: int) -> None:
    version_matches = step.get("format_version") == 1
    if not version_matches or step["step_index"] != index:
        raise ValueError(f"Step identity mismatch: steps/step_{index:03d}/manifest.json")
    if (
        step["config_sha256"] != state["config_sha256"]
        or step["batch"] != record["batches"][index]
        or step["model"] != record["parameters"]["model"]
    ):
        raise ValueError(f"Step identity mismatch: steps/step_{index:03d}/manifest.json")


def read_completed_steps(run_dir: Path | str) -> tuple[Metadata, ...]:
    """Read committed steps and verify their small, portable report files.

    Original stream inputs, the mutable reference and model binaries are not
    needed. Uncommitted step directories are ignored.

    Args:
        run_dir (Path | str): Existing run directory.

    Returns:
        tuple[Metadata, ...]: Committed manifests in stream order.

    Raises:
        ValueError: A report digest differs from its committed identity.
    """
    run_dir = Path(run_dir).resolve()
    record, state = load_run(run_dir)
    steps: list[Metadata] = []
    for index, saved in enumerate(state["steps"]):
        step = _completed_step(run_dir, saved, index)
        _check_step(step, record, state, index)
        for filename, expected in step["files"].items():
            if digest(contained(run_dir, filename)) != expected:
                raise ValueError(f"Step report SHA-256 mismatch: {filename}")
        steps.append(step)
    return tuple(steps)
