"""Sequential, single-ranker runs over a prepared stream of batches.

A run owns its physical reference and progress. Only a fully completed step
advances progress; exceptions propagate and partial steps require manual repair.
"""

from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

from threadpoolctl import threadpool_limits

from buy_today.auto_eda import DriftThresholds
from buy_today.clustering.models.temporal import train_temporal
from buy_today.clustering.report import report_clustering
from buy_today.clustering.training import train_clustering
from buy_today.generation import generate_dataset
from buy_today.generation.histories import check_parameters
from buy_today.progress import stage
from buy_today.ranking import RandomRanker, SVDRanker, report_ranking, train_ranker
from buy_today.update import initialize_reference, update_reference


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    model: str = "svd"
    random_state: int = 42
    temperature: float = 86400.0
    initial_users: int = 2000
    additional_users: int = 250
    split_sizes: tuple[int, int, int] = (70, 15, 15)
    k: int = 10
    svd_n_components: int = 32
    svd_n_iter: int = 7
    temporal_n_clusters: int | None = None
    max_evaluation_rows: int = 1000
    thresholds: DriftThresholds = DriftThresholds()

    def __post_init__(self):
        if self.model not in ("random", "svd"):
            raise ValueError("model must be random or svd")
        for users in (self.initial_users, self.additional_users):
            check_parameters(
                n_users=users, batch_index=0, temperature=self.temperature,
                random_state=self.random_state, split_sizes=self.split_sizes,
            )
        for name in ("k", "svd_n_components", "svd_n_iter", "max_evaluation_rows", "temporal_n_clusters"):
            value = getattr(self, name)
            if name == "temporal_n_clusters" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.thresholds, DriftThresholds):
            raise ValueError("thresholds must be DriftThresholds")


@dataclass(frozen=True)
class StepResult:
    step_index: int
    output_dir: Path
    manifest_path: Path


def _digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    """Replace one metadata file; this does not make a whole step transactional."""
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(text)
            temporary.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _config_dict(config):
    value = asdict(config)
    value["split_sizes"] = list(config.split_sizes)
    return value


def _parse_config(value):
    value = dict(value)
    value["split_sizes"] = tuple(value["split_sizes"])
    value["thresholds"] = DriftThresholds(**value["thresholds"])
    return PipelineConfig(**value)


def _contained(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root not in path.parents:
        raise ValueError(f"Expected a relative path within {root}: {relative}")
    return path


def _initialize(run_dir, data_dir, config):
    data_dir = Path(data_dir).resolve()
    manifest_path = data_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("format_version") != 1 or not manifest.get("batches"):
        raise ValueError("Expected a preparation manifest with format_version=1 and batches")
    batches = []
    for index, batch in enumerate(manifest["batches"]):
        if batch["id"] != f"batch_{index:03d}":
            raise ValueError("Stream batch IDs must be consecutive, starting at batch_000")
        path = _contained(data_dir, batch["path"])
        if isinstance(batch["rows"], bool) or not isinstance(batch["rows"], int) or batch["rows"] <= 0:
            raise ValueError("Batch rows must be a positive integer")
        batches.append({"id": batch["id"], "path": str(path), "rows": batch["rows"], "sha256": _digest(path)})
    if len({batch["path"] for batch in batches}) != len(batches):
        raise ValueError("Stream batch paths must be unique")
    # A new run cannot contain or replace any stream input.
    if run_dir == data_dir or run_dir in data_dir.parents:
        raise ValueError("Run directory must not contain the prepared stream")
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config.json", {
        "format_version": 1,
        "stream": {"path": str(manifest_path), "sha256": _digest(manifest_path)},
        "batches": batches, "parameters": _config_dict(config),
        "thread_limit": 1,
    })
    _write_json(run_dir / "state.json", {
        "format_version": 1, "config_sha256": _digest(run_dir / "config.json"),
        "next_batch_index": 0, "reference_sha256": None, "steps": [],
    })


def _load_run(run_dir):
    record = _read_json(run_dir / "config.json")
    state = _read_json(run_dir / "state.json")
    if record.get("format_version") != 1 or state.get("format_version") != 1:
        raise ValueError("Expected run format_version=1")
    if _digest(run_dir / "config.json") != state["config_sha256"]:
        raise ValueError("Run configuration SHA-256 mismatch")
    index = state["next_batch_index"]
    if (isinstance(index, bool) or not isinstance(index, int)
            or not 0 <= index <= len(record["batches"]) or index != len(state["steps"])):
        raise ValueError("Inconsistent run progress")
    return record, state


def read_pipeline_config(run_dir: Path | str) -> PipelineConfig:
    """Return the frozen, integrity-checked parameters of an existing run."""
    record, _ = _load_run(Path(run_dir).resolve())
    return _parse_config(record["parameters"])


def read_completed_steps(run_dir: Path | str) -> tuple[dict, ...]:
    """Read committed step records and verify their small, portable report files.

    Neither original stream inputs, the mutable reference, nor model binaries
    are needed to build a summary. Uncommitted step directories are ignored.
    """
    run_dir = Path(run_dir).resolve()
    record, state = _load_run(run_dir)
    steps = []
    for index, saved in enumerate(state["steps"]):
        relative = f"steps/step_{index:03d}/manifest.json"
        if saved["path"] != relative:
            raise ValueError("Completed step paths must be consecutive")
        path = _contained(run_dir, relative)
        if _digest(path) != saved["sha256"]:
            raise ValueError(f"Step manifest SHA-256 mismatch: {relative}")
        step = _read_json(path)
        if (step.get("format_version") != 1 or step["step_index"] != index
                or step["config_sha256"] != state["config_sha256"]
                or step["batch"] != record["batches"][index]
                or step["model"] != record["parameters"]["model"]):
            raise ValueError(f"Step identity mismatch: {relative}")
        for filename, digest in step["files"].items():
            if _digest(_contained(run_dir, filename)) != digest:
                raise ValueError(f"Step report SHA-256 mismatch: {filename}")
        steps.append(step)
    return tuple(steps)


def run_next_batch(
    run_dir: Path | str, *, data_dir: Path | str | None = None,
    config: PipelineConfig | None = None,
) -> StepResult | None:
    """Create/continue a run, process exactly one batch; None means stream exhausted.

    data_dir is required for a new run; config then defaults to PipelineConfig().
    Supplied arguments on continuation must match the saved configuration.
    Errors are raised unchanged, without retries or recovery. State advances
    only after a completed batch; later logging failures do not undo the commit.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        if data_dir is None:
            raise ValueError("data_dir is required for a new run")
        with stage("pipeline.initialize", run_dir=run_dir):
            _initialize(run_dir, data_dir, PipelineConfig() if config is None else config)
    with stage("pipeline.integrity", run_dir=run_dir):
        record, state = _load_run(run_dir)
        settings = _parse_config(record["parameters"])
        if config is not None and _config_dict(config) != record["parameters"]:
            raise ValueError("Run parameters are fixed; create a new run to change them")
        source = Path(record["stream"]["path"])
        if data_dir is not None and Path(data_dir).resolve() != source.parent:
            raise ValueError("Run stream differs from data_dir")
        # Completed steps are checked before any writes, including on exhaustion.
        read_completed_steps(run_dir)
        index = state["next_batch_index"]
        if index < len(record["batches"]):
            if _digest(source) != record["stream"]["sha256"]:
                raise ValueError("Preparation manifest SHA-256 mismatch")
            batch = record["batches"][index]
            if _digest(batch["path"]) != batch["sha256"]:
                raise ValueError("Batch SHA-256 mismatch")
            reference = run_dir / "reference.csv"
            if index:
                if _digest(reference) != state["reference_sha256"]:
                    raise ValueError("Reference SHA-256 mismatch; unfinished update or external change")
            elif reference.exists():
                raise ValueError("Unexpected reference before the first completed step")
    if index == len(record["batches"]):
        _logger.info(
            "event=stream_exhausted batch_index=%s run_dir=%s", index, run_dir,
            extra={"event": "stream_exhausted", "batch_index": index},
        )
        return None
    _logger.info(
        "event=batch_start batch_index=%s run_dir=%s", index, run_dir,
        extra={"event": "batch_start", "batch_index": index},
    )
    output_dir = run_dir / f"steps/step_{index:03d}"
    # An exclusive step directory also prevents a second writer/repeated append.
    output_dir.mkdir(parents=True, exist_ok=False)
    durations = {}
    started = perf_counter()

    def timed(name, function, *args, **kwargs):
        with stage(name, batch_index=index):
            start = perf_counter()
            result = function(*args, **kwargs)
            durations[name] = perf_counter() - start
            return result

    if index == 0:
        timed("reference", initialize_reference, batch["path"], reference, output_dir)
    else:
        timed("reference", update_reference, batch["path"], reference, output_dir, thresholds=settings.thresholds)
    temporal = {}
    if settings.temporal_n_clusters is not None:
        temporal["n_clusters"] = settings.temporal_n_clusters
    cluster_dir = output_dir / "clustering"
    clustering = timed(
        "clustering_train", train_clustering, reference, cluster_dir,
        strategy=partial(train_temporal, **temporal),
    )
    timed(
        "clustering_evaluation", report_clustering, reference, batch["path"], cluster_dir,
        max_evaluation_rows=settings.max_evaluation_rows, random_state=settings.random_state,
        thread_limit=record["thread_limit"],
    )
    histories = output_dir / "histories"
    previous = None if index == 0 else run_dir / f"steps/step_{index - 1:03d}/histories"
    timed(
        "generation", generate_dataset, batch["path"], clustering.distance_path, histories,
        previous_dir=previous, temperature=settings.temperature,
        n_users=settings.initial_users if index == 0 else settings.additional_users,
        split_sizes=settings.split_sizes, random_state=settings.random_state,
    )
    model = (
        SVDRanker(n_components=settings.svd_n_components, n_iter=settings.svd_n_iter,
                  random_state=settings.random_state)
        if settings.model == "svd" else RandomRanker(random_state=settings.random_state)
    )
    ranking = output_dir / "ranking"
    # Limit BLAS/OpenMP for stable SVD fits and per-user prediction cost.
    with threadpool_limits(limits=1):
        timed("ranking_train", train_ranker, histories, ranking, ranker=model)
        for split in ("validation", "test"):
            timed(split, report_ranking, histories, ranking, ranking / split, split=split, k=settings.k)
    eda = _read_json(output_dir / "eda/metrics.json")
    expected_rows = sum(item["rows"] for item in record["batches"][:index + 1])
    if eda["rows"] != expected_rows:
        raise ValueError("Reference rows differ from the prepared stream manifest")
    prefix = output_dir.relative_to(run_dir).as_posix()
    artifacts = {
        "eda": f"{prefix}/eda/report.html",
        "eda_metrics": f"{prefix}/eda/metrics.json",
        "clustering": f"{prefix}/clustering/report.html",
        "clustering_metrics": f"{prefix}/clustering/metrics.json",
        "histories": f"{prefix}/histories",
        "histories_manifest": f"{prefix}/histories/generator/manifest.json",
        "ranking": f"{prefix}/ranking",
        "ranking_manifest": f"{prefix}/ranking/manifest.json",
        "validation": f"{prefix}/ranking/validation/metrics.json",
        "test": f"{prefix}/ranking/test/metrics.json",
    }
    if index:
        artifacts.update(deda=f"{prefix}/deda/report.html", deda_metrics=f"{prefix}/deda/metrics.json")
    durations["total"] = perf_counter() - started
    step = {
        "format_version": 1, "step_index": index, "model": settings.model,
        "config_sha256": state["config_sha256"], "batch": batch,
        "reference": {"path": "reference.csv", "sha256": _digest(reference), "rows": eda["rows"]},
        "artifacts": artifacts, "durations_seconds": durations,
        "files": {name: _digest(run_dir / name) for name in artifacts.values() if name.endswith(".json")},
    }
    manifest_path = output_dir / "manifest.json"
    with stage("pipeline.commit", batch_index=index):
        _write_json(manifest_path, step)
        state["steps"].append({"path": f"{prefix}/manifest.json", "sha256": _digest(manifest_path)})
        state.update(next_batch_index=index + 1, reference_sha256=step["reference"]["sha256"])
        _write_json(run_dir / "state.json", state)
    _logger.info(
        "event=batch_committed batch_index=%s run_dir=%s", index, run_dir,
        extra={"event": "batch_committed", "batch_index": index},
    )
    return StepResult(index, output_dir, manifest_path)


def run_pipeline(
    run_dir: Path | str, *, data_dir: Path | str | None = None,
    config: PipelineConfig | None = None, all_batches: bool = False,
) -> tuple[StepResult, ...]:
    """Run one next batch, or all remaining batches in their manifest order."""
    results = []
    while True:
        result = run_next_batch(run_dir, data_dir=data_dir, config=config)
        if result is None:
            break
        results.append(result)
        if not all_batches:
            break
    return tuple(results)
