"""Sequential single-ranker runs; only a fully completed step advances progress."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
import logging
from pathlib import Path
from time import perf_counter

from threadpoolctl import threadpool_limits

from buy_today.clustering.models.temporal import train_temporal
from buy_today.clustering.report import report_clustering
from buy_today.clustering.training import train_clustering
from buy_today.generation import generate_dataset
from buy_today.pipeline_config import PipelineConfig, SVD_MODEL, config_dict, parse_config
from buy_today.pipeline_state import Metadata, digest, initialize, load_run, read_json
from buy_today.pipeline_state import read_completed_steps, read_pipeline_config
from buy_today.pipeline_state import write_json as _write_json
from buy_today.progress import stage
from buy_today.ranking import RandomRanker, SVDRanker, report_ranking, train_ranker
from buy_today.update import initialize_reference, update_reference

_logger = logging.getLogger(__name__)
__all__ = [
    "PipelineConfig",
    "StepResult",
    "read_completed_steps",
    "read_pipeline_config",
    "run_next_batch",
    "run_pipeline",
]


@dataclass(frozen=True)
class StepResult:
    step_index: int
    output_dir: Path
    manifest_path: Path


@dataclass
class _Step:
    run_dir: Path
    record: Metadata
    state: Metadata
    settings: PipelineConfig
    durations: dict[str, float] = field(default_factory=dict[str, float])
    started: float = field(default_factory=perf_counter)

    @property
    def index(self) -> int:
        return self.state["next_batch_index"]

    @property
    def batch_path(self) -> str:
        return self.record["batches"][self.index]["path"]

    @property
    def reference(self) -> Path:
        return self.run_dir / "reference.csv"

    @property
    def output(self) -> Path:
        return self.run_dir / f"steps/step_{self.index:03d}"

    def timed[**P, T](
        self, name: str, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        with stage(name, batch_index=self.index):
            started = perf_counter()
            result = function(*args, **kwargs)
            self.durations[name] = perf_counter() - started
            return result


def _check_inputs(step: _Step) -> None:
    if digest(step.record["stream"]["path"]) != step.record["stream"]["sha256"]:
        raise ValueError("Preparation manifest SHA-256 mismatch")
    if digest(step.batch_path) != step.record["batches"][step.index]["sha256"]:
        raise ValueError("Batch SHA-256 mismatch")
    if step.index:
        if digest(step.reference) != step.state["reference_sha256"]:
            raise ValueError("Reference SHA-256 mismatch; unfinished update or external change")
        return
    if step.reference.exists():
        raise ValueError("Unexpected reference before the first completed step")


def _prepare_step(
    run_dir: Path, data_dir: Path | str | None, config: PipelineConfig | None
) -> _Step:
    if not run_dir.exists():
        if data_dir is None:
            raise ValueError("data_dir is required for a new run")
        with stage("pipeline.initialize", run_dir=run_dir):
            initialize(run_dir, data_dir, PipelineConfig() if config is None else config)
    with stage("pipeline.integrity", run_dir=run_dir):
        record, state = load_run(run_dir)
        settings = parse_config(record["parameters"])
        if config is not None and config_dict(config) != record["parameters"]:
            raise ValueError("Run parameters are fixed; create a new run to change them")
        source = Path(record["stream"]["path"])
        if data_dir is not None and Path(data_dir).resolve() != source.parent:
            raise ValueError("Run stream differs from data_dir")
        _ = read_completed_steps(run_dir)
        step = _Step(run_dir, record, state, settings)
        if step.index < len(record["batches"]):
            _check_inputs(step)
    return step


def _reference(step: _Step) -> None:
    if not step.index:
        _ = step.timed(
            "reference", initialize_reference, step.batch_path, step.reference, step.output
        )
        return
    _ = step.timed(
        "reference",
        update_reference,
        step.batch_path,
        step.reference,
        step.output,
        thresholds=step.settings.thresholds,
    )


def _histories(step: _Step) -> None:
    settings = step.settings
    temporal = (
        {} if settings.temporal_n_clusters is None else {"n_clusters": settings.temporal_n_clusters}
    )
    cluster_dir = step.output / "clustering"
    clustering = step.timed(
        "clustering_train",
        train_clustering,
        step.reference,
        cluster_dir,
        strategy=partial(train_temporal, **temporal),
    )
    _ = step.timed(
        "clustering_evaluation",
        report_clustering,
        step.reference,
        step.batch_path,
        cluster_dir,
        max_evaluation_rows=settings.max_evaluation_rows,
        random_state=settings.random_state,
        thread_limit=step.record["thread_limit"],
    )
    previous = (
        None if not step.index else step.run_dir / f"steps/step_{step.index - 1:03d}/histories"
    )
    _ = step.timed(
        "generation",
        generate_dataset,
        step.batch_path,
        clustering.distance_path,
        step.output / "histories",
        previous_dir=previous,
        temperature=settings.temperature,
        n_users=settings.additional_users if step.index else settings.initial_users,
        split_sizes=settings.split_sizes,
        random_state=settings.random_state,
    )


def _ranking(step: _Step) -> None:
    settings = step.settings
    model = (
        SVDRanker(
            n_components=settings.svd_n_components,
            n_iter=settings.svd_n_iter,
            random_state=settings.random_state,
        )
        if settings.model == SVD_MODEL
        else RandomRanker(random_state=settings.random_state)
    )
    histories, ranking = step.output / "histories", step.output / "ranking"
    with threadpool_limits(limits=1):
        _ = step.timed("ranking_train", train_ranker, histories, ranking, ranker=model)
        for split in ("validation", "test"):
            _ = step.timed(
                split,
                report_ranking,
                histories,
                ranking,
                ranking / split,
                split=split,
                k=settings.k,
            )


def _artifacts(prefix: str, index: int) -> dict[str, str]:
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
        artifacts.update(
            deda=f"{prefix}/deda/report.html", deda_metrics=f"{prefix}/deda/metrics.json"
        )
    return artifacts


def _manifest(step: _Step) -> Metadata:
    eda = read_json(step.output / "eda/metrics.json")
    expected_rows = sum(item["rows"] for item in step.record["batches"][: step.index + 1])
    if eda["rows"] != expected_rows:
        raise ValueError("Reference rows differ from the prepared stream manifest")
    artifacts = _artifacts(step.output.relative_to(step.run_dir).as_posix(), step.index)
    step.durations["total"] = perf_counter() - step.started
    return {
        "format_version": 1,
        "step_index": step.index,
        "model": step.settings.model,
        "config_sha256": step.state["config_sha256"],
        "batch": step.record["batches"][step.index],
        "reference": {
            "path": "reference.csv",
            "sha256": digest(step.reference),
            "rows": eda["rows"],
        },
        "artifacts": artifacts,
        "durations_seconds": step.durations,
        "files": {
            name: digest(step.run_dir / name)
            for name in artifacts.values()
            if name.endswith(".json")
        },
    }


def _commit(step: _Step) -> StepResult:
    manifest = _manifest(step)
    manifest_path = step.output / "manifest.json"
    index = step.index
    with stage("pipeline.commit", batch_index=index):
        _write_json(manifest_path, manifest)
        step.state["steps"].append(
            {
                "path": manifest_path.relative_to(step.run_dir).as_posix(),
                "sha256": digest(manifest_path),
            }
        )
        step.state.update(
            next_batch_index=index + 1, reference_sha256=manifest["reference"]["sha256"]
        )
        _write_json(step.run_dir / "state.json", step.state)
    _log_event("batch_committed", index, step.run_dir)
    return StepResult(index, manifest_path.parent, manifest_path)


def _log_event(event: str, index: int, run_dir: Path) -> None:
    _logger.info(
        "event=%s batch_index=%s run_dir=%s",
        event,
        index,
        run_dir,
        extra={"event": event, "batch_index": index},
    )


def run_next_batch(
    run_dir: Path | str,
    *,
    data_dir: Path | str | None = None,
    config: PipelineConfig | None = None,
) -> StepResult | None:
    """Create/continue a run and process exactly one batch.

    Errors propagate without recovery. Only completion advances state; later
    logging failures do not undo the commit. Continuation arguments must match.

    Args:
        run_dir (Path | str): Run's output and progress directory.
        data_dir (Path | str | None, default=None): Prepared stream; required for a new run.
        config (PipelineConfig | None, default=None): Frozen settings for a new run.

    Returns:
        StepResult | None: Completed step, or None when the stream is exhausted.
    """
    step = _prepare_step(Path(run_dir).resolve(), data_dir, config)
    if step.index == len(step.record["batches"]):
        _log_event("stream_exhausted", step.index, step.run_dir)
        return None
    _log_event("batch_start", step.index, step.run_dir)
    step.output.mkdir(parents=True, exist_ok=False)
    step.started = perf_counter()
    _reference(step)
    _histories(step)
    _ranking(step)
    return _commit(step)


def run_pipeline(
    run_dir: Path | str,
    *,
    data_dir: Path | str | None = None,
    config: PipelineConfig | None = None,
    all_batches: bool = False,
) -> tuple[StepResult, ...]:
    """Run one next batch, or all remaining batches in their manifest order.

    Args:
        run_dir (Path | str): Run's output and progress directory.
        data_dir (Path | str | None, default=None): Prepared stream for initialization.
        config (PipelineConfig | None, default=None): Settings to freeze or check on continuation.
        all_batches (bool, default=False): Process the remaining stream instead of one step.

    Returns:
        tuple[StepResult, ...]: Steps completed during this call.
    """
    results: list[StepResult] = []
    advance = partial(run_next_batch, run_dir, data_dir=data_dir, config=config)
    for result in iter(advance, None):
        results.append(result)
        if not all_batches:
            break
    return tuple(results)
