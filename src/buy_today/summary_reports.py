"""Validate portable report identities and assemble summary step records."""

from pathlib import Path
from urllib.parse import quote

from buy_today.pipeline_config import SVD_MODEL
from buy_today.pipeline_state import Metadata, read_json

_JSON_SUFFIX = ".json"
_PARENT = ".."
_DRIFT_METRICS = "deda_metrics"


def relative_path(run_dir: Path, relative: str) -> Path:
    path = Path(relative)
    resolved = (run_dir / path).resolve()
    outside = path.is_absolute() or _PARENT in path.parts or run_dir not in resolved.parents
    if outside or resolved == run_dir / "summary" or run_dir / "summary" in resolved.parents:
        raise ValueError(f"Expected an input path within the run, outside summary: {relative}")
    return path


def _report(run_dir: Path, step: Metadata, name: str, *, kind: str | None = None) -> Metadata:
    relative = step["artifacts"][name]
    path = relative_path(run_dir, relative)
    if relative not in step["files"] or path.suffix != _JSON_SUFFIX:
        raise ValueError(f"Step {step['step_index']}: unhashed JSON artifact {name}")
    value = read_json(run_dir / path)
    if value.get("format_version") != 1 or (kind and value.get("kind") != kind):
        raise ValueError(f"Step {step['step_index']}: unexpected report format for {name}")
    return value


def _check_model(training: Metadata, parameters: Metadata) -> None:
    selected = parameters["model"]
    expected_class = {
        "random": "buy_today.ranking.random.RandomRanker",
        SVD_MODEL: "buy_today.ranking.svd.SVDRanker",
    }[selected]
    expected_parameters = {"random_state": parameters["random_state"]}
    if selected == SVD_MODEL:
        expected_parameters.update(
            n_components=parameters["svd_n_components"],
            n_iter=parameters["svd_n_iter"],
        )
    model = training["model"]
    if model["class"] != expected_class or model["parameters"] != expected_parameters:
        raise ValueError("Ranking model parameters differ from frozen run parameters")
    train, catalog = training["inputs"]["train"], training["inputs"]["catalog"]
    if Path(catalog["path"]) != Path(train["path"]).with_name("catalog.csv"):
        raise ValueError("Ranking catalog and train snapshot paths differ")


def _check_evaluation(
    split: str, report: Metadata, training: Metadata, parameters: Metadata
) -> None:
    if report["split"] != split or set(report["inputs"]) != {split, "catalog"}:
        raise ValueError(f"Ranking {split} split identity mismatch")
    if report["k"] != parameters["k"] or not 1 <= report["k"] <= report["n_catalog"]:
        raise ValueError(f"Ranking {split} k differs from frozen parameters or catalog size")
    if report["n_users"] != training["n_users"] or report["n_users"] <= 0:
        raise ValueError(f"Ranking {split} user count differs from training")
    catalog = training["inputs"]["catalog"]
    if report["inputs"]["catalog"] != catalog or report["n_catalog"] != catalog["rows"]:
        raise ValueError(f"Ranking {split} catalog identity mismatch")
    model = training["model"]
    if (
        report["model"]["sha256"] != model["sha256"]
        or Path(report["model"]["path"]).name != model["file"]
    ):
        raise ValueError(f"Ranking {split} trained model identity mismatch")
    _check_events(split, report, training)


def _check_events(split: str, report: Metadata, training: Metadata) -> None:
    source = report["inputs"][split]
    if (
        Path(source["path"]) != Path(training["inputs"]["train"]["path"]).with_name(f"{split}.csv")
        or source["rows"] != report["n_events"]
        or report["n_events"] < report["n_users"]
    ):
        raise ValueError(f"Ranking {split} input identity or event count mismatch")
    for metric in ("recall_at_k", "ndcg_at_k"):
        if not 0 <= report["metrics"][metric] <= 1:
            raise ValueError(f"Ranking {split} {metric} must be in [0, 1]")


def _check_ranking(
    training: Metadata, evaluations: dict[str, Metadata], parameters: Metadata
) -> None:
    """Compare identities without opening models or history snapshots.

    Args:
        training (Metadata): Saved training manifest.
        evaluations (dict[str, Metadata]): Evaluation reports by split.
        parameters (Metadata): Frozen run settings.

    Raises:
        ValueError: Evaluation reports refer to different trained models.
    """
    _check_model(training, parameters)
    for split, report in evaluations.items():
        _check_evaluation(split, report, training, parameters)
    if evaluations["validation"]["model"] != evaluations["test"]["model"]:
        raise ValueError("Ranking validation/test trained model identities differ")


def _check_reference(
    step: Metadata, eda: Metadata, clustering: Metadata, previous: Metadata | None
) -> None:
    reference, batch = step["reference"], step["batch"]
    expected_rows = batch["rows"] + (previous["reference"]["rows"] if previous else 0)
    rows_match = eda["rows"] == reference["rows"] == expected_rows
    if (
        not rows_match
        or eda["input"]["sha256"] != reference["sha256"]
        or clustering["inputs"]["dataset"] != eda["input"]
        or clustering["inputs"]["new_batch"]["sha256"] != batch["sha256"]
    ):
        raise ValueError(f"Step {step['step_index']}: reference/batch report identity mismatch")


def _drift_report(
    run_dir: Path,
    step: Metadata,
    parameters: Metadata,
    previous: Metadata | None,
) -> Metadata | None:
    if _DRIFT_METRICS not in step["artifacts"]:
        if previous:
            raise ValueError(f"Step {step['step_index']}: missing DEDA metrics")
        return None
    drift = _report(run_dir, step, _DRIFT_METRICS, kind="deda")
    if not previous:
        raise ValueError(f"Step {step['step_index']}: drift input identity or thresholds mismatch")
    if (
        drift["inputs"]["reference"]["sha256"] != previous["reference"]["sha256"]
        or drift["inputs"]["reference"]["rows"] != previous["reference"]["rows"]
        or drift["inputs"]["batch"]
        != {key: step["batch"][key] for key in ("path", "sha256", "rows")}
        or drift["thresholds"] != parameters["thresholds"]
    ):
        raise ValueError(f"Step {step['step_index']}: drift input identity or thresholds mismatch")
    return drift


def step_record(
    run_dir: Path, step: Metadata, parameters: Metadata, previous: Metadata | None
) -> Metadata:
    eda = _report(run_dir, step, "eda_metrics", kind="eda")
    clustering = _report(run_dir, step, "clustering_metrics", kind="clustering")
    training = _report(run_dir, step, "ranking_manifest")
    evaluations = {split: _report(run_dir, step, split) for split in ("validation", "test")}
    _check_ranking(training, evaluations, parameters)
    _check_reference(step, eda, clustering, previous)
    return {
        "step_index": step["step_index"],
        "model": step["model"],
        "config_sha256": step["config_sha256"],
        "batch": step["batch"],
        "reference": step["reference"],
        "data_quality": eda,
        "drift": _drift_report(run_dir, step, parameters, previous),
        "clustering": clustering,
        "ranking": {"training": training, **evaluations},
        "durations_seconds": step["durations_seconds"],
        "links": {
            name: "../" + quote(relative_path(run_dir, relative).as_posix(), safe="/")
            for name, relative in step["artifacts"].items()
        },
    }
