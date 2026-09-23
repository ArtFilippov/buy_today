"""Container contracts with real preparation, histories, SVD state and summaries.

Only notebook report boundaries are replaced in the compact training cases.
The Docker acceptance script and existing notebook integration tests execute
the real report kernels and HTML exporters.
"""

from dataclasses import asdict
import hashlib
import logging
from pathlib import Path
import runpy
import shutil
from typing import Never

import joblib
import nbformat
import pandas as pd
import pytest
import fixture_types as ft

from container_support import (
    INIT_ARGS,
    KEEP_FILES,
    RESET_FILES,
    SMALL_PARAMETERS,
    SUMMARY_FILES,
    check_exports,
    file_hashes,
    read_json,
    seed_workspace,
    small_raw_tables,
    write_bundle,
)

from buy_today import container_cli
from buy_today.auto_eda.checks import check_combination, check_dataset
from buy_today.auto_eda.drift import evaluate_drift
from buy_today.auto_eda import DriftThresholds
from buy_today.auto_eda.notebook import ReportPaths, write_metrics
from buy_today.generation import read_history_dataset
from buy_today.pipeline import read_completed_steps, read_pipeline_config
from buy_today.progress import stage
from buy_today.schema import read_dataset
from buy_today.summary import SummaryPaths


@pytest.fixture
def container_environment(
    tmp_path: Path,
    raw_category_tables: ft.CategoryTables,
    monkeypatch: ft.MonkeyPatch,
) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace with spaces"
    bundle = write_bundle(tmp_path / "bundled raw tables", small_raw_tables(raw_category_tables))
    monkeypatch.setattr(container_cli, "WORKSPACE", workspace)
    monkeypatch.setattr(container_cli, "BUNDLED_DATA", bundle)
    return workspace, bundle


@pytest.fixture
def compact_reports(monkeypatch: ft.MonkeyPatch) -> None:
    """Small report doubles with genuine input identities and quality/drift checks.

    Keep production summary validation, real clustering training, generation,
    SVD training, ranking evaluation and persistence. No notebook kernels/t-SNE.

    Args:
        monkeypatch (ft.MonkeyPatch): Fixture that installs the report doubles.
    """

    def identity(path: Path | str) -> dict[str, str]:
        path = Path(path)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def save(output: Path | str, **metrics: object) -> ReportPaths:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        write_metrics({"format_version": 1, **metrics}, output / "metrics.json")
        nbformat.write(
            nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_markdown_cell(
                        "Notebook rendering omitted in container unit tests."
                    ),
                ]
            ),
            output / "report.ipynb",
        )
        (output / "report.html").write_text("<html>Report test double</html>", encoding="utf-8")
        return ReportPaths(output / "report.ipynb", output / "report.html")

    def eda(dataset: Path | str, output: Path | str) -> ReportPaths:
        frame = read_dataset(dataset)
        return save(
            output,
            kind="eda",
            input=identity(dataset),
            rows=len(frame),
            n_orders=frame.order_id.nunique(),
            n_customers=frame.customer_id.nunique(),
            n_products=frame.product_id.nunique(),
            checks=check_dataset(frame).to_dict("records"),
        )

    def deda(
        batch: Path | str,
        reference: Path | str,
        output: Path | str,
        *,
        thresholds: DriftThresholds,
    ) -> ReportPaths:
        old, new = read_dataset(reference), read_dataset(batch)
        check_combination(old, new)
        result = evaluate_drift(old, new, thresholds=thresholds)
        metrics = result.metrics.rename(
            columns={
                "Признак": "feature",
                "Мера": "measure",
                "Значение": "value",
                "Порог": "threshold",
                "Дрейф": "drift",
            }
        ).to_dict("records")
        return save(
            output,
            kind="deda",
            inputs={
                "reference": {**identity(reference), "rows": len(old)},
                "batch": {**identity(batch), "rows": len(new)},
            },
            thresholds=asdict(thresholds),
            metrics=metrics,
            drift_detected=result.drift_detected,
        )

    def clustering(
        dataset: Path | str,
        batch: Path | str,
        output: Path | str,
        **kwargs: object,
    ) -> ReportPaths:
        return save(
            output,
            kind="clustering",
            inputs={"dataset": identity(dataset), "new_batch": identity(batch)},
            silhouette=None,
            silhouette_reason="Notebook evaluation omitted in this unit test",
            model={"class": "TemporalClusterer"},
            distance={"class": "TimestampDistance"},
        )

    monkeypatch.setattr("buy_today.update.report_dataset", eda)
    monkeypatch.setattr("buy_today.update.report_drift", deda)
    monkeypatch.setattr("buy_today.pipeline.report_clustering", clustering)


@pytest.fixture
def initialized(
    container_environment: tuple[Path, Path],
    compact_reports: None,
) -> tuple[Path, Path]:
    workspace, bundle = container_environment
    container_cli.main(INIT_ARGS)
    return workspace, bundle


def invocation_log(workspace: Path, command: str) -> str:
    (path,) = (workspace / "logs").glob(f"*_{command}.log")
    return path.read_text(encoding="utf-8")


def forbidden(*args: object, **kwargs: object) -> Never:
    pytest.fail("This command must not initialize, export raw data, prepare, or train")


def test_init_resets_only_owned_outputs_and_overwrites_all_nine_raw_csvs(
    container_environment: tuple[Path, Path],
    compact_reports: None,
) -> None:
    workspace, bundle = container_environment
    seed_workspace(workspace)
    source_before = file_hashes(bundle)

    container_cli.main(INIT_ARGS)

    check_exports(workspace, bundle)
    assert file_hashes(bundle) == source_before
    assert all(not (workspace / relative).exists() for relative in RESET_FILES)
    for relative, content in KEEP_FILES.items():
        assert (workspace / relative).read_bytes() == content
    assert not list((workspace / "recommendations").rglob("*.csv"))
    assert read_json(workspace / "run/state.json")["next_batch_index"] == 1
    assert [step["step_index"] for step in read_completed_steps(workspace / "run")] == [0]
    assert read_json(workspace / "run/config.json")["parameters"] == SMALL_PARAMETERS
    assert [batch["rows"] for batch in read_json(workspace / "data/manifest.json")["batches"]] == [
        12,
        12,
    ]
    assert all((workspace / name).is_file() for name in SUMMARY_FILES)
    assert len(list((workspace / "logs").glob("*.log"))) == 2


@pytest.mark.parametrize(
    "filename,problem",
    [
        ("olist_order_reviews_dataset.csv", "missing"),  # An unused table is still mandatory.
        ("product_category_name_translation.csv", "empty"),  # Last source checked.
    ],
)
def test_invalid_bundle_is_rejected_before_any_reset_or_raw_overwrite(
    container_environment: tuple[Path, Path],
    monkeypatch: ft.MonkeyPatch,
    filename: str,
    problem: str,
) -> None:
    workspace, bundle = container_environment
    seed_workspace(workspace)
    before = file_hashes(workspace)
    if problem == "missing":
        (bundle / filename).unlink()
    else:
        (bundle / filename).write_bytes(b"")
    monkeypatch.setattr(container_cli, "prepare_data", forbidden)
    monkeypatch.setattr(container_cli, "run_next_batch", forbidden)

    with pytest.raises(ValueError, match="Missing or empty bundled Olist CSV"):
        container_cli.main(INIT_ARGS)

    after = file_hashes(workspace)
    assert {name: after.get(name) for name in before} == before
    log = invocation_log(workspace, "init")
    assert filename in log and "event=command_failed" in log
    assert "stage=reset" not in log


@pytest.mark.parametrize(
    "options,error",
    [
        (["--batch-size", "0"], SystemExit),
        (["--model", "random", "--svd-n-components", "2"], ValueError),
        (["--initial-users", "1", "--svd-n-components", "2"], ValueError),
    ],
)
def test_bad_config_preserves_existing_experiment_before_reset(
    container_environment: tuple[Path, Path],
    monkeypatch: ft.MonkeyPatch,
    options: list[str],
    error: type[BaseException],
) -> None:
    workspace, _ = container_environment
    seed_workspace(workspace)
    before = file_hashes(workspace)
    monkeypatch.setattr(container_cli, "prepare_data", forbidden)
    monkeypatch.setattr(container_cli, "run_next_batch", forbidden)

    with pytest.raises(error):
        container_cli.main(["init", *options])

    after = file_hashes(workspace)
    assert {name: after.get(name) for name in before} == before
    assert "event=command_failed" in invocation_log(workspace, "init")


@pytest.mark.parametrize(
    "arguments,diagnostic",
    [
        (["inference", "--user-id", "someone"], "--model-dir"),
        (["update", "--unknown-option"], "--unknown-option"),
    ],
)
def test_argument_failure_has_a_persistent_diagnostic(
    container_environment: tuple[Path, Path],
    arguments: list[str],
    diagnostic: str,
) -> None:
    workspace, _ = container_environment
    with pytest.raises(SystemExit) as caught:
        container_cli.main(arguments)
    assert caught.value.code == 2
    text = invocation_log(workspace, arguments[0])
    assert "Invalid arguments:" in text and diagnostic in text
    assert "event=command_failed" in text


def test_help_does_not_require_a_writable_workspace(
    container_environment: tuple[Path, Path],
) -> None:
    workspace, _ = container_environment
    with pytest.raises(SystemExit) as caught:
        container_cli.main(["--help"])
    assert not caught.value.code
    assert not workspace.exists()


@pytest.mark.parametrize("name", ["dataset", "data", "run", "recommendations"])
def test_symlink_destination_is_rejected_before_deleting_any_existing_outputs(
    container_environment: tuple[Path, Path],
    tmp_path: Path,
    name: str,
) -> None:
    workspace, _ = container_environment
    seed_workspace(workspace)
    outside = tmp_path / "outside workspace"
    (workspace / name).rename(outside)
    (workspace / name).symlink_to(outside, target_is_directory=True)
    before, outside_before = file_hashes(workspace), file_hashes(outside)

    with pytest.raises(ValueError, match="symlink"):
        container_cli.main(INIT_ARGS)

    assert (workspace / name).is_symlink()
    assert file_hashes(outside) == outside_before
    after = file_hashes(workspace)
    assert {key: after.get(key) for key in before} == before


def test_update_resumes_frozen_config_without_export_or_preparation(
    initialized: tuple[Path, Path],
    monkeypatch: ft.MonkeyPatch,
) -> None:
    workspace, bundle = initialized
    run = workspace / "run"
    first = file_hashes(run / "steps/step_000")
    prepared = file_hashes(workspace / "data")
    config = (run / "config.json").read_bytes()
    first_history = read_history_dataset(run / "steps/step_000/histories")
    # Neither the source image bundle nor exported raw CSVs are needed to resume.
    shutil.rmtree(bundle)
    shutil.rmtree(workspace / "dataset")
    monkeypatch.setattr(container_cli, "prepare_data", forbidden)
    monkeypatch.setattr(container_cli, "_reset_and_export", forbidden)

    container_cli.main(["update"])

    assert not (workspace / "dataset").exists()
    assert file_hashes(workspace / "data") == prepared
    assert (run / "config.json").read_bytes() == config
    settings = asdict(read_pipeline_config(run))
    settings["split_sizes"] = list(settings["split_sizes"])
    assert settings == SMALL_PARAMETERS
    assert read_json(run / "state.json")["next_batch_index"] == 2
    assert file_hashes(run / "steps/step_000") == first
    second = read_history_dataset(run / "steps/step_001/histories")
    pd.testing.assert_frame_equal(
        second.events.loc[second.events.batch_index.eq(0)].reset_index(drop=True),
        first_history.events,
    )
    assert len(second.anchors) == 9 and len(second.catalog) == 6
    assert [step["step_index"] for step in read_json(run / "summary/summary.json")["steps"]] == [
        0,
        1,
    ]
    log = invocation_log(workspace, "update")
    assert "event=batch_committed batch_index=1" in log
    assert "stage=preparation" not in log and "stage=export_olist" not in log


def test_csv_destination_directory_is_rejected_before_reset(
    container_environment: tuple[Path, Path],
) -> None:
    workspace, _ = container_environment
    seed_workspace(workspace)
    destination = workspace / "dataset/olist_customers_dataset.csv"
    destination.unlink()
    destination.mkdir()
    before = file_hashes(workspace)

    with pytest.raises(ValueError, match="Expected a CSV destination"):
        container_cli.main(INIT_ARGS)

    after = file_hashes(workspace)
    assert {key: after.get(key) for key in before} == before
    assert "stage=reset" not in invocation_log(workspace, "init")


def test_inference_uses_explicit_saved_model_and_unique_default_output_without_histories(
    initialized: tuple[Path, Path],
    monkeypatch: ft.MonkeyPatch,
) -> None:
    workspace, _ = initialized
    container_cli.main(["update"])
    run = workspace / "run"
    shutil.rmtree(workspace / "data")
    shutil.rmtree(workspace / "dataset")
    for index in range(2):
        shutil.rmtree(run / f"steps/step_{index:03d}/histories")
    before = file_hashes(run)
    monkeypatch.setattr(container_cli, "prepare_data", forbidden)
    monkeypatch.setattr(container_cli, "run_next_batch", forbidden)
    monkeypatch.setattr(container_cli, "report_summary", forbidden)

    for index, user in enumerate(("user_000000_000000", "user_000001_000000")):
        relative = Path(f"run/steps/step_{index:03d}/ranking")
        model = joblib.load(workspace / relative / "model.joblib")
        existing = set((workspace / "recommendations").glob("*.csv"))
        selected = workspace / relative if index else relative
        container_cli.main(
            ["inference", "--model-dir", str(selected), "--user-id", user, "--k", "2"]
        )
        (output,) = set((workspace / "recommendations").glob("*.csv")) - existing
        table = pd.read_csv(output)
        expected = pd.DataFrame(
            {"user_id": [user] * 2, "rank": [1, 2], "product_id": model.predict(user, 2)}
        )
        pd.testing.assert_frame_equal(table, expected)

    outputs_before = file_hashes(workspace / "recommendations")
    assert len(outputs_before) == 2
    # A latest-model fallback would incorrectly succeed for this newly added user.
    with pytest.raises(ValueError, match="Unknown user_id"):
        container_cli.main(
            [
                "inference",
                "--model-dir",
                "run/steps/step_000/ranking",
                "--user-id",
                "user_000001_000000",
                "--k",
                "2",
            ]
        )
    assert file_hashes(workspace / "recommendations") == outputs_before
    assert file_hashes(run) == before


@pytest.mark.parametrize("destination", ["../run/config.json", "symlink"])
def test_inference_cannot_write_outside_recommendations(
    container_environment: tuple[Path, Path],
    monkeypatch: ft.MonkeyPatch,
    destination: str,
) -> None:
    workspace, _ = container_environment
    seed_workspace(workspace)
    victim = workspace / "run/config.json"
    victim.write_bytes(b"existing run configuration")
    if destination == "symlink":
        (workspace / "recommendations/alias.csv").symlink_to(victim)
        destination = "alias.csv"
    before = victim.read_bytes()
    monkeypatch.setattr(container_cli, "export_recommendations", forbidden)

    with pytest.raises(ValueError, match="inside /workspace/recommendations"):
        container_cli.main(
            [
                "inference",
                "--model-dir",
                "run/steps/step_000/ranking",
                "--user-id",
                "known",
                "--output",
                destination,
            ]
        )

    assert victim.read_bytes() == before


def test_inference_does_not_overwrite_run_file_through_hardlink(
    container_environment: tuple[Path, Path],
    monkeypatch: ft.MonkeyPatch,
) -> None:
    workspace, _ = container_environment
    seed_workspace(workspace)
    victim = workspace / "run/config.json"
    victim.write_bytes(b"existing run configuration")
    (workspace / "recommendations/alias.csv").hardlink_to(victim)
    monkeypatch.setattr(container_cli, "export_recommendations", forbidden)

    with pytest.raises(ValueError, match="file alias"):
        container_cli.main(
            [
                "inference",
                "--model-dir",
                "run/steps/step_000/ranking",
                "--user-id",
                "known",
                "--output",
                "alias.csv",
            ]
        )

    assert victim.read_bytes() == b"existing run configuration"


def test_summary_failure_keeps_committed_step_logs_progress_and_reraises_original(
    container_environment: tuple[Path, Path],
    compact_reports: None,
    monkeypatch: ft.MonkeyPatch,
) -> None:
    workspace, _ = container_environment
    error = OSError("summary destination full")

    def fail_summary(run: Path) -> Never:
        assert read_json(run / "state.json")["next_batch_index"] == 1
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(container_cli, "report_summary", fail_summary)
        with pytest.raises(OSError) as caught:
            container_cli.main(INIT_ARGS)
    assert caught.value is error
    run = workspace / "run"
    assert [step["step_index"] for step in read_completed_steps(run)] == [0]
    first = file_hashes(run / "steps/step_000")
    log = invocation_log(workspace, "init")
    assert log.index("event=batch_committed batch_index=0") < log.index(
        "stage=summary event=failed"
    )
    assert "Батч 000 завершён и сохранён" in log
    assert "Следующий update обработает следующий батч" in log
    assert "OSError: summary destination full" in log and "event=command_failed" in log
    assert "event=command_succeeded" not in log

    container_cli.main(["update"])

    assert read_json(run / "state.json")["next_batch_index"] == 2
    assert file_hashes(run / "steps/step_000") == first
    assert [step["step_index"] for step in read_json(run / "summary/summary.json")["steps"]] == [
        0,
        1,
    ]


def test_exhausted_update_rebuilds_summary_without_changing_training_state(
    initialized: tuple[Path, Path],
    monkeypatch: ft.MonkeyPatch,
    capsys: ft.CaptureFixture[str],
) -> None:
    workspace, _ = initialized
    container_cli.main(["update"])
    before = file_hashes(workspace / "run")
    for relative in SUMMARY_FILES:
        (workspace / relative).unlink()
    monkeypatch.setattr("buy_today.pipeline.train_ranker", forbidden)
    monkeypatch.setattr("buy_today.pipeline.generate_dataset", forbidden)
    monkeypatch.setattr(container_cli, "prepare_data", forbidden)
    old_logs = set((workspace / "logs").iterdir())
    capsys.readouterr()

    container_cli.main(["update"])

    assert file_hashes(workspace / "run") == before
    assert "Поток завершён" in capsys.readouterr().out
    (log,) = set((workspace / "logs").iterdir()) - old_logs
    text = log.read_text(encoding="utf-8")
    assert "event=stream_exhausted batch_index=2" in text
    assert "stage=summary event=done" in text
    assert "event=batch_start" not in text


@pytest.mark.parametrize("verbose", [False, True])
def test_stages_are_live_when_verbose_and_always_in_file_log(
    container_environment: tuple[Path, Path],
    monkeypatch: ft.MonkeyPatch,
    capsys: ft.CaptureFixture[str],
    verbose: bool,
) -> None:
    workspace, _ = container_environment
    monkeypatch.setattr(container_cli, "run_next_batch", exhausted_run)
    observed: list[bool] = []

    def summarize(run: Path) -> SummaryPaths:
        # Inspect while the summary call is still in progress, not after main.
        captured = capsys.readouterr()
        assert str(workspace / "logs") in captured.out
        assert ("stage=summary event=start" in captured.err) is verbose
        assert "stage=summary event=done" not in captured.err
        text = invocation_log(workspace, "update")
        assert "stage=summary event=start" in text and "stage=summary event=done" not in text
        observed.append(True)
        with stage("summary_details"):
            pass
        return SummaryPaths(
            *(run / f"summary/summary.{suffix}" for suffix in ("json", "csv", "html"))
        )

    monkeypatch.setattr(container_cli, "report_summary", summarize)
    container_cli.main(["update", *(["--verbose"] if verbose else [])])

    assert observed == [True]
    captured = capsys.readouterr()
    if verbose:
        assert "stage=summary event=done elapsed_seconds=" in captured.err
        assert "stage=summary_details event=start" in captured.err
    else:
        assert not captured.err
    log = invocation_log(workspace, "update")
    assert "stage=summary event=done elapsed_seconds=" in log
    assert "stage=summary_details event=start" in log and "event=command_succeeded" in log


def test_handlers_levels_and_propagation_restored_after_success_failure_and_repeated_calls(
    container_environment: tuple[Path, Path],
    monkeypatch: ft.MonkeyPatch,
) -> None:
    workspace, _ = container_environment
    logger = logging.getLogger("buy_today")
    old_handler = logging.NullHandler()
    monkeypatch.setattr(logger, "handlers", [old_handler])
    monkeypatch.setattr(logger, "level", logging.DEBUG)
    monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(container_cli, "run_next_batch", exhausted_run)
    active_handlers: list[logging.Handler] = []
    error = RuntimeError("summary failed between calls")

    def summarize(run: Path) -> SummaryPaths:
        assert old_handler not in logger.handlers
        active_handlers.extend(logger.handlers)
        if len(active_handlers) == 4:
            raise error
        return SummaryPaths(*(run / f"summary.{suffix}" for suffix in ("json", "csv", "html")))

    monkeypatch.setattr(container_cli, "report_summary", summarize)
    saved_logs = {}
    for index in range(3):
        if index == 1:
            with pytest.raises(RuntimeError) as caught:
                container_cli.main(["update"])
            assert caught.value is error
        else:
            container_cli.main(["update"])
        assert logger.handlers == [old_handler]
        assert logger.level == logging.DEBUG and logger.propagate is True
        assert all(
            handler.stream is None
            for handler in active_handlers
            if isinstance(handler, logging.FileHandler)
        )
        current = file_hashes(workspace / "logs")
        assert {key: current[key] for key in saved_logs} == saved_logs
        assert len(current) == index + 1
        saved_logs = current
    texts = [path.read_text(encoding="utf-8") for path in (workspace / "logs").glob("*.log")]
    assert all(text.count("event=command_start") == 1 for text in texts)
    assert sum("event=command_failed" in text for text in texts) == 1
    assert sum("event=command_succeeded" in text for text in texts) == 2


def test_update_requires_initialized_run_without_exporting_or_preparing(
    container_environment: tuple[Path, Path],
    monkeypatch: ft.MonkeyPatch,
) -> None:
    workspace, _ = container_environment
    monkeypatch.setattr(container_cli, "prepare_data", forbidden)
    monkeypatch.setattr(container_cli, "_reset_and_export", forbidden)

    with pytest.raises(ValueError, match="data_dir is required"):
        container_cli.main(["update"])

    assert not (workspace / "run").exists()
    assert not (workspace / "data").exists()
    assert not (workspace / "dataset").exists()
    assert "event=command_failed" in invocation_log(workspace, "update")


def test_logging_restores_effective_child_level_after_command(
    container_environment: tuple[Path, Path],
    monkeypatch: ft.MonkeyPatch,
) -> None:
    logger = logging.getLogger("buy_today")
    previous = logger.level
    child = logging.getLogger("buy_today.progress")
    error = RuntimeError("stop before training")

    def fail(run: Path) -> Never:
        assert child.isEnabledFor(logging.INFO)
        raise error

    def check_child_logging() -> None:
        logger.setLevel(logging.WARNING)
        assert not child.isEnabledFor(logging.INFO)
        with pytest.raises(RuntimeError) as caught:
            container_cli.main(["update"])
        assert caught.value is error
        assert not child.isEnabledFor(logging.INFO)

    monkeypatch.setattr(container_cli, "run_next_batch", fail)
    try:
        check_child_logging()
    finally:
        logger.setLevel(previous)


def test_entrypoint_module_name_does_not_change_command_logging(
    container_environment: tuple[Path, Path],
    monkeypatch: ft.MonkeyPatch,
) -> None:
    workspace, _ = container_environment
    # Like python -m, load the file under a different __name__, then call its
    # actual entry point with the workspace redirected to the fixture.
    namespace = runpy.run_path(container_cli.__file__, run_name="container_entrypoint")
    entrypoint = namespace["main"]
    monkeypatch.setitem(entrypoint.__globals__, "WORKSPACE", workspace)
    with pytest.raises(SystemExit) as caught:
        entrypoint(["update", "--unknown-option"])
    assert caught.value.code == 2
    text = invocation_log(workspace, "update")
    assert "event=command_start" in text
    assert "Invalid arguments:" in text and "--unknown-option" in text
    assert "event=command_failed" in text


def exhausted_run(run: Path) -> None:
    assert run.name == "run"
