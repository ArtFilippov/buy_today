"""Progress is observable during work and never implies an uncommitted batch."""

import json
import logging
from pathlib import Path

import nbformat
import pytest

from buy_today import pipeline
from buy_today.auto_eda.checks import check_dataset
from buy_today.auto_eda.notebook import ReportPaths, execute_report
from buy_today.progress import stage
from buy_today.schema import DATE_FORMAT, read_dataset


def progress_records(caplog):
    return [record for record in caplog.records if record.name == "buy_today.progress"]


def test_start_is_visible_inside_stage_and_completion_has_elapsed_time(caplog, monkeypatch):
    # Configure only the root: the library's NullHandler must allow propagation.
    caplog.set_level(logging.INFO)
    times = iter([10.0, 12.5])
    monkeypatch.setattr("buy_today.progress.perf_counter", lambda: next(times))
    with stage("training", batch_index=3, message="caller context"):
        start, = progress_records(caplog)
        assert start.event == "start"
        assert start.stage == "training"
        assert start.stage_fields == {"batch_index": 3, "message": "caller context"}
        assert start.levelno == logging.INFO
        assert not hasattr(start, "elapsed_seconds")
    start, done = progress_records(caplog)
    assert done.event == "done"
    assert done.stage == start.stage
    assert done.stage_fields == start.stage_fields
    assert done.elapsed_seconds == 2.5
    assert done.levelno == logging.INFO
    assert done.exc_info is None
    # Ordinary formatters must expose useful context without custom extras.
    assert "batch_index=3" in start.getMessage()
    assert "elapsed_seconds=" in done.getMessage()


@pytest.mark.parametrize("error", [ValueError("bad input"), OSError("write failed"), KeyboardInterrupt()])
def test_failure_has_duration_traceback_and_preserves_exact_exception(caplog, monkeypatch, error):
    caplog.set_level(logging.INFO, logger="buy_today")
    times = iter([20.0, 23.0])
    monkeypatch.setattr("buy_today.progress.perf_counter", lambda: next(times))
    with pytest.raises(type(error)) as caught:
        with stage("export", output_path="report.html"):
            raise error
    assert caught.value is error
    start, failed = progress_records(caplog)
    assert start.event == "start"
    assert failed.event == "failed"
    assert failed.levelno == logging.ERROR
    assert failed.elapsed_seconds == 3.0
    assert failed.exc_info[1] is error
    assert failed.exc_info[2] is not None
    assert "Traceback (most recent call last)" in caplog.text


def test_error_logging_failure_cannot_replace_primary_exception(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="buy_today")

    class BrokenHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                raise OSError("log disk full")

    monkeypatch.setattr(logging.getLogger("buy_today.progress"), "handlers", [BrokenHandler()])
    error = RuntimeError("primary failure")
    with pytest.raises(RuntimeError) as caught:
        with stage("work"):
            raise error
    assert caught.value is error


def test_unconfigured_library_does_not_use_last_resort(monkeypatch, capsys):
    root = logging.getLogger()
    library = logging.getLogger("buy_today")
    monkeypatch.setattr(root, "handlers", [])
    # Simulate a caller without application handlers, keeping library defaults.
    monkeypatch.setattr(library, "handlers", [
        handler for handler in library.handlers if isinstance(handler, logging.NullHandler)
    ])
    monkeypatch.setattr(logging.getLogger("buy_today.progress"), "handlers", [])
    error = RuntimeError("must not appear on stderr")
    with pytest.raises(RuntimeError) as caught:
        with stage("unconfigured"):
            raise error
    assert caught.value is error
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.fixture
def pipeline_case(tmp_path, working_frame, monkeypatch):
    """Use real training and persistence; omit notebook kernels as in pipeline tests."""
    stream = tmp_path / "stream"
    stream.mkdir()
    frame = working_frame.copy()
    frame["product_id"] = [f"product_{number % 6:02d}" for number in range(len(frame))]
    frame.to_csv(stream / "batch.csv", index=False, date_format=DATE_FORMAT)
    (stream / "manifest.json").write_text(json.dumps({
        "format_version": 1,
        "batches": [{"id": "batch_000", "path": "batch.csv", "rows": len(frame)}],
    }), encoding="utf-8")

    def save_report(output, **metrics):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (output / "report.html").write_text("report", encoding="utf-8")
        return ReportPaths(output / "report.ipynb", output / "report.html")

    def eda(dataset, output):
        frame = read_dataset(dataset)
        check_dataset(frame)
        return save_report(output, rows=len(frame))

    def clustering(dataset, batch, output, **kwargs):
        return save_report(output)

    monkeypatch.setattr("buy_today.update.report_dataset", eda)
    monkeypatch.setattr(pipeline, "report_clustering", clustering)
    config = pipeline.PipelineConfig(
        model="random", initial_users=6, split_sizes=(4, 2, 2), k=3,
        temporal_n_clusters=3, max_evaluation_rows=8,
    )
    return tmp_path / "run", stream, config


def test_batch_commit_is_observed_only_after_state_is_saved(pipeline_case, caplog, monkeypatch):
    run, stream, config = pipeline_case
    caplog.set_level(logging.INFO, logger="buy_today")
    observed = []

    class ObserveCommit(logging.Handler):
        def emit(self, record):
            if getattr(record, "event", None) == "batch_committed":
                state = json.loads((run / "state.json").read_text(encoding="utf-8"))
                observed.append((record.batch_index, state["next_batch_index"]))

    logger = logging.getLogger("buy_today.pipeline")
    monkeypatch.setattr(logger, "handlers", [*logger.handlers, ObserveCommit()])
    result = pipeline.run_next_batch(run, data_dir=stream, config=config)
    assert observed == [(0, 1)]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    timed_names = {
        "reference", "clustering_train", "clustering_evaluation", "generation",
        "ranking_train", "validation", "test",
    }
    assert set(manifest["durations_seconds"]) == timed_names | {"total"}
    assert all(value >= 0 for value in manifest["durations_seconds"].values())
    for name in timed_names:
        start, done = [record for record in progress_records(caplog) if record.stage == name]
        assert (start.event, done.event) == ("start", "done")
        assert start.stage_fields["batch_index"] == done.stage_fields["batch_index"] == 0
    caplog.clear()
    assert pipeline.run_next_batch(run) is None
    assert observed == [(0, 1)]
    exhausted, = [record for record in caplog.records if record.name == "buy_today.pipeline"]
    assert exhausted.event == "stream_exhausted"
    assert exhausted.levelno == logging.INFO


@pytest.mark.parametrize("failure", ["work", "state_write"])
def test_failed_batch_never_logs_commit(pipeline_case, caplog, monkeypatch, failure):
    run, stream, config = pipeline_case
    caplog.set_level(logging.INFO, logger="buy_today")
    error = OSError("injected failure")
    write_json = pipeline._write_json

    def fail_work(*args, **kwargs):
        raise error

    def fail_state_write(path, value):
        if path == run / "state.json" and value["next_batch_index"] == 1:
            raise error
        return write_json(path, value)

    if failure == "work":
        monkeypatch.setattr(pipeline, "train_ranker", fail_work)
    else:
        monkeypatch.setattr(pipeline, "_write_json", fail_state_write)
    with pytest.raises(OSError) as caught:
        pipeline.run_next_batch(run, data_dir=stream, config=config)
    assert caught.value is error
    assert json.loads((run / "state.json").read_text(encoding="utf-8"))["next_batch_index"] == 0
    assert any(getattr(record, "event", None) == "batch_start" for record in caplog.records)
    assert not any(getattr(record, "event", None) == "batch_committed" for record in caplog.records)
    failed, = [record for record in progress_records(caplog) if record.event == "failed"]
    assert failed.stage == ("ranking_train" if failure == "work" else "pipeline.commit")
    assert failed.exc_info[1] is error


def test_log_io_error_after_commit_propagates_without_rolling_back(pipeline_case, caplog, monkeypatch):
    run, stream, config = pipeline_case
    caplog.set_level(logging.INFO, logger="buy_today")
    error = OSError("log destination failed after commit")

    class FailAfterCommit(logging.Handler):
        def emit(self, record):
            if getattr(record, "event", None) == "batch_committed":
                raise error

    monkeypatch.setattr(logging.getLogger("buy_today.pipeline"), "handlers", [FailAfterCommit()])
    with pytest.raises(OSError) as caught:
        pipeline.run_next_batch(run, data_dir=stream, config=config)
    assert caught.value is error
    assert json.loads((run / "state.json").read_text(encoding="utf-8"))["next_batch_index"] == 1
    assert len(pipeline.read_completed_steps(run)) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_notebook_parent_progress_and_serialization_without_streaming_cells(
    tmp_path, caplog, monkeypatch, fail,
):
    caplog.set_level(logging.INFO, logger="buy_today")
    error = RuntimeError("kernel failed")
    write = nbformat.write

    def serialize(notebook, path):
        active = progress_records(caplog)[-1]
        assert (active.stage, active.event) == ("notebook.serialize", "start")
        write(notebook, path)

    def execute(client, **kwargs):
        active = progress_records(caplog)[-1]
        assert (active.stage, active.event) == ("notebook.execute", "start")
        client.nb.cells[0].outputs = [nbformat.v4.new_output(
            "stream", name="stdout", text="cell output remains in notebook",
        )]
        if fail:
            raise error

    def render(exporter, notebook):
        active = progress_records(caplog)[-1]
        assert (active.stage, active.event) == ("notebook.render", "start")
        return "<p>report</p>", {}

    monkeypatch.setattr("buy_today.auto_eda.notebook.nbformat.write", serialize)
    monkeypatch.setattr("buy_today.auto_eda.notebook.NotebookClient.execute", execute)
    monkeypatch.setattr("buy_today.auto_eda.notebook.HTMLExporter.from_notebook_node", render)
    cells = [nbformat.v4.new_code_cell("print('cell output remains in notebook')")]
    if fail:
        with pytest.raises(RuntimeError) as caught:
            execute_report(cells, tmp_path, title="Progress")
        assert caught.value is error
        assert not (tmp_path / "report.html").exists()
    else:
        execute_report(cells, tmp_path, title="Progress")
        assert (tmp_path / "report.html").is_file()
    notebook = nbformat.read(tmp_path / "report.ipynb", as_version=4)
    assert notebook.cells[0].outputs[0].text == "cell output remains in notebook"
    assert "cell output remains in notebook" not in caplog.text
    records = progress_records(caplog)
    assert records[-1].event == "done"
    if fail:
        failed, = [record for record in records if record.event == "failed"]
        assert failed.stage == "notebook.execute"
        assert failed.exc_info[1] is error
        assert not any(record.stage == "notebook.render" for record in records)
