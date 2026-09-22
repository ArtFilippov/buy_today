from dataclasses import replace
import json
from pathlib import Path
import shutil

import joblib
import pandas as pd
import pytest

from buy_today.auto_eda.checks import check_combination, check_dataset
from buy_today.auto_eda.notebook import ReportPaths
from buy_today.cli import main
from buy_today.generation import generate_dataset, read_history_dataset
from buy_today.pipeline import PipelineConfig, read_completed_steps, read_pipeline_config, run_next_batch, run_pipeline
from buy_today.ranking import recommend
from buy_today.schema import DATE_FORMAT, read_dataset
from buy_today.summary import report_summary


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def stream(tmp_path, working_frame, new_batch):
    root = tmp_path / "prepared stream"
    (root / "batches").mkdir(parents=True)
    batches = []
    for index, frame in enumerate((working_frame, new_batch)):
        frame = frame.copy()
        frame["product_id"] = [f"product_{number % 6 + index:02d}" for number in range(len(frame))]
        relative = f"batches/batch_{index:03d}.csv"
        frame.to_csv(root / relative, index=False, date_format=DATE_FORMAT)
        batches.append({"id": f"batch_{index:03d}", "path": relative, "rows": len(frame)})
    (root / "manifest.json").write_text(json.dumps({"format_version": 1, "batches": batches}))
    (root / "state.json").write_text('{"next_batch_index": 0}\n')
    return root


@pytest.fixture
def small_config():
    return PipelineConfig(
        initial_users=6, additional_users=3, split_sizes=(4, 2, 2), k=3,
        svd_n_components=2, svd_n_iter=3, temporal_n_clusters=3, max_evaluation_rows=8,
    )


@pytest.fixture
def fast_reports(monkeypatch):
    """Skip kernel startup only; real references, histories and rankers still run."""
    def save(output, payload):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps({"format_version": 1, **payload}))
        (output / "report.html").write_text("report")
        return ReportPaths(output / "report.ipynb", output / "report.html")

    def eda(dataset, output):
        frame = read_dataset(dataset)
        check_dataset(frame)
        return save(output, {"kind": "eda", "rows": len(frame)})

    def deda(batch, reference, output, **kwargs):
        check_combination(read_dataset(reference), read_dataset(batch))
        return save(output, {"kind": "deda"})

    def clustering(dataset, batch, output, **kwargs):
        return save(output, {"kind": "clustering"})

    monkeypatch.setattr("buy_today.update.report_dataset", eda)
    monkeypatch.setattr("buy_today.update.report_drift", deda)
    monkeypatch.setattr("buy_today.pipeline.report_clustering", clustering)


@pytest.mark.parametrize("model", ["random", "svd"])
def test_two_steps_real_training_preserves_old_histories_and_owns_progress(
    tmp_path, stream, small_config, fast_reports, model,
):
    run = tmp_path / "run"
    first = run_next_batch(run, data_dir=stream, config=replace(small_config, model=model))
    assert first.step_index == 0
    before = {path.relative_to(first.output_dir): path.read_bytes()
              for path in first.output_dir.rglob("*") if path.is_file()}
    first_history = read_history_dataset(first.output_dir / "histories")
    second, = run_pipeline(run, all_batches=True)
    assert second.step_index == 1
    assert before == {path.relative_to(first.output_dir): path.read_bytes()
                      for path in first.output_dir.rglob("*") if path.is_file()}
    second_history = read_history_dataset(second.output_dir / "histories")
    old_events = second_history.events[second_history.events.batch_index.eq(0)].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_events, first_history.events)
    assert len(second_history.anchors) == 9
    assert len(second_history.catalog) == 7
    assert read_json(run / "state.json")["next_batch_index"] == 2
    assert read_json(stream / "state.json") == {"next_batch_index": 0}
    assert len(read_dataset(run / "reference.csv")) == 24
    steps = read_completed_steps(run)
    assert [step["model"] for step in steps] == [model, model]
    assert "deda" not in steps[0]["artifacts"]
    assert "deda" in steps[1]["artifacts"]
    training = read_json(second.output_dir / "ranking/manifest.json")
    assert training["n_users"] == 9
    assert training["inputs"]["train"]["rows"] == 36
    assert training["model"]["class"].endswith("SVDRanker" if model == "svd" else "RandomRanker")
    for split in ("validation", "test"):
        metrics = read_json(second.output_dir / f"ranking/{split}/metrics.json")
        assert metrics["n_users"] == 9
        assert metrics["n_events"] == 18
        assert metrics["model"]["sha256"] == training["model"]["sha256"]
    state_before = (run / "state.json").read_bytes()
    assert run_pipeline(run, all_batches=True) == ()
    assert (run / "state.json").read_bytes() == state_before
    assert read_pipeline_config(run) == replace(small_config, model=model)


def test_full_two_batch_notebook_pipeline_summary_and_inference(tmp_path, stream, small_config):
    run = tmp_path / "full run"
    results = run_pipeline(run, data_dir=stream, config=small_config, all_batches=True)
    assert [result.step_index for result in results] == [0, 1]
    summary = report_summary(run)
    report = read_json(summary.json_path)
    assert [step["data_quality"]["rows"] for step in report["steps"]] == [12, 24]
    assert report["steps"][0]["drift"] is None
    assert report["steps"][1]["drift"]["inputs"]["reference"]["rows"] == 12
    assert all(step["ranking"]["test"]["k"] == 3 for step in report["steps"])
    ranking = results[-1].output_dir / "ranking"
    table = recommend(ranking, "user_000000_000000", 3)
    model = joblib.load(ranking / "model.joblib")
    assert table.product_id.tolist() == model.predict("user_000000_000000", 3).tolist()
    main(["summary", "--run-dir", str(run)])
    csv = tmp_path / "recommendations.csv"
    main(["inference", "--model-dir", str(ranking), "--user-id", "user_000000_000000",
          "--k", "3", "--output", str(csv)])
    pd.testing.assert_frame_equal(pd.read_csv(csv), table)
    for result in results:
        assert (result.output_dir / "clustering/report.ipynb").is_file()
        assert (result.output_dir / "eda/report.html").is_file()


@pytest.mark.parametrize("stage", [
    "update_reference", "train_clustering", "generate_dataset", "train_ranker", "report_ranking",
])
def test_stage_exception_is_propagated_without_advancing_progress(
    tmp_path, stream, small_config, fast_reports, monkeypatch, stage,
):
    run = tmp_path / "run"
    run_next_batch(run, data_dir=stream, config=small_config)
    state_before = (run / "state.json").read_bytes()
    error = RuntimeError(f"failure in {stage}")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(f"buy_today.pipeline.{stage}", fail)
    with pytest.raises(RuntimeError) as caught:
        run_pipeline(run, all_batches=True)
    assert caught.value is error
    assert (run / "state.json").read_bytes() == state_before
    assert len(read_completed_steps(run)) == 1
    assert not (run / "steps/step_001/manifest.json").exists()
    assert not (run / "steps/step_002").exists()
    with pytest.raises((FileExistsError, ValueError)):
        run_next_batch(run)


def test_failed_post_update_eda_leaves_reference_but_cannot_append_again(
    tmp_path, stream, small_config, fast_reports, monkeypatch,
):
    run = tmp_path / "run"
    run_next_batch(run, data_dir=stream, config=small_config)
    error = OSError("EDA export failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("buy_today.update.report_dataset", fail)
    with pytest.raises(OSError) as caught:
        run_next_batch(run)
    assert caught.value is error
    assert len(read_dataset(run / "reference.csv")) == 24
    assert read_json(run / "state.json")["next_batch_index"] == 1
    with pytest.raises(ValueError, match="Reference SHA-256"):
        run_next_batch(run)
    assert len(read_dataset(run / "reference.csv")) == 24


@pytest.mark.parametrize("changed", ["manifest", "batch", "reference", "config", "report"])
def test_changed_inputs_are_rejected_before_starting_next_step(
    tmp_path, stream, small_config, fast_reports, changed,
):
    run = tmp_path / "run"
    run_next_batch(run, data_dir=stream, config=small_config)
    paths = {
        "manifest": stream / "manifest.json", "batch": stream / "batches/batch_001.csv",
        "reference": run / "reference.csv", "config": run / "config.json",
        "report": run / "steps/step_000/ranking/test/metrics.json",
    }
    path = paths[changed]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        run_next_batch(run)
    assert not (run / "steps/step_001").exists()


def test_valid_rebuilt_history_snapshot_cannot_replace_committed_lineage(
    tmp_path, stream, small_config, fast_reports,
):
    run = tmp_path / "run"
    first = run_next_batch(run, data_dir=stream, config=small_config)
    alternate = tmp_path / "different histories"
    generate_dataset(
        stream / "batches/batch_000.csv", first.output_dir / "clustering/distance.joblib", alternate,
        temperature=small_config.temperature, n_users=small_config.initial_users,
        split_sizes=small_config.split_sizes, random_state=7,
    )
    shutil.copytree(alternate, first.output_dir / "histories", dirs_exist_ok=True)
    # The replacement is internally valid, but it is not this run's snapshot.
    read_history_dataset(first.output_dir / "histories")
    with pytest.raises(ValueError, match="SHA-256.*generator/manifest"):
        run_next_batch(run)
    assert not (run / "steps/step_001").exists()


def test_frozen_run_parameters_and_data_source(tmp_path, stream, small_config, fast_reports):
    run = tmp_path / "run"
    run_next_batch(run, data_dir=stream, config=small_config)
    with pytest.raises(ValueError, match="fixed"):
        run_next_batch(run, config=replace(small_config, model="random"))
    with pytest.raises(ValueError, match="differs"):
        run_next_batch(run, data_dir=tmp_path / "other")
    assert not (run / "steps/step_001").exists()


def test_new_run_requires_stream_and_rejects_existing_directories(tmp_path, stream, small_config):
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="data_dir"):
        run_next_batch(run)
    assert not run.exists()
    run.mkdir()
    (run / "user.txt").write_text("keep me")
    with pytest.raises(FileNotFoundError):
        run_next_batch(run, data_dir=stream, config=small_config)
    assert (run / "user.txt").read_text() == "keep me"


@pytest.mark.parametrize("options", [
    {"model": "other"}, {"initial_users": 0}, {"additional_users": True},
    {"temperature": float("inf")}, {"split_sizes": (1, 0, 1)}, {"k": 0},
    {"svd_n_components": 0}, {"svd_n_iter": -1}, {"temporal_n_clusters": 0},
    {"max_evaluation_rows": False}, {"random_state": -1}, {"thresholds": {}},
])
def test_invalid_run_settings(options):
    with pytest.raises(ValueError):
        PipelineConfig(**options)


def test_two_fresh_runs_reproduce_histories_models_and_scores(tmp_path, stream, small_config, fast_reports):
    first = run_pipeline(tmp_path / "a", data_dir=stream, config=small_config, all_batches=True)
    second = run_pipeline(tmp_path / "b", data_dir=stream, config=small_config, all_batches=True)
    for left, right in zip(first, second, strict=True):
        for relative in ("histories/train.csv", "histories/validation.csv", "histories/test.csv",
                         "histories/catalog.csv", "ranking/model.joblib", "clustering/labels.csv"):
            assert (left.output_dir / relative).read_bytes() == (right.output_dir / relative).read_bytes()
        for split in ("validation", "test"):
            a = read_json(left.output_dir / f"ranking/{split}/metrics.json")
            b = read_json(right.output_dir / f"ranking/{split}/metrics.json")
            assert a["metrics"] == b["metrics"]


def test_inconsistent_progress_is_rejected(tmp_path, stream, small_config, fast_reports):
    run = tmp_path / "run"
    run_next_batch(run, data_dir=stream, config=small_config)
    state = read_json(run / "state.json")
    state["next_batch_index"] = 0
    (run / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="progress"):
        run_next_batch(run)


def test_cli_run_configuration_continuation_and_exhaustion(tmp_path, stream, fast_reports, capsys):
    run = tmp_path / "cli run"
    main([
        "run", "--data-dir", str(stream), "--run-dir", str(run), "--model", "svd",
        "--initial-users", "6", "--additional-users", "3", "--split-sizes", "4", "2", "2",
        "--svd-n-components", "2", "--svd-n-iter", "3", "--k", "3",
        "--temporal-n-clusters", "3", "--max-evaluation-rows", "8", "--price-threshold", "0.2",
    ])
    assert "Шаг 000:" in capsys.readouterr().out
    assert read_pipeline_config(run).thresholds.price == .2
    main(["run", "--run-dir", str(run), "--model", "svd", "--all"])
    assert "Шаг 001:" in capsys.readouterr().out
    main(["run", "--run-dir", str(run)])
    assert "Поток завершён" in capsys.readouterr().out
    with pytest.raises(ValueError, match="fixed"):
        main(["run", "--run-dir", str(run), "--temperature", "100"])


@pytest.mark.parametrize("command,function,args", [
    ("run", "run_pipeline", ["--run-dir", "unused"]),
    ("inference", "export_recommendations", ["--model-dir", "unused", "--user-id", "u", "--output", "unused.csv"]),
    ("summary", "report_summary", ["--run-dir", "unused"]),
])
def test_new_cli_commands_propagate_original_exceptions(monkeypatch, command, function, args):
    error = RuntimeError("original stage failure")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(f"buy_today.cli.{function}", fail)
    with pytest.raises(RuntimeError) as caught:
        main([command, *args])
    assert caught.value is error


@pytest.mark.parametrize("command", ["run", "inference", "summary"])
def test_new_cli_help(command, capsys):
    with pytest.raises(SystemExit) as caught:
        main([command, "--help"])
    assert caught.value.code == 0
    assert "buy_today" in capsys.readouterr().out


def test_cli_rejects_svd_options_for_random(tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(["run", "--run-dir", str(tmp_path / "run"), "--model", "random", "--svd-n-components", "2"])
    assert caught.value.code == 2
