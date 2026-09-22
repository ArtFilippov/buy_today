import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import nbformat
import pandas as pd
import pytest

from buy_today.auto_eda import DriftThresholds
from buy_today.auto_eda.notebook import ReportPaths
from buy_today.cli import main
from buy_today.schema import SORT_KEY, read_dataset
from buy_today.update import UpdateReports


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(Path(sys.executable).with_name("buy_today")), *args],
        cwd=cwd, text=True, capture_output=True, check=False,
    )


@pytest.mark.parametrize(("extra_args", "sizes"), [
    ([], [6]),
    (["--batch-size", "2"], [6, 2, 2, 2]),
])
def test_prepare_command(raw_tables, write_raw, tmp_path, extra_args, sizes):
    raw_dir = write_raw(raw_tables)
    result = run_cli(
        "prepare", "--raw-dir", raw_dir.name, "--output-dir", "prepared stream",
        "--min-category-count", "1", *extra_args, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "prepared stream/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert [batch["rows"] for batch in manifest["batches"]] == sizes
    assert manifest["parameters"]["min_category_count"] == 1
    assert str(manifest_path) in result.stdout
    assert "Строк после очистки: 12" in result.stdout
    assert "Удалено неполных строк: 0" in result.stdout
    assert "Удалено строк редких категорий: 0" in result.stdout
    assert "Строк после фильтрации категорий: 12" in result.stdout
    assert "Батчей:" in result.stdout
    assert (tmp_path / "prepared stream/working_dataset.csv").is_file()


def test_prepare_default_category_threshold(raw_category_tables, write_raw, tmp_path):
    tables = raw_category_tables({"rare": 999, "boundary": 1000, "common": 1001})
    tables["olist_orders_dataset.csv"].loc[0, "order_approved_at"] = None
    raw_dir = write_raw(tables)
    result = run_cli(
        "prepare", "--raw-dir", raw_dir.name, "--output-dir", "stream", cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "Удалено неполных строк: 1" in result.stdout
    assert "Удалено строк редких категорий: 998" in result.stdout
    assert "Категорий: 3 → 2" in result.stdout
    assert "Строк после фильтрации категорий: 2,001" in result.stdout
    manifest = json.loads((tmp_path / "stream/manifest.json").read_text())
    assert manifest["parameters"]["min_category_count"] == 1000
    assert read_dataset(tmp_path / "stream/working_dataset.csv")[
        "product_category_name"
    ].value_counts().to_dict() == {"boundary": 1000, "common": 1001}


@pytest.mark.parametrize("threshold", ["0", "-1", "1.5"])
def test_prepare_invalid_category_threshold(tmp_path, threshold):
    result = run_cli(
        "prepare", "--raw-dir", "absent", "--output-dir", "stream",
        "--min-category-count", threshold, cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "--min-category-count" in result.stderr


def test_prepare_no_categories_is_failure(raw_tables, write_raw, tmp_path):
    raw_dir = write_raw(raw_tables)
    result = run_cli(
        "prepare", "--raw-dir", raw_dir.name, "--output-dir", "stream", cwd=tmp_path,
    )
    assert result.returncode == 1
    assert "min_category_count=1000, retained rows=0" in result.stderr
    assert not (tmp_path / "stream").exists()


@pytest.mark.parametrize("args", [
    ["--help"], ["prepare", "--help"], ["eda", "--help"], ["init", "--help"],
    ["deda", "--help"], ["update", "--help"], ["cluster", "--help"], ["evaluate", "--help"],
    ["generate", "--help"],
])
def test_help(tmp_path, args):
    result = run_cli(*args, cwd=tmp_path)
    assert result.returncode == 0
    assert "buy_today" in result.stdout
    if args[0] == "prepare":
        assert "--min-category-count" in result.stdout
    if args[0] == "cluster":
        assert "--temporal-n-clusters" in result.stdout
        assert "--n-clusters" not in result.stdout
    if args[0] in {"deda", "update"}:
        for name in ("batch", "reference", "output-dir", "price-threshold", "category-threshold", "state-threshold"):
            assert f"--{name}" in result.stdout


def test_invalid_batch_size(tmp_path):
    result = run_cli(
        "prepare", "--raw-dir", "absent", "--output-dir", "stream",
        "--batch-size", "0", cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "--batch-size" in result.stderr


def test_missing_source_is_failure(tmp_path):
    result = run_cli(
        "prepare", "--raw-dir", "absent", "--output-dir", "stream", cwd=tmp_path
    )
    assert result.returncode == 1
    assert "Ошибка подготовки" in result.stderr
    assert "olist_order_items_dataset.csv" in result.stderr


@pytest.mark.parametrize("command", ["eda", "init"])
def test_report_commands_from_another_cwd(working_frame, write_dataset, tmp_path, command):
    dataset = write_dataset(working_frame)
    before = dataset.read_bytes()
    if command == "eda":
        inputs = ["--dataset", dataset.name]
        report_dir = tmp_path / "report dir"
    else:
        inputs = ["--batch", dataset.name, "--reference", "saved reference/reference.csv"]
        report_dir = tmp_path / "report dir/eda"
    result = run_cli(command, *inputs, "--output-dir", "report dir", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (report_dir / "report.ipynb").is_file()
    assert (report_dir / "report.html").is_file()
    assert str(report_dir / "report.html") in result.stdout
    assert dataset.read_bytes() == before
    if command == "init":
        from buy_today.schema import read_dataset
        import pandas as pd
        pd.testing.assert_frame_equal(
            read_dataset(tmp_path / "saved reference/reference.csv"), working_frame,
        )


@pytest.mark.parametrize("args", [
    ["eda"], ["init"], ["eda", "--dataset", "input.csv"],
    ["init", "--batch", "input.csv", "--output-dir", "out"],
    ["eda", "--dataset", "input.csv", "--output-dir", "out", "--price-threshold", "0.1"],
    ["init", "--batch", "input.csv", "--reference", "ref.csv", "--output-dir", "out", "--state-threshold", "0.1"],
])
def test_report_invalid_arguments_exit_two(tmp_path, args):
    assert run_cli(*args, cwd=tmp_path).returncode == 2


@pytest.mark.parametrize("command", ["eda", "init"])
def test_report_missing_input_exits_one(tmp_path, command):
    inputs = ["--dataset", "absent.csv"] if command == "eda" else [
        "--batch", "absent.csv", "--reference", "ref.csv",
    ]
    result = run_cli(command, *inputs, "--output-dir", "reports", cwd=tmp_path)
    assert result.returncode == 1
    assert f"Ошибка {command}" in result.stderr
    assert "absent.csv" in result.stderr


@pytest.mark.parametrize("command", ["eda", "init"])
def test_report_invalid_data_exits_one(working_frame, write_dataset, tmp_path, command):
    working_frame.loc[0, "price"] = -1
    dataset = write_dataset(working_frame)
    inputs = ["--dataset", dataset.name] if command == "eda" else [
        "--batch", dataset.name, "--reference", "ref.csv",
    ]
    result = run_cli(command, *inputs, "--output-dir", "reports", cwd=tmp_path)
    assert result.returncode == 1
    assert "Конечная положительная цена" in result.stderr
    assert not (tmp_path / "ref.csv").exists()


@pytest.mark.parametrize("command", ["deda", "update"])
def test_drift_commands_from_another_cwd_with_real_drift(
    working_frame, new_batch, write_dataset, tmp_path, command,
):
    new_batch["price"] *= 10
    new_batch["product_category_name"] = "new_category"
    new_batch["customer_state"] = "AM"
    reference = write_dataset(working_frame, "reference ' old \" data.csv")
    batch = write_dataset(new_batch, "batch ' new \" data.csv")
    reference_before, batch_before = reference.read_bytes(), batch.read_bytes()
    report_dir = tmp_path / "reports ' current \" step"
    deda_dir = report_dir if command == "deda" else report_dir / "deda"
    ignored = {}
    if command == "update":
        # A new step must use the CSVs even when history/state is unreadable.
        old_reports = tmp_path / "old reports"
        old_reports.mkdir()
        for path in (
            tmp_path / "state.json", tmp_path / "manifest.json",
            old_reports / "report.ipynb", old_reports / "report.html",
        ):
            path.write_bytes(b"not a valid report or JSON\x00")
            ignored[path] = path.read_bytes()
    # Reuse a nonempty destination, as the report contract allows.
    deda_dir.mkdir(parents=True)
    for name in ("report.ipynb", "report.html"):
        (deda_dir / name).write_text("old report", encoding="utf-8")

    result = run_cli(
        command, "--batch", batch.name, "--reference", reference.name,
        "--output-dir", report_dir.name, "--price-threshold", "2.3e-1",
        "--category-threshold", "0.47", "--state-threshold", "0.89", cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert batch.read_bytes() == batch_before
    for path, content in ignored.items():
        assert path.read_bytes() == content
    notebook = nbformat.read(deda_dir / "report.ipynb", as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    source = "\n".join(cell.source for cell in code)
    for path in (batch, reference):
        assert repr(str(path)) in source
    for name, value in (("price", 0.23), ("category", 0.47), ("state", 0.89)):
        assert f"{name}={value}" in source
    outputs = [output for cell in code for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    text = "\n".join(output.get("text", "") for output in outputs)
    assert str(batch) in text and str(reference) in text
    assert hashlib.sha256(reference_before).hexdigest() in text
    assert hashlib.sha256(batch_before).hexdigest() in text
    assert "drift_detected = True" in text
    assert "Эталон до добавления: 12 позиций; батч: 12 позиций." in text
    metric_tables = [
        output.get("data", {}).get("text/html", "") for output in outputs
        if "<th>Порог</th>" in output.get("data", {}).get("text/html", "")
    ]
    assert len(metric_tables) == 1
    values = re.findall(r"<td>(.*?)</td>", metric_tables[0])
    rows = [values[index:index + 5] for index in range(0, len(values), 5)]
    assert {row[0]: float(row[3]) for row in rows} == {
        "price": 0.23, "product_category_name": 0.47, "customer_state": 0.89,
    }
    for row in rows:
        assert float(row[2]) == 1.0
        assert row[4] == "True"

    report_dirs = [deda_dir]
    if command == "update":
        expected = pd.concat([working_frame, new_batch], ignore_index=True).sort_values(
            list(SORT_KEY), ignore_index=True,
        )
        pd.testing.assert_frame_equal(read_dataset(reference), expected, check_exact=True)
        assert f"Эталон: {reference}" in result.stdout
        report_dirs.append(report_dir / "eda")
        eda = nbformat.read(report_dir / "eda/report.ipynb", as_version=4)
        eda_text = "\n".join(
            output.get("text", "") for cell in eda.cells if cell.cell_type == "code"
            for output in cell.outputs
        )
        assert "24 позиций × 35 колонок" in eda_text
        assert hashlib.sha256(reference.read_bytes()).hexdigest() in eda_text
        assert {path.name for path in report_dir.iterdir()} == {"deda", "eda"}
    else:
        assert reference.read_bytes() == reference_before
        assert not (tmp_path / "state.json").exists()
        assert not (tmp_path / "manifest.json").exists()
    for directory in report_dirs:
        assert {path.name for path in directory.iterdir()} == {"report.ipynb", "report.html", "metrics.json"}
        for name in ("report.ipynb", "report.html"):
            assert str(directory / name) in result.stdout
        assert "Итог" in (directory / "report.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("command", ["deda", "update"])
@pytest.mark.parametrize("missing", ["batch", "reference", "output-dir"])
def test_drift_commands_require_every_path(command, missing, capsys):
    args = [command]
    for name in ("batch", "reference", "output-dir"):
        if name != missing:
            args.extend([f"--{name}", "unused"])
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert f"--{missing}" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["deda", "update"])
@pytest.mark.parametrize("feature", ["price", "category", "state"])
@pytest.mark.parametrize("value", ["-0.01", "1.01", "nan", "inf", "-inf", "1e999", "not-a-number"])
def test_invalid_drift_thresholds_exit_two(command, feature, value, capsys):
    with pytest.raises(SystemExit) as error:
        main([
            command, "--batch", "batch.csv", "--reference", "reference.csv",
            "--output-dir", "reports", f"--{feature}-threshold={value}",
        ])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert f"--{feature}-threshold" in stderr
    assert "конечное число в [0, 1]" in stderr


@pytest.mark.parametrize("command", ["deda", "update"])
@pytest.mark.parametrize("value", [None, 0.0, 1.0])
def test_drift_commands_forward_default_and_boundary_thresholds(
    tmp_path, monkeypatch, command, value,
):
    expected = DriftThresholds() if value is None else DriftThresholds(value, value, value)
    report = ReportPaths(tmp_path / "report.ipynb", tmp_path / "report.html")
    calls = []

    def run_report(batch_path, reference_path, output_dir, *, thresholds):
        calls.append((batch_path, reference_path, output_dir, thresholds))
        return report if command == "deda" else UpdateReports(deda=report, eda=report)

    target = "report_drift" if command == "deda" else "update_reference"
    monkeypatch.setattr(f"buy_today.cli.{target}", run_report)
    args = [command, "--batch", "batch.csv", "--reference", "ref.csv", "--output-dir", "reports"]
    if value is not None:
        for feature in ("price", "category", "state"):
            args.extend([f"--{feature}-threshold", str(value)])
    main(args)
    assert calls == [(Path("batch.csv"), Path("ref.csv"), Path("reports"), expected)]


@pytest.mark.parametrize("command", ["deda", "update"])
@pytest.mark.parametrize("failure", [
    ValueError("invalid input"), RuntimeError("kernel failed"), OSError("HTML export failed"),
])
def test_drift_commands_report_technical_errors_as_exit_one(
    monkeypatch, capsys, command, failure,
):
    def fail_report(*args, **kwargs):
        raise failure

    target = "report_drift" if command == "deda" else "update_reference"
    monkeypatch.setattr(f"buy_today.cli.{target}", fail_report)
    with pytest.raises(SystemExit) as error:
        main([command, "--batch", "batch.csv", "--reference", "ref.csv", "--output-dir", "reports"])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert f"Ошибка {command}: {failure}" in captured.err


@pytest.mark.parametrize("command", ["deda", "update"])
@pytest.mark.parametrize(("args", "code", "message"), [
    (["--batch", "absent.csv", "--output-dir", "reports"], 2, "--reference"),
    (["--batch", "absent.csv", "--reference", "missing.csv", "--output-dir", "reports",
      "--state-threshold", "nan"], 2, "--state-threshold"),
    (["--batch", "absent.csv", "--reference", "missing.csv", "--output-dir", "reports"], 1, "missing.csv"),
])
def test_drift_command_exit_codes_in_subprocess(tmp_path, command, args, code, message):
    result = run_cli(command, *args, cwd=tmp_path)
    assert result.returncode == code
    assert message in result.stderr
    assert not result.stdout
    assert not (tmp_path / "missing.csv").exists()


def test_cluster_then_evaluate_from_another_cwd(working_frame, write_dataset, tmp_path):
    dataset = write_dataset(working_frame, 'data \' with " quotes.csv')
    model_dir = tmp_path / "model ' dir"
    result = run_cli(
        "cluster", "--batch", dataset.name, "--output-dir", model_dir.name,
        "--model", "temporal", "--distance", "timestamp", "--temporal-n-clusters", "3", cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    artifacts = {path: path.read_bytes() for path in model_dir.iterdir()}
    assert {path.name for path in artifacts} == {"model.joblib", "distance.joblib", "labels.csv"}
    assert str(model_dir / "labels.csv") in result.stdout
    result = run_cli(
        "evaluate", "--dataset", dataset.name, "--new-batch", dataset.name,
        "--model-dir", model_dir.name, "--max-evaluation-rows", "8", "--random-state", "17", cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert str(model_dir / "report.html") in result.stdout
    html = (model_dir / "report.html").read_text()
    assert "Выбрано строк: 8; новых: 8; прежних: 0" in html
    assert "random_state: 17" in html
    assert {path: path.read_bytes() for path in artifacts} == artifacts


@pytest.mark.parametrize("args", [
    ["cluster"], ["evaluate"],
    ["cluster", "--batch", "a", "--output-dir", "b", "--temporal-n-clusters", "0"],
    ["cluster", "--batch", "a", "--output-dir", "b", "--temporal-n-clusters", "-1"],
    ["cluster", "--batch", "a", "--output-dir", "b", "--temporal-n-clusters", "1.5"],
    ["cluster", "--batch", "a", "--output-dir", "b", "--n-clusters", "3"],
    ["cluster", "--batch", "a", "--output-dir", "b", "--model", "unknown"],
    ["cluster", "--batch", "a", "--output-dir", "b", "--distance", "unknown"],
    ["cluster", "--batch", "a", "--output-dir", "b", "--timestamp-column", "time"],
    ["cluster", "--batch", "a", "--output-dir", "b", "--max-evaluation-rows", "5"],
    ["evaluate", "--dataset", "a", "--model-dir", "b"],
    ["evaluate", "--dataset", "a", "--new-batch", "a", "--model-dir", "b", "--max-evaluation-rows", "0"],
    ["evaluate", "--dataset", "a", "--new-batch", "a", "--model-dir", "b", "--random-state", "-1"],
    ["evaluate", "--dataset", "a", "--new-batch", "a", "--model-dir", "b", "--random-state", "4294967296"],
    ["evaluate", "--dataset", "a", "--new-batch", "a", "--model-dir", "b", "--random-state", "1.5"],
])
def test_clustering_invalid_arguments_exit_two(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["cluster", "evaluate"])
def test_clustering_missing_input_exits_one(tmp_path, command):
    args = ["--batch", "missing.csv", "--output-dir", "model"] if command == "cluster" else [
        "--dataset", "missing.csv", "--new-batch", "missing.csv", "--model-dir", "model",
    ]
    result = run_cli(command, *args, cwd=tmp_path)
    assert result.returncode == 1
    assert f"Ошибка {command}" in result.stderr


def test_cluster_too_many_clusters_is_data_error(working_frame, write_dataset, tmp_path):
    dataset = write_dataset(working_frame)
    result = run_cli("cluster", "--batch", dataset.name, "--output-dir", "model", cwd=tmp_path)
    assert result.returncode == 1
    assert "n_samples must be >= n_clusters" in result.stderr


@pytest.mark.parametrize("override", [None, 3])
def test_cluster_leaves_default_to_temporal(monkeypatch, tmp_path, override):
    from buy_today.clustering.training import TrainingPaths

    calls = []

    def strategy(frame, **kwargs):
        calls.append(kwargs)

    def train(batch_path, output_dir, *, strategy):
        strategy(object())
        return TrainingPaths(*(tmp_path / name for name in ("model", "distance", "labels")))

    monkeypatch.setattr("buy_today.cli.train_temporal", strategy)
    monkeypatch.setattr("buy_today.cli.train_clustering", train)
    args = ["cluster", "--batch", "batch.csv", "--output-dir", "model"]
    if override is not None:
        args.extend(["--temporal-n-clusters", str(override)])
    main(args)
    assert len(calls) == 1
    calls[0].pop("distance")
    assert calls[0] == ({} if override is None else {"n_clusters": override})
