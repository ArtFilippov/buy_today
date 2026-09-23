import json
from typing import Any, Never

import fixture_types as ft
from nbclient.exceptions import CellExecutionError
import pandas as pd
import pytest

from buy_today.auto_eda import DriftThresholds
from buy_today.auto_eda.notebook import ReportPaths
from buy_today.schema import DATE_FORMAT, ROW_KEY, SORT_KEY, read_dataset
from buy_today.update import UpdateReports, initialize_reference, update_reference


@pytest.mark.parametrize("existing", [False, True])
def test_init_saves_exact_reference_before_reporting(
    working_frame: ft.DataFrame,
    write_dataset: ft.DatasetWriter,
    tmp_path: ft.Path,
    monkeypatch: ft.MonkeyPatch,
    existing: bool,
) -> None:
    reference = tmp_path / "reference dir/reference.csv"
    if existing:
        reference.parent.mkdir()
        reference.write_text("old reference", encoding="utf-8")
    # Ties may arrive in any key order; the persisted CSV uses SORT_KEY.
    frame = working_frame.iloc[[1, 0, *range(2, 12)]]
    batch = write_dataset(frame)
    batch_before = batch.read_bytes()
    expected = frame.sort_values(list(SORT_KEY)).reset_index(drop=True)
    reports = tmp_path / "step"
    result = ReportPaths(reports / "eda/report.ipynb", reports / "eda/report.html")
    calls: list[tuple[ft.Path | str, ft.Path | str]] = []

    def report_saved(dataset_path: ft.Path | str, output_dir: ft.Path | str) -> ReportPaths:
        calls.append((dataset_path, output_dir))
        pd.testing.assert_frame_equal(read_dataset(dataset_path), expected)
        return result

    monkeypatch.setattr("buy_today.update.report_dataset", report_saved)
    assert initialize_reference(str(batch), str(reference), str(reports)) == result
    assert calls == [(reference, reports / "eda")]
    assert batch.read_bytes() == batch_before
    assert read_dataset(reference)["customer_zip_code_prefix"].iloc[0] == "00123"


def test_invalid_init_leaves_old_reference(
    working_frame: ft.DataFrame, write_dataset: ft.DatasetWriter, tmp_path: ft.Path
) -> None:
    reference = tmp_path / "reference.csv"
    reference.write_bytes(b"existing reference")
    working_frame.loc[1, "order_item_id"] = 1
    with pytest.raises(ValueError, match="Повторы ключа"):
        initialize_reference(write_dataset(working_frame), reference, tmp_path / "reports")
    assert reference.read_bytes() == b"existing reference"
    assert not (tmp_path / "reports").exists()


def test_failed_eda_keeps_newly_written_reference(
    working_frame: ft.DataFrame,
    write_dataset: ft.DatasetWriter,
    tmp_path: ft.Path,
    monkeypatch: ft.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.csv"
    reference.write_bytes(b"old reference")

    def fail_report(*args: object) -> Never:
        raise RuntimeError("EDA failed")

    monkeypatch.setattr("buy_today.update.report_dataset", fail_report)
    with pytest.raises(RuntimeError, match="EDA failed"):
        initialize_reference(write_dataset(working_frame), reference, tmp_path / "reports")
    pd.testing.assert_frame_equal(read_dataset(reference), working_frame)


def test_update_reports_before_and_after_exact_sorted_merge(
    working_frame: ft.DataFrame,
    new_batch: ft.DataFrame,
    write_dataset: ft.DatasetWriter,
    tmp_path: ft.Path,
    monkeypatch: ft.MonkeyPatch,
) -> None:
    # Both inputs are chronological, but their times interleave. Equal times
    # also require sorting by order_id and order_item_id, not just appending.
    new_batch["order_purchase_timestamp"] = working_frame[
        "order_purchase_timestamp"
    ] + pd.Timedelta(minutes=30)
    new_batch.loc[:1, "order_purchase_timestamp"] = working_frame.loc[0, "order_purchase_timestamp"]
    reference = write_dataset(working_frame.iloc[[1, 0, *range(2, 12)]], "reference.csv")
    batch = write_dataset(new_batch.iloc[[1, 0, *range(2, 12)]])
    reference_before, batch_before = reference.read_bytes(), batch.read_bytes()
    expected = pd.concat([working_frame, new_batch], ignore_index=True).sort_values(
        list(SORT_KEY),
        ignore_index=True,
    )
    reports = tmp_path / "step"
    deda = ReportPaths(reports / "deda/report.ipynb", reports / "deda/report.html")
    eda = ReportPaths(reports / "eda/report.ipynb", reports / "eda/report.html")
    thresholds = DriftThresholds(price=0.2, category=0.3, state=0.4)
    calls: list[str] = []

    def report_before(
        batch_path: ft.Path | str,
        reference_path: ft.Path | str,
        output_dir: ft.Path | str,
        *,
        thresholds: DriftThresholds,
    ) -> ReportPaths:
        assert (batch_path, reference_path, output_dir) == (str(batch), reference, reports / "deda")
        assert thresholds == DriftThresholds(price=0.2, category=0.3, state=0.4)
        assert reference.read_bytes() == reference_before
        calls.append("deda")
        # Paths are enough: the coordinator must not inspect notebook outputs.
        return deda

    def read_after_deda(path: ft.Path | str) -> ft.DataFrame:
        assert calls == ["deda"]
        return read_dataset(path)

    def report_after(dataset_path: ft.Path | str, output_dir: ft.Path | str) -> ReportPaths:
        assert (dataset_path, output_dir) == (reference, reports / "eda")
        assert calls == ["deda"]
        pd.testing.assert_frame_equal(read_dataset(dataset_path), expected, check_exact=True)
        calls.append("eda")
        return eda

    monkeypatch.setattr("buy_today.update.report_drift", report_before)
    monkeypatch.setattr("buy_today.update.read_dataset", read_after_deda)
    monkeypatch.setattr("buy_today.update.report_dataset", report_after)
    result = update_reference(str(batch), str(reference), str(reports), thresholds=thresholds)
    assert isinstance(result, UpdateReports)
    assert (result.deda, result.eda) == (deda, eda)
    assert calls == ["deda", "eda"]
    assert batch.read_bytes() == batch_before
    assert reference.read_text(encoding="utf-8") == expected.to_csv(
        index=False, date_format=DATE_FORMAT
    )
    assert not reports.exists()
    assert not (tmp_path / "state.json").exists()


def test_update_missing_reference_fails_before_deda(
    tmp_path: ft.Path, monkeypatch: ft.MonkeyPatch
) -> None:
    reference = tmp_path / "absent/reference.csv"

    def unexpected_call(*args: object, **kwargs: object) -> Never:
        pytest.fail("Missing reference must fail before reports or CSV reads")

    monkeypatch.setattr("buy_today.update.report_drift", unexpected_call)
    monkeypatch.setattr("buy_today.update.report_dataset", unexpected_call)
    monkeypatch.setattr("buy_today.update.read_dataset", unexpected_call)
    with pytest.raises(FileNotFoundError, match="reference.csv"):
        update_reference(tmp_path / "also absent.csv", reference, tmp_path / "step")
    assert not reference.parent.exists()
    assert not (tmp_path / "step").exists()


def test_update_deda_technical_failure_preserves_reference(
    working_frame: ft.DataFrame,
    new_batch: ft.DataFrame,
    write_dataset: ft.DatasetWriter,
    tmp_path: ft.Path,
    monkeypatch: ft.MonkeyPatch,
) -> None:
    reference = write_dataset(working_frame, "reference.csv")
    batch = write_dataset(new_batch)
    before = reference.read_bytes()

    def fail_deda(*args: object, **kwargs: object) -> Never:
        raise RuntimeError("DEDA kernel failed")

    def unexpected_call(*args: object, **kwargs: object) -> Never:
        pytest.fail("Failed DEDA must stop before CSV reads or EDA")

    monkeypatch.setattr("buy_today.update.report_drift", fail_deda)
    monkeypatch.setattr("buy_today.update.read_dataset", unexpected_call)
    monkeypatch.setattr("buy_today.update.report_dataset", unexpected_call)
    with pytest.raises(RuntimeError, match="DEDA kernel failed"):
        update_reference(batch, reference, tmp_path / "step")
    assert reference.read_bytes() == before


def test_update_html_export_failure_preserves_reference(
    working_frame: ft.DataFrame,
    new_batch: ft.DataFrame,
    write_dataset: ft.DatasetWriter,
    tmp_path: ft.Path,
    monkeypatch: ft.MonkeyPatch,
) -> None:
    reference = write_dataset(working_frame, "reference.csv")
    batch = write_dataset(new_batch)
    before = reference.read_bytes()
    reports = tmp_path / "step"
    exported: list[Any] = []

    def fail_export(self: object, notebook: Any, *args: object, **kwargs: object) -> Never:
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in code)
        assert reference.read_bytes() == before
        exported.append(notebook)
        raise OSError("HTML export failed")

    def unexpected_eda(*args: object, **kwargs: object) -> Never:
        pytest.fail("HTML export must complete before writing the reference or EDA")

    monkeypatch.setattr("buy_today.auto_eda.notebook.HTMLExporter.from_notebook_node", fail_export)
    monkeypatch.setattr("buy_today.update.report_dataset", unexpected_eda)
    with pytest.raises(OSError, match="HTML export failed"):
        update_reference(batch, reference, reports)
    assert len(exported) == 1
    assert reference.read_bytes() == before
    assert (reports / "deda/report.ipynb").is_file()
    assert not (reports / "deda/report.html").exists()
    assert not (reports / "eda").exists()


def test_update_failed_final_eda_keeps_merged_reference(
    working_frame: ft.DataFrame,
    new_batch: ft.DataFrame,
    write_dataset: ft.DatasetWriter,
    tmp_path: ft.Path,
    monkeypatch: ft.MonkeyPatch,
) -> None:
    reference = write_dataset(working_frame, "reference.csv")
    batch = write_dataset(new_batch)
    expected = pd.concat([working_frame, new_batch], ignore_index=True)
    reports = tmp_path / "step"

    def successful_deda(*args: object, **kwargs: object) -> ReportPaths:
        return ReportPaths(reports / "deda/report.ipynb", reports / "deda/report.html")

    def fail_eda(dataset_path: ft.Path | str, output_dir: ft.Path | str) -> Never:
        assert (dataset_path, output_dir) == (reference, reports / "eda")
        pd.testing.assert_frame_equal(read_dataset(dataset_path), expected, check_exact=True)
        raise RuntimeError("Final EDA failed")

    monkeypatch.setattr("buy_today.update.report_drift", successful_deda)
    monkeypatch.setattr("buy_today.update.report_dataset", fail_eda)
    with pytest.raises(RuntimeError, match="Final EDA failed"):
        update_reference(batch, reference, reports)
    pd.testing.assert_frame_equal(read_dataset(reference), expected, check_exact=True)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("duplicate", "в объединении: фактически 12"),
        ("partial overlap", "в объединении: фактически 1"),
        ("reference", "Конечная положительная цена"),
        ("batch", "Expected the 35 working dataset columns"),
    ],
)
def test_update_real_deda_rejects_invalid_inputs_before_writing(
    working_frame: ft.DataFrame,
    new_batch: ft.DataFrame,
    write_dataset: ft.DatasetWriter,
    tmp_path: ft.Path,
    monkeypatch: ft.MonkeyPatch,
    invalid: str,
    message: str,
) -> None:
    if invalid == "duplicate":
        new_batch = working_frame.copy()
    elif invalid == "partial overlap":
        new_batch.loc[0, list(ROW_KEY)] = working_frame.loc[0, list(ROW_KEY)]
    elif invalid == "reference":
        working_frame.loc[0, "price"] = -1
    else:
        new_batch = new_batch.drop(columns="price")
    reference = write_dataset(working_frame, "reference.csv")
    batch = write_dataset(new_batch)
    reference_before, batch_before = reference.read_bytes(), batch.read_bytes()
    reports = tmp_path / "step"

    def unexpected_eda(*args: object, **kwargs: object) -> Never:
        pytest.fail("Rejected inputs must not reach EDA")

    monkeypatch.setattr("buy_today.update.report_dataset", unexpected_eda)
    with pytest.raises(CellExecutionError, match=message):
        update_reference(batch, reference, reports)
    assert reference.read_bytes() == reference_before
    assert batch.read_bytes() == batch_before
    assert not (reports / "eda").exists()
    assert not (reports / "deda/report.html").exists()
    notebook = json.loads((reports / "deda/report.ipynb").read_text(encoding="utf-8"))
    outputs = [
        output
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for output in cell["outputs"]
    ]
    assert any(output["output_type"] == "error" for output in outputs)
    assert not any("image/png" in output.get("data", {}) for output in outputs)
