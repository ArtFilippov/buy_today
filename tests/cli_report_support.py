"""Assertions for CLI reports executed by real notebook kernels."""

import hashlib
from pathlib import Path
import re

import nbformat
import pandas as pd

from buy_today.schema import SORT_KEY, read_dataset


def check_drift_notebook(
    directory: Path,
    batch: Path,
    reference: Path,
    originals: tuple[bytes, bytes],
) -> None:
    reference_before, batch_before = originals
    notebook = nbformat.read(directory / "report.ipynb", as_version=4)
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
        output.get("data", {}).get("text/html", "")
        for output in outputs
        if "<th>Порог</th>" in output.get("data", {}).get("text/html", "")
    ]
    assert len(metric_tables) == 1
    values = re.findall(r"<td>(.*?)</td>", metric_tables[0])
    rows = [values[index : index + 5] for index in range(0, len(values), 5)]
    assert {row[0]: float(row[3]) for row in rows} == {
        "price": 0.23,
        "product_category_name": 0.47,
        "customer_state": 0.89,
    }
    for row in rows:
        assert float(row[2]) == 1.0
        assert row[4] == "True"


def check_updated_reference(
    reference: Path,
    report_dir: Path,
    working_frame: pd.DataFrame,
    new_batch: pd.DataFrame,
) -> None:
    expected = pd.concat([working_frame, new_batch], ignore_index=True).sort_values(
        list(SORT_KEY),
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(read_dataset(reference), expected, check_exact=True)
    eda = nbformat.read(report_dir / "eda/report.ipynb", as_version=4)
    text = "\n".join(
        output.get("text", "")
        for cell in eda.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
    assert "24 позиций × 35 колонок" in text
    assert hashlib.sha256(reference.read_bytes()).hexdigest() in text
    assert {path.name for path in report_dir.iterdir()} == {"deda", "eda"}
