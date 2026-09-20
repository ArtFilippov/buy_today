import json
from pathlib import Path
import subprocess
import sys

import pytest


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(Path(sys.executable).with_name("prak")), *args],
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
        *extra_args, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "prepared stream/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert [batch["rows"] for batch in manifest["batches"]] == sizes
    assert str(manifest_path) in result.stdout
    assert "Строк после очистки: 12" in result.stdout
    assert "Батчей:" in result.stdout
    assert (tmp_path / "prepared stream/working_dataset.csv").is_file()


@pytest.mark.parametrize("args", [["--help"], ["prepare", "--help"]])
def test_help(tmp_path, args):
    result = run_cli(*args, cwd=tmp_path)
    assert result.returncode == 0
    assert "prepare" in result.stdout


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
