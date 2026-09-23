import json
from pathlib import Path
import subprocess
import sys

import pytest

from buy_today.cli import main


def test_cli_from_another_cwd_with_quoted_paths_and_iid_options(
    quality_snapshot: tuple[Path, Path], tmp_path: Path,
) -> None:
    dataset, reference = quality_snapshot
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("buy_today")), "history-quality",
            "--dataset-dir", dataset.name, "--reference", reference.name,
            "--output-dir", "quality ' report", "--iid-repeats", "7", "--random-state", "17",
        ],
        cwd=tmp_path, text=True, capture_output=True, check=False,
    )
    assert not result.returncode, result.stderr
    output = tmp_path / "quality ' report/metrics.json"
    assert str(output) in result.stdout
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["iid"]["repeats"] == 7
    assert metrics["iid"]["random_state"] == 17


@pytest.mark.parametrize("missing", ["dataset-dir", "reference", "output-dir"])
def test_cli_requires_all_paths(missing: str) -> None:
    arguments = ["history-quality"]
    for option in ("dataset-dir", "reference", "output-dir"):
        if option != missing:
            arguments.extend([f"--{option}", "path"])
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("arguments", [
    ["--iid-repeats", "0"], ["--iid-repeats", "-1"], ["--iid-repeats", "1.5"],
    ["--random-state", "-1"], ["--random-state", str(2**32)], ["--split", "train"],
    ["--reference", "second.csv"],
])
def test_cli_rejects_invalid_options(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main([
            "history-quality", "--dataset-dir", "snapshot", "--reference", "reference",
            "--output-dir", "output", *arguments,
        ])
    assert error.value.code == 2


def test_cli_input_error_exits_one_without_success_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main([
            "history-quality", "--dataset-dir", str(tmp_path / "absent"),
            "--reference", "reference", "--output-dir", str(tmp_path / "output"),
        ])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "Ошибка history-quality" in captured.err
    assert not captured.out
    assert not (tmp_path / "output").exists()
