from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import fixture_types as ft

from buy_today.cli import main
from buy_today.ranking import read_latest_benchmark_runs


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(Path(sys.executable).with_name("buy_today")), *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def test_benchmark_and_report_commands_from_another_cwd(ranking_snapshot: Path, tmp_path: Path):
    second = tmp_path / "metric b"
    shutil.copytree(ranking_snapshot, second)
    result = run_cli(
        "benchmark-ranking",
        "--output",
        "benchmark results",
        "--dataset",
        ranking_snapshot.name,
        "--dataset",
        second.name,
        "--model",
        "random",
        "--model",
        "svd --n-components 2 --random-state 73",
        "--k",
        "3",
        "--no-save-model",
        cwd=tmp_path,
    )
    assert not result.returncode, result.stderr
    selected = read_latest_benchmark_runs(tmp_path / "benchmark results")
    assert len(selected) == 2 and all(str(path) in result.stdout for path, _ in selected)
    assert selected[1][1]["model"]["parameters"] == {
        "n_components": 2,
        "n_iter": 7,
        "random_state": 73,
    }
    assert not list((tmp_path / "benchmark results").rglob("model.joblib"))
    shutil.rmtree(ranking_snapshot)
    shutil.rmtree(second)
    report = run_cli(
        "benchmark-ranking-report",
        "benchmark results",
        "--output",
        "reports/compare ranking.html",
        cwd=tmp_path,
    )
    assert not report.returncode, report.stderr
    for suffix in ("html", "csv"):
        path = tmp_path / "reports" / f"compare ranking.{suffix}"
        assert path.is_file() and str(path) in report.stdout


def test_cli_defaults_save_model_and_k(ranking_snapshot: Path, tmp_path: Path):
    result = run_cli(
        "benchmark-ranking",
        "--output",
        "out",
        "--dataset",
        ranking_snapshot.name,
        "--model",
        "random",
        cwd=tmp_path,
    )
    assert not result.returncode, result.stderr
    selected = read_latest_benchmark_runs(tmp_path / "out")
    assert selected[0][1]["k"] == 10 and selected[0][1]["save_model"]
    assert list(selected[0][0].rglob("model.joblib"))


@pytest.mark.parametrize(
    "extra",
    [
        ["--model", "unknown"],
        ["--model", "random --unknown 1"],
        ["--model", "svd --n-iter 0"],
        ["--model", "random", "--k", "0"],
        ["--model", "random", "--model", "random --random-state 42"],
        ["--model", "random", "--dataset", "other/missing"],
    ],
)
def test_bad_benchmark_syntax_exits_two(
    extra: list[str], tmp_path: Path, capsys: ft.CaptureFixture[str]
):
    with pytest.raises(SystemExit) as error:
        main(
            ["benchmark-ranking", "--output", str(tmp_path / "out"), "--dataset", "missing", *extra]
        )
    assert error.value.code == 2
    assert capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "command,args",
    [
        ("benchmark-ranking", ["--dataset", "missing", "--model", "random", "--output", "out"]),
        ("benchmark-ranking-report", ["missing", "--output", "report.html"]),
    ],
)
def test_execution_failures_exit_one(command: str, args: list[str], tmp_path: Path):
    result = run_cli(command, *args, cwd=tmp_path)
    assert result.returncode == 1
    assert f"Ошибка {command}:" in result.stderr
    assert not (tmp_path / "out").exists() and not (tmp_path / "report.html").exists()


@pytest.mark.parametrize("command", ["benchmark-ranking", "benchmark-ranking-report"])
def test_command_help(command: str, tmp_path: Path):
    result = run_cli(command, "--help", cwd=tmp_path)
    assert not result.returncode
    assert "--output" in result.stdout
