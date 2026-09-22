import json
from pathlib import Path
import subprocess
import sys

import pytest

from prak.cli import main


def run_cli(*args, cwd):
    return subprocess.run(
        [str(Path(sys.executable).with_name("prak")), *args],
        cwd=cwd, text=True, capture_output=True, check=False,
    )


@pytest.mark.parametrize("model,options,parameters", [
    ("random", [], {"random_state": 73}),
    ("svd", ["--svd-n-components", "1", "--svd-n-iter", "5"],
     {"random_state": 73, "n_components": 1, "n_iter": 5}),
])
def test_rank_and_both_evaluation_splits_from_another_cwd(
    ranking_snapshot, tmp_path, model, options, parameters,
):
    trained = run_cli(
        "rank", "--dataset-dir", ranking_snapshot.name, "--output-dir", "model dir",
        "--model", model, "--random-state", "73", *options, cwd=tmp_path,
    )
    assert trained.returncode == 0, trained.stderr
    model_dir = tmp_path / "model dir"
    assert str(model_dir / "model.joblib") in trained.stdout
    manifest = json.loads((model_dir / "manifest.json").read_text())
    assert manifest["model"]["parameters"] == parameters
    assert manifest["model"]["class"].endswith("SVDRanker" if model == "svd" else "RandomRanker")
    for split in ("validation", "test"):
        result = run_cli(
            "evaluate-ranking", "--dataset-dir", ranking_snapshot.name,
            "--model-dir", "model dir", "--output-dir", f"report {split}",
            "--split", split, cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        output = tmp_path / f"report {split}"
        assert str(output / "metrics.json") in result.stdout
        assert str(output / "per_user.csv") in result.stdout
        metrics = json.loads((output / "metrics.json").read_text())
        assert metrics["split"] == split and metrics["k"] == 10
    repeat = run_cli(
        "rank", "--dataset-dir", ranking_snapshot.name, "--output-dir", "model dir", cwd=tmp_path,
    )
    assert repeat.returncode == 1 and "already exists" in repeat.stderr


@pytest.mark.parametrize("command", ["rank", "evaluate-ranking"])
def test_help(command, tmp_path):
    result = run_cli(command, "--help", cwd=tmp_path)
    assert result.returncode == 0
    assert "--dataset-dir" in result.stdout


@pytest.mark.parametrize("args", [
    ["rank"],
    ["rank", "--dataset-dir", "missing", "--output-dir", "out", "--random-state", "-1"],
    ["rank", "--dataset-dir", "missing", "--output-dir", "out", "--model", "unknown"],
    ["rank", "--dataset-dir", "missing", "--output-dir", "out", "--model", "svd", "--svd-n-components", "0"],
    ["rank", "--dataset-dir", "missing", "--output-dir", "out", "--model", "svd", "--svd-n-iter", "1.5"],
    ["rank", "--dataset-dir", "missing", "--output-dir", "out", "--svd-n-components", "1"],
    ["rank", "--dataset-dir", "missing", "--output-dir", "out", "--svd-n-iter", "7"],
    ["evaluate-ranking", "--dataset-dir", "missing", "--model-dir", "model", "--output-dir", "out"],
    ["evaluate-ranking", "--dataset-dir", "missing", "--model-dir", "model", "--output-dir", "out", "--split", "train"],
    ["evaluate-ranking", "--dataset-dir", "missing", "--model-dir", "model", "--output-dir", "out", "--split", "test", "--k", "0"],
])
def test_invalid_arguments_exit_two(args, capsys):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert capsys.readouterr().err


@pytest.mark.parametrize("command", ["rank", "evaluate-ranking"])
def test_missing_inputs_exit_one(command, tmp_path):
    extra = [] if command == "rank" else ["--model-dir", "missing", "--split", "validation"]
    result = run_cli(
        command, "--dataset-dir", "missing", "--output-dir", "out", *extra, cwd=tmp_path,
    )
    assert result.returncode == 1
    assert f"Ошибка {command}:" in result.stderr
    assert not (tmp_path / "out").exists()


def test_k_larger_than_catalog_exits_one(ranking_snapshot, tmp_path, capsys):
    main(["rank", "--dataset-dir", str(ranking_snapshot), "--output-dir", str(tmp_path / "model")])
    with pytest.raises(SystemExit) as error:
        main([
            "evaluate-ranking", "--dataset-dir", str(ranking_snapshot),
            "--model-dir", str(tmp_path / "model"), "--output-dir", str(tmp_path / "report"),
            "--split", "test", "--k", "13",
        ])
    assert error.value.code == 1
    assert "k must" in capsys.readouterr().err
    assert not (tmp_path / "report").exists()


def test_svd_defaults_and_dimension_error(ranking_snapshot, tmp_path, capsys):
    output = tmp_path / "svd"
    args = ["rank", "--dataset-dir", str(ranking_snapshot), "--output-dir", str(output),
            "--model", "svd"]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 1
    assert "n_components" in capsys.readouterr().err
    assert not output.exists()
    main([*args, "--svd-n-components", "1"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model"]["parameters"] == {"n_components": 1, "n_iter": 7, "random_state": 42}
