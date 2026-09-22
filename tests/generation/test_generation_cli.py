from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd
import pytest

from buy_today.cli import main
from buy_today.clustering.distances import TimestampDistance
from buy_today.generation import read_history_dataset


def test_generate_two_steps_from_another_cwd(working_frame, new_batch, write_dataset, tmp_path):
    batch0 = write_dataset(working_frame, "first ' batch.csv")
    batch1 = write_dataset(new_batch, "second batch.csv")
    distance = tmp_path / 'distance " ready.joblib'
    joblib.dump(TimestampDistance(), distance)
    for index, batch in enumerate((batch0, batch1)):
        args = [
            str(Path(sys.executable).with_name("buy_today")), "generate",
            "--batch", batch.name, "--distance", distance.name,
            "--output-dir", f"step {index}", "--temperature", "86400", "--n-users", "2",
            "--random-state", "17",
        ]
        args += ["--split-sizes", "3", "2", "1"] if index == 0 else ["--previous-dir", "step 0"]
        result = subprocess.run(args, cwd=tmp_path, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert str(tmp_path / f"step {index}" / "generator/manifest.json") in result.stdout
    first = read_history_dataset(tmp_path / "step 0")
    second = read_history_dataset(tmp_path / "step 1")
    pd.testing.assert_frame_equal(second.events.iloc[:len(first.events)], first.events)
    assert len(second.events) == 24
    assert second.manifest["split_sizes"] == [3, 2, 1]
    assert [record["random_state"] for record in second.manifest["batches"]] == [17, 17]


@pytest.mark.parametrize("args", [
    [], ["--temperature", "0"], ["--temperature", "-1"], ["--temperature", "nan"],
    ["--temperature", "inf"], ["--temperature", "text"],
    ["--temperature", "1", "--n-users", "0"],
    ["--temperature", "1", "--n-users", "1.5"],
    ["--temperature", "1", "--split-sizes", "1", "0", "1"],
    ["--temperature", "1", "--split-sizes", "1", "1"],
    ["--temperature", "1", "--random-state", "-1"],
    ["--temperature", "1", "--n-clusters", "20"],
])
def test_invalid_generate_arguments_exit_two(args):
    with pytest.raises(SystemExit) as error:
        main(["generate", "--batch", "batch", "--distance", "distance", "--output-dir", "output", *args])
    assert error.value.code == 2


@pytest.mark.parametrize("missing", ["batch", "distance", "output-dir"])
def test_generate_requires_paths(missing):
    args = ["generate", "--temperature", "86400"]
    for name in ("batch", "distance", "output-dir"):
        if name != missing:
            args.extend([f"--{name}", "path"])
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_generation_data_error_exits_one(tmp_path, capsys):
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        main([
            "generate", "--batch", str(tmp_path / "absent.csv"),
            "--distance", "absent.joblib", "--output-dir", str(output), "--temperature", "86400",
        ])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "Ошибка generate" in captured.err
    assert not captured.out
    assert not output.exists()
