import hashlib
import json
from pathlib import Path

import fixture_types as ft
import joblib
import pandas as pd
import pytest

from buy_today.clustering.distances import TimestampDistance
from buy_today.generation import generate_dataset
from buy_today.history_quality import (
    HistoryQuality, evaluate_history_quality, report_history_quality,
)
from buy_today.schema import DATE_FORMAT, read_dataset


def contents(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*") if path.is_file()
    }


def test_report_uses_all_splits_and_source_keys_and_preserves_inputs(
    quality_snapshot: tuple[Path, Path], tmp_path: Path,
) -> None:
    dataset, reference = quality_snapshot
    before, reference_before = contents(dataset), reference.read_bytes()
    path = report_history_quality(dataset, reference, tmp_path / "report")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "report/metrics.json"
    assert list(path.parent.iterdir()) == [path]
    assert result["format_version"] == 1
    assert result["scope"] == "train+validation+test"
    assert result["quantile_method"] == "linear"
    assert result["n_events"] == 12
    assert result["n_users"] == 2
    assert result["n_reference_rows"] == 2
    assert len(result["metrics"]) == 20
    assert result["metrics"]["user_top_category_share_mean"] == pytest.approx(5 / 6)
    assert result["metrics"]["user_effective_category_count_mean"] == pytest.approx(36 / 26)
    assert result["metrics"]["anchor_category_event_share"] == pytest.approx(5 / 6)
    assert not result["metrics"]["category_frequency_tvd"]
    assert result["iid"]["repeats"] == 100
    assert result["iid"]["random_state"] == 42
    for name, checksum in result["inputs"]["dataset"]["files"].items():
        assert checksum == hashlib.sha256(before[name]).hexdigest()
    assert set(result["inputs"]["dataset"]["files"]) == set(before)
    assert result["inputs"]["reference"]["sha256"] == hashlib.sha256(reference_before).hexdigest()
    repeated = report_history_quality(dataset, reference, tmp_path / "repeat")
    assert path.read_bytes() == repeated.read_bytes()
    assert contents(dataset) == before
    assert reference.read_bytes() == reference_before


def test_cumulative_reference_weights_source_rows_not_batch_user_counts(
    working_frame: pd.DataFrame, new_batch: pd.DataFrame, write_dataset: ft.DatasetWriter,
    tmp_path: Path,
) -> None:
    first = working_frame.iloc[:2].assign(product_category_name="A")
    second = new_batch.iloc[:6].assign(product_category_name="B")
    batch0, batch1 = write_dataset(first, "zero.csv"), write_dataset(second, "one.csv")
    distance = tmp_path / "distance.joblib"
    joblib.dump(TimestampDistance(), distance)
    step0 = generate_dataset(
        batch0, distance, tmp_path / "step0", temperature=1, n_users=3, split_sizes=(1, 1, 1),
    )
    step1 = generate_dataset(
        batch1, distance, tmp_path / "step1", previous_dir=step0.output_dir,
        temperature=1, n_users=1,
    )
    reference = write_dataset(pd.concat([second, first], ignore_index=True), "combined.csv")
    # Only the supplied CSV is needed: original paths in the manifest may be gone.
    batch0.unlink()
    batch1.unlink()
    distance.unlink()
    path = report_history_quality(step1.output_dir, reference, tmp_path / "report")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["n_reference_rows"] == 8
    assert result["n_users"] == 4
    assert result["metrics"]["category_frequency_tvd"] == 0.5
    assert result["metrics"]["anchor_category_event_share"] == 1
    with pytest.raises(ValueError, match="extra=6"):
        report_history_quality(step0.output_dir, reference, tmp_path / "wrong-report")
    assert not (tmp_path / "wrong-report").exists()


@pytest.mark.parametrize(
    "problem", ["missing", "extra", "duplicate", "product", "time", "category", "schema"],
)
def test_reference_must_match_complete_source_set_and_content(
    quality_snapshot: tuple[Path, Path], tmp_path: Path, problem: str,
) -> None:
    dataset, reference = quality_snapshot
    frame = read_dataset(reference)
    if problem == "missing":
        frame = frame.iloc[:1]
    elif problem == "extra":
        frame = pd.concat([frame, frame.iloc[:1].assign(order_id="later")])
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "product":
        frame.loc[0, "product_id"] = "changed"
    elif problem == "time":
        frame["order_purchase_timestamp"] += pd.Timedelta(days=1)
    elif problem == "category":
        frame.loc[0, "product_category_name"] = " "
    else:
        frame = frame.drop(columns="product_category_name")
    frame.to_csv(reference, index=False, date_format=DATE_FORMAT)
    before = contents(dataset)
    with pytest.raises(ValueError):
        report_history_quality(dataset, reference, tmp_path / "report")
    assert not (tmp_path / "report").exists()
    assert contents(dataset) == before


def test_changed_reference_cannot_produce_report_with_stale_hash(
    quality_snapshot: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, reference = quality_snapshot

    def change_reference(
        events: pd.DataFrame, anchors: pd.DataFrame, working: pd.DataFrame,
        *, iid_repeats: int, random_state: int,
    ) -> HistoryQuality:
        result = evaluate_history_quality(
            events, anchors, working, iid_repeats=iid_repeats, random_state=random_state,
        )
        reference.write_bytes(reference.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(
        "buy_today.history_quality.report.evaluate_history_quality", change_reference,
    )
    with pytest.raises(ValueError, match="inputs changed"):
        report_history_quality(dataset, reference, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("destination", ["existing", "snapshot", "inside", "symlink"])
def test_destination_must_be_new_and_outside_snapshot(
    quality_snapshot: tuple[Path, Path], tmp_path: Path, destination: str,
) -> None:
    dataset, reference = quality_snapshot
    before = contents(dataset)
    output = tmp_path / "report"
    if destination == "existing":
        output.mkdir()
    elif destination == "snapshot":
        output = dataset
    elif destination == "inside":
        output = dataset / "nested/report"
    else:
        (tmp_path / "link").symlink_to(dataset, target_is_directory=True)
        output = tmp_path / "link/report"
    with pytest.raises(ValueError, match="already exists|outside"):
        report_history_quality(dataset, reference, output)
    assert contents(dataset) == before


def test_corrupt_snapshot_is_rejected(
    quality_snapshot: tuple[Path, Path], tmp_path: Path,
) -> None:
    dataset, reference = quality_snapshot
    (dataset / "test.csv").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch: test.csv"):
        report_history_quality(dataset, reference, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_write_failure_cleans_staging_and_never_publishes_partial_report(
    quality_snapshot: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, reference = quality_snapshot
    before = contents(dataset)

    def fail(*args: object, **kwargs: object) -> int:
        raise OSError("write failed")

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(OSError, match="write failed"):
        report_history_quality(dataset, reference, tmp_path / "report")
    assert not (tmp_path / "report").exists()
    assert not list(tmp_path.glob(".history-quality-*"))
    assert contents(dataset) == before
