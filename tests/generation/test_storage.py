import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from buy_today.clustering.distances import TimestampDistance
from buy_today.generation import generate_dataset, generate_histories, read_history_dataset
from buy_today.schema import ROW_KEY


class AnchorOnlyDistance:
    def pairwise(self, X, Y):
        same_key = (Y.order_id == X.order_id.iloc[0]) & (Y.order_item_id == X.order_item_id.iloc[0])
        return np.where(same_key.to_numpy()[None, :], 0, 1e6)


class BrokenDistance:
    def pairwise(self, X, Y):
        raise RuntimeError("distance failed")


@pytest.fixture
def distance_path(tmp_path):
    path = tmp_path / "distance.joblib"
    joblib.dump(TimestampDistance(), path)
    return path


def contents(directory):
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_two_steps_roundtrip_preserve_old_histories_with_changed_distance(
    working_frame, new_batch, write_dataset, distance_path, tmp_path,
):
    working_frame["product_id"] = pd.array([f"product_{i:02d}" for i in range(12)], dtype="string")
    new_batch["product_id"] = pd.array([f"product_{i:02d}" for i in range(6, 18)], dtype="string")
    batch0, batch1 = write_dataset(working_frame, "zero.csv"), write_dataset(new_batch, "one.csv")
    original_inputs = [path.read_bytes() for path in (batch0, batch1, distance_path)]
    first = generate_dataset(batch0, distance_path, tmp_path / "step0", temperature=86400, n_users=3, split_sizes=(3, 2, 1))
    saved = read_history_dataset(first.output_dir)
    expected = generate_histories(
        working_frame, distance=TimestampDistance(), n_users=3, batch_index=0,
        temperature=86400, split_sizes=(3, 2, 1),
    )
    pd.testing.assert_frame_equal(saved.events, expected.events)
    pd.testing.assert_frame_equal(saved.anchors, expected.anchors)
    assert set(saved.catalog.product_id) == set(working_frame.product_id)
    assert all(path.is_absolute() for path in vars(first).values())
    old_files = contents(first.output_dir)
    assert set(old_files) == {
        "train.csv", "validation.csv", "test.csv", "catalog.csv", "generator/manifest.json",
        "generator/sources.csv", "generator/anchors.csv", "generator/distances/batch_000.joblib",
    }
    changed_distance = tmp_path / "changed.joblib"
    joblib.dump(AnchorOnlyDistance(), changed_distance)
    later = generate_dataset(
        batch1, changed_distance, tmp_path / "step1", previous_dir=first.output_dir,
        temperature=1, n_users=2,
    )
    accumulated = read_history_dataset(later.output_dir)
    assert contents(first.output_dir) == old_files
    assert [path.read_bytes() for path in (batch0, batch1, distance_path)] == original_inputs
    pd.testing.assert_frame_equal(accumulated.events[accumulated.events.batch_index == 0], saved.events)
    pd.testing.assert_frame_equal(accumulated.anchors.iloc[:3], saved.anchors)
    pd.testing.assert_frame_equal(accumulated.sources.iloc[:12], saved.sources)
    assert accumulated.events.event_id.is_unique
    assert len(accumulated.anchors) == 5
    assert len(accumulated.events) == 30
    assert set(accumulated.catalog.product_id) == set(working_frame.product_id) | set(new_batch.product_id)
    for user, events in accumulated.events[accumulated.events.batch_index == 1].groupby("user_id"):
        anchor = accumulated.anchors.set_index("user_id").loc[user]
        assert events.order_id.eq(anchor.order_id).all()
        assert events.order_item_id.eq(anchor.order_item_id).all()
        assert events.order_id.str.startswith("new_").all()
    manifest = accumulated.manifest
    assert manifest["format_version"] == 1
    assert manifest["split_sizes"] == [3, 2, 1]
    assert [record["n_users"] for record in manifest["batches"]] == [3, 2]
    for record, source, distance in zip(manifest["batches"], (batch0, batch1), (distance_path, changed_distance)):
        assert record["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert record["distance"]["sha256"] == hashlib.sha256(distance.read_bytes()).hexdigest()
        assert (later.output_dir / record["distance"]["snapshot"]).read_bytes() == distance.read_bytes()
        assert record["random_state"] == 42

    # Repeating the same two-step experiment preserves every serialized file.
    copy0 = generate_dataset(batch0, distance_path, tmp_path / "copy0", temperature=86400, n_users=3, split_sizes=(3, 2, 1))
    copy1 = generate_dataset(batch1, changed_distance, tmp_path / "copy1", previous_dir=copy0.output_dir, temperature=1, n_users=2)
    assert contents(copy0.output_dir) == old_files
    assert contents(copy1.output_dir) == contents(later.output_dir)


def test_default_user_counts_and_catalog_includes_unsampled_products(working_frame, new_batch, write_dataset, distance_path, tmp_path):
    batch0, batch1 = write_dataset(working_frame, "zero.csv"), write_dataset(new_batch, "one.csv")
    first = generate_dataset(batch0, distance_path, tmp_path / "step0", temperature=86400, split_sizes=(1, 1, 1))
    second = generate_dataset(batch1, distance_path, tmp_path / "step1", temperature=86400, previous_dir=first.output_dir)
    data = read_history_dataset(second.output_dir)
    assert data.anchors.groupby("batch_index").size().to_dict() == {0: 2000, 1: 250}
    assert data.events.groupby("split").size().to_dict() == {"train": 2250, "validation": 2250, "test": 2250}

    working_frame["product_id"] = pd.array([str(i) for i in range(12)], dtype="string")
    small = generate_dataset(write_dataset(working_frame), distance_path, tmp_path / "small", temperature=86400, n_users=1, split_sizes=(1, 1, 1))
    data = read_history_dataset(small.output_dir)
    assert set(data.catalog.product_id) == set(working_frame.product_id)
    assert set(data.catalog.product_id) - set(data.events.product_id)


def test_continuation_does_not_require_original_files(working_frame, new_batch, write_dataset, distance_path, tmp_path):
    batch = write_dataset(working_frame)
    first = generate_dataset(batch, distance_path, tmp_path / "step0", temperature=86400, n_users=2)
    batch.unlink()
    distance_path.unlink()
    snapshot_distance = first.output_dir / "generator/distances/batch_000.joblib"
    second = generate_dataset(
        write_dataset(new_batch), snapshot_distance, tmp_path / "step1",
        previous_dir=first.output_dir, temperature=86400, n_users=1,
    )
    assert len(read_history_dataset(second.output_dir).events) == 300


@pytest.mark.parametrize("partial", [False, True])
def test_repeat_or_overlap_even_unsampled_source_is_rejected(working_frame, write_dataset, distance_path, tmp_path, partial):
    first = generate_dataset(write_dataset(working_frame), distance_path, tmp_path / "step0", temperature=86400, n_users=1, split_sizes=(1, 1, 1))
    old = contents(first.output_dir)
    frame = working_frame
    if partial:
        events = read_history_dataset(first.output_dir).events
        sampled = pd.MultiIndex.from_frame(events[list(ROW_KEY)])
        keys = pd.MultiIndex.from_frame(working_frame[list(ROW_KEY)])
        frame = working_frame.loc[~keys.isin(sampled)].iloc[:1]
    with pytest.raises(ValueError, match="overlaps"):
        generate_dataset(write_dataset(frame, "renamed.csv"), distance_path, tmp_path / "step1", previous_dir=first.output_dir, temperature=86400)
    assert not (tmp_path / "step1").exists()
    assert contents(first.output_dir) == old


def test_cannot_change_splits_or_write_into_previous_snapshot(working_frame, new_batch, write_dataset, distance_path, tmp_path):
    first = generate_dataset(write_dataset(working_frame), distance_path, tmp_path / "step0", temperature=86400, n_users=1)
    old = contents(first.output_dir)
    for output, sizes, message in [
        (tmp_path / "step1", (60, 20, 20), "split_sizes"),
        (first.output_dir, None, "already exists"),
        (first.output_dir / "step1", None, "outside"),
    ]:
        with pytest.raises(ValueError, match=message):
            generate_dataset(write_dataset(new_batch), distance_path, output, previous_dir=first.output_dir, temperature=86400, split_sizes=sizes)
    assert contents(first.output_dir) == old


def test_corrupted_snapshot_fails_before_new_sampling(working_frame, new_batch, write_dataset, distance_path, tmp_path):
    first = generate_dataset(write_dataset(working_frame), distance_path, tmp_path / "step0", temperature=86400, n_users=1)
    first.train_path.write_text(first.train_path.read_text() + "corruption\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch: train.csv"):
        generate_dataset(write_dataset(new_batch), distance_path, tmp_path / "step1", previous_dir=first.output_dir, temperature=86400)
    assert not (tmp_path / "step1").exists()


@pytest.mark.parametrize("problem", ["split", "product", "source_key", "duplicate_event", "catalog", "anchor"])
def test_snapshot_semantics_checked_even_with_updated_checksums(working_frame, write_dataset, distance_path, tmp_path, problem):
    paths = generate_dataset(write_dataset(working_frame), distance_path, tmp_path / "step0", temperature=86400, n_users=2)
    name = {"catalog": "catalog.csv", "anchor": "generator/anchors.csv"}.get(problem, "train.csv")
    path = paths.output_dir / name
    table = pd.read_csv(path)
    if problem == "split":
        table.loc[0, "split"] = "test"
    elif problem == "product":
        table.loc[0, "product_id"] = "absent"
    elif problem == "source_key":
        table.loc[0, "order_id"] = "absent"
    elif problem == "duplicate_event":
        table.iloc[1] = table.iloc[0]
    elif problem == "catalog":
        table.loc[0, "product_id"] = "absent"
    else:
        table.loc[0, "source_position"] = 1000
    table.to_csv(path, index=False)
    manifest = json.loads(paths.manifest_path.read_text())
    manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    paths.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_history_dataset(paths.output_dir)


@pytest.mark.parametrize("failure", ["distance", "write"])
def test_failure_does_not_publish_next_snapshot(working_frame, new_batch, write_dataset, distance_path, tmp_path, monkeypatch, failure):
    first = generate_dataset(write_dataset(working_frame), distance_path, tmp_path / "step0", temperature=86400, n_users=1)
    old = contents(first.output_dir)
    batch1 = write_dataset(new_batch)
    if failure == "distance":
        joblib.dump(BrokenDistance(), distance_path)
    else:
        original = pd.DataFrame.to_csv
        calls = []

        def failed_write(self, path, **kwargs):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("write failed")
            return original(self, path, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_csv", failed_write)
    with pytest.raises((RuntimeError, OSError), match="failed"):
        generate_dataset(batch1, distance_path, tmp_path / "step1", previous_dir=first.output_dir, temperature=86400, n_users=1)
    assert not (tmp_path / "step1").exists()
    assert not list(tmp_path.glob(".histories-*"))
    assert contents(first.output_dir) == old
