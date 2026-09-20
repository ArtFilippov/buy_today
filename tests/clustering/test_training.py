from functools import partial

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from prak.clustering.models import ModelSnapshot
from prak.clustering.models.temporal import train_temporal
from prak.clustering.report import align_assignments, read_labels
from prak.clustering.training import train_clustering
from prak.schema import ROW_KEY


def test_training_roundtrip_and_keyed_alignment(working_frame, write_dataset, tmp_path):
    output = tmp_path / "models"
    paths = train_clustering(write_dataset(working_frame), output, strategy=partial(train_temporal, n_clusters=3))
    assert set(path.name for path in output.iterdir()) == {"model.joblib", "distance.joblib", "labels.csv"}
    assert paths.model_path.is_absolute()
    model, distance = joblib.load(paths.model_path), joblib.load(paths.distance_path)
    np.testing.assert_array_equal(model.labels_, [0] * 4 + [1] * 4 + [2] * 4)
    assert distance.pairwise(working_frame).shape == (12, 12)
    table = read_labels(paths.labels_path).sample(frac=1, random_state=42)
    table.to_csv(paths.labels_path, index=False)
    labels, mask = align_assignments(working_frame, read_labels(paths.labels_path), working_frame.iloc[-3:])
    np.testing.assert_array_equal(labels, model.labels_)
    np.testing.assert_array_equal(mask, [False] * 9 + [True] * 3)
    # Refit changes all temporal groups and overwrites the three artifacts.
    train_clustering(write_dataset(working_frame), output, strategy=partial(train_temporal, n_clusters=2))
    assert joblib.load(paths.model_path).n_clusters == 2
    assert read_labels(paths.labels_path).cluster_id.nunique() == 2


def test_coordinator_delegates_policy_and_can_save_accumulated_labels(working_frame, new_batch, write_dataset, tmp_path):
    accumulated = pd.concat([working_frame, new_batch], ignore_index=True)
    table = accumulated[list(ROW_KEY)].assign(cluster_id=np.arange(len(accumulated)) % 3)
    calls = []

    def retain_previous_state(batch):
        calls.append(batch)
        # Neither object has fit/predict; the coordinator only persists them.
        return ModelSnapshot(BaseEstimator(), BaseEstimator(), table)

    paths = train_clustering(write_dataset(new_batch), tmp_path / "model", strategy=retain_previous_state)
    pd.testing.assert_frame_equal(calls[0], new_batch)
    pd.testing.assert_frame_equal(read_labels(paths.labels_path), table)


@pytest.mark.parametrize(("problem", "message"), [
    ("missing", "missing=1"), ("extra", "extra=1"), ("duplicate", "Duplicate"),
    ("float", "integers"), ("null", "missing"), ("columns", "exactly"),
    ("outside_batch", "outside dataset"), ("duplicate_batch", "duplicate"),
    ("duplicate_dataset", "duplicate"),
])
def test_invalid_assignments_and_batch_keys(working_frame, new_batch, problem, message):
    table = working_frame[list(ROW_KEY)].assign(cluster_id=0)
    batch = working_frame.iloc[-2:].copy()
    if problem == "missing":
        table = table.iloc[1:]
    elif problem == "extra":
        table = pd.concat([table, new_batch.iloc[:1][list(ROW_KEY)].assign(cluster_id=0)])
    elif problem == "duplicate":
        table = pd.concat([table, table.iloc[:1]])
    elif problem == "float":
        table["cluster_id"] = 0.5
    elif problem == "null":
        table.loc[0, "order_id"] = pd.NA
    elif problem == "columns":
        table["extra"] = 0
    elif problem == "outside_batch":
        batch = new_batch
    elif problem == "duplicate_batch":
        batch = pd.concat([batch, batch])
    else:
        working_frame = pd.concat([working_frame, working_frame.iloc[:1]])
    with pytest.raises(ValueError, match=message):
        align_assignments(working_frame, table, batch)


def test_invalid_dataset_does_not_call_strategy_or_overwrite(working_frame, write_dataset, tmp_path):
    working_frame.loc[0, "price"] = 0

    def must_not_run(frame):
        raise AssertionError("Bad dataset must be checked first")

    with pytest.raises(ValueError, match="цена"):
        train_clustering(write_dataset(working_frame), tmp_path / "model", strategy=must_not_run)
    assert not (tmp_path / "model").exists()
