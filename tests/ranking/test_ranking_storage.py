"""Persisted ranking models and independent file-based evaluation."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from buy_today.clustering.distances import TimestampDistance
from buy_today.generation import generate_dataset
from buy_today.ranking import RandomRanker, SVDRanker, read_ranking_data, report_ranking, train_ranker


def contents(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("prototype", [
    RandomRanker(random_state=np.int64(17)),
    SVDRanker(n_components=1, random_state=np.int64(17)),
], ids=["random", "svd"])
def test_training_reads_only_train_and_catalog_and_records_provenance(
    ranking_snapshot, tmp_path, monkeypatch, prototype,
):
    (ranking_snapshot / "validation.csv").unlink()
    (ranking_snapshot / "test.csv").unlink()
    before = contents(ranking_snapshot)
    fitted = []
    original_fit = type(prototype).fit

    def record_fit(self, X, y=None):
        fitted.append(X)
        return original_fit(self, X, y)

    monkeypatch.setattr(type(prototype), "fit", record_fit)
    paths = train_ranker(ranking_snapshot, tmp_path / "model", ranker=prototype)
    assert not hasattr(prototype, "catalog_")
    assert len(fitted) == 1
    assert list(fitted[0].interactions.columns) == ["user_id", "product_id"]
    assert len(fitted[0].interactions) == 6
    assert fitted[0].interactions.product_id.tolist() == ["p00", "p00", "p01"] * 2
    model = joblib.load(paths.model_path)
    assert set(model.user_ids_) == {"001", "NA"}
    assert set(model.predict("001", 12)) == set(fitted[0].catalog.product_id)
    assert contents(ranking_snapshot) == before
    assert all(path.is_absolute() for path in vars(paths).values())
    assert set(contents(paths.model_path.parent)) == {"model.joblib", "manifest.json"}
    manifest = json.loads(paths.manifest_path.read_text())
    assert manifest["format_version"] == 1
    assert manifest["n_users"] == 2
    assert manifest["model"]["parameters"] == prototype.get_params()
    assert manifest["model"]["sha256"] == hashlib.sha256(paths.model_path.read_bytes()).hexdigest()
    for name, count in [("train", 6), ("catalog", 12)]:
        source = ranking_snapshot / f"{name}.csv"
        assert manifest["inputs"][name] == {
            "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "rows": count,
        }


@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize("ranker", [RandomRanker(), SVDRanker(n_components=1)], ids=["random", "svd"])
def test_saved_evaluation_needs_only_heldout_and_catalog_and_never_fits(
    ranking_snapshot, tmp_path, monkeypatch, split, ranker,
):
    model_dir = tmp_path / "model"
    train_ranker(ranking_snapshot, model_dir, ranker=ranker)
    (ranking_snapshot / "train.csv").unlink()
    (ranking_snapshot / ("test.csv" if split == "validation" else "validation.csv")).unlink()
    before_data, before_model = contents(ranking_snapshot), contents(model_dir)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("Evaluation must never call fit")

    monkeypatch.setattr(type(ranker), "fit", forbidden_fit)
    paths = report_ranking(ranking_snapshot, model_dir, model_dir / split, split=split, k=12)
    metrics = json.loads(paths.metrics_path.read_text())
    users = pd.read_csv(paths.per_user_path, dtype={"user_id": "string"}, keep_default_na=False)
    assert metrics["split"] == split
    assert metrics["k"] == 12
    assert metrics["n_users"] == 2 and metrics["n_events"] == 4 and metrics["n_catalog"] == 12
    assert metrics["metrics"]["recall_at_k"] == 1
    assert metrics["metrics"]["ndcg_at_k"] == pytest.approx(users.ndcg_at_k.mean())
    assert set(users.user_id) == {"001", "NA"}
    assert users.n_relevant.tolist() == ([1, 1] if split == "validation" else [2, 2])
    assert set(metrics["inputs"]) == {split, "catalog"}
    assert metrics["inputs"][split]["sha256"] == hashlib.sha256(before_data[f"{split}.csv"]).hexdigest()
    assert metrics["model"]["sha256"] == hashlib.sha256(before_model["model.joblib"]).hexdigest()
    assert contents(ranking_snapshot) == before_data
    for name, value in before_model.items():
        assert (model_dir / name).read_bytes() == value
    repeated = report_ranking(ranking_snapshot, model_dir, tmp_path / "again", split=split, k=12)
    assert repeated.metrics_path.read_bytes() == paths.metrics_path.read_bytes()
    assert repeated.per_user_path.read_bytes() == paths.per_user_path.read_bytes()


@pytest.mark.parametrize("problem", ["catalog", "users", "model", "version", "split", "k"])
def test_incompatible_evaluation_fails_without_report(ranking_snapshot, tmp_path, problem):
    model_dir, output = tmp_path / "model", tmp_path / "report"
    paths = train_ranker(ranking_snapshot, model_dir)
    split, k = "validation", 10
    if problem == "catalog":
        with (ranking_snapshot / "catalog.csv").open("a") as file:
            file.write("other\n")
        message = "catalog differs"
    elif problem == "users":
        path = ranking_snapshot / "validation.csv"
        table = pd.read_csv(path, keep_default_na=False)
        table = table[table.user_id == "001"]
        table.to_csv(path, index=False)
        message = "users differ"
    elif problem == "model":
        paths.model_path.write_bytes(b"not a model")
        message = "SHA-256"
    elif problem == "version":
        paths.manifest_path.write_text('{"format_version": 99}')
        message = "format_version"
    elif problem == "split":
        split, message = "train", "validation or test"
    else:
        k, message = 13, "k must"
    with pytest.raises(ValueError, match=message):
        report_ranking(ranking_snapshot, model_dir, output, split=split, k=k)
    assert not output.exists()


@pytest.mark.parametrize("operation", ["train", "evaluate"])
@pytest.mark.parametrize("location", ["existing", "inside_snapshot"])
def test_destinations_protect_prior_artifacts(ranking_snapshot, tmp_path, operation, location):
    model_dir = tmp_path / "model"
    train_ranker(ranking_snapshot, model_dir)
    before = contents(tmp_path)
    output = model_dir if location == "existing" else ranking_snapshot / "new"
    with pytest.raises(ValueError, match="already exists|outside"):
        if operation == "train":
            train_ranker(ranking_snapshot, output)
        else:
            report_ranking(ranking_snapshot, model_dir, output, split="validation")
    assert contents(tmp_path) == before


@pytest.mark.parametrize("operation", ["train", "evaluate"])
def test_write_failure_never_publishes_partial_artifacts(ranking_snapshot, tmp_path, monkeypatch, operation):
    model_dir, output = tmp_path / "model", tmp_path / "failed"
    train_ranker(ranking_snapshot, model_dir)
    before = contents(tmp_path)

    def failed_write(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr("buy_today.ranking.storage._write_json", failed_write)
    with pytest.raises(OSError, match="write failed"):
        if operation == "train":
            train_ranker(ranking_snapshot, output)
        else:
            report_ranking(ranking_snapshot, model_dir, output, split="validation")
    assert not output.exists()
    assert not list(tmp_path.glob(".ranking-*"))
    assert contents(tmp_path) == before


@pytest.mark.parametrize("ranker", [RandomRanker(), SVDRanker(n_components=2)], ids=["random", "svd"])
def test_generated_cumulative_snapshots_train_and_evaluate_independently(
    working_frame, new_batch, write_dataset, tmp_path, ranker,
):
    working_frame["product_id"] = pd.array([f"p{i}" for i in range(12)], dtype="string")
    new_batch["product_id"] = pd.array([f"p{i}" for i in range(6, 18)], dtype="string")
    distance = tmp_path / "distance.joblib"
    joblib.dump(TimestampDistance(), distance)
    first = generate_dataset(
        write_dataset(working_frame, "zero.csv"), distance, tmp_path / "step0",
        temperature=86400, n_users=3, split_sizes=(3, 2, 1),
    )
    second = generate_dataset(
        write_dataset(new_batch, "one.csv"), distance, tmp_path / "step1",
        previous_dir=first.output_dir, temperature=86400, n_users=2,
    )
    old_data = read_ranking_data(first.output_dir)
    accumulated = read_ranking_data(second.output_dir)
    pd.testing.assert_frame_equal(accumulated.interactions.iloc[:9], old_data.interactions)
    for index, snapshot, count, catalog_size in [(0, first, 3, 12), (1, second, 5, 18)]:
        paths = train_ranker(snapshot.output_dir, tmp_path / f"model{index}", ranker=ranker)
        assert len(joblib.load(paths.model_path).user_ids_) == count
        report = report_ranking(
            snapshot.output_dir, paths.model_path.parent, tmp_path / f"report{index}",
            split="validation", k=10,
        )
        metrics = json.loads(report.metrics_path.read_text())
        assert metrics["n_users"] == count
        assert metrics["n_catalog"] == catalog_size
