"""Saved-model inference, integrity checks and non-destructive CSV exports."""

import csv
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from buy_today.ranking import RandomRanker, SVDRanker, train_ranker
from buy_today.ranking.inference import export_recommendations, recommend


def contents(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture(params=["random", "svd"])
def saved_model(ranking_snapshot, tmp_path, request):
    ranker = RandomRanker(random_state=17) if request.param == "random" else SVDRanker(n_components=1)
    model_dir = tmp_path / "saved model"
    paths = train_ranker(ranking_snapshot, model_dir, ranker=ranker)
    return model_dir, joblib.load(paths.model_path)


def test_recommend_needs_only_saved_model_and_preserves_inputs(
    saved_model, ranking_snapshot, tmp_path, monkeypatch,
):
    model_dir, model = saved_model
    # The recorded training paths no longer exist, but their contents remain
    # available for a byte-for-byte preservation check.
    history = ranking_snapshot.rename(tmp_path / "unavailable history")
    before_history, before_model = contents(history), contents(model_dir)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("Inference must never call fit")

    monkeypatch.setattr(type(model), "fit", forbidden_fit)
    for user in ("001", "NA"):
        for k in (1, np.int64(12), 5):
            expected = pd.DataFrame({
                "user_id": [user] * k,
                "rank": range(1, k + 1),
                "product_id": model.predict(user, k),
            })
            pd.testing.assert_frame_equal(recommend(model_dir, user, k), expected)
        default = recommend(str(model_dir), user)
        assert list(default.columns) == ["user_id", "rank", "product_id"]
        assert default.user_id.tolist() == [user] * 10
        assert default["rank"].tolist() == list(range(1, 11))
        assert default.product_id.tolist() == model.predict(user, 10).tolist()
    assert contents(history) == before_history
    assert contents(model_dir) == before_model


@pytest.mark.parametrize("user,k,message", [
    ("unknown", 1, "Unknown user_id"), ("1", 1, "Unknown user_id"),
    (None, 1, "Unknown user_id"), (1, 1, "Unknown user_id"),
    ("001", 0, "k must"), ("001", -1, "k must"), ("001", 13, "k must"),
    ("001", True, "k must"), ("001", np.bool_(True), "k must"),
    ("001", 1.5, "k must"), ("001", "3", "k must"), ("001", None, "k must"),
])
def test_predict_validation_propagates_without_output(saved_model, tmp_path, user, k, message):
    model_dir, _ = saved_model
    output = tmp_path / "new output" / "recommendations.csv"
    with pytest.raises(ValueError, match=message):
        recommend(model_dir, user, k)
    with pytest.raises(ValueError, match=message):
        export_recommendations(model_dir, user, output, k=k)
    assert not output.parent.exists()


@pytest.mark.parametrize("problem", ["checksum", "version"])
def test_integrity_is_checked_before_deserialization(saved_model, tmp_path, monkeypatch, problem):
    model_dir, _ = saved_model
    if problem == "checksum":
        with (model_dir / "model.joblib").open("ab") as file:
            file.write(b"tampered")
        message = "SHA-256 mismatch"
    else:
        manifest_path = model_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["format_version"] = 99
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        message = "format_version=1"
    before = contents(model_dir)

    def forbidden_load(*args, **kwargs):
        pytest.fail("Invalid artifacts must be rejected before joblib.load")

    monkeypatch.setattr("buy_today.ranking.inference.joblib.load", forbidden_load)
    with pytest.raises(ValueError, match=message):
        recommend(model_dir, "001")
    output = tmp_path / "recommendations.csv"
    with pytest.raises(ValueError, match=message):
        export_recommendations(model_dir, "001", output)
    assert not output.exists()
    assert contents(model_dir) == before


def test_export_writes_exact_ordered_columns_and_preserves_inputs(
    saved_model, ranking_snapshot, tmp_path,
):
    model_dir, model = saved_model
    before_history, before_model = contents(ranking_snapshot), contents(model_dir)
    output = tmp_path / "new output" / "recommendations.csv"
    result = export_recommendations(str(model_dir), "001", str(output))
    assert result == output.resolve()
    with result.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["user_id", "rank", "product_id"]
        assert list(reader) == [
            {"user_id": "001", "rank": str(rank), "product_id": product}
            for rank, product in enumerate(model.predict("001", 10), start=1)
        ]
    assert contents(ranking_snapshot) == before_history
    assert contents(model_dir) == before_model


def test_export_preserves_utf8_and_csv_quoting(ranking_snapshot, tmp_path):
    user, product = "пользователь", 'товар, "00"'
    for name in ("train.csv", "catalog.csv"):
        path = ranking_snapshot / name
        table = pd.read_csv(path, dtype="string", keep_default_na=False)
        table["product_id"] = table.product_id.replace({"p00": product})
        if name == "train.csv":
            table["user_id"] = table.user_id.replace({"NA": user})
        table.to_csv(path, index=False, encoding="utf-8")
    model_dir = tmp_path / "model"
    train_ranker(ranking_snapshot, model_dir)
    output = export_recommendations(model_dir, user, tmp_path / "output.csv", k=12)
    assert user.encode("utf-8") in output.read_bytes()
    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["user_id"] for row in rows] == [user] * 12
    assert [row["product_id"] for row in rows] == recommend(model_dir, user, 12).product_id.tolist()
    assert product in [row["product_id"] for row in rows]


@pytest.mark.parametrize("name", ["model.joblib", "manifest.json", "reports/metrics.json"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink", "directory_symlink"])
def test_export_protects_model_directory_files_and_aliases(saved_model, tmp_path, name, alias):
    model_dir, _ = saved_model
    reports = model_dir / "reports"
    reports.mkdir()
    (reports / "metrics.json").write_text('{"keep": true}', encoding="utf-8")
    target = model_dir / name
    if alias == "direct":
        output = target
    elif alias == "directory_symlink":
        directory = tmp_path / "model alias"
        directory.symlink_to(model_dir, target_is_directory=True)
        output = directory / name
    else:
        output = tmp_path / "output.csv"
        if alias == "symlink":
            output.symlink_to(target)
        else:
            output.hardlink_to(target)
    before = contents(model_dir)
    with pytest.raises(ValueError, match="model directory"):
        export_recommendations(model_dir, "001", output)
    assert contents(model_dir) == before


@pytest.mark.parametrize("name", [".", "recommendations.csv", "new/recommendations.csv"])
def test_export_keeps_output_outside_model_directory(saved_model, name):
    model_dir, _ = saved_model
    before = contents(model_dir)
    with pytest.raises(ValueError, match="outside the model directory"):
        export_recommendations(model_dir, "001", model_dir / name)
    assert contents(model_dir) == before


@pytest.mark.parametrize("stage", ["load", "predict", "write"])
def test_export_propagates_underlying_errors(saved_model, tmp_path, monkeypatch, stage):
    model_dir, model = saved_model
    error = OSError(f"{stage} failed")

    def fail(*args, **kwargs):
        raise error

    if stage == "load":
        monkeypatch.setattr("buy_today.ranking.inference.joblib.load", fail)
    elif stage == "predict":
        monkeypatch.setattr(type(model), "predict", fail)
    else:
        monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    output = tmp_path / "output.csv"
    with pytest.raises(OSError) as caught:
        export_recommendations(model_dir, "001", output)
    assert caught.value is error
    assert not output.exists()
