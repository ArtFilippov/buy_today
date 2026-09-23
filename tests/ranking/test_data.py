from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import fixture_types as ft

from buy_today.ranking.data import RankingData, check_ranking_data, read_ranking_data


@pytest.fixture
def ranking_data() -> RankingData:
    return RankingData(
        pd.DataFrame(
            {
                "user_id": ["007", "007", "NA", "007"],
                "product_id": ["001", "001", "NA", "002"],
            },
            dtype="string",
        ),
        pd.DataFrame({"product_id": ["001", "002", "NA", "unsampled"]}, dtype="string"),
    )


@pytest.fixture
def events(ranking_data: RankingData) -> pd.DataFrame:
    return ranking_data.interactions.assign(
        event_id=["0001", "0002", "NA", "0004"],
        event_index=[0, 1, 0, 2],
        split="train",
    )


def write_snapshot(
    directory: Path, events: pd.DataFrame, catalog: pd.DataFrame, split: str = "train"
) -> None:
    events.to_csv(directory / f"{split}.csv", index=False)
    catalog.to_csv(directory / "catalog.csv", index=False)


def test_ranking_data_retains_repeated_purchases_and_unobserved_candidates(
    ranking_data: RankingData,
):
    interactions = ranking_data.interactions.copy(deep=True)
    catalog = ranking_data.catalog.copy(deep=True)

    check_ranking_data(ranking_data)

    pd.testing.assert_frame_equal(ranking_data.interactions, interactions)
    pd.testing.assert_frame_equal(ranking_data.catalog, catalog)
    assert list(ranking_data.interactions.columns) == ["user_id", "product_id"]
    assert ranking_data.interactions.duplicated().sum() == 1
    assert ranking_data.catalog.product_id.is_unique
    assert "unsampled" in set(ranking_data.catalog.product_id)
    assert "unsampled" not in set(ranking_data.interactions.product_id)


@pytest.mark.parametrize(
    ("table", "problem", "message"),
    [
        ("interactions", "missing_column", "exactly"),
        ("interactions", "extra_column", "exactly"),
        ("catalog", "extra_column", "exactly"),
        ("interactions", "empty", "nonempty"),
        ("catalog", "empty", "nonempty"),
        ("catalog", "duplicate", "Duplicate"),
        ("catalog", "missing_product", "outside catalog"),
    ],
)
def test_ranking_data_rejects_invalid_tables(
    ranking_data: RankingData, table: str, problem: str, message: str
):
    frame = {"interactions": ranking_data.interactions, "catalog": ranking_data.catalog}[table]
    if problem == "missing_column":
        frame = frame.drop(columns="product_id")
    elif problem == "extra_column":
        frame = frame.assign(cluster_id=0)
    elif problem == "empty":
        frame = frame.iloc[:0]
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    else:
        frame = frame.iloc[1:]
    tables = {
        "interactions": ranking_data.interactions,
        "catalog": ranking_data.catalog,
        table: frame,
    }

    with pytest.raises(ValueError, match=message):
        check_ranking_data(RankingData(**tables))


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [
        ("interactions", "user_id", None),
        ("interactions", "user_id", "  "),
        ("interactions", "product_id", 1),
        ("catalog", "product_id", None),
        ("catalog", "product_id", ""),
        ("catalog", "product_id", 1),
    ],
)
def test_identifiers_must_be_present_nonempty_strings(
    ranking_data: RankingData, table: str, column: str, value: Any
):
    tables = {
        "interactions": ranking_data.interactions.astype(object),
        "catalog": ranking_data.catalog.astype(object),
    }
    tables[table].loc[0, column] = value

    with pytest.raises(ValueError, match="missing|string identifiers"):
        check_ranking_data(RankingData(**tables))


def test_plain_interactions_are_not_a_complete_ranking_input(ranking_data: RankingData):
    with pytest.raises(ValueError, match="RankingData"):
        check_ranking_data(ranking_data.interactions)


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_reader_opens_only_selected_split_and_catalog_and_preserves_ids(
    tmp_path: Path,
    monkeypatch: ft.MonkeyPatch,
    ranking_data: RankingData,
    events: pd.DataFrame,
    split: str,
):
    # Other splits and all generator state are absent. Provenance is not a model input.
    events = events.assign(
        split=split,
        order_purchase_timestamp="not a timestamp",
        cluster_id=99,
        anchor_order_id="private generator information",
    )
    write_snapshot(tmp_path, events, ranking_data.catalog, split)
    opened: list[Path] = []
    original_read_csv = pd.read_csv

    def record_read(path: Path | str, **kwargs: Any) -> pd.DataFrame:
        opened.append(Path(path))
        result = original_read_csv(path, iterator=False, chunksize=None, **kwargs)
        assert isinstance(result, pd.DataFrame)
        return result

    monkeypatch.setattr(pd, "read_csv", record_read)
    loaded = read_ranking_data(str(tmp_path), split=split)

    assert len(opened) == 2
    assert set(opened) == {tmp_path / f"{split}.csv", tmp_path / "catalog.csv"}
    assert isinstance(loaded, RankingData)
    pd.testing.assert_frame_equal(loaded.interactions, ranking_data.interactions)
    pd.testing.assert_frame_equal(loaded.catalog, ranking_data.catalog)


def test_reader_rejects_unknown_split_before_opening_files(tmp_path: Path):
    with pytest.raises(ValueError, match="split"):
        read_ranking_data(tmp_path, split="holdout")


@pytest.mark.parametrize(
    ("column", "row", "value", "message"),
    [
        ("split", 0, "test", "split"),
        ("event_id", 1, "0001", "Duplicate"),
        ("event_index", 1, 0, "Duplicate"),
        ("event_index", 0, -1, "nonnegative"),
        ("event_index", 0, "0.5", None),
        ("event_index", 0, "", None),
        ("event_id", 0, "", "identifiers"),
        ("user_id", 0, "  ", "identifiers"),
    ],
)
def test_reader_rejects_invalid_events(
    tmp_path: Path,
    ranking_data: RankingData,
    events: pd.DataFrame,
    column: str,
    row: int,
    value: Any,
    message: str | None,
):
    events = events.astype(object)
    events.loc[row, column] = value
    write_snapshot(tmp_path, events, ranking_data.catalog)

    with pytest.raises(ValueError, match=message):
        read_ranking_data(tmp_path)


@pytest.mark.parametrize(
    ("products", "message"),
    [
        (["001", "002", "NA", "001"], "Duplicate"),
        (["001", "002", "unsampled"], "outside catalog"),
    ],
)
def test_reader_requires_unique_catalog_covering_every_purchase(
    tmp_path: Path, events: pd.DataFrame, products: list[str], message: str
):
    write_snapshot(tmp_path, events, pd.DataFrame({"product_id": products}))

    with pytest.raises(ValueError, match=message):
        read_ranking_data(tmp_path)
