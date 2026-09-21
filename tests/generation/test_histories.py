import numpy as np
import pandas as pd
import pytest

from prak.clustering.distances import TimestampDistance
from prak.generation.histories import GeneratedHistories, generate_histories
from prak.schema import ROW_KEY


PROVENANCE_COLUMNS = [
    "product_id", "order_id", "order_item_id", "order_purchase_timestamp",
]


class ReadyDistance:
    """A ready, key-addressed distance with no clustering or fitting contract."""

    def __init__(self, batch, values):
        self.keys = pd.MultiIndex.from_frame(batch[list(ROW_KEY)])
        self.values = np.asarray(values)
        self.calls = []

    def pairwise(self, X, Y):
        self.calls.append((X.copy(deep=True), Y.copy(deep=True)))
        left = self.keys.get_indexer(pd.MultiIndex.from_frame(X[list(ROW_KEY)]))
        right = self.keys.get_indexer(pd.MultiIndex.from_frame(Y[list(ROW_KEY)]))
        return self.values[np.ix_(left, right)]

    def fit(self, *args, **kwargs):
        raise AssertionError("Generation must use the ready distance without fitting")

    def fit_predict(self, *args, **kwargs):
        raise AssertionError("Generation must not perform clustering")

    @property
    def labels_(self):
        raise AssertionError("Generation must not consult cluster labels")

    @property
    def n_clusters(self):
        raise AssertionError("Generation must not depend on the number of clusters")


@pytest.fixture
def sampling_batch(working_frame):
    batch = working_frame.iloc[:4].copy()
    batch["product_id"] = pd.array(["shared", "other", "shared", "third"], dtype="string")
    batch["order_purchase_timestamp"] = pd.to_datetime([
        "2018-01-01", "2018-01-02", "2018-01-04", "2018-01-07",
    ]).as_unit("ns")
    batch.index = pd.Index([17, 17, -5, 99], name="source_row")
    return batch


def reference_draws(probabilities, *, n_users, batch_index, random_state=42, split_sizes=(70, 15, 15)):
    """The specified RNG protocol, with independently supplied probabilities."""
    rng = np.random.default_rng(np.random.SeedSequence([random_state, batch_index]))
    anchors = rng.integers(len(probabilities), size=n_users)
    histories = np.array([
        rng.choice(len(probabilities), size=sum(split_sizes), replace=True, p=probabilities[anchor])
        for anchor in anchors
    ])
    return anchors, histories


def assert_draws(result, batch, anchor_positions, history_positions):
    np.testing.assert_array_equal(result.anchors["source_position"], anchor_positions)
    pd.testing.assert_frame_equal(
        result.anchors[list(ROW_KEY)],
        batch.iloc[anchor_positions][list(ROW_KEY)].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        result.events[PROVENANCE_COLUMNS],
        batch.iloc[history_positions.ravel()][PROVENANCE_COLUMNS].reset_index(drop=True),
    )
    np.testing.assert_array_equal(
        result.events["user_id"], np.repeat(result.anchors["user_id"], history_positions.shape[1]),
    )


def test_default_schema_weighted_sampling_and_ready_rectangular_distance(sampling_batch):
    # exp(-log(2) * k) gives exact, independently known relative weights 2**-k.
    distance = ReadyDistance(sampling_batch, np.log(2) * np.array([
        [0, 1, 2, 3], [1, 0, 2, 4], [2, 2, 0, 1], [3, 4, 1, 0],
    ]))
    weights = np.array([
        [1, 1 / 2, 1 / 4, 1 / 8], [1 / 2, 1, 1 / 4, 1 / 16],
        [1 / 4, 1 / 4, 1, 1 / 2], [1 / 8, 1 / 16, 1 / 2, 1],
    ])
    anchors, histories = reference_draws(
        weights / weights.sum(axis=1, keepdims=True), n_users=6, batch_index=7,
    )
    before = sampling_batch.copy(deep=True)
    result = generate_histories(sampling_batch, distance=distance, batch_index=7, n_users=6, temperature=1)

    assert isinstance(result, GeneratedHistories)
    event_schema = {
        "event_id": "string", "user_id": "string", "event_index": "int64",
        "split": "string", "batch_index": "int64", "product_id": "string",
        "order_id": "string", "order_item_id": "int64",
        "order_purchase_timestamp": "datetime64[ns]",
    }
    anchor_schema = {
        "user_id": "string", "batch_index": "int64", "source_position": "int64",
        "order_id": "string", "order_item_id": "int64",
    }
    for frame, schema in [(result.events, event_schema), (result.anchors, anchor_schema)]:
        assert list(frame.columns) == list(schema)
        assert frame.dtypes.astype(str).to_dict() == schema
        assert frame["batch_index"].eq(7).all()
        assert not frame.isna().any().any()
    assert_draws(result, sampling_batch, anchors, histories)
    assert result.anchors["user_id"].is_unique
    assert result.events["event_id"].is_unique
    assert set(result.anchors["user_id"]).isdisjoint(sampling_batch["customer_unique_id"])
    np.testing.assert_array_equal(result.events["event_index"], np.tile(np.arange(100), 6))
    assert result.events["split"].tolist() == (["train"] * 70 + ["validation"] * 15 + ["test"] * 15) * 6
    # Sampling order is synthetic time, even when the next source purchase is older.
    assert (result.events.groupby("user_id")["order_purchase_timestamp"].diff() < pd.Timedelta(0)).any()
    assert result.events.duplicated(["user_id", *ROW_KEY]).any()
    assert len(distance.calls) == 6
    for (anchor, candidates), position in zip(distance.calls, anchors):
        pd.testing.assert_frame_equal(anchor, before.iloc[[position]])
        pd.testing.assert_frame_equal(candidates, before)
    pd.testing.assert_frame_equal(sampling_batch, before)


@pytest.mark.parametrize("use_timestamp_distance", [False, True], ids=["custom", "timestamp"])
@pytest.mark.parametrize("temperature", [86400, 172800])
def test_temperature_is_in_seconds_and_custom_splits_follow_draw_order(
    sampling_batch, use_timestamp_distance, temperature,
):
    days = np.array([
        [0, 1, 3, 6], [1, 0, 2, 5], [3, 2, 0, 3], [6, 5, 3, 0],
    ])
    distance = TimestampDistance() if use_timestamp_distance else ReadyDistance(sampling_batch, days * 86400)
    weights = np.exp(-days / (temperature / 86400))
    args = dict(n_users=4, batch_index=2, random_state=13, split_sizes=(7, 3, 2))
    anchors, histories = reference_draws(weights / weights.sum(axis=1, keepdims=True), **args)
    result = generate_histories(sampling_batch, distance=distance, temperature=temperature, **args)

    assert_draws(result, sampling_batch, anchors, histories)
    np.testing.assert_array_equal(result.events["event_index"], np.tile(np.arange(12), 4))
    assert result.events["split"].tolist() == (["train"] * 7 + ["validation"] * 3 + ["test"] * 2) * 4


@pytest.mark.parametrize("n_rows", [1, 3])
def test_zero_distances_sample_rows_with_replacement_and_keep_repeated_products(sampling_batch, n_rows):
    batch = sampling_batch.iloc[:n_rows].copy()
    args = dict(n_users=6, batch_index=0, split_sizes=(4, 2, 2))
    anchors, histories = reference_draws(np.full((n_rows, n_rows), 1 / n_rows), **args)
    result = generate_histories(
        batch, distance=ReadyDistance(batch, np.zeros((n_rows, n_rows))), temperature=1, **args,
    )

    assert_draws(result, batch, anchors, histories)
    assert len(result.anchors) == 6
    assert result.anchors["user_id"].is_unique
    assert result.anchors["source_position"].duplicated().any()
    assert len(result.events) == result.events["event_id"].nunique() == 48
    assert result.events.duplicated(["user_id", *ROW_KEY]).any()
    shared = result.events.loc[result.events["product_id"].eq("shared")]
    assert len(shared[list(ROW_KEY)].drop_duplicates()) == (1 if n_rows == 1 else 2)


@pytest.mark.parametrize("change", [{"random_state": 43}, {"batch_index": 1}], ids=["seed", "batch"])
def test_reproducibility_and_independent_seed_sequence_streams(sampling_batch, change):
    distance = ReadyDistance(sampling_batch, np.zeros((4, 4)))
    args = dict(n_users=7, batch_index=0, random_state=42, split_sizes=(4, 2, 1))
    first = generate_histories(sampling_batch, distance=distance, temperature=1, **args)
    changed_args = args | change
    changed = generate_histories(sampling_batch, distance=distance, temperature=1, **changed_args)
    repeated = generate_histories(sampling_batch, distance=distance, temperature=1, **args)

    pd.testing.assert_frame_equal(first.events, repeated.events)
    pd.testing.assert_frame_equal(first.anchors, repeated.anchors)
    for result, parameters in [(first, args), (changed, changed_args)]:
        anchors, histories = reference_draws(np.full((4, 4), 0.25), **parameters)
        assert_draws(result, sampling_batch, anchors, histories)
    # Compare the sampled sources, not IDs that necessarily include the batch number.
    assert not first.events[list(ROW_KEY)].equals(changed.events[list(ROW_KEY)])
    assert not first.anchors["source_position"].equals(changed.anchors["source_position"])


def test_next_batch_has_new_ids_and_only_its_own_sources(working_frame, new_batch):
    args = dict(distance=TimestampDistance(), temperature=86400, split_sizes=(3, 1, 1))
    first = generate_histories(working_frame, batch_index=0, n_users=3, **args)
    previous_events, previous_anchors = first.events.copy(deep=True), first.anchors.copy(deep=True)
    later = generate_histories(new_batch, batch_index=1, n_users=2, **args)

    assert set(first.anchors["user_id"]).isdisjoint(later.anchors["user_id"])
    assert set(first.events["event_id"]).isdisjoint(later.events["event_id"])
    assert len(later.events) == 10
    assert len(later.anchors) == 2
    for result, source, batch_index in [(first, working_frame, 0), (later, new_batch, 1)]:
        source_keys = set(source[list(ROW_KEY)].itertuples(index=False, name=None))
        for frame in (result.events, result.anchors):
            assert set(frame[list(ROW_KEY)].itertuples(index=False, name=None)) <= source_keys
            assert frame["batch_index"].eq(batch_index).all()
    pd.testing.assert_frame_equal(first.events, previous_events)
    pd.testing.assert_frame_equal(first.anchors, previous_anchors)


@pytest.mark.parametrize(("distances", "temperature", "probabilities"), [
    pytest.param([1e308] * 4, np.finfo(float).tiny, [0.25] * 4, id="all-large-equal"),
    pytest.param([1e308, 5e307, 5e307, 1e308], np.finfo(float).tiny, [0, 0.5, 0.5, 0], id="large-tied-minima"),
    pytest.param([0, 1, 2, 3], np.nextafter(0.0, 1.0), [1, 0, 0, 0], id="subnormal-temperature"),
    pytest.param([0, 1, 2, 3], np.finfo(float).max, [0.25] * 4, id="huge-temperature"),
    pytest.param(
        [1e308, 0, 1e308, 0], 1e308,
        np.array([np.exp(-1), 1, np.exp(-1), 1]) / (2 + 2 * np.exp(-1)),
        id="large-distance-and-temperature",
    ),
])
def test_extreme_temperatures_have_stable_normalized_sampling(sampling_batch, distances, temperature, probabilities):
    args = dict(n_users=3, batch_index=0, split_sizes=(4, 2, 2))
    anchors, histories = reference_draws(np.tile(probabilities, (4, 1)), **args)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = generate_histories(
            sampling_batch, distance=ReadyDistance(sampling_batch, np.tile(distances, (4, 1))),
            temperature=temperature, **args,
        )
    assert_draws(result, sampling_batch, anchors, histories)


@pytest.mark.parametrize("random_state", [0, 2**32 - 1])
def test_numpy_integer_parameters_and_list_split_sizes_are_accepted(sampling_batch, random_state):
    args = dict(
        n_users=np.int64(3), batch_index=np.int64(2), random_state=np.int64(random_state),
        split_sizes=[np.int64(2), np.int64(1), np.int64(1)],
    )
    anchors, histories = reference_draws(np.full((4, 4), 0.25), **args)
    result = generate_histories(
        sampling_batch, distance=ReadyDistance(sampling_batch, np.zeros((4, 4))),
        temperature=np.float64(1), **args,
    )
    assert_draws(result, sampling_batch, anchors, histories)


@pytest.mark.parametrize(("parameter", "value"), [
    ("n_users", 0), ("n_users", -1), ("n_users", 1.5), ("n_users", "2"),
    ("n_users", True), ("n_users", np.bool_(True)), ("n_users", None),
    ("batch_index", -1), ("batch_index", 1.5), ("batch_index", "0"),
    ("batch_index", True), ("batch_index", np.bool_(False)), ("batch_index", None),
    ("random_state", -1), ("random_state", 2**32), ("random_state", 42.0),
    ("random_state", True), ("random_state", np.bool_(True)), ("random_state", None),
    ("temperature", 0), ("temperature", -1), ("temperature", np.nan),
    ("temperature", np.inf), ("temperature", -np.inf), ("temperature", "1"),
    ("temperature", True), ("temperature", np.bool_(True)), ("temperature", None),
    ("temperature", 1 + 0j),
    ("split_sizes", None), ("split_sizes", "1,1,1"), ("split_sizes", np.array([1, 1, 1])),
    ("split_sizes", ()), ("split_sizes", (1, 1)), ("split_sizes", (1, 1, 1, 1)),
    ("split_sizes", (0, 1, 1)), ("split_sizes", (1, 0, 1)), ("split_sizes", (1, 1, 0)),
    ("split_sizes", (-1, 1, 1)), ("split_sizes", (1, 1.5, 1)), ("split_sizes", (1, 1, "1")),
    ("split_sizes", (True, 1, 1)), ("split_sizes", (1, np.bool_(True), 1)),
])
def test_invalid_parameters_fail_before_pairwise(sampling_batch, parameter, value):
    distance = ReadyDistance(sampling_batch, np.zeros((4, 4)))
    args = dict(n_users=2, batch_index=0, temperature=1, random_state=42, split_sizes=(2, 1, 1))
    args[parameter] = value
    before = sampling_batch.copy(deep=True)
    with pytest.raises(ValueError, match="split_sizes|size" if parameter == "split_sizes" else parameter):
        generate_histories(sampling_batch, distance=distance, **args)
    assert distance.calls == []
    pd.testing.assert_frame_equal(sampling_batch, before)


@pytest.mark.parametrize("matrix", [
    pytest.param(0, id="scalar"),
    pytest.param([0, 1, 2, 3], id="vector"),
    pytest.param([[0], [1], [2], [3]], id="column"),
    pytest.param(np.zeros((4, 4)), id="square"),
    pytest.param(np.zeros((1, 3)), id="too-short"),
    pytest.param(np.zeros((1, 5)), id="too-long"),
    pytest.param(np.zeros((0, 4)), id="empty"),
    pytest.param(np.zeros((1, 1, 4)), id="three-dimensional"),
    pytest.param([[0j, 1j, 2j, 3j]], id="complex"),
    pytest.param([["0", "1", "2", "3"]], id="strings"),
    pytest.param([[True, False, True, False]], id="boolean"),
    pytest.param(np.array([[0, 1, 2, 3]], dtype=object), id="object"),
    pytest.param([[0, -1, 2, 3]], id="negative"),
    pytest.param([[0, np.nan, 2, 3]], id="nan"),
    pytest.param([[0, np.inf, 2, 3]], id="infinite"),
    pytest.param([[0, -np.inf, 2, 3]], id="negative-infinite"),
])
def test_malformed_distances_fail_on_first_anchor(sampling_batch, matrix):
    class MalformedDistance:
        def __init__(self):
            self.calls = 0

        def pairwise(self, X, Y):
            self.calls += 1
            assert X.shape == (1, sampling_batch.shape[1])
            pd.testing.assert_frame_equal(Y, sampling_batch)
            return matrix

    distance = MalformedDistance()
    before = sampling_batch.copy(deep=True)
    with pytest.raises(ValueError, match="pairwise|Distances"):
        generate_histories(
            sampling_batch, distance=distance, batch_index=0, n_users=3,
            temperature=1, split_sizes=(2, 1, 1),
        )
    assert distance.calls == 1
    pd.testing.assert_frame_equal(sampling_batch, before)


@pytest.mark.parametrize(("problem", "message"), [
    ("missing_column", "Схема"), ("extra_column", "Схема"), ("column_order", "Схема"),
    ("item_dtype", "Типы и даты"), ("text_timestamp", "Типы и даты"),
    ("empty", "Непустой датасет"), ("missing_product", "Пропуски"),
    ("missing_timestamp", "Пропуски"), ("missing_customer", "Пропуски"),
    ("duplicate_key", "Повторы ключа"), ("unsorted", "Хронологический порядок"),
])
def test_invalid_olist_data_fail_before_pairwise(sampling_batch, problem, message):
    distance = ReadyDistance(sampling_batch, np.zeros((4, 4)))
    batch = sampling_batch.copy()
    if problem == "missing_column":
        batch = batch.drop(columns="customer_city")
    elif problem == "extra_column":
        batch["extra"] = 1
    elif problem == "column_order":
        batch = batch[batch.columns[::-1]]
    elif problem == "item_dtype":
        batch["order_item_id"] = batch["order_item_id"].astype("float64")
    elif problem == "text_timestamp":
        batch["order_purchase_timestamp"] = batch["order_purchase_timestamp"].astype("string")
    elif problem == "empty":
        batch = batch.iloc[:0]
    elif problem == "missing_product":
        batch.loc[-5, "product_id"] = pd.NA
    elif problem == "missing_timestamp":
        batch.loc[-5, "order_purchase_timestamp"] = pd.NaT
    elif problem == "missing_customer":
        batch.loc[-5, "customer_city"] = pd.NA
    elif problem == "duplicate_key":
        batch.iloc[1, batch.columns.get_loc("order_item_id")] = 1
    else:
        batch = batch.iloc[::-1]
    before = batch.copy(deep=True)
    with pytest.raises(ValueError, match=message):
        generate_histories(batch, distance=distance, batch_index=0, n_users=2, temperature=1)
    assert distance.calls == []
    pd.testing.assert_frame_equal(batch, before)


@pytest.mark.parametrize("price", [0, -1, np.inf, -np.inf, np.nan])
def test_invalid_olist_prices_fail_before_pairwise(sampling_batch, price):
    sampling_batch.loc[-5, "price"] = price
    distance = ReadyDistance(sampling_batch, np.zeros((4, 4)))
    with pytest.raises(ValueError, match="Пропуски|Конечная положительная цена"):
        generate_histories(sampling_batch, distance=distance, batch_index=0, n_users=2, temperature=1)
    assert distance.calls == []
