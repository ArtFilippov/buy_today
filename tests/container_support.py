"""Shared tiny Olist bundle and assertions for the Windows Docker acceptance run.

Run with the tests image's Python, not pytest. Runtime commands remain separate
containers with their ordinary entrypoint. Only ``create`` unwraps the existing
pure fixture factories in conftest; pytest callers pass the injected factory.
All acceptance writes are below the newly created host root mounted at /acceptance.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import nbformat
import pandas as pd

from prak.bundled_data import OLIST_FILES
from prak.generation import read_history_dataset
from prak.pipeline import read_completed_steps


CATEGORY_COUNTS = {name: 4 for name in ("toys", "books", "games", "garden", "sports", "music")}
INIT_ARGS = [
    "init", "--batch-size", "12", "--min-category-count", "1", "--model", "svd",
    "--initial-users", "6", "--additional-users", "3", "--split-sizes", "4", "2", "2",
    "--svd-n-components", "2", "--svd-n-iter", "3", "--k", "2",
    "--temporal-n-clusters", "3", "--max-evaluation-rows", "8", "--random-state", "17",
    "--temperature", "3600", "--price-threshold", "0.23",
    "--category-threshold", "0.47", "--state-threshold", "0.89",
]
SMALL_PARAMETERS = {
    "model": "svd", "initial_users": 6, "additional_users": 3, "split_sizes": [4, 2, 2],
    "svd_n_components": 2, "svd_n_iter": 3, "k": 2, "temporal_n_clusters": 3,
    "max_evaluation_rows": 8, "random_state": 17, "temperature": 3600.0,
    "thresholds": {"price": 0.23, "category": 0.47, "state": 0.89},
}
WORKSPACE_NAME = "workspace with spaces"
RAW_NAME = "raw fixture"
KEEP_FILES = {
    "notes/keep.txt": b"unrelated workspace notes\x00\xff\r\n",
    "dataset/keep.txt": b"unrelated raw directory file\r\n",
    "logs/old.log": b"old log bytes, not necessarily UTF-8\x00\xff\r\n",
}
RESET_FILES = tuple(f"{name}/obsolete/nested.txt" for name in ("data", "run", "recommendations"))
SUMMARY_FILES = tuple(f"run/summary/summary.{suffix}" for suffix in ("html", "json", "csv"))


def small_raw_tables(raw_category_tables):
    """24 positions, two batches, three initial products and three new products."""
    tables = raw_category_tables(CATEGORY_COUNTS)
    order_ids = tables["olist_orders_dataset.csv"]["order_id"]
    # Preparation ignores these two tables; the container must still export them.
    tables["olist_order_payments_dataset.csv"] = pd.DataFrame({
        "order_id": order_ids, "payment_sequential": 1, "payment_type": "credit_card",
        "payment_installments": 1, "payment_value": 106.11,
    })
    tables["olist_order_reviews_dataset.csv"] = pd.DataFrame({
        "review_id": [f"review_{index:05d}" for index in range(len(order_ids))],
        "order_id": order_ids, "review_score": 5,
        "review_comment_title": "Good", "review_comment_message": "Delivered",
        "review_creation_date": "2018-01-06 00:00:00",
        "review_answer_timestamp": "2018-01-07 00:00:00",
    })
    assert len(tables) == 9 and set(tables) == set(OLIST_FILES)
    return tables


def write_bundle(directory, tables):
    directory.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        frame.to_csv(directory / name, index=False, encoding="utf-8")
    return directory


def seed_workspace(workspace):
    for relative, content in KEEP_FILES.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    seed_reset_targets(workspace)


def seed_reset_targets(workspace):
    for relative in RESET_FILES:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must be removed by init")
    for name in OLIST_FILES:
        (workspace / "dataset" / name).write_bytes(b"old raw CSV must be overwritten\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_hashes(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*")) if path.is_file()
    }


def assert_preserved(directory, expected):
    actual = file_hashes(directory)
    for name, digest in expected.items():
        assert actual.get(name) == digest, f"Changed or missing file: {directory / name}"


def check_exports(workspace, bundle):
    exported = {path.name for path in (workspace / "dataset").glob("*.csv")}
    assert exported == set(OLIST_FILES) and len(exported) == 9, exported
    for name in OLIST_FILES:
        assert (workspace / "dataset" / name).read_bytes() == (bundle / name).read_bytes(), name
    for name, content in KEEP_FILES.items():
        assert (workspace / name).read_bytes() == content, name


def check_reports(workspace, count):
    run = workspace / "run"
    assert read_json(run / "state.json")["next_batch_index"] == count
    assert read_json(workspace / "data/state.json") == {"next_batch_index": 0}
    assert read_json(run / "config.json")["parameters"] == SMALL_PARAMETERS
    assert [step["step_index"] for step in read_completed_steps(run)] == list(range(count))
    assert sorted(path.name for path in (run / "steps").iterdir()) == [
        f"step_{index:03d}" for index in range(count)
    ]
    prepared = read_json(workspace / "data/manifest.json")
    assert [batch["rows"] for batch in prepared["batches"]] == [12, 12]
    summary = read_json(run / "summary/summary.json")
    assert [step["step_index"] for step in summary["steps"]] == list(range(count))
    assert [step["data_quality"]["rows"] for step in summary["steps"]] == [12 * (i + 1) for i in range(count)]
    assert [step["ranking"]["test"]["n_users"] for step in summary["steps"]] == [6 + 3 * i for i in range(count)]
    csv = pd.read_csv(run / "summary/summary.csv")
    assert csv.step_index.tolist() == list(range(count))
    assert csv.reference_rows.tolist() == [12 * (i + 1) for i in range(count)]
    assert "<html" in (run / "summary/summary.html").read_text(encoding="utf-8").lower()
    for index in range(count):
        step = run / f"steps/step_{index:03d}"
        for report in ("eda", "clustering", *(("deda",) if index else ())):
            directory = step / report
            assert (directory / "report.html").stat().st_size > 0
            assert read_json(directory / "metrics.json")["format_version"] == 1
            notebook = nbformat.read(directory / "report.ipynb", as_version=4)
            nbformat.validate(notebook)
            cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
            assert cells and all(cell.execution_count is not None for cell in cells)
            assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
        training = read_json(step / "ranking/manifest.json")
        assert training["model"]["class"] == "prak.ranking.svd.SVDRanker"
        model = step / "ranking/model.joblib"
        assert hashlib.sha256(model.read_bytes()).hexdigest() == training["model"]["sha256"]
        read_history_dataset(step / "histories")
    if count == 2:
        first = read_history_dataset(run / "steps/step_000/histories")
        second = read_history_dataset(run / "steps/step_001/histories")
        pd.testing.assert_frame_equal(
            second.events.loc[second.events.batch_index.eq(0)].reset_index(drop=True), first.events,
        )


def check_log(workspace, command, stages, previous=()):
    candidates = [path for path in (workspace / "logs").glob(f"*_{command}.log")
                  if path.name not in previous]
    assert len(candidates) == 1, candidates
    path, = candidates
    text = path.read_text(encoding="utf-8")
    assert f"command={command}" in text
    assert "event=command_start" in text and "event=command_succeeded" in text
    assert "event=command_failed" not in text and "event=failed" not in text
    for name in stages:
        assert f"stage={name} event=start" in text, (path, name)
        assert f"stage={name} event=done elapsed_seconds=" in text, (path, name)
    return text


def create(root):
    if any(root.iterdir()):
        raise ValueError("Acceptance root must be empty; refusing to reuse existing data")
    # These fixtures are pure factories. Unwrapping here avoids duplicating seven
    # source tables; pytest tests use the injected raw_category_tables fixture.
    from conftest import raw_category_tables, raw_tables

    factory = raw_category_tables.__wrapped__(raw_tables.__wrapped__())
    write_bundle(root / RAW_NAME, small_raw_tables(factory))
    seed_workspace(root / WORKSPACE_NAME)
    write_json(root / "owned.json", {"purpose": "synthetic Docker acceptance", "format_version": 1})


def snapshot(root, workspace, name):
    value = {"workspace": file_hashes(workspace), "logs": file_hashes(workspace / "logs")}
    write_json(root / "snapshots" / f"{name}.json", value)
    if name == "exhausted":
        # An exhausted update must actually reconstruct the missing summary.
        for relative in SUMMARY_FILES:
            (workspace / relative).unlink()
    elif name == "reinit":
        seed_reset_targets(workspace)


def check(root, workspace, phase):
    count = 1 if phase in ("init", "reinit") else 2
    check_exports(workspace, root / RAW_NAME)
    check_reports(workspace, count)
    stages = ["pipeline.integrity", "summary"]
    training_stages = ["reference", "clustering_train", "clustering_evaluation", "generation",
                       "ranking_train", "validation", "test", "pipeline.commit"]
    if phase in ("init", "reinit"):
        for relative in RESET_FILES:
            assert not (workspace / relative).exists(), relative
        assert not list((workspace / "recommendations").rglob("*.csv"))
        previous = read_json(root / "snapshots/reinit.json") if phase == "reinit" else {"logs": {}}
        assert_preserved(workspace / "logs", previous["logs"])
        text = check_log(workspace, "init", stages + training_stages + [
            "reset", "export_olist", "preparation", "preparation.read", "pipeline.initialize",
            "notebook.execute", "notebook.render",
        ], previous["logs"])
        assert "event=batch_committed batch_index=0" in text
        assert not (workspace / "run/steps/step_001").exists()
    else:
        name = {"update": "first", "inference": "second", "exhausted": "exhausted"}[phase]
        previous = read_json(root / "snapshots" / f"{name}.json")
        assert_preserved(workspace / "logs", previous["logs"])
        if phase == "update":
            protected = {key: value for key, value in previous["workspace"].items()
                         if key.startswith(("run/steps/step_000/", "data/", "dataset/"))
                         or key == "run/config.json"}
            assert_preserved(workspace, protected)
            text = check_log(workspace, "update", stages + training_stages + ["reference.deda"], previous["logs"])
            assert "event=batch_committed batch_index=1" in text
            assert "stage=preparation" not in text and "stage=export_olist" not in text
        elif phase == "inference":
            assert_preserved(workspace, previous["workspace"])
            paths = list((workspace / "recommendations").glob("*.csv"))
            assert len(paths) == 1, paths
            model = joblib.load(workspace / "run/steps/step_001/ranking/model.joblib")
            user = "user_000001_000000"  # Exists only in the explicitly selected step_001.
            table = pd.read_csv(paths[0])
            assert list(table.columns) == ["user_id", "rank", "product_id"]
            assert table.user_id.tolist() == [user, user] and table["rank"].tolist() == [1, 2]
            assert table.product_id.tolist() == model.predict(user, 2).tolist()
            check_log(workspace, "inference", ["inference", "inference.load", "inference.predict", "inference.export"])
        else:
            # Every pre-existing byte, including reconstructed summaries, survives.
            assert_preserved(workspace, previous["workspace"])
            current = {key: value for key, value in file_hashes(workspace).items() if not key.startswith("logs/")}
            expected = {key: value for key, value in previous["workspace"].items() if not key.startswith("logs/")}
            assert current == expected
            text = check_log(workspace, "update", stages, previous["logs"])
            assert "event=stream_exhausted batch_index=2" in text
            assert "event=batch_start" not in text and "stage=ranking_train" not in text
    hashes = file_hashes(workspace)
    write_json(root / "checks" / f"{phase}.json", {
        "passed": True, "phase": phase, "next_batch_index": count,
        "logs": file_hashes(workspace / "logs"),
        "summaries": {name: hashes[name] for name in SUMMARY_FILES},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "snapshot", "check"))
    parser.add_argument("--root", type=Path, default=Path("/acceptance"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--name", choices=("first", "second", "exhausted", "reinit"))
    parser.add_argument("--phase", choices=("init", "update", "inference", "exhausted", "reinit"))
    args = parser.parse_args()
    if args.action == "create":
        create(args.root)
    else:
        assert read_json(args.root / "owned.json")["purpose"] == "synthetic Docker acceptance"
        if args.action == "snapshot":
            if not args.name:
                parser.error("snapshot requires --name")
            snapshot(args.root, args.workspace, args.name)
        else:
            if not args.phase:
                parser.error("check requires --phase")
            check(args.root, args.workspace, args.phase)
    print(f"PASS: {args.action} {args.phase or args.name or 'synthetic nine-table bundle'}", flush=True)


if __name__ == "__main__":
    main()
