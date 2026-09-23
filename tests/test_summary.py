"""Summary integration tests using real completed-step verification, without training."""

import csv
from dataclasses import asdict
from html.parser import HTMLParser
import json
from pathlib import Path
from typing import Any, IO, override
from urllib.parse import unquote

import fixture_types as ft
import pytest
from summary_fixtures import (
    digest,
    drift_report,
    prepared_stream,
    save_reports,
    step_reports,
    write_json,
)

from buy_today.pipeline import PipelineConfig, read_completed_steps
from buy_today.summary import SummaryPaths, report_summary


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def completed_run(tmp_path: Path, *, model: str = "random", count: int = 2) -> Path:
    """Build the same small JSON contracts emitted by pipeline and report writers.

    Args:
        tmp_path (Path): Parent directory of the run and prepared source.
        model (str, default="random"): Ranker identity to record.
        count (int, default=2): Number of completed steps to create.

    Returns:
        Path: Directory containing the completed run.
    """
    run = tmp_path / "запуск ' & <demo>"
    batches, stream = prepared_stream(tmp_path / 'stream " & <source>')
    parameters = asdict(
        PipelineConfig(
            model=model,
            initial_users=2,
            additional_users=1,
            split_sizes=(3, 2, 1),
            k=2,
            svd_n_components=2,
        )
    )
    parameters["split_sizes"] = list(parameters["split_sizes"])
    config_hash = write_json(
        run / "config.json",
        {
            "format_version": 1,
            "parameters": parameters,
            "batches": batches,
            "stream": stream,
            "thread_limit": 1,
        },
    )
    state: dict[str, Any] = {
        "format_version": 1,
        "config_sha256": config_hash,
        "next_batch_index": count,
        "reference_sha256": None,
        "steps": [],
    }
    previous_reference: dict[str, Any] = {}
    for index in range(count):
        prefix = f"steps/step_{index:03d}"
        batch = batches[index]
        reference_path = run / "reference.csv"
        reference_path.write_text("product_id\n" + "p0\n" * (12 * (index + 1)), encoding="utf-8")
        reference: dict[str, Any] = {
            "path": "reference.csv",
            "sha256": digest(reference_path.read_bytes()),
            "rows": 12 * (index + 1),
        }
        reports = step_reports(run, prefix, index, model, batch, reference)
        artifacts = {
            "eda": f"{prefix}/eda/report.html",
            "clustering": f"{prefix}/clustering/report.html",
            "histories": f"{prefix}/histories",
            "ranking": f"{prefix}/ranking",
        }
        if index:
            reports["deda_metrics"] = (
                "deda/metrics.json",
                drift_report(previous_reference, reference_path, batch, parameters),
            )
            artifacts["deda"] = f"{prefix}/deda/report.html"
        files = save_reports(run, prefix, reports, artifacts)
        step = {
            "format_version": 1,
            "step_index": index,
            "model": model,
            "config_sha256": config_hash,
            "batch": batch,
            "reference": reference,
            "artifacts": artifacts,
            "files": files,
            "durations_seconds": {
                "reference": 1.5 + index,
                "clustering_train": 0.2,
                "clustering_evaluation": 2.0,
                "generation": 0.3,
                "ranking_train": 0.4,
                "validation": 0.5,
                "test": 0.6,
                "total": 6.0 + index,
            },
        }
        manifest = f"{prefix}/manifest.json"
        state["steps"].append({"path": manifest, "sha256": write_json(run / manifest, step)})
        state["reference_sha256"] = reference["sha256"]
        previous_reference = reference
    write_json(run / "state.json", state)
    assert len(read_completed_steps(run)) == count
    return run


def resave_step(run: Path, step: dict[str, Any]) -> None:
    """Re-sign deliberate metadata changes to exercise semantic checks.

    Args:
        run (Path): Existing run directory.
        step (dict[str, Any]): Altered step manifest to save and sign.
    """
    step["files"] = {name: digest((run / name).read_bytes()) for name in step["files"]}
    state = read_json(run / "state.json")
    saved = state["steps"][step["step_index"]]
    saved["sha256"] = write_json(run / saved["path"], step)
    write_json(run / "state.json", state)


def test_two_step_aggregation_and_repeatable_outputs(tmp_path: Path) -> None:
    run = completed_run(tmp_path)
    paths = report_summary(run)
    assert isinstance(paths, SummaryPaths)
    assert all(path.is_absolute() for path in vars(paths).values())
    assert {path.name for path in paths.json_path.parent.iterdir()} == {
        "summary.json",
        "summary.csv",
        "summary.html",
    }
    summary = read_json(paths.json_path)
    assert summary["format_version"] == 1 and summary["model"] == "random"
    first, second = summary["steps"]
    assert first["drift"] is None and "deda" not in first["links"]
    assert first["clustering"]["silhouette"] is None
    assert second["drift"]["drift_detected"] is True
    assert second["clustering"]["silhouette"] == 0.45
    with paths.csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert [row["step_index"] for row in rows] == ["0", "1"]
    assert [row["reference_rows"] for row in rows] == ["12", "24"]
    assert [row["n_users"] for row in rows] == ["2", "3"]
    assert [row["n_catalog"] for row in rows] == ["4", "5"]
    assert rows[0]["drift_detected"] == rows[0]["price_ks"] == rows[0]["silhouette"] == ""
    assert rows[0]["silhouette_reason"] == first["clustering"]["silhouette_reason"]
    assert rows[1]["price_ks"] == "0.2" and rows[1]["category_drift"] == "False"
    assert rows[1]["state_tvd"] == "0.15"
    assert rows[1]["validation_recall"] == "0.6" and rows[1]["test_recall"] == "0.35"
    assert rows[0]["validation_ndcg"] == "0.6" and rows[0]["test_ndcg"] == "0.3"
    assert rows[1]["validation_n_events"] == "6" and rows[1]["test_n_events"] == "3"
    assert rows[1]["reference_seconds"] == "2.5" and rows[1]["total_seconds"] == "7.0"
    assert rows[0]["ranking_train_seconds"] == "0.4"
    assert rows[0]["validation_seconds"] == "0.5" and rows[0]["test_seconds"] == "0.6"
    assert rows[0]["quality_checks"] == "7" and rows[0]["quality_checks_failed"] == "0"
    before = {path: path.read_bytes() for path in vars(paths).values()}
    assert report_summary(run) == paths
    assert {path: path.read_bytes() for path in vars(paths).values()} == before


def test_parameters_provenance_relative_links_and_html_escaping(
    tmp_path: Path, monkeypatch: ft.MonkeyPatch
) -> None:
    run = completed_run(tmp_path, model="svd")
    step = read_completed_steps(run)[0]
    step["artifacts"]["clustering"] = 'steps/step_000/clustering/a " & <report>.html'
    resave_step(run, step)
    monkeypatch.chdir(tmp_path)
    paths = report_summary(run.name)
    summary = read_json(paths.json_path)
    config = read_json(run / "config.json")
    assert summary["model"] == "svd" and summary["parameters"] == config["parameters"]
    assert summary["stream"] == config["stream"]
    result = summary["steps"][0]
    assert result["batch"] == step["batch"] and result["reference"] == step["reference"]
    assert result["config_sha256"] == step["config_sha256"]
    assert result["data_quality"] == read_json(run / step["artifacts"]["eda_metrics"])
    for name in ("validation", "test"):
        assert result["ranking"][name] == read_json(run / step["artifacts"][name])
    assert result["ranking"]["training"] == read_json(run / step["artifacts"]["ranking_manifest"])

    class Page(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.tags: list[str] = []
            self.links: list[str] = []

        @override
        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            self.tags.append(tag)
            if tag == "a":
                href = dict(attrs)["href"]
                assert href is not None
                self.links.append(href)

    html = paths.html_path.read_text(encoding="utf-8")
    page = Page()
    page.feed(html)
    assert "script" not in page.tags and "img" not in page.tags
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html
    assert "&lt;source&gt;" in html and "&amp;" in html
    assert all(link.startswith("../steps/") for link in page.links)
    assert unquote(result["links"]["clustering"]) == "../" + step["artifacts"]["clustering"]
    assert result["links"]["clustering"] in page.links
    assert "Validation Recall@K" in html and "Test Recall@K" in html
    assert "Время выполнения, секунды" in html and "Гиперпараметры" in html
    assert "n_components" in html and "Пропуски" in html


def test_no_completed_steps_has_useful_error_and_no_output(tmp_path: Path) -> None:
    run = completed_run(tmp_path, count=0)
    with pytest.raises(ValueError, match="No completed steps.*complete a pipeline step"):
        report_summary(run)
    assert not (run / "summary").exists()


def test_tampered_report_fails_verification_and_preserves_summary(tmp_path: Path) -> None:
    run = completed_run(tmp_path)
    paths = report_summary(run)
    before = {path: path.read_bytes() for path in vars(paths).values()}
    report = run / "steps/step_000/ranking/test/metrics.json"
    report.write_text(report.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Step report SHA-256 mismatch"):
        report_summary(run)
    assert {path: path.read_bytes() for path in vars(paths).values()} == before


def test_partial_step_not_in_state_is_ignored(tmp_path: Path) -> None:
    run = completed_run(tmp_path, count=1)
    partial = run / "steps/step_001"
    partial.mkdir()
    (partial / "manifest.json").write_text("unfinished and invalid JSON", encoding="utf-8")
    (run / "reference.csv").write_text("partially updated reference", encoding="utf-8")
    summary = read_json(report_summary(run).json_path)
    assert [step["batch"]["id"] for step in summary["steps"]] == ["batch_000"]
    assert (partial / "manifest.json").read_text() == "unfinished and invalid JSON"


def test_summary_needs_only_completed_json_after_sources_are_removed(
    tmp_path: Path, monkeypatch: ft.MonkeyPatch
) -> None:
    run = completed_run(tmp_path)
    config = read_json(run / "config.json")
    for source in (config["stream"], *config["batches"]):
        Path(source["path"]).unlink()
    (run / "reference.csv").unlink()
    steps = read_completed_steps(run)
    allowed = {run / "config.json", run / "state.json"}
    for step in steps:
        allowed.add(run / f"steps/step_{step['step_index']:03d}/manifest.json")
        allowed.update(run / name for name in step["files"])
    original_open = Path.open

    def only_metadata(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        if "r" in mode:
            assert path in allowed, f"Summary reopened an input: {path}"
        return original_open(path, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", only_metadata)
    paths = report_summary(run)
    assert paths.html_path.is_file() and paths.csv_path.is_file() and paths.json_path.is_file()


@pytest.mark.parametrize(
    "problem,message",
    [
        ("k", "k differs"),
        ("users", "user count differs"),
        ("catalog", "catalog identity mismatch"),
        ("model", "trained model identity mismatch"),
        ("split", "input identity or event count mismatch"),
    ],
)
def test_incoherent_evaluations_are_rejected_even_with_valid_digests(
    tmp_path: Path, problem: str, message: str
) -> None:
    run = completed_run(tmp_path, count=1)
    step = read_completed_steps(run)[0]
    path = run / step["artifacts"]["test"]
    report = read_json(path)
    if problem == "k":
        report["k"] = 3
    elif problem == "users":
        report["n_users"] = 1
    elif problem == "catalog":
        report["inputs"]["catalog"]["sha256"] = "f" * 64
    elif problem == "model":
        report["model"]["sha256"] = "f" * 64
    else:
        report["inputs"]["test"] = dict(
            read_json(run / step["artifacts"]["validation"])["inputs"]["validation"]
        )
        report["n_events"] = report["inputs"]["test"]["rows"]
    write_json(path, report)
    resave_step(run, step)
    assert len(read_completed_steps(run)) == 1
    with pytest.raises(ValueError, match=message):
        report_summary(run)
    assert not (run / "summary").exists()


def test_summary_output_aliases_cannot_overwrite_inputs_in_partial_run(tmp_path: Path) -> None:
    run = completed_run(tmp_path, count=1)
    partial = run / "steps/step_001"
    partial.mkdir()
    target = partial / "summary.json"
    target.write_text("unfinished input", encoding="utf-8")
    output = run / "summary"
    output.symlink_to(partial, target_is_directory=True)
    with pytest.raises(ValueError, match="output directory.*symlink"):
        report_summary(run)
    assert target.read_text() == "unfinished input"
    assert list(partial.iterdir()) == [target]
    output.unlink()
    output.mkdir()
    alias = output / "summary.html"
    alias.symlink_to(run / "config.json")
    config_before = (run / "config.json").read_bytes()
    with pytest.raises(ValueError, match="symlink or file alias"):
        report_summary(run)
    assert list(output.iterdir()) == [alias]
    alias.unlink()
    alias.hardlink_to(run / "config.json")
    with pytest.raises(ValueError, match="symlink or file alias"):
        report_summary(run)
    assert (run / "config.json").read_bytes() == config_before
    assert list(output.iterdir()) == [alias]
