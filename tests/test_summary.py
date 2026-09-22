"""Summary integration tests using real completed-step verification, without training."""

import csv
from dataclasses import asdict
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote

import pytest

from prak.pipeline import PipelineConfig, read_completed_steps
from prak.summary import SummaryPaths, report_summary


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return digest(path.read_bytes())


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def completed_run(tmp_path, *, model="random", count=2):
    """The same small JSON contracts emitted by pipeline and report writers."""
    run = tmp_path / "запуск ' & <demo>"
    source = tmp_path / 'stream " & <source>'
    source.mkdir()
    batches = []
    for index in range(2):
        path = source / f"batch_{index:03d}.csv"
        path.write_text("product_id\n" + f"p{index}\n" * 12, encoding="utf-8")
        batches.append({"id": path.stem, "path": str(path), "rows": 12, "sha256": digest(path.read_bytes())})
    stream_path = source / "manifest.json"
    stream_hash = write_json(stream_path, {"format_version": 1, "batches": batches})
    parameters = asdict(PipelineConfig(
        model=model, initial_users=2, additional_users=1, split_sizes=(3, 2, 1),
        k=2, svd_n_components=2,
    ))
    parameters["split_sizes"] = list(parameters["split_sizes"])
    config_hash = write_json(run / "config.json", {
        "format_version": 1, "parameters": parameters, "batches": batches,
        "stream": {"path": str(stream_path), "sha256": stream_hash}, "thread_limit": 1,
    })
    state = {"format_version": 1, "config_sha256": config_hash,
             "next_batch_index": count, "reference_sha256": None, "steps": []}
    previous_reference = None
    for index in range(count):
        prefix = f"steps/step_{index:03d}"
        batch, users, catalog_rows = batches[index], 2 + index, 4 + index
        reference_path = run / "reference.csv"
        reference_path.write_text("product_id\n" + "p0\n" * (12 * (index + 1)), encoding="utf-8")
        reference = {"path": "reference.csv", "sha256": digest(reference_path.read_bytes()),
                     "rows": 12 * (index + 1)}
        reference_input = {"path": str(reference_path), "sha256": reference["sha256"]}
        histories = run / prefix / "histories"

        def identity(name, rows):
            return {"path": str(histories / f"{name}.csv"), "rows": rows,
                    "sha256": digest(f"{index}:{name}:{rows}".encode())}

        training = {
            "format_version": 1, "inputs": {"train": identity("train", users * 3),
                                             "catalog": identity("catalog", catalog_rows)},
            "n_users": users, "user_ids_sha256": digest(json.dumps([f"u{i}" for i in range(users)]).encode()),
            "versions": {"numpy": "2.1.0", "scikit_learn": "1.6.0"},
            "model": {"file": "model.joblib", "sha256": digest(f"model:{index}".encode()),
                      "class": f"prak.ranking.{model}.{'RandomRanker' if model == 'random' else 'SVDRanker'}",
                      "parameters": {"random_state": 42}},
        }
        if model == "svd":
            training["model"]["parameters"].update(n_components=2, n_iter=7)
        eda = {
            "format_version": 1, "kind": "eda", "input": reference_input,
            "rows": reference["rows"], "n_columns": 35, "n_orders": 6 * (index + 1),
            "n_customers": 3 + index, "n_products": catalog_rows,
            "period_start": "2018-01-01 00:00:00", "period_end": f"2018-01-0{index + 2} 00:00:00",
            "checks": [
                {"Проверка": name, "Результат": "OK", "Фактически": actual, "Ожидается": expected}
                for name, actual, expected in (
                    ("Схема и порядок колонок", "35 колонок, порядок совпадает", "35 колонок в порядке schema.COLUMNS"),
                    ("Типы и даты", "35 типов совпадают", "типы schema.DTYPES"),
                    ("Непустой датасет", str(reference["rows"]), "число строк > 0"),
                    ("Пропуски", "0", "число пропущенных значений = 0"),
                    ("Повторы ключа (order_id, order_item_id)", "0", "число повторений сверх первого = 0"),
                    ("Хронологический порядок покупок", "0", "число переходов назад во времени = 0"),
                    ("Конечная положительная цена", "0", "число цен, нарушающих 0 < price < inf, = 0"),
                )
            ],
        }
        clustering = {
            "format_version": 1, "kind": "clustering",
            "inputs": {"dataset": reference_input,
                       "new_batch": {key: batch[key] for key in ("path", "sha256")},
                       "labels": {"path": str(run / prefix / "clustering/labels.csv"),
                                  "sha256": digest(f"labels:{index}".encode())}},
            "silhouette": None if index == 0 else 0.45,
            "silhouette_reason": 'Один кластер <script>alert("x")</script>' if index == 0 else None,
            "sample_rows": reference["rows"], "new_count": 12, "previous_count": index * 12,
            "max_evaluation_rows": 1000, "random_state": 42,
            "model": {"class": "TemporalClusterer", "parameters": {"n_clusters": 1 + index}},
            "distance": {"class": "TimestampDistance", "parameters": {}},
        }
        reports = {"eda_metrics": ("eda/metrics.json", eda),
                   "clustering_metrics": ("clustering/metrics.json", clustering),
                   "ranking_manifest": ("ranking/manifest.json", training)}
        for split, events_per_user, recall, ndcg in (("validation", 2, 0.5, 0.6), ("test", 1, 0.25, 0.3)):
            reports[split] = (f"ranking/{split}/metrics.json", {
                "format_version": 1, "split": split, "k": 2, "n_users": users,
                "n_catalog": catalog_rows, "n_events": users * events_per_user,
                "metrics": {"recall_at_k": recall + index * 0.1, "ndcg_at_k": ndcg + index * 0.1},
                "inputs": {split: identity(split, users * events_per_user),
                           "catalog": training["inputs"]["catalog"]},
                "model": {"path": str(run / prefix / "ranking/model.joblib"),
                          "sha256": training["model"]["sha256"]},
            })
        artifacts = {"eda": f"{prefix}/eda/report.html", "clustering": f"{prefix}/clustering/report.html",
                     "histories": f"{prefix}/histories", "ranking": f"{prefix}/ranking"}
        if index:
            reports["deda_metrics"] = ("deda/metrics.json", {
                "format_version": 1, "kind": "deda", "drift_detected": True,
                "inputs": {"reference": {**previous_reference, "path": str(reference_path)},
                           "batch": {key: batch[key] for key in ("path", "sha256", "rows")}},
                "thresholds": parameters["thresholds"],
                "metrics": [
                    {"feature": feature, "measure": measure, "value": value, "threshold": 0.1, "drift": value >= 0.1}
                    for feature, measure, value in (("price", "KS D", 0.2),
                                                    ("product_category_name", "TVD", 0.05),
                                                    ("customer_state", "TVD", 0.15))
                ],
            })
            artifacts["deda"] = f"{prefix}/deda/report.html"
        files = {}
        for name, (relative, report) in reports.items():
            artifacts[name] = f"{prefix}/{relative}"
            files[artifacts[name]] = write_json(run / artifacts[name], report)
        step = {
            "format_version": 1, "step_index": index, "model": model,
            "config_sha256": config_hash, "batch": batch, "reference": reference,
            "artifacts": artifacts, "files": files,
            "durations_seconds": {"reference": 1.5 + index, "clustering_train": 0.2,
                                  "clustering_evaluation": 2.0, "generation": 0.3,
                                  "ranking_train": 0.4, "validation": 0.5, "test": 0.6,
                                  "total": 6.0 + index},
        }
        manifest = f"{prefix}/manifest.json"
        state["steps"].append({"path": manifest, "sha256": write_json(run / manifest, step)})
        state["reference_sha256"] = reference["sha256"]
        previous_reference = reference
    write_json(run / "state.json", state)
    assert len(read_completed_steps(run)) == count
    return run


def resave_step(run, step):
    """Re-sign deliberate metadata changes to exercise semantic, not hash, checks."""
    step["files"] = {name: digest((run / name).read_bytes()) for name in step["files"]}
    state = read_json(run / "state.json")
    saved = state["steps"][step["step_index"]]
    saved["sha256"] = write_json(run / saved["path"], step)
    write_json(run / "state.json", state)


def test_two_step_aggregation_and_repeatable_outputs(tmp_path):
    run = completed_run(tmp_path)
    paths = report_summary(run)
    assert isinstance(paths, SummaryPaths)
    assert all(path.is_absolute() for path in vars(paths).values())
    assert {path.name for path in paths.json_path.parent.iterdir()} == {"summary.json", "summary.csv", "summary.html"}
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


def test_parameters_provenance_relative_links_and_html_escaping(tmp_path, monkeypatch):
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
        def __init__(self):
            super().__init__()
            self.tags, self.links = [], []

        def handle_starttag(self, tag, attrs):
            self.tags.append(tag)
            if tag == "a":
                self.links.append(dict(attrs)["href"])

    html = paths.html_path.read_text(encoding="utf-8")
    page = Page()
    page.feed(html)
    assert "script" not in page.tags and "img" not in page.tags
    assert '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;' in html
    assert "&lt;source&gt;" in html and "&amp;" in html
    assert all(link.startswith("../steps/") for link in page.links)
    assert unquote(result["links"]["clustering"]) == "../" + step["artifacts"]["clustering"]
    assert result["links"]["clustering"] in page.links
    assert "Validation Recall@K" in html and "Test Recall@K" in html
    assert "Время выполнения, секунды" in html and "Гиперпараметры" in html
    assert "n_components" in html and "Пропуски" in html


def test_no_completed_steps_has_useful_error_and_no_output(tmp_path):
    run = completed_run(tmp_path, count=0)
    with pytest.raises(ValueError, match="No completed steps.*complete a pipeline step"):
        report_summary(run)
    assert not (run / "summary").exists()


def test_tampered_report_fails_verification_and_preserves_summary(tmp_path):
    run = completed_run(tmp_path)
    paths = report_summary(run)
    before = {path: path.read_bytes() for path in vars(paths).values()}
    report = run / "steps/step_000/ranking/test/metrics.json"
    report.write_text(report.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Step report SHA-256 mismatch"):
        report_summary(run)
    assert {path: path.read_bytes() for path in vars(paths).values()} == before


def test_partial_step_not_in_state_is_ignored(tmp_path):
    run = completed_run(tmp_path, count=1)
    partial = run / "steps/step_001"
    partial.mkdir()
    (partial / "manifest.json").write_text("unfinished and invalid JSON", encoding="utf-8")
    (run / "reference.csv").write_text("partially updated reference", encoding="utf-8")
    summary = read_json(report_summary(run).json_path)
    assert [step["batch"]["id"] for step in summary["steps"]] == ["batch_000"]
    assert (partial / "manifest.json").read_text() == "unfinished and invalid JSON"


def test_summary_needs_only_completed_json_after_sources_are_removed(tmp_path, monkeypatch):
    run = completed_run(tmp_path)
    config = read_json(run / "config.json")
    for source in [config["stream"], *config["batches"]]:
        Path(source["path"]).unlink()
    (run / "reference.csv").unlink()
    steps = read_completed_steps(run)
    allowed = {run / "config.json", run / "state.json"}
    for step in steps:
        allowed.add(run / f"steps/step_{step['step_index']:03d}/manifest.json")
        allowed.update(run / name for name in step["files"])
    original_open = Path.open

    def only_metadata(path, mode="r", *args, **kwargs):
        if "r" in mode:
            assert path in allowed, f"Summary reopened an input: {path}"
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", only_metadata)
    paths = report_summary(run)
    assert paths.html_path.is_file() and paths.csv_path.is_file() and paths.json_path.is_file()


@pytest.mark.parametrize("problem,message", [
    ("k", "k differs"), ("users", "user count differs"),
    ("catalog", "catalog identity mismatch"), ("model", "trained model identity mismatch"),
    ("split", "input identity or event count mismatch"),
])
def test_incoherent_evaluations_are_rejected_even_with_valid_digests(tmp_path, problem, message):
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
        report["inputs"]["test"] = dict(read_json(run / step["artifacts"]["validation"])["inputs"]["validation"])
        report["n_events"] = report["inputs"]["test"]["rows"]
    write_json(path, report)
    resave_step(run, step)
    assert len(read_completed_steps(run)) == 1
    with pytest.raises(ValueError, match=message):
        report_summary(run)
    assert not (run / "summary").exists()


def test_summary_output_aliases_cannot_overwrite_inputs_in_partial_run(tmp_path):
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
