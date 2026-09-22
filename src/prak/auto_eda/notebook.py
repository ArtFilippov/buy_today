"""Execute report notebooks in the caller's Python and export HTML and metrics."""

from dataclasses import dataclass
from html import escape
import json
import os
from pathlib import Path
import sys

from nbclient import NotebookClient
from nbconvert import HTMLExporter
import nbformat
import numpy as np

from prak.progress import stage


@dataclass(frozen=True)
class ReportPaths:
    notebook_path: Path
    html_path: Path

    @property
    def metrics_path(self) -> Path:
        return self.notebook_path.with_name("metrics.json")


def write_metrics(metrics: dict, path: Path | str = "metrics.json") -> None:
    """Serialize computed values from a notebook's final cell, without NaN/Inf."""
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__} as report metrics")

    content = json.dumps(
        metrics, ensure_ascii=False, allow_nan=False, indent=2, default=convert,
    )
    Path(path).write_text(content + "\n", encoding="utf-8")


def execute_report(
    cells: list[nbformat.NotebookNode], output_dir: Path | str, *, title: str,
    thread_limit: int | None = None,
) -> ReportPaths:
    """Write source, execute in a fresh kernel, save outputs, then export HTML.

    Errors propagate to the caller; a partially executed notebook is retained
    for inspection. No rollback of the report directory or input data is done.
    """
    kernel_options = {}
    if thread_limit is not None:
        if isinstance(thread_limit, bool) or not isinstance(thread_limit, int) or thread_limit <= 0:
            raise ValueError("thread_limit must be a positive integer")
        kernel_options["env"] = os.environ | {
            name: str(thread_limit) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
            )
        }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ReportPaths(output_dir / "report.ipynb", output_dir / "report.html")
    paths.metrics_path.unlink(missing_ok=True)
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {
            "name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python",
        },
        "title": title,
    })
    with stage("notebook.serialize", output_dir=output_dir, phase="source"):
        nbformat.write(notebook, paths.notebook_path)
    client = NotebookClient(
        notebook, timeout=180, kernel_name="python3", allow_errors=False,
        resources={"metadata": {"path": str(output_dir)}},
    )
    # Override the installed kernelspec rather than trusting its Python path.
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
    ]
    try:
        with stage("notebook.execute", output_dir=output_dir):
            client.execute(**kernel_options)
    finally:
        with stage("notebook.serialize", output_dir=output_dir, phase="executed"):
            nbformat.write(notebook, paths.notebook_path)

    # The basic template is an HTML fragment with inline PNGs, without CDN JS,
    # MathJax or external stylesheets. Supply a small, entirely local document.
    with stage("notebook.render", output_dir=output_dir):
        exporter = HTMLExporter(
            template_name="basic", exclude_input=True,
            exclude_input_prompt=True, exclude_output_prompt=True,
        )
        body, _ = exporter.from_notebook_node(notebook)
        html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; color: #202b33; background: white;
        max-width: 1200px; margin: 2rem auto; padding: 0 1.5rem; }}
h1, h2, h3 {{ line-height: 1.25; }}
table {{ border-collapse: collapse; font-size: 0.9rem; margin: 1rem 0; }}
th, td {{ padding: 0.4rem 0.7rem; border-bottom: 1px solid #d9e1e5; text-align: left; }}
th {{ background: #f1f5f7; }}
img {{ max-width: 100%; height: auto; }}
.output_subarea {{ overflow-x: auto; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
</style></head><body>{body}</body></html>
"""
        paths.html_path.write_text(html, encoding="utf-8")
    return paths
