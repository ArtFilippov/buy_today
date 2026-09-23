"""Execute report notebooks in the caller's Python and export HTML and metrics."""

from html import escape
import json
import os
from pathlib import Path
import sys
from typing import Any, cast

from nbclient import NotebookClient
from ipykernel.kernelspec import make_ipkernel_cmd
from nbconvert import HTMLExporter
import nbformat
import numpy as np

from buy_today.progress import stage
from buy_today.artifacts import ReportPaths


def write_metrics(metrics: dict[str, Any], path: Path | str = "metrics.json") -> None:
    """Serialize computed values from a notebook's final cell, without NaN/Inf.

    Args:
        metrics (dict[str, Any]): JSON values, including numpy scalars or arrays.
        path (Path | str, default='metrics.json'): Destination for the metrics document.
    """

    def convert(value: object) -> object:
        if isinstance(value, np.generic):
            return cast("np.generic[object]", value).item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__} as report metrics")

    content = json.dumps(
        metrics,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        default=convert,
    )
    _ = Path(path).write_text(content + "\n", encoding="utf-8")


def _kernel_options(thread_limit: object) -> dict[str, dict[str, str]]:
    if thread_limit is None:
        return {}
    if isinstance(thread_limit, bool) or not isinstance(thread_limit, int) or thread_limit <= 0:
        raise ValueError("thread_limit must be a positive integer")
    return {
        "env": os.environ
        | {
            name: str(thread_limit)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        }
    }


def _execute_kernel(
    client: NotebookClient, output_dir: Path, kernel_options: dict[str, dict[str, str]]
) -> None:
    with stage("notebook.execute", output_dir=output_dir):
        _ = client.execute(**kernel_options)


def execute_report(
    cells: list[nbformat.NotebookNode],
    output_dir: Path | str,
    *,
    title: str,
    thread_limit: int | None = None,
) -> ReportPaths:
    """Write source, execute in a fresh kernel, save outputs, then export HTML.

    Errors propagate to the caller; a partially executed notebook is retained
    for inspection. No rollback of the report directory or input data is done.

    Args:
        cells (list[nbformat.NotebookNode]): Report source cells in execution order.
        output_dir (Path | str): Destination for notebook, metrics and HTML.
        title (str): Notebook and standalone HTML title.
        thread_limit (int | None, default=None): Optional native thread count for the fresh kernel.

    Returns:
        ReportPaths: Executed notebook and standalone HTML locations.
    """
    kernel_options = _kernel_options(thread_limit)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ReportPaths(output_dir / "report.ipynb", output_dir / "report.html")
    paths.metrics_path.unlink(missing_ok=True)
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
            },
            "title": title,
        },
    )
    with stage("notebook.serialize", output_dir=output_dir, phase="source"):
        nbformat.write(notebook, paths.notebook_path)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(output_dir)}},
    )
    # Override the installed kernelspec rather than trusting its Python path.
    client.km = client.create_kernel_manager()
    kernel_spec = client.km.kernel_spec
    assert kernel_spec is not None
    kernel_spec.argv = make_ipkernel_cmd(executable=sys.executable)
    try:
        _execute_kernel(client, output_dir, kernel_options)
    finally:
        with stage("notebook.serialize", output_dir=output_dir, phase="executed"):
            nbformat.write(notebook, paths.notebook_path)

    with stage("notebook.render", output_dir=output_dir):
        _ = paths.html_path.write_text(_render_html(notebook, title), encoding="utf-8")
    return paths


def _render_html(notebook: nbformat.NotebookNode, title: str) -> str:
    # The basic template is an HTML fragment with inline PNGs, without CDN JS,
    # MathJax or external stylesheets. Supply a small, entirely local document.
    exporter = HTMLExporter(
        template_name="basic",
        exclude_input=True,
        exclude_input_prompt=True,
        exclude_output_prompt=True,
    )
    body, _ = exporter.from_notebook_node(notebook)
    return f"""<!doctype html>
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
