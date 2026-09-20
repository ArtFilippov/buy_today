"""Execute report notebooks in the caller's Python and export standalone HTML."""

from dataclasses import dataclass
from html import escape
from pathlib import Path
import sys

from nbclient import NotebookClient
from nbconvert import HTMLExporter
import nbformat


@dataclass(frozen=True)
class ReportPaths:
    notebook_path: Path
    html_path: Path


def execute_report(
    cells: list[nbformat.NotebookNode], output_dir: Path | str, *, title: str
) -> ReportPaths:
    """Write source, execute in a fresh kernel, save outputs, then export HTML.

    Errors propagate to the caller; a partially executed notebook is retained
    for inspection. No rollback of the report directory or input data is done.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ReportPaths(output_dir / "report.ipynb", output_dir / "report.html")
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {
            "name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python",
        },
        "title": title,
    })
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
        client.execute()
    finally:
        nbformat.write(notebook, paths.notebook_path)

    # The basic template is an HTML fragment with inline PNGs, without CDN JS,
    # MathJax or external stylesheets. Supply a small, entirely local document.
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
