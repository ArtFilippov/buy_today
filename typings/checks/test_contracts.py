"""Verify stub precision, fallback resolution and observed runtime behavior."""

from __future__ import annotations

from io import BytesIO, StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import joblib
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
import nbformat
from nbformat._struct import Struct
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PYRIGHT = ROOT / ".venv/bin/pyright"


class StubContracts(unittest.TestCase):
    def test_positive_types_and_installed_module_fallback(self) -> None:
        result = subprocess.run(
            [str(PYRIGHT), "--pythonpath", sys.executable, "-p", str(HERE / "pyrightconfig.json")],
            text=True, capture_output=True, check=False, cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_invalid_calls_are_still_rejected(self) -> None:
        source = (HERE / "negative.py.txt").read_text(encoding="utf-8")
        expected = {index for index, line in enumerate(source.splitlines()) if line.endswith("# error")}
        with TemporaryDirectory(prefix="stub-contracts-", dir="/tmp/opencode") as directory:
            target = Path(directory)
            _ = (target / "negative.py").write_text(source, encoding="utf-8")
            _ = (target / "pyrightconfig.json").write_text(json.dumps({
                "extends": str(HERE / "pyrightconfig.json"),
                "include": ["negative.py"],
                "executionEnvironments": [{"root": str(target)}],
            }), encoding="utf-8")
            result = subprocess.run(
                [str(PYRIGHT), "--pythonpath", sys.executable, "-p", str(target), "--outputjson"],
                text=True, capture_output=True, check=False, cwd=ROOT,
            )
        report = json.loads(result.stdout)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        diagnostics = report["generalDiagnostics"]
        self.assertEqual({item["range"]["start"]["line"] for item in diagnostics}, expected,
                         result.stdout)
        self.assertTrue(all(item["severity"] == "error" for item in diagnostics))
        self.assertFalse(any(item["rule"].startswith("reportMissing") for item in diagnostics))

    def test_pickle_paths_streams_and_round_trip(self) -> None:
        value = {"array": np.arange(3), "parameter": 17}
        stream = BytesIO()
        self.assertIsNone(joblib.dump(value, stream))
        _ = stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream)["array"], value["array"])
        with TemporaryDirectory(prefix="stub-pickle-", dir="/tmp/opencode") as directory:
            path = Path(directory) / "model.joblib"
            self.assertEqual(joblib.dump(value, path, compress=("gzip", 1)), [str(path)])
            self.assertEqual(joblib.load(path, ensure_native_byte_order=True)["parameter"], 17)

    def test_notebook_fields_and_struct_copy(self) -> None:
        cell = nbformat.v4.new_code_cell("print(1)", execution_count=1,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="1\n")])
        notebook = nbformat.v4.new_notebook(cells=[cell], metadata={"extension": {"enabled": True}})
        stream = StringIO()
        nbformat.write(notebook, stream)
        _ = stream.seek(0)
        restored = nbformat.read(stream, as_version=4)
        nbformat.validate(restored)
        self.assertEqual(restored.cells[0].source, "print(1)")
        self.assertEqual(restored.cells[0].outputs[0].text, "1\n")
        self.assertTrue(restored.metadata.extension.enabled)
        # NotebookNode inherits Struct.copy, which deliberately returns Struct.
        self.assertIs(type(cell.copy()), Struct)

    def test_clone_preserves_types_and_fitted_state_contract(self) -> None:
        scaler = StandardScaler().fit([[1.0], [2.0], [3.0]])
        copied = clone(scaler)
        self.assertIsInstance(copied, BaseEstimator)
        self.assertIs(type(copied), StandardScaler)
        self.assertIsNot(copied, scaler)
        self.assertEqual(copied.get_params(), scaler.get_params())
        with self.assertRaises(NotFittedError):
            check_is_fitted(copied, ["scale_"])
        self.assertIs(copied.set_params(with_mean=False), copied)
        self.assertIs(type(clone(TruncatedSVD())), TruncatedSVD)
        self.assertEqual(clone({"model": scaler})["model"].get_params(), scaler.get_params())

    def test_threadpool_context_and_decorator(self) -> None:
        with threadpool_limits(limits=1, user_api="blas") as limiter:
            self.assertIsInstance(limiter.get_original_num_threads(), dict)

        @threadpool_limits.wrap(limits=1, user_api="blas")
        def limited(value: str) -> int:
            return len(value)

        self.assertEqual(limited("abc"), 3)

    def test_matplotlib_concrete_results_and_rc_restoration(self) -> None:
        original = mpl.rcParams["figure.dpi"]
        with mpl.rc_context({"figure.dpi": 72}):
            figure = Figure(figsize=(2, 2))
            _canvas = FigureCanvasAgg(figure)
            axes = figure.subplots()
            self.assertIsInstance(axes, Axes)
            self.assertIsInstance(axes.text(0, 0, "caption", color="red"), Text)
            mappable = axes.imshow([[0.0, 1.0]], cmap="viridis")
            _ = figure.colorbar(mappable, ax=axes, ticks=[0, 1])
            _ = figure.suptitle("Title", fontsize=8)
            image = BytesIO()
            figure.savefig(image, format="png")
            self.assertTrue(image.getvalue().startswith(b"\x89PNG"))
        self.assertEqual(mpl.rcParams["figure.dpi"], original)


if __name__ == "__main__":
    unittest.main()
