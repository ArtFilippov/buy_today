"""Strict consumer checks, including unmodified installed package modules."""

from typing import assert_type
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import joblib
import nbformat
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

estimator = BaseEstimator()
_ = assert_type(clone(estimator), BaseEstimator)
_ = assert_type(clone(TruncatedSVD()), TruncatedSVD)
_ = assert_type(clone(TSNE()), TSNE)
_ = assert_type(clone(StandardScaler()), StandardScaler)
_ = assert_type(StandardScaler().get_params(), dict[str, object])

def accepts_estimator(model: BaseEstimator) -> BaseEstimator:
    return model

_ = accepts_estimator(StandardScaler())
_ = accepts_estimator(TruncatedSVD())
_ = assert_type(csr_matrix((2, 3)), csr_matrix[np.float64])

check_is_fitted(estimator, ["coef_"], all_or_any=any)
_ = assert_type(silhouette_score([[0.0], [1.0], [5.0]], [0, 0, 1]), float)
_ = assert_type(joblib.dump(estimator, Path("model.joblib")), list[str])
_ = assert_type(joblib.dump(estimator, BytesIO()), None)
cell = nbformat.v4.new_code_cell("print(1)")
notebook = nbformat.v4.new_notebook(cells=[cell], metadata={"title": "Report"})
_ = assert_type(notebook.cells, list[nbformat.NotebookNode])
_ = assert_type(cell.source, str)
_ = assert_type(cell.execution_count, int | None)
_ = assert_type(nbformat.read(Path("report.ipynb"), as_version=4), nbformat.NotebookNode)
nbformat.write(notebook, StringIO())
nbformat.validate(notebook)
with threadpool_limits(limits=1, user_api="blas") as limit:
    _ = assert_type(limit, threadpool_limits)

@threadpool_limits.wrap(limits=1)
def limited(value: str) -> int:
    return len(value)

_ = assert_type(limited("value"), int)
figure = Figure()
axes = figure.subplots()
_ = assert_type(axes, Axes)
with mpl.rc_context({"figure.dpi": 120}):
    _ = assert_type(axes.text(0, 0, "label", color="black"), Text)
    _ = assert_type(axes.scatter([0], [1], alpha=0.3), PathCollection)
    mappable = axes.imshow([[0, 1]], cmap="viridis")
    _ = assert_type(mappable, AxesImage)
    _ = axes.hist([0, 1], bins=2, color="green")
    bars = axes.barh([0], [1], color="blue")
    _ = axes.bar_label(bars, padding=2, fontsize=8)
    _ = axes.stairs([1, 2], [0, 1, 2], color="red")
    axes.set_xscale("log", base=10)
    _ = axes.set_xticks([1], ["one"], rotation=45)
    _ = axes.set_yticks([1], ["one"], fontsize=8)
    _ = axes.set_title("Title", color="red")
    _ = axes.set_xlabel("Value", fontsize=8)
    axes.grid(axis="x", alpha=0.2)
    _ = axes.legend(loc="upper right", frameon=False)
    _ = figure.text(0, 0, "caption", ha="left")
    _ = figure.suptitle("Title", fontsize=10)
    _ = figure.colorbar(mappable, ax=axes, ticks=[0, 1])
    figure.savefig(BytesIO(), format="png")
_ = assert_type(PercentFormatter(), PercentFormatter)
_ = assert_type(plt.get_fignums(), list[int])
