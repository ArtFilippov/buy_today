"""Shared report cell construction and explanatory source fragments."""

from textwrap import dedent

import nbformat


DATASET_SCOPE = """Подготовленный Olist — ретроспективная выборка доставленных покупок товаров
с единственным продавцом, после удаления строк с любым пропуском.
Распределения описывают сохранённые покупки, а не весь рынок или каталог.
"""

REPORT_IMPORTS = """from pathlib import Path
import hashlib
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from buy_today.schema import read_dataset
"""


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip())


def markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source)


def setup_code(source: str) -> nbformat.NotebookNode:
    return code(REPORT_IMPORTS + dedent(source))
