from collections.abc import Mapping
from typing import Any, Literal, TypedDict, Unpack
from ..notebooknode import NotebookNode

class _CellOptions(TypedDict, total=False):
    id: str
    metadata: Mapping[str, Any]

class _CodeOptions(_CellOptions, total=False):
    cell_type: Literal["code"]
    execution_count: int | None
    outputs: list[NotebookNode]

class _MarkdownOptions(_CellOptions, total=False):
    cell_type: Literal["markdown"]
    attachments: Mapping[str, Any]

class _RawOptions(_CellOptions, total=False):
    cell_type: Literal["raw"]
    attachments: Mapping[str, Any]

class _NotebookOptions(TypedDict, total=False):
    cells: list[NotebookNode]
    metadata: Mapping[str, Any]
    nbformat: int
    nbformat_minor: int

class _OutputOptions(TypedDict, total=False):
    name: Literal["stdout", "stderr"]
    text: str
    metadata: Mapping[str, Any]
    execution_count: int | None
    ename: str
    evalue: str
    traceback: list[str]

def new_code_cell(source: str = "", **kwargs: Unpack[_CodeOptions]) -> NotebookNode: ...
def new_markdown_cell(source: str = "", **kwargs: Unpack[_MarkdownOptions]) -> NotebookNode: ...
def new_raw_cell(source: str = "", **kwargs: Unpack[_RawOptions]) -> NotebookNode: ...
def new_notebook(**kwargs: Unpack[_NotebookOptions]) -> NotebookNode: ...
def new_output(output_type: Literal["stream", "display_data", "execute_result", "error"],
               data: Mapping[str, Any] | None = None,
               **kwargs: Unpack[_OutputOptions]) -> NotebookNode: ...
def output_from_msg(msg: Mapping[str, Any]) -> NotebookNode: ...
def validate(node: NotebookNode, ref: str | None = None) -> None: ...
nbformat: int
nbformat_minor: int
nbformat_schema: dict[tuple[int | None, int | None], str]
