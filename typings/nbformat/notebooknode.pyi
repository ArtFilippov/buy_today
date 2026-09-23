"""Normalized notebook fields plus the format's extensible JSON attribute map."""

from typing import Any, Literal, Self, overload
from ._struct import Struct

class NotebookNode(Struct):
    cells: list[NotebookNode]
    metadata: NotebookNode
    nbformat: int
    nbformat_minor: int
    cell_type: Literal["code", "markdown", "raw"]
    id: str
    source: str
    execution_count: int | None
    outputs: list[NotebookNode]
    output_type: Literal["stream", "display_data", "execute_result", "error"]
    data: NotebookNode
    name: Literal["stdout", "stderr"]
    text: str
    ename: str
    evalue: str
    traceback: list[str]
    attachments: NotebookNode

    def __getattr__(self, key: str) -> Any: ...
    def __setattr__(self, key: str, value: Any) -> None: ...
    def __delattr__(self, key: str) -> None: ...
    def dict(self) -> dict[str, Any]: ...
    def allow_new_attr(self, allow: bool = True) -> None: ...
    def hasattr(self, key: str) -> bool: ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...

@overload
def from_dict(d: dict[str, Any]) -> NotebookNode: ...
@overload
def from_dict(d: list[Any] | tuple[Any, ...]) -> list[Any]: ...
@overload
def from_dict(d: Any) -> Any: ...
