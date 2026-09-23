"""nbformat 5.11 public notebook I/O and validation contracts."""

from os import PathLike
from types import ModuleType
from typing import Any, Protocol

from jsonschema.exceptions import ValidationError as ValidationError
from nbformat import v1 as v1, v2 as v2, v3 as v3, v4 as v4
from .notebooknode import NotebookNode as NotebookNode, from_dict as from_dict
from nbformat.sentinel import Sentinel as Sentinel
from .validator import validate as validate

class _TextReader(Protocol):
    def read(self) -> str: ...

class _TextWriter(Protocol):
    def write(self, text: str, /) -> object: ...

class NBFormatError(ValueError): ...

NO_CONVERT: Sentinel
current_nbformat: int
current_nbformat_minor: int
__version__: str
version_info: tuple[int, int, int]
versions: dict[int, ModuleType]

def read(fp: str | PathLike[str] | _TextReader, as_version: int | Sentinel,
         capture_validation_error: dict[str, ValidationError] | None = None,
         **kwargs: Any) -> NotebookNode: ...
def reads(s: str, as_version: int | Sentinel,
          capture_validation_error: dict[str, ValidationError] | None = None,
          **kwargs: Any) -> NotebookNode: ...
def write(nb: NotebookNode, fp: str | PathLike[str] | _TextWriter,
          version: int | Sentinel = ...,
          capture_validation_error: dict[str, ValidationError] | None = None,
          **kwargs: Any) -> None: ...
def writes(nb: NotebookNode, version: int | Sentinel = ...,
           capture_validation_error: dict[str, ValidationError] | None = None,
           **kwargs: Any) -> str: ...
def convert(nb: NotebookNode, to_version: int) -> NotebookNode: ...
