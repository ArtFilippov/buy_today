"""Validation is also an intentional boundary for malformed JSON inputs."""

from collections.abc import Iterator
from typing import Any
from jsonschema.exceptions import ValidationError as ValidationError
from .notebooknode import NotebookNode

def validate(nbdict: object = None, ref: str | None = None, version: int | None = None,
             version_minor: int | None = None, relax_add_props: bool = False,
             nbjson: object = None) -> None: ...
def isvalid(nbjson: object, ref: str | None = None, version: int | None = None,
            version_minor: int | None = None) -> bool: ...
def iter_validate(nbdict: object, ref: str | None = None, version: int | None = None,
                  version_minor: int | None = None,
                  relax_add_props: bool = False) -> Iterator[ValidationError]: ...
