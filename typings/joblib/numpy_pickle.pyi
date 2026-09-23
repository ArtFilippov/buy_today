"""joblib 1.6 persistence: filenames are returned only for path destinations."""

from os import PathLike
from typing import Any, BinaryIO, Literal, overload

type _Compression = int | str | tuple[str, int]

@overload
def dump(value: object, filename: str | PathLike[str], compress: _Compression = 0,
         protocol: int | None = None) -> list[str]: ...
@overload
def dump(value: object, filename: BinaryIO, compress: _Compression = 0,
         protocol: int | None = None) -> None: ...

def load(filename: str | PathLike[str] | BinaryIO,
         mmap_mode: Literal["r", "r+", "w+", "c"] | None = None,
         ensure_native_byte_order: bool | Literal["auto"] = "auto") -> Any: ...
