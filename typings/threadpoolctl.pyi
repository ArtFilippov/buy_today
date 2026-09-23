"""threadpoolctl 3.7 limits, context-manager restoration and decorators."""

from contextlib import ContextDecorator
from types import TracebackType
from typing import Literal, Self, TypedDict

type _UserAPI = Literal["blas", "openmp"]
type _Limits = (
    int | dict[str, int | None] | Literal["sequential_blas_under_openmp"]
    | list[_ThreadpoolInfo] | ThreadpoolController | None
)

class _ThreadpoolInfo(TypedDict, total=False):
    user_api: str
    internal_api: str
    num_threads: int
    prefix: str
    filepath: str
    version: str | None
    threading_layer: str
    architecture: str

class LibController:
    filepath: str
    prefix: str
    user_api: str
    internal_api: str
    version: str | None
    def get_num_threads(self) -> int: ...
    def set_num_threads(self, num_threads: int) -> None: ...
    def info(self, debugging_info: bool = False) -> _ThreadpoolInfo: ...

class _ThreadpoolLimiter:
    def __enter__(self) -> Self: ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None) -> None: ...
    def restore_original_limits(self) -> None: ...
    def unregister(self) -> None: ...
    def get_original_num_threads(self) -> dict[str, int | None]: ...

class _ThreadpoolLimiterDecorator(_ThreadpoolLimiter, ContextDecorator): ...

class threadpool_limits(_ThreadpoolLimiter):
    def __init__(self, limits: _Limits = None, user_api: _UserAPI | None = None) -> None: ...
    @classmethod
    def wrap(cls, limits: _Limits = None,
             user_api: _UserAPI | None = None) -> _ThreadpoolLimiterDecorator: ...

class ThreadpoolController:
    lib_controllers: list[LibController]
    def __init__(self) -> None: ...
    def info(self, debugging_info: bool = False) -> list[_ThreadpoolInfo]: ...
    def select(self, **kwargs: object) -> ThreadpoolController: ...
    def limit(self, *, limits: _Limits = None,
              user_api: _UserAPI | None = None) -> _ThreadpoolLimiter: ...
    def wrap(self, *, limits: _Limits = None,
             user_api: _UserAPI | None = None) -> _ThreadpoolLimiterDecorator: ...

def threadpool_info(debugging_info: bool = False) -> list[_ThreadpoolInfo]: ...
def register(controller: type[LibController]) -> None: ...
__version__: str
