# importing arkjit.numba_ext will install the relevant hooks with numba
import arkjit.numba_ext           # noqa: F401

from ._version import __version__
from .compiler import ArkoudaCompiler, optimize

__all__ = [
    "__version__",
    "optimize",
    "ArkoudaCompiler",
]
