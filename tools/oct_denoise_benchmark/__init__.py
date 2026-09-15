"""Leakage-aware OCT denoising benchmark.

Keep the legacy adapter registry lazy: protocol/audit entry points do not use
the historical wavelet baseline and should not fail before argument parsing
merely because its optional runtime has not been installed yet.
"""

from typing import Any

__all__ = ["ADAPTERS", "denoise"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import adapters

        return getattr(adapters, name)
    raise AttributeError(name)
