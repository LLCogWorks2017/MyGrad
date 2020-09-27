"""
Provides user interface for suspending computational graph tracking and back-propagation
"""
from functools import wraps
from typing import Callable

import numpy as np

__all__ = ["no_autodiff"]


# If `False`, suspends all computational graph tracking and backprop
TRACK_GRAPH = True  # type: bool


class _NoAutoDiff:
    """ Serves as a context manager and decorator for suspending
    all computational graph tracking."""

    # tracks context depth
    _depth = 0  # type: int

    def __enter__(self):
        """Suspends graph-tracking"""
        global TRACK_GRAPH
        self._depth += 1
        TRACK_GRAPH = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restores graph-tracking when context depth returns to 0"""
        global TRACK_GRAPH
        self._depth -= 1

        TRACK_GRAPH = self._depth == 0

    def __call__(self, func: Callable, to_numpy: bool = False) -> Callable:
        """Decorates a function so that it will have graph-tracking suspended
        during its execution.

        Parameters
        ----------
        func : Callable
            The function to be decorated

        to_numpy : bool, optional (default=False)
            If true, the output is assumed to be array-like and
            will be cast to a numpy array

        Returns
        -------
        decorated_func : Callable"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                out = func(*args, **kwargs)
            return out if not to_numpy else np.asarray(out)

        return wrapper


no_autodiff = _NoAutoDiff()