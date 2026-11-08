"""Scope native BLAS threading to the small, bounded spline solves."""

from collections.abc import Callable
from functools import wraps
from threading import RLock

from threadpoolctl import threadpool_limits  # pyright: ignore[reportMissingTypeStubs]

# BLAS limits can be process-wide. Serialize our scoped changes so concurrent
# API callers cannot restore one another's limits out of order. This does not
# change environment variables or permanently configure the host's libraries.
_BLAS_LOCK = RLock()


def single_threaded_blas[**P, R](operation: Callable[P, R]) -> Callable[P, R]:
    """Avoid native thread-team overhead; restore prior limits on every exit."""

    @wraps(operation)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with _BLAS_LOCK, threadpool_limits(limits=1, user_api="blas"):
            return operation(*args, **kwargs)

    return wrapped
