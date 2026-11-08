"""Native thread limits are scoped, restored, and safe across API callers."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import cast

import numpy as np
import pytest
from threadpoolctl import threadpool_info  # pyright: ignore[reportMissingTypeStubs]

from panelchemy.numeric_threads import single_threaded_blas


def blas_threads() -> dict[str, int]:
    pools = cast("list[dict[str, object]]", threadpool_info())
    return {
        cast("str", pool["filepath"]): cast("int", pool["num_threads"])
        for pool in pools
        if pool["user_api"] == "blas"
    }


def test_small_solve_uses_one_thread_and_restores_loaded_libraries() -> None:
    before = blas_threads()

    @single_threaded_blas
    def solve() -> np.ndarray:
        assert all(value == 1 for value in blas_threads().values())
        return np.linalg.solve(np.array([[2.0, 1.0], [1.0, 3.0]]), np.array([4.0, 7.0]))

    assert np.allclose(solve(), [1, 2])
    assert blas_threads() == before


def test_exception_restores_prior_limits() -> None:
    before = blas_threads()

    @single_threaded_blas
    def fail() -> None:
        assert all(value == 1 for value in blas_threads().values())
        raise RuntimeError("solver failed")

    with pytest.raises(RuntimeError, match="solver failed"):
        fail()
    assert blas_threads() == before


def test_nested_scope_restores_outer_limit_before_host_limit() -> None:
    before = blas_threads()

    @single_threaded_blas
    def inner() -> None:
        assert all(value == 1 for value in blas_threads().values())

    @single_threaded_blas
    def outer() -> None:
        inner()
        assert all(value == 1 for value in blas_threads().values())

    outer()
    assert blas_threads() == before


def test_concurrent_scopes_cannot_restore_each_others_limits() -> None:
    before = blas_threads()
    first_entered, second_started, second_entered, release = (Event() for _ in range(4))

    @single_threaded_blas
    def first() -> None:
        first_entered.set()
        assert release.wait(5)
        assert all(value == 1 for value in blas_threads().values())

    @single_threaded_blas
    def second() -> None:
        second_entered.set()
        assert all(value == 1 for value in blas_threads().values())

    def invoke_second() -> None:
        second_started.set()
        second()

    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(first)
        assert first_entered.wait(5)
        b = pool.submit(invoke_second)
        assert second_started.wait(5)
        try:
            assert not second_entered.wait(0.05)
        finally:
            release.set()
        a.result(timeout=5)
        b.result(timeout=5)
    assert second_entered.is_set()
    assert blas_threads() == before
