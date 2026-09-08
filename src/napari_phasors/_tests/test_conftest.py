"""Tests for helpers defined directly in ``_tests/conftest.py``."""

import sys
from unittest.mock import patch

import pytest
import requests

from napari_phasors._tests.conftest import (
    configure_phasorpy_retries,
    network_flake_reason,
    pytest_runtest_call,
    pytest_runtest_setup,
    skip_if_network_flake,
)


def _http_error(status):
    """A ``requests`` HTTPError carrying ``status``, as pooch would raise."""
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(f"HTTP {status}", response=response)


def test_configure_phasorpy_retries_sets_retry_on_all_repositories():
    """Every phasorpy dataset repository gets retries enabled."""
    import phasorpy.datasets

    for repo in phasorpy.datasets.REPOSITORIES.values():
        repo.retry_if_failed = 0

    configure_phasorpy_retries()

    assert phasorpy.datasets.REPOSITORIES
    for repo in phasorpy.datasets.REPOSITORIES.values():
        assert repo.retry_if_failed == 3


def test_configure_phasorpy_retries_swallows_import_error():
    """A missing/incompatible phasorpy.datasets must not raise or fail setup."""
    # Setting a module to None in sys.modules makes the next `import`/`from
    # ... import` statement for it raise ImportError, simulating an
    # environment where ``phasorpy.datasets`` is unavailable.
    with patch.dict(sys.modules, {"phasorpy.datasets": None}):
        configure_phasorpy_retries()  # must not raise


def test_configure_phasorpy_retries_swallows_attribute_error():
    """An unexpected REPOSITORIES shape (e.g. no ``.values()``) is ignored."""
    with patch("phasorpy.datasets.REPOSITORIES", None):
        configure_phasorpy_retries()  # must not raise


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.ReadTimeout(
            "HTTPSConnectionPool(host='zenodo.org', port=443): Read timed "
            "out. (read timeout=30)"
        ),
        requests.exceptions.ConnectTimeout("connect timed out"),
        requests.exceptions.ConnectionError("connection aborted"),
        requests.exceptions.ChunkedEncodingError("truncated response"),
        ConnectionResetError("connection reset by peer"),
        _http_error(503),
        _http_error(429),
    ],
)
def test_network_flake_reason_recognises_transient_failures(exc):
    """Transport errors and retryable HTTP statuses are reported as flakes."""
    assert network_flake_reason(exc) is not None


@pytest.mark.parametrize(
    "exc",
    [
        AssertionError("assert 1 == 2"),
        # pytest-qt's ``waitUntil`` raises a bare ``TimeoutError``; that is a
        # real failure, not a network one.
        TimeoutError("waitUntil timed out"),
        ValueError("bad data"),
        # A wrong URL or a stale registry entry must keep failing.
        _http_error(404),
    ],
)
def test_network_flake_reason_ignores_real_failures(exc):
    """Anything that is not a transient network problem stays a failure."""
    assert network_flake_reason(exc) is None


def test_network_flake_reason_walks_the_exception_chain():
    """A timeout wrapped in another exception is still recognised."""
    try:
        try:
            raise requests.exceptions.ReadTimeout("read timed out")
        except requests.exceptions.ReadTimeout as timeout:
            raise RuntimeError("download failed") from timeout
    except RuntimeError as exc:
        assert "ReadTimeout" in network_flake_reason(exc)


def test_network_flake_reason_survives_a_self_referencing_chain():
    """A cyclic ``__context__`` chain must not loop forever."""
    exc = ValueError("boom")
    exc.__context__ = exc
    assert network_flake_reason(exc) is None


def test_skip_if_network_flake_skips_on_timeout():
    with pytest.raises(pytest.skip.Exception, match="ReadTimeout"):
        skip_if_network_flake(requests.exceptions.ReadTimeout("timed out"))


def test_skip_if_network_flake_reraises_other_errors():
    with pytest.raises(ValueError, match="bad data"):
        skip_if_network_flake(ValueError("bad data"))


@pytest.mark.parametrize("hook", [pytest_runtest_setup, pytest_runtest_call])
def test_hooks_turn_a_timeout_into_a_skip(hook):
    """A download timeout raised by setup or by the test body is skipped."""
    wrapper = hook(item=None)
    next(wrapper)
    with pytest.raises(pytest.skip.Exception, match="transient network"):
        wrapper.throw(requests.exceptions.ReadTimeout("timed out"))


@pytest.mark.parametrize("hook", [pytest_runtest_setup, pytest_runtest_call])
def test_hooks_let_real_failures_through(hook):
    """Non-network errors keep propagating, so the test still fails."""
    wrapper = hook(item=None)
    next(wrapper)
    with pytest.raises(AssertionError, match="assert 1 == 2"):
        wrapper.throw(AssertionError("assert 1 == 2"))


@pytest.mark.parametrize("hook", [pytest_runtest_setup, pytest_runtest_call])
def test_hooks_are_transparent_when_nothing_raises(hook):
    """The wrapper hands back the wrapped hook's result unchanged."""
    wrapper = hook(item=None)
    next(wrapper)
    with pytest.raises(StopIteration) as excinfo:
        wrapper.send("hook result")
    assert excinfo.value.value == "hook result"
