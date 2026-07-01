"""Tests for helpers defined directly in ``_tests/conftest.py``."""

import sys
from unittest.mock import patch

from napari_phasors._tests.conftest import configure_phasorpy_retries


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
