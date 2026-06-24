"""Repo-root pytest configuration.

This conftest is loaded by pytest before plugins (including pytest-cov) start
and before the ``napari_phasors`` package is imported for coverage source
resolution. We use that early hook to pre-register vispy's Qt backend.
"""

# DIAGNOSTIC (temporary): capture a C-level traceback per xdist worker when a
# PySide6 teardown segfault kills the process. ``faulthandler`` writes via a
# raw file descriptor from its signal handler, so the dump survives a crash
# even though pytest-xdist discards the dead worker's stderr. Each worker
# writes its own file; CI prints them on failure. Remove once the crash is
# located. Controlled by ``NP_FAULTHANDLER_DIR`` so it is a no-op locally.
try:  # pragma: no cover - diagnostic only
    import os as _os

    _fh_dir = _os.environ.get("NP_FAULTHANDLER_DIR")
    if _fh_dir:
        import faulthandler as _faulthandler

        _worker = _os.environ.get("PYTEST_XDIST_WORKER", "main")
        _os.makedirs(_fh_dir, exist_ok=True)
        _fh_file = (
            open(  # noqa: SIM115 - must stay open for the process lifetime
                _os.path.join(_fh_dir, f"faulthandler_{_worker}.log"), "w"
            )
        )
        _faulthandler.enable(file=_fh_file, all_threads=True)
except Exception:  # noqa: BLE001
    pass

# vispy resolves its backend with ``getattr(vispy.app.backends, "_pyside6")``,
# which raises ``AttributeError`` unless that submodule has already been
# imported. Depending on import ordering (e.g. when coverage instruments the
# package, triggering ``import napari_phasors`` -> biaplotter -> vispy
# ``use_app`` before the lazy backend import has run), collection fails with a
# flaky ``module 'vispy.app.backends' has no attribute '_pyside6'`` error.
# Importing the backend submodule here makes backend resolution deterministic.
try:  # pragma: no cover - import guard only
    import importlib

    from qtpy import API_NAME

    if API_NAME:
        importlib.import_module(f"vispy.app.backends._{API_NAME.lower()}")
except Exception:  # noqa: BLE001
    pass


try:  # pragma: no cover - environment-dependent mitigation
    import gc

    from qtpy import API_NAME as _QT_API_NAME

    if (_QT_API_NAME or "").lower() == "pyside6":
        gc.disable()
except Exception:  # noqa: BLE001
    pass
