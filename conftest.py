"""Repo-root pytest configuration.

This conftest is loaded by pytest before plugins (including pytest-cov) start
and before the ``napari_phasors`` package is imported for coverage source
resolution. We use that early hook to pre-register vispy's Qt backend.
"""

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
