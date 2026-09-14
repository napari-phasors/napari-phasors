"""Repo-root pytest configuration.

This conftest is loaded by pytest before plugins (including pytest-cov) start
and before the ``napari_phasors`` package is imported for coverage source
resolution. We use that early hook to pre-register vispy's Qt backend.
"""

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
