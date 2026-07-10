"""Unobtrusive background check for newer ``napari-phasors`` releases.

On startup (once per session, and at most once a week) this queries PyPI in a
background thread and, if a newer stable release is available, shows a small
dialog with upgrade instructions. The dialog offers a "Don't notify me about
updates again" checkbox that permanently disables the check.

Everything here fails silently: no network, an odd PyPI response, or an
unwritable config directory must never disrupt the plugin.
"""

import json
import os
import time
import urllib.request
from pathlib import Path

from packaging.version import InvalidVersion, Version
from platformdirs import user_config_dir

# pragma: no cover - version file always present in a built/installed package
try:
    from ._version import version as __installed_version__
except ImportError:
    __installed_version__ = "unknown"

#: PyPI JSON endpoint for the project.
PYPI_URL = "https://pypi.org/pypi/napari-phasors/json"

#: Minimum time between actual network checks (one week).
CHECK_INTERVAL_SECONDS = 7 * 24 * 60 * 60

#: Network timeout for the PyPI request, in seconds.
_REQUEST_TIMEOUT = 3.0

#: Guard so the check runs at most once per interpreter session, regardless of
#: how many plugin widgets the user opens.
_checked_this_session = False


def _config_path() -> Path:
    """Return the path to the persistent update-check config file."""
    return Path(user_config_dir("napari-phasors")) / "update_check.json"


def _load_config() -> dict:
    """Load the persisted config, returning an empty dict on any failure."""
    try:
        with open(_config_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_config(config: dict) -> None:
    """Persist the config, silently ignoring any write failure."""
    try:
        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f)
    except OSError:
        pass


def _fetch_latest_version() -> str | None:
    """Return the latest stable version string from PyPI, or ``None``."""
    try:
        request = urllib.request.Request(
            PYPI_URL, headers={"User-Agent": "napari-phasors-update-check"}
        )
        with urllib.request.urlopen(
            request, timeout=_REQUEST_TIMEOUT
        ) as response:
            data = json.load(response)
        version = data.get("info", {}).get("version")
        return version if isinstance(version, str) else None
    except Exception:  # noqa: BLE001 - never let a check break startup
        return None


def _is_source_install(installed: Version) -> bool:
    """Whether this looks like an editable/source build (not a PyPI release).

    ``setuptools_scm`` stamps such builds with a local version segment (e.g.
    ``0.4.2.dev75+gb0c98f1``). We never nag developers running from source.
    """
    return installed.local is not None or installed.is_devrelease


def _upgrade_instructions() -> str:
    """Return upgrade instructions, ordered by the likely install method."""
    pip_line = "• pip:   pip install --upgrade napari-phasors"
    conda_line = "• conda: conda update -c conda-forge napari-phasors"
    napari_line = "• napari: Plugins ▸ Install/Uninstall Plugins… ▸ Update"
    # Inside a conda environment, surface the conda command first.
    if os.environ.get("CONDA_PREFIX"):
        lines = [conda_line, pip_line, napari_line]
    else:
        lines = [pip_line, conda_line, napari_line]
    return "\n".join(lines)


def _show_update_dialog(latest: str, installed: str, parent=None) -> None:
    """Show the (non-modal) update dialog with a "don't ask again" option."""
    from qtpy.QtWidgets import QCheckBox, QMessageBox

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Information)
    box.setWindowTitle("napari-phasors update available")
    box.setText(
        f"A new version of napari-phasors is available.\n\n"
        f"Installed: {installed}\nLatest: {latest}"
    )
    box.setInformativeText(
        "To update, use the method matching your installation:\n\n"
        f"{_upgrade_instructions()}\n\n"
        "(Restart napari after updating.)"
    )
    box.setStandardButtons(QMessageBox.Ok)

    dont_ask = QCheckBox("Don't notify me about updates again")
    box.setCheckBox(dont_ask)

    def _on_finished(_result):
        if dont_ask.isChecked():
            config = _load_config()
            config["disabled"] = True
            _save_config(config)

    box.finished.connect(_on_finished)
    # Non-modal so it never blocks the UI; keep a reference alive so it is not
    # garbage-collected before the user interacts with it.
    box.setModal(False)
    box.show()
    global _live_dialog
    _live_dialog = box


#: Keeps the non-modal dialog alive until dismissed.
_live_dialog = None


def _handle_latest_version(latest: str | None, installed: Version, parent):
    """Record the check and show the dialog if a newer release exists."""
    config = _load_config()
    config["last_check"] = time.time()
    _save_config(config)

    if latest is None:
        return
    try:
        if Version(latest) > installed:
            _show_update_dialog(latest, str(installed), parent)
    except InvalidVersion:
        pass


def maybe_check_for_update(parent=None) -> None:
    """Kick off an unobtrusive background check for a newer release.

    Safe to call from multiple widget constructors: it runs at most once per
    session and no more than once per :data:`CHECK_INTERVAL_SECONDS`. Skips
    entirely for source/dev installs and when the user has opted out.
    """
    global _checked_this_session
    if _checked_this_session:
        return
    _checked_this_session = True

    try:
        installed = Version(__installed_version__)
    except InvalidVersion:
        return

    if _is_source_install(installed):
        return

    config = _load_config()
    if config.get("disabled"):
        return

    last_check = config.get("last_check", 0)
    if (
        isinstance(last_check, (int, float))
        and time.time() - last_check < CHECK_INTERVAL_SECONDS
    ):
        return

    try:
        from napari.qt.threading import thread_worker
    except ImportError:
        return

    @thread_worker
    def _worker():
        return _fetch_latest_version()

    worker = _worker()
    worker.returned.connect(
        lambda latest: _handle_latest_version(latest, installed, parent)
    )
    worker.start()
