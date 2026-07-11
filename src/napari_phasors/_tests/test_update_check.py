import json
import time
from pathlib import Path

import pytest
from packaging.version import Version
from qtpy.QtWidgets import QMessageBox

from napari_phasors import _update_check as uc


class _FakeResponse:
    """Minimal stand-in for the object returned by ``urlopen``."""

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeSignal:
    """Minimal stand-in for a Qt signal: single-slot connect + sync emit."""

    def __init__(self):
        self._callback = None

    def connect(self, callback):
        self._callback = callback

    def emit(self, value):
        if self._callback is not None:
            self._callback(value)


class _FakeWorker:
    """Runs the wrapped function synchronously instead of on a thread."""

    def __init__(self, func):
        self._func = func
        self.returned = _FakeSignal()

    def start(self):
        self.returned.emit(self._func())


def _fake_thread_worker(func):
    def factory(*args, **kwargs):
        return _FakeWorker(lambda: func(*args, **kwargs))

    return factory


@pytest.fixture(autouse=True)
def _isolate_update_check(tmp_path, monkeypatch):
    """Isolate persisted config and session-guard state for every test."""
    monkeypatch.setattr(uc, "user_config_dir", lambda name: str(tmp_path))
    monkeypatch.setattr(uc, "_checked_this_session", False)
    monkeypatch.setattr(uc, "_live_dialog", None)
    yield


# --- _config_path -----------------------------------------------------


def test_config_path_uses_platformdirs(tmp_path):
    assert uc._config_path() == Path(str(tmp_path)) / "update_check.json"


# --- _load_config / _save_config ---------------------------------------


def test_load_config_missing_file_returns_empty_dict():
    assert uc._load_config() == {}


def test_save_and_load_config_round_trip():
    uc._save_config({"skipped_version": "1.2.3", "last_check": 1.0})
    assert uc._load_config() == {
        "skipped_version": "1.2.3",
        "last_check": 1.0,
    }


def test_load_config_invalid_json_returns_empty_dict():
    path = uc._config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json {{{")
    assert uc._load_config() == {}


def test_load_config_non_dict_json_returns_empty_dict():
    path = uc._config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]")
    assert uc._load_config() == {}


def test_save_config_swallows_oserror(monkeypatch):
    def boom(self, *args, **kwargs):
        raise OSError("cannot create directory")

    monkeypatch.setattr(Path, "mkdir", boom)
    uc._save_config({"a": 1})  # must not raise


# --- _fetch_latest_version ----------------------------------------------


def test_fetch_latest_version_success(monkeypatch):
    payload = {"info": {"version": "1.2.3"}}
    monkeypatch.setattr(
        uc.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(payload),
    )
    assert uc._fetch_latest_version() == "1.2.3"


def test_fetch_latest_version_returns_none_on_network_error(monkeypatch):
    def raise_error(request, timeout):
        raise OSError("network unreachable")

    monkeypatch.setattr(uc.urllib.request, "urlopen", raise_error)
    assert uc._fetch_latest_version() is None


def test_fetch_latest_version_returns_none_for_non_string_version(
    monkeypatch,
):
    payload = {"info": {"version": 123}}
    monkeypatch.setattr(
        uc.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(payload),
    )
    assert uc._fetch_latest_version() is None


# --- _is_source_install ---------------------------------------------------


def test_is_source_install_true_for_local_version_segment():
    assert uc._is_source_install(Version("0.4.2+gabcdef1")) is True


def test_is_source_install_true_for_devrelease():
    assert uc._is_source_install(Version("0.4.2.dev5")) is True


def test_is_source_install_false_for_plain_release():
    assert uc._is_source_install(Version("0.4.2")) is False


# --- _upgrade_instructions -------------------------------------------------


def test_upgrade_instructions_pip_first_outside_conda(monkeypatch):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    lines = uc._upgrade_instructions().splitlines()
    assert lines[0].startswith("• pip")
    assert lines[1].startswith("• conda")
    assert lines[2].startswith("• napari")


def test_upgrade_instructions_conda_first_inside_conda_env(monkeypatch):
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/napari-phasors")
    lines = uc._upgrade_instructions().splitlines()
    assert lines[0].startswith("• conda")
    assert lines[1].startswith("• pip")
    assert lines[2].startswith("• napari")


# --- _show_update_dialog ----------------------------------------------


def test_show_update_dialog_builds_expected_content(qtbot):
    uc._show_update_dialog("1.2.3", "1.0.0")
    box = uc._live_dialog
    assert box is not None
    # NOTE: box.windowTitle() is intentionally not asserted here: on macOS,
    # Qt ignores/clears QMessageBox window titles per platform HIG, so the
    # getter returns "" there regardless of what setWindowTitle() was given.
    assert "Installed: 1.0.0" in box.text()
    assert "Latest: 1.2.3" in box.text()
    assert box.checkBox() is not None
    assert box.checkBox().text() == "Don't show this message again"
    box.close()


def test_show_update_dialog_checkbox_checked_persists_skipped_version(
    qtbot,
):
    uc._show_update_dialog("2.0.0", "1.0.0")
    box = uc._live_dialog
    box.checkBox().setChecked(True)
    box.done(QMessageBox.Ok)
    assert uc._load_config().get("skipped_version") == "2.0.0"


def test_show_update_dialog_checkbox_unchecked_does_not_persist(qtbot):
    uc._show_update_dialog("2.0.0", "1.0.0")
    box = uc._live_dialog
    box.done(QMessageBox.Ok)
    assert "skipped_version" not in uc._load_config()


# --- _handle_latest_version -------------------------------------------


def test_handle_latest_version_records_last_check_when_latest_is_none():
    uc._handle_latest_version(None, Version("1.0.0"), None)
    assert "last_check" in uc._load_config()


def test_handle_latest_version_no_dialog_when_latest_is_none(monkeypatch):
    shown = []
    monkeypatch.setattr(
        uc, "_show_update_dialog", lambda *a, **k: shown.append(a)
    )
    uc._handle_latest_version(None, Version("1.0.0"), None)
    assert shown == []


def test_handle_latest_version_skips_dismissed_version(monkeypatch):
    uc._save_config({"skipped_version": "2.0.0"})
    shown = []
    monkeypatch.setattr(
        uc, "_show_update_dialog", lambda *a, **k: shown.append(a)
    )
    uc._handle_latest_version("2.0.0", Version("1.0.0"), None)
    assert shown == []


def test_handle_latest_version_no_dialog_when_not_newer(monkeypatch):
    shown = []
    monkeypatch.setattr(
        uc, "_show_update_dialog", lambda *a, **k: shown.append(a)
    )
    uc._handle_latest_version("1.0.0", Version("1.0.0"), None)
    assert shown == []


def test_handle_latest_version_shows_dialog_when_newer(monkeypatch):
    shown = []
    monkeypatch.setattr(
        uc, "_show_update_dialog", lambda *a, **k: shown.append(a)
    )
    parent = object()
    uc._handle_latest_version("2.0.0", Version("1.0.0"), parent)
    assert shown == [("2.0.0", "1.0.0", parent)]


def test_handle_latest_version_still_notifies_for_later_release(
    monkeypatch,
):
    """A version skipped once must not suppress a *later* release."""
    uc._save_config({"skipped_version": "2.0.0"})
    shown = []
    monkeypatch.setattr(
        uc, "_show_update_dialog", lambda *a, **k: shown.append(a)
    )
    uc._handle_latest_version("3.0.0", Version("1.0.0"), None)
    assert shown == [("3.0.0", "1.0.0", None)]


def test_handle_latest_version_handles_invalid_version_string(monkeypatch):
    shown = []
    monkeypatch.setattr(
        uc, "_show_update_dialog", lambda *a, **k: shown.append(a)
    )
    uc._handle_latest_version("not-a-version", Version("1.0.0"), None)
    assert shown == []


# --- maybe_check_for_update ------------------------------------------


def test_maybe_check_for_update_runs_once_per_session(monkeypatch):
    calls = []
    monkeypatch.setattr(
        uc, "_is_source_install", lambda v: calls.append(v) or True
    )
    uc.maybe_check_for_update()
    uc.maybe_check_for_update()
    assert len(calls) == 1


def test_maybe_check_for_update_returns_for_invalid_installed_version(
    monkeypatch,
):
    monkeypatch.setattr(uc, "__installed_version__", "not-a-version")
    called = []
    monkeypatch.setattr(
        uc, "_is_source_install", lambda v: called.append(v) or False
    )
    uc.maybe_check_for_update()
    assert called == []


def test_maybe_check_for_update_skips_source_install(monkeypatch):
    monkeypatch.setattr(uc, "__installed_version__", "0.4.2.dev1+gabcdef1")
    fetch_called = []
    monkeypatch.setattr(
        uc, "_fetch_latest_version", lambda: fetch_called.append(True)
    )
    uc.maybe_check_for_update()
    assert fetch_called == []


def test_maybe_check_for_update_throttled_by_recent_check(monkeypatch):
    monkeypatch.setattr(uc, "__installed_version__", "1.0.0")
    uc._save_config({"last_check": time.time()})
    fetch_called = []
    monkeypatch.setattr(
        uc, "_fetch_latest_version", lambda: fetch_called.append(True)
    )
    uc.maybe_check_for_update()
    assert fetch_called == []


def test_maybe_check_for_update_runs_after_stale_check(monkeypatch):
    monkeypatch.setattr(uc, "__installed_version__", "1.0.0")
    uc._save_config(
        {"last_check": time.time() - uc.CHECK_INTERVAL_SECONDS - 1}
    )
    monkeypatch.setattr(
        "napari.qt.threading.thread_worker", _fake_thread_worker
    )
    monkeypatch.setattr(uc, "_fetch_latest_version", lambda: "2.0.0")
    shown = []
    monkeypatch.setattr(
        uc, "_show_update_dialog", lambda *a, **k: shown.append(a)
    )
    uc.maybe_check_for_update(parent=None)
    assert shown == [("2.0.0", "1.0.0", None)]


def test_maybe_check_for_update_returns_when_thread_worker_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(uc, "__installed_version__", "1.0.0")
    import napari.qt.threading as napari_threading

    monkeypatch.delattr(napari_threading, "thread_worker")
    fetch_called = []
    monkeypatch.setattr(
        uc, "_fetch_latest_version", lambda: fetch_called.append(True)
    )
    uc.maybe_check_for_update()
    assert fetch_called == []


def test_maybe_check_for_update_end_to_end_shows_dialog(monkeypatch):
    """Full happy path through the real (fake-threaded) worker plumbing."""
    monkeypatch.setattr(uc, "__installed_version__", "1.0.0")
    monkeypatch.setattr(
        "napari.qt.threading.thread_worker", _fake_thread_worker
    )
    monkeypatch.setattr(uc, "_fetch_latest_version", lambda: "2.0.0")
    shown = []
    monkeypatch.setattr(
        uc, "_show_update_dialog", lambda *a, **k: shown.append(a)
    )

    parent = object()
    uc.maybe_check_for_update(parent=parent)

    assert shown == [("2.0.0", "1.0.0", parent)]
    assert "last_check" in uc._load_config()


def test_maybe_check_for_update_end_to_end_no_dialog_when_up_to_date(
    monkeypatch,
):
    monkeypatch.setattr(uc, "__installed_version__", "1.0.0")
    monkeypatch.setattr(
        "napari.qt.threading.thread_worker", _fake_thread_worker
    )
    monkeypatch.setattr(uc, "_fetch_latest_version", lambda: "1.0.0")
    shown = []
    monkeypatch.setattr(
        uc, "_show_update_dialog", lambda *a, **k: shown.append(a)
    )
    uc.maybe_check_for_update()
    assert shown == []


# --- widget wiring ------------------------------------------------------


def test_phasor_transform_triggers_update_check(
    monkeypatch, make_viewer_model
):
    from napari_phasors import _widget as widget_mod

    calls = []
    monkeypatch.setattr(
        widget_mod,
        "maybe_check_for_update",
        lambda parent=None: calls.append(parent),
    )
    viewer = make_viewer_model()
    widget = widget_mod.PhasorTransform(viewer)
    assert calls == [widget]


def test_plotter_widget_triggers_update_check(monkeypatch, make_viewer_model):
    from napari_phasors import plotter as plotter_mod

    calls = []
    monkeypatch.setattr(
        plotter_mod,
        "maybe_check_for_update",
        lambda parent=None: calls.append(parent),
    )
    viewer = make_viewer_model()
    widget = plotter_mod.PlotterWidget(viewer)
    assert calls == [widget]
