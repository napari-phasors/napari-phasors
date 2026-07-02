"""Tests for the napari-phasors -> FLIMari interoperability bridge."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from napari.layers import Image

from napari_phasors._flimari import (
    FlimariNotAvailable,
    build_flimari_dataset,
    get_layer_histogram_bins,
    has_phasor_data,
    histogram_bins_from_raw_file,
    send_layers_to_flimari,
)
from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)

N_TIME_BINS = 256


def _phasor_layer(harmonic=None, name="FLIM data"):
    """Create an intensity layer with phasor metadata for testing."""
    raw = make_raw_flim_data(n_time_bins=N_TIME_BINS, shape=(4, 5))
    return make_intensity_layer_with_phasors(
        raw, harmonic=harmonic or [1, 2], name=name
    )


class _FakeBridge:
    """Stand-in for ``flimari.core.bridge`` capturing the received payload."""

    def __init__(self):
        self.received = None

    def import_from_napari_phasors(self, data_list):
        self.received = data_list


class _FlakyBridge:
    """Bridge stub that reports "not ready" once, then accepts data.

    Mimics FLIMari's real bridge, which raises ``RuntimeError`` from
    ``import_from_napari_phasors`` until its dock widget has been opened
    (which registers the import callback).
    """

    def __init__(self):
        self.calls = 0
        self.received = None

    def import_from_napari_phasors(self, data_list):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError(
                "FLIMari is not ready to receive data. "
                "Open the FLIMari dock widget first."
            )
        self.received = data_list


def test_build_recovers_photon_counts():
    """counts == round(mean * K), where K is the number of histogram bins."""
    layer = _phasor_layer()
    payload = build_flimari_dataset(layer)

    expected = np.rint(layer.metadata["original_mean"] * N_TIME_BINS)
    assert "counts" in payload
    np.testing.assert_allclose(payload["counts"], expected)
    # Counts must never be smaller than the mean intensity.
    assert np.all(payload["counts"] >= payload["mean"] - 1e-6)


def test_build_schema_keys_and_shapes():
    """The payload matches FLIMari's documented schema."""
    layer = _phasor_layer()
    payload = build_flimari_dataset(layer)

    for key in (
        "name",
        "channel",
        "g",
        "s",
        "g_original",
        "s_original",
        "mean",
        "min_count",
        "max_count",
        "filter_size",
        "filter_repeat",
    ):
        assert key in payload

    assert payload["name"] == "FLIM data Intensity Image"
    # Phasor arrays are [Harmonics, Y, X].
    assert payload["g"].shape == (2, 4, 5)
    assert payload["s"].shape == (2, 4, 5)
    assert payload["mean"].shape == (4, 5)
    # Identity calibration (napari-phasors already baked calibration into G).
    assert payload["calibration_phase"] == [0.0, 0.0]
    assert payload["calibration_modulation"] == [1.0, 1.0]


def test_build_caps_harmonics_at_two():
    """FLIMari requires harmonic [1, 2]; extra harmonics are dropped."""
    layer = _phasor_layer(harmonic=[1, 2, 3])
    payload = build_flimari_dataset(layer)

    assert payload["g"].shape[0] == 2
    assert payload["s"].shape[0] == 2
    assert len(payload["calibration_phase"]) == 2


def test_build_converts_thresholds_to_counts():
    """Mean-intensity thresholds are scaled to photon-count units by K."""
    layer = _phasor_layer()
    layer.metadata["settings"] = {
        "threshold": 3.0,
        "threshold_upper": 10.0,
        "frequency": 40.0,
        "channel": 1,
        "filter": {"size": 5, "repeat": 2},
    }

    payload = build_flimari_dataset(layer)

    assert payload["min_count"] == round(3.0 * N_TIME_BINS)
    assert payload["max_count"] == round(10.0 * N_TIME_BINS)
    assert payload["frequency"] == 40.0
    assert payload["channel"] == 1
    assert payload["filter_size"] == 5
    assert payload["filter_repeat"] == 2


def test_build_without_histogram_bins_uses_mean_thresholds():
    """No summed_signal: counts omitted, thresholds stay in mean units."""
    layer = _phasor_layer()
    del layer.metadata["summed_signal"]
    layer.metadata["settings"] = {"threshold": 4.0}

    payload = build_flimari_dataset(layer)

    assert "counts" not in payload
    assert payload["min_count"] == 4


def test_build_uses_histogram_bins_override():
    """An explicit K recovers counts even when the metadata lacks it."""
    layer = _phasor_layer()
    del layer.metadata["summed_signal"]
    assert get_layer_histogram_bins(layer) is None

    payload = build_flimari_dataset(layer, histogram_bins=100)

    assert "counts" in payload
    np.testing.assert_allclose(
        payload["counts"], np.rint(layer.metadata["original_mean"] * 100)
    )


def test_get_layer_histogram_bins():
    """get_layer_histogram_bins reads K from summed_signal, else None."""
    layer = _phasor_layer()
    assert get_layer_histogram_bins(layer) == N_TIME_BINS
    del layer.metadata["summed_signal"]
    assert get_layer_histogram_bins(layer) is None


def test_histogram_bins_from_raw_file(tmp_path):
    """K is recovered from an original raw file and validated against mean."""
    import tifffile

    raw = make_raw_flim_data(n_time_bins=64, shape=(4, 5))
    path = str(tmp_path / "raw.tif")
    tifffile.imwrite(path, raw)

    # A layer built from the same data has a matching mean-intensity image.
    layer = make_intensity_layer_with_phasors(raw)

    k = histogram_bins_from_raw_file(
        path, reference_mean=layer.metadata["original_mean"]
    )
    assert k == 64


def test_histogram_bins_from_raw_file_rejects_mismatched_file(tmp_path):
    """A file whose image shape/intensity doesn't match the layer is rejected."""
    import tifffile

    raw = make_raw_flim_data(n_time_bins=64, shape=(4, 5))
    path = str(tmp_path / "raw.tif")
    tifffile.imwrite(path, raw)

    with pytest.raises(ValueError):
        histogram_bins_from_raw_file(path, reference_mean=np.zeros((10, 10)))


def test_send_layers_uses_histogram_bins_override(monkeypatch):
    """Per-layer K overrides recover counts for layers lacking summed_signal."""
    import napari_phasors._flimari as flimari_mod

    fake = _FakeBridge()
    monkeypatch.setattr(flimari_mod, "_import_bridge", lambda: fake)

    layer = _phasor_layer()
    del layer.metadata["summed_signal"]

    sent, skipped, _ = send_layers_to_flimari(
        [layer], histogram_bins={layer: 128}
    )

    assert "counts" in sent[0]
    np.testing.assert_allclose(
        sent[0]["counts"], np.rint(layer.metadata["original_mean"] * 128)
    )


def test_build_returns_none_without_phasor_metadata():
    """A plain image layer (no phasor features) yields no payload."""
    layer = Image(np.zeros((4, 5)), name="plain")
    assert build_flimari_dataset(layer) is None


def test_has_phasor_data():
    """has_phasor_data distinguishes phasor layers from derived-analysis ones."""
    phasor_layer = _phasor_layer()
    assert has_phasor_data(phasor_layer) is True

    plain_layer = Image(np.zeros((4, 5)), name="plain")
    assert has_phasor_data(plain_layer) is False

    # A component-analysis fraction layer only carries
    # 'fraction_data_original', never raw phasor coordinates.
    component_layer = Image(
        np.zeros((4, 5)),
        name="Component 1 Fractions",
        metadata={"fraction_data_original": np.zeros((4, 5))},
    )
    assert has_phasor_data(component_layer) is False


def test_send_layers_calls_bridge(monkeypatch):
    """send_layers_to_flimari forwards built payloads to the bridge."""
    import napari_phasors._flimari as flimari_mod

    fake = _FakeBridge()
    monkeypatch.setattr(flimari_mod, "_import_bridge", lambda: fake)

    phasor = _phasor_layer(name="good")
    plain = Image(np.zeros((4, 5)), name="plain")

    sent, skipped, opened_dock = send_layers_to_flimari([phasor, plain])

    assert fake.received is sent
    assert len(sent) == 1
    assert skipped == ["plain"]
    assert opened_dock is False


def test_send_layers_skips_component_analysis_layers(monkeypatch):
    """Mixing a phasor layer with a component-analysis one only sends the former."""
    import napari_phasors._flimari as flimari_mod

    fake = _FakeBridge()
    monkeypatch.setattr(flimari_mod, "_import_bridge", lambda: fake)

    phasor = _phasor_layer(name="good")
    component_layer = Image(
        np.zeros((4, 5)),
        name="Component 1 Fractions",
        metadata={"fraction_data_original": np.zeros((4, 5))},
    )

    sent, skipped, _ = send_layers_to_flimari([phasor, component_layer])

    assert len(sent) == 1
    assert skipped == ["Component 1 Fractions"]


def test_send_layers_raises_when_no_phasor_layers(monkeypatch):
    """A ValueError is raised when nothing has phasor data to send."""
    import napari_phasors._flimari as flimari_mod

    monkeypatch.setattr(flimari_mod, "_import_bridge", lambda: _FakeBridge())
    plain = Image(np.zeros((4, 5)), name="plain")

    with pytest.raises(ValueError):
        send_layers_to_flimari([plain])


def test_send_layers_raises_when_flimari_missing():
    """When FLIMari is not installed, FlimariNotAvailable propagates."""
    # FLIMari is not a test dependency, so the real import must fail.
    with pytest.raises(FlimariNotAvailable):
        send_layers_to_flimari([_phasor_layer()])


def test_send_layers_opens_flimari_dock_when_not_ready(monkeypatch):
    """If FLIMari's dock isn't open, it is opened automatically and retried."""
    import napari_phasors._flimari as flimari_mod

    fake = _FlakyBridge()
    monkeypatch.setattr(flimari_mod, "_import_bridge", lambda: fake)

    mock_viewer = MagicMock()
    layer = _phasor_layer()

    sent, skipped, opened_dock = send_layers_to_flimari(
        [layer], viewer=mock_viewer
    )

    assert opened_dock is True
    assert fake.calls == 2
    assert fake.received is sent
    assert skipped == []
    mock_viewer.window.add_plugin_dock_widget.assert_called_once_with(
        plugin_name="flimari", widget_name="Phasor Analysis"
    )


def test_send_layers_without_viewer_reraises_runtime_error(monkeypatch):
    """Without a viewer to open the dock with, the original error propagates."""
    import napari_phasors._flimari as flimari_mod

    fake = _FlakyBridge()
    monkeypatch.setattr(flimari_mod, "_import_bridge", lambda: fake)

    with pytest.raises(RuntimeError):
        send_layers_to_flimari([_phasor_layer()])


def test_send_layers_raises_if_dock_cannot_be_opened(monkeypatch):
    """If opening the dock itself fails, a clear FlimariNotAvailable is raised."""
    import napari_phasors._flimari as flimari_mod

    fake = _FlakyBridge()
    monkeypatch.setattr(flimari_mod, "_import_bridge", lambda: fake)

    mock_viewer = MagicMock()
    mock_viewer.window.add_plugin_dock_widget.side_effect = KeyError("flimari")

    with pytest.raises(FlimariNotAvailable):
        send_layers_to_flimari([_phasor_layer()], viewer=mock_viewer)
