import numpy as np
import pooch
import pytest

from napari_phasors._sample_data import (
    DOWNLOAD_RETRIES,
    convallaria_FLIM_sample_data,
    embryo_FLIM_sample_data,
    fret_FLIM_sample_data,
    paramecium_HSI_sample_data,
)


@pytest.mark.parametrize(
    "sample_data_function",
    [
        convallaria_FLIM_sample_data,
        embryo_FLIM_sample_data,
        paramecium_HSI_sample_data,
        fret_FLIM_sample_data,
    ],
)
def test_sample_data_downloaders_retry_transient_failures(
    sample_data_function, monkeypatch
):
    """Every loader asks pooch to retry a failed download before giving up.

    Without ``retry_if_failed`` a single read timeout against GitHub/Zenodo
    aborts the whole sample-data command (and, on CI, the test).
    """
    created_kwargs = []
    real_create = pooch.create

    def spy_create(*args, **kwargs):
        created_kwargs.append(kwargs)
        return real_create(*args, **kwargs)

    class _StopHere(Exception):
        """Sentinel: stop before any actual download happens."""

    def no_download(self, *args, **kwargs):
        raise _StopHere

    monkeypatch.setattr(pooch, "create", spy_create)
    monkeypatch.setattr(pooch.Pooch, "fetch", no_download)

    with pytest.raises(_StopHere):
        sample_data_function()

    assert created_kwargs
    for kwargs in created_kwargs:
        assert kwargs["retry_if_failed"] == DOWNLOAD_RETRIES


def test_convallaria_FLIM_sample_data(make_viewer_model, qtbot):
    """Test the convallaria FLIM sample data"""
    layer_data_list = convallaria_FLIM_sample_data()
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 2
    # Convallaria image
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (256, 256)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"] == "Convallaria_$EI0S [Phasor] Intensity"
    )
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata
    # Check G and S are NumPy arrays with correct shape (n_harmonics, height, width)
    assert isinstance(metadata["G"], np.ndarray)
    assert isinstance(metadata["S"], np.ndarray)
    assert metadata["G"].shape == (2, 256, 256)
    assert metadata["S"].shape == (2, 256, 256)
    assert list(metadata["harmonics"]) == [1, 2]
    # Calibration
    layer_data_tuple = layer_data_list[1]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (256, 256)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"]
        == "Calibration_Rhodamine110_$EI0S [Phasor] Intensity"
    )
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata
    # Check G and S are NumPy arrays with correct shape (n_harmonics, height, width)
    assert isinstance(metadata["G"], np.ndarray)
    assert isinstance(metadata["S"], np.ndarray)
    assert metadata["G"].shape == (2, 256, 256)
    assert metadata["S"].shape == (2, 256, 256)
    assert list(metadata["harmonics"]) == [1, 2]


def test_embryo_FLIM_sample_data(make_viewer_model, qtbot):
    """Test the embryo FLIM sample data"""
    layer_data_list = embryo_FLIM_sample_data()
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 2
    # Embryo image
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (512, 512)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert layer_data_tuple[1]["name"] == "Embryo [Phasor] Intensity"
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata
    # Check G and S are NumPy arrays with correct shape (n_harmonics, height, width)
    assert isinstance(metadata["G"], np.ndarray)
    assert isinstance(metadata["S"], np.ndarray)
    assert metadata["G"].shape == (2, 512, 512)
    assert metadata["S"].shape == (2, 512, 512)
    assert list(metadata["harmonics"]) == [1, 2]
    # Calibration
    layer_data_tuple = layer_data_list[1]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (512, 512)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert (
        layer_data_tuple[1]["name"] == "Fluorescein_Embryo [Phasor] Intensity"
    )
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata
    # Check G and S are NumPy arrays with correct shape (n_harmonics, height, width)
    assert isinstance(metadata["G"], np.ndarray)
    assert isinstance(metadata["S"], np.ndarray)
    assert metadata["G"].shape == (2, 512, 512)
    assert metadata["S"].shape == (2, 512, 512)
    assert list(metadata["harmonics"]) == [1, 2]


def test_paramecium_HSI_sample_data(make_viewer_model, qtbot):
    """Test the paramecium HSI sample data"""
    layer_data_list = paramecium_HSI_sample_data()
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 1
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
    assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
        layer_data_tuple[1], dict
    )
    assert layer_data_tuple[0].shape == (512, 512)
    assert "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
    assert layer_data_tuple[1]["name"] == "paramecium [Phasor] Intensity"
    metadata = layer_data_tuple[1]["metadata"]
    assert "G" in metadata
    assert "S" in metadata
    assert "G_original" in metadata
    assert "S_original" in metadata
    assert "harmonics" in metadata
    assert "original_mean" in metadata
    assert "settings" in metadata
    # Check G and S are NumPy arrays with correct shape (n_harmonics, height, width)
    assert isinstance(metadata["G"], np.ndarray)
    assert isinstance(metadata["S"], np.ndarray)
    assert metadata["G"].shape == (2, 512, 512)
    assert metadata["S"].shape == (2, 512, 512)
    assert list(metadata["harmonics"]) == [1, 2]


def test_fret_FLIM_sample_data(make_viewer_model, qtbot):
    """Test the FLIM-FRET training sample data"""
    layer_data_list = fret_FLIM_sample_data()
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 4
    expected_names = [
        "Donor_Only [Phasor] Intensity",
        "Background_Autofluorescence [Phasor] Intensity",
        "FRET_Construct_1 [Phasor] Intensity",
        "FRET_Construct_2 [Phasor] Intensity",
    ]
    for layer_data_tuple, expected_name in zip(
        layer_data_list, expected_names, strict=True
    ):
        assert (
            isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 2
        )
        assert isinstance(layer_data_tuple[0], np.ndarray) and isinstance(
            layer_data_tuple[1], dict
        )
        assert layer_data_tuple[0].shape == (256, 256)
        assert (
            "name" in layer_data_tuple[1] and "metadata" in layer_data_tuple[1]
        )
        assert layer_data_tuple[1]["name"] == expected_name
        metadata = layer_data_tuple[1]["metadata"]
        assert "G" in metadata
        assert "S" in metadata
        assert "G_original" in metadata
        assert "S_original" in metadata
        assert "harmonics" in metadata
        assert "original_mean" in metadata
        assert "settings" in metadata
        # Check G and S are NumPy arrays with correct shape
        # (n_harmonics, height, width)
        assert isinstance(metadata["G"], np.ndarray)
        assert isinstance(metadata["S"], np.ndarray)
        assert metadata["G"].shape == (2, 256, 256)
        assert metadata["S"].shape == (2, 256, 256)
        assert list(metadata["harmonics"]) == [1, 2]
        # Files are exported already calibrated
        assert metadata["settings"]["calibrated"] is True
