import json
import logging
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr
from phasorpy.datasets import fetch
from phasorpy.io import (
    phasor_from_ometiff,
    signal_from_fbd,
    signal_from_lsm,
    signal_from_sdt,
)
from qtpy.QtWidgets import QWidget

from napari_phasors._reader import napari_get_reader
from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors._tests.test_data_utils import get_test_file_path
from napari_phasors._utils import CollapsibleSection, natural_sort_key
from napari_phasors._widget import (
    AdvancedOptionsWidget,
    CziWidget,
    FbdWidget,
    LsmWidget,
    OmeTifWidget,
    PhasorTransform,
    PtuWidget,
    SdtWidget,
    WriterWidget,
)

TEST_FORMATS = [
    (".fbd", FbdWidget),
    (".ptu", PtuWidget),
    (".lsm", LsmWidget),
    (".sdt", SdtWidget),
    (".ome.tif", None),
]


def test_phasor_transform_widget(make_viewer_model, qtbot):
    """Test PhasorTransform widget behavior with mocked QFileDialog."""
    viewer = make_viewer_model()
    widget = PhasorTransform(viewer)

    assert widget.viewer is viewer
    assert isinstance(widget, QWidget)

    for extension, expected_widget_class in widget.reader_options.items():
        # Handle special cases for specific extensions
        if extension == ".fbd":
            test_file_path = get_test_file_path("test_file$EI0S.fbd")
        elif extension == ".sdt":
            test_file_path = get_test_file_path(
                "seminal_receptacle_FLIM_single_image.sdt"
            )
        elif extension == ".lsm":
            test_file_path = get_test_file_path("test_file.lsm")
        elif extension == ".ptu":
            test_file_path = get_test_file_path("test_file.ptu")
        elif extension == ".ome.tif":
            test_file_path = get_test_file_path("test_file.ome.tif")
        elif extension == ".czi":
            test_file_path = get_test_file_path("test_file.czi")
        else:
            continue

        with (
            patch(
                "napari_phasors._widget.QFileDialog.getOpenFileNames",
                return_value=([test_file_path], ""),
            ),
            patch("napari_phasors._widget.show_info"),
            patch("napari_phasors._widget.show_error"),
        ):

            # Simulate button click to open file dialog
            widget.search_button.click()

            # Verify the save path was updated
            assert widget.save_path.text() == test_file_path

            # Verify dynamic widget layout is updated
            if expected_widget_class:
                assert widget.dynamic_widget_layout.count() == 1
                added_widget = widget.dynamic_widget_layout.itemAt(0).widget()
                assert isinstance(added_widget, expected_widget_class)
            else:
                assert widget.dynamic_widget_layout.count() == 0


def test_phasor_transform_widget_multi_file_grouped_mode(
    make_viewer_model, qtbot
):
    """Test grouped mode creates collapsible widgets and a scrollable path list."""
    viewer = make_viewer_model()
    widget = PhasorTransform(viewer)

    file_paths = [
        get_test_file_path("test_file.ptu"),
        get_test_file_path("test_file.lsm"),
        get_test_file_path("test_file$EI0S.fbd"),
        get_test_file_path("seminal_receptacle_FLIM_single_image.sdt"),
    ]

    with (
        patch(
            "napari_phasors._widget.QFileDialog.getOpenFileNames",
            return_value=(file_paths, ""),
        ),
        patch("napari_phasors._widget.show_info"),
        patch("napari_phasors._widget.show_error"),
    ):
        widget.search_button.click()

    assert widget.save_path.isHidden()
    assert widget.selected_paths_list.isHidden() is False
    assert widget.selected_paths_list.count() == len(file_paths)
    assert [
        widget.selected_paths_list.item(i).text()
        for i in range(widget.selected_paths_list.count())
    ] == sorted(file_paths, key=natural_sort_key)

    # Four different file formats should create four collapsible groups plus a stretch at the bottom.
    assert widget.dynamic_widget_layout.count() == 5

    first_group = widget.dynamic_widget_layout.itemAt(0).widget()
    assert isinstance(first_group, CollapsibleSection)
    assert first_group._toggle_button.isChecked() is True
    assert first_group._content.isHidden() is False

    first_group._toggle_button.setChecked(False)
    first_group._on_toggle()
    assert first_group._content.isHidden() is True
    first_group._toggle_button.setChecked(True)
    first_group._on_toggle()
    assert first_group._content.isHidden() is False

    expected_groups = [
        ".fbd",
        ".lsm",
        ".ptu",
        ".sdt",
    ]
    expected_widget_types = [
        FbdWidget,
        LsmWidget,
        PtuWidget,
        SdtWidget,
    ]
    for index, (expected_group, expected_type) in enumerate(
        zip(expected_groups, expected_widget_types, strict=True)
    ):
        container = widget.dynamic_widget_layout.itemAt(index).widget()
        assert isinstance(container, CollapsibleSection)
        assert expected_group in container._toggle_button.text()

        group_widget = container._content.layout().itemAt(0).widget()
        assert isinstance(group_widget, expected_type)


def test_collapsible_group_container_toggle(make_viewer_model, qtbot):
    """Test collapsible group container hides and shows its content."""
    container = CollapsibleSection(
        title="Group .ptu (2 file(s))",
        initially_collapsed=False,
    )
    label = QWidget()
    container.add_widget(label)

    assert container._toggle_button.isChecked() is True
    assert container._content.isHidden() is False

    container._toggle_button.setChecked(False)
    container._on_toggle()
    assert container._content.isHidden() is True

    container._toggle_button.setChecked(True)
    container._on_toggle()
    assert container._content.isHidden() is False


def test_multi_file_preview_is_averaged(make_viewer_model, qtbot):
    """Test grouped preview signal averages over all selected files."""
    viewer = make_viewer_model()
    widget = LsmWidget(viewer, path=get_test_file_path("test_file.lsm"))

    file_paths = ["file_a.lsm", "file_b.lsm"]
    signals = {
        "file_a.lsm": np.array([1.0, 2.0, 3.0]),
        "file_b.lsm": np.array([3.0, 4.0, 5.0]),
    }
    widget._grouped_file_paths = file_paths
    # Also set multi paths to verify grouped mode takes precedence.
    widget._multi_file_paths = ["other_a.lsm", "other_b.lsm"]

    original_path = widget.path

    def fake_get_signal_data():
        return signals[widget.path]

    with patch.object(
        widget, "_get_signal_data", side_effect=fake_get_signal_data
    ):
        preview = widget._get_preview_signal_data()

    assert widget.path == original_path
    np.testing.assert_array_equal(preview, np.array([2.0, 3.0, 4.0]))


def test_phasor_transform_fbd_widget(make_viewer_model, qtbot):
    """Test FbdWidget from PhasorTransfrom widget."""
    viewer = make_viewer_model()
    PhasorTransform(viewer)
    test_file_path = get_test_file_path("test_file$EI0S.fbd")
    widget = FbdWidget(viewer, path=test_file_path)
    assert widget.viewer is viewer
    # Init values
    assert isinstance(widget, AdvancedOptionsWidget)
    assert widget.path == test_file_path
    assert widget.reader_options == {"frame": -1, "channel": None}
    assert widget.harmonics == [1, 2]
    assert widget.all_frames == 9
    assert widget.all_channels == 2
    assert widget.harmonic_slider.value() == (1, 2)
    # Modify harmonic values
    widget.harmonic_slider.setValue((2, 2))
    assert widget.harmonic_slider.value() == (2, 2)
    assert widget.harmonics == [2]
    widget.harmonic_slider.setValue((2, 3))
    assert widget.harmonic_slider.value() == (2, 3)
    assert widget.harmonics == [2, 3]
    # Init frames
    frames_combobox_values = [
        widget.frames.itemText(i) for i in range(widget.frames.count())
    ]
    assert frames_combobox_values == [
        "Average all frames",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
    ]
    assert widget.frames.currentIndex() == 0
    # Modify frames
    widget.frames.setCurrentIndex(1)
    assert widget.reader_options == {"frame": 0, "channel": None}
    # Init channels
    channels_combobox_values = [
        widget.channels.itemText(i) for i in range(widget.channels.count())
    ]
    assert channels_combobox_values == ["All channels", "0", "1"]
    assert widget.channels.currentIndex() == 0
    # Modify channels
    widget.channels.setCurrentIndex(1)
    assert widget.reader_options == {"frame": 0, "channel": 0}
    # Click button of phasor transform and check layers
    widget.btn.click()
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "test_file$EI0S Intensity Image"
    assert viewer.layers[0].data.shape == (256, 256)
    # Check phasor data in metadata
    assert "G" in viewer.layers[0].metadata
    assert "S" in viewer.layers[0].metadata
    assert "harmonics" in viewer.layers[0].metadata
    assert viewer.layers[0].metadata["G"].shape == (2, 256, 256)
    assert list(viewer.layers[0].metadata["harmonics"]) == [2, 3]
    # Modify channels and harmonics and phasor transform again
    widget.channels.setCurrentIndex(0)
    widget.harmonic_slider.setValue((2, 2))
    widget.btn.click()
    assert len(viewer.layers) == 3
    assert viewer.layers[2].name == "test_file$EI0S Intensity Image: Channel 1"
    assert viewer.layers[2].data.shape == (256, 256)
    assert viewer.layers[2].metadata["G"].shape == (1, 256, 256)
    assert list(viewer.layers[2].metadata["harmonics"]) == [2]
    # TODO: test laser factor parameter


def test_phasor_transform_ptu_widget(make_viewer_model, qtbot, caplog):
    """Test PtuWidget from PhasorTransfrom widget."""
    viewer = make_viewer_model()
    PhasorTransform(viewer)
    test_file_path = get_test_file_path("test_file.ptu")
    caplog.set_level(logging.ERROR, logger="ptufile")
    widget = PtuWidget(viewer, path=test_file_path)
    assert widget.viewer is viewer
    # Init values
    assert isinstance(widget, AdvancedOptionsWidget)
    assert widget.path == test_file_path
    assert widget.reader_options == {"frame": -1, "channel": None}
    assert widget.harmonics == [1, 2]
    assert widget.all_frames == 5
    assert widget.all_channels == 1
    assert widget.harmonic_slider.value() == (1, 2)
    # Modify harmonic values
    widget.harmonic_slider.setValue((2, 2))
    assert widget.harmonic_slider.value() == (2, 2)
    assert widget.harmonics == [2]
    widget.harmonic_slider.setValue((2, 3))
    assert widget.harmonic_slider.value() == (2, 3)
    assert widget.harmonics == [2, 3]
    # Init frames
    frames_combobox_values = [
        widget.frames.itemText(i) for i in range(widget.frames.count())
    ]
    assert frames_combobox_values == [
        "Average all frames",
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    assert widget.frames.currentIndex() == 0
    # Modify frames
    widget.frames.setCurrentIndex(1)
    assert widget.reader_options == {"frame": 0, "channel": None}
    # Init channels
    assert widget.channels is None
    assert widget.reader_options == {"frame": 0, "channel": None}
    # Click button of phasor transform and check layers
    widget.btn.click()
    assert not any(
        "tag with index not in tags" in record.message
        for record in caplog.records
    )
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "test_file Intensity Image: Channel 0"
    assert viewer.layers[0].data.shape == (256, 256)
    # Check phasor data in metadata
    assert "G" in viewer.layers[0].metadata
    assert "S" in viewer.layers[0].metadata
    assert "harmonics" in viewer.layers[0].metadata
    assert viewer.layers[0].metadata["G"].shape == (2, 256, 256)
    assert list(viewer.layers[0].metadata["harmonics"]) == [2, 3]
    # Modify frames and harmonics and phasor transform again
    widget.frames.setCurrentIndex(0)
    widget.harmonic_slider.setValue((2, 2))
    widget.btn.click()
    assert len(viewer.layers) == 2
    assert viewer.layers[1].name == "test_file Intensity Image: Channel 0 [1]"
    assert viewer.layers[1].data.shape == (256, 256)
    assert viewer.layers[1].metadata["G"].shape == (1, 256, 256)
    assert list(viewer.layers[1].metadata["harmonics"]) == [2]
    # TODO: test dtime parameter


def test_phasor_transform_sdt_widget(make_viewer_model, qtbot):
    """Test SdtWidget from PhasorTransfrom widget."""
    viewer = make_viewer_model()
    file_path = get_test_file_path("seminal_receptacle_FLIM_single_image.sdt")
    PhasorTransform(viewer)
    widget = SdtWidget(viewer, path=file_path)
    assert widget.viewer is viewer
    # Init values
    assert isinstance(widget, AdvancedOptionsWidget)
    assert widget.path == file_path
    assert widget.reader_options == {}
    assert widget.harmonics == [1, 2]
    assert widget.harmonic_slider.value() == (1, 2)
    # Modify harmonic values
    widget.harmonic_slider.setValue((2, 2))
    assert widget.harmonic_slider.value() == (2, 2)
    assert widget.harmonics == [2]
    widget.harmonic_slider.setValue((2, 3))
    assert widget.harmonic_slider.value() == (2, 3)
    assert widget.harmonics == [2, 3]
    # Init index parameter
    assert widget.index.text() == "0"
    # Click button of phasor transform and check layers
    widget.btn.click()
    assert widget.reader_options == {"index": 0}
    assert len(viewer.layers) == 1
    assert (
        viewer.layers[0].name
        == "seminal_receptacle_FLIM_single_image Intensity Image: Channel 0"
    )
    assert viewer.layers[0].data.shape == (512, 512)
    # Check phasor data in metadata
    assert "G" in viewer.layers[0].metadata
    assert "S" in viewer.layers[0].metadata
    assert "harmonics" in viewer.layers[0].metadata
    assert viewer.layers[0].metadata["G"].shape == (2, 512, 512)
    assert list(viewer.layers[0].metadata["harmonics"]) == [2, 3]
    # Modify harmonics and phasor transform again
    widget.harmonic_slider.setValue((2, 2))
    widget.btn.click()
    assert len(viewer.layers) == 2
    assert (
        viewer.layers[1].name
        == "seminal_receptacle_FLIM_single_image Intensity Image: Channel 0 [1]"
    )
    assert viewer.layers[1].data.shape == (512, 512)
    assert viewer.layers[1].metadata["G"].shape == (1, 512, 512)
    assert list(viewer.layers[1].metadata["harmonics"]) == [2]
    # TODO: test index parameter


def test_phasor_transform_lsm_widget(make_viewer_model, qtbot):
    """Test LsmWidget from PhasorTransfrom widget."""
    viewer = make_viewer_model()
    PhasorTransform(viewer)
    test_file_path = get_test_file_path("test_file.lsm")
    widget = LsmWidget(viewer, path=test_file_path)
    assert widget.viewer is viewer
    # Init values
    assert isinstance(widget, AdvancedOptionsWidget)
    assert widget.path == test_file_path
    assert widget.reader_options == {}
    assert widget.harmonics == [1, 2]
    assert widget.harmonic_slider.value() == (1, 2)
    # Modify harmonic values
    widget.harmonic_slider.setValue((2, 2))
    assert widget.harmonic_slider.value() == (2, 2)
    assert widget.harmonics == [2]
    widget.harmonic_slider.setValue((2, 3))
    assert widget.harmonic_slider.value() == (2, 3)
    assert widget.harmonics == [2, 3]
    # Click button of phasor transform and check layers
    widget.btn.click()
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "test_file Intensity Image"
    assert viewer.layers[0].data.shape == (512, 512)
    # Check phasor data in metadata
    assert "G" in viewer.layers[0].metadata
    assert "S" in viewer.layers[0].metadata
    assert "harmonics" in viewer.layers[0].metadata
    assert viewer.layers[0].metadata["G"].shape == (2, 512, 512)
    assert list(viewer.layers[0].metadata["harmonics"]) == [2, 3]
    # Modify harmonics and phasor transform again
    widget.harmonic_slider.setValue((2, 2))
    widget.btn.click()
    assert len(viewer.layers) == 2
    assert viewer.layers[1].name == "test_file Intensity Image [1]"
    assert viewer.layers[1].data.shape == (512, 512)
    assert viewer.layers[1].metadata["G"].shape == (1, 512, 512)
    assert list(viewer.layers[1].metadata["harmonics"]) == [2]


def test_lsm_widget_axis_selection_updates_signal_plot(
    make_viewer_model, qtbot, monkeypatch
):
    """Changing the axis combobox should update the selected axis and refresh the preview plot."""
    viewer = make_viewer_model()
    fake_signal = xr.DataArray(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        dims=("Z", "Y", "H"),
    )

    widget = LsmWidget(viewer, path=get_test_file_path("test_file.lsm"))
    monkeypatch.setattr(
        widget, "_get_preview_signal_data", lambda: fake_signal
    )
    widget._update_axis_options()
    widget._update_signal_plot()

    assert widget.axis_combo is not None
    assert widget.axis_combo.currentText() == "Auto"
    assert widget.reader_options.get("phasor_axis") is None
    assert widget.axis_combo.count() == 4

    initial_plot = widget.ax.get_lines()[0].get_ydata().copy()

    widget.axis_combo.setCurrentIndex(2)

    assert widget.reader_options["phasor_axis"] == 1
    updated_plot = widget.ax.get_lines()[0].get_ydata()
    expected_plot = fake_signal.values.sum(axis=(0, 2))

    np.testing.assert_array_equal(updated_plot, expected_plot)
    assert not np.array_equal(initial_plot, updated_plot)


def test_tiff_widget_axis_selection_updates_signal_plot(
    make_viewer_model, qtbot, monkeypatch
):
    """Changing the axis combobox for a TIFF file should update the selected axis and refresh the preview plot."""
    viewer = make_viewer_model()
    fake_signal = xr.DataArray(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        dims=("Z", "Y", "H"),
    )

    # LsmWidget handles both LSM and TIFF files; passing a TIFF extension (like "example.tif")
    # will set self._is_lsm = False.
    widget = LsmWidget(viewer, path="example.tif")
    monkeypatch.setattr(
        widget, "_get_preview_signal_data", lambda: fake_signal
    )
    widget._update_axis_options()
    widget._update_signal_plot()

    assert widget.axis_combo is not None
    assert widget.axis_combo.currentText() == "Auto"
    assert widget.reader_options.get("phasor_axis") is None
    assert widget.axis_combo.count() == 4

    initial_plot = widget.ax.get_lines()[0].get_ydata().copy()

    widget.axis_combo.setCurrentIndex(2)

    assert widget.reader_options["phasor_axis"] == 1
    updated_plot = widget.ax.get_lines()[0].get_ydata()
    expected_plot = fake_signal.values.sum(axis=(0, 2))

    np.testing.assert_array_equal(updated_plot, expected_plot)
    assert not np.array_equal(initial_plot, updated_plot)


def test_phasor_transform_czi_widget(make_viewer_model, qtbot):
    """Test CziWidget from PhasorTransform widget."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file.czi")
    widget = CziWidget(viewer, path=test_file_path)
    try:
        assert widget.viewer is viewer
        # Init values
        assert isinstance(widget, AdvancedOptionsWidget)
        assert widget.path == test_file_path
        assert widget.reader_options == {}
        assert widget.harmonics == [1, 2]
        assert widget.harmonic_slider.value() == (1, 2)
        # Harmonic range: (1, 14) for 28 channels
        assert widget.max_harmonic == 14
        assert widget.harmonic_slider.maximum() == 14

        # Modify harmonic values
        widget.harmonic_slider.setValue((2, 2))
        assert widget.harmonic_slider.value() == (2, 2)
        assert widget.harmonics == [2]

        # Click button of phasor transform and check layers
        widget.btn.click()
        assert len(viewer.layers) == 1
        assert viewer.layers[0].name == "test_file Intensity Image"
        # Shape after squeeze was (28, 512, 512), so spatial is (512, 512)
        assert viewer.layers[0].data.shape == (512, 512)
        # Check phasor data in metadata
        assert "G" in viewer.layers[0].metadata
        assert "S" in viewer.layers[0].metadata
        assert "harmonics" in viewer.layers[0].metadata
        # Harmonic 2 requested -> shape (1, 512, 512)
        assert viewer.layers[0].metadata["G"].shape == (1, 512, 512)
        assert list(viewer.layers[0].metadata["harmonics"]) == [2]

        # Modify harmonics and phasor transform again
        widget.harmonic_slider.setValue((2, 3))
        widget.btn.click()
        assert len(viewer.layers) == 2
        assert viewer.layers[1].name == "test_file Intensity Image [1]"
        assert viewer.layers[1].data.shape == (512, 512)
        assert viewer.layers[1].metadata["G"].shape == (2, 512, 512)
        assert list(viewer.layers[1].metadata["harmonics"]) == [2, 3]
    finally:
        widget.deleteLater()


def test_phasor_transform_ome_tif_widget(make_viewer_model, qtbot):
    """Test OmeTifWidget from PhasorTransform widget."""
    viewer = make_viewer_model()
    PhasorTransform(viewer)
    test_file_path = get_test_file_path("test_file.ome.tif")
    widget = OmeTifWidget(viewer, path=test_file_path)
    assert widget.viewer is viewer

    # Init values
    assert isinstance(widget, AdvancedOptionsWidget)
    assert widget.path == test_file_path
    assert widget.reader_options == {}
    assert widget.harmonics == [1, 2]
    assert widget.harmonic_slider.value() == (1, 2)

    # Test signal plot initialization
    assert widget.canvas is not None
    assert widget.ax is not None

    # Modify harmonic values
    widget.harmonic_slider.setValue((2, 2))
    assert widget.harmonic_slider.value() == (2, 2)
    assert widget.harmonics == [2]

    # Click button of phasor transform and check layers
    widget.btn.click()
    assert len(viewer.layers) == 1
    assert "Intensity Image" in viewer.layers[0].name
    # Check phasor data in metadata
    assert "G" in viewer.layers[0].metadata
    assert "S" in viewer.layers[0].metadata
    assert "harmonics" in viewer.layers[0].metadata
    assert list(viewer.layers[0].metadata["harmonics"]) == [2]


def test_phasor_transform_ome_tif_preserves_axis_kwargs(
    make_viewer_model, qtbot
):
    """OME-TIFF loaded from widget should preserve reader axis kwargs."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file.ome.tif")

    reader = napari_get_reader(test_file_path, harmonics=[1])
    assert reader is not None
    expected_layer_data = reader(test_file_path)[0]
    expected_data, expected_kwargs = expected_layer_data

    widget = OmeTifWidget(viewer, path=test_file_path)
    widget.harmonic_slider.setValue((1, 1))
    widget.btn.click()

    assert len(viewer.layers) == 1
    loaded_layer = viewer.layers[0]
    assert loaded_layer.data.shape == expected_data.shape

    if "axis_labels" in expected_kwargs:
        assert tuple(loaded_layer.axis_labels) == tuple(
            expected_kwargs["axis_labels"]
        )

    if "scale" in expected_kwargs:
        np.testing.assert_allclose(
            loaded_layer.scale, expected_kwargs["scale"]
        )


def test_fbd_laser_factor_triggers_plot_only_on_editing_finished(
    make_viewer_model,
    qtbot,
):
    """FbdWidget laser_factor should only update the signal plot on editingFinished, not on each keystroke."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file$EI0S.fbd")
    widget = FbdWidget(viewer, path=test_file_path)

    with patch.object(widget, "_update_signal_plot") as mock_update:
        # Simulate typing character by character (textChanged-like)
        widget.laser_factor.setText("0")
        widget.laser_factor.setText("0.")
        widget.laser_factor.setText("0.0")
        # None of the setText calls should have triggered a redraw
        mock_update.assert_not_called()

        # Finishing the edit should trigger exactly one redraw
        widget.laser_factor.editingFinished.emit()
        mock_update.assert_called_once()


def test_ptu_dtime_triggers_plot_only_on_editing_finished(
    make_viewer_model, qtbot, caplog
):
    """PtuWidget dtime should only update the signal plot on editingFinished, not on each keystroke."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file.ptu")
    caplog.set_level(logging.ERROR, logger="ptufile")
    widget = PtuWidget(viewer, path=test_file_path)

    with patch.object(widget, "_update_signal_plot") as mock_update:
        widget.dtime.setText("1")
        widget.dtime.setText("10")
        widget.dtime.setText("100")
        mock_update.assert_not_called()

        widget.dtime.editingFinished.emit()
        mock_update.assert_called_once()


def test_sdt_index_triggers_plot_only_on_editing_finished(
    make_viewer_model, qtbot
):
    """SdtWidget index should only update the signal plot on editingFinished, not on each keystroke."""
    viewer = make_viewer_model()
    file_path = get_test_file_path("seminal_receptacle_FLIM_single_image.sdt")
    widget = SdtWidget(viewer, path=file_path)

    with patch.object(widget, "_update_signal_plot") as mock_update:
        widget.index.setText("0")
        widget.index.setText("1")
        mock_update.assert_not_called()

        widget.index.editingFinished.emit()
        mock_update.assert_called_once()


def test_json_channel_triggers_plot_only_on_editing_finished(
    make_viewer_model,
    qtbot,
):
    """JsonWidget channel_entry should only update the signal plot on editingFinished, not on each keystroke."""
    from phasorpy.datasets import fetch

    from napari_phasors._widget import JsonWidget

    viewer = make_viewer_model()
    file_path = fetch("Fluorescein_Calibration_m2_1740751189_imaging.json")
    widget = JsonWidget(viewer, path=file_path)

    with patch.object(widget, "_update_signal_plot") as mock_update:
        widget.channel_entry.setText("0")
        widget.channel_entry.setText("1")
        mock_update.assert_not_called()

        widget.channel_entry.editingFinished.emit()
        mock_update.assert_called_once()


def test_harmonic_range_slider_functionality(make_viewer_model, qtbot):
    """Test QRangeSlider functionality for harmonics in all widget types."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file$EI0S.fbd")
    widget = FbdWidget(viewer, path=test_file_path)

    # Test initial slider values
    assert widget.harmonic_slider.value() == (1, 2)
    assert widget.harmonic_start_edit.text() == "1"
    assert widget.harmonic_end_edit.text() == str(2)

    # Test slider value change
    widget.harmonic_slider.setValue((2, 4))
    assert widget.harmonics == [2, 3, 4]
    assert widget.harmonic_start_edit.text() == "2"
    assert widget.harmonic_end_edit.text() == "4"

    # Test line edit changes
    widget.harmonic_start_edit.setText("3")
    widget.harmonic_start_edit.editingFinished.emit()
    assert widget.harmonic_slider.value()[0] == 3
    assert widget.harmonics == [3, 4]

    widget.harmonic_end_edit.setText("5")
    widget.harmonic_end_edit.editingFinished.emit()
    assert widget.harmonic_slider.value()[1] == 5
    assert widget.harmonics == [3, 4, 5]


def test_signal_plot_fbd_single_channel_all_frames(make_viewer_model, qtbot):
    """Test signal plot for FBD widget with single channel, all frames."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file$EI0S.fbd")
    widget = FbdWidget(viewer, path=test_file_path)

    # Set to single channel (channel 0), all frames
    widget.channels.setCurrentIndex(1)  # Channel 0
    widget.frames.setCurrentIndex(0)  # All frames

    # Get signal data for verification
    signal_data = signal_from_fbd(test_file_path, frame=-1, channel=0)
    # Summ over all axis except last one
    signal_data = signal_data.sum(axis=(0, 1))
    assert signal_data is not None

    # Update plot and verify it contains expected data
    widget._update_signal_plot()

    # Check that plot has data
    lines = widget.ax.get_lines()
    assert len(lines) > 0

    # Verify plot data matches expected signal
    plot_x_data = lines[0].get_xdata()
    plot_y_data = lines[0].get_ydata()

    # Expected data should match what _get_signal_data returns
    np.testing.assert_array_almost_equal(plot_y_data, signal_data)
    assert len(plot_x_data) == len(signal_data)


def test_signal_plot_fbd_single_channel_single_frame(make_viewer_model, qtbot):
    """Test signal plot for FBD widget with single channel, single frame."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file$EI0S.fbd")
    widget = FbdWidget(viewer, path=test_file_path)

    # Set to single channel (channel 1), single frame (frame 2)
    widget.channels.setCurrentIndex(2)  # Channel 1
    widget.frames.setCurrentIndex(3)  # Frame 2

    # Get signal data for verification
    signal_data = signal_from_fbd(test_file_path, frame=2, channel=1)
    signal_data = signal_data.sum(axis=(0, 1))
    assert signal_data is not None

    # Update plot and verify
    widget._update_signal_plot()

    lines = widget.ax.get_lines()
    assert len(lines) > 0

    plot_y_data = lines[0].get_ydata()
    np.testing.assert_array_almost_equal(plot_y_data, signal_data)


def test_signal_plot_fbd_all_channels_single_frame(make_viewer_model, qtbot):
    """Test signal plot for FBD widget with all channels, single frame."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file$EI0S.fbd")
    widget = FbdWidget(viewer, path=test_file_path)

    # Set to all channels, single frame (frame 1)
    widget.channels.setCurrentIndex(0)  # All channels
    widget.frames.setCurrentIndex(2)  # Frame 1

    # Get signal data for verification
    signal_data_channel_0 = signal_from_fbd(test_file_path, frame=1, channel=0)
    signal_data_channel_1 = signal_from_fbd(test_file_path, frame=1, channel=1)

    assert signal_data_channel_0 is not None
    assert signal_data_channel_1 is not None

    signal_data = np.array(
        [
            signal_data_channel_0.sum(axis=(0, 1)),
            signal_data_channel_1.sum(axis=(0, 1)),
        ]
    )

    # Update plot
    widget._update_signal_plot()

    lines = widget.ax.get_lines()
    # Should have multiple lines for multiple channels
    assert len(lines) == widget.all_channels

    # Verify each line has correct data
    for i, line in enumerate(lines):
        plot_y_data = line.get_ydata()
        # Signal data should be averaged across channels or individual channel data
        np.testing.assert_array_almost_equal(plot_y_data, signal_data[i])


def test_signal_plot_sdt_widget(make_viewer_model, qtbot):
    """Test signal plot for SDT widget."""
    viewer = make_viewer_model()
    file_path = get_test_file_path("seminal_receptacle_FLIM_single_image.sdt")
    widget = SdtWidget(viewer, path=file_path)

    # Get signal data and verify plot
    signal_data = signal_from_sdt(file_path)
    signal_data = signal_data.sum(axis=(0, 1))
    assert signal_data is not None

    widget._update_signal_plot()

    lines = widget.ax.get_lines()
    assert len(lines) > 0

    plot_y_data = lines[0].get_ydata()
    np.testing.assert_array_almost_equal(plot_y_data, signal_data)


def test_signal_plot_ometif_widget(make_viewer_model, qtbot):
    """Test signal plot for OME-TIF widget."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file.ome.tif")
    widget = OmeTifWidget(viewer, path=test_file_path)

    signal_data = None

    # Use phasorpy to read the OME-TIFF metadata
    _, _, _, attrs = phasor_from_ometiff(test_file_path, harmonic='all')

    # Get harmonics from attrs
    if "harmonic" in attrs:
        harmonics = attrs["harmonic"]
    if "description" in attrs:
        description = json.loads(attrs["description"])
        if sys.getsizeof(description) > 512 * 512:  # Threshold: 256 KB
            raise ValueError("Description dictionary is too large.")
        if "napari_phasors_settings" in description:
            settings = json.loads(description["napari_phasors_settings"])

            # Check if we have summed_signal data
            if 'summed_signal' in settings:
                signal_data = (np.array(settings['summed_signal']),)

    assert signal_data is not None

    widget._update_signal_plot()

    lines = widget.ax.get_lines()
    assert len(lines) > 0

    plot_y_data = lines[0].get_ydata()
    np.testing.assert_array_almost_equal(plot_y_data, signal_data[0])

    # Assert max number of harmonics is correct
    assert widget.harmonic_slider.maximum() == np.max(harmonics)


def test_signal_plot_lsm_widget(make_viewer_model, qtbot):
    """Test signal plot for LSM widget."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file.lsm")
    widget = LsmWidget(viewer, path=test_file_path)

    # Get signal data and verify plot
    signal_data = signal_from_lsm(test_file_path)
    signal_data = signal_data.sum(axis=(1, 2))
    assert signal_data is not None

    widget._update_signal_plot()

    lines = widget.ax.get_lines()
    assert len(lines) > 0

    plot_y_data = lines[0].get_ydata()
    np.testing.assert_array_almost_equal(plot_y_data, signal_data)


def test_signal_plot_error_handling(make_viewer_model, qtbot):
    """Test signal plot error handling when data cannot be loaded."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file$EI0S.fbd")
    widget = FbdWidget(viewer, path=test_file_path)

    # Mock _get_signal_data to raise an exception
    with (
        patch.object(
            widget, '_get_signal_data', side_effect=Exception("Test error")
        ),
        patch("napari_phasors._widget.show_error"),
    ):
        # Should not raise; the plot must end up with no data lines.
        widget._update_signal_plot()
        assert len(widget.ax.get_lines()) == 0


def test_phasor_transform_with_ome_tif_reader_option(make_viewer_model, qtbot):
    """Test PhasorTransform widget includes OmeTifWidget in reader options."""
    viewer = make_viewer_model()
    widget = PhasorTransform(viewer)

    try:
        # Verify OME-TIF reader option is included
        assert ".ome.tif" in widget.reader_options

        # Test with OME-TIF file
        test_file_path = get_test_file_path("test_file.ome.tif")

        with patch(
            "napari_phasors._widget.QFileDialog.getOpenFileNames",
            return_value=([test_file_path], ""),
        ):
            widget.search_button.click()

            # Verify OmeTifWidget was added
            assert widget.dynamic_widget_layout.count() == 1
            added_widget = widget.dynamic_widget_layout.itemAt(0).widget()
            assert isinstance(added_widget, OmeTifWidget)
    finally:
        widget.deleteLater()


def test_signal_plot_canvas_properties(make_viewer_model, qtbot):
    """Test signal plot canvas has correct properties."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file$EI0S.fbd")
    widget = FbdWidget(viewer, path=test_file_path)

    # Test canvas properties
    assert widget.canvas.height() == 300
    assert widget.figure.patch.get_alpha() == 0.0
    assert widget.ax.get_facecolor() == (0.0, 0.0, 0.0, 0.0)

    # Test spine colors are grey (matplotlib converts 'grey' to RGBA)
    for spine in widget.ax.spines.values():
        assert spine.get_edgecolor() == (
            0.5019607843137255,
            0.5019607843137255,
            0.5019607843137255,
            1.0,
        )

    # Test tick and label colors are grey (these return string values)
    assert widget.ax.xaxis.label.get_color() == 'grey'
    assert widget.ax.yaxis.label.get_color() == 'grey'
    assert widget.ax.title.get_color() == 'grey'


def test_signal_plot_data_consistency_across_widgets(make_viewer_model, qtbot):
    """Test signal plot data consistency across different widget types."""
    viewer = make_viewer_model()

    # Test each widget type has consistent signal data format
    test_files = [
        ("test_file$EI0S.fbd", FbdWidget),
        ("test_file.ptu", PtuWidget),
        ("seminal_receptacle_FLIM_single_image.sdt", SdtWidget),
        ("test_file.lsm", LsmWidget),
    ]

    for filename, widget_class in test_files:
        test_file_path = get_test_file_path(filename)
        widget = widget_class(viewer, path=test_file_path)

        signal_data = widget._get_signal_data()
        assert signal_data is not None

        # Handle both numpy arrays and xarray DataArrays
        if hasattr(signal_data, 'values'):  # xarray.DataArray
            signal_array = signal_data.values
        else:  # numpy.ndarray
            signal_array = signal_data

        assert isinstance(signal_array, np.ndarray)
        assert len(signal_array) > 0
        assert not np.isnan(signal_array).all()


def test_phasor_transform_flif_widget(make_viewer_model, qtbot):
    viewer = make_viewer_model()
    file_path = fetch("flimfast.flif")
    PhasorTransform(viewer)
    from napari_phasors._widget import FlifWidget

    widget = FlifWidget(viewer, path=file_path)
    widget.btn.click()
    assert len(viewer.layers) > 0


def test_phasor_transform_bh_widget(make_viewer_model, qtbot):
    viewer = make_viewer_model()
    file_path = fetch("simfcs.b&h")
    PhasorTransform(viewer)
    from napari_phasors._widget import BhWidget

    widget = BhWidget(viewer, path=file_path)
    widget.btn.click()
    assert len(viewer.layers) > 0


def test_phasor_transform_bhz_widget(make_viewer_model, qtbot):
    viewer = make_viewer_model()
    file_path = fetch("simfcs.bhz")
    PhasorTransform(viewer)
    from napari_phasors._widget import BhWidget

    widget = BhWidget(viewer, path=file_path)
    widget.btn.click()
    assert len(viewer.layers) > 0


def test_phasor_transform_json_widget(make_viewer_model, qtbot):
    viewer = make_viewer_model()
    file_path = fetch("Fluorescein_Calibration_m2_1740751189_imaging.json")
    PhasorTransform(viewer)
    from napari_phasors._widget import JsonWidget

    widget = JsonWidget(viewer, path=file_path)
    widget.btn.click()
    assert len(viewer.layers) > 0


def test_phasor_transform_simfcs_widget(make_viewer_model, qtbot):
    viewer = make_viewer_model()
    file_path = fetch("simfcs.r64")
    PhasorTransform(viewer)
    from napari_phasors._widget import SimfcsWidget

    widget = SimfcsWidget(viewer, path=file_path)
    widget.btn.click()
    assert len(viewer.layers) > 0


def test_writer_widget(make_viewer_model, qtbot, tmp_path):
    """Test the WriterWidget class."""
    # Intialize viewer and add intensity image layer with phasors data
    viewer = make_viewer_model()
    main_widget = WriterWidget(viewer)
    assert main_widget.viewer is viewer
    assert isinstance(main_widget, QWidget)
    # Check init values are empty
    assert main_widget.export_layer_combobox.count() == 0
    # FLIMari export button is disabled while no layer is selected
    assert not main_widget.flimari_button.isEnabled()
    # Check error messages if there are no phasor layers
    with patch("napari_phasors._widget.show_error") as mock_show_error:
        main_widget.search_button.click()
        mock_show_error.assert_called_once_with("No layer selected")
    # Create a synthetic FLIM data and an intensity image layer with phasors
    raw_flim_data = make_raw_flim_data()
    harmonic = [1, 2, 3]
    sample_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )
    viewer.add_layer(sample_image_layer)
    # Assert combobox is populated and file name is set
    assert (
        main_widget.export_layer_combobox.itemText(0)
        == sample_image_layer.name
    )
    # Still disabled: layer is present but not checked yet
    assert not main_widget.flimari_button.isEnabled()
    # Select the layer in the CheckableComboBox
    main_widget.export_layer_combobox.selectAll()
    assert main_widget.export_layer_combobox.checkedItems() == [
        sample_image_layer.name
    ]
    # FLIMari export button becomes enabled once a layer is checked
    assert main_widget.flimari_button.isEnabled()
    main_widget.export_layer_combobox.deselectAll()
    assert not main_widget.flimari_button.isEnabled()
    main_widget.export_layer_combobox.selectAll()

    # Simulate saving as OME-TIFF
    with (
        patch(
            "napari_phasors._widget.QFileDialog.getSaveFileName",
            return_value=(
                str(tmp_path / "test.ome.tif"),
                "Phasor as OME-TIFF (*.ome.tif)",
            ),
        ),
        patch("napari_phasors._widget.show_info") as mock_show_info,
    ):

        # Simulate button click
        main_widget.search_button.click()

        # Verify file saving logic
        export_layer_name = sample_image_layer.name
        export_path = str(tmp_path / "test.ome.tif")
        mock_show_info.assert_called_once_with(
            f"Exported {export_layer_name} to {export_path}"
        )

        # Check if the file was created and has expected data when read
        assert os.path.exists(export_path)
        reader = napari_get_reader(export_path, harmonics=harmonic)
        layer_data_list = reader(export_path)
        layer_data_tuple = layer_data_list[0]
        assert len(layer_data_tuple) == 2
        np.testing.assert_array_almost_equal(
            layer_data_tuple[0], sample_image_layer.data
        )
        # Check phasor data in metadata
        metadata = layer_data_tuple[1]["metadata"]
        assert "G" in metadata
        assert "S" in metadata
        assert "harmonics" in metadata
        assert list(metadata["harmonics"]) == [1, 2, 3]

    # Simulate saving as CSV
    with (
        patch(
            "napari_phasors._widget.QFileDialog.getSaveFileName",
            return_value=(
                str(tmp_path / "test.csv"),
                "Layer data as CSV (*.csv)",
            ),
        ),
        patch("napari_phasors._widget.show_info") as mock_show_info,
    ):

        # Simulate button click
        main_widget.search_button.click()

        # Verify CSV export logic
        export_layer_name = sample_image_layer.name
        export_path = str(tmp_path / "test.csv")
        mock_show_info.assert_called_once_with(
            f"Exported {export_layer_name} to {export_path}"
        )

        # Check if the file was created and has expected data when read
        assert os.path.exists(export_path)
        exported_table = pd.read_csv(export_path)
        # Check the CSV has the expected structure for phasor data
        assert len(exported_table) > 0
        # The CSV should contain coordinate columns and phasor values
        assert (
            'G' in exported_table.columns or 'value' in exported_table.columns
        )


def test_writer_widget_image_exports(make_viewer_model, qtbot, tmp_path):
    """Test image export functionality in WriterWidget."""
    viewer = make_viewer_model()
    main_widget = WriterWidget(viewer)

    # Create synthetic data
    raw_flim_data = make_raw_flim_data()
    harmonic = [1]
    sample_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )
    viewer.add_layer(sample_image_layer)
    main_widget.export_layer_combobox.selectAll()

    # Test PNG export with colorbar
    with (
        patch(
            "napari_phasors._widget.QFileDialog.getSaveFileName",
            return_value=(
                str(tmp_path / "test_with_colorbar.png"),
                "Layer as PNG image (*.png)",
            ),
        ),
        patch("napari_phasors._widget.show_info") as mock_show_info,
    ):
        main_widget.colorbar_checkbox.setChecked(True)
        main_widget.search_button.click()

        export_path = str(tmp_path / "test_with_colorbar.png")
        assert os.path.exists(export_path)
        mock_show_info.assert_called_with(
            f"Exported {sample_image_layer.name} to {export_path}"
        )

    # Test PNG export without colorbar
    with (
        patch(
            "napari_phasors._widget.QFileDialog.getSaveFileName",
            return_value=(
                str(tmp_path / "test_no_colorbar.png"),
                "Layer as PNG image (*.png)",
            ),
        ),
        patch("napari_phasors._widget.show_info") as mock_show_info,
    ):
        main_widget.colorbar_checkbox.setChecked(False)
        main_widget.search_button.click()

        export_path = str(tmp_path / "test_no_colorbar.png")
        assert os.path.exists(export_path)
        mock_show_info.assert_called_with(
            f"Exported {sample_image_layer.name} to {export_path}"
        )

    # Test JPG export
    with (
        patch(
            "napari_phasors._widget.QFileDialog.getSaveFileName",
            return_value=(
                str(tmp_path / "test.jpg"),
                "Layer as JPEG image (*.jpg)",
            ),
        ),
        patch("napari_phasors._widget.show_info") as mock_show_info,
    ):
        main_widget.colorbar_checkbox.setChecked(True)
        main_widget.search_button.click()

        export_path = str(tmp_path / "test.jpg")
        assert os.path.exists(export_path)
        mock_show_info.assert_called_with(
            f"Exported {sample_image_layer.name} to {export_path}"
        )

    # Test TIFF image export (not OME-TIFF)
    with (
        patch(
            "napari_phasors._widget.QFileDialog.getSaveFileName",
            return_value=(
                str(tmp_path / "test_image.tif"),
                "Layer as TIFF image (*.tif)",
            ),
        ),
        patch("napari_phasors._widget.show_info") as mock_show_info,
    ):
        main_widget.colorbar_checkbox.setChecked(False)
        main_widget.search_button.click()

        export_path = str(tmp_path / "test_image.tif")
        assert os.path.exists(export_path)
        mock_show_info.assert_called_with(
            f"Exported {sample_image_layer.name} to {export_path}"
        )


def test_writer_widget_colormap_applied(make_viewer_model, qtbot, tmp_path):
    """Test that the napari layer's colormap is correctly applied to exported images."""
    from PIL import Image

    viewer = make_viewer_model()
    main_widget = WriterWidget(viewer)

    # Create synthetic data
    raw_flim_data = make_raw_flim_data()
    harmonic = [1]
    sample_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )

    # Set a specific colormap
    sample_image_layer.colormap = 'viridis'
    viewer.add_layer(sample_image_layer)
    main_widget.export_layer_combobox.selectAll()

    # Export as PNG
    with patch(
        "napari_phasors._widget.QFileDialog.getSaveFileName",
        return_value=(
            str(tmp_path / "test_colormap.png"),
            "Layer as PNG image (*.png)",
        ),
    ):
        main_widget.colorbar_checkbox.setChecked(False)
        main_widget.search_button.click()

        export_path = str(tmp_path / "test_colormap.png")
        assert os.path.exists(export_path)

        # Verify the image was created and is not empty
        with Image.open(export_path) as img:
            assert img.size[0] > 0
            assert img.size[1] > 0


def test_writer_widget_file_extension_handling(
    make_viewer_model, qtbot, tmp_path
):
    """Test that file extensions are correctly handled for all export formats."""
    viewer = make_viewer_model()
    main_widget = WriterWidget(viewer)

    # Create synthetic data
    raw_flim_data = make_raw_flim_data()
    harmonic = [1]
    sample_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )
    viewer.add_layer(sample_image_layer)
    main_widget.export_layer_combobox.selectAll()

    # Test cases: (input_name, selected_filter, expected_output_name)
    test_cases = [
        ("test", "Phasor as OME-TIFF (*.ome.tif)", "test.ome.tif"),
        ("test.tif", "Phasor as OME-TIFF (*.ome.tif)", "test.ome.tif"),
        ("test", "Layer data as CSV (*.csv)", "test.csv"),
        ("test.csv", "Layer data as CSV (*.csv)", "test.csv"),
        ("test", "Layer as PNG image (*.png)", "test.png"),
        ("test.png", "Layer as PNG image (*.png)", "test.png"),
        ("test", "Layer as JPEG image (*.jpg)", "test.jpg"),
        ("test.jpg", "Layer as JPEG image (*.jpg)", "test.jpg"),
        ("test", "Layer as TIFF image (*.tif)", "test.tif"),
        ("test.ome.tif", "Layer as TIFF image (*.tif)", "test.tif"),
    ]

    for input_name, selected_filter, expected_output in test_cases:
        with patch(
            "napari_phasors._widget.QFileDialog.getSaveFileName",
            return_value=(
                str(tmp_path / input_name),
                selected_filter,
            ),
        ):
            main_widget.search_button.click()

            export_path = str(tmp_path / expected_output)
            assert os.path.exists(
                export_path
            ), f"Failed for {input_name} -> {expected_output}"


def test_writer_widget_colorbar_checkbox_state(make_viewer_model, qtbot):
    """Test that the colorbar checkbox is properly initialized and responsive."""
    viewer = make_viewer_model()
    main_widget = WriterWidget(viewer)

    # Check initial state
    assert main_widget.colorbar_checkbox.isChecked() is True
    assert (
        main_widget.colorbar_checkbox.text()
        == "Include colorbar (for image exports)"
    )

    # Test checkbox state changes
    main_widget.colorbar_checkbox.setChecked(False)
    assert main_widget.colorbar_checkbox.isChecked() is False

    main_widget.colorbar_checkbox.setChecked(True)
    assert main_widget.colorbar_checkbox.isChecked() is True


def test_writer_widget_csv_export_2d_no_phasor(
    make_viewer_model, qtbot, tmp_path
):
    """Test CSV export for 2D image without phasor metadata."""
    viewer = make_viewer_model()
    widget = WriterWidget(viewer)

    # Create a simple 2D image layer without phasor metadata
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    viewer.add_image(data, name="test_2d_image")

    # Export as CSV
    csv_path = tmp_path / "test_2d.csv"
    widget._save_file(
        str(csv_path),
        "Layer data as CSV (*.csv)",
        False,
        selected_layers=["test_2d_image"],
    )

    # Verify CSV content
    df = pd.read_csv(csv_path)

    # Check columns
    assert list(df.columns) == ['y', 'x', 'value']

    # Check shape
    assert len(df) == 9  # 3x3 = 9 pixels

    # Check coordinates and values
    expected_y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    expected_x = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    expected_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    np.testing.assert_array_equal(df['y'].values, expected_y)
    np.testing.assert_array_equal(df['x'].values, expected_x)
    np.testing.assert_array_equal(df['value'].values, expected_values)


def test_writer_widget_csv_export_4d_no_phasor(
    make_viewer_model, qtbot, tmp_path
):
    """Test CSV export for 4D image without phasor metadata."""
    viewer = make_viewer_model()
    widget = WriterWidget(viewer)

    # Create a 4D image layer
    data = np.arange(48).reshape(2, 2, 3, 4)
    viewer.add_image(data, name="test_4d_image")

    # Export as CSV
    csv_path = tmp_path / "test_4d.csv"
    widget._save_file(
        str(csv_path),
        "Layer data as CSV (*.csv)",
        False,
        selected_layers=["test_4d_image"],
    )

    # Verify CSV content
    df = pd.read_csv(csv_path)

    # Check columns
    assert list(df.columns) == ['dim_0', 'dim_1', 'dim_2', 'dim_3', 'value']

    # Check shape
    assert len(df) == 48

    # Verify coordinate ranges
    assert df['dim_0'].min() == 0 and df['dim_0'].max() == 1
    assert df['dim_1'].min() == 0 and df['dim_1'].max() == 1
    assert df['dim_2'].min() == 0 and df['dim_2'].max() == 2
    assert df['dim_3'].min() == 0 and df['dim_3'].max() == 3


def test_writer_widget_csv_coordinates_consistency_2d(
    make_viewer_model, qtbot, tmp_path
):
    """Test that CSV export maintains coordinate consistency for 2D images."""
    viewer = make_viewer_model()
    widget = WriterWidget(viewer)

    # Create 2D image with known pattern
    data = np.array([[10, 20], [30, 40]])
    viewer.add_image(data, name="test_pattern")

    # Export as CSV
    csv_path = tmp_path / "test_pattern.csv"
    widget._save_file(
        str(csv_path),
        "Layer data as CSV (*.csv)",
        False,
        selected_layers=["test_pattern"],
    )

    # Verify specific coordinate-value mappings
    df = pd.read_csv(csv_path)

    # Check (0,0) -> 10
    val_00 = df[(df['y'] == 0) & (df['x'] == 0)]['value'].values[0]
    assert val_00 == 10

    # Check (0,1) -> 20
    val_01 = df[(df['y'] == 0) & (df['x'] == 1)]['value'].values[0]
    assert val_01 == 20

    # Check (1,0) -> 30
    val_10 = df[(df['y'] == 1) & (df['x'] == 0)]['value'].values[0]
    assert val_10 == 30

    # Check (1,1) -> 40
    val_11 = df[(df['y'] == 1) & (df['x'] == 1)]['value'].values[0]
    assert val_11 == 40


def test_writer_widget_excludes_labels_layer(make_viewer_model, qtbot):
    """Test that WriterWidget excludes Labels layers from populate combobox."""
    viewer = make_viewer_model()
    viewer.add_image(np.random.random((10, 10)), name="my_image")
    viewer.add_labels(np.zeros((10, 10), dtype=int), name="my_labels")

    widget = WriterWidget(viewer)

    items = [
        widget.export_layer_combobox.itemText(i)
        for i in range(widget.export_layer_combobox.count())
    ]
    assert "my_image" in items
    assert "my_labels" not in items


def test_writer_widget_flimari_button_requires_phasor_layer(
    make_viewer_model, qtbot
):
    """FLIMari button stays disabled unless a *phasor* layer is checked.

    Layers from other analyses (e.g. component analysis fractions) are
    valid targets for OME-TIFF/CSV export but must not enable the FLIMari
    export button, since they carry no phasor coordinates to send.
    """
    viewer = make_viewer_model()
    raw_flim_data = make_raw_flim_data()
    phasor_layer = make_intensity_layer_with_phasors(raw_flim_data)
    viewer.add_layer(phasor_layer)
    # Simulate a component-analysis fraction layer: an Image layer with
    # unrelated metadata and no phasor coordinates.
    viewer.add_image(
        np.zeros((5, 5)),
        name="Component 1 Fractions",
        metadata={"fraction_data_original": np.zeros((5, 5))},
    )

    widget = WriterWidget(viewer)

    # Only the non-phasor layer selected: button must stay disabled.
    widget.export_layer_combobox.setCheckedItems(["Component 1 Fractions"])
    assert not widget.flimari_button.isEnabled()

    # Adding the phasor layer to the selection enables it.
    widget.export_layer_combobox.setCheckedItems(
        ["Component 1 Fractions", phasor_layer.name]
    )
    assert widget.flimari_button.isEnabled()

    # Sending only forwards the phasor layer and skips the other one.
    from unittest.mock import MagicMock

    with (
        patch("napari_phasors._flimari._import_bridge") as mock_import_bridge,
        patch("napari_phasors._widget.show_info") as mock_show_info,
    ):
        fake_bridge = MagicMock()
        mock_import_bridge.return_value = fake_bridge

        widget.flimari_button.click()

        fake_bridge.import_from_napari_phasors.assert_called_once()
        (sent_payloads,) = fake_bridge.import_from_napari_phasors.call_args[0]
        assert len(sent_payloads) == 1
        assert sent_payloads[0]["name"] == phasor_layer.name
        message = mock_show_info.call_args[0][0]
        assert "Sent 1 layer(s)" in message
        assert "Component 1 Fractions" in message


def test_writer_widget_flimari_prompts_for_raw_file(make_viewer_model, qtbot):
    """A phasor layer without summed_signal prompts for the original raw file.

    The recovered histogram-bin count is used to reconstruct photon counts,
    which are then included in the payload sent to FLIMari.
    """
    from unittest.mock import MagicMock

    viewer = make_viewer_model()
    raw_flim_data = make_raw_flim_data()
    layer = make_intensity_layer_with_phasors(raw_flim_data)
    # Simulate a layer loaded from a processed format without raw histogram.
    del layer.metadata["summed_signal"]
    viewer.add_layer(layer)

    widget = WriterWidget(viewer)
    widget.export_layer_combobox.setCheckedItems([layer.name])

    with (
        patch(
            "napari_phasors._widget.QFileDialog.getOpenFileName",
            return_value=("/fake/original.ptu", ""),
        ) as mock_dialog,
        patch(
            "napari_phasors._flimari.histogram_bins_from_raw_file",
            return_value=64,
        ) as mock_bins,
        patch("napari_phasors._flimari._import_bridge") as mock_import_bridge,
        patch("napari_phasors._widget.show_info"),
    ):
        fake_bridge = MagicMock()
        mock_import_bridge.return_value = fake_bridge

        widget.flimari_button.click()

        # User was prompted for the raw file for this layer.
        mock_dialog.assert_called_once()
        mock_bins.assert_called_once()

        (sent_payloads,) = fake_bridge.import_from_napari_phasors.call_args[0]
        assert "counts" in sent_payloads[0]
        np.testing.assert_allclose(
            sent_payloads[0]["counts"],
            np.rint(layer.metadata["original_mean"] * 64),
        )


def test_writer_widget_flimari_skips_prompt_when_counts_present(
    make_viewer_model, qtbot
):
    """A layer that already has summed_signal is not prompted for a raw file."""
    from unittest.mock import MagicMock

    viewer = make_viewer_model()
    raw_flim_data = make_raw_flim_data()
    layer = make_intensity_layer_with_phasors(raw_flim_data)
    viewer.add_layer(layer)

    widget = WriterWidget(viewer)
    widget.export_layer_combobox.setCheckedItems([layer.name])

    with (
        patch(
            "napari_phasors._widget.QFileDialog.getOpenFileName"
        ) as mock_dialog,
        patch("napari_phasors._flimari._import_bridge") as mock_import_bridge,
        patch("napari_phasors._widget.show_info"),
    ):
        mock_import_bridge.return_value = MagicMock()
        widget.flimari_button.click()

        mock_dialog.assert_not_called()


def test_send_to_flimari_shows_error_when_nothing_selected(
    make_viewer_model, qtbot
):
    """_send_to_flimari guards against being invoked with no selection.

    The button itself is disabled in this state, so the handler is called
    directly to exercise the guard clause.
    """
    viewer = make_viewer_model()
    widget = WriterWidget(viewer)

    with patch("napari_phasors._widget.show_error") as mock_show_error:
        widget._send_to_flimari()

    mock_show_error.assert_called_once_with("No layer selected")


def test_send_to_flimari_reports_flimari_not_available(
    make_viewer_model, qtbot
):
    """FlimariNotAvailable from the bridge is surfaced via show_error."""
    from napari_phasors._flimari import FlimariNotAvailable

    viewer = make_viewer_model()
    raw_flim_data = make_raw_flim_data()
    layer = make_intensity_layer_with_phasors(raw_flim_data)
    viewer.add_layer(layer)

    widget = WriterWidget(viewer)
    widget.export_layer_combobox.setCheckedItems([layer.name])

    with (
        patch(
            "napari_phasors._flimari.send_layers_to_flimari",
            side_effect=FlimariNotAvailable("FLIMari is not installed."),
        ),
        patch("napari_phasors._widget.show_error") as mock_show_error,
    ):
        widget._send_to_flimari()

    mock_show_error.assert_called_once_with("FLIMari is not installed.")


def test_send_to_flimari_reports_value_error(make_viewer_model, qtbot):
    """A ValueError from send_layers_to_flimari is surfaced via show_error."""
    viewer = make_viewer_model()
    raw_flim_data = make_raw_flim_data()
    layer = make_intensity_layer_with_phasors(raw_flim_data)
    viewer.add_layer(layer)

    widget = WriterWidget(viewer)
    widget.export_layer_combobox.setCheckedItems([layer.name])

    with (
        patch(
            "napari_phasors._flimari.send_layers_to_flimari",
            side_effect=ValueError("no phasor data to send."),
        ),
        patch("napari_phasors._widget.show_error") as mock_show_error,
    ):
        widget._send_to_flimari()

    mock_show_error.assert_called_once_with("no phasor data to send.")


def test_send_to_flimari_reports_runtime_error(make_viewer_model, qtbot):
    """A RuntimeError still raised after the dock-open retry is surfaced."""
    viewer = make_viewer_model()
    raw_flim_data = make_raw_flim_data()
    layer = make_intensity_layer_with_phasors(raw_flim_data)
    viewer.add_layer(layer)

    widget = WriterWidget(viewer)
    widget.export_layer_combobox.setCheckedItems([layer.name])

    with (
        patch(
            "napari_phasors._flimari.send_layers_to_flimari",
            side_effect=RuntimeError("FLIMari is not ready to receive data."),
        ),
        patch("napari_phasors._widget.show_error") as mock_show_error,
    ):
        widget._send_to_flimari()

    mock_show_error.assert_called_once_with(
        "FLIMari is not ready to receive data."
    )


def test_send_to_flimari_reports_unexpected_error(make_viewer_model, qtbot):
    """Any other exception is caught and reported with context."""
    viewer = make_viewer_model()
    raw_flim_data = make_raw_flim_data()
    layer = make_intensity_layer_with_phasors(raw_flim_data)
    viewer.add_layer(layer)

    widget = WriterWidget(viewer)
    widget.export_layer_combobox.setCheckedItems([layer.name])

    with (
        patch(
            "napari_phasors._flimari.send_layers_to_flimari",
            side_effect=TypeError("boom"),
        ),
        patch("napari_phasors._widget.show_error") as mock_show_error,
    ):
        widget._send_to_flimari()

    mock_show_error.assert_called_once_with("Error sending to FLIMari: boom")


def test_recover_missing_counts_reports_error_for_bad_raw_file(
    make_viewer_model, qtbot
):
    """A raw file that fails validation is reported and the layer is skipped.

    The layer must then be exported without a histogram-bin override, so it
    falls back to sending mean intensity as FLIMari's counts proxy.
    """
    viewer = make_viewer_model()
    raw_flim_data = make_raw_flim_data()
    layer = make_intensity_layer_with_phasors(raw_flim_data)
    del layer.metadata["summed_signal"]
    viewer.add_layer(layer)

    widget = WriterWidget(viewer)

    with (
        patch(
            "napari_phasors._widget.QFileDialog.getOpenFileName",
            return_value=("/fake/wrong.ptu", ""),
        ),
        patch(
            "napari_phasors._flimari.histogram_bins_from_raw_file",
            side_effect=ValueError("the selected file does not match."),
        ),
        patch("napari_phasors._widget.show_error") as mock_show_error,
    ):
        overrides = widget._recover_missing_counts([layer])

    assert overrides == {}
    mock_show_error.assert_called_once_with(
        f"Could not recover photon counts for "
        f"'{layer.name}': the selected file does not match."
    )


def test_export_labels_layer_as_colored_image(
    make_viewer_model, qtbot, tmp_path
):
    """Test that Labels layers are exported as colored images without colorbar."""
    from PIL import Image as PILImage

    from napari_phasors._writer import export_layer_as_image

    viewer = make_viewer_model()

    # Create a labels layer with distinct labels
    labels_data = np.zeros((10, 10), dtype=np.int32)
    labels_data[2:5, 2:5] = 1
    labels_data[6:9, 6:9] = 2
    labels_layer = viewer.add_labels(labels_data, name="test_labels")

    # Export labels layer as image
    export_path = str(tmp_path / "test_labels_export.png")
    export_layer_as_image(export_path, labels_layer, include_colorbar=True)

    assert os.path.exists(export_path)

    # Open and verify the image has color representation (not black and white)
    with PILImage.open(export_path) as img:
        img_data = np.array(img)

    assert img_data.ndim == 3
    assert img_data.shape[-1] in (3, 4)

    # Check that it's colored (not grayscale where R=G=B for all pixels)
    is_colored = np.any(img_data[..., 0] != img_data[..., 1]) or np.any(
        img_data[..., 1] != img_data[..., 2]
    )
    assert is_colored, "Exported labels image is grayscale or black and white!"


def test_writer_widget_mask_checkbox(make_viewer_model, qtbot, tmp_path):
    """Test that WriterWidget mask checkbox behaves dynamically based on layer mask presence."""
    viewer = make_viewer_model()

    # 1. Create a layer without phasor/mask data
    data = np.random.random((10, 10))
    layer = viewer.add_image(data, name="test_image")

    widget = WriterWidget(viewer)

    # Select layer
    widget.export_layer_combobox.selectAll()

    # Verify checkbox is hidden
    assert widget.mask_checkbox.isHidden() is True

    # 2. Add a mask to the metadata and trigger metadata event
    layer.metadata["mask"] = np.ones((10, 10))
    layer.events.metadata()

    # Verify checkbox is visible
    assert widget.mask_checkbox.isHidden() is False

    # 3. Remove the mask and trigger metadata event
    del layer.metadata["mask"]
    layer.events.metadata()

    # Verify checkbox is hidden again
    assert widget.mask_checkbox.isHidden() is True

    # 4. Check saving calls write_ome_tiff with correct export_masked parameter
    # Restore mask
    layer.metadata["mask"] = np.ones((10, 10))
    layer.events.metadata()
    widget.mask_checkbox.setChecked(True)

    with patch("napari_phasors._widget.write_ome_tiff") as mock_write:
        widget._save_file(
            str(tmp_path / "output.ome.tif"),
            "Phasor as OME-TIFF (*.ome.tif)",
            selected_layers=["test_image"],
            export_masked=True,
        )
        mock_write.assert_called_once_with(
            str(tmp_path / "output.ome.tif"),
            layer,
            export_masked=True,
        )

    # Clean up
    widget.close()


def test_advanced_options_widget_kwargs(make_viewer_model, qtbot):
    """Test the dynamic kwargs functionality in AdvancedOptionsWidget."""
    viewer = make_viewer_model()
    widget = FbdWidget(viewer, path=get_test_file_path("test_file$EI0S.fbd"))

    # Initially, kwargs_widgets should be empty
    assert hasattr(widget, "kwargs_widgets")
    assert len(widget.kwargs_widgets) == 0

    # 1. Add a kwarg row
    widget.add_kwarg_btn.click()
    assert len(widget.kwargs_widgets) == 1

    key_edit, val_edit, row_widget = widget.kwargs_widgets[0]
    key_edit.setText("test_key")
    val_edit.setText("123")

    options = {}
    widget._apply_kwargs(options)
    assert options == {"test_key": 123}

    # 2. Add another kwarg row with a list
    widget.add_kwarg_btn.click()
    assert len(widget.kwargs_widgets) == 2

    key_edit2, val_edit2, row_widget2 = widget.kwargs_widgets[1]
    key_edit2.setText("list_key")
    val_edit2.setText("[1, 2, 'three']")

    options = {}
    widget._apply_kwargs(options)
    assert options == {
        "test_key": 123,
        "list_key": [1, 2, "three"],
    }

    # 3. Add one more with string value that fails ast.literal_eval
    widget.add_kwarg_btn.click()
    key_edit3, val_edit3, row_widget3 = widget.kwargs_widgets[2]
    key_edit3.setText("str_key")
    val_edit3.setText("hello_world")

    options = {}
    widget._apply_kwargs(options)
    assert options == {
        "test_key": 123,
        "list_key": [1, 2, "three"],
        "str_key": "hello_world",
    }

    # 4. Remove a kwarg row (the first one)
    layout = row_widget.layout()
    del_btn = layout.itemAt(2).widget()
    del_btn.click()

    assert len(widget.kwargs_widgets) == 2
    # Verify the remaining widgets are row_widget2 and row_widget3
    assert widget.kwargs_widgets[0][2] == row_widget2
    assert widget.kwargs_widgets[1][2] == row_widget3

    options = {}
    widget._apply_kwargs(options)
    assert options == {
        "list_key": [1, 2, "three"],
        "str_key": "hello_world",
    }

    widget.close()


def test_ifli_widget(make_viewer_model, qtbot):
    """Test IfliWidget UI initialization and kwargs integration."""
    from napari_phasors._widget import IfliWidget, ProcessedOnlyWidget

    viewer = make_viewer_model()
    widget = IfliWidget(viewer, path="dummy.ifli")

    # Check that UI elements are correctly set up
    assert widget.channel_entry.text() == "0"
    assert widget.reader_options["channel"] == 0
    assert hasattr(widget, "kwargs_widgets")

    # Add a kwarg row
    widget.add_kwarg_btn.click()
    key_edit, val_edit, _ = widget.kwargs_widgets[0]
    key_edit.setText("ifli_kwarg")
    val_edit.setText("'yes'")

    # Change the channel entry
    widget.channel_entry.setText("2")

    # Mock super()._on_click to see what arguments it receives
    with patch.object(ProcessedOnlyWidget, "_on_click") as mock_on_click:
        widget.btn.click()

        mock_on_click.assert_called_once()
        args, kwargs = mock_on_click.call_args
        assert args[0] == "dummy.ifli"
        assert args[1]["channel"] == 2
        assert args[1]["ifli_kwarg"] == "yes"

    # Test fallback to channel 0 when channel entry is empty
    widget.channel_entry.setText("")
    with patch.object(ProcessedOnlyWidget, "_on_click") as mock_on_click:
        widget.btn.click()
        assert widget.reader_options["channel"] == 0
        assert widget.channel_entry.text() == ""

    # Test fallback to channel 0 when channel entry is invalid
    widget.channel_entry.setText("abc")
    with patch.object(ProcessedOnlyWidget, "_on_click") as mock_on_click:
        widget.btn.click()
        assert widget.reader_options["channel"] == 0
        assert widget.channel_entry.text() == "0"

    widget.close()


def test_lsm_widget_get_signal_data_with_kwargs(make_viewer_model, qtbot):
    """Test that LSM file reading forwards kwargs."""
    viewer = make_viewer_model()
    widget = LsmWidget(viewer, path="dummy.lsm")
    widget._is_lsm = True

    # Add a custom kwarg row
    widget.add_kwarg_btn.click()
    key_edit, val_edit, _ = widget.kwargs_widgets[0]
    key_edit.setText("custom_param")
    val_edit.setText("99")

    with patch("phasorpy.io.signal_from_lsm") as mock_signal_from_lsm:
        mock_signal_from_lsm.return_value = np.zeros((2, 2, 2))
        widget._get_signal_data()

        mock_signal_from_lsm.assert_called_once_with(
            "dummy.lsm", custom_param=99
        )

    widget.close()


def test_tiff_widget_get_signal_data_with_kwargs(make_viewer_model, qtbot):
    """Test that TIFF file reading forwards kwargs."""
    viewer = make_viewer_model()
    widget = LsmWidget(viewer, path="dummy.tif")
    widget._is_lsm = False

    # Add a custom kwarg row
    widget.add_kwarg_btn.click()
    key_edit, val_edit, _ = widget.kwargs_widgets[0]
    key_edit.setText("custom_tiff_param")
    val_edit.setText("True")

    with patch("tifffile.imread") as mock_imread:
        mock_imread.return_value = np.zeros((2, 2, 2))
        widget._get_signal_data()

        mock_imread.assert_called_once_with(
            "dummy.tif", custom_tiff_param=True
        )

    widget.close()


def test_czi_widget_get_signal_data_with_kwargs(make_viewer_model, qtbot):
    """Test that CZI file reading forwards kwargs."""
    viewer = make_viewer_model()
    widget = CziWidget(viewer, path="dummy.czi")

    # Add a custom kwarg row
    widget.add_kwarg_btn.click()
    key_edit, val_edit, _ = widget.kwargs_widgets[0]
    key_edit.setText("czi_param")
    val_edit.setText("4.5")

    with patch("phasorpy.io.signal_from_czi") as mock_signal_from_czi:
        mock_signal_from_czi.return_value = np.zeros((2, 2, 2))
        widget._get_signal_data()

        mock_signal_from_czi.assert_called_once_with(
            "dummy.czi", czi_param=4.5
        )

    widget.close()


def test_bh_widget_init(make_viewer_model, qtbot):
    from unittest.mock import patch

    from napari_phasors._widget import BhWidget

    viewer = make_viewer_model()
    with patch("napari_phasors._widget.BhWidget._update_signal_plot"):
        widget = BhWidget(viewer, path="test.bh")
    assert widget.path == "test.bh"
    assert widget.reader_options == {}


def test_pqbin_widget_init(make_viewer_model, qtbot):
    from unittest.mock import patch

    from napari_phasors._widget import PqbinWidget

    viewer = make_viewer_model()
    with patch("napari_phasors._widget.PqbinWidget._update_signal_plot"):
        widget = PqbinWidget(viewer, path="test.bin")
    assert widget.path == "test.bin"


def test_simfcs_widget_init(make_viewer_model, qtbot):
    from unittest.mock import patch

    from napari_phasors._widget import SimfcsWidget

    viewer = make_viewer_model()
    with patch("napari_phasors._widget.SimfcsWidget._update_signal_plot"):
        widget = SimfcsWidget(viewer, path="test.r64")
    assert widget.path == "test.r64"


def test_ifli_widget_init(make_viewer_model, qtbot):
    from unittest.mock import patch

    from napari_phasors._widget import IfliWidget

    viewer = make_viewer_model()
    with patch("napari_phasors._widget.IfliWidget._update_signal_plot"):
        widget = IfliWidget(viewer, path="test.ifli")
    assert widget.path == "test.ifli"


def test_flif_widget_init(make_viewer_model, qtbot):
    from unittest.mock import patch

    from napari_phasors._widget import FlifWidget

    viewer = make_viewer_model()
    with patch("napari_phasors._widget.FlifWidget._update_signal_plot"):
        widget = FlifWidget(viewer, path="test.flif")
    assert widget.path == "test.flif"


def test_lif_widget_init(make_viewer_model, qtbot):
    from unittest.mock import patch

    from napari_phasors._widget import LifWidget

    viewer = make_viewer_model()
    with patch("napari_phasors._widget.LifWidget._update_signal_plot"):
        widget = LifWidget(viewer, path="test.lif")
    assert widget.path == "test.lif"


def test_json_widget_init(make_viewer_model, qtbot):
    from unittest.mock import patch

    from napari_phasors._widget import JsonWidget

    viewer = make_viewer_model()
    with patch("napari_phasors._widget.JsonWidget._update_signal_plot"):
        widget = JsonWidget(viewer, path="test.json")
    assert widget.path == "test.json"


def test_phasor_transform_open_multi_file_dialog_single_file(
    make_viewer_model, qtbot, monkeypatch
):
    from unittest.mock import patch

    from napari_phasors._widget import PhasorTransform

    viewer = make_viewer_model()
    widget = PhasorTransform(viewer)

    mock_files = (["/fake/path/test1.lsm"], "")
    monkeypatch.setattr(
        "napari_phasors._widget.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: mock_files,
    )

    class MockFileOrderDialog:
        Accepted = 1

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return 1

        def get_ordered_paths(self):
            return ["/fake/path/test1.lsm"]

        def get_z_spacing(self):
            return 1.0

        def get_axis_order(self):
            return [0, 1]

        def get_axis_labels(self):
            return ["Y", "X"]

    monkeypatch.setattr(
        "napari_phasors._widget.FileOrderDialog", MockFileOrderDialog
    )
    monkeypatch.setattr("os.path.isfile", lambda p: True)

    with patch("napari_phasors._widget.LsmWidget._update_signal_plot"):
        widget.multi_file_button.click()

    assert widget.dynamic_widget_layout.count() == 1


def test_phasor_transform_open_multi_file_dialog_multiple_files(
    make_viewer_model, qtbot, monkeypatch
):
    from unittest.mock import patch

    from napari_phasors._widget import PhasorTransform

    viewer = make_viewer_model()
    widget = PhasorTransform(viewer)

    mock_files = (["/fake/path/test2.lsm", "/fake/path/test1.lsm"], "")
    monkeypatch.setattr(
        "napari_phasors._widget.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: mock_files,
    )

    class MockFileOrderDialog:
        Accepted = 1

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return 1

        def get_ordered_paths(self):
            return ["/fake/path/test1.lsm", "/fake/path/test2.lsm"]

        def get_z_spacing(self):
            return 1.0

        def get_axis_order(self):
            return [0, 1]

        def get_axis_labels(self):
            return ["Y", "X"]

    monkeypatch.setattr(
        "napari_phasors._widget.FileOrderDialog", MockFileOrderDialog
    )
    monkeypatch.setattr("os.path.isfile", lambda p: True)

    with patch("napari_phasors._widget.LsmWidget._update_signal_plot"):
        widget.multi_file_button.click()

    assert widget.dynamic_widget_layout.count() == 1
    added_widget = widget.dynamic_widget_layout.itemAt(0).widget()
    assert added_widget._multi_file_paths == [
        "/fake/path/test1.lsm",
        "/fake/path/test2.lsm",
    ]


def test_phasor_transform_open_multi_file_dialog_cancel(
    make_viewer_model, qtbot, monkeypatch
):
    from napari_phasors._widget import PhasorTransform

    viewer = make_viewer_model()
    widget = PhasorTransform(viewer)

    mock_files = (["/fake/path/test1.lsm"], "")
    monkeypatch.setattr(
        "napari_phasors._widget.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: mock_files,
    )

    class MockFileOrderDialog:
        Accepted = 1
        Rejected = 0

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return 0

    monkeypatch.setattr(
        "napari_phasors._widget.FileOrderDialog", MockFileOrderDialog
    )
    monkeypatch.setattr("os.path.isfile", lambda p: True)

    widget.multi_file_button.click()
    assert widget.dynamic_widget_layout.count() == 0


def test_phasor_transform_open_multi_file_dialog_mismatched_extensions(
    make_viewer_model, qtbot, monkeypatch
):
    from unittest.mock import MagicMock

    from napari_phasors._widget import PhasorTransform

    viewer = make_viewer_model()
    widget = PhasorTransform(viewer)

    mock_files = (["/fake/path/test1.lsm", "/fake/path/test2.ptu"], "")
    monkeypatch.setattr(
        "napari_phasors._widget.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: mock_files,
    )
    monkeypatch.setattr("os.path.isfile", lambda p: True)

    mock_show_error = MagicMock()
    monkeypatch.setattr("napari_phasors._widget.show_error", mock_show_error)

    widget.multi_file_button.click()
    mock_show_error.assert_called_once()
    assert (
        "All selected files must have the same extension"
        in mock_show_error.call_args[0][0]
    )


def test_writer_widget_export_multiple_layers(
    make_viewer_model, qtbot, monkeypatch, tmp_path
):
    from unittest.mock import patch

    import numpy as np

    from napari_phasors._widget import WriterWidget

    viewer = make_viewer_model()
    viewer.add_image(np.random.rand(10, 10), name="Layer1")
    viewer.add_image(np.random.rand(10, 10), name="Layer2")

    widget = WriterWidget(viewer)

    # Mock file dialog to return a directory and base name
    mock_save_path = str(tmp_path / "export_test")
    monkeypatch.setattr(
        "napari_phasors._widget.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (mock_save_path, "Layer data as CSV (*.csv)"),
    )

    # Select multiple layers
    widget.export_layer_combobox.setCheckedItems(["Layer1", "Layer2"])

    with patch("napari_phasors._widget.export_layer_as_csv") as mock_csv:
        widget._open_file_dialog()
        assert mock_csv.call_count == 2

    # Also test error handling
    mock_save_path2 = str(tmp_path / "export_test2")
    monkeypatch.setattr(
        "napari_phasors._widget.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (mock_save_path2, "Layer data as CSV (*.csv)"),
    )
    with patch(
        "napari_phasors._widget.export_layer_as_csv",
        side_effect=Exception("Test Error"),
    ):
        with patch("napari_phasors._widget.show_error") as mock_error:
            widget._open_file_dialog()
            assert mock_error.call_count == 2


def test_widget_pure_helpers(tmp_path):
    """Consolidated coverage of widget-free helper functions: axis
    transform, z-scale, OME-TIFF z-spacing readout and axis labels."""
    import numpy as np
    import tifffile

    from napari_phasors._widget import (
        AdvancedOptionsWidget,
        _default_axis_labels,
        _estimate_result_shape,
        _try_get_z_spacing_from_ome_tiff,
    )

    # --- _apply_axis_transform ---
    data = np.arange(24).reshape(2, 3, 4)
    meta = {
        "original_mean": data.copy(),
        "G": np.stack([data, data]),  # ndim + 1 (harmonic axis)
        "S": None,
    }
    kwargs = {"metadata": meta}
    out = AdvancedOptionsWidget._apply_axis_transform(
        kwargs, data, axis_order=[2, 0, 1], axis_labels=["A", "B", "C"]
    )
    assert out.shape == (4, 2, 3)
    assert meta["original_mean"].shape == (4, 2, 3)
    assert meta["G"].shape == (2, 4, 2, 3)
    assert kwargs["axis_labels"] == ("A", "B", "C")
    # Identity order and invalid order are no-ops.
    same = AdvancedOptionsWidget._apply_axis_transform(
        {"metadata": {}}, data, axis_order=[0, 1, 2], axis_labels=None
    )
    assert same.shape == data.shape
    bad = AdvancedOptionsWidget._apply_axis_transform(
        {"metadata": {}}, data, axis_order=[0, 0, 1], axis_labels=None
    )
    assert bad.shape == data.shape

    # --- _set_layer_z_scale ---
    kw = {}
    AdvancedOptionsWidget._set_layer_z_scale(kw, np.ones((3, 4, 4)), 2.5)
    assert kw["scale"] == (2.5, 1.0, 1.0)
    # Existing scale shorter than ndim is padded; longer is truncated.
    kw = {"scale": (1.0,)}
    AdvancedOptionsWidget._set_layer_z_scale(kw, np.ones((3, 4, 4)), 2.0)
    assert kw["scale"] == (2.0, 1.0, 1.0)
    kw = {"scale": (1.0, 1.0, 1.0, 1.0)}
    AdvancedOptionsWidget._set_layer_z_scale(kw, np.ones((3, 4, 4)), 2.0)
    assert len(kw["scale"]) == 3
    # Guards: None, 2D data, non-numeric, non-positive.
    for args in (
        ({}, np.ones((3, 4, 4)), None),
        ({}, np.ones((4, 4)), 2.0),
        ({}, np.ones((3, 4, 4)), "abc"),
        ({}, np.ones((3, 4, 4)), -1.0),
    ):
        kw = args[0]
        AdvancedOptionsWidget._set_layer_z_scale(*args)
        assert "scale" not in kw

    # --- _try_get_z_spacing_from_ome_tiff ---
    assert _try_get_z_spacing_from_ome_tiff("file.ptu") is None
    # OME metadata with PhysicalSizeZ (written like the project writer does).
    from phasorpy.io import phasor_to_ometiff

    p1 = str(tmp_path / "z1.ome.tif")
    phasor_to_ometiff(
        p1,
        np.ones((2, 4, 4)),
        np.zeros((1, 2, 4, 4)),
        np.zeros((1, 2, 4, 4)),
        harmonic=[1],
        dims="ZYX",
        metadata={"PhysicalSizeZ": 3.5, "PhysicalSizeZUnit": "µm"},
    )
    assert _try_get_z_spacing_from_ome_tiff(p1) == 3.5
    # Fallback to napari-phasors settings in the description.
    import json as _json

    p2 = str(tmp_path / "z2.ome.tif")
    desc = _json.dumps(
        {"napari_phasors_settings": _json.dumps({"z_spacing_um": 4.5})}
    )
    tifffile.imwrite(p2, np.ones((4, 4), dtype=np.float32), description=desc)
    assert _try_get_z_spacing_from_ome_tiff(p2) == 4.5
    # Unreadable file returns None.
    p3 = str(tmp_path / "junk.ome.tif")
    with open(p3, "w") as fh:
        fh.write("not a tiff")
    assert _try_get_z_spacing_from_ome_tiff(p3) is None

    # --- _default_axis_labels ---
    assert _default_axis_labels(2) == ["Y", "X"]
    assert _default_axis_labels(3) == ["Z", "Y", "X"]
    assert _default_axis_labels(4) == ["T", "Z", "Y", "X"]
    assert _default_axis_labels(5)[0] == "Axis 0"

    # --- _estimate_result_shape guards ---
    assert _estimate_result_shape([]) is None
    assert _estimate_result_shape(["nope.unsupported"]) is None


def test_fbd_widget_stack_z_spacing_and_harmonic_edits(
    make_viewer_model, qtbot
):
    """Piggyback on one FbdWidget build to cover stack z-spacing UI
    lifecycle and harmonic line-edit clamping branches."""
    viewer = make_viewer_model()
    test_file_path = get_test_file_path("test_file$EI0S.fbd")
    widget = FbdWidget(viewer, path=test_file_path)

    # --- stack z-spacing editor: created in stack mode, removed otherwise ---
    widget._multi_file_paths = ["a.fbd", "b.fbd"]
    widget._sync_stack_z_spacing_widget_visibility()
    assert widget._stack_z_spacing_layout is not None
    assert widget._stack_z_spacing_edit is not None

    # Valid, invalid and non-positive edits.
    widget._stack_z_spacing_edit.setText("2.5")
    widget._on_stack_z_spacing_changed()
    assert widget._stack_z_spacing == 2.5
    widget._stack_z_spacing_edit.setText("oops")
    widget._on_stack_z_spacing_changed()
    assert widget._stack_z_spacing == 1.0
    widget._stack_z_spacing_edit.setText("-3")
    widget._on_stack_z_spacing_changed()
    assert widget._stack_z_spacing == 1.0

    # Leaving stack mode tears the editor down.
    widget._multi_file_paths = []
    widget._sync_stack_z_spacing_widget_visibility()
    assert widget._stack_z_spacing_layout is None

    # --- harmonic line edits: clamping below 1, above max, start > end ---
    if widget.harmonic_start_edit is not None:
        widget.harmonic_start_edit.setText("0")
        widget.harmonic_end_edit.setText("2")
        widget._on_harmonic_edit_changed()
        assert widget.harmonics[0] == 1

        widget.harmonic_end_edit.setText(str(widget.max_harmonic + 10))
        widget._on_harmonic_edit_changed()
        assert widget.harmonics[-1] == widget.max_harmonic

        widget.harmonic_start_edit.setText("5")
        widget.harmonic_end_edit.setText("2")
        widget._on_harmonic_edit_changed()
        assert widget.harmonics[0] <= widget.harmonics[-1]

        # Non-numeric input falls back to the slider values.
        widget.harmonic_start_edit.setText("abc")
        widget._on_harmonic_edit_changed()
        assert len(widget.harmonics) >= 1
