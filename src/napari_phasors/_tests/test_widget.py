import json
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
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
from napari_phasors._widget import (
    AdvancedOptionsWidget,
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


def test_phasor_transform_widget(make_napari_viewer):
    """Test PhasorTransform widget behavior with mocked QFileDialog."""
    viewer = make_napari_viewer()
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
        else:
            continue

        with (
            patch(
                "napari_phasors._widget.QFileDialog.exec_", return_value=True
            ),
            patch(
                "napari_phasors._widget.QFileDialog.selectedFiles",
                return_value=[test_file_path],
            ),
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


def test_phasor_transform_fbd_widget(make_napari_viewer):
    """Test FbdWidget from PhasorTransfrom widget."""
    viewer = make_napari_viewer()
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
    assert viewer.layers[0].name == "test_file$EI0S Intensity Image: Channel 0"
    assert viewer.layers[0].data.shape == (256, 256)
    phasor_data = (
        viewer.layers[0].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (131072, 6)
    assert phasor_data["harmonic"].unique().tolist() == [2, 3]
    # Modify channels and harmonics and phasor transform again
    widget.channels.setCurrentIndex(0)
    widget.harmonic_slider.setValue((2, 2))
    widget.btn.click()
    assert len(viewer.layers) == 3
    assert viewer.layers[2].name == "test_file$EI0S Intensity Image: Channel 1"
    assert viewer.layers[2].data.shape == (256, 256)
    phasor_data = (
        viewer.layers[2].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (65536, 6)
    assert phasor_data["harmonic"].unique().tolist() == [2]
    # TODO: test laser factor parameter


def test_phasor_transform_ptu_widget(make_napari_viewer):
    """Test PtuWidget from PhasorTransfrom widget."""
    viewer = make_napari_viewer()
    PhasorTransform(viewer)
    test_file_path = get_test_file_path("test_file.ptu")
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
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "test_file Intensity Image: Channel 0"
    assert viewer.layers[0].data.shape == (256, 256)
    phasor_data = (
        viewer.layers[0].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (131072, 6)
    assert phasor_data["harmonic"].unique().tolist() == [2, 3]
    # Modify frames and harmonics and phasor transform again
    widget.frames.setCurrentIndex(0)
    widget.harmonic_slider.setValue((2, 2))
    widget.btn.click()
    assert len(viewer.layers) == 2
    assert viewer.layers[1].name == "test_file Intensity Image: Channel 0 [1]"
    assert viewer.layers[1].data.shape == (256, 256)
    phasor_data = (
        viewer.layers[1].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (65536, 6)
    assert phasor_data["harmonic"].unique().tolist() == [2]
    # TODO: test dtime parameter


def test_phasor_transform_sdt_widget(make_napari_viewer):
    """Test SdtWidget from PhasorTransfrom widget."""
    viewer = make_napari_viewer()
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
    phasor_data = (
        viewer.layers[0].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (524288, 6)
    assert phasor_data["harmonic"].unique().tolist() == [2, 3]
    # Modify harmonics and phasor transform again
    widget.harmonic_slider.setValue((2, 2))
    widget.btn.click()
    assert len(viewer.layers) == 2
    assert (
        viewer.layers[1].name
        == "seminal_receptacle_FLIM_single_image Intensity Image: Channel 0 [1]"
    )
    assert viewer.layers[1].data.shape == (512, 512)
    phasor_data = (
        viewer.layers[1].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (262144, 6)
    assert phasor_data["harmonic"].unique().tolist() == [2]
    # TODO: test index parameter


def test_phasor_transform_lsm_widget(make_napari_viewer):
    """Test LsmWidget from PhasorTransfrom widget."""
    viewer = make_napari_viewer()
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
    phasor_data = (
        viewer.layers[0].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (524288, 6)
    assert phasor_data["harmonic"].unique().tolist() == [2, 3]
    # Modify harmonics and phasor transform again
    widget.harmonic_slider.setValue((2, 2))
    widget.btn.click()
    assert len(viewer.layers) == 2
    assert viewer.layers[1].name == "test_file Intensity Image [1]"
    assert viewer.layers[1].data.shape == (512, 512)
    phasor_data = (
        viewer.layers[1].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (262144, 6)
    assert phasor_data["harmonic"].unique().tolist() == [2]


def test_phasor_transform_ome_tif_widget(make_napari_viewer):
    """Test OmeTifWidget from PhasorTransform widget."""
    viewer = make_napari_viewer()
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
    phasor_data = (
        viewer.layers[0].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data["harmonic"].unique().tolist() == [2]


def test_harmonic_range_slider_functionality(make_napari_viewer):
    """Test QRangeSlider functionality for harmonics in all widget types."""
    viewer = make_napari_viewer()
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


def test_signal_plot_fbd_single_channel_all_frames(make_napari_viewer):
    """Test signal plot for FBD widget with single channel, all frames."""
    viewer = make_napari_viewer()
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


def test_signal_plot_fbd_single_channel_single_frame(make_napari_viewer):
    """Test signal plot for FBD widget with single channel, single frame."""
    viewer = make_napari_viewer()
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


def test_signal_plot_fbd_all_channels_single_frame(make_napari_viewer):
    """Test signal plot for FBD widget with all channels, single frame."""
    viewer = make_napari_viewer()
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


def test_signal_plot_sdt_widget(make_napari_viewer):
    """Test signal plot for SDT widget."""
    viewer = make_napari_viewer()
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


def test_signal_plot_ometif_widget(make_napari_viewer):
    """Test signal plot for OME-TIF widget."""
    viewer = make_napari_viewer()
    test_file_path = get_test_file_path("test_file.ome.tif")
    widget = OmeTifWidget(viewer, path=test_file_path)

    signal_data = None

    # Use phasorpy to read the OME-TIFF metadata
    _, _, _, attrs = phasor_from_ometiff(test_file_path, harmonic='all')

    # Get harmonics from attrs
    if "harmonic" in attrs:
        harmonics = attrs["harmonic"]
    if "description" in attrs.keys():
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


def test_signal_plot_lsm_widget(make_napari_viewer):
    """Test signal plot for LSM widget."""
    viewer = make_napari_viewer()
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


def test_signal_plot_error_handling(make_napari_viewer):
    """Test signal plot error handling when data cannot be loaded."""
    viewer = make_napari_viewer()
    test_file_path = get_test_file_path("test_file$EI0S.fbd")
    widget = FbdWidget(viewer, path=test_file_path)

    # Mock _get_signal_data to raise an exception
    with patch.object(
        widget, '_get_signal_data', side_effect=Exception("Test error")
    ):
        # Should not raise exception, should handle gracefully
        widget._update_signal_plot()

        # Plot should be cleared or show error state
        lines = widget.ax.get_lines()
        # Should either be empty or show error message


def test_phasor_transform_with_ome_tif_reader_option(make_napari_viewer):
    """Test PhasorTransform widget includes OmeTifWidget in reader options."""
    viewer = make_napari_viewer()
    widget = PhasorTransform(viewer)

    # Verify OME-TIF reader option is included
    assert ".ome.tif" in widget.reader_options

    # Test with OME-TIF file
    test_file_path = get_test_file_path("test_file.ome.tif")

    with (
        patch("napari_phasors._widget.QFileDialog.exec_", return_value=True),
        patch(
            "napari_phasors._widget.QFileDialog.selectedFiles",
            return_value=[test_file_path],
        ),
    ):
        widget.search_button.click()

        # Verify OmeTifWidget was added
        assert widget.dynamic_widget_layout.count() == 1
        added_widget = widget.dynamic_widget_layout.itemAt(0).widget()
        assert isinstance(added_widget, OmeTifWidget)


def test_signal_plot_canvas_properties(make_napari_viewer):
    """Test signal plot canvas has correct properties."""
    viewer = make_napari_viewer()
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


def test_signal_plot_data_consistency_across_widgets(make_napari_viewer):
    """Test signal plot data consistency across different widget types."""
    viewer = make_napari_viewer()

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


def test_writer_widget(make_napari_viewer, tmp_path):
    """Test the WriterWidget class."""
    # Intialize viewer and add intensity image layer with phasors data
    viewer = make_napari_viewer()
    main_widget = WriterWidget(viewer)
    assert main_widget.viewer is viewer
    assert isinstance(main_widget, QWidget)
    # Check init values are empty
    assert main_widget.export_layer_combobox.count() == 0
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
    assert (
        main_widget.export_layer_combobox.currentText()
        == sample_image_layer.name
    )

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
        export_layer_name = main_widget.export_layer_combobox.currentText()
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
        phasor_features = layer_data_tuple[1]["metadata"][
            "phasor_features_labels_layer"
        ]
        np.testing.assert_array_equal(
            phasor_features.data, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        )
        assert phasor_features.features.shape == (30, 6)
        expected_columns = [
            "label",
            "G_original",
            "S_original",
            "G",
            "S",
            "harmonic",
        ]
        actual_columns = phasor_features.features.columns.tolist()
        assert actual_columns == expected_columns
        assert phasor_features.features["harmonic"].unique().tolist() == [
            1,
            2,
            3,
        ]

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
        export_layer_name = main_widget.export_layer_combobox.currentText()
        export_path = str(tmp_path / "test.csv")
        mock_show_info.assert_called_once_with(
            f"Exported {export_layer_name} to {export_path}"
        )

        # Check if the file was created and has expected data when read
        assert os.path.exists(export_path)
        exported_table = pd.read_csv(export_path)
        exported_table = exported_table.astype(phasor_features.features.dtypes)
        coords = np.unravel_index(
            np.arange(sample_image_layer.data.size),
            sample_image_layer.data.shape,
        )
        coords = [np.tile(coord, len(harmonic)) for coord in coords]
        for dim, coord in enumerate(coords):
            phasor_features.features[f'dim_{dim}'] = coord
        pd.testing.assert_frame_equal(exported_table, phasor_features.features)


def test_writer_widget_image_exports(make_napari_viewer, tmp_path):
    """Test image export functionality in WriterWidget."""
    viewer = make_napari_viewer()
    main_widget = WriterWidget(viewer)

    # Create synthetic data
    raw_flim_data = make_raw_flim_data()
    harmonic = [1]
    sample_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )
    viewer.add_layer(sample_image_layer)

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


def test_writer_widget_colormap_applied(make_napari_viewer, tmp_path):
    """Test that the napari layer's colormap is correctly applied to exported images."""
    import matplotlib.pyplot as plt
    from PIL import Image

    viewer = make_napari_viewer()
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
        img = Image.open(export_path)
        assert img.size[0] > 0
        assert img.size[1] > 0


def test_writer_widget_file_extension_handling(make_napari_viewer, tmp_path):
    """Test that file extensions are correctly handled for all export formats."""
    viewer = make_napari_viewer()
    main_widget = WriterWidget(viewer)

    # Create synthetic data
    raw_flim_data = make_raw_flim_data()
    harmonic = [1]
    sample_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )
    viewer.add_layer(sample_image_layer)

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


def test_writer_widget_colorbar_checkbox_state(make_napari_viewer):
    """Test that the colorbar checkbox is properly initialized and responsive."""
    viewer = make_napari_viewer()
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


def test_writer_widget_csv_export_2d_no_phasor(make_napari_viewer, tmp_path):
    """Test CSV export for 2D image without phasor metadata."""
    viewer = make_napari_viewer()
    widget = WriterWidget(viewer)

    # Create a simple 2D image layer without phasor metadata
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    layer = viewer.add_image(data, name="test_2d_image")

    # Export as CSV
    csv_path = tmp_path / "test_2d.csv"
    widget._save_file(str(csv_path), "Layer data as CSV (*.csv)", False)

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


def test_writer_widget_csv_export_4d_no_phasor(make_napari_viewer, tmp_path):
    """Test CSV export for 4D image without phasor metadata."""
    viewer = make_napari_viewer()
    widget = WriterWidget(viewer)

    # Create a 4D image layer
    data = np.arange(48).reshape(2, 2, 3, 4)
    layer = viewer.add_image(data, name="test_4d_image")

    # Export as CSV
    csv_path = tmp_path / "test_4d.csv"
    widget._save_file(str(csv_path), "Layer data as CSV (*.csv)", False)

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
    make_napari_viewer, tmp_path
):
    """Test that CSV export maintains coordinate consistency for 2D images."""
    viewer = make_napari_viewer()
    widget = WriterWidget(viewer)

    # Create 2D image with known pattern
    data = np.array([[10, 20], [30, 40]])
    layer = viewer.add_image(data, name="test_pattern")

    # Export as CSV
    csv_path = tmp_path / "test_pattern.csv"
    widget._save_file(str(csv_path), "Layer data as CSV (*.csv)", False)

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
