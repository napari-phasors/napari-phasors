import os
from unittest.mock import patch

import numpy as np
import pandas as pd
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
    assert widget.harmonics == [1]
    assert widget.all_frames == 9
    assert widget.all_channels == 2
    assert widget.harmonic_start.value() == 1
    assert widget.harmonic_end.value() == 1
    # Modify harmonic values
    widget.harmonic_start.setValue(2)
    assert (
        widget.harmonic_start.value() == 2 and widget.harmonic_end.value() == 2
    )
    assert widget.harmonics == [2]
    widget.harmonic_end.setValue(3)
    assert (
        widget.harmonic_start.value() == 2 and widget.harmonic_end.value() == 3
    )
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
    assert viewer.layers[0].data.shape == (1, 256, 256)
    phasor_data = (
        viewer.layers[0].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (131072, 6)
    assert phasor_data["harmonic"].unique().tolist() == [2, 3]
    # Modify channels and harmonics and phasor transform again
    widget.channels.setCurrentIndex(0)
    widget.harmonic_end.setValue(2)
    widget.btn.click()
    assert len(viewer.layers) == 3
    assert viewer.layers[2].name == "test_file$EI0S Intensity Image: Channel 1"
    assert viewer.layers[2].data.shape == (1, 256, 256)
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
    assert widget.harmonics == [1]
    assert widget.all_frames == 5
    assert widget.all_channels == 1
    assert widget.harmonic_start.value() == 1
    assert widget.harmonic_end.value() == 1
    # Modify harmonic values
    widget.harmonic_start.setValue(2)
    assert (
        widget.harmonic_start.value() == 2 and widget.harmonic_end.value() == 2
    )
    assert widget.harmonics == [2]
    widget.harmonic_end.setValue(3)
    assert (
        widget.harmonic_start.value() == 2 and widget.harmonic_end.value() == 3
    )
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
    channels_combobox_values = [
        widget.channels.itemText(i) for i in range(widget.channels.count())
    ]
    assert channels_combobox_values == ["All channels", "0"]
    assert widget.channels.currentIndex() == 0
    # Modify channels
    widget.channels.setCurrentIndex(1)
    assert widget.reader_options == {"frame": 0, "channel": 0}
    # Click button of phasor transform and check layers
    widget.btn.click()
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "test_file Intensity Image: Channel 0"
    assert viewer.layers[0].data.shape == (1, 256, 256)
    phasor_data = (
        viewer.layers[0].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (131072, 6)
    assert phasor_data["harmonic"].unique().tolist() == [2, 3]
    # Modify frames and harmonics and phasor transform again
    widget.frames.setCurrentIndex(0)
    widget.harmonic_end.setValue(2)
    widget.btn.click()
    assert len(viewer.layers) == 2
    assert viewer.layers[1].name == "test_file Intensity Image: Channel 0 [1]"
    assert viewer.layers[1].data.shape == (1, 256, 256)
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
    assert widget.harmonics == [1]
    assert widget.harmonic_start.value() == 1
    assert widget.harmonic_end.value() == 1
    # Modify harmonic values
    widget.harmonic_start.setValue(2)
    assert (
        widget.harmonic_start.value() == 2 and widget.harmonic_end.value() == 2
    )
    assert widget.harmonics == [2]
    widget.harmonic_end.setValue(3)
    assert (
        widget.harmonic_start.value() == 2 and widget.harmonic_end.value() == 3
    )
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
    widget.harmonic_end.setValue(2)
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
    assert widget.harmonics == [1]
    assert widget.harmonic_start.value() == 1
    assert widget.harmonic_end.value() == 1
    # Modify harmonic values
    widget.harmonic_start.setValue(2)
    assert (
        widget.harmonic_start.value() == 2 and widget.harmonic_end.value() == 2
    )
    assert widget.harmonics == [2]
    widget.harmonic_end.setValue(3)
    assert (
        widget.harmonic_start.value() == 2 and widget.harmonic_end.value() == 3
    )
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
    widget.harmonic_end.setValue(2)
    widget.btn.click()
    assert len(viewer.layers) == 2
    assert viewer.layers[1].name == "test_file Intensity Image [1]"
    assert viewer.layers[1].data.shape == (512, 512)
    phasor_data = (
        viewer.layers[1].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (262144, 6)
    assert phasor_data["harmonic"].unique().tolist() == [2]


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
        mock_show_error.assert_called_once_with(
            "No layer with phasor data selected"
        )
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
        patch("napari_phasors._widget.QFileDialog.exec_", return_value=True),
        patch(
            "napari_phasors._widget.QFileDialog.selectedFiles",
            return_value=[str(tmp_path / "test.ome.tif")],
        ),
        patch(
            "napari_phasors._widget.QFileDialog.selectedNameFilter",
            return_value="Phasor as OME-TIFF (*.ome.tif)",
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
        patch("napari_phasors._widget.QFileDialog.exec_", return_value=True),
        patch(
            "napari_phasors._widget.QFileDialog.selectedFiles",
            return_value=[str(tmp_path / "test.csv")],
        ),
        patch(
            "napari_phasors._widget.QFileDialog.selectedNameFilter",
            return_value="Phasor table as CSV (*.csv)",
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
