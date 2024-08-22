from unittest.mock import MagicMock, patch

from PyQt5.QtCore import QModelIndex
from qtpy.QtWidgets import QWidget

from napari_phasors._widget import (
    AdvancedOptionsWidget,
    FbdWidget,
    LsmWidget,
    PhasorTransform,
    PtuWidget,
)

TEST_FORMATS = [
    (".fbd", FbdWidget),
    (".ptu", PtuWidget),
    (".lsm", LsmWidget),
    (".ome.tif", None),
]


def test_phasor_trasfrom_widget(make_napari_viewer):
    """Test PhasorTransform widget call for specific file formats."""
    viewer = make_napari_viewer()
    widget = PhasorTransform(viewer)
    assert widget.viewer is viewer
    assert isinstance(widget, QWidget)
    model = MagicMock()
    current = MagicMock(spec=QModelIndex)

    for extension, expected_widget_class in TEST_FORMATS:
        with patch(
            "napari_phasors._widget._get_filename_extension",
            return_value=("filename", extension),
        ):
            if extension == ".fbd":
                model.filePath.return_value = (
                    "src/napari_phasors/_tests/test_data/test_file$EI0S.fbd"
                )
            else:
                model.filePath.return_value = (
                    f"src/napari_phasors/_tests/test_data/test_file{extension}"
                )
            for i in reversed(range(widget.dynamic_widget_layout.count())):
                widget_item = widget.dynamic_widget_layout.takeAt(i).widget()
                if widget_item:
                    widget_item.deleteLater()
            widget._on_change(current, model)
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
    widget = FbdWidget(
        viewer, path="src/napari_phasors/_tests/test_data/test_file$EI0S.fbd"
    )
    assert widget.viewer is viewer
    # Init values
    assert isinstance(widget, AdvancedOptionsWidget)
    assert (
        widget.path == "src/napari_phasors/_tests/test_data/test_file$EI0S.fbd"
    )
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
    assert (
        viewer.layers[0].name
        == "test_file$EI0S.fbd Intensity Image: Channel 0"
    )
    assert viewer.layers[0].data.shape == (1, 256, 256)
    phasor_data = (
        viewer.layers[0].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (131072, 4)
    assert phasor_data["harmonic"].unique().tolist() == [2, 3]
    # Modify channels and harmonics and phasor transform again
    widget.channels.setCurrentIndex(0)
    widget.harmonic_end.setValue(2)
    widget.btn.click()
    assert len(viewer.layers) == 3
    assert (
        viewer.layers[2].name
        == "test_file$EI0S.fbd Intensity Image: Channel 1"
    )
    assert viewer.layers[2].data.shape == (1, 256, 256)
    phasor_data = (
        viewer.layers[2].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (65536, 4)
    assert phasor_data["harmonic"].unique().tolist() == [2]
    # TODO: test laser factor parameter


def test_phasor_transform_ptu_widget(make_napari_viewer):
    """Test PtuWidget from PhasorTransfrom widget."""
    viewer = make_napari_viewer()
    PhasorTransform(viewer)
    widget = PtuWidget(
        viewer, path="src/napari_phasors/_tests/test_data/test_file.ptu"
    )
    assert widget.viewer is viewer
    # Init values
    assert isinstance(widget, AdvancedOptionsWidget)
    assert widget.path == "src/napari_phasors/_tests/test_data/test_file.ptu"
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
    assert viewer.layers[0].name == "test_file.ptu Intensity Image: Channel 0"
    assert viewer.layers[0].data.shape == (1, 256, 256)
    phasor_data = (
        viewer.layers[0].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (131072, 4)
    assert phasor_data["harmonic"].unique().tolist() == [2, 3]
    # Modify frames and harmonics and phasor transform again
    widget.frames.setCurrentIndex(0)
    widget.harmonic_end.setValue(2)
    widget.btn.click()
    assert len(viewer.layers) == 2
    assert (
        viewer.layers[1].name == "test_file.ptu Intensity Image: Channel 0 [1]"
    )
    assert viewer.layers[1].data.shape == (1, 256, 256)
    phasor_data = (
        viewer.layers[1].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (65536, 4)
    assert phasor_data["harmonic"].unique().tolist() == [2]
    # TODO: test dtime parameter


def test_phasor_transform_lsm_widget(make_napari_viewer):
    """Test LsmWidget from PhasorTransfrom widget."""
    viewer = make_napari_viewer()
    PhasorTransform(viewer)
    widget = LsmWidget(
        viewer, path="src/napari_phasors/_tests/test_data/test_file.lsm"
    )
    assert widget.viewer is viewer
    # Init values
    assert isinstance(widget, AdvancedOptionsWidget)
    assert widget.path == "src/napari_phasors/_tests/test_data/test_file.lsm"
    assert widget.reader_options is None
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
    assert viewer.layers[0].name == "test_file.lsm Intensity Image"
    assert viewer.layers[0].data.shape == (512, 512)
    phasor_data = (
        viewer.layers[0].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (524288, 4)
    assert phasor_data["harmonic"].unique().tolist() == [2, 3]
    # Modify harmonics and phasor transform again
    widget.harmonic_end.setValue(2)
    widget.btn.click()
    assert len(viewer.layers) == 2
    assert viewer.layers[1].name == "test_file.lsm Intensity Image [1]"
    assert viewer.layers[1].data.shape == (512, 512)
    phasor_data = (
        viewer.layers[1].metadata["phasor_features_labels_layer"].features
    )
    assert phasor_data.shape == (262144, 4)
    assert phasor_data["harmonic"].unique().tolist() == [2]
