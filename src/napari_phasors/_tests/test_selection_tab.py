from unittest.mock import Mock, patch

import numpy as np
from napari.layers import Labels

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.plotter import PlotterWidget
from napari_phasors.selection_tab import DATA_COLUMNS


def test_selection_widget_initialization_values(make_napari_viewer):
    """Test the initialization of the SelectionWidget."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    # Basic widget structure tests
    assert widget.viewer == viewer
    assert widget.parent_widget == parent
    assert widget.layout().count() > 0

    # Test initial attribute values
    assert widget._current_selection_id == "None"
    assert widget.selection_id == None
    assert widget._phasors_selected_layer is None

    # Test combobox initialization
    combobox = widget.selection_input_widget.phasor_selection_id_combobox
    assert combobox.count() == 2
    assert combobox.itemText(0) == "None"
    assert combobox.itemText(1) == "MANUAL SELECTION #1"
    assert combobox.currentText() == "None"


def test_selection_widget_with_layer_data(make_napari_viewer):
    """Test selection widget behavior with actual layer data."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    # Create and add an intensity image layer with phasors
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    # Test initial attribute values
    assert widget._current_selection_id == "None"
    assert widget.selection_id == None
    assert widget._phasors_selected_layer is None

    # Test combobox initialization
    combobox = widget.selection_input_widget.phasor_selection_id_combobox
    assert combobox.count() == 2
    assert combobox.itemText(0) == "None"
    assert combobox.itemText(1) == "MANUAL SELECTION #1"
    assert combobox.currentText() == "None"

    # Test that the features table does not have any selections yet
    labels_layer = intensity_image_layer.metadata[
        "phasor_features_labels_layer"
    ]
    assert "MANUAL SELECTION #1" not in labels_layer.features.columns

    # Make a manual selection
    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    widget.manual_selection_changed(manual_selection)
    assert widget.selection_id == "MANUAL SELECTION #1"

    # Check that the selection was added to the features table
    assert "MANUAL SELECTION #1" in labels_layer.features.columns

    # Create expected column: [1, 0, 1, 0, 1, 0, 0, 0, 0, 0] repeated 3 times
    expected_column = np.concatenate(
        [manual_selection, manual_selection, manual_selection]
    )  # Repeat 3 times for the three harmonics

    # Convert pandas Series to numpy array for comparison
    actual_column = labels_layer.features["MANUAL SELECTION #1"].values
    assert np.array_equal(actual_column, expected_column)


def test_selection_id_property_getter(make_napari_viewer):
    """Test the selection_id property getter."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    # Test with "None" selected
    assert widget.selection_id == None

    # Test with empty combobox
    widget.selection_input_widget.phasor_selection_id_combobox.clear()
    assert widget.selection_id is None

    # Test with valid selection
    widget.selection_input_widget.phasor_selection_id_combobox.addItem(
        "test_selection"
    )
    widget.selection_input_widget.phasor_selection_id_combobox.setCurrentText(
        "test_selection"
    )
    assert widget.selection_id == "test_selection"

    # Test with empty string
    widget.selection_input_widget.phasor_selection_id_combobox.setCurrentText(
        ""
    )
    assert widget.selection_id is None


@patch('napari_phasors.selection_tab.notifications')
def test_selection_id_property_setter_invalid(
    mock_notifications, make_napari_viewer
):
    """Test the selection_id property setter with invalid values."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    # Test setting invalid selection ID (from DATA_COLUMNS)
    widget.selection_id = "label"  # This is in DATA_COLUMNS

    # Should show warning notification
    mock_notifications.WarningNotification.assert_called_once()
    # Selection ID should remain unchanged
    assert widget.selection_id is None


def test_add_selection_id_to_features_valid(make_napari_viewer):
    """Test add_selection_id_to_features with valid column name."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    parent = PlotterWidget(viewer)
    parent._labels_layer_with_phasor_features = intensity_image_layer.metadata[
        "phasor_features_labels_layer"
    ]
    widget = parent.selection_tab

    # Add new selection ID
    widget.add_selection_id_to_features("new_selection")

    # Should add new column to features
    labels_layer = parent._labels_layer_with_phasor_features
    assert "new_selection" in labels_layer.features.columns
    assert len(labels_layer.features["new_selection"]) > 0
    assert all(labels_layer.features["new_selection"] == 0)


def test_find_phasors_layer_by_name(make_napari_viewer):
    """Test finding phasors layer by name."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    # Add test layers
    test_layer = Labels(np.zeros((10, 10), dtype=int), name="test_layer")
    viewer.add_layer(test_layer)

    # Test finding existing layer
    found_layer = widget._find_phasors_layer_by_name("test_layer")
    assert found_layer == test_layer

    # Test finding non-existing layer
    not_found = widget._find_phasors_layer_by_name("non_existing")
    assert not_found is None


def test_manual_selection_changed_during_update(make_napari_viewer):
    """Test manual_selection_changed during plot update."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    # Mock the methods that would be called if not during update
    with (
        patch.object(
            widget, 'add_selection_id_to_features'
        ) as mock_add_selection,
        patch.object(widget, 'update_phasors_layer') as mock_update_layer,
    ):

        # Set the flag to simulate during update
        parent._updating_plot = True

        result = widget.manual_selection_changed([1, 2, 3])

        # Should return None and not call any methods
        assert result is None
        mock_add_selection.assert_not_called()
        mock_update_layer.assert_not_called()


def test_manual_selection_changed_while_switching(make_napari_viewer):
    """Test manual_selection_changed while switching selection IDs."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    # Mock the methods that would be called if not switching
    with (
        patch.object(
            widget, 'add_selection_id_to_features'
        ) as mock_add_selection,
        patch.object(widget, 'update_phasors_layer') as mock_update_layer,
    ):

        # Set the flag to simulate switching selection ID
        widget._switching_selection_id = True

        result = widget.manual_selection_changed([1, 2, 3])

        # Should return None and not call any methods
        assert result is None
        mock_add_selection.assert_not_called()
        mock_update_layer.assert_not_called()


def test_get_next_available_selection_id_no_layer(make_napari_viewer):
    """Test _get_next_available_selection_id when no layer is available."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    result = widget._get_next_available_selection_id()
    assert result == "MANUAL SELECTION #1"

    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    # Make selection to add ID to table
    widget.manual_selection_changed(np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]))
    assert widget.selection_id == "MANUAL SELECTION #1"

    # Add a second call to ensure it increments
    result = widget._get_next_available_selection_id()
    assert result == "MANUAL SELECTION #2"

    # Make another manual selection
    widget.manual_selection_changed(np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0]))
    # selected id should be the same
    assert widget.selection_id == "MANUAL SELECTION #1"

    # Go back to "None" selection
    widget.selection_input_widget.phasor_selection_id_combobox.setCurrentText(
        "None"
    )
    assert widget.selection_id is None

    # Make new selection and check next available ID increments
    widget.manual_selection_changed(np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0]))
    assert widget.selection_id == "MANUAL SELECTION #2"

    # Get next available ID again
    result = widget._get_next_available_selection_id()
    assert result == "MANUAL SELECTION #3"


def test_get_next_available_selection_id_with_existing(make_napari_viewer):
    """Test _get_next_available_selection_id with existing selections."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    parent = PlotterWidget(viewer)
    labels_layer = intensity_image_layer.metadata[
        "phasor_features_labels_layer"
    ]

    # Add existing manual selections to test incrementing
    labels_layer.features["MANUAL SELECTION #1"] = np.zeros(
        len(labels_layer.features)
    )
    labels_layer.features["MANUAL SELECTION #2"] = np.zeros(
        len(labels_layer.features)
    )

    parent._labels_layer_with_phasor_features = labels_layer
    widget = parent.selection_tab

    result = widget._get_next_available_selection_id()
    assert result == "MANUAL SELECTION #3"


def test_update_phasor_plot_no_layer(make_napari_viewer):
    """Test update_phasor_plot_with_selection_id when no layer is available."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    # Mock the plot method that would be called if layer was available
    with patch.object(parent, 'plot') as mock_plot:
        result = widget.update_phasor_plot_with_selection_id("test_selection")

        # Should return early and not call plot
        assert result is None
        mock_plot.assert_not_called()


def test_update_phasor_plot_during_update(make_napari_viewer):
    """Test update_phasor_plot_with_selection_id during plot update."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    parent._updating_plot = True
    widget = parent.selection_tab

    # Mock the plot method that would be called if not during update
    with patch.object(parent, 'plot') as mock_plot:
        result = widget.update_phasor_plot_with_selection_id("test_selection")

        # Should return early and not call plot
        assert result is None
        mock_plot.assert_not_called()


def test_create_phasors_selected_layer_no_layer(make_napari_viewer):
    """Test create_phasors_selected_layer when no layer is available."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    parent._labels_layer_with_phasor_features = None
    widget = parent.selection_tab

    # Mock methods that would be called if layer was available
    with (
        patch('napari_phasors.selection_tab.map_array') as mock_map_array,
        patch(
            'napari_phasors.selection_tab.colormap_to_dict'
        ) as mock_colormap_to_dict,
    ):

        result = widget.create_phasors_selected_layer()

        # Should return early and not call any methods
        assert result is None
        mock_map_array.assert_not_called()
        mock_colormap_to_dict.assert_not_called()


@patch('napari_phasors.selection_tab.map_array')
@patch('napari_phasors.selection_tab.colormap_to_dict')
def test_create_phasors_selected_layer_with_data(
    mock_colormap_to_dict, mock_map_array, make_napari_viewer
):
    """Test create_phasors_selected_layer with actual data."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    parent._labels_layer_with_phasor_features = intensity_image_layer.metadata[
        "phasor_features_labels_layer"
    ]
    parent._colormap = Mock()
    parent._colormap.N = 10
    widget = parent.selection_tab

    # Set up a valid selection ID first
    widget.add_selection_id_to_features("custom_selection")
    widget.selection_id = "custom_selection"

    # Mock return values
    mock_map_array.return_value = np.ones((100, 100), dtype=int)
    mock_colormap_to_dict.return_value = {1: [1, 0, 0], 2: [0, 1, 0]}

    widget.create_phasors_selected_layer()

    # Should create new layer and add to viewer
    mock_map_array.assert_called_once()
    mock_colormap_to_dict.assert_called_once()

    # Check if layer was added to viewer
    layer_names = [layer.name for layer in viewer.layers]
    assert "Selection: custom_selection" in layer_names


def test_no_selection_processing_during_plot_update(make_napari_viewer):
    """Test that selection processing is skipped during plot updates."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    parent._updating_plot = True
    widget = parent.selection_tab

    # These should all return early due to _updating_plot flag
    assert widget.manual_selection_changed([1, 2, 3]) is None
    assert widget.update_phasor_plot_with_selection_id("test") is None


def test_delete_labels_layer_and_recreate(make_napari_viewer):
    """Test that delete_labels_layer_and_recreate works as expected."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    # Make selection to create labels layer
    widget.manual_selection_changed(manual_selection)

    # Check that a layer with the specific name exists
    assert "Selection: MANUAL SELECTION #1" in [
        layer.name for layer in viewer.layers
    ]

    # Store the original layer data for comparison
    selection_layer = viewer.layers["Selection: MANUAL SELECTION #1"]
    original_data = selection_layer.data.copy()

    # Delete the layer from the viewer
    viewer.layers.remove("Selection: MANUAL SELECTION #1")
    assert "Selection: MANUAL SELECTION #1" not in [
        layer.name for layer in viewer.layers
    ]

    # Mock the plot update methods to prevent the error
    with patch.object(
        widget, 'update_phasor_plot_with_selection_id'
    ) as mock_update_plot:
        # Select None ID to trigger recreation
        widget.selection_input_widget.phasor_selection_id_combobox.setCurrentText(
            "None"
        )
        widget.on_selection_id_changed()  # Explicitly trigger the event

        # Select again MANUAL SELECTION #1 to recreate the layer
        widget.selection_input_widget.phasor_selection_id_combobox.setCurrentText(
            "MANUAL SELECTION #1"
        )
        widget.on_selection_id_changed()  # Explicitly trigger the event

    # Check that the layer is recreated
    assert "Selection: MANUAL SELECTION #1" in [
        layer.name for layer in viewer.layers
    ]

    # Check that the recreated layer has the same values as before
    recreated_layer = viewer.layers["Selection: MANUAL SELECTION #1"]
    np.testing.assert_array_equal(recreated_layer.data, original_data)


def test_data_columns_constant():
    """Test that DATA_COLUMNS constant is properly defined."""
    expected_columns = [
        "label",
        "G_original",
        "S_original",
        "G",
        "S",
        "harmonic",
    ]
    assert DATA_COLUMNS == expected_columns
