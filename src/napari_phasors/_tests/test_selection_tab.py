from unittest.mock import Mock, patch

import numpy as np
from napari.layers import Labels

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.plotter import PlotterWidget


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

    # Test that the metadata does not have any selections yet
    assert "selections" not in intensity_image_layer.metadata or len(
        intensity_image_layer.metadata.get("selections", {})
    ) == 0

    # Make a manual selection
    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    widget.manual_selection_changed(manual_selection)
    assert widget.selection_id == "MANUAL SELECTION #1"

    # Check that the selection was added to the metadata
    assert "selections" in intensity_image_layer.metadata
    assert "MANUAL SELECTION #1" in intensity_image_layer.metadata["selections"]

    # The selection is stored as a 2D array matching the image shape
    selection_map = intensity_image_layer.metadata["selections"]["MANUAL SELECTION #1"]
    assert isinstance(selection_map, np.ndarray)


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


def test_ensure_selection_storage_valid(make_napari_viewer):
    """Test _ensure_selection_storage with valid selection name."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    # Add new selection ID
    widget._ensure_selection_storage("new_selection")

    # Should add new selection to metadata
    assert "selections" in intensity_image_layer.metadata
    assert "new_selection" in intensity_image_layer.metadata["selections"]
    selection_map = intensity_image_layer.metadata["selections"]["new_selection"]
    assert isinstance(selection_map, np.ndarray)
    assert np.all(selection_map == 0)  # Initially all zeros


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
            widget, '_ensure_selection_storage'
        ) as mock_ensure_storage,
        patch.object(widget, 'update_phasors_layer') as mock_update_layer,
    ):

        # Set the flag to simulate during update
        parent._updating_plot = True

        result = widget.manual_selection_changed([1, 2, 3])

        # Should return None and not call any methods
        assert result is None
        mock_ensure_storage.assert_not_called()
        mock_update_layer.assert_not_called()


def test_manual_selection_changed_while_switching(make_napari_viewer):
    """Test manual_selection_changed while switching selection IDs."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    # Mock the methods that would be called if not switching
    with (
        patch.object(
            widget, '_ensure_selection_storage'
        ) as mock_ensure_storage,
        patch.object(widget, 'update_phasors_layer') as mock_update_layer,
    ):

        # Set the flag to simulate switching selection ID
        widget._switching_selection_id = True

        result = widget.manual_selection_changed([1, 2, 3])

        # Should return None and not call any methods
        assert result is None
        mock_ensure_storage.assert_not_called()
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
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    # Add existing manual selections to test incrementing
    intensity_image_layer.metadata["selections"] = {
        "MANUAL SELECTION #1": np.zeros((10, 10), dtype=np.uint32),
        "MANUAL SELECTION #2": np.zeros((10, 10), dtype=np.uint32),
    }

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
    widget = parent.selection_tab

    # Mock methods that would be called if layer was available
    with (
        patch(
            'napari_phasors.selection_tab.colormap_to_dict'
        ) as mock_colormap_to_dict,
    ):

        result = widget.create_phasors_selected_layer()

        # Should return early and not call any methods
        assert result is None
        mock_colormap_to_dict.assert_not_called()


@patch('napari_phasors.selection_tab.colormap_to_dict')
def test_create_phasors_selected_layer_with_data(
    mock_colormap_to_dict, make_napari_viewer
):
    """Test create_phasors_selected_layer with actual data."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    parent._colormap = Mock()
    parent._colormap.N = 10
    widget = parent.selection_tab

    # Set up a valid selection ID first - use _ensure_selection_storage instead
    widget._ensure_selection_storage("custom_selection")
    widget.selection_id = "custom_selection"

    # Mock return values
    mock_colormap_to_dict.return_value = {1: [1, 0, 0], 2: [0, 1, 0]}

    widget.create_phasors_selected_layer()

    # Should create new layer and add to viewer
    mock_colormap_to_dict.assert_called_once()

    # Check if layer was added to viewer
    layer_names = [layer.name for layer in viewer.layers]
    assert (
        f"Selection custom_selection: {intensity_image_layer.name}"
        in layer_names
    )


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
    assert f"Selection MANUAL SELECTION #1: {intensity_image_layer.name}" in [
        layer.name for layer in viewer.layers
    ]

    # Store the original layer data for comparison
    selection_layer = viewer.layers[
        f"Selection MANUAL SELECTION #1: {intensity_image_layer.name}"
    ]
    original_data = selection_layer.data.copy()

    # Delete the layer from the viewer
    viewer.layers.remove(
        f"Selection MANUAL SELECTION #1: {intensity_image_layer.name}"
    )
    assert (
        f"Selection MANUAL SELECTION #1: {intensity_image_layer.name}"
        not in [layer.name for layer in viewer.layers]
    )

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
    assert f"Selection MANUAL SELECTION #1: {intensity_image_layer.name}" in [
        layer.name for layer in viewer.layers
    ]

    # Check that the recreated layer has the same values as before
    recreated_layer = viewer.layers[
        f"Selection MANUAL SELECTION #1: {intensity_image_layer.name}"
    ]
    np.testing.assert_array_equal(recreated_layer.data, original_data)
