from unittest.mock import Mock, patch

import numpy as np
from napari.layers import Labels
from qtpy.QtWidgets import QDoubleSpinBox, QTableWidget

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

    # Test mode combobox initialization
    mode_combobox = widget.selection_mode_combobox
    assert mode_combobox.count() == 2
    assert mode_combobox.itemText(0) == "Circular Cursor"
    assert mode_combobox.itemText(1) == "Manual Selection"
    assert mode_combobox.currentIndex() == 0  # Circular Cursor is default

    # Test stacked widget has both mode widgets
    assert widget.stacked_widget.count() == 2
    assert (
        widget.stacked_widget.currentIndex() == 0
    )  # Circular mode is default

    # Test circular cursor widget exists
    assert hasattr(widget, 'circular_cursor_widget')
    assert widget.circular_cursor_widget is not None

    # Test manual selection widget exists
    assert hasattr(widget, 'manual_selection_widget')
    assert widget.manual_selection_widget is not None

    # Test manual selection combobox initialization (even though not default)
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

    # Switch to manual selection mode to test manual selection functionality
    widget.selection_mode_combobox.setCurrentIndex(1)

    # Test combobox initialization
    combobox = widget.selection_input_widget.phasor_selection_id_combobox
    assert combobox.count() == 2
    assert combobox.itemText(0) == "None"
    assert combobox.itemText(1) == "MANUAL SELECTION #1"
    assert combobox.currentText() == "None"

    # Test that the metadata does not have any selections yet
    assert (
        "selections" not in intensity_image_layer.metadata
        or len(intensity_image_layer.metadata.get("selections", {})) == 0
    )

    # Make a manual selection
    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    widget.manual_selection_changed(manual_selection)
    assert widget.selection_id == "MANUAL SELECTION #1"

    # Check that the selection was added to the metadata
    assert "selections" in intensity_image_layer.metadata
    assert (
        "MANUAL SELECTION #1" in intensity_image_layer.metadata["selections"]
    )

    # The selection is stored as a 2D array matching the image shape
    selection_map = intensity_image_layer.metadata["selections"][
        "MANUAL SELECTION #1"
    ]
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
    selection_map = intensity_image_layer.metadata["selections"][
        "new_selection"
    ]
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


def test_selection_mode_switching(make_napari_viewer):
    """Test switching between circular cursor and manual selection modes."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    # Initially should be in circular cursor mode
    assert widget.selection_mode_combobox.currentIndex() == 0
    assert widget.stacked_widget.currentIndex() == 0
    assert not widget.is_manual_selection_mode()

    # Switch to manual selection mode
    widget.selection_mode_combobox.setCurrentIndex(1)
    assert widget.stacked_widget.currentIndex() == 1
    assert widget.is_manual_selection_mode()

    # Switch back to circular cursor mode
    widget.selection_mode_combobox.setCurrentIndex(0)
    assert widget.stacked_widget.currentIndex() == 0
    assert not widget.is_manual_selection_mode()


def test_circular_cursor_widget_initialization(make_napari_viewer):
    """Test the initialization of the CircularCursorWidget."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Basic widget structure tests
    assert widget.viewer == viewer
    assert widget.parent_widget == parent
    assert widget.layout().count() > 0

    # Test table initialization
    assert isinstance(widget.cursor_table, QTableWidget)
    assert (
        widget.cursor_table.columnCount() == 5
    )  # G, S, Radius, Color, Remove
    assert widget.cursor_table.rowCount() == 0  # No cursors initially

    # Test header labels (first 4 columns, last is remove button)
    headers = [
        widget.cursor_table.horizontalHeaderItem(i).text()
        for i in range(4)  # Check first 4 headers
    ]
    assert headers == ['G', 'S', 'Radius', 'Color']

    # Test button initialization
    assert hasattr(widget, 'add_cursor_button')
    assert hasattr(widget, 'clear_all_button')
    assert widget.add_cursor_button.text() == 'Add Cursor'
    assert widget.clear_all_button.text() == 'Clear All'

    # Test internal state
    assert widget._cursors == []
    assert widget._dragging_cursor is None
    assert widget._drag_offset == (0, 0)  # Initialized to (0, 0) not None


def test_circular_cursor_add_cursor(make_napari_viewer):
    """Test adding circular cursors."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Initially no cursors
    assert len(widget._cursors) == 0
    assert widget.cursor_table.rowCount() == 0

    # Add first cursor
    widget._add_cursor()
    assert len(widget._cursors) == 1
    assert widget.cursor_table.rowCount() == 1

    # Check first cursor properties
    cursor = widget._cursors[0]
    assert 'g' in cursor
    assert 's' in cursor
    assert 'radius' in cursor
    assert 'color' in cursor
    assert 'patch' in cursor
    assert cursor['g'] == 0.5
    assert cursor['s'] == 0.5  # Default s value
    assert cursor['radius'] == 0.05  # Default radius

    # Add second cursor
    widget._add_cursor()
    assert len(widget._cursors) == 2
    assert widget.cursor_table.rowCount() == 2

    # Check second cursor has different color (Set1 colormap)
    assert widget._cursors[0]['color'] != widget._cursors[1]['color']


def test_circular_cursor_remove_cursor(make_napari_viewer):
    """Test removing circular cursors."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add some cursors
    widget._add_cursor()
    widget._add_cursor()
    widget._add_cursor()
    assert len(widget._cursors) == 3

    # Remove first cursor
    widget._remove_cursor(0)
    assert len(widget._cursors) == 2
    assert widget.cursor_table.rowCount() == 2

    # Remove all cursors
    widget._remove_cursor(0)
    widget._remove_cursor(0)  # After first removal, next item moves to index 0
    assert len(widget._cursors) == 0
    assert widget.cursor_table.rowCount() == 0


def test_circular_cursor_set1_colormap(make_napari_viewer):
    """Test that cursors use Set1 colormap colors."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add 9 cursors (Set1 has 9 colors)
    for _ in range(9):
        widget._add_cursor()

    # Verify each cursor has a unique color from Set1
    colors = [cursor['color'] for cursor in widget._cursors]
    assert len(colors) == 9
    # Check uniqueness by converting QColor to RGB tuples
    color_tuples = [(c.red(), c.green(), c.blue()) for c in colors]
    assert len(set(color_tuples)) == 9  # All unique

    # Add 10th cursor, should cycle back to first color
    widget._add_cursor()
    assert len(widget._cursors) == 10
    # Color 10 should match color 0 (modulo)
    color_0 = widget._cursors[0]['color']
    color_9 = widget._cursors[9]['color']
    assert color_0.red() == color_9.red()
    assert color_0.green() == color_9.green()
    assert color_0.blue() == color_9.blue()


def test_circular_cursor_table_updates(make_napari_viewer):
    """Test that table updates when cursor parameters change."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add a cursor
    widget._add_cursor()

    # Get table spinboxes
    g_spinbox = widget.cursor_table.cellWidget(0, 0)
    s_spinbox = widget.cursor_table.cellWidget(0, 1)
    radius_spinbox = widget.cursor_table.cellWidget(0, 2)

    assert isinstance(g_spinbox, QDoubleSpinBox)
    assert isinstance(s_spinbox, QDoubleSpinBox)
    assert isinstance(radius_spinbox, QDoubleSpinBox)

    # Check initial values
    assert g_spinbox.value() == 0.5
    assert s_spinbox.value() == 0.5  # Default s value
    assert radius_spinbox.value() == 0.05  # Default radius

    # Change values
    g_spinbox.setValue(0.7)
    s_spinbox.setValue(0.4)
    radius_spinbox.setValue(0.15)

    # Verify cursor data updated
    assert widget._cursors[0]['g'] == 0.7
    assert widget._cursors[0]['s'] == 0.4
    assert widget._cursors[0]['radius'] == 0.15


def test_circular_cursor_last_radius_used(make_napari_viewer):
    """Test that new cursors use the radius from the last cursor."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add first cursor with default radius
    widget._add_cursor()
    assert widget._cursors[0]['radius'] == 0.05

    # Change first cursor radius
    radius_spinbox = widget.cursor_table.cellWidget(0, 2)
    radius_spinbox.setValue(0.25)
    assert widget._cursors[0]['radius'] == 0.25

    # Add second cursor, should use last cursor's radius
    widget._add_cursor()
    assert widget._cursors[1]['radius'] == 0.25


def test_circular_cursor_creates_labels_layer(make_napari_viewer):
    """Test that circular cursors create a labels layer."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add a cursor
    widget._add_cursor()

    # Apply selection should create labels layer
    widget._apply_selection()

    # Check that labels layer was created
    expected_layer_name = (
        f"Selection CIRCULAR CURSOR SELECTION: {intensity_image_layer.name}"
    )
    layer_names = [layer.name for layer in viewer.layers]
    assert expected_layer_name in layer_names

    # Get the labels layer
    labels_layer = viewer.layers[expected_layer_name]
    assert isinstance(labels_layer, Labels)
    assert labels_layer.data.shape == intensity_image_layer.data.shape


def test_circular_cursor_labels_layer_visibility(make_napari_viewer):
    """Test that labels layer visibility is managed when switching modes."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    selection_widget = parent.selection_tab

    # Add a circular cursor and apply
    circular_widget = selection_widget.circular_cursor_widget
    circular_widget._add_cursor()
    circular_widget._apply_selection()

    # Get circular cursor layer
    circular_layer_name = (
        f"Selection CIRCULAR CURSOR SELECTION: {intensity_image_layer.name}"
    )
    circular_layer = viewer.layers[circular_layer_name]
    assert circular_layer.visible is True

    # Switch to manual selection mode
    selection_widget.selection_mode_combobox.setCurrentIndex(1)

    # Circular cursor layer should now be hidden
    assert circular_layer.visible is False

    # Switch back to circular cursor mode
    selection_widget.selection_mode_combobox.setCurrentIndex(0)

    # Circular cursor layer should be visible again
    assert circular_layer.visible is True


def test_circular_cursor_clear_patches(make_napari_viewer):
    """Test clearing all circular cursor patches."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add some cursors
    widget._add_cursor()
    widget._add_cursor()

    # Verify patches exist
    assert len(widget._cursors) == 2
    for cursor in widget._cursors:
        assert cursor['patch'] is not None
        # Patch should be visible initially
        assert cursor['patch'].get_visible()

    # Clear patches
    widget.clear_all_patches()

    # Patches should be removed from axes
    assert len(widget._cursors) == 2
    for cursor in widget._cursors:
        # Patch is removed from axes but cursor data remains
        assert cursor['patch'] is None or not cursor['patch'].get_visible()


def test_circular_cursor_redraw_patches(make_napari_viewer):
    """Test redrawing circular cursor patches."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add cursors
    widget._add_cursor()
    widget._add_cursor()

    # Clear patches
    widget.clear_all_patches()
    for cursor in widget._cursors:
        # After clear, patches should be None or hidden
        assert cursor['patch'] is None or not cursor['patch'].get_visible()

    # Redraw patches
    widget.redraw_all_patches()

    # Patches should be created and visible again
    for cursor in widget._cursors:
        assert cursor['patch'] is not None
        assert cursor['patch'].get_visible()


def test_manual_selection_layers_hidden_in_circular_mode(make_napari_viewer):
    """Test that manual selection layers are hidden when in circular cursor mode."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    selection_widget = parent.selection_tab

    # Switch to manual selection mode
    selection_widget.selection_mode_combobox.setCurrentIndex(1)

    # Make a manual selection
    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    selection_widget.manual_selection_changed(manual_selection)

    # Get manual selection layer
    manual_layer_name = (
        f"Selection MANUAL SELECTION #1: {intensity_image_layer.name}"
    )
    manual_layer = viewer.layers[manual_layer_name]
    assert manual_layer.visible is True

    # Switch to circular cursor mode
    selection_widget.selection_mode_combobox.setCurrentIndex(0)

    # Manual selection layer should be hidden
    assert manual_layer.visible is False

    # Switch back to manual selection mode
    selection_widget.selection_mode_combobox.setCurrentIndex(1)

    # Manual selection layer should be visible again
    assert manual_layer.visible is True


def test_circular_cursor_drag_initialization(make_napari_viewer):
    """Test cursor drag initialization with pick event."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add a cursor
    widget._add_cursor()
    cursor = widget._cursors[0]
    patch = cursor['patch']

    # Create mock pick event
    mock_event = Mock()
    mock_event.artist = patch
    mock_event.mouseevent.xdata = 0.5
    mock_event.mouseevent.ydata = 0.3

    # Initially not dragging
    assert widget._dragging_cursor is None
    # Note: _drag_offset is initialized to (0, 0) not None
    assert widget._drag_offset == (0, 0)

    # Trigger pick event
    widget._on_pick(mock_event)

    # Should now be in dragging state
    assert widget._dragging_cursor == 0
    assert widget._drag_offset is not None
    assert len(widget._drag_offset) == 2


def test_circular_cursor_drag_motion(make_napari_viewer):
    """Test cursor motion during drag."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add a cursor at initial position
    widget._add_cursor()
    initial_g = widget._cursors[0]['g']
    initial_s = widget._cursors[0]['s']

    # Simulate picking the cursor
    widget._dragging_cursor = 0
    widget._drag_offset = (0, 0)

    # Create mock motion event
    mock_event = Mock()
    mock_event.xdata = 0.7  # New G position
    mock_event.ydata = 0.4  # New S position

    # Trigger motion event
    widget._on_motion(mock_event)

    # Cursor position should be updated
    assert widget._cursors[0]['g'] == 0.7
    assert widget._cursors[0]['s'] == 0.4
    assert widget._cursors[0]['g'] != initial_g
    assert widget._cursors[0]['s'] != initial_s

    # Table should also be updated
    g_spinbox = widget.cursor_table.cellWidget(0, 0)
    s_spinbox = widget.cursor_table.cellWidget(0, 1)
    assert g_spinbox.value() == 0.7
    assert s_spinbox.value() == 0.4


def test_circular_cursor_drag_release(make_napari_viewer):
    """Test cursor drag release."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add a cursor and simulate dragging
    widget._add_cursor()
    widget._dragging_cursor = 0
    widget._drag_offset = (0, 0)

    # Create mock release event
    mock_event = Mock()

    # Trigger release event
    widget._on_release(mock_event)

    # Should no longer be in dragging state
    assert widget._dragging_cursor is None
    # Note: _drag_offset remains (0, 0) after release
    assert widget._drag_offset == (0, 0)


def test_circular_cursor_drag_without_pick(make_napari_viewer):
    """Test that motion without picking doesn't move cursor."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add a cursor
    widget._add_cursor()
    initial_g = widget._cursors[0]['g']
    initial_s = widget._cursors[0]['s']

    # Create mock motion event WITHOUT picking first
    mock_event = Mock()
    mock_event.xdata = 0.9
    mock_event.ydata = 0.5

    # Trigger motion event
    widget._on_motion(mock_event)

    # Cursor position should NOT change
    assert widget._cursors[0]['g'] == initial_g
    assert widget._cursors[0]['s'] == initial_s


def test_circular_cursor_drag_updates_patch_position(make_napari_viewer):
    """Test that dragging updates the patch center position."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add a cursor
    widget._add_cursor()
    patch = widget._cursors[0]['patch']
    initial_center = patch.center

    # Simulate dragging
    widget._dragging_cursor = 0
    widget._drag_offset = (0, 0)

    # Move cursor
    mock_event = Mock()
    mock_event.xdata = 0.8
    mock_event.ydata = 0.45

    widget._on_motion(mock_event)

    # Patch center should be updated
    new_center = patch.center
    assert new_center != initial_center
    assert new_center == (0.8, 0.45)


def test_circular_cursor_no_auto_apply_during_drag(make_napari_viewer):
    """Test that selection is not auto-applied during drag operations."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.circular_cursor_widget

    # Add a cursor
    widget._add_cursor()

    # Mock _apply_selection to track calls
    with patch.object(widget, '_apply_selection') as mock_apply:
        # Simulate dragging
        widget._dragging_cursor = 0
        widget._drag_offset = (0, 0)

        # Change cursor parameter via table spinbox during drag
        g_spinbox = widget.cursor_table.cellWidget(0, 0)
        g_spinbox.setValue(0.8)

        # _apply_selection should NOT be called during drag
        mock_apply.assert_not_called()

        # End drag
        mock_event = Mock()
        widget._on_release(mock_event)

        # Now parameter changes should trigger apply
        # Note: first call was from setValue(0.8) after drag ended
        initial_call_count = mock_apply.call_count
        g_spinbox.setValue(0.7)
        # Should have one more call after the second setValue
        assert mock_apply.call_count == initial_call_count + 1
