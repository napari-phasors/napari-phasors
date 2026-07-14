from unittest.mock import Mock, patch

import numpy as np
import pytest
from napari.layers import Labels
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QLabel

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.plotter import PlotterWidget


def _visible_rows(cw):
    """Number of cursor rows currently shown (current-harmonic cursors)."""
    return sum(not c['row'].isHidden() for c in cw._cursors)


# ---------------------------------------------------------------------------
# SelectionWidget structure / modes
# ---------------------------------------------------------------------------


def test_selection_widget_initialization_values(make_viewer_model, qtbot):
    """Test the initialization of the SelectionWidget."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    assert widget.viewer == viewer
    assert widget.parent_widget == parent
    assert widget.layout().count() > 0

    assert widget._current_selection_id == "None"
    assert widget.selection_id is None
    assert widget._phasors_selected_layer is None

    # Mode combobox now has three options.
    mode_combobox = widget.selection_mode_combobox
    assert mode_combobox.count() == 3
    assert mode_combobox.itemText(0) == "Cursor Selection"
    assert mode_combobox.itemText(1) == "Automatic Clustering"
    assert mode_combobox.itemText(2) == "Manual Selection"
    assert mode_combobox.currentIndex() == 0  # Cursor Selection is default

    assert widget.stacked_widget.count() == 3
    assert widget.stacked_widget.currentIndex() == 0

    assert hasattr(widget, 'cursor_selection_widget')
    assert widget.cursor_selection_widget is not None
    assert hasattr(widget, 'manual_selection_widget')
    assert widget.manual_selection_widget is not None
    assert hasattr(widget, 'automatic_clustering_widget')
    assert widget.automatic_clustering_widget is not None

    combobox = widget.selection_input_widget.phasor_selection_id_combobox
    assert combobox.count() == 2
    assert combobox.itemText(0) == "None"
    assert combobox.itemText(1) == "MANUAL SELECTION #1"
    assert combobox.currentText() == "None"


def test_selection_widget_with_layer_data(make_viewer_model, qtbot):
    """Test selection widget behavior with actual layer data."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    assert widget._current_selection_id == "None"
    assert widget.selection_id is None
    assert widget._phasors_selected_layer is None

    widget.selection_mode_combobox.setCurrentText("Manual Selection")

    combobox = widget.selection_input_widget.phasor_selection_id_combobox
    assert combobox.count() == 2
    assert combobox.itemText(0) == "None"
    assert combobox.currentText() == "None"

    assert (
        "selections" not in intensity_image_layer.metadata
        or len(intensity_image_layer.metadata.get("selections", {})) == 0
    )

    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    widget.manual_selection_changed(manual_selection)
    assert widget.selection_id == "MANUAL SELECTION #1"

    assert f"MANUAL SELECTION #1: {intensity_image_layer.name}" in [
        layer.name for layer in viewer.layers
    ]


def test_selection_id_property_getter(make_viewer_model, qtbot):
    """Test the selection_id property getter."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    assert widget.selection_id is None

    widget.selection_input_widget.phasor_selection_id_combobox.clear()
    assert widget.selection_id is None

    widget.selection_input_widget.phasor_selection_id_combobox.addItem(
        "test_selection"
    )
    widget.selection_input_widget.phasor_selection_id_combobox.setCurrentText(
        "test_selection"
    )
    assert widget.selection_id == "test_selection"

    widget.selection_input_widget.phasor_selection_id_combobox.setCurrentText(
        ""
    )
    assert widget.selection_id is None


def test_find_phasors_layer_by_name(make_viewer_model, qtbot):
    """Test finding phasors layer by name."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    test_layer = Labels(np.zeros((10, 10), dtype=int), name="test_layer")
    viewer.add_layer(test_layer)

    found_layer = widget._find_phasors_layer_by_name("test_layer")
    assert found_layer == test_layer

    not_found = widget._find_phasors_layer_by_name("non_existing")
    assert not_found is None


def test_get_next_available_selection_with_layer(make_viewer_model, qtbot):
    """Test _get_next_available_selection_id."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    result = widget._get_next_available_selection_id()
    assert result == "MANUAL SELECTION #1"

    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    widget.manual_selection_changed(np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]))
    assert widget.selection_id == "MANUAL SELECTION #1"

    result = widget._get_next_available_selection_id()
    assert result == "MANUAL SELECTION #2"

    widget.manual_selection_changed(np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0]))
    assert widget.selection_id == "MANUAL SELECTION #1"

    widget.selection_input_widget.phasor_selection_id_combobox.setCurrentText(
        "None"
    )
    assert widget.selection_id is None

    widget.manual_selection_changed(np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0]))
    assert widget.selection_id == "MANUAL SELECTION #2"

    result = widget._get_next_available_selection_id()
    assert result == "MANUAL SELECTION #3"


def test_update_phasor_plot_no_layer(make_viewer_model, qtbot):
    """Test update_phasor_plot_with_selection_id when no layer is available."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    with patch.object(parent, 'plot') as mock_plot:
        result = widget.update_phasor_plot_with_selection_id("test_selection")
        assert result is None
        mock_plot.assert_not_called()


def test_update_phasor_plot_during_update(make_viewer_model, qtbot):
    """Test update_phasor_plot_with_selection_id during plot update."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    parent._updating_plot = True
    widget = parent.selection_tab

    with patch.object(parent, 'plot') as mock_plot:
        result = widget.update_phasor_plot_with_selection_id("test_selection")
        assert result is None
        mock_plot.assert_not_called()


def test_create_phasors_selected_layer_no_layer(make_viewer_model, qtbot):
    """Test create_phasors_selected_layer when no layer is available."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    with patch(
        'napari_phasors.selection_tab.colormap_to_dict'
    ) as mock_colormap_to_dict:
        result = widget.create_phasors_selected_layer()
        assert result is None
        mock_colormap_to_dict.assert_not_called()


@patch('napari_phasors.selection_tab.colormap_to_dict')
def test_create_phasors_selected_layer_with_data(
    mock_colormap_to_dict, make_viewer_model, qtbot
):
    """Test create_phasors_selected_layer with actual data."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    parent._colormap = Mock()
    parent._colormap.N = 10
    widget = parent.selection_tab

    widget.selection_mode_combobox.setCurrentText("Manual Selection")
    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    widget.manual_selection_changed(manual_selection)
    widget.selection_id = "custom_selection"

    mock_colormap_to_dict.return_value = {1: [1, 0, 0], 2: [0, 1, 0]}

    widget.create_phasors_selected_layer()

    assert mock_colormap_to_dict.call_count >= 1
    layer_names = [layer.name for layer in viewer.layers]
    assert f"custom_selection: {intensity_image_layer.name}" in layer_names


def test_no_selection_processing_during_plot_update(make_viewer_model, qtbot):
    """Test that selection processing is skipped during plot updates."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    parent._updating_plot = True
    widget = parent.selection_tab

    assert widget.manual_selection_changed([1, 2, 3]) is None
    assert widget.update_phasor_plot_with_selection_id("test") is None


def test_selection_mode_switching(make_viewer_model, qtbot):
    """Test switching between cursor, clustering and manual modes."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    assert widget.selection_mode_combobox.currentIndex() == 0
    assert widget.stacked_widget.currentIndex() == 0
    assert not widget.is_manual_selection_mode()

    widget.selection_mode_combobox.setCurrentText("Automatic Clustering")
    assert widget.stacked_widget.currentIndex() == 1
    assert not widget.is_manual_selection_mode()

    widget.selection_mode_combobox.setCurrentText("Manual Selection")
    assert widget.stacked_widget.currentIndex() == 2
    assert widget.is_manual_selection_mode()

    widget.selection_mode_combobox.setCurrentText("Cursor Selection")
    assert widget.stacked_widget.currentIndex() == 0
    assert not widget.is_manual_selection_mode()


def test_selection_mode_changed_all_modes(make_viewer_model, qtbot):
    """Switching across every selection mode index exercises each branch."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    sel = parent.selection_tab
    # 0=cursor, 1=clustering, 2=manual
    for index in (1, 2, 0):
        sel._on_selection_mode_changed(index)
        assert sel.stacked_widget.currentIndex() == index


# ---------------------------------------------------------------------------
# CursorSelectionWidget - structure & adding cursors
# ---------------------------------------------------------------------------


def test_cursor_selection_widget_initialization(make_viewer_model, qtbot):
    """Test the initialization of the CursorSelectionWidget."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    assert widget.viewer == viewer
    assert widget.parent_widget == parent
    assert widget.layout().count() > 0

    assert hasattr(widget, 'add_cursor_button')
    assert hasattr(widget, 'calculate_button')
    assert hasattr(widget, 'autoupdate_check')
    assert "Add Cursor" in widget.add_cursor_button.text()
    assert "Calculate" in widget.calculate_button.text()
    assert widget.autoupdate_check.text() == 'Autoupdate'
    assert not widget.autoupdate_check.isChecked()

    assert widget._cursors == []
    assert widget._dragging_cursor is None
    assert widget._drag_offset == (0, 0)
    assert not widget._autoupdate_enabled


def test_cursor_add_circular(make_viewer_model, qtbot):
    """Test adding circular cursors (the default shape)."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    assert len(widget._cursors) == 0

    widget._add_cursor()
    assert len(widget._cursors) == 1
    assert _visible_rows(widget) == 1

    cursor = widget._cursors[0]
    assert cursor['type'] == 'circular'
    assert {'g', 's', 'radius', 'color', 'patch'} <= set(cursor)
    xlim = parent.canvas_widget.axes.get_xlim()
    ylim = parent.canvas_widget.axes.get_ylim()
    expected_g = max(-1.5, min(1.5, (xlim[0] + xlim[1]) / 2.0))
    expected_s = max(-1.5, min(1.5, (ylim[0] + ylim[1]) / 2.0))
    assert np.isclose(cursor['g'], expected_g)
    assert np.isclose(cursor['s'], expected_s)
    assert cursor['radius'] == 0.05

    widget._add_cursor()
    assert len(widget._cursors) == 2
    assert widget._cursors[0]['color'] != widget._cursors[1]['color']


def test_cursor_add_elliptic_defaults(make_viewer_model, qtbot):
    """Elliptic cursors keep their distinct radius defaults."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(cursor_type="elliptic")
    cursor = widget._cursors[0]
    assert cursor['type'] == 'elliptic'
    assert cursor['radius'] == 0.1
    assert cursor['radius_minor'] == 0.05
    assert cursor['angle'] == 0.0


def test_cursor_add_polar_defaults(make_viewer_model, qtbot):
    """Polar cursors derive phase/modulation from the axes center."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(cursor_type="polar")
    cursor = widget._cursors[0]
    assert cursor['type'] == 'polar'

    xlim = parent.canvas_widget.axes.get_xlim()
    ylim = parent.canvas_widget.axes.get_ylim()
    center_g = (xlim[0] + xlim[1]) / 2.0
    center_s = (ylim[0] + ylim[1]) / 2.0
    center_phase = np.rad2deg(np.arctan2(center_s, center_g))
    center_modulation = np.sqrt(center_g**2 + center_s**2)
    expected_mod_min = max(0.0, min(1.0, center_modulation - 0.1))
    expected_mod_max = max(0.0, min(1.0, center_modulation + 0.1))

    assert np.allclose(cursor['phase_min'], center_phase - 10.0)
    assert np.allclose(cursor['phase_max'], center_phase + 10.0)
    assert np.allclose(cursor['modulation_min'], expected_mod_min)
    assert np.allclose(cursor['modulation_max'], expected_mod_max)


def test_cursor_type_change_field_visibility(make_viewer_model, qtbot):
    """Changing the shape combobox shows/hides the matching editor fields."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    cursor = widget._cursors[0]
    combo = cursor['type_combo']
    assert isinstance(combo, QComboBox)

    # Circular: center fields shown, others hidden.
    assert cursor['center_widget'].isVisibleTo(cursor['detail'])
    assert not cursor['elliptic_widget'].isVisibleTo(cursor['detail'])
    assert not cursor['polar_widget'].isVisibleTo(cursor['detail'])

    # Switch to elliptical.
    combo.setCurrentIndex(combo.findData("elliptic"))
    assert cursor['type'] == 'elliptic'
    assert cursor['center_widget'].isVisibleTo(cursor['detail'])
    assert cursor['elliptic_widget'].isVisibleTo(cursor['detail'])
    assert not cursor['polar_widget'].isVisibleTo(cursor['detail'])

    # Switch to polar.
    combo.setCurrentIndex(combo.findData("polar"))
    assert cursor['type'] == 'polar'
    assert not cursor['center_widget'].isVisibleTo(cursor['detail'])
    assert not cursor['elliptic_widget'].isVisibleTo(cursor['detail'])
    assert cursor['polar_widget'].isVisibleTo(cursor['detail'])


def test_cursor_selection_follows_add_and_row_click(make_viewer_model, qtbot):
    """Adding a cursor selects it; left-clicking a row selects that cursor."""
    viewer = make_viewer_model()
    viewer.add_layer(create_image_layer_with_phasors())
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    # No cursors yet: the editor is hidden.
    assert widget._selected_cursor is None
    assert not widget._editor_box.isVisible()

    # Adding a cursor selects it and shows its editor page.
    widget._add_cursor()
    first = widget._cursors[0]
    assert widget._selected_cursor is first
    assert widget._details_stack.currentWidget() is first['detail']
    assert first['row'].property("selected")

    # Adding a second cursor moves the selection to the new cursor.
    widget._add_cursor()
    second = widget._cursors[1]
    assert widget._selected_cursor is second
    assert not first['row'].property("selected")
    assert second['row'].property("selected")

    # Left-clicking the first cursor's row selects it again (exercises
    # ClickableFrame.mousePressEvent -> clicked -> _select_cursor).
    qtbot.mouseClick(first['row'], Qt.LeftButton)
    assert widget._selected_cursor is first
    assert first['row'].property("selected")
    assert widget._details_stack.currentWidget() is first['detail']
    # The editor title reflects the selected cursor's number and shape.
    assert "Circular" in widget._editor_box.title()


def test_select_cursor_ignores_foreign_cursor(make_viewer_model, qtbot):
    """``_select_cursor`` ignores a cursor dict that is not in the list."""
    viewer = make_viewer_model()
    viewer.add_layer(create_image_layer_with_phasors())
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    selected = widget._selected_cursor
    assert selected is not None

    # Passing a dict that is not one of the widget's cursors is a no-op.
    widget._select_cursor({'not': 'a real cursor'})
    assert widget._selected_cursor is selected


def test_refresh_editor_title_without_selection(make_viewer_model, qtbot):
    """``_refresh_editor_title`` is a no-op when nothing is selected."""
    viewer = make_viewer_model()
    viewer.add_layer(create_image_layer_with_phasors())
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    # No cursors -> no selection: must not raise.
    assert widget._selected_cursor is None
    widget._refresh_editor_title()


def test_editor_title_updates_with_shape(make_viewer_model, qtbot):
    """The editor title tracks the selected cursor's shape."""
    viewer = make_viewer_model()
    viewer.add_layer(create_image_layer_with_phasors())
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    cursor = widget._cursors[0]
    assert "Circular" in widget._editor_box.title()

    combo = cursor['type_combo']
    combo.setCurrentIndex(combo.findData("polar"))
    assert "Polar" in widget._editor_box.title()


def test_make_spinbox_default_width_ref(make_viewer_model, qtbot):
    """``_make_spinbox`` sizes itself from its own range when no ref given."""
    from napari_phasors.selection_tab import CursorSelectionWidget

    spin = CursorSelectionWidget._make_spinbox(-1.5, 1.5, 0.5, 2, 0.01)
    assert isinstance(spin, QDoubleSpinBox)
    assert spin.value() == 0.5
    # A width was derived and applied as a hard minimum (default ref path).
    assert spin.minimumWidth() >= 70


def test_cursor_remove(make_viewer_model, qtbot):
    """Test removing cursors by index."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    widget._add_cursor()
    widget._add_cursor()
    assert len(widget._cursors) == 3

    widget._remove_cursor(0)
    assert len(widget._cursors) == 2
    assert _visible_rows(widget) == 2

    widget._remove_cursor(0)
    widget._remove_cursor(0)
    assert len(widget._cursors) == 0


def test_cursor_set1_colormap(make_viewer_model, qtbot):
    """Test that cursors cycle through the Set1 colormap colors."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    for _ in range(9):
        widget._add_cursor()

    colors = [cursor['color'] for cursor in widget._cursors]
    color_tuples = [(c.red(), c.green(), c.blue()) for c in colors]
    assert len(set(color_tuples)) == 9

    widget._add_cursor()
    assert len(widget._cursors) == 10
    color_0 = widget._cursors[0]['color']
    color_9 = widget._cursors[9]['color']
    assert (color_0.red(), color_0.green(), color_0.blue()) == (
        color_9.red(),
        color_9.green(),
        color_9.blue(),
    )


def test_cursor_row_spinbox_updates(make_viewer_model, qtbot):
    """Test that editing the row spinboxes updates the cursor data."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    cursor = widget._cursors[0]
    assert isinstance(cursor['g_spin'], QDoubleSpinBox)

    cursor['g_spin'].setValue(0.7)
    cursor['s_spin'].setValue(0.4)
    cursor['radius_spin'].setValue(0.15)

    assert cursor['g'] == 0.7
    assert cursor['s'] == 0.4
    assert cursor['radius'] == 0.15


def test_cursor_last_radius_used(make_viewer_model, qtbot):
    """Test that new circular cursors reuse the last cursor's radius."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    assert widget._cursors[0]['radius'] == 0.05

    widget._cursors[0]['radius_spin'].setValue(0.25)
    assert widget._cursors[0]['radius'] == 0.25

    widget._add_cursor()
    assert widget._cursors[1]['radius'] == 0.25


def test_cursor_creates_labels_layer(make_viewer_model, qtbot):
    """Test that cursors create the combined labels layer on Calculate."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()

    expected_layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    assert expected_layer_name not in [ly.name for ly in viewer.layers]

    widget.calculate_button.click()

    assert expected_layer_name in [ly.name for ly in viewer.layers]
    labels_layer = viewer.layers[expected_layer_name]
    assert isinstance(labels_layer, Labels)
    assert labels_layer.data.shape == intensity_image_layer.data.shape


def test_cursor_combined_layer_mixed_shapes(make_viewer_model, qtbot):
    """A table mixing shapes builds a single combined selection layer."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(cursor_type="circular", g=0.5, s=0.5, radius=0.4)
    widget._add_cursor(cursor_type="polar")
    widget._add_cursor(
        cursor_type="elliptic", g=0.4, s=0.3, radius=0.3, radius_minor=0.2
    )
    widget._apply_selection()

    layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    layer_names = [ly.name for ly in viewer.layers]
    # Exactly one combined layer (no per-shape layers).
    assert layer_names.count(layer_name) == 1
    labels_layer = viewer.layers[layer_name]
    assert (
        labels_layer.metadata['napari_phasors_selection_type']
        == 'cursor_selection'
    )

    # All three shape metadata keys are persisted (Batch Analysis interop).
    selections = intensity_image_layer.metadata["settings"]["selections"]
    assert len(selections["circular_cursors"]) == 1
    assert len(selections["polar_cursors"]) == 1
    assert len(selections["elliptical_cursors"]) == 1


def test_cursor_labels_layer_visibility(make_viewer_model, qtbot):
    """Labels layer visibility is managed when switching modes."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    selection_widget = parent.selection_tab

    cw = selection_widget.cursor_selection_widget
    cw._add_cursor()
    cw._apply_selection()

    layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    layer = viewer.layers[layer_name]
    assert layer.visible is True

    selection_widget.selection_mode_combobox.setCurrentText("Manual Selection")
    assert layer.visible is False
    assert selection_widget.is_manual_selection_mode()

    selection_widget.selection_mode_combobox.setCurrentText("Cursor Selection")
    assert layer.visible is True


def test_cursor_autoupdate_checkbox(make_viewer_model, qtbot):
    """Test the autoupdate toggle behavior."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    assert not widget.autoupdate_check.isChecked()
    assert not widget._autoupdate_enabled
    assert widget.calculate_button.isEnabled()

    widget._add_cursor()
    expected_layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    assert expected_layer_name not in [ly.name for ly in viewer.layers]

    widget.autoupdate_check.setChecked(True)
    assert widget._autoupdate_enabled
    assert not widget.calculate_button.isEnabled()
    assert expected_layer_name in [ly.name for ly in viewer.layers]

    widget.autoupdate_check.setChecked(False)
    assert not widget._autoupdate_enabled
    assert widget.calculate_button.isEnabled()


def test_cursor_calculate_button(make_viewer_model, qtbot):
    """Test calculate button creates a populated labels layer."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(g=0.5, s=0.3, radius=0.1)
    widget._add_cursor(g=0.6, s=0.4, radius=0.15)

    expected_layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    assert expected_layer_name not in [ly.name for ly in viewer.layers]

    widget.calculate_button.click()

    assert expected_layer_name in [ly.name for ly in viewer.layers]
    labels_layer = viewer.layers[expected_layer_name]
    assert isinstance(labels_layer, Labels)
    assert np.any(labels_layer.data > 0)


def test_cursor_count_and_percentage(make_viewer_model, qtbot):
    """Test that count and percentage labels are populated."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(g=0.5, s=0.5, radius=0.5)
    cursor = widget._cursors[0]

    assert isinstance(cursor['count_label'], QLabel)
    assert isinstance(cursor['percentage_label'], QLabel)

    count_text = cursor['count_label'].text()
    percentage_text = cursor['percentage_label'].text()
    assert count_text != "-"
    assert percentage_text != "-"
    assert int(count_text) >= 0
    assert "." in percentage_text or percentage_text.isdigit()

    widget._update_cursor_statistics()
    assert cursor['count_label'].text() != "-"


def test_cursor_statistics_update_on_change(make_viewer_model, qtbot):
    """Statistics update when a cursor parameter changes (no autoupdate)."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    assert not widget._autoupdate_enabled
    widget._add_cursor(g=0.5, s=0.5, radius=0.1)
    cursor = widget._cursors[0]
    assert cursor['count_label'].text() != "-"

    cursor['radius_spin'].setValue(0.5)
    assert cursor['count_label'].text() != "-"


def test_cursor_clear_and_redraw_patches(make_viewer_model, qtbot):
    """Test clearing and redrawing cursor patches."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    widget._add_cursor()
    assert len(widget._cursors) == 2
    for cursor in widget._cursors:
        assert cursor['patch'] is not None

    widget.clear_all_patches()
    for cursor in widget._cursors:
        assert cursor['patch'] is None

    widget.redraw_all_patches()
    for cursor in widget._cursors:
        assert cursor['patch'] is not None
        assert cursor['patch'].get_visible()


def test_cursor_clear_all(make_viewer_model, qtbot):
    """Clear All removes all cursors and the selection layer."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget.autoupdate_check.setChecked(True)
    widget._add_cursor(g=0.5, s=0.5, radius=0.5)
    layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    assert layer_name in [ly.name for ly in viewer.layers]

    widget._clear_all_cursors()
    assert len(widget._cursors) == 0
    assert layer_name not in [ly.name for ly in viewer.layers]


def test_manual_selection_layers_hidden_in_cursor_mode(
    make_viewer_model, qtbot
):
    """Manual selection layers hidden when in cursor selection mode."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    selection_widget = parent.selection_tab

    selection_widget.selection_mode_combobox.setCurrentText("Manual Selection")
    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    selection_widget.manual_selection_changed(manual_selection)

    manual_layer_name = f"MANUAL SELECTION #1: {intensity_image_layer.name}"
    manual_layer = viewer.layers[manual_layer_name]
    assert manual_layer.visible is True

    selection_widget.selection_mode_combobox.setCurrentText("Cursor Selection")
    assert manual_layer.visible is False
    assert not selection_widget.is_manual_selection_mode()

    selection_widget.selection_mode_combobox.setCurrentText("Manual Selection")
    assert manual_layer.visible is True


# ---------------------------------------------------------------------------
# CursorSelectionWidget - dragging
# ---------------------------------------------------------------------------


def test_cursor_drag_initialization(make_viewer_model, qtbot):
    """Test cursor drag initialization with a pick event."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    cursor = widget._cursors[0]

    mock_event = Mock()
    mock_event.artist = cursor['patch']
    mock_event.mouseevent.xdata = 0.5
    mock_event.mouseevent.ydata = 0.3

    assert widget._dragging_cursor is None
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.NoModifier
    ):
        widget._on_pick(mock_event)

    assert widget._dragging_cursor is cursor
    assert len(widget._drag_offset) == 2


def test_cursor_drag_motion(make_viewer_model, qtbot):
    """Test circular cursor motion during drag."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    cursor = widget._cursors[0]
    initial_g = cursor['g']

    widget._dragging_cursor = cursor
    widget._drag_mode = 'translate'
    widget._drag_offset = (0, 0)

    mock_event = Mock()
    mock_event.xdata = 0.7
    mock_event.ydata = 0.4
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.NoModifier
    ):
        widget._on_motion(mock_event)

    assert cursor['g'] == 0.7
    assert cursor['s'] == 0.4
    assert cursor['g'] != initial_g
    assert cursor['g_spin'].value() == 0.7
    assert cursor['s_spin'].value() == 0.4


def test_cursor_drag_release(make_viewer_model, qtbot):
    """Test cursor drag release resets dragging state."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    widget._dragging_cursor = widget._cursors[0]
    widget._drag_offset = (0, 0)

    widget._on_release(Mock())
    assert widget._dragging_cursor is None


def test_cursor_drag_without_pick(make_viewer_model, qtbot):
    """Motion without picking first does not move a cursor."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    cursor = widget._cursors[0]
    initial_g = cursor['g']
    initial_s = cursor['s']

    mock_event = Mock()
    mock_event.xdata = 0.9
    mock_event.ydata = 0.5
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.NoModifier
    ):
        widget._on_motion(mock_event)

    assert cursor['g'] == initial_g
    assert cursor['s'] == initial_s


def test_cursor_drag_updates_patch_position(make_viewer_model, qtbot):
    """Dragging a circular cursor updates the patch center."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    cursor = widget._cursors[0]
    circle_patch = cursor['patch']
    initial_center = circle_patch.center

    widget._dragging_cursor = cursor
    widget._drag_mode = 'translate'
    widget._drag_offset = (0, 0)

    mock_event = Mock()
    mock_event.xdata = 0.8
    mock_event.ydata = 0.45
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.NoModifier
    ):
        widget._on_motion(mock_event)

    assert circle_patch.center != initial_center
    assert circle_patch.center == (0.8, 0.45)


def test_cursor_no_auto_apply_during_drag(make_viewer_model, qtbot):
    """Selection is not auto-applied while dragging."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    cursor = widget._cursors[0]

    with patch.object(widget, '_apply_selection') as mock_apply:
        widget._dragging_cursor = cursor
        widget._drag_offset = (0, 0)
        cursor['g_spin'].setValue(0.8)
        mock_apply.assert_not_called()
        widget._on_release(Mock())


def test_elliptical_cursor_drag_and_rotate_logic(make_viewer_model, qtbot):
    """Test translation and shift-rotation of an elliptical cursor."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(cursor_type="elliptic", g=0.5, s=0.5, angle=0.0)
    cursor = widget._cursors[0]

    widget._dragging_cursor = cursor
    widget._drag_offset = (0, 0)
    widget._drag_mode = 'translate'

    mock_event = Mock()
    mock_event.xdata = 0.6
    mock_event.ydata = 0.6
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.NoModifier
    ):
        widget._on_motion(mock_event)
    assert cursor['g'] == 0.6
    assert cursor['s'] == 0.6

    widget._dragging_cursor = cursor
    widget._drag_mode = 'rotate'
    widget._drag_start_angle = 0.0
    widget._drag_start_cursor_angle = 0.0

    mock_rotate_event = Mock()
    mock_rotate_event.xdata = 0.6
    mock_rotate_event.ydata = 0.7
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.ShiftModifier
    ):
        widget._on_motion(mock_rotate_event)
    assert np.isclose(cursor['angle'], 90.0)


def test_elliptical_cursor_drag_cycle(make_viewer_model, qtbot):
    """Exercise the pick/motion/release drag handlers for an ellipse."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(cursor_type="elliptic")
    cursor = widget._cursors[0]

    pick = Mock()
    pick.artist = cursor['patch']
    pick.mouseevent.xdata = 0.5
    pick.mouseevent.ydata = 0.3
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.NoModifier
    ):
        widget._on_pick(pick)

    motion = Mock()
    motion.xdata = 0.6
    motion.ydata = 0.4
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.NoModifier
    ):
        widget._on_motion(motion)

    widget._on_release(Mock())
    assert len(widget._cursors) == 1


def test_polar_cursor_not_translated_but_edge_editable(
    make_viewer_model, qtbot
):
    """Polar cursors are not translated; their edges are pickable/editable."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(
        cursor_type="polar",
        phase_min=10.0,
        phase_max=30.0,
        modulation_min=0.4,
        modulation_max=0.6,
    )
    cursor = widget._cursors[0]
    # The polar patch is pickable (so its edges can be grabbed).
    assert cursor['patch'].get_picker() is True


def test_polar_closest_edge_detection(make_viewer_model, qtbot):
    """The nearest polar boundary is identified from a click position."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget
    widget._add_cursor(
        cursor_type="polar",
        phase_min=10.0,
        phase_max=30.0,
        modulation_min=0.4,
        modulation_max=0.8,
    )
    cursor = widget._cursors[0]

    def point(angle_deg, r):
        return (
            r * np.cos(np.deg2rad(angle_deg)),
            r * np.sin(np.deg2rad(angle_deg)),
        )

    assert (
        widget._closest_polar_edge(cursor, point(20, 0.8)) == 'modulation_max'
    )
    assert (
        widget._closest_polar_edge(cursor, point(20, 0.4)) == 'modulation_min'
    )
    assert widget._closest_polar_edge(cursor, point(10, 0.6)) == 'phase_min'
    assert widget._closest_polar_edge(cursor, point(30, 0.6)) == 'phase_max'


def test_polar_edge_drag_changes_value_not_center(make_viewer_model, qtbot):
    """Dragging a polar edge changes its bound, leaving the wedge centered."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget
    widget._add_cursor(
        cursor_type="polar",
        phase_min=10.0,
        phase_max=30.0,
        modulation_min=0.4,
        modulation_max=0.6,
    )
    cursor = widget._cursors[0]

    # Pick the outer arc (modulation_max) at 20 degrees, r=0.6.
    pick = Mock()
    pick.artist = cursor['patch']
    pick.mouseevent.xdata = 0.6 * np.cos(np.deg2rad(20))
    pick.mouseevent.ydata = 0.6 * np.sin(np.deg2rad(20))
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.NoModifier
    ):
        widget._on_pick(pick)
    assert widget._drag_mode == 'polar_edge'
    assert widget._polar_edge == 'modulation_max'

    # Drag outward to r=0.85 at the same angle.
    motion = Mock()
    motion.xdata = 0.85 * np.cos(np.deg2rad(20))
    motion.ydata = 0.85 * np.sin(np.deg2rad(20))
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.NoModifier
    ):
        widget._on_motion(motion)

    assert np.isclose(cursor['modulation_max'], 0.85, atol=1e-6)
    assert np.isclose(cursor['mod_max_spin'].value(), 0.85, atol=1e-2)
    # Phase bounds and the modulation_min are untouched.
    assert cursor['phase_min'] == 10.0
    assert cursor['phase_max'] == 30.0
    assert cursor['modulation_min'] == 0.4

    widget._on_release(Mock())
    assert widget._dragging_cursor is None


def test_polar_edge_drag_changes_phase(make_viewer_model, qtbot):
    """Dragging the phase_max edge changes the phase bound to the new angle."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget
    widget._add_cursor(
        cursor_type="polar",
        phase_min=10.0,
        phase_max=30.0,
        modulation_min=0.4,
        modulation_max=0.6,
    )
    cursor = widget._cursors[0]

    pick = Mock()
    pick.artist = cursor['patch']
    pick.mouseevent.xdata = 0.5 * np.cos(np.deg2rad(30))
    pick.mouseevent.ydata = 0.5 * np.sin(np.deg2rad(30))
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.NoModifier
    ):
        widget._on_pick(pick)
    assert widget._polar_edge == 'phase_max'

    motion = Mock()
    motion.xdata = 0.5 * np.cos(np.deg2rad(45))
    motion.ydata = 0.5 * np.sin(np.deg2rad(45))
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.NoModifier
    ):
        widget._on_motion(motion)

    assert np.isclose(cursor['phase_max'], 45.0, atol=1e-6)
    assert cursor['phase_min'] == 10.0


# ---------------------------------------------------------------------------
# CursorSelectionWidget - metadata persistence / harmonic handling
# ---------------------------------------------------------------------------


def test_cursor_storage_in_metadata(make_viewer_model, qtbot):
    """Circular cursors are stored in the circular_cursors metadata key."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(g=0.5, s=0.3, radius=0.1)
    widget._add_cursor(g=0.6, s=0.4, radius=0.15)
    widget._add_cursor(g=0.7, s=0.2, radius=0.08)
    widget._apply_selection()

    selections = intensity_image_layer.metadata["settings"]["selections"]
    cursors = selections["circular_cursors"]
    assert len(cursors) == 3
    assert cursors[0]['g'] == 0.5
    assert cursors[0]['s'] == 0.3
    assert cursors[0]['radius'] == 0.1
    assert len(cursors[0]['color']) == 4
    assert cursors[1]['g'] == 0.6
    assert cursors[2]['radius'] == 0.08


def test_cursor_restoration_from_metadata(make_viewer_model, qtbot):
    """Cursors are restored from metadata on image-layer change."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(g=0.5, s=0.3, radius=0.1)
    widget._add_cursor(g=0.6, s=0.4, radius=0.15)
    widget._apply_selection()

    widget._clear_all_cursors()
    assert len(widget._cursors) == 0

    parent.image_layer_with_phasor_features_combobox.setCurrentText("")
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        intensity_image_layer.name
    )
    parent.on_image_layer_changed()

    assert len(widget._cursors) == 2
    assert abs(widget._cursors[0]['g'] - 0.5) < 0.001
    assert abs(widget._cursors[0]['s'] - 0.3) < 0.001
    assert abs(widget._cursors[0]['radius'] - 0.1) < 0.001
    assert abs(widget._cursors[1]['g'] - 0.6) < 0.001


def test_cursor_metadata_updates_on_change(make_viewer_model, qtbot):
    """Metadata reflects edited cursor parameters after re-applying."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(g=0.5, s=0.3, radius=0.1)
    widget._apply_selection()
    selections = intensity_image_layer.metadata["settings"]["selections"]
    assert selections["circular_cursors"][0]['g'] == 0.5

    widget._cursors[0]['g_spin'].setValue(0.7)
    widget._apply_selection()
    cursors = intensity_image_layer.metadata["settings"]["selections"][
        "circular_cursors"
    ]
    assert cursors[0]['g'] == 0.7
    assert cursors[0]['s'] == 0.3


def test_polar_cursor_restore_from_metadata(make_viewer_model, qtbot):
    """Polar cursors are restored from layer metadata."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata.setdefault("settings", {})["selections"] = {
        "polar_cursors": [
            {
                "phase_min": 10.0,
                "phase_max": 40.0,
                "modulation_min": 0.3,
                "modulation_max": 0.7,
                "color": [255, 0, 0, 255],
                "harmonic": 1,
            }
        ]
    }
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    w = parent.selection_tab.cursor_selection_widget
    w._on_image_layer_changed()
    assert len(w._cursors) == 1
    assert w._cursors[0]['type'] == 'polar'


def test_elliptical_cursor_restore_from_metadata(make_viewer_model, qtbot):
    """Elliptical cursors are restored from layer metadata."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata.setdefault("settings", {})["selections"] = {
        "elliptical_cursors": [
            {
                "g": 0.5,
                "s": 0.3,
                "radius": 0.2,
                "radius_minor": 0.1,
                "angle": 0.0,
                "color": [0, 255, 0, 255],
                "harmonic": 1,
            }
        ]
    }
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    w = parent.selection_tab.cursor_selection_widget
    w._on_image_layer_changed()
    assert len(w._cursors) == 1
    assert w._cursors[0]['type'] == 'elliptic'


def test_cursor_harmonic_storage(make_viewer_model, qtbot):
    """Cursors record the harmonic they were created on."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    parent.harmonic_spinbox.setValue(1)
    widget._add_cursor()
    assert widget._cursors[0]['harmonic'] == 1

    parent.harmonic_spinbox.setValue(2)
    widget._add_cursor()
    assert widget._cursors[1]['harmonic'] == 2


def test_cursor_harmonic_visibility(make_viewer_model, qtbot):
    """Cursor patches only exist for the current harmonic."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    parent.harmonic_spinbox.setValue(1)
    widget._add_cursor()
    assert widget._cursors[0]['patch'] is not None

    parent.harmonic_spinbox.setValue(2)
    widget.on_harmonic_changed()
    assert widget._cursors[0]['patch'] is None

    widget._add_cursor()
    assert widget._cursors[1]['patch'] is not None

    parent.harmonic_spinbox.setValue(1)
    widget.on_harmonic_changed()
    assert widget._cursors[0]['patch'] is not None
    assert widget._cursors[1]['patch'] is None


def test_cursor_harmonic_row_filtering(make_viewer_model, qtbot):
    """Only rows for the current harmonic are shown."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    parent.harmonic_spinbox.setValue(1)
    widget._add_cursor()
    assert _visible_rows(widget) == 1
    assert len(widget._cursors) == 1

    parent.harmonic_spinbox.setValue(2)
    widget.on_harmonic_changed()
    assert _visible_rows(widget) == 0
    assert len(widget._cursors) == 1

    widget._add_cursor()
    assert _visible_rows(widget) == 1
    assert len(widget._cursors) == 2

    parent.harmonic_spinbox.setValue(1)
    widget.on_harmonic_changed()
    assert _visible_rows(widget) == 1
    assert len(widget._cursors) == 2


def test_cursor_harmonic_color_indexing(make_viewer_model, qtbot):
    """Cursor colors are indexed per-harmonic."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    parent.harmonic_spinbox.setValue(1)
    widget._add_cursor()
    h1_color1 = widget._cursors[0]['color']

    parent.harmonic_spinbox.setValue(2)
    widget._add_cursor()
    h2_color1 = widget._cursors[1]['color']

    assert (h1_color1.red(), h1_color1.green(), h1_color1.blue()) == (
        h2_color1.red(),
        h2_color1.green(),
        h2_color1.blue(),
    )

    widget._add_cursor()
    h2_color2 = widget._cursors[2]['color']
    assert h2_color1 != h2_color2


def test_cursor_remove_updates_rows(make_viewer_model, qtbot):
    """Removing a cursor updates the visible rows."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    parent.harmonic_spinbox.setValue(1)
    widget._add_cursor()
    widget._add_cursor()
    widget._add_cursor()
    assert _visible_rows(widget) == 3

    widget._remove_cursor(1)
    assert len(widget._cursors) == 2
    assert _visible_rows(widget) == 2


def test_cursor_labels_layer_per_harmonic(make_viewer_model, qtbot):
    """Labels layer updates as the harmonic changes (with autoupdate)."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget.autoupdate_check.setChecked(True)

    parent.harmonic_spinbox.setValue(1)
    widget._add_cursor(g=0.5, s=0.5, radius=0.5)

    layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    assert layer_name in [ly.name for ly in viewer.layers]
    assert np.count_nonzero(viewer.layers[layer_name].data) > 0

    parent.harmonic_spinbox.setValue(2)
    widget.on_harmonic_changed()
    if layer_name in [ly.name for ly in viewer.layers]:
        assert np.count_nonzero(viewer.layers[layer_name].data) == 0


def test_polar_cursor_clamping_and_validation(make_viewer_model, qtbot):
    """Polar cursor modulation ranges are clamped/validated."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(
        cursor_type="polar", modulation_min=0.8, modulation_max=0.4
    )
    cursor = widget._cursors[-1]
    assert cursor['modulation_min'] == 0.4
    assert cursor['modulation_max'] == 0.8

    widget._add_cursor(
        cursor_type="polar", modulation_min=-0.5, modulation_max=1.5
    )
    cursor = widget._cursors[-1]
    assert cursor['modulation_min'] == 0.0
    assert cursor['modulation_max'] == 1.0

    widget._add_cursor(
        cursor_type="polar", modulation_min=1.2, modulation_max=1.0
    )
    cursor = widget._cursors[-1]
    assert cursor['modulation_min'] == 0.99
    assert cursor['modulation_max'] == 1.0

    widget._add_cursor(
        cursor_type="polar", modulation_min=0.5, modulation_max=0.5
    )
    cursor = widget._cursors[-1]
    assert np.isclose(cursor['modulation_min'], 0.5)
    assert np.isclose(cursor['modulation_max'], 0.51)


def test_cursor_coordinate_clipping(make_viewer_model, qtbot):
    """Circular and elliptical cursor centers are clipped to [-1.5, 1.5]."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(cursor_type="circular", g=2.0, s=-2.0)
    cursor = widget._cursors[-1]
    assert cursor['g'] == 1.5
    assert cursor['s'] == -1.5

    widget._add_cursor(cursor_type="elliptic", g=-2.5, s=3.0)
    cursor = widget._cursors[-1]
    assert cursor['g'] == -1.5
    assert cursor['s'] == 1.5


def test_cursor_added_at_custom_limits(make_viewer_model, qtbot):
    """Cursors are added at the center of custom/zoomed plot limits."""
    viewer = make_viewer_model()
    intensity_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_layer)
    parent = PlotterWidget(viewer)

    parent.canvas_widget.axes.set_xlim([0.2, 0.8])
    parent.canvas_widget.axes.set_ylim([0.1, 0.5])

    xlim = parent.canvas_widget.axes.get_xlim()
    ylim = parent.canvas_widget.axes.get_ylim()
    expected_g = max(-1.5, min(1.5, (xlim[0] + xlim[1]) / 2.0))
    expected_s = max(-1.5, min(1.5, (ylim[0] + ylim[1]) / 2.0))

    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(cursor_type="circular")
    assert np.isclose(widget._cursors[-1]['g'], expected_g)
    assert np.isclose(widget._cursors[-1]['s'], expected_s)

    widget._add_cursor(cursor_type="elliptic")
    assert np.isclose(widget._cursors[-1]['g'], expected_g)
    assert np.isclose(widget._cursors[-1]['s'], expected_s)


@pytest.mark.parametrize("cursor_type", ["circular", "elliptic", "polar"])
def test_cursor_lifecycle(make_viewer_model, qtbot, cursor_type):
    """Exercise add/remove/statistics/harmonic/apply for each shape."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    assert widget._cursors == []
    widget._add_cursor(cursor_type=cursor_type)
    widget._add_cursor(cursor_type=cursor_type)
    assert len(widget._cursors) == 2
    assert widget._cursors[0]['color'] != widget._cursors[1]['color']

    widget._update_cursor_statistics()
    widget.on_harmonic_changed()
    widget._on_cursor_changed(widget._cursors[0])
    widget.redraw_all_patches()
    widget._on_calculate_clicked()

    widget._remove_cursor(0)
    assert len(widget._cursors) == 1
    widget.clear_all_patches()


@pytest.mark.parametrize("cursor_type", ["circular", "elliptic", "polar"])
def test_cursor_autoupdate_applies_on_add(
    make_viewer_model, qtbot, cursor_type
):
    """With autoupdate enabled, adding a cursor applies the selection."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._on_autoupdate_changed(True)
    widget._add_cursor(cursor_type=cursor_type)
    assert len(widget._cursors) == 1
    widget._on_autoupdate_changed(False)
    widget._remove_selection_layer()


def test_cursor_statistics_with_autoupdate_and_hover(make_viewer_model, qtbot):
    """Autoupdate stats + hover cursor handling over real data."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    w = parent.selection_tab.cursor_selection_widget

    w._on_autoupdate_changed(True)
    w._add_cursor(
        cursor_type="elliptic", g=0.5, s=0.3, radius=0.3, radius_minor=0.2
    )
    w._update_cursor_statistics()

    ell_patch = w._cursors[0]['patch']
    ev = Mock()
    ev.inaxes = ell_patch.axes
    ev.xdata = w._cursors[0]['g']
    ev.ydata = w._cursors[0]['s']
    with patch.object(
        QApplication, 'keyboardModifiers', return_value=Qt.NoModifier
    ):
        w._update_hover_cursor(ev)
        ev2 = Mock()
        ev2.inaxes = ell_patch.axes
        ev2.xdata = 5.0
        ev2.ydata = 5.0
        w._update_hover_cursor(ev2)

    w._clear_all_cursors()
    assert len(w._cursors) == 0


def test_cursor_remove_keeps_remaining_functional(make_viewer_model, qtbot):
    """Removing a cursor leaves the remaining row's signals working."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    w = parent.selection_tab.cursor_selection_widget
    w._add_cursor(g=0.4, s=0.3, radius=0.2)
    w._add_cursor(g=0.6, s=0.2, radius=0.2)
    w._remove_cursor(0)
    assert len(w._cursors) == 1
    w._on_cursor_changed(w._cursors[0])


def test_cursor_multi_statistics(make_viewer_model, qtbot):
    """Two overlapping cursors exercise the per-cursor statistics loop."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    w = parent.selection_tab.cursor_selection_widget
    w._on_autoupdate_changed(True)
    w._add_cursor(
        cursor_type="elliptic", g=0.5, s=0.3, radius=0.4, radius_minor=0.3
    )
    w._add_cursor(
        cursor_type="elliptic", g=0.4, s=0.25, radius=0.4, radius_minor=0.3
    )
    w._update_cursor_statistics()
    w._cursors[0]['radius_spin'].setValue(0.5)
    w._update_cursor_statistics()
    assert len(w._cursors) == 2


def test_cursor_row_edit_updates_patch(make_viewer_model, qtbot):
    """Editing a row spinbox updates the cursor and its patch."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    w = parent.selection_tab.cursor_selection_widget
    w._add_cursor(
        cursor_type="elliptic", g=0.5, s=0.3, radius=0.2, radius_minor=0.1
    )
    cursor = w._cursors[0]
    cursor['g_spin'].setValue(0.6)
    cursor['s_spin'].setValue(0.25)
    cursor['radius_spin'].setValue(0.25)
    assert cursor['g'] == 0.6
    assert cursor['patch'] is not None


# ---------------------------------------------------------------------------
# Automatic clustering (unchanged widget) and SelectionWidget integration
# ---------------------------------------------------------------------------


def test_automatic_clustering_widget_initialization(make_viewer_model, qtbot):
    """Test the initialization of the AutomaticClusteringWidget."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.automatic_clustering_widget

    assert widget.viewer == viewer
    assert widget.parent_widget == parent
    assert widget.layout().count() > 0

    assert hasattr(widget, 'clustering_method_combobox')
    assert widget.clustering_method_combobox.count() == 1
    assert (
        widget.clustering_method_combobox.itemText(0)
        == "GMM (Gaussian Mixture Model)"
    )

    assert hasattr(widget, 'num_clusters_spinbox')
    assert widget.num_clusters_spinbox.minimum() == 2
    assert widget.num_clusters_spinbox.maximum() == 100
    assert widget.num_clusters_spinbox.value() == 2

    assert hasattr(widget, 'apply_button')
    assert hasattr(widget, 'clear_button')
    assert widget.apply_button.text() == "Apply Clustering"
    assert widget.clear_button.text() == "Clear Clusters"
    assert not widget.clear_button.isEnabled()

    assert widget._clusters == []
    assert widget._ellipse_patches == []

    from qtpy.QtWidgets import QTableWidget

    assert isinstance(widget.cluster_table, QTableWidget)
    assert widget.cluster_table.columnCount() == 8
    assert widget.cluster_table.rowCount() == 0


def test_automatic_clustering_apply_gmm(make_viewer_model, qtbot):
    """Test applying GMM clustering."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.automatic_clustering_widget

    assert len(widget._clusters) == 0
    widget.num_clusters_spinbox.setValue(3)
    widget._apply_clustering()

    assert len(widget._clusters) == 3
    cluster = widget._clusters[0]
    assert {
        'g',
        's',
        'radius',
        'radius_minor',
        'angle',
        'color',
        'harmonic',
    } <= set(cluster)
    assert cluster['harmonic'] == 1
    assert len(widget._ellipse_patches) == 3
    assert widget.cluster_table.rowCount() == 3
    assert widget.clear_button.isEnabled()

    layer_name = f"Cluster Selection: {intensity_image_layer.name}"
    assert layer_name in [layer.name for layer in viewer.layers]


def test_automatic_clustering_clear_clusters(make_viewer_model, qtbot):
    """Test clearing automatic clusters."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.automatic_clustering_widget

    widget.num_clusters_spinbox.setValue(3)
    widget._apply_clustering()
    assert len(widget._clusters) > 0

    widget._clear_clusters()
    assert len(widget._clusters) == 0
    assert len(widget._ellipse_patches) == 0
    assert not widget.clear_button.isEnabled()
    assert widget.cluster_table.rowCount() == 0


def test_automatic_clustering_harmonic_storage(make_viewer_model, qtbot):
    """Test that automatic clustering stores harmonic information."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.automatic_clustering_widget

    parent.harmonic_spinbox.setValue(2)
    widget.num_clusters_spinbox.setValue(3)
    widget._apply_clustering()

    assert len(widget._clusters) == 3
    for cluster in widget._clusters:
        assert cluster['harmonic'] == 2


def test_automatic_clustering_remove_cluster(make_viewer_model, qtbot):
    """Test removing individual clusters."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.automatic_clustering_widget

    widget.num_clusters_spinbox.setValue(3)
    widget._apply_clustering()
    assert len(widget._clusters) == 3

    widget._remove_cluster(0)
    assert len(widget._clusters) == 2
    assert widget.cluster_table.rowCount() == 2

    widget._remove_cluster(0)
    widget._remove_cluster(0)
    assert len(widget._clusters) == 0
    assert not widget.clear_button.isEnabled()


def test_automatic_clustering_count_and_percentage(make_viewer_model, qtbot):
    """Count and percentage columns populated in clustering table."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.automatic_clustering_widget

    widget.num_clusters_spinbox.setValue(2)
    widget._apply_clustering()

    for row in range(2):
        count_label = widget.cluster_table.cellWidget(row, 5)
        percentage_label = widget.cluster_table.cellWidget(row, 6)
        assert isinstance(count_label, QLabel)
        assert isinstance(percentage_label, QLabel)
        assert count_label.text() != "-"
        assert percentage_label.text() != "-"
        assert int(count_label.text()) >= 0


def test_automatic_clustering_lifecycle(make_viewer_model, qtbot):
    """Apply GMM clustering then exercise statistics/recolour/remove paths."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.automatic_clustering_widget

    widget.num_clusters_spinbox.setValue(3)
    widget._apply_clustering()
    assert len(widget._clusters) == 3

    widget._update_cluster_statistics()
    widget.on_harmonic_changed()
    widget._redraw_cluster_ellipse(0)
    widget._on_cluster_color_changed(0, QColor(255, 0, 0, 255))
    widget._reapply_clustering_to_layers()

    widget._remove_cluster(0)
    assert len(widget._clusters) == 2

    widget._clear_clusters()
    assert len(widget._clusters) == 0


def test_automatic_clustering_reapply_variants(make_viewer_model, qtbot):
    """Re-running clustering with a different cluster count rebuilds."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    w = parent.selection_tab.automatic_clustering_widget
    w.num_clusters_spinbox.setValue(2)
    w._apply_clustering()
    assert len(w._clusters) == 2
    w.num_clusters_spinbox.setValue(4)
    w._apply_clustering()
    assert len(w._clusters) == 4


def test_on_harmonic_changed_only_updates_active_mode(
    make_viewer_model, qtbot
):
    """Harmonic changes only act on the active selection mode."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    selection_widget = parent.selection_tab

    parent.harmonic_spinbox.setValue(1)
    cursor_widget = selection_widget.cursor_selection_widget
    cursor_widget._add_cursor()

    selection_widget.selection_mode_combobox.setCurrentText(
        "Automatic Clustering"
    )
    clustering_widget = selection_widget.automatic_clustering_widget
    clustering_widget.num_clusters_spinbox.setValue(2)
    clustering_widget._apply_clustering()

    parent.harmonic_spinbox.setValue(2)
    selection_widget.on_harmonic_changed()

    selection_widget.selection_mode_combobox.setCurrentText("Cursor Selection")
    parent.harmonic_spinbox.setValue(1)
    selection_widget.on_harmonic_changed()

    assert cursor_widget._cursors[0]['patch'] is not None


def test_labels_layer_visibility_on_tab_toggle(make_viewer_model, qtbot):
    """All selection layers are hidden when the selection tab is hidden."""
    viewer = make_viewer_model()
    intensity_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    widget.cursor_selection_widget._add_cursor()
    widget.cursor_selection_widget._apply_selection()

    widget.selection_mode_combobox.setCurrentText("Manual Selection")
    widget.manual_selection_changed(np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]))

    cursor_layer_name = f"Cursor Selection: {intensity_layer.name}"
    man_layer_name = f"MANUAL SELECTION #1: {intensity_layer.name}"

    assert viewer.layers[man_layer_name].visible is True
    assert viewer.layers[cursor_layer_name].visible is False

    widget._set_labels_layer_visibility(False)
    assert viewer.layers[man_layer_name].visible is False
    assert viewer.layers[cursor_layer_name].visible is False

    widget._set_labels_layer_visibility(True)
    assert viewer.layers[man_layer_name].visible is True
    assert viewer.layers[cursor_layer_name].visible is False

    widget.selection_mode_combobox.setCurrentText("Cursor Selection")
    widget._set_labels_layer_visibility(True)
    assert viewer.layers[man_layer_name].visible is False
    assert viewer.layers[cursor_layer_name].visible is True


def test_selection_widget_image_layer_and_selection_id(
    make_viewer_model, qtbot
):
    """Exercise SelectionWidget layer-change and selection-id update paths."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    sel = parent.selection_tab

    sel._on_image_layer_changed()
    sel.on_selection_id_changed()
    sel.update_phasor_plot_with_selection_id(sel.selection_id)
    assert sel.selection_id in ("", "None", None) or isinstance(
        sel.selection_id, str
    )


def test_selection_widget_manual_overlay_and_selection_id(
    make_viewer_model, qtbot
):
    """Manual selection creates a layer; overlay visibility toggles."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    sel = parent.selection_tab

    sel._on_show_color_overlay(True)

    sel.selection_mode_combobox.setCurrentText("Manual Selection")
    sel.manual_selection_changed(np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]))
    sel.selection_id = "sel_a"
    sel.create_phasors_selected_layer()
    overlay_name = f"sel_a: {layer.name}"
    assert overlay_name in [ly.name for ly in viewer.layers]

    sel._on_show_color_overlay(True)
    assert viewer.layers[overlay_name].visible is True
    sel._on_show_color_overlay(False)
    assert viewer.layers[overlay_name].visible is False

    sel.update_phasor_plot_with_selection_id("sel_a")


def test_recreate_manual_selection_layer(make_viewer_model, qtbot):
    """Recreating a stored manual selection adds a hidden Labels layer once."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    sel = parent.selection_tab

    selection_map = np.zeros(layer.data.shape, dtype=np.uint32)
    selection_map[0, 0] = 1
    sel._recreate_manual_selection_layer("stored_sel", selection_map)

    name = f"stored_sel: {layer.name}"
    assert name in [ly.name for ly in viewer.layers]
    recreated = viewer.layers[name]
    assert recreated.visible is False
    assert recreated.metadata["napari_phasors_selection_type"] == "manual"

    n_layers = len(viewer.layers)
    sel._recreate_manual_selection_layer("stored_sel", selection_map)
    assert len(viewer.layers) == n_layers


# ---------------------------------------------------------------------------
# CursorSelectionWidget - tooltips
# ---------------------------------------------------------------------------


def test_cursor_widget_control_tooltips(make_viewer_model, qtbot):
    """The add/calculate/autoupdate controls expose tooltips."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    assert widget.add_cursor_button.toolTip() != ""
    assert widget.calculate_button.toolTip() != ""
    assert widget.autoupdate_check.toolTip() != ""


def test_cursor_row_parameter_tooltips(make_viewer_model, qtbot):
    """Every per-cursor parameter widget exposes an explanatory tooltip."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    cursor = widget._cursors[0]

    for key in (
        'type_combo',
        'color_button',
        'g_spin',
        's_spin',
        'radius_spin',
        'radius_minor_spin',
        'angle_spin',
        'phase_min_spin',
        'phase_max_spin',
        'mod_min_spin',
        'mod_max_spin',
        'count_label',
        'percentage_label',
        'visibility_button',
        'remove_button',
    ):
        assert cursor[key].toolTip() != "", f"missing tooltip on {key}"


# ---------------------------------------------------------------------------
# CursorSelectionWidget - visibility (eye) toggle
# ---------------------------------------------------------------------------


def test_cursor_visibility_button_defaults(make_viewer_model, qtbot):
    """New cursors are visible by default with a checked eye button."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    cursor = widget._cursors[0]
    assert cursor['visible'] is True
    # The eye button is a plain (non-checkable) button, like the remove "×".
    assert cursor['visibility_button'].isCheckable() is False
    # Plain (un-crossed) white eye icon shown when visible.
    assert not cursor['visibility_button'].icon().isNull()
    assert cursor['visibility_button'].property("eyeCrossed") is False


def test_cursor_visibility_toggle_hides_patch(make_viewer_model, qtbot):
    """Hiding a cursor removes its patch; showing it recreates it."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor()
    cursor = widget._cursors[0]
    assert cursor['patch'] is not None

    cursor['visibility_button'].click()
    assert cursor['visible'] is False
    assert cursor['patch'] is None
    # Crossed-out white eye when hidden (same colour, slashed shape).
    assert cursor['visibility_button'].property("eyeCrossed") is True

    # Showing it again clears the slash and recreates the patch.
    cursor['visibility_button'].click()
    assert cursor['visible'] is True
    assert cursor['patch'] is not None
    assert cursor['visibility_button'].property("eyeCrossed") is False


def test_cursor_visibility_excludes_from_selection(make_viewer_model, qtbot):
    """A hidden cursor is excluded from the computed selection map."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget.autoupdate_check.setChecked(True)
    widget._add_cursor(g=0.5, s=0.5, radius=0.5)
    widget._add_cursor(g=0.4, s=0.4, radius=0.5)

    layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    labels_layer = viewer.layers[layer_name]
    # Both cursors present -> ids {0, 1, 2} possible.
    assert set(np.unique(labels_layer.data)) <= {0, 1, 2}
    assert 2 in np.unique(labels_layer.data)

    # The first cursor's own mask count (independent of overlap).
    g, s = widget._layer_harmonic_arrays(intensity_image_layer, 1)
    cursor0_mask_count = int(
        np.sum(widget._cursor_mask(widget._cursors[0], g, s))
    )

    # Hide the second cursor -> only the first contributes (id 1).
    widget._cursors[1]['visibility_button'].click()
    assert set(np.unique(labels_layer.data)) == {0, 1}
    assert int(np.count_nonzero(labels_layer.data)) == cursor0_mask_count

    # Hidden cursor's statistics are blanked.
    assert widget._cursors[1]['count_label'].text() == "-"
    assert widget._cursors[1]['percentage_label'].text() == "-"

    # Show it again -> id 2 returns.
    widget._cursors[1]['visibility_button'].click()
    assert 2 in np.unique(labels_layer.data)


def test_cursor_visibility_overlap_logic(make_viewer_model, qtbot):
    """Hiding the top cursor reassigns overlapping pixels to the lower one."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget.autoupdate_check.setChecked(True)
    # Two heavily overlapping cursors covering the data.
    widget._add_cursor(g=0.5, s=0.5, radius=0.6)
    widget._add_cursor(g=0.5, s=0.5, radius=0.6)

    layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    labels_layer = viewer.layers[layer_name]
    g, s = widget._layer_harmonic_arrays(intensity_image_layer, 1)
    cursor0_mask = widget._cursor_mask(widget._cursors[0], g, s)

    # With both visible, the second (id 2) overwrites the overlap.
    assert 2 in np.unique(labels_layer.data)

    # Hide the second: overlap pixels fall back to the first cursor (id 1).
    widget._cursors[1]['visibility_button'].click()
    assert set(np.unique(labels_layer.data)) == {0, 1}
    assert int(np.count_nonzero(labels_layer.data)) == int(
        np.sum(cursor0_mask)
    )


def test_cursor_visibility_recompute_when_layer_exists(
    make_viewer_model, qtbot
):
    """Without autoupdate, toggling recomputes if a selection layer exists."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    assert not widget._autoupdate_enabled
    widget._add_cursor(g=0.5, s=0.5, radius=0.5)
    widget.calculate_button.click()

    layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    assert layer_name in [ly.name for ly in viewer.layers]

    # Hiding the only cursor removes the (now empty) selection layer.
    widget._cursors[0]['visibility_button'].click()
    assert layer_name not in [ly.name for ly in viewer.layers]

    # Showing it again recreates the layer.
    widget._cursors[0]['visibility_button'].click()
    assert layer_name in [ly.name for ly in viewer.layers]


def test_cursor_visibility_no_recompute_without_layer(
    make_viewer_model, qtbot
):
    """Toggling with no layer and no autoupdate only updates stats/patch."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(g=0.5, s=0.5, radius=0.5)
    layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    assert layer_name not in [ly.name for ly in viewer.layers]

    widget._cursors[0]['visibility_button'].click()
    # No selection layer is created just by toggling.
    assert layer_name not in [ly.name for ly in viewer.layers]
    assert widget._cursors[0]['patch'] is None


def test_cursor_visibility_persisted_in_metadata(make_viewer_model, qtbot):
    """Cursor visibility is written to the per-shape metadata entries."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.cursor_selection_widget

    widget._add_cursor(cursor_type="circular", g=0.5, s=0.3, radius=0.1)
    widget._add_cursor(cursor_type="circular", g=0.6, s=0.4, radius=0.1)
    widget._cursors[1]['visibility_button'].click()
    widget._apply_selection()

    cursors = intensity_image_layer.metadata["settings"]["selections"][
        "circular_cursors"
    ]
    assert len(cursors) == 2
    assert cursors[0]['visible'] is True
    assert cursors[1]['visible'] is False


def test_cursor_visibility_restored_from_metadata(make_viewer_model, qtbot):
    """A hidden cursor stored in metadata is restored hidden."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata.setdefault("settings", {})["selections"] = {
        "circular_cursors": [
            {
                "g": 0.5,
                "s": 0.3,
                "radius": 0.1,
                "color": [255, 0, 0, 255],
                "visible": False,
            }
        ]
    }
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    w = parent.selection_tab.cursor_selection_widget
    w._on_image_layer_changed()

    assert len(w._cursors) == 1
    assert w._cursors[0]['visible'] is False
    assert w._cursors[0]['visibility_button'].property("eyeCrossed") is True
    assert w._cursors[0]['patch'] is None


def test_cursor_visibility_missing_metadata_defaults_true(
    make_viewer_model, qtbot
):
    """Cursors stored before the visibility feature default to visible."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata.setdefault("settings", {})["selections"] = {
        "circular_cursors": [
            {"g": 0.5, "s": 0.3, "radius": 0.1, "color": [0, 255, 0, 255]}
        ]
    }
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    w = parent.selection_tab.cursor_selection_widget
    w._on_image_layer_changed()

    assert len(w._cursors) == 1
    assert w._cursors[0]['visible'] is True
