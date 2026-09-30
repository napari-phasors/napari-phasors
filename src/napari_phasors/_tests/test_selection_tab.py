from unittest.mock import Mock, patch

import numpy as np
import pytest
from napari.layers import Labels
from phasorpy.cluster import phasor_cluster_kmeans
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QColor, QFocusEvent, QValidator
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QSizePolicy,
)

from napari_phasors._tests.test_plotter import (
    assert_run_row_is_pinned,
    create_image_layer_with_phasors,
)
from napari_phasors._utils import analysis_layer_name
from napari_phasors.plotter import PlotterWidget
from napari_phasors.selection_tab import (
    ClickableFrame,
    ColorButton,
    MixedValueSpinBox,
)


def _visible_rows(cw):
    """Number of cursor rows currently shown (current-harmonic cursors)."""
    return sum(not c["row"].isHidden() for c in cw._cursors)


# ---------------------------------------------------------------------------
# SelectionWidget structure / modes
# ---------------------------------------------------------------------------


def _cursor_widget(make_viewer_model, settings=None):
    """Return ``(viewer, layer, plotter, cursor widget)`` for one layer."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    if settings is not None:
        layer.metadata["settings"] = settings
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    return viewer, layer, parent, parent.selection_tab.cursor_selection_widget


def test_selection_widget_initial_state(make_viewer_model, qtbot):
    """A fresh Selection tab, its three modes and their run buttons."""
    from qtpy.QtWidgets import QTableWidget

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

    assert hasattr(widget, "cursor_selection_widget")
    assert widget.cursor_selection_widget is not None
    assert hasattr(widget, "manual_selection_widget")
    assert widget.manual_selection_widget is not None
    assert hasattr(widget, "automatic_clustering_widget")
    assert widget.automatic_clustering_widget is not None

    combobox = widget.selection_input_widget.phasor_selection_id_combobox
    assert combobox.count() == 2
    assert combobox.itemText(0) == "None"
    assert combobox.itemText(1) == "MANUAL SELECTION #1"
    assert combobox.currentText() == "None"

    # The cursor selection widget.
    cursor = widget.cursor_selection_widget
    assert cursor.viewer == viewer
    assert cursor.parent_widget == parent
    assert cursor.layout().count() > 0
    assert hasattr(cursor, "add_cursor_button")
    assert hasattr(cursor, "calculate_button")
    assert hasattr(cursor, "autoupdate_check")
    assert "Add Cursor" in cursor.add_cursor_button.text()
    assert "Calculate" in cursor.calculate_button.text()
    assert cursor.autoupdate_check.text() == "Autoupdate"
    assert not cursor.autoupdate_check.isChecked()
    assert cursor._cursors == []
    assert cursor._dragging_cursor is None
    assert cursor._drag_offset == (0, 0)
    assert not cursor._autoupdate_enabled
    # The add/calculate/autoupdate controls expose tooltips.
    assert cursor.add_cursor_button.toolTip() != ""
    assert cursor.calculate_button.toolTip() != ""
    assert cursor.autoupdate_check.toolTip() != ""

    # The automatic clustering widget.
    clustering = widget.automatic_clustering_widget
    assert clustering.viewer == viewer
    assert clustering.parent_widget == parent
    assert clustering.layout().count() > 0
    assert hasattr(clustering, "clustering_method_combobox")
    assert clustering.clustering_method_combobox.count() == 2
    assert (
        clustering.clustering_method_combobox.itemText(0)
        == "GMM (Gaussian Mixture Model)"
    )
    assert clustering.clustering_method_combobox.itemText(1) == "K-means"
    assert clustering.color_plot_toggle.isChecked()
    assert clustering.show_centroids_toggle.isChecked()
    assert hasattr(clustering, "num_clusters_spinbox")
    assert clustering.num_clusters_spinbox.minimum() == 2
    assert clustering.num_clusters_spinbox.maximum() == 100
    assert clustering.num_clusters_spinbox.value() == 2
    assert hasattr(clustering, "apply_button")
    assert hasattr(clustering, "clear_button")
    assert clustering.apply_button.text() == "Apply Clustering"
    assert clustering.clear_button.text() == "Clear Clusters"
    assert not clustering.clear_button.isEnabled()
    assert clustering._clusters == []
    assert clustering._ellipse_patches == []
    assert isinstance(clustering.cluster_table, QTableWidget)
    assert clustering.cluster_table.columnCount() == 8
    assert clustering.cluster_table.rowCount() == 0

    # Manual selection starts with Selection 1 and no drawing tool.
    assert hasattr(widget, "_manual_selections")
    assert len(widget._manual_selections) == 1
    sel1 = widget._manual_selections[0]
    assert sel1["class_id"] == 1
    assert sel1["visible"] is True
    assert widget._selected_class_id == 1
    assert hasattr(widget, "selection_tool_buttons")
    assert set(widget.selection_tool_buttons.keys()) == {
        "LASSO",
        "ELLIPSE",
        "RECTANGLE",
        "BRUSH",
        "ERASER",
    }
    for btn in widget.selection_tool_buttons.values():
        assert not btn.isChecked()
    # Canvas selectors are set to class 1.
    for selector in parent.canvas_widget.selectors.values():
        assert selector.class_value == 1

    # Switching between cursor, clustering and manual modes.
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

    # Each mode's own button is pinned, and manual selection has none.
    assert_run_row_is_pinned(
        widget, cursor.calculate_button, cursor.autoupdate_checkbox
    )
    assert_run_row_is_pinned(widget, clustering.apply_button)
    # The pinned row follows the mode the tab is showing.
    assert widget._run_row_stack.currentWidget() is cursor.run_row
    widget.selection_mode_combobox.setCurrentIndex(1)
    assert widget._run_row_stack.currentWidget() is clustering.run_row
    assert widget._run_row_stack.isVisibleTo(widget)
    # Manual selection has nothing to run, so the row takes no space.
    widget.selection_mode_combobox.setCurrentIndex(2)
    assert not widget._run_row_stack.isVisibleTo(widget)
    # It is one row of buttons, not a panel: the settings above it keep the
    # space, as in every other tab.
    assert (
        widget._run_row_stack.sizePolicy().verticalPolicy()
        == QSizePolicy.Fixed
    )
    assert widget.layout().stretch(0) == 1


def test_manual_selections_on_a_layer(make_viewer_model, qtbot):
    """Manual selections get the next free id, their own labels layer, and
    follow the selection mode, the overlay toggle and plot updates."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    assert widget._get_next_available_selection_id() == "MANUAL SELECTION #1"

    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    assert widget._current_selection_id == "None"
    assert widget.selection_id is None
    assert widget._phasors_selected_layer is None
    # With no selection there is no overlay to show.
    widget._on_show_color_overlay(True)

    widget.selection_mode_combobox.setCurrentText("Manual Selection")
    combobox = widget.selection_input_widget.phasor_selection_id_combobox
    assert combobox.count() == 2
    assert combobox.itemText(0) == "None"
    assert combobox.currentText() == "None"
    assert (
        "selections" not in intensity_image_layer.metadata
        or len(intensity_image_layer.metadata.get("selections", {})) == 0
    )

    widget.manual_selection_changed(np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]))
    assert widget.selection_id == "MANUAL SELECTION #1"
    assert analysis_layer_name(
        "MANUAL SELECTION #1", intensity_image_layer.name
    ) in [layer.name for layer in viewer.layers]

    assert widget._get_next_available_selection_id() == "MANUAL SELECTION #2"
    widget.manual_selection_changed(np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0]))
    assert widget.selection_id == "MANUAL SELECTION #1"
    combobox.setCurrentText("None")
    assert widget.selection_id is None
    widget.manual_selection_changed(np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0]))
    assert widget.selection_id == "MANUAL SELECTION #2"
    assert widget._get_next_available_selection_id() == "MANUAL SELECTION #3"

    # Scalar 0, None, or non-sequences return None without raising.
    assert widget.selection_id is not None
    assert widget.manual_selection_changed(0) is None
    assert widget.manual_selection_changed(None) is None
    assert widget.manual_selection_changed(1.5) is None
    assert widget.manual_selection_changed("invalid") is None

    # Selection processing is skipped during plot updates.
    parent._updating_plot = True
    assert widget.manual_selection_changed([1, 2, 3]) is None
    assert widget.update_phasor_plot_with_selection_id("test") is None
    parent._updating_plot = False

    # Manual selection layers are hidden in cursor mode.
    manual_layer = viewer.layers[
        analysis_layer_name(widget.selection_id, intensity_image_layer.name)
    ]
    assert manual_layer.visible is True
    widget.selection_mode_combobox.setCurrentText("Cursor Selection")
    assert manual_layer.visible is False
    assert not widget.is_manual_selection_mode()
    widget.selection_mode_combobox.setCurrentText("Manual Selection")
    assert manual_layer.visible is True

    # Switching across every selection mode index exercises each branch.
    for index in (1, 2, 0):
        widget._on_selection_mode_changed(index)
        assert widget.stacked_widget.currentIndex() == index

    # Layer-change and selection-id update paths.
    widget._on_image_layer_changed()
    widget.on_selection_id_changed()
    widget.update_phasor_plot_with_selection_id(widget.selection_id)
    assert widget.selection_id in ("", "None", None) or isinstance(
        widget.selection_id, str
    )

    # Recreating a stored manual selection adds a hidden Labels layer once.
    selection_map = np.zeros(intensity_image_layer.data.shape, dtype=np.uint32)
    selection_map[0, 0] = 1
    widget._recreate_manual_selection_layer("stored_sel", selection_map)
    name = analysis_layer_name("stored_sel", intensity_image_layer.name)
    assert name in [ly.name for ly in viewer.layers]
    recreated = viewer.layers[name]
    assert recreated.visible is False
    assert recreated.metadata["napari_phasors_selection_type"] == "manual"
    n_layers = len(viewer.layers)
    widget._recreate_manual_selection_layer("stored_sel", selection_map)
    assert len(viewer.layers) == n_layers

    # A named selection creates a layer whose visibility follows the colour
    # overlay toggle.
    widget._on_show_color_overlay(True)
    widget.selection_mode_combobox.setCurrentText("Manual Selection")
    widget.manual_selection_changed(np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]))
    widget.selection_id = "sel_a"
    widget.create_phasors_selected_layer()
    overlay_name = analysis_layer_name("sel_a", intensity_image_layer.name)
    assert overlay_name in [ly.name for ly in viewer.layers]
    widget._on_show_color_overlay(True)
    assert viewer.layers[overlay_name].visible is True
    widget._on_show_color_overlay(False)
    assert viewer.layers[overlay_name].visible is False
    widget.update_phasor_plot_with_selection_id("sel_a")

    # A selection layer is "<image> [<selection id>]", not "<id>: <image>".
    assert intensity_image_layer.name == "FLIM data Intensity [Phasor]"
    parent._colormap = Mock()
    parent._colormap.N = 10
    widget.manual_selection_changed(np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]))
    widget.selection_id = "custom_selection"
    with patch(
        "napari_phasors.selection_tab.colormap_to_dict",
        return_value={1: [1, 0, 0], 2: [0, 1, 0]},
    ) as mock_colormap_to_dict:
        widget.create_phasors_selected_layer()
    assert mock_colormap_to_dict.call_count >= 1
    layer_names = [layer.name for layer in viewer.layers]
    assert (
        analysis_layer_name("custom_selection", intensity_image_layer.name)
        in layer_names
    )
    assert "FLIM data Intensity [custom_selection]" in layer_names
    assert not any(n.startswith("custom_selection: ") for n in layer_names)


def test_selection_tools_without_a_layer(make_viewer_model, qtbot):
    """Drawing tools, brush size, manual rows and the no-layer paths."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab
    cw = parent.canvas_widget

    # Finding a phasors layer by name.
    test_layer = Labels(np.zeros((10, 10), dtype=int), name="test_layer")
    viewer.add_layer(test_layer)
    assert widget._find_phasors_layer_by_name("test_layer") == test_layer
    assert widget._find_phasors_layer_by_name("non_existing") is None

    # Without an image layer there is nothing to plot or create.
    with patch.object(parent, "plot") as mock_plot:
        result = widget.update_phasor_plot_with_selection_id("test_selection")
        assert result is None
        mock_plot.assert_not_called()
    with patch(
        "napari_phasors.selection_tab.colormap_to_dict"
    ) as mock_colormap_to_dict:
        assert widget.create_phasors_selected_layer() is None
        mock_colormap_to_dict.assert_not_called()

    # A manual selection with no image layer selected is ignored.
    assert widget.manual_selection_changed(np.array([1, 0, 1])) is None

    # Nor while the plot is updating.
    parent._updating_plot = True
    with patch.object(parent, "plot") as mock_plot:
        result = widget.update_phasor_plot_with_selection_id("test_selection")
        assert result is None
        mock_plot.assert_not_called()
    parent._updating_plot = False

    # The size slider appears only for the painting tools and drives them.
    assert widget.brush_size_row.isHidden()
    widget.selection_tool_buttons["BRUSH"].click()
    assert cw.active_selector.name == "Interactive Brush Selector"
    assert not widget.brush_size_row.isHidden()
    widget.brush_size_slider.setValue(25)
    assert cw.brush_size == 25
    assert cw.selectors["ERASER"].size_px == 25
    assert widget.brush_size_value_label.text() == "25 px"
    # The eraser shares the slider, the marquee tools hide it again
    widget.selection_tool_buttons["ERASER"].click()
    assert cw.active_selector.name == "Interactive Eraser Selector"
    assert not widget.brush_size_row.isHidden()
    widget.selection_tool_buttons["LASSO"].click()
    assert widget.brush_size_row.isHidden()
    # Unchecking the brush hides the slider too
    widget.selection_tool_buttons["BRUSH"].click()
    assert not widget.brush_size_row.isHidden()
    widget.selection_tool_buttons["BRUSH"].click()
    assert cw.active_selector is None
    assert widget.brush_size_row.isHidden()

    # Tool buttons and the canvas stay in sync both ways.
    lasso_btn = widget.selection_tool_buttons["LASSO"]
    lasso_btn.click()
    assert lasso_btn.isChecked()
    assert cw.active_selector is not None
    assert cw.active_selector.name == "Interactive Lasso Selector"
    rect_btn = widget.selection_tool_buttons["RECTANGLE"]
    rect_btn.click()
    assert rect_btn.isChecked()
    assert not lasso_btn.isChecked()
    assert cw.active_selector is not None
    assert cw.active_selector.name == "Interactive Rectangle Selector"
    # Setting active_selector on canvas directly updates tab buttons
    cw.active_selector = "ELLIPSE"
    assert widget.selection_tool_buttons["ELLIPSE"].isChecked()
    assert not rect_btn.isChecked()
    assert cw.active_selector.name == "Interactive Ellipse Selector"
    # Escape deactivates all
    cw._on_escape(None)
    assert cw.active_selector is None
    assert not widget.selection_tool_buttons["ELLIPSE"].isChecked()
    assert not widget.selection_tool_buttons["RECTANGLE"].isChecked()
    assert not widget.selection_tool_buttons["LASSO"].isChecked()

    # Unchecking a tool button deactivates the active selector.
    btn = widget.selection_tool_buttons["RECTANGLE"]
    btn.click()
    assert btn.isChecked()
    assert cw.active_selector is not None
    btn.click()
    assert not btn.isChecked()
    assert cw.active_selector is None

    # Null parent edge cases; the slider stays usable without a canvas.
    widget.parent_widget = None
    widget._on_tool_btn_clicked("RECTANGLE")
    widget._on_tool_btn_clicked("BRUSH")
    widget.brush_size_slider.setValue(40)
    assert widget.brush_size_value_label.text() == "40 px"
    widget.parent_widget = parent
    cw._on_escape(None)

    # Selecting a row that is not in _manual_selections is ignored, and the
    # colormaps survive having no selection rows.
    widget._select_manual_row({"class_id": 999})
    assert widget._selected_class_id != 999
    saved_sels = widget._manual_selections
    widget._manual_selections = []
    widget._update_manual_colormaps()
    widget._manual_selections = saved_sels

    # Adding Selection 2 selects it, and the canvas selectors follow.
    widget.add_manual_selection_button.click()
    assert len(widget._manual_selections) == 2
    sel2 = widget._manual_selections[1]
    assert sel2["class_id"] == 2
    assert widget._selected_class_id == 2
    for selector in cw.selectors.values():
        assert selector.class_value == 2
    sel1 = widget._manual_selections[0]
    widget._select_manual_row(sel1)
    assert widget._selected_class_id == 1
    for selector in cw.selectors.values():
        assert selector.class_value == 1

    # The brush cursor takes the selected row's colour.
    widget.selection_mode_combobox.setCurrentText("Manual Selection")
    brush = cw.selectors["BRUSH"]
    assert brush.color == sel1["color"]
    sel3 = widget._add_manual_selection()
    assert widget._selected_class_id == sel3["class_id"]
    assert brush.color == sel3["color"]
    widget._select_manual_row(sel1)
    assert widget._selected_class_id == sel1["class_id"]
    assert brush.color == sel1["color"]
    new_color = QColor("#00ff00")
    widget._on_manual_color_changed(sel1, new_color)
    assert brush.color == new_color
    widget.selection_tool_buttons["BRUSH"].click()
    assert cw.active_selector is brush
    # Cursor has the new color with 0.5 transparency
    img = brush.cursor().pixmap().toImage()
    c = img.pixelColor(img.width() // 2, img.height() // 2)
    assert 115 <= c.alpha() <= 140
    assert c.green() > 200

    # Brush and eraser cursors persist while moving over the canvas.
    from matplotlib.backend_bases import MouseEvent

    assert not cw.canvas.cursor().pixmap().isNull()
    event = MouseEvent("motion_notify_event", cw.canvas, 100, 100)
    cw.canvas.callbacks.process("motion_notify_event", event)
    assert not cw.canvas.cursor().pixmap().isNull()
    widget.selection_tool_buttons["ERASER"].click()
    assert not cw.canvas.cursor().pixmap().isNull()
    event2 = MouseEvent("motion_notify_event", cw.canvas, 120, 120)
    cw.canvas.callbacks.process("motion_notify_event", event2)
    assert not cw.canvas.cursor().pixmap().isNull()
    cw._on_escape(None)

    # The selection_id property getter.
    combobox = widget.selection_input_widget.phasor_selection_id_combobox
    widget.selection_mode_combobox.setCurrentText("Cursor Selection")
    combobox.setCurrentText("None")
    assert widget.selection_id is None
    combobox.clear()
    assert widget.selection_id is None
    combobox.addItem("test_selection")
    combobox.setCurrentText("test_selection")
    assert widget.selection_id == "test_selection"
    combobox.setCurrentText("")
    assert widget.selection_id is None


# ---------------------------------------------------------------------------
# CursorSelectionWidget - structure & adding cursors
# ---------------------------------------------------------------------------


def test_cursor_adding_and_shapes(make_viewer_model, qtbot):
    """Adding cursors: selection, defaults per shape, colours, radius reuse,
    shape switching, clamping and clipping."""
    viewer, layer, parent, widget = _cursor_widget(make_viewer_model)

    # No cursors yet: the editor is hidden and has no title to refresh.
    assert widget._selected_cursor is None
    assert not widget._editor_box.isVisible()
    widget._refresh_editor_title()

    # Adding a cursor selects it and shows its editor page.
    widget._add_cursor()
    first = widget._cursors[0]
    assert widget._selected_cursor is first
    assert widget._details_stack.currentWidget() is first["detail"]
    assert first["row"].property("selected")

    # Adding a second cursor moves the selection to the new cursor.
    widget._add_cursor()
    second = widget._cursors[1]
    assert widget._selected_cursor is second
    assert not first["row"].property("selected")
    assert second["row"].property("selected")

    # Left-clicking the first cursor's row selects it again (exercises
    # ClickableFrame.mousePressEvent -> clicked -> _select_cursor).
    qtbot.mouseClick(first["row"], Qt.LeftButton)
    assert widget._selected_cursor is first
    assert first["row"].property("selected")
    assert widget._details_stack.currentWidget() is first["detail"]
    # The editor title reflects the selected cursor's number and shape.
    assert "Circular" in widget._editor_box.title()

    # Passing a dict that is not one of the widget's cursors is a no-op.
    widget._select_cursor({"not": "a real cursor"})
    assert widget._selected_cursor is first

    # Circular cursors (the default shape) start at the axes center.
    widget._clear_all_cursors()
    assert len(widget._cursors) == 0
    widget._add_cursor()
    assert len(widget._cursors) == 1
    assert _visible_rows(widget) == 1
    cursor = widget._cursors[0]
    assert cursor["type"] == "circular"
    assert {"g", "s", "radius", "color", "patch"} <= set(cursor)
    xlim = parent.canvas_widget.axes.get_xlim()
    ylim = parent.canvas_widget.axes.get_ylim()
    expected_g = max(-1.5, min(1.5, (xlim[0] + xlim[1]) / 2.0))
    expected_s = max(-1.5, min(1.5, (ylim[0] + ylim[1]) / 2.0))
    assert np.isclose(cursor["g"], expected_g)
    assert np.isclose(cursor["s"], expected_s)
    assert cursor["radius"] == 0.05

    # New circular cursors reuse the last cursor's radius.
    widget._cursors[0]["radius_spin"].setValue(0.25)
    assert widget._cursors[0]["radius"] == 0.25
    widget._add_cursor()
    assert widget._cursors[1]["radius"] == 0.25
    assert widget._cursors[0]["color"] != widget._cursors[1]["color"]

    # Cursors cycle through the Set1 colormap colors.
    widget._clear_all_cursors()
    for _ in range(9):
        widget._add_cursor()
    colors = [c["color"] for c in widget._cursors]
    color_tuples = [(c.red(), c.green(), c.blue()) for c in colors]
    assert len(set(color_tuples)) == 9
    widget._add_cursor()
    assert len(widget._cursors) == 10
    color_0 = widget._cursors[0]["color"]
    color_9 = widget._cursors[9]["color"]
    assert (color_0.red(), color_0.green(), color_0.blue()) == (
        color_9.red(),
        color_9.green(),
        color_9.blue(),
    )

    # Elliptic cursors keep their distinct radius defaults.
    widget._clear_all_cursors()
    widget._add_cursor(cursor_type="elliptic")
    cursor = widget._cursors[0]
    assert cursor["type"] == "elliptic"
    assert cursor["radius"] == 0.1
    assert cursor["radius_minor"] == 0.05
    assert cursor["angle"] == 0.0

    # Polar cursors take phase from the axes center, modulation 0.2-0.5.
    widget._add_cursor(cursor_type="polar")
    cursor = widget._cursors[-1]
    assert cursor["type"] == "polar"
    center_g = (xlim[0] + xlim[1]) / 2.0
    center_s = (ylim[0] + ylim[1]) / 2.0
    center_phase = np.rad2deg(np.arctan2(center_s, center_g))
    assert np.allclose(cursor["phase_min"], center_phase - 10.0)
    assert np.allclose(cursor["phase_max"], center_phase + 10.0)
    assert np.allclose(cursor["modulation_min"], 0.2)
    assert np.allclose(cursor["modulation_max"], 0.5)
    assert np.allclose(cursor["mod_min_spin"].value(), 0.2)
    assert np.allclose(cursor["mod_max_spin"].value(), 0.5)

    # Changing the shape combobox shows/hides the matching editor fields,
    # and the editor title tracks the shape.
    widget._clear_all_cursors()
    widget._add_cursor()
    cursor = widget._cursors[0]
    combo = cursor["type_combo"]
    assert isinstance(combo, QComboBox)
    assert "Circular" in widget._editor_box.title()
    assert cursor["center_widget"].isVisibleTo(cursor["detail"])
    assert not cursor["elliptic_widget"].isVisibleTo(cursor["detail"])
    assert not cursor["polar_widget"].isVisibleTo(cursor["detail"])
    combo.setCurrentIndex(combo.findData("elliptic"))
    assert cursor["type"] == "elliptic"
    assert cursor["center_widget"].isVisibleTo(cursor["detail"])
    assert cursor["elliptic_widget"].isVisibleTo(cursor["detail"])
    assert not cursor["polar_widget"].isVisibleTo(cursor["detail"])
    combo.setCurrentIndex(combo.findData("polar"))
    assert cursor["type"] == "polar"
    assert not cursor["center_widget"].isVisibleTo(cursor["detail"])
    assert not cursor["elliptic_widget"].isVisibleTo(cursor["detail"])
    assert cursor["polar_widget"].isVisibleTo(cursor["detail"])
    assert "Polar" in widget._editor_box.title()

    # Polar cursor modulation ranges are clamped/validated.
    widget._add_cursor(
        cursor_type="polar", modulation_min=0.8, modulation_max=0.4
    )
    cursor = widget._cursors[-1]
    assert cursor["modulation_min"] == 0.4
    assert cursor["modulation_max"] == 0.8
    widget._add_cursor(
        cursor_type="polar", modulation_min=-0.5, modulation_max=1.5
    )
    cursor = widget._cursors[-1]
    assert cursor["modulation_min"] == 0.0
    assert cursor["modulation_max"] == 1.0
    widget._add_cursor(
        cursor_type="polar", modulation_min=1.2, modulation_max=1.0
    )
    cursor = widget._cursors[-1]
    assert cursor["modulation_min"] == 0.99
    assert cursor["modulation_max"] == 1.0
    widget._add_cursor(
        cursor_type="polar", modulation_min=0.5, modulation_max=0.5
    )
    cursor = widget._cursors[-1]
    assert np.isclose(cursor["modulation_min"], 0.5)
    assert np.isclose(cursor["modulation_max"], 0.51)

    # Circular and elliptical cursor centers are clipped to [-1.5, 1.5].
    widget._add_cursor(cursor_type="circular", g=2.0, s=-2.0)
    cursor = widget._cursors[-1]
    assert cursor["g"] == 1.5
    assert cursor["s"] == -1.5
    widget._add_cursor(cursor_type="elliptic", g=-2.5, s=3.0)
    cursor = widget._cursors[-1]
    assert cursor["g"] == -1.5
    assert cursor["s"] == 1.5

    # Cursors are added at the center of custom/zoomed plot limits.
    parent.canvas_widget.axes.set_xlim([0.2, 0.8])
    parent.canvas_widget.axes.set_ylim([0.1, 0.5])
    xlim = parent.canvas_widget.axes.get_xlim()
    ylim = parent.canvas_widget.axes.get_ylim()
    expected_g = max(-1.5, min(1.5, (xlim[0] + xlim[1]) / 2.0))
    expected_s = max(-1.5, min(1.5, (ylim[0] + ylim[1]) / 2.0))
    widget._add_cursor(cursor_type="circular")
    assert np.isclose(widget._cursors[-1]["g"], expected_g)
    assert np.isclose(widget._cursors[-1]["s"], expected_s)
    widget._add_cursor(cursor_type="elliptic")
    assert np.isclose(widget._cursors[-1]["g"], expected_g)
    assert np.isclose(widget._cursors[-1]["s"], expected_s)


def test_cursor_rows_statistics_and_style(make_viewer_model, qtbot):
    """Row editors, counts and percentages, removal, patches, tooltips and
    the outline style dialog."""
    viewer, layer, parent, widget = _cursor_widget(make_viewer_model)

    # Every per-cursor parameter widget exposes an explanatory tooltip.
    widget._add_cursor()
    cursor = widget._cursors[0]
    for key in (
        "type_combo",
        "color_button",
        "g_spin",
        "s_spin",
        "radius_spin",
        "radius_minor_spin",
        "angle_spin",
        "phase_min_spin",
        "phase_max_spin",
        "mod_min_spin",
        "mod_max_spin",
        "count_label",
        "percentage_label",
        "visibility_button",
        "remove_button",
    ):
        assert cursor[key].toolTip() != "", f"missing tooltip on {key}"

    # Editing the row spinboxes updates the cursor data.
    assert isinstance(cursor["g_spin"], QDoubleSpinBox)
    cursor["g_spin"].setValue(0.7)
    cursor["s_spin"].setValue(0.4)
    cursor["radius_spin"].setValue(0.15)
    assert cursor["g"] == 0.7
    assert cursor["s"] == 0.4
    assert cursor["radius"] == 0.15

    # Removing cursors by index.
    widget._add_cursor()
    widget._add_cursor()
    assert len(widget._cursors) == 3
    widget._remove_cursor(0)
    assert len(widget._cursors) == 2
    assert _visible_rows(widget) == 2
    widget._remove_cursor(0)
    widget._remove_cursor(0)
    assert len(widget._cursors) == 0

    # Count and percentage labels are populated, and update on a change
    # (no autoupdate).
    assert not widget._autoupdate_enabled
    widget._add_cursor(g=0.5, s=0.5, radius=0.1)
    cursor = widget._cursors[0]
    assert isinstance(cursor["count_label"], QLabel)
    assert isinstance(cursor["percentage_label"], QLabel)
    assert cursor["count_label"].text() != "-"
    cursor["radius_spin"].setValue(0.5)
    count_text = cursor["count_label"].text()
    percentage_text = cursor["percentage_label"].text()
    assert count_text != "-"
    assert percentage_text != "-"
    assert int(count_text) >= 0
    assert "." in percentage_text or percentage_text.isdigit()
    widget._update_cursor_statistics()
    assert cursor["count_label"].text() != "-"

    # Clearing and redrawing cursor patches.
    widget._add_cursor()
    assert len(widget._cursors) == 2
    for cursor in widget._cursors:
        assert cursor["patch"] is not None
    widget.clear_all_patches()
    for cursor in widget._cursors:
        assert cursor["patch"] is None
    widget.redraw_all_patches()
    for cursor in widget._cursors:
        assert cursor["patch"] is not None
        assert cursor["patch"].get_visible()

    # Removing a cursor leaves the remaining row's signals working.
    widget._clear_all_cursors()
    widget._add_cursor(g=0.4, s=0.3, radius=0.2)
    widget._add_cursor(g=0.6, s=0.2, radius=0.2)
    widget._remove_cursor(0)
    assert len(widget._cursors) == 1
    widget._on_cursor_changed(widget._cursors[0])

    # Editing a row spinbox updates an elliptic cursor and its patch.
    widget._clear_all_cursors()
    widget._add_cursor(
        cursor_type="elliptic", g=0.5, s=0.3, radius=0.2, radius_minor=0.1
    )
    cursor = widget._cursors[0]
    cursor["g_spin"].setValue(0.6)
    cursor["s_spin"].setValue(0.25)
    cursor["radius_spin"].setValue(0.25)
    assert cursor["g"] == 0.6
    assert cursor["patch"] is not None

    # Two overlapping cursors exercise the per-cursor statistics loop.
    widget._clear_all_cursors()
    widget._on_autoupdate_changed(True)
    widget._add_cursor(
        cursor_type="elliptic", g=0.5, s=0.3, radius=0.4, radius_minor=0.3
    )
    widget._add_cursor(
        cursor_type="elliptic", g=0.4, s=0.25, radius=0.4, radius_minor=0.3
    )
    widget._update_cursor_statistics()
    widget._cursors[0]["radius_spin"].setValue(0.5)
    widget._update_cursor_statistics()
    assert len(widget._cursors) == 2

    # Autoupdate stats + hover cursor handling over real data.
    widget._clear_all_cursors()
    widget._add_cursor(
        cursor_type="elliptic", g=0.5, s=0.3, radius=0.3, radius_minor=0.2
    )
    widget._update_cursor_statistics()
    ell_patch = widget._cursors[0]["patch"]
    ev = Mock()
    ev.inaxes = ell_patch.axes
    ev.xdata = widget._cursors[0]["g"]
    ev.ydata = widget._cursors[0]["s"]
    with patch.object(
        QApplication, "keyboardModifiers", return_value=Qt.NoModifier
    ):
        widget._update_hover_cursor(ev)
        ev2 = Mock()
        ev2.inaxes = ell_patch.axes
        ev2.xdata = 5.0
        ev2.ydata = 5.0
        widget._update_hover_cursor(ev2)
    widget._clear_all_cursors()
    assert len(widget._cursors) == 0
    widget._on_autoupdate_changed(False)

    # Outline width and transparency apply to every cursor shape.
    widget._add_cursor()
    widget._add_cursor(cursor_type="elliptic")
    widget.cursor_style_button.click()
    dialog = widget.cursor_style_dialog
    assert dialog.isVisible()
    widget.cursor_style_button.click()
    assert widget.cursor_style_dialog is dialog
    widget.cursor_width_spin.setValue(4.5)
    widget.cursor_transparency_spin.setValue(0.3)
    dialog.close()
    for cursor in widget._cursors:
        patch_ = cursor["patch"]
        assert patch_.get_linewidth() == pytest.approx(4.5)
        assert patch_.get_edgecolor()[3] == pytest.approx(0.7)
    # New and redrawn cursors use the same style.
    widget._add_cursor(cursor_type="polar")
    widget.redraw_all_patches()
    for cursor in widget._cursors:
        patch_ = cursor["patch"]
        assert patch_.get_linewidth() == pytest.approx(4.5)
        assert patch_.get_edgecolor()[3] == pytest.approx(0.7)

    # Reset restores the default outline width and transparency.
    widget.cursor_style_button.click()
    widget.cursor_width_spin.setValue(6.0)
    widget.cursor_transparency_spin.setValue(0.5)
    widget.cursor_style_reset_button.click()
    widget.cursor_style_dialog.close()
    assert widget.cursor_outline_width == 2.0
    assert widget.cursor_outline_alpha == 1.0
    assert widget.cursor_width_slider.value() == 20
    assert widget.cursor_transparency_slider.value() == 0
    patch_ = widget._cursors[0]["patch"]
    assert patch_.get_linewidth() == pytest.approx(2.0)
    assert patch_.get_edgecolor()[3] == pytest.approx(1.0)


def test_cursor_multi_selection(make_viewer_model, qtbot):
    """Ctrl toggles and Shift extends the selection; removals keep it tidy."""
    viewer, layer, parent, widget = _cursor_widget(make_viewer_model)

    widget._add_cursor()
    widget._add_cursor()
    widget._add_cursor()
    c0, c1, c2 = widget._cursors

    # Normal click c0: single selection
    qtbot.mouseClick(c0['row'], Qt.LeftButton)
    assert widget._selected_cursors == [c0]
    assert c0['row'].property("selected")
    assert not c1['row'].property("selected")
    assert not c2['row'].property("selected")

    # Ctrl-click c1 and c2: toggles them into the selection
    qtbot.mouseClick(c1['row'], Qt.LeftButton, Qt.ControlModifier)
    assert widget._selected_cursors == [c0, c1]
    assert c0['row'].property("selected")
    assert c1['row'].property("selected")
    assert not c2['row'].property("selected")
    qtbot.mouseClick(c2['row'], Qt.LeftButton, Qt.ControlModifier)
    assert widget._selected_cursors == [c0, c1, c2]
    assert c0['row'].property("selected")
    assert c1['row'].property("selected")
    assert c2['row'].property("selected")

    # Ctrl-click c1: toggles c1 out of selection
    qtbot.mouseClick(c1['row'], Qt.LeftButton, Qt.ControlModifier)
    assert widget._selected_cursors == [c0, c2]
    assert c0['row'].property("selected")
    assert not c1['row'].property("selected")
    assert c2['row'].property("selected")

    # Shift-click selects a range from the clicked anchor.
    qtbot.mouseClick(c0['row'], Qt.LeftButton)
    assert widget._selected_cursors == [c0]
    qtbot.mouseClick(c2['row'], Qt.LeftButton, Qt.ShiftModifier)
    assert widget._selected_cursors == [c0, c1, c2]
    assert all(c['row'].property("selected") for c in (c0, c1, c2))

    # Removing a cursor while several are selected updates the selection.
    widget._remove_cursor(c1)
    assert widget._selected_cursors == [c0, c2]
    assert c0['row'].property("selected")
    assert c2['row'].property("selected")

    # A deselected row must not anchor the next Shift-click range.
    widget._clear_all_cursors()
    for _ in range(4):
        widget._add_cursor()
    c0, c1, c2, c3 = widget._cursors
    widget._select_cursor(c0)
    widget._select_cursor(c2, modifiers=Qt.ControlModifier)
    widget._select_cursor(c2, modifiers=Qt.ControlModifier)
    assert widget._selected_cursors == [c0]
    assert widget._last_clicked_cursor is c0
    # The range runs from the still-selected row, not from c2.
    widget._select_cursor(c3, modifiers=Qt.ShiftModifier)
    assert widget._selected_cursors == [c0, c1, c2, c3]
    # Deselecting the last selected cursor clears the anchor too.
    widget._select_cursor(c0, modifiers=Qt.ControlModifier)
    widget._select_cursor(c1, modifiers=Qt.ControlModifier)
    widget._select_cursor(c2, modifiers=Qt.ControlModifier)
    widget._select_cursor(c3, modifiers=Qt.ControlModifier)
    assert widget._selected_cursors == []
    assert widget._last_clicked_cursor is None

    # A queued edit for an already-removed cursor is a no-op.
    widget._clear_all_cursors()
    widget._add_cursor(radius=0.05)
    cursor = widget._cursors[0]
    widget._remove_cursor(cursor)
    widget._on_param_changed(cursor, 'radius', 0.4)
    assert cursor['radius'] == pytest.approx(0.05)


def test_cursor_batch_editing_and_mixed_values(make_viewer_model, qtbot):
    """Editing several selected cursors at once: shared parameters, dashed
    (mixed) fields, typing, stepping, tooltips and autoupdate."""
    viewer, layer, parent, widget = _cursor_widget(make_viewer_model)

    # Batch editing two circular cursors updates shared parameters.
    widget._add_cursor(radius=0.05)
    widget._add_cursor(radius=0.08)
    c0, c1 = widget._cursors
    widget._select_cursor(c0)
    widget._select_cursor(c1, modifiers=Qt.ControlModifier)
    assert widget._selected_cursors == [c0, c1]
    assert widget._shared_params(widget._selected_cursors) == {
        'g',
        's',
        'radius',
    }
    c1['radius_spin'].setValue(0.12)
    assert abs(c1['radius'] - 0.12) < 1e-5
    assert abs(c0['radius'] - 0.12) < 1e-5
    assert abs(c0['radius_spin'].value() - 0.12) < 1e-5
    c1['g_spin'].setValue(0.65)
    assert abs(c1['g'] - 0.65) < 1e-5
    assert abs(c0['g'] - 0.65) < 1e-5
    assert abs(c0['g_spin'].value() - 0.65) < 1e-5

    # Circular + elliptic cursors only share their common parameters.
    widget._clear_all_cursors()
    widget._add_cursor("circular", radius=0.05)
    widget._add_cursor("elliptic", radius=0.10, radius_minor=0.04, angle=30.0)
    c_circ, c_ellip = widget._cursors
    radius_tip = c_ellip['radius_spin'].toolTip()
    angle_tip = c_ellip['angle_spin'].toolTip()
    assert radius_tip and angle_tip
    widget._select_cursor(c_circ)
    widget._select_cursor(c_ellip, modifiers=Qt.ControlModifier)
    assert widget._shared_params(widget._selected_cursors) == {
        'g',
        's',
        'radius',
    }
    # Non-shared inputs (minor radius, angle) are disabled in editor
    assert not c_ellip['radius_minor_spin'].isEnabled()
    assert not c_ellip['angle_spin'].isEnabled()
    assert c_ellip['radius_spin'].isEnabled()
    assert c_ellip['radius_spin'].toolTip() == widget.BATCH_TOOLTIP
    assert c_ellip['angle_spin'].toolTip() == widget.UNSHARED_TOOLTIP
    # A parameter only some cursors have is never dashed: angle is not
    # shared, so it is disabled and shows its own value.
    assert not c_ellip['angle_spin'].isMixed()
    assert c_ellip['angle_spin'].value() == pytest.approx(30.0)
    # Changing radius updates both
    c_ellip['radius_spin'].setValue(0.18)
    assert abs(c_ellip['radius'] - 0.18) < 1e-5
    assert abs(c_circ['radius'] - 0.18) < 1e-5
    # Changing angle directly on elliptic cursor does not affect circular
    c_ellip['angle_spin'].setValue(75.0)
    assert abs(c_ellip['angle'] - 75.0) < 1e-5
    assert 'angle' not in c_circ or c_circ.get('angle') == 0.0
    # Back to one cursor: every parameter is editable again, so the widgets
    # must describe themselves rather than the batch state.
    widget._select_cursor(c_ellip)
    assert c_ellip['angle_spin'].isEnabled()
    assert c_ellip['radius_spin'].toolTip() == radius_tip
    assert c_ellip['angle_spin'].toolTip() == angle_tip

    # Circular and polar cursors share no parameters.
    widget._clear_all_cursors()
    widget._add_cursor("circular")
    widget._add_cursor("polar")
    c_circ, c_polar = widget._cursors
    widget._select_cursor(c_circ)
    widget._select_cursor(c_polar, modifiers=Qt.ControlModifier)
    assert widget._shared_params(widget._selected_cursors) == set()
    assert "No shared parameters" in widget._editor_box.title()

    # Parameters the selected cursors disagree on show a dash.
    widget._clear_all_cursors()
    widget._add_cursor("circular", g=0.5, s=0.3, radius=0.05)
    widget._add_cursor("circular", g=0.6, s=0.3, radius=0.05)
    c0, c1 = widget._cursors
    _multi_select(widget, [c0, c1])
    assert c1['g_spin'].isMixed()
    assert c1['g_spin'].text() == MixedValueSpinBox.MIXED_TEXT
    # S and the radius agree, so their common value is shown as usual.
    assert not c1['s_spin'].isMixed()
    assert c1['s_spin'].value() == pytest.approx(0.3)
    assert not c1['radius_spin'].isMixed()
    # Values that round to the same displayed text are not a disagreement.
    c0['s'] = 0.3 + 10 ** -(c1['s_spin'].decimals() + 2)
    widget._update_editor_for_selection()
    assert not c1['s_spin'].isMixed()

    # Focus alone must not push one cursor's value onto the others.
    c1['g_spin'].setFocus()
    c1['g_spin'].editingFinished.emit()
    assert c1['g_spin'].isMixed()
    assert c0['g'] == pytest.approx(0.5)
    assert c1['g'] == pytest.approx(0.6)

    # One selected cursor always shows its own values, never a dash.
    widget._select_cursor(c1)
    assert not c1['g_spin'].isMixed()
    assert c1['g_spin'].value() == pytest.approx(0.6)

    # The typed value may be the one the field already holds (the editor
    # shows c1's page, so its G is what the box holds).
    _multi_select(widget, [c0, c1])
    assert c1['g_spin'].isMixed()
    assert c1['g_spin'].value() == pytest.approx(0.6)
    _type_into(qtbot, c1['g_spin'], "0.60")
    assert c0['g'] == pytest.approx(0.6)
    assert c1['g'] == pytest.approx(0.6)

    # A value typed into a dashed field is set on every selected cursor.
    widget._clear_all_cursors()
    widget._add_cursor("circular", g=0.5)
    widget._add_cursor("circular", g=0.6)
    c0, c1 = widget._cursors
    _multi_select(widget, [c0, c1])
    spin = c1['g_spin']
    spin.setFocus()
    spin.lineEdit().selectAll()
    qtbot.keyClicks(spin, "0.75")
    # Half-typed values must not reach the cursors, and the text being typed
    # must not be overwritten by the dash.
    assert spin.text() == "0.75"
    assert c0['g'] == pytest.approx(0.5)
    qtbot.keyClick(spin, Qt.Key_Return)
    assert not spin.isMixed()
    assert c0['g'] == pytest.approx(0.75)
    assert c1['g'] == pytest.approx(0.75)
    assert c0['g_spin'].value() == pytest.approx(0.75)

    # The arrows resolve a dashed field too, from the value on show.
    widget._clear_all_cursors()
    widget._add_cursor("circular", radius=0.05)
    widget._add_cursor("circular", radius=0.08)
    c0, c1 = widget._cursors
    _multi_select(widget, [c0, c1])
    spin = c1['radius_spin']
    step = spin.singleStep()
    spin.stepBy(1)
    assert not spin.isMixed()
    assert c1['radius'] == pytest.approx(0.08 + step)
    assert c0['radius'] == pytest.approx(0.08 + step)

    # With auto-update on, one batch edit recomputes the selection once.
    widget._clear_all_cursors()
    widget._add_cursor(radius=0.05)
    widget._add_cursor(radius=0.05)
    c0, c1 = widget._cursors
    widget._select_cursor(c0)
    widget._select_cursor(c1, modifiers=Qt.ControlModifier)
    # Enabling auto-update applies once by itself; count only what the batch
    # edit triggers.
    widget.autoupdate_check.setChecked(True)
    assert widget._autoupdate_enabled
    calls = []
    widget._apply_selection = lambda: calls.append(True)
    c1['radius_spin'].setValue(0.2)
    assert len(calls) == 1
    assert c0['radius'] == pytest.approx(0.2)
    assert c1['radius'] == pytest.approx(0.2)


def _multi_select(widget, cursors):
    """Select every cursor in *cursors*, in order."""
    widget._select_cursor(cursors[0])
    for cursor in cursors[1:]:
        widget._select_cursor(cursor, modifiers=Qt.ControlModifier)


def _type_into(qtbot, spin, text):
    """Replace a spin box's contents with *text* and press Enter."""
    spin.setFocus()
    spin.lineEdit().selectAll()
    qtbot.keyClicks(spin, text)
    qtbot.keyClick(spin, Qt.Key_Return)


def test_mixed_spinbox_widget_behaviour(qtbot):
    """The spin box itself: dash, round trip, and explicit assignment."""
    spin = MixedValueSpinBox()
    qtbot.addWidget(spin)
    spin.setRange(-1.5, 1.5)
    spin.setDecimals(2)
    spin.setValue(0.25)

    assert not spin.isMixed()
    assert spin.text() == "0.25"

    spin.setMixed(True)
    assert spin.isMixed()
    assert spin.text() == MixedValueSpinBox.MIXED_TEXT
    # The held value is untouched, and reading the dash back yields it.
    assert spin.value() == pytest.approx(0.25)
    assert spin.valueFromText(MixedValueSpinBox.MIXED_TEXT) == pytest.approx(
        0.25
    )
    assert (
        spin.validate(MixedValueSpinBox.MIXED_TEXT, 0)[0]
        == QValidator.Acceptable
    )
    # Per-keystroke interpretation is off, so typing survives on screen.
    assert not spin.keyboardTracking()

    # Setting it again is a no-op rather than a redundant repaint.
    spin.setMixed(True)
    assert spin.text() == MixedValueSpinBox.MIXED_TEXT

    # An explicit assignment resolves the state.
    spin.setValue(0.4)
    assert not spin.isMixed()
    assert spin.text() == "0.40"
    assert spin.keyboardTracking()


def test_mixed_spinbox_forgets_edits_from_a_previous_visit(qtbot):
    """Entering the field starts a fresh edit, so no stale commit fires."""
    spin = MixedValueSpinBox()
    qtbot.addWidget(spin)
    spin.setRange(-1.5, 1.5)
    spin.setValue(0.25)
    spin.setMixed(True)

    committed = []
    spin.valueCommitted.connect(committed.append)

    # Pretend an earlier visit left an edit pending without committing it.
    spin._edited_while_mixed = True
    QApplication.sendEvent(
        spin, QFocusEvent(QEvent.FocusIn, Qt.MouseFocusReason)
    )

    spin.editingFinished.emit()

    assert committed == []
    assert spin.isMixed()


def test_shared_params_without_cursors():
    """No selection shares no parameters."""
    from napari_phasors.selection_tab import CursorSelectionWidget

    assert CursorSelectionWidget._shared_params([]) == set()


def test_make_spinbox_default_width_ref(make_viewer_model, qtbot):
    """``_make_spinbox`` sizes itself from its own range when no ref given."""
    from napari_phasors.selection_tab import CursorSelectionWidget

    spin = CursorSelectionWidget._make_spinbox(-1.5, 1.5, 0.5, 2, 0.01)
    assert isinstance(spin, QDoubleSpinBox)
    assert spin.value() == 0.5
    # A width was derived and applied as a hard minimum (default ref path).
    assert spin.minimumWidth() >= 70


def test_cursor_selection_layer_and_metadata(make_viewer_model, qtbot):
    """Calculate, autoupdate and Clear All manage the combined selection
    layer; cursors are stored in, and restored from, the layer metadata."""
    viewer, intensity_image_layer, parent, widget = _cursor_widget(
        make_viewer_model
    )
    selection_widget = parent.selection_tab
    layer_name = analysis_layer_name(
        "Cursor Selection", intensity_image_layer.name
    )

    def layer_names():
        return [ly.name for ly in viewer.layers]

    # Calculate creates the combined labels layer.
    widget._add_cursor()
    assert layer_name not in layer_names()
    widget.calculate_button.click()
    assert layer_name in layer_names()
    labels_layer = viewer.layers[layer_name]
    assert isinstance(labels_layer, Labels)
    assert labels_layer.data.shape == intensity_image_layer.data.shape

    # Labels layer visibility is managed when switching modes.
    widget._apply_selection()
    labels_layer = viewer.layers[layer_name]
    assert labels_layer.visible is True
    selection_widget.selection_mode_combobox.setCurrentText("Manual Selection")
    assert labels_layer.visible is False
    assert selection_widget.is_manual_selection_mode()
    selection_widget.selection_mode_combobox.setCurrentText("Cursor Selection")
    assert labels_layer.visible is True

    # Changing a cursor's colour recolours an already computed selection.
    widget._clear_all_cursors()
    widget.autoupdate_check.setChecked(False)
    widget._add_cursor(cursor_type="polar")
    cursor = widget._cursors[0]
    # With no computed selection, a colour change creates no labels.
    cursor["color_button"].set_color(QColor(10, 200, 30))
    cursor["color_button"].color_changed.emit(QColor(10, 200, 30))
    assert cursor["color"] == QColor(10, 200, 30)
    assert layer_name not in viewer.layers
    widget._apply_selection()
    cursor["color_button"].set_color(QColor(10, 200, 30))
    cursor["color_button"].color_changed.emit(QColor(10, 200, 30))
    color = viewer.layers[layer_name].colormap.color_dict[1]
    np.testing.assert_allclose(color[:3], (10 / 255, 200 / 255, 30 / 255))
    # A removed cursor's stale signal is ignored.
    widget._remove_cursor(cursor)
    widget._on_cursor_color_changed(cursor)

    # Calculate with two cursors fills the labels layer.
    widget._clear_all_cursors()
    widget._add_cursor(g=0.5, s=0.3, radius=0.1)
    widget._add_cursor(g=0.6, s=0.4, radius=0.15)
    assert layer_name not in layer_names()
    widget.calculate_button.click()
    assert layer_name in layer_names()
    labels_layer = viewer.layers[layer_name]
    assert isinstance(labels_layer, Labels)
    assert np.any(labels_layer.data > 0)

    # The autoupdate toggle.
    widget._clear_all_cursors()
    assert not widget.autoupdate_check.isChecked()
    assert not widget._autoupdate_enabled
    assert widget.calculate_button.isEnabled()
    widget._add_cursor()
    assert layer_name not in layer_names()
    widget.autoupdate_check.setChecked(True)
    assert widget._autoupdate_enabled
    assert not widget.calculate_button.isEnabled()
    assert layer_name in layer_names()
    widget.autoupdate_check.setChecked(False)
    assert not widget._autoupdate_enabled
    assert widget.calculate_button.isEnabled()

    # Clear All removes all cursors and the selection layer.
    widget.autoupdate_check.setChecked(True)
    widget._add_cursor(g=0.5, s=0.5, radius=0.5)
    assert layer_name in layer_names()
    widget._clear_all_cursors()
    assert len(widget._cursors) == 0
    assert layer_name not in layer_names()
    widget.autoupdate_check.setChecked(False)

    # A table mixing shapes builds a single combined selection layer.
    widget._add_cursor(cursor_type="circular", g=0.5, s=0.5, radius=0.4)
    widget._add_cursor(cursor_type="polar")
    widget._add_cursor(
        cursor_type="elliptic", g=0.4, s=0.3, radius=0.3, radius_minor=0.2
    )
    widget._apply_selection()
    assert layer_names().count(layer_name) == 1
    labels_layer = viewer.layers[layer_name]
    assert (
        labels_layer.metadata["napari_phasors_selection_type"]
        == "cursor_selection"
    )
    # All three shape metadata keys are persisted (Batch Analysis interop).
    selections = intensity_image_layer.metadata["settings"]["selections"]
    assert len(selections["circular_cursors"]) == 1
    assert len(selections["polar_cursors"]) == 1
    assert len(selections["elliptical_cursors"]) == 1

    # Circular cursors are stored in the circular_cursors metadata key, with
    # their visibility.
    widget._clear_all_cursors()
    widget._add_cursor(g=0.5, s=0.3, radius=0.1)
    widget._add_cursor(g=0.6, s=0.4, radius=0.15)
    widget._add_cursor(g=0.7, s=0.2, radius=0.08)
    widget._cursors[1]["visibility_button"].click()
    widget._apply_selection()
    selections = intensity_image_layer.metadata["settings"]["selections"]
    cursors = selections["circular_cursors"]
    assert len(cursors) == 3
    assert cursors[0]["g"] == 0.5
    assert cursors[0]["s"] == 0.3
    assert cursors[0]["radius"] == 0.1
    assert len(cursors[0]["color"]) == 4
    assert cursors[1]["g"] == 0.6
    assert cursors[2]["radius"] == 0.08
    assert cursors[0]["visible"] is True
    assert cursors[1]["visible"] is False

    # Metadata reflects edited cursor parameters after re-applying.
    widget._cursors[0]["g_spin"].setValue(0.7)
    widget._apply_selection()
    cursors = intensity_image_layer.metadata["settings"]["selections"][
        "circular_cursors"
    ]
    assert cursors[0]["g"] == 0.7
    assert cursors[0]["s"] == 0.3

    # Cursors are restored from metadata on image-layer change.
    widget._clear_all_cursors()
    assert len(widget._cursors) == 0
    parent.image_layer_with_phasor_features_combobox.setCurrentText("")
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        intensity_image_layer.name
    )
    parent.on_image_layer_changed()
    assert len(widget._cursors) == 3
    assert abs(widget._cursors[0]["g"] - 0.7) < 0.001
    assert abs(widget._cursors[0]["s"] - 0.3) < 0.001
    assert abs(widget._cursors[0]["radius"] - 0.1) < 0.001
    assert abs(widget._cursors[1]["g"] - 0.6) < 0.001


# ---------------------------------------------------------------------------
# CursorSelectionWidget - dragging
# ---------------------------------------------------------------------------


def test_cursor_dragging(make_viewer_model, qtbot):
    """Picking, translating, rotating and releasing circular and elliptic
    cursors."""
    viewer, layer, parent, widget = _cursor_widget(make_viewer_model)

    def modifiers(mod):
        return patch.object(
            QApplication, "keyboardModifiers", return_value=mod
        )

    widget._add_cursor()
    cursor = widget._cursors[0]
    initial_g = cursor["g"]
    initial_s = cursor["s"]

    # Motion without picking first does not move a cursor.
    mock_event = Mock()
    mock_event.xdata = 0.9
    mock_event.ydata = 0.5
    with modifiers(Qt.NoModifier):
        widget._on_motion(mock_event)
    assert cursor["g"] == initial_g
    assert cursor["s"] == initial_s

    # A pick event starts the drag.
    mock_event = Mock()
    mock_event.artist = cursor["patch"]
    mock_event.mouseevent.xdata = 0.5
    mock_event.mouseevent.ydata = 0.3
    assert widget._dragging_cursor is None
    with modifiers(Qt.NoModifier):
        widget._on_pick(mock_event)
    assert widget._dragging_cursor is cursor
    assert len(widget._drag_offset) == 2

    # Motion during a drag moves the cursor and its patch.
    circle_patch = cursor["patch"]
    initial_center = circle_patch.center
    widget._dragging_cursor = cursor
    widget._drag_mode = "translate"
    widget._drag_offset = (0, 0)
    mock_event = Mock()
    mock_event.xdata = 0.7
    mock_event.ydata = 0.4
    with modifiers(Qt.NoModifier):
        widget._on_motion(mock_event)
    assert cursor["g"] == 0.7
    assert cursor["s"] == 0.4
    assert cursor["g"] != initial_g
    assert cursor["g_spin"].value() == 0.7
    assert cursor["s_spin"].value() == 0.4
    mock_event.xdata = 0.8
    mock_event.ydata = 0.45
    with modifiers(Qt.NoModifier):
        widget._on_motion(mock_event)
    assert circle_patch.center != initial_center
    assert circle_patch.center == (0.8, 0.45)

    # Selection is not auto-applied while dragging.
    with patch.object(widget, "_apply_selection") as mock_apply:
        cursor["g_spin"].setValue(0.8)
        mock_apply.assert_not_called()
        # Releasing resets the dragging state.
        widget._on_release(Mock())
    assert widget._dragging_cursor is None

    # Translation and shift-rotation of an elliptical cursor.
    widget._clear_all_cursors()
    widget._add_cursor(cursor_type="elliptic", g=0.5, s=0.5, angle=0.0)
    cursor = widget._cursors[0]
    widget._dragging_cursor = cursor
    widget._drag_offset = (0, 0)
    widget._drag_mode = "translate"
    mock_event = Mock()
    mock_event.xdata = 0.6
    mock_event.ydata = 0.6
    with modifiers(Qt.NoModifier):
        widget._on_motion(mock_event)
    assert cursor["g"] == 0.6
    assert cursor["s"] == 0.6
    widget._dragging_cursor = cursor
    widget._drag_mode = "rotate"
    widget._drag_start_angle = 0.0
    widget._drag_start_cursor_angle = 0.0
    mock_rotate_event = Mock()
    mock_rotate_event.xdata = 0.6
    mock_rotate_event.ydata = 0.7
    with modifiers(Qt.ShiftModifier):
        widget._on_motion(mock_rotate_event)
    assert np.isclose(cursor["angle"], 90.0)
    widget._on_release(Mock())

    # The pick/motion/release drag handlers for an ellipse.
    widget._clear_all_cursors()
    widget._add_cursor(cursor_type="elliptic")
    cursor = widget._cursors[0]
    pick = Mock()
    pick.artist = cursor["patch"]
    pick.mouseevent.xdata = 0.5
    pick.mouseevent.ydata = 0.3
    with modifiers(Qt.NoModifier):
        widget._on_pick(pick)
    motion = Mock()
    motion.xdata = 0.6
    motion.ydata = 0.4
    with modifiers(Qt.NoModifier):
        widget._on_motion(motion)
    widget._on_release(Mock())
    assert len(widget._cursors) == 1


def test_polar_cursor_edges(make_viewer_model, qtbot):
    """Polar cursors are not translated: their nearest edge is picked and
    dragging it changes that bound only."""
    viewer, layer, parent, widget = _cursor_widget(make_viewer_model)

    def modifiers(mod):
        return patch.object(
            QApplication, "keyboardModifiers", return_value=mod
        )

    def point(angle_deg, r):
        return (
            r * np.cos(np.deg2rad(angle_deg)),
            r * np.sin(np.deg2rad(angle_deg)),
        )

    widget._add_cursor(
        cursor_type="polar",
        phase_min=10.0,
        phase_max=30.0,
        modulation_min=0.4,
        modulation_max=0.8,
    )
    cursor = widget._cursors[0]
    # The polar patch is pickable (so its edges can be grabbed).
    assert cursor["patch"].get_picker() is True

    # The nearest polar boundary is identified from a click position.
    assert (
        widget._closest_polar_edge(cursor, point(20, 0.8)) == "modulation_max"
    )
    assert (
        widget._closest_polar_edge(cursor, point(20, 0.4)) == "modulation_min"
    )
    assert widget._closest_polar_edge(cursor, point(10, 0.6)) == "phase_min"
    assert widget._closest_polar_edge(cursor, point(30, 0.6)) == "phase_max"

    # Dragging the outer arc changes modulation_max, leaving the wedge
    # centered.
    widget._clear_all_cursors()
    widget._add_cursor(
        cursor_type="polar",
        phase_min=10.0,
        phase_max=30.0,
        modulation_min=0.4,
        modulation_max=0.6,
    )
    cursor = widget._cursors[0]
    pick = Mock()
    pick.artist = cursor["patch"]
    pick.mouseevent.xdata, pick.mouseevent.ydata = point(20, 0.6)
    with modifiers(Qt.NoModifier):
        widget._on_pick(pick)
    assert widget._drag_mode == "polar_edge"
    assert widget._polar_edge == "modulation_max"
    motion = Mock()
    motion.xdata, motion.ydata = point(20, 0.85)
    with modifiers(Qt.NoModifier):
        widget._on_motion(motion)
    assert np.isclose(cursor["modulation_max"], 0.85, atol=1e-6)
    assert np.isclose(cursor["mod_max_spin"].value(), 0.85, atol=1e-2)
    # Phase bounds and the modulation_min are untouched.
    assert cursor["phase_min"] == 10.0
    assert cursor["phase_max"] == 30.0
    assert cursor["modulation_min"] == 0.4
    widget._on_release(Mock())
    assert widget._dragging_cursor is None

    # Dragging the phase_max edge changes the phase bound to the new angle.
    pick = Mock()
    pick.artist = cursor["patch"]
    pick.mouseevent.xdata, pick.mouseevent.ydata = point(30, 0.5)
    with modifiers(Qt.NoModifier):
        widget._on_pick(pick)
    assert widget._polar_edge == "phase_max"
    motion = Mock()
    motion.xdata, motion.ydata = point(45, 0.5)
    with modifiers(Qt.NoModifier):
        widget._on_motion(motion)
    assert np.isclose(cursor["phase_max"], 45.0, atol=1e-6)
    assert cursor["phase_min"] == 10.0


# ---------------------------------------------------------------------------
# CursorSelectionWidget - metadata persistence / harmonic handling
# ---------------------------------------------------------------------------


def test_cursors_restored_from_layer_metadata(make_viewer_model, qtbot):
    """Every cursor shape is restored from the layer's metadata, a hidden one
    restored hidden, and one stored before visibility existed as visible."""
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
        ],
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
        ],
        "circular_cursors": [
            {
                "g": 0.5,
                "s": 0.3,
                "radius": 0.1,
                "color": [255, 0, 0, 255],
                "visible": False,
            },
            {"g": 0.5, "s": 0.3, "radius": 0.1, "color": [0, 255, 0, 255]},
        ],
    }
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    w = parent.selection_tab.cursor_selection_widget
    w._on_image_layer_changed()

    assert len(w._cursors) == 4
    by_type = {}
    for cursor in w._cursors:
        by_type.setdefault(cursor["type"], []).append(cursor)
    assert len(by_type["polar"]) == 1
    assert len(by_type["elliptic"]) == 1
    hidden, legacy = by_type["circular"]
    assert hidden["visible"] is False
    assert hidden["visibility_button"].property("eyeCrossed") is True
    assert hidden["patch"] is None
    assert legacy["visible"] is True


def test_cursors_per_harmonic(make_viewer_model, qtbot):
    """Cursors belong to the harmonic they were created on: storage, rows,
    patches, colours, labels and which selection mode reacts."""
    viewer, intensity_image_layer, parent, widget = _cursor_widget(
        make_viewer_model
    )

    parent.harmonic_spinbox.setValue(1)
    widget._add_cursor()
    assert widget._cursors[0]["harmonic"] == 1
    assert widget._cursors[0]["patch"] is not None
    assert _visible_rows(widget) == 1
    h1_color1 = widget._cursors[0]["color"]

    parent.harmonic_spinbox.setValue(2)
    widget.on_harmonic_changed()
    assert widget._cursors[0]["patch"] is None
    assert _visible_rows(widget) == 0
    assert len(widget._cursors) == 1

    widget._add_cursor()
    assert widget._cursors[1]["harmonic"] == 2
    assert widget._cursors[1]["patch"] is not None
    assert _visible_rows(widget) == 1
    assert len(widget._cursors) == 2
    # Cursor colors are indexed per-harmonic.
    h2_color1 = widget._cursors[1]["color"]
    assert (h1_color1.red(), h1_color1.green(), h1_color1.blue()) == (
        h2_color1.red(),
        h2_color1.green(),
        h2_color1.blue(),
    )
    widget._add_cursor()
    h2_color2 = widget._cursors[2]["color"]
    assert h2_color1 != h2_color2

    parent.harmonic_spinbox.setValue(1)
    widget.on_harmonic_changed()
    assert widget._cursors[0]["patch"] is not None
    assert widget._cursors[1]["patch"] is None
    assert _visible_rows(widget) == 1
    assert len(widget._cursors) == 3

    # Removing a cursor updates the visible rows.
    widget._clear_all_cursors()
    widget._add_cursor()
    widget._add_cursor()
    widget._add_cursor()
    assert _visible_rows(widget) == 3
    widget._remove_cursor(1)
    assert len(widget._cursors) == 2
    assert _visible_rows(widget) == 2

    # The labels layer updates as the harmonic changes (with autoupdate).
    widget._clear_all_cursors()
    widget.autoupdate_check.setChecked(True)
    widget._add_cursor(g=0.5, s=0.5, radius=0.5)
    layer_name = analysis_layer_name(
        "Cursor Selection", intensity_image_layer.name
    )
    assert layer_name in [ly.name for ly in viewer.layers]
    assert np.count_nonzero(viewer.layers[layer_name].data) > 0
    parent.harmonic_spinbox.setValue(2)
    widget.on_harmonic_changed()
    if layer_name in [ly.name for ly in viewer.layers]:
        assert np.count_nonzero(viewer.layers[layer_name].data) == 0
    widget.autoupdate_check.setChecked(False)

    # Harmonic changes only act on the active selection mode.
    widget._clear_all_cursors()
    selection_widget = parent.selection_tab
    parent.harmonic_spinbox.setValue(1)
    widget._add_cursor()
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
    assert widget._cursors[0]["patch"] is not None


def test_cursor_lifecycle_for_every_shape(make_viewer_model, qtbot):
    """Add/remove/statistics/harmonic/apply for each shape, and autoupdate
    applying the selection when a cursor is added."""
    viewer, layer, parent, widget = _cursor_widget(
        make_viewer_model, settings={"frequency": 80.0}
    )

    for cursor_type in ("circular", "elliptic", "polar"):
        assert widget._cursors == []
        widget._add_cursor(cursor_type=cursor_type)
        widget._add_cursor(cursor_type=cursor_type)
        assert len(widget._cursors) == 2
        assert widget._cursors[0]["color"] != widget._cursors[1]["color"]

        widget._update_cursor_statistics()
        widget.on_harmonic_changed()
        widget._on_cursor_changed(widget._cursors[0])
        widget.redraw_all_patches()
        widget._on_calculate_clicked()

        widget._remove_cursor(0)
        assert len(widget._cursors) == 1
        widget.clear_all_patches()
        widget._clear_all_cursors()

        # With autoupdate enabled, adding a cursor applies the selection.
        widget._on_autoupdate_changed(True)
        widget._add_cursor(cursor_type=cursor_type)
        assert len(widget._cursors) == 1
        widget._on_autoupdate_changed(False)
        widget._remove_selection_layer()
        widget._clear_all_cursors()


# ---------------------------------------------------------------------------
# Automatic clustering (unchanged widget) and SelectionWidget integration
# ---------------------------------------------------------------------------


def test_automatic_clustering(make_viewer_model, qtbot):
    """GMM clustering: the table, statistics, removal, clearing, recolouring,
    re-running and the harmonic it is stored under."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.automatic_clustering_widget

    # The cluster table never scrolls internally: it grows to fit all rows
    # so the tab's own scroll area is what scrolls the table into view.
    assert (
        widget.cluster_table.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    )
    empty_height = widget.cluster_table.maximumHeight()
    widget.num_clusters_spinbox.setValue(5)
    widget._apply_clustering()
    assert widget.cluster_table.rowCount() == 5
    assert widget.cluster_table.maximumHeight() > empty_height
    widget._clear_clusters()
    # Allow a small tolerance for frame-metric jitter across Qt style
    # recalculations; the important invariant is that it shrinks back down.
    assert widget.cluster_table.maximumHeight() <= empty_height + 5

    # Applying GMM clustering.
    assert len(widget._clusters) == 0
    widget.num_clusters_spinbox.setValue(3)
    widget._apply_clustering()
    assert len(widget._clusters) == 3
    cluster = widget._clusters[0]
    assert {
        "g",
        "s",
        "radius",
        "radius_minor",
        "angle",
        "color",
        "harmonic",
    } <= set(cluster)
    assert cluster["harmonic"] == 1
    assert len(widget._ellipse_patches) == 3
    assert widget.cluster_table.rowCount() == 3
    assert widget.clear_button.isEnabled()
    layer_name = analysis_layer_name(
        "Cluster Selection", intensity_image_layer.name
    )
    assert layer_name in [layer.name for layer in viewer.layers]

    # Count and percentage columns are populated.
    for row in range(3):
        count_label = widget.cluster_table.cellWidget(row, 5)
        percentage_label = widget.cluster_table.cellWidget(row, 6)
        assert isinstance(count_label, QLabel)
        assert isinstance(percentage_label, QLabel)
        assert count_label.text() != "-"
        assert percentage_label.text() != "-"
        assert int(count_label.text()) >= 0

    # Statistics, recolouring and re-application paths.
    widget._update_cluster_statistics()
    widget.on_harmonic_changed()
    widget._redraw_cluster_ellipse(0)
    widget._on_cluster_color_changed(0, QColor(255, 0, 0, 255))
    widget._reapply_clustering_to_layers()

    # Removing individual clusters.
    widget._remove_cluster(0)
    assert len(widget._clusters) == 2
    assert widget.cluster_table.rowCount() == 2
    widget._remove_cluster(0)
    widget._remove_cluster(0)
    assert len(widget._clusters) == 0
    assert not widget.clear_button.isEnabled()

    # Clearing all clusters.
    widget.num_clusters_spinbox.setValue(3)
    widget._apply_clustering()
    assert len(widget._clusters) > 0
    widget._clear_clusters()
    assert len(widget._clusters) == 0
    assert len(widget._ellipse_patches) == 0
    assert not widget.clear_button.isEnabled()
    assert widget.cluster_table.rowCount() == 0

    # Re-running with a different cluster count rebuilds.
    widget.num_clusters_spinbox.setValue(2)
    widget._apply_clustering()
    assert len(widget._clusters) == 2
    widget.num_clusters_spinbox.setValue(4)
    widget._apply_clustering()
    assert len(widget._clusters) == 4

    # Clusters store the harmonic they were computed on.
    parent.harmonic_spinbox.setValue(2)
    widget.num_clusters_spinbox.setValue(3)
    widget._apply_clustering()
    assert len(widget._clusters) == 3
    for cluster in widget._clusters:
        assert cluster["harmonic"] == 2


# ---------------------------------------------------------------------------
# Automatic clustering: k-means and phasor plot colouring
# ---------------------------------------------------------------------------


def _clustering_setup(make_viewer_model, method="K-means", n_clusters=3):
    """Show the clustering mode of the Selection tab and apply *method*."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    parent.tab_widget.setCurrentWidget(parent.selection_tab)
    parent.selection_tab.selection_mode_combobox.setCurrentText(
        "Automatic Clustering"
    )
    widget = parent.selection_tab.automatic_clustering_widget
    widget.clustering_method_combobox.setCurrentText(method)
    widget.num_clusters_spinbox.setValue(n_clusters)
    widget._apply_clustering()
    return viewer, layer, parent, widget


def _labels_layer(viewer, layer):
    return viewer.layers[analysis_layer_name("Cluster Selection", layer.name)]


def _plotted_ids(parent, layer, labels_data):
    """Cluster id of every plotted point, in plotted order."""
    g = layer.metadata["G"][0]
    s = layer.metadata["S"][0]
    valid = ~np.isnan(g.ravel()) & ~np.isnan(s.ravel())
    return labels_data.ravel()[valid]


def test_kmeans_apply_labels_every_valid_pixel(make_viewer_model, qtbot):
    """K-means assigns each valid pixel to a cluster and fills the table."""
    viewer, layer, _parent, widget = _clustering_setup(make_viewer_model)

    assert widget._cluster_method == widget.METHOD_KMEANS
    assert len(widget._clusters) == 3
    assert all("radius" not in cluster for cluster in widget._clusters)
    assert widget._ellipse_patches == []
    assert widget.clear_button.isEnabled()

    labels = _labels_layer(viewer, layer).data
    g = layer.metadata["G"][0]
    s = layer.metadata["S"][0]
    finite = np.isfinite(g) & np.isfinite(s)
    assert labels.shape == g.shape
    assert np.all(labels[finite] >= 1)
    assert np.all(labels[finite] <= 3)
    assert np.all(labels[~finite] == 0)

    # The radii columns do not apply to k-means clusters.
    assert widget.cluster_table.rowCount() == 3
    assert widget.cluster_table.isColumnHidden(2)
    assert widget.cluster_table.isColumnHidden(3)
    assert widget.cluster_table.cellWidget(0, 2).text() == "-"

    counts = [
        int(widget.cluster_table.cellWidget(r, 5).text()) for r in range(3)
    ]
    assert counts == [int(np.sum(labels == i + 1)) for i in range(3)]
    percentages = [
        float(widget.cluster_table.cellWidget(r, 6).text()) for r in range(3)
    ]
    assert sum(percentages) == pytest.approx(100, abs=0.2)


def test_kmeans_labels_match_phasorpy(make_viewer_model, qtbot):
    """The labels layer holds phasorpy's k-means labels, shifted by one."""
    viewer, layer, _parent, widget = _clustering_setup(make_viewer_model)
    g = layer.metadata["G"][0]
    s = layer.metadata["S"][0]

    center_real, center_imag, labels = phasor_cluster_kmeans(
        g, s, clusters=3, random_state=0
    )

    np.testing.assert_array_equal(
        _labels_layer(viewer, layer).data, labels.astype(np.int64) + 1
    )
    np.testing.assert_allclose([c["g"] for c in widget._clusters], center_real)
    np.testing.assert_allclose([c["s"] for c in widget._clusters], center_imag)


def test_kmeans_gmm_table_switch_restores_radius_columns(
    make_viewer_model, qtbot
):
    """Running GMM after k-means shows the radii columns again."""
    _viewer, _layer, _parent, widget = _clustering_setup(make_viewer_model)
    assert widget.cluster_table.isColumnHidden(2)

    widget.clustering_method_combobox.setCurrentText(
        "GMM (Gaussian Mixture Model)"
    )
    widget._apply_clustering()

    assert widget._cluster_method == widget.METHOD_GMM
    assert not widget.cluster_table.isColumnHidden(2)
    assert not widget.cluster_table.isColumnHidden(3)
    assert len(widget._ellipse_patches) == 3


def test_kmeans_colors_histogram_bins(make_viewer_model, qtbot):
    """Histogram bins are painted with the colour of their cluster."""
    viewer, layer, parent, widget = _clustering_setup(make_viewer_model)
    parent.plot_type = "HISTOGRAM2D"
    parent.plot()
    artist = parent.canvas_widget.artists["HISTOGRAM2D"]

    expected = _plotted_ids(parent, layer, _labels_layer(viewer, layer).data)
    np.testing.assert_array_equal(artist.color_indices, expected)
    assert "overlay_histogram_image" in artist._mpl_artists
    color = widget._clusters[1]["color"]
    np.testing.assert_allclose(
        artist.overlay_colormap(2),
        (color.redF(), color.greenF(), color.blueF(), 1.0),
    )
    assert artist.overlay_colormap(0)[3] == 0
    assert parent._cluster_coloring_active


def test_kmeans_color_toggle_restores_plot(make_viewer_model, qtbot):
    """Turning the colouring off restores the original overlay colormap."""
    viewer, layer, parent, widget = _clustering_setup(make_viewer_model)
    artist = parent.canvas_widget.artists["HISTOGRAM2D"]
    widget.color_plot_toggle.setChecked(False)
    original = artist.overlay_colormap

    widget.color_plot_toggle.setChecked(True)
    assert parent._cluster_coloring_active
    assert artist.overlay_colormap is not original

    widget.color_plot_toggle.setChecked(False)
    assert not parent._cluster_coloring_active
    assert artist.overlay_colormap is original
    assert np.all(np.asarray(artist.color_indices) == 0)
    assert "overlay_histogram_image" not in artist._mpl_artists


def test_kmeans_colors_scatter_points(make_viewer_model, qtbot):
    """Scatter points take the colour of their cluster."""
    viewer, layer, parent, widget = _clustering_setup(make_viewer_model)
    parent.plot_type = "SCATTER"
    parent.plot()
    artist = parent.canvas_widget.artists["SCATTER"]

    ids = _plotted_ids(parent, layer, _labels_layer(viewer, layer).data)
    np.testing.assert_array_equal(artist.color_indices, ids)
    facecolors = artist._mpl_artists["scatter"].get_facecolors()
    first = int(np.flatnonzero(ids == 1)[0])
    color = widget._clusters[0]["color"]
    np.testing.assert_allclose(
        facecolors[first][:3], (color.redF(), color.greenF(), color.blueF())
    )


def test_kmeans_colors_contours(make_viewer_model, qtbot):
    """In contour mode each cluster gets contours in its own colour."""
    _viewer, _layer, parent, widget = _clustering_setup(make_viewer_model)
    parent.plot_type = "CONTOUR"
    parent.plot()
    contour = parent.canvas_widget.artists["CONTOUR"]

    assert sorted(contour._grouped_data) == [1, 2, 3]
    assert contour._group_styles[1]["mode"] == "solid"
    color = widget._clusters[0]["color"]
    assert contour._group_styles[1]["color"] == pytest.approx(
        (color.redF(), color.greenF(), color.blueF())
    )
    assert parent._contour_collections

    widget.color_plot_toggle.setChecked(False)
    assert contour._grouped_data is None
    assert contour.data is not None


def test_gmm_contours_keep_unassigned_points(make_viewer_model, qtbot):
    """Points outside every GMM ellipse keep the contour colormap."""
    _viewer, _layer, parent, _widget = _clustering_setup(
        make_viewer_model, method="GMM (Gaussian Mixture Model)", n_clusters=2
    )
    parent.plot_type = "CONTOUR"
    ids = np.zeros(4, dtype=np.uint32)
    ids[2:] = 1
    parent._selection_contour_colors = {1: (1.0, 0.0, 0.0)}
    cmap = parent._resolve_contour_colormap()
    x = np.array([0.1, 0.12, 0.8, 0.82])
    y = np.array([0.1, 0.12, 0.3, 0.32])

    assert parent._render_contour_by_class(x, y, ids, cmap)
    contour = parent.canvas_widget.artists["CONTOUR"]
    assert contour._group_styles[0] == {"mode": "colormap"}
    assert contour._group_styles[1]["mode"] == "solid"

    # A mismatched id array is not drawn per class.
    assert not parent._render_contour_by_class(x, y, ids[:2], cmap)


def test_kmeans_cluster_color_change(make_viewer_model, qtbot):
    """A new cluster colour reaches the plot, labels layer and centroid."""
    viewer, layer, parent, widget = _clustering_setup(make_viewer_model)
    red = QColor(255, 0, 0)

    widget.cluster_table.cellWidget(0, 4).color_changed.emit(red)

    assert widget._clusters[0]["color"] == red
    labels_layer = _labels_layer(viewer, layer)
    np.testing.assert_allclose(
        labels_layer.colormap.color_dict[1], (1.0, 0.0, 0.0, 1.0)
    )
    artist = parent.canvas_widget.artists["HISTOGRAM2D"]
    np.testing.assert_allclose(artist.overlay_colormap(1), (1, 0, 0, 1))
    assert widget._centroid_artists[0].get_markerfacecolor()[:3] == (
        1.0,
        0.0,
        0.0,
    )


def test_kmeans_centroids_toggle(make_viewer_model, qtbot):
    """Centroid markers sit on the cluster centers and can be hidden."""
    _viewer, _layer, parent, widget = _clustering_setup(make_viewer_model)
    ax = parent.canvas_widget.axes
    xlim = ax.get_xlim()

    assert len(widget._centroid_artists) == 3
    for marker, cluster in zip(
        widget._centroid_artists, widget._clusters, strict=True
    ):
        assert marker.axes is ax
        assert marker.get_marker() == "o"
        assert marker.get_markeredgecolor() == "black"
        color = cluster["color"]
        assert marker.get_markerfacecolor()[:3] == pytest.approx(
            (color.redF(), color.greenF(), color.blueF())
        )
        assert marker.get_xdata()[0] == pytest.approx(cluster["g"])
        assert marker.get_ydata()[0] == pytest.approx(cluster["s"])
    assert ax.get_xlim() == xlim

    widget.show_centroids_toggle.setChecked(False)
    assert widget._centroid_artists == []

    widget.show_centroids_toggle.setChecked(True)
    assert len(widget._centroid_artists) == 3


def test_kmeans_remove_cluster_relabels(make_viewer_model, qtbot):
    """Removing a k-means cluster unassigns its pixels and shifts the ids."""
    viewer, layer, parent, widget = _clustering_setup(make_viewer_model)
    before = _labels_layer(viewer, layer).data.copy()
    second_center = widget._clusters[1]["g"]

    widget.cluster_table.cellWidget(0, 7).click()

    after = _labels_layer(viewer, layer).data
    assert len(widget._clusters) == 2
    assert widget._clusters[0]["g"] == second_center
    assert np.all(after[before == 1] == 0)
    assert np.all(after[before == 2] == 1)
    assert np.all(after[before == 3] == 2)
    assert len(widget._centroid_artists) == 2
    assert int(widget.cluster_table.cellWidget(0, 5).text()) == int(
        np.sum(before == 2)
    )
    # Percentages stay relative to all valid pixels, not renormalised.
    percentages = [
        float(widget.cluster_table.cellWidget(r, 6).text()) for r in range(2)
    ]
    assert sum(percentages) < 99.9
    artist = parent.canvas_widget.artists["HISTOGRAM2D"]
    assert np.max(artist.color_indices) == 2

    widget._remove_cluster(0)
    widget._remove_cluster(0)
    assert widget._clusters == []
    assert widget._cluster_method is None
    assert widget._cluster_maps == {}
    assert widget._centroid_artists == []
    assert not widget.clear_button.isEnabled()
    assert not parent._cluster_coloring_active
    assert (
        analysis_layer_name("Cluster Selection", layer.name)
        not in viewer.layers
    )


def test_kmeans_coloring_follows_mode_and_tab(make_viewer_model, qtbot):
    """Cluster colours and centroids only show in the clustering mode."""
    _viewer, _layer, parent, widget = _clustering_setup(make_viewer_model)
    selection_tab = parent.selection_tab
    artist = parent.canvas_widget.artists["HISTOGRAM2D"]

    selection_tab.selection_mode_combobox.setCurrentText("Cursor Selection")
    assert not parent._cluster_coloring_active
    assert widget._centroid_artists == []

    selection_tab.selection_mode_combobox.setCurrentText(
        "Automatic Clustering"
    )
    assert parent._cluster_coloring_active
    assert len(widget._centroid_artists) == 3
    assert np.max(artist.color_indices) == 3

    parent.tab_widget.setCurrentWidget(parent.components_tab)
    assert not parent._cluster_coloring_active
    assert widget._centroid_artists == []

    parent.tab_widget.setCurrentWidget(parent.selection_tab)
    assert parent._cluster_coloring_active
    assert len(widget._centroid_artists) == 3


def test_kmeans_coloring_and_centroids_follow_harmonic(
    make_viewer_model, qtbot
):
    """Clusters of another harmonic neither colour the plot nor show."""
    _viewer, _layer, parent, widget = _clustering_setup(make_viewer_model)

    parent.harmonic_spinbox.setValue(2)
    assert not parent._cluster_coloring_active
    assert widget._centroid_artists == []

    parent.harmonic_spinbox.setValue(1)
    assert parent._cluster_coloring_active
    assert len(widget._centroid_artists) == 3


def test_kmeans_coloring_with_layer_missing_cluster_map(
    make_viewer_model, qtbot
):
    """Layers without matching cluster ids contribute unassigned points."""
    _viewer, layer, parent, widget = _clustering_setup(make_viewer_model)

    widget._cluster_maps[layer.name] = np.zeros((2, 2), dtype=np.uint32)
    ids, colors = widget.phasor_plot_selection_data()
    assert len(ids) == len(parent.get_features()[0])
    assert not np.any(ids)
    assert len(colors) == 3

    # With no usable ids anywhere the statistics show dashes.
    widget._update_cluster_statistics()
    assert widget.cluster_table.cellWidget(0, 5).text() == "-"
    assert widget.cluster_table.cellWidget(0, 6).text() == "-"


def test_kmeans_selection_data_without_phasor_layers(make_viewer_model, qtbot):
    """No selected layer with phasors means nothing to colour."""
    _viewer, _layer, parent, widget = _clustering_setup(make_viewer_model)

    with patch.object(widget, "_get_selected_layers", return_value=[]):
        assert widget.phasor_plot_selection_data() is None


def test_kmeans_ignores_infinite_coordinates(make_viewer_model, qtbot):
    """Infinite phasor coordinates are left out of the clusters."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.automatic_clustering_widget
    widget.clustering_method_combobox.setCurrentText("K-means")
    layer.metadata["G"][0].flat[0] = np.inf
    # The phasor plot itself cannot bin infinite values; only the
    # clustering is under test here.
    with patch.object(parent, "plot"):
        widget._apply_clustering()

    labels = _labels_layer(viewer, layer).data
    assert labels.flat[0] == 0
    assert labels.flat[1] > 0


def test_kmeans_failure_clears_previous_clusters(
    make_viewer_model, qtbot, capsys
):
    """A failing k-means run leaves no stale clusters behind."""
    viewer, layer, _parent, widget = _clustering_setup(make_viewer_model)

    with patch(
        "napari_phasors.selection_tab.phasor_cluster_kmeans",
        side_effect=ValueError("too few points"),
    ):
        widget._apply_clustering()

    assert "too few points" in capsys.readouterr().out
    assert widget._clusters == []
    assert widget._cluster_maps == {}
    assert not widget.clear_button.isEnabled()
    assert (
        analysis_layer_name("Cluster Selection", layer.name)
        not in viewer.layers
    )


def test_kmeans_refresh_without_phasor_data(make_viewer_model, qtbot):
    """Refreshing or releasing the colouring without data is a no-op."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab.automatic_clustering_widget

    with patch.object(parent, "plot") as plot:
        widget.refresh_phasor_plot_coloring()
        widget.release_phasor_plot_coloring()
    plot.assert_not_called()
    assert widget._update_kmeans_statistics([]) is None


def test_gmm_colors_phasor_plot(make_viewer_model, qtbot):
    """GMM clusters colour the phasor plot and show centroids too."""
    viewer, layer, parent, widget = _clustering_setup(
        make_viewer_model, method="GMM (Gaussian Mixture Model)", n_clusters=2
    )
    artist = parent.canvas_widget.artists["HISTOGRAM2D"]

    expected = _plotted_ids(parent, layer, _labels_layer(viewer, layer).data)
    np.testing.assert_array_equal(artist.color_indices, expected)
    assert len(widget._centroid_artists) == 2


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

    assert cursor_widget._cursors[0]["patch"] is not None


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

    cursor_layer_name = analysis_layer_name(
        "Cursor Selection", intensity_layer.name
    )
    man_layer_name = analysis_layer_name(
        "MANUAL SELECTION #1", intensity_layer.name
    )

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


def test_labels_layer_visibility_multi_layer_on_tab_toggle(
    make_viewer_model, qtbot
):
    """All selection layers across multiple layers are hidden when selection tab is hidden."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer1.name = "Layer 1"
    layer2 = create_image_layer_with_phasors()
    layer2.name = "Layer 2"
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    parent = PlotterWidget(viewer)
    parent.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )
    parent._process_layer_selection_change()
    widget = parent.selection_tab

    widget.cursor_selection_widget._add_cursor()
    widget.cursor_selection_widget._apply_selection()

    cursor1_name = analysis_layer_name("Cursor Selection", layer1.name)
    cursor2_name = analysis_layer_name("Cursor Selection", layer2.name)
    assert cursor1_name in viewer.layers
    assert cursor2_name in viewer.layers
    assert viewer.layers[cursor1_name].visible is True
    assert viewer.layers[cursor2_name].visible is True

    widget._set_labels_layer_visibility(False)
    assert viewer.layers[cursor1_name].visible is False
    assert viewer.layers[cursor2_name].visible is False

    widget._set_labels_layer_visibility(True)
    assert viewer.layers[cursor1_name].visible is True
    assert viewer.layers[cursor2_name].visible is True

    # Now test manual selection mode with multiple layers
    widget.selection_mode_combobox.setCurrentText("Manual Selection")
    widget.manual_selection_changed(np.array([1, 0] * 10))

    man1_name = analysis_layer_name("MANUAL SELECTION #1", layer1.name)
    man2_name = analysis_layer_name("MANUAL SELECTION #1", layer2.name)
    assert man1_name in viewer.layers
    assert man2_name in viewer.layers
    assert viewer.layers[man1_name].visible is True
    assert viewer.layers[man2_name].visible is True
    assert viewer.layers[cursor1_name].visible is False
    assert viewer.layers[cursor2_name].visible is False

    widget._set_labels_layer_visibility(False)
    assert viewer.layers[man1_name].visible is False
    assert viewer.layers[man2_name].visible is False
    assert viewer.layers[cursor1_name].visible is False
    assert viewer.layers[cursor2_name].visible is False

    widget._set_labels_layer_visibility(True)
    assert viewer.layers[man1_name].visible is True
    assert viewer.layers[man2_name].visible is True
    assert viewer.layers[cursor1_name].visible is False
    assert viewer.layers[cursor2_name].visible is False

    # Switch back to cursor selection
    widget.selection_mode_combobox.setCurrentText("Cursor Selection")
    widget._set_labels_layer_visibility(True)
    assert viewer.layers[man1_name].visible is False
    assert viewer.layers[man2_name].visible is False
    assert viewer.layers[cursor1_name].visible is True
    assert viewer.layers[cursor2_name].visible is True

    # Test via PlotterWidget tab transition methods
    parent._hide_all_tab_artists()
    assert viewer.layers[cursor1_name].visible is False
    assert viewer.layers[cursor2_name].visible is False
    parent._show_tab_artists(widget)
    assert viewer.layers[cursor1_name].visible is True
    assert viewer.layers[cursor2_name].visible is True


# ---------------------------------------------------------------------------
# CursorSelectionWidget - tooltips
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CursorSelectionWidget - visibility (eye) toggle
# ---------------------------------------------------------------------------


def test_cursor_visibility_toggle(make_viewer_model, qtbot):
    """The eye button hides a cursor's patch, excludes it from the selection
    and recomputes the selection layer only when there is one."""
    viewer, intensity_image_layer, parent, widget = _cursor_widget(
        make_viewer_model
    )
    layer_name = analysis_layer_name(
        "Cursor Selection", intensity_image_layer.name
    )

    # New cursors are visible by default with a plain (non-checkable) eye
    # button, like the remove "x".
    widget._add_cursor(g=0.5, s=0.5, radius=0.5)
    cursor = widget._cursors[0]
    assert cursor["visible"] is True
    assert cursor["visibility_button"].isCheckable() is False
    assert not cursor["visibility_button"].icon().isNull()
    assert cursor["visibility_button"].property("eyeCrossed") is False
    assert cursor["patch"] is not None
    assert layer_name not in [ly.name for ly in viewer.layers]

    # Hiding removes its patch (crossed-out eye) and, with no layer and no
    # autoupdate, only updates stats/patch.
    cursor["visibility_button"].click()
    assert cursor["visible"] is False
    assert cursor["patch"] is None
    assert cursor["visibility_button"].property("eyeCrossed") is True
    assert layer_name not in [ly.name for ly in viewer.layers]
    # Showing it again clears the slash and recreates the patch.
    cursor["visibility_button"].click()
    assert cursor["visible"] is True
    assert cursor["patch"] is not None
    assert cursor["visibility_button"].property("eyeCrossed") is False

    # Without autoupdate, toggling recomputes if a selection layer exists:
    # hiding the only cursor removes the (now empty) selection layer, and
    # showing it again recreates it.
    assert not widget._autoupdate_enabled
    widget.calculate_button.click()
    assert layer_name in [ly.name for ly in viewer.layers]
    cursor["visibility_button"].click()
    assert layer_name not in [ly.name for ly in viewer.layers]
    cursor["visibility_button"].click()
    assert layer_name in [ly.name for ly in viewer.layers]

    # A hidden cursor is excluded from the computed selection map.
    widget._clear_all_cursors()
    widget.autoupdate_check.setChecked(True)
    widget._add_cursor(g=0.5, s=0.5, radius=0.5)
    widget._add_cursor(g=0.4, s=0.4, radius=0.5)
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
    widget._cursors[1]["visibility_button"].click()
    assert set(np.unique(labels_layer.data)) == {0, 1}
    assert int(np.count_nonzero(labels_layer.data)) == cursor0_mask_count
    # Hidden cursor's statistics are blanked.
    assert widget._cursors[1]["count_label"].text() == "-"
    assert widget._cursors[1]["percentage_label"].text() == "-"
    # Show it again -> id 2 returns.
    widget._cursors[1]["visibility_button"].click()
    assert 2 in np.unique(labels_layer.data)

    # Hiding the top cursor reassigns overlapping pixels to the lower one.
    widget._clear_all_cursors()
    widget._add_cursor(g=0.5, s=0.5, radius=0.6)
    widget._add_cursor(g=0.5, s=0.5, radius=0.6)
    labels_layer = viewer.layers[layer_name]
    cursor0_mask = widget._cursor_mask(widget._cursors[0], g, s)
    # With both visible, the second (id 2) overwrites the overlap.
    assert 2 in np.unique(labels_layer.data)
    widget._cursors[1]["visibility_button"].click()
    assert set(np.unique(labels_layer.data)) == {0, 1}
    assert int(np.count_nonzero(labels_layer.data)) == int(
        np.sum(cursor0_mask)
    )


# ===========================================================================
# Manual Selection UI tests
# ===========================================================================


def test_manual_selection_rows(make_viewer_model, qtbot):
    """A selection row's colour and visibility reach both the plot overlay
    and the labels layer."""
    viewer = make_viewer_model()
    intensity_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab
    cw = parent.canvas_widget

    widget.selection_mode_combobox.setCurrentText("Manual Selection")
    widget.selection_id = "MANUAL SELECTION #1"
    widget.create_phasors_selected_layer()
    sel_layer = viewer.layers[
        analysis_layer_name("MANUAL SELECTION #1", intensity_layer.name)
    ]

    sel1 = widget._manual_selections[0]
    new_color = QColor(255, 0, 128)
    widget._on_manual_color_changed(sel1, new_color)
    assert sel1["color"] == new_color
    # The artist colormap has the updated color...
    overlay_cmap = cw.artists["HISTOGRAM2D"].overlay_colormap
    rgba = overlay_cmap(1)
    assert np.isclose(rgba[0], 255 / 255.0, atol=1e-2)
    assert np.isclose(rgba[1], 0.0, atol=1e-2)
    assert np.isclose(rgba[2], 128 / 255.0, atol=1e-2)
    # ...and so does the napari labels layer colormap.
    assert 1 in sel_layer.colormap.color_dict
    layer_rgba = sel_layer.colormap.color_dict[1]
    assert np.isclose(layer_rgba[0], 255 / 255.0, atol=1e-2)
    assert np.isclose(layer_rgba[1], 0.0, atol=1e-2)
    assert np.isclose(layer_rgba[2], 128 / 255.0, atol=1e-2)

    # Toggle to hidden: colormaps have alpha 0 for class 1.
    assert sel1["visible"] is True
    widget._toggle_manual_visibility(sel1)
    assert sel1["visible"] is False
    overlay_cmap = cw.artists["HISTOGRAM2D"].overlay_colormap
    assert overlay_cmap(1)[3] == 0.0
    assert sel_layer.colormap.color_dict[1][3] == 0.0
    assert sel1["count_label"].text() == "-"
    assert sel1["percentage_label"].text() == "-"
    # Toggle back to visible
    widget._toggle_manual_visibility(sel1)
    assert sel1["visible"] is True
    overlay_cmap = cw.artists["HISTOGRAM2D"].overlay_colormap
    assert overlay_cmap(1)[3] == 1.0
    assert sel_layer.colormap.color_dict[1][3] == 1.0


def test_manual_selection_classes(make_viewer_model, qtbot):
    """Statistics per class, removing classes and syncing them from the
    layer's stored selection map."""
    viewer = make_viewer_model()
    intensity_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    widget.selection_mode_combobox.setCurrentText("Manual Selection")
    widget.selection_id = "MANUAL SELECTION #1"
    widget._add_manual_selection(class_id=2)
    assert len(widget._manual_selections) == 2

    # Statistics for multiple manual selection classes: 4 pixels to class 1,
    # 6 pixels to class 2.
    g = intensity_layer.metadata["G"]
    s = intensity_layer.metadata["S"]
    g_harm = g[0] if g.ndim > intensity_layer.data.ndim else g
    s_harm = s[0] if s.ndim > intensity_layer.data.ndim else s
    total_valid = int(np.sum(np.isfinite(g_harm) & np.isfinite(s_harm)))
    s_map = np.zeros_like(intensity_layer.data, dtype=np.uint32)
    s_map.flat[0:4] = 1
    s_map.flat[4:10] = 2
    intensity_layer.metadata.setdefault("settings", {}).setdefault(
        "selections", {}
    ).setdefault("manual_selections", {})["MANUAL SELECTION #1"] = s_map
    widget._update_manual_selection_statistics()
    sel1 = [s for s in widget._manual_selections if s["class_id"] == 1][0]
    sel2 = [s for s in widget._manual_selections if s["class_id"] == 2][0]
    assert sel1["count_label"].text() == "4"
    assert sel2["count_label"].text() == "6"
    pct1 = float(sel1["percentage_label"].text())
    pct2 = float(sel2["percentage_label"].text())
    assert np.isclose(pct1, 4 / total_valid * 100, atol=0.1)
    assert np.isclose(pct2, 6 / total_valid * 100, atol=0.1)

    # Removing a class resets its pixels and updates the UI.
    manual_data = np.zeros(10, dtype=np.uint32)
    manual_data[0:3] = 1
    manual_data[3:6] = 2
    widget.manual_selection_changed(manual_data)
    s_map = intensity_layer.metadata["settings"]["selections"][
        "manual_selections"
    ]["MANUAL SELECTION #1"]
    assert np.any(s_map == 1)
    assert np.any(s_map == 2)
    widget._remove_manual_selection(sel1)
    assert len(widget._manual_selections) == 1
    assert widget._manual_selections[0]["class_id"] == 2
    assert not np.any(s_map == 1)
    assert np.any(s_map == 2)

    # Removing the selected row selects the first remaining one.
    sel1 = widget._add_manual_selection(class_id=1)
    sel3 = widget._add_manual_selection(class_id=3)
    sel2 = [s for s in widget._manual_selections if s["class_id"] == 2][0]
    assert len(widget._manual_selections) == 3
    widget._select_manual_row(sel1)
    assert widget._selected_class_id == 1
    widget._remove_manual_selection(sel1)
    assert (
        widget._selected_class_id == widget._manual_selections[0]["class_id"]
    )
    # Removing until empty recreates Selection 1.
    widget._remove_manual_selection(sel2)
    widget._remove_manual_selection(sel3)
    assert len(widget._manual_selections) == 1
    assert widget._manual_selections[0]["class_id"] == 1

    # Metadata sync with a layer map holding classes 1 and 4.
    s_map = np.zeros_like(intensity_layer.data, dtype=np.uint32)
    s_map.flat[0] = 1
    s_map.flat[1] = 4
    intensity_layer.metadata["settings"]["selections"]["manual_selections"][
        "MANUAL SELECTION #1"
    ] = s_map
    widget._sync_manual_selections_from_layer()
    class_ids = {s["class_id"] for s in widget._manual_selections}
    assert 4 in class_ids

    # Changing the selection ID with _processing_initial_selection False.
    widget._current_selection_id = "MANUAL SELECTION #1"
    widget._processing_initial_selection = False
    widget.selection_input_widget.phasor_selection_id_combobox.addItem(
        "MANUAL SELECTION #2"
    )
    widget.selection_input_widget.phasor_selection_id_combobox.setCurrentText(
        "MANUAL SELECTION #2"
    )
    widget.on_selection_id_changed()
    assert widget._current_selection_id == "MANUAL SELECTION #2"


def test_manual_selection_coloring_removed_on_tab_change(
    make_viewer_model, qtbot
):
    """Test that manual selection histogram coloring is removed when switching tabs."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab
    hist = parent.canvas_widget.artists["HISTOGRAM2D"]

    widget.selection_mode_combobox.setCurrentText("Manual Selection")

    # Draw a selection on the canvas
    parent.canvas_widget.active_selector = "RECTANGLE"
    rect = parent.canvas_widget.selectors["RECTANGLE"]

    class MockEvent:
        def __init__(self, x, y):
            self.xdata = x
            self.ydata = y

    rect.on_select(MockEvent(0.0, 0.0), MockEvent(1.0, 1.0))
    rect.apply_selection()

    assert "overlay_histogram_image" in hist._mpl_artists
    assert isinstance(hist.color_indices, np.ndarray)

    # Switch away from selection tab to components tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)
    assert "overlay_histogram_image" not in hist._mpl_artists
    assert hist.color_indices == 0

    # Switch back to selection tab
    parent.tab_widget.setCurrentWidget(parent.selection_tab)
    assert "overlay_histogram_image" in hist._mpl_artists
    assert isinstance(hist.color_indices, np.ndarray)

    # Switch to cursor mode within selection tab
    widget.selection_mode_combobox.setCurrentText("Cursor Selection")
    assert "overlay_histogram_image" not in hist._mpl_artists
    assert hist.color_indices == 0


def test_color_button_and_clickable_frame(qtbot):
    """Test ColorButton clicking, color setting, and ClickableFrame."""
    btn = ColorButton(QColor(255, 0, 0))
    btn.set_color(QColor(0, 255, 0))
    assert btn.color() == QColor(0, 255, 0)

    with patch(
        "qtpy.QtWidgets.QColorDialog.getColor", return_value=QColor(0, 0, 255)
    ):
        with qtbot.waitSignal(btn.color_changed):
            btn._on_clicked()
        assert btn.color() == QColor(0, 0, 255)

    with patch(
        "qtpy.QtWidgets.QColorDialog.getColor", return_value=QColor()
    ):  # invalid
        btn._on_clicked()
        assert btn.color() == QColor(0, 0, 255)

    frame = ClickableFrame()
    with qtbot.waitSignal(frame.clicked):
        qtbot.mouseClick(frame, Qt.LeftButton)


def test_manual_selection_statistics_and_labels_edge_cases(
    make_viewer_model, qtbot
):
    """Test manual selection statistics edge cases and hidden layer colormap."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab

    widget.selection_mode_combobox.setCurrentText("Manual Selection")
    widget.selection_id = "MANUAL SELECTION #1"

    # Test layer without G/S metadata (line 752)
    layer_no_phasors = layer.data.copy()
    no_phasors_layer = viewer.add_image(layer_no_phasors, name="no_phasors")
    widget._update_manual_selection_statistics()
    viewer.layers.remove(no_phasors_layer)

    # Test total_valid_pixels == 0
    layer.metadata["G"] = np.full_like(layer.data, np.nan, dtype=float)
    layer.metadata["S"] = np.full_like(layer.data, np.nan, dtype=float)
    widget._update_manual_selection_statistics()
    for sel in widget._manual_selections:
        assert sel["count_label"].text() == "-"
        assert sel["percentage_label"].text() == "-"

    # Restore finite data
    layer.metadata["G"] = np.zeros_like(layer.data, dtype=float)
    layer.metadata["S"] = np.zeros_like(layer.data, dtype=float)

    # Test layer with non-matching harmonic
    layer.metadata["harmonics"] = np.array([99])
    parent.harmonic = 1
    widget._update_manual_selection_statistics()

    # Test layer without harmonics array (2D G/S)
    del layer.metadata["harmonics"]
    widget._update_manual_selection_statistics()

    # Test parent._colormap branch in create_phasors_selected_layer and recreate
    import matplotlib.pyplot as plt

    parent._colormap = plt.get_cmap("viridis")

    # Test hidden selection row color in create_phasors_selected_layer and recreate
    sel1 = widget._manual_selections[0]
    sel1["visible"] = False
    widget.create_phasors_selected_layer()
    label_layer = viewer.layers[
        analysis_layer_name("MANUAL SELECTION #1", layer.name)
    ]
    assert label_layer.colormap.color_dict[1][3] == 0.0

    # Recreate manual selection layer with hidden selection and parent colormap
    viewer.layers.remove(label_layer)
    widget._recreate_manual_selection_layer(
        "MANUAL SELECTION #1", np.zeros_like(layer.data, dtype=np.uint32)
    )
    recreated = viewer.layers[
        analysis_layer_name("MANUAL SELECTION #1", layer.name)
    ]
    assert recreated.colormap.color_dict[1][3] == 0.0


def test_cursor_selection_widget_edge_cases_and_interactions(
    make_viewer_model, qtbot
):
    """Test cursor selection drag modes, hover cursor, and polar edge dragging."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab
    w_cursor = widget.cursor_selection_widget

    # Add circular, elliptic, and polar cursors first
    w_cursor._add_cursor(
        cursor_type="circular",
        g=0.5,
        s=0.5,
        radius=0.1,
        color=QColor(255, 0, 0),
    )
    w_cursor._add_cursor(
        cursor_type="elliptic",
        g=0.5,
        s=0.5,
        radius=0.1,
        radius_minor=0.05,
        angle=30.0,
        color=QColor(0, 255, 0),
    )
    w_cursor._add_cursor(
        cursor_type="polar",
        phase_min=10.0,
        phase_max=50.0,
        modulation_min=0.2,
        modulation_max=0.8,
        color=QColor(0, 0, 255),
    )

    # 1. Total valid pixels == 0 in cursor statistics with cursors present
    layer.metadata["G"] = np.full_like(layer.data, np.nan, dtype=float)
    layer.metadata["S"] = np.full_like(layer.data, np.nan, dtype=float)
    w_cursor._update_cursor_statistics()
    for cursor in w_cursor._cursors:
        assert cursor["count_label"].text() == "-"
        assert cursor["percentage_label"].text() == "-"

    # Restore finite data
    layer.metadata["G"] = np.zeros_like(layer.data, dtype=float)
    layer.metadata["S"] = np.zeros_like(layer.data, dtype=float)

    # 2. Test redraw_all_patches
    w_cursor.redraw_all_patches()

    # 3. Test draw_selection_overlay standalone function for polar cursor
    from napari_phasors.selection_tab import draw_selection_overlay

    draw_selection_overlay(
        parent.canvas_widget.axes,
        [
            {
                "type": "polar",
                "modulation_max": 0.8,
                "modulation_min": 0.2,
                "phase_min": 10.0,
                "phase_max": 45.0,
                "color": "#00ff00",
            }
        ],
        mode="cursor",
    )

    # 4. Test _on_image_layer_changed cleans up existing patches
    w_cursor._on_image_layer_changed()

    # Re-add cursors
    w_cursor._add_cursor(
        cursor_type="elliptic",
        g=0.5,
        s=0.5,
        radius=0.1,
        radius_minor=0.05,
        angle=30.0,
        color=QColor(0, 255, 0),
    )
    w_cursor._add_cursor(
        cursor_type="polar",
        phase_min=10.0,
        phase_max=50.0,
        modulation_min=0.2,
        modulation_max=0.8,
        color=QColor(0, 0, 255),
    )
    w_cursor.redraw_all_patches()

    # 5. Elliptic cursor shift-drag (rotate) and hover
    elliptic_cursor = [
        c for c in w_cursor._cursors if c["type"] == "elliptic"
    ][0]
    polar_cursor = [c for c in w_cursor._cursors if c["type"] == "polar"][0]

    class MockPickEvent:
        def __init__(self, artist, x, y):
            self.artist = artist
            self.mouseevent = Mock(xdata=x, ydata=y)

    # Pick with shift modifier
    with patch(
        "qtpy.QtWidgets.QApplication.keyboardModifiers",
        return_value=Qt.ShiftModifier,
    ):
        w_cursor._on_pick(MockPickEvent(elliptic_cursor["patch"], 0.6, 0.6))
        assert w_cursor._drag_mode == "rotate"

        # Hover with shift modifier (reset dragging_cursor first)
        w_cursor._dragging_cursor = None
        mock_hover_event = Mock(
            inaxes=parent.canvas_widget.axes, xdata=0.5, ydata=0.5
        )
        with patch.object(
            elliptic_cursor["patch"], "contains", return_value=(True, {})
        ):
            w_cursor._update_hover_cursor(mock_hover_event)

    # 6. Polar cursor edge drag for phase_min and modulation_min
    w_cursor._polar_edge = "phase_min"
    w_cursor._drag_polar_edge(polar_cursor, 0.5, 0.2)
    assert polar_cursor["phase_min"] != 10.0

    w_cursor._polar_edge = "modulation_min"
    w_cursor._drag_polar_edge(polar_cursor, 0.3, 0.3)
    assert polar_cursor["modulation_min"] != 0.2


class _BrushEvent:
    """Minimal stand-in for a Matplotlib mouse event."""

    def __init__(self, x, y, inaxes, button=1):
        self.xdata = x
        self.ydata = y
        self.inaxes = inaxes
        self.button = button


def _brush_stroke(selector, axes, points):
    """Press, drag through ``points`` and release on the last one."""
    selector._on_press(_BrushEvent(points[0][0], points[0][1], axes))
    for x, y in points[1:]:
        selector._on_motion(_BrushEvent(x, y, axes))
    selector._on_release(_BrushEvent(points[-1][0], points[-1][1], axes))


def test_manual_selection_brush_and_eraser(make_viewer_model, qtbot):
    """The brush paints the selected class into the layer and the eraser
    takes it back out."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    widget = parent.selection_tab
    widget.selection_mode_combobox.setCurrentText("Manual Selection")

    axes = parent.canvas_widget.axes
    data = parent.canvas_widget.active_artist_object.data
    x, y = float(data[0, 0]), float(data[0, 1])

    widget.selection_tool_buttons["BRUSH"].click()
    widget.brush_size_slider.setValue(20)
    _brush_stroke(parent.canvas_widget.active_selector, axes, [(x, y)])
    selections = layer.metadata["settings"]["selections"]["manual_selections"]
    painted = selections[widget.selection_id]
    assert np.count_nonzero(painted) > 0
    assert set(np.unique(painted)) <= {0, widget._selected_class_id}

    widget.selection_tool_buttons["ERASER"].click()
    widget.brush_size_slider.setValue(64)
    _brush_stroke(parent.canvas_widget.active_selector, axes, [(x, y)])
    selections = layer.metadata["settings"]["selections"]["manual_selections"]
    assert np.count_nonzero(selections[widget.selection_id]) == 0

    # The brush paints whichever selection row is currently highlighted.
    widget._add_manual_selection()  # Selection 2, selected on creation
    assert widget._selected_class_id == 2
    assert parent.canvas_widget.selectors["BRUSH"].paint_value == 2
    # The eraser always clears, whatever class is selected
    assert parent.canvas_widget.selectors["ERASER"].paint_value == 0
    widget.selection_tool_buttons["BRUSH"].click()
    widget.brush_size_slider.setValue(20)
    _brush_stroke(parent.canvas_widget.active_selector, axes, [(x, y)])
    selections = layer.metadata["settings"]["selections"]["manual_selections"]
    assert 2 in np.unique(selections[widget.selection_id])


def test_manual_selection_brush_and_eraser_cursor_persists_after_painting(
    make_viewer_model, qtbot
):
    """Verify brush and eraser cursors persist after a painting stroke."""
    from qtpy.QtCore import QEvent, QPointF, Qt
    from qtpy.QtGui import QMouseEvent

    viewer = make_viewer_model()
    data = np.random.rand(10, 10)
    layer = viewer.add_image(data, name="test")
    layer.metadata["G"] = np.random.rand(10, 10) * 0.5 + 0.2
    layer.metadata["S"] = np.random.rand(10, 10) * 0.5 + 0.1
    layer.metadata["harmonics"] = 1

    parent = PlotterWidget(viewer)
    widget = parent.selection_tab
    widget.selection_mode_combobox.setCurrentText("Manual Selection")

    cw = parent.canvas_widget
    bbox = cw.axes.bbox
    px = (bbox.x0 + bbox.x1) / 2.0
    py = (bbox.y0 + bbox.y1) / 2.0
    pt = QPointF(px, cw.canvas.height() - py)

    for tool_name in ("BRUSH", "ERASER"):
        widget.selection_tool_buttons[tool_name].click()
        assert cw.canvas.cursor().shape() == Qt.CursorShape.BitmapCursor

        # Paint stroke
        cw.canvas.mousePressEvent(
            QMouseEvent(
                QEvent.MouseButtonPress,
                pt,
                Qt.LeftButton,
                Qt.LeftButton,
                Qt.NoModifier,
            )
        )
        pt2 = QPointF(pt.x() + 5, pt.y() + 5)
        cw.canvas.mouseMoveEvent(
            QMouseEvent(
                QEvent.MouseMove,
                pt2,
                Qt.LeftButton,
                Qt.LeftButton,
                Qt.NoModifier,
            )
        )
        cw.canvas.mouseReleaseEvent(
            QMouseEvent(
                QEvent.MouseButtonRelease,
                pt2,
                Qt.LeftButton,
                Qt.NoButton,
                Qt.NoModifier,
            )
        )

        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()

        # Cursor after stroke release must still be the bitmap cursor
        assert cw.canvas.cursor().shape() == Qt.CursorShape.BitmapCursor
        assert not cw.canvas.cursor().pixmap().isNull()

        # Hover move after stroke
        pt3 = QPointF(pt.x() + 10, pt.y() + 10)
        cw.canvas.mouseMoveEvent(
            QMouseEvent(
                QEvent.MouseMove, pt3, Qt.NoButton, Qt.NoButton, Qt.NoModifier
            )
        )
        QApplication.processEvents()
        assert cw.canvas.cursor().shape() == Qt.CursorShape.BitmapCursor
        assert not cw.canvas.cursor().pixmap().isNull()
