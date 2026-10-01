import copy
from unittest.mock import MagicMock, patch

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from napari.layers import Image, Labels
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from phasorpy.component import phasor_component_fraction
from phasorpy.lifetime import phasor_from_lifetime
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import QColorDialog

from napari_phasors._mapping_filters import (
    COMPONENT_FRACTION,
    MODULATION,
    get_filters,
    new_filter,
    serialize_filter_applies,
)
from napari_phasors._tests.test_plotter import (
    assert_run_row_is_pinned,
    create_image_layer_with_phasors,
)
from napari_phasors._utils import (
    StatisticsTableWidget,
    analysis_layer_name,
    component_analysis_label,
    concentration_analysis_label,
    is_component_fit_label,
    split_analysis_layer_name,
)
from napari_phasors.components_tab import (
    ABSOLUTE_CONCENTRATION,
    COMPONENT_LABELS_TAG,
    LABELS_DOMINANT,
    LABELS_PER_COMPONENT,
    MANUAL_REFERENCE,
    TOTAL_CONCENTRATION,
    CenterFillSlider,
    ComponentsWidget,
    _as_hex,
    _finite_or_nan,
    _label_color_dict,
    component_concentrations,
    component_label_map,
    dominant_component_label_map,
    draw_components_overlay,
    draw_fraction_histogram_overlay,
    harmonic_plane,
    phasor_reference_from_layer,
)
from napari_phasors.plotter import PlotterWidget


def _lp_name(component, source):
    """Default name of a Linear Projection fraction layer."""
    return analysis_layer_name(component_analysis_label(component), source)


def _fit_name(component, source):
    """Default name of a Component Fit fraction layer."""
    return analysis_layer_name(
        component_analysis_label(component, fit=True), source
    )


def _setup_linear_projection(comp_widget):
    """Configure two components and run a Linear Projection analysis."""
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")
    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(1)
    comp_widget._run_analysis()


def _type_lifetime(comp, text):
    """Type *text* into a component's lifetime box and commit it."""
    comp.lifetime_edit.setText(text)
    comp.lifetime_edit.setModified(True)
    comp.lifetime_edit.editingFinished.emit()


def _rename_component(comp_widget, idx, name):
    """Type *name* into a component's field and commit it (as Enter does)."""
    comp_widget.components[idx].name_edit.setText(name)
    comp_widget.components[idx].name_edit.editingFinished.emit()


def _histogram_toggle(comp_widget, name):
    """Return the histogram/statistics toggle on *name*'s component card."""
    for idx in range(len(comp_widget.components)):
        if comp_widget._component_display_name(idx) == name:
            return comp_widget.components[idx].histogram_checkbox
    return None


def _available_histogram_components(comp_widget):
    """Return the component names whose card toggle is enabled."""
    return [
        comp_widget._component_display_name(idx)
        for idx, comp in enumerate(comp_widget.components)
        if comp.histogram_checkbox.isEnabled()
    ]


def _check_histogram_components(comp_widget, names):
    """Check exactly *names* by clicking the component cards' toggles."""
    for comp in comp_widget.components:
        if comp.histogram_checkbox.isChecked():
            comp.histogram_checkbox.setChecked(False)
    for name in names:
        toggle = _histogram_toggle(comp_widget, name)
        assert toggle is not None, f"no component card named {name!r}"
        toggle.setChecked(True)


def test_components_widget_initial_state(make_viewer_model, qtbot):
    """A fresh Components tab: its state, layout and docks, then the edits a
    tab without any layer has to cope with."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QComboBox, QLabel, QScrollArea

    from napari_phasors.components_tab import COMPONENT_CARD_STYLE

    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # Internal flags and dialogs start unset.
    assert comp_widget._updating_from_lifetime is False
    assert comp_widget._updating_settings is False
    assert comp_widget._analysis_attempted is False
    assert comp_widget.drag_events_connected is False
    assert comp_widget.dragging_component_idx is None
    assert comp_widget.dragging_label_idx is None
    assert comp_widget.plot_dialog is None
    assert comp_widget.style_dialog is None

    # Label and line styles.
    assert comp_widget.label_fontsize == 10
    assert comp_widget.label_bold is False
    assert comp_widget.label_italic is False
    assert comp_widget.label_color == 'black'
    assert comp_widget.show_colormap_line is True
    assert comp_widget.show_component_dots is True
    assert comp_widget.line_offset == 0.0
    assert comp_widget.line_width == 3.0
    assert comp_widget.line_alpha == 1
    assert comp_widget.default_component_color == 'dimgray'

    # Component colours and colormaps.
    assert len(comp_widget.component_colors) == 9
    assert comp_widget.component_colors[0] == 'magenta'
    assert comp_widget.component_colors[1] == 'cyan'
    assert len(comp_widget.component_colormap_names) == 9
    assert comp_widget.component_colormap_names[0] == 'magenta'
    assert comp_widget.component_colormap_names[1] == 'cyan'

    # ComponentState objects.
    for idx in (0, 1):
        state = comp_widget.components[idx]
        assert state.idx == idx
        assert state.dot is None
        assert state.text is None
        assert state.label == f"Component {idx + 1}"
        assert state.text_offset == (0.02, 0.02)
    comp = comp_widget.components[0]
    assert hasattr(comp, 'ui_elements')
    assert 'lifetime_label' in comp.ui_elements
    assert 'comp_layout' in comp.ui_elements
    assert comp.ui_elements['lifetime_label'] is not None
    assert comp.ui_elements['comp_layout'] is not None

    # Default settings structure.
    default_settings = comp_widget._get_default_components_settings()
    assert 'analysis_type' in default_settings
    assert default_settings['analysis_type'] == 'Linear Projection'
    assert 'components' in default_settings
    assert isinstance(default_settings['components'], dict)
    # The default keys must match the ones edits are persisted under, so the
    # restore path re-applies them (see ``_restore_line_and_label_settings``).
    line_key = 'two_component_line_settings'
    assert line_key in default_settings
    assert 'show_colormap_line' in default_settings[line_key]
    assert 'show_component_dots' in default_settings[line_key]
    assert 'line_offset' in default_settings[line_key]
    assert 'line_width' in default_settings[line_key]
    assert 'line_alpha' in default_settings[line_key]
    assert 'default_component_color' in default_settings[line_key]
    assert 'show_fraction_histogram' in default_settings[line_key]
    assert 'histogram_overlay_height' in default_settings[line_key]
    assert 'histogram_offset' in default_settings[line_key]
    assert 'histogram_alpha' in default_settings[line_key]
    label_key = 'two_components_label_settings'
    assert label_key in default_settings
    assert 'fontsize' in default_settings[label_key]
    assert 'bold' in default_settings[label_key]
    assert 'italic' in default_settings[label_key]
    assert 'color' in default_settings[label_key]

    # The layout holds a scroll area whose horizontal scrollbar only appears
    # once the content genuinely can't shrink further.
    assert comp_widget.layout() is not None
    assert comp_widget.layout().count() > 0
    scroll_area = comp_widget.findChild(QScrollArea)
    assert scroll_area is not None
    assert scroll_area.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded

    # The primary action stays reachable however long the settings get.
    assert_run_row_is_pinned(
        comp_widget,
        comp_widget.calculate_button,
        comp_widget.autoupdate_container,
    )

    # The histogram dock is sized like every other tab's dock: components are
    # picked with the toggle on each card, so there is no selector row.
    components_min = parent.components_histogram_dock_widget.minimumHeight()
    mapping_min = parent.phasor_map_histogram_dock_widget.minimumHeight()
    assert components_min == mapping_min
    for dock in (
        parent.components_histogram_dock_widget,
        parent.components_statistics_dock_widget,
    ):
        labels = [
            widget.text()
            for widget in dock.findChildren(QLabel)
            if widget.text() == "Component:"
        ]
        assert labels == []
        assert dock.findChild(QComboBox) is None

    parent.tab_widget.setCurrentWidget(comp_widget)

    assert comp_widget.viewer is viewer
    assert comp_widget.parent_widget is parent
    assert len(comp_widget.components) == 2
    assert comp_widget.component_line is None
    assert comp_widget.component_polygon is None
    assert comp_widget.comp1_fractions_layer is None
    assert comp_widget.fraction_layers == []
    assert comp_widget.fractions_colormap is None
    assert comp_widget.colormap_contrast_limits is None
    assert comp_widget.analysis_type == "Linear Projection"
    assert comp_widget.current_harmonic == 1
    assert comp_widget.analysis_type_combo is not None
    assert comp_widget.add_component_btn is not None
    assert comp_widget.calculate_button is not None
    for comp in comp_widget.components:
        assert comp is not None
        assert comp.name_edit is not None
        assert comp.g_edit is not None
        assert comp.s_edit is not None
        assert comp.select_button is not None
        assert comp.lifetime_edit is not None
    assert not comp_widget.histogram_widget.isHidden()

    # The selected card is highlighted in dodgerblue, rgb(30, 144, 255); the
    # old green must be gone.
    assert "rgba(30, 144, 255, 0.85)" in COMPONENT_CARD_STYLE
    assert "rgba(30, 144, 255, 0.06)" in COMPONENT_CARD_STYLE
    assert "0, 193, 140" not in COMPONENT_CARD_STYLE
    comp_widget._select_component_item(0)
    assert comp_widget.components[0].card_frame.property("selected") is True
    assert (
        comp_widget.components[1].card_frame.property("selected") is not True
    )
    comp_widget._select_component_item(1)
    assert comp_widget.components[1].card_frame.property("selected") is True

    # The harmonic spinbox is connected to the tab.
    parent.harmonic_spinbox.setValue(2)
    assert comp_widget.current_harmonic == 2

    # With no primary layer, restore clears the component input fields.
    comp_widget.components[0].g_edit.setText("0.5")
    comp_widget.components[0].name_edit.setText("stale")
    comp_widget._restore_on_layer_change()
    assert comp_widget.components[0].g_edit.text() == ""
    assert comp_widget.components[0].name_edit.text() == ""

    # A selected component without fraction layers cannot retain old data.
    comp_widget.histogram_widget.update_data(np.array([0.1, 0.2]))
    comp_widget._histogram_components = ["Missing component"]
    comp_widget.update_component_histogram()
    assert comp_widget.histogram_widget.counts is None
    assert comp_widget.histogram_widget._datasets == {}
    assert not comp_widget.histogram_widget.isHidden()

    # Every card's text fields are bold, including cards added later.
    comp_widget._add_component()
    assert len(comp_widget.components) >= 3
    for comp in comp_widget.components:
        for edit in (
            comp.name_edit,
            comp.g_edit,
            comp.s_edit,
            comp.lifetime_edit,
        ):
            edit.ensurePolished()
            assert edit.font().weight() >= QFont.DemiBold


def test_components_widget_lifetime_inputs_visibility_no_frequency(
    make_viewer_model,
    qtbot,
):
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    # Ensure no frequency
    layer.metadata.pop("settings", None)
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # Lifetime widgets should be hidden - check first component
    comp = comp_widget.components[0]
    assert not comp.ui_elements['lifetime_label'].isVisible()
    assert not comp.lifetime_edit.isVisible()


def test_components_lifetime_input_places_the_component(
    make_viewer_model, qtbot
):
    """With a frequency, a typed lifetime sets G/S and moves the dot.

    The dot lands exactly on the universal circle for every harmonic: the G/S
    boxes show three decimals, and placing the component from those strings
    left it up to ~5e-4 off the circle, visible once the plot is zoomed in.
    A component placed by typing its lifetime moves with the frequency;
    moving it by other means pins it, after which it stays put.
    """
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    comp = comp_widget.components[0]

    # Lifetime widgets visible - check first component
    assert not comp.ui_elements['lifetime_label'].isHidden()
    assert not comp.lifetime_edit.isHidden()

    # Enter lifetime and verify G,S updated
    comp.lifetime_edit.setText("3.0")
    comp_widget._update_component_from_lifetime(0)
    expected_g, expected_s = phasor_from_lifetime(80.0, 3.0)
    assert abs(float(comp.g_edit.text()) - expected_g) < 1e-3
    assert abs(float(comp.s_edit.text()) - expected_s) < 1e-3

    parent.tab_widget.setCurrentWidget(comp_widget)

    # Typing a lifetime places the component dot on the universal circle.
    canvas = parent.canvas_widget.canvas
    with patch.object(canvas, "draw", wraps=canvas.draw) as forced_draw:
        _type_lifetime(comp, "1.0")
    expected_g, expected_s = phasor_from_lifetime(80.0, 1.0)
    x, y = comp.dot.get_data()
    assert abs(x[0] - expected_g) < 1e-3
    assert abs(y[0] - expected_s) < 1e-3
    # A dot moved only with ``draw_idle`` can be repainted from a stale blit
    # background, leaving it visually at its old position.
    assert forced_draw.call_count >= 1

    # Moving it again keeps it on the circle and leaves the guard flag clear.
    _type_lifetime(comp, "3.0")
    expected_g, expected_s = phasor_from_lifetime(80.0, 3.0)
    x, y = comp.dot.get_data()
    assert abs(x[0] - expected_g) < 1e-3
    assert abs(y[0] - expected_s) < 1e-3
    assert comp_widget._updating_from_lifetime is False

    # The typed component follows a frequency change.
    parent._broadcast_frequency_value_across_tabs("40")
    expected_g, expected_s = phasor_from_lifetime(40.0, 3.0)
    x, y = comp.dot.get_data()
    assert abs(x[0] - expected_g) < 1e-12
    assert abs(y[0] - expected_s) < 1e-12

    # The run keeps the mark, so it still follows after being stored.
    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.2")
    comp_widget._on_component_coords_changed(1)
    comp_widget._run_analysis()
    parent._broadcast_frequency_value_across_tabs("80")
    expected_g, expected_s = phasor_from_lifetime(80.0, 3.0)
    x, y = comp.dot.get_data()
    assert abs(x[0] - expected_g) < 1e-12
    assert abs(y[0] - expected_s) < 1e-12

    # Typing coordinates pins the component.
    comp.g_edit.setText("0.3")
    comp.s_edit.setText("0.2")
    comp_widget._on_component_coords_changed(0)
    parent._broadcast_frequency_value_across_tabs("40")
    x, y = comp.dot.get_data()
    assert abs(x[0] - 0.3) < 1e-9
    assert abs(y[0] - 0.2) < 1e-9

    for harmonic in (1, 2, 3):
        parent.harmonic = harmonic
        for lifetime in (0.1, 0.5, 1.0, 3.0, 8.0, 20.0):
            _type_lifetime(comp, str(lifetime))

            x, y = comp.dot.get_data()
            assert abs(np.hypot(x[0] - 0.5, y[0]) - 0.5) < 1e-12

            # An unsaved setting of the layer until the analysis runs.
            stored = parent.layer_settings(layer)["component_analysis"][
                "components"
            ]["0"]["gs_harmonics"][str(harmonic)]
            assert abs(np.hypot(stored["g"] - 0.5, stored["s"]) - 0.5) < 1e-12
            # Stored coordinates are serialised to JSON on export.
            assert isinstance(stored["g"], float)
            assert isinstance(stored["s"], float)


def test_components_lifetime_without_harmonics_metadata(
    make_viewer_model, qtbot
):
    """A layer that reports no harmonics still accepts a typed lifetime.

    ``_get_available_harmonics`` returns an empty list for such layers, which
    used to make the lifetime input a silent no-op while the G/S inputs kept
    working.
    """
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    layer.metadata["harmonics"] = None
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)
    assert comp_widget._get_available_harmonics() == []

    comp = comp_widget.components[0]
    _type_lifetime(comp, "3.0")

    expected_g, expected_s = phasor_from_lifetime(80.0, 3.0)
    assert abs(float(comp.g_edit.text()) - expected_g) < 1e-3
    assert abs(float(comp.s_edit.text()) - expected_s) < 1e-3
    assert comp.dot is not None


def test_components_widget_dots_lines_and_labels(make_viewer_model, qtbot):
    """Placed components draw dots, a line and name labels on the plot."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    # Set component coordinates; a component with a name gets a label.
    comp_widget.components[0].name_edit.setText("A")
    comp_widget.components[0].g_edit.setText("0.25")
    comp_widget.components[0].s_edit.setText("0.15")
    comp_widget._on_component_coords_changed(0)

    comp_widget.components[1].g_edit.setText("0.75")
    comp_widget.components[1].s_edit.setText("0.45")
    comp_widget._on_component_coords_changed(1)

    # Dots created
    assert comp_widget.components[0].dot is not None
    assert comp_widget.components[1].dot is not None
    # Line should exist
    assert comp_widget.component_line is not None

    # Verify coordinates
    x0, y0 = comp_widget.components[0].dot.get_data()
    x1, y1 = comp_widget.components[1].dot.get_data()
    assert abs(x0[0] - 0.25) < 1e-9 and abs(y0[0] - 0.15) < 1e-9
    assert abs(x1[0] - 0.75) < 1e-9 and abs(y1[0] - 0.45) < 1e-9

    # The dots are only shown while the Components tab is active.
    assert comp_widget.components[0].dot.get_visible() is True
    parent.tab_widget.setCurrentIndex(
        parent.tab_widget.indexOf(parent.fret_tab)
    )
    assert comp_widget.components[0].dot.get_visible() is False
    parent.tab_widget.setCurrentWidget(parent.components_tab)
    assert comp_widget.components[0].dot.get_visible() is True

    # The label is styled from the label dialog.
    assert comp_widget.components[0].text is not None

    comp_widget._open_label_style_dialog()
    assert comp_widget.style_dialog.isVisible()

    comp_widget.fontsize_spin.setValue(14)
    comp_widget.bold_checkbox.setChecked(True)
    comp_widget.italic_checkbox.setChecked(True)
    comp_widget._on_label_style_changed()

    txt = comp_widget.components[0].text
    assert txt.get_fontsize() == 14
    assert txt.get_fontweight() == 'bold'
    assert txt.get_fontstyle() == 'italic'

    # Setting/clearing a name creates, repositions and removes the label.
    assert comp_widget.components[1].text is None
    comp_widget.components[1].name_edit.setText("Alpha")
    comp_widget._on_component_name_changed(1)
    assert comp_widget.components[1].text is not None
    # Renaming again exercises the previous-position branch.
    comp_widget.components[1].name_edit.setText("Beta")
    comp_widget._on_component_name_changed(1)
    assert comp_widget.components[1].text is not None
    # Clearing the name removes the label.
    comp_widget.components[1].name_edit.setText("")
    comp_widget._on_component_name_changed(1)

    # Components are stored per harmonic.
    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    _rename_component(comp_widget, 0, "Test Component 1")
    comp_widget._on_component_coords_changed(0)

    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.5")
    _rename_component(comp_widget, 1, "Test Component 2")
    comp_widget._on_component_coords_changed(1)

    # Switch to harmonic 2: components are cleared in the UI...
    comp_widget._on_harmonic_changed(2)
    assert comp_widget.components[0].g_edit.text() == ""
    assert comp_widget.components[1].g_edit.text() == ""

    # ...but kept (unsaved until the analysis runs) under harmonic 1
    components_settings = parent.layer_settings(layer)['component_analysis']
    assert '0' in components_settings['components']
    stored_comp1 = components_settings['components']['0']
    assert '1' in stored_comp1['gs_harmonics']
    harmonic_1_data = stored_comp1['gs_harmonics']['1']
    assert abs(harmonic_1_data['g'] - 0.2) < 1e-6
    assert abs(harmonic_1_data['s'] - 0.1) < 1e-6
    assert stored_comp1['name'] == "Test Component 1"

    # Switch back to harmonic 1: components are restored.
    comp_widget._on_harmonic_changed(1)
    assert abs(float(comp_widget.components[0].g_edit.text()) - 0.2) < 1e-6
    assert abs(float(comp_widget.components[0].s_edit.text()) - 0.1) < 1e-6
    assert comp_widget.components[0].name_edit.text() == "Test Component 1"

    # Clearing all components removes the dots and empties every field.
    assert comp_widget.components[0].dot is not None
    assert comp_widget.components[1].dot is not None
    comp_widget._clear_components()
    assert comp_widget.components[0].dot is None
    assert comp_widget.components[1].dot is None
    assert comp_widget.components[0].g_edit.text() == ""
    assert comp_widget.components[0].name_edit.text() == ""
    assert comp_widget.components[1].g_edit.text() == ""
    assert comp_widget.components[1].name_edit.text() == ""


def test_components_linear_projection_fractions_and_colormap(
    make_viewer_model,
    qtbot,
):
    """Linear Projection writes the expected fraction layer, whose colormap
    and contrast edits are followed by the tab."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    # Ensure Linear Projection is selected
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")

    # Define two components
    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget._on_component_coords_changed(0)

    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(1)

    # Calculate expected fractions using the new array-based metadata
    metadata = layer.metadata
    G_image = metadata["G"]
    S_image = metadata["S"]
    harmonics = metadata.get("harmonics", [1])

    # Get the harmonic index (0-based for array access)
    harmonic = parent.harmonic
    if isinstance(harmonics, (list, np.ndarray)) and len(harmonics) > 1:
        # Multi-harmonic case: G and S have shape (n_harmonics, ...)
        harmonic_idx = (
            list(harmonics).index(harmonic) if harmonic in harmonics else 0
        )
        real = G_image[harmonic_idx].flatten()
        imag = S_image[harmonic_idx].flatten()
    else:
        # Single harmonic case
        real = G_image.flatten()
        imag = S_image.flatten()

    expected_comp1_fractions = phasor_component_fraction(
        np.array(real),
        np.array(imag),
        (0.2, 0.8),
        (0.1, 0.5),
    )
    expected_comp1_fractions = expected_comp1_fractions.reshape(
        layer.data.shape
    )
    # Apply NaN masking for invalid values (outside 0-1 range)
    expected_comp1_fractions = np.where(
        np.isfinite(expected_comp1_fractions)
        & (expected_comp1_fractions >= 0)
        & (expected_comp1_fractions <= 1),
        expected_comp1_fractions,
        np.nan,
    )

    comp_widget._run_analysis()

    # Only comp1 fractions layer should be created (comp2 is no longer created)
    assert comp_widget.comp1_fractions_layer in viewer.layers

    # Check data
    comp1_data = comp_widget.comp1_fractions_layer.data
    np.testing.assert_allclose(
        comp1_data,
        expected_comp1_fractions,
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )

    # Check initial colormap
    assert comp_widget.comp1_fractions_layer.colormap.name == 'jet'

    assert isinstance(comp_widget.component_line, LineCollection)

    # Changing the colormap on the comp1 layer updates the stored gradient.
    orig_comp1_colors = (
        comp_widget.comp1_fractions_layer.colormap.colors.copy()
    )
    orig_colors = comp_widget.fractions_colormap.copy()
    comp_widget.comp1_fractions_layer.colormap = 'viridis'
    comp1_colors = comp_widget.comp1_fractions_layer.colormap.colors
    assert not np.allclose(orig_comp1_colors, comp1_colors)
    assert comp_widget.fractions_colormap is not None
    assert comp_widget.comp1_fractions_layer.colormap.name == 'viridis'
    assert not np.array_equal(orig_colors, comp_widget.fractions_colormap)

    # A built-in colormap name (which might not have a colors attribute)
    # falls back to default colormaps without crashing.
    comp_widget.comp1_fractions_layer.colormap = 'gray'
    assert comp_widget.comp1_fractions_layer.colormap is not None

    # Changing the fraction layer's colormap/contrast triggers handlers.
    fl = comp_widget.comp1_fractions_layer
    fl.contrast_limits = (0.1, 0.9)
    fl.colormap = "viridis"
    assert comp_widget.colormap_contrast_limits is not None


def test_components_fraction_histogram_overlay_and_line_settings(
    make_viewer_model,
    qtbot,
):
    """The fraction histogram overlay drawn along the line, and the line
    settings dialog."""
    from matplotlib.image import AxesImage

    viewer, layer, parent, comp_widget = _setup_components(make_viewer_model)
    _setup_linear_projection(comp_widget)

    # No overlay by default.
    assert comp_widget.show_fraction_histogram is False
    assert comp_widget.component_histogram is None

    # Enable the overlay and redraw.
    comp_widget.show_fraction_histogram = True
    comp_widget.draw_line_between_components()
    assert comp_widget.component_histogram is not None
    # The overlay is a single seamless gradient image (no outline).
    assert len(comp_widget.component_histogram) == 1
    fill = comp_widget.component_histogram[0]
    assert isinstance(fill, AxesImage)
    assert fill in comp_widget.get_all_artists()
    assert fill.get_alpha() == comp_widget.histogram_alpha

    # Disable again -> overlay removed on next draw.
    comp_widget.show_fraction_histogram = False
    comp_widget.draw_line_between_components()
    assert comp_widget.component_histogram is None

    comp_widget.show_fraction_histogram = True

    # Overlay height scales the histogram profile.
    def _profile_height():
        comp_widget.draw_line_between_components()
        # The gradient image extent is in the local (u, v) frame; its top edge
        # (v_max) scales linearly with the height setting.
        im = comp_widget.component_histogram[0]
        return im.get_extent()[3]

    comp_widget.histogram_overlay_height = 0.1
    small = _profile_height()
    comp_widget.histogram_overlay_height = 0.6
    large = _profile_height()
    assert large > small
    assert np.isclose(large / small, 6.0, rtol=1e-3)

    # A negative histogram offset mirrors the overlay to the other side.
    def _apex_display():
        comp_widget.draw_line_between_components()
        im = comp_widget.component_histogram[0]
        # Map the local apex (u=0, top of the profile) through the image
        # transform to display coordinates.
        v_max = im.get_extent()[3]
        return im.get_transform().transform((0.0, v_max))

    comp_widget.histogram_offset = 0.1
    pos_apex = _apex_display()
    comp_widget.histogram_offset = -0.1
    neg_apex = _apex_display()
    assert not np.allclose(pos_apex, neg_apex)

    # Offset and transparency edits are stored (unsaved until the analysis
    # runs again) and redrawn.
    comp_widget._on_histogram_offset_changed(0.3)
    assert comp_widget.histogram_offset == 0.3
    settings = parent.layer_settings(layer)["component_analysis"]
    assert settings["two_component_line_settings"]["histogram_offset"] == 0.3

    comp_widget._on_histogram_transparency_changed(0.4)
    assert abs(comp_widget.histogram_alpha - 0.6) < 1e-9
    settings = parent.layer_settings(layer)["component_analysis"]
    assert (
        abs(settings["two_component_line_settings"]["histogram_alpha"] - 0.6)
        < 1e-9
    )

    # The line settings dialog.
    comp_widget._open_plot_settings_dialog()
    assert comp_widget.plot_dialog.isVisible()

    comp_widget.colormap_line_checkbox.setChecked(False)
    comp_widget._on_plot_setting_changed()
    assert not comp_widget.show_colormap_line
    comp_widget.colormap_line_checkbox.setChecked(True)
    comp_widget._on_plot_setting_changed()
    assert comp_widget.show_colormap_line

    # Change offset (slider uses 3-decimal factor: 120 -> 0.120)
    comp_widget.line_offset_slider.setValue(120)
    assert abs(comp_widget.line_offset - 0.12) < 1e-6
    # Values can also be typed directly via the spinbox, which drives the
    # slider.
    comp_widget.line_offset_spin.setValue(-0.25)
    assert abs(comp_widget.line_offset + 0.25) < 1e-6
    assert comp_widget.line_offset_slider.value() == -250

    comp_widget.line_width_spin.setValue(5.0)
    assert comp_widget.line_width == 5.0

    # Transparency (inverse of alpha): 0.45 transparency -> 0.55 opacity
    comp_widget.line_transparency_spin.setValue(0.45)
    assert abs(comp_widget.line_alpha - 0.55) < 1e-6


def test_components_adding_and_removing_components(make_viewer_model, qtbot):
    """Component count drives the remove buttons, numbering and methods."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    # Initially should have 2 components
    assert len(comp.components) == 2
    assert comp.add_component_btn.isEnabled()
    assert not comp.components[0].remove_button.isEnabled()

    # With 2 components, every two-component method is available.
    assert [
        comp.analysis_type_combo.itemText(i)
        for i in range(comp.analysis_type_combo.count())
    ] == ["Linear Projection", "Component Fit", ABSOLUTE_CONCENTRATION]

    # The run button's text follows the analysis type.
    assert comp.calculate_button.text() == "Display Component Fraction Images"
    comp.analysis_type_combo.setCurrentText("Component Fit")
    assert comp.calculate_button.text() == "Run Multi-Component Analysis"
    comp.analysis_type_combo.setCurrentText("Linear Projection")
    assert comp.calculate_button.text() == "Display Component Fraction Images"

    # Adding a component enables removal and leaves only Component Fit.
    comp._add_component()
    assert len(comp.components) == 3
    assert comp.components[0].remove_button.isEnabled()
    assert comp.analysis_type_combo.count() == 1
    assert comp.analysis_type_combo.itemText(0) == "Component Fit"

    # Removing brings it back to the minimum.
    comp._remove_component()
    assert len(comp.components) == 2
    assert not comp.components[0].remove_button.isEnabled()

    # Clicking the remove button on a specific component removes it and
    # renumbers the remaining ones.
    comp._add_component()
    assert len(comp.components) == 3
    comp.components[0].name_edit.setText("Comp A")
    comp.components[1].name_edit.setText("Comp B")
    comp.components[2].name_edit.setText("Comp C")
    for c in comp.components:
        assert c.remove_button.isEnabled()
    comp.components[1].remove_button.click()
    assert len(comp.components) == 2
    assert comp.components[0].name_edit.text() == "Comp A"
    assert comp.components[0].idx == 0
    assert comp.components[0].number_label.text() == "1."
    assert comp.components[1].name_edit.text() == "Comp C"
    assert comp.components[1].idx == 1
    assert comp.components[1].number_label.text() == "2."
    assert not comp.components[0].remove_button.isEnabled()
    assert not comp.components[1].remove_button.isEnabled()

    # Cannot remove when count <= 2
    comp._remove_component()
    assert len(comp.components) == 2

    comp._add_component()
    comp._add_component()
    assert len(comp.components) == 4

    # Remove with invalid indices (no-ops)
    comp._remove_component(-1)
    comp._remove_component(100)
    assert len(comp.components) == 4

    # Select component 0, remove component 2 (was_selected is False)
    comp._select_component_item(0)
    comp._remove_component(2)
    assert len(comp.components) == 3
    assert comp._selected_component is comp.components[0]
    assert comp.components[0].card_frame.property("selected")

    # Remove passing ComponentState instance
    comp_to_remove = comp.components[2]
    comp._remove_component(comp_to_remove)
    assert len(comp.components) == 2

    # Add back to 3 and remove with idx=None (removes last component)
    comp._add_component()
    assert len(comp.components) == 3
    comp._select_component_item(2)  # last component selected
    comp._remove_component(None)
    assert len(comp.components) == 2
    # Since was_selected was True on index 2, new selection is
    # min(2, len - 1) = 1
    assert comp._selected_component is comp.components[1]
    assert comp.components[1].card_frame.property("selected")

    # Backward compatibility stubs execute without raising.
    comp._refresh_editor_title()
    comp._update_row_coords_label(0)
    comp._update_all_row_coords_labels()

    # Absolute concentration: an incomplete calibration warns and stores
    # nothing.
    _concentration_ready(viewer, layer, parent, comp, reference=False)
    with patch("napari_phasors.components_tab.show_warning") as warn:
        comp._run_analysis()
    warn.assert_called_once()
    assert "reference solution" in warn.call_args[0][0]
    assert _concentration_maps(viewer) == {}
    assert 'component_analysis' not in (layer.metadata.get('settings') or {})

    # Choosing the method swaps in its inputs, labels and histogram axis.
    assert not comp.concentration_box.isHidden()
    assert comp.add_component_btn.isHidden()
    assert comp.calculate_button.text() == "Calculate Absolute Concentrations"
    assert comp.histogram_widget.xlabel == "Concentration (mM)"
    assert comp.histogram_widget.range_label.text() == (
        "Concentration range (mM):"
    )
    assert [
        comp.calibrated_component_combo.itemText(i)
        for i in range(comp.calibrated_component_combo.count())
    ] == ["Component 1", "Component 2"]
    assert "Component 2" in comp.second_component_checkbox.text()
    assert [
        comp.reference_source_combo.itemText(i)
        for i in range(comp.reference_source_combo.count())
    ] == [MANUAL_REFERENCE, layer.name, "reference"]
    assert not comp.reference_mean_edit.isReadOnly()
    assert not comp.brightness_ratio_edit.isEnabled()
    comp.second_component_checkbox.setChecked(True)
    assert comp.brightness_ratio_edit.isEnabled()
    comp.calibrated_component_combo.setCurrentIndex(1)
    assert "Component 1" in comp.second_component_checkbox.text()
    comp.reference_source_combo.setCurrentText("reference")
    assert comp.reference_mean_edit.isReadOnly()
    comp.analysis_type_combo.setCurrentText("Linear Projection")
    assert comp.concentration_box.isHidden()
    assert comp.histogram_widget.xlabel == "Fraction"
    assert comp.histogram_widget.range_factor == 1000
    assert not comp.add_component_btn.isHidden()
    comp.analysis_type_combo.setCurrentText(ABSOLUTE_CONCENTRATION)
    assert comp.add_component_btn.isHidden()
    comp._add_component()
    assert comp.analysis_type == "Component Fit"
    assert comp.concentration_box.isHidden()
    assert not comp.add_component_btn.isHidden()

    # The run button says which calibration input is missing or wrong.
    _concentration_ready(viewer, layer, parent, comp, reference=False)
    assert "reference solution's layer" in comp._components_validation()
    _type_reference(comp, -3, 0.8, 0.3)
    assert "intensity must be a positive" in comp._components_validation()
    _type_reference(comp, 3, 0.8, 0.3)
    assert comp._components_validation() is None
    comp.reference_concentration_edit.setText("0")
    assert "reference concentration" in comp._components_validation()
    comp.reference_concentration_edit.setText("1")
    comp.second_component_checkbox.setChecked(True)
    comp.brightness_ratio_edit.setText("abc")
    assert "brightness ratio" in comp._components_validation()
    comp.brightness_ratio_edit.setText("1.5")
    assert comp._components_validation() is None
    # The model needs the two components apart along G.
    comp.components[1].g_edit.setText(str(_CONC_COMPONENTS[0][0]))
    comp._on_component_coords_changed(1)
    assert "different G" in comp._components_validation()
    comp.components[1].g_edit.setText(str(_CONC_COMPONENTS[1][0]))
    comp._on_component_coords_changed(1)
    # A reference layer with nothing to measure is named, and so is an
    # entry naming a layer that is not there.
    with patch(
        "napari_phasors.components_tab.phasor_reference_from_layer",
        return_value=None,
    ):
        comp.reference_source_combo.setCurrentText("reference")
    assert "No usable phasor data in reference" in (
        comp._components_validation()
    )
    assert not comp.reference_note.isHidden()
    assert comp.reference_mean_edit.text() == ""
    comp.reference_source_combo.addItem("closed layer")
    comp.reference_source_combo.setCurrentText("closed layer")
    assert comp._reference_layer() is None
    assert "No usable phasor data in closed layer" in (
        comp._components_validation()
    )

    # Fraction filters stay fraction filters on the two components.
    _concentration_ready(viewer, layer, parent, comp)
    params = comp._component_filter_params(0)
    assert params['analysis_type'] == "Linear Projection"
    assert params['component_real'] == [0.9, 0.25]
    assert [index for index, *_ in comp._filterable_components()] == [0, 1]
    assert "Run the component analysis" in comp._filter_enable_blocked_reason()
    _enable_second_component(comp)
    comp._run_analysis()
    assert comp._has_analysed_fractions()
    assert comp._filter_enable_blocked_reason() is None

    # Concentration maps are never another method's data.
    total = viewer.layers[_cname(layer, "Total")]
    assert comp._find_component_index_for_layer(total) is None
    assert comp._layer_matches_analysis_type(total)
    comp.analysis_type_combo.setCurrentText("Component Fit")
    assert not comp._layer_matches_analysis_type(total)
    assert comp._get_component_names_from_fraction_layers() == []
    comp.analysis_type_combo.setCurrentText("Linear Projection")
    assert not comp._layer_matches_analysis_type(total)
    assert comp._get_all_layers_for_component(0) == []


def test_components_three_component_fit(make_viewer_model, qtbot):
    """Three components draw a polygon, and the fit writes one fraction
    layer each; re-displaying keeps manual colormap/contrast/gamma."""
    viewer, layer, parent, comp_widget = _setup_components(make_viewer_model)

    # Add third component
    comp_widget._add_component()

    # Set analysis type to Component Fit
    comp_widget.analysis_type_combo.setCurrentText("Component Fit")
    assert comp_widget.analysis_type == "Component Fit"

    # Define three components
    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget._on_component_coords_changed(0)

    comp_widget.components[1].g_edit.setText("0.5")
    comp_widget.components[1].s_edit.setText("0.3")
    comp_widget._on_component_coords_changed(1)

    comp_widget.components[2].g_edit.setText("0.8")
    comp_widget.components[2].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(2)

    # Should create polygon instead of line
    assert comp_widget.component_polygon is not None
    assert comp_widget.component_line is None

    # Run analysis - this should create fraction layers using component fit
    comp_widget._run_analysis()
    assert len(comp_widget.fraction_layers) == 3

    # Component-colour helper for both the Component Fit (>=2 layers) path
    # and a smaller count.
    assert comp_widget._get_component_colors_for_count(3) is not None
    assert comp_widget._get_component_colors_for_count(2) is not None

    # Record the default colormap of an untouched layer, then manually tweak
    # a different one's colormap, contrast limits and gamma.
    other_layer = comp_widget.fraction_layers[0]
    other_colormap = other_layer.colormap.name

    tweaked_layer = comp_widget.fraction_layers[1]
    tweaked_layer.colormap = "magma"
    tweaked_layer.contrast_limits = (0.15, 0.85)
    tweaked_layer.gamma = 0.4

    # Press "Display Component Fraction Images" again.
    comp_widget._run_analysis()

    new_tweaked = viewer.layers[tweaked_layer.name]
    assert new_tweaked.colormap.name == "magma"
    assert np.allclose(new_tweaked.contrast_limits, (0.15, 0.85))
    assert new_tweaked.gamma == 0.4

    # The untouched layer must keep its (default) colormap, not reset.
    new_other = viewer.layers[other_layer.name]
    assert new_other.colormap.name == other_colormap

    # Removing the last layer clears the drawn polygon.
    for fraction in list(comp_widget.fraction_layers):
        viewer.layers.remove(fraction)
    viewer.layers.remove(layer)
    assert comp_widget.component_polygon is None


def test_components_fraction_range_clips_from_the_original_data(
    make_viewer_model,
    qtbot,
):
    """The range slider clips the fraction layer from its original values,
    in first- or second-component space, and spans every checked one."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)
    _setup_linear_projection(comp_widget)

    name1, name2 = comp_widget._linear_projection_component_names()
    fraction_layer = comp_widget.comp1_fractions_layer

    # The slider covers all checked components, not just the first one.
    _check_histogram_components(comp_widget, [name1, name2])
    original = np.asarray(
        fraction_layer.metadata.get(
            'fraction_data_original', fraction_layer.data
        ),
        dtype=float,
    )
    original = original[np.isfinite(original)]
    # The second component is 1 - the first, so together they span both ends.
    expected_min = min(float(original.min()), float(1.0 - original.max()))
    expected_max = max(float(original.max()), float(1.0 - original.min()))
    factor = comp_widget.histogram_widget.range_factor
    slider = comp_widget.histogram_widget.range_slider
    assert slider.minimum() / factor == pytest.approx(expected_min, abs=1e-3)
    assert slider.maximum() / factor == pytest.approx(expected_max, abs=1e-3)
    # The handles are not clipped to the first component's contrast limits
    # either, so neither distribution is cut off.
    range_min, range_max = comp_widget.histogram_widget.get_range()
    assert range_min == pytest.approx(expected_min, abs=1e-2)
    assert range_max == pytest.approx(expected_max, abs=1e-2)

    # The shared fraction layer is clipped once when both components show:
    # the first component owns the layer, so it is clipped to the slider
    # range rather than to the second component's mirrored range.
    original = fraction_layer.metadata.get(
        'fraction_data_original', fraction_layer.data
    ).copy()
    comp_widget._on_fraction_range_changed(0.2, 0.8)
    np.testing.assert_allclose(
        fraction_layer.data,
        np.clip(original, 0.2, 0.8),
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )
    assert len(comp_widget.histogram_widget._datasets) == 2
    comp_widget._on_fraction_range_changed(0.0, 1.0)

    # One component: the layer is clipped, and expanding the range restores
    # values from the original data, not from the already-clipped layer.
    _check_histogram_components(comp_widget, [name1])
    original_data = fraction_layer.data.copy()
    min_val, max_val = 0.2, 0.8
    comp_widget._on_fraction_range_changed(min_val, max_val)
    np.testing.assert_allclose(
        fraction_layer.data,
        np.clip(original_data, min_val, max_val),
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )
    assert tuple(fraction_layer.contrast_limits) == (min_val, max_val)
    comp_widget._on_fraction_range_changed(0.0, 1.0)
    np.testing.assert_allclose(
        fraction_layer.data,
        np.clip(original_data, 0.0, 1.0),
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )

    # Clipping the second-component fraction to [0.2, 0.6] is equivalent to
    # clipping the underlying first-component layer to [0.4, 0.8].
    _check_histogram_components(comp_widget, [name2])
    original_data = fraction_layer.data.copy()
    comp_widget._on_fraction_range_changed(0.2, 0.6)
    np.testing.assert_allclose(
        fraction_layer.data,
        np.clip(original_data, 0.4, 0.8),
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(fraction_layer.contrast_limits),
        [0.4, 0.8],
        atol=1e-9,
    )
    # The phasor-line gradient stays in first-component fraction space.
    np.testing.assert_allclose(
        comp_widget.colormap_contrast_limits, [0.4, 0.8], atol=1e-9
    )
    # The histogram shows the inverted distribution with the reversed
    # colormap.
    np.testing.assert_allclose(
        comp_widget.histogram_widget.colormap_colors,
        np.asarray(fraction_layer.colormap.colors)[::-1],
    )
    displayed = comp_widget.histogram_widget._raw_valid_data
    expected = 1.0 - np.clip(original_data, 0.4, 0.8)
    expected = expected[np.isfinite(expected)]
    np.testing.assert_allclose(
        np.sort(displayed), np.sort(expected), rtol=1e-6, atol=1e-9
    )


def test_components_second_component_and_several_curves(
    make_viewer_model,
    qtbot,
):
    """The second component is 1 - the first with a reversed colormap, and
    checking both plots two curves in their layers' colormaps."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)
    _setup_linear_projection(comp_widget)

    name1, name2 = comp_widget._linear_projection_component_names()
    assert name1 is not None and name2 is not None
    fraction_layer = comp_widget.comp1_fractions_layer
    histogram = comp_widget.histogram_widget

    # Both components must be toggleable on their cards.
    assert _histogram_toggle(comp_widget, name1).isEnabled()
    assert _histogram_toggle(comp_widget, name2).isEnabled()

    fraction_layers_map, invert = comp_widget._resolve_histogram_component(
        name2
    )
    assert invert is True
    assert fraction_layers_map  # underlying first-component layers

    # Selecting the second component displays the inverted distribution.
    _check_histogram_components(comp_widget, [name2])
    comp_widget.update_component_histogram()
    displayed = histogram._raw_valid_data
    expected = 1.0 - fraction_layer.data
    expected = expected[np.isfinite(expected)]
    np.testing.assert_allclose(
        np.sort(displayed),
        np.sort(expected),
        rtol=1e-6,
        atol=1e-9,
    )
    # The histogram colormap must be the reversed first-component colormap.
    np.testing.assert_allclose(
        histogram.colormap_colors,
        np.asarray(fraction_layer.colormap.colors)[::-1],
    )

    # Changing the first component's colormap reverses it on the histogram.
    fraction_layer.colormap = "viridis"
    np.testing.assert_allclose(
        histogram.colormap_colors,
        np.asarray(fraction_layer.colormap.colors)[::-1],
    )
    # Changing the contrast limits flips them into second-component space.
    fraction_layer.contrast_limits = [0.2, 0.7]
    np.testing.assert_allclose(
        np.asarray(histogram.contrast_limits),
        [1.0 - 0.7, 1.0 - 0.2],
        atol=1e-9,
    )
    # With no component selected, a first-layer colormap change is a no-op
    # for the overlay (the refresh guard short-circuits).
    _check_histogram_components(comp_widget, [])
    previous_colors = histogram.colormap_colors
    fraction_layer.colormap = "magma"
    assert histogram.colormap_colors is previous_colors

    # One component: its colormap gradient carries the fraction scale.
    _check_histogram_components(comp_widget, [name1])
    assert histogram._series_style == "colormap"

    # Checking two components plots both fraction distributions together.
    _check_histogram_components(comp_widget, [name1, name2])
    datasets = histogram._datasets
    assert len(datasets) == 2
    first = np.asarray(fraction_layer.data, dtype=float).ravel()
    first = first[np.isfinite(first)]
    second = 1.0 - first
    plotted = sorted(datasets, key=lambda label: name2 in label)
    np.testing.assert_allclose(
        np.sort(datasets[plotted[0]]), np.sort(first), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.sort(datasets[plotted[1]]), np.sort(second), rtol=1e-6
    )

    # Both distributions trace back to the same analysed image, so grouping
    # made in any tab applies to both of them.
    sources = histogram._dataset_sources
    assert set(sources) == set(datasets)
    assert set(sources.values()) == {layer.name}

    # The statistics dock keeps one row per analysed layer and adds a column
    # block per component instead of repeating the layer once per component.
    stats_table = parent.components_statistics_dock_widget.layer_stats_table
    assert stats_table.rowCount() == 1
    assert stats_table.item(0, 0).text() == layer.name
    headers = [
        stats_table.horizontalHeaderItem(col).text()
        for col in range(stats_table.columnCount())
    ]
    block = len(StatisticsTableWidget.COLUMNS) - 1
    assert headers[1].startswith(name1)
    assert headers[1 + block].startswith(name2)

    # Merged mode pools layers, not components: each component keeps its own
    # curve. Their colormaps are mirror images, so gradients would only
    # confuse: solid colours instead, one per component.
    assert histogram.display_mode == "Merged"
    assert histogram._series_style == "solid"
    curves = histogram.ax.lines
    assert len(curves) == 2
    assert [line.get_label() for line in curves] == [name1, name2]

    # Each component curve reads its layer's colormap, the second one
    # reversed to match its inverted fraction scale.
    stored = histogram._series_colormaps[name1]
    np.testing.assert_allclose(stored[0], fraction_layer.colormap.colors)
    np.testing.assert_allclose(
        histogram._series_colormaps[name2][0],
        np.asarray(fraction_layer.colormap.colors)[::-1],
    )

    # A choice made in the settings dialog is not overridden afterwards.
    histogram._series_style = "colormap"
    histogram._series_style_explicit = True
    comp_widget.update_component_histogram()
    assert histogram._series_style == "colormap"

    # Asking for the colormap draws each curve as a gradient in its own
    # colormap.
    histogram.set_default_series_style("colormap")
    histogram._render()
    curves = [
        artist
        for artist in histogram.ax.collections
        if isinstance(artist, LineCollection)
    ]
    assert len(curves) == 2

    # Changing the layer's colormap updates the curve without a re-analysis.
    fraction_layer.colormap = "viridis"
    np.testing.assert_allclose(
        histogram._series_colormaps[name1][0],
        fraction_layer.colormap.colors,
    )
    assert not np.allclose(histogram._series_colormaps[name1][0], stored[0])

    # Unchecking one goes back to a single distribution.
    _histogram_toggle(comp_widget, name2).setChecked(False)
    assert len(histogram._datasets) == 1


def test_components_card_toggles_drive_histogram_and_statistics(
    make_viewer_model,
    qtbot,
    monkeypatch,
):
    """The card toggle is the only selector, and it feeds both docks."""
    from qtpy.QtWidgets import QDialog

    from napari_phasors._utils import HistogramSettingsDialog

    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)
    _setup_linear_projection(comp_widget)

    # The old checkable comboboxes are gone from the widget and from both
    # docks; the per-card toggle replaces them.
    assert not hasattr(comp_widget, "histogram_component_combobox")
    assert not hasattr(comp_widget, "stats_component_combobox")

    names = _available_histogram_components(comp_widget)
    assert len(names) == 2

    # Every card carries a labelled toggle.
    for comp in comp_widget.components:
        assert comp.histogram_checkbox is not None
        assert (
            comp.histogram_checkbox.text()
            == "Show in histogram and statistics"
        )

    # The first component is checked by default, so neither dock starts blank.
    assert comp_widget._selected_histogram_components() == [names[0]]

    stats_table = parent.components_statistics_dock_widget.layer_stats_table

    def stats_headers():
        return [
            stats_table.horizontalHeaderItem(col).text()
            for col in range(stats_table.columnCount())
        ]

    # Checking the second card adds its curve and its statistics columns.
    _histogram_toggle(comp_widget, names[1]).setChecked(True)
    assert comp_widget._selected_histogram_components() == names
    assert comp_widget.histogram_widget._series_names() == names
    assert any(header.startswith(names[1]) for header in stats_headers())

    # Unchecking the first card drops it from both the plot and the table.
    _histogram_toggle(comp_widget, names[0]).setChecked(False)
    assert comp_widget._selected_histogram_components() == [names[1]]
    assert comp_widget.histogram_widget._series_names() == [names[1]]
    assert not any(header.startswith(names[0]) for header in stats_headers())

    name1, name2 = comp_widget._linear_projection_component_names()
    _check_histogram_components(comp_widget, [name1])

    # Toggle syncing survives missing cards, missing toggles and reentrancy.
    # A hole in the component list (left by teardown) resolves to no name.
    original = comp_widget.components[1]
    comp_widget.components[1] = None
    try:
        assert comp_widget._component_display_name(1) == ""
        comp_widget._sync_component_histogram_toggles()
    finally:
        comp_widget.components[1] = original

    # A card without a toggle is skipped rather than raising.
    checkbox = comp_widget.components[1].histogram_checkbox
    comp_widget.components[1].histogram_checkbox = None
    try:
        comp_widget._sync_component_histogram_toggles()
    finally:
        comp_widget.components[1].histogram_checkbox = checkbox

    # A sync triggered from inside a sync is a no-op, so a toggle already
    # being written cannot be clobbered halfway through.
    comp_widget._syncing_component_toggles = True
    try:
        checkbox.setChecked(True)
        comp_widget._sync_component_histogram_toggles()
        assert checkbox.isChecked()
    finally:
        comp_widget._syncing_component_toggles = False
    assert comp_widget._selected_histogram_components() == [name1]
    checkbox.blockSignals(True)
    checkbox.setChecked(False)
    checkbox.blockSignals(False)

    # Time-lapse frame refreshes are skipped while nothing is plotted.
    with patch.object(comp_widget, "update_component_histogram") as mock:
        comp_widget.refresh_for_frame_change()
        mock.assert_called_once()
    comp_widget._histogram_components = []
    with patch.object(comp_widget, "update_component_histogram") as mock:
        comp_widget.refresh_for_frame_change()
        mock.assert_not_called()

    # Re-emitting the current state, or a missing card, is a no-op.
    _check_histogram_components(comp_widget, [name1])
    comp_widget._on_component_histogram_toggled(0, True)
    assert comp_widget._selected_histogram_components() == [name1]
    comp_widget._on_component_histogram_toggled(1, False)
    assert comp_widget._selected_histogram_components() == [name1]

    # An out-of-range card index resolves to no name and is ignored.
    assert comp_widget._component_display_name(99) == ""
    assert comp_widget._component_display_name(-1) == ""
    comp_widget._on_component_histogram_toggled(99, True)
    assert comp_widget._selected_histogram_components() == [name1]

    # While the toggles are being synced, user-facing handlers stay inert.
    comp_widget._syncing_component_toggles = True
    try:
        comp_widget._on_component_histogram_toggled(1, True)
    finally:
        comp_widget._syncing_component_toggles = False
    assert comp_widget._selected_histogram_components() == [name1]

    # A committed rename carries the selection onto the new display name.
    _rename_component(comp_widget, 0, "Free NADH")
    assert comp_widget._selected_histogram_components() == ["Free NADH"]
    assert comp_widget.components[0].histogram_checkbox.isChecked()
    assert _histogram_toggle(comp_widget, name2) is not None
    assert _histogram_toggle(comp_widget, "Not a component") is None

    # Grouping the fraction histogram tags the analysed image layer: groups
    # are keyed by the analysed image layer, not by the fraction curve.
    histogram = comp_widget.histogram_widget
    histogram._group_assignments = {layer.name: 1}
    histogram._group_names = {1: 'Ctrl'}
    histogram._group_colors = {1: (1.0, 0.0, 0.0)}

    def fake_exec(self):
        self.mode_combo.setCurrentText('Grouped')
        return QDialog.Accepted

    monkeypatch.setattr(HistogramSettingsDialog, 'exec', fake_exec)
    histogram._open_settings_dialog()
    group = viewer.layers[layer.name].metadata['settings']['group']
    assert group['name'] == 'Ctrl'

    # Typing a name updates the card only; Enter / focus-out applies it.
    name_edit = comp_widget.components[0].name_edit
    old_layer_name = comp_widget.comp1_fractions_layer.name
    # Typing letter by letter must not rename layers, rewrite metadata or
    # redraw the histogram once per keystroke.
    with patch.object(
        comp_widget, "_refresh_histogram_after_rename"
    ) as mock_refresh:
        for i in range(1, len("Free") + 1):
            name_edit.setText("Free"[:i])
    mock_refresh.assert_not_called()
    assert comp_widget.comp1_fractions_layer.name == old_layer_name
    assert comp_widget._selected_histogram_components() != ["Free"]
    # The card's own title still follows every keystroke.
    assert comp_widget._component_display_name(0) == "Free"
    # Committing the edit (Enter, or leaving the field) applies it everywhere.
    name_edit.editingFinished.emit()
    assert comp_widget.comp1_fractions_layer.name.endswith(
        "[(Linear Projection) Free]"
    )
    assert comp_widget._selected_histogram_components() == ["Free"]
    # Re-committing an unchanged name is a no-op.
    with patch.object(
        comp_widget, "_propagate_component_name"
    ) as mock_propagate:
        name_edit.editingFinished.emit()
    mock_propagate.assert_not_called()

    # Without data the histogram keeps its empty axes instead of hiding:
    # removing the only fraction layer leaves the selection unresolvable.
    hw = comp_widget.histogram_widget
    viewer.layers.remove(comp_widget.comp1_fractions_layer)
    with patch.object(hw, "hide") as mock_hide:
        comp_widget.update_component_histogram()
    mock_hide.assert_not_called()
    assert not hw.isHidden()
    assert hw.counts is None
    assert hw._datasets == {}
    # The axes are still drawn: spines, ticks and labels are all in place.
    assert hw.ax.get_xlabel() == "Fraction"
    assert hw.ax.get_ylabel() == "Pixel count"
    assert hw.ax.spines["bottom"].get_visible()
    assert len(hw.ax.lines) == 0
    # An empty selection is drawn the same way.
    comp_widget._histogram_components = []
    with patch.object(hw, "hide") as mock_hide:
        comp_widget.update_component_histogram()
    mock_hide.assert_not_called()
    assert not hw.isHidden()
    assert hw.counts is None


def test_components_histogram_multi_layer_linear_projection(
    make_viewer_model,
    qtbot,
):
    """Linear projection over multiple layers feeds a per-layer histogram."""
    viewer = make_viewer_model()
    layer_a = create_image_layer_with_phasors()
    layer_a.name = "layer_a"
    layer_b = create_image_layer_with_phasors()
    layer_b.name = "layer_b"
    viewer.add_layer(layer_a)
    viewer.add_layer(layer_b)

    parent = PlotterWidget(viewer)
    parent.image_layers_checkable_combobox.setCheckedItems(
        [layer_a.name, layer_b.name]
    )
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")
    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(1)

    with patch.object(
        parent, "get_selected_layers", return_value=[layer_a, layer_b]
    ):
        comp_widget._run_analysis()

    name1, _ = comp_widget._linear_projection_component_names()
    fraction_layers_map, invert = comp_widget._resolve_histogram_component(
        name1
    )
    assert invert is False
    assert set(fraction_layers_map.keys()) == {"layer_a", "layer_b"}

    _check_histogram_components(comp_widget, [name1])
    comp_widget.update_component_histogram()

    # Each selected layer feeds its own dataset (as in the FRET tab), so the
    # Merged / Individual layers / Grouped display modes and per-row
    # statistics all work. Rows are named after the analysis fraction layers.
    expected_keys = {
        _lp_name(name1, "layer_a"),
        _lp_name(name1, "layer_b"),
    }
    assert set(comp_widget.histogram_widget._datasets.keys()) == expected_keys

    # The range slider clips every layer and keeps the per-layer datasets.
    comp_widget._on_fraction_range_changed(0.2, 0.8)
    assert set(comp_widget.histogram_widget._datasets.keys()) == expected_keys

    # Early returns: an empty and an unresolved selection are both no-ops.
    comp_widget._histogram_components = []
    comp_widget._on_fraction_range_changed(0.1, 0.9)

    comp_widget._histogram_components = ["Ghost component"]
    comp_widget._on_fraction_range_changed(0.1, 0.9)


def test_components_gamma_links_layers_and_histogram(
    make_viewer_model,
    qtbot,
):
    """Changing gamma on one fraction layer syncs siblings and the histogram."""
    viewer = make_viewer_model()
    layer_a = create_image_layer_with_phasors()
    layer_a.name = "layer_a"
    layer_b = create_image_layer_with_phasors()
    layer_b.name = "layer_b"
    viewer.add_layer(layer_a)
    viewer.add_layer(layer_b)

    parent = PlotterWidget(viewer)
    parent.image_layers_checkable_combobox.setCheckedItems(
        [layer_a.name, layer_b.name]
    )
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")
    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(1)

    with patch.object(
        parent, "get_selected_layers", return_value=[layer_a, layer_b]
    ):
        comp_widget._run_analysis()

    # The first component owns one fraction layer per analyzed image.
    first_component_layers = comp_widget._get_all_layers_for_component(0)
    assert len(first_component_layers) == 2

    name1, _ = comp_widget._linear_projection_component_names()
    _check_histogram_components(comp_widget, [name1])

    # Changing gamma on one layer propagates to the sibling layer, the stored
    # gradient gamma, and the histogram widget.
    first_component_layers[0].gamma = 0.5

    assert first_component_layers[1].gamma == 0.5
    assert comp_widget.fractions_gamma == 0.5
    assert comp_widget.histogram_widget.gamma == 0.5


def test_components_rename_follows_into_histogram_and_statistics(
    make_napari_viewer,
):
    """Renaming a component relabels its curve, columns and every image."""
    viewer = make_napari_viewer()
    for index in range(2):
        layer = create_image_layer_with_phasors()
        layer.name = f"img{index}"
        viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    parent.image_layers_checkable_combobox.setCheckedItems(["img0", "img1"])
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)

    comp_widget.analysis_type_combo.setCurrentText("Component Fit")
    for index, (g, s_coord) in enumerate(((0.15, 0.1), (0.85, 0.3))):
        comp_widget.components[index].g_edit.setText(str(g))
        comp_widget.components[index].s_edit.setText(str(s_coord))
        comp_widget._on_component_coords_changed(index)
    comp_widget._run_analysis()

    names = _available_histogram_components(comp_widget)
    _check_histogram_components(comp_widget, names)

    table = parent.components_statistics_dock_widget.layer_stats_table

    def headers():
        return [
            table.horizontalHeaderItem(col).text()
            for col in range(table.columnCount())
        ]

    assert headers()[1] == "Component 1 Center of Mass"

    _rename_component(comp_widget, 0, "Free NADH")

    # The renamed component stays checked, and is offered only once: every
    # analysed image reports the new name, not just the primary one.
    assert _available_histogram_components(comp_widget) == [
        "Free NADH",
        "Component 2",
    ]
    assert comp_widget._selected_histogram_components() == [
        "Free NADH",
        "Component 2",
    ]
    assert comp_widget.histogram_widget._series_names() == [
        "Free NADH",
        "Component 2",
    ]
    assert headers()[1] == "Free NADH Center of Mass"

    # Absolute concentration: renaming the reference, the sample or a
    # component carries through.
    comp = comp_widget
    sample = viewer.layers["img0"]
    _concentration_ready(viewer, sample, parent, comp)
    comp._run_analysis()
    # An unsaved edit names the reference too.
    comp.reference_concentration_edit.setText("2")
    comp.reference_concentration_edit.editingFinished.emit()
    assert parent.settings_store.has_draft(sample, ['component_analysis'])
    viewer.layers['reference'].name = "NADH 1 mM"
    assert comp.reference_source_combo.currentText() == "NADH 1 mM"
    assert _stored_concentration(sample)['reference_layer'] == "NADH 1 mM"
    draft = parent.layer_settings(sample)['component_analysis'][
        'concentration'
    ]
    assert draft['reference_layer'] == "NADH 1 mM"
    assert draft['reference_concentration'] == 2.0
    sample.name = "cell"
    maps = _concentration_maps(viewer)
    assert list(maps) == [_cname(sample, "Component 1")]
    tag = maps[_cname(sample, "Component 1")].metadata[
        'phasor_component_fraction'
    ]
    assert tag['source_layer'] == "cell"
    _rename_component(comp, 0, "Free NADH")
    assert list(_concentration_maps(viewer)) == [_cname(sample, "Free NADH")]
    assert comp.calibrated_component_combo.itemText(0) == "Free NADH"
    assert "Free NADH" in comp._available_histogram_components


def test_components_layer_visibility_follows_checked_components(
    make_napari_viewer,
):
    """Only the fraction layers of the checked components stay visible."""
    viewer = make_napari_viewer()
    layer = create_image_layer_with_phasors()
    layer.name = "img0"
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    parent.image_layers_checkable_combobox.setCheckedItems(["img0"])
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)

    comp_widget.analysis_type_combo.setCurrentText("Component Fit")
    for idx, (g, s_coord) in enumerate(((0.2, 0.1), (0.8, 0.5))):
        comp_widget.components[idx].g_edit.setText(str(g))
        comp_widget.components[idx].s_edit.setText(str(s_coord))
        comp_widget._on_component_coords_changed(idx)
    comp_widget._run_analysis()

    names = _available_histogram_components(comp_widget)
    assert len(names) >= 2

    def visible(component_name):
        layers_map = comp_widget._get_fraction_layers_for_component(
            component_name
        )
        return [fl.visible for fl in layers_map.values()]

    _check_histogram_components(comp_widget, [names[0]])
    assert all(visible(names[0]))
    assert not any(visible(names[1]))

    _check_histogram_components(comp_widget, [names[1]])
    assert not any(visible(names[0]))
    assert all(visible(names[1]))

    # Checking both shows both.
    _check_histogram_components(comp_widget, names[:2])
    assert all(visible(names[0]))
    assert all(visible(names[1]))


def test_components_card_toggle_disabled_without_fraction_data(
    make_viewer_model,
    qtbot,
):
    """Toggles only become usable once a component has fraction data."""
    from napari_phasors.components_tab import (
        HISTOGRAM_TOGGLE_DISABLED_TOOLTIP,
        HISTOGRAM_TOGGLE_TOOLTIP,
    )

    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    # Before any analysis there is nothing to plot.
    for comp in comp_widget.components:
        assert not comp.histogram_checkbox.isEnabled()
        assert not comp.histogram_checkbox.isChecked()
        assert comp.histogram_checkbox.toolTip() == (
            HISTOGRAM_TOGGLE_DISABLED_TOOLTIP
        )

    # Clicking a disabled toggle cannot change the selection.
    comp_widget._on_component_histogram_toggled(0, True)
    assert comp_widget._selected_histogram_components() == []

    _setup_linear_projection(comp_widget)

    for comp in comp_widget.components:
        assert comp.histogram_checkbox.isEnabled()
        assert comp.histogram_checkbox.toolTip() == HISTOGRAM_TOGGLE_TOOLTIP

    # A newly added component has no fraction data of its own yet.
    comp_widget.analysis_type_combo.setCurrentText("Component Fit")
    comp_widget._add_component()
    assert not comp_widget.components[-1].histogram_checkbox.isEnabled()

    # Removing it renumbers the cards without disturbing the selection.
    selected = comp_widget._selected_histogram_components()
    comp_widget._remove_component()
    assert comp_widget._selected_histogram_components() == selected
    assert comp_widget.components[0].histogram_checkbox.isChecked()


def test_components_on_image_layer_changed_runs_teardown_and_restore(
    make_viewer_model,
    qtbot,
):
    """test that _on_image_layer_changed calls both teardown and
    restore methods to properly handle layer changes"""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    comp_widget = parent.components_tab

    from unittest.mock import patch as _patch

    with (
        _patch.object(
            comp_widget, '_teardown_on_layer_change'
        ) as mock_teardown,
        _patch.object(comp_widget, '_restore_on_layer_change') as mock_restore,
    ):
        comp_widget._on_image_layer_changed()
        mock_teardown.assert_called_once()
        mock_restore.assert_called_once()


def test_components_selecting_on_the_plot(make_viewer_model, qtbot):
    """Picking a component on the plot: Escape cancels, a click or a drag
    places it and fills its lifetime, and the lifetime box only reacts to
    real typing."""
    from phasorpy.lifetime import phasor_to_normal_lifetime

    viewer, layer, parent, comp_widget = _setup_components(make_viewer_model)
    comp = comp_widget.components[0]

    # Escape through the Matplotlib key press event cancels the selection.
    comp_widget._select_component(0)
    assert comp.select_button.text() == "Click on plot..."
    assert not comp.select_button.isEnabled()
    assert comp_widget._active_select_cid is not None
    assert comp_widget._active_select_key_cid is not None
    assert comp_widget._active_select_shortcut is not None
    assert comp_widget._active_select_idx == 0

    class DummyKeyEvent:
        def __init__(self, key):
            self.key = key

    comp_widget._handle_select_key_press_event(DummyKeyEvent('escape'))
    assert comp.select_button.text() == "Select"
    assert comp.select_button.isEnabled()
    assert comp_widget._active_select_cid is None
    assert comp_widget._active_select_key_cid is None
    assert comp_widget._active_select_shortcut is None
    assert comp_widget._active_select_idx is None

    # So does the Qt QShortcut.
    comp_widget._select_component(0)
    assert comp_widget._active_select_shortcut is not None
    comp_widget._active_select_shortcut.activated.emit()
    assert comp.select_button.text() == "Select"
    assert comp.select_button.isEnabled()
    assert comp_widget._active_select_cid is None
    assert comp_widget._active_select_key_cid is None
    assert comp_widget._active_select_shortcut is None

    # Clicking on the plot creates the component and calculates its lifetime.
    class Event:
        inaxes = True

    event = Event()
    event.xdata, event.ydata = 0.5, 0.5
    comp.name_edit.setText("First")
    comp_widget._select_component(0)
    comp_widget._handle_component_selection_event(event)
    assert comp.dot is not None
    expected_lifetime = phasor_to_normal_lifetime(0.5, 0.5, frequency=80.0)
    assert comp.lifetime_edit.text() != ""
    assert abs(float(comp.lifetime_edit.text()) - expected_lifetime) < 1e-3

    # Dragging it updates the lifetime too.
    comp_widget.dragging_component_idx = 0

    class DragEvent:
        inaxes = True
        xdata = 0.2
        ydata = 0.1

    comp_widget._on_motion(DragEvent())
    expected_drag_lifetime = phasor_to_normal_lifetime(
        0.2, 0.1, frequency=80.0
    )
    assert comp.lifetime_edit.text() != ""
    assert (
        abs(float(comp.lifetime_edit.text()) - expected_drag_lifetime) < 1e-3
    )
    comp_widget.dragging_component_idx = None

    # A second selection event updates the existing dot instead of creating
    # a new one.
    comp_widget._select_component(0)
    event.xdata, event.ydata = 0.3, 0.2
    comp_widget._handle_component_selection_event(event)
    assert comp.dot is not None
    assert comp.lifetime_edit.text() != ""

    # Leaving the lifetime box unedited keeps the placed component in place:
    # Qt reports leaving the box as a finished edit, which used to place the
    # component from the lifetime shown for it, onto the semicircle.
    comp.lifetime_edit.editingFinished.emit()
    x, y = comp.dot.get_data()
    assert abs(x[0] - 0.3) < 1e-9
    assert abs(y[0] - 0.2) < 1e-9

    comp.lifetime_edit.clear()
    qtbot.keyClicks(comp.lifetime_edit, "3.0")
    qtbot.keyClick(comp.lifetime_edit, Qt.Key_Return)
    expected_g, expected_s = phasor_from_lifetime(80.0, 3.0)
    x, y = comp.dot.get_data()
    assert abs(x[0] - expected_g) < 1e-12
    assert abs(y[0] - expected_s) < 1e-12

    # Committing the same text again is not a new edit.
    comp_widget._apply_component_coords(0, 0.3, 0.2)
    comp.lifetime_edit.setText("3.0")
    comp.lifetime_edit.editingFinished.emit()
    x, y = comp.dot.get_data()
    assert abs(x[0] - 0.3) < 1e-9


@pytest.mark.parametrize(
    "analysis_type", ["Linear Projection", "Component Fit"]
)
def test_components_inside_semicircle_stay_after_run(
    make_viewer_model, qtbot, analysis_type
):
    """Running the analysis leaves components placed inside the semicircle.

    Placing a component fills its lifetime box with the projection onto the
    semicircle; the refresh after the run used to re-place every component
    from that lifetime, pulling them all onto the semicircle.
    """
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)
    comp_widget.analysis_type_combo.setCurrentText(analysis_type)

    positions = [(0.3, 0.2), (0.7, 0.25)]

    class Event:
        inaxes = True

    for idx, (g, s) in enumerate(positions):
        event = Event()
        event.xdata, event.ydata = g, s
        comp_widget._select_component(idx)
        comp_widget._handle_component_selection_event(event)
        assert comp_widget.components[idx].lifetime_edit.text() != ""

    comp_widget._run_analysis()
    # Switching layers restores the tab through the same refresh.
    comp_widget._restore_on_layer_change()

    for idx, (g, s) in enumerate(positions):
        x, y = comp_widget.components[idx].dot.get_data()
        assert abs(x[0] - g) < 1e-9
        assert abs(y[0] - s) < 1e-9
        stored = parent.layer_settings(layer)["component_analysis"][
            "components"
        ][str(idx)]["gs_harmonics"]["1"]
        assert abs(stored["g"] - g) < 1e-9
        assert abs(stored["s"] - s) < 1e-9

    # Nor does changing the frequency move a pinned component.
    parent._broadcast_frequency_value_across_tabs("40")
    for idx, (g, s) in enumerate(positions):
        x, y = comp_widget.components[idx].dot.get_data()
        assert abs(x[0] - g) < 1e-9
        assert abs(y[0] - s) < 1e-9


def test_components_auto_placement_and_select_menu(make_viewer_model, qtbot):
    """Auto placement on the semicircle, and every entry of the select menu."""
    from qtpy.QtWidgets import QDialog, QMenu

    viewer, layer, parent, comp_widget = _setup_components(make_viewer_model)

    # Out-of-range indices return early.
    comp_widget._auto_place_component_by_index(0)
    comp_widget._auto_place_component_by_index(99)
    # Previous component not set -> warning + early return.
    comp_widget._auto_place_component_by_index(1)

    # Set component 0, then auto-place component 1 on the universal circle,
    # which also calculates its lifetime.
    comp_widget.components[0].g_edit.setText("0.3")
    comp_widget.components[0].s_edit.setText("0.2")
    comp_widget._on_component_coords_changed(0)
    comp_widget._auto_place_component_by_index(1)
    comp2 = comp_widget.components[1]
    assert comp2.g_edit.text() != ""
    assert comp2.lifetime_edit.text() != ""
    assert float(comp2.lifetime_edit.text()) > 0

    # Without a frequency the method warns and returns.
    layer.metadata["settings"].pop("frequency", None)
    comp_widget.components[0].g_edit.setText("0.4")
    comp_widget.components[0].s_edit.setText("0.2")
    comp_widget._auto_place_component_by_index(1)
    layer.metadata["settings"]["frequency"] = 80.0

    # With no cursors the submenu says so, and only later components offer
    # to auto intersect the semicircle.
    menu0 = QMenu()
    comp_widget._populate_select_menu(0, menu0)
    texts0 = [a.text() for a in menu0.actions()]
    assert "Select from layer(s) phasor center" in texts0
    assert "Auto intersect semicircle" not in texts0
    menu1 = QMenu()
    comp_widget._populate_select_menu(1, menu1)
    texts1 = [a.text() for a in menu1.actions()]
    assert "Select from layer(s) phasor center" in texts1
    assert "Auto intersect semicircle" in texts1

    # The select menu lists every cursor and cluster of the Selection tab.
    selection_tab = parent.selection_tab
    cursor_widget = selection_tab.cursor_selection_widget
    cursor_widget._add_cursor(
        cursor_type="circular", g=0.4, s=0.3, radius=0.05
    )
    cursor_widget._add_cursor(
        cursor_type="polar",
        phase_min=10.0,
        phase_max=30.0,
        modulation_min=0.4,
        modulation_max=0.6,
    )
    cursor_widget._add_cursor(cursor_type="elliptic", g=0.35, s=0.25)
    cluster_widget = selection_tab.automatic_clustering_widget
    cluster_widget._clusters.append(
        {
            'g': 0.45,
            's': 0.35,
            'harmonic': 1,
            'color': 'magenta',
            'patch': None,
        }
    )

    menu = QMenu()
    comp_widget._populate_select_menu(0, menu)
    actions = menu.actions()
    assert len(actions) > 0
    assert actions[0].text() == "Select on plot"
    # The third action is the "Select from cursor center" submenu
    cursor_submenu = actions[2].menu()
    assert cursor_submenu is not None
    cursor_actions = cursor_submenu.actions()
    assert len(cursor_actions) == 4
    assert "Circular 1" in cursor_actions[0].text()
    assert "Polar 2" in cursor_actions[1].text()
    assert "Elliptical 3" in cursor_actions[2].text()
    assert "Cluster 1" in cursor_actions[3].text()

    # Selecting from a cursor center sets coordinates and lifetime.
    comp_widget._set_component_coords_from_menu(0, 0.4, 0.3)
    comp1 = comp_widget.components[0]
    assert comp1.g_edit.text() == "0.400"
    assert comp1.s_edit.text() == "0.300"
    assert comp1.lifetime_edit.text() != ""
    assert float(comp1.lifetime_edit.text()) > 0
    comp_widget._set_component_coords_from_menu(0, 0.5, 0.4)
    assert comp1.g_edit.text() == "0.500"
    assert comp1.s_edit.text() == "0.400"

    # Selecting from the layer's phasor center fills the coordinates.
    comp_widget._clear_components()
    with patch(
        "napari_phasors.components_tab.PhasorCenterSelectionDialog"
    ) as MockDialog:
        mock_dialog_instance = MockDialog.return_value
        mock_dialog_instance.exec.return_value = QDialog.Accepted
        mock_dialog_instance.get_selected_layers.return_value = [layer.name]
        comp_widget._select_from_phasor_center(0)
    comp0 = comp_widget.components[0]
    assert comp0.g_edit.text() != ""
    assert comp0.s_edit.text() != ""
    assert float(comp0.g_edit.text()) > 0
    assert float(comp0.s_edit.text()) > 0

    # Auto placement generalizes to later components: component 3
    # intersects from component 2.
    comp_widget._add_component()
    assert len(comp_widget.components) == 3
    comp_widget.components[1].g_edit.setText("0.6")
    comp_widget.components[1].s_edit.setText("0.4")
    comp_widget._on_component_coords_changed(1)
    comp_widget._auto_place_component_by_index(2)
    comp3 = comp_widget.components[2]
    assert comp3.g_edit.text() != ""
    assert comp3.s_edit.text() != ""
    assert float(comp3.g_edit.text()) > 0


def test_components_phasor_center_remembers_selection_and_rename(
    make_viewer_model,
    qtbot,
):
    """Selecting phasor-center layers is remembered per component and survives renames."""
    from unittest.mock import patch

    from qtpy.QtWidgets import QDialog

    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)

    comp0 = comp_widget.components[0]
    # No selection remembered initially.
    assert comp0.phasor_center_layers == []

    # Select the layer for component 1's phasor center.
    with patch(
        "napari_phasors.components_tab.PhasorCenterSelectionDialog"
    ) as MockDialog:
        mock_dialog_instance = MockDialog.return_value
        mock_dialog_instance.exec.return_value = QDialog.Accepted
        mock_dialog_instance.get_selected_layers.return_value = [layer.name]

        comp_widget._select_from_phasor_center(0)

    # The selection is remembered on the component and in the layer's
    # (unsaved) settings.
    assert comp0.phasor_center_layers == [layer.name]
    settings = parent.layer_settings(layer)["component_analysis"]
    assert settings["components"]["0"]["phasor_center_layers"] == [layer.name]

    # Reopening the dialog pre-selects the previously chosen layers.
    with patch(
        "napari_phasors.components_tab.PhasorCenterSelectionDialog"
    ) as MockDialog:
        mock_dialog_instance = MockDialog.return_value
        mock_dialog_instance.exec.return_value = QDialog.Rejected
        comp_widget._select_from_phasor_center(0)
        _, kwargs = MockDialog.call_args
        assert kwargs.get("preselected") == [layer.name]

    # Renaming the layer updates the remembered selection instead of dropping it.
    new_name = "renamed_layer"
    comp_widget.rename_layer(layer.name, new_name)
    assert comp0.phasor_center_layers == [new_name]
    assert settings["components"]["0"]["phasor_center_layers"] == [new_name]


def test_color_action_widget_and_dialog(make_viewer_model, qtbot):
    """Test ColorActionWidget color conversions, mouse events, and PhasorCenterSelectionDialog."""
    from qtpy.QtCore import QPointF, Qt
    from qtpy.QtGui import QColor, QMouseEvent
    from qtpy.QtWidgets import QMenu, QWidgetAction

    from napari_phasors.components_tab import (
        ColorActionWidget,
        PhasorCenterSelectionDialog,
    )

    # 1. Test ColorActionWidget color representations
    action = QWidgetAction(None)

    # QColor
    w1 = ColorActionWidget("Text", QColor(255, 0, 0), action)
    assert "color: #ff0000" in w1.styleSheet().lower()

    # FakeColor with getRgb but no name
    class FakeColor:
        def getRgb(self):
            return (0, 255, 0, 255)

    w2 = ColorActionWidget("Text", FakeColor(), action)
    assert "color: rgba(0, 255, 0, 1.0)" in w2.styleSheet().lower()

    # Float tuple
    w3 = ColorActionWidget("Text", (0.0, 0.0, 1.0, 0.5), action)
    assert "color: rgba(0, 0, 255, 0.5)" in w3.styleSheet().lower()

    # Int tuple
    w4 = ColorActionWidget("Text", (128, 128, 128, 255), action)
    assert "color: rgba(128, 128, 128, 1.0)" in w4.styleSheet().lower()

    # String color name
    w5 = ColorActionWidget("Text", "yellow", action)
    assert "color: yellow" in w5.styleSheet().lower()

    # 2. Test ColorActionWidget mouse release event
    from qtpy.QtWidgets import QWidget

    parent_menu = QMenu()
    action = QWidgetAction(parent_menu)
    action_widget = ColorActionWidget("Text", "blue", action, parent_menu)
    action.setDefaultWidget(action_widget)
    # Nest action_widget inside a container widget under parent_menu to cover parent hierarchy traversal
    container = QWidget(parent_menu)
    action_widget.setParent(container)

    # Track action trigger
    triggered = False

    def on_triggered():
        nonlocal triggered
        triggered = True

    action.triggered.connect(on_triggered)

    # Simulate left click (pass globalPos for the non-deprecated overload)
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(5, 5),
        QPointF(5, 5),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    action_widget.mouseReleaseEvent(event)
    assert triggered is True

    # 3. Test PhasorCenterSelectionDialog
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    dialog = PhasorCenterSelectionDialog([layer.name], parent=comp_widget)
    assert dialog.windowTitle() == "Select Phasor Center Layers"
    # By default nothing is checked on init.
    assert dialog.get_selected_layers() == []

    # When preselected layers are provided, they are checked on init.
    dialog_pre = PhasorCenterSelectionDialog(
        [layer.name], parent=comp_widget, preselected=[layer.name]
    )
    assert layer.name in dialog_pre.get_selected_layers()


# ---------------------------------------------------------------------------
# Metadata restore, plot-settings reset, artists, lifetime, colormap helpers
# ---------------------------------------------------------------------------


def _setup_components(make_viewer_model, freq=80.0):
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": freq}
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    comp = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp)
    return viewer, layer, parent, comp


def test_linear_projection_layer_names(make_viewer_model, qtbot):
    """Output layers are "<image> [(Linear Projection) <name>]" and follow
    component and source renames; labels layers are "<image> [...]"."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    assert (
        comp._label_layer_name(layer.name, None)
        == "FLIM data Intensity [Dominant component]"
    )
    assert (
        comp._label_layer_name(layer.name, 0)
        == "FLIM data Intensity [Component 1 filtered]"
    )

    assert layer.name == "FLIM data Intensity [Phasor]"
    _setup_linear_projection(comp)
    name1, _ = comp._linear_projection_component_names()
    expected = f"FLIM data Intensity [(Linear Projection) {name1}]"
    assert comp.comp1_fractions_layer.name == expected
    assert expected in viewer.layers
    assert f"{name1} fractions: {layer.name}" not in viewer.layers

    # Renaming component 1 renames the layer inside the brackets.
    comp.components[0].name_edit.setText("Donor")
    comp._on_component_name_changed(0)
    assert (
        comp.comp1_fractions_layer.name
        == "FLIM data Intensity [(Linear Projection) Donor]"
    )

    # Renaming the source keeps the bracketed analysis of its outputs.
    name1, _ = comp._linear_projection_component_names()
    comp.rename_layer(layer.name, "renamed Intensity [Phasor]")
    assert (
        comp.comp1_fractions_layer.name
        == f"renamed Intensity [(Linear Projection) {name1}]"
    )

    # Absolute concentration: a run adds one tagged map per quantity and
    # stores its calibration.
    reference = _concentration_ready(viewer, layer, parent, comp)
    _enable_second_component(comp)
    comp._run_analysis()
    maps = _concentration_maps(viewer)
    assert list(maps) == [
        _cname(layer, "Component 1"),
        _cname(layer, "Component 2"),
        _cname(layer, "Total"),
    ]
    first, second, total = _expected_concentrations(
        layer, reference, ratio=2.0
    )
    for name, want in zip(maps, (first, second, total), strict=True):
        np.testing.assert_array_equal(maps[name].data, want)
        np.testing.assert_array_equal(
            maps[name].metadata['fraction_data_original'], want
        )
    total_map = maps[_cname(layer, "Total")]
    tag = total_map.metadata['phasor_component_fraction']
    assert tag['total'] is True
    assert tag['component_index'] is None
    assert tag['units'] == "mM"
    assert tag['harmonic'] == 1
    assert tag['source_layer'] == layer.name
    assert total_map.colormap.name == "viridis"
    block = layer.metadata['settings']['component_analysis']
    assert block['analysis_type'] == ABSOLUTE_CONCENTRATION
    stored = block['concentration']
    measured = phasor_reference_from_layer(reference, 1)
    assert stored['reference_layer'] == "reference"
    assert stored['reference_mean'] == measured[0]
    assert stored['reference_gs_harmonics'] == {
        '1': {'g': measured[1], 's': measured[2]}
    }
    assert stored['calibrated_component'] == 0
    assert stored['reference_concentration'] == 1.0
    assert stored['units'] == "mM"
    assert stored['second_component'] is True
    assert stored['brightness_ratio'] == 2.0
    # How each map is shown is stored too: the components' with their
    # coordinates, the total's on its own.
    display = block['components']['0']['gs_harmonics']['1']
    assert display['analysis_type'] == ABSOLUTE_CONCENTRATION
    assert display['colormap_name'] == comp.component_colormap_names[0]
    assert stored['total_display']['colormap_name'] == "viridis"
    assert comp.fraction_layers == [
        maps[_cname(layer, "Component 1")],
        maps[_cname(layer, "Component 2")],
    ]
    # The histogram offers both components and the total.
    assert comp._available_histogram_components == [
        "Component 1",
        "Component 2",
        TOTAL_CONCENTRATION,
    ]
    assert comp.histogram_widget.xlabel == "Concentration (mM)"
    assert comp.total_histogram_checkbox.isEnabled()
    comp.total_histogram_checkbox.setChecked(True)
    assert comp._histogram_components == ["Component 1", TOTAL_CONCENTRATION]
    comp.total_histogram_checkbox.setChecked(False)
    assert comp._histogram_components == ["Component 1"]

    # A rerun updates the maps in place, keeping their identity, name and
    # display until the concentration scale changes.
    first_map = viewer.layers[_cname(layer, "Component 1")]
    first_map.contrast_limits = (0.1, 0.2)
    comp._run_analysis()
    assert viewer.layers[_cname(layer, "Component 1")] is first_map
    assert tuple(first_map.contrast_limits) == pytest.approx((0.1, 0.2))
    old = np.array(first_map.data)
    comp.reference_concentration_edit.setText("1000")
    comp.reference_concentration_edit.editingFinished.emit()
    comp.concentration_units_combo.setCurrentText("µM")
    comp.concentration_units_combo.lineEdit().editingFinished.emit()
    comp._run_analysis()
    np.testing.assert_allclose(first_map.data, old * 1000, rtol=1e-12)
    assert first_map.contrast_limits[1] == pytest.approx(
        np.nanmax(first_map.data)
    )
    assert first_map.metadata['phasor_component_fraction']['units'] == "µM"
    assert comp.histogram_widget.xlabel == "Concentration (µM)"
    # A map still named after a component is named after it again, and one
    # the user renamed keeps its name.
    first_map.name = _cname(layer, "Other")
    comp._run_analysis()
    assert first_map.name == _cname(layer, "Component 1")
    first_map.name = "my free map"
    comp._run_analysis()
    assert viewer.layers["my free map"] is first_map
    # Without the second component, its maps and the total go.
    comp.second_component_checkbox.setChecked(False)
    comp._run_analysis()
    assert list(_concentration_maps(viewer)) == ["my free map"]
    assert comp.fraction_layers == []

    # Calibrating on the second component puts it first in the model.
    _concentration_ready(viewer, layer, parent, comp)
    comp.calibrated_component_combo.setCurrentIndex(1)
    comp._run_analysis()
    maps = _concentration_maps(viewer)
    assert list(maps) == [_cname(layer, "Component 2")]
    want, _second, _total = _expected_concentrations(
        layer, reference, order=(1, 0)
    )
    only = maps[_cname(layer, "Component 2")]
    np.testing.assert_array_equal(only.data, want)
    assert only.metadata['phasor_component_fraction']['component_index'] == 1
    assert comp._find_component_index_for_layer(only) == 1
    assert _stored_concentration(layer)['calibrated_component'] == 1

    # The histogram range slider stays within its integer range at any
    # magnitude.
    _concentration_ready(viewer, layer, parent, comp)
    comp.reference_concentration_edit.setText("1000000")
    comp.reference_concentration_edit.editingFinished.emit()
    comp.concentration_units_combo.setCurrentText("nM")
    comp.concentration_units_combo.lineEdit().editingFinished.emit()
    comp._run_analysis()
    histogram = comp.histogram_widget
    assert histogram.xlabel == "Concentration (nM)"
    assert histogram.range_label.text() == "Concentration range (nM):"
    assert histogram.range_slider.maximum() <= 10_000_000
    assert histogram.range_factor < 1000
    assert comp._histogram_range_factor(0.0, 1.47) == 1e5
    assert comp._histogram_range_factor(0.0, 0.0) == 1000
    assert comp._histogram_range_factor(0.0, np.nan) == 1000
    comp.analysis_type_combo.setCurrentText("Component Fit")
    assert comp._histogram_range_factor(0.0, 1e6) == 1000


def test_component_fit_layer_names_do_not_collide_with_projection(
    make_viewer_model, qtbot
):
    """Every fit layer is "<image> [(Component Fit) <name>]", and both
    methods can keep their layers for the same image."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_linear_projection(comp)
    _setup_component_fit(comp)

    fit_names = sorted(
        lyr.name
        for lyr in viewer.layers
        if is_component_fit_label(split_analysis_layer_name(lyr.name)[1])
    )
    assert fit_names == [
        "FLIM data Intensity [(Component Fit) Component 1]",
        "FLIM data Intensity [(Component Fit) Component 2]",
    ]

    lp_name = _lp_name("Component 1", layer.name)
    fit_name = _fit_name("Component 1", layer.name)
    assert lp_name != fit_name
    assert lp_name in viewer.layers
    assert fit_name in viewer.layers


def _linear_projection_settings():
    return {
        "analysis_type": "Linear Projection",
        "last_analysis_harmonic": 1,
        "components": {
            "0": {
                "name": "Comp A",
                "gs_harmonics": {"1": {"g": 0.6, "s": 0.3, "lifetime": 1.5}},
            },
            "1": {
                "name": "Comp B",
                "gs_harmonics": {"1": {"g": 0.3, "s": 0.2, "lifetime": 4.0}},
            },
        },
        "line_settings": {
            "show_colormap_line": False,
            "show_component_dots": True,
            "line_offset": 0.1,
            "line_width": 2.0,
            "line_alpha": 0.5,
        },
        "label_settings": {
            "fontsize": 12,
            "bold": True,
            "italic": True,
            "color": "red",
        },
    }


def test_components_restore_from_metadata(make_viewer_model, qtbot):
    """Stored component settings rebuild the cards, line and labels."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    # No 'component_analysis' key -> the restore returns early.
    comp._restore_components_ui_only_from_metadata()
    assert len(comp.components) == 2

    # Restoring fewer components than shown trims the extra cards.
    comp._add_component()
    comp._add_component()
    assert len(comp.components) == 4
    layer.metadata["settings"]["component_analysis"] = {
        "analysis_type": "Linear Projection",
        "components": {
            "0": {"name": "A", "gs_harmonics": {"1": {"g": 0.2, "s": 0.1}}},
            "1": {"name": "B", "gs_harmonics": {"1": {"g": 0.8, "s": 0.5}}},
        },
    }
    comp._restore_components_ui_only_from_metadata()
    assert len(comp.components) == 2
    assert comp.components[0].name_edit.text() == "A"
    assert comp.components[1].name_edit.text() == "B"
    assert comp._selected_component is comp.components[0]

    layer.metadata["settings"][
        "component_analysis"
    ] = _linear_projection_settings()
    comp._restore_components_ui_only_from_metadata()
    assert comp.components[0].name_edit.text() == "Comp A"
    assert comp.components[1].name_edit.text() == "Comp B"
    # Both components are drawn on the plot.
    created = [
        c for c in comp.components if c is not None and c.dot is not None
    ]
    assert len(created) == 2
    assert comp.label_color == "red"
    assert comp.label_bold is True
    assert comp.line_width == 2.0
    assert comp.show_colormap_line is False

    # Line + fraction-histogram overlay settings round-trip via metadata.
    # User edits are persisted under ``two_component_line_settings``; the
    # restore path must read that key (regression: it previously only read
    # the unused ``line_settings`` key, so overlay settings were never
    # re-applied).
    layer.metadata["settings"]["component_analysis"] = {
        "analysis_type": "Linear Projection",
        "last_analysis_harmonic": 1,
        "components": {
            "0": {
                "name": "Comp A",
                "gs_harmonics": {"1": {"g": 0.6, "s": 0.3}},
            },
            "1": {
                "name": "Comp B",
                "gs_harmonics": {"1": {"g": 0.3, "s": 0.2}},
            },
        },
        "two_component_line_settings": {
            "show_colormap_line": True,
            "show_component_dots": False,
            "line_offset": 0.07,
            "line_width": 4.5,
            "line_alpha": 0.6,
            "default_component_color": "#abcdef",
            "show_fraction_histogram": True,
            "histogram_overlay_height": 0.42,
            "histogram_offset": -0.15,
            "histogram_alpha": 0.55,
        },
        "two_components_label_settings": {
            "fontsize": 16,
            "bold": True,
            "italic": True,
            "color": "red",
        },
    }
    comp._restore_components_ui_only_from_metadata()
    assert comp.show_component_dots is False
    assert comp.line_offset == 0.07
    assert comp.line_width == 4.5
    assert comp.line_alpha == 0.6
    assert comp.default_component_color == "#abcdef"
    assert comp.show_fraction_histogram is True
    assert comp.histogram_overlay_height == 0.42
    assert comp.histogram_offset == -0.15
    assert comp.histogram_alpha == 0.55
    assert comp.label_fontsize == 16
    assert comp.label_bold is True
    assert comp.label_italic is True
    assert comp.label_color == "red"

    # Removing a component from the settings re-indexes the rest.
    comp._add_component()
    layer.metadata["settings"]["component_analysis"] = {
        "components": {
            "0": {"name": "Comp 0", "gs_harmonics": {}},
            "1": {"name": "Comp 1", "gs_harmonics": {}},
            "2": {"name": "Comp 2", "gs_harmonics": {}},
        }
    }
    # Remove index 1 from settings: old '2' should become new '1'. The
    # removal is an unsaved edit until the analysis runs.
    comp._remove_component_from_settings(1)
    settings = parent.layer_settings(layer)["component_analysis"]["components"]
    assert "0" in settings and settings["0"]["name"] == "Comp 0"
    assert "1" in settings and settings["1"]["name"] == "Comp 2"
    assert "2" not in settings
    stored = layer.metadata["settings"]["component_analysis"]["components"]
    assert len(stored) == 3

    # Remove with idx=None removes the last component
    comp._remove_component_from_settings(None)
    settings = parent.layer_settings(layer)["component_analysis"]["components"]
    assert len(settings) == 1
    assert "0" in settings

    # Calling with empty or missing components setting does nothing
    layer.metadata["settings"]["component_analysis"]["components"] = {}
    comp._remove_component_from_settings(0)

    # Absolute concentration: typed reference phasors are drafts kept per
    # harmonic, and only a run stores them.
    _concentration_ready(viewer, layer, parent, comp, reference=False)
    _type_reference(comp, 2.5, 0.8, 0.3)
    draft = parent.layer_settings(layer)['component_analysis']
    assert draft['concentration']['reference_gs_harmonics'] == {
        '1': {'g': 0.8, 's': 0.3}
    }
    assert draft['concentration']['reference_mean'] == 2.5
    assert draft['concentration']['reference_layer'] is None
    assert 'component_analysis' not in (layer.metadata.get('settings') or {})
    parent.harmonic_spinbox.setValue(2)
    assert comp.reference_g_edit.text() == ""
    assert comp.reference_mean_edit.text() == "2.5"
    _type_reference(comp, 2.5, 0.6, 0.35)
    parent.harmonic_spinbox.setValue(1)
    assert comp._reference_values() == (2.5, 0.8, 0.3)
    comp._run_analysis()
    assert _stored_concentration(layer)['reference_gs_harmonics'] == {
        '1': {'g': 0.8, 's': 0.3},
        '2': {'g': 0.6, 's': 0.35},
    }

    # A reference layer is measured at whichever harmonic is shown.
    reference = _concentration_ready(viewer, layer, parent, comp)
    assert comp._reference_values() == phasor_reference_from_layer(
        reference, 1
    )
    parent.harmonic_spinbox.setValue(2)
    assert comp._reference_values() == phasor_reference_from_layer(
        reference, 2
    )
    comp.analysis_type_combo.setCurrentText("Linear Projection")
    parent.harmonic_spinbox.setValue(1)
    comp.analysis_type_combo.setCurrentText(ABSOLUTE_CONCENTRATION)
    assert comp._reference_values() == phasor_reference_from_layer(
        reference, 1
    )

    # A reopened concentration shows, and reproduces, what was run.
    _enable_second_component(comp, "3")
    comp.concentration_units_combo.setCurrentText("µM")
    comp.concentration_units_combo.lineEdit().editingFinished.emit()
    comp.calibrated_component_combo.setCurrentIndex(1)
    comp._run_analysis()
    stored = copy.deepcopy(layer.metadata['settings']['component_analysis'])
    measured = phasor_reference_from_layer(reference, 1)
    other = PlotterWidget(viewer)
    _select_layers(other, [layer.name])
    restored = other.components_tab
    other.tab_widget.setCurrentWidget(restored)
    restored._on_image_layer_changed()
    assert restored.analysis_type == ABSOLUTE_CONCENTRATION
    assert not restored.concentration_box.isHidden()
    assert restored.calibrated_component_combo.currentIndex() == 1
    assert restored.reference_source_combo.currentText() == "reference"
    assert restored._reference_values() == measured
    assert restored.second_component_checkbox.isChecked()
    assert restored.brightness_ratio_edit.isEnabled()
    assert restored.brightness_ratio_edit.text() == "3"
    assert restored._concentration_units() == "µM"
    # Showing an analysis writes nothing.
    assert layer.metadata['settings']['component_analysis'] == stored
    # Without its reference layer, the stored measurement stands in.
    viewer.layers.remove(reference)
    assert restored.reference_source_combo.currentText() == MANUAL_REFERENCE
    assert restored._reference_values() == measured
    assert "was removed" in restored.reference_note.text()
    # Reopened without the edits made since, the stored settings still name
    # the missing layer.
    other.settings_store.discard_drafts([layer])
    restored._on_image_layer_changed()
    assert restored.reference_source_combo.currentText() == MANUAL_REFERENCE
    assert restored._reference_values() == measured
    assert "is not open" in restored.reference_note.text()
    # ...and reproduces the maps exactly.
    name = _cname(layer, "Component 2")
    before = np.array(viewer.layers[name].data)
    restored._run_analysis()
    np.testing.assert_array_equal(viewer.layers[name].data, before)
    # Recreating from the metadata runs the analysis again.
    for old in list(_concentration_maps(viewer).values()):
        viewer.layers.remove(old)
    restored._restore_and_recreate_components_from_metadata()
    assert set(_concentration_maps(viewer)) == {
        _cname(layer, "Component 1"),
        _cname(layer, "Component 2"),
        _cname(layer, "Total"),
    }


def test_components_artists_and_plot_settings_dialog(make_viewer_model, qtbot):
    """Artists, the plot settings dialog and its reset, and the per-harmonic
    coordinate helpers, on one widget."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    # Per-harmonic coordinate and harmonic-listing helpers.
    comp._create_component_at_coordinates(0, 0.5, 0.3)
    comp._create_component_at_coordinates(1, 0.3, 0.2)
    comp.components[0].name_edit.setText("A")
    current = parent.harmonic
    g, s, names = comp._get_component_coords_for_harmonic(current)
    assert len(g) == 2 and len(s) == 2 and len(names) == 2
    layer.metadata["settings"]["component_analysis"] = {
        "components": {
            "0": {"name": "A", "gs_harmonics": {"2": {"g": 0.4, "s": 0.25}}},
            "1": {"name": "B", "gs_harmonics": {"2": {"g": 0.2, "s": 0.15}}},
        }
    }
    comp.current_image_layer_name = layer.name
    g2, _, _ = comp._get_component_coords_for_harmonic(2)
    assert g2 == [0.4, 0.2]
    assert current in comp._get_harmonics_with_components()

    # Artists: listed, hidden and shown.
    comp._create_component_at_coordinates(0, 0.6, 0.3)
    comp._create_component_at_coordinates(1, 0.3, 0.2)
    comp.draw_line_between_components()
    artists = comp.get_all_artists()
    assert len(artists) >= 2
    comp.set_artists_visible(False)
    comp.set_artists_visible(True)

    # The reset routine manipulates widgets created by the settings dialog.
    comp._open_plot_settings_dialog()
    comp.line_width = 5.0
    comp.line_alpha = 0.2
    comp._reset_plot_settings()
    assert comp.line_width == 3.0
    assert comp.line_alpha == 1
    assert comp.show_colormap_line is True

    # Picking a valid color from the dialog updates the button, the stored
    # default color, and the persisted metadata setting.
    with patch.object(
        QColorDialog, "getColor", return_value=QColor("#ff0000")
    ):
        comp._on_color_button_clicked()
    assert comp.default_component_color == "#ff0000"
    settings = parent.layer_settings(layer)["component_analysis"]
    assert (
        settings["two_component_line_settings"]["default_component_color"]
        == "#ff0000"
    )
    comp.plot_dialog.close()
    comp.plot_dialog = None

    comp.clear_artists()

    # The fraction-histogram overlay group is only enabled for a
    # two-component Linear Projection.
    comp.analysis_type_combo.setCurrentText("Linear Projection")
    comp._open_plot_settings_dialog()
    assert comp.fraction_histogram_checkbox.isEnabled()
    comp.plot_dialog.close()
    comp.plot_dialog = None
    comp.analysis_type_combo.setCurrentText("Component Fit")
    comp._open_plot_settings_dialog()
    assert not comp.fraction_histogram_checkbox.isEnabled()
    comp.plot_dialog.close()

    # ``get_all_artists``/``set_artists_visible`` wrap a non-list/tuple
    # ``component_histogram`` into a single-element list before use.
    fake_artist = MagicMock()
    comp.component_histogram = fake_artist
    assert fake_artist in comp.get_all_artists()
    comp.set_artists_visible(True)
    fake_artist.set_visible.assert_called_once_with(True)

    # A single artist whose ``remove`` raises is still dropped.
    comp.component_histogram = _RemoveRaisesArtist()
    comp._remove_histogram_overlay()
    assert comp.component_histogram is None

    # ``_draw_colormap_line`` falls back to ``plt.cm.jet`` when
    # ``fractions_colormap`` is unset.
    comp.fractions_colormap = None
    fig, ax = plt.subplots()
    try:
        comp._draw_colormap_line(ax, 0.0, 0.0, 1.0, 1.0)
    finally:
        plt.close(fig)


def test_components_component_fit_prompts_for_more_harmonics(
    make_viewer_model, qtbot
):
    """A 4-component fit needs 2 harmonics; with locations only in harmonic 1
    the analysis prompts the user to also place them in the next harmonic."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    comp._add_component()
    comp._add_component()
    assert len(comp.components) == 4
    comp.analysis_type_combo.setCurrentText("Component Fit")

    coords = [
        ("0.2", "0.1"),
        ("0.4", "0.25"),
        ("0.6", "0.35"),
        ("0.8", "0.45"),
    ]
    for i, (g, s) in enumerate(coords):
        comp.components[i].g_edit.setText(g)
        comp.components[i].s_edit.setText(s)
        comp._on_component_coords_changed(i)

    start_harmonic = parent.harmonic
    comp._run_analysis()
    # The widget advanced to the next harmonic to collect more locations.
    assert parent.harmonic != start_harmonic


def test_components_teardown_and_restore_for_harmonic(
    make_viewer_model, qtbot
):
    """Cover teardown-on-layer-change and per-harmonic component restore."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    comp.components[0].g_edit.setText("0.2")
    comp.components[0].s_edit.setText("0.1")
    comp._on_component_coords_changed(0)
    comp.components[1].g_edit.setText("0.8")
    comp.components[1].s_edit.setText("0.5")
    comp._on_component_coords_changed(1)
    comp.components[0].name_edit.setText("A")
    comp._on_component_name_changed(0)
    comp._run_analysis()
    assert comp.comp1_fractions_layer is not None

    # Restore component coordinates stored for a different harmonic.
    layer.metadata["settings"]["component_analysis"] = {
        "components": {
            "0": {"name": "A", "gs_harmonics": {"2": {"g": 0.4, "s": 0.25}}},
            "1": {"name": "B", "gs_harmonics": {"2": {"g": 0.2, "s": 0.15}}},
        }
    }
    comp.current_image_layer_name = layer.name
    comp._restore_components_for_harmonic(2)

    # Tearing down on a layer change removes all artists and disconnects events.
    comp._teardown_on_layer_change()
    assert comp.components[0].dot is None
    assert comp.component_line is None


def test_components_widget_harmonics_none_fallback(make_viewer_model, qtbot):
    """Test component analysis when harmonics metadata is None (e.g. loaded .R64 files)."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    # Modify metadata to simulate a .R64 file where harmonics is None and G/S are 2D arrays (ndim matches data)
    layer.metadata["harmonics"] = None
    layer.metadata["G"] = layer.metadata["G"][0]
    layer.metadata["S"] = layer.metadata["S"][0]

    # 1. Test Linear Projection
    comp.analysis_type_combo.setCurrentText("Linear Projection")
    comp.components[0].g_edit.setText("0.2")
    comp.components[0].s_edit.setText("0.1")
    comp._on_component_coords_changed(0)
    comp.components[1].g_edit.setText("0.8")
    comp.components[1].s_edit.setText("0.5")
    comp._on_component_coords_changed(1)

    comp._run_analysis()
    assert comp.comp1_fractions_layer is not None

    # 2. Test Component Fit
    comp._add_component()
    comp.analysis_type_combo.setCurrentText("Component Fit")
    comp.components[2].g_edit.setText("0.5")
    comp.components[2].s_edit.setText("0.3")
    comp._on_component_coords_changed(2)

    comp._run_analysis()
    assert len(comp.fraction_layers) == 3


def test_components_widget_exceptions(make_viewer_model, qtbot):
    """Test component analysis handles missing metadata and IndexError."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    # Test Linear Projection with missing G array
    comp.analysis_type_combo.setCurrentText("Linear Projection")
    layer.metadata["G"] = None
    comp._run_analysis()  # Should return early

    # Test Linear Projection with missing harmonic (IndexError)
    layer.metadata["G"] = np.ones((2, 10, 10))
    layer.metadata["S"] = np.ones((2, 10, 10))
    layer.metadata["harmonics"] = np.array([999])
    comp._run_analysis()  # Should return early

    # Test Component Fit with missing G array
    comp._add_component()
    comp.analysis_type_combo.setCurrentText("Component Fit")
    layer.metadata["G"] = None
    comp._run_analysis()  # Should return early

    # Test Component Fit with missing harmonic (IndexError)
    layer.metadata["G"] = np.ones((2, 10, 10))
    layer.metadata["harmonics"] = np.array([999])
    comp._run_analysis()  # Should return early


def test_center_fill_slider_paint_event_zero_span(qtbot):
    """paintEvent should no-op (not raise) when minimum == maximum."""
    slider = CenterFillSlider()
    qtbot.addWidget(slider)
    slider.setMinimum(0)
    slider.setMaximum(0)
    slider.resize(100, 20)

    # QWidget.grab() forces a synchronous paint (unlike repaint(), which
    # only schedules one on the offscreen platform used in tests); the
    # span==0 guard must return early without error.
    slider.grab()


def _is_connected(emitter, bound_method):
    """Check whether ``bound_method`` is connected to a napari ``emitter``.

    Napari event emitters store callbacks for bound methods as
    ``(weakref_to_instance, method_name)`` tuples rather than the bound
    method object itself, so a plain ``in`` check does not work.
    """
    for cb in emitter.callbacks:
        if (
            isinstance(cb, tuple)
            and cb[0]() is bound_method.__self__
            and cb[1] == bound_method.__name__
        ):
            return True
        if cb is bound_method:
            return True
    return False


class _RemoveRaisesArtist:
    """Fake artist whose ``remove`` raises, exercising the suppressed
    ValueError/AttributeError branch in ``_remove_histogram_overlay``."""

    def remove(self):
        raise ValueError("boom")


def test_get_first_component_fraction_values(make_viewer_model, qtbot):
    """First-component fractions are pooled across the selected images."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    # With no fraction layers and no comp1_fractions_layer, an empty array
    # is returned.
    assert comp.comp1_fractions_layer is None
    assert comp._get_first_component_fraction_values().size == 0

    _setup_linear_projection(comp)
    assert comp.comp1_fractions_layer is not None
    comp1_name = comp.components[0].name_edit.text().strip() or "Component 1"
    original_size = np.asarray(
        comp.comp1_fractions_layer.data, dtype=float
    ).size

    # Fraction layers of deselected images are excluded from the pool.
    viewer.add_layer(
        Image(
            np.array([[0.25, 0.75], [np.nan, 0.5]]),
            name=_lp_name(comp1_name, "never_selected"),
        )
    )
    values = comp._get_first_component_fraction_values()
    assert values.size == original_size

    # A second analysed image must be selected for its fractions to be
    # pooled, and non-finite values are dropped.
    other_image = create_image_layer_with_phasors()
    other_image.name = "other_image"
    viewer.add_layer(other_image)
    parent.image_layers_checkable_combobox.setCheckedItems(
        [layer.name, other_image.name]
    )
    viewer.add_layer(
        Image(
            np.array([[0.25, 0.75], [np.nan, 0.5]]),
            name=_lp_name(comp1_name, "other_image"),
        )
    )
    values = comp._get_first_component_fraction_values()
    assert not np.any(np.isnan(values))
    assert values.size == original_size + 3

    # When no fraction layer matches the live component name, values fall
    # back to ``comp1_fractions_layer`` directly.
    comp.components[0].name_edit.setText("Renamed Component")
    values = comp._get_first_component_fraction_values()
    expected = np.asarray(comp.comp1_fractions_layer.data, dtype=float).ravel()
    expected = expected[np.isfinite(expected)]
    assert values.size == expected.size
    assert values.size > 0


def test_components_internal_helpers(make_viewer_model, qtbot):
    """Guards, layer reconnection, selection lookups and label updates."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    # ``_sync_component_layers_gamma`` returns immediately (without touching
    # any layers) while ``_updating_linked_layers`` is already True.
    comp._updating_linked_layers = True
    with patch.object(comp, "_get_all_layers_for_component") as mock_get:
        comp._sync_component_layers_gamma(0, 1.5)
    mock_get.assert_not_called()
    comp._updating_linked_layers = False

    # Reconnecting via the expected layer name connects the gamma event.
    fraction_layer = Image(
        np.zeros((5, 5)), name=_lp_name("Component 1", "img1")
    )
    viewer.add_layer(fraction_layer)
    comp._find_and_reconnect_layer(
        _lp_name("Component 1", "img1"), "Component 1", "img1", 0
    )
    assert comp.comp1_fractions_layer is fraction_layer
    assert _is_connected(
        fraction_layer.events.gamma, comp._on_colormap_changed
    )

    # Reconnecting via one of the fallback naming conventions renames the
    # layer and connects the gamma event.
    fraction_layer = Image(
        np.zeros((5, 5)), name=_lp_name("Component 1", "img2")
    )
    viewer.add_layer(fraction_layer)
    comp._find_and_reconnect_layer(
        _lp_name("Component 1", "RENAMED"), "Component 1", "img2", 0
    )
    assert comp.comp1_fractions_layer is fraction_layer
    assert fraction_layer.name == _lp_name("Component 1", "RENAMED")
    assert _is_connected(
        fraction_layer.events.gamma, comp._on_colormap_changed
    )

    # Without a parent widget there is no selection to honour.
    with patch.object(comp, "parent_widget", None):
        assert comp._get_selected_image_layer_names() == set()
    # A deleted Qt combobox is treated as an empty selection, not an error.
    with patch.object(
        parent,
        "get_selected_layers",
        side_effect=RuntimeError("wrapped C/C++ object has been deleted"),
    ):
        assert comp._get_selected_image_layer_names() == set()
    with patch.object(
        parent, "get_selected_layers", side_effect=AttributeError
    ):
        assert comp._get_selected_image_layer_names() == set()

    # Component name changes update name_label if present.
    comp.components[0].name_label = MagicMock()
    comp.components[0].name_edit.setText("NewName")
    comp.components[0].name_label.setText.assert_called_with("NewName")
    comp.components[0].name_edit.setText("")
    comp.components[0].name_label.setText.assert_called_with("Component 1")

    # Clearing components clears fields and updates labels.
    comp.components[0].name_edit.setText("MyComp")
    comp.components[0].g_edit.setText("0.4")
    comp.components[0].s_edit.setText("0.3")
    comp.components[0].coords_label = MagicMock()
    comp.components[0].name_label = MagicMock()
    comp._clear_components()
    comp.components[0].coords_label.setText.assert_called_with("G: -, S: -")
    comp.components[0].name_label.setText.assert_called_with("Component 1")
    assert comp.components[0].name_edit.text() == ""

    # Absolute concentration: the model is the wrapped phasorpy function,
    # with non-finite values left empty.
    from phasorpy.component import phasor_component_concentration

    mean = np.linspace(1, 9, 12).reshape(3, 4)
    real = np.linspace(0.3, 0.6, 12).reshape(3, 4)
    imag = np.linspace(0.2, 0.4, 12).reshape(3, 4)
    model = ([0.9, 0.25], [0.25, 0.43])
    calibration = (5.0, 0.8, 0.28)
    first, second, total = component_concentrations(
        mean, real, imag, *model, calibration, 2.0, brightness_ratio=1.5
    )
    want_first, want_second = phasor_component_concentration(
        mean, real, imag, *model, *calibration, 2.0, brightness_ratio=1.5
    )
    np.testing.assert_array_equal(first, want_first)
    np.testing.assert_array_equal(second, want_second)
    np.testing.assert_array_equal(total, first + second)
    only, no_second, no_total = component_concentrations(
        mean, real, imag, *model, calibration, 2.0
    )
    np.testing.assert_array_equal(only, want_first)
    assert no_second is None and no_total is None
    np.testing.assert_array_equal(
        _finite_or_nan(np.array([1.0, np.inf, -np.inf, np.nan])),
        [1.0, np.nan, np.nan, np.nan],
    )

    # A plane and a reference are only read at a harmonic that exists.
    from phasorpy.phasor import phasor_center

    plane_real, plane_imag = harmonic_plane(layer, 2)
    np.testing.assert_array_equal(plane_real, layer.metadata['G'][1])
    np.testing.assert_array_equal(plane_imag, layer.metadata['S'][1])
    assert harmonic_plane(layer, 5) == (None, None)
    centre = phasor_center(
        np.asarray(layer.data, dtype=float),
        layer.metadata['G'][0],
        layer.metadata['S'][0],
    )
    assert phasor_reference_from_layer(layer, 1) == pytest.approx(
        tuple(float(v) for v in centre)
    )
    assert phasor_reference_from_layer(layer, 5) is None
    # One plane is the only plane there is, whatever the harmonic asked;
    # stacked planes cannot be told apart without their harmonics.
    single = Image(
        np.asarray(layer.data),
        metadata={'G': layer.metadata['G'][0], 'S': layer.metadata['S'][0]},
    )
    np.testing.assert_array_equal(
        harmonic_plane(single, 3)[0], layer.metadata['G'][0]
    )
    unlabelled = Image(
        np.asarray(layer.data),
        metadata={'G': layer.metadata['G'], 'S': layer.metadata['S']},
    )
    assert harmonic_plane(unlabelled, 1) == (None, None)
    assert harmonic_plane(Image(np.zeros((4, 4))), 1) == (None, None)
    # Nothing measurable, or an intensity that does not match the phasor,
    # gives no reference.
    blank = Image(
        np.full(np.shape(layer.data), np.nan), metadata=dict(layer.metadata)
    )
    assert phasor_reference_from_layer(blank, 1) is None
    mismatched = Image(
        np.ones((3, 3)),
        metadata={'G': layer.metadata['G'][0], 'S': layer.metadata['S'][0]},
    )
    assert phasor_reference_from_layer(mismatched, 1) is None

    # Display ranges never collapse, and stored displays match the method.
    assert ComponentsWidget._value_limits(np.full(4, np.nan)) == (0.0, 1.0)
    assert ComponentsWidget._value_limits(np.array([])) == (0.0, 1.0)
    low, high = ComponentsWidget._value_limits(np.array([2.0, 2.0]))
    assert low == 2.0 and high > low
    assert ComponentsWidget._value_limits(np.array([0.5, np.nan, 3.0])) == (
        0.5,
        3.0,
    )
    assert ComponentsWidget._concentration_display(None, 0, 1) is None
    fraction_entry = {'colormap_name': "jet", 'analysis_type': "Component Fit"}
    settings = {
        'components': {'0': {'gs_harmonics': {'1': fraction_entry}}},
        'concentration': {
            'total_display': {'analysis_type': ABSOLUTE_CONCENTRATION}
        },
    }
    assert ComponentsWidget._concentration_display(settings, 0, 1) is None
    assert ComponentsWidget._concentration_display(settings, None, 1) == {
        'analysis_type': ABSOLUTE_CONCENTRATION
    }

    # A run replaces only its own harmonic's reference phasor.
    merge = ComponentsWidget._components_merge_rule([2])
    merged = merge(
        {
            'components': {},
            'concentration': {
                'reference_gs_harmonics': {
                    '1': {'g': 0.1, 's': 0.2},
                    '2': {'g': 0.3, 's': 0.4},
                }
            },
        },
        {
            'components': {},
            'concentration': {
                'reference_gs_harmonics': {'2': {'g': 0.5, 's': 0.6}}
            },
        },
    )
    assert merged['concentration']['reference_gs_harmonics'] == {
        '1': {'g': 0.1, 's': 0.2},
        '2': {'g': 0.5, 's': 0.6},
    }


def test_linear_projection_gamma_is_restored_and_preserved(
    make_viewer_model, qtbot
):
    """A gamma saved in metadata under the first component's harmonic data
    is picked up when the fraction layer is first created, and re-running
    the analysis preserves a manually-set gamma."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    layer.metadata["settings"]["component_analysis"] = {
        "components": {
            "0": {
                "name": "Component 1",
                "gs_harmonics": {
                    "1": {
                        "analysis_type": "Linear Projection",
                        "colormap_name": "viridis",
                        "gamma": 2.75,
                    }
                },
            },
            "1": {"name": "Component 2", "gs_harmonics": {}},
        },
    }

    _setup_linear_projection(comp)
    assert comp.comp1_fractions_layer is not None
    assert comp.comp1_fractions_layer.gamma == 2.75

    comp.comp1_fractions_layer.gamma = 2.5
    comp._run_analysis()
    assert comp.comp1_fractions_layer.gamma == 2.5


def test_component_fit_restores_saved_gamma_for_non_first_component(
    make_viewer_model, qtbot
):
    """Gamma saved under a non-first component's harmonic data is restored
    when its fraction layer is (re-)created during Component Fit."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    layer.metadata["settings"]["component_analysis"] = {
        "components": {
            "0": {"name": "Component 1", "gs_harmonics": {}},
            "1": {
                "name": "Component 2",
                "gs_harmonics": {
                    "1": {
                        "analysis_type": "Component Fit",
                        "colormap_name": "viridis",
                        "gamma": 3.25,
                    }
                },
            },
        },
    }

    comp.analysis_type_combo.setCurrentText("Component Fit")
    comp.components[0].g_edit.setText("0.2")
    comp.components[0].s_edit.setText("0.1")
    comp._on_component_coords_changed(0)
    comp.components[1].g_edit.setText("0.8")
    comp.components[1].s_edit.setText("0.5")
    comp._on_component_coords_changed(1)
    comp._run_analysis()

    assert len(comp.fraction_layers) == 2
    assert comp.fraction_layers[1].gamma == 3.25


def _valid_histogram_values():
    return np.concatenate(
        [np.full(20, 0.3), np.full(30, 0.6), np.full(5, 0.9)]
    )


def test_draw_fraction_histogram_overlay_empty_values_returns_none():
    fig, ax = plt.subplots()
    try:
        result = draw_fraction_histogram_overlay(
            ax, 0.0, 0.0, 1.0, 1.0, np.array([np.nan, np.nan]), None
        )
    finally:
        plt.close(fig)
    assert result is None


def test_draw_fraction_histogram_overlay_all_values_out_of_range():
    """Values entirely outside the fixed [0, 1] histogram range produce an
    all-zero histogram (``counts.max() == 0``)."""
    fig, ax = plt.subplots()
    try:
        result = draw_fraction_histogram_overlay(
            ax, 0.0, 0.0, 1.0, 1.0, np.full(10, 5.0), None
        )
    finally:
        plt.close(fig)
    assert result is None


def test_draw_fraction_histogram_overlay_smoothed_peak_not_positive():
    """If the (best-effort) smoothing step collapses the histogram to all
    zeros, the function bails out instead of drawing a degenerate curve."""
    fig, ax = plt.subplots()
    try:
        with patch(
            "scipy.ndimage.gaussian_filter1d",
            return_value=np.zeros(150),
        ):
            result = draw_fraction_histogram_overlay(
                ax, 0.0, 0.0, 1.0, 1.0, _valid_histogram_values(), None
            )
    finally:
        plt.close(fig)
    assert result is None


def test_draw_fraction_histogram_overlay_zero_length_line():
    fig, ax = plt.subplots()
    try:
        result = draw_fraction_histogram_overlay(
            ax, 0.5, 0.5, 0.5, 0.5, _valid_histogram_values(), None
        )
    finally:
        plt.close(fig)
    assert result is None


def test_draw_fraction_histogram_overlay_normal_flip_sign():
    """A line direction whose default normal points 'down' is flipped so it
    still points up (the ``ny < 0`` branch)."""
    fig, ax = plt.subplots()
    try:
        # dx = -1, dy = 0 -> default normal (0, -1) has ny < 0 and gets flipped.
        result = draw_fraction_histogram_overlay(
            ax, 1.0, 0.0, 0.0, 0.0, _valid_histogram_values(), None
        )
    finally:
        plt.close(fig)
    assert result is not None


def test_draw_fraction_histogram_overlay_zero_height_returns_none():
    """A zero overlay height collapses every bar to zero (``v_max <= 0``)."""
    fig, ax = plt.subplots()
    try:
        result = draw_fraction_histogram_overlay(
            ax,
            0.0,
            0.0,
            1.0,
            1.0,
            _valid_histogram_values(),
            None,
            height=0.0,
        )
    finally:
        plt.close(fig)
    assert result is None


def test_draw_fraction_histogram_overlay_degenerate_contrast_limits():
    """``vmax <= vmin`` in the contrast limits is nudged apart instead of
    raising in the normalization step."""
    fig, ax = plt.subplots()
    try:
        result = draw_fraction_histogram_overlay(
            ax,
            0.0,
            0.0,
            1.0,
            1.0,
            _valid_histogram_values(),
            None,
            contrast_limits=(0.5, 0.5),
        )
    finally:
        plt.close(fig)
    assert result is not None


def test_draw_fraction_histogram_overlay_gamma_power_norm():
    from matplotlib.colors import PowerNorm

    fig, ax = plt.subplots()
    try:
        result = draw_fraction_histogram_overlay(
            ax,
            0.0,
            0.0,
            1.0,
            1.0,
            _valid_histogram_values(),
            None,
            contrast_limits=(0.0, 1.0),
            gamma=2.0,
        )
    finally:
        plt.close(fig)
    assert result is not None
    assert isinstance(result[0].norm, PowerNorm)


def test_draw_fraction_histogram_overlay_fractions_colormap_branches():
    """Small (<=32 entries) colormaps use a smoothly-interpolated colormap;
    an unset colormap falls back to ``jet``."""
    fig, ax = plt.subplots()
    try:
        small_colormap = [
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
        result_small = draw_fraction_histogram_overlay(
            ax, 0.0, 0.0, 1.0, 1.0, _valid_histogram_values(), small_colormap
        )
        assert result_small is not None

        result_none = draw_fraction_histogram_overlay(
            ax, 0.0, 0.0, 1.0, 1.0, _valid_histogram_values(), None
        )
        assert result_none is not None
        assert result_none[0].get_cmap() is plt.cm.jet
    finally:
        plt.close(fig)


def test_draw_components_overlay_draws_fraction_histogram():
    """``draw_components_overlay`` delegates to
    ``draw_fraction_histogram_overlay`` when Linear Projection, the fraction
    histogram setting, fraction data, and a fractions colormap are all
    present."""
    fig, ax = plt.subplots()
    try:
        settings = {
            "show_fraction_histogram": True,
            "fraction_data": _valid_histogram_values(),
            "fractions_colormap": [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            "colormap_contrast_limits": (0.0, 1.0),
        }
        with patch(
            "napari_phasors.components_tab.draw_fraction_histogram_overlay"
        ) as mock_draw:
            draw_components_overlay(
                ax,
                [0.6, 0.3],
                [0.3, 0.2],
                names=["Component 1", "Component 2"],
                analysis_type="Linear Projection",
                settings=settings,
            )
        mock_draw.assert_called_once()
    finally:
        plt.close(fig)

    # Concentrations draw a plain line, never a fraction gradient.
    fig, ax = plt.subplots()
    try:
        draw_components_overlay(
            ax,
            [0.9, 0.25],
            [0.25, 0.43],
            ["Free", "Bound"],
            None,
            ABSOLUTE_CONCENTRATION,
            {
                "show_labels": True,
                "show_colormap_line": True,
                "fractions_colormap": plt.get_cmap("jet")(
                    np.linspace(0, 1, 8)
                ),
            },
        )
        assert not any(isinstance(c, LineCollection) for c in ax.collections)
        assert not any(line.get_marker() == '*' for line in ax.lines)
    finally:
        plt.close(fig)


def _setup_component_fit(comp, coords=(("0.2", "0.1"), ("0.8", "0.5"))):
    """Configure components and run a Component Fit analysis."""
    comp.analysis_type_combo.setCurrentText("Component Fit")
    for i, (g, s) in enumerate(coords):
        comp.components[i].g_edit.setText(g)
        comp.components[i].s_edit.setText(s)
        comp._on_component_coords_changed(i)
    comp._run_analysis()


def _setup_two_image_layers(make_viewer_model):
    """Two phasor image layers plus a plotter, for multi-layer tests."""
    viewer = make_viewer_model()
    layer_a = create_image_layer_with_phasors()
    layer_a.name = "img_a"
    layer_b = create_image_layer_with_phasors()
    layer_b.name = "img_b"
    layer_a.metadata["settings"] = {"frequency": 80.0}
    layer_b.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer_a)
    viewer.add_layer(layer_b)
    parent = PlotterWidget(viewer)
    # Both layers checked in "phasor layers", as when analysing them together.
    parent.image_layers_checkable_combobox.setCheckedItems(
        [layer_a.name, layer_b.name]
    )
    comp = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp)
    return viewer, layer_a, layer_b, parent, comp


def test_component_fit_rename_replaces_layers(make_viewer_model, qtbot):
    """Renaming components between runs relabels the fraction layers instead
    of leaving stale default-named duplicates behind."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_component_fit(comp)
    first_run = sorted(
        lyr.name
        for lyr in viewer.layers
        if is_component_fit_label(split_analysis_layer_name(lyr.name)[1])
    )
    assert len(first_run) == 2

    comp.components[0].name_edit.setText("Alpha")
    comp._on_component_name_changed(0)
    comp.components[1].name_edit.setText("Beta")
    comp._on_component_name_changed(1)
    comp._run_analysis()

    second_run = sorted(
        lyr.name
        for lyr in viewer.layers
        if is_component_fit_label(split_analysis_layer_name(lyr.name)[1])
    )
    assert len(second_run) == 2, second_run
    assert all("Alpha" in n or "Beta" in n for n in second_run)
    assert not any(
        "Component 1" in n or "Component 2" in n for n in second_run
    )


def test_manually_renamed_fraction_layer_updated_in_place(
    make_viewer_model, qtbot
):
    """A manually renamed fraction layer is matched by its metadata tag on
    re-run: data updates in place, the custom name is kept, no duplicate."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_component_fit(comp)

    tagged = [
        lyr
        for lyr in viewer.layers
        if lyr.metadata.get('phasor_component_fraction')
    ]
    assert len(tagged) == 2
    target = tagged[1]
    assert target.metadata['phasor_component_fraction']['component_index'] == 1
    target.name = "MyRenamedFractions"
    data_before = np.array(target.data, copy=True)

    # Nudge a component and re-run: the renamed layer must be found via
    # _find_component_fraction_layer's tag path and replaced in place.
    comp.components[1].g_edit.setText("0.75")
    comp._on_component_coords_changed(1)
    comp._run_analysis()

    names = [lyr.name for lyr in viewer.layers]
    assert "MyRenamedFractions" in names
    tagged_after = [
        lyr
        for lyr in viewer.layers
        if lyr.metadata.get('phasor_component_fraction')
    ]
    assert len(tagged_after) == 2, [lyr.name for lyr in tagged_after]
    renamed = viewer.layers["MyRenamedFractions"]
    assert (
        renamed.metadata['phasor_component_fraction']['component_index'] == 1
    )
    assert not np.array_equal(renamed.data, data_before)


def test_source_rename_updates_fraction_tags(make_viewer_model, qtbot):
    """rename_layer keeps the metadata tag's source_layer in sync, including
    for fraction layers the user renamed manually."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_component_fit(comp)
    fracs = [
        lyr
        for lyr in viewer.layers
        if lyr.metadata.get('phasor_component_fraction')
    ]
    fracs[0].name = "CustomFrac"

    old = layer.name
    comp.rename_layer(old, "renamed_source")
    layer.name = "renamed_source"

    for lyr in [viewer.layers["CustomFrac"], fracs[1]]:
        tag = lyr.metadata['phasor_component_fraction']
        assert tag['source_layer'] == "renamed_source", (lyr.name, tag)
    # Default-named layer's name suffix followed the source rename too.
    assert fracs[1].name == _fit_name("Component 2", "renamed_source"), fracs[
        1
    ].name

    # Re-run: the custom-named layer is still matched (no duplicate).
    comp.components[1].g_edit.setText("0.75")
    comp._on_component_coords_changed(1)
    comp._run_analysis()
    tagged = [
        lyr
        for lyr in viewer.layers
        if lyr.metadata.get('phasor_component_fraction')
    ]
    assert len(tagged) == 2, [lyr.name for lyr in tagged]
    assert "CustomFrac" in [lyr.name for lyr in viewer.layers]


def test_renamed_fraction_layer_stays_in_histogram_combobox(
    make_viewer_model, qtbot
):
    """Discovery and resolution are metadata-aware: a renamed fraction layer
    keeps its component listed and selectable in the histogram combobox."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_component_fit(comp)

    frac2 = comp._get_fraction_layers_for_component("Component 2")
    target = next(iter(frac2.values()))
    target.name = "SomethingCustom"

    assert "Component 2" in comp._get_component_names_from_fraction_layers()
    resolved = comp._get_fraction_layers_for_component("Component 2")
    assert any(lyr.name == "SomethingCustom" for lyr in resolved.values())

    comp._update_histogram_component_toggles()
    assert _histogram_toggle(comp, "Component 2").isEnabled()
    _check_histogram_components(comp, ["Component 2"])
    comp.update_component_histogram()
    assert list(comp.histogram_widget._datasets.keys()) == ["SomethingCustom"]


def test_component_fraction_layer_lookup_by_tag(make_viewer_model, qtbot):
    """The tag -> display-name helper, and _find_component_fraction_layer
    matching by tag, falling back to the default name, or returning None."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    # Non-dict tags resolve to None.
    assert comp._component_display_name_from_tag(None) is None
    assert comp._component_display_name_from_tag("bogus") is None
    # Default name when the source layer has no custom component name.
    tag = {'source_layer': layer.name, 'component_index': 1}
    assert comp._component_display_name_from_tag(tag) == "Component 2"
    assert (
        comp._find_component_fraction_layer(layer.name, 0, "missing") is None
    )

    _setup_component_fit(comp)
    default_name = _fit_name("Component 1", layer.name)
    found = comp._find_component_fraction_layer(layer.name, 0, default_name)
    assert found is not None and found.name == default_name
    # Tag path: still found after a manual rename.
    found.name = "Camouflaged"
    refound = comp._find_component_fraction_layer(layer.name, 0, default_name)
    assert refound is not None and refound.name == "Camouflaged"
    # Fallback path: an untagged layer matching the default name is found.
    del refound.metadata['phasor_component_fraction']
    refound.name = default_name
    fallback = comp._find_component_fraction_layer(layer.name, 0, default_name)
    assert fallback is not None and fallback.name == default_name

    # Custom name from the source layer's component settings.
    comp.components[1].name_edit.setText("Bound")
    comp._on_component_name_changed(1)
    assert comp._component_display_name_from_tag(tag) == "Bound"
    # Unknown component index without a source falls back to the default.
    assert (
        comp._component_display_name_from_tag({'component_index': 4})
        == "Component 5"
    )


def test_component_fit_multi_layer_per_row_datasets(make_viewer_model, qtbot):
    """Multi-image Component Fit feeds one histogram dataset per image,
    labelled after the analysis fraction layers."""
    viewer, layer_a, layer_b, parent, comp = _setup_two_image_layers(
        make_viewer_model
    )
    with patch.object(
        parent, "get_selected_layers", return_value=[layer_a, layer_b]
    ):
        _setup_component_fit(comp)
    _check_histogram_components(comp, ["Component 1"])
    comp.update_component_histogram()
    assert set(comp.histogram_widget._datasets.keys()) == {
        _fit_name("Component 1", "img_a"),
        _fit_name("Component 1", "img_b"),
    }


def test_linear_projection_second_component_per_row(make_viewer_model, qtbot):
    """The Linear Projection second component (fraction = 1 - first) gets
    per-image datasets with virtual '<component> fractions: <image>' labels."""
    viewer, layer_a, layer_b, parent, comp = _setup_two_image_layers(
        make_viewer_model
    )
    comp.analysis_type_combo.setCurrentText("Linear Projection")
    for i, (g, s) in enumerate([("0.2", "0.1"), ("0.8", "0.5")]):
        comp.components[i].g_edit.setText(g)
        comp.components[i].s_edit.setText(s)
        comp._on_component_coords_changed(i)
    with patch.object(
        parent, "get_selected_layers", return_value=[layer_a, layer_b]
    ):
        comp._run_analysis()
    name1, name2 = comp._linear_projection_component_names()
    comp._update_histogram_component_toggles()
    _check_histogram_components(comp, [name2])
    comp.update_component_histogram()
    assert set(comp.histogram_widget._datasets.keys()) == {
        _lp_name(name2, "img_a"),
        _lp_name(name2, "img_b"),
    }


def test_switch_to_linear_projection_hides_stale_component_fit(
    make_viewer_model, qtbot
):
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    # 3-component fit -> Component 1/2/3 fraction layers (tagged).
    comp._add_component()
    _setup_component_fit(
        comp, coords=(("0.2", "0.1"), ("0.5", "0.3"), ("0.8", "0.5"))
    )
    assert len(comp.fraction_layers) == 3

    # In Component Fit mode all three components are offered.
    comp._update_histogram_component_toggles()
    fit_names = _available_histogram_components(comp)
    assert {"Component 1", "Component 2", "Component 3"} <= set(fit_names)

    # Drop the 3rd component so Linear Projection becomes available, switch,
    # and run it. The component-fit fraction layers remain in the viewer.
    comp._remove_component()
    comp._update_analysis_options()
    comp.analysis_type_combo.setCurrentText("Linear Projection")
    assert comp.analysis_type == "Linear Projection"
    comp._run_analysis()

    comp._update_histogram_component_toggles()
    lp_names = _available_histogram_components(comp)
    assert "Component 1" in lp_names
    assert "Component 2" in lp_names
    assert "Component 3" not in lp_names

    # Component 2 must resolve to the complementary (invert=True) using the
    # linear-projection Component 1 layer, NOT the stale component-fit layer.
    fmap, invert = comp._resolve_histogram_component("Component 2")
    assert invert is True, "Component 2 should be the complementary fraction"
    only_layer = next(iter(fmap.values()))
    assert only_layer.metadata.get('phasor_component_fraction') is None
    assert only_layer.name == _lp_name("Component 1", layer.name)


def _setup_analysed_layers(viewer, count=3):
    """Add `count` phasor layers, select them all and run a 2-component fit.

    Returns
    -------
    tuple of (PlotterWidget, ComponentsWidget, str)
        The plotter, its components tab and the analysed component name.
    """
    for i in range(count):
        layer = create_image_layer_with_phasors()
        layer.name = f"img{i}"
        viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    parent.image_layers_checkable_combobox.setCheckedItems(
        [f"img{i}" for i in range(count)]
    )

    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")

    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(1)
    comp_widget._run_analysis()

    return (
        parent,
        comp_widget,
        comp_widget._primary_histogram_component(),
    )


def _select_layers(parent, names):
    """Check `names` and run the plotter's debounced selection handler now."""
    parent.image_layers_checkable_combobox.setCheckedItems(names)
    parent._layer_selection_timer.stop()
    parent._process_layer_selection_change()


def _click_phasor_layer(qtbot, parent, layer_name):
    """Toggle a layer through the real checkable-combobox popup."""
    combo = parent.image_layers_checkable_combobox
    row = next(
        row
        for row in range(combo.model().rowCount())
        if combo.model().item(row).text() == layer_name
    )
    combo.showPopup()
    view = combo.view()
    rect = view.visualRect(combo.model().index(row, 0))
    position = rect.center()
    position.setX(rect.left() + 5)
    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=position)
    combo.hidePopup()


def _histogram_legend_labels(histogram):
    """Return the labels currently rendered in the histogram legend."""
    legend = histogram.ax.get_legend()
    return [] if legend is None else [text.get_text() for text in legend.texts]


def _assert_individual_histogram(histogram, expected_labels):
    """Assert internal and rendered state for Individual layers mode."""
    assert list(histogram._datasets) == expected_labels
    assert list(histogram._counts_per_dataset) == expected_labels
    assert len(histogram.ax.lines) == len(expected_labels)
    assert _histogram_legend_labels(histogram) == expected_labels


def test_components_histogram_follows_layer_selection(
    make_napari_viewer, qtbot
):
    """Histogram and fraction layers track the selected layers (issue #358).

    Deselected layers are hidden, not deleted, are left untouched by the
    range slider, and are not resurrected by re-running the analysis.
    """
    viewer = make_napari_viewer()
    parent, comp_widget, comp_name = _setup_analysed_layers(viewer)

    def fraction_sources():
        return sorted(
            comp_widget._get_fraction_layers_for_component(comp_name)
        )

    assert fraction_sources() == ["img0", "img1", "img2"]
    assert len(comp_widget.histogram_widget._datasets) == 3

    # The real selectionChanged -> debounce -> components tab wiring works:
    # emit the real signal and let the 300 ms debounce timer expire.
    parent.image_layers_checkable_combobox.setCheckedItems(["img0", "img1"])
    qtbot.waitUntil(
        lambda: len(comp_widget.histogram_widget._datasets) == 2,
        timeout=5000,
    )
    assert fraction_sources() == ["img0", "img1"]
    assert viewer.layers[_lp_name(comp_name, "img2")].visible is False

    deselected = viewer.layers[_lp_name(comp_name, "img1")]
    untouched_data = deselected.data.copy()

    _select_layers(parent, ["img0"])
    assert fraction_sources() == ["img0"]
    assert len(comp_widget.histogram_widget._datasets) == 1
    assert viewer.layers[_lp_name(comp_name, "img0")].visible is True
    assert viewer.layers[_lp_name(comp_name, "img1")].visible is False
    assert viewer.layers[_lp_name(comp_name, "img2")].visible is False

    # The fraction range slider must leave deselected layers untouched.
    selected = viewer.layers[_lp_name(comp_name, "img0")]
    selected_original = selected.data.copy()
    comp_widget._on_fraction_range_changed(0.2, 0.8)
    np.testing.assert_allclose(
        selected.data,
        np.clip(selected_original, 0.2, 0.8),
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )
    np.testing.assert_array_equal(deselected.data, untouched_data)

    # Re-running the analysis must not resurrect deselected layers: their
    # fraction layers still exist in the viewer, but must not contribute.
    comp_widget._run_analysis()
    assert _lp_name(comp_name, "img1") in viewer.layers
    assert fraction_sources() == ["img0"]
    assert len(comp_widget.histogram_widget._datasets) == 1

    _select_layers(parent, ["img0", "img1", "img2"])
    assert fraction_sources() == ["img0", "img1", "img2"]
    assert len(comp_widget.histogram_widget._datasets) == 3
    assert viewer.layers[_lp_name(comp_name, "img1")].visible is True
    assert viewer.layers[_lp_name(comp_name, "img2")].visible is True

    # Unchecking every layer empties the fraction histogram.
    _select_layers(parent, [])
    assert _available_histogram_components(comp_widget) == []
    assert comp_widget._selected_histogram_components() == []
    assert comp_widget.histogram_widget.counts is None
    assert comp_widget.histogram_widget._datasets == {}
    assert not comp_widget.histogram_widget.isHidden()


def test_components_individual_histogram_follows_real_popup_click(
    make_napari_viewer, qtbot
):
    """A real non-primary uncheck removes its Linear Projection curve."""
    viewer = make_napari_viewer()
    parent, comp, comp_name = _setup_analysed_layers(viewer, count=2)
    qtbot.addWidget(parent)
    parent.show()
    qtbot.wait(50)
    histogram = comp.histogram_widget
    histogram.display_mode = "Individual layers"
    labels = [
        _lp_name(comp_name, "img0"),
        _lp_name(comp_name, "img1"),
    ]
    _assert_individual_histogram(histogram, labels)

    selection_events = []
    parent.image_layers_checkable_combobox.selectionChanged.connect(
        lambda: selection_events.append(True)
    )

    _click_phasor_layer(qtbot, parent, "img1")

    qtbot.waitUntil(
        lambda: parent.get_selected_layer_names() == ["img0"]
        and len(histogram._datasets) == 1,
        timeout=5000,
    )

    assert selection_events
    assert sorted(comp._get_fraction_layers_for_component(comp_name)) == [
        "img0"
    ]
    _assert_individual_histogram(histogram, labels[:1])
    assert viewer.layers[labels[1]].visible is False

    comp._run_analysis()

    _assert_individual_histogram(histogram, labels[:1])


def test_component_fit_individual_histogram_follows_primary_popup_click(
    make_napari_viewer, qtbot
):
    """A real primary uncheck removes its tagged Component Fit curve."""
    viewer, _, _, parent, comp = _setup_two_image_layers(make_napari_viewer)
    qtbot.addWidget(parent)
    parent.show()
    qtbot.wait(50)
    _setup_component_fit(comp)
    comp_name = comp._primary_histogram_component()
    histogram = comp.histogram_widget
    histogram.display_mode = "Individual layers"
    labels = [
        _fit_name(comp_name, "img_a"),
        _fit_name(comp_name, "img_b"),
    ]
    _assert_individual_histogram(histogram, labels)

    selection_events = []
    primary_events = []
    combo = parent.image_layers_checkable_combobox
    combo.selectionChanged.connect(lambda: selection_events.append(True))
    combo.primaryLayerChanged.connect(primary_events.append)

    _click_phasor_layer(qtbot, parent, "img_a")

    qtbot.waitUntil(
        lambda: parent.get_selected_layer_names() == ["img_b"]
        and len(histogram._datasets) == 1,
        timeout=5000,
    )

    assert selection_events
    assert primary_events[-1] == "img_b"
    assert sorted(comp._get_fraction_layers_for_component(comp_name)) == [
        "img_b"
    ]
    _assert_individual_histogram(histogram, labels[1:])
    assert viewer.layers[labels[0]].visible is False

    comp._run_analysis()

    _assert_individual_histogram(histogram, labels[1:])


def test_sync_fraction_layer_visibility_only_touches_fraction_layers(
    make_viewer_model, qtbot
):
    """Fraction layers follow the selection; other layers are left alone."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_linear_projection(comp)
    fraction_layer = comp.comp1_fractions_layer
    assert fraction_layer is not None
    assert fraction_layer.visible is True

    unrelated = Image(np.zeros((2, 2)), name="unrelated")
    viewer.add_layer(unrelated)

    with patch.object(parent, "get_selected_layers", return_value=[]):
        comp._sync_fraction_layer_visibility()

    assert fraction_layer.visible is False
    # Neither the source image nor an unrelated layer is touched.
    assert unrelated.visible is True
    assert layer.visible is True
    # The guard flag is always released.
    assert comp._updating_linked_layers is False

    comp._sync_fraction_layer_visibility()

    assert fraction_layer.visible is True


def _projection_widget(viewer, name="proj_layer"):
    """Return a components tab configured for a Linear Projection run."""
    layer = create_image_layer_with_phasors()
    layer.name = name
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")
    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(1)
    return parent, comp_widget, layer


def test_linear_projection_per_layer_failures(make_viewer_model, qtbot):
    """Layers the projection cannot use are skipped or reported by name,
    and the per-layer entry point works on its own."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _projection_widget(viewer)
    real = layer.metadata["G"]
    harmonics = layer.metadata["harmonics"]
    c1 = comp_widget.components[0]

    # A layer whose G/S went missing yields no fraction layer, from the run
    # or from the per-layer entry point.
    layer.metadata["G"] = None
    comp_widget._run_analysis()
    assert comp_widget.comp1_fractions_layer is None
    comp_widget._run_linear_projection_for_layer(
        layer,
        np.array([0.2, 0.8]),
        np.array([0.1, 0.5]),
        c1,
    )
    assert comp_widget.comp1_fractions_layer is None
    layer.metadata["G"] = real

    # A layer that never computed the selected harmonic is skipped.
    layer.metadata["harmonics"] = np.array([97])
    comp_widget._run_analysis()
    assert comp_widget.comp1_fractions_layer is None
    layer.metadata["harmonics"] = harmonics

    # A projection that raises names the layer instead of failing silently.
    errors = []

    def explode(*args, **kwargs):
        raise RuntimeError("projection boom")

    with (
        patch("napari_phasors.components_tab.show_error", errors.append),
        patch(
            "napari_phasors.components_tab.phasor_component_fraction",
            explode,
        ),
    ):
        comp_widget._run_analysis()
    assert any("projection boom" in message for message in errors)
    assert any("proj_layer" in message for message in errors)

    # The per-layer entry point still works without a precomputed map.
    comp_widget._run_analysis()
    comp_widget.comp1_fractions_layer = None
    comp_widget._run_linear_projection_for_layer(
        layer,
        np.array([0.2, 0.8]),
        np.array([0.1, 0.5]),
        c1,
    )
    assert comp_widget.comp1_fractions_layer is not None

    # Absolute concentration: a model error or a missing harmonic is
    # reported, never raised.
    comp = comp_widget
    _concentration_ready(viewer, layer, parent, comp)
    with (
        patch(
            "napari_phasors.components_tab.component_concentrations",
            side_effect=ValueError("invalid g_cal=nan"),
        ),
        patch("napari_phasors.components_tab.show_error") as error,
    ):
        comp._run_analysis()
    # A rerun after an earlier analysis also re-applies the filter stack,
    # which runs the analysis once more and reports the same error.
    assert error.call_count >= 1
    assert all("invalid g_cal" in call[0][0] for call in error.call_args_list)
    assert _concentration_maps(viewer) == {}
    with (
        patch(
            "napari_phasors.components_tab.harmonic_plane",
            return_value=(None, None),
        ),
        patch("napari_phasors.components_tab.show_warning") as warn,
    ):
        comp._run_concentration()
    warn.assert_called_once()
    assert "no phasor data at harmonic 1" in warn.call_args[0][0]
    # Missing inputs make no parameters, and no run.
    comp.reference_concentration_edit.setText("")
    assert comp._concentration_parameters() is None
    comp._run_concentration()
    assert _concentration_maps(viewer) == {}
    # Neither does an empty selection.
    comp.reference_concentration_edit.setText("1")
    with patch.object(parent, "get_selected_layers", return_value=[]):
        comp._run_concentration()
    assert _concentration_maps(viewer) == {}
    # Restoring the components from settings never stages the section.
    parent.settings_store.discard_drafts([layer])
    comp._updating_settings = True
    try:
        comp._stage_concentration_settings()
    finally:
        comp._updating_settings = False
    assert not parent.settings_store.has_draft(layer)
    # An intensity that does not match the phasor cannot be analysed.
    odd = Image(
        np.ones((3, 3)),
        metadata={'G': np.zeros((5, 5)), 'S': np.zeros((5, 5))},
    )
    assert ComponentsWidget._compute_layer_concentrations(odd, {}, 1) is None


def _fit_widget(make_viewer_model, name="fit_layer"):
    """Return a components tab configured for a three-component fit."""
    viewer, layer, parent, comp_widget = _setup_components(make_viewer_model)
    layer.name = name
    comp_widget._add_component()
    comp_widget.analysis_type_combo.setCurrentText("Component Fit")
    for index, (g_value, s_value) in enumerate(
        [("0.2", "0.1"), ("0.5", "0.3"), ("0.8", "0.5")]
    ):
        comp_widget.components[index].g_edit.setText(g_value)
        comp_widget.components[index].s_edit.setText(s_value)
        comp_widget._on_component_coords_changed(index)
    return parent, comp_widget, layer


def test_component_fit_per_layer_failures(make_viewer_model, qtbot):
    """A fit that raises is reported rather than swallowed, and the per-layer
    path prepares, fits and bails out on its own."""
    parent, comp_widget, layer = _fit_widget(make_viewer_model)

    errors = []
    with (
        patch("napari_phasors.components_tab.show_error", errors.append),
        patch(
            "napari_phasors.components_tab.phasor_component_fit",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fit boom")),
        ),
    ):
        comp_widget._run_analysis()
    assert any("fit boom" in message for message in errors)

    comp_widget._run_analysis()
    active = [c for c in comp_widget.components if c is not None and c.dot]
    harmonic = getattr(parent, "harmonic", 1)
    required = comp_widget._get_required_harmonics(len(active))

    # Called without a precomputed fit, the per-layer path does the work.
    comp_widget._run_component_fit_for_layer(
        layer, len(active), harmonic, required
    )
    assert layer.metadata["settings"]["component_analysis"]

    # A harmonic with no component positions is skipped without raising:
    # nothing was ever placed on harmonic 7, so there is nothing to fit.
    before = dict(layer.metadata["settings"]["component_analysis"])
    comp_widget._run_component_fit_for_layer(layer, 2, 7, 1)
    assert layer.metadata["settings"]["component_analysis"] == before

    # The standalone path reports a failing fit the same way.
    errors = []
    with (
        patch("napari_phasors.components_tab.show_error", errors.append),
        patch(
            "napari_phasors.components_tab.phasor_component_fit",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("solo boom")),
        ),
    ):
        comp_widget._run_component_fit_for_layer(
            layer, len(active), harmonic, required
        )
    assert any("solo boom" in message for message in errors)


def test_components_card_selection(make_viewer_model, qtbot):
    """Cards are selected by clicking them, their inputs, or the canvas."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    assert len(comp.components) == 2
    assert comp._selected_component is comp.components[0]
    assert comp.components[0].card_frame.property("selected")
    assert not comp.components[1].card_frame.property("selected")

    # Click second component's card frame
    qtbot.mouseClick(comp.components[1].card_frame, Qt.LeftButton)
    assert comp._selected_component is comp.components[1]
    assert not comp.components[0].card_frame.property("selected")
    assert comp.components[1].card_frame.property("selected")

    # Focusing or moving the cursor in any input field selects that card.
    comp._select_component_item(0)
    assert comp._selected_component is comp.components[0]
    assert comp.components[0].card_frame.property("selected")
    comp.components[1].name_edit.cursorPositionChanged.emit(0, 1)
    assert comp._selected_component is comp.components[1]
    assert comp.components[1].card_frame.property("selected")
    comp.components[0].g_edit.cursorPositionChanged.emit(0, 1)
    assert comp._selected_component is comp.components[0]
    comp.components[1].s_edit.cursorPositionChanged.emit(0, 1)
    assert comp._selected_component is comp.components[1]
    comp.components[0].lifetime_edit.cursorPositionChanged.emit(0, 1)
    assert comp._selected_component is comp.components[0]

    # Invalid indices deselect; a ComponentState selects itself; a foreign
    # component is ignored.
    comp._select_component_item(-1)
    assert comp._selected_component is None
    for c in comp.components:
        assert not c.card_frame.property("selected")
    comp._select_component_item(999)
    assert comp._selected_component is None
    comp._select_component_item(comp.components[1])
    assert comp._selected_component is comp.components[1]
    assert comp.components[1].card_frame.property("selected")
    foreign_comp = MagicMock()
    comp._select_component_item(foreign_comp)
    assert comp._selected_component is comp.components[1]

    # Each card embeds its inputs directly, in two rows.
    c0 = comp.components[0]
    assert c0.number_label.text() == "1."
    assert c0.name_edit is not None
    assert c0.select_button is not None
    assert c0.remove_button is not None
    assert c0.g_edit is not None
    assert c0.s_edit is not None
    assert c0.lifetime_edit is not None
    c0.name_edit.setText("Donor Fluorophore")
    assert c0.name_edit.text() == "Donor Fluorophore"
    c0.g_edit.setText("0.600")
    c0.s_edit.setText("0.400")
    comp._on_component_coords_changed(0)
    assert c0.g_edit.text() == "0.600"
    assert c0.s_edit.text() == "0.400"
    c0.lifetime_edit.setText("2.500")
    comp._update_component_from_lifetime(0)
    assert c0.lifetime_edit.text() == "2.500"

    # Clicking a component's text label selects its card and starts a drag.
    comp.components[0].g_edit.setText("0.3")
    comp.components[0].s_edit.setText("0.2")
    comp._on_component_coords_changed(0)
    comp.components[1].name_edit.setText("Comp 2")
    comp.components[1].g_edit.setText("0.7")
    comp.components[1].s_edit.setText("0.4")
    comp._on_component_coords_changed(1)
    assert comp.components[1].text is not None
    comp._select_component_item(0)
    assert comp._selected_component is comp.components[0]
    mock_event = MagicMock()
    mock_event.inaxes = parent.canvas_widget.axes
    with patch.object(
        comp.components[1].text, "contains", return_value=(True, {})
    ):
        comp._on_press(mock_event)
    assert comp._selected_component is comp.components[1]
    assert comp.components[1].card_frame.property("selected")
    assert comp.dragging_label_idx == 1
    comp.dragging_label_idx = None

    # Clicking a component dot on the canvas selects its card.
    comp.components[0].g_edit.setText("0.2")
    comp.components[0].s_edit.setText("0.1")
    comp._on_component_coords_changed(0)
    comp.components[1].g_edit.setText("0.8")
    comp.components[1].s_edit.setText("0.5")
    comp._on_component_coords_changed(1)
    comp._select_component_item(0)
    assert comp._selected_component is comp.components[0]
    assert comp.components[0].card_frame.property("selected")
    mock_event = MagicMock()
    mock_event.inaxes = parent.canvas_widget.axes
    with patch.object(
        comp.components[1].dot, "contains", return_value=(True, {})
    ):
        comp._on_press(mock_event)
    assert comp._selected_component is comp.components[1]
    assert comp.components[1].card_frame.property("selected")
    assert not comp.components[0].card_frame.property("selected")


# --------------------------------------------------------- fraction filters


def _components_tab(viewer, layer=None):
    """Return a plotter showing the Components tab for one analysed layer."""
    layer = layer if layer is not None else create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)
    return parent, comp_widget, layer


def _enable_fraction_filter(comp_widget, index, minimum, maximum):
    """Switch on one component's fraction filter over ``[minimum, maximum]``.

    Driven through the card's own widgets, as the user would.
    """
    card = comp_widget.filter_list._cards[index]
    card.min_edit.setText(str(minimum))
    card.max_edit.setText(str(maximum))
    card.min_edit.editingFinished.emit()
    card.enabled_check.setChecked(True)
    return comp_widget.filter_list._cards[index]


def test_component_label_map_paints_only_the_kept_pixels():
    """A component's labels layer is its own range, not the leftovers."""
    keep = np.array([[True, False], [True, True]])
    measurable = np.array([[True, True], [False, True]])
    labels = component_label_map(2, keep, measurable)
    assert labels.dtype == np.uint16
    np.testing.assert_array_equal(labels, [[3, 0], [0, 3]])


def test_dominant_component_label_map_names_the_biggest_fraction():
    """Each surviving pixel is labelled after the component it holds most of."""
    fractions = {
        0: np.array([[0.8, 0.2], [np.nan, 0.4]]),
        1: np.array([[0.2, 0.8], [np.nan, 0.6]]),
    }
    keep = {
        0: np.array([[True, True], [True, True]]),
        1: np.array([[True, True], [True, False]]),
    }
    measurable = np.ones((2, 2), dtype=bool)
    labels = dominant_component_label_map(fractions, keep, measurable)
    # (1, 1) is kept by component 1's range but not by component 2's, so no
    # phasor coordinates are left there to name anything after.
    np.testing.assert_array_equal(labels, [[1, 2], [0, 0]])


def test_dominant_component_label_map_with_nothing_to_label():
    """No components, or nothing kept, is an empty labels layer."""
    measurable = np.ones((2, 2), dtype=bool)
    empty = dominant_component_label_map({}, {}, measurable)
    np.testing.assert_array_equal(empty, np.zeros((2, 2)))
    assert dominant_component_label_map({0: None}, {}, measurable).max() == 0

    nothing_kept = dominant_component_label_map(
        {0: np.ones((2, 2))},
        {0: np.zeros((2, 2), dtype=bool)},
        measurable,
    )
    np.testing.assert_array_equal(nothing_kept, np.zeros((2, 2)))


def test_fraction_filter_cards_follow_the_components(make_viewer_model):
    """One card per component that the current analysis has a fraction for."""
    viewer = make_viewer_model()
    parent, comp_widget, _layer = _components_tab(viewer)

    # Nothing placed yet: nothing to filter on, and the cards say why.
    assert comp_widget._filterable_components() == []
    assert "at least two components" in (
        comp_widget._filter_enable_blocked_reason()
    )

    _setup_linear_projection(comp_widget)
    assert comp_widget.analysis_type == "Linear Projection"
    assert sorted(comp_widget.filter_list._cards) == [0, 1]
    assert comp_widget._filter_enable_blocked_reason() is None

    # A third component makes it a fit, which has a fraction for each one.
    comp_widget._add_component()
    comp_widget.components[2].g_edit.setText("0.5")
    comp_widget.components[2].s_edit.setText("0.45")
    comp_widget._on_component_coords_changed(2)
    assert comp_widget.analysis_type == "Component Fit"
    assert sorted(comp_widget.filter_list._cards) == [0, 1, 2]

    # Back to a projection, which only ever has two fractions.
    comp_widget._remove_component(2)
    assert comp_widget.analysis_type == "Linear Projection"
    assert sorted(comp_widget.filter_list._cards) == [0, 1]


def test_a_fraction_filter_blanks_pixels_and_reports_its_share(
    make_viewer_model,
):
    """A filtered pixel loses its phasor coordinates, as a threshold does,
    and the statistics say which range was counted and how much was kept."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)
    table = parent.components_statistics_dock_widget.layer_stats_table

    # The table says which pixels it counted, not just how many.
    assert comp_widget._series_range_labels() == {}
    _enable_fraction_filter(comp_widget, 0, 0.2, 0.8)
    assert comp_widget._series_range_labels() == {
        "Component 1": "in 0.2 – 0.8"
    }
    headers = [
        table.horizontalHeaderItem(col).text()
        for col in range(table.columnCount())
    ]
    assert "Component 1 Pixels in 0.2 – 0.8" in headers
    assert "Component 1 % in 0.2 – 0.8" in headers

    # Excluding a range says so rather than reading as keeping it.
    card = comp_widget.filter_list._cards[0]
    card.mode_combobox.setCurrentIndex(1)
    assert comp_widget._series_range_labels() == {
        "Component 1": "outside 0.2 – 0.8"
    }

    # A filter that is switched off is not a range the counts were taken over.
    card = comp_widget.filter_list._cards[0]
    card.enabled_check.setChecked(False)
    assert comp_widget._series_range_labels() == {}
    with patch.object(comp_widget, "_primary_filter_layer", return_value=None):
        assert comp_widget._series_range_labels() == {}
    comp_widget.filter_list._cards[0].mode_combobox.setCurrentIndex(0)

    # What the filters are measured against: the threshold, the median filter
    # and the mask applied, the criteria not.
    card = _enable_fraction_filter(comp_widget, 0, 0.0, 0.4)
    measurable = comp_widget._reference_pixel_count(layer)
    assert measurable > 0

    stored = get_filters(layer)
    assert [f['metric'] for f in stored] == [COMPONENT_FRACTION]
    assert stored[0]['params']['component_index'] == 0
    assert stored[0]['params']['analysis_type'] == "Linear Projection"

    kept = int(np.isfinite(layer.data).sum())
    assert kept < measurable
    assert np.isnan(layer.metadata['G'][0][np.isnan(layer.data)]).all()
    assert "keeps" in card.stat_label.text()
    assert "1 of 1 on" in comp_widget.filter_list.summary_label.text()

    # Switching it off brings back exactly the pixels it had hidden, which
    # is the whole of the baseline again.
    comp_widget.filter_list._cards[0].enabled_check.setChecked(False)
    assert int(np.isfinite(layer.data).sum()) == measurable

    # The table says how much of the layer the filters left behind.
    baseline = comp_widget._reference_pixel_count(layer)
    assert baseline == int(np.isfinite(layer.data).sum())
    _enable_fraction_filter(comp_widget, 0, 0.0, 0.5)
    columns = StatisticsTableWidget.COLUMNS
    pixels = int(table.item(0, columns.index("Pixels")).text())
    share = table.item(0, columns.index("% Pixels")).text()
    assert pixels == int(np.isfinite(layer.data).sum())
    assert share == f"{100.0 * pixels / baseline:.1f}%"

    # The denominator is cached, and forgotten when the layer changes.
    assert comp_widget._reference_pixel_counts[layer.name] == baseline
    assert comp_widget._reference_pixel_count(layer) == baseline
    comp_widget._invalidate_pixel_counts()
    assert comp_widget._reference_pixel_counts == {}


def test_fraction_filters_combine_with_the_other_tabs(make_viewer_model):
    """A pixel survives only when every criterion, anywhere, keeps it."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)
    _enable_fraction_filter(comp_widget, 0, 0.0, 0.9)
    after_fraction = int(np.isfinite(layer.data).sum())

    mapping_tab = parent.phasor_mapping_tab
    mapping_tab.filter_list.set_current_metric(MODULATION)
    mapping_tab._sync_filter_ui()
    mapping_tab.filter_list.add_filter(new_filter(MODULATION, 0.0, 0.5))

    metrics = {f['metric'] for f in get_filters(layer)}
    assert metrics == {COMPONENT_FRACTION, MODULATION}
    assert int(np.isfinite(layer.data).sum()) <= after_fraction

    # The fraction criterion is edited where its component is defined, so
    # the Phasor Mapping tab lists only its own -- without disturbing it.
    assert [
        card.entry['metric']
        for card in mapping_tab.filter_list._cards.values()
    ] == [MODULATION]

    # The Components tab still shows its own criterion after the other tab
    # rewrote the shared stack.
    assert len(comp_widget.filter_list.filters()) == 1


def test_removing_a_component_takes_its_filter_with_it(make_viewer_model):
    """A criterion with no card left would hide pixels nothing accounts for."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    comp_widget.analysis_type_combo.setCurrentText("Component Fit")
    comp_widget._add_component()
    for idx, (g, s) in enumerate(
        [("0.2", "0.1"), ("0.5", "0.45"), ("0.8", "0.3")]
    ):
        comp_widget.components[idx].g_edit.setText(g)
        comp_widget.components[idx].s_edit.setText(s)
        comp_widget._on_component_coords_changed(idx)
    comp_widget._run_analysis()

    _enable_fraction_filter(comp_widget, 2, 0.1, 0.9)
    assert [
        f['params']['component_index']
        for f in get_filters(layer)
        if f['metric'] == COMPONENT_FRACTION
    ] == [2]

    comp_widget._remove_component(2)

    assert [
        f['params']['component_index']
        for f in get_filters(layer)
        if f['metric'] == COMPONENT_FRACTION
    ] == []
    assert sorted(comp_widget.filter_list._cards) == [0, 1]


def test_component_labels_layers(make_viewer_model):
    """Labels layers paint each component's kept pixels in its own colour,
    or name the dominant component, and follow filters and renames."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)

    def labels_layers():
        return [lyr for lyr in viewer.layers if isinstance(lyr, Labels)]

    # One layer per component, in that component's own colour.
    _enable_fraction_filter(comp_widget, 0, 0.0, 0.5)
    comp_widget.create_labels_checkbox.setChecked(True)
    assert comp_widget.labels_mode_combobox.isEnabled()
    labels = labels_layers()
    assert len(labels) == 2
    names = {lyr.name for lyr in labels}
    assert names == {
        analysis_layer_name("Component 1 filtered", layer.name),
        analysis_layer_name("Component 2 filtered", layer.name),
    }
    first = viewer.layers[
        analysis_layer_name("Component 1 filtered", layer.name)
    ]
    assert set(np.unique(first.data)) <= {0, 1}
    assert first.metadata[COMPONENT_LABELS_TAG]['source_layer'] == layer.name
    assert first.metadata[COMPONENT_LABELS_TAG]['component_index'] == 0
    # Component 1's layer shows exactly the pixels its own range keeps, which
    # is what its card reports.
    assert first.data.astype(bool).sum() == int(np.isfinite(layer.data).sum())

    comp_widget.create_labels_checkbox.setChecked(False)
    assert labels_layers() == []
    comp_widget.filter_list._cards[0].enabled_check.setChecked(False)

    # The other layout is one layer naming each pixel's main component.
    comp_widget.create_labels_checkbox.setChecked(True)
    comp_widget.labels_mode_combobox.setCurrentText(LABELS_DOMINANT)
    labels = labels_layers()
    assert len(labels) == 1
    combined = labels[0]
    assert combined.name == analysis_layer_name(
        "Dominant component", layer.name
    )
    assert set(np.unique(combined.data)) <= {0, 1, 2}
    # Every measurable pixel belongs to one of the two components.
    assert combined.data.astype(bool).sum() == int(
        np.isfinite(layer.data).sum()
    )
    # A colour per component, plus transparent for the unlabelled pixels.
    assert {1, 2}.issubset(combined.colormap.color_dict)

    # Switching layout replaces the layers rather than accumulating them.
    comp_widget.labels_mode_combobox.setCurrentText(LABELS_PER_COMPONENT)
    assert len(labels_layers()) == 2

    # A pixel with no phasor coordinates left cannot be labelled.
    combined_before = (
        viewer.layers[analysis_layer_name("Component 1 filtered", layer.name)]
        .data.astype(bool)
        .sum()
    )
    mapping_tab = parent.phasor_mapping_tab
    mapping_tab.filter_list.set_current_metric(MODULATION)
    mapping_tab._sync_filter_ui()
    mapping_tab.filter_list.add_filter(new_filter(MODULATION, 0.0, 0.5))
    comp_widget._update_label_layers()
    after = (
        viewer.layers[analysis_layer_name("Component 1 filtered", layer.name)]
        .data.astype(bool)
        .sum()
    )
    assert after < combined_before

    # Editing a filter refreshes the layer instead of replacing it.
    first = viewer.layers[
        analysis_layer_name("Component 1 filtered", layer.name)
    ]
    before = first.data.astype(bool).sum()
    _enable_fraction_filter(comp_widget, 0, 0.0, 0.2)
    assert (
        viewer.layers[analysis_layer_name("Component 1 filtered", layer.name)]
        is first
    )
    assert first.data.astype(bool).sum() <= before

    # A renamed component renames its filter card, the stored criterion and
    # its labels layer.
    _rename_component(comp_widget, 0, "Free NADH")
    assert (
        comp_widget.filter_list._cards[0].metric_label.text()
        == "Free NADH Filter"
    )
    assert [
        f['params']['component_name']
        for f in get_filters(layer)
        if f['metric'] == COMPONENT_FRACTION
    ] == ["Free NADH"]
    assert (
        analysis_layer_name("Free NADH filtered", layer.name) in viewer.layers
    )

    # A renamed image keeps its labels layer recognisable.
    old_name = layer.name
    comp_widget.rename_layer(old_name, "renamed")
    assert "renamed [Free NADH filtered]" in viewer.layers
    renamed = viewer.layers["renamed [Free NADH filtered]"]
    assert renamed.metadata[COMPONENT_LABELS_TAG]['source_layer'] == "renamed"
    assert ("renamed", 0) in comp_widget._component_label_layers
    assert old_name not in comp_widget._reference_pixel_counts


def test_a_component_labels_colour_can_be_picked(
    make_viewer_model,
):
    """A chosen colour reaches the card, the labels layer and the layer."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)
    comp_widget.create_labels_checkbox.setChecked(True)

    inherited = _as_hex(comp_widget._component_filter_colors()[0])
    card = comp_widget.filter_list._cards[0]
    assert card.accent_color == inherited

    comp_widget._on_component_color_changed(0, "#ff0000")

    # The card, the labels layer and the stored settings all agree.
    assert comp_widget._component_label_colors == {0: "#ff0000"}
    assert comp_widget.filter_list._cards[0].accent_color == "#ff0000"
    assert _as_hex(comp_widget._component_filter_colors()[0]) == "#ff0000"
    assert (
        mcolors.to_hex(comp_widget.components[0].dot.get_color()) == "#ff0000"
    )
    # A redraw of the plot does not put the inherited colour back.
    comp_widget._update_component_colors()
    assert (
        mcolors.to_hex(comp_widget.components[0].dot.get_color()) == "#ff0000"
    )
    painted = viewer.layers[
        analysis_layer_name("Component 1 filtered", layer.name)
    ]
    assert np.allclose(painted.colormap.color_dict[1], (1.0, 0.0, 0.0, 1.0))
    # Like a rename, the choice is a draft until the analysis runs...
    draft = comp_widget._read_component_settings(layer)['components']
    assert draft['0']['label_color'] == "#ff0000"
    # ...and the run stores it on the layer.
    comp_widget._run_analysis()
    stored = layer.metadata['settings']['component_analysis']['components']
    assert stored['0']['label_color'] == "#ff0000"
    # Only the component that was picked moves; the other keeps its own.
    assert _as_hex(comp_widget._component_filter_colors()[1]) != "#ff0000"

    # Picking the same colour again is not a change.
    assert comp_widget._on_component_color_changed(0, "#ff0000") is None


def test_the_last_colour_change_wins_between_colormap_and_card(
    make_viewer_model, monkeypatch
):
    """A colormap change and a card colour each replace the other.

    The dot, the card swatch, the filter card and the histogram's solid
    colour all show whichever was changed last.
    """
    from qtpy.QtWidgets import QDialog

    from napari_phasors._utils import HistogramSettingsDialog

    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    comp_widget._add_component()
    _setup_component_fit(
        comp_widget, (("0.2", "0.1"), ("0.8", "0.4"), ("0.5", "0.3"))
    )
    _check_histogram_components(
        comp_widget, ["Component 1", "Component 2", "Component 3"]
    )
    histogram = comp_widget.histogram_widget
    comp = comp_widget.components[0]

    # Pressing OK in the histogram settings freezes no colour.
    monkeypatch.setattr(
        HistogramSettingsDialog, 'exec', lambda self: QDialog.Accepted
    )
    histogram._open_settings_dialog()

    def shown():
        swatch = comp.color_button.styleSheet()
        return {
            "dot": mcolors.to_hex(comp.dot.get_color()),
            "swatch": swatch.split("background-color: ")[-1].rstrip(";"),
            "card": comp_widget.filter_list._cards[0].accent_color,
            "histogram": _as_hex(
                histogram._series_color(
                    "Component 1",
                    histogram._series_names().index("Component 1"),
                )
            ),
        }

    comp_widget.fraction_layers[0].colormap = "red"
    assert set(shown().values()) == {"#ff0000"}

    comp_widget._on_component_color_changed(0, "#00ff00")
    assert set(shown().values()) == {"#00ff00"}

    comp_widget.fraction_layers[0].colormap = "cyan"
    assert set(shown().values()) == {"#00ffff"}
    assert comp_widget._component_label_colors == {}
    draft = comp_widget._read_component_settings(layer)['components']
    assert 'label_color' not in draft['0']
    # The other components are left alone.
    assert _as_hex(comp_widget._component_filter_colors()[1]) == "#00ffff"
    assert _as_hex(comp_widget._component_filter_colors()[2]) == "#ffff00"

    # A card colour also replaces one picked in the histogram settings.
    histogram._series_color_overrides["Component 1"] = (0.0, 0.0, 1.0)
    comp_widget._on_component_color_changed(0, "#ff8800")
    assert set(shown().values()) == {"#ff8800"}


def test_a_card_colour_draws_the_fraction_layer_from_black_to_it(
    make_viewer_model,
):
    """Picking a card colour recolours the component's fraction layer."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    comp_widget._add_component()
    coords = (("0.2", "0.1"), ("0.8", "0.4"), ("0.5", "0.3"))
    _setup_component_fit(comp_widget, coords)
    fraction = comp_widget.fraction_layers[1]

    comp_widget._on_component_color_changed(1, "#00ff00")
    assert np.allclose(fraction.colormap.colors[0], (0, 0, 0, 1))
    assert np.allclose(fraction.colormap.colors[-1], (0, 1, 0, 1))
    # The card colour stays: it is what set the colormap.
    assert comp_widget._component_label_colors == {1: "#00ff00"}
    # The colormap is stored by its colours, since its name only exists in
    # this session.
    stored = comp_widget._read_component_settings(layer)['components']['1']
    entry = stored['gs_harmonics']['1']
    assert entry['colormap_name'] is None
    assert np.allclose(entry['colormap_colors'][-1], (0, 1, 0, 1))
    # The other components keep their colormaps.
    assert comp_widget.fraction_layers[0].colormap.name == "magenta"

    # A re-run keeps it.
    comp_widget._run_analysis()
    fraction = comp_widget.fraction_layers[1]
    assert np.allclose(fraction.colormap.colors[-1], (0, 1, 0, 1))

    # A colour picked before the layers exist colours them once made.
    comp_widget._on_component_color_changed(2, "#ff8800")
    comp_widget._run_analysis()
    fraction = comp_widget.fraction_layers[2]
    assert np.allclose(fraction.colormap.colors[0], (0, 0, 0, 1))
    assert _as_hex(fraction.colormap.colors[-1]) == "#ff8800"


def test_card_colours_set_the_linear_projection_colormap_ends(
    make_viewer_model,
):
    """Clicking the swatch asks for a colour, cancelling leaves it alone, and
    Linear Projection's one layer runs from one card colour to the other."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)
    fraction = comp_widget.comp1_fractions_layer
    jet = np.asarray(fraction.colormap.colors)
    comp = comp_widget.components[1]

    with patch.object(
        QColorDialog, "getColor", return_value=QColor("#123456")
    ):
        comp.color_button.click()
    assert comp_widget._component_label_colors == {1: "#123456"}
    assert comp_widget.filter_list._cards[1].accent_color == "#123456"

    # An invalid colour is what a cancelled dialog returns.
    with patch.object(QColorDialog, "getColor", return_value=QColor()):
        comp.color_button.click()
    assert comp_widget._component_label_colors == {1: "#123456"}

    comp_widget._on_component_color_changed(1, "#00ff00")
    colors = np.asarray(fraction.colormap.colors)
    assert np.allclose(colors[0], (0, 1, 0, 1))
    assert np.allclose(colors[-1], jet[-1])

    comp_widget._on_component_color_changed(0, "#ff0000")
    colors = np.asarray(fraction.colormap.colors)
    assert np.allclose(colors[0], (0, 1, 0, 1))
    assert np.allclose(colors[-1], (1, 0, 0, 1))
    assert comp_widget._component_label_colors == {
        0: "#ff0000",
        1: "#00ff00",
    }

    # The two stay in step on a re-run.
    comp_widget._run_analysis()
    colors = np.asarray(comp_widget.comp1_fractions_layer.colormap.colors)
    assert np.allclose(colors[0], (0, 1, 0, 1))
    assert np.allclose(colors[-1], (1, 0, 0, 1))


def _stored_display(layer, index):
    """Return how component *index*'s fraction layer is stored in *layer*."""
    components = layer.metadata['settings']['component_analysis']
    return components['components'][str(index)]['gs_harmonics']['1']


def test_a_custom_colormap_on_a_component_fit_layer_is_stored_by_its_colours(
    make_viewer_model, monkeypatch
):
    """A colormap only this session knows by name is kept as its colours."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_component_fit(comp_widget)
    ramp = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0]])

    # A built-in colormap is stored by name.
    comp_widget.fraction_layers[0].colormap = "viridis"
    assert _stored_display(layer, 0)['colormap_name'] == "viridis"
    assert _stored_display(layer, 0)['colormap_colors'] is None

    # napari registers a colormap's name as soon as a layer uses it, but
    # that name means nothing in another session.
    comp_widget.fraction_layers[1].colormap = Colormap(
        colors=ramp, name="fit ramp"
    )
    assert _stored_display(layer, 1)['colormap_name'] is None
    assert np.allclose(_stored_display(layer, 1)['colormap_colors'], ramp)

    # A layer recreated where the name is unknown gets the same colours.
    viewer.layers.remove(comp_widget.fraction_layers[1])
    monkeypatch.delitem(AVAILABLE_COLORMAPS, "fit ramp")
    comp_widget._run_analysis()
    fraction = comp_widget.fraction_layers[1]
    assert np.allclose(fraction.colormap.colors, ramp)
    assert np.allclose(_stored_display(layer, 1)['colormap_colors'], ramp)

    # The recreated colormap's name is again this session's only, so a
    # display change keeps storing the colours.
    fraction.gamma = 0.5
    assert _stored_display(layer, 1)['colormap_name'] is None
    assert np.allclose(_stored_display(layer, 1)['colormap_colors'], ramp)
    assert comp_widget.fraction_layers[0].colormap.name == "viridis"


def test_a_custom_colormap_on_a_linear_projection_is_stored_by_its_colours(
    make_viewer_model, monkeypatch
):
    """The Linear Projection layer keeps a custom colormap as its colours."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)
    ramp = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]])

    comp_widget.comp1_fractions_layer.colormap = Colormap(
        colors=ramp, name="projection ramp"
    )
    assert _stored_display(layer, 0)['colormap_name'] is None
    assert np.allclose(_stored_display(layer, 0)['colormap_colors'], ramp)

    # A re-run keeps the layer's colormap and stores it the same way.
    comp_widget._run_analysis()
    fraction = comp_widget.comp1_fractions_layer
    assert fraction.colormap.name == "projection ramp"
    assert _stored_display(layer, 0)['colormap_name'] is None
    assert np.allclose(_stored_display(layer, 0)['colormap_colors'], ramp)

    # A layer recreated where the name is unknown gets the same colours.
    viewer.layers.remove(fraction)
    monkeypatch.delitem(AVAILABLE_COLORMAPS, "projection ramp")
    comp_widget._run_analysis()
    fraction = comp_widget.comp1_fractions_layer
    assert np.allclose(fraction.colormap.colors, ramp)
    assert np.allclose(comp_widget.fractions_colormap, ramp)


@pytest.mark.parametrize(
    ("analysis_type", "index", "default"),
    [("Linear Projection", 0, "jet"), ("Component Fit", 1, "cyan")],
)
def test_a_colormap_name_this_session_lacks_falls_back_to_the_default(
    make_viewer_model, analysis_type, index, default
):
    """A fraction layer stored under an unknown colormap name still loads.

    Older versions stored some custom colormaps by a name napari only knew
    in the session that used it, with no colours to rebuild it from.
    """
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    components = {
        str(i): {"name": f"C{i}", "gs_harmonics": {"1": {"g": g, "s": s}}}
        for i, (g, s) in enumerate(((0.2, 0.1), (0.8, 0.5)))
    }
    components[str(index)]["gs_harmonics"]["1"].update(
        colormap_name="lost ramp",
        colormap_colors=None,
        contrast_limits=[0.1, 0.9],
        analysis_type=analysis_type,
    )
    layer.metadata["settings"]["component_analysis"] = {
        "analysis_type": analysis_type,
        "last_analysis_harmonic": 1,
        "components": components,
    }

    def fraction():
        if analysis_type == "Linear Projection":
            return comp.comp1_fractions_layer
        return comp.fraction_layers[index]

    # Running the analysis on the reopened layer shows the default colormap
    # with the stored contrast limits, says why, and stores the default in
    # place of the unknown name.
    comp._restore_components_ui_only_from_metadata()
    with patch("napari_phasors.components_tab.show_warning") as warn:
        comp._run_analysis()
    assert fraction().colormap.name == default
    assert tuple(fraction().contrast_limits) == pytest.approx((0.1, 0.9))
    warn.assert_called_once()
    assert "lost ramp" in warn.call_args[0][0]
    stored = _stored_display(layer, index)
    assert stored['colormap_name'] == default
    assert stored['colormap_colors'] is None

    # So a layer made again from the metadata has nothing to warn about.
    viewer.layers.remove(fraction())
    with patch("napari_phasors.components_tab.show_warning") as warn:
        comp._run_analysis()
    assert fraction().colormap.name == default
    warn.assert_not_called()


def test_a_picked_colour_stays_with_its_component_and_is_restored(
    make_viewer_model,
):
    """Removing a component shifts the colours of the ones below it."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    comp_widget._add_component()
    _setup_linear_projection(comp_widget)
    comp_widget._on_component_color_changed(1, "#00ff00")
    assert comp_widget._component_label_colors == {1: "#00ff00"}

    comp_widget._remove_component(0)
    # The component that was second is now first, and kept its colour.
    assert comp_widget._component_label_colors == {0: "#00ff00"}

    # A colour stored with the component's settings comes back with them.
    comp_widget._component_label_colors = {}
    settings = comp_widget._edit_component_settings(layer)
    settings['components']['0']['label_color'] = "#0000ff"
    comp_widget._component_settings_edited(layer)
    comp_widget._restore_components_for_harmonic(
        settings.get('last_analysis_harmonic', 1)
    )
    assert comp_widget._component_label_colors == {0: "#0000ff"}

    # Absolute concentration: a map shares its look across images and gets
    # it back when re-made.
    comp = comp_widget
    other = create_image_layer_with_phasors()
    other.name = "b"
    viewer.add_layer(other)
    _concentration_ready(viewer, layer, parent, comp, samples=(other,))
    viewer.add_labels(np.zeros((4, 4), dtype=np.uint8), name="mask")
    _enable_second_component(comp)
    comp._run_analysis()
    total_a = viewer.layers[_cname(layer, "Total")]
    total_b = viewer.layers[_cname(other, "Total")]
    total_a.colormap = "magma"
    total_a.gamma = 0.5
    total_a.contrast_limits = (0.0, 0.5)
    assert total_b.colormap.name == "magma"
    assert total_b.gamma == 0.5
    assert tuple(total_b.contrast_limits) == pytest.approx((0.0, 0.5))
    for sample in (layer, other):
        display = _stored_concentration(sample)['total_display']
        assert display['colormap_name'] == "magma"
        assert display['gamma'] == 0.5
        assert display['contrast_limits'] == pytest.approx([0.0, 0.5])
    first_a = viewer.layers[_cname(layer, "Component 1")]
    first_a.colormap = "inferno"
    assert viewer.layers[_cname(other, "Component 1")].colormap.name == (
        "inferno"
    )
    entry = layer.metadata['settings']['component_analysis']['components'][
        '0'
    ]['gs_harmonics']['1']
    assert entry['colormap_name'] == "inferno"
    assert entry['analysis_type'] == ABSOLUTE_CONCENTRATION
    viewer.layers.remove(total_a)
    viewer.layers.remove(first_a)
    comp._run_analysis()
    total_a = viewer.layers[_cname(layer, "Total")]
    assert total_a.colormap.name == "magma"
    assert total_a.gamma == 0.5
    assert tuple(total_a.contrast_limits) == pytest.approx((0.0, 0.5))
    assert (
        viewer.layers[_cname(layer, "Component 1")].colormap.name == "inferno"
    )
    # A total shown in the histogram follows its colormap there too.
    comp._histogram_components = [TOTAL_CONCENTRATION]
    comp._on_histogram_component_changed()
    total_a.colormap = "plasma"
    np.testing.assert_allclose(
        comp.histogram_widget.colormap_colors, total_a.colormap.colors
    )

    # A custom colormap is stored by its colours, and a map re-made from it
    # gets those colours back.
    _concentration_ready(viewer, layer, parent, comp)
    comp._run_analysis()
    custom = viewer.layers[_cname(layer, "Component 1")]
    custom.colormap = Colormap(
        colors=np.array([[0, 0, 0, 1], [1, 0.5, 0, 1]], dtype=float),
        name="my ramp",
    )
    entry = comp._layer_display_entry(custom)
    assert entry['colormap_name'] is None
    assert len(entry['colormap_colors']) == 2
    viewer.layers.remove(custom)
    comp._run_analysis()
    remade = viewer.layers[_cname(layer, "Component 1")]
    np.testing.assert_allclose(remade.colormap.colors[-1], [1, 0.5, 0, 1])


def test_derived_arrays_are_measured_once_per_edit(make_viewer_model):
    """The baseline and the fractions are derived once, not once per card,
    and the scratch pad expires with the interaction."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)

    comp_widget._expire_derived_arrays()
    first = comp_widget._baseline_for(layer)
    assert comp_widget._baseline_for(layer) is first
    # Everything measured on those arrays shares one memo, so the fit behind
    # a fraction runs once however many cards ask for it.
    context = comp_widget._context_for(layer)
    assert comp_widget._context_for(layer) is context
    maps = comp_widget._component_fraction_maps(layer)
    assert set(maps) == {0, 1}
    with patch("napari_phasors.components_tab.compute_metric") as recompute:
        assert set(comp_widget._component_fraction_maps(layer)) == {0, 1}
    recompute.assert_not_called()

    # The maps handed out are copies: a caller cannot empty the cache.
    comp_widget._component_fraction_maps(layer).clear()
    assert set(comp_widget._component_fraction_maps(layer)) == {0, 1}

    # They are a scratch pad for one edit, not a memory of the layer.
    assert comp_widget._baseline_cache
    assert comp_widget._fraction_map_cache
    # Adding and removing layers pumps the event loop, so the expiry can
    # land in the middle of the edit it was meant to speed up: it waits.
    comp_widget._applying_mapping_filter = True
    comp_widget._expire_derived_arrays()
    assert comp_widget._baseline_cache
    comp_widget._applying_mapping_filter = False
    comp_widget._expire_derived_arrays()
    assert comp_widget._baseline_cache == {}
    assert comp_widget._fraction_map_cache == {}
    assert comp_widget._metric_context_cache == {}
    assert comp_widget._derived_cache_timer.isActive() is False

    # Replacing the fraction layers must not redraw the histogram each time.
    comp_widget._applying_mapping_filter = True
    with patch.object(ComponentsWidget, 'update_component_histogram') as draw:
        comp_widget.on_layer_selection_changed()
        comp_widget.on_layer_selection_changed()
    draw.assert_not_called()
    assert comp_widget._deferred_selection_refresh is True
    comp_widget._applying_mapping_filter = False
    with patch.object(ComponentsWidget, 'update_component_histogram') as draw:
        comp_widget.on_layer_selection_changed()
    assert draw.call_count == 1
    assert comp_widget._deferred_selection_refresh is False

    # An edit answers the postponed refresh itself.
    comp_widget._deferred_selection_refresh = False
    _enable_fraction_filter(comp_widget, 0, 0.0, 0.5)
    assert comp_widget._deferred_selection_refresh is False


def test_a_measurement_is_never_served_for_stale_arrays(make_viewer_model):
    """A changed threshold, components or mask re-derives everything, and
    the shared baseline is never handed to the layer itself."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)

    before = comp_widget._baseline_for(layer)
    maps_before = comp_widget._component_fraction_maps(layer)
    layer.metadata['settings']['threshold'] = 1e9
    after = comp_widget._baseline_for(layer)
    assert after is not before
    assert np.isfinite(after[0]).sum() < np.isfinite(before[0]).sum()

    # The arrays the fractions were measured on are gone, so they are too.
    del layer.metadata['settings']['threshold']
    comp_widget.components[0].g_edit.setText("0.35")
    comp_widget._on_component_coords_changed(0)
    maps_after = comp_widget._component_fraction_maps(layer)
    assert not np.allclose(maps_after[0], maps_before[0], equal_nan=True)

    # Switched off: nothing is masked, so the rebuild has no new array to
    # write and would otherwise hand the layer the cached baseline itself.
    comp_widget._run_analysis()
    _enable_fraction_filter(comp_widget, 0, 0.0, 1.0)
    comp_widget.filter_list._cards[0].enabled_check.setChecked(False)
    mean, real, imag = comp_widget._baseline_for(layer)
    assert layer.data is not mean
    assert layer.metadata['G'] is not real
    assert layer.metadata['S'] is not imag

    # The baseline outlives one edit now, so a mask has to invalidate it.
    unmasked = comp_widget._baseline_for(layer)
    assert comp_widget._baseline_for(layer) is unmasked
    mask = np.zeros(layer.data.shape, dtype=int)
    mask[..., :1] = 1
    layer.metadata['mask'] = mask
    masked = comp_widget._baseline_for(layer)
    assert masked is not unmasked
    assert np.isfinite(masked[0]).sum() < np.isfinite(unmasked[0]).sum()

    # Selecting labels within the mask, and inverting it, are mask edits too.
    layer.metadata['mask_invert'] = True
    inverted = comp_widget._baseline_for(layer)
    assert inverted is not masked
    assert np.isfinite(inverted[0]).sum() != np.isfinite(masked[0]).sum()
    layer.metadata['mask_labels'] = [1]
    assert comp_widget._baseline_for(layer) is not inverted

    # The layer's arrays are its own, whoever measured them first.
    layer.data[:] = np.nan
    assert np.isfinite(comp_widget._baseline_for(layer)[0]).any()


def test_component_filter_helpers(make_viewer_model):
    """The filter section's helpers before and after an analysis, and every
    input they have to survive."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)

    # A refresh must not raise on a layer whose phasor data is unusable.
    broken = Image(
        np.random.random((4, 4)),
        name="broken",
        metadata={"G": np.array([1]), "S": np.array([1]), "G_original": 1},
    )
    assert comp_widget._baseline_for(broken) == (None, None, None)
    assert comp_widget._measurable_mask(broken) is None
    assert comp_widget._reference_pixel_count(broken) is None
    assert comp_widget._component_fraction_maps(broken) == {}

    # Every entry point is safe before a layer is selected.
    comp_widget.parent_widget = None
    assert comp_widget._filter_layers() == []
    assert comp_widget._primary_filter_layer() is None
    assert comp_widget._layer_filter_params(None) == {}
    assert comp_widget._component_filter_colors() == {}
    comp_widget._apply_filter_stack()
    comp_widget._sync_filter_ui()
    assert comp_widget.filter_list.filters() == []
    comp_widget.parent_widget = parent
    # A parent that cannot answer is the same as no layers at all.
    with patch.object(
        parent, "get_selected_layers", side_effect=RuntimeError("gone")
    ):
        assert comp_widget._filter_layers() == []
        comp_widget._apply_filter_stack()

    # Nothing placed: the identifying keys only, so the card stays unusable.
    bare = comp_widget._component_filter_params(0)
    assert bare == {
        'component_index': 0,
        'component_name': "Component 1",
        'analysis_type': "Linear Projection",
    }
    # Persisting the names is a no-op when there is nothing to rename.
    comp_widget._persist_component_filter_names()
    assert get_filters(layer) == []

    _setup_linear_projection(comp_widget)
    comp_widget._persist_component_filter_names()
    assert get_filters(layer) == []

    # A criterion stores what it takes to reproduce the fraction it tests.
    params = comp_widget._component_filter_params(1)
    assert params['component_real'] == [0.2, 0.8]
    assert params['component_imag'] == [0.1, 0.5]
    assert params['harmonics'] == [1]
    # A projection has no third fraction to filter on.
    assert 'component_real' not in comp_widget._component_filter_params(2)

    # The labels layers measure the same fractions the filters do.
    maps = comp_widget._component_fraction_maps(layer)
    assert sorted(maps) == [0, 1]
    np.testing.assert_allclose(maps[0] + maps[1], 1.0)
    for values in maps.values():
        assert values.shape == layer.data.shape
    # A layer with no phasor arrays has no fractions and no labels.
    empty = Image(np.ones((2, 2)), name="plain", metadata={'settings': {}})
    assert comp_widget._component_fraction_maps(empty) == {}
    assert comp_widget._component_label_maps(empty, LABELS_DOMINANT) == {}

    # Greying the plot dots must not collapse every label to one colour.
    colors = comp_widget._component_filter_colors()
    assert set(colors) == {0, 1}
    comp_widget.show_colormap_line = False
    fallback = comp_widget._component_filter_colors()
    assert len(set(map(str, fallback.values()))) == 2
    comp_widget.show_colormap_line = True

    # A component the analysis cannot place is left out of the labels.
    with patch.object(
        comp_widget, "_component_filter_params", return_value={}
    ):
        assert comp_widget._component_fraction_maps(layer) == {}
    with patch.object(
        comp_widget, "_baseline_for", return_value=(None, None, None)
    ):
        assert comp_widget._component_label_maps(layer, LABELS_DOMINANT) == {}

    # A layer with no phasor data, or none at all, contributes no total.
    assert comp_widget._reference_pixel_totals({"x": "no such layer"}) == {}
    totals = comp_widget._reference_pixel_totals({"x": layer.name})
    assert totals[layer.name] > 0
    viewer.add_layer(empty)
    with patch.object(
        comp_widget, "_baseline_for", return_value=(None, None, None)
    ):
        assert comp_widget._reference_pixel_count(empty) is None
        assert comp_widget._reference_pixel_totals({"x": "plain"}) == {}

    # Nothing to measure is not the same as a range of zero.
    with patch.object(comp_widget, "_primary_filter_layer", return_value=None):
        assert comp_widget._component_fraction_bounds(0) is None
        comp_widget._refresh_filter_bounds()
    # A component the analysis has no fraction for cannot be measured.
    assert comp_widget._component_fraction_bounds(7) is None
    with patch.object(
        comp_widget,
        "_component_fraction_maps",
        return_value={0: np.full((2, 2), np.nan)},
    ):
        assert comp_widget._component_fraction_bounds(0) is None

    # Applying the stack re-runs the analysis; that must not re-apply it.
    comp_widget._applying_mapping_filter = True
    try:
        assert comp_widget._refresh_component_filter_params() is False
    finally:
        comp_widget._applying_mapping_filter = False

    comp_widget.analysis_type_combo.setCurrentText("Component Fit")
    fit = comp_widget._component_filter_params(0)
    assert fit['analysis_type'] == "Component Fit"
    assert fit['component_real'] == [0.2, 0.8]
    assert 'component_real' not in comp_widget._component_filter_params(5)
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")

    # A projection with only one placed component has no line to project on.
    comp_widget.components[1].dot.remove()
    comp_widget.components[1].dot = None
    assert 'component_real' not in comp_widget._component_filter_params(0)


def test_colour_helpers_fall_back_rather_than_raise():
    """A missing or unusable colour must not stop a card from being drawn."""
    assert _as_hex(None) is None
    assert _as_hex("not a colour") is None
    assert _as_hex((1.0, 0.0, 0.0)) == "#ff0000"

    # A component with no colour of its own still gets a distinct one.
    colors = _label_color_dict([0, 1], {0: "red"})
    assert colors[1] == (1.0, 0.0, 0.0, 1.0)
    assert colors[2] != colors[1]
    assert colors[None] == (0.0, 0.0, 0.0, 0.0)


def test_component_filter_params_span_the_harmonics_a_fit_needs(
    make_viewer_model,
):
    """Above three components the positions are stored per harmonic."""
    viewer = make_viewer_model()
    parent, comp_widget, _layer = _components_tab(viewer)
    comp_widget.analysis_type_combo.setCurrentText("Component Fit")
    for _ in range(2):
        comp_widget._add_component()
    coords = [("0.2", "0.1"), ("0.4", "0.4"), ("0.6", "0.45"), ("0.8", "0.3")]
    for idx, (g, s) in enumerate(coords):
        comp_widget.components[idx].g_edit.setText(g)
        comp_widget.components[idx].s_edit.setText(s)
        comp_widget._on_component_coords_changed(idx)

    # Only the first harmonic has been placed, so nothing can be evaluated.
    assert 'component_real' not in comp_widget._component_filter_params(0)
    assert "every harmonic" in comp_widget._filter_enable_blocked_reason()

    parent.harmonic_spinbox.setValue(2)
    for idx, (g, s) in enumerate(coords):
        comp_widget.components[idx].g_edit.setText(g)
        comp_widget.components[idx].s_edit.setText(s)
        comp_widget._on_component_coords_changed(idx)

    params = comp_widget._component_filter_params(0)
    assert params['harmonics'] == [1, 2]
    assert len(params['component_real']) == 2
    assert len(params['component_real'][0]) == 4


def test_a_criterion_that_cannot_be_evaluated_warns_and_hides_nothing(
    make_viewer_model,
):
    """A stored criterion the data no longer supports is reported, not applied."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)
    before = int(np.isfinite(layer.data).sum())

    broken = new_filter(
        COMPONENT_FRACTION,
        0.0,
        0.5,
        params={
            'analysis_type': "Component Fit",
            'component_index': 0,
            'component_name': "Component 1",
            # Two harmonics that do not describe the same components: a
            # fit that cannot be solved, as one read back from a layer whose
            # components have since changed would be.
            'component_real': [[0.1, 0.9], [0.2]],
            'component_imag': [[0.05, 0.45], [0.1]],
            'harmonics': [1, 2],
        },
    )
    with patch("napari_phasors.components_tab.show_warning") as warn:
        comp_widget._apply_filter_stack([broken])

    assert warn.called
    assert "Component 1" in warn.call_args[0][0]
    assert int(np.isfinite(layer.data).sum()) == before


def test_a_fraction_cannot_be_filtered_before_it_is_computed(
    make_viewer_model,
):
    """A filter tests the fractions the analysis produced, so it waits for it."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(1)

    # The positions are there, but nothing has computed a fraction from them.
    assert not comp_widget._has_analysed_fractions()
    reason = comp_widget._filter_enable_blocked_reason()
    assert "Run the component analysis" in reason
    card = comp_widget.filter_list._cards[0]
    assert not card.enabled_check.isEnabled()
    assert card.enabled_check.toolTip() == reason
    assert get_filters(layer) == []

    comp_widget._run_analysis()

    assert comp_widget._has_analysed_fractions()
    assert comp_widget._filter_enable_blocked_reason() is None
    assert comp_widget.filter_list._cards[0].enabled_check.isEnabled()


def test_criteria_follow_the_analysis_they_were_made_for(make_viewer_model):
    """Moving a component, or changing how it is fitted, moves its filter."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)
    _enable_fraction_filter(comp_widget, 0, 0.0, 0.9)

    def params():
        return get_filters(layer)[0]['params']

    assert params()['component_real'] == [0.2, 0.8]
    assert params()['analysis_type'] == "Linear Projection"

    # A component that moves takes its filter with it, but only once the
    # analysis it tests has actually been re-run.
    comp_widget.components[1].g_edit.setText("0.6")
    comp_widget.components[1].s_edit.setText("0.4")
    comp_widget._on_component_coords_changed(1)
    assert params()['component_real'] == [0.2, 0.8]
    comp_widget._run_analysis()
    assert params()['component_real'] == [0.2, 0.6]

    # So does a change of method ...
    comp_widget._add_component()
    comp_widget.components[2].g_edit.setText("0.5")
    comp_widget.components[2].s_edit.setText("0.45")
    comp_widget._on_component_coords_changed(2)
    comp_widget._run_analysis()
    assert params()['analysis_type'] == "Component Fit"
    assert params()['component_real'] == [0.2, 0.6, 0.5]

    # ... and a change of harmonic.
    parent.harmonic_spinbox.setValue(2)
    for idx, (g, s) in enumerate(
        [("0.3", "0.2"), ("0.7", "0.45"), ("0.5", "0.4")]
    ):
        comp_widget.components[idx].g_edit.setText(g)
        comp_widget.components[idx].s_edit.setText(s)
        comp_widget._on_component_coords_changed(idx)
    comp_widget._run_analysis()
    assert params()['harmonics'] == [2]
    assert params()['component_real'] == [0.3, 0.7, 0.5]


def test_fraction_filter_range_spans_what_a_fit_actually_produces(
    make_viewer_model,
):
    """A component fit is unconstrained: its fractions leave ``[0, 1]``.

    Regression: the cards were fixed to 0-1, so the pixels a fit placed
    below 0 or above 1 could not be filtered on at all.
    """
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    comp_widget.analysis_type_combo.setCurrentText("Component Fit")
    comp_widget._add_component()
    for idx, (g, s) in enumerate(
        [("0.2", "0.1"), ("0.5", "0.45"), ("0.8", "0.3")]
    ):
        comp_widget.components[idx].g_edit.setText(g)
        comp_widget.components[idx].s_edit.setText(s)
        comp_widget._on_component_coords_changed(idx)
    comp_widget._run_analysis()

    maps = comp_widget._component_fraction_maps(layer)
    outside = [
        index
        for index, values in maps.items()
        if np.nanmin(values) < 0 or np.nanmax(values) > 1
    ]
    assert outside, "this fixture should produce fractions outside [0, 1]"

    for index in outside:
        values = maps[index]
        low, high = comp_widget._component_fraction_bounds(index)
        assert low == pytest.approx(float(np.nanmin(values)))
        assert high == pytest.approx(float(np.nanmax(values)))
        card = comp_widget.filter_list._cards[index]
        assert card.range_slider.minimum() / card.scale <= low
        assert card.range_slider.maximum() / card.scale >= high

    # And a range outside [0, 1] really does filter.
    index = outside[0]
    values = maps[index]
    _enable_fraction_filter(comp_widget, index, float(np.nanmin(values)), 0.0)
    stored = get_filters(layer)[0]
    assert stored['min'] < 0
    assert int(np.isfinite(layer.data).sum()) < layer.data.size


def test_a_nested_filter_apply_runs_once_after_the_first(make_viewer_model):
    """The second edit wins, and neither rebuild runs inside the other."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)
    entry = dict(comp_widget.filter_list._entries[0], enabled=True)

    seen = []
    body = ComponentsWidget._apply_filter_stack.__wrapped__

    def recording_apply(self, filters=None, layers=None):
        seen.append([f['max'] for f in filters])
        if len(seen) == 1:
            # What a queued editingFinished does once the apply, which
            # re-runs the analysis and rewrites layers, pumps the event loop.
            comp_widget._apply_filter_stack([dict(entry, max=0.25)])
        return body(self, filters, layers)

    with patch.object(
        ComponentsWidget,
        "_apply_filter_stack",
        serialize_filter_applies(recording_apply),
    ):
        comp_widget._apply_filter_stack([dict(entry, max=0.75)])

    # The interrupting edit ran once, after the first, not inside it.
    assert seen == [[0.75], [0.25]]
    assert get_filters(layer)[0]['max'] == pytest.approx(0.25)


def test_the_share_is_of_the_pixels_that_were_still_valid(make_viewer_model):
    """Not of the frame: a pixel another filter removed was never a candidate."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)
    without_others = comp_widget._reference_pixel_count(layer)
    assert without_others == int(np.isfinite(layer.data).sum())

    # A criterion owned by another tab shrinks what a fraction is a share of.
    mapping_tab = parent.phasor_mapping_tab
    mapping_tab.filter_list.set_current_metric(MODULATION)
    mapping_tab._sync_filter_ui()
    mapping_tab.filter_list.add_filter(new_filter(MODULATION, 0.0, 0.6))

    comp_widget._invalidate_pixel_counts()
    with_others = comp_widget._reference_pixel_count(layer)
    assert with_others < without_others
    assert with_others == int(comp_widget._measurable_mask(layer).sum())


def test_the_share_is_measured_when_the_analysis_ran(make_viewer_model):
    """A denominator that moved on ahead would read as more than 100%."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)
    _enable_fraction_filter(comp_widget, 0, 0.0, 0.9)
    pinned = comp_widget._reference_pixel_count(layer)

    # Another tab narrows the data, but this tab's fractions still come from
    # the run before it, so the number they are a share of does too.
    mapping_tab = parent.phasor_mapping_tab
    mapping_tab.filter_list.set_current_metric(MODULATION)
    mapping_tab._sync_filter_ui()
    mapping_tab.filter_list.add_filter(new_filter(MODULATION, 0.0, 0.6))
    assert comp_widget._reference_pixel_count(layer) == pinned

    # Re-running re-measures both together.
    comp_widget._run_analysis()
    assert comp_widget._reference_pixel_count(layer) < pinned


def test_every_criterion_follows_the_same_analysis_run(make_viewer_model):
    """One run, one method: the criteria never split across two of them."""
    viewer = make_viewer_model()
    parent, comp_widget, layer = _components_tab(viewer)
    _setup_linear_projection(comp_widget)
    _enable_fraction_filter(comp_widget, 0, 0.2, 0.8)
    _enable_fraction_filter(comp_widget, 1, 0.2, 0.8)

    stored = get_filters(layer)
    assert {f['params']['analysis_type'] for f in stored} == {
        "Linear Projection"
    }

    # A third component forces a fit. Until it has run, the criteria keep
    # testing the fractions the projection produced.
    comp_widget._add_component()
    comp_widget.components[2].g_edit.setText("0.5")
    comp_widget.components[2].s_edit.setText("0.45")
    comp_widget._on_component_coords_changed(2)
    assert {f['params']['analysis_type'] for f in get_filters(layer)} == {
        "Linear Projection"
    }
    # And the fraction the fit has not computed yet cannot be filtered on.
    assert comp_widget._filter_enable_blocked_reason() is not None

    # The run moves all of them at once, so they are never mixed.
    comp_widget._run_analysis()
    stored = get_filters(layer)
    assert len(stored) == 2
    assert {f['params']['analysis_type'] for f in stored} == {"Component Fit"}
    assert len({tuple(f['params']['harmonics']) for f in stored}) == 1


# --------------------------------------------------------------------------
# Absolute concentration helpers. The scenarios are sections of the tests
# above, which keeps one viewer and plotter for each of them.
# --------------------------------------------------------------------------

#: Positions of the two components the concentration sections place.
_CONC_COMPONENTS = ((0.9, 0.25), (0.25, 0.43))


def _cname(layer, component):
    """Name of *layer*'s concentration map of *component* (or the total)."""
    return analysis_layer_name(
        concentration_analysis_label(component), layer.name
    )


def _enable_second_component(comp, ratio="2"):
    """Also compute the second component, with brightness *ratio*."""
    comp.second_component_checkbox.setChecked(True)
    comp.brightness_ratio_edit.setText(ratio)
    comp.brightness_ratio_edit.editingFinished.emit()


def _type_reference(comp, mean, g, s):
    """Type the reference solution's values and commit each field."""
    for edit, value in (
        (comp.reference_mean_edit, mean),
        (comp.reference_g_edit, g),
        (comp.reference_s_edit, s),
    ):
        edit.setText(str(value))
        edit.editingFinished.emit()


def _concentration_maps(viewer):
    """Return ``{name: layer}`` of the concentration maps in *viewer*."""
    return {
        layer.name: layer
        for layer in viewer.layers
        if (layer.metadata.get('phasor_component_fraction') or {}).get(
            'analysis_type'
        )
        == ABSOLUTE_CONCENTRATION
    }


def _concentration_ready(
    viewer, layer, parent, comp, reference=True, samples=()
):
    """Set *comp* up for an absolute concentration of *layer* and *samples*.

    Adds the ``reference`` layer once, then clears whatever an earlier
    section left (maps, drafts, stored settings, extra components, typed
    inputs) so a section can follow another on the same widget. The two
    components are placed and named, and the reference solution is
    measured on the ``reference`` layer, or typed in when *reference* is
    false. Returns the ``reference`` layer.
    """
    if "reference" not in viewer.layers:
        solution = create_image_layer_with_phasors()
        solution.name = "reference"
        viewer.add_layer(solution)
    for old in list(_concentration_maps(viewer).values()):
        viewer.layers.remove(old)
    analysed = (layer, *samples)
    for sample in analysed:
        parent.settings_store.discard_drafts([sample])
        (sample.metadata.get('settings') or {}).pop('component_analysis', None)
    _select_layers(parent, [sample.name for sample in analysed])
    parent.tab_widget.setCurrentWidget(comp)
    while len(comp.components) > 2:
        comp._remove_component()
    comp.analysis_type_combo.setCurrentText(ABSOLUTE_CONCENTRATION)
    for index, (g, s) in enumerate(_CONC_COMPONENTS):
        _rename_component(comp, index, f"Component {index + 1}")
        comp.components[index].g_edit.setText(str(g))
        comp.components[index].s_edit.setText(str(s))
        comp._on_component_coords_changed(index)
    comp.second_component_checkbox.setChecked(False)
    comp.calibrated_component_combo.setCurrentIndex(0)
    comp.reference_concentration_edit.setText("1")
    comp.reference_concentration_edit.editingFinished.emit()
    comp.concentration_units_combo.setCurrentText("mM")
    comp.concentration_units_combo.lineEdit().editingFinished.emit()
    if reference:
        comp.reference_source_combo.setCurrentText("reference")
    else:
        comp.reference_source_combo.setCurrentText(MANUAL_REFERENCE)
        _type_reference(comp, "", "", "")
    return viewer.layers["reference"]


def _expected_concentrations(sample, reference, ratio=None, order=(0, 1)):
    """Return what the tab should compute for *sample*, straight from the
    wrapped model, with the components ordered calibrated first."""
    real, imag = harmonic_plane(sample, 1)
    return component_concentrations(
        np.asarray(sample.data),
        real,
        imag,
        [_CONC_COMPONENTS[i][0] for i in order],
        [_CONC_COMPONENTS[i][1] for i in order],
        phasor_reference_from_layer(reference, 1),
        1.0,
        ratio,
    )


def _stored_concentration(layer):
    """Return the ``concentration`` settings committed to *layer*."""
    return layer.metadata['settings']['component_analysis']['concentration']
