from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from napari.layers import Image
from phasorpy.component import phasor_component_fraction
from phasorpy.lifetime import phasor_from_lifetime
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QColorDialog

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.components_tab import (
    CenterFillSlider,
    draw_components_overlay,
    draw_fraction_histogram_overlay,
)
from napari_phasors.plotter import PlotterWidget


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


def test_components_widget_initialization_values(make_viewer_model, qtbot):
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)

    assert comp_widget.viewer is viewer
    assert comp_widget.parent_widget is parent
    assert len(comp_widget.components) == 2
    # Initial state
    assert comp_widget.component_line is None
    assert comp_widget.component_polygon is None
    assert comp_widget.comp1_fractions_layer is None
    assert comp_widget.comp2_fractions_layer is None
    assert comp_widget.fraction_layers == []
    assert comp_widget.fractions_colormap is None
    assert comp_widget.colormap_contrast_limits is None
    assert comp_widget.analysis_type == "Linear Projection"
    assert comp_widget.current_harmonic == 1
    # UI elements exist - updated for new structure
    assert comp_widget.analysis_type_combo is not None
    assert comp_widget.add_component_btn is not None
    assert comp_widget.remove_component_btn is not None
    assert comp_widget.clear_components_btn is not None
    assert comp_widget.calculate_button is not None
    # Check first component exists
    assert comp_widget.components[0] is not None
    assert comp_widget.components[0].name_edit is not None
    assert comp_widget.components[0].g_edit is not None
    assert comp_widget.components[0].s_edit is not None
    assert comp_widget.components[0].select_button is not None
    assert comp_widget.components[0].lifetime_edit is not None
    # Check second component exists
    assert comp_widget.components[1] is not None
    assert comp_widget.components[1].name_edit is not None
    assert comp_widget.components[1].g_edit is not None
    assert comp_widget.components[1].s_edit is not None
    assert comp_widget.components[1].select_button is not None
    assert comp_widget.components[1].lifetime_edit is not None
    assert not comp_widget.histogram_widget.isHidden()


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


def test_components_widget_lifetime_inputs_visibility_with_frequency(
    make_viewer_model,
    qtbot,
):
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # Lifetime widgets visible - check first component
    comp = comp_widget.components[0]
    assert not comp.ui_elements['lifetime_label'].isHidden()
    assert not comp.lifetime_edit.isHidden()

    # Enter lifetime and verify G,S updated
    comp.lifetime_edit.setText("3.0")
    comp_widget._update_component_from_lifetime(0)
    g_val = float(comp.g_edit.text())
    s_val = float(comp.s_edit.text())
    expected_g, expected_s = phasor_from_lifetime(80.0, 3.0)
    assert abs(g_val - expected_g) < 1e-3
    assert abs(s_val - expected_s) < 1e-3


def test_components_widget_component_creation_and_line(
    make_viewer_model, qtbot
):
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    # Set component coordinates
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


def test_components_widget_fraction_calculation_creates_both_layers(
    make_viewer_model,
    qtbot,
):
    """Test that fraction calculation creates both comp1 and comp2 layers."""
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
    assert comp_widget.comp2_fractions_layer is None

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


def test_components_widget_colormap_change(make_viewer_model, qtbot):
    """Test that changing colormap on comp1 layer works correctly."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    # Ensure Linear Projection is selected
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")

    # Create components and calculate fractions
    comp_widget.components[0].g_edit.setText("0.3")
    comp_widget.components[0].s_edit.setText("0.2")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.9")
    comp_widget.components[1].s_edit.setText("0.6")
    comp_widget._on_component_coords_changed(1)
    comp_widget._run_analysis()

    # Store original colormap
    orig_comp1_colors = (
        comp_widget.comp1_fractions_layer.colormap.colors.copy()
    )

    # Change colormap on comp1 layer
    comp_widget.comp1_fractions_layer.colormap = 'viridis'

    # comp1 should have new colormap
    comp1_colors = comp_widget.comp1_fractions_layer.colormap.colors

    # Should be different from original
    assert not np.allclose(orig_comp1_colors, comp1_colors)

    # comp2_fractions_layer should not exist
    assert comp_widget.comp2_fractions_layer is None


def test_components_widget_colormap_update_legacy(make_viewer_model, qtbot):
    """Legacy test updated for new dual-layer structure."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    # Ensure Linear Projection is selected
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")

    # Create components
    comp_widget.components[0].g_edit.setText("0.3")
    comp_widget.components[0].s_edit.setText("0.2")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.9")
    comp_widget.components[1].s_edit.setText("0.6")
    comp_widget._on_component_coords_changed(1)
    comp_widget._run_analysis()

    # Original colormap snapshot
    orig_colors = comp_widget.fractions_colormap.copy()

    # Change colormap on comp1 layer
    comp_widget.comp1_fractions_layer.colormap = 'viridis'

    # Ensure updated
    assert comp_widget.fractions_colormap is not None
    assert comp_widget.comp1_fractions_layer.colormap.name == 'viridis'
    # Colors changed
    assert not np.array_equal(orig_colors, comp_widget.fractions_colormap)


def test_components_widget_colormap_fallback_handling(
    make_viewer_model, qtbot
):
    """Test that colormap sync handles colormaps without colors attribute gracefully."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    # Ensure Linear Projection is selected
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")

    # Create components and calculate fractions
    comp_widget.components[0].g_edit.setText("0.5")
    comp_widget.components[0].s_edit.setText("0.3")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.6")
    comp_widget._on_component_coords_changed(1)
    comp_widget._run_analysis()

    # Use a built-in colormap name (these might not have colors attribute)
    comp_widget.comp1_fractions_layer.colormap = 'gray'

    # Should fall back to default colormaps without crashing
    assert comp_widget.comp1_fractions_layer.colormap is not None
    assert comp_widget.comp2_fractions_layer is None


def test_components_widget_visibility_toggle(make_viewer_model, qtbot):
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # Create components
    comp_widget.components[0].g_edit.setText("0.4")
    comp_widget.components[0].s_edit.setText("0.25")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.7")
    comp_widget.components[1].s_edit.setText("0.45")
    comp_widget._on_component_coords_changed(1)

    parent.tab_widget.setCurrentWidget(parent.components_tab)
    assert comp_widget.components[0].dot.get_visible() is True

    # Switch to another tab
    parent.tab_widget.setCurrentIndex(
        parent.tab_widget.indexOf(parent.fret_tab)
    )
    assert comp_widget.components[0].dot.get_visible() is False

    # Back to components
    parent.tab_widget.setCurrentWidget(parent.components_tab)
    assert comp_widget.components[0].dot.get_visible() is True


def test_components_widget_line_settings_dialog_effects(
    make_viewer_model, qtbot
):
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    # Ensure Linear Projection is selected
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")

    # Create components
    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.15")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.6")
    comp_widget.components[1].s_edit.setText("0.45")
    comp_widget._on_component_coords_changed(1)
    comp_widget._run_analysis()

    comp_widget._open_plot_settings_dialog()
    assert comp_widget.plot_dialog.isVisible()

    # Toggle settings
    comp_widget.colormap_line_checkbox.setChecked(False)
    comp_widget._on_plot_setting_changed()
    assert not comp_widget.show_colormap_line

    comp_widget.colormap_line_checkbox.setChecked(True)
    comp_widget._on_plot_setting_changed()
    assert comp_widget.show_colormap_line

    # Change offset (slider uses 3-decimal factor: 120 -> 0.120)
    comp_widget.line_offset_slider.setValue(120)
    assert abs(comp_widget.line_offset - 0.12) < 1e-6

    # Values can also be typed directly via the spinbox, which drives the slider.
    comp_widget.line_offset_spin.setValue(-0.25)
    assert abs(comp_widget.line_offset + 0.25) < 1e-6
    assert comp_widget.line_offset_slider.value() == -250

    # Change width
    comp_widget.line_width_spin.setValue(5.0)
    assert comp_widget.line_width == 5.0

    # Transparency (inverse of alpha): 0.45 transparency -> 0.55 opacity
    comp_widget.line_transparency_spin.setValue(0.45)
    assert abs(comp_widget.line_alpha - 0.55) < 1e-6


def test_components_widget_label_style_dialog(make_viewer_model, qtbot):
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    # Create a component with a name (label drawn)
    comp_widget.components[0].name_edit.setText("A")
    comp_widget.components[0].g_edit.setText("0.3")
    comp_widget.components[0].s_edit.setText("0.2")
    comp_widget._on_component_coords_changed(0)

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


def test_components_widget_add_remove_components(make_viewer_model, qtbot):
    """Test adding and removing components."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # Initially should have 2 components
    assert len(comp_widget.components) == 2
    assert comp_widget.add_component_btn.isEnabled()
    assert (
        not comp_widget.remove_component_btn.isEnabled()
    )  # Can't remove when only 2

    # Add a component
    comp_widget._add_component()
    assert len(comp_widget.components) == 3
    assert comp_widget.remove_component_btn.isEnabled()  # Now can remove

    # Remove a component
    comp_widget._remove_component()
    assert len(comp_widget.components) == 2
    assert not comp_widget.remove_component_btn.isEnabled()  # Back to minimum


def test_components_widget_analysis_type_changes(make_viewer_model, qtbot):
    """Test analysis type changes based on component count."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # With 2 components, both options available
    assert comp_widget.analysis_type_combo.count() == 2
    assert "Linear Projection" in [
        comp_widget.analysis_type_combo.itemText(i)
        for i in range(comp_widget.analysis_type_combo.count())
    ]
    assert "Component Fit" in [
        comp_widget.analysis_type_combo.itemText(i)
        for i in range(comp_widget.analysis_type_combo.count())
    ]

    # Add third component
    comp_widget._add_component()
    # Should only have Component Fit available
    assert comp_widget.analysis_type_combo.count() == 1
    assert comp_widget.analysis_type_combo.itemText(0) == "Component Fit"


def test_components_widget_multi_component_analysis(make_viewer_model, qtbot):
    """Test multi-component analysis with component fit."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

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

    # Should create multiple fraction layers
    assert len(comp_widget.fraction_layers) == 3


def test_components_widget_polygon_visualization(make_viewer_model, qtbot):
    """Test that 3+ components create a polygon instead of a line."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    # Add third component
    comp_widget._add_component()

    # Set three component coordinates
    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget._on_component_coords_changed(0)

    comp_widget.components[1].g_edit.setText("0.5")
    comp_widget.components[1].s_edit.setText("0.3")
    comp_widget._on_component_coords_changed(1)

    comp_widget.components[2].g_edit.setText("0.8")
    comp_widget.components[2].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(2)

    # Should have polygon, not line
    assert comp_widget.component_polygon is not None
    assert comp_widget.component_line is None


def test_components_widget_harmonic_storage_and_switching(
    make_viewer_model, qtbot
):
    """Test component storage across harmonic changes."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # Set components for harmonic 1
    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget.components[0].name_edit.setText("Test Component 1")
    comp_widget._on_component_coords_changed(0)

    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.5")
    comp_widget.components[1].name_edit.setText("Test Component 2")
    comp_widget._on_component_coords_changed(1)

    # Switch to harmonic 2
    comp_widget._on_harmonic_changed(2)

    # Components should be cleared in UI
    assert comp_widget.components[0].g_edit.text() == ""
    assert comp_widget.components[1].g_edit.text() == ""

    # But stored in metadata under harmonic 1
    components_settings = layer.metadata['settings']['component_analysis']
    assert '0' in components_settings['components']  # Changed from 1 to '0'
    stored_comp1 = components_settings['components'][
        '0'
    ]  # Changed from [1][0] to ['0']

    # Access the gs_harmonics for harmonic 1
    assert '1' in stored_comp1['gs_harmonics']
    harmonic_1_data = stored_comp1['gs_harmonics']['1']

    assert abs(harmonic_1_data['g'] - 0.2) < 1e-6
    assert abs(harmonic_1_data['s'] - 0.1) < 1e-6
    assert stored_comp1['name'] == "Test Component 1"

    # Switch back to harmonic 1
    comp_widget._on_harmonic_changed(1)

    # Components should be restored
    assert abs(float(comp_widget.components[0].g_edit.text()) - 0.2) < 1e-6
    assert abs(float(comp_widget.components[0].s_edit.text()) - 0.1) < 1e-6
    assert comp_widget.components[0].name_edit.text() == "Test Component 1"


def test_components_widget_clear_all_components(make_viewer_model, qtbot):
    """Test clearing all components."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(
        parent.components_tab
    )  # Make sure tab is active

    # Set components
    comp_widget.components[0].g_edit.setText("0.3")
    comp_widget.components[0].s_edit.setText("0.2")
    comp_widget.components[0].name_edit.setText("Test")
    comp_widget._on_component_coords_changed(0)

    comp_widget.components[1].g_edit.setText("0.7")
    comp_widget.components[1].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(1)

    # Verify components exist
    assert comp_widget.components[0].dot is not None
    assert comp_widget.components[1].dot is not None

    # Clear all
    comp_widget._clear_components()

    # Verify cleared
    assert comp_widget.components[0].dot is None
    assert comp_widget.components[1].dot is None
    assert comp_widget.components[0].g_edit.text() == ""
    assert comp_widget.components[0].name_edit.text() == ""
    assert comp_widget.components[1].g_edit.text() == ""
    assert comp_widget.components[1].name_edit.text() == ""


def test_components_widget_calculate_button_text_changes(
    make_viewer_model, qtbot
):
    """Test that calculate button text changes based on analysis type."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # Initially should be Linear Projection
    assert (
        comp_widget.calculate_button.text()
        == "Display Component Fraction Images"
    )

    # Change to Component Fit
    comp_widget.analysis_type_combo.setCurrentText("Component Fit")
    assert (
        comp_widget.calculate_button.text() == "Run Multi-Component Analysis"
    )

    # Change back to Linear Projection
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")
    assert (
        comp_widget.calculate_button.text()
        == "Display Component Fraction Images"
    )


def test_components_widget_component_state_initialization(
    make_viewer_model, qtbot
):
    """Test that ComponentState objects are properly initialized."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # Check first component state
    comp0 = comp_widget.components[0]
    assert comp0.idx == 0
    assert comp0.dot is None
    assert comp0.text is None
    assert comp0.label == "Component 1"
    assert comp0.text_offset == (0.02, 0.02)

    # Check second component state
    comp1 = comp_widget.components[1]
    assert comp1.idx == 1
    assert comp1.dot is None
    assert comp1.text is None
    assert comp1.label == "Component 2"
    assert comp1.text_offset == (0.02, 0.02)


def test_components_widget_ui_elements_stored(make_viewer_model, qtbot):
    """Test that ui_elements dictionary is properly stored in ComponentState."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    comp = comp_widget.components[0]
    assert hasattr(comp, 'ui_elements')
    assert 'lifetime_label' in comp.ui_elements
    assert 'comp_layout' in comp.ui_elements
    assert comp.ui_elements['lifetime_label'] is not None
    assert comp.ui_elements['comp_layout'] is not None


def test_components_widget_default_settings_structure(
    make_viewer_model, qtbot
):
    """Test the structure of default components settings."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

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


def test_components_widget_style_state_initialization(
    make_viewer_model, qtbot
):
    """Test that style state variables are properly initialized."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    assert comp_widget.label_fontsize == 10
    assert comp_widget.label_bold is False
    assert comp_widget.label_italic is False
    assert comp_widget.label_color == 'black'


def test_components_widget_line_settings_initialization(
    make_viewer_model, qtbot
):
    """Test that line settings are properly initialized."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    assert comp_widget.show_colormap_line is True
    assert comp_widget.show_component_dots is True
    assert comp_widget.line_offset == 0.0
    assert comp_widget.line_width == 3.0
    assert comp_widget.line_alpha == 1
    assert comp_widget.default_component_color == 'dimgray'


def test_components_widget_flags_initialization(make_viewer_model, qtbot):
    """Test that internal flags are properly initialized."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    assert comp_widget._updating_from_lifetime is False
    assert comp_widget._updating_settings is False
    assert comp_widget._analysis_attempted is False
    assert comp_widget.drag_events_connected is False
    assert comp_widget.dragging_component_idx is None
    assert comp_widget.dragging_label_idx is None


def test_components_widget_dialogs_initialization(make_viewer_model, qtbot):
    """Test that dialog references are initialized as None."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    assert comp_widget.plot_dialog is None
    assert comp_widget.style_dialog is None


def test_components_widget_component_colors_list(make_viewer_model, qtbot):
    """Test that component colors lists are properly initialized."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    assert len(comp_widget.component_colors) == 9
    assert comp_widget.component_colors[0] == 'magenta'
    assert comp_widget.component_colors[1] == 'cyan'

    assert len(comp_widget.component_colormap_names) == 9
    assert comp_widget.component_colormap_names[0] == 'magenta'
    assert comp_widget.component_colormap_names[1] == 'cyan'


def test_components_widget_harmonic_signal_connection(
    make_viewer_model, qtbot
):
    """Test that harmonic spinbox signal is connected."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # Change harmonic and verify method is called
    parent.harmonic_spinbox.setValue(2)
    assert comp_widget.current_harmonic == 2


def test_components_widget_scroll_area_exists(make_viewer_model, qtbot):
    """Test that a scroll area is created for the components widget."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # The layout should contain a scroll area
    assert comp_widget.layout() is not None
    assert comp_widget.layout().count() > 0


def test_components_fraction_range_updates_layer_and_is_reversible(
    make_viewer_model,
    qtbot,
):
    """Range slider should clip fraction layer data from original values."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")

    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(1)
    comp_widget._run_analysis()

    comp_name = comp_widget.components[0].name_edit.text().strip()
    if not comp_name:
        comp_name = "Component 1"
    comp_widget.histogram_component_combobox.setCurrentText(comp_name)

    fraction_layer = comp_widget.comp1_fractions_layer
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

    # Expanding the range should restore values from original data, not from
    # the already-clipped layer data.
    comp_widget._on_fraction_range_changed(0.0, 1.0)
    np.testing.assert_allclose(
        fraction_layer.data,
        np.clip(original_data, 0.0, 1.0),
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )


def test_components_second_component_histogram_inverts_fraction(
    make_viewer_model,
    qtbot,
):
    """Second component option shows 1 - first fraction with reversed colormap."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")

    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(1)
    comp_widget._run_analysis()

    name1, name2 = comp_widget._linear_projection_component_names()
    assert name1 is not None and name2 is not None

    # Both components must be selectable in the histogram combobox.
    assert comp_widget.histogram_component_combobox.findText(name1) >= 0
    assert comp_widget.histogram_component_combobox.findText(name2) >= 0

    fraction_layers_map, invert = comp_widget._resolve_histogram_component(
        name2
    )
    assert invert is True
    assert fraction_layers_map  # underlying first-component layers

    first_layer = comp_widget.comp1_fractions_layer

    # Selecting the second component displays the inverted distribution.
    comp_widget.histogram_component_combobox.setCurrentText(name2)
    comp_widget.update_component_histogram()

    displayed = comp_widget.histogram_widget._raw_valid_data
    expected = 1.0 - first_layer.data
    expected = expected[np.isfinite(expected)]
    np.testing.assert_allclose(
        np.sort(displayed),
        np.sort(expected),
        rtol=1e-6,
        atol=1e-9,
    )

    # The histogram colormap must be the reversed first-component colormap.
    np.testing.assert_allclose(
        comp_widget.histogram_widget.colormap_colors,
        np.asarray(first_layer.colormap.colors)[::-1],
    )


def test_components_second_component_fraction_range_inverts(
    make_viewer_model,
    qtbot,
):
    """Range slider on the second component clips the first-component layer."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)
    _setup_linear_projection(comp_widget)

    name1, name2 = comp_widget._linear_projection_component_names()
    comp_widget.histogram_component_combobox.setCurrentText(name2)

    fraction_layer = comp_widget.comp1_fractions_layer
    original_data = fraction_layer.data.copy()

    # Clipping the second-component fraction to [0.2, 0.6] is equivalent to
    # clipping the underlying first-component layer to [0.4, 0.8].
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

    # The histogram shows the inverted distribution with the reversed colormap.
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


def test_components_second_component_colormap_follows_first_layer(
    make_viewer_model,
    qtbot,
):
    """First-layer colormap/contrast edits update the inverted overlay."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)
    _setup_linear_projection(comp_widget)

    _name1, name2 = comp_widget._linear_projection_component_names()
    comp_widget.histogram_component_combobox.setCurrentText(name2)

    fraction_layer = comp_widget.comp1_fractions_layer

    # Changing the first component's colormap reverses it on the histogram.
    fraction_layer.colormap = "viridis"
    np.testing.assert_allclose(
        comp_widget.histogram_widget.colormap_colors,
        np.asarray(fraction_layer.colormap.colors)[::-1],
    )

    # Changing the contrast limits flips them into second-component space.
    fraction_layer.contrast_limits = [0.2, 0.7]
    np.testing.assert_allclose(
        np.asarray(comp_widget.histogram_widget.contrast_limits),
        [1.0 - 0.7, 1.0 - 0.2],
        atol=1e-9,
    )

    # With no component selected, a first-layer colormap change is a no-op for
    # the overlay (the refresh guard short-circuits).
    comp_widget.histogram_component_combobox.blockSignals(True)
    comp_widget.histogram_component_combobox.clear()
    comp_widget.histogram_component_combobox.blockSignals(False)
    previous_colors = comp_widget.histogram_widget.colormap_colors
    fraction_layer.colormap = "magma"
    assert comp_widget.histogram_widget.colormap_colors is previous_colors


def test_components_histogram_hidden_without_fraction_data(
    make_viewer_model,
    qtbot,
):
    """Histogram is hidden when the selection resolves to no data."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)
    _setup_linear_projection(comp_widget)

    hw = comp_widget.histogram_widget
    name1, _ = comp_widget._linear_projection_component_names()
    comp_widget.histogram_component_combobox.setCurrentText(name1)

    # Removing the only fraction layer leaves the selection unresolvable.
    viewer.layers.remove(comp_widget.comp1_fractions_layer)
    with patch.object(hw, "hide") as mock_hide:
        comp_widget.update_component_histogram()
        mock_hide.assert_called_once()

    # An empty selection is also hidden.
    comp_widget.histogram_component_combobox.blockSignals(True)
    comp_widget.histogram_component_combobox.clear()
    comp_widget.histogram_component_combobox.blockSignals(False)
    with patch.object(hw, "hide") as mock_hide:
        comp_widget.update_component_histogram()
        mock_hide.assert_called_once()


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

    comp_widget.histogram_component_combobox.setCurrentText(name1)
    comp_widget.update_component_histogram()

    # All selected layers are pooled into a single merged histogram (rather
    # than a per-layer mean +/- SD that looks like just one layer).
    assert set(comp_widget.histogram_widget._datasets.keys()) == {"Layer"}
    assert comp_widget.histogram_widget._show_sd is False

    # The range slider clips every layer and keeps the merged histogram.
    comp_widget._on_fraction_range_changed(0.2, 0.8)
    assert set(comp_widget.histogram_widget._datasets.keys()) == {"Layer"}

    # Early returns: an empty and an unresolved selection are both no-ops.
    comp_widget.histogram_component_combobox.blockSignals(True)
    comp_widget.histogram_component_combobox.clear()
    comp_widget.histogram_component_combobox.blockSignals(False)
    comp_widget._on_fraction_range_changed(0.1, 0.9)

    comp_widget.histogram_component_combobox.blockSignals(True)
    comp_widget.histogram_component_combobox.addItem("Ghost component")
    comp_widget.histogram_component_combobox.setCurrentText("Ghost component")
    comp_widget.histogram_component_combobox.blockSignals(False)
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
    comp_widget.histogram_component_combobox.setCurrentText(name1)

    # Changing gamma on one layer propagates to the sibling layer, the stored
    # gradient gamma, and the histogram widget.
    first_component_layers[0].gamma = 0.5

    assert first_component_layers[1].gamma == 0.5
    assert comp_widget.fractions_gamma == 0.5
    assert comp_widget.histogram_widget.gamma == 0.5


def test_components_stats_combobox_mirrors_histogram_combobox(
    make_viewer_model,
    qtbot,
):
    """Statistics-dock component selector stays in sync with the histogram one."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")

    comp_widget.components[0].g_edit.setText("0.2")
    comp_widget.components[0].s_edit.setText("0.1")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.8")
    comp_widget.components[1].s_edit.setText("0.5")
    comp_widget._on_component_coords_changed(1)
    comp_widget._run_analysis()

    hist_combo = comp_widget.histogram_component_combobox
    stats_combo = comp_widget.stats_component_combobox

    # Both comboboxes expose the same entries.
    hist_items = [hist_combo.itemText(i) for i in range(hist_combo.count())]
    stats_items = [stats_combo.itemText(i) for i in range(stats_combo.count())]
    assert hist_items == stats_items
    assert len(hist_items) >= 2

    # Changing the statistics combobox updates the histogram combobox.
    stats_combo.setCurrentText(hist_items[1])
    assert hist_combo.currentText() == hist_items[1]

    # Changing the histogram combobox updates the statistics combobox.
    hist_combo.setCurrentText(hist_items[0])
    assert stats_combo.currentText() == hist_items[0]


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


def test_components_selection_calculates_lifetime(make_viewer_model, qtbot):
    """Test that selecting/clicking or dragging a component calculates and updates the lifetime."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)

    comp = comp_widget.components[0]

    # 1. Test selection via _handle_component_selection_event
    class DummyEvent:
        inaxes = True
        xdata = 0.5
        ydata = 0.5

    event = DummyEvent()
    comp_widget._select_component(0)
    comp_widget._handle_component_selection_event(event)

    from phasorpy.lifetime import phasor_to_normal_lifetime

    expected_lifetime = phasor_to_normal_lifetime(0.5, 0.5, frequency=80.0)

    assert comp.lifetime_edit.text() != ""
    assert abs(float(comp.lifetime_edit.text()) - expected_lifetime) < 1e-3

    # 2. Test dragging via _on_motion
    comp_widget.dragging_component_idx = 0
    if comp.dot is None:
        comp_widget._create_component_at_coordinates(0, 0.5, 0.5)

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


def test_components_auto_placement_calculates_lifetime(
    make_viewer_model, qtbot
):
    """Test that auto-placing the second component calculates and updates the lifetime edit."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)

    # Set first component's position
    comp_widget.components[0].g_edit.setText("0.3")
    comp_widget.components[0].s_edit.setText("0.2")
    comp_widget._on_component_coords_changed(0)

    # Trigger auto place for second component
    comp_widget._auto_place_second_component()

    # The lifetime edit for Component 2 should not be empty and should have a valid lifetime
    comp2 = comp_widget.components[1]
    assert comp2.lifetime_edit.text() != ""
    assert float(comp2.lifetime_edit.text()) > 0


def test_components_dropdown_menu_and_cursor_selection(
    make_viewer_model, qtbot
):
    """Test that the dropdown menu is populated correctly and selection from a cursor sets coordinates and lifetime."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)

    # 1. Setup mock cursors in Selection tab
    selection_tab = parent.selection_tab

    # A-C. Cursor-selection cursors (circular / polar / elliptical)
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

    # D. GMM Cluster
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

    # 2. Re-populate/verify the select menu for Component 1 (idx=0)
    from qtpy.QtWidgets import QMenu

    menu = QMenu()
    comp_widget._populate_select_menu(0, menu)

    # Verify menu has the right actions/submenus
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

    # 3. Trigger selecting from cursor center (each type)
    comp_widget._set_component_coords_from_menu(0, 0.4, 0.3)
    comp1 = comp_widget.components[0]
    assert comp1.g_edit.text() == "0.400"
    assert comp1.s_edit.text() == "0.300"
    assert comp1.lifetime_edit.text() != ""
    assert float(comp1.lifetime_edit.text()) > 0

    comp_widget._set_component_coords_from_menu(0, 0.5, 0.4)
    assert comp1.g_edit.text() == "0.500"
    assert comp1.s_edit.text() == "0.400"


def test_components_selection_escape_cancellation(make_viewer_model, qtbot):
    """Test that pressing Escape cancels the active plot selection and restores button state."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)

    comp = comp_widget.components[0]

    # Test path 1: Matplotlib key press event 'escape'
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

    event = DummyKeyEvent('escape')
    comp_widget._handle_select_key_press_event(event)

    assert comp.select_button.text() == "Select"
    assert comp.select_button.isEnabled()
    assert comp_widget._active_select_cid is None
    assert comp_widget._active_select_key_cid is None
    assert comp_widget._active_select_shortcut is None
    assert comp_widget._active_select_idx is None

    # Test path 2: Qt QShortcut activated signal
    comp_widget._select_component(0)
    assert comp_widget._active_select_shortcut is not None

    # Trigger shortcut activation
    comp_widget._active_select_shortcut.activated.emit()

    assert comp.select_button.text() == "Select"
    assert comp.select_button.isEnabled()
    assert comp_widget._active_select_cid is None
    assert comp_widget._active_select_key_cid is None
    assert comp_widget._active_select_shortcut is None


def test_components_select_from_phasor_center_and_generalized_auto_place(
    make_viewer_model,
    qtbot,
):
    """Test selection from phasor center dialog and generalized auto intersect placement."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)

    # 1. Verify select menu options for idx=0 (Component 1) and idx=1 (Component 2)
    from qtpy.QtWidgets import QMenu

    menu0 = QMenu()
    comp_widget._populate_select_menu(0, menu0)
    texts0 = [a.text() for a in menu0.actions()]
    assert "Select from layer(s) phasor center" in texts0
    assert (
        "Auto intersect semicircle" not in texts0
    )  # idx=0 should NOT have auto intersect

    menu1 = QMenu()
    comp_widget._populate_select_menu(1, menu1)
    texts1 = [a.text() for a in menu1.actions()]
    assert "Select from layer(s) phasor center" in texts1
    assert (
        "Auto intersect semicircle" in texts1
    )  # idx=1 should have auto intersect

    # 2. Test select from phasor center using mock dialog
    from unittest.mock import patch

    from qtpy.QtWidgets import QDialog

    # Mock the dialog execution
    with patch(
        "napari_phasors.components_tab.PhasorCenterSelectionDialog"
    ) as MockDialog:
        mock_dialog_instance = MockDialog.return_value
        mock_dialog_instance.exec.return_value = QDialog.Accepted
        mock_dialog_instance.get_selected_layers.return_value = [layer.name]

        comp_widget._select_from_phasor_center(0)

    # Verify component 1 coordinates are populated from the layer's phasor center
    comp0 = comp_widget.components[0]
    assert comp0.g_edit.text() != ""
    assert comp0.s_edit.text() != ""
    assert float(comp0.g_edit.text()) > 0
    assert float(comp0.s_edit.text()) > 0

    # 3. Add a third component and verify generalized auto-placement
    comp_widget._add_component()  # Adds component 3 (idx=2)
    assert len(comp_widget.components) == 3

    # Set component 2 (idx=1) coordinates manually
    comp_widget.components[1].g_edit.setText("0.6")
    comp_widget.components[1].s_edit.setText("0.4")
    comp_widget._on_component_coords_changed(1)

    # Trigger auto-placement on component 3 (idx=2) which should intersect from component 2 (idx=1)
    comp_widget._auto_place_component_by_index(2)

    comp2 = comp_widget.components[2]
    assert comp2.g_edit.text() != ""
    assert comp2.s_edit.text() != ""
    assert float(comp2.g_edit.text()) > 0


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

    # The selection is remembered on the component and persisted to metadata.
    assert comp0.phasor_center_layers == [layer.name]
    settings = layer.metadata["settings"]["component_analysis"]
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


def test_components_histogram_dock_reserves_selector_row_height(
    make_viewer_model, qtbot
):
    """The components histogram dock is taller than tabs without a selector row.

    The docked histogram area is clamped to its minimum height, and the
    Components tab uniquely adds a "Component:" selector row above the plot, so
    its dock must reserve extra height to keep the whole canvas visible.
    """
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)

    components_min = parent.components_histogram_dock_widget.minimumHeight()
    mapping_min = parent.phasor_map_histogram_dock_widget.minimumHeight()

    assert components_min > mapping_min


def _setup_components(make_viewer_model, freq=80.0):
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": freq}
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    comp = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp)
    return viewer, layer, parent, comp


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


def test_components_restore_ui_only_from_metadata(make_viewer_model, qtbot):
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    layer.metadata["settings"][
        "component_analysis"
    ] = _linear_projection_settings()

    comp._restore_components_ui_only_from_metadata()

    assert comp.components[0].name_edit.text() == "Comp A"
    assert comp.components[1].name_edit.text() == "Comp B"
    assert comp.label_color == "red"
    assert comp.label_bold is True
    assert comp.line_width == 2.0
    assert comp.show_colormap_line is False


def test_components_restore_line_and_histogram_overlay_settings(
    make_viewer_model, qtbot
):
    """Line + fraction-histogram overlay settings round-trip via metadata.

    User edits are persisted under ``two_component_line_settings``; the restore
    path must read that key (regression: it previously only read the unused
    ``line_settings`` key, so overlay settings were never re-applied).
    """
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
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


def test_components_restore_and_recreate_linear_projection(
    make_viewer_model, qtbot
):
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    layer.metadata["settings"][
        "component_analysis"
    ] = _linear_projection_settings()

    comp._restore_and_recreate_components_from_metadata()

    assert comp.components[0].name_edit.text() == "Comp A"
    # Two components were recreated with dots.
    created = [
        c for c in comp.components if c is not None and c.dot is not None
    ]
    assert len(created) == 2


def test_components_restore_no_component_analysis_is_noop(
    make_viewer_model, qtbot
):
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    # No 'component_analysis' key -> both restore methods return early.
    comp._restore_components_ui_only_from_metadata()
    comp._restore_and_recreate_components_from_metadata()
    assert len(comp.components) == 2


def test_components_reset_plot_settings_and_artists(make_viewer_model, qtbot):
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
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

    comp.clear_artists()


def test_components_pure_helpers(make_viewer_model, qtbot):
    """Consolidated coverage of pure helper methods using a single widget
    build: per-harmonic coordinate/harmonic helpers, inverted colormap, and
    lifetime->phasor conversion."""
    import numpy as np
    from napari.utils.colormaps import Colormap

    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    # --- per-harmonic coordinate + harmonic-listing helpers ---
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

    # --- inverted colormap (name, reversed name, Colormap object) ---
    assert comp._get_inverted_colormap("viridis") is not None
    assert comp._get_inverted_colormap("viridis_r") is not None
    assert (
        comp._get_inverted_colormap(
            Colormap(
                colors=np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
                name="grayish",
            )
        )
        is not None
    )

    # --- lifetime -> phasor (valid, non-numeric, missing frequency) ---
    gg, ss = comp._compute_phasor_from_lifetime("2.0", harmonic=1)
    assert gg is not None and ss is not None
    assert comp._compute_phasor_from_lifetime("not-a-number") == (None, None)
    layer.metadata["settings"].pop("frequency", None)
    assert comp._compute_phasor_from_lifetime("2.0") == (None, None)


def test_components_restore_on_layer_change_without_layer(
    make_viewer_model, qtbot
):
    """With no primary layer, restore clears the component input fields."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    comp = parent.components_tab
    comp.components[0].g_edit.setText("0.5")
    comp.components[0].name_edit.setText("stale")

    comp._restore_on_layer_change()

    assert comp.components[0].g_edit.text() == ""
    assert comp.components[0].name_edit.text() == ""


def test_components_selection_event_updates_existing_dot(
    make_viewer_model, qtbot
):
    """A second selection event updates the existing dot/label instead of
    creating a new one."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    comp.components[0].name_edit.setText("First")
    comp._select_component(0)

    class _Event:
        inaxes = True
        xdata = 0.5
        ydata = 0.4

    comp._handle_component_selection_event(_Event())
    assert comp.components[0].dot is not None

    # Second event updates the existing dot (the else branch).
    comp._select_component(0)
    ev2 = _Event()
    ev2.xdata, ev2.ydata = 0.3, 0.2
    comp._handle_component_selection_event(ev2)
    assert comp.components[0].dot is not None


def test_components_component_fit_three_and_colors(make_viewer_model, qtbot):
    """Run a 3-component Component Fit analysis and query component colours."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    comp._add_component()
    comp.analysis_type_combo.setCurrentText("Component Fit")
    for i, (g, s) in enumerate(
        [("0.2", "0.1"), ("0.5", "0.3"), ("0.8", "0.5")]
    ):
        comp.components[i].g_edit.setText(g)
        comp.components[i].s_edit.setText(s)
        comp._on_component_coords_changed(i)

    comp._run_analysis()
    assert len(comp.fraction_layers) == 3

    # Component-colour helper for both the Component Fit (>=2 layers) path
    # and a smaller count.
    assert comp._get_component_colors_for_count(3) is not None
    assert comp._get_component_colors_for_count(2) is not None


def test_components_redisplay_preserves_display_settings(
    make_viewer_model, qtbot
):
    """Re-displaying fraction images keeps manual colormap/contrast/gamma."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    comp._add_component()
    comp.analysis_type_combo.setCurrentText("Component Fit")
    for i, (g, s) in enumerate(
        [("0.2", "0.1"), ("0.5", "0.3"), ("0.8", "0.5")]
    ):
        comp.components[i].g_edit.setText(g)
        comp.components[i].s_edit.setText(s)
        comp._on_component_coords_changed(i)

    comp._run_analysis()
    assert len(comp.fraction_layers) == 3

    # Record the default colormap of an untouched layer, then manually tweak
    # a different one's colormap, contrast limits and gamma.
    other_layer = comp.fraction_layers[0]
    other_colormap = other_layer.colormap.name

    tweaked_layer = comp.fraction_layers[1]
    tweaked_layer.colormap = "magma"
    tweaked_layer.contrast_limits = (0.15, 0.85)
    tweaked_layer.gamma = 0.4

    # Press "Display Component Fraction Images" again.
    comp._run_analysis()

    new_tweaked = viewer.layers[tweaked_layer.name]
    assert new_tweaked.colormap.name == "magma"
    assert np.allclose(new_tweaked.contrast_limits, (0.15, 0.85))
    assert new_tweaked.gamma == 0.4

    # The untouched layer must keep its (default) colormap, not reset.
    new_other = viewer.layers[other_layer.name]
    assert new_other.colormap.name == other_colormap


def test_components_auto_place_by_index(make_viewer_model, qtbot):
    """Cover _auto_place_component_by_index guards and the success path."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    # Out-of-range indices return early.
    comp._auto_place_component_by_index(0)
    comp._auto_place_component_by_index(99)

    # Previous component not set -> warning + early return.
    comp._auto_place_component_by_index(1)

    # Set component 0, then auto-place component 1 on the universal circle.
    comp.components[0].g_edit.setText("0.5")
    comp.components[0].s_edit.setText("0.3")
    comp._on_component_coords_changed(0)
    comp._auto_place_component_by_index(1)
    assert comp.components[1].g_edit.text() != ""

    # Without a frequency the method warns and returns.
    layer.metadata["settings"].pop("frequency", None)
    comp.components[0].g_edit.setText("0.4")
    comp.components[0].s_edit.setText("0.2")
    comp._auto_place_component_by_index(1)


def test_components_fraction_layer_colormap_and_contrast_events(
    make_viewer_model, qtbot
):
    """Changing the fraction layer's colormap/contrast triggers handlers."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    comp.components[0].g_edit.setText("0.2")
    comp.components[0].s_edit.setText("0.1")
    comp._on_component_coords_changed(0)
    comp.components[1].g_edit.setText("0.8")
    comp.components[1].s_edit.setText("0.5")
    comp._on_component_coords_changed(1)
    comp._run_analysis()

    fl = comp.comp1_fractions_layer
    assert fl is not None
    fl.contrast_limits = (0.1, 0.9)
    fl.colormap = "viridis"
    assert comp.colormap_contrast_limits is not None


def test_components_name_change_creates_and_moves_label(
    make_viewer_model, qtbot
):
    """Setting/clearing a component name creates, repositions and removes its
    text label."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    comp._create_component_at_coordinates(0, 0.5, 0.3)

    comp.components[0].name_edit.setText("Alpha")
    comp._on_component_name_changed(0)
    assert comp.components[0].text is not None

    # Renaming again exercises the previous-position branch.
    comp.components[0].name_edit.setText("Beta")
    comp._on_component_name_changed(0)
    assert comp.components[0].text is not None

    # Clearing the name removes the label.
    comp.components[0].name_edit.setText("")
    comp._on_component_name_changed(0)


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


def test_components_apply_saved_colormap_settings(make_viewer_model, qtbot):
    """Apply saved colormap settings (name and explicit colours) to the
    fraction layer."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    comp.components[0].g_edit.setText("0.2")
    comp.components[0].s_edit.setText("0.1")
    comp._on_component_coords_changed(0)
    comp.components[1].g_edit.setText("0.8")
    comp.components[1].s_edit.setText("0.5")
    comp._on_component_coords_changed(1)
    comp._run_analysis()
    assert comp.comp1_fractions_layer is not None

    # Saved as a named colormap.
    comp._saved_colormap_name = "viridis"
    comp._saved_colormap_colors = None
    comp._saved_contrast_limits = (0.0, 1.0)
    comp._apply_saved_colormap_settings()

    # Saved as an explicit colour list.
    comp._saved_colormap_colors = [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    comp._saved_contrast_limits = [0.0, 1.0]
    comp._apply_saved_colormap_settings()


def test_components_recreate_variants_and_colormaps(make_viewer_model, qtbot):
    """One widget, two recreate flows: Linear Projection with fraction-layer
    colormap restore, then Component Fit (which also adds a 3rd component)."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    settings = layer.metadata["settings"]

    # 1. Linear Projection recreate with per-component colormap settings.
    settings["component_analysis"] = {
        "analysis_type": "Linear Projection",
        "last_analysis_harmonic": 1,
        "components": {
            "0": {
                "name": "A",
                "gs_harmonics": {
                    "1": {
                        "g": 0.6,
                        "s": 0.3,
                        "colormap_name": "viridis",
                        "contrast_limits": [0.0, 1.0],
                    }
                },
            },
            "1": {
                "name": "B",
                "gs_harmonics": {
                    "1": {
                        "g": 0.3,
                        "s": 0.2,
                        "colormap_colors": [
                            [0.0, 0.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0],
                        ],
                        "contrast_limits": [0.1, 0.9],
                    }
                },
            },
        },
        "line_settings": {"show_colormap_line": True, "line_width": 2.0},
        "label_settings": {"fontsize": 11, "color": "red"},
    }
    comp._restore_and_recreate_components_from_metadata()
    assert comp.comp1_fractions_layer is not None

    # 2. Component Fit recreate with three components (adds a component).
    settings["component_analysis"] = {
        "analysis_type": "Component Fit",
        "last_analysis_harmonic": 1,
        "components": {
            "0": {"name": "C0", "gs_harmonics": {"1": {"g": 0.2, "s": 0.1}}},
            "1": {"name": "C1", "gs_harmonics": {"1": {"g": 0.5, "s": 0.3}}},
            "2": {"name": "C2", "gs_harmonics": {"1": {"g": 0.8, "s": 0.5}}},
        },
    }
    comp._restore_and_recreate_components_from_metadata()
    assert len(comp.fraction_layers) == 3


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


def test_components_widget_fraction_histogram_overlay(
    make_viewer_model,
    qtbot,
):
    """Fraction histogram overlay is drawn/removed with the setting toggle."""
    from matplotlib.image import AxesImage

    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)

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


def test_components_widget_fraction_histogram_height_setting(
    make_viewer_model,
    qtbot,
):
    """Overlay height scales the histogram profile."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)

    _setup_linear_projection(comp_widget)

    comp_widget.show_fraction_histogram = True

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


def test_components_widget_fraction_histogram_offset_flips_side(
    make_viewer_model,
    qtbot,
):
    """A negative histogram offset mirrors the overlay to the other side."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(comp_widget)

    _setup_linear_projection(comp_widget)
    comp_widget.show_fraction_histogram = True

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

    # Flipping the sign puts the apex on the opposite side of the line.
    assert not np.allclose(pos_apex, neg_apex)


def test_center_fill_slider_paint_event_zero_span(qtbot):
    """paintEvent should no-op (not raise) when minimum == maximum."""
    slider = CenterFillSlider()
    qtbot.addWidget(slider)
    slider.setMinimum(0)
    slider.setMaximum(0)

    slider.show()
    # Forces a synchronous call to paintEvent; the span==0 guard must return
    # early without error.
    slider.repaint()
    slider.close()


def test_restore_fraction_layer_colormaps_applies_gamma(
    make_viewer_model, qtbot
):
    """Saved settings with a non-None gamma are applied to the fraction
    layer by ``_restore_fraction_layer_colormaps``."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_linear_projection(comp)
    assert comp.comp1_fractions_layer is not None

    settings = {
        "components": {
            "0": {
                "name": "Component 1",
                "gs_harmonics": {
                    "1": {
                        "colormap_name": "viridis",
                        "gamma": 2.0,
                    }
                },
            }
        }
    }
    comp._restore_fraction_layer_colormaps(settings, "1")

    assert comp.comp1_fractions_layer.gamma == 2.0


def test_on_color_button_clicked_updates_color_and_metadata(
    make_viewer_model, qtbot
):
    """Picking a valid color from the dialog updates the button, the stored
    default color, and the persisted metadata setting."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    comp._open_plot_settings_dialog()

    with patch.object(
        QColorDialog, "getColor", return_value=QColor("#ff0000")
    ):
        comp._on_color_button_clicked()

    assert comp.default_component_color == "#ff0000"
    settings = layer.metadata["settings"]["component_analysis"]
    assert (
        settings["two_component_line_settings"]["default_component_color"]
        == "#ff0000"
    )


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


def test_apply_saved_colormap_settings_exception_reconnects_gamma(
    make_viewer_model, qtbot
):
    """When applying saved colormap settings raises, the except-branch must
    still reconnect the colormap/contrast_limits/gamma events."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_linear_projection(comp)
    assert comp.comp1_fractions_layer is not None

    # An invalid colormap name raises inside the try block, forcing the
    # except branch (which re-connects the events) to run.
    comp._saved_colormap_name = "not_a_real_colormap_xyz"
    comp._saved_colormap_colors = None
    comp._saved_contrast_limits = (0.0, 1.0)

    comp._apply_saved_colormap_settings()

    assert _is_connected(
        comp.comp1_fractions_layer.events.gamma, comp._on_colormap_changed
    )
    assert _is_connected(
        comp.comp1_fractions_layer.events.colormap, comp._on_colormap_changed
    )


def test_on_histogram_offset_changed_updates_metadata_and_redraws(
    make_viewer_model, qtbot
):
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_linear_projection(comp)
    comp.show_fraction_histogram = True

    comp._on_histogram_offset_changed(0.3)

    assert comp.histogram_offset == 0.3
    settings = layer.metadata["settings"]["component_analysis"]
    assert settings["two_component_line_settings"]["histogram_offset"] == 0.3


def test_on_histogram_transparency_changed_updates_metadata_and_redraws(
    make_viewer_model, qtbot
):
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_linear_projection(comp)
    comp.show_fraction_histogram = True

    comp._on_histogram_transparency_changed(0.4)

    assert abs(comp.histogram_alpha - 0.6) < 1e-9
    settings = layer.metadata["settings"]["component_analysis"]
    assert (
        abs(settings["two_component_line_settings"]["histogram_alpha"] - 0.6)
        < 1e-9
    )


class _RemoveRaisesArtist:
    """Fake artist whose ``remove`` raises, exercising the suppressed
    ValueError/AttributeError branch in ``_remove_histogram_overlay``."""

    def remove(self):
        raise ValueError("boom")


def test_remove_histogram_overlay_single_artist(make_viewer_model, qtbot):
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    # Single artist (not a list/tuple) - exercises the wrap-into-list branch.
    comp.component_histogram = _RemoveRaisesArtist()
    comp._remove_histogram_overlay()
    assert comp.component_histogram is None


def test_get_all_artists_and_set_artists_visible_single_histogram_artist(
    make_viewer_model, qtbot
):
    """``get_all_artists``/``set_artists_visible`` wrap a non-list/tuple
    ``component_histogram`` into a single-element list before use."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    fake_artist = MagicMock()
    comp.component_histogram = fake_artist

    artists = comp.get_all_artists()
    assert fake_artist in artists

    comp.set_artists_visible(True)
    fake_artist.set_visible.assert_called_once_with(True)


def test_get_first_component_fraction_values_pools_multiple_layers(
    make_viewer_model, qtbot
):
    """Fraction values are pooled (and non-finite values dropped) across
    every fraction layer matching the first component's name."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_linear_projection(comp)
    assert comp.comp1_fractions_layer is not None

    comp1_name = comp.components[0].name_edit.text().strip() or "Component 1"
    extra_layer = Image(
        np.array([[0.25, 0.75], [np.nan, 0.5]]),
        name=f"{comp1_name} fractions: other_image",
    )
    viewer.add_layer(extra_layer)

    values = comp._get_first_component_fraction_values()

    assert not np.any(np.isnan(values))
    original_size = np.asarray(
        comp.comp1_fractions_layer.data, dtype=float
    ).size
    assert values.size == original_size + 3


def test_get_first_component_fraction_values_fallback_to_comp1_layer(
    make_viewer_model, qtbot
):
    """When no fraction layer matches the live component name, values fall
    back to ``comp1_fractions_layer`` directly."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_linear_projection(comp)
    assert comp.comp1_fractions_layer is not None

    # Renaming after analysis means no viewer layer matches the new name.
    comp.components[0].name_edit.setText("Renamed Component")

    values = comp._get_first_component_fraction_values()
    expected = np.asarray(comp.comp1_fractions_layer.data, dtype=float).ravel()
    expected = expected[np.isfinite(expected)]
    assert values.size == expected.size
    assert values.size > 0


def test_get_first_component_fraction_values_returns_empty_with_no_data(
    make_viewer_model, qtbot
):
    """With no fraction layers and no comp1_fractions_layer, an empty array
    is returned."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    assert comp.comp1_fractions_layer is None

    values = comp._get_first_component_fraction_values()
    assert values.size == 0


def test_draw_colormap_line_jet_fallback(make_viewer_model, qtbot):
    """``_draw_colormap_line`` falls back to ``plt.cm.jet`` when
    ``fractions_colormap`` is unset."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    comp.fractions_colormap = None

    fig, ax = plt.subplots()
    try:
        comp._draw_colormap_line(ax, 0.0, 0.0, 1.0, 1.0)
    finally:
        plt.close(fig)


def test_sync_component_layers_gamma_guard_returns_early(
    make_viewer_model, qtbot
):
    """``_sync_component_layers_gamma`` returns immediately (without
    touching any layers) while ``_updating_linked_layers`` is already
    True."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    comp._updating_linked_layers = True

    with patch.object(comp, "_get_all_layers_for_component") as mock_get:
        comp._sync_component_layers_gamma(0, 1.5)

    mock_get.assert_not_called()


def test_find_and_reconnect_layer_expected_name(make_viewer_model, qtbot):
    """Reconnecting via the expected layer name connects the gamma event."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    fraction_layer = Image(
        np.zeros((5, 5)), name="Component 1 fractions: img1"
    )
    viewer.add_layer(fraction_layer)

    comp._find_and_reconnect_layer(
        "Component 1 fractions: img1", "Component 1", "img1", 0
    )

    assert comp.comp1_fractions_layer is fraction_layer
    assert _is_connected(
        fraction_layer.events.gamma, comp._on_colormap_changed
    )


def test_find_and_reconnect_layer_possible_names(make_viewer_model, qtbot):
    """Reconnecting via one of the fallback naming conventions renames the
    layer and connects the gamma event."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    fraction_layer = Image(
        np.zeros((5, 5)), name="Component 1 fractions: img2"
    )
    viewer.add_layer(fraction_layer)

    comp._find_and_reconnect_layer(
        "Component 1 fractions: RENAMED", "Component 1", "img2", 0
    )

    assert comp.comp1_fractions_layer is fraction_layer
    assert fraction_layer.name == "Component 1 fractions: RENAMED"
    assert _is_connected(
        fraction_layer.events.gamma, comp._on_colormap_changed
    )


def test_get_inverted_colormap_name_fallback(make_viewer_model, qtbot):
    """Non-standard colormap names without explicit colors fall back to the
    ``_r``-suffix inversion convention (the fallback name is a literal
    ``jet``/``jet_r`` pair keyed only off whether the input already ends in
    ``_r``)."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)

    assert comp._get_inverted_colormap("zzz_not_a_real_colormap") == "jet_r"
    assert comp._get_inverted_colormap("zzz_not_a_real_colormap_r") == "jet"


def test_linear_projection_preserves_gamma_on_redisplay(
    make_viewer_model, qtbot
):
    """Re-triggering the same analysis preserves a manually-set gamma on the
    already-displayed comp1 fraction layer."""
    viewer, layer, parent, comp = _setup_components(make_viewer_model)
    _setup_linear_projection(comp)
    assert comp.comp1_fractions_layer is not None

    comp.comp1_fractions_layer.gamma = 2.5
    comp._run_analysis()

    assert comp.comp1_fractions_layer.gamma == 2.5


def test_linear_projection_restores_saved_gamma_from_metadata(
    make_viewer_model, qtbot
):
    """A gamma value saved in metadata under the first component's harmonic
    data is picked up when the fraction layer is first created."""
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
