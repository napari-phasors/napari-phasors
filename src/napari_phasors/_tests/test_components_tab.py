import numpy as np
from matplotlib.collections import LineCollection
from phasorpy.component import phasor_component_fraction
from phasorpy.lifetime import phasor_from_lifetime

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.plotter import PlotterWidget


def test_components_widget_initialization_values(make_napari_viewer):
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

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
    assert comp_widget.component_locations == {}
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


def test_components_widget_lifetime_inputs_visibility_no_frequency(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    layer = create_image_layer_with_phasors()
    # Ensure no frequency
    layer.metadata.pop("settings", None)
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # Lifetime widgets should be hidden - check first component
    comp = comp_widget.components[0]
    assert comp.ui_elements['lifetime_label'].isHidden()
    assert comp.lifetime_edit.isHidden()


def test_components_widget_lifetime_inputs_visibility_with_frequency(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
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
    expected_g, expected_s = phasor_from_lifetime(80.0 * parent.harmonic, 3.0)
    assert abs(g_val - expected_g) < 1e-6
    assert abs(s_val - expected_s) < 1e-6


def test_components_widget_component_creation_and_line(make_napari_viewer):
    viewer = make_napari_viewer()
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
    make_napari_viewer,
):
    """Test that fraction calculation creates both comp1 and comp2 layers."""
    viewer = make_napari_viewer()
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

    # Calculate expected fractions
    phasor_labels_layer = layer.metadata["phasor_features_labels_layer"]
    features = phasor_labels_layer.features
    harmonic_mask = features["harmonic"] == parent.harmonic
    real = features.loc[harmonic_mask, "G"]
    imag = features.loc[harmonic_mask, "S"]
    expected_comp1_fractions = phasor_component_fraction(
        np.array(real),
        np.array(imag),
        (0.2, 0.8),
        (0.1, 0.5),
    )
    expected_comp1_fractions = expected_comp1_fractions.reshape(
        phasor_labels_layer.data.shape
    )
    expected_comp2_fractions = 1.0 - expected_comp1_fractions

    comp_widget._run_analysis()

    # Both fractions layers should be created
    assert comp_widget.comp1_fractions_layer in viewer.layers
    assert comp_widget.comp2_fractions_layer in viewer.layers

    # Check data
    comp1_data = comp_widget.comp1_fractions_layer.data
    comp2_data = comp_widget.comp2_fractions_layer.data
    np.testing.assert_allclose(
        comp1_data, expected_comp1_fractions, rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        comp2_data, expected_comp2_fractions, rtol=1e-6, atol=1e-9
    )

    # Check initial colormaps
    assert comp_widget.comp1_fractions_layer.colormap.name == 'PiYG'
    assert comp_widget.comp2_fractions_layer.colormap.name == 'PiYG_r'

    assert isinstance(comp_widget.component_line, LineCollection)


def test_components_widget_colormap_synchronization(make_napari_viewer):
    """Test that changing colormap on one layer updates the other with inverted colors."""
    viewer = make_napari_viewer()
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

    # Store original colormaps
    orig_comp1_colors = (
        comp_widget.comp1_fractions_layer.colormap.colors.copy()
    )
    orig_comp2_colors = (
        comp_widget.comp2_fractions_layer.colormap.colors.copy()
    )

    # Change colormap on comp1 layer
    comp_widget.comp1_fractions_layer.colormap = 'viridis'

    # comp2 should automatically update with inverted colors
    comp1_colors = comp_widget.comp1_fractions_layer.colormap.colors
    comp2_colors = comp_widget.comp2_fractions_layer.colormap.colors

    # Colors should be inverted
    np.testing.assert_allclose(comp1_colors, comp2_colors[::-1], rtol=1e-6)

    # Both should be different from original
    assert not np.allclose(orig_comp1_colors, comp1_colors)
    assert not np.allclose(orig_comp2_colors, comp2_colors)


def test_components_widget_colormap_synchronization_from_comp2(
    make_napari_viewer,
):
    """Test that changing colormap on comp2 layer updates comp1 with inverted colors."""
    viewer = make_napari_viewer()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab
    parent.tab_widget.setCurrentWidget(parent.components_tab)

    # Ensure Linear Projection is selected
    comp_widget.analysis_type_combo.setCurrentText("Linear Projection")

    # Create components and calculate fractions
    comp_widget.components[0].g_edit.setText("0.4")
    comp_widget.components[0].s_edit.setText("0.25")
    comp_widget._on_component_coords_changed(0)
    comp_widget.components[1].g_edit.setText("0.7")
    comp_widget.components[1].s_edit.setText("0.45")
    comp_widget._on_component_coords_changed(1)
    comp_widget._run_analysis()

    # Change colormap on comp2 layer
    comp_widget.comp2_fractions_layer.colormap = 'plasma'

    # comp1 should automatically update with inverted colors
    comp1_colors = comp_widget.comp1_fractions_layer.colormap.colors
    comp2_colors = comp_widget.comp2_fractions_layer.colormap.colors

    # Colors should be inverted
    np.testing.assert_allclose(comp1_colors, comp2_colors[::-1], rtol=1e-6)


def test_components_widget_colormap_update_legacy(make_napari_viewer):
    """Legacy test updated for new dual-layer structure."""
    viewer = make_napari_viewer()
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


def test_components_widget_colormap_fallback_handling(make_napari_viewer):
    """Test that colormap sync handles colormaps without colors attribute gracefully."""
    viewer = make_napari_viewer()
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
    assert comp_widget.comp2_fractions_layer.colormap is not None


def test_components_widget_visibility_toggle(make_napari_viewer):
    viewer = make_napari_viewer()
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


def test_components_widget_line_settings_dialog_effects(make_napari_viewer):
    viewer = make_napari_viewer()
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

    # Change offset
    comp_widget.line_offset_slider.setValue(120)  # 0.120
    assert abs(comp_widget.line_offset - 0.12) < 1e-6

    # Change width
    comp_widget.line_width_spin.setValue(5.0)
    assert comp_widget.line_width == 5.0

    # Change alpha
    comp_widget.line_alpha_slider.setValue(55)
    assert abs(comp_widget.line_alpha - 0.55) < 1e-6


def test_components_widget_label_style_dialog(make_napari_viewer):
    viewer = make_napari_viewer()
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


def test_components_widget_add_remove_components(make_napari_viewer):
    """Test adding and removing components."""
    viewer = make_napari_viewer()
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


def test_components_widget_analysis_type_changes(make_napari_viewer):
    """Test analysis type changes based on component count."""
    viewer = make_napari_viewer()
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


def test_components_widget_multi_component_analysis(make_napari_viewer):
    """Test multi-component analysis with component fit."""
    viewer = make_napari_viewer()
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


def test_components_widget_polygon_visualization(make_napari_viewer):
    """Test that 3+ components create a polygon instead of a line."""
    viewer = make_napari_viewer()
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


def test_components_widget_harmonic_storage_and_switching(make_napari_viewer):
    """Test component storage across harmonic changes."""
    viewer = make_napari_viewer()
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

    # But stored in component_locations
    assert 1 in comp_widget.component_locations
    stored_comp1 = comp_widget.component_locations[1][0]
    assert abs(stored_comp1['g'] - 0.2) < 1e-6
    assert abs(stored_comp1['s'] - 0.1) < 1e-6
    assert stored_comp1['name'] == "Test Component 1"

    # Switch back to harmonic 1
    comp_widget._on_harmonic_changed(1)

    # Components should be restored
    assert abs(float(comp_widget.components[0].g_edit.text()) - 0.2) < 1e-6
    assert abs(float(comp_widget.components[0].s_edit.text()) - 0.1) < 1e-6
    assert comp_widget.components[0].name_edit.text() == "Test Component 1"


def test_components_widget_clear_all_components(make_napari_viewer):
    """Test clearing all components."""
    viewer = make_napari_viewer()
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


def test_components_widget_calculate_button_text_changes(make_napari_viewer):
    """Test that calculate button text changes based on analysis type."""
    viewer = make_napari_viewer()
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
