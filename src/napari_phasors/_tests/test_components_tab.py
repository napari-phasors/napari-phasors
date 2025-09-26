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
    assert comp_widget.comp1_fractions_layer is None
    assert comp_widget.comp2_fractions_layer is None  # Updated
    assert comp_widget.fractions_colormap is None
    assert comp_widget.colormap_contrast_limits is None
    # UI elements exist
    assert comp_widget.comp1_name_edit is not None
    assert comp_widget.first_button is not None
    assert comp_widget.first_edit1 is not None
    assert comp_widget.first_edit2 is not None
    assert comp_widget.comp2_name_edit is not None
    assert comp_widget.second_button is not None
    assert comp_widget.second_edit1 is not None
    assert comp_widget.second_edit2 is not None
    assert comp_widget.calculate_button is not None


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

    # Lifetime widgets should be hidden
    assert comp_widget.first_lifetime_label.isHidden()
    assert comp_widget.first_lifetime_edit.isHidden()
    assert comp_widget.second_lifetime_label.isHidden()
    assert comp_widget.second_lifetime_edit.isHidden()


def test_components_widget_lifetime_inputs_visibility_with_frequency(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    comp_widget = parent.components_tab

    # Lifetime widgets visible
    assert not comp_widget.first_lifetime_label.isHidden()
    assert not comp_widget.first_lifetime_edit.isHidden()
    assert not comp_widget.second_lifetime_label.isHidden()
    assert not comp_widget.second_lifetime_edit.isHidden()

    # Enter lifetime and verify G,S updated
    comp_widget.first_lifetime_edit.setText("3.0")
    comp_widget._update_component_from_lifetime(0)
    g_val = float(comp_widget.first_edit1.text())
    s_val = float(comp_widget.first_edit2.text())
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
    comp_widget.first_edit1.setText("0.25")
    comp_widget.first_edit2.setText("0.15")
    comp_widget._on_component_coords_changed(0)

    comp_widget.second_edit1.setText("0.75")
    comp_widget.second_edit2.setText("0.45")
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

    # Define two components
    comp_widget.first_edit1.setText("0.2")
    comp_widget.first_edit2.setText("0.1")
    comp_widget._on_component_coords_changed(0)

    comp_widget.second_edit1.setText("0.8")
    comp_widget.second_edit2.setText("0.5")
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

    comp_widget.on_calculate_button_clicked()

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

    # Create components and calculate fractions
    comp_widget.first_edit1.setText("0.3")
    comp_widget.first_edit2.setText("0.2")
    comp_widget._on_component_coords_changed(0)
    comp_widget.second_edit1.setText("0.9")
    comp_widget.second_edit2.setText("0.6")
    comp_widget._on_component_coords_changed(1)
    comp_widget.on_calculate_button_clicked()

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

    # Create components and calculate fractions
    comp_widget.first_edit1.setText("0.4")
    comp_widget.first_edit2.setText("0.25")
    comp_widget._on_component_coords_changed(0)
    comp_widget.second_edit1.setText("0.7")
    comp_widget.second_edit2.setText("0.45")
    comp_widget._on_component_coords_changed(1)
    comp_widget.on_calculate_button_clicked()

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

    # Create components
    comp_widget.first_edit1.setText("0.3")
    comp_widget.first_edit2.setText("0.2")
    comp_widget._on_component_coords_changed(0)
    comp_widget.second_edit1.setText("0.9")
    comp_widget.second_edit2.setText("0.6")
    comp_widget._on_component_coords_changed(1)
    comp_widget.on_calculate_button_clicked()

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

    # Create components and calculate fractions
    comp_widget.first_edit1.setText("0.5")
    comp_widget.first_edit2.setText("0.3")
    comp_widget._on_component_coords_changed(0)
    comp_widget.second_edit1.setText("0.8")
    comp_widget.second_edit2.setText("0.6")
    comp_widget._on_component_coords_changed(1)
    comp_widget.on_calculate_button_clicked()

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
    comp_widget.first_edit1.setText("0.4")
    comp_widget.first_edit2.setText("0.25")
    comp_widget._on_component_coords_changed(0)
    comp_widget.second_edit1.setText("0.7")
    comp_widget.second_edit2.setText("0.45")
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

    # Create components
    comp_widget.first_edit1.setText("0.2")
    comp_widget.first_edit2.setText("0.15")
    comp_widget._on_component_coords_changed(0)
    comp_widget.second_edit1.setText("0.6")
    comp_widget.second_edit2.setText("0.45")
    comp_widget._on_component_coords_changed(1)
    comp_widget.on_calculate_button_clicked()

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
    comp_widget.comp1_name_edit.setText("A")
    comp_widget.first_edit1.setText("0.3")
    comp_widget.first_edit2.setText("0.2")
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
