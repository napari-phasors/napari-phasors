from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from phasorpy.lifetime import phasor_from_fret_donor
from phasorpy.phasor import phasor_nearest_neighbor

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.plotter import PlotterWidget


def test_fret_widget_initialization(make_napari_viewer):
    """Test the initialization of the FretWidget."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Basic widget structure tests
    assert widget.viewer == viewer
    assert widget.parent_widget == parent
    assert widget.layout().count() > 0

    # Test initial UI state
    assert widget.donor_line_edit.text() == ""
    assert widget.frequency_input.text() == ""
    assert widget.background_real_edit.text() == "0.0"
    assert widget.background_imag_edit.text() == "0.0"
    assert (
        widget.calculate_fret_efficiency_button.text()
        == "Calculate FRET efficiency"
    )

    # Test source selectors are set to Manual by default
    assert widget.donor_source_selector.currentText() == "Manual"
    assert widget.bg_source_selector.currentText() == "Manual"

    # Test stacked widgets show manual inputs by default
    assert widget.donor_stack.currentIndex() == 0  # Manual page
    assert widget.bg_stack.currentIndex() == 0  # Manual page

    # Test donor lifetime combobox
    assert widget.donor_lifetime_combobox.currentText() in [
        "Select layer...",
        "",
    ]
    assert (
        widget.lifetime_type_combobox.currentText()
        == "Apparent Phase Lifetime"
    )
    assert widget.lifetime_type_combobox.count() == 3

    # Verify all lifetime modes are present
    lifetime_modes = [
        widget.lifetime_type_combobox.itemText(i)
        for i in range(widget.lifetime_type_combobox.count())
    ]
    expected_modes = [
        "Apparent Phase Lifetime",
        "Apparent Modulation Lifetime",
        "Normal Lifetime",
    ]
    assert lifetime_modes == expected_modes

    # Test slider initial values
    assert widget.background_slider.value() == 10  # 0.1 * 100
    assert widget.fretting_slider.value() == 100  # 1.0 * 100
    assert widget.colormap_checkbox.isChecked() is True

    # Test dynamic labels show default text
    assert widget.donor_label.text() == "Donor lifetime (ns):"
    assert widget.background_position_label.text() == "Background position:"


def test_fret_widget_parameter_updates(make_napari_viewer):
    """Test that parameters update correctly when UI changes."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Create and add layer
    test_layer = create_image_layer_with_phasors()
    viewer.add_layer(test_layer)

    # Test donor lifetime input
    widget.donor_line_edit.setText("2.5")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.2")
    widget.background_imag_edit.setText("0.3")

    # Trigger parameter change
    widget._on_parameters_changed()

    assert widget.donor_lifetime == 2.5
    assert widget.frequency == 80 * parent.harmonic
    assert widget.background_real == 0.2
    assert widget.background_imag == 0.3


def test_fret_widget_background_slider(make_napari_viewer):
    """Test background slider functionality."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Test slider value change
    widget.background_slider.setValue(25)  # 0.25
    widget._on_background_slider_changed()

    assert widget.donor_background == 0.25
    assert widget.background_label.text() == "0.25"


def test_fret_widget_fretting_slider(make_napari_viewer):
    """Test fretting proportion slider functionality."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Test slider value change
    widget.fretting_slider.setValue(75)  # 0.75
    widget._on_fretting_slider_changed()

    assert widget.donor_fretting_proportion == 0.75
    assert widget.fretting_label.text() == "0.75"


def test_fret_widget_colormap_checkbox(make_napari_viewer):
    """Test colormap checkbox functionality."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Test unchecking colormap
    widget.colormap_checkbox.setChecked(False)
    widget._on_colormap_checkbox_changed()

    assert widget.use_colormap is False

    # Test checking colormap
    widget.colormap_checkbox.setChecked(True)
    widget._on_colormap_checkbox_changed()

    assert widget.use_colormap is True


def test_calculate_background_position_no_layer(make_napari_viewer):
    """Test background calculation with no layer selected."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # No layer selected
    parent._labels_layer_with_phasor_features = None

    # Should return without error
    widget._calculate_background_position()

    # Values should remain at defaults
    assert widget.background_real_edit.text() == "0.0"
    assert widget.background_imag_edit.text() == "0.0"


def test_calculate_background_position_with_layer(make_napari_viewer):
    """Test background calculation with a valid layer."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Update the background combobox to include the new layer
    widget._update_background_combobox()

    # Check that the combobox has been populated correctly
    combobox_items = [
        widget.background_image_combobox.itemText(i)
        for i in range(widget.background_image_combobox.count())
    ]
    assert "Select layer..." in combobox_items
    assert "test_layer" in combobox_items

    # Switch to "From layer" mode and select the test layer
    widget.bg_source_selector.setCurrentText("From layer")
    widget.background_image_combobox.setCurrentText("test_layer")

    # Set up the parent widget properly
    parent.harmonic = 1

    # Calculate background position
    widget._calculate_background_position()

    # Check that values were updated (they should be different from defaults)
    real_text = widget.background_real_edit.text()
    imag_text = widget.background_imag_edit.text()

    # Values should be numeric strings
    assert (
        real_text != "0.0" or imag_text != "0.0"
    )  # At least one should change
    assert float(real_text) >= 0
    assert float(imag_text) >= 0

    # Check that the position was stored for the current harmonic
    assert parent.harmonic in widget.background_positions_by_harmonic
    stored_position = widget.background_positions_by_harmonic[parent.harmonic]

    # Compare with proper precision handling - the stored values should match
    # the displayed text when formatted to 3 decimal places
    assert abs(stored_position['real'] - float(real_text)) < 0.001
    assert abs(stored_position['imag'] - float(imag_text)) < 0.001

    # Check that the label shows the calculated values in "From layer" mode
    expected_label = f"Background position: G={stored_position['real']:.2f}, S={stored_position['imag']:.2f}"
    assert widget.background_position_label.text() == expected_label

    # Test with "Select layer..." selected - should not crash
    widget.background_image_combobox.setCurrentText("Select layer...")
    widget._calculate_background_position()

    # Label should revert to default when no valid layer is selected
    assert widget.background_position_label.text() == "Background position:"


def test_plot_donor_trajectory_no_parameters(make_napari_viewer):
    """Test plotting donor trajectory with missing parameters."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Try to plot without setting parameters
    widget.plot_donor_trajectory()

    # Should not create a line
    assert widget.current_donor_line is None


def test_plot_donor_trajectory_with_parameters(make_napari_viewer):
    """Test plotting donor trajectory with valid parameters."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Set valid parameters
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    # Mock the canvas and figure
    parent.canvas_widget = Mock()
    parent.canvas_widget.figure = Mock()
    ax_mock = Mock()
    parent.canvas_widget.figure.gca.return_value = ax_mock
    parent.canvas_widget.canvas = Mock()

    # Plot trajectory
    widget.plot_donor_trajectory()

    # Should create a line
    assert ax_mock.plot.called


def test_calculate_fret_efficiency_no_layer(make_napari_viewer):
    """Test FRET efficiency calculation with no layer selected."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # No layer selected
    parent._labels_layer_with_phasor_features = None

    # Should return without error
    widget.calculate_fret_efficiency()


def test_calculate_fret_efficiency_with_layer(make_napari_viewer):
    """Test FRET efficiency calculation with valid data."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Setup widget parameters
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    # Get values of G and S from the test layer using new array-based metadata
    metadata = test_layer.metadata
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

    # Get trajectory data
    donor_trajectory_real, donor_trajectory_imag = phasor_from_fret_donor(
        80,
        2,
        fret_efficiency=widget._fret_efficiencies,
        donor_background=widget.donor_background,
        background_imag=0.1,
        background_real=0.1,
        donor_fretting=widget.donor_fretting_proportion,
    )

    # Calculate expected FRET efficiency
    expected_fret_efficiency = phasor_nearest_neighbor(
        np.array(real),
        np.array(imag),
        donor_trajectory_real,
        donor_trajectory_imag,
        values=widget._fret_efficiencies,
    )

    # Click on calculate button
    widget.calculate_fret_efficiency_button.click()

    # Check that a FRET layer was added
    fret_layer_name = f"FRET efficiency: test_layer"
    assert fret_layer_name in [layer.name for layer in viewer.layers]
    assert widget.fret_layer is not None

    # Check that the created layer has the expected FRET values
    fret_layer = viewer.layers[fret_layer_name]
    assert_array_equal(fret_layer.data.flatten(), expected_fret_efficiency)

    # Change values and recalculate
    widget.donor_line_edit.setText("1.5")

    # Get trajectory data
    donor_trajectory_real, donor_trajectory_imag = phasor_from_fret_donor(
        80,
        1.5,
        fret_efficiency=widget._fret_efficiencies,
        donor_background=widget.donor_background,
        background_imag=0.1,
        background_real=0.1,
        donor_fretting=widget.donor_fretting_proportion,
    )

    # Calculate expected FRET efficiency
    expected_fret_efficiency = phasor_nearest_neighbor(
        np.array(real),
        np.array(imag),
        donor_trajectory_real,
        donor_trajectory_imag,
        values=widget._fret_efficiencies,
    )

    # Click on calculate button
    widget.calculate_fret_efficiency_button.click()

    # Check that the existing FRET layer was updated
    assert fret_layer_name in [layer.name for layer in viewer.layers]
    fret_layer = viewer.layers[fret_layer_name]
    assert_array_equal(fret_layer.data.flatten(), expected_fret_efficiency)


def test_artist_management(make_napari_viewer):
    """Test artist visibility and management."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Initially no artists
    assert len(widget.get_all_artists()) == 0

    # Add values and plot
    widget.frequency_input.setText("80")
    widget.donor_line_edit.setText("2.0")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")
    widget.plot_donor_trajectory()

    # Should have one artist now
    assert len(widget.get_all_artists()) == 3
    assert widget.current_donor_line is not None
    assert widget.current_donor_line in widget.get_all_artists()

    # Artist should be visible initially
    assert widget.current_donor_line.get_visible() is True

    # Simulate switching to another tab (artist should be hidden)
    widget.set_artists_visible(False)
    assert widget.current_donor_line.get_visible() is False

    # Simulate switching back to FRET tab (artist should be visible again)
    widget.set_artists_visible(True)
    assert widget.current_donor_line.get_visible() is True

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)


def test_colormap_events(make_napari_viewer):
    """Test colormap and contrast limit event handling."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Setup FRET calculation parameters
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    # Calculate FRET efficiency to create the FRET layer
    widget.calculate_fret_efficiency_button.click()

    fret_layer_name = f"FRET efficiency: test_layer"
    assert fret_layer_name in [layer.name for layer in viewer.layers]

    # Get the created FRET layer
    fret_layer = viewer.layers[fret_layer_name]
    widget.fret_layer = fret_layer

    # Test initial colormap properties
    initial_colormap = fret_layer.colormap.name
    initial_contrast_limits = fret_layer.contrast_limits

    # Verify initial state is captured
    assert initial_colormap is not None
    assert initial_contrast_limits is not None
    assert len(initial_contrast_limits) == 2

    # Test colormap change
    new_colormap = 'viridis'
    # Ensure we're actually changing to a different colormap
    if initial_colormap == new_colormap:
        new_colormap = 'plasma'

    fret_layer.colormap = new_colormap
    mock_event = Mock()
    mock_event.source = fret_layer
    widget._on_colormap_changed(mock_event)

    # Check that the widget's colormap property was updated
    assert (
        widget.fret_colormap is not None
    )  # Should be updated with new colors
    assert widget.fret_layer.colormap.name == new_colormap
    assert (
        widget.fret_layer.colormap.name != initial_colormap
    )  # Should have changed

    # Test contrast limits change
    new_contrast_limits = [0.2, 0.8]
    # Ensure we're actually changing the contrast limits
    if initial_contrast_limits == new_contrast_limits:
        new_contrast_limits = [0.3, 0.9]

    fret_layer.contrast_limits = new_contrast_limits
    widget._on_contrast_limits_changed(mock_event)

    # Check that the widget's contrast limits were updated
    assert widget.colormap_contrast_limits == new_contrast_limits
    assert (
        widget.colormap_contrast_limits != initial_contrast_limits
    )  # Should have changed


def test_draw_colormap_trajectory(make_napari_viewer):
    """Test drawing trajectory with colormap."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Create mock axes
    ax_mock = Mock()

    # Test trajectory data
    trajectory_real = np.linspace(0.1, 0.9, 100)
    trajectory_imag = np.linspace(0.1, 0.5, 100)

    # Set up FRET layer for colormap
    mock_layer = Mock()
    mock_layer.contrast_limits = (0.0, 1.0)
    widget.fret_layer = mock_layer
    widget.colormap_contrast_limits = (0.0, 1.0)

    # Draw colormap trajectory
    widget._draw_colormap_trajectory(ax_mock, trajectory_real, trajectory_imag)

    # Should call add_collection on axes
    assert ax_mock.add_collection.called


def test_fret_widget_layer_replacement(make_napari_viewer):
    """Test that existing FRET layers are replaced when calculating new ones."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Setup widget
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    # Calculate FRET efficiency first time
    widget.calculate_fret_efficiency_button.click()

    fret_layer_name = f"FRET efficiency: test_layer"
    assert fret_layer_name in [layer.name for layer in viewer.layers]

    # Count layers before second calculation
    initial_layer_count = len(viewer.layers)

    # Change values
    widget.donor_line_edit.setText("1.5")
    widget.frequency_input.setText("90")
    widget.background_real_edit.setText("0.2")
    widget.background_imag_edit.setText("0.2")

    # Calculate again - should replace existing layer
    widget.calculate_fret_efficiency_button.click()

    # Should not have added an extra layer
    assert len(viewer.layers) == initial_layer_count
    assert fret_layer_name in [layer.name for layer in viewer.layers]


def test_harmonic_change_updates_trajectory(make_napari_viewer):
    """Test that changing harmonics updates the donor trajectory."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Set up initial parameters
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    # Mock the canvas and figure
    parent.canvas_widget = Mock()
    parent.canvas_widget.figure = Mock()
    ax_mock = Mock()
    parent.canvas_widget.figure.gca.return_value = ax_mock
    parent.canvas_widget.canvas = Mock()

    # Set initial harmonic to 1
    parent.harmonic = 1
    widget.current_harmonic = 1

    # Plot initial trajectory
    widget.plot_donor_trajectory()
    initial_frequency = widget.frequency

    # Verify initial state
    assert widget.current_harmonic == 1
    assert initial_frequency == 80.0  # base_frequency * harmonic (80 * 1)

    # Change harmonic to 2
    parent.harmonic = 2
    widget._on_harmonic_changed()

    # Verify harmonic updated
    assert widget.current_harmonic == 2
    assert widget.frequency == 160.0  # base_frequency * harmonic (80 * 2)

    # Change harmonic to 3
    parent.harmonic = 3
    widget._on_harmonic_changed()

    # Verify harmonic updated again
    assert widget.current_harmonic == 3
    assert widget.frequency == 240.0  # base_frequency * harmonic (80 * 3)


def test_background_position_storage_by_harmonic(make_napari_viewer):
    """Test that background positions are stored and retrieved by harmonic."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Set initial harmonic and background position
    parent.harmonic = 1
    widget.current_harmonic = 1
    widget.background_real_edit.setText("0.2")
    widget.background_imag_edit.setText("0.3")

    # Store current position for harmonic 1
    widget._store_current_background_position()

    # Verify position was stored
    assert 1 in widget.background_positions_by_harmonic
    assert widget.background_positions_by_harmonic[1]['real'] == 0.2
    assert widget.background_positions_by_harmonic[1]['imag'] == 0.3

    # Change to harmonic 2 (should get default position)
    parent.harmonic = 2
    widget._on_harmonic_changed()

    # Should have default position for harmonic 2
    assert widget.background_real_edit.text() == "0.000"
    assert widget.background_imag_edit.text() == "0.000"
    assert widget.current_harmonic == 2

    # Set different position for harmonic 2
    widget.background_real_edit.setText("0.5")
    widget.background_imag_edit.setText("0.6")
    widget._store_current_background_position()

    # Verify position stored for harmonic 2
    assert 2 in widget.background_positions_by_harmonic
    assert widget.background_positions_by_harmonic[2]['real'] == 0.5
    assert widget.background_positions_by_harmonic[2]['imag'] == 0.6

    # Switch back to harmonic 1
    parent.harmonic = 1
    widget._on_harmonic_changed()

    # Should restore original position for harmonic 1
    assert widget.background_real_edit.text() == "0.200"
    assert widget.background_imag_edit.text() == "0.300"
    assert widget.current_harmonic == 1


def test_trajectory_calculation_with_different_harmonics(make_napari_viewer):
    """Test that trajectory calculations use effective frequency (base * harmonic)."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Set base parameters
    base_frequency = 80.0
    donor_lifetime = 2.0
    widget.donor_line_edit.setText(str(donor_lifetime))
    widget.frequency_input.setText(str(base_frequency))
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    # Test harmonic 1
    parent.harmonic = 1
    widget._on_parameters_changed()
    trajectory_h1_real, trajectory_h1_imag = phasor_from_fret_donor(
        base_frequency * 1,  # effective frequency
        donor_lifetime,
        fret_efficiency=widget._fret_efficiencies,
        donor_background=widget.donor_background,
        background_imag=0.1,
        background_real=0.1,
        donor_fretting=widget.donor_fretting_proportion,
    )

    # Test harmonic 2
    parent.harmonic = 2
    widget._on_parameters_changed()
    trajectory_h2_real, trajectory_h2_imag = phasor_from_fret_donor(
        base_frequency * 2,  # effective frequency
        donor_lifetime,
        fret_efficiency=widget._fret_efficiencies,
        donor_background=widget.donor_background,
        background_imag=0.1,
        background_real=0.1,
        donor_fretting=widget.donor_fretting_proportion,
    )

    # Test harmonic 3
    parent.harmonic = 3
    widget._on_parameters_changed()
    trajectory_h3_real, trajectory_h3_imag = phasor_from_fret_donor(
        base_frequency * 3,  # effective frequency
        donor_lifetime,
        fret_efficiency=widget._fret_efficiencies,
        donor_background=widget.donor_background,
        background_imag=0.1,
        background_real=0.1,
        donor_fretting=widget.donor_fretting_proportion,
    )

    # Trajectories should be different for different harmonics
    assert not np.array_equal(trajectory_h1_real, trajectory_h2_real)
    assert not np.array_equal(trajectory_h1_real, trajectory_h3_real)
    assert not np.array_equal(trajectory_h2_real, trajectory_h3_real)


def test_fret_efficiency_calculation_with_harmonics(make_napari_viewer):
    """Test FRET efficiency calculation respects harmonic changes."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Setup widget parameters
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    # Test with harmonic 1
    parent.harmonic = 1
    widget._on_harmonic_changed()
    widget.calculate_fret_efficiency_button.click()

    fret_layer_name_h1 = f"FRET efficiency: test_layer"
    assert fret_layer_name_h1 in [layer.name for layer in viewer.layers]
    fret_data_h1 = viewer.layers[fret_layer_name_h1].data.copy()

    # Change to harmonic 2 and recalculate
    parent.harmonic = 2
    widget._on_harmonic_changed()
    widget.calculate_fret_efficiency_button.click()

    # Same layer name but data should be different
    fret_data_h2 = viewer.layers[fret_layer_name_h1].data.copy()

    # FRET efficiency should be different for different harmonics
    assert not np.array_equal(fret_data_h1, fret_data_h2)


def test_background_position_manual_changes_stored_by_harmonic(
    make_napari_viewer,
):
    """Test that manual background position changes are stored per harmonic."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Start with harmonic 1
    parent.harmonic = 1
    widget.current_harmonic = 1

    # Manually change background position
    widget.background_real_edit.setText("0.15")
    widget.background_imag_edit.setText("0.25")
    widget._on_background_position_changed()

    # Verify stored for harmonic 1
    assert 1 in widget.background_positions_by_harmonic
    assert widget.background_positions_by_harmonic[1]['real'] == 0.15
    assert widget.background_positions_by_harmonic[1]['imag'] == 0.25

    # Switch to harmonic 2
    parent.harmonic = 2
    widget._on_harmonic_changed()

    # Should be default for harmonic 2
    assert widget.background_real_edit.text() == "0.000"
    assert widget.background_imag_edit.text() == "0.000"

    # Set different values for harmonic 2
    widget.background_real_edit.setText("0.35")
    widget.background_imag_edit.setText("0.45")
    widget._on_background_position_changed()

    # Switch back to harmonic 1
    parent.harmonic = 1
    widget._on_harmonic_changed()

    # Should restore harmonic 1 values
    assert float(widget.background_real_edit.text()) == 0.15
    assert float(widget.background_imag_edit.text()) == 0.25


@pytest.mark.parametrize(
    "donor_lifetime,frequency,expected_message,message_type",
    [
        # Empty values - these should be warnings
        ("", "80", "Enter a Donor lifetime value.", "warning"),
        ("2.0", "", "Enter a frequency value.", "warning"),
        ("", "", "Enter a Donor lifetime value.", "warning"),
        # Whitespace only values - these should be warnings
        ("   ", "80", "Enter a Donor lifetime value.", "warning"),
        ("2.0", "   ", "Enter a frequency value.", "warning"),
        ("   ", "   ", "Enter a Donor lifetime value.", "warning"),
        # Invalid numeric values - these should be errors
        (
            "not_a_number",
            "80",
            "Enter valid numeric values for donor lifetime and frequency.",
            "error",
        ),
        (
            "2.0",
            "invalid_frequency",
            "Enter valid numeric values for donor lifetime and frequency.",
            "error",
        ),
        (
            "invalid_lifetime",
            "invalid_frequency",
            "Enter valid numeric values for donor lifetime and frequency.",
            "error",
        ),
        (
            "abc",
            "xyz",
            "Enter valid numeric values for donor lifetime and frequency.",
            "error",
        ),
        # Mixed invalid cases - empty/whitespace takes precedence, so warnings
        ("", "invalid_frequency", "Enter a Donor lifetime value.", "warning"),
        ("   ", "not_a_number", "Enter a Donor lifetime value.", "warning"),
    ],
)
def test_calculate_fret_efficiency_invalid_inputs(
    make_napari_viewer,
    donor_lifetime,
    frequency,
    expected_message,
    message_type,
):
    """Test FRET efficiency calculation with various invalid inputs."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    widget.donor_line_edit.setText(donor_lifetime)
    widget.frequency_input.setText(frequency)
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    initial_layer_count = len(viewer.layers)

    if message_type == "warning":
        with patch(
            'napari_phasors.fret_tab.show_warning'
        ) as mock_show_warning:
            widget.calculate_fret_efficiency()
            mock_show_warning.assert_called_once_with(expected_message)
    else:  # message_type == "error"
        with patch('napari_phasors.fret_tab.show_error') as mock_show_error:
            widget.calculate_fret_efficiency()
            mock_show_error.assert_called_once_with(expected_message)

    assert len(viewer.layers) == initial_layer_count

    fret_layer_name = f"FRET efficiency: test_layer"
    assert fret_layer_name not in [layer.name for layer in viewer.layers]

    assert widget.fret_layer is None


def test_donor_lifetime_combobox_initialization(make_napari_viewer):
    """Test that donor lifetime combobox is properly initialized."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Initially should have "Select layer..." option
    assert widget.donor_lifetime_combobox.count() >= 1
    assert widget.donor_lifetime_combobox.itemText(0) == "Select layer..."


def test_donor_lifetime_combobox_updates_with_layers(make_napari_viewer):
    """Test that donor lifetime combobox updates when layers are added/removed."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Initial state
    initial_count = widget.donor_lifetime_combobox.count()

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Combobox should be updated
    new_count = widget.donor_lifetime_combobox.count()
    assert new_count == initial_count + 1

    # Check that the new layer is in the combobox
    combobox_items = [
        widget.donor_lifetime_combobox.itemText(i)
        for i in range(widget.donor_lifetime_combobox.count())
    ]
    assert "test_layer" in combobox_items

    # Remove layer
    viewer.layers.remove(test_layer)

    # Combobox should be updated back to original count
    assert widget.donor_lifetime_combobox.count() == initial_count


def test_lifetime_type_combobox_modes(make_napari_viewer):
    """Test the lifetime type combobox modes."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Test changing modes
    widget.lifetime_type_combobox.setCurrentText(
        "Apparent Modulation Lifetime"
    )
    assert (
        widget.lifetime_type_combobox.currentText()
        == "Apparent Modulation Lifetime"
    )

    widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")
    assert widget.lifetime_type_combobox.currentText() == "Normal Lifetime"

    widget.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
    assert (
        widget.lifetime_type_combobox.currentText()
        == "Apparent Phase Lifetime"
    )


def test_calculate_donor_lifetime_no_layer_selected(make_napari_viewer):
    """Test donor lifetime calculation with no layer selected."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Set frequency for calculations
    widget.frequency_input.setText("80")

    # Should return without error when "None" is selected
    widget.donor_lifetime_combobox.setCurrentText("None")
    initial_lifetime = widget.donor_line_edit.text()

    widget._calculate_donor_lifetime()

    # Lifetime should remain unchanged
    assert widget.donor_line_edit.text() == initial_lifetime


def test_calculate_donor_lifetime_no_frequency(make_napari_viewer):
    """Test donor lifetime calculation with no frequency set."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Select the layer but don't set frequency
    widget.donor_lifetime_combobox.setCurrentText("test_layer")
    widget.frequency_input.setText("")  # No frequency

    initial_lifetime = widget.donor_line_edit.text()

    # Should return without error
    widget._calculate_donor_lifetime()

    # Lifetime should remain unchanged
    assert widget.donor_line_edit.text() == initial_lifetime


def test_calculate_donor_lifetime_apparent_phase(make_napari_viewer):
    """Test donor lifetime calculation using apparent phase lifetime."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Set up parameters
    widget.frequency_input.setText("80")
    widget.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
    widget.donor_lifetime_combobox.setCurrentText("test_layer")

    # Set up parent widget properly
    parent.harmonic = 1

    # Calculate donor lifetime
    widget._calculate_donor_lifetime()

    # Should have updated the donor lifetime
    assert widget.donor_line_edit.text() != ""
    lifetime_value = float(widget.donor_line_edit.text())
    assert lifetime_value > 0
    assert widget.donor_lifetime == lifetime_value


def test_calculate_donor_lifetime_apparent_modulation(make_napari_viewer):
    """Test donor lifetime calculation using apparent modulation lifetime."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Set up parameters
    widget.frequency_input.setText("80")
    widget.lifetime_type_combobox.setCurrentText(
        "Apparent Modulation Lifetime"
    )
    widget.donor_lifetime_combobox.setCurrentText("test_layer")

    # Set up parent widget properly
    parent.harmonic = 1

    # Calculate donor lifetime
    widget._calculate_donor_lifetime()

    # Should have updated the donor lifetime
    assert widget.donor_line_edit.text() != ""
    lifetime_value = float(widget.donor_line_edit.text())
    assert lifetime_value > 0
    assert widget.donor_lifetime == lifetime_value


def test_calculate_donor_lifetime_normal_lifetime(make_napari_viewer):
    """Test donor lifetime calculation using normal lifetime."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Set up parameters
    widget.frequency_input.setText("80")
    widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")
    widget.donor_lifetime_combobox.setCurrentText("test_layer")

    # Set up parent widget properly
    parent.harmonic = 1

    # Calculate donor lifetime
    widget._calculate_donor_lifetime()

    # Should have updated the donor lifetime
    assert widget.donor_line_edit.text() != ""
    lifetime_value = float(widget.donor_line_edit.text())
    assert lifetime_value > 0
    assert widget.donor_lifetime == lifetime_value


def test_calculate_donor_lifetime_different_harmonics(make_napari_viewer):
    """Test donor lifetime calculation with different harmonics."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Set up parameters
    widget.frequency_input.setText("80")
    widget.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
    widget.donor_lifetime_combobox.setCurrentText("test_layer")

    # Test with harmonic 1
    parent.harmonic = 1
    widget._calculate_donor_lifetime()
    lifetime_h1 = widget.donor_line_edit.text()

    # Test with harmonic 2
    parent.harmonic = 2
    widget._calculate_donor_lifetime()
    lifetime_h2 = widget.donor_line_edit.text()

    # Test with harmonic 3
    parent.harmonic = 3
    widget._calculate_donor_lifetime()
    lifetime_h3 = widget.donor_line_edit.text()

    # Lifetimes should be different for different harmonics
    assert lifetime_h1 != lifetime_h2
    assert lifetime_h1 != lifetime_h3
    assert lifetime_h2 != lifetime_h3


def test_calculate_donor_lifetime_mode_differences(make_napari_viewer):
    """Test that different lifetime modes give different results."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Set up parameters
    widget.frequency_input.setText("80")
    widget.donor_lifetime_combobox.setCurrentText("test_layer")
    parent.harmonic = 1

    # Test apparent phase lifetime
    widget.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
    widget._calculate_donor_lifetime()
    phase_lifetime = widget.donor_line_edit.text()

    # Test apparent modulation lifetime
    widget.lifetime_type_combobox.setCurrentText(
        "Apparent Modulation Lifetime"
    )
    widget._calculate_donor_lifetime()
    mod_lifetime = widget.donor_line_edit.text()

    # Test normal lifetime
    widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")
    widget._calculate_donor_lifetime()
    normal_lifetime = widget.donor_line_edit.text()

    # All should be valid numbers
    assert phase_lifetime != ""
    assert mod_lifetime != ""
    assert normal_lifetime != ""

    phase_val = float(phase_lifetime)
    mod_val = float(mod_lifetime)
    normal_val = float(normal_lifetime)

    assert phase_val > 0
    assert mod_val > 0
    assert normal_val > 0

    # They should generally be different (though could be close)
    # At least one should be different from the others
    assert not (phase_val == mod_val == normal_val)


def test_donor_lifetime_combobox_layer_selection_persistence(
    make_napari_viewer,
):
    """Test that donor lifetime combobox selection persists when layers change."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add first layer
    test_layer1 = create_image_layer_with_phasors()
    test_layer1.name = "test_layer1"
    viewer.add_layer(test_layer1)

    # Select first layer
    widget.donor_lifetime_combobox.setCurrentText("test_layer1")
    assert widget.donor_lifetime_combobox.currentText() == "test_layer1"

    # Add second layer
    test_layer2 = create_image_layer_with_phasors()
    test_layer2.name = "test_layer2"
    viewer.add_layer(test_layer2)

    # Selection should persist
    assert widget.donor_lifetime_combobox.currentText() == "test_layer1"

    # Change to second layer
    widget.donor_lifetime_combobox.setCurrentText("test_layer2")
    assert widget.donor_lifetime_combobox.currentText() == "test_layer2"

    # Remove first layer
    viewer.layers.remove(test_layer1)

    # Selection should still be test_layer2
    assert widget.donor_lifetime_combobox.currentText() == "test_layer2"

    # Remove second layer
    viewer.layers.remove(test_layer2)

    # Should revert to "Select layer..."
    assert widget.donor_lifetime_combobox.currentText() == "Select layer..."


def test_calculate_donor_lifetime_error_handling(make_napari_viewer):
    """Test error handling in donor lifetime calculation."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Create a layer without proper phasor data
    import numpy as np
    from napari.layers import Image

    bad_layer = Image(np.random.random((10, 10)), name="bad_layer")
    viewer.add_layer(bad_layer)

    # Try to calculate with bad layer
    widget.frequency_input.setText("80")
    widget.donor_lifetime_combobox.setCurrentText("bad_layer")
    parent.harmonic = 1

    initial_lifetime = widget.donor_line_edit.text()

    # Should handle error gracefully
    widget._calculate_donor_lifetime()

    # Should not crash and lifetime should remain unchanged
    assert widget.donor_line_edit.text() == initial_lifetime


def test_donor_source_selector_functionality(make_napari_viewer):
    """Test the donor source selector switches between Manual and From layer modes."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Initially should be in Manual mode
    assert widget.donor_source_selector.currentText() == "Manual"
    assert widget.donor_stack.currentIndex() == 0  # Manual page
    assert widget.donor_label.text() == "Donor lifetime (ns):"

    # Switch to "From layer" mode
    widget.donor_source_selector.setCurrentText("From layer")
    widget._on_donor_source_changed(1)

    assert widget.donor_stack.currentIndex() == 1  # From layer page
    assert widget.donor_label.text() == "Donor lifetime (ns):"

    # Switch back to Manual mode
    widget.donor_source_selector.setCurrentText("Manual")
    widget._on_donor_source_changed(0)

    assert widget.donor_stack.currentIndex() == 0  # Manual page
    assert widget.donor_label.text() == "Donor lifetime (ns):"


def test_background_source_selector_functionality(make_napari_viewer):
    """Test the background source selector switches between Manual and From layer modes."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Initially should be in Manual mode
    assert widget.bg_source_selector.currentText() == "Manual"
    assert widget.bg_stack.currentIndex() == 0  # Manual page
    assert widget.background_position_label.text() == "Background position:"

    # Switch to "From layer" mode
    widget.bg_source_selector.setCurrentText("From layer")
    widget._on_bg_source_changed(1)

    assert widget.bg_stack.currentIndex() == 1  # From layer page
    assert widget.background_position_label.text() == "Background position:"

    # Switch back to Manual mode
    widget.bg_source_selector.setCurrentText("Manual")
    widget._on_bg_source_changed(0)

    assert widget.bg_stack.currentIndex() == 0  # Manual page
    assert widget.background_position_label.text() == "Background position:"


def test_donor_lifetime_label_updates_with_layer_calculation(
    make_napari_viewer,
):
    """Test that donor lifetime label updates when calculated from layer."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Set up parameters
    widget.frequency_input.setText("80")
    parent.harmonic = 1

    # Switch to "From layer" mode
    widget.donor_source_selector.setCurrentText("From layer")
    widget._on_donor_source_changed(1)

    # Select the test layer
    widget.donor_lifetime_combobox.setCurrentText("test_layer")

    # Calculate donor lifetime
    widget._calculate_donor_lifetime()

    # Label should now show the calculated lifetime value
    label_text = widget.donor_label.text()
    assert "Donor lifetime (from layer):" in label_text
    assert "ns" in label_text

    # Switch back to Manual mode
    widget.donor_source_selector.setCurrentText("Manual")
    widget._on_donor_source_changed(0)

    # Label should revert to default
    assert widget.donor_label.text() == "Donor lifetime (ns):"


def test_background_position_label_updates_with_layer_calculation(
    make_napari_viewer,
):
    """Test that background position label updates when calculated from layer."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Set up parameters
    parent.harmonic = 1

    # Switch to "From layer" mode
    widget.bg_source_selector.setCurrentText("From layer")
    widget._on_bg_source_changed(1)

    # Select the test layer
    widget.background_image_combobox.setCurrentText("test_layer")

    # Calculate background position
    widget._calculate_background_position()

    # Label should now show the calculated G and S values
    label_text = widget.background_position_label.text()
    assert "Background position: G=" in label_text
    assert "S=" in label_text

    # Switch back to Manual mode
    widget.bg_source_selector.setCurrentText("Manual")
    widget._on_bg_source_changed(0)

    # Label should revert to default
    assert widget.background_position_label.text() == "Background position:"


def test_layer_selection_with_no_valid_layers(make_napari_viewer):
    """Test behavior when no valid layers are available for selection."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add a layer without phasor data
    import numpy as np
    from napari.layers import Image

    invalid_layer = Image(np.random.random((10, 10)), name="invalid_layer")
    viewer.add_layer(invalid_layer)

    # Update comboboxes
    widget._update_donor_lifetime_combobox()
    widget._update_background_combobox()

    # Should only have "Select layer..." option
    assert widget.donor_lifetime_combobox.count() == 1
    assert widget.donor_lifetime_combobox.itemText(0) == "Select layer..."
    assert widget.background_image_combobox.count() == 1
    assert widget.background_image_combobox.itemText(0) == "Select layer..."


def test_calculate_values_with_select_layer_option(make_napari_viewer):
    """Test calculation behavior when 'Select layer...' is selected."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Set frequency for donor calculation
    widget.frequency_input.setText("80")
    parent.harmonic = 1

    # Switch to From layer modes
    widget.donor_source_selector.setCurrentText("From layer")
    widget._on_donor_source_changed(1)
    widget.bg_source_selector.setCurrentText("From layer")
    widget._on_bg_source_changed(1)

    # Ensure "Select layer..." is selected
    widget.donor_lifetime_combobox.setCurrentText("Select layer...")
    widget.background_image_combobox.setCurrentText("Select layer...")

    # Store initial values
    initial_donor_text = widget.donor_line_edit.text()
    initial_donor_label = widget.donor_label.text()
    initial_bg_real = widget.background_real_edit.text()
    initial_bg_imag = widget.background_imag_edit.text()
    initial_bg_label = widget.background_position_label.text()

    # Try to calculate - should not crash and should not change values
    widget._calculate_donor_lifetime()
    widget._calculate_background_position()

    # Values should remain unchanged
    assert widget.donor_line_edit.text() == initial_donor_text
    assert widget.donor_label.text() == initial_donor_label
    assert widget.background_real_edit.text() == initial_bg_real
    assert widget.background_imag_edit.text() == initial_bg_imag
    assert widget.background_position_label.text() == initial_bg_label


def test_ui_mode_switching_preserves_manual_values(make_napari_viewer):
    """Test that switching between modes preserves manually entered values."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Set manual values
    widget.donor_line_edit.setText("2.5")
    widget.background_real_edit.setText("0.3")
    widget.background_imag_edit.setText("0.4")

    # Switch to From layer mode and back
    widget.donor_source_selector.setCurrentText("From layer")
    widget._on_donor_source_changed(1)
    widget.donor_source_selector.setCurrentText("Manual")
    widget._on_donor_source_changed(0)

    widget.bg_source_selector.setCurrentText("From layer")
    widget._on_bg_source_changed(1)
    widget.bg_source_selector.setCurrentText("Manual")
    widget._on_bg_source_changed(0)

    # Manual values should be preserved
    assert widget.donor_line_edit.text() == "2.5"
    assert widget.background_real_edit.text() == "0.3"
    assert widget.background_imag_edit.text() == "0.4"


def test_metadata_initialization(make_napari_viewer):
    """Test that FRET settings are only initialized when analysis is performed."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer with phasor data
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    # Select the layer
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "test_layer"
    )

    # Trigger layer change - should NOT initialize FRET metadata
    widget._on_image_layer_changed()

    # Check that FRET settings were NOT initialized
    if 'settings' in test_layer.metadata:
        assert 'fret' not in test_layer.metadata['settings']

    # Now perform FRET analysis
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    # Calculate FRET efficiency - this should initialize metadata
    widget.calculate_fret_efficiency_button.click()

    # Now check that settings were initialized
    assert 'settings' in test_layer.metadata
    assert 'fret' in test_layer.metadata['settings']

    # Verify default values
    fret_settings = test_layer.metadata['settings']['fret']
    assert fret_settings['donor_lifetime'] == 2.0
    assert test_layer.metadata['settings']['frequency'] == '80'
    assert fret_settings['donor_background'] == 0.1
    assert fret_settings['donor_fretting_proportion'] == 1.0
    assert fret_settings['use_colormap'] is True
    assert fret_settings['background_positions_by_harmonic'] == {
        1: {'imag': 0.1, 'real': 0.1}
    }

    # Check colormap settings
    assert 'colormap_settings' in fret_settings
    assert fret_settings['colormap_settings']['colormap_name'] == 'viridis'
    assert fret_settings['colormap_settings']['colormap_colors'] is None
    assert fret_settings['colormap_settings']['contrast_limits'] == [0, 1]
    assert fret_settings['colormap_settings']['colormap_changed'] is False


def test_metadata_storage_manual_values(make_napari_viewer):
    """Test that manual values are correctly stored in metadata."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "test_layer"
    )
    widget._on_image_layer_changed()

    # Set manual values
    widget.donor_line_edit.setText("2.5")
    widget.frequency_input.setText("85")
    widget.background_real_edit.setText("0.15")
    widget.background_imag_edit.setText("0.25")
    widget.background_slider.setValue(30)  # 0.3
    widget.fretting_slider.setValue(75)  # 0.75
    widget.colormap_checkbox.setChecked(False)

    # Trigger updates
    parent._broadcast_frequency_value_across_tabs('85')
    widget._on_parameters_changed()
    widget._on_background_position_changed()  # Explicitly store background position
    widget._on_background_slider_changed()
    widget._on_fretting_slider_changed()
    widget._on_colormap_checkbox_changed()

    # Check metadata
    fret_settings = test_layer.metadata['settings']['fret']
    assert fret_settings['donor_lifetime'] == 2.5
    assert test_layer.metadata['settings']['frequency'] == 85.0
    assert fret_settings['donor_background'] == 0.3
    assert fret_settings['donor_fretting_proportion'] == 0.75
    assert fret_settings['use_colormap'] is False
    # Background positions should be stored in background_positions_by_harmonic
    assert 1 in fret_settings['background_positions_by_harmonic']
    assert fret_settings['background_positions_by_harmonic'][1]['real'] == 0.15
    assert fret_settings['background_positions_by_harmonic'][1]['imag'] == 0.25


def test_metadata_storage_background_positions_by_harmonic(make_napari_viewer):
    """Test that background positions by harmonic are correctly stored."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "test_layer"
    )
    widget._on_image_layer_changed()

    # Set background for harmonic 1
    parent.harmonic = 1
    widget._on_harmonic_changed()
    widget.background_real_edit.setText("0.2")
    widget.background_imag_edit.setText("0.3")
    widget._on_background_position_changed()

    # Set background for harmonic 2
    parent.harmonic = 2
    widget._on_harmonic_changed()
    widget.background_real_edit.setText("0.5")
    widget.background_imag_edit.setText("0.6")
    widget._on_background_position_changed()

    # Check metadata
    fret_settings = test_layer.metadata['settings']['fret']
    bg_positions = fret_settings['background_positions_by_harmonic']

    assert 1 in bg_positions
    assert bg_positions[1]['real'] == 0.2
    assert bg_positions[1]['imag'] == 0.3

    assert 2 in bg_positions
    assert bg_positions[2]['real'] == 0.5
    assert bg_positions[2]['imag'] == 0.6


def test_metadata_storage_from_layer_mode(make_napari_viewer):
    """Test that 'From layer' mode settings are stored in metadata."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layers
    donor_layer = create_image_layer_with_phasors()
    donor_layer.name = "donor_layer"
    viewer.add_layer(donor_layer)

    bg_layer = create_image_layer_with_phasors()
    bg_layer.name = "bg_layer"
    viewer.add_layer(bg_layer)

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "donor_layer"
    )
    widget._on_image_layer_changed()
    widget.frequency_input.setText("80")

    # Switch to From layer mode for donor
    widget.donor_source_selector.setCurrentText("From layer")
    widget._on_donor_source_changed(1)
    widget.donor_lifetime_combobox.setCurrentText("donor_layer")
    widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")

    # Switch to From layer mode for background
    widget.bg_source_selector.setCurrentText("From layer")
    widget._on_bg_source_changed(1)
    widget.background_image_combobox.setCurrentText("bg_layer")

    # Check metadata
    fret_settings = donor_layer.metadata['settings']['fret']
    assert fret_settings['donor_source'] == 'From layer'
    assert fret_settings['donor_layer_name'] == 'donor_layer'
    assert fret_settings['donor_lifetime_type'] == 'Normal Lifetime'
    assert fret_settings['background_source'] == 'From layer'
    assert fret_settings['background_layer_name'] == 'bg_layer'


def test_metadata_restoration_manual_values(make_napari_viewer):
    """Test that manual values are correctly restored from metadata."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "test_layer"
    )
    widget._on_image_layer_changed()

    # Set and store values
    widget.donor_line_edit.setText("3.5")
    widget.frequency_input.setText("90")
    widget.background_slider.setValue(40)  # 0.4
    widget.fretting_slider.setValue(85)  # 0.85
    widget.colormap_checkbox.setChecked(False)

    parent._broadcast_frequency_value_across_tabs('90')
    widget._on_parameters_changed()
    widget._on_background_slider_changed()
    widget._on_fretting_slider_changed()
    widget._on_colormap_checkbox_changed()

    parent.harmonic = 1
    widget._on_harmonic_changed()
    widget.background_real_edit.setText("0.35")
    widget.background_imag_edit.setText("0.45")
    widget._on_background_position_changed()

    # Switch to another layer and back
    parent.image_layer_with_phasor_features_combobox.setCurrentText("")
    widget._on_image_layer_changed()

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "test_layer"
    )
    widget._on_image_layer_changed()

    # Verify restoration (comparing float values, not exact string format)
    assert float(widget.donor_line_edit.text()) == 3.5
    assert float(widget.frequency_input.text()) == 90.0
    assert widget.background_slider.value() == 40
    assert widget.background_label.text() == "0.40"
    assert widget.fretting_slider.value() == 85
    assert widget.fretting_label.text() == "0.85"
    assert widget.colormap_checkbox.isChecked() is False
    assert float(widget.background_real_edit.text()) == 0.35
    assert float(widget.background_imag_edit.text()) == 0.45


def test_metadata_restoration_background_positions_by_harmonic(
    make_napari_viewer,
):
    """Test that background positions by harmonic are correctly restored."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "test_layer"
    )
    widget._on_image_layer_changed()

    # Set positions for different harmonics
    parent.harmonic = 1
    widget._on_harmonic_changed()
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.2")
    widget._on_background_position_changed()

    parent.harmonic = 2
    widget._on_harmonic_changed()
    widget.background_real_edit.setText("0.3")
    widget.background_imag_edit.setText("0.4")
    widget._on_background_position_changed()

    parent.harmonic = 3
    widget._on_harmonic_changed()
    widget.background_real_edit.setText("0.5")
    widget.background_imag_edit.setText("0.6")
    widget._on_background_position_changed()

    # Switch to another layer and back
    parent.image_layer_with_phasor_features_combobox.setCurrentText("")
    widget._on_image_layer_changed()

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "test_layer"
    )
    widget._on_image_layer_changed()

    # Verify restoration for each harmonic
    parent.harmonic = 1
    widget._on_harmonic_changed()
    assert float(widget.background_real_edit.text()) == 0.1
    assert float(widget.background_imag_edit.text()) == 0.2

    parent.harmonic = 2
    widget._on_harmonic_changed()
    assert float(widget.background_real_edit.text()) == 0.3
    assert float(widget.background_imag_edit.text()) == 0.4

    parent.harmonic = 3
    widget._on_harmonic_changed()
    assert float(widget.background_real_edit.text()) == 0.5
    assert float(widget.background_imag_edit.text()) == 0.6


def test_metadata_restoration_from_layer_mode(make_napari_viewer):
    """Test that 'From layer' mode settings are correctly restored."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layers
    donor_layer = create_image_layer_with_phasors()
    donor_layer.name = "donor_layer"
    viewer.add_layer(donor_layer)

    bg_layer = create_image_layer_with_phasors()
    bg_layer.name = "bg_layer"
    viewer.add_layer(bg_layer)

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "donor_layer"
    )
    widget._on_image_layer_changed()
    widget.frequency_input.setText("80")

    # Configure From layer mode
    widget.donor_source_selector.setCurrentText("From layer")
    widget._on_donor_source_changed(1)
    widget.donor_lifetime_combobox.setCurrentText("donor_layer")
    widget.lifetime_type_combobox.setCurrentText(
        "Apparent Modulation Lifetime"
    )

    widget.bg_source_selector.setCurrentText("From layer")
    widget._on_bg_source_changed(1)
    widget.background_image_combobox.setCurrentText("bg_layer")

    # Switch away and back
    parent.image_layer_with_phasor_features_combobox.setCurrentText("")
    widget._on_image_layer_changed()

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "donor_layer"
    )
    widget._on_image_layer_changed()

    # Verify restoration
    assert widget.donor_source_selector.currentText() == "From layer"
    assert widget.donor_lifetime_combobox.currentText() == "donor_layer"
    assert (
        widget.lifetime_type_combobox.currentText()
        == "Apparent Modulation Lifetime"
    )
    assert widget.bg_source_selector.currentText() == "From layer"
    assert widget.background_image_combobox.currentText() == "bg_layer"


def test_metadata_restoration_reverts_to_manual_when_layer_missing(
    make_napari_viewer,
):
    """Test that settings revert to manual mode when selected layers are missing."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layers
    donor_layer = create_image_layer_with_phasors()
    donor_layer.name = "donor_layer"
    viewer.add_layer(donor_layer)

    bg_layer = create_image_layer_with_phasors()
    bg_layer.name = "bg_layer"
    viewer.add_layer(bg_layer)

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "donor_layer"
    )
    widget._on_image_layer_changed()

    # Configure From layer mode
    widget.frequency_input.setText("80")
    widget.donor_line_edit.setText("2.5")
    widget.donor_source_selector.setCurrentText("From layer")
    widget._on_donor_source_changed(1)
    widget.donor_lifetime_combobox.setCurrentText("donor_layer")

    widget.bg_source_selector.setCurrentText("From layer")
    widget._on_bg_source_changed(1)
    widget.background_image_combobox.setCurrentText("bg_layer")

    # Remove the referenced layer
    viewer.layers.remove(bg_layer)

    # Switch away and back
    parent.image_layer_with_phasor_features_combobox.setCurrentText("")
    widget._on_image_layer_changed()

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "donor_layer"
    )
    widget._on_image_layer_changed()

    # Background should revert to Manual since bg_layer is missing
    assert widget.bg_source_selector.currentText() == "Manual"

    # Donor should still be From layer since donor_layer exists
    assert widget.donor_source_selector.currentText() == "From layer"


def test_metadata_colormap_settings_storage(make_napari_viewer):
    """Test that colormap settings are correctly stored in metadata."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add layer
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "test_layer"
    )
    widget._on_image_layer_changed()

    # Calculate FRET to create FRET layer
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")
    widget.calculate_fret_efficiency()

    # Change colormap settings
    widget.fret_layer.colormap = 'plasma'
    widget.fret_layer.contrast_limits = (0.2, 0.8)

    # Trigger colormap change events
    mock_event = Mock()
    mock_event.source = widget.fret_layer
    widget._on_colormap_changed(mock_event)
    widget._on_contrast_limits_changed(mock_event)

    # Check metadata
    colormap_settings = test_layer.metadata['settings']['fret'][
        'colormap_settings'
    ]
    assert colormap_settings['colormap_name'] == 'plasma'
    assert colormap_settings['contrast_limits'] == [0.2, 0.8]
    assert colormap_settings['colormap_changed'] is True


def test_metadata_persistence_across_layer_switches(make_napari_viewer):
    """Test that metadata persists correctly when switching between layers."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Add two layers
    layer1 = create_image_layer_with_phasors()
    layer1.name = "layer1"
    viewer.add_layer(layer1)

    layer2 = create_image_layer_with_phasors()
    layer2.name = "layer2"
    viewer.add_layer(layer2)

    # Configure layer 1
    parent.image_layer_with_phasor_features_combobox.setCurrentText("layer1")
    widget._on_image_layer_changed()
    widget.donor_line_edit.setText("2.5")
    widget.frequency_input.setText("80.0")
    widget.background_slider.setValue(30)
    parent._broadcast_frequency_value_across_tabs('80.0')
    widget._on_parameters_changed()
    widget._on_background_slider_changed()

    # Configure layer 2 with different values
    parent.image_layer_with_phasor_features_combobox.setCurrentText("layer2")
    widget._on_image_layer_changed()
    widget.donor_line_edit.setText("3.5")
    widget.frequency_input.setText("90.0")
    widget.background_slider.setValue(50)
    parent._broadcast_frequency_value_across_tabs('90.0')
    widget._on_parameters_changed()
    widget._on_background_slider_changed()

    # Switch back to layer 1
    parent.image_layer_with_phasor_features_combobox.setCurrentText("layer1")
    widget._on_image_layer_changed()

    # Verify layer 1 settings were restored
    assert widget.donor_line_edit.text() == "2.5"
    assert widget.frequency_input.text() == "80.0"
    assert widget.background_slider.value() == 30

    # Switch to layer 2
    parent.image_layer_with_phasor_features_combobox.setCurrentText("layer2")
    widget._on_image_layer_changed()

    # Verify layer 2 settings were restored
    assert widget.donor_line_edit.text() == "3.5"
    assert widget.frequency_input.text() == "90.0"
    assert widget.background_slider.value() == 50
