from unittest.mock import Mock

import numpy as np
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
    assert widget.background_real_edit.text() == "0.1"
    assert widget.background_imag_edit.text() == "0.1"
    assert (
        widget.calculate_fret_efficiency_button.text()
        == "Calculate FRET efficiency"
    )

    # Test slider initial values
    assert widget.background_slider.value() == 10  # 0.1 * 100
    assert widget.fretting_slider.value() == 100  # 1.0 * 100
    assert widget.colormap_checkbox.isChecked() is True


def test_fret_widget_parameter_updates(make_napari_viewer):
    """Test that parameters update correctly when UI changes."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Test donor lifetime input
    widget.donor_line_edit.setText("2.5")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.2")
    widget.background_imag_edit.setText("0.3")

    # Trigger parameter change
    widget._on_parameters_changed()

    assert widget.donor_lifetime == 2.5
    assert widget.frequency == 80 * parent.harmonic  # frequency * harmonic
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
    assert widget.background_real_edit.text() == "0.1"
    assert widget.background_imag_edit.text() == "0.1"


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
    assert "None" in combobox_items
    assert "test_layer" in combobox_items

    # Select the test layer in the background combobox
    widget.background_image_combobox.setCurrentText("test_layer")

    # Set up the parent widget properly
    parent.harmonic = 1
    parent._labels_layer_with_phasor_features = test_layer.metadata[
        'phasor_features_labels_layer'
    ]

    # Calculate background position
    widget._calculate_background_position()

    # Check that values were updated (they should be different from defaults)
    real_text = widget.background_real_edit.text()
    imag_text = widget.background_imag_edit.text()

    # Values should be numeric strings
    assert (
        real_text != "0.1" or imag_text != "0.1"
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

    # Alternative: Check that the formatted stored values match the display
    assert f"{stored_position['real']:.3f}" == real_text
    assert f"{stored_position['imag']:.3f}" == imag_text

    # Test with "None" selected - should not crash
    widget.background_image_combobox.setCurrentText("None")
    initial_real = widget.background_real_edit.text()
    initial_imag = widget.background_imag_edit.text()

    widget._calculate_background_position()

    # Values should remain unchanged when "None" is selected
    assert widget.background_real_edit.text() == initial_real
    assert widget.background_imag_edit.text() == initial_imag


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

    # Get values of G and S from the test layer
    phasor_data = parent._labels_layer_with_phasor_features.features
    harmonic_mask = phasor_data['harmonic'] == parent.harmonic
    real = phasor_data.loc[harmonic_mask, 'G']
    imag = phasor_data.loc[harmonic_mask, 'S']

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
        real.values,
        imag.values,
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
        real.values,
        imag.values,
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
