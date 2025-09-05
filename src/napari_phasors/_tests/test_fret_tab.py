from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
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
    assert widget.calculate_fret_efficiency_button.text() == "Calculate FRET efficiency"

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
    
    # Mock the parent's layer selection
    parent.image_layer_with_phasor_features_combobox = Mock()
    parent.image_layer_with_phasor_features_combobox.currentText.return_value = "test_layer"
    parent._labels_layer_with_phasor_features = test_layer.metadata['phasor_features_labels_layer']

    # Calculate background position
    widget._calculate_background_position()

    # Check that values were updated (they should be different from defaults)
    real_text = widget.background_real_edit.text()
    imag_text = widget.background_imag_edit.text()
    
    # Values should be numeric strings
    assert real_text != "0.1" or imag_text != "0.1"  # At least one should change
    assert float(real_text) >= 0
    assert float(imag_text) >= 0


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
    
    # Mock parent's layer selection
    parent.image_layer_with_phasor_features_combobox = Mock()
    parent.image_layer_with_phasor_features_combobox.currentText.return_value = "test_layer"
    parent._labels_layer_with_phasor_features = test_layer.metadata['phasor_features_labels_layer']

    # Create a mock donor line for trajectory data
    widget.current_donor_line = Mock()
    trajectory_data = np.array([[0.1, 0.2, 0.3], [0.1, 0.15, 0.2]]).T
    widget.current_donor_line.get_xydata.return_value = trajectory_data

    # Calculate FRET efficiency
    widget.calculate_fret_efficiency()

    # Check that a FRET layer was added
    fret_layer_name = f"FRET efficiency: test_layer"
    assert fret_layer_name in [layer.name for layer in viewer.layers]
    assert widget.fret_layer is not None


def test_fret_efficiency_calculation_with_line_collection(make_napari_viewer):
    """Test FRET efficiency calculation when using LineCollection (colormap mode)."""
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
    widget.frequency = 80.0 * parent.harmonic
    widget.donor_lifetime = 2.0
    
    # Mock parent's layer selection
    parent.image_layer_with_phasor_features_combobox = Mock()
    parent.image_layer_with_phasor_features_combobox.currentText.return_value = "test_layer"
    parent._labels_layer_with_phasor_features = test_layer.metadata['phasor_features_labels_layer']

    # Create a mock LineCollection (doesn't have get_xydata method)
    widget.current_donor_line = Mock()
    del widget.current_donor_line.get_xydata  # Remove the method to simulate LineCollection

    # Calculate FRET efficiency
    widget.calculate_fret_efficiency()

    # Check that a FRET layer was added
    fret_layer_name = f"FRET efficiency: test_layer"
    assert fret_layer_name in [layer.name for layer in viewer.layers]


def test_artist_management(make_napari_viewer):
    """Test artist visibility and management."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Initially no artists
    assert len(widget.get_all_artists()) == 0

    # Add a mock artist
    mock_artist = Mock()
    widget.current_donor_line = mock_artist

    # Check artists list
    artists = widget.get_all_artists()
    assert len(artists) == 1
    assert mock_artist in artists

    # Test visibility setting
    widget.set_artists_visible(False)
    mock_artist.set_visible.assert_called_with(False)

    widget.set_artists_visible(True)
    mock_artist.set_visible.assert_called_with(True)


def test_colormap_events(make_napari_viewer):
    """Test colormap and contrast limit event handling."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Create a mock FRET layer
    mock_layer = Mock()
    mock_layer.colormap.colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mock_layer.contrast_limits = (0.0, 1.0)
    widget.fret_layer = mock_layer

    # Create mock event
    mock_event = Mock()
    mock_event.source = mock_layer

    # Test colormap change
    widget._on_colormap_changed(mock_event)
    assert widget.fret_colormap == mock_layer.colormap.colors
    assert widget.colormap_contrast_limits == mock_layer.contrast_limits

    # Test contrast limits change
    mock_layer.contrast_limits = (0.2, 0.8)
    widget._on_contrast_limits_changed(mock_event)
    assert widget.colormap_contrast_limits == (0.2, 0.8)


def test_parameter_validation(make_napari_viewer):
    """Test parameter validation with invalid inputs."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # Test with invalid numeric inputs
    widget.donor_line_edit.setText("invalid")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    # Should not raise exception
    widget._on_parameters_changed()

    # Frequency should not be updated due to invalid donor lifetime
    assert widget.frequency == 0.0


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
    parent.image_layer_with_phasor_features_combobox = Mock()
    parent.image_layer_with_phasor_features_combobox.currentText.return_value = "test_layer"
    parent._labels_layer_with_phasor_features = test_layer.metadata['phasor_features_labels_layer']
    
    # Mock donor line
    widget.current_donor_line = Mock()
    trajectory_data = np.array([[0.1, 0.2], [0.1, 0.15]]).T
    widget.current_donor_line.get_xydata.return_value = trajectory_data

    # Calculate FRET efficiency first time
    widget.calculate_fret_efficiency()
    
    fret_layer_name = f"FRET efficiency: test_layer"
    assert fret_layer_name in [layer.name for layer in viewer.layers]
    
    # Count layers before second calculation
    initial_layer_count = len(viewer.layers)
    
    # Calculate again - should replace existing layer
    widget.calculate_fret_efficiency()
    
    # Should not have added an extra layer
    assert len(viewer.layers) == initial_layer_count
    assert fret_layer_name in [layer.name for layer in viewer.layers]
