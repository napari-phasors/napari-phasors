from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure
from napari.layers import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.lifetime_tab import LifetimeWidget
from napari_phasors.plotter import PlotterWidget

from phasorpy.phasor import phasor_to_apparent_lifetime, phasor_to_normal_lifetime


def test_lifetime_widget_initialization_values(make_napari_viewer):
    """Test the initialization of the Lifetime Widget."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab

    # Basic widget structure tests
    assert lifetime_widget.viewer == viewer
    assert lifetime_widget.parent_widget == parent
    assert isinstance(lifetime_widget.layout(), QVBoxLayout)

    # Test initial attribute values
    assert lifetime_widget.frequency == 0.0
    assert lifetime_widget.lifetime_data is None
    assert lifetime_widget.lifetime_data_original is None
    assert lifetime_widget.lifetime_layer is None
    assert lifetime_widget.min_lifetime is None
    assert lifetime_widget.max_lifetime is None
    assert lifetime_widget.lifetime_colormap is None
    assert lifetime_widget.colormap_contrast_limits is None
    assert lifetime_widget.lifetime_type is None
    assert lifetime_widget.lifetime_range_factor == 1000
    assert lifetime_widget._slider_being_dragged is False

    # Test histogram figure initialization
    assert isinstance(lifetime_widget.hist_fig, Figure)
    assert lifetime_widget.hist_ax is not None

    # Test UI components
    assert hasattr(lifetime_widget, 'frequency_input')
    assert isinstance(lifetime_widget.frequency_input, QLineEdit)

    assert hasattr(lifetime_widget, 'lifetime_type_combobox')
    assert isinstance(lifetime_widget.lifetime_type_combobox, QComboBox)
    expected_items = ["None", "Apparent Phase Lifetime", "Apparent Modulation Lifetime", "Normal Lifetime"]
    actual_items = [lifetime_widget.lifetime_type_combobox.itemText(i) 
                   for i in range(lifetime_widget.lifetime_type_combobox.count())]
    assert actual_items == expected_items
    assert lifetime_widget.lifetime_type_combobox.currentText() == "None"

    assert hasattr(lifetime_widget, 'lifetime_range_label')
    assert isinstance(lifetime_widget.lifetime_range_label, QLabel)
    assert lifetime_widget.lifetime_range_label.text() == "Lifetime range (ns): 0.0 - 100.0"

    assert hasattr(lifetime_widget, 'lifetime_min_edit')
    assert isinstance(lifetime_widget.lifetime_min_edit, QLineEdit)
    assert lifetime_widget.lifetime_min_edit.text() == "0.0"

    assert hasattr(lifetime_widget, 'lifetime_max_edit')
    assert isinstance(lifetime_widget.lifetime_max_edit, QLineEdit)
    assert lifetime_widget.lifetime_max_edit.text() == "100.0"

    assert hasattr(lifetime_widget, 'lifetime_range_slider')
    assert isinstance(lifetime_widget.lifetime_range_slider, QRangeSlider)
    assert lifetime_widget.lifetime_range_slider.orientation() == Qt.Orientation.Horizontal
    assert lifetime_widget.lifetime_range_slider.minimum() == 0
    assert lifetime_widget.lifetime_range_slider.maximum() == 100
    assert lifetime_widget.lifetime_range_slider.value() == (0, 100)

    # Test scroll area
    scroll_areas = lifetime_widget.findChildren(QScrollArea)
    assert len(scroll_areas) == 1
    scroll_area = scroll_areas[0]
    assert scroll_area.widgetResizable() == True

    # Test histogram widget initially hidden
    assert lifetime_widget.histogram_widget.isHidden() == True


def test_lifetime_widget_histogram_styling(make_napari_viewer):
    """Test that histogram styling is applied correctly."""
    viewer = make_napari_viewer()

    # Test that style_histogram_axes method exists and is called
    with patch.object(LifetimeWidget, 'style_histogram_axes') as mock_style:
        parent = PlotterWidget(viewer)
        lifetime_widget = parent.lifetime_tab
        mock_style.assert_called_once()

    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab

    # Check axes styling
    assert lifetime_widget.hist_ax.patch.get_alpha() == 0
    assert lifetime_widget.hist_fig.patch.get_alpha() == 0

    # Check spine colors - use numpy.allclose for RGBA comparison
    import matplotlib.colors as mcolors

    grey_rgba = mcolors.to_rgba('grey')

    for spine in lifetime_widget.hist_ax.spines.values():
        np.testing.assert_array_almost_equal(spine.get_edgecolor(), grey_rgba)
        assert spine.get_linewidth() == 1

    # Check labels
    assert lifetime_widget.hist_ax.get_ylabel() == "Pixel count"
    assert lifetime_widget.hist_ax.get_xlabel() == "Lifetime (ns)"


def test_lifetime_widget_frequency_input_validation(make_napari_viewer):
    """Test frequency input validation."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab

    # Test that frequency input has double validator
    validator = lifetime_widget.frequency_input.validator()
    assert validator is not None

    # Test valid input
    lifetime_widget.frequency_input.setText("80.0")
    assert lifetime_widget.frequency_input.text() == "80.0"


def test_lifetime_widget_slider_drag_state(make_napari_viewer):
    """Test slider drag state tracking."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab

    # Initially not being dragged
    assert lifetime_widget._slider_being_dragged is False

    # Simulate slider press
    lifetime_widget._on_slider_pressed()
    assert lifetime_widget._slider_being_dragged is True

    # Simulate slider release
    lifetime_widget._on_slider_released()
    assert lifetime_widget._slider_being_dragged is False


def test_lifetime_widget_range_label_update(make_napari_viewer):
    """Test lifetime range label update."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab

    # Test label update
    test_value = (25000, 75000)  # Represents 25.0 - 75.0 ns with factor 1000
    lifetime_widget._on_lifetime_range_label_update(test_value)
    
    assert lifetime_widget.lifetime_range_label.text() == "Lifetime range (ns): 25.00 - 75.00"
    assert lifetime_widget.lifetime_min_edit.text() == "25.00"
    assert lifetime_widget.lifetime_max_edit.text() == "75.00"


def test_lifetime_widget_calculate_lifetimes_no_layer(make_napari_viewer):
    """Test calculate_lifetimes when no layer is available."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab
    
    # Should return early without error
    lifetime_widget.calculate_lifetimes()
    assert lifetime_widget.lifetime_data_original is None


def test_lifetime_widget_plot_histogram_no_data(make_napari_viewer):
    """Test plotting histogram when no data is available."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab

    # Should hide histogram widget and return early
    lifetime_widget.plot_lifetime_histogram()
    assert lifetime_widget.histogram_widget.isHidden() == True


def test_lifetime_widget_ui_layout(make_napari_viewer):
    """Test the UI layout structure."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab

    # Check main layout
    main_layout = lifetime_widget.layout()
    assert isinstance(main_layout, QVBoxLayout)

    # Check scroll area exists
    scroll_areas = lifetime_widget.findChildren(QScrollArea)
    assert len(scroll_areas) == 1

    # Check horizontal layouts exist for range controls
    h_layouts = lifetime_widget.findChildren(QHBoxLayout)
    assert len(h_layouts) >= 1  # At least one for the min/max edit controls


def test_lifetime_widget_canvas_properties(make_napari_viewer):
    """Test canvas and figure properties."""
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab

    # Check figure size
    assert lifetime_widget.hist_fig.get_figwidth() == 8
    assert lifetime_widget.hist_fig.get_figheight() == 4

    # Check that constrained_layout is used
    assert lifetime_widget.hist_fig.get_constrained_layout()

    canvas_widgets = lifetime_widget.findChildren(FigureCanvasQTAgg)
    assert len(canvas_widgets) == 1

    canvas = canvas_widgets[0]
    assert canvas.height() == 150  # Fixed height as set in setup_ui


def test_lifetime_widget_type_changed_to_none(make_napari_viewer):
    """Test behavior when lifetime type is changed to None."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab

    # Initially the histogram should be hidden
    assert lifetime_widget.histogram_widget.isHidden() == True
    
    # Show the histogram first by setting some data and a lifetime type
    lifetime_widget.lifetime_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    lifetime_widget.histogram_widget.show()
    assert lifetime_widget.histogram_widget.isHidden() == False
    
    # Change to None and verify histogram is hidden
    lifetime_widget._on_lifetime_type_changed("None")
    assert lifetime_widget.histogram_widget.isHidden() == True


def test_lifetime_widget_type_changed_no_frequency(make_napari_viewer):
    """Test behavior when lifetime type is changed but no frequency is set."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab

    with patch('napari_phasors.lifetime_tab.show_error') as mock_error:
        lifetime_widget._on_lifetime_type_changed("Apparent Phase Lifetime")
        mock_error.assert_called_once_with("Enter frequency")


def test_lifetime_widget_slider_range_update(make_napari_viewer):
    """Test updating slider range based on lifetime data."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab

    # Create mock lifetime data
    lifetime_widget.lifetime_data_original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    lifetime_widget._update_lifetime_range_slider()
    
    # Check that min/max are set correctly
    assert lifetime_widget.min_lifetime == 1.0
    assert lifetime_widget.max_lifetime == 5.0
    
    # Check slider range
    assert lifetime_widget.lifetime_range_slider.minimum() == 0
    assert lifetime_widget.lifetime_range_slider.maximum() == 5000  # 5.0 * 1000
    assert lifetime_widget.lifetime_range_slider.value() == (1000, 5000)  # (1.0 * 1000, 5.0 * 1000)


def test_lifetime_widget_slider_range_update_no_valid_data(make_napari_viewer):
    """Test updating slider range when no valid data exists."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab

    # Create data with only invalid values
    lifetime_widget.lifetime_data_original = np.array([np.nan, 0, np.inf, -1])
    
    lifetime_widget._update_lifetime_range_slider()
    
    # Check that defaults are used
    assert lifetime_widget.min_lifetime == 0.0
    assert lifetime_widget.max_lifetime == 10.0
    assert lifetime_widget.lifetime_range_slider.maximum() == 10000


def test_lifetime_widget_min_max_edit_callbacks(make_napari_viewer):
    """Test manual entry of min/max values."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab
    
    # Set some initial max lifetime for validation
    lifetime_widget.max_lifetime = 10.0
    
    # Test min edit
    lifetime_widget.lifetime_min_edit.setText("2.5")
    lifetime_widget.lifetime_max_edit.setText("7.5")
    
    with patch.object(lifetime_widget, '_on_lifetime_range_changed') as mock_range_changed:
        lifetime_widget._on_lifetime_min_edit()
        mock_range_changed.assert_called_once()
    
    with patch.object(lifetime_widget, '_on_lifetime_range_changed') as mock_range_changed:
        lifetime_widget._on_lifetime_max_edit()
        mock_range_changed.assert_called_once()


def test_lifetime_widget_image_layer_changed_with_settings(make_napari_viewer):
    """Test behavior when image layer changes and has frequency settings."""
    viewer = make_napari_viewer()
    intensity_image_layer = create_image_layer_with_phasors()
    intensity_image_layer.metadata["settings"] = {"frequency": 80.0}
    viewer.add_layer(intensity_image_layer)
    
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab
    
    # Mock the harmonic value
    parent.harmonic = 1
    
    lifetime_widget._on_image_layer_changed()
    
    # Check that frequency is loaded from settings
    assert lifetime_widget.frequency_input.text() == "80.0"


def test_lifetime_widget_image_layer_changed_no_layer(make_napari_viewer):
    """Test behavior when no layer is selected."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab
    
    # Mock empty layer name
    parent.image_layer_with_phasor_features_combobox.setCurrentText("")
    
    lifetime_widget._on_image_layer_changed()
    
    # Should hide histogram
    assert lifetime_widget.histogram_widget.isHidden() == True


def test_lifetime_widget_colormap_changed_callback(make_napari_viewer):
    """Test colormap change callback."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab
    
    # Create mock event and layer
    mock_layer = MagicMock()
    mock_layer.colormap.colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mock_layer.contrast_limits = (1.0, 5.0)
    
    mock_event = MagicMock()
    mock_event.source = mock_layer
    
    # Set initial contrast limits
    lifetime_widget.colormap_contrast_limits = (0.0, 10.0)
    
    with patch.object(lifetime_widget, '_update_slider_from_contrast_limits') as mock_update_slider:
        with patch.object(lifetime_widget, '_update_lifetime_histogram') as mock_update_hist:
            lifetime_widget._on_colormap_changed(mock_event)
            
            # Check that attributes are updated
            np.testing.assert_array_equal(lifetime_widget.lifetime_colormap, mock_layer.colormap.colors)
            assert lifetime_widget.colormap_contrast_limits == (1.0, 5.0)
            
            # Check that methods are called
            mock_update_slider.assert_called_once()
            mock_update_hist.assert_called_once()

def test_lifetime_widget_calculate_lifetimes_with_real_data(make_napari_viewer):
    """Test calculating different lifetime types with real phasor data and compare with expected values."""
    import pandas as pd
    from phasorpy.phasor import phasor_to_apparent_lifetime, phasor_to_normal_lifetime
    
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab
    
    # Set up test data
    parent.harmonic = 1
    frequency = 80.0  # MHz
    
    # Create realistic phasor data
    real_values = np.array([0.5, 0.6, 0.7, 0.8])
    imag_values = np.array([0.3, 0.4, 0.5, 0.6])
    
    # Create mock labels layer with features
    mock_labels_layer = MagicMock()
    mock_labels_layer.data = np.ones((2, 2))  # 2x2 shape to match test data
    mock_labels_layer.scale = (1.0, 1.0)
    
    # Create phasor features DataFrame
    mock_features = pd.DataFrame({
        'harmonic': [1, 1, 1, 1],
        'G': real_values,
        'S': imag_values
    })
    mock_labels_layer.features = mock_features
    
    parent._labels_layer_with_phasor_features = mock_labels_layer
    parent.image_layer_with_phasor_features_combobox.setCurrentText("test_layer")
    lifetime_widget.frequency_input.setText(str(frequency))
    
    # Test Apparent Phase Lifetime
    lifetime_widget.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
    
    # Calculate expected values directly
    expected_frequency = frequency * parent.harmonic
    expected_phase_lifetime, expected_mod_lifetime = phasor_to_apparent_lifetime(
        real_values, imag_values, frequency=expected_frequency
    )
    expected_phase_clipped = np.clip(expected_phase_lifetime, a_min=0, a_max=None)
    expected_phase_reshaped = np.reshape(expected_phase_clipped, (2, 2))
    
    # Calculate using widget
    lifetime_widget.calculate_lifetimes()
    
    # Compare results
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data_original, 
        expected_phase_reshaped,
        decimal=10
    )
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data, 
        expected_phase_reshaped,
        decimal=10
    )
    
    # Test Apparent Modulation Lifetime
    lifetime_widget.lifetime_type_combobox.setCurrentText("Apparent Modulation Lifetime")
    
    # Calculate expected values
    expected_mod_clipped = np.clip(expected_mod_lifetime, a_min=0, a_max=None)
    expected_mod_reshaped = np.reshape(expected_mod_clipped, (2, 2))
    
    # Calculate using widget
    lifetime_widget.calculate_lifetimes()
    
    # Compare results
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data_original, 
        expected_mod_reshaped,
        decimal=10
    )
    
    # Test Normal Lifetime
    lifetime_widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")
    
    # Calculate expected values directly
    expected_normal_lifetime = phasor_to_normal_lifetime(
        real_values, imag_values, frequency=expected_frequency
    )
    expected_normal_reshaped = np.reshape(expected_normal_lifetime, (2, 2))
    
    # Calculate using widget
    lifetime_widget.calculate_lifetimes()
    
    # Compare results
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data_original, 
        expected_normal_reshaped,
        decimal=10
    )


def test_lifetime_widget_full_workflow_with_real_calculations(make_napari_viewer):
    """Test the complete workflow with real lifetime calculations and layer creation."""
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab
    
    # Create and add synthetic layer
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    # Set frequency
    lifetime_widget.frequency_input.setText(str("80"))

    # Select lifetime type
    lifetime_widget.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")

    # Check lifetime layer was added
    assert lifetime_widget.lifetime_layer in viewer.layers

    harmonic = parent.harmonic

    phasor_features = layer.metadata['phasor_features_labels_layer'].features
    harmonic_mask = phasor_features['harmonic'] == harmonic
    real = phasor_features.loc[harmonic_mask, 'G']
    imag = phasor_features.loc[harmonic_mask, 'S']

    expected_phase_lifetime, expected_mod_lifetime = phasor_to_apparent_lifetime(real, imag, frequency=80)

    mean_shape = layer.data.shape
    
    # reshape
    expected_phase_lifetime = np.reshape(expected_phase_lifetime, mean_shape)
    expected_mod_lifetime = np.reshape(expected_mod_lifetime, mean_shape)

    lifetime_layer = viewer.layers[lifetime_widget.lifetime_layer.name]

    # Verify expected lifetime values
    np.testing.assert_array_almost_equal(
        lifetime_layer.data,
        expected_phase_lifetime,
        decimal=10
    )

    # Change lifetime type to Modulation Lifetime
    lifetime_widget.lifetime_type_combobox.setCurrentText("Apparent Modulation Lifetime")
    lifetime_layer = viewer.layers[lifetime_widget.lifetime_layer.name]

    # Verify expected lifetime values
    np.testing.assert_array_almost_equal(
        lifetime_layer.data,
        expected_mod_lifetime,
        decimal=10
    )

    # Change lifetime type to Normal Lifetime
    lifetime_widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")
    lifetime_layer = viewer.layers[lifetime_widget.lifetime_layer.name]

    expected_normal_lifetime = phasor_to_normal_lifetime(
        real, imag, frequency=80
    )
    expected_normal_reshaped = np.reshape(expected_normal_lifetime, mean_shape)

    # Verify expected lifetime values
    np.testing.assert_array_almost_equal(
        lifetime_layer.data,
        expected_normal_reshaped,
        decimal=10
    )


def test_lifetime_widget_range_clipping_with_real_data(make_napari_viewer):
    """Test range clipping functionality with real calculated lifetime data."""
    import pandas as pd
    from phasorpy.phasor import phasor_to_normal_lifetime
    
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab
    
    # Set up test data
    parent.harmonic = 1
    frequency = 80.0
    
    # Create phasor data that will generate a known range of lifetimes
    real_values = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.05])
    imag_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
    
    mock_labels_layer = MagicMock()
    mock_labels_layer.data = np.ones((2, 3))
    mock_labels_layer.scale = (1.0, 1.0)
    
    mock_features = pd.DataFrame({
        'harmonic': [1, 1, 1, 1, 1, 1],
        'G': real_values,
        'S': imag_values
    })
    mock_labels_layer.features = mock_features
    
    parent._labels_layer_with_phasor_features = mock_labels_layer
    parent.image_layer_with_phasor_features_combobox.setCurrentText("test_layer")
    lifetime_widget.frequency_input.setText(str(frequency))
    
    # Calculate expected normal lifetime values
    expected_frequency = frequency * parent.harmonic
    expected_lifetime = phasor_to_normal_lifetime(
        real_values, imag_values, frequency=expected_frequency
    )
    expected_reshaped = np.reshape(expected_lifetime, (2, 3))
    
    # Set normal lifetime type and calculate
    lifetime_widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")
    lifetime_widget.calculate_lifetimes()
    
    # Verify initial calculation
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data_original, 
        expected_reshaped,
        decimal=10
    )
    
    # Test range clipping
    # Find reasonable clipping range from the calculated data
    min_lifetime = np.min(expected_lifetime)
    max_lifetime = np.max(expected_lifetime)
    clip_range = max_lifetime - min_lifetime
    
    # Clip to middle 50% of the range
    clip_min = min_lifetime + 0.25 * clip_range
    clip_max = max_lifetime - 0.25 * clip_range
    
    # Apply clipping via slider
    min_slider = int(clip_min * lifetime_widget.lifetime_range_factor)
    max_slider = int(clip_max * lifetime_widget.lifetime_range_factor)
    lifetime_widget._on_lifetime_range_changed((min_slider, max_slider))
    
    # Calculate expected clipped data
    expected_clipped = np.clip(expected_reshaped, clip_min, clip_max)
    
    # Verify clipping worked correctly
    np.testing.assert_array_almost_equal(
        lifetime_widget.lifetime_data, 
        expected_clipped,
        decimal=10
    )
    
    # If layer exists, check it was updated
    if lifetime_widget.lifetime_layer is not None:
        np.testing.assert_array_almost_equal(
            lifetime_widget.lifetime_layer.data, 
            expected_clipped,
            decimal=10
        )
        assert lifetime_widget.lifetime_layer.contrast_limits == [clip_min, clip_max]


def test_lifetime_widget_different_harmonics_and_frequencies(make_napari_viewer):
    """Test lifetime calculations with different harmonic and frequency combinations."""
    import pandas as pd
    from phasorpy.phasor import phasor_to_apparent_lifetime
    
    viewer = make_napari_viewer()
    parent = PlotterWidget(viewer)
    lifetime_widget = parent.lifetime_tab
    
    # Test data
    real_values = np.array([0.6, 0.7, 0.8, 0.5])
    imag_values = np.array([0.4, 0.3, 0.2, 0.5])
    
    mock_labels_layer = MagicMock()
    mock_labels_layer.data = np.ones((2, 2))
    mock_labels_layer.scale = (1.0, 1.0)
    
    parent._labels_layer_with_phasor_features = mock_labels_layer
    parent.image_layer_with_phasor_features_combobox.setCurrentText("test_layer")
    
    # Test different combinations
    test_cases = [
        (1, 80.0),   # 1st harmonic, 80 MHz
        (2, 80.0),   # 2nd harmonic, 80 MHz  
        (1, 40.0),   # 1st harmonic, 40 MHz
        (3, 160.0),  # 3rd harmonic, 160 MHz
    ]
    
    for harmonic, base_frequency in test_cases:
        # Set up for this test case
        parent.harmonic = harmonic
        lifetime_widget.frequency_input.setText(str(base_frequency))
        
        # Create features with correct harmonic
        mock_features = pd.DataFrame({
            'harmonic': [harmonic] * 4,
            'G': real_values,
            'S': imag_values
        })
        mock_labels_layer.features = mock_features
        
        # Calculate expected values
        expected_frequency = base_frequency * harmonic
        expected_phase_lifetime, _ = phasor_to_apparent_lifetime(
            real_values, imag_values, frequency=expected_frequency
        )
        expected_clipped = np.clip(expected_phase_lifetime, a_min=0, a_max=None)
        expected_reshaped = np.reshape(expected_clipped, (2, 2))
        
        # Calculate using widget
        lifetime_widget.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
        lifetime_widget.calculate_lifetimes()
        
        # Verify results for this combination
        np.testing.assert_array_almost_equal(
            lifetime_widget.lifetime_data_original,
            expected_reshaped,
            decimal=10,
            err_msg=f"Failed for harmonic={harmonic}, frequency={base_frequency}"
        )
        
        # Verify frequency was calculated correctly
        assert lifetime_widget.frequency == base_frequency