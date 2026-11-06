from unittest.mock import Mock, patch

import numpy as np
import pytest
from napari.utils.colormaps import Colormap
from numpy.testing import assert_array_equal
from phasorpy.lifetime import phasor_from_fret_donor
from phasorpy.phasor import phasor_nearest_neighbor
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from superqt import QToggleSwitch

from napari_phasors._mapping_filters import (
    FRET_EFFICIENCY,
    MappingFilterList,
    get_filters,
    new_filter,
    set_filters,
)
from napari_phasors._tests.test_plotter import (
    assert_run_row_is_pinned,
    create_image_layer_with_phasors,
)
from napari_phasors._utils import analysis_layer_name
from napari_phasors.fret_tab import draw_fret_trajectory_overlay
from napari_phasors.plotter import PlotterWidget


def test_fret_widget_initial_state_and_controls(make_viewer_model, qtbot):
    """A fresh FRET tab, its controls, and every action with no layer."""
    from napari.layers import Image

    viewer = make_viewer_model()
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

    # Source selectors are Manual, showing the manual inputs, by default.
    assert widget.donor_source_selector.currentText() == "Manual"
    assert widget.bg_source_selector.currentText() == "Manual"
    assert widget.donor_stack.currentIndex() == 0  # Manual page
    assert widget.bg_stack.currentIndex() == 0  # Manual page

    # Donor lifetime combobox (CheckableComboBox: starts empty, no placeholder
    # item) and the lifetime modes.
    assert widget.donor_lifetime_combobox.count() == 0
    assert widget.donor_lifetime_combobox.currentText() == ""
    assert (
        widget.lifetime_type_combobox.currentText()
        == "Apparent Phase Lifetime"
    )
    assert widget.lifetime_type_combobox.count() == 3
    lifetime_modes = [
        widget.lifetime_type_combobox.itemText(i)
        for i in range(widget.lifetime_type_combobox.count())
    ]
    assert lifetime_modes == [
        "Apparent Phase Lifetime",
        "Apparent Modulation Lifetime",
        "Normal Lifetime",
    ]
    for mode in (
        "Apparent Modulation Lifetime",
        "Normal Lifetime",
        "Apparent Phase Lifetime",
    ):
        widget.lifetime_type_combobox.setCurrentText(mode)
        assert widget.lifetime_type_combobox.currentText() == mode

    # Slider initial values and the colormap toggle.
    assert widget.background_slider.value() == 10  # 0.1 * 100
    assert widget.fretting_slider.value() == 100  # 1.0 * 100
    assert isinstance(widget.colormap_checkbox, QToggleSwitch)
    assert (
        widget.colormap_checkbox.text()
        == "Overlay colormap on donor trajectory"
    )
    assert widget.colormap_checkbox.isChecked() is True

    # Dynamic labels show default text
    assert widget.donor_label.text() == "Donor lifetime (ns):"
    assert widget.background_position_label.text() == "Background position:"

    # The primary action stays reachable however long the settings get.
    assert_run_row_is_pinned(
        widget,
        widget.calculate_fret_efficiency_button,
        widget.autoupdate_container,
    )

    # One efficiency filter: always shown, nothing to add or remove.
    assert isinstance(widget.filter_list, MappingFilterList)
    assert not widget.filter_list.add_button.isVisibleTo(widget.filter_list)
    assert not hasattr(widget.filter_list, 'clear_button')
    card = _efficiency_card(widget)
    assert card.metric_label.text() == FRET_EFFICIENCY
    assert not card.remove_button.isVisibleTo(card)
    assert not card.entry['enabled']
    assert widget.filter_list.filters() == []

    # The style button leads into the filter section; the toggle is in its
    # dialog, which opens once however often the button is clicked.
    content_layout = widget.filter_box.parentWidget().layout()
    display_box = widget.trajectory_style_btn.parentWidget()
    style = content_layout.indexOf(display_box)
    assert style >= 0
    assert content_layout.indexOf(widget.filter_box) > style
    assert widget.colormap_checkbox.window() is widget.trajectory_style_dialog
    widget.trajectory_style_btn.click()
    dialog = widget.trajectory_style_dialog
    assert dialog.isVisible()
    widget.trajectory_style_btn.click()
    assert widget.trajectory_style_dialog is dialog
    dialog.close()

    # Sliders and the colormap toggle.
    widget.background_slider.setValue(25)  # 0.25
    widget._on_background_slider_changed()
    assert widget.donor_background == 0.25
    assert widget.background_label.text() == "0.25"
    widget.background_slider.setValue(10)
    widget._on_background_slider_changed()
    widget.fretting_slider.setValue(75)  # 0.75
    widget._on_fretting_slider_changed()
    assert widget.donor_fretting_proportion == 0.75
    assert widget.fretting_label.text() == "0.75"
    widget.fretting_slider.setValue(100)
    widget._on_fretting_slider_changed()
    widget.colormap_checkbox.setChecked(False)
    widget._on_colormap_checkbox_changed()
    assert widget.use_colormap is False
    widget.colormap_checkbox.setChecked(True)
    widget._on_colormap_checkbox_changed()
    assert widget.use_colormap is True

    # The donor and background source selectors switch pages.
    widget.donor_source_selector.setCurrentText("From layer(s)")
    widget._on_donor_source_changed(1)
    assert widget.donor_stack.currentIndex() == 1  # From layer(s) page
    assert widget.donor_label.text() == "Donor lifetime (ns):"
    widget.donor_source_selector.setCurrentText("Manual")
    widget._on_donor_source_changed(0)
    assert widget.donor_stack.currentIndex() == 0
    assert widget.donor_label.text() == "Donor lifetime (ns):"
    widget.bg_source_selector.setCurrentText("From layer(s)")
    widget._on_bg_source_changed(1)
    assert widget.bg_stack.currentIndex() == 1
    assert widget.background_position_label.text() == "Background position:"
    widget.bg_source_selector.setCurrentText("Manual")
    widget._on_bg_source_changed(0)
    assert widget.bg_stack.currentIndex() == 0
    assert widget.background_position_label.text() == "Background position:"

    # With no layer, nothing is calculated.
    parent._labels_layer_with_phasor_features = None
    widget._calculate_background_position()
    assert widget.background_real_edit.text() == "0.0"
    assert widget.background_imag_edit.text() == "0.0"
    widget.plot_donor_trajectory()
    assert widget.current_donor_line is None
    widget.calculate_fret_efficiency()
    assert widget.fret_layer is None
    assert len(viewer.layers) == 0
    widget.frequency_input.setText("80")
    initial_lifetime = widget.donor_line_edit.text()
    widget._calculate_donor_lifetime()
    assert widget.donor_line_edit.text() == initial_lifetime

    # "From layer(s)" with nothing checked changes nothing.
    parent.harmonic = 1
    widget.donor_source_selector.setCurrentText("From layer(s)")
    widget._on_donor_source_changed(1)
    widget.bg_source_selector.setCurrentText("From layer(s)")
    widget._on_bg_source_changed(1)
    initial_donor_text = widget.donor_line_edit.text()
    initial_donor_label = widget.donor_label.text()
    initial_bg_real = widget.background_real_edit.text()
    initial_bg_imag = widget.background_imag_edit.text()
    initial_bg_label = widget.background_position_label.text()
    widget._calculate_donor_lifetime()
    widget._calculate_background_position()
    assert widget.donor_line_edit.text() == initial_donor_text
    assert widget.donor_label.text() == initial_donor_label
    assert widget.background_real_edit.text() == initial_bg_real
    assert widget.background_imag_edit.text() == initial_bg_imag
    assert widget.background_position_label.text() == initial_bg_label
    widget.donor_source_selector.setCurrentText("Manual")
    widget._on_donor_source_changed(0)
    widget.bg_source_selector.setCurrentText("Manual")
    widget._on_bg_source_changed(0)

    # Switching between modes preserves manually entered values.
    widget.donor_line_edit.setText("2.5")
    widget.background_real_edit.setText("0.3")
    widget.background_imag_edit.setText("0.4")
    widget.donor_source_selector.setCurrentText("From layer(s)")
    widget._on_donor_source_changed(1)
    widget.donor_source_selector.setCurrentText("Manual")
    widget._on_donor_source_changed(0)
    widget.bg_source_selector.setCurrentText("From layer(s)")
    widget._on_bg_source_changed(1)
    widget.bg_source_selector.setCurrentText("Manual")
    widget._on_bg_source_changed(0)
    assert widget.donor_line_edit.text() == "2.5"
    assert widget.background_real_edit.text() == "0.3"
    assert widget.background_imag_edit.text() == "0.4"

    # Layers without phasor data are never offered.
    invalid_layer = Image(np.random.random((10, 10)), name="invalid_layer")
    viewer.add_layer(invalid_layer)
    widget._update_donor_lifetime_combobox()
    widget._update_background_combobox()
    assert widget.donor_lifetime_combobox.count() == 0
    assert widget.background_image_combobox.count() == 0
    assert "invalid_layer" not in widget.donor_lifetime_combobox.allItems()
    # With no checked layers the donor calculation is a no-op.
    initial_lifetime = widget.donor_line_edit.text()
    widget._calculate_donor_lifetime()
    assert widget.donor_line_edit.text() == initial_lifetime


def test_fret_efficiency_calculation(make_viewer_model, qtbot):
    """FRET efficiency matches phasorpy, is written to one layer per source,
    stores its settings and colormap, and follows the harmonic."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)
    fret_layer_name = "test_layer [FRET efficiency]"
    hw = widget.histogram_widget

    # FRET settings are only initialized when the analysis is performed.
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "test_layer"
    )
    widget._on_image_layer_changed()
    if 'settings' in test_layer.metadata:
        assert 'fret' not in test_layer.metadata['settings']

    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    # Expected FRET efficiency from the layer's phasor coordinates.
    metadata = test_layer.metadata
    harmonics = metadata.get("harmonics", [1])
    harmonic = parent.harmonic
    if isinstance(harmonics, (list, np.ndarray)) and len(harmonics) > 1:
        harmonic_idx = (
            list(harmonics).index(harmonic) if harmonic in harmonics else 0
        )
        real = metadata["G"][harmonic_idx].flatten()
        imag = metadata["S"][harmonic_idx].flatten()
    else:
        real = metadata["G"].flatten()
        imag = metadata["S"].flatten()

    def expected_efficiency(donor_lifetime):
        trajectory_real, trajectory_imag = phasor_from_fret_donor(
            80,
            donor_lifetime,
            fret_efficiency=widget._fret_efficiencies,
            donor_background=widget.donor_background,
            background_imag=0.1,
            background_real=0.1,
            donor_fretting=widget.donor_fretting_proportion,
        )
        return phasor_nearest_neighbor(
            np.array(real),
            np.array(imag),
            trajectory_real,
            trajectory_imag,
            values=widget._fret_efficiencies,
        )

    expected = expected_efficiency(2)
    widget.calculate_fret_efficiency_button.click()
    assert fret_layer_name in [layer.name for layer in viewer.layers]
    assert widget.fret_layer is not None
    assert_array_equal(viewer.layers[fret_layer_name].data.flatten(), expected)
    # The settings were initialized with the analysis.
    assert 'settings' in test_layer.metadata
    assert 'fret' in test_layer.metadata['settings']
    fret_settings = test_layer.metadata['settings']['fret']
    assert fret_settings['donor_lifetime'] == 2.0
    assert test_layer.metadata['settings']['frequency'] == 80.0
    assert fret_settings['donor_background'] == 0.1
    assert fret_settings['donor_fretting_proportion'] == 1.0
    assert fret_settings['use_colormap'] is True
    assert fret_settings['background_positions_by_harmonic'] == {
        1: {'imag': 0.1, 'real': 0.1}
    }
    assert 'colormap_settings' in fret_settings
    assert fret_settings['colormap_settings']['colormap_name'] == 'viridis'
    assert fret_settings['colormap_settings']['colormap_colors'] is None
    assert fret_settings['colormap_settings']['contrast_limits'] == [0, 1]
    assert fret_settings['colormap_settings']['colormap_changed'] is False

    # Parameters update from the UI.
    widget.donor_line_edit.setText("2.5")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.2")
    widget.background_imag_edit.setText("0.3")
    widget._on_parameters_changed()
    assert widget.donor_lifetime == 2.5
    assert widget.frequency == 80 * parent.harmonic
    assert widget.background_real == 0.2
    assert widget.background_imag == 0.3
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    # The histogram dataset is labelled with the FRET output layer's name.
    assert list(hw._datasets.keys()) == [fret_layer_name]
    # The range-changed path re-labels the same way.
    widget._on_fret_range_changed(0.1, 0.9)
    assert list(hw._datasets.keys()) == [fret_layer_name]
    # Clip the FRET layers to a sub-range and refresh the histogram.
    widget._on_fret_range_changed(0.1, 0.5)
    widget._update_fret_histogram()
    widget._on_fret_range_changed(0.0, 1.0)

    # A new donor lifetime updates the existing FRET layer.
    widget.donor_line_edit.setText("1.5")
    expected = expected_efficiency(1.5)
    widget.calculate_fret_efficiency_button.click()
    assert fret_layer_name in [layer.name for layer in viewer.layers]
    assert_array_equal(viewer.layers[fret_layer_name].data.flatten(), expected)

    # Recalculating with other values replaces the layer, not adds one.
    initial_layer_count = len(viewer.layers)
    widget.donor_line_edit.setText("1.5")
    widget.frequency_input.setText("90")
    widget.background_real_edit.setText("0.2")
    widget.background_imag_edit.setText("0.2")
    widget.calculate_fret_efficiency_button.click()
    assert len(viewer.layers) == initial_layer_count
    assert fret_layer_name in [layer.name for layer in viewer.layers]

    # Running FRET again keeps the colormap, limits and gamma of the layer.
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")
    widget.calculate_fret_efficiency()
    assert widget.fret_layer.colormap.name == 'viridis'
    widget.fret_layer.colormap = 'magma'
    widget.fret_layer.contrast_limits = (0.1, 0.9)
    widget.fret_layer.gamma = 0.8
    widget.calculate_fret_efficiency()
    assert widget.fret_layer.colormap.name == 'magma'
    assert tuple(widget.fret_layer.contrast_limits) == pytest.approx(
        (0.1, 0.9)
    )
    assert widget.fret_layer.gamma == pytest.approx(0.8)
    colormap_settings = test_layer.metadata['settings']['fret'][
        'colormap_settings'
    ]
    assert colormap_settings['colormap_name'] == 'magma'
    assert colormap_settings['gamma'] == pytest.approx(0.8)

    # Colormap and contrast limit events update the tab.
    fret_layer = viewer.layers[fret_layer_name]
    widget.fret_layer = fret_layer
    initial_colormap = fret_layer.colormap.name
    initial_contrast_limits = fret_layer.contrast_limits
    assert initial_colormap is not None
    assert initial_contrast_limits is not None
    assert len(initial_contrast_limits) == 2
    new_colormap = 'viridis'
    fret_layer.colormap = new_colormap
    mock_event = Mock()
    mock_event.source = fret_layer
    widget._on_colormap_changed(mock_event)
    assert widget.fret_colormap is not None
    assert widget.fret_layer.colormap.name == new_colormap
    assert widget.fret_layer.colormap.name != initial_colormap
    new_contrast_limits = [0.2, 0.8]
    fret_layer.contrast_limits = new_contrast_limits
    widget._on_contrast_limits_changed(mock_event)
    assert widget.colormap_contrast_limits == new_contrast_limits
    assert widget.colormap_contrast_limits != initial_contrast_limits

    # Colormap settings are stored in the metadata.
    widget.fret_layer.colormap = 'plasma'
    widget.fret_layer.contrast_limits = (0.2, 0.8)
    widget._on_colormap_changed(mock_event)
    widget._on_contrast_limits_changed(mock_event)
    colormap_settings = test_layer.metadata['settings']['fret'][
        'colormap_settings'
    ]
    assert colormap_settings['colormap_name'] == 'plasma'
    assert colormap_settings['contrast_limits'] == [0.2, 0.8]
    assert colormap_settings['colormap_changed'] is True
    # Regression: a built-in colormap picked after the run was stored with
    # all its colours, while the run itself stores it by name only.
    assert colormap_settings['colormap_colors'] is None

    # A custom colormap keeps its colours, even though napari knows it by
    # name once the layer has used it, so it can come back in another
    # session.
    custom = Colormap(
        colors=[[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        name='fret test black to red',
    )
    widget.fret_layer.colormap = custom
    widget._on_colormap_changed(mock_event)
    colormap_settings = test_layer.metadata['settings']['fret'][
        'colormap_settings'
    ]
    assert colormap_settings['colormap_name'] == 'fret test black to red'
    np.testing.assert_allclose(
        colormap_settings['colormap_colors'],
        np.asarray(widget.fret_layer.colormap.colors),
    )

    # FRET efficiency respects harmonic changes.
    parent.harmonic = 1
    widget._on_harmonic_changed()
    widget.calculate_fret_efficiency_button.click()
    fret_data_h1 = viewer.layers[fret_layer_name].data.copy()
    parent.harmonic = 2
    widget._on_harmonic_changed()
    widget.calculate_fret_efficiency_button.click()
    fret_data_h2 = viewer.layers[fret_layer_name].data.copy()
    assert not np.array_equal(fret_data_h1, fret_data_h2)


def test_fret_histogram_stays_empty_before_any_analysis(
    make_viewer_model, qtbot
):
    """Without FRET layers the histogram keeps its empty axes on screen.

    Every other tab shows a styled, dataless plot before an analysis is run,
    so the FRET dock must not collapse to nothing either.
    """
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    w = parent.fret_tab
    parent.tab_widget.setCurrentWidget(w)

    hw = w.histogram_widget
    with patch.object(hw, "hide") as mock_hide:
        w._update_fret_histogram()
    mock_hide.assert_not_called()

    assert not hw.isHidden()
    assert hw.counts is None
    assert hw._datasets == {}
    # The axes are still drawn: spines, ticks and labels are all in place.
    assert hw.ax.get_xlabel() == "FRET efficiency"
    assert hw.ax.get_ylabel() == "Pixel count"
    assert hw.ax.spines["bottom"].get_visible()
    assert len(hw.ax.lines) == 0

    # Running an analysis fills the same, still-visible axes.
    w.donor_line_edit.setText("2.0")
    w.frequency_input.setText("80")
    w.calculate_fret_efficiency()
    assert not hw.isHidden()
    assert hw.counts is not None

    # Dropping the layers again empties the plot without hiding it.
    for fret_layer in list(w.fret_layers):
        viewer.layers.remove(fret_layer)
    w._update_fret_histogram()
    assert not hw.isHidden()
    assert hw.counts is None


def test_fret_donor_and_background_from_layers(make_viewer_model, qtbot):
    """The donor lifetime and background position computed from layers, for
    every lifetime type and harmonic, and the labels that report them."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # The donor combobox follows layers being added and removed.
    initial_count = widget.donor_lifetime_combobox.count()
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)
    assert widget.donor_lifetime_combobox.count() == initial_count + 1
    combobox_items = [
        widget.donor_lifetime_combobox.itemText(i)
        for i in range(widget.donor_lifetime_combobox.count())
    ]
    assert "test_layer" in combobox_items
    viewer.layers.remove(test_layer)
    assert widget.donor_lifetime_combobox.count() == initial_count
    viewer.add_layer(test_layer)

    # Without a frequency the donor lifetime is left alone.
    widget.donor_lifetime_combobox.setCheckedItems(["test_layer"])
    widget.frequency_input.setText("")
    initial_lifetime = widget.donor_line_edit.text()
    widget._calculate_donor_lifetime()
    assert widget.donor_line_edit.text() == initial_lifetime

    # Every lifetime type gives a positive donor lifetime.
    widget.frequency_input.setText("80")
    parent.harmonic = 1
    values = {}
    for lifetime_type in (
        "Apparent Phase Lifetime",
        "Apparent Modulation Lifetime",
        "Normal Lifetime",
    ):
        widget.lifetime_type_combobox.setCurrentText(lifetime_type)
        widget._calculate_donor_lifetime()
        assert widget.donor_line_edit.text() != ""
        lifetime_value = float(widget.donor_line_edit.text())
        assert lifetime_value > 0
        assert widget.donor_lifetime == lifetime_value
        values[lifetime_type] = lifetime_value
    # They should generally be different (though could be close).
    assert len(set(values.values())) > 1

    # Different harmonics give different lifetimes.
    widget.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
    lifetimes = []
    for harmonic in (1, 2, 3):
        parent.harmonic = harmonic
        widget._calculate_donor_lifetime()
        lifetimes.append(widget.donor_line_edit.text())
    assert lifetimes[0] != lifetimes[1]
    assert lifetimes[0] != lifetimes[2]
    assert lifetimes[1] != lifetimes[2]
    parent.harmonic = 1

    # "From layer(s)": a finite averaged lifetime for each type, and the
    # label shows it until switching back to Manual.
    widget.donor_source_selector.setCurrentIndex(1)  # From layer(s)
    widget._on_donor_source_changed(1)
    widget._update_donor_lifetime_combobox()
    widget.donor_lifetime_combobox.setCheckedItems([test_layer.name])
    for lifetime_type in (
        "Apparent Phase Lifetime",
        "Apparent Modulation Lifetime",
        "Normal Lifetime",
    ):
        widget.lifetime_type_combobox.setCurrentText(lifetime_type)
        widget._calculate_donor_lifetime()
        assert widget.donor_lifetime is not None and widget.donor_lifetime > 0
        assert widget.donor_line_edit.text() != ""
    label_text = widget.donor_label.text()
    assert "Donor lifetime (from layer(s)):" in label_text
    assert "ns" in label_text
    widget.donor_source_selector.setCurrentText("Manual")
    widget._on_donor_source_changed(0)
    assert widget.donor_label.text() == "Donor lifetime (ns):"

    # The background position from a layer.
    widget._update_background_combobox()
    combobox_items = [
        widget.background_image_combobox.itemText(i)
        for i in range(widget.background_image_combobox.count())
    ]
    assert "Select layer..." not in combobox_items
    assert "test_layer" in combobox_items
    widget.bg_source_selector.setCurrentText("From layer(s)")
    widget._on_bg_source_changed(1)
    widget.background_image_combobox.setCheckedItems(["test_layer"])
    widget._calculate_background_position()
    real_text = widget.background_real_edit.text()
    imag_text = widget.background_imag_edit.text()
    assert real_text != "0.0" or imag_text != "0.0"
    assert float(real_text) >= 0
    assert float(imag_text) >= 0
    # The position was stored for the current harmonic, matching the
    # displayed text to 3 decimal places.
    assert parent.harmonic in widget.background_positions_by_harmonic
    stored_position = widget.background_positions_by_harmonic[parent.harmonic]
    assert abs(stored_position['real'] - float(real_text)) < 0.001
    assert abs(stored_position['imag'] - float(imag_text)) < 0.001
    # The label shows the calculated values in "From layer" mode.
    expected_label = (
        f"Background position: G={stored_position['real']:.2f}, "
        f"S={stored_position['imag']:.2f}"
    )
    assert widget.background_position_label.text() == expected_label
    assert "Background position: G=" in expected_label
    # With no layer checked the label reverts to its default.
    widget.background_image_combobox.deselectAll()
    widget._calculate_background_position()
    assert widget.background_position_label.text() == "Background position:"
    # Back to Manual, the label reverts too.
    widget.background_image_combobox.setCheckedItems(["test_layer"])
    widget._calculate_background_position()
    assert "S=" in widget.background_position_label.text()
    widget.bg_source_selector.setCurrentText("Manual")
    widget._on_bg_source_changed(0)
    assert widget.background_position_label.text() == "Background position:"


def test_fret_donor_trajectory_on_a_mocked_canvas(make_viewer_model, qtbot):
    """The trajectory is plotted, follows the harmonic, and falls back to
    plt.cm.jet when a FRET layer is active but no colormap was set."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

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

    # A failure while drawing is reported instead of raised.
    with patch("napari_phasors.fret_tab.show_error") as mock_error:
        widget.plot_donor_trajectory()
    assert ax_mock.plot.called
    mock_error.assert_called_once()

    # Make ax.plot() return a list-like object that can be subscripted
    ax_mock.plot.return_value = [Mock()]
    parent.canvas_widget.axes = ax_mock
    histogram_mock = Mock()
    histogram_mock.histogram = None
    parent.canvas_widget.artists = {'HISTOGRAM2D': histogram_mock}

    parent.harmonic = 1
    widget.current_harmonic = 1
    widget.plot_donor_trajectory()
    assert ax_mock.plot.called
    assert widget.current_harmonic == 1
    assert widget.frequency == 80.0  # base_frequency * harmonic (80 * 1)

    parent.harmonic = 2
    widget._on_harmonic_changed()
    assert widget.current_harmonic == 2
    assert widget.frequency == 160.0

    parent.harmonic = 3
    widget._on_harmonic_changed()
    assert widget.current_harmonic == 3
    assert widget.frequency == 240.0

    # An active FRET layer with colormap coloring enabled and no custom
    # colormap set falls back to plt.cm.jet.
    parent.harmonic = 1
    widget._on_harmonic_changed()
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")
    mock_fret_layer = Mock()
    mock_fret_layer.contrast_limits = (0.0, 1.0)
    widget.fret_layer = mock_fret_layer
    widget.use_colormap = True
    widget.fret_colormap = None
    ax_mock.add_collection.reset_mock()
    widget.plot_donor_trajectory()
    assert widget.current_donor_circle is not None
    assert widget.current_background_circle is not None
    assert ax_mock.add_collection.called
    widget.fret_layer = None


def test_fret_trajectory_and_filters_without_an_analysis(
    make_viewer_model, qtbot
):
    """Background positions per harmonic, the donor trajectory and its
    artists, and the efficiency filter before any FRET map exists."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    # With no usable donor lifetime there is nothing to freeze or refresh.
    widget.donor_line_edit.setText("")
    widget.frequency_input.setText("")
    assert widget._fret_filter_params() == {}
    assert widget._refresh_fret_filter_params([]) is False

    # Nothing selected, nothing to filter.
    widget._apply_filter_stack([])
    assert widget.filter_list.filters() == []
    widget._sync_filter_ui()
    assert widget.filter_list.summary_label.text() == ""
    assert widget._primary_filter_layer() is None

    # Without a parent plotter there is nothing to filter, and no crash.
    widget.parent_widget = None
    try:
        assert widget._filter_layers() == []
        assert widget._layer_filter_params(object()) == {}
        widget._apply_filter_stack([new_filter(FRET_EFFICIENCY, 0.0, 1.0)])
    finally:
        widget.parent_widget = parent

    # A selector that has already been destroyed reads as "nothing selected".
    widget.parent_widget = _BrokenSelector()
    try:
        assert widget._filter_layers() == []
        assert widget._primary_filter_layer() is None
    finally:
        # The tab's own teardown still needs a real plotter behind it.
        widget.parent_widget = parent

    # The card says why it cannot be switched on yet.
    widget._refresh_filter_enable_state()
    card = _efficiency_card(widget)
    assert not card.enabled_check.isEnabled()
    assert "Select at least one" in card.enabled_check.toolTip()

    # Artists: none until the trajectory is plotted; they follow the tab.
    assert len(widget.get_all_artists()) == 0
    widget.frequency_input.setText("80")
    widget.donor_line_edit.setText("2.0")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")
    widget.plot_donor_trajectory()
    assert len(widget.get_all_artists()) == 3
    assert widget.current_donor_line is not None
    assert widget.current_donor_line in widget.get_all_artists()
    assert widget.current_donor_line.get_visible() is True
    widget.set_artists_visible(False)
    assert widget.current_donor_line.get_visible() is False
    widget.set_artists_visible(True)
    assert widget.current_donor_line.get_visible() is True

    # Background positions are stored and retrieved by harmonic.
    parent.harmonic = 1
    widget.current_harmonic = 1
    widget.background_real_edit.setText("0.2")
    widget.background_imag_edit.setText("0.3")
    widget._store_current_background_position()
    assert 1 in widget.background_positions_by_harmonic
    assert widget.background_positions_by_harmonic[1]['real'] == 0.2
    assert widget.background_positions_by_harmonic[1]['imag'] == 0.3
    # Harmonic 2 starts from the default position.
    parent.harmonic = 2
    widget._on_harmonic_changed()
    assert widget.background_real_edit.text() == "0.000"
    assert widget.background_imag_edit.text() == "0.000"
    assert widget.current_harmonic == 2
    widget.background_real_edit.setText("0.5")
    widget.background_imag_edit.setText("0.6")
    widget._store_current_background_position()
    assert 2 in widget.background_positions_by_harmonic
    assert widget.background_positions_by_harmonic[2]['real'] == 0.5
    assert widget.background_positions_by_harmonic[2]['imag'] == 0.6
    # Switching back restores harmonic 1's position.
    parent.harmonic = 1
    widget._on_harmonic_changed()
    assert widget.background_real_edit.text() == "0.200"
    assert widget.background_imag_edit.text() == "0.300"
    assert widget.current_harmonic == 1

    # Manual background position changes are stored per harmonic too.
    widget.background_real_edit.setText("0.15")
    widget.background_imag_edit.setText("0.25")
    widget._on_background_position_changed()
    assert widget.background_positions_by_harmonic[1]['real'] == 0.15
    assert widget.background_positions_by_harmonic[1]['imag'] == 0.25
    parent.harmonic = 3
    widget._on_harmonic_changed()
    assert widget.background_real_edit.text() == "0.000"
    assert widget.background_imag_edit.text() == "0.000"
    widget.background_real_edit.setText("0.35")
    widget.background_imag_edit.setText("0.45")
    widget._on_background_position_changed()
    parent.harmonic = 1
    widget._on_harmonic_changed()
    assert float(widget.background_real_edit.text()) == 0.15
    assert float(widget.background_imag_edit.text()) == 0.25

    # Trajectory calculations use the effective frequency (base * harmonic).
    base_frequency = 80.0
    donor_lifetime = 2.0
    widget.donor_line_edit.setText(str(donor_lifetime))
    widget.frequency_input.setText(str(base_frequency))
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")
    trajectories = []
    for harmonic in (1, 2, 3):
        parent.harmonic = harmonic
        widget._on_parameters_changed()
        trajectories.append(
            phasor_from_fret_donor(
                base_frequency * harmonic,
                donor_lifetime,
                fret_efficiency=widget._fret_efficiencies,
                donor_background=widget.donor_background,
                background_imag=0.1,
                background_real=0.1,
                donor_fretting=widget.donor_fretting_proportion,
            )[0]
        )
    assert not np.array_equal(trajectories[0], trajectories[1])
    assert not np.array_equal(trajectories[0], trajectories[2])
    assert not np.array_equal(trajectories[1], trajectories[2])
    parent.harmonic = 1
    widget._on_harmonic_changed()
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    # Drawing the trajectory with a colormap adds a line collection.
    ax_mock = Mock()
    mock_layer = Mock()
    mock_layer.contrast_limits = (0.0, 1.0)
    widget.fret_layer = mock_layer
    widget.colormap_contrast_limits = (0.0, 1.0)
    widget._draw_colormap_trajectory(
        ax_mock, np.linspace(0.1, 0.9, 100), np.linspace(0.1, 0.5, 100)
    )
    assert ax_mock.add_collection.called
    widget.fret_layer = None

    # With a layer but no donor trajectory, the card says so; a complete
    # trajectory makes it available.
    layer = create_image_layer_with_phasors()
    layer.name = "no_donor"
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    widget._on_image_layer_changed()
    widget.donor_line_edit.setText("")
    widget.frequency_input.setText("")
    widget._refresh_filter_enable_state()
    card = _efficiency_card(widget)
    assert not card.enabled_check.isEnabled()
    assert "donor lifetime" in card.enabled_check.toolTip()
    widget.frequency_input.setText("80.0")
    widget.donor_line_edit.setText("4.2")
    assert _efficiency_card(widget).enabled_check.isEnabled()

    # When restoring saved colormap settings on the FRET layer raises, the
    # except-handler still reconnects the colormap/contrast_limits/gamma
    # events instead of leaving the layer without callbacks.
    mock_layer = Mock()
    mock_layer.events.colormap.disconnect.side_effect = RuntimeError("boom")
    widget.fret_layer = mock_layer
    widget._saved_colormap_name = "viridis"
    widget._saved_colormap_colors = None
    widget._saved_contrast_limits = [0.0, 1.0]
    widget._saved_gamma = 1.0
    widget._apply_saved_fret_colormap_settings()
    assert mock_layer.events.colormap.connect.called
    assert mock_layer.events.contrast_limits.connect.called
    assert mock_layer.events.gamma.connect.called
    widget.fret_layer = None


def test_fret_layers_share_gamma_and_are_torn_down(make_viewer_model, qtbot):
    """Changing gamma on one FRET layer syncs its siblings and the histogram,
    and a layer change disconnects every FRET layer."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    layer_a = create_image_layer_with_phasors()
    layer_a.name = "layer_a"
    layer_b = create_image_layer_with_phasors()
    layer_b.name = "layer_b"
    viewer.add_layer(layer_a)
    viewer.add_layer(layer_b)

    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    with patch.object(
        parent, "get_selected_layers", return_value=[layer_a, layer_b]
    ):
        widget.calculate_fret_efficiency_button.click()

    assert len(widget.fret_layers) == 2

    # Gamma propagates to the sibling layer, the stored gamma, and the
    # histogram widget.
    widget.fret_layers[0].gamma = 0.7
    assert widget.fret_layers[1].gamma == 0.7
    assert widget.colormap_gamma == 0.7
    assert widget.histogram_widget.gamma == 0.7

    widget._teardown_on_layer_change()
    assert widget.fret_layer is None
    assert widget.fret_layers == []


def test_fret_efficiency_calculation_single_harmonic_layer(
    make_viewer_model, qtbot
):
    """A layer with a single harmonic (no leading harmonic axis in G/S)
    should produce a FRET efficiency map matching the image shape, not a
    malformed slice of G/S."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    test_layer = create_image_layer_with_phasors(harmonic=1)
    assert test_layer.metadata["G"].ndim == test_layer.data.ndim
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)

    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")

    parent.harmonic = 1
    widget._on_harmonic_changed()
    widget.calculate_fret_efficiency_button.click()

    fret_layer_name = "test_layer [FRET efficiency]"
    assert fret_layer_name in [layer.name for layer in viewer.layers]
    assert viewer.layers[fret_layer_name].data.shape == test_layer.data.shape


def test_fret_efficiency_rejects_bad_inputs_and_layers(
    make_viewer_model, qtbot
):
    """Invalid inputs are reported without creating a layer; layers without
    phasor arrays or the harmonic are skipped; a failing one is reported."""
    viewer = make_viewer_model()
    parent, widget = _ready_fret_widget(viewer)
    layer = viewer.layers[0]

    cases = [
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
    ]
    widget.background_real_edit.setText("0.1")
    widget.background_imag_edit.setText("0.1")
    initial_layer_count = len(viewer.layers)
    for donor_lifetime, frequency, expected_message, message_type in cases:
        widget.donor_line_edit.setText(donor_lifetime)
        widget.frequency_input.setText(frequency)
        target = "show_warning" if message_type == "warning" else "show_error"
        with patch(f'napari_phasors.fret_tab.{target}') as mock_show:
            widget.calculate_fret_efficiency()
            mock_show.assert_called_once_with(expected_message)
        assert len(viewer.layers) == initial_layer_count
        assert "layer_a [FRET efficiency]" not in [
            lyr.name for lyr in viewer.layers
        ]
        assert widget.fret_layer is None

    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")

    def no_fret_output():
        return not any(
            lyr.name.startswith("FRET efficiency") for lyr in viewer.layers
        )

    # A layer whose G/S went missing is skipped, not treated as an error.
    real = layer.metadata["G"]
    layer.metadata["G"] = None
    widget.calculate_fret_efficiency()
    assert no_fret_output()
    layer.metadata["G"] = real

    # A layer that never computed the selected harmonic is skipped.
    harmonics = layer.metadata["harmonics"]
    layer.metadata["harmonics"] = np.array([97])
    widget.calculate_fret_efficiency()
    assert no_fret_output()
    layer.metadata["harmonics"] = harmonics

    # A layer whose computation raises is reported by name, not swallowed.
    errors = []

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    with (
        patch("napari_phasors.fret_tab.show_error", errors.append),
        patch("napari_phasors.fret_tab.phasor_nearest_neighbor", explode),
    ):
        widget.calculate_fret_efficiency()
    assert any("boom" in message for message in errors)
    assert any("layer_a" in message for message in errors)
    assert no_fret_output()


def test_fret_settings_follow_layer_switches(make_viewer_model, qtbot):
    """Each layer keeps its own settings, and the donor combobox keeps its
    checked layers while layers come and go."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    layer1 = create_image_layer_with_phasors()
    layer1.name = "layer1"
    viewer.add_layer(layer1)
    layer2 = create_image_layer_with_phasors()
    layer2.name = "layer2"
    viewer.add_layer(layer2)

    # Configure layer 1, then layer 2 with different values.
    for name, donor, frequency, background in (
        ("layer1", "2.5", "80.0", 30),
        ("layer2", "3.5", "90.0", 50),
    ):
        parent.image_layer_with_phasor_features_combobox.setCurrentText(name)
        widget._on_image_layer_changed()
        widget.donor_line_edit.setText(donor)
        widget.frequency_input.setText(frequency)
        widget.background_slider.setValue(background)
        parent._broadcast_frequency_value_across_tabs(frequency)
        widget._on_parameters_changed()
        widget._on_background_slider_changed()

    # Each layer's settings are restored when it is selected again.
    for name, donor, frequency, background in (
        ("layer1", "2.5", "80.0", 30),
        ("layer2", "3.5", "90.0", 50),
    ):
        parent.image_layer_with_phasor_features_combobox.setCurrentText(name)
        widget._on_image_layer_changed()
        assert widget.donor_line_edit.text() == donor
        assert widget.frequency_input.text() == frequency
        assert widget.background_slider.value() == background

    # The donor combobox's checked selection persists as layers change.
    test_layer1 = create_image_layer_with_phasors()
    test_layer1.name = "test_layer1"
    viewer.add_layer(test_layer1)
    widget.donor_lifetime_combobox.setCheckedItems(["test_layer1"])
    assert widget.donor_lifetime_combobox.checkedItems() == ["test_layer1"]
    test_layer2 = create_image_layer_with_phasors()
    test_layer2.name = "test_layer2"
    viewer.add_layer(test_layer2)
    assert "test_layer1" in widget.donor_lifetime_combobox.checkedItems()
    widget.donor_lifetime_combobox.setCheckedItems(["test_layer2"])
    assert widget.donor_lifetime_combobox.checkedItems() == ["test_layer2"]
    viewer.layers.remove(test_layer1)
    assert widget.donor_lifetime_combobox.checkedItems() == ["test_layer2"]
    viewer.layers.remove(test_layer2)
    assert widget.donor_lifetime_combobox.checkedItems() == []
    assert widget.donor_lifetime_combobox.currentText() == ""


def test_fret_manual_settings_are_stored_and_restored(
    make_viewer_model, qtbot
):
    """Manual values and per-harmonic background positions are unsaved
    settings of the layer, restored when it is selected again."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "test_layer"
    )
    widget._on_image_layer_changed()

    def reselect():
        parent.image_layer_with_phasor_features_combobox.setCurrentText("")
        widget._on_image_layer_changed()
        parent.image_layer_with_phasor_features_combobox.setCurrentText(
            "test_layer"
        )
        widget._on_image_layer_changed()

    # Manual values are stored as unsaved settings of the layer.
    widget.donor_line_edit.setText("2.5")
    widget.frequency_input.setText("85")
    widget.background_real_edit.setText("0.15")
    widget.background_imag_edit.setText("0.25")
    widget.background_slider.setValue(30)  # 0.3
    widget.fretting_slider.setValue(75)  # 0.75
    widget.colormap_checkbox.setChecked(False)
    parent._broadcast_frequency_value_across_tabs('85')
    widget._on_parameters_changed()
    widget._on_background_position_changed()
    widget._on_background_slider_changed()
    widget._on_fretting_slider_changed()
    widget._on_colormap_checkbox_changed()
    # Edits are unsaved settings of the layer until FRET is calculated.
    assert 'fret' not in test_layer.metadata.get('settings', {})
    settings = parent.layer_settings(test_layer)
    fret_settings = settings['fret']
    assert fret_settings['donor_lifetime'] == 2.5
    assert settings['frequency'] == 85.0
    assert fret_settings['donor_background'] == 0.3
    assert fret_settings['donor_fretting_proportion'] == 0.75
    assert fret_settings['use_colormap'] is False
    assert 1 in fret_settings['background_positions_by_harmonic']
    assert fret_settings['background_positions_by_harmonic'][1]['real'] == 0.15
    assert fret_settings['background_positions_by_harmonic'][1]['imag'] == 0.25

    # They are restored after switching away and back.
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
    reselect()
    assert float(widget.donor_line_edit.text()) == 3.5
    assert float(widget.frequency_input.text()) == 90.0
    assert widget.background_slider.value() == 40
    assert widget.background_label.text() == "0.40"
    assert widget.fretting_slider.value() == 85
    assert widget.fretting_label.text() == "0.85"
    assert widget.colormap_checkbox.isChecked() is False
    assert float(widget.background_real_edit.text()) == 0.35
    assert float(widget.background_imag_edit.text()) == 0.45

    # Background positions are stored per harmonic...
    positions = {1: (0.1, 0.2), 2: (0.3, 0.4), 3: (0.5, 0.6)}
    for harmonic, (real, imag) in positions.items():
        parent.harmonic = harmonic
        widget._on_harmonic_changed()
        widget.background_real_edit.setText(str(real))
        widget.background_imag_edit.setText(str(imag))
        widget._on_background_position_changed()
    bg_positions = parent.layer_settings(test_layer)['fret'][
        'background_positions_by_harmonic'
    ]
    for harmonic, (real, imag) in positions.items():
        assert harmonic in bg_positions
        assert bg_positions[harmonic]['real'] == real
        assert bg_positions[harmonic]['imag'] == imag

    # ...and restored per harmonic.
    reselect()
    for harmonic, (real, imag) in positions.items():
        parent.harmonic = harmonic
        widget._on_harmonic_changed()
        assert float(widget.background_real_edit.text()) == real
        assert float(widget.background_imag_edit.text()) == imag


def test_fret_from_layer_settings_are_stored_and_restored(
    make_viewer_model, qtbot
):
    """'From layer(s)' sources are stored, restored, and fall back to Manual
    when a referenced layer is gone."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

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
    widget.donor_line_edit.setText("2.5")

    widget.donor_source_selector.setCurrentText("From layer(s)")
    widget._on_donor_source_changed(1)
    widget.donor_lifetime_combobox.setCheckedItems(["donor_layer"])
    widget.lifetime_type_combobox.setCurrentText("Normal Lifetime")
    widget.bg_source_selector.setCurrentText("From layer(s)")
    widget._on_bg_source_changed(1)
    widget.background_image_combobox.setCheckedItems(["bg_layer"])

    # The layer's (unsaved) settings record the sources.
    fret_settings = parent.layer_settings(donor_layer)['fret']
    assert fret_settings['donor_source'] == 'From layer(s)'
    assert fret_settings['donor_layer_names'] == ['donor_layer']
    assert fret_settings['donor_lifetime_type'] == 'Normal Lifetime'
    assert fret_settings['background_source'] == 'From layer(s)'
    assert fret_settings['background_layer_names'] == ['bg_layer']

    def reselect():
        parent.image_layer_with_phasor_features_combobox.setCurrentText("")
        widget._on_image_layer_changed()
        parent.image_layer_with_phasor_features_combobox.setCurrentText(
            "donor_layer"
        )
        widget._on_image_layer_changed()

    # They are restored after switching away and back.
    widget.lifetime_type_combobox.setCurrentText(
        "Apparent Modulation Lifetime"
    )
    reselect()
    assert widget.donor_source_selector.currentText() == "From layer(s)"
    assert widget.donor_lifetime_combobox.checkedItems() == ["donor_layer"]
    assert (
        widget.lifetime_type_combobox.currentText()
        == "Apparent Modulation Lifetime"
    )
    assert widget.bg_source_selector.currentText() == "From layer(s)"
    assert widget.background_image_combobox.checkedItems() == ["bg_layer"]

    # A missing background layer reverts the background to Manual; the donor
    # stays From layer(s) since its layer exists.
    viewer.layers.remove(bg_layer)
    reselect()
    assert widget.bg_source_selector.currentText() == "Manual"
    assert widget.donor_source_selector.currentText() == "From layer(s)"


def test_fret_recreate_restore_and_colormap_from_metadata(
    make_viewer_model, qtbot
):
    """Cover FRET metadata restore/recreate and saved-colormap application."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    w = parent.fret_tab

    w.donor_line_edit.setText("2.0")
    w.frequency_input.setText("80")
    w.calculate_fret_efficiency()
    assert w.fret_layer is not None

    # Apply saved colormap settings to the existing FRET layer.
    w._saved_colormap_name = "viridis"
    w._saved_colormap_colors = None
    w._saved_contrast_limits = (0.0, 1.0)
    w._apply_saved_fret_colormap_settings()

    # Inject colormap settings into metadata and restore them.
    fret_settings = layer.metadata.setdefault("settings", {}).setdefault(
        "fret", {}
    )
    fret_settings["colormap_settings"] = {
        "colormap_name": "magma",
        "colormap_colors": None,
        "contrast_limits": (0.0, 1.0),
        "colormap_changed": True,
    }
    w._restore_fret_settings_from_metadata()
    assert w._saved_colormap_name == "magma"

    # Recreate the FRET analysis from metadata (re-runs the calculation).
    w._recreate_fret_from_metadata()


def test_fret_harmonics_none_fallback(make_viewer_model, qtbot):
    """Cover FRET calculations when harmonics metadata is None (e.g. loaded .R64 files)."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {"frequency": 80.0}
    # Simulate a .R64 file where harmonics is None and G/S are 2D arrays
    layer.metadata["harmonics"] = None
    layer.metadata["G"] = layer.metadata["G"][0]
    layer.metadata["S"] = layer.metadata["S"][0]
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    w = parent.fret_tab
    parent.tab_widget.setCurrentWidget(w)

    w.frequency_input.setText("80")

    # 1. Test _calculate_background_position with harmonics is None
    w.bg_source_selector.setCurrentText("From layer(s)")
    w._update_background_combobox()
    w.background_image_combobox.setCheckedItems([layer.name])
    w._calculate_background_position()
    assert len(w.background_positions_by_harmonic) > 0

    # 2. Test _calculate_donor_lifetime with harmonics is None
    w.donor_source_selector.setCurrentIndex(1)  # From layer(s)
    w._update_donor_lifetime_combobox()
    w.donor_lifetime_combobox.setCheckedItems([layer.name])
    w.lifetime_type_combobox.setCurrentText("Normal Lifetime")
    w._calculate_donor_lifetime()
    assert w.donor_lifetime is not None and w.donor_lifetime > 0

    # 3. Test main FRET efficiency trajectory calculation loop with harmonics is None
    w.calculate_fret_efficiency()
    assert w.fret_layer is not None


def test_fret_widget_exceptions(make_viewer_model, qtbot):
    """Test FRET module handles missing metadata, IndexError, and Exception gracefully."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    parent = PlotterWidget(viewer)
    w = parent.fret_tab
    parent.tab_widget.setCurrentWidget(w)

    w.frequency_input.setText("80")

    # 1. Background position exceptions
    w.bg_source_selector.setCurrentText("From layer(s)")
    w._update_background_combobox()
    w.background_image_combobox.setCheckedItems([layer.name])

    # Missing G/S arrays
    layer.metadata["G"] = None
    w._calculate_background_position()  # Should continue/return

    # Harmonic not found
    layer.metadata["G"] = np.ones((2, 10, 10))
    layer.metadata["S"] = np.ones((2, 10, 10))
    layer.metadata["harmonics"] = np.array([999])
    w._calculate_background_position()  # Should continue/return

    # Mismatched shapes causing Exception in phasor_center
    layer.metadata["G"] = np.ones(
        (10, 10)
    )  # Intentionally 2D when expected 3D or vice versa
    layer.metadata["S"] = np.ones((2, 10, 10))
    layer.metadata["harmonics"] = np.array([1])
    parent.harmonic = 1
    w._calculate_background_position()  # Should catch Exception

    # Same for harmonics is None
    layer.metadata["harmonics"] = None
    w._calculate_background_position()  # Should catch Exception

    # 2. Donor lifetime exceptions
    w.donor_source_selector.setCurrentIndex(1)  # From layer(s)
    w._update_donor_lifetime_combobox()
    w.donor_lifetime_combobox.setCheckedItems([layer.name])

    layer.metadata["G"] = None
    w._calculate_donor_lifetime()

    layer.metadata["G"] = np.ones((2, 10, 10))
    layer.metadata["harmonics"] = np.array([999])
    w._calculate_donor_lifetime()

    layer.metadata["G"] = np.ones((10, 10))
    layer.metadata["S"] = np.ones((2, 10, 10))
    layer.metadata["harmonics"] = np.array([1])
    w._calculate_donor_lifetime()

    # 3. Main trajectory loop exceptions
    # Reset G/S to valid matching shapes before triggering plotter update via combobox
    layer.metadata["G"] = np.ones((2, 10, 10))
    layer.metadata["S"] = np.ones((2, 10, 10))
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    layer.metadata["G"] = None
    w.calculate_fret_efficiency()

    layer.metadata["G"] = np.ones((2, 10, 10))
    layer.metadata["S"] = np.ones((2, 10, 10))
    layer.metadata["harmonics"] = np.array([999])
    w.calculate_fret_efficiency()


def test_reconnect_existing_fret_layer_direct(make_viewer_model, qtbot):
    """_reconnect_existing_fret_layer connects events directly and caches
    the layer's current colormap/contrast_limits/gamma when there are no
    saved colormap settings to restore."""
    import numpy as np
    from napari.layers import Image

    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab

    layer_name = "test_layer"
    source_layer = create_image_layer_with_phasors()
    source_layer.name = layer_name
    viewer.add_layer(source_layer)
    fret_layer_name = analysis_layer_name("FRET efficiency", layer_name)
    layer = Image(np.random.random((10, 10)), name=fret_layer_name)
    viewer.add_layer(layer)

    assert not hasattr(widget, '_saved_colormap_name')

    widget._reconnect_existing_fret_layer(layer_name)

    assert widget.fret_layer is layer
    assert widget.colormap_gamma == layer.gamma
    assert widget.colormap_contrast_limits == layer.contrast_limits
    np.testing.assert_array_equal(widget.fret_colormap, layer.colormap.colors)


def test_draw_fret_trajectory_overlay_uses_jet_colormap_by_default():
    """The standalone draw_fret_trajectory_overlay falls back to plt.cm.jet
    when no fret_colormap is supplied in settings."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        trajectory_real = np.linspace(0.1, 0.9, 20)
        trajectory_imag = np.linspace(0.1, 0.5, 20)
        fret_efficiencies = np.linspace(0.0, 1.0, 20)

        draw_fret_trajectory_overlay(
            ax, trajectory_real, trajectory_imag, fret_efficiencies
        )

        # A colormap trajectory (LineCollection) plus donor/background
        # circles should have been added to the axes.
        assert len(ax.collections) == 1
        assert len(ax.patches) == 2
    finally:
        plt.close(fig)


def _ready_fret_widget(viewer, layer_names=("layer_a",)):
    """Return a FRET tab with valid inputs and *layer_names* selected."""
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab
    for name in layer_names:
        layer = create_image_layer_with_phasors()
        layer.name = name
        viewer.add_layer(layer)
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    return parent, widget


def _setup_fret_selection_workflow(make_napari_viewer, qtbot):
    """Create two selected source layers and calculate FRET efficiency."""
    viewer = make_napari_viewer()
    layers = []
    for name in ("fret_a", "fret_b"):
        layer = create_image_layer_with_phasors()
        layer.name = name
        viewer.add_layer(layer)
        layers.append(layer)

    parent = PlotterWidget(viewer)
    qtbot.addWidget(parent)
    parent.show()
    parent.image_layers_checkable_combobox.setCheckedItems(
        [layer.name for layer in layers]
    )
    fret = parent.fret_tab
    parent.tab_widget.setCurrentWidget(fret)
    fret.donor_line_edit.setText("2.0")
    fret.frequency_input.setText("80")
    fret.background_real_edit.setText("0.1")
    fret.background_imag_edit.setText("0.1")
    fret.calculate_fret_efficiency_button.click()
    fret.histogram_widget.display_mode = "Individual layers"
    return viewer, parent, fret, layers


def _click_fret_source(qtbot, parent, source_name):
    """Toggle a Phasor Layers row through the visible popup."""
    combo = parent.image_layers_checkable_combobox
    row = next(
        row
        for row in range(combo.model().rowCount())
        if combo.model().item(row).text() == source_name
    )
    combo.showPopup()
    view = combo.view()
    rect = view.visualRect(combo.model().index(row, 0))
    point = rect.center()
    point.setX(rect.left() + 5)
    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=point)
    combo.hidePopup()


def test_fret_outputs_follow_real_source_selection(make_napari_viewer, qtbot):
    """FRET curves, statistics, and outputs follow real popup clicks, and a
    tagged output stays authoritative after a manual rename."""
    viewer, parent, fret, _ = _setup_fret_selection_workflow(
        make_napari_viewer, qtbot
    )
    output_a = "fret_a [FRET efficiency]"
    output_b = "fret_b [FRET efficiency]"
    stats = parent.fret_statistics_dock_widget.layer_stats_table

    assert list(fret.histogram_widget._datasets) == [output_a, output_b]
    assert len(fret.histogram_widget.ax.lines) == 2
    assert stats.rowCount() == 2
    assert viewer.layers[output_a].metadata['phasor_fret_output'] == {
        'source_layer': 'fret_a'
    }

    _click_fret_source(qtbot, parent, "fret_b")
    qtbot.waitUntil(
        lambda: parent.get_selected_layer_names() == ["fret_a"]
        and list(fret.histogram_widget._datasets) == [output_a],
        timeout=5000,
    )
    assert len(fret.histogram_widget.ax.lines) == 1
    assert stats.rowCount() == 1
    assert viewer.layers[output_a].visible is True
    assert viewer.layers[output_b].visible is False

    _click_fret_source(qtbot, parent, "fret_b")
    qtbot.waitUntil(
        lambda: parent.get_selected_layer_names() == ["fret_a", "fret_b"]
        and len(fret.histogram_widget._datasets) == 2,
        timeout=5000,
    )
    assert viewer.layers[output_b].visible is True
    assert len(fret.histogram_widget.ax.lines) == 2
    assert stats.rowCount() == 2

    # Tagged FRET outputs remain authoritative after manual renaming.
    output = viewer.layers[output_a]
    output.name = "Custom FRET result"
    output_id = id(output)
    fret.calculate_fret_efficiency()
    assert id(viewer.layers["Custom FRET result"]) == output_id
    assert output_a not in viewer.layers
    assert fret._fret_output_layers()['fret_a'] is output
    fret.rename_layer("fret_a", "fret_a_renamed")
    assert output.name == "Custom FRET result"
    assert output.metadata['phasor_fret_output'] == {
        'source_layer': 'fret_a_renamed'
    }


def test_fret_range_and_empty_source_selection(make_napari_viewer, qtbot):
    """Range clipping leaves deselected outputs untouched, a disjoint prior
    range resets to the new output's bounds, and clearing Phasor Layers
    removes stale FRET statistics and curves."""
    viewer, parent, fret, _ = _setup_fret_selection_workflow(
        make_napari_viewer, qtbot
    )
    output_a = viewer.layers["fret_a [FRET efficiency]"]
    output_b = viewer.layers["fret_b [FRET efficiency]"]
    output_a_original = output_a.metadata['fret_data_original'].copy()
    output_b_before = output_b.data.copy()
    slider_max_before = fret.histogram_widget.range_slider.maximum()

    def select(names):
        parent.image_layers_checkable_combobox.setCheckedItems(names)
        parent._layer_selection_timer.stop()
        parent._process_layer_selection_change()

    select(["fret_a"])
    fret.histogram_widget.set_range(0.2, 0.8)
    fret._on_fret_range_changed(0.2, 0.8)
    np.testing.assert_allclose(
        output_a.data,
        np.clip(output_a_original, 0.2, 0.8),
        equal_nan=True,
    )
    np.testing.assert_array_equal(output_b.data, output_b_before)
    assert fret.histogram_widget.range_slider.maximum() == slider_max_before

    select(["fret_a", "fret_b"])
    np.testing.assert_allclose(
        output_b.data,
        np.clip(output_b.metadata['fret_data_original'], 0.2, 0.8),
        equal_nan=True,
    )
    assert fret.histogram_widget.get_range() == (0.2, 0.8)

    # A disjoint prior range resets to the newly selected output bounds.
    replacement = np.linspace(0.6, 1.0, output_b.data.size).reshape(
        output_b.data.shape
    )
    output_b.metadata['fret_data_original'] = replacement.copy()
    output_b.data = replacement.copy()
    fret.histogram_widget.set_range(0.2, 0.4)
    select(["fret_b"])
    assert fret.histogram_widget.get_range() == (0.6, 1.0)
    np.testing.assert_allclose(output_b.data, replacement)

    # Clearing Phasor Layers removes stale FRET statistics and curves.
    select([])
    assert fret.histogram_widget.counts is None
    assert fret.histogram_widget._datasets == {}
    assert parent.fret_statistics_dock_widget.layer_stats_table.rowCount() == 0
    assert output_a.visible is False
    assert output_b.visible is False


def test_existing_fret_output_initializes_full_range_before_clipping(
    make_viewer_model, qtbot
):
    """Restored tagged output is not clipped by the slider's default range."""
    viewer = make_viewer_model()
    source = create_image_layer_with_phasors()
    source.name = "restored_source"
    viewer.add_layer(source)
    original = np.linspace(0.0, 1.0, 10).reshape(2, 5)
    output = viewer.add_image(
        original.copy(),
        name="Custom restored FRET",
        metadata={
            'fret_data_original': original.copy(),
            'phasor_fret_output': {'source_layer': source.name},
        },
    )

    parent = PlotterWidget(viewer)

    np.testing.assert_array_equal(output.data, original)
    assert parent.fret_tab._fret_range_initialized is True
    assert parent.fret_tab.histogram_widget.get_range() == (0.0, 1.0)


def test_fret_reconnects_an_existing_canonical_output(
    make_viewer_model, qtbot
):
    """A canonical output found at start-up is registered once, reconnected
    with its saved colormap, and renamed with its source."""
    viewer = make_viewer_model()
    source = create_image_layer_with_phasors()
    source.name = "legacy_fret_source"
    viewer.add_layer(source)
    output = viewer.add_image(
        np.linspace(0.0, 1.0, 10).reshape(2, 5),
        name="legacy_fret_source [FRET efficiency]",
    )
    parent = PlotterWidget(viewer)
    fret = parent.fret_tab

    # Direct reconnect uses the lifecycle-managed FRET registry.
    fret._reconnect_existing_fret_layer(source.name)
    fret._reconnect_existing_fret_layer(source.name)
    assert fret.fret_layer is output
    assert fret.fret_layers == [output]

    # Defensive selection paths.
    with patch.object(fret, "parent_widget", None):
        assert fret._get_selected_source_names() == set()
    with patch.object(
        parent, "get_selected_layers", side_effect=AttributeError
    ):
        assert fret._get_selected_source_names() == set()

    # A saved colormap is applied on reconnect.
    fret._saved_colormap_name = "viridis"
    with patch.object(fret, "_apply_saved_fret_colormap_settings") as apply:
        fret._reconnect_existing_fret_layer(source.name)
    apply.assert_called_once()

    fret.rename_layer("legacy_fret_source", "renamed_fret_source")
    assert output.name == "renamed_fret_source [FRET efficiency]"
    assert output.metadata['phasor_fret_output'] == {
        'source_layer': 'renamed_fret_source'
    }


# --------------------------------------------------------- efficiency filter


def _ready_fret_filter_widget(viewer, name="fret_layer"):
    """Return a FRET tab with a calculated efficiency map."""
    parent = PlotterWidget(viewer)
    widget = parent.fret_tab
    layer = create_image_layer_with_phasors()
    layer.name = name
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    widget._on_image_layer_changed()
    widget.frequency_input.setText("80.0")
    widget.donor_line_edit.setText("4.2")
    widget.calculate_fret_efficiency()
    return parent, widget, layer


def _efficiency_card(widget):
    """Return the Fret tab's single efficiency card."""
    (card,) = widget.filter_list._cards.values()
    return card


def _add_efficiency_filter(widget, low, high):
    """Set the efficiency card's range and switch it on."""
    card = _efficiency_card(widget)
    card.min_edit.setText(f"{low:.4f}")
    card.max_edit.setText(f"{high:.4f}")
    card._on_edits_changed()
    card = _efficiency_card(widget)
    card.enabled_check.setChecked(True)
    return _efficiency_card(widget)


def _plotted_trajectory_widget(make_viewer_model):
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    widget = parent.fret_tab
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.plot_donor_trajectory()
    return parent, widget, layer


def test_fret_trajectory_style(make_viewer_model, qtbot):
    """Width, transparency, dots and colour of the donor trajectory reach the
    plot, are kept in the layer settings, and reset to their defaults."""
    parent, widget, layer = _plotted_trajectory_widget(make_viewer_model)

    # Width, transparency and end-dot radius reach the drawn artists.
    widget.trajectory_width_spin.setValue(6.5)
    widget.trajectory_transparency_spin.setValue(0.4)
    widget.trajectory_dot_spin.setValue(0.05)
    assert widget.trajectory_linewidth == 6.5
    assert widget.trajectory_alpha == pytest.approx(0.6)
    assert widget.trajectory_dot_radius == pytest.approx(0.05)
    # The slider follows the spinbox.
    assert widget.trajectory_width_slider.value() == 65
    line = widget.current_donor_line
    assert line.get_linewidth() == pytest.approx(6.5)
    assert line.get_alpha() == pytest.approx(0.6)
    for circle in (
        widget.current_donor_circle,
        widget.current_background_circle,
    ):
        assert circle.get_radius() == pytest.approx(0.05)
        assert circle.get_alpha() == pytest.approx(0.6)

    # The style is a setting of the layer and is restored with it.
    widget.trajectory_width_spin.setValue(5.0)
    widget.trajectory_transparency_spin.setValue(0.25)
    widget.trajectory_dot_spin.setValue(0.04)
    fret_settings = parent.layer_settings(layer)['fret']
    assert fret_settings['trajectory_linewidth'] == 5.0
    assert fret_settings['trajectory_alpha'] == pytest.approx(0.75)
    assert fret_settings['trajectory_dot_radius'] == pytest.approx(0.04)
    widget.trajectory_linewidth = 3.0
    widget.trajectory_alpha = 1.0
    widget.trajectory_dot_radius = 0.02
    widget._restore_fret_settings_from_metadata()
    assert widget.trajectory_linewidth == 5.0
    assert widget.trajectory_alpha == pytest.approx(0.75)
    assert widget.trajectory_dot_radius == pytest.approx(0.04)
    assert widget.trajectory_width_spin.value() == 5.0
    assert widget.trajectory_transparency_spin.value() == pytest.approx(0.25)
    assert widget.trajectory_dot_slider.value() == 40
    widget.donor_line_edit.setText("2.0")
    widget.frequency_input.setText("80")
    widget.plot_donor_trajectory()

    # The flat color is offered only while the colormap overlay is off.
    assert widget.trajectory_color_row.isHidden()
    widget.colormap_checkbox.setChecked(False)
    assert not widget.trajectory_color_row.isHidden()
    with patch(
        'napari_phasors.fret_tab.QColorDialog.getColor',
        return_value=QColor('#ff0000'),
    ):
        widget.trajectory_color_button.click()
    assert widget.trajectory_color == '#ff0000'
    assert widget.current_donor_line.get_color() == '#ff0000'
    assert widget.current_donor_circle.get_facecolor()[:3] == (1.0, 0.0, 0.0)
    settings = widget.parent_widget.layer_settings(layer)['fret']
    assert settings['trajectory_color'] == '#ff0000'
    widget.colormap_checkbox.setChecked(True)
    assert widget.trajectory_color_row.isHidden()

    # Unchecking the dots removes them and disables their radius.
    assert widget.current_donor_circle is not None
    widget.trajectory_dots_checkbox.setChecked(False)
    assert widget.current_donor_circle is None
    assert widget.current_background_circle is None
    assert not widget.trajectory_dot_spin.isEnabled()
    assert (
        parent.layer_settings(layer)['fret']['show_trajectory_dots'] is False
    )
    widget.trajectory_dots_checkbox.setChecked(True)
    assert widget.current_donor_circle is not None
    assert widget.trajectory_dot_spin.isEnabled()

    # Reset restores every trajectory style default.
    widget.colormap_checkbox.setChecked(False)
    widget.trajectory_width_spin.setValue(8.0)
    widget.trajectory_transparency_spin.setValue(0.5)
    widget.trajectory_dot_spin.setValue(0.07)
    widget.trajectory_dots_checkbox.setChecked(False)
    widget.trajectory_color = '#00ff00'
    widget.trajectory_style_reset_button.click()
    assert widget.use_colormap is True
    assert widget.colormap_checkbox.isChecked()
    assert widget.trajectory_color_row.isHidden()
    assert widget.trajectory_linewidth == 3.0
    assert widget.trajectory_alpha == 1.0
    assert widget.trajectory_dot_radius == 0.02
    assert widget.trajectory_color == 'dimgray'
    assert widget.show_trajectory_dots is True
    assert widget.trajectory_width_spin.value() == 3.0
    assert widget.trajectory_width_slider.value() == 30
    assert widget.trajectory_dots_checkbox.isChecked()
    assert widget.current_donor_circle is not None
    settings = parent.layer_settings(layer)['fret']
    assert settings['trajectory_linewidth'] == 3.0
    assert settings['trajectory_color'] == 'dimgray'
    assert settings['show_trajectory_dots'] is True
    assert settings['use_colormap'] is True


def test_draw_fret_trajectory_overlay_uses_style_settings():
    """The standalone overlay honours the trajectory style settings."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    real = np.linspace(0.8, 0.2, 10)
    imag = np.linspace(0.3, 0.1, 10)
    draw_fret_trajectory_overlay(
        ax,
        real,
        imag,
        np.linspace(0, 1, 10),
        {
            "use_colormap": False,
            "trajectory_linewidth": 7.0,
            "trajectory_alpha": 0.3,
            "trajectory_dot_radius": 0.06,
            "trajectory_color": "#0000ff",
        },
    )
    assert ax.lines[0].get_color() == "#0000ff"
    assert ax.lines[0].get_linewidth() == pytest.approx(7.0)
    assert ax.lines[0].get_alpha() == pytest.approx(0.3)
    assert all(p.get_radius() == pytest.approx(0.06) for p in ax.patches)
    plt.close(fig)

    fig, ax = plt.subplots()
    draw_fret_trajectory_overlay(
        ax,
        real,
        imag,
        np.linspace(0, 1, 10),
        {"use_colormap": False, "show_trajectory_dots": False},
    )
    assert len(ax.patches) == 0
    plt.close(fig)


def headers_of(table):
    """Return the column headings of *table*."""
    return [
        table.horizontalHeaderItem(col).text()
        for col in range(table.columnCount())
    ]


def test_fret_efficiency_filter(make_viewer_model, qtbot):
    """The efficiency criterion: frozen parameters, pixels removed everywhere,
    histogram, report, toggling, following the trajectory, other tabs'
    criteria, and the layer's output name."""
    viewer = make_viewer_model()
    parent, widget, layer = _ready_fret_filter_widget(
        viewer, name="sample Intensity [Phasor]"
    )

    # The FRET map replaces the source's [Phasor] tag with [FRET efficiency].
    assert "sample Intensity [FRET efficiency]" in viewer.layers
    assert "FRET efficiency: sample Intensity [Phasor]" not in viewer.layers
    output = viewer.layers["sample Intensity [FRET efficiency]"]
    assert widget._fret_output_source(output) == layer.name

    baseline = layer.metadata['G'].copy()
    before_points = sum(
        len(values) for values in widget.histogram_widget._datasets.values()
    )
    median = float(np.nanmedian(output.metadata['fret_data_original']))

    # A new criterion captures the donor trajectory it was created with.
    card = _add_efficiency_filter(widget, 0.0, 1.0)
    params = card.entry['params']
    assert params['frequency'] == 80.0
    assert params['donor_lifetime'] == 4.2
    assert params['donor_fretting'] == widget.donor_fretting_proportion
    assert params['donor_background'] == widget.donor_background
    assert card.entry['harmonic'] == parent.harmonic
    assert widget._positive_float("abc") is None
    assert widget._positive_float("-1") is None
    assert widget._positive_float("2.5") == 2.5

    # Filtering by efficiency removes the pixels everywhere at once.
    before = np.isnan(layer.metadata['G']).sum()
    card = _add_efficiency_filter(widget, median, 1.0)
    assert np.isnan(layer.metadata['G']).sum() > before
    assert (
        np.isnan(layer.metadata['S']).sum()
        == np.isnan(layer.metadata['G']).sum()
    )
    output = viewer.layers[analysis_layer_name("FRET efficiency", layer.name)]
    assert np.isnan(output.data).any()
    survivors = output.data[np.isfinite(output.data)]
    assert survivors.min() >= median - 1e-9
    (stored,) = get_filters(layer)
    assert stored['metric'] == FRET_EFFICIENCY

    # The efficiency histogram only shows what survives the filter.
    plotted = np.concatenate(list(widget.histogram_widget._datasets.values()))
    assert 0 < len(plotted) < before_points
    assert plotted.min() >= median - 1e-9

    # Its pixel counts show beside the statistics, as a share of the pixels
    # the criterion started from.
    table = parent.fret_statistics_dock_widget.layer_stats_table
    headers = headers_of(table)
    pixels = next(i for i, name in enumerate(headers) if "Pixels in" in name)
    assert headers[pixels + 1].startswith("FRET efficiency % in 0.74")
    assert int(table.item(0, pixels).text()) == len(plotted)
    assert 0 < float(table.item(0, pixels + 1).text().rstrip("%")) < 100

    # The card and the summary say how much of the image survives; a single
    # filter needs no stack summary.
    assert "keeps" in card.stat_label.text()
    assert not widget.filter_list.summary_label.isVisibleTo(widget.filter_list)

    # Clearing the criterion restores every pixel it had hidden.
    card.enabled_check.setChecked(False)
    np.testing.assert_allclose(layer.metadata['G'], baseline)
    assert not any("Pixels" in name for name in headers_of(table))
    card.enabled_check.setChecked(True)
    assert np.isnan(layer.metadata['G']).any()
    card.enabled_check.setChecked(False)
    np.testing.assert_allclose(layer.metadata['G'], baseline)
    # Switched off, the criterion is kept (range and all) but hides nothing.
    (stored,) = get_filters(layer)
    assert stored['enabled'] is False

    # Recalculating with a new donor lifetime re-points the criterion.
    _add_efficiency_filter(widget, median, 1.0)
    widget.donor_line_edit.setText("2.0")
    widget.calculate_fret_efficiency()
    (stored,) = get_filters(layer)
    assert stored['params']['donor_lifetime'] == 2.0
    # A criterion already pointing at the current trajectory is left alone.
    assert widget._refresh_fret_filter_params([layer]) is False

    # Calling apply with no argument uses whatever the list currently holds.
    widget.filter_list.set_filters(
        [
            new_filter(
                FRET_EFFICIENCY,
                median,
                1.0,
                params=widget._fret_filter_params(),
            )
        ]
    )
    widget._apply_filter_stack()
    assert np.isnan(layer.metadata['G']).any()
    assert len(get_filters(layer)) == 1

    # A criterion with no donor trajectory is reported, not applied.
    warnings = []
    widget.donor_line_edit.setText("")
    with patch("napari_phasors.fret_tab.show_warning", warnings.append):
        widget._apply_filter_stack(
            [{'metric': FRET_EFFICIENCY, 'min': 0.2, 'max': 0.8}]
        )
    assert any(FRET_EFFICIENCY in message for message in warnings)
    assert not np.isnan(layer.metadata['G']).all()
    widget.donor_line_edit.setText("4.2")

    # Phasor Mapping criteria stay out of this list, and out of its way.
    set_filters(
        layer,
        [new_filter("Normal Lifetime", 0.0, 1.0, params={'frequency': 80.0})],
    )
    widget._sync_filter_ui()
    assert widget.filter_list.filters() == []
    assert len(widget.filter_list._cards) == 1
    assert _efficiency_card(widget).entry['metric'] == FRET_EFFICIENCY
    _add_efficiency_filter(widget, 0.0, 1.0)
    assert [f['metric'] for f in get_filters(layer)] == [
        "Normal Lifetime",
        FRET_EFFICIENCY,
    ]

    # Renaming the source renames the output.
    widget.rename_layer(layer.name, "renamed Intensity [Phasor]")
    assert output.name == "renamed Intensity [FRET efficiency]"

    # With the layer gone, the card falls back to a fresh, switched-off
    # criterion measured on the current harmonic.
    viewer.layers.remove(layer)
    assert widget.filter_list.filters() == []
    assert not _efficiency_card(widget).entry['enabled']
    assert _efficiency_card(widget).entry['harmonic'] == parent.harmonic


class _BrokenSelector:
    """Stands in for a plotter whose Qt selector has already been destroyed."""

    def get_selected_layers(self):
        """Raise the way a deleted Qt widget does when it is queried."""
        raise RuntimeError("wrapped C/C++ object has been deleted")
