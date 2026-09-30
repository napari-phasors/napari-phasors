import contextlib
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure
from napari.layers import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
)
from superqt import QRangeSlider, QToggleSwitch

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.plotter import PlotterWidget


def _applied_pairs(mock_apply):
    """Return the ``(layer, params)`` pairs handed to the filtering helper.

    ``apply_filter_and_threshold_to_layers`` takes one list of pairs rather
    than being called once per layer, so tests unpack that list instead of
    reading per-call keyword arguments.
    """
    return mock_apply.call_args[0][0]


def test_filter_widget_without_a_phasor_layer(make_viewer_model, qtbot):
    """The Filter tab builds its controls and histogram before any phasor
    layer exists, reacts to its own controls without replotting, ignores
    layers without phasors, and picks up the first phasor layer added."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    # Basic widget structure tests
    assert filter_widget.viewer == viewer
    assert filter_widget.parent_widget == parent
    assert isinstance(filter_widget.layout(), QVBoxLayout)

    # Test initial attribute values
    assert filter_widget._phasors_selected_layer is None
    assert filter_widget.threshold_factor == 1
    assert filter_widget.threshold_line_lower is None
    assert filter_widget.threshold_line_upper is None
    assert filter_widget.threshold_area_lower is None
    assert filter_widget.threshold_area_upper is None

    # Test histogram figure initialization
    assert isinstance(filter_widget.hist_fig, Figure)
    assert filter_widget.hist_ax is not None

    # Test filter method combobox
    assert isinstance(filter_widget.filter_method_combobox, QComboBox)
    assert filter_widget.filter_method_combobox.count() == 3
    assert filter_widget.filter_method_combobox.itemText(0) == "None"
    assert filter_widget.filter_method_combobox.itemText(1) == "Median"
    assert (
        filter_widget.filter_method_combobox.itemText(2)
        == "Wavelet (binlet pawFLIM)"
    )
    assert filter_widget.filter_method_combobox.currentText() == "None"

    # Test threshold method combobox
    assert isinstance(filter_widget.threshold_method_combobox, QComboBox)
    assert filter_widget.threshold_method_combobox.count() == 5
    assert filter_widget.threshold_method_combobox.itemText(0) == "None"
    assert filter_widget.threshold_method_combobox.itemText(1) == "Manual"
    assert filter_widget.threshold_method_combobox.itemText(2) == "Otsu"
    assert filter_widget.threshold_method_combobox.itemText(3) == "Li"
    assert filter_widget.threshold_method_combobox.itemText(4) == "Yen"
    assert filter_widget.threshold_method_combobox.currentText() == "None"

    # Test log scale checkbox, which starts unchecked
    assert isinstance(filter_widget.log_scale_checkbox, QToggleSwitch)
    assert filter_widget.log_scale_checkbox.text() == "Log Scale Histogram"
    assert not filter_widget.log_scale_checkbox.isChecked()

    # Test median filter UI components
    assert isinstance(filter_widget.median_filter_label, QLabel)
    assert filter_widget.median_filter_label.text() == "Kernel Size: 3 x 3"
    assert isinstance(filter_widget.median_filter_spinbox, QSpinBox)
    assert filter_widget.median_filter_spinbox.minimum() == 2
    assert filter_widget.median_filter_spinbox.maximum() == 99
    assert filter_widget.median_filter_spinbox.value() == 3
    assert isinstance(filter_widget.median_filter_repetition_spinbox, QSpinBox)
    assert filter_widget.median_filter_repetition_spinbox.minimum() == 1
    assert filter_widget.median_filter_repetition_spinbox.value() == 1

    # Test wavelet filter UI components
    assert isinstance(filter_widget.wavelet_sigma_spinbox, QDoubleSpinBox)
    assert filter_widget.wavelet_sigma_spinbox.minimum() == 0.1
    assert filter_widget.wavelet_sigma_spinbox.maximum() == 10.0
    assert filter_widget.wavelet_sigma_spinbox.value() == 2.0
    assert isinstance(filter_widget.wavelet_levels_spinbox, QSpinBox)
    assert filter_widget.wavelet_levels_spinbox.minimum() == 1
    assert filter_widget.wavelet_levels_spinbox.maximum() == 10
    assert filter_widget.wavelet_levels_spinbox.value() == 1

    # Test warning label for harmonics
    assert isinstance(filter_widget.harmonic_warning_label, QLabel)
    assert filter_widget.harmonic_warning_label.isHidden()

    # Test threshold editable text fields (min and max intensity)
    assert isinstance(filter_widget.min_threshold_edit, QLineEdit)
    assert filter_widget.min_threshold_edit.text() == "0.00"
    assert isinstance(filter_widget.max_threshold_edit, QLineEdit)
    assert filter_widget.max_threshold_edit.text() == "0.00"

    # Test threshold range slider
    assert isinstance(filter_widget.threshold_slider, QRangeSlider)
    assert filter_widget.threshold_slider.orientation() == Qt.Horizontal
    assert filter_widget.threshold_slider.minimum() == 0
    assert filter_widget.threshold_slider.maximum() == 100
    assert filter_widget.threshold_slider.value() == (0, 100)

    # Test apply button
    assert isinstance(filter_widget.apply_button, QPushButton)
    assert filter_widget.apply_button.text() == "Apply"

    # One resizable scroll area, with the Apply button outside it.
    scroll_areas = filter_widget.findChildren(QScrollArea)
    assert len(scroll_areas) == 1
    assert scroll_areas[0].widgetResizable()
    apply_buttons = filter_widget.findChildren(QPushButton)
    apply_button = [btn for btn in apply_buttons if btn.text() == "Apply"][0]
    assert apply_button == filter_widget.apply_button
    assert len(filter_widget.findChildren(QHBoxLayout)) >= 5

    # Test initial visibility of filter widgets
    assert filter_widget.median_filter_widget.isHidden()
    assert filter_widget.wavelet_filter_widget.isHidden()

    # The histogram is transparent with grey spines and labelled axes.
    import matplotlib.colors as mcolors

    assert filter_widget.hist_ax.patch.get_alpha() == 0
    assert filter_widget.hist_fig.patch.get_alpha() == 0
    grey_rgba = mcolors.to_rgba('grey')
    for spine in filter_widget.hist_ax.spines.values():
        np.testing.assert_array_almost_equal(spine.get_edgecolor(), grey_rgba)
        assert spine.get_linewidth() == 1
    assert filter_widget.hist_ax.get_ylabel() == "Count"
    assert filter_widget.hist_ax.get_xlabel() == "Mean Intensity"

    # Test canvas and figure properties
    assert filter_widget.hist_fig.get_figwidth() == 3.5
    assert filter_widget.hist_fig.get_figheight() == 1.5
    assert filter_widget.hist_fig.get_constrained_layout()
    canvas_widgets = filter_widget.findChildren(FigureCanvasQTAgg)
    assert len(canvas_widgets) == 1
    assert canvas_widgets[0].height() == 150

    # Without a layer there is nothing to plot or draw lines for.
    filter_widget.plot_mean_histogram()
    assert len(filter_widget.hist_ax.get_children()) >= 0
    filter_widget.update_threshold_lines()
    assert filter_widget.threshold_line_lower is None
    assert filter_widget.threshold_line_upper is None

    # Regression: _update_histogram_if_needed() used to connect signals on
    # every call (each tab switch), so callbacks fired N times after N
    # switches.
    def _receiver_count(combo_box):
        signal = combo_box.currentTextChanged
        with contextlib.suppress(TypeError):
            return combo_box.receivers(signal)

        # PySide variants expect a signal signature string.
        for signal_name in (
            "currentTextChanged(str)",
            "currentTextChanged(QString)",
        ):
            with contextlib.suppress(TypeError):
                return combo_box.receivers(signal_name)

        raise AssertionError(
            "Could not query signal receivers for currentTextChanged"
        )

    initial_receivers = _receiver_count(
        filter_widget.threshold_method_combobox
    )
    for _ in range(5):
        filter_widget._update_histogram_if_needed()
    final_receivers = _receiver_count(filter_widget.threshold_method_combobox)
    assert final_receivers == initial_receivers, (
        f"Signal has {final_receivers} receivers after 5 tab switches "
        f"(started with {initial_receivers}). "
        f"Connections are accumulating on each call to "
        f"_update_histogram_if_needed()."
    )

    # Switching between median and wavelet shows the matching parameters.
    for method, median_shown, wavelet_shown in (
        ("Median", True, False),
        ("Wavelet (binlet pawFLIM)", False, True),
        ("Median", True, False),
    ):
        filter_widget.filter_method_combobox.setCurrentText(method)
        filter_widget.on_filter_method_changed()
        assert filter_widget.median_filter_widget.isHidden() is not (
            median_shown
        )
        assert filter_widget.wavelet_filter_widget.isHidden() is not (
            wavelet_shown
        )

    # Changing a spinbox or the slider does not replot until Apply.
    with patch.object(parent, 'plot') as mock_plot:
        filter_widget.median_filter_spinbox.setValue(5)
        filter_widget.threshold_slider.setValue((5, 20))
        mock_plot.assert_not_called()

    filter_widget.on_median_kernel_size_change()
    assert filter_widget.median_filter_label.text() == "Kernel Size: 5 x 5"

    # The slider values are shown in intensity units.
    filter_widget.threshold_factor = 10
    filter_widget.threshold_slider.setValue((20, 80))
    filter_widget.on_threshold_slider_change()
    assert filter_widget.min_threshold_edit.text() == f"{20 / 10:.2f}"
    assert filter_widget.max_threshold_edit.text() == f"{80 / 10:.2f}"

    # With no layer selected, Apply does nothing.
    with patch.object(parent, 'plot') as mock_plot:
        with patch(
            'napari_phasors.filter_tab.apply_filter_and_threshold_to_layers'
        ):
            filter_widget.apply_button.click()
        mock_plot.assert_not_called()
    filter_widget.plot_mean_histogram()
    filter_widget.update_threshold_lines()

    # The log scale can be toggled without a layer.
    filter_widget.log_scale_checkbox.setChecked(True)
    filter_widget.on_log_scale_changed(True)
    assert filter_widget.log_scale_checkbox.isChecked()
    filter_widget.log_scale_checkbox.setChecked(False)
    filter_widget.on_log_scale_changed(False)

    # Automatic thresholds are non-negative, and 0 without usable data.
    test_data = np.random.rand(100, 100) * 100
    for method in ("Otsu", "Li", "Yen"):
        lower = filter_widget.calculate_automatic_threshold(method, test_data)
        assert isinstance(lower, (int, float))
        assert lower >= 0
    empty = np.array([])
    assert filter_widget.calculate_automatic_threshold("Otsu", empty) == 0
    nan_data = np.full((10, 10), np.nan)
    assert filter_widget.calculate_automatic_threshold("Otsu", nan_data) == 0

    # Layers without phasor features are not offered, and draw nothing.
    viewer.add_layer(Image(np.random.rand(10, 10), name="no_phasor_layer"))
    viewer.add_layer(Image(np.random.random((10, 10))))
    assert len(parent.image_layers_checkable_combobox.checkedItems()) == 0
    assert filter_widget.threshold_line_lower is None
    assert filter_widget.threshold_line_upper is None
    assert filter_widget.threshold_slider.maximum() == 100

    # Adding a phasor layer updates the slider range, the threshold lines
    # and the histogram once the tab shows it...
    intensity_image_layer1 = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer1)
    filter_tab_index = parent.tab_widget.indexOf(filter_widget)
    parent.tab_widget.setCurrentIndex(filter_tab_index)
    expected_max1 = int(
        np.ceil(
            np.nanmax(intensity_image_layer1.metadata["original_mean"])
            * filter_widget.threshold_factor
        )
    )
    assert filter_widget.threshold_slider.maximum() == expected_max1
    assert filter_widget.threshold_line_lower is not None
    assert filter_widget.threshold_line_upper is not None
    assert len(filter_widget.hist_ax.patches) > 0

    # ...but a second one only once it is selected.
    intensity_image_layer2 = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer2)
    assert filter_widget.threshold_slider.maximum() == expected_max1
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        intensity_image_layer2.name
    )
    parent.tab_widget.setCurrentIndex(filter_tab_index)
    expected_max2 = int(
        np.ceil(
            np.nanmax(intensity_image_layer2.metadata["original_mean"])
            * filter_widget.threshold_factor
        )
    )
    assert filter_widget.threshold_slider.maximum() == expected_max2
    assert filter_widget.threshold_line_lower is not None
    assert filter_widget.threshold_line_upper is not None
    assert len(filter_widget.hist_ax.patches) > 0


def test_threshold_methods(make_viewer_model, qtbot):
    """The slider spans the layer's intensities in a scaled integer range;
    None leaves it unconstrained, Otsu/Li/Yen place the lower bound, and
    moving the slider yourself switches to Manual. The chosen method is
    handed to the filtering helper on Apply."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    max_mean_value = np.nanmax(intensity_image_layer.metadata["original_mean"])
    expected_magnitude = int(np.log10(max_mean_value))
    expected_threshold_factor = (
        10 ** (2 - expected_magnitude) if expected_magnitude <= 2 else 1
    )
    assert filter_widget.threshold_factor == expected_threshold_factor
    expected_max = int(
        np.ceil(max_mean_value * filter_widget.threshold_factor)
    )
    assert filter_widget.threshold_slider.maximum() == expected_max

    # With "None" as default, slider should be at full range, and a slider
    # left at full range does not switch to Manual.
    assert filter_widget.threshold_slider.value() == (0, expected_max)
    assert filter_widget.threshold_method_combobox.currentText() == "None"
    filter_widget.on_threshold_method_changed()
    filter_widget.on_threshold_slider_change()
    assert filter_widget.threshold_method_combobox.currentText() == "None"

    # Automatic methods place the lower bound above zero.
    for method in ("Otsu", "Li", "Yen"):
        filter_widget.threshold_method_combobox.setCurrentText(method)
        filter_widget.on_threshold_method_changed()
        lower, _upper = filter_widget.threshold_slider.value()
        assert lower > 0, method

    # Manually changing the slider switches to Manual mode.
    filter_widget.threshold_method_combobox.setCurrentText("Otsu")
    filter_widget.on_threshold_method_changed()
    assert filter_widget.threshold_method_combobox.currentText() == "Otsu"
    filter_widget.threshold_slider.setValue((42, 80))
    filter_widget.on_threshold_slider_change()
    assert filter_widget.threshold_method_combobox.currentText() == "Manual"

    # Choosing None drops the lower bound again.
    filter_widget.threshold_slider.setValue((10, 37))
    filter_widget.on_threshold_slider_change()
    assert filter_widget.threshold_slider.value() == (10, 37)
    filter_widget.threshold_method_combobox.setCurrentText("None")
    filter_widget.on_threshold_method_changed()
    lower_val, _upper_val = filter_widget.threshold_slider.value()
    assert lower_val == 0
    assert filter_widget.min_threshold_edit.text() == "0.00"

    # The threshold method is passed through to the filtering helper.
    filter_widget.threshold_method_combobox.setCurrentText("Li")
    with patch(
        'napari_phasors.filter_tab.apply_filter_and_threshold_to_layers'
    ) as mock_apply:
        mock_apply.side_effect = lambda pairs, **kwargs: [None] * len(pairs)
        filter_widget.apply_button_clicked()
        mock_apply.assert_called_once()
        _, params = _applied_pairs(mock_apply)[0]
        assert params['threshold_method'] == "Li"


def create_image_layer_with_incompatible_harmonics():
    """Create an image layer with incompatible harmonics for wavelet filtering."""
    layer = create_image_layer_with_phasors()

    # Update harmonics in the new array-based metadata structure
    # Incompatible harmonics are non-consecutive (e.g., [1, 3, 5] instead of [1, 2])
    layer.metadata['harmonics'] = [1, 3, 5]

    return layer


def test_apply_button(make_viewer_model, qtbot):
    """Apply hands every selected layer to the filtering helper with the
    chosen median or wavelet parameters, stores unconstrained bounds as
    None, names layers that failed, and invalidates cached features and
    deferred tabs."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    intensity_image_layer.name = "bad_layer"
    # Set compatible harmonics (consecutive: 1, 2)
    intensity_image_layer.metadata['harmonics'] = [1, 2]
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab

    def patched_apply():
        return patch(
            'napari_phasors.filter_tab.apply_filter_and_threshold_to_layers'
        )

    # Compatible harmonics show the wavelet parameters, not a warning.
    filter_widget.filter_method_combobox.setCurrentText(
        "Wavelet (binlet pawFLIM)"
    )
    filter_widget.on_filter_method_changed()
    assert filter_widget.harmonic_warning_label.isHidden()
    assert not filter_widget.wavelet_params_widget.isHidden()

    filter_widget.wavelet_sigma_spinbox.setValue(1.5)
    filter_widget.wavelet_levels_spinbox.setValue(2)
    filter_widget.threshold_slider.setValue((10, 90))
    filter_widget.threshold_method_combobox.setCurrentText("Manual")
    with (
        patched_apply() as mock_apply,
        patch.object(parent, 'plot') as mock_plot,
    ):
        mock_apply.side_effect = lambda pairs, **kwargs: [None] * len(pairs)
        filter_widget.apply_button_clicked()
        mock_apply.assert_called_once()
        layer, params = _applied_pairs(mock_apply)[0]
        assert layer == intensity_image_layer
        assert params['filter_method'] == 'wavelet'
        assert params['sigma'] == 1.5
        assert params['levels'] == 2
        assert 'harmonics' in params
        mock_plot.assert_called_once()

    filter_widget.filter_method_combobox.setCurrentText("Median")
    filter_widget.on_filter_method_changed()
    filter_widget.median_filter_spinbox.setValue(5)
    filter_widget.median_filter_repetition_spinbox.setValue(2)
    with patched_apply() as mock_apply, patch.object(parent, 'plot'):
        mock_apply.side_effect = lambda pairs, **kwargs: [None] * len(pairs)
        filter_widget.apply_button_clicked()
        mock_apply.assert_called_once()
        layer, params = _applied_pairs(mock_apply)[0]
        assert layer == intensity_image_layer
        assert params['filter_method'] == 'median'
        assert params['size'] == 5
        assert params['repeat'] == 2

    # Regression: handles left at the slider extremes persist as None.
    # Storing the current data max as an explicit upper bound froze a
    # masked max, which then persisted after the mask was removed.
    slider = filter_widget.threshold_slider
    slider.setValue((slider.minimum(), slider.maximum()))
    filter_widget.threshold_method_combobox.setCurrentText("Manual")
    with patched_apply() as mock_apply:
        mock_apply.side_effect = lambda pairs, **kwargs: [None] * len(pairs)
        filter_widget.apply_button_clicked()
        _, params = _applied_pairs(mock_apply)[0]
        assert params['threshold'] is None
        assert params['threshold_upper'] is None
    # A constrained max should still be persisted as a concrete value.
    upper = slider.maximum() - 1
    slider.setValue((slider.minimum(), upper))
    with patched_apply() as mock_apply:
        mock_apply.side_effect = lambda pairs, **kwargs: [None] * len(pairs)
        filter_widget.apply_button_clicked()
        _, params = _applied_pairs(mock_apply)[0]
        assert params['threshold_upper'] == (
            upper / filter_widget.threshold_factor
        )

    # A layer whose filtering raised is named in a single grouped error.
    errors = []
    with (
        patched_apply() as mock_apply,
        patch('napari_phasors.filter_tab.show_error', errors.append),
        patch.object(parent, 'plot'),
    ):
        mock_apply.side_effect = lambda pairs, **kwargs: [
            RuntimeError("kernel exploded") for _ in pairs
        ]
        filter_widget.apply_button_clicked()
    assert len(errors) == 1
    assert "Could not filter 1 layer(s)" in errors[0]
    assert "bad_layer" in errors[0]
    assert "kernel exploded" in errors[0]

    # Regression: the plotter caches merged features keyed on (selected
    # layer names, harmonic), which a filter does not change, so
    # refresh_phasor_data() must drop the stale cache.
    assert parent.get_merged_features() is not None
    assert parent._features_cache is not None
    sentinel = ('stale-sentinel', 'data')
    parent._features_cache = sentinel
    parent.refresh_phasor_data()
    assert parent._features_cache is not sentinel, (
        "refresh_phasor_data() did not invalidate stale features cache "
        "— threshold/filter changes will not be reflected in the plot."
    )

    # Regression: deferred tabs must be marked stale after a real
    # filter/threshold is applied, so they refresh when next visible.
    for tab_attr in ('phasor_mapping_tab', 'components_tab', 'fret_tab'):
        getattr(parent, tab_attr)._needs_update = False
    # A non-deferrable tab, so deferred tabs don't auto-restore.
    parent.tab_widget.setCurrentWidget(filter_widget)
    filter_widget.filter_method_combobox.setCurrentText("None")
    filter_widget.threshold_method_combobox.setCurrentText("Manual")
    filter_widget.threshold_slider.setValue((20, 90))
    filter_widget.apply_button_clicked()
    for tab_attr in ('phasor_mapping_tab', 'components_tab', 'fret_tab'):
        assert getattr(parent, tab_attr)._needs_update is True, (
            f"{tab_attr} was not marked for deferred update after "
            f"filter/threshold was applied."
        )


def test_filter_settings_are_restored_per_layer(make_viewer_model, qtbot):
    """Each layer's stored filter method, parameters and thresholds are
    restored when it becomes the current layer. Wavelet filtering on
    harmonics it cannot handle falls back to median with a warning, and a
    layer whose intensities start above zero bounds the slider minimum."""
    viewer = make_viewer_model()

    def add_layer(name, settings, harmonic=None, harmonics=None):
        layer = create_image_layer_with_phasors(harmonic=harmonic)
        layer.name = name
        if harmonics is not None:
            layer.metadata['harmonics'] = harmonics
        layer.metadata["settings"] = settings
        viewer.add_layer(layer)
        return layer

    thresholds = {
        "threshold": 0.1,
        "threshold_upper": 5.0,
        "threshold_method": "Li",
    }
    wavelet = add_layer(
        "wavelet",
        {
            **thresholds,
            "filter": {"method": "wavelet", "sigma": 3.5, "levels": 4},
        },
        harmonic=[1, 2],
        harmonics=[1, 2],
    )
    median = add_layer(
        "median",
        {**thresholds, "filter": {"method": "median", "size": 7, "repeat": 3}},
    )
    # Incompatible harmonics are non-consecutive (e.g. [1, 3, 5]).
    incompatible = add_layer(
        "incompatible",
        {"filter": {"method": "wavelet", "sigma": 3.5, "levels": 4}},
        harmonics=[1, 3, 5],
    )
    full = add_layer(
        "full",
        {
            "filter": {
                "method": "wavelet",
                "size": 5,
                "repeat": 2,
                "sigma": 3.0,
                "levels": 4,
            },
            "threshold": 2.0,
            "threshold_upper": 8.0,
        },
    )
    # An already-thresholded intensity image: nothing below 5.0, and no
    # stored threshold.
    shifted = add_layer("shifted", {})
    mean = shifted.metadata["original_mean"].astype(float)
    mean[:] = np.linspace(5.0, 40.0, mean.size).reshape(mean.shape)
    shifted.metadata["original_mean"] = mean

    parent = PlotterWidget(viewer)
    fw = parent.filter_tab

    def select(layer):
        parent.image_layers_checkable_combobox.setCheckedItems([layer.name])
        parent._process_layer_selection_change()
        fw._on_image_layer_changed()

    def assert_thresholds_restored():
        lower_val, upper_val = fw.threshold_slider.value()
        assert lower_val == int(0.1 * fw.threshold_factor)
        # 5.0 is far above the data maximum, so it is clamped to it.
        assert upper_val == min(
            int(5.0 * fw.threshold_factor), fw.threshold_slider.maximum()
        )
        assert fw.threshold_method_combobox.currentText() == "Li"

    select(wavelet)
    assert (
        fw.filter_method_combobox.currentText() == "Wavelet (binlet pawFLIM)"
    )
    assert fw.wavelet_sigma_spinbox.value() == 3.5
    assert fw.wavelet_levels_spinbox.value() == 4
    assert_thresholds_restored()

    select(median)
    assert fw.filter_method_combobox.currentText() == "Median"
    assert fw.median_filter_spinbox.value() == 7
    assert fw.median_filter_repetition_spinbox.value() == 3
    assert_thresholds_restored()

    select(incompatible)
    assert fw.filter_method_combobox.currentText() == "Median"
    # Picking wavelet anyway warns instead of offering its parameters.
    fw.filter_method_combobox.setCurrentText("Wavelet (binlet pawFLIM)")
    fw.on_filter_method_changed()
    assert not fw.harmonic_warning_label.isHidden()
    assert fw.wavelet_params_widget.isHidden()
    warning_text = fw.harmonic_warning_label.text()
    assert "Warning: Harmonics" in warning_text
    assert "not compatible" in warning_text

    # A layer carrying full filter+threshold settings restores all widgets.
    select(full)
    assert fw.median_filter_spinbox.value() == 5
    assert fw.median_filter_repetition_spinbox.value() == 2
    assert fw.wavelet_sigma_spinbox.value() == 3.0
    assert fw.wavelet_levels_spinbox.value() == 4

    # Regression: importing an already-thresholded image left the lower
    # threshold line to the left of the visible histogram because the
    # slider minimum was hard-coded to zero.
    select(shifted)
    expected_min = int(5.0 * fw.threshold_factor)
    assert fw.threshold_slider.minimum() == expected_min
    lower_val, _ = fw.threshold_slider.value()
    assert lower_val == expected_min
    assert fw.min_threshold_edit.text() == f"{5.0:.2f}"


def test_threshold_edits_and_lines(make_viewer_model, qtbot):
    """The min/max fields, the slider and the draggable histogram lines stay
    in step: edits move the slider (switching to Manual), are clamped to
    each other, and invalid text is reset."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    fw = parent.filter_tab
    fw._on_image_layer_changed()
    fw.plot_mean_histogram()
    factor = fw.threshold_factor

    # Invalid input in the threshold fields resets to the current value.
    lower_val, upper_val = fw.threshold_slider.value()
    expected_min = f"{lower_val / factor:.2f}"
    expected_max = f"{upper_val / factor:.2f}"
    fw.min_threshold_edit.setText("invalid")
    fw.on_min_threshold_edit_changed()
    assert fw.min_threshold_edit.text() == expected_min
    fw.max_threshold_edit.setText("also_invalid")
    fw.on_max_threshold_edit_changed()
    assert fw.max_threshold_edit.text() == expected_max

    # Changing the slider value moves the threshold lines.
    fw.threshold_slider.setValue((5, 20))
    fw.on_threshold_slider_change()
    assert fw.threshold_line_lower is not None
    assert fw.threshold_line_upper is not None
    line_data_lower = fw.threshold_line_lower.get_xdata()
    line_data_upper = fw.threshold_line_upper.get_xdata()
    assert line_data_lower[0] == line_data_lower[1] == 5 / factor
    assert line_data_upper[0] == line_data_upper[1] == 20 / factor

    # The lines can be dragged with the mouse.
    class Ev:
        def __init__(self, x, inaxes):
            self.xdata = x
            self.ydata = 0.0
            self.inaxes = inaxes
            self.button = 1

    lower_x = 5 / factor
    upper_x = 20 / factor
    # Press on the lower line, drag it, release.
    fw.on_mouse_press(Ev(lower_x, fw.hist_ax))
    assert fw._dragging_line == "lower"
    fw.on_mouse_move(Ev(10 / factor, fw.hist_ax))
    fw.on_mouse_release(Ev(10 / factor, fw.hist_ax))
    assert fw._dragging_line is None
    # Press on the upper line, drag it, release.
    fw.on_mouse_press(Ev(upper_x, fw.hist_ax))
    assert fw._dragging_line == "upper"
    fw.on_mouse_move(Ev(15 / factor, fw.hist_ax))
    fw.on_mouse_release(Ev(15 / factor, fw.hist_ax))
    assert fw._dragging_line is None
    # Press outside the axes does nothing.
    fw.on_mouse_press(Ev(lower_x, None))
    assert fw._dragging_line is None
    # Move while not dragging is a no-op.
    fw.on_mouse_move(Ev(lower_x, fw.hist_ax))

    # Editing the minimum moves the slider and the line, and switches to
    # Manual.
    fw.threshold_method_combobox.setCurrentText("None")
    fw.threshold_slider.setValue((10, 80))
    fw.on_threshold_slider_change()
    fw.min_threshold_edit.setText("0.15")
    fw.on_min_threshold_edit_changed()
    lower_val, _ = fw.threshold_slider.value()
    assert lower_val == int(0.15 * factor)
    assert fw.threshold_method_combobox.currentText() == "Manual"

    fw.threshold_slider.setValue((10, 80))
    fw.on_threshold_slider_change()
    fw.min_threshold_edit.setText("0.2")
    fw.on_min_threshold_edit_changed()
    assert fw.threshold_line_lower is not None
    expected_x = int(0.2 * factor) / factor
    assert abs(fw.threshold_line_lower.get_xdata()[0] - expected_x) < 0.01

    # Editing the maximum moves the slider too. The value is rounded when
    # shown with .2f and converted back.
    fw.threshold_method_combobox.setCurrentText("None")
    max_mean = np.nanmax(intensity_image_layer.metadata["original_mean"])
    fw.max_threshold_edit.setText(f"{max_mean * 0.8:.2f}")
    fw.on_max_threshold_edit_changed()
    _, upper_val = fw.threshold_slider.value()
    assert upper_val == int(float(f"{max_mean * 0.8:.2f}") * factor)
    assert fw.threshold_method_combobox.currentText() == "Manual"

    # The minimum cannot exceed the maximum, nor the maximum drop below the
    # minimum.
    fw.threshold_slider.setValue((10, 50))
    fw.on_threshold_slider_change()
    fw.min_threshold_edit.setText("100.0")
    fw.on_min_threshold_edit_changed()
    lower_val, upper_val = fw.threshold_slider.value()
    assert lower_val <= upper_val

    fw.threshold_slider.setValue((30, 80))
    fw.on_threshold_slider_change()
    fw.max_threshold_edit.setText("0.5")
    fw.on_max_threshold_edit_changed()
    lower_val, upper_val = fw.threshold_slider.value()
    assert upper_val >= lower_val


def test_log_scale_histogram(make_viewer_model, qtbot):
    """The histogram's log scale can be toggled, and a redraw for a newly
    selected layer keeps it."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    parent = PlotterWidget(viewer)
    filter_widget = parent.filter_tab
    parent.tab_widget.setCurrentWidget(filter_widget)

    filter_widget.plot_mean_histogram()
    assert filter_widget.hist_ax.get_yscale() == 'linear'
    assert not filter_widget.log_scale_checkbox.isChecked()

    filter_widget.log_scale_checkbox.setChecked(True)
    filter_widget.on_log_scale_changed(2)  # Qt.Checked = 2
    assert filter_widget.hist_ax.get_yscale() == 'log'

    filter_widget.log_scale_checkbox.setChecked(False)
    filter_widget.on_log_scale_changed(0)  # Qt.Unchecked = 0
    assert filter_widget.hist_ax.get_yscale() == 'linear'

    filter_widget.log_scale_checkbox.setChecked(True)
    filter_widget.on_log_scale_changed(2)
    second_layer = create_image_layer_with_phasors()
    viewer.add_layer(second_layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        second_layer.name
    )
    filter_widget._on_image_layer_changed()
    assert filter_widget.log_scale_checkbox.isChecked()
    assert filter_widget.hist_ax.get_yscale() == 'log'


def test_thresholds_survive_masking(make_viewer_model, qtbot):
    """A mask narrows the intensity range the slider spans, but thresholds
    the user set outside it are remembered and re-applied, and the full
    range comes back once the mask is removed.

    Regression: masking raised the slider minimum (or lowered the maximum),
    the restored handle was clamped to it, and applying then stored None,
    silently discarding the user's threshold once the mask was removed.
    """
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    fw = parent.filter_tab

    # With a mask the masked region is used; without it the full image.
    layer.metadata['mask'] = (layer.metadata["original_mean"] > 0).astype(int)
    fw._on_image_layer_changed()
    assert fw.threshold_slider.maximum() > 0
    del layer.metadata['mask']
    fw._on_image_layer_changed()
    assert fw.threshold_slider.maximum() > 0

    om = layer.metadata["original_mean"]
    # A lower threshold between the full-data minimum and the masked one.
    bright = om >= np.nanpercentile(om, 70)
    lower = float(np.nanpercentile(om, 40))
    fw.threshold_method_combobox.setCurrentText("Manual")
    fw.min_threshold_edit.setText(f"{lower:.2f}")
    fw.on_min_threshold_edit_changed()
    fw.apply_button_clicked()
    stored = layer.metadata["settings"]["threshold"]
    assert stored is not None
    assert stored < om[bright].min()

    # Mask in only the bright pixels, then re-apply as the mask handlers do.
    layer.metadata["mask"] = bright.astype(int)
    fw._on_image_layer_changed()
    assert fw._offscreen_threshold_lower == stored
    fw.apply_button_clicked()
    assert layer.metadata["settings"]["threshold"] == stored

    # Moving the handle yourself supersedes the remembered value.
    fw.threshold_slider.setValue(
        (fw.threshold_slider.minimum(), fw.threshold_slider.maximum())
    )
    assert fw._offscreen_threshold_lower is None
    fw.apply_button_clicked()
    assert layer.metadata["settings"]["threshold"] is None
    del layer.metadata["mask"]
    fw._on_image_layer_changed()

    # The upper bound gets the same treatment: keeping only the dimmest
    # pixels drops the slider maximum well below the user's upper
    # threshold, whose handle then sits at the extreme, which would
    # otherwise read as "no limit".
    dim = om <= np.nanpercentile(om, 30)
    upper = float(np.nanpercentile(om, 85))
    fw.threshold_method_combobox.setCurrentText("Manual")
    # A lower bound too, since the two are restored together.
    fw.min_threshold_edit.setText("0.01")
    fw.on_min_threshold_edit_changed()
    fw.max_threshold_edit.setText(f"{upper:.2f}")
    fw.on_max_threshold_edit_changed()
    fw.apply_button_clicked()
    stored = layer.metadata["settings"]["threshold_upper"]
    assert stored is not None
    assert stored > om[dim].max()
    layer.metadata["mask"] = dim.astype(int)
    fw._on_image_layer_changed()
    assert fw._offscreen_threshold_upper == stored
    fw.apply_button_clicked()
    assert layer.metadata["settings"]["threshold_upper"] == stored

    # A layer that carries no settings starts from an unconstrained slider.
    del layer.metadata["mask"]
    fw._offscreen_threshold_lower = 1.0
    fw._offscreen_threshold_upper = 2.0
    del layer.metadata["settings"]
    fw._on_image_layer_changed()
    assert fw._offscreen_threshold_lower is None
    assert fw._offscreen_threshold_upper is None
    assert fw.threshold_method_combobox.currentText() == "None"

    # Deselecting all phasor layers clears the intensity histogram.
    fw.plot_mean_histogram()
    assert fw._histogram_data is not None
    assert len(fw.hist_ax.patches) > 0
    with patch.object(parent, 'get_selected_layers', return_value=[]):
        fw._on_image_layer_changed()
    assert fw._histogram_data is None
    assert len(fw.hist_ax.patches) == 0


def test_unconstrained_upper_threshold_follows_the_data(
    make_viewer_model, qtbot
):
    """A stored upper threshold of None pins the handle to the slider
    maximum, with or without a mask, and removing a mask brings the
    full-data maximum back.

    Regression: because the slider maximum is ``ceil``-rounded while the
    upper value was ``int``-truncated, an unconstrained max landed one tick
    below the maximum, was mistaken for a user constraint, and got frozen;
    and a reduced max stuck after the mask was removed.
    """
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    layer.metadata["settings"] = {
        "threshold": 0.05,
        "threshold_upper": None,
        "threshold_method": "Manual",
    }
    viewer.add_layer(layer)
    parent = PlotterWidget(viewer)
    fw = parent.filter_tab
    fw._on_image_layer_changed()

    om = layer.metadata["original_mean"]
    full_max_real = np.nanmax(om)
    full_factor = fw.threshold_factor
    full_max = fw.threshold_slider.maximum()
    _, upper_full = fw.threshold_slider.value()
    assert upper_full == full_max

    # Mask in only the below-median pixels so the visible max drops (in
    # real units: the scaled value is not comparable because
    # ``threshold_factor`` rescales with the magnitude).
    masked_max_real = np.nanmax(om[om <= np.median(om)])
    layer.metadata["mask"] = (om <= np.median(om)).astype(int)
    fw._on_image_layer_changed()
    assert masked_max_real < full_max_real
    _, upper_masked = fw.threshold_slider.value()
    assert upper_masked == fw.threshold_slider.maximum()
    assert upper_masked / fw.threshold_factor < full_max_real

    # Remove the mask: the full range and unconstrained max must return.
    del layer.metadata["mask"]
    fw._on_image_layer_changed()
    assert fw.threshold_factor == full_factor
    assert fw.threshold_slider.maximum() == full_max
    _, upper_restored = fw.threshold_slider.value()
    assert upper_restored == full_max


def test_apply_filter_to_no_layers_is_a_no_op():
    """An empty batch returns an empty result rather than starting a pool."""
    from napari_phasors._utils import apply_filter_and_threshold_to_layers

    assert apply_filter_and_threshold_to_layers([]) == []
