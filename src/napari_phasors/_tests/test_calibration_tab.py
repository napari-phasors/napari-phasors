from unittest.mock import patch

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from phasorpy.lifetime import (
    phasor_calibrate,
    phasor_from_lifetime,
    polar_from_reference_phasor,
)
from phasorpy.phasor import phasor_center

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.calibration_tab import _HTML_LABEL_ROLE
from napari_phasors.plotter import PlotterWidget


def test_calibration_widget_without_a_phasor_layer(make_viewer_model, qtbot):
    """Before a sample is selected the tab shows empty inputs, refuses to
    calibrate, fills the lifetime from the fluorophore picker (which a
    manual edit resets) and tracks phasor layers as they come and go."""
    from qtpy.QtGui import QPainter, QPixmap
    from qtpy.QtWidgets import QStyle, QStyleOptionViewItem

    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab
    calibration_widget = widget.calibration_widget

    # Basic widget structure tests
    assert widget.viewer == viewer
    assert widget.parent_widget == parent
    assert widget.layout().count() > 0

    # Test initial UI state
    assert calibration_widget.frequency_input.text() == ""
    assert calibration_widget.lifetime_line_edit_widget.text() == ""
    assert calibration_widget.calibrate_push_button.text() == "Calibrate"
    combobox = calibration_widget.calibration_layer_combobox
    assert combobox.count() == 0

    with patch("napari_phasors.calibration_tab.show_error") as mock_show_error:
        widget._on_click()
        mock_show_error.assert_called_once_with(
            "Select sample and calibration layers"
        )
    assert widget._uncalibrate_layer("") is None
    widget._update_button_state()
    assert calibration_widget.calibrate_push_button.text() == "Calibrate"

    # Inverting the calibration parameters.
    phi_inv, mod_inv = widget._invert_calibration_parameters(0.5, 2.0)
    assert phi_inv == -0.5
    assert mod_inv == 0.5

    # A re-entrant populate returns early and leaves the guard set.
    widget._populating_comboboxes = True
    widget._populate_comboboxes()
    assert widget._populating_comboboxes is True
    widget._populating_comboboxes = False

    # Selecting a reference fluorophore fills the lifetime edit; the first
    # item is the placeholder and carries no lifetime.
    fluorophores = calibration_widget.fluorophore_combobox
    lifetime_edit = calibration_widget.lifetime_line_edit_widget
    assert fluorophores.itemData(0) is None
    assert fluorophores.count() > 1
    fluorophores.setCurrentIndex(1)
    assert lifetime_edit.text() == f"{fluorophores.itemData(1):g}"

    # Editing the lifetime by hand resets the fluorophore selection
    # (textEdited fires only on user input).
    lifetime_edit.setText("2.5")
    widget._on_lifetime_edited("2.5")
    assert fluorophores.currentIndex() == 0

    # Selecting the placeholder leaves the lifetime edit untouched.
    lifetime_edit.setText("3.14")
    widget._on_fluorophore_selected(0)
    assert lifetime_edit.text() == "3.14"

    # Programmatic lifetime edits (from the combo) must not reset the combo:
    # while the guard flag is set, _on_lifetime_edited returns early.
    fluorophores.setCurrentIndex(1)
    assert fluorophores.currentIndex() == 1
    widget._setting_lifetime_from_combo = True
    widget._on_lifetime_edited("9.9")
    assert fluorophores.currentIndex() == 1
    widget._setting_lifetime_from_combo = False

    # The delegate renders an item's HTML label (also while selected, for
    # the highlighted-text color), and falls back to the default painting
    # for the placeholder, which has none.
    delegate = fluorophores.itemDelegate()
    model = fluorophores.model()
    pixmap = QPixmap(200, 20)
    painter = QPainter(pixmap)
    try:
        option = QStyleOptionViewItem()
        option.rect = pixmap.rect()
        plain_index = model.index(0, 0)
        assert plain_index.data(_HTML_LABEL_ROLE) is None
        delegate.paint(painter, option, plain_index)
        html_index = model.index(1, 0)
        assert html_index.data(_HTML_LABEL_ROLE) is not None
        delegate.paint(painter, option, html_index)
        option.state |= QStyle.State_Selected
        delegate.paint(painter, option, html_index)
    finally:
        painter.end()

    # Filling the lifetime from metadata must not look like the user
    # typing: a feedback loop would clear the fluorophore selection.
    edits = []
    lifetime_edit.textChanged.connect(edits.append)
    parent._set_calibration_lifetime_from_metadata("3.5")
    assert lifetime_edit.text() == "3.5"
    assert edits == []
    lifetime_edit.textChanged.disconnect(edits.append)

    # The calibration layer combobox follows phasor layers being added and
    # removed.
    test_layer = create_image_layer_with_phasors()
    test_layer.name = "test_layer"
    viewer.add_layer(test_layer)
    assert combobox.count() == 1
    assert combobox.itemText(0) == "test_layer"
    viewer.layers.remove("test_layer")
    assert combobox.count() == 0

    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    calibration_layer = create_image_layer_with_phasors()
    calibration_layer.name = "calibration_layer"
    viewer.add_layer(sample_layer)
    viewer.add_layer(calibration_layer)
    assert combobox.count() == 2
    layer_names = [combobox.itemText(i) for i in range(combobox.count())]
    assert "sample_layer" in layer_names
    assert "calibration_layer" in layer_names

    # Closing unhooks the widget; a second close is harmless.
    class MockEvent:
        accepted = False

        def accept(self):
            self.accepted = True

    event = MockEvent()
    widget.closeEvent(event)
    widget.closeEvent(event)


def test_calibration_click_reports_missing_inputs(make_viewer_model, qtbot):
    """Calibrating needs a calibration layer, a frequency, a reference
    lifetime and matching harmonics; uncalibrating needs a calibrated
    layer. The button reads Uncalibrate for a calibrated layer."""
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab
    calibration_widget = widget.calibration_widget

    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    calibration_layer = create_image_layer_with_phasors()
    calibration_layer.name = "calibration_layer"
    mismatched_layer = create_image_layer_with_phasors()
    mismatched_layer.name = "mismatched_layer"
    mismatched_layer.metadata["harmonics"] = [
        h + 1 for h in sample_layer.metadata["harmonics"]
    ]
    viewer.add_layer(sample_layer)
    viewer.add_layer(calibration_layer)
    viewer.add_layer(mismatched_layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(
        "sample_layer"
    )
    combobox = calibration_widget.calibration_layer_combobox

    def assert_click_reports(message):
        with patch(
            "napari_phasors.calibration_tab.show_error"
        ) as mock_show_error:
            widget._on_click()
            mock_show_error.assert_called_once_with(message)

    combobox.setCurrentIndex(-1)
    assert_click_reports("Select sample and calibration layers")

    combobox.setCurrentText("calibration_layer")
    assert_click_reports("Enter frequency")

    calibration_widget.frequency_input.setText("80")
    assert_click_reports("Enter reference lifetime")

    combobox.setCurrentText("mismatched_layer")
    calibration_widget.lifetime_line_edit_widget.setText("2")
    assert_click_reports(
        "Harmonics in sample and calibration layers do not match"
    )

    # The button follows the selected layer's stored calibration status,
    # read straight from its metadata (the no-layer case is covered above).
    button = calibration_widget.calibrate_push_button
    widget._update_button_state()
    assert button.text() == "Calibrate"

    # A layer is not calibrated until it has both a phase and a modulation.
    with patch("napari_phasors.calibration_tab.show_error") as mock_show_error:
        widget._uncalibrate_layer("sample_layer")
        mock_show_error.assert_called_once_with("Layer is not calibrated")
    sample_layer.metadata["settings"].update(
        {"calibrated": True, "calibration_phase": 0.5}
    )
    with patch("napari_phasors.calibration_tab.show_error") as mock_show_error:
        widget._uncalibrate_layer("sample_layer")
        mock_show_error.assert_called_once_with("Layer is not calibrated")
    widget._update_button_state()
    assert button.text() == "Uncalibrate"

    sample_layer.metadata["settings"] = {}
    widget._update_button_state()
    assert button.text() == "Calibrate"


def test_calibrate_and_uncalibrate_a_layer(make_viewer_model, qtbot):
    """Calibrating matches phasorpy's reference calibration and leaves the
    intensity untouched; uncalibrating restores the original phasors, and
    both keep the layer's median or wavelet filter and thresholds."""
    from napari_phasors._utils import apply_filter_and_threshold

    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab
    calibration_widget = widget.calibration_widget

    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    viewer.add_layer(sample_layer)
    calibration_widget.calibration_layer_combobox.setCurrentText(
        "sample_layer"
    )
    calibration_widget.frequency_input.setText("80")
    calibration_widget.lifetime_line_edit_widget.setText("2.0")
    button = calibration_widget.calibrate_push_button

    # Make copies of the original data before calibration modifies it
    original_image = sample_layer.data.copy()
    g_original = sample_layer.metadata['G_original'].copy()
    s_original = sample_layer.metadata['S_original'].copy()
    mean_original = sample_layer.metadata['original_mean'].copy()
    mean_shape = mean_original.shape
    harmonic = sample_layer.metadata['harmonics']
    frequency = [80 * h for h in harmonic]
    lifetime = 2.0

    # Reshape for phasor_calibrate if needed
    if g_original.ndim == mean_original.ndim + 1:
        g_reshaped = g_original
        s_reshaped = s_original
    else:
        g_reshaped = g_original.reshape((len(harmonic),) + mean_shape)
        s_reshaped = s_original.reshape((len(harmonic),) + mean_shape)

    # Calculate expected values using copies
    real, imag = phasor_calibrate(
        g_reshaped,
        s_reshaped,
        mean_original,
        g_reshaped,
        s_reshaped,
        frequency=frequency,
        lifetime=lifetime,
    )
    _, real_center, imag_center = phasor_center(
        mean_original, g_reshaped, s_reshaped
    )
    known_re, known_im = phasor_from_lifetime(frequency, lifetime)
    phi, mod = polar_from_reference_phasor(
        real_center, imag_center, known_re, known_im
    )

    button.click()
    assert sample_layer.metadata["settings"]["calibrated"] is True
    assert_array_equal(
        sample_layer.metadata["settings"]["calibration_phase"], phi
    )
    assert_array_equal(
        sample_layer.metadata["settings"]["calibration_modulation"], mod
    )
    assert_array_equal(sample_layer.metadata['G'], real)
    assert_array_equal(sample_layer.metadata['S'], imag)
    assert not np.array_equal(g_original, sample_layer.metadata['G'])
    assert not np.array_equal(s_original, sample_layer.metadata['S'])
    assert_array_equal(sample_layer.metadata['original_mean'], mean_original)
    assert_array_equal(sample_layer.data, original_image)

    # Uncalibrate
    button.click()
    settings = sample_layer.metadata["settings"]
    assert settings["calibrated"] is False
    assert "calibration_phase" not in settings
    assert "calibration_modulation" not in settings
    assert_almost_equal(sample_layer.metadata['G'], g_original)
    assert_almost_equal(sample_layer.metadata['S'], s_original)
    assert_almost_equal(sample_layer.metadata['original_mean'], mean_original)
    assert_almost_equal(sample_layer.data, original_image)

    def assert_filter_kept(filter_kwargs, expected):
        settings = sample_layer.metadata["settings"]
        for key, value in expected.items():
            assert settings["filter"][key] == value
        assert settings["threshold"] == filter_kwargs["threshold"]
        assert settings["threshold_method"] == (
            filter_kwargs["threshold_method"]
        )
        if "threshold_upper" in filter_kwargs:
            assert settings["threshold_upper"] == (
                filter_kwargs["threshold_upper"]
            )

    # Filters and thresholds are preserved through calibration and
    # uncalibration: median, wavelet with sigma, and wavelet with levels.
    for filter_kwargs, expected in (
        (
            {
                "threshold": 0.1,
                "threshold_upper": 0.9,
                "threshold_method": "Manual",
                "filter_method": "median",
                "size": 3,
                "repeat": 1,
            },
            {"method": "median", "size": 3, "repeat": 1},
        ),
        (
            {
                "threshold": 0.05,
                "threshold_method": "Otsu",
                "filter_method": "wavelet",
                "sigma": 2.5,
                "levels": 2,
            },
            {"method": "wavelet", "sigma": 2.5, "levels": 2},
        ),
        (
            {
                "threshold": 0.02,
                "threshold_upper": 0.95,
                "threshold_method": "Manual",
                "filter_method": "wavelet",
                "sigma": 1.5,
                "levels": 4,
            },
            {"method": "wavelet", "sigma": 1.5, "levels": 4},
        ),
    ):
        apply_filter_and_threshold(sample_layer, **filter_kwargs)
        assert_filter_kept(filter_kwargs, expected)
        button.click()
        assert sample_layer.metadata["settings"]["calibrated"] is True
        assert_filter_kept(filter_kwargs, expected)
        button.click()
        assert sample_layer.metadata["settings"]["calibrated"] is False
        assert_filter_kept(filter_kwargs, expected)


def test_calibration_inputs_filled_from_layer_metadata(
    make_viewer_model, qtbot
):
    """Files that record their acquisition parameters fill the tab in.

    Calibrated BrightEyes-MCS files carry the reference lifetime alongside
    the laser frequency, so neither has to be retyped; selecting a layer
    that records no lifetime must not keep the old one.
    """
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    calibration_widget = parent.calibration_tab.calibration_widget

    _add_layer_with_settings(
        viewer,
        parent,
        {'frequency': 80.0, 'reference_lifetime_ns': 2.7},
        'mcs calibrated',
    )
    assert calibration_widget.frequency_input.text() == "80.0"
    assert calibration_widget.lifetime_line_edit_widget.text() == "2.7"
    parent._sync_frequency_inputs_from_metadata()
    assert calibration_widget.frequency_input.text() == "80.0"
    assert calibration_widget.lifetime_line_edit_widget.text() == "2.7"

    _add_layer_with_settings(
        viewer, parent, {'frequency': 40.0}, 'plain layer'
    )
    # The stale 2.7 ns would otherwise be applied to the wrong acquisition.
    assert calibration_widget.lifetime_line_edit_widget.text() == ""
    assert calibration_widget.frequency_input.text() == "40.0"


def test_calibration_with_an_already_calibrated_reference(
    make_viewer_model, qtbot, monkeypatch
):
    """An already-calibrated reference is only used once the user agrees to
    use its original data; it then gets its own calibration back."""
    from qtpy.QtWidgets import QMessageBox

    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab

    sample_layer = create_image_layer_with_phasors()
    sample_layer.name = "sample_layer"
    calibration_layer = create_image_layer_with_phasors()
    calibration_layer.name = "calibration_layer"
    calibration_layer.metadata["settings"] = {
        "calibrated": True,
        "calibration_phase": [0.1],
        "calibration_modulation": [1.1],
    }
    viewer.add_layer(sample_layer)
    viewer.add_layer(calibration_layer)
    widget.calibration_widget.calibration_layer_combobox.setCurrentText(
        "calibration_layer"
    )

    # Cancelling calibrates nothing.
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Cancel
    )
    widget._on_click()
    assert not sample_layer.metadata.get("settings", {}).get(
        "calibrated", False
    )

    widget.calibration_widget.frequency_input.setText("80")
    widget.calibration_widget.lifetime_line_edit_widget.setText("2.5")
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes
    )
    widget._on_click()
    assert sample_layer.metadata["settings"]["calibrated"] is True
    # The reference layer was only uncalibrated to be used as reference; it
    # gets its own calibration back afterwards.
    settings = calibration_layer.metadata["settings"]
    assert settings["calibrated"] is True
    assert settings["calibration_phase"] == [0.1]
    assert settings["calibration_modulation"] == [1.1]


def test_apply_phasor_transformation_lists_and_scalars(
    make_viewer_model, qtbot
):
    viewer = make_viewer_model()
    parent = PlotterWidget(viewer)
    widget = parent.calibration_tab

    sample_layer = create_image_layer_with_phasors()
    # Mock it to be 1D
    sample_layer.metadata["G_original"] = np.array([0.5])
    sample_layer.metadata["S_original"] = np.array([0.5])
    sample_layer.metadata["G"] = np.array([0.5])
    sample_layer.metadata["S"] = np.array([0.5])
    sample_layer.metadata["harmonics"] = [1]

    sample_layer.name = "sample_layer"
    viewer.add_layer(sample_layer)

    # Test lists
    widget._apply_phasor_transformation("sample_layer", [0.1], [1.1])

    assert sample_layer.metadata["G_original"] is not None


def _add_layer_with_settings(viewer, parent, settings, name):
    """Add a phasor layer carrying *settings* and select it in *parent*."""
    layer = create_image_layer_with_phasors()
    layer.name = name
    layer.metadata.setdefault('settings', {}).update(settings)
    viewer.add_layer(layer)
    parent.image_layer_with_phasor_features_combobox.setCurrentText(layer.name)
    return layer
