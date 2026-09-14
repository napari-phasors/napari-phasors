"""Tests for the Autoupdate toggles of the analysis tabs.

The Components, Phasor Mapping and FRET tabs each own an "Autoupdate" switch.
While it is on, the tab re-runs its analysis whenever something the result
depends on changes: its own inputs, and the external events the plotter
forwards (the filter or calibration tab rewriting the phasor data, a new
harmonic, a different layer selection).
"""

from unittest.mock import patch

import pytest

from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.plotter import PlotterWidget

AUTOUPDATE_TABS = ("components_tab", "phasor_mapping_tab", "fret_tab")


def _plotter_with_layer(viewer, harmonic=None):
    """Return a plotter showing one freshly added phasor layer."""
    plotter = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors(harmonic=harmonic)
    viewer.add_layer(layer)
    plotter.image_layer_with_phasor_features_combobox.setCurrentText(
        layer.name
    )
    plotter.on_image_layer_changed()
    return plotter, layer


def _prepare_components(plotter):
    """Give the components tab two valid components."""
    tab = plotter.components_tab
    tab.components[0].g_edit.setText("0.2")
    tab.components[0].s_edit.setText("0.3")
    tab.components[1].g_edit.setText("0.6")
    tab.components[1].s_edit.setText("0.4")
    return tab


def _prepare_mapping(plotter):
    """Give the mapping tab a valid frequency and output type."""
    tab = plotter.phasor_mapping_tab
    tab.output_mode_combobox.setCurrentText("Phase")
    tab.frequency_input.setText("80")
    return tab


def _prepare_fret(plotter):
    """Give the FRET tab a valid donor lifetime and frequency."""
    tab = plotter.fret_tab
    tab.donor_line_edit.setText("4.2")
    tab.frequency_input.setText("80")
    return tab


PREPARE = {
    "components_tab": _prepare_components,
    "phasor_mapping_tab": _prepare_mapping,
    "fret_tab": _prepare_fret,
}


@pytest.mark.parametrize("tab_name", AUTOUPDATE_TABS)
def test_tab_exposes_an_autoupdate_toggle(make_viewer_model, tab_name):
    """Each analysis tab owns an Autoupdate switch, off by default."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    tab = getattr(plotter, tab_name)

    assert tab.autoupdate_check.text() == "Autoupdate"
    assert not tab.autoupdate_check.isChecked()
    assert not tab.autoupdate_enabled()
    assert tab.autoupdate_check.toolTip()
    # The toggle sits in the tab's own layout, under the primary button.
    assert tab.autoupdate_container.parent() is not None


@pytest.mark.parametrize("tab_name", AUTOUPDATE_TABS)
def test_toggle_disables_the_primary_button(make_viewer_model, tab_name):
    """The manual Run button steps aside while autoupdate is on."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = PREPARE[tab_name](plotter)
    button = tab._autoupdate_run_button

    assert button.isEnabled()
    tab.autoupdate_check.setChecked(True)
    assert not button.isEnabled()
    tab.autoupdate_check.setChecked(False)
    assert button.isEnabled()


@pytest.mark.parametrize("tab_name", AUTOUPDATE_TABS)
def test_filter_change_triggers_an_autoupdate(make_viewer_model, tab_name):
    """Applying a threshold in the Filter tab re-runs the analysis."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = PREPARE[tab_name](plotter)
    tab.autoupdate_check.setChecked(True)

    with patch.object(tab, "_autoupdate_action") as action:
        plotter.filter_tab.threshold_slider.setValue((5, 90))
        plotter.filter_tab.apply_button_clicked()

    action.assert_called()


@pytest.mark.parametrize("tab_name", AUTOUPDATE_TABS)
def test_no_autoupdate_while_the_toggle_is_off(make_viewer_model, tab_name):
    """The same filter change is ignored while autoupdate is off."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = PREPARE[tab_name](plotter)

    with patch.object(tab, "_autoupdate_action") as action:
        plotter.filter_tab.threshold_slider.setValue((5, 90))
        plotter.filter_tab.apply_button_clicked()

    action.assert_not_called()


@pytest.mark.parametrize("tab_name", AUTOUPDATE_TABS)
def test_calibration_triggers_an_autoupdate(make_viewer_model, tab_name):
    """A calibration rewrites G/S, so the analysis is recomputed."""
    viewer = make_viewer_model()
    plotter, sample = _plotter_with_layer(viewer)
    calibration = create_image_layer_with_phasors()
    calibration.name = "calibration"
    viewer.add_layer(calibration)

    tab = PREPARE[tab_name](plotter)
    tab.autoupdate_check.setChecked(True)

    calibration_tab = plotter.calibration_tab
    calibration_tab.calibration_widget.frequency_input.setText("80")
    calibration_tab.calibration_widget.lifetime_line_edit_widget.setText("2")
    calibration_tab.calibration_widget.calibration_layer_combobox.setCurrentText(
        calibration.name
    )

    with patch.object(tab, "_autoupdate_action") as action:
        calibration_tab._on_click()

    assert "G_original" in sample.metadata
    action.assert_called()


@pytest.mark.parametrize("tab_name", AUTOUPDATE_TABS)
def test_harmonic_change_requests_an_autoupdate(make_viewer_model, tab_name):
    """A new harmonic moves the phasor coordinates the analysis uses.

    Whether the tab then *runs* depends on its own inputs still being valid
    for that harmonic (the components tab, for instance, keeps a separate set
    of components per harmonic), so what is asserted here is the request.
    """
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = PREPARE[tab_name](plotter)
    tab.autoupdate_check.setChecked(True)

    with patch.object(tab, "request_autoupdate") as request:
        plotter.harmonic_spinbox.setValue(2)

    request.assert_called()


@pytest.mark.parametrize("tab_name", AUTOUPDATE_TABS)
def test_no_harmonic_autoupdate_while_restoring_settings(
    make_viewer_model, tab_name
):
    """Restoring a layer's stored harmonic is not a user-driven change."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = PREPARE[tab_name](plotter)
    tab.autoupdate_check.setChecked(True)

    plotter._updating_settings = True
    try:
        with patch.object(tab, "request_autoupdate") as request:
            plotter.harmonic_spinbox.setValue(2)
    finally:
        plotter._updating_settings = False

    request.assert_not_called()


def test_request_analysis_autoupdates_reports_the_tabs_that_ran(
    make_viewer_model,
):
    """The plotter helper returns only the tabs that actually recomputed."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    components = _prepare_components(plotter)
    mapping = _prepare_mapping(plotter)
    _prepare_fret(plotter)

    assert plotter.request_analysis_autoupdates() == []

    components.autoupdate_check.setChecked(True)
    mapping.autoupdate_check.setChecked(True)

    with (
        patch.object(components, "_autoupdate_action"),
        patch.object(mapping, "_autoupdate_action"),
    ):
        updated = plotter.request_analysis_autoupdates()
    assert updated == [mapping, components]

    with patch.object(components, "_autoupdate_action"):
        updated = plotter.request_analysis_autoupdates(tabs=[components])
    assert updated == [components]


def test_layer_selection_change_updates_only_the_visible_tab(
    make_viewer_model,
):
    """A hidden tab is torn down on layer change, so it must not re-run."""
    viewer = make_viewer_model()
    plotter, first = _plotter_with_layer(viewer)
    second = create_image_layer_with_phasors()
    second.name = "second"
    viewer.add_layer(second)

    mapping = _prepare_mapping(plotter)
    fret = _prepare_fret(plotter)
    mapping.autoupdate_check.setChecked(True)
    fret.autoupdate_check.setChecked(True)

    plotter.tab_widget.setCurrentWidget(mapping)

    with patch.object(mapping, "request_autoupdate") as mapping_request:
        with patch.object(fret, "request_autoupdate") as fret_request:
            plotter.image_layer_with_phasor_features_combobox.setCurrentText(
                second.name
            )
            plotter.on_image_layer_changed()

    mapping_request.assert_called()
    fret_request.assert_not_called()


def test_deferred_tab_updates_when_it_is_brought_forward(make_viewer_model):
    """A tab marked stale catches up the moment the user selects it."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    mapping = _prepare_mapping(plotter)
    fret = _prepare_fret(plotter)

    plotter.tab_widget.setCurrentWidget(mapping)
    fret.autoupdate_check.setChecked(True)
    fret._needs_update = True

    with patch.object(fret, "request_autoupdate") as request:
        plotter.tab_widget.setCurrentWidget(fret)

    request.assert_called()
    assert not fret._needs_update


def test_components_autoupdate_follows_the_component_inputs(
    make_viewer_model,
):
    """Editing, adding, removing or dragging components re-runs the fit."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = _prepare_components(plotter)
    tab.autoupdate_check.setChecked(True)

    with patch.object(tab, "_autoupdate_action") as action:
        tab.components[1].g_edit.setText("0.55")
        tab.components[1].g_edit.editingFinished.emit()
    action.assert_called()

    # A freshly added component has no coordinates yet, so the analysis is
    # incomplete and must not run until it is filled in.
    with patch.object(tab, "_autoupdate_action") as action:
        tab._add_component()
    action.assert_not_called()

    with patch.object(tab, "_autoupdate_action") as action:
        tab.components[2].g_edit.setText("0.4")
        tab.components[2].s_edit.setText("0.45")
        tab.components[2].s_edit.editingFinished.emit()
    action.assert_called()

    with patch.object(tab, "_autoupdate_action") as action:
        tab._remove_component()
    action.assert_called()

    # A drag only recomputes on release, not on every mouse-move.
    with patch.object(tab, "_autoupdate_action") as action:
        tab.dragging_component_idx = None
        tab._on_release(None)
    action.assert_not_called()

    with patch.object(tab, "_autoupdate_action") as action:
        tab.dragging_component_idx = 0
        tab._on_release(None)
    action.assert_called()


def test_components_autoupdate_follows_the_analysis_type(make_viewer_model):
    """Switching between projection and fit recomputes the result."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = _prepare_components(plotter)
    tab.autoupdate_check.setChecked(True)

    with patch.object(tab, "_autoupdate_action") as action:
        tab._on_analysis_type_changed("Linear Projection")
    action.assert_called()


def test_mapping_autoupdate_follows_its_own_inputs(make_viewer_model):
    """The output type and the committed frequency drive the recalculation."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = _prepare_mapping(plotter)
    tab.autoupdate_check.setChecked(True)

    with patch.object(tab, "_autoupdate_action") as action:
        tab.output_mode_combobox.setCurrentText("Modulation")
    action.assert_called()

    with patch.object(tab, "_autoupdate_action") as action:
        tab.output_mode_combobox.setCurrentText("Lifetime")
        tab.lifetime_type_combobox.setCurrentText("Normal Lifetime")
    action.assert_called()

    # Typing is not committing: only editingFinished triggers a run, so "8"
    # on the way to "80" does not recalculate the whole image.
    with patch.object(tab, "_autoupdate_action") as action:
        tab.frequency_input.setText("4")
    action.assert_not_called()

    with patch.object(tab, "_autoupdate_action") as action:
        tab.frequency_input.editingFinished.emit()
    action.assert_called()


def test_mapping_autoupdate_absorbs_the_debounced_refresh(make_viewer_model):
    """The tab's own refresh timer is dropped when autoupdate just ran."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = _prepare_mapping(plotter)
    tab.autoupdate_check.setChecked(True)

    tab._output_refresh_timer.start()
    tab._autoupdate_calculate_output()
    assert not tab._output_refresh_timer.isActive()


def test_mapping_autoupdate_does_not_pop_warnings(make_viewer_model):
    """An automatic run is silent; only a click may raise a warning."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = _prepare_mapping(plotter)

    with patch.object(tab, "_calculate_and_display_output") as calculate:
        calculate.return_value = True
        tab._autoupdate_calculate_output()
    calculate.assert_called_once_with(show_warnings=False)
    assert tab._has_calculated_output

    with patch.object(tab, "_calculate_and_display_output") as calculate:
        calculate.return_value = True
        tab._on_calculate_lifetime_clicked()
    calculate.assert_called_once_with(show_warnings=True)


def test_fret_autoupdate_follows_its_own_inputs(make_viewer_model):
    """Committed donor/frequency values and released sliders trigger a run."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = _prepare_fret(plotter)
    tab.autoupdate_check.setChecked(True)

    with patch.object(tab, "_autoupdate_action") as action:
        tab.donor_line_edit.setText("3.1")
    action.assert_not_called()

    with patch.object(tab, "_autoupdate_action") as action:
        tab.donor_line_edit.editingFinished.emit()
    action.assert_called()

    with patch.object(tab, "_autoupdate_action") as action:
        tab.frequency_input.editingFinished.emit()
    action.assert_called()

    with patch.object(tab, "_autoupdate_action") as action:
        tab.background_real_edit.editingFinished.emit()
        tab.background_imag_edit.editingFinished.emit()
    assert action.call_count == 2

    with patch.object(tab, "_autoupdate_action") as action:
        tab.fretting_slider.setValue(50)
        tab.fretting_slider.sliderReleased.emit()
    action.assert_called_once()

    with patch.object(tab, "_autoupdate_action") as action:
        tab.background_slider.setValue(30)
        tab.background_slider.sliderReleased.emit()
    action.assert_called_once()


def test_fret_autoupdate_follows_the_layer_derived_sources(make_viewer_model):
    """Deriving donor/background from layers fills fields programmatically."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = _prepare_fret(plotter)
    tab.autoupdate_check.setChecked(True)

    with patch.object(tab, "_autoupdate_action") as action:
        tab.donor_lifetime_combobox.selectionChanged.emit()
    action.assert_called()

    with patch.object(tab, "_autoupdate_action") as action:
        tab.background_image_combobox.selectionChanged.emit()
    action.assert_called()


@pytest.mark.parametrize("tab_name", AUTOUPDATE_TABS)
def test_incomplete_inputs_block_the_autoupdate(make_viewer_model, tab_name):
    """A tab with missing required inputs never runs on its own."""
    viewer = make_viewer_model()
    plotter, _ = _plotter_with_layer(viewer)
    tab = getattr(plotter, tab_name)

    tab.autoupdate_check.setChecked(True)
    assert tab._autoupdate_validator() is not None

    with patch.object(tab, "_autoupdate_action") as action:
        plotter.filter_tab.apply_button_clicked()
    action.assert_not_called()
