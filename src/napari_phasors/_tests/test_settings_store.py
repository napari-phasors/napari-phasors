"""Tests for per-layer analysis settings: drafts, commits and notes."""

import copy

import numpy as np
from phasorpy.lifetime import phasor_to_apparent_lifetime

from napari_phasors._settings_store import (
    LayerSettingsStore,
    merge_keyed_path,
    replace_keyed_entries,
    settings_equal,
)
from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.plotter import PlotterWidget


class _Layer:
    """Minimal stand-in for a napari layer (weak-referenceable)."""

    def __init__(self, settings=None):
        self.metadata = {} if settings is None else {"settings": settings}


# ---------------------------------------------------------------------------
# Store unit tests
# ---------------------------------------------------------------------------


def test_settings_equal_tolerates_round_trip_differences():
    """Values read back from an OME-TIFF compare equal to the originals."""
    assert settings_equal({1: (0.1, 0.2)}, {"1": [0.1, 0.2]})
    assert settings_equal(np.array([1.0, np.nan]), [1.0, float("nan")])
    assert settings_equal(80, 80.0)
    assert not settings_equal({"a": 1}, {"a": 1, "b": 2})
    assert not settings_equal(True, 1.5)


def test_replace_keyed_entries_only_replaces_run_keys():
    """Entries of keys the run did not use are kept."""
    old = {"1": {"real": 0.1}, "2": {"real": 0.2}}
    new = {1: {"real": 0.9}, 3: {"real": 0.3}}
    assert replace_keyed_entries(old, new, [1]) == {
        "2": {"real": 0.2},
        1: {"real": 0.9},
    }


def test_merge_keyed_path_replaces_the_rest_of_the_block():
    """Only the dict at the path is merged; other values are replaced."""
    merge = merge_keyed_path(("per_harmonic",), [2])
    old = {"value": 1, "per_harmonic": {"1": "a", "2": "b"}}
    new = {"value": 5, "per_harmonic": {"1": "x", "2": "y"}}
    assert merge(old, new) == {
        "value": 5,
        "per_harmonic": {"1": "a", "2": "y"},
    }


def test_drafts_are_dropped_when_they_match_the_metadata():
    """Restoring a widget to the stored value leaves no draft behind."""
    layer = _Layer({"frequency": 80})
    store = LayerSettingsStore()

    store.set_draft(layer, "frequency", 40.0)
    assert store.get(layer, "frequency") == 40.0
    assert layer.metadata["settings"]["frequency"] == 80

    store.set_draft(layer, "frequency", 80.0)
    assert not store.has_draft(layer)


def test_commit_copies_values_and_discards_drafts():
    """Each layer gets its own copy; the committed keys' drafts go away."""
    a, b = _Layer(), _Layer({"filter": {"method": "median"}})
    notified = []
    store = LayerSettingsStore(on_change=lambda: notified.append(True))
    store.set_draft(a, "fret", {"donor_lifetime": 1.0})
    store.set_draft(a, "frequency", 80.0)

    store.commit([a, b], {"fret": {"donor_lifetime": 2.0}})

    assert a.metadata["settings"]["fret"] == {"donor_lifetime": 2.0}
    assert b.metadata["settings"]["fret"] == {"donor_lifetime": 2.0}
    assert a.metadata["settings"]["fret"] is not b.metadata["settings"]["fret"]
    # Other analyses and unrelated drafts are left alone.
    assert b.metadata["settings"]["filter"] == {"method": "median"}
    assert store.drafts(a) == {"frequency": 80.0}
    assert notified


def test_commit_applies_merge_rules_to_existing_values():
    """A merge rule keeps the parts of a stored value the run did not use."""
    layer = _Layer({"fret": {"bg": {"2": "old"}, "donor": 1}})
    store = LayerSettingsStore()
    store.commit(
        [layer],
        {"fret": {"bg": {1: "new"}, "donor": 3}},
        merge={"fret": merge_keyed_path(("bg",), [1])},
    )
    assert layer.metadata["settings"]["fret"] == {
        "bg": {"2": "old", 1: "new"},
        "donor": 3,
    }


def test_update_committed_patches_existing_drafts():
    """A change after the run is not undone by an older unsaved edit."""
    layer = _Layer({"fret": {"colormap": "viridis", "donor": 1}})
    store = LayerSettingsStore()
    store.set_draft_path(layer, "fret", ("donor",), 2)

    store.update_committed([layer], "fret", ("colormap",), "magma")

    assert layer.metadata["settings"]["fret"]["colormap"] == "magma"
    assert store.get(layer, "fret") == {"colormap": "magma", "donor": 2}


def test_overwritten_layers():
    """Only layers storing different values count as overwritten."""
    same = _Layer({"fret": {"donor": 2}})
    different = _Layer({"fret": {"donor": 5}})
    empty = _Layer()
    store = LayerSettingsStore()

    result = store.overwritten_layers(
        [same, different, empty], ["fret"], {"fret": {"donor": 2}}
    )
    assert result == [different]
    # Unknown values (nothing to compare with) count as overwritten.
    assert store.overwritten_layers([same, empty], ["fret"], {}) == [same]


# ---------------------------------------------------------------------------
# Plotter integration
# ---------------------------------------------------------------------------


def _select(plotter, names, primary):
    """Check *names* with *primary* as primary and process the change."""
    combo = plotter.image_layers_checkable_combobox
    combo.setCheckedItems(names)
    combo.setPrimaryLayer(primary)
    plotter._layer_selection_timer.stop()
    plotter._process_layer_selection_change()


def _plotter_with_layers(make_viewer_model, names, primary=None):
    """Return a plotter with one phasor layer per name, all selected."""
    viewer = make_viewer_model()
    layers = []
    for name in names:
        layer = create_image_layer_with_phasors()
        layer.name = name
        viewer.add_layer(layer)
        layers.append(layer)
    plotter = PlotterWidget(viewer)
    _select(plotter, list(names), primary or names[0])
    return plotter, layers


def test_fret_run_stores_its_settings_in_every_selected_layer(
    make_viewer_model, qtbot
):
    """A run replaces its own settings in all layers, nothing else."""
    plotter, (a, b) = _plotter_with_layers(make_viewer_model, ["A", "B"])
    b.metadata["settings"]["filter"] = {
        "method": "median",
        "size": 3,
        "repeat": 1,
    }
    b.metadata["settings"]["fret"] = {
        "donor_lifetime": 4.0,
        "background_positions_by_harmonic": {"2": {"real": 0.3, "imag": 0.2}},
    }

    fret = plotter.fret_tab
    fret.frequency_input.setText("80")
    fret.donor_line_edit.setText("2.5")
    fret.calculate_fret_efficiency()

    for layer in (a, b):
        settings = layer.metadata["settings"]
        assert settings["fret"]["donor_lifetime"] == 2.5
        assert settings["frequency"] == 80.0
    assert a.metadata["settings"]["fret"] is not b.metadata["settings"]["fret"]
    # Another analysis stored in B is untouched.
    assert b.metadata["settings"]["filter"] == {
        "method": "median",
        "size": 3,
        "repeat": 1,
    }
    # The run used harmonic 1; B keeps its background for harmonic 2.
    positions = b.metadata["settings"]["fret"][
        "background_positions_by_harmonic"
    ]
    assert positions["2"] == {"real": 0.3, "imag": 0.2}


def test_unsaved_edits_follow_the_primary_layer(make_viewer_model, qtbot):
    """Edits stay out of the metadata and come back with their layer."""
    plotter, (a, b) = _plotter_with_layers(make_viewer_model, ["A", "B"])
    b.metadata["settings"]["fret"] = {"donor_lifetime": 1.0}
    fret = plotter.fret_tab
    plotter.tab_widget.setCurrentWidget(fret)

    fret.donor_line_edit.setText("3.3")
    fret._on_parameters_changed()
    assert "fret" not in a.metadata["settings"]
    assert plotter.layer_settings(a)["fret"]["donor_lifetime"] == 3.3

    plotter.image_layers_checkable_combobox.setPrimaryLayer("B")
    assert fret.donor_line_edit.text() == "1.0"

    plotter.image_layers_checkable_combobox.setPrimaryLayer("A")
    assert fret.donor_line_edit.text() == "3.3"
    assert "fret" not in a.metadata["settings"]


def test_changing_selection_or_primary_writes_no_settings(
    make_viewer_model, qtbot
):
    """Only running an analysis writes settings, never selecting layers."""
    plotter, layers = _plotter_with_layers(make_viewer_model, ["A", "B", "C"])
    layers[1].metadata["settings"]["frequency"] = 40.0
    before = {
        layer.name: copy.deepcopy(layer.metadata["settings"])
        for layer in layers
    }

    _select(plotter, ["A", "B", "C"], "B")
    _select(plotter, ["B", "C"], "C")
    _select(plotter, ["A", "B", "C"], "A")

    for layer in layers:
        assert settings_equal(layer.metadata["settings"], before[layer.name])


def test_plot_setting_change_is_stored_in_all_selected_layers(
    make_viewer_model, qtbot
):
    """The plot redraws every selected layer, so all of them store it."""
    plotter, (a, b) = _plotter_with_layers(make_viewer_model, ["A", "B"])
    plotter.plotter_inputs_widget.number_of_bins_spinbox.setValue(77)
    assert a.metadata["settings"]["number_of_bins"] == 77
    assert b.metadata["settings"]["number_of_bins"] == 77


def test_mapping_uses_each_layers_own_frequency(make_viewer_model, qtbot):
    """Stored frequencies are kept and used; missing ones are filled."""
    plotter, (a, b, c) = _plotter_with_layers(
        make_viewer_model, ["A", "B", "C"]
    )
    b.metadata["settings"]["frequency"] = 40.0
    mapping = plotter.phasor_mapping_tab
    plotter.tab_widget.setCurrentWidget(mapping)
    mapping.output_mode_combobox.setCurrentText("Lifetime")
    mapping.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
    mapping.frequency_input.setText("80")

    mapping._refresh_settings_note()
    note = mapping._settings_note.text()
    assert "B (40 MHz)" in note
    assert "No frequency is stored in C" in note

    mapping._on_calculate_lifetime_clicked()

    assert a.metadata["settings"]["frequency"] == 80.0
    assert b.metadata["settings"]["frequency"] == 40.0
    assert c.metadata["settings"]["frequency"] == 80.0

    real = b.metadata["G"][0]
    imag = b.metadata["S"][0]
    with np.errstate(divide="ignore", invalid="ignore"):
        expected, _ = phasor_to_apparent_lifetime(real, imag, frequency=40.0)
    expected = np.clip(expected, 0, None)
    # The output is clipped to the (rounded) display range, hence rtol; at
    # 80 MHz every lifetime would be half of these.
    np.testing.assert_allclose(
        mapping.per_layer_metric_data["B"], expected, rtol=1e-3, equal_nan=True
    )
    for layer in (a, b, c):
        mapping_settings = layer.metadata["settings"]["phasor_mapping"]
        assert mapping_settings["output_type"] == "Apparent Phase Lifetime"


def test_overwrite_note_names_layers_with_other_settings(
    make_viewer_model, qtbot
):
    """The note lists layers whose stored settings a run would replace."""
    plotter, (_, b) = _plotter_with_layers(make_viewer_model, ["A", "B"])
    b.metadata["settings"]["fret"] = {"donor_lifetime": 4.0}
    b.metadata["settings"]["frequency"] = 80.0
    fret = plotter.fret_tab
    plotter.tab_widget.setCurrentWidget(fret)
    fret.frequency_input.setText("80")
    fret.donor_line_edit.setText("2.5")

    fret._refresh_settings_note()
    assert "FRET parameters stored in: B" in fret._settings_note.text()
    assert not fret._settings_note.isHidden()

    fret.calculate_fret_efficiency()
    fret._refresh_settings_note()
    assert fret._settings_note.isHidden()


def test_filter_apply_without_filter_drops_the_stored_one(
    make_viewer_model, qtbot
):
    """A layer filtered with 'None' no longer claims a median filter."""
    plotter, (a,) = _plotter_with_layers(make_viewer_model, ["A"])
    a.metadata["settings"]["filter"] = {
        "method": "median",
        "size": 3,
        "repeat": 1,
    }
    plotter.filter_tab.filter_method_combobox.setCurrentText("None")
    plotter.filter_tab.apply_button_clicked()
    assert "filter" not in a.metadata["settings"]


def test_calibration_stores_its_reference(make_viewer_model, qtbot):
    """The reference used is stored, so the calibration can be repeated."""
    plotter, (sample, _) = _plotter_with_layers(
        make_viewer_model, ["sample", "reference"]
    )
    _select(plotter, ["sample"], "sample")
    widget = plotter.calibration_tab.calibration_widget
    widget.calibration_layer_combobox.setCurrentText("reference")
    widget.frequency_input.setText("80")
    widget.lifetime_line_edit_widget.setText("2.5")

    plotter.calibration_tab._on_click()

    settings = sample.metadata["settings"]
    assert settings["calibrated"] is True
    assert settings["calibration_reference"] == {
        "reference_layer": "reference",
        "reference_lifetime": 2.5,
        "frequency": 80.0,
    }
