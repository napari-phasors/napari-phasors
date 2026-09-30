"""Tests for per-layer analysis settings: drafts, commits and notes."""

import copy

import numpy as np
from phasorpy.lifetime import phasor_to_apparent_lifetime

from napari_phasors._settings_store import (
    ANALYSIS_SETTINGS_KEYS,
    LayerSettingsStore,
    format_layer_list,
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


def test_stored_values_are_read_live():
    """The store keeps no copy of the metadata: direct edits are seen at
    once, and only a draft of the same key wins until it is discarded."""
    layer = _Layer({"frequency": 80})
    store = LayerSettingsStore()

    layer.metadata["settings"]["frequency"] = 40.0
    assert store.get(layer, "frequency") == 40.0
    layer.metadata["settings"] = {"frequency": 20.0}
    assert store.effective(layer) == {"frequency": 20.0}

    store.set_draft(layer, "frequency", 60.0)
    layer.metadata["settings"]["frequency"] = 10.0
    assert store.get(layer, "frequency") == 60.0
    store.discard_drafts([layer])
    assert store.get(layer, "frequency") == 10.0


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


def test_settings_across_two_selected_layers(make_viewer_model, qtbot):
    """Plot settings reach every selected layer; analysis edits stay drafts
    of the primary until a run, which stores them in all layers (keeping
    each layer's own frequency and unrelated settings), and the notes name
    the layers whose stored settings a run would replace."""
    plotter, (a, b) = _plotter_with_layers(make_viewer_model, ["A", "B"])
    fret = plotter.fret_tab

    # The plot redraws every selected layer, so all of them store it.
    plotter.plotter_inputs_widget.number_of_bins_spinbox.setValue(77)
    assert a.metadata["settings"]["number_of_bins"] == 77
    assert b.metadata["settings"]["number_of_bins"] == 77

    # The selection note names the layers whose cursors a run replaces.
    b.metadata["settings"]["selections"] = {
        "circular_cursors": [{"center": [0.5, 0.2], "radius": 0.1}]
    }
    cursor = plotter.selection_tab.cursor_selection_widget
    cursor._refresh_settings_note()
    assert "cursors stored in: B" in cursor._settings_note.text()
    cursor.parent_widget = None
    try:
        cursor._refresh_settings_note()
    finally:
        cursor.parent_widget = plotter

    # The values a run would store are the primary's, edits included.
    b.metadata["settings"]["fret"] = {"donor_lifetime": 4.0}
    plotter.stage_setting("fret", {"donor_lifetime": 2.0})
    assert plotter.pending_settings_values("fret_tab") == {
        "fret": {"donor_lifetime": 2.0}
    }
    message = plotter.settings_overwrite_message("fret_tab")
    assert "FRET parameters stored in: B" in message
    plotter.settings_store.discard_drafts([a])

    # Edits stay out of the metadata and come back with their layer.
    b.metadata["settings"]["fret"] = {"donor_lifetime": 1.0}
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

    # The note lists layers whose stored settings a run would replace, and
    # goes away once the run has stored the same values everywhere.
    b.metadata["settings"]["fret"] = {"donor_lifetime": 4.0}
    b.metadata["settings"]["frequency"] = 80.0
    fret.frequency_input.setText("80")
    fret.donor_line_edit.setText("2.5")
    fret._refresh_settings_note()
    assert "FRET parameters stored in: B" in fret._settings_note.text()
    assert not fret._settings_note.isHidden()
    fret.calculate_fret_efficiency()
    fret._refresh_settings_note()
    assert fret._settings_note.isHidden()

    # A run replaces its own settings in all layers, nothing else.
    b.metadata["settings"]["filter"] = {
        "method": "median",
        "size": 3,
        "repeat": 1,
    }
    b.metadata["settings"]["fret"] = {
        "donor_lifetime": 4.0,
        "background_positions_by_harmonic": {"2": {"real": 0.3, "imag": 0.2}},
    }
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

    # A layer acquired at another frequency keeps and uses it.
    b.metadata["settings"]["frequency"] = 40.0
    fret.calculate_fret_efficiency()
    assert a.metadata["settings"]["frequency"] == 80.0
    assert b.metadata["settings"]["frequency"] == 40.0
    fret._needs_update = True
    fret._refresh_settings_note()
    fret._needs_update = False


def test_settings_across_three_selected_layers(make_viewer_model, qtbot):
    """Selecting layers never writes settings; component settings are edited
    as drafts only for the primary; and the mapping keeps and uses each
    layer's own frequency, filling in the missing ones."""
    plotter, layers = _plotter_with_layers(make_viewer_model, ["A", "B", "C"])
    a, b, c = layers

    # Only running an analysis writes settings, never selecting layers.
    b.metadata["settings"]["frequency"] = 40.0
    before = {
        layer.name: copy.deepcopy(layer.metadata["settings"])
        for layer in layers
    }
    _select(plotter, ["A", "B", "C"], "B")
    _select(plotter, ["B", "C"], "C")
    _select(plotter, ["A", "B", "C"], "A")
    for layer in layers:
        assert settings_equal(layer.metadata["settings"], before[layer.name])

    # Stored frequencies are kept and used; missing ones are filled.
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
    for layer in layers:
        mapping_settings = layer.metadata["settings"]["phasor_mapping"]
        assert mapping_settings["output_type"] == "Apparent Phase Lifetime"

    # Only the primary layer's component settings are edited as drafts.
    tab = plotter.components_tab
    assert tab._read_component_settings(None) is None
    assert tab._edit_component_settings(None) is None
    assert tab._read_component_settings(a) is None
    b.metadata["settings"]["component_analysis"] = {"components": {}}
    assert tab._read_component_settings(b) == {"components": {}}
    # Another selected layer without settings: created in its metadata.
    assert tab._edit_component_settings(c, create=False) is None
    block = tab._edit_component_settings(c)
    assert c.metadata["settings"]["component_analysis"] is block
    assert tab._components_merge_rule([1])({}, "replaced") == "replaced"
    tab.parent_widget = None
    try:
        assert tab._analysed_component_layers() == []
    finally:
        tab.parent_widget = plotter
    tab._needs_update = True
    tab._refresh_settings_note()
    tab._needs_update = False


def test_settings_helpers_with_one_layer(make_viewer_model, qtbot):
    """With one layer: plot settings fill only missing keys, unusable
    frequencies are ignored, unsaved filter and mapping edits stay out of
    the metadata until a run, legacy lifetime settings merge, and nothing
    is pending once no layer is selected."""
    plotter, (a,) = _plotter_with_layers(make_viewer_model, ["A"])

    # A layer keeps the plot settings it has and gets the ones it lacks.
    a.metadata["settings"]["number_of_bins"] = 12
    plotter._initialize_plot_settings_in_metadata(a)
    assert a.metadata["settings"]["number_of_bins"] == 12
    assert set(ANALYSIS_SETTINGS_KEYS["settings_tab"]) <= set(
        a.metadata["settings"]
    )

    # A frequency that is not a positive number is never stored or read.
    plotter.commit_frequency([a], "not a number")
    assert "frequency" not in a.metadata["settings"]
    a.metadata["settings"]["frequency"] = "not a number"
    plotter.image_layer_with_phasor_features_combobox.setCurrentText("A")
    assert plotter._get_frequency_from_layer() is None
    del a.metadata["settings"]["frequency"]

    # Choosing 'None' reads as "no filter" without touching the metadata...
    stored = {"method": "median", "size": 3, "repeat": 1}
    a.metadata["settings"]["filter"] = dict(stored)
    plotter.filter_tab.filter_method_combobox.setCurrentText("None")
    plotter.filter_tab._stage_ui_settings()
    assert plotter.layer_settings(a)["filter"] == {}
    assert a.metadata["settings"]["filter"] == stored
    # ...and applying it drops the stored median filter.
    plotter.filter_tab.apply_button_clicked()
    assert "filter" not in a.metadata["settings"]

    # Editing the displayed mapping range is kept as an unsaved edit.
    mapping = plotter.phasor_mapping_tab
    mapping._update_lifetime_setting_in_metadata("range_min", 0.5)
    mapping._update_lifetime_setting_in_metadata("range_max", 4.5)
    staged = plotter.layer_settings(a)["phasor_mapping"]
    assert staged["lifetime_range_min"] == 0.5
    assert staged["lifetime_range_max"] == 4.5
    assert "phasor_mapping" not in (a.metadata.get("settings") or {})

    # Without the plotter the tab reads metadata and stores nothing.
    assert mapping._get_phasor_mapping_settings(None) is None
    a.metadata["settings"]["phasor_mapping"] = {"output_type": "Phase"}
    mapping.parent_widget = None
    try:
        assert mapping._get_phasor_mapping_settings(a) == {
            "output_type": "Phase"
        }
        mapping._stage_mapping_values({"output_type": "Modulation"})
        mapping._commit_mapping_settings([a])
        mapping._refresh_settings_note()
    finally:
        mapping.parent_widget = plotter
    assert a.metadata["settings"]["phasor_mapping"] == {"output_type": "Phase"}

    # A layer written before the rename keeps one set of settings.
    del a.metadata["settings"]["phasor_mapping"]
    a.metadata["settings"]["lifetime"] = {"colormap": "turbo"}
    mapping._commit_mapping_settings([a])
    stored = a.metadata["settings"]["phasor_mapping"]
    assert stored["output_type"] == mapping._get_selected_output_type()
    assert a.metadata["settings"]["lifetime"] == stored

    # Refreshing the notes while closing, or without a label, is a no-op.
    plotter._is_closing = True
    plotter._refresh_settings_notes()
    plotter._is_closing = False
    plotter._plot_settings_note = None
    plotter._refresh_plot_settings_note()

    # Nothing is pending when no layer is selected.
    plotter.image_layers_checkable_combobox.setCheckedItems([])
    plotter._layer_selection_timer.stop()
    plotter._process_layer_selection_change()
    assert plotter.get_primary_layer() is None
    assert plotter.pending_settings_values("fret_tab") == {}


def test_calibration_reference_settings(make_viewer_model, qtbot):
    """The reference used is stored, so the calibration can be repeated, and
    the tab shows the reference the primary layer was calibrated with."""
    plotter, (sample, _) = _plotter_with_layers(
        make_viewer_model, ["sample", "reference"]
    )
    _select(plotter, ["sample"], "sample")
    tab = plotter.calibration_tab
    widget = tab.calibration_widget
    widget.calibration_layer_combobox.setCurrentText("reference")
    widget.frequency_input.setText("80")
    widget.lifetime_line_edit_widget.setText("2.5")

    tab._on_click()
    settings = sample.metadata["settings"]
    assert settings["calibrated"] is True
    assert settings["calibration_reference"] == {
        "reference_layer": "reference",
        "reference_lifetime": 2.5,
        "frequency": 80.0,
    }

    # Move the widgets away (discarding those edits) so the restore shows.
    widget.calibration_layer_combobox.setCurrentText("sample")
    widget.lifetime_line_edit_widget.setText("1")
    plotter.settings_store.discard_drafts([sample])
    sample.metadata["settings"]["calibration_reference"] = {
        "reference_layer": "reference",
        "reference_lifetime": 2.5,
    }
    tab._restore_reference_from_primary()
    assert widget.calibration_layer_combobox.currentText() == "reference"
    assert widget.lifetime_line_edit_widget.text() == "2.5"

    # A layer with no stored reference leaves what is being typed alone.
    sample.metadata["settings"]["calibration_reference"] = "not a reference"
    widget.lifetime_line_edit_widget.setText("9")
    tab._restore_reference_from_primary()
    assert widget.lifetime_line_edit_widget.text() == "9"

    # Restoring the widgets does not stage what it puts in them.
    plotter.settings_store.discard_drafts([sample])
    tab._restoring_settings = True
    try:
        tab._stage_reference()
    finally:
        tab._restoring_settings = False
    assert not plotter.settings_store.has_draft(
        sample, ["calibration_reference"]
    )


# ---------------------------------------------------------------------------
# Store edge cases
# ---------------------------------------------------------------------------


def test_settings_equal_array_edge_cases():
    """Values that cannot be compared elementwise are not equal."""
    # Ragged: NumPy cannot build an array to compare against.
    assert not settings_equal(np.array([1.0]), [[1, 2], [3]])
    assert not settings_equal(np.array([1.0, 2.0]), [1.0])
    assert settings_equal(np.array([1, 2]), [1, 2])
    assert settings_equal(float("nan"), float("nan"))


def test_merge_keyed_path_without_a_stored_dict():
    """There is nothing to keep when the old settings have no dict there."""
    merge = merge_keyed_path(("a", "b"), [1])
    assert merge({"a": 1}, "replaced") == "replaced"
    assert merge({"a": 5}, {"a": {"b": {1: "new"}}}) == {
        "a": {"b": {1: "new"}}
    }


def test_format_layer_list_summarises_long_lists():
    """A note names a few layers and counts the rest."""
    assert format_layer_list(["A", "B"]) == "A, B"
    assert (
        format_layer_list([f"L{i}" for i in range(6)])
        == "L0, L1, L2, L3 and 2 more"
    )


def test_set_draft_path_creates_missing_levels():
    """A nested edit builds the dicts it needs, without touching metadata."""
    layer = _Layer({"fret": {"donor": 1}})
    store = LayerSettingsStore()

    store.set_draft_path(layer, "fret", ("colormap_settings", "gamma"), 2.0)

    assert store.get(layer, "fret")["colormap_settings"] == {"gamma": 2.0}
    assert layer.metadata["settings"]["fret"] == {"donor": 1}


def test_store_ignores_layers_it_has_nothing_for():
    """Calls without a layer, or without a draft, are no-ops."""
    store = LayerSettingsStore()
    assert store.committed(None) == {}
    assert store.drafts(None) == {}
    store.set_draft(None, "frequency", 80.0)
    store.set_draft_path(None, "fret", ("donor",), 1)

    layer = _Layer({"fret": {"donor": 1}})
    store.settle_draft(layer, "fret")
    assert not store.has_draft(layer, ["fret"])

    store.set_draft(layer, "frequency", 80.0)
    assert store.has_draft(layer, ["frequency"])
    assert not store.has_draft(layer, ["fret"])


def test_update_committed_replaces_a_non_dict_value():
    """A key stored as something else is replaced by the patched dict."""
    layer = _Layer({"fret": 5})
    store = LayerSettingsStore()

    store.update_committed([layer], "fret", ("colormap", "gamma"), 2.0)

    assert layer.metadata["settings"]["fret"] == {"colormap": {"gamma": 2.0}}


# ---------------------------------------------------------------------------
# Plotter helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tab helpers
# ---------------------------------------------------------------------------
