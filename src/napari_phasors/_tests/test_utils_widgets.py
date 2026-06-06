import csv

import numpy as np

from napari_phasors._utils import (
    HistogramDockWidget,
    HistogramWidget,
    StatisticsDockWidget,
    StatisticsTableWidget,
    build_group_styles_from_layer_metadata,
    build_groups_from_layer_metadata,
    save_groups_to_layer_metadata,
)


def test_histogram_widget_update_data_and_clear(qtbot):
    """HistogramWidget should compute bins and reset cleanly."""
    widget = HistogramWidget(bins=8)
    qtbot.addWidget(widget)

    # Include invalid values to verify filtering behavior.
    data = np.array([1.0, 2.0, 3.0, np.nan, np.inf, -1.0, 0.0])
    widget.update_data(data)

    assert widget.counts is not None
    assert widget.bin_edges is not None
    assert widget.bin_centers is not None
    assert len(widget.counts) == 8
    assert 0.0 in widget._raw_valid_data
    assert -1.0 in widget._raw_valid_data
    assert np.all(np.isfinite(widget._raw_valid_data))
    assert widget._settings_button.isEnabled()
    assert widget.save_png_button.isEnabled()
    assert widget.save_csv_button.isEnabled()

    widget.clear()

    assert widget.counts is None
    assert widget.bin_edges is None
    assert widget.bin_centers is None
    assert widget._datasets == {}
    assert widget._raw_valid_data is None
    assert not widget._settings_button.isEnabled()
    assert not widget.save_png_button.isEnabled()
    assert not widget.save_csv_button.isEnabled()


def test_histogram_widget_save_csv(qtbot, tmp_path, monkeypatch):
    """HistogramWidget should export CSV accurately in all modes."""
    from qtpy.QtWidgets import QFileDialog

    widget = HistogramWidget(bins=2)
    qtbot.addWidget(widget)

    # Put some data
    data1 = np.array([1.0, 1.0, 2.0])
    data2 = np.array([1.0, 2.0, 2.0])

    csv_file = tmp_path / "test.csv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(csv_file), ""),
    )

    # 1. Test Merged (default) mode with multiple datasets
    widget.update_multi_data({"Layer 1": data1, "Layer 2": data2})
    widget._display_mode = "Merged"
    widget._save_histogram_csv()

    assert csv_file.exists()

    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[0] == ['Bin Center', 'Mean Counts', 'Std Counts']
    assert len(rows) == 3  # Header + 2 bins

    # 2. Test Grouped mode
    widget._display_mode = "Grouped"
    widget._group_assignments = {"Layer 1": 1, "Layer 2": 1}
    widget._group_names = {1: "MyGroup"}
    widget._save_histogram_csv()

    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[0] == ['Bin Center', 'MyGroup Mean', 'MyGroup Std']
    assert len(rows) == 3

    # 3. Test Individual mode
    widget._display_mode = "Individual layers"
    widget._save_histogram_csv()

    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[0] == ['Bin Center', 'Layer 1', 'Layer 2']
    assert len(rows) == 3


def test_histogram_widget_default_filter_keeps_zero(qtbot):
    """Default filtering should keep zero values (drop only NaN/Inf)."""
    widget = HistogramWidget(bins=5)
    qtbot.addWidget(widget)
    widget.update_data(np.array([0.0, 0.25, np.nan, np.inf]))

    assert widget.counts is not None
    assert np.any(np.isclose(widget._raw_valid_data, 0.0))


def test_histogram_widget_exclude_nonpositive_option(qtbot):
    """Optional non-positive filtering should preserve old behavior."""
    widget = HistogramWidget(bins=5, exclude_nonpositive=True)
    qtbot.addWidget(widget)
    widget.update_data(np.array([-1.0, 0.0, 0.25, np.nan, np.inf]))

    assert widget.counts is not None
    assert np.all(widget._raw_valid_data > 0)
    assert np.allclose(widget._raw_valid_data, np.array([0.25]))


def test_histogram_widget_update_multi_data_autosd(qtbot):
    """Multi-dataset updates should auto-enable SD shading when needed."""
    widget = HistogramWidget(bins=10)
    qtbot.addWidget(widget)

    datasets = {
        "Layer A": np.array([1, 2, 3, 4], dtype=float),
        "Layer B": np.array([2, 3, 4, 5], dtype=float),
    }
    widget.update_multi_data(datasets)

    assert set(widget._datasets.keys()) == {"Layer A", "Layer B"}
    assert set(widget._counts_per_dataset.keys()) == {"Layer A", "Layer B"}
    assert widget._show_sd is True


def test_statistics_table_widget_populates_rows(qtbot):
    """StatisticsTableWidget should populate rows for each dataset."""
    table = StatisticsTableWidget()
    qtbot.addWidget(table)

    datasets = {
        "Layer A": np.array([1.0, 2.0, 3.0]),
        "Layer B": np.array([2.0, 4.0, 6.0]),
    }
    bin_edges = np.linspace(1.0, 6.0, 6)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    table.update_statistics(
        datasets, bin_centers=bin_centers, bin_edges=bin_edges
    )

    assert table.rowCount() == 2
    assert table.columnCount() == len(StatisticsTableWidget.COLUMNS)
    row_names = {table.item(row, 0).text() for row in range(table.rowCount())}
    assert row_names == {"Layer A", "Layer B"}


def test_statistics_table_widget_handles_empty_histogram_bins(qtbot):
    """StatisticsTableWidget should not crash when histogram counts are zero."""
    table = StatisticsTableWidget()
    qtbot.addWidget(table)

    datasets = {"Layer A": np.array([10.0, 11.0, 12.0])}
    bin_edges = np.array([0.0, 1.0, 2.0])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    table.update_statistics(
        datasets, bin_centers=bin_centers, bin_edges=bin_edges
    )

    assert table.rowCount() == 1
    assert table.item(0, 1).text() == "nan"


def test_statistics_dock_widget_updates_for_single_and_grouped_data(qtbot):
    """StatisticsDockWidget should toggle sections by histogram mode/data."""
    histogram_widget = HistogramWidget(bins=12)
    qtbot.addWidget(histogram_widget)
    stats_dock = StatisticsDockWidget(histogram_widget)
    qtbot.addWidget(stats_dock)

    histogram_widget.update_data(np.array([1.0, 2.0, 3.0, 4.0]))
    assert not stats_dock.layer_stats_section.isHidden()
    assert stats_dock.group_stats_section.isHidden()
    assert stats_dock.export_csv_button.isEnabled()
    assert stats_dock.layer_stats_table.rowCount() == 1

    histogram_widget.update_multi_data(
        {
            "Layer A": np.array([1.0, 2.0, 3.0]),
            "Layer B": np.array([2.0, 3.0, 4.0]),
        }
    )
    histogram_widget.display_mode = "Grouped"
    histogram_widget._group_assignments = {"Layer A": 1, "Layer B": 2}
    histogram_widget._group_names = {1: "Group 1", 2: "Group 2"}
    stats_dock._update_statistics()

    assert not stats_dock.group_stats_section.isHidden()
    assert stats_dock.group_stats_table.rowCount() == 2


def test_histogram_dock_widget_links_statistics_dock(qtbot):
    """HistogramDockWidget should store a linked statistics dock."""
    histogram_widget = HistogramWidget()
    qtbot.addWidget(histogram_widget)
    hist_dock = HistogramDockWidget(histogram_widget)
    qtbot.addWidget(hist_dock)
    stats_dock = StatisticsDockWidget(histogram_widget)
    qtbot.addWidget(stats_dock)

    hist_dock.link_statistics_dock(stats_dock)

    assert hist_dock._stats_dock is stats_dock


def test_phasor_center_statistics_widget_update(qtbot):
    """PhasorCenterStatisticsWidget table should populate and be formatted correctly."""
    from napari_phasors.plotter import PhasorCenterStatisticsWidget

    widget = PhasorCenterStatisticsWidget()
    qtbot.addWidget(widget)

    # name -> (G, S, phase, mod)
    data = {"Layer X": (0.5, 0.4, 45.0, 0.7071)}
    widget.update_centers(data)

    table = widget._layer_table
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == "Layer X"
    assert table.item(0, 1).text() == "0.500000"
    assert table.item(0, 2).text() == "0.400000"
    assert table.item(0, 3).text() == "45.0000"
    assert table.item(0, 4).text() == "0.707100"


def test_histogram_widget_respects_range_slider_limits(qtbot):
    """HistogramWidget axes should match the range slider even if data is outside."""
    # Use explicit range_factor for easy testing
    widget = HistogramWidget(range_slider_enabled=True, range_factor=100)
    qtbot.addWidget(widget)

    # Data from 0 to 1
    data = np.array([0.1, 0.5, 0.9])

    # Set range slider to 2 to 3 (outside data) using set_range helper
    # We must also update the slider's absolute limits (slider_min/max)
    # otherwise it will be clipped to the default 0-100 (which is 1.0 with factor 100)
    widget.set_range(2.0, 3.0, slider_min=0.0, slider_max=5.0)
    widget.update_data(data)

    # X-axis should be exactly 2 to 3
    xlim = widget.ax.get_xlim()
    assert xlim[0] == 2.0
    assert xlim[1] == 3.0


def test_histogram_widget_rename_dataset(qtbot):
    """HistogramWidget should update internal tracking dicts when a dataset is renamed."""
    widget = HistogramWidget(bins=10)
    qtbot.addWidget(widget)

    datasets = {
        "Layer A": np.array([1, 2, 3], dtype=float),
    }
    widget.update_multi_data(datasets)

    # Simulate user changing some settings for the layer
    widget._layer_colors["Layer A"] = (1.0, 0.0, 0.0)
    widget._group_assignments["Layer A"] = 2

    # Rename the dataset
    widget.rename_dataset("Layer A", "Layer B")

    assert "Layer A" not in widget._datasets
    assert "Layer B" in widget._datasets

    assert "Layer A" not in widget._counts_per_dataset
    assert "Layer B" in widget._counts_per_dataset

    assert "Layer A" not in widget._layer_colors
    assert "Layer B" in widget._layer_colors
    assert widget._layer_colors["Layer B"] == (1.0, 0.0, 0.0)

    assert "Layer A" not in widget._group_assignments
    assert "Layer B" in widget._group_assignments
    assert widget._group_assignments["Layer B"] == 2


# ---------------------------------------------------------------------------
# Helpers shared by group-metadata tests
# ---------------------------------------------------------------------------


class _FakeLayer:
    """Minimal stand-in for a napari layer."""

    def __init__(self, metadata=None):
        self.metadata = metadata if metadata is not None else {}


class _FakeLayerList:
    def __init__(self, layers):
        self._layers = layers

    def __getitem__(self, name):
        return self._layers[name]


class _FakeViewer:
    def __init__(self, layers):
        self.layers = _FakeLayerList(layers)


# ---------------------------------------------------------------------------
# build_groups_from_layer_metadata
# ---------------------------------------------------------------------------


def test_build_groups_from_layer_metadata_basic():
    """Layers with settings['group'] are grouped by name and assigned IDs."""
    layers = {
        'A': _FakeLayer(
            {
                'settings': {
                    'group': {'name': 'Control', 'color': [1.0, 0.0, 0.0]}
                }
            }
        ),
        'B': _FakeLayer(
            {
                'settings': {
                    'group': {'name': 'Treatment', 'color': [0.0, 0.0, 1.0]}
                }
            }
        ),
        'C': _FakeLayer(
            {
                'settings': {
                    'group': {'name': 'Control', 'color': [1.0, 0.0, 0.0]}
                }
            }
        ),
    }
    viewer = _FakeViewer(layers)

    assignments, names, colors = build_groups_from_layer_metadata(
        viewer, ['A', 'B', 'C']
    )

    assert assignments == {'A': 1, 'C': 1, 'B': 2}
    assert names == {1: 'Control', 2: 'Treatment'}
    assert colors[1] == (1.0, 0.0, 0.0)
    assert colors[2] == (0.0, 0.0, 1.0)


def test_build_groups_from_layer_metadata_skips_missing():
    """Layers without group metadata are silently skipped."""
    layers = {
        'A': _FakeLayer(
            {'settings': {'group': {'name': 'Ctrl', 'color': [1.0, 0.0, 0.0]}}}
        ),
        'B': _FakeLayer({}),
        'C': _FakeLayer({'settings': {}}),
    }
    viewer = _FakeViewer(layers)

    assignments, names, colors = build_groups_from_layer_metadata(
        viewer, ['A', 'B', 'C']
    )

    assert list(assignments.keys()) == ['A']
    assert 'B' not in assignments
    assert 'C' not in assignments


def test_build_groups_from_layer_metadata_missing_layer_name():
    """Layer names that don't exist in the viewer are silently skipped."""
    layers = {
        'A': _FakeLayer(
            {'settings': {'group': {'name': 'G', 'color': [0.0, 1.0, 0.0]}}}
        ),
    }
    viewer = _FakeViewer(layers)

    assignments, names, colors = build_groups_from_layer_metadata(
        viewer, ['A', 'nonexistent']
    )

    assert 'nonexistent' not in assignments
    assert 'A' in assignments


def test_build_groups_from_layer_metadata_empty_input():
    """Empty layer-name list returns three empty dicts."""
    viewer = _FakeViewer({})
    assignments, names, colors = build_groups_from_layer_metadata(viewer, [])
    assert assignments == {}
    assert names == {}
    assert colors == {}


def test_build_groups_from_layer_metadata_backward_compat_top_level():
    """Layers tagged at the top-level metadata['group'] key are still read."""
    layers = {
        'X': _FakeLayer(
            {'group': {'name': 'Legacy', 'color': [0.5, 0.5, 0.0]}}
        ),
    }
    viewer = _FakeViewer(layers)

    assignments, names, colors = build_groups_from_layer_metadata(
        viewer, ['X']
    )

    assert assignments == {'X': 1}
    assert names == {1: 'Legacy'}
    assert colors[1] == (0.5, 0.5, 0.0)


def test_build_groups_from_layer_metadata_settings_takes_priority():
    """settings['group'] takes priority over top-level metadata['group']."""
    layers = {
        'X': _FakeLayer(
            {
                'group': {'name': 'OldName', 'color': [0.0, 0.0, 0.0]},
                'settings': {
                    'group': {'name': 'NewName', 'color': [1.0, 1.0, 1.0]}
                },
            }
        ),
    }
    viewer = _FakeViewer(layers)

    assignments, names, _ = build_groups_from_layer_metadata(viewer, ['X'])

    assert names[1] == 'NewName'


# ---------------------------------------------------------------------------
# build_group_styles_from_layer_metadata
# ---------------------------------------------------------------------------


def test_build_group_styles_colormap():
    """Layers with colormap style produce correct group_styles dict."""
    layers = {
        'A': _FakeLayer(
            {
                'settings': {
                    'group': {
                        'name': 'G1',
                        'color': [1.0, 0.0, 0.0],
                        'colormap': 'turbo',
                        'style': 'colormap',
                    }
                }
            }
        ),
        'B': _FakeLayer(
            {
                'settings': {
                    'group': {
                        'name': 'G2',
                        'color': [0.0, 1.0, 0.0],
                        'style': 'solid',
                    }
                }
            }
        ),
    }
    viewer = _FakeViewer(layers)

    assignments, names, colors, styles = (
        build_group_styles_from_layer_metadata(viewer, ['A', 'B'])
    )

    assert styles[1]['style'] == 'colormap'
    assert styles[1]['colormap'] == 'turbo'
    assert styles[2]['style'] == 'solid'


def test_build_group_styles_defaults_to_solid_when_no_style_field():
    """Layers without a 'style' key default to 'solid' when no colormap."""
    layers = {
        'A': _FakeLayer(
            {'settings': {'group': {'name': 'G', 'color': [1.0, 0.0, 0.0]}}}
        ),
    }
    viewer = _FakeViewer(layers)

    _, _, _, styles = build_group_styles_from_layer_metadata(viewer, ['A'])

    assert styles[1]['style'] == 'solid'


# ---------------------------------------------------------------------------
# save_groups_to_layer_metadata
# ---------------------------------------------------------------------------


def test_save_groups_writes_to_settings():
    """save_groups_to_layer_metadata stores data inside metadata['settings']."""
    layers = {
        'A': _FakeLayer({}),
        'B': _FakeLayer({'settings': {}}),
    }
    viewer = _FakeViewer(layers)

    save_groups_to_layer_metadata(
        viewer,
        ['A', 'B'],
        {'A': 1, 'B': 2},
        {1: 'Alpha', 2: 'Beta'},
        {1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0)},
    )

    assert 'settings' in layers['A'].metadata
    grp_a = layers['A'].metadata['settings']['group']
    assert grp_a['name'] == 'Alpha'
    assert grp_a['color'] == [1.0, 0.0, 0.0]

    grp_b = layers['B'].metadata['settings']['group']
    assert grp_b['name'] == 'Beta'
    assert grp_b['color'] == [0.0, 1.0, 0.0]


def test_save_groups_unassigned_layer_clears_group():
    """A layer not in group_assignments has its group entry removed."""
    layers = {
        'A': _FakeLayer({'settings': {'group': {'name': 'Old'}}}),
    }
    viewer = _FakeViewer(layers)

    save_groups_to_layer_metadata(viewer, ['A'], {}, {}, {})

    assert 'group' not in layers['A'].metadata['settings']


def test_save_groups_with_colormap_style():
    """When group_styles is provided, colormap and style are persisted."""
    layers = {'A': _FakeLayer({})}
    viewer = _FakeViewer(layers)

    save_groups_to_layer_metadata(
        viewer,
        ['A'],
        {'A': 1},
        {1: 'G1'},
        {1: (1.0, 0.0, 0.0)},
        group_styles={
            1: {
                'style': 'colormap',
                'colormap': 'plasma',
                'color': (1.0, 0.0, 0.0),
            }
        },
    )

    grp = layers['A'].metadata['settings']['group']
    assert grp['style'] == 'colormap'
    assert grp['colormap'] == 'plasma'


def test_save_groups_solid_style_omits_colormap():
    """Solid-style groups do not store a colormap key."""
    layers = {'A': _FakeLayer({})}
    viewer = _FakeViewer(layers)

    save_groups_to_layer_metadata(
        viewer,
        ['A'],
        {'A': 1},
        {1: 'G1'},
        {1: (0.0, 1.0, 0.0)},
        group_styles={
            1: {
                'style': 'solid',
                'colormap': 'turbo',
                'color': (0.0, 1.0, 0.0),
            }
        },
    )

    grp = layers['A'].metadata['settings']['group']
    assert grp['style'] == 'solid'
    assert 'colormap' not in grp


def test_save_groups_skips_nonexistent_layer():
    """save_groups_to_layer_metadata silently ignores unknown layer names."""
    layers = {'A': _FakeLayer({})}
    viewer = _FakeViewer(layers)

    # 'ghost' is not in the viewer — should not raise
    save_groups_to_layer_metadata(
        viewer,
        ['A', 'ghost'],
        {'A': 1, 'ghost': 1},
        {1: 'G'},
        {1: (1.0, 0.0, 0.0)},
    )

    assert layers['A'].metadata['settings']['group']['name'] == 'G'


def test_save_then_build_roundtrip():
    """save followed by build restores identical assignments, names, and colors."""
    layers = {'A': _FakeLayer({}), 'B': _FakeLayer({})}
    viewer = _FakeViewer(layers)

    original_assignments = {'A': 1, 'B': 2}
    original_names = {1: 'Control', 2: 'Treated'}
    original_colors = {1: (1.0, 0.0, 0.0), 2: (0.0, 0.0, 1.0)}

    save_groups_to_layer_metadata(
        viewer,
        ['A', 'B'],
        original_assignments,
        original_names,
        original_colors,
    )

    assignments, names, colors = build_groups_from_layer_metadata(
        viewer, ['A', 'B']
    )

    assert assignments == original_assignments
    assert names == original_names
    assert colors[1] == (1.0, 0.0, 0.0)
    assert colors[2] == (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# HistogramWidget — group metadata integration
# ---------------------------------------------------------------------------


def test_histogram_widget_restores_groups_from_layer_metadata(
    qtbot, monkeypatch
):
    """When no group assignments exist, the settings dialog is pre-populated
    from each layer's settings['group'] metadata."""
    from qtpy.QtWidgets import QDialog

    layers = {
        'LayerA': _FakeLayer(
            {'settings': {'group': {'name': 'Ctrl', 'color': [1.0, 0.0, 0.0]}}}
        ),
        'LayerB': _FakeLayer(
            {'settings': {'group': {'name': 'Trt', 'color': [0.0, 0.0, 1.0]}}}
        ),
    }
    viewer = _FakeViewer(layers)

    widget = HistogramWidget(bins=10, viewer=viewer)
    qtbot.addWidget(widget)

    data = {'LayerA': np.array([1.0, 2.0]), 'LayerB': np.array([3.0, 4.0])}
    widget.update_multi_data(data)

    captured = {}

    def fake_exec(self):
        # Record what group_assignments the dialog received
        captured['assignments'] = {
            row['name_edit'].text(): row['layer_combo'].checkedItems()
            for row in self._group_row_data
        }
        return QDialog.Rejected  # don't apply, just inspect

    from napari_phasors._utils import HistogramSettingsDialog

    monkeypatch.setattr(HistogramSettingsDialog, 'exec', fake_exec)

    widget._open_settings_dialog()

    # Dialog should have been pre-populated with the two groups
    assert 'Ctrl' in captured['assignments']
    assert 'Trt' in captured['assignments']
    assert 'LayerA' in captured['assignments']['Ctrl']
    assert 'LayerB' in captured['assignments']['Trt']


def test_histogram_widget_saves_groups_to_layer_metadata(qtbot, monkeypatch):
    """After the settings dialog is accepted, each layer receives a
    settings['group'] entry matching the user's selections."""
    from qtpy.QtWidgets import QDialog

    layers = {
        'LayerA': _FakeLayer({}),
        'LayerB': _FakeLayer({}),
    }
    viewer = _FakeViewer(layers)

    widget = HistogramWidget(bins=10, viewer=viewer)
    qtbot.addWidget(widget)

    data = {'LayerA': np.array([1.0, 2.0]), 'LayerB': np.array([3.0, 4.0])}
    widget.update_multi_data(data)

    def fake_exec(self):
        # Simulate user choosing Grouped mode and confirming
        self.mode_combo.setCurrentText('Grouped')
        return QDialog.Accepted

    from napari_phasors._utils import HistogramSettingsDialog

    monkeypatch.setattr(HistogramSettingsDialog, 'exec', fake_exec)

    # Pre-load group state as if the dialog had been filled in
    widget._group_assignments = {'LayerA': 1, 'LayerB': 2}
    widget._group_names = {1: 'GroupX', 2: 'GroupY'}
    widget._group_colors = {1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0)}

    widget._open_settings_dialog()

    grp_a = layers['LayerA'].metadata['settings']['group']
    grp_b = layers['LayerB'].metadata['settings']['group']
    assert grp_a['name'] == 'GroupX'
    assert grp_b['name'] == 'GroupY'
    assert grp_a['color'] == [1.0, 0.0, 0.0]


def test_histogram_widget_no_viewer_does_not_crash(qtbot, monkeypatch):
    """HistogramWidget without a viewer still opens the settings dialog normally."""
    from qtpy.QtWidgets import QDialog

    widget = HistogramWidget(bins=10)  # no viewer
    qtbot.addWidget(widget)

    widget.update_multi_data(
        {
            'A': np.array([1.0, 2.0]),
            'B': np.array([3.0, 4.0]),
        }
    )

    from napari_phasors._utils import HistogramSettingsDialog

    monkeypatch.setattr(
        HistogramSettingsDialog, 'exec', lambda self: QDialog.Rejected
    )

    # Should not raise even though _viewer is None
    widget._open_settings_dialog()


# ---------------------------------------------------------------------------
# CheckableComboBox tests
# ---------------------------------------------------------------------------


def _make_combo(qtbot, items=None, **kwargs):
    """Helper: create a CheckableComboBox, add items, and register with qtbot."""
    from napari_phasors._utils import CheckableComboBox

    combo = CheckableComboBox(**kwargs)
    qtbot.addWidget(combo)
    if items:
        combo.blockSignals(True)
        combo.addItems(items)
        combo.blockSignals(False)
    return combo


class TestCheckableComboBoxBasics:
    """Unit tests for CheckableComboBox core API."""

    def test_default_parameters(self, qtbot):
        """Default combo has no items, no headers, plain-int role values."""
        from napari_phasors._utils import CheckableComboBox

        combo = CheckableComboBox()
        qtbot.addWidget(combo)
        assert combo._header_count == 0
        assert combo.allItems() == []
        assert combo.checkedItems() == []
        # Role constants must be plain ints, not Qt enum values
        assert type(combo._CONTROL_ROLE) is int
        from napari_phasors._utils import _PrimaryLayerDelegate

        assert type(_PrimaryLayerDelegate.PRIMARY_ROLE) is int

    def test_add_items_and_all_items(self, qtbot):
        """addItems populates allItems(); header rows are excluded."""
        combo = _make_combo(qtbot, items=["a", "b", "c"])
        assert combo.allItems() == ["a", "b", "c"]

    def test_checked_items_default_unchecked(self, qtbot):
        """Items added without explicit check start unchecked."""
        combo = _make_combo(qtbot, items=["x", "y"])
        assert combo.checkedItems() == []

    def test_set_checked_items(self, qtbot):
        """setCheckedItems marks only the named items as checked."""
        combo = _make_combo(qtbot, items=["1", "2", "3"])
        combo.setCheckedItems(["2"])
        assert combo.checkedItems() == ["2"]
        assert combo.allItems() == ["1", "2", "3"]

    def test_select_all_and_deselect_all(self, qtbot):
        """selectAll / deselectAll flip all data items."""
        combo = _make_combo(qtbot, items=["a", "b", "c"])
        combo.selectAll()
        assert combo.checkedItems() == ["a", "b", "c"]
        combo.deselectAll()
        assert combo.checkedItems() == []

    def test_clear_resets_state(self, qtbot):
        """clear() empties model and resets _header_count."""
        combo = _make_combo(qtbot, items=["a", "b"], show_select_all_none=True)
        assert combo._header_count == 2
        combo.clear()
        assert combo._header_count == 0
        assert combo.allItems() == []
        assert combo.checkedItems() == []

    def test_primary_layer_delegate_paint_custom_color(self, qtbot):
        """Test that _PrimaryLayerDelegate.paint handles custom ForegroundRole colors without crashing."""
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QColor, QImage, QPainter
        from qtpy.QtWidgets import QStyleOptionViewItem

        from napari_phasors._utils import CheckableComboBox

        combo = CheckableComboBox()
        qtbot.addWidget(combo)
        combo.addItems(["test_item"])

        item = combo.model().item(0)
        item.setForeground(QColor(255, 0, 0))
        item.setCheckState(Qt.Checked)

        delegate = combo.itemDelegate()
        option = QStyleOptionViewItem()
        option.widget = combo
        option.rect = combo.rect()

        image = QImage(100, 30, QImage.Format_ARGB32)
        painter = QPainter(image)
        try:
            # Should paint without crashing (handles the custom color path, checked)
            delegate.paint(painter, option, combo.model().index(0, 0))

            # Handles the custom color path, unchecked
            item.setCheckState(Qt.Unchecked)
            delegate.paint(painter, option, combo.model().index(0, 0))
        finally:
            painter.end()


class TestCheckableComboBoxHeaderControls:
    """Tests for the show_select_all_none 'All' / 'None' header rows."""

    def test_header_rows_are_added(self, qtbot):
        """When show_select_all_none=True, 2 header rows are prepended."""
        combo = _make_combo(
            qtbot, items=["1", "2", "3"], show_select_all_none=True
        )
        assert combo._header_count == 2
        # Total model rows = 2 headers + 3 data
        assert combo.model().rowCount() == 5
        # First two rows are "All" and "None"
        assert combo.model().item(0).text() == "All"
        assert combo.model().item(1).text() == "None"

    def test_all_items_excludes_header_rows(self, qtbot):
        """allItems() must return only the data rows, not 'All'/'None'."""
        combo = _make_combo(
            qtbot, items=["1", "2", "3"], show_select_all_none=True
        )
        assert combo.allItems() == ["1", "2", "3"]

    def test_checked_items_excludes_header_rows(self, qtbot):
        """checkedItems() must not include the non-checkable header rows."""
        combo = _make_combo(
            qtbot, items=["1", "2", "3"], show_select_all_none=True
        )
        combo.selectAll()
        checked = combo.checkedItems()
        assert "All" not in checked
        assert "None" not in checked
        assert checked == ["1", "2", "3"]

    def test_set_checked_items_skips_headers(self, qtbot):
        """setCheckedItems must not attempt to check the header rows."""
        combo = _make_combo(
            qtbot, items=["1", "2", "3"], show_select_all_none=True
        )
        # Should not raise; header items have no CheckStateRole
        combo.setCheckedItems(["2"])
        assert combo.checkedItems() == ["2"]

    def test_select_all_skips_headers(self, qtbot):
        """selectAll() must only check data rows, not the 'All'/'None' rows."""
        combo = _make_combo(qtbot, items=["a", "b"], show_select_all_none=True)
        combo.selectAll()
        from qtpy.QtCore import Qt

        # Header items should NOT have a checkable check state
        for row in range(combo._header_count):
            item = combo.model().item(row)
            assert not (item.flags() & Qt.ItemIsUserCheckable)

    def test_deselect_all_skips_headers(self, qtbot):
        """deselectAll() must only uncheck data rows."""
        combo = _make_combo(qtbot, items=["a", "b"], show_select_all_none=True)
        combo.selectAll()
        combo.deselectAll()
        assert combo.checkedItems() == []

    def test_is_header_row(self, qtbot):
        """_is_header_row correctly identifies header vs data rows."""
        combo = _make_combo(qtbot, items=["1"], show_select_all_none=True)
        assert combo._is_header_row(0)  # "All"
        assert combo._is_header_row(1)  # "None"
        assert not combo._is_header_row(2)  # "1"

    def test_no_header_rows_without_flag(self, qtbot):
        """Without show_select_all_none, no header rows are added."""
        combo = _make_combo(qtbot, items=["a", "b"])
        assert combo._header_count == 0
        assert combo.model().rowCount() == 2

    def test_clear_then_repopulate_adds_headers_once(self, qtbot):
        """After clear+addItems a fresh set of 2 headers is added."""
        combo = _make_combo(qtbot, items=["1", "2"], show_select_all_none=True)
        combo.clear()
        combo.addItems(["x", "y", "z"])
        assert combo._header_count == 2
        assert combo.model().rowCount() == 5  # 2 headers + 3 data
        assert combo.allItems() == ["x", "y", "z"]


class TestCheckableComboBoxDisplayText:
    """Tests for _update_display_text with all parameter combinations."""

    def test_no_selection_shows_placeholder_by_default(self, qtbot):
        """Without no_selection_text, an empty selection shows the placeholder."""
        combo = _make_combo(
            qtbot, items=["a", "b"], placeholder="Pick something"
        )
        combo.deselectAll()
        assert combo.lineEdit().placeholderText() == "Pick something"
        assert combo.lineEdit().text() == ""

    def test_no_selection_shows_no_selection_text(self, qtbot):
        """When no_selection_text is set, it is shown when nothing is checked."""
        combo = _make_combo(
            qtbot,
            items=["1", "2"],
            no_selection_text="No labels",
        )
        combo.deselectAll()
        assert combo.lineEdit().text() == "No labels"

    def test_all_selected_shows_placeholder_in_no_primary_mode(self, qtbot):
        """All items checked → placeholder shown (e.g. 'All Labels')."""
        combo = _make_combo(
            qtbot,
            items=["1", "2", "3"],
            enable_primary_layer=False,
            placeholder="All Labels",
        )
        combo.selectAll()
        assert combo.lineEdit().text() == ""
        assert combo.lineEdit().placeholderText() == "All Labels"

    def test_single_item_selected_shows_item_text(self, qtbot):
        """Exactly one item checked → display that item's text."""
        combo = _make_combo(
            qtbot, items=["alpha", "beta"], enable_primary_layer=False
        )
        combo.setCheckedItems(["alpha"])
        assert combo.lineEdit().text() == "alpha"

    def test_partial_selection_shows_count_with_unit(self, qtbot):
        """Multiple (not all) items → show 'N <unit> selected'."""
        combo = _make_combo(
            qtbot,
            items=["1", "2", "3"],
            enable_primary_layer=False,
            unit="labels",
        )
        combo.setCheckedItems(["1", "3"])
        assert combo.lineEdit().text() == "2 labels selected"

    def test_partial_selection_singular_unit(self, qtbot):
        """Single item shows singular unit (strips trailing 's')."""
        combo = _make_combo(
            qtbot,
            items=["1", "2", "3"],
            enable_primary_layer=False,
            unit="labels",
        )
        combo.setCheckedItems(["2"])
        # single item → item text, not count string
        assert combo.lineEdit().text() == "2"

    def test_all_selected_with_headers_uses_correct_all_count(self, qtbot):
        """all_count excludes header rows so 'all checked' triggers correctly."""
        combo = _make_combo(
            qtbot,
            items=["1", "2"],
            enable_primary_layer=False,
            placeholder="All Labels",
            show_select_all_none=True,
        )
        combo.selectAll()
        # all_count = 2 (not 4), len(checked)=2, so placeholder should show
        assert combo.lineEdit().text() == ""
        assert combo.lineEdit().placeholderText() == "All Labels"

    def test_no_selection_with_headers_shows_no_selection_text(self, qtbot):
        """no_selection_text works correctly when headers are present."""
        combo = _make_combo(
            qtbot,
            items=["1", "2", "3"],
            enable_primary_layer=False,
            placeholder="All Labels",
            show_select_all_none=True,
            no_selection_text="No labels",
        )
        combo.deselectAll()
        assert combo.lineEdit().text() == "No labels"

    def test_deselect_all_then_select_all_text_cycles(self, qtbot):
        """Cycling between all-selected and none-selected updates text correctly."""
        combo = _make_combo(
            qtbot,
            items=["1", "2"],
            enable_primary_layer=False,
            placeholder="All Labels",
            show_select_all_none=True,
            no_selection_text="No labels",
        )
        combo.selectAll()
        assert combo.lineEdit().text() == ""  # placeholder mode
        assert combo.lineEdit().placeholderText() == "All Labels"

        combo.deselectAll()
        assert combo.lineEdit().text() == "No labels"

        combo.selectAll()
        assert combo.lineEdit().text() == ""
        assert combo.lineEdit().placeholderText() == "All Labels"

    def test_select_deselect_while_signals_blocked(self, qtbot):
        """selectAll / deselectAll still update display when signals are blocked."""
        combo = _make_combo(
            qtbot,
            items=["1", "2", "3"],
            enable_primary_layer=False,
            placeholder="All Labels",
            no_selection_text="No labels",
        )
        combo.blockSignals(True)
        combo.deselectAll()
        combo.blockSignals(False)
        assert combo.lineEdit().text() == "No labels"

        combo.blockSignals(True)
        combo.selectAll()
        combo.blockSignals(False)
        assert combo.lineEdit().text() == ""
        assert combo.lineEdit().placeholderText() == "All Labels"


def test_check_state_value_normalises_qt_enum():
    """_check_state_value returns the integer value for both PyQt5 ints and PyQt6 enums."""
    from qtpy.QtCore import Qt

    from napari_phasors._utils import _check_state_value

    assert _check_state_value(Qt.Checked) == 2
    assert _check_state_value(Qt.Unchecked) == 0
