import csv

import matplotlib.pyplot as plt
import numpy as np

from napari_phasors._utils import (
    CurrentPageStackedWidget,
    HistogramDockWidget,
    HistogramWidget,
    StatisticsDockWidget,
    StatisticsTableWidget,
    build_group_styles_from_layer_metadata,
    build_groups_from_layer_metadata,
    save_groups_to_layer_metadata,
)


def _clicked_receiver_count(button):
    """Return the number of receivers connected to ``button.clicked``.

    PyQt's ``QObject.receivers`` accepts the bound ``SignalInstance``
    directly, while PySide6 requires the signal signature wrapped with the
    ``SIGNAL`` macro (a bare instance or signature string returns 0 there).
    """
    try:
        from qtpy.QtCore import SIGNAL  # PySide6 only
    except ImportError:
        return button.receivers(button.clicked)
    return button.receivers(SIGNAL("clicked()"))


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
    assert widget.save_button.isEnabled()

    widget.clear()

    assert widget.counts is None
    assert widget.bin_edges is None
    assert widget.bin_centers is None
    assert widget._datasets == {}
    assert widget._raw_valid_data is None
    assert not widget._settings_button.isEnabled()
    assert not widget.save_button.isEnabled()


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


def test_histogram_widget_gamma_uses_power_norm(qtbot):
    """A non-unity gamma renders the colormap through a PowerNorm."""
    from matplotlib.colors import PowerNorm

    widget = HistogramWidget(bins=10)
    qtbot.addWidget(widget)

    widget.update_data(np.linspace(0.0, 1.0, 100))

    colors = np.array([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=float)

    # Default gamma keeps a plain linear normalisation.
    widget.update_colormap(colormap_colors=colors, contrast_limits=[0.0, 1.0])
    _, norm = widget._get_cmap_and_norm()
    assert not isinstance(norm, PowerNorm)

    # A non-unity gamma switches to a matching PowerNorm.
    widget.update_colormap(
        colormap_colors=colors, contrast_limits=[0.0, 1.0], gamma=0.4
    )
    assert widget.gamma == 0.4
    _, norm = widget._get_cmap_and_norm()
    assert isinstance(norm, PowerNorm)
    assert norm.gamma == 0.4

    # Omitting gamma on a later update preserves the stored value.
    widget.update_colormap(colormap_colors=colors, contrast_limits=[0.0, 2.0])
    assert widget.gamma == 0.4


def test_histogram_widget_grouped_sd_band(qtbot):
    """Grouped mode draws a shaded SD band per multi-file group when enabled."""
    from matplotlib.collections import PolyCollection

    widget = HistogramWidget(bins=10)
    qtbot.addWidget(widget)

    rng = np.random.default_rng(0)
    # Two files per group so an across-file standard deviation exists.
    datasets = {
        "G1::a": rng.normal(0.3, 0.05, 500),
        "G1::b": rng.normal(0.32, 0.05, 500),
        "G2::a": rng.normal(0.7, 0.05, 500),
        "G2::b": rng.normal(0.68, 0.05, 500),
    }
    widget._group_assignments = {
        "G1::a": 1,
        "G1::b": 1,
        "G2::a": 2,
        "G2::b": 2,
    }
    widget._group_names = {1: "Group 1", 2: "Group 2"}
    widget.display_mode = "Grouped"
    widget.update_multi_data(datasets)

    widget.show_sd = True
    bands = [c for c in widget.ax.collections if isinstance(c, PolyCollection)]
    assert len(bands) >= 2  # one SD band per group

    widget.show_sd = False
    bands = [c for c in widget.ax.collections if isinstance(c, PolyCollection)]
    assert bands == []  # no bands when SD shading is off


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


# ---------------------------------------------------------------------------
# natural_sort_key
# ---------------------------------------------------------------------------


def test_natural_sort_key_orders_numerically():
    from napari_phasors._utils import natural_sort_key

    paths = ["/d/img10.tif", "/d/img2.tif", "/d/img1.tif"]
    assert sorted(paths, key=natural_sort_key) == [
        "/d/img1.tif",
        "/d/img2.tif",
        "/d/img10.tif",
    ]


# ---------------------------------------------------------------------------
# FileOrderDialog
# ---------------------------------------------------------------------------


def test_file_order_dialog_reorder_and_getters(qtbot):
    from napari_phasors._utils import FileOrderDialog

    paths = ["/d/a.lsm", "/d/b.lsm", "/d/c.lsm"]
    dlg = FileOrderDialog(paths, estimated_shape=(3, 4, 4))
    qtbot.addWidget(dlg)

    assert dlg.file_list.count() == 3
    assert dlg.get_ordered_paths() == paths
    assert dlg.get_axis_labels() == ["Z", "Y", "X"]
    assert dlg.get_axis_order() is None
    assert dlg.get_z_spacing() == 1.0
    assert "(3, 4, 4)" in dlg.shape_label.text()

    # Move the first item down.
    dlg.file_list.setCurrentRow(0)
    dlg._move_down()
    assert dlg.get_ordered_paths() == ["/d/b.lsm", "/d/a.lsm", "/d/c.lsm"]

    # Move the last item up.
    dlg.file_list.setCurrentRow(2)
    dlg._move_up()
    assert dlg.get_ordered_paths() == ["/d/b.lsm", "/d/c.lsm", "/d/a.lsm"]

    # Boundary no-ops: move-up on first row and move-down on last row.
    dlg.file_list.setCurrentRow(0)
    dlg._move_up()
    dlg.file_list.setCurrentRow(2)
    dlg._move_down()
    assert dlg.get_ordered_paths() == ["/d/b.lsm", "/d/c.lsm", "/d/a.lsm"]


def test_file_order_dialog_items_not_drop_enabled(qtbot):
    """Items must not carry ``Qt.ItemIsDropEnabled``.

    Combined with ``QAbstractItemView.InternalMove``, a per-item drop flag
    lets Qt treat a drag as landing *on* another item instead of *between*
    rows, which can make the dragged item vanish (``takeItem`` fires but the
    reinsert doesn't land in the right spot). See the ``FileOrderDialog``
    docstring / PR history for the reported bug.
    """
    from qtpy.QtCore import Qt

    from napari_phasors._utils import FileOrderDialog

    dlg = FileOrderDialog(["/d/a.lsm", "/d/b.lsm"], estimated_shape=(4, 4))
    qtbot.addWidget(dlg)

    for row in range(dlg.file_list.count()):
        item = dlg.file_list.item(row)
        assert not (item.flags() & Qt.ItemIsDropEnabled)
        assert item.flags() & Qt.ItemIsDragEnabled
        assert item.flags() & Qt.ItemIsSelectable
        assert item.flags() & Qt.ItemIsEnabled


def test_file_order_dialog_axis_label_defaults(qtbot):
    from napari_phasors._utils import FileOrderDialog

    d2 = FileOrderDialog(["a.tif", "b.tif"], estimated_shape=(4, 4))
    qtbot.addWidget(d2)
    assert d2.get_axis_labels() == ["Y", "X"]

    d4 = FileOrderDialog(["a"], estimated_shape=(2, 3, 4, 4))
    qtbot.addWidget(d4)
    assert d4.get_axis_labels() == ["T", "Z", "Y", "X"]

    d5 = FileOrderDialog(["a"], estimated_shape=(2, 2, 3, 4, 4))
    qtbot.addWidget(d5)
    assert d5.get_axis_labels()[0].startswith("Axis")

    # No estimated shape -> empty labels edit -> None, and "Unavailable" shape.
    dn = FileOrderDialog(["a"])
    qtbot.addWidget(dn)
    assert dn.get_axis_labels() is None
    assert "Unavailable" in dn.shape_label.text()

    # Empty axis labels edit returns None.
    d4.axis_labels_edit.setText("   ")
    assert d4.get_axis_labels() is None


def test_file_order_dialog_z_spacing_parsing(qtbot):
    from napari_phasors._utils import FileOrderDialog

    dlg = FileOrderDialog(["a"], initial_z_spacing=2.5)
    qtbot.addWidget(dlg)
    assert dlg.get_z_spacing() == 2.5

    dlg.z_spacing_edit.setText("not-a-number")
    assert dlg.get_z_spacing() == 1.0

    dlg.z_spacing_edit.setText("-3")
    assert dlg.get_z_spacing() == 1.0


# ---------------------------------------------------------------------------
# StatisticsTableWidget - copy / context menu / key handling
# ---------------------------------------------------------------------------


def test_statistics_table_copy_all_and_selection(qtbot):
    from qtpy.QtWidgets import QApplication

    table = StatisticsTableWidget()
    qtbot.addWidget(table)
    table.update_statistics({"A": np.array([1.0, 2.0, 3.0])})

    # No selection -> copy the whole table.
    table.clearSelection()
    table._copy_selection()
    clip = QApplication.clipboard().text()
    assert "A" in clip

    # With headers.
    table.selectAll()
    table._copy_selection(include_headers=True)
    clip = QApplication.clipboard().text()
    assert "Name" in clip.splitlines()[0]


def test_statistics_table_keypress_shortcuts(qtbot):
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QKeyEvent
    from qtpy.QtWidgets import QApplication

    table = StatisticsTableWidget()
    qtbot.addWidget(table)
    table.update_statistics({"A": np.array([1.0, 2.0])})

    # Ctrl+A selects all.
    ev_a = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_A, Qt.ControlModifier)
    table.keyPressEvent(ev_a)
    assert len(table.selectedItems()) > 0

    # Ctrl+C copies the selection to the clipboard.
    ev_c = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_C, Qt.ControlModifier)
    table.keyPressEvent(ev_c)
    assert "A" in QApplication.clipboard().text()

    # A non-shortcut key is passed through without error.
    ev_x = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_X, Qt.NoModifier)
    table.keyPressEvent(ev_x)


def test_statistics_table_context_menu_actions(qtbot, monkeypatch):
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QApplication, QMenu

    import napari_phasors._utils as utils_mod

    table = StatisticsTableWidget()
    qtbot.addWidget(table)
    table.update_statistics({"A": np.array([1.0, 2.0])})
    table.selectAll()

    # Patching ``QMenu.exec`` on the class does not intercept PySide6's
    # native menu (the real menu would open and block ~20s per call), so
    # replace the QMenu *name* used inside _utils with a Python subclass
    # whose exec returns the requested action without showing anything.
    class _NonBlockingMenu(QMenu):
        chosen = None

        def exec(self, *args, **kwargs):  # noqa: A003
            for act in self.actions():
                if act.text() == _NonBlockingMenu.chosen:
                    return act
            return None

    monkeypatch.setattr(utils_mod, "QMenu", _NonBlockingMenu)

    for label in ("Copy", "Copy with Headers", "Select All"):
        _NonBlockingMenu.chosen = label
        table._show_context_menu(QPoint(1, 1))

    assert "A" in QApplication.clipboard().text()


def test_statistics_table_update_group_statistics(qtbot):
    table = StatisticsTableWidget()
    qtbot.addWidget(table)
    datasets = {
        "l1": np.array([1.0, 2.0, 3.0]),
        "l2": np.array([4.0, 5.0, 6.0]),
        "l3": np.array([7.0, 8.0, 9.0]),
    }
    assignments = {"l1": 1, "l2": 1, "l3": 2}
    table.update_group_statistics(
        datasets, assignments, group_names={1: "First", 2: "Second"}
    )
    assert table.rowCount() == 2
    names = {table.item(r, 0).text() for r in range(table.rowCount())}
    assert names == {"First", "Second"}


# ---------------------------------------------------------------------------
# Colormap helper functions
# ---------------------------------------------------------------------------


def test_resolve_colormap_by_name_variants(qtbot):
    from matplotlib.colors import Colormap

    from napari_phasors._utils import resolve_colormap_by_name

    # Sentinel / None / non-string -> None.
    assert resolve_colormap_by_name("Select color...") is None
    assert resolve_colormap_by_name(None) is None
    assert resolve_colormap_by_name(123) is None
    # A matplotlib colormap name resolves to a Colormap.
    assert isinstance(resolve_colormap_by_name("viridis"), Colormap)
    # An unknown name returns None.
    assert resolve_colormap_by_name("definitely-not-a-cmap") is None


def test_create_colormaps_from_qcolor(qtbot):
    from matplotlib.colors import LinearSegmentedColormap
    from qtpy.QtGui import QColor

    from napari_phasors._utils import (
        create_mpl_colormap_from_qcolor,
        create_napari_colormap_from_qcolor,
    )

    color = QColor(255, 0, 0, 255)
    nap = create_napari_colormap_from_qcolor(color, name="red")
    assert nap.name == "red"
    assert len(nap.colors) == 2

    mpl = create_mpl_colormap_from_qcolor(color, name="red")
    assert isinstance(mpl, LinearSegmentedColormap)


def test_resolve_napari_layer_colormap(qtbot):
    from qtpy.QtGui import QColor

    from napari_phasors._utils import resolve_napari_layer_colormap

    # A normal name passes through unchanged.
    assert resolve_napari_layer_colormap("viridis") == "viridis"
    # Sentinel with a custom colour returns a Colormap object.
    cmap = resolve_napari_layer_colormap(
        "Select color...", custom_color=QColor(0, 255, 0, 255)
    )
    assert hasattr(cmap, "colors")
    # Sentinel without a colour returns None.
    assert (
        resolve_napari_layer_colormap("Select color...", custom_color=None)
        is None
    )


def test_create_colormap_icon(qtbot):
    from qtpy.QtGui import QIcon

    from napari_phasors._utils import create_colormap_icon

    # Valid colormap -> a non-null icon.
    icon = create_colormap_icon("viridis", width=20, height=8)
    assert isinstance(icon, QIcon)
    assert not icon.isNull()
    # Invalid colormap still returns an (empty) icon without raising.
    assert isinstance(create_colormap_icon("nope-cmap"), QIcon)


def test_populate_colormap_combobox(qtbot):
    from qtpy.QtWidgets import QComboBox

    from napari_phasors._utils import populate_colormap_combobox

    combo = QComboBox()
    qtbot.addWidget(combo)
    populate_colormap_combobox(
        combo,
        include_select_color=True,
        available_colormaps=["viridis", "magma"],
        selected="magma",
    )
    assert combo.itemText(0) == "Select color..."
    assert combo.currentText() == "magma"

    # Without the sentinel entry and with default selection.
    combo2 = QComboBox()
    qtbot.addWidget(combo2)
    populate_colormap_combobox(
        combo2,
        include_select_color=False,
        available_colormaps=["viridis", "magma"],
    )
    assert combo2.count() == 2
    assert combo2.currentIndex() == 0


def test_register_extra_colormaps_idempotent():
    """Calling register_extra_colormaps twice skips already-registered names."""
    from napari.utils.colormaps import AVAILABLE_COLORMAPS

    from napari_phasors._utils import (
        EXTRA_MATPLOTLIB_COLORMAPS,
        register_extra_colormaps,
    )

    # First call (possibly a no-op if already registered at plugin import).
    register_extra_colormaps()
    for name in EXTRA_MATPLOTLIB_COLORMAPS:
        assert name in AVAILABLE_COLORMAPS

    # Second call must hit the "already registered" skip branch for every
    # name without raising or duplicating entries.
    register_extra_colormaps()
    for name in EXTRA_MATPLOTLIB_COLORMAPS:
        assert name in AVAILABLE_COLORMAPS


def test_available_colormap_names_includes_extras():
    from napari_phasors._utils import (
        EXTRA_MATPLOTLIB_COLORMAPS,
        available_colormap_names,
    )

    names = available_colormap_names()
    for name in EXTRA_MATPLOTLIB_COLORMAPS:
        assert name in names
    # No duplicates from colormaps already present in napari's own list.
    assert len(names) == len(set(names))


def test_colormap_legend_proxy_and_handler(qtbot):
    from matplotlib.transforms import IdentityTransform

    from napari_phasors._utils import (
        ColormapLegendHandler,
        ColormapLegendProxy,
    )

    cmap = plt.get_cmap("viridis")
    # n_colors is clamped to a minimum of 2.
    proxy = ColormapLegendProxy(cmap, linewidth=2, style="full", n_colors=1)
    assert proxy.n_colors == 2

    handler = ColormapLegendHandler()
    common = {
        "legend": None,
        "xdescent": 0.0,
        "ydescent": 0.0,
        "width": 10.0,
        "height": 4.0,
        "fontsize": 8,
        "trans": IdentityTransform(),
    }
    full_artists = handler.create_artists(orig_handle=proxy, **common)
    assert len(full_artists) == 1

    cat_proxy = ColormapLegendProxy(
        cmap, linewidth=2, style="categorical", n_colors=5
    )
    cat_artists = handler.create_artists(orig_handle=cat_proxy, **common)
    assert len(cat_artists) == 1


# ---------------------------------------------------------------------------
# HistogramWidget rendering modes & central tendency
# ---------------------------------------------------------------------------


def test_histogram_compute_central_tendency():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    edges = np.linspace(0.0, 5.0, 6)
    centers = (edges[:-1] + edges[1:]) / 2.0

    assert HistogramWidget._compute_central_tendency(data, "Mean") == 2.5
    assert HistogramWidget._compute_central_tendency(data, "Median") == 2.5
    com = HistogramWidget._compute_central_tendency(
        data, "Center of mass", centers, edges
    )
    assert com is not None
    # Center of mass without bins falls back to the mean.
    assert (
        HistogramWidget._compute_central_tendency(data, "Center of mass")
        == 2.5
    )
    # Empty data and unknown methods return None.
    assert (
        HistogramWidget._compute_central_tendency(np.array([]), "Mean") is None
    )
    assert HistogramWidget._compute_central_tendency(data, "Nope") is None


def test_histogram_widget_render_all_modes(qtbot):
    """Exercise every display mode and central-tendency option."""
    widget = HistogramWidget(bins=5)
    qtbot.addWidget(widget)
    datasets = {
        "A": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "B": np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
    }
    widget.update_multi_data(datasets)
    widget._group_assignments = {"A": 1, "B": 2}
    widget._group_names = {1: "G1", 2: "G2"}

    # Toggle background and SD shading (each triggers a re-render).
    widget.show_sd = True
    widget.white_background = True
    widget.white_background = False

    for ct in ("Mean", "Median", "Center of mass"):
        widget._central_tendency = ct
        for mode in ("Individual layers", "Grouped", "Merged"):
            widget.display_mode = mode
            widget._render()

    assert widget.counts is not None


def test_histogram_widget_render_single_dataset_central_tendency(qtbot):
    """A single-dataset render with central tendency draws a marker line."""
    widget = HistogramWidget(bins=5)
    qtbot.addWidget(widget)
    widget.update_data(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    widget._central_tendency = "Mean"
    widget._render()
    assert widget.counts is not None


def test_histogram_widget_center_of_mass_uses_raw_unsmoothed_data(qtbot):
    """The center-of-mass line comes from the raw histogram, not the smoothed
    display curve, so smoothing on/off must not move it."""
    rng = np.random.default_rng(0)
    data = np.concatenate(
        [rng.normal(2.0, 0.3, 5000), rng.normal(5.0, 0.5, 1500)]
    )
    widget = HistogramWidget(bins=150)
    qtbot.addWidget(widget)
    widget._central_tendency = "Center of mass"
    widget.update_data(data)

    expected = float(np.average(widget.bin_centers, weights=widget.counts))

    def _drawn_com():
        widget._render()
        lines = [
            ln for ln in widget.ax.get_lines() if ln.get_linestyle() == "--"
        ]
        assert len(lines) == 1
        return float(lines[0].get_xdata()[0])

    widget._smooth_curves = True
    com_smoothed = _drawn_com()
    widget._smooth_curves = False
    com_raw = _drawn_com()

    # Identical regardless of the display smoothing, and equal to the raw
    # histogram's center of mass.
    assert np.isclose(com_smoothed, expected)
    assert np.isclose(com_raw, expected)
    assert np.isclose(com_smoothed, com_raw)


def test_histogram_widget_taller_default_canvas_height(qtbot):
    """The histogram canvas has an increased minimum height."""
    widget = HistogramWidget()
    qtbot.addWidget(widget)
    assert widget.fig.canvas.minimumHeight() >= 180


def test_histogram_widget_save_menu_dispatches_to_png_and_csv(qtbot):
    """The merged "Save Histogram…" button opens a menu that dispatches to
    the correct export routine depending on which action is chosen."""
    from unittest.mock import MagicMock, patch

    widget = HistogramWidget(bins=4)
    qtbot.addWidget(widget)
    widget.update_data(np.array([1.0, 2.0, 3.0]))
    assert widget.save_button.isEnabled()

    def _mock_menu_returning(chosen_index):
        mock_menu = MagicMock()
        png_action = MagicMock(name="png_action")
        csv_action = MagicMock(name="csv_action")
        mock_menu.addAction.side_effect = [png_action, csv_action]
        actions = [png_action, csv_action]
        mock_menu.exec.return_value = (
            actions[chosen_index] if chosen_index is not None else None
        )
        return mock_menu

    # Choosing "Save as PNG" calls the PNG export only.
    with (
        patch.object(widget, '_save_histogram_png') as mock_png,
        patch.object(widget, '_save_histogram_csv') as mock_csv,
        patch('napari_phasors._utils.QMenu') as mock_menu_cls,
    ):
        mock_menu_cls.return_value = _mock_menu_returning(0)
        widget._show_save_menu()
        mock_png.assert_called_once()
        mock_csv.assert_not_called()

    # Choosing "Save as CSV" calls the CSV export only.
    with (
        patch.object(widget, '_save_histogram_png') as mock_png,
        patch.object(widget, '_save_histogram_csv') as mock_csv,
        patch('napari_phasors._utils.QMenu') as mock_menu_cls,
    ):
        mock_menu_cls.return_value = _mock_menu_returning(1)
        widget._show_save_menu()
        mock_csv.assert_called_once()
        mock_png.assert_not_called()

    # Dismissing the menu without a choice triggers neither export.
    with (
        patch.object(widget, '_save_histogram_png') as mock_png,
        patch.object(widget, '_save_histogram_csv') as mock_csv,
        patch('napari_phasors._utils.QMenu') as mock_menu_cls,
    ):
        mock_menu_cls.return_value = _mock_menu_returning(None)
        widget._show_save_menu()
        mock_png.assert_not_called()
        mock_csv.assert_not_called()


def test_histogram_widget_save_button_wired_to_save_menu(qtbot):
    """The merged save button's ``clicked`` signal is wired to the
    export-format menu handler (not to the old per-format buttons)."""
    widget = HistogramWidget(bins=4)
    qtbot.addWidget(widget)
    widget.update_data(np.array([1.0, 2.0, 3.0]))

    assert _clicked_receiver_count(widget.save_button) >= 1
    assert not hasattr(widget, 'save_png_button')
    assert not hasattr(widget, 'save_csv_button')


def test_current_page_stacked_widget_sizes_to_active_page(qtbot):
    """``CurrentPageStackedWidget`` must not let a hidden page's larger size
    hint force the container wider than the currently active page needs.

    Regression test for the FRET/Selection tab bug where a plain
    ``QStackedWidget`` reported the max size hint across *all* pages
    (including hidden ones), preventing the tab from shrinking below the
    widest page even when a much narrower page was on display.
    """
    from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

    stack = CurrentPageStackedWidget()
    qtbot.addWidget(stack)

    small_page = QLineEdit()

    big_page = QWidget()
    big_layout = QHBoxLayout(big_page)
    big_layout.setContentsMargins(0, 0, 0, 0)
    big_layout.addWidget(QLabel("A" * 60))
    big_layout.addWidget(QLineEdit())
    big_layout.addWidget(QLineEdit())

    stack.addWidget(small_page)
    stack.addWidget(big_page)
    stack.show()

    stack.setCurrentIndex(1)
    big_page_min_width = stack.minimumSizeHint().width()
    assert big_page_min_width > small_page.minimumSizeHint().width()

    stack.setCurrentIndex(0)
    # The stack's own hint must match the small page alone, not the max
    # across all pages (the default QStackedWidget behavior).
    assert (
        stack.minimumSizeHint().width() == small_page.minimumSizeHint().width()
    )
    assert stack.minimumSizeHint().width() < big_page_min_width


def test_statistics_dock_export_csv(qtbot, tmp_path, monkeypatch):
    """Export the statistics tables to CSV (and handle the cancel path)."""
    import os

    from qtpy.QtWidgets import QFileDialog

    hw = HistogramWidget(bins=12)
    qtbot.addWidget(hw)
    dock = StatisticsDockWidget(hw)
    qtbot.addWidget(dock)
    hw.update_data(np.array([1.0, 2.0, 3.0, 4.0]))

    # The low-level writer produces a CSV with header + rows.
    out = str(tmp_path / "stats.csv")
    dock._write_table_to_csv(dock.layer_stats_table, out)
    assert os.path.exists(out)
    with open(out) as f:
        assert f.readline().strip() != ""

    # The export entry point (sections are not on-screen in headless mode,
    # so it just walks the branch logic) plus the cancelled-dialog no-op.
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(tmp_path / "e"), ""),
    )
    dock._export_table_csv_impl()
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: ("", "")
    )
    dock._export_table_csv_impl()


def test_histogram_widget_save_png(qtbot, tmp_path, monkeypatch):
    """Save the histogram figure as a PNG (and handle the cancel path)."""
    from qtpy.QtWidgets import QFileDialog

    w = HistogramWidget(bins=5)
    qtbot.addWidget(w)
    w.update_data(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    out = tmp_path / "hist.png"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "")
    )
    w._save_histogram_png()
    assert out.exists()

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: ("", "")
    )
    w._save_histogram_png()


def test_histogram_widget_square_aspect_keeps_data(
    qtbot, tmp_path, monkeypatch
):
    """The square option squares the axes box without squashing the data.

    The x axis is in data units and y is in counts, so a 1:1 *data* aspect
    collapses the x range. Only the box aspect may be constrained, and it
    must survive an export so the viewer plot stays square.
    """
    from qtpy.QtWidgets import QFileDialog

    w = HistogramWidget(bins=50)
    qtbot.addWidget(w)
    # Counts far larger than the x range: the regime a data aspect ruins.
    w.update_data(np.repeat(np.linspace(0.0, 10.0, 50), 400))

    w._aspect_ratio = "equal"
    w._render()
    expected_xlim = w.ax.get_xlim()

    assert w.ax.get_box_aspect() == 1
    assert w.ax.get_aspect() == "auto"

    out = tmp_path / "square.png"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "")
    )
    w._save_histogram_png()
    assert out.exists()

    # The exported figure keeps the full x range, and the on-screen plot is
    # still square once the export has finished.
    assert w.ax.get_xlim() == expected_xlim
    assert w.ax.get_box_aspect() == 1
    assert w.ax.get_aspect() == "auto"

    # Switching back to auto releases the box constraint.
    w._aspect_ratio = "auto"
    w._render()
    assert w.ax.get_box_aspect() is None


def test_checkable_combobox_event_filter(qtbot):
    """Drive the CheckableComboBox event filter: line-edit clicks, hover,
    header All/None clicks, item toggle and leave."""
    from qtpy.QtCore import QEvent, QPointF, Qt
    from qtpy.QtGui import QMouseEvent

    combo = _make_combo(
        qtbot, items=["a", "b", "c"], show_select_all_none=True
    )
    le = combo.lineEdit()

    def mouse(etype, x, y):
        # Pass globalPos explicitly to select the non-deprecated overload.
        return QMouseEvent(
            etype,
            QPointF(x, y),
            QPointF(x, y),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )

    # Line-edit press/release branches (release opens the popup).
    assert combo.eventFilter(le, mouse(QEvent.MouseButtonPress, 2, 2)) is True
    assert (
        combo.eventFilter(le, mouse(QEvent.MouseButtonRelease, 2, 2)) is True
    )

    combo.showPopup()
    view = combo.view()
    vp = view.viewport()

    def center_of(row):
        rect = view.visualRect(combo.model().index(row, 0))
        c = rect.center()
        return c.x(), c.y()

    # Hover move over the viewport.
    mx, my = center_of(combo._header_count)
    combo.eventFilter(vp, mouse(QEvent.MouseMove, mx, my))

    # Click the "All" then "None" header rows.
    ax, ay = center_of(0)
    combo.eventFilter(vp, mouse(QEvent.MouseButtonRelease, ax, ay))
    nx, ny = center_of(1)
    combo.eventFilter(vp, mouse(QEvent.MouseButtonRelease, nx, ny))

    # Toggle a data item.
    dx, dy = center_of(combo._header_count)
    combo.eventFilter(vp, mouse(QEvent.MouseButtonRelease, dx, dy))

    # Leave clears the hover state.
    combo.eventFilter(vp, QEvent(QEvent.Leave))


def test_primary_layer_delegate_paint(qtbot):
    """Render the _PrimaryLayerDelegate for header, plain, coloured and
    primary rows to cover its paint/sizeHint/labelRect code."""
    from qtpy.QtCore import QRect, Qt
    from qtpy.QtGui import QBrush, QColor, QImage, QPainter
    from qtpy.QtWidgets import QStyleOptionViewItem

    combo = _make_combo(qtbot, items=["a", "b"], show_select_all_none=True)
    delegate = combo._delegate
    model = combo.model()

    img = QImage(200, 120, QImage.Format_ARGB32)
    painter = QPainter(img)
    try:
        # Includes the "All"/"None" header rows, which have no check state:
        # painting them used to raise TypeError in _check_state_value(None).
        for row in range(model.rowCount()):
            idx = model.index(row, 0)
            opt = QStyleOptionViewItem()
            opt.rect = QRect(0, 0, 200, 25)
            delegate.paint(painter, opt, idx)
            delegate.sizeHint(opt, idx)
            delegate.labelRect(opt, idx)

        # Give a data item an explicit colour + checked + primary, then repaint
        # to cover the coloured-checkbox and primary-label branches.
        combo.setCheckedItems(["a"])
        data_row = combo._header_count
        item = model.item(data_row)
        item.setData(QBrush(QColor(255, 0, 0)), Qt.ForegroundRole)
        combo._set_primary_by_name("a")
        idx = model.index(data_row, 0)
        opt = QStyleOptionViewItem()
        opt.rect = QRect(0, 0, 200, 25)
        delegate.paint(painter, opt, idx)

        # An unchecked coloured item covers the faded-checkbox branch.
        combo.setCheckedItems([])
        delegate.paint(painter, opt, idx)
    finally:
        painter.end()


def test_popout_window_mixin(qtbot):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QDockWidget, QMainWindow, QWidget

    from napari_phasors._utils import PopoutWindowMixin

    class MyPopoutWidget(PopoutWindowMixin, QWidget):
        _popout_title = "Test Popout"

    main_window = QMainWindow()
    qtbot.addWidget(main_window)

    dock = QDockWidget("Dock", main_window)
    main_window.addDockWidget(Qt.RightDockWidgetArea, dock)

    widget = MyPopoutWidget()
    dock.setWidget(widget)

    from qtpy.QtGui import QShowEvent

    widget.showEvent(QShowEvent())

    assert widget._floated is True

    # wait for singleShot
    qtbot.wait(50)

    # After popout, it should detach from the dock parent
    assert widget.parent() is None

    # fallback sets it to Qt.Window
    assert int(widget.windowFlags() & Qt.Window) == int(Qt.Window)
    assert widget.windowTitle() == "Test Popout"


def test_popout_window_mixin_with_napari_viewer(qtbot):
    from qtpy.QtWidgets import QMainWindow, QWidget

    from napari_phasors._utils import PopoutWindowMixin

    class MyPopoutWidget(PopoutWindowMixin, QWidget):
        pass

    widget = MyPopoutWidget()

    class MockWindow:
        def __init__(self):
            self._qt_window = QMainWindow()
            self.removed_widget = None

        def remove_dock_widget(self, w):
            self.removed_widget = w
            # normally this removes parent, but here we just mock it

    class MockViewer:
        def __init__(self):
            self.window = MockWindow()

    widget.viewer = MockViewer()

    widget._popout_to_window()

    assert widget.viewer.window.removed_widget is widget
    assert widget.parent() is widget.viewer.window._qt_window


def test_update_data_label_parameter(qtbot):
    """update_data stores the dataset under the given label (defaulting to
    'Layer'), which drives the statistics Name column."""
    widget = HistogramWidget(bins=8)
    qtbot.addWidget(widget)
    data = np.linspace(0.0, 1.0, 50)

    widget.update_data(data)
    assert list(widget._datasets.keys()) == ["Layer"]
    assert list(widget._counts_per_dataset.keys()) == ["Layer"]

    widget.update_data(data, label="Lifetime: my image")
    assert list(widget._datasets.keys()) == ["Lifetime: my image"]
    assert list(widget._counts_per_dataset.keys()) == ["Lifetime: my image"]
