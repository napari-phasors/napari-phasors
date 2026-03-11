import numpy as np

from napari_phasors._utils import (
    HistogramDockWidget,
    HistogramWidget,
    StatisticsDockWidget,
    StatisticsTableWidget,
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

    widget.clear()

    assert widget.counts is None
    assert widget.bin_edges is None
    assert widget.bin_centers is None
    assert widget._datasets == {}
    assert widget._raw_valid_data is None
    assert not widget._settings_button.isEnabled()
    assert not widget.save_png_button.isEnabled()


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
