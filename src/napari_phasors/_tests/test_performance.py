"""Performance benchmarks for napari-phasors.

This module measures execution times and call counts for the key
operations described in issue #225.  Run with:

    python -m pytest src/napari_phasors/_tests/test_performance.py -v -s

The ``-s`` flag is required so that the benchmark summary printed at the
end of every test is visible.
"""

import time
from unittest.mock import patch

from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors.plotter import PlotterWidget

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_layer(name=None, harmonic=None):
    """Return an Image layer with synthetic phasor metadata."""
    if harmonic is None:
        harmonic = [1, 2, 3]
    raw = make_raw_flim_data(
        time_constants=[0.1, 1, 2, 3, 4, 5, 10],
    )
    layer = make_intensity_layer_with_phasors(raw, harmonic=harmonic)
    if name is not None:
        layer.name = name
    return layer


def _select_layers(plotter, layer_names):
    """Programmatically select layers by name in the checkable combobox."""
    from qtpy.QtCore import Qt

    cb = plotter.image_layers_checkable_combobox
    cb.blockSignals(True)
    for i in range(cb.count()):
        item_text = cb.itemText(i)
        state = Qt.Checked if item_text in layer_names else Qt.Unchecked
        cb.setItemCheckState(i, state)
    cb.blockSignals(False)
    cb.selectionChanged.emit()


def _measure(func, *args, **kwargs):
    """Call *func* and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ---------------------------------------------------------------------------
# Benchmark: layer selection change (on_image_layer_changed end-to-end)
# ---------------------------------------------------------------------------


class TestLayerSelectionPerformance:
    """Measure time for layer-selection workflows."""

    def test_single_layer_selection(self, make_napari_viewer, qtbot):
        """Baseline: select 1 layer."""
        viewer = make_napari_viewer()
        plotter = PlotterWidget(viewer)
        qtbot.addWidget(plotter)

        layer = _create_layer(name="layer_0")
        viewer.add_layer(layer)

        _, elapsed = _measure(plotter.on_image_layer_changed)
        print(
            f"\n[BENCHMARK] single_layer_selection: "
            f"{elapsed * 1000:.1f} ms"
        )

    def test_three_layer_selection(self, make_napari_viewer, qtbot):
        """Select 3 layers and trigger on_image_layer_changed."""
        viewer = make_napari_viewer()
        plotter = PlotterWidget(viewer)
        qtbot.addWidget(plotter)

        for i in range(3):
            viewer.add_layer(_create_layer(name=f"layer_{i}"))

        names = {f"layer_{i}" for i in range(3)}
        _select_layers(plotter, names)

        _, elapsed = _measure(plotter.on_image_layer_changed)
        print(f"\n[BENCHMARK] three_layer_selection: {elapsed * 1000:.1f} ms")

    def test_five_layer_selection(self, make_napari_viewer, qtbot):
        """Select 5 layers and trigger on_image_layer_changed."""
        viewer = make_napari_viewer()
        plotter = PlotterWidget(viewer)
        qtbot.addWidget(plotter)

        for i in range(5):
            viewer.add_layer(_create_layer(name=f"layer_{i}"))

        names = {f"layer_{i}" for i in range(5)}
        _select_layers(plotter, names)

        _, elapsed = _measure(plotter.on_image_layer_changed)
        print(f"\n[BENCHMARK] five_layer_selection: {elapsed * 1000:.1f} ms")


# ---------------------------------------------------------------------------
# Benchmark: tab switch
# ---------------------------------------------------------------------------


class TestTabSwitchPerformance:
    """Measure time for switching between tabs."""

    def test_tab_switch_with_data(self, make_napari_viewer, qtbot):
        """Switch to each tab after loading a layer."""
        viewer = make_napari_viewer()
        plotter = PlotterWidget(viewer)
        qtbot.addWidget(plotter)

        viewer.add_layer(_create_layer(name="layer_0"))

        results = {}
        for i in range(plotter.tab_widget.count()):
            tab_name = plotter.tab_widget.tabText(i)
            _, elapsed = _measure(plotter._on_tab_changed, i)
            results[tab_name] = elapsed

        for name, elapsed in results.items():
            print(
                f"\n[BENCHMARK] tab_switch({name}): "
                f"{elapsed * 1000:.1f} ms"
            )


# ---------------------------------------------------------------------------
# Benchmark: plot() duration
# ---------------------------------------------------------------------------


class TestPlotPerformance:
    """Measure time for a single plot() invocation."""

    def test_plot_single_layer(self, make_napari_viewer, qtbot):
        """Time plot() with 1 selected layer."""
        viewer = make_napari_viewer()
        plotter = PlotterWidget(viewer)
        qtbot.addWidget(plotter)

        viewer.add_layer(_create_layer(name="layer_0"))
        # Warm up: ensure initial plot happened
        plotter.on_image_layer_changed()

        _, elapsed = _measure(plotter.plot)
        print(f"\n[BENCHMARK] plot_single_layer: {elapsed * 1000:.1f} ms")

    def test_plot_three_layers(self, make_napari_viewer, qtbot):
        """Time plot() with 3 selected layers."""
        viewer = make_napari_viewer()
        plotter = PlotterWidget(viewer)
        qtbot.addWidget(plotter)

        for i in range(3):
            viewer.add_layer(_create_layer(name=f"layer_{i}"))

        names = {f"layer_{i}" for i in range(3)}
        _select_layers(plotter, names)
        plotter.on_image_layer_changed()

        _, elapsed = _measure(plotter.plot)
        print(f"\n[BENCHMARK] plot_three_layers: {elapsed * 1000:.1f} ms")


# ---------------------------------------------------------------------------
# Benchmark: visual parameter change propagation
# ---------------------------------------------------------------------------


class TestVisualParamPerformance:
    """Measure time for visual-only parameter changes."""

    def _setup_plotter(self, make_napari_viewer, qtbot):
        """Create a plotter with one layer loaded."""
        viewer = make_napari_viewer()
        plotter = PlotterWidget(viewer)
        qtbot.addWidget(plotter)

        viewer.add_layer(_create_layer(name="layer_0"))
        plotter.on_image_layer_changed()
        return plotter

    def test_colormap_change(self, make_napari_viewer, qtbot):
        plotter = self._setup_plotter(make_napari_viewer, qtbot)
        _, elapsed = _measure(
            setattr, plotter, 'histogram_colormap', 'viridis'
        )
        print(f"\n[BENCHMARK] colormap_change: {elapsed * 1000:.1f} ms")

    def test_bins_change(self, make_napari_viewer, qtbot):
        plotter = self._setup_plotter(make_napari_viewer, qtbot)
        _, elapsed = _measure(setattr, plotter, 'histogram_bins', 128)
        print(f"\n[BENCHMARK] bins_change: {elapsed * 1000:.1f} ms")

    def test_log_scale_toggle(self, make_napari_viewer, qtbot):
        plotter = self._setup_plotter(make_napari_viewer, qtbot)
        _, elapsed = _measure(setattr, plotter, 'log_scale', True)
        print(f"\n[BENCHMARK] log_scale_toggle: {elapsed * 1000:.1f} ms")


# ---------------------------------------------------------------------------
# Call-count audit
# ---------------------------------------------------------------------------


class TestCallCountAudit:
    """Count how many times key methods fire per user action."""

    def test_call_counts_on_add_layer(self, make_napari_viewer, qtbot):
        """Adding a single layer: count plot/get_features calls."""
        viewer = make_napari_viewer()
        plotter = PlotterWidget(viewer)
        qtbot.addWidget(plotter)

        with (
            patch.object(
                plotter,
                'plot',
                wraps=plotter.plot,
            ) as mock_plot,
            patch.object(
                plotter,
                'get_features',
                wraps=plotter.get_features,
            ) as mock_features,
        ):
            viewer.add_layer(_create_layer(name="layer_0"))
            # Let debounce timers fire
            qtbot.wait(500)

            print(
                f"\n[BENCHMARK] add_layer call counts: "
                f"plot={mock_plot.call_count}, "
                f"get_features={mock_features.call_count}"
            )

    def test_call_counts_on_select_three_layers(
        self, make_napari_viewer, qtbot
    ):
        """Selecting 3 layers: count calls."""
        viewer = make_napari_viewer()
        plotter = PlotterWidget(viewer)
        qtbot.addWidget(plotter)

        for i in range(3):
            viewer.add_layer(_create_layer(name=f"layer_{i}"))
        qtbot.wait(500)

        with (
            patch.object(
                plotter,
                'plot',
                wraps=plotter.plot,
            ) as mock_plot,
            patch.object(
                plotter,
                'get_features',
                wraps=plotter.get_features,
            ) as mock_features,
        ):
            names = {f"layer_{i}" for i in range(3)}
            _select_layers(plotter, names)
            # Let debounce timers fire
            qtbot.wait(500)

            print(
                f"\n[BENCHMARK] select_3_layers call counts: "
                f"plot={mock_plot.call_count}, "
                f"get_features={mock_features.call_count}"
            )

    def test_tab_update_call_counts_on_layer_change(
        self, make_napari_viewer, qtbot
    ):
        """Count how many tab._on_image_layer_changed() calls fire."""
        viewer = make_napari_viewer()
        plotter = PlotterWidget(viewer)
        qtbot.addWidget(plotter)

        viewer.add_layer(_create_layer(name="layer_0"))
        qtbot.wait(500)

        tab_names = [
            'filter_tab',
            'calibration_tab',
            'selection_tab',
            'phasor_mapping_tab',
            'components_tab',
            'fret_tab',
        ]

        mock_ctx_managers = []
        for name in tab_names:
            tab = getattr(plotter, name, None)
            if tab is not None and hasattr(tab, '_on_image_layer_changed'):
                p = patch.object(
                    tab,
                    '_on_image_layer_changed',
                    wraps=tab._on_image_layer_changed,
                )
                mock_ctx_managers.append((name, p))

        # Enter all patches
        mocks = {}
        for name, p in mock_ctx_managers:
            mocks[name] = p.start()

        try:
            plotter.on_image_layer_changed()
            qtbot.wait(200)

            counts = {name: mocks[name].call_count for name in mocks}
            total = sum(counts.values())
            print(
                f"\n[BENCHMARK] tab_update_calls on layer_change: "
                f"total={total}, per_tab={counts}"
            )
        finally:
            for _name, p in mock_ctx_managers:
                p.stop()
