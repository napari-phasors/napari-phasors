"""Unit tests for the native PhasorCanvasWidget and artists."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.path import Path as mplPath
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPainter

from napari_phasors._canvas import (
    TOOLBAR_ICON_MARGIN,
    Contour,
    Histogram2D,
    Histogram2DArtist,
    InteractiveEllipseSelector,
    InteractiveLassoSelector,
    InteractiveRectangleSelector,
    PhasorCanvasWidget,
    Scatter,
    ScatterArtist,
    SelectionGeometry,
    _inset_icon_image,
    _opaque_bounds,
    load_toolbar_icon,
)


def test_selection_geometry_rectangle():
    geom = SelectionGeometry("rectangle", (0.2, 0.8, 0.1, 0.7))
    pts = np.array(
        [
            [0.5, 0.5],  # Inside
            [0.1, 0.5],  # Outside (x low)
            [0.9, 0.5],  # Outside (x high)
            [0.5, 0.0],  # Outside (y low)
            [0.5, 0.8],  # Outside (y high)
        ]
    )
    mask = geom.contains_points(pts)
    assert np.array_equal(mask, [True, False, False, False, False])
    assert len(geom.contains_points(np.empty((0, 2)))) == 0


def test_selection_geometry_ellipse():
    geom = SelectionGeometry("ellipse", (0.5, 0.5, 0.2, 0.2))  # circle r=0.2
    pts = np.array(
        [
            [0.5, 0.5],  # Center (inside)
            [0.6, 0.5],  # Inside (dist 0.1)
            [0.8, 0.5],  # Outside (dist 0.3)
        ]
    )
    mask = geom.contains_points(pts)
    assert np.array_equal(mask, [True, True, False])

    # Degenerate ellipse
    geom_zero = SelectionGeometry("ellipse", (0.5, 0.5, 0.0, 0.0))
    assert not np.any(geom_zero.contains_points(pts))


def test_selection_geometry_lasso():
    # Square polygon [0,0] to [1,1]
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    geom = SelectionGeometry("lasso", mplPath(vertices))
    pts = np.array(
        [
            [0.5, 0.5],  # Inside
            [1.5, 0.5],  # Outside
        ]
    )
    mask = geom.contains_points(pts)
    assert np.array_equal(mask, [True, False])

    geom_bad = SelectionGeometry("unknown", None)
    assert not geom_bad.contains_points(pts).any()


def test_canvas_initialization(make_viewer_model):
    viewer = make_viewer_model()
    canvas_widget = PhasorCanvasWidget(viewer)

    assert hasattr(canvas_widget, "axes")
    assert hasattr(canvas_widget, "figure")
    assert hasattr(canvas_widget, "canvas")
    assert hasattr(canvas_widget, "toolbar")
    assert hasattr(canvas_widget, "selection_toolbar")
    assert hasattr(canvas_widget, "class_spinbox")
    assert canvas_widget.class_spinbox.value == 1

    assert "HISTOGRAM2D" in canvas_widget.artists
    assert "SCATTER" in canvas_widget.artists
    assert canvas_widget.active_artist == "HISTOGRAM2D"
    assert canvas_widget.active_selector is None

    assert "RECTANGLE" in canvas_widget.selectors
    assert "ELLIPSE" in canvas_widget.selectors
    assert "LASSO" in canvas_widget.selectors

    assert canvas_widget._is_click_inside_axes(None) is False


def test_histogram_artist(make_viewer_model):
    viewer = make_viewer_model()
    canvas_widget = PhasorCanvasWidget(viewer)
    hist_artist: Histogram2DArtist = canvas_widget.artists["HISTOGRAM2D"]

    # Assign data
    data = np.random.rand(200, 2)
    hist_artist.data = data
    assert hist_artist.histogram is not None
    counts, xedges, yedges = hist_artist.histogram
    assert counts.shape == (100, 100)

    # Change bins
    hist_artist.bins = (30, 30)
    counts, _, _ = hist_artist.histogram
    assert counts.shape == (30, 30)

    # Change colormap and norm
    hist_artist.histogram_colormap = "viridis"
    hist_artist.histogram_color_normalization_method = "log"
    assert hist_artist.histogram_color_normalization_method == "log"

    # Color indices / overlay
    indices = np.zeros(200, dtype=np.uint32)
    indices[:50] = 1
    indices[50:100] = 2
    hist_artist.color_indices = indices
    assert hist_artist._mpl_artists["overlay_histogram_image"] is not None

    # Fixed range for timelapse
    hist_artist._napari_phasors_fixed_counts_range = (1.0, 50.0)
    assert hist_artist._fixed_counts_range == (1.0, 50.0)

    # Visibility
    hist_artist.visible = False
    assert hist_artist.visible is False
    assert hist_artist._mpl_artists["histogram_image"].get_visible() is False

    hist_artist.visible = True
    assert hist_artist.visible is True


def test_scatter_artist(make_viewer_model):
    viewer = make_viewer_model()
    canvas_widget = PhasorCanvasWidget(viewer)
    scatter_artist: ScatterArtist = canvas_widget.artists["SCATTER"]

    data = np.random.rand(100, 2)
    scatter_artist.data = data
    scatter_artist.size = 42.0
    scatter_artist.alpha = 0.75

    assert scatter_artist.size == 42.0
    assert scatter_artist.alpha == 0.75

    # New data should not reset size and alpha
    new_data = np.random.rand(80, 2)
    scatter_artist.data = new_data
    assert scatter_artist.size == 42.0
    assert scatter_artist.alpha == 0.75

    # Color indices
    indices = np.zeros(80, dtype=np.uint32)
    indices[:20] = 1
    scatter_artist.color_indices = indices
    assert scatter_artist.color_indices is not None

    # Visibility
    scatter_artist.visible = False
    assert scatter_artist.visible is False

    # Color property and no outline
    scatter_artist.visible = True
    scatter_artist.color_indices = None
    mpl_sc = scatter_artist._mpl_artists["scatter"]
    assert len(mpl_sc.get_edgecolors()) == 0
    assert mpl_sc.get_linewidths()[0] == 0

    scatter_artist.color = "#ff0000"
    assert scatter_artist.color == "#ff0000"
    facecolors = mpl_sc.get_facecolors()
    np.testing.assert_allclose(facecolors[0][:3], [1.0, 0.0, 0.0], atol=1e-3)
    assert len(mpl_sc.get_edgecolors()) == 0
    assert mpl_sc.get_linewidths()[0] == 0


def test_selector_activation_and_geometry(make_viewer_model):
    viewer = make_viewer_model()
    canvas_widget = PhasorCanvasWidget(viewer)

    # Add data to active artist
    data = np.array([[0.2, 0.2], [0.8, 0.8], [0.5, 0.5]])
    canvas_widget.artists["HISTOGRAM2D"].data = data

    # Activate Rectangle
    canvas_widget.selection_toolbar.buttons["RECTANGLE"].click()
    assert canvas_widget.active_selector is not None
    assert canvas_widget.selection_toolbar.buttons["RECTANGLE"].isChecked()
    assert not canvas_widget.selection_toolbar.buttons["LASSO"].isChecked()

    # Simulate Rectangle selection
    rect_selector = canvas_widget.selectors["RECTANGLE"]

    class MockEvent:
        def __init__(self, x, y):
            self.xdata = x
            self.ydata = y

    rect_selector.on_select(MockEvent(0.0, 0.0), MockEvent(0.6, 0.6))
    assert rect_selector.last_geometry is not None

    region_predicate = canvas_widget.active_selection_region()
    assert region_predicate is not None
    mask = region_predicate(data)
    # [0.2, 0.2] and [0.5, 0.5] inside, [0.8, 0.8] outside
    assert np.array_equal(mask, [True, False, True])

    # Activate Lasso (should deactivate Rectangle)
    canvas_widget.selection_toolbar.buttons["LASSO"].click()
    assert canvas_widget.selection_toolbar.buttons["LASSO"].isChecked()
    assert not canvas_widget.selection_toolbar.buttons["RECTANGLE"].isChecked()


def test_navigation_deactivates_selector(make_viewer_model):
    viewer = make_viewer_model()
    canvas_widget = PhasorCanvasWidget(viewer)

    # Activate selector
    canvas_widget.selection_toolbar.buttons["ELLIPSE"].click()
    assert canvas_widget.active_selector is not None
    assert canvas_widget.selection_toolbar.buttons["ELLIPSE"].isChecked()

    # Pan activation deactivates selector
    canvas_widget.toolbar.pan()
    assert canvas_widget.active_selector is None
    assert not canvas_widget.selection_toolbar.buttons["ELLIPSE"].isChecked()

    # Activate again and test zoom
    canvas_widget.selection_toolbar.buttons["RECTANGLE"].click()
    assert canvas_widget.active_selector is not None
    assert canvas_widget.selection_toolbar.buttons["RECTANGLE"].isChecked()

    # Zoom activation deactivates selector
    canvas_widget.toolbar.zoom()
    assert canvas_widget.active_selector is None
    assert not canvas_widget.selection_toolbar.buttons["RECTANGLE"].isChecked()

    # Activate again and test escape
    canvas_widget.selection_toolbar.buttons["LASSO"].click()
    assert canvas_widget.active_selector is not None
    canvas_widget._on_escape()
    assert canvas_widget.active_selector is None
    assert not canvas_widget.selection_toolbar.buttons["LASSO"].isChecked()


def test_selectors_base():
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.add_subplot(111)
    canvas = PhasorCanvasWidget(None)

    sel = InteractiveRectangleSelector(ax, canvas)
    sel.data = np.array([[0, 0], [1, 1]])
    assert (sel.data == np.array([[0, 0], [1, 1]])).all()
    sel.class_value = 2
    assert sel.class_value == 2
    sel.selected_indices = np.array([1])
    assert (sel.selected_indices == np.array([1])).all()
    assert sel.last_geometry is None


def test_apply_selection():
    canvas = PhasorCanvasWidget(None)
    artist = Histogram2D(canvas.axes)
    artist.data = np.array([[0, 0], [1, 1], [2, 2]])
    canvas.artists = {"HISTOGRAM2D": artist}
    canvas.active_artist = "HISTOGRAM2D"

    sel = InteractiveRectangleSelector(canvas.axes, canvas)
    sel.data = artist.data
    sel.class_value = 5
    sel.selected_indices = np.array([0, 2])
    sel.apply_selection()

    assert artist.color_indices[0] == 5
    assert artist.color_indices[1] == 0
    assert artist.color_indices[2] == 5


def test_ellipse_and_lasso_selectors():
    canvas = PhasorCanvasWidget(None)
    artist = Scatter(canvas.axes)
    artist.data = np.array([[0, 0], [1, 1]])
    canvas.artists = {"SCATTER": artist}
    canvas.active_artist = "SCATTER"

    el = InteractiveEllipseSelector(canvas.axes, canvas)
    el.create_selector()

    # Mock event objects since Matplotlib selectors expect them
    class MockEvent:
        def __init__(self, x, y):
            self.xdata = x
            self.ydata = y

    el.on_select(MockEvent(0, 0), MockEvent(1, 1))

    la = InteractiveLassoSelector(canvas.axes, canvas)
    la.create_selector()
    la.on_select([(0, 0), (0, 1), (1, 1)])


def test_toolbar_theme_icons():
    canvas = PhasorCanvasWidget(None)
    # By default, in headless / dark theme, icons should point to white
    assert "white" in str(canvas.toolbar._get_path_to_icon())

    # Check that toolbar actions have valid non-null icons
    for action in canvas.toolbar.actions():
        if action.text():
            assert not action.icon().isNull()
            img = action.icon().pixmap(28, 28).toImage()
            # Verify the icons are light (white)
            colors = {
                img.pixelColor(x, y).name()
                for x in range(img.width())
                for y in range(img.height())
                if img.pixelColor(x, y).alpha() > 200
            }
            assert "#ffffff" in colors

    # Test toggling Pan updates icon
    canvas.toolbar.pan()
    assert canvas.toolbar._actions["pan"].isChecked()
    canvas.toolbar.pan()
    assert not canvas.toolbar._actions["pan"].isChecked()

    # Test toggling Zoom updates icon
    canvas.toolbar.zoom()
    assert canvas.toolbar._actions["zoom"].isChecked()
    canvas.toolbar.zoom()
    assert not canvas.toolbar._actions["zoom"].isChecked()

    # Test theme change callback
    canvas._on_napari_theme_changed()


def test_apply_selection_when_indices_already_array():
    """Test apply_selection when color_indices is already an array and empty cases."""
    canvas = PhasorCanvasWidget(None)
    artist = Histogram2D(canvas.axes)
    artist.data = np.array([[0, 0], [1, 1], [2, 2]])
    # Pre-set color_indices as array
    artist.color_indices = np.array([1, 1, 1], dtype=np.uint32)
    canvas.artists = {"HISTOGRAM2D": artist}
    canvas.active_artist = "HISTOGRAM2D"

    sel = InteractiveRectangleSelector(canvas.axes, canvas)
    sel.data = artist.data
    sel.class_value = 7
    # Case 1: _selected_indices is None or empty
    sel._selected_indices = None
    sel.apply_selection()
    assert np.array_equal(artist.color_indices, [1, 1, 1])

    sel._selected_indices = np.array([], dtype=int)
    sel.apply_selection()
    assert np.array_equal(artist.color_indices, [1, 1, 1])

    # Case 2: color_indices is already an array and valid selection is applied
    sel._selected_indices = np.array([0, 2])
    sel.apply_selection()
    assert artist.color_indices[0] == 7
    assert artist.color_indices[1] == 1
    assert artist.color_indices[2] == 7


def test_selectors_edge_cases():
    """Test edge cases for rectangle, ellipse, and lasso selectors."""
    canvas = PhasorCanvasWidget(None)
    artist = Histogram2D(canvas.axes)
    artist.data = np.array([[0.5, 0.5]])
    canvas.artists = {"HISTOGRAM2D": artist}
    canvas.active_artist = "HISTOGRAM2D"

    class MockEvent:
        def __init__(self, x=None, y=None, button=1, inaxes=None):
            self.xdata = x
            self.ydata = y
            self.button = button
            self.inaxes = inaxes

    # Rectangle selector edge cases
    rect = InteractiveRectangleSelector(canvas.axes, canvas)
    assert rect.on_select(MockEvent(None, 0.5), MockEvent(0.5, 0.5)) is None
    rect.data = np.empty((0, 2))
    assert rect.on_select(MockEvent(0.1, 0.1), MockEvent(0.9, 0.9)) is None

    # Rectangle button 3 triggers apply_selection
    rect.data = artist.data
    rect._selected_indices = np.array([0])
    rect.class_value = 3
    rect._on_button_press(MockEvent(button=3))
    assert artist.color_indices[0] == 3

    # Ellipse selector edge cases
    ellipse = InteractiveEllipseSelector(canvas.axes, canvas)
    assert ellipse.on_select(MockEvent(None, 0.5), MockEvent(0.5, 0.5)) is None
    ellipse.data = np.empty((0, 2))
    assert ellipse.on_select(MockEvent(0.1, 0.1), MockEvent(0.9, 0.9)) is None

    # Ellipse button 3 triggers apply_selection
    ellipse.data = artist.data
    ellipse._selected_indices = np.array([0])
    ellipse.class_value = 4
    ellipse._on_button_press(MockEvent(button=3))
    assert artist.color_indices[0] == 4

    # Lasso selector edge cases
    lasso = InteractiveLassoSelector(canvas.axes, canvas)
    assert lasso.on_select(None) is None
    assert lasso.on_select([(0, 0), (1, 1)]) is None  # len < 3
    lasso.data = np.empty((0, 2))
    assert lasso.on_select([(0, 0), (1, 0), (1, 1), (0, 0)]) is None


def test_histogram_artist_edge_cases():
    """Test edge cases and uncovered getters/branches in Histogram2D."""
    canvas = PhasorCanvasWidget(None)
    hist = Histogram2D(canvas.axes)

    # __eq__ comparisons
    assert hist == "HISTOGRAM2D"
    assert hist == "histogram_2d"
    assert hist != "SCATTER"
    assert hist != 123

    # Getters and setters
    assert hist.bins == 100
    assert hist.cmin == 1
    hist.cmin = 2
    assert hist.cmin == 2
    assert hist.histogram_colormap is not None
    assert hist.overlay_colormap is not None
    assert hist._napari_phasors_fixed_counts_range is None
    assert hist.overlay_visible is True

    # overlay_visible setter with overlay present
    hist.data = np.array([[0.2, 0.2], [0.8, 0.8]])
    hist.color_indices = np.array([1, 2])
    hist.overlay_visible = False
    assert hist.overlay_visible is False
    hist.overlay_visible = True
    assert hist.overlay_visible is True

    # _refresh with color_indices present
    hist._refresh()

    # Mismatched length with overlay present
    hist.color_indices = np.array([1, 2, 3])
    assert "overlay_histogram_image" not in hist._mpl_artists

    # Scalar non-zero color_indices
    hist.color_indices = 3
    assert "overlay_histogram_image" in hist._mpl_artists

    # Scalar zero with overlay present
    hist.color_indices = np.array([1, 2])
    hist.color_indices = 0
    assert "overlay_histogram_image" not in hist._mpl_artists

    # All zeros with overlay present
    hist.color_indices = np.array([1, 2])
    hist.color_indices = np.array([0, 0])
    assert "overlay_histogram_image" not in hist._mpl_artists

    # None with overlay present
    hist.color_indices = np.array([1, 2])
    hist.color_indices = None
    assert "overlay_histogram_image" not in hist._mpl_artists

    # LinearSegmentedColormap overlay colormap
    import matplotlib.colors as mcolors

    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_test", ["black", "yellow"]
    )
    hist.overlay_colormap = custom_cmap
    hist.color_indices = np.array([1, 2])
    assert "overlay_histogram_image" in hist._mpl_artists

    # _get_normalization branches
    norm_overlay = hist._get_normalization(is_overlay=True)
    assert norm_overlay.vmin == 0 and norm_overlay.vmax == 1

    fresh_hist = Histogram2D(canvas.axes)
    norm_empty = fresh_hist._get_normalization(values=None)
    assert norm_empty.vmin == 0 and norm_empty.vmax == 1

    hist._napari_phasors_fixed_counts_range = (5.0, 50.0)
    assert hist._napari_phasors_fixed_counts_range == (5.0, 50.0)
    norm_fixed = hist._get_normalization(values=np.array([[1, 2], [3, 4]]))
    assert norm_fixed.vmin == 5.0 and norm_fixed.vmax == 50.0
    hist._napari_phasors_fixed_counts_range = None

    # _refresh with empty data
    hist.data = None
    assert hist.data is None
    hist._refresh()
    assert "histogram_image" not in hist._mpl_artists

    # Points outside histogram extent with overlay present
    hist.data = np.array([[0.2, 0.2], [0.8, 0.8]])
    hist.color_indices = np.array([1, 2])
    assert "overlay_histogram_image" in hist._mpl_artists
    hist._histogram = (
        np.zeros((2, 2)),
        np.array([10.0, 11.0, 12.0]),
        np.array([10.0, 11.0, 12.0]),
    )
    hist._colorize(np.array([1, 2]))
    assert "overlay_histogram_image" not in hist._mpl_artists


def test_scatter_artist_edge_cases():
    """Test edge cases and uncovered branches in Scatter."""
    import matplotlib.colors as mcolors

    canvas = PhasorCanvasWidget(None)
    scatter = Scatter(canvas.axes)

    # __eq__ comparisons
    assert scatter == "SCATTER"
    assert scatter == "scatter"
    assert scatter != "HISTOGRAM2D"
    assert scatter != 456

    # overlay_colormap getter and setter
    assert scatter.overlay_colormap is not None
    scatter.overlay_colormap = "viridis"
    assert scatter.overlay_colormap == "viridis"

    # _refresh with empty data
    scatter.data = None
    scatter._refresh()
    assert "scatter" not in scatter._mpl_artists

    # Add data
    scatter.data = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert "scatter" in scatter._mpl_artists

    # _refresh with color_indices
    scatter.color_indices = np.array([1, 2])
    scatter._refresh()

    # Scalar zero
    scatter.color_indices = 0

    # Scalar non-zero
    scatter.color_indices = 2

    # None
    scatter.color_indices = None

    # Mismatched length
    scatter.color_indices = np.array([1, 2, 3, 4])

    # LinearSegmentedColormap
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_scatter", ["black", "cyan"]
    )
    scatter.overlay_colormap = custom_cmap
    scatter.color_indices = np.array([1, 2])


def test_canvas_widget_methods_and_theme_handling():
    """Test PhasorCanvasWidget artist/selector setters and theme fallbacks."""
    canvas = PhasorCanvasWidget(None)

    # active_artist setter with None and NONE
    canvas.active_artist = None
    assert canvas.active_artist is None
    canvas.active_artist = "NONE"
    assert canvas.active_artist is None

    # Set active_artist by passing artist instance
    hist_obj = canvas.artists["HISTOGRAM2D"]
    canvas.active_artist = hist_obj
    assert canvas.active_artist == "HISTOGRAM2D"

    # Set active_artist with unknown name
    canvas.active_artist = "NONEXISTENT_ARTIST"
    assert canvas.active_artist is None

    # active_selector setter with None
    canvas.active_selector = "RECTANGLE"
    assert canvas.active_selector is not None
    canvas.active_selector = None
    assert canvas.active_selector is None

    # active_selector setter with selector object
    rect_obj = canvas.selectors["RECTANGLE"]
    canvas.active_selector = rect_obj
    assert canvas.active_selector is not None

    # active_selector with unknown selector
    canvas.active_selector = "NONEXISTENT_SELECTOR"
    assert canvas.active_selector is None

    # _on_selector_button_clicked when button unchecked
    canvas._on_selector_button_clicked("RECTANGLE")
    assert canvas.active_selector is None

    # active_selection_region when no active selector or geometry
    assert canvas.active_selection_region() is None

    # _is_click_inside_axes with other axes
    class MockAxesEvent:
        def __init__(self, inaxes, x=0.5, y=0.5):
            self.inaxes = inaxes
            self.xdata = x
            self.ydata = y

    assert (
        canvas._is_click_inside_axes(MockAxesEvent(inaxes="other_axes"))
        is False
    )
    assert (
        canvas._is_click_inside_axes(MockAxesEvent(inaxes=canvas.axes)) is True
    )

    # _apply_theme fallback exception branch
    class BadViewer:
        theme = "nonexistent_theme_error"

    canvas.viewer = BadViewer()
    canvas._apply_theme()  # Exercises exception handler

    # Test light theme detection in toolbar
    class MockTheme:
        type = "light"

    class LightViewer:
        theme = "light"

    canvas.viewer = LightViewer()
    with patch("napari.utils.theme.get_theme", return_value=MockTheme()):
        assert canvas.toolbar._napari_theme_has_light_bg() is True
        assert "black" in str(canvas.toolbar._get_path_to_icon())

    # Test toolbar exception branch in _napari_theme_has_light_bg
    with patch(
        "napari.utils.theme.get_theme", side_effect=RuntimeError("theme error")
    ):
        assert canvas.toolbar._napari_theme_has_light_bg() is False


def test_contour_artist(make_viewer_model):
    """Test Contour artist initialization, properties, setters, and grouped mode."""
    viewer = make_viewer_model()
    canvas = PhasorCanvasWidget(viewer)

    assert "CONTOUR" in canvas.artists
    contour = canvas.artists["CONTOUR"]
    assert isinstance(contour, Contour)
    assert contour == "CONTOUR"
    assert contour == "contour_plot"
    assert contour.color_indices is None
    contour.color_indices = [1, 2]  # Should be no-op
    assert contour.overlay_colormap is None
    contour.overlay_colormap = "jet"  # Should be no-op

    assert contour.bins == 150
    assert contour.levels == 5
    assert contour.linewidths == 1.5
    assert contour.colormap == "jet"
    assert contour.log_norm is True
    assert contour.visible is True
    assert contour.histogram is None

    # Setting data
    rng = np.random.default_rng(42)
    pts = rng.normal(loc=0.5, scale=0.1, size=(200, 2))
    contour.data = pts
    assert contour.data is not None
    assert len(contour.data) == 200
    assert contour.histogram is not None
    assert len(contour._contour_collections) > 0

    # Test property setters triggering refresh
    contour.bins = 50
    assert contour.bins == 50
    assert contour.histogram[0].shape[0] == 50

    contour.bins = (40, 40)
    assert contour.bins == (40, 40)
    assert contour.histogram[0].shape == (40, 40)

    contour.levels = 3
    assert contour.levels == 3

    contour.linewidths = 2.0
    assert contour.linewidths == 2.0

    contour.colormap = "viridis"
    assert contour.colormap == "viridis"

    contour.log_norm = False
    assert contour.log_norm is False

    # Test visibility toggle
    contour.visible = False
    assert contour.visible is False
    for cs in contour._contour_collections:
        if hasattr(cs, "collections"):
            for col in cs.collections:
                assert col.get_visible() is False

    contour.visible = True
    assert contour.visible is True

    # Test grouped mode rendering on pooled grid
    grp1 = rng.normal(loc=0.3, scale=0.05, size=(100, 2))
    grp2 = rng.normal(loc=0.7, scale=0.05, size=(100, 2))
    grouped_data = {1: (grp1[:, 0], grp1[:, 1]), 2: (grp2[:, 0], grp2[:, 1])}
    styles = {
        1: {"mode": "colormap", "colormap": "magma"},
        2: {"mode": "solid", "color": (0.0, 1.0, 0.0)},
    }
    names = {1: "Cluster A", 2: "Cluster B"}

    contour.set_grouped_data(
        grouped_dict=grouped_data,
        styles_dict=styles,
        group_names=names,
        show_legend=True,
    )
    assert len(contour._contour_collections) == 2
    assert contour.histogram is not None
    assert canvas.axes.get_legend() is not None

    # Test active_artist switching
    canvas.active_artist = "CONTOUR"
    assert canvas.active_artist == "CONTOUR"
    assert contour.visible is True
    assert canvas.artists["HISTOGRAM2D"].visible is False
    assert canvas.artists["SCATTER"].visible is False

    canvas.active_artist = "HISTOGRAM2D"
    assert canvas.active_artist == "HISTOGRAM2D"
    assert contour.visible is False
    assert canvas.artists["HISTOGRAM2D"].visible is True

    contour._remove_artists()
    assert len(contour._contour_collections) == 0
    assert contour.histogram is None
    assert canvas.axes.get_legend() is None


def test_contour_artist_edge_cases(make_viewer_model):
    """Test edge cases and branches in Contour artist."""
    viewer = make_viewer_model()
    canvas = PhasorCanvasWidget(viewer)
    contour = canvas.artists["CONTOUR"]

    # __eq__ with non-string
    assert (contour == 123) is False
    assert (contour == contour) is True

    # setting data to None and empty array
    contour.data = None
    assert contour.data is None
    assert len(contour._contour_collections) == 0

    contour.data = np.empty((0, 2))
    assert len(contour._contour_collections) == 0

    # Custom bins with ndarray edges
    rng = np.random.default_rng(42)
    pts = rng.normal(loc=0.5, scale=0.1, size=(50, 2))
    contour.bins = (np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    contour.data = pts
    assert contour.histogram is not None

    # Aspect <= 1 in _compute_grid
    canvas.axes.set_xlim(0, 1)
    canvas.axes.set_ylim(0, 5)
    contour.bins = 30
    contour._refresh()

    # Explicit levels sequence
    contour.levels = [2.0, 5.0, 10.0]
    contour._refresh()
    assert list(contour.levels) == [2.0, 5.0, 10.0]

    # _compute_clean_levels edge cases (vmax <= 0, vmax <= 1)
    contour.levels = 5
    assert contour._compute_clean_levels(np.zeros((10, 10))) is None
    low_counts = np.zeros((10, 10))
    low_counts[0, 0] = 1.0
    levs = contour._compute_clean_levels(low_counts)
    assert np.array_equal(levs, np.array([1.0]))

    # _refresh when data is None or levels is None
    contour._data = None
    contour._refresh()
    contour._data = np.zeros(
        (5, 2)
    )  # all same points, clean levels will be None
    contour.levels = 5
    contour._refresh()

    # set_grouped_data edge cases
    # 1. empty dict
    contour.set_grouped_data({})
    assert contour._grouped_data is None

    # 2. all empty groups
    contour.set_grouped_data({1: (np.array([]), np.array([]))})
    assert len(contour._contour_collections) == 0

    # 3. group where levels is None, colormap fallback, solid fallback
    pts1 = rng.normal(loc=0.5, scale=0.1, size=(50, 2))
    grouped = {
        1: (pts1[:, 0], pts1[:, 1]),
        2: (np.array([0.5]), np.array([0.5])),
    }
    styles = {
        1: {"mode": "colormap", "colormap": "nonexistent_cmap_xyz"},
        2: {"mode": "solid"},
    }
    contour.set_grouped_data(
        grouped_dict=grouped,
        styles_dict=styles,
        show_legend=True,
    )
    assert len(contour._contour_collections) >= 1


def _icon_paths():
    """Every packaged toolbar icon, both themes."""
    root = Path(__file__).parent.parent / "icons"
    return sorted(root.glob("*/*.png"))


def _clearance(image):
    """Smallest empty border, in pixels, between the glyph and the edges."""
    bounds = _opaque_bounds(image)
    assert bounds is not None
    left, top, right, bottom = bounds
    return min(
        left, top, image.width() - 1 - right, image.height() - 1 - bottom
    )


def _solid_image(rect, size=48):
    """Return a ``size``x``size`` ARGB image with one opaque white ``rect``."""
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.fillRect(rect, Qt.GlobalColor.white)
    painter.end()
    return image


def test_opaque_bounds_finds_the_glyph_box():
    """The bounds are the inclusive box of the non-transparent pixels."""
    from qtpy.QtCore import QRect

    image = _solid_image(QRect(4, 6, 10, 20))
    assert _opaque_bounds(image) == (4, 6, 13, 25)


def test_opaque_bounds_of_a_blank_image_is_none():
    """A fully transparent image has no glyph to bound."""
    image = QImage(8, 8, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    assert _opaque_bounds(image) is None


def test_inset_icon_image_leaves_a_clear_glyph_untouched():
    """An icon that already clears the margin is returned unchanged."""
    from qtpy.QtCore import QRect

    margin = TOOLBAR_ICON_MARGIN
    image = _solid_image(
        QRect(margin, margin, 48 - 2 * margin, 48 - 2 * margin)
    )
    assert _inset_icon_image(image) is image


def test_inset_icon_image_pulls_an_edge_to_edge_glyph_in():
    """A glyph touching the canvas edge is scaled down and re-centred."""
    from qtpy.QtCore import QRect

    image = _solid_image(QRect(0, 0, 48, 48))
    assert _clearance(image) == 0

    inset = _inset_icon_image(image)
    assert inset is not image
    assert inset.size() == image.size()
    assert _clearance(inset) >= TOOLBAR_ICON_MARGIN


def test_inset_icon_image_keeps_a_blank_image():
    """Nothing to inset when the image has no visible pixels."""
    image = QImage(8, 8, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    assert _inset_icon_image(image) is image


def test_inset_icon_image_skips_images_smaller_than_the_margin():
    """An image with no room for the margin is left alone."""
    from qtpy.QtCore import QRect

    image = _solid_image(QRect(0, 0, 4, 4), size=4)
    assert _inset_icon_image(image) is image


def test_load_toolbar_icon_returns_null_icon_for_a_missing_file(tmp_path):
    """A path that is not a readable image yields an empty icon."""
    missing = tmp_path / "not-an-icon.png"
    missing.write_text("not a png")
    assert load_toolbar_icon(missing).isNull()


@pytest.mark.parametrize("icon_path", _icon_paths(), ids=lambda p: p.stem)
def test_toolbar_icons_are_never_clipped(icon_path):
    """Every packaged glyph keeps a margin, so none reads as cut off.

    ``Pan`` and ``Zoom`` (the drag and zoom tools) were drawn edge to edge and
    were the visible symptom: their arrow tips and the magnifier crown landed
    on the icon-box boundary.
    """
    icon = load_toolbar_icon(icon_path)
    assert not icon.isNull()
    rendered = icon.pixmap(48, 48).toImage()
    rendered = rendered.convertToFormat(QImage.Format.Format_ARGB32)
    assert _clearance(rendered) >= TOOLBAR_ICON_MARGIN


def test_toolbar_icons_are_cached_per_path():
    """Repeated loads reuse the processed image instead of re-scanning it."""
    from napari_phasors import _canvas

    path = _icon_paths()[0]
    _canvas._TOOLBAR_ICON_CACHE.pop(str(path), None)
    load_toolbar_icon(path)
    assert str(path) in _canvas._TOOLBAR_ICON_CACHE

    with patch.object(_canvas, "_inset_icon_image") as inset:
        load_toolbar_icon(path)
    inset.assert_not_called()


def test_toolbar_actions_use_the_inset_icons():
    """The toolbar's Pan/Zoom actions, checked or not, keep their margin."""
    canvas = PhasorCanvasWidget(None)
    toolbar = canvas.toolbar

    for name in ("pan", "zoom"):
        action = toolbar._actions[name]
        for checked in (False, True):
            if action.isChecked() != checked:
                getattr(toolbar, name)()
            rendered = action.icon().pixmap(48, 48).toImage()
            rendered = rendered.convertToFormat(QImage.Format.Format_ARGB32)
            assert _clearance(rendered) >= TOOLBAR_ICON_MARGIN
        if action.isChecked():
            getattr(toolbar, name)()
