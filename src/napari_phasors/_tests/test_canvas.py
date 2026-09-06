"""Unit tests for the native PhasorCanvasWidget and artists."""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as mplPath

from napari_phasors._canvas import (
    Histogram2D,
    Histogram2DArtist,
    InteractiveEllipseSelector,
    InteractiveLassoSelector,
    InteractiveRectangleSelector,
    PhasorCanvasWidget,
    Scatter,
    ScatterArtist,
    SelectionGeometry,
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
