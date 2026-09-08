"""Native Matplotlib canvas and interactive selection tools for napari-phasors.

The architecture is adapted from `biaplotter` and `nap-plot-tools`, licensed
under BSD-3-Clause.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.colors import ListedColormap, LogNorm, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.path import Path as mplPath
from matplotlib.widgets import (
    EllipseSelector,
    LassoSelector,
    RectangleSelector,
)
from psygnal import Signal
from qtpy.QtCore import QPointF, QRectF, QSize, Qt
from qtpy.QtGui import (
    QColor,
    QCursor,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from qtpy.QtWidgets import (
    QHBoxLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari


def _build_default_overlay_colormaps() -> (
    tuple[ListedColormap, ListedColormap]
):
    """Create default categorical colormaps based on Matplotlib's tab10."""
    tab10_colors = [to_rgba(c) for c in plt.get_cmap("tab10").colors]
    opaque_cmap = ListedColormap(
        [to_rgba("#e6e6fa")] + tab10_colors, name="tab10_opaque"
    )

    trans_colors = [(0.0, 0.0, 0.0, 0.0)] + tab10_colors
    first_trans_cmap = ListedColormap(
        trans_colors, name="tab10_first_transparent"
    )
    return opaque_cmap, first_trans_cmap


default_overlay_cmap, default_overlay_cmap_first_transparent = (
    _build_default_overlay_colormaps()
)

# Backward-compatibility aliases
cat10_mod_cmap = default_overlay_cmap
cat10_mod_cmap_first_transparent = default_overlay_cmap_first_transparent

#: Default diameter, in screen pixels, of the brush and eraser tools.
DEFAULT_BRUSH_SIZE_PX = 14.0


def _capsule_mask(
    px: np.ndarray,
    py: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rx: float,
    ry: float,
) -> np.ndarray:
    """Return a mask of points within ``(rx, ry)`` of a segment.

    The stroke a brush leaves between two mouse positions is the set of
    points closer than the brush radius to the segment joining them.
    Distances are measured in units of the radii, so the footprint stays
    round on screen even when the two axes have different scales.
    """
    ux = (np.asarray(px, dtype=float) - x0) / rx
    uy = (np.asarray(py, dtype=float) - y0) / ry
    vx = (x1 - x0) / rx
    vy = (y1 - y0) / ry
    vv = vx * vx + vy * vy
    # A zero-length segment (a single dab) collapses to its start point.
    t = np.clip((ux * vx + uy * vy) / vv, 0.0, 1.0) if vv > 0 else 0.0
    dx = ux - t * vx
    dy = uy - t * vy
    return (dx * dx + dy * dy) <= 1.0


class SelectionGeometry:
    """Stores the mathematical shape of a user-drawn selection region.

    Unlike biaplotter, which computed point indices on the currently plotted
    frame and immediately discarded the geometry, this class preserves the
    actual shape parameters so selections can be evaluated across any dataset
    or timelapse stack without monkey-patching.
    """

    def __init__(self, shape_type: str, params: Any):
        self.shape_type = shape_type
        self.params = params

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask for which ``points`` (N, 2) fall inside the region."""
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            return np.zeros(0, dtype=bool)

        x = points[:, 0]
        y = points[:, 1]

        if self.shape_type == "rectangle":
            xmin, xmax, ymin, ymax = self.params
            return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

        if self.shape_type == "ellipse":
            cx, cy, rx, ry = self.params
            if rx <= 0 or ry <= 0:
                return np.zeros(len(points), dtype=bool)
            return (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2) <= 1.0

        if self.shape_type == "lasso":
            path: mplPath = self.params
            return path.contains_points(points)

        if self.shape_type == "brush":
            return self._brush_contains(x, y)

        return np.zeros(len(points), dtype=bool)

    def _brush_contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return a mask of points covered by a stored brush stroke.

        ``params`` holds one ``(x0, y0, x1, y1, rx, ry)`` row per painted
        segment; a point belongs to the stroke when any segment covers it.
        """
        segments = np.atleast_2d(np.asarray(self.params, dtype=float))
        inside = np.zeros(len(x), dtype=bool)
        if segments.size == 0:
            return inside

        # Only points inside the stroke bounding box can be covered, and
        # a long stroke holds many segments, so prefilter before looping.
        rx = segments[:, 4]
        ry = segments[:, 5]
        xmin = np.min(np.minimum(segments[:, 0], segments[:, 2]) - rx)
        xmax = np.max(np.maximum(segments[:, 0], segments[:, 2]) + rx)
        ymin = np.min(np.minimum(segments[:, 1], segments[:, 3]) - ry)
        ymax = np.max(np.maximum(segments[:, 1], segments[:, 3]) + ry)
        near = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        if not np.any(near):
            return inside

        near_x = x[near]
        near_y = y[near]
        covered = np.zeros(len(near_x), dtype=bool)
        for x0, y0, x1, y1, seg_rx, seg_ry in segments:
            remaining = ~covered
            if not np.any(remaining):
                break
            covered[remaining] = _capsule_mask(
                near_x[remaining],
                near_y[remaining],
                x0,
                y0,
                x1,
                y1,
                seg_rx,
                seg_ry,
            )
        inside[near] = covered
        return inside


def _render_selector_pixmap(shape: str, color: str, size: int = 24) -> QPixmap:
    """Render a pixmap for a selector shape in a given color."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    pen = QPen(QColor(color))
    pen.setWidthF(1.8)
    # The drawing tools are outlined like the region they leave behind:
    # dashed for the marquee shapes, solid for the painting tools.
    if shape in ("brush", "eraser"):
        pen.setWidthF(1.5)
    else:
        pen.setStyle(Qt.DashLine)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    margin = 3.5
    rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)

    if shape == "rectangle":
        painter.drawRect(rect)
    elif shape == "ellipse":
        painter.drawEllipse(rect)
    elif shape == "lasso":
        from qtpy.QtGui import QPainterPath

        path = QPainterPath()
        path.moveTo(margin + 3, size - margin - 2)
        path.cubicTo(
            margin - 2,
            margin + 4,
            size - margin - 2,
            margin - 1,
            size - margin,
            margin + 8,
        )
        path.cubicTo(
            size - margin + 2,
            size - margin - 2,
            margin + 8,
            size - margin,
            margin + 3,
            size - margin - 2,
        )
        path.lineTo(margin, size - margin + 1)
        painter.drawPath(path)
    elif shape == "brush":
        scale = size / 24.0

        def pt(x, y):
            return QPointF(x * scale, y * scale)

        painter.drawLine(pt(20.5, 3.5), pt(13.0, 11.0))  # handle
        painter.drawLine(pt(10.0, 9.5), pt(14.5, 14.0))  # ferrule
        bristles = QPolygonF([pt(10.0, 11.0), pt(13.0, 14.0), pt(5.0, 19.0)])
        painter.setBrush(QColor(color))
        painter.drawPolygon(bristles)
        painter.setBrush(Qt.NoBrush)
    elif shape == "eraser":
        scale = size / 24.0

        def pt(x, y):
            return QPointF(x * scale, y * scale)

        body = QPolygonF(
            [pt(4.0, 15.5), pt(11.5, 5.0), pt(20.0, 5.0), pt(12.5, 15.5)]
        )
        painter.drawPolygon(body)
        painter.drawLine(pt(7.75, 10.25), pt(16.25, 10.25))  # rubber band
        painter.drawLine(pt(3.0, 19.5), pt(21.0, 19.5))  # erased line

    painter.end()
    return pixmap


def _make_selector_icon(
    shape: str,
    normal_color: str = "#ffffff",
    checked_color: str = "#00c18c",
    size: int = 24,
) -> QIcon:
    """Render a crisp vector QIcon for a selection tool.

    ``shape`` is one of ``rectangle``, ``ellipse``, ``lasso``, ``brush``
    or ``eraser``; both the normal and the checked state are rendered.
    """
    icon = QIcon()
    pixmap_off = _render_selector_pixmap(shape, normal_color, size)
    pixmap_on = _render_selector_pixmap(shape, checked_color, size)
    icon.addPixmap(pixmap_off, QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(pixmap_on, QIcon.Mode.Normal, QIcon.State.On)
    return icon


def _make_brush_cursor(
    size_px: float, color: str, widget: QWidget | None = None
) -> QCursor:
    """Return a circular outline cursor matching the brush footprint.

    Drawing the brush outline into the mouse cursor rather than into the
    figure keeps hovering free: no Matplotlib redraw is needed to move it.
    """
    ratio = 1.0
    if widget is not None:
        with contextlib.suppress(Exception):
            ratio = float(widget.devicePixelRatioF())
    if not np.isfinite(ratio) or ratio <= 0:
        ratio = 1.0

    diameter = float(np.clip(size_px, 4.0, 96.0))
    total = int(np.ceil(diameter)) + 6
    pixmap = QPixmap(int(round(total * ratio)), int(round(total * ratio)))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(Qt.NoBrush)
    offset = (total - diameter) / 2.0
    rect = QRectF(offset, offset, diameter, diameter)

    # A dark halo underneath keeps the outline readable on any background.
    halo = QPen(QColor(0, 0, 0, 170))
    halo.setWidthF(2.6)
    painter.setPen(halo)
    painter.drawEllipse(rect)

    pen = QPen(QColor(color))
    pen.setWidthF(1.2)
    painter.setPen(pen)
    painter.drawEllipse(rect)

    # A small crosshair marks the exact spot the stroke is centred on; it
    # shrinks with the brush so it never outgrows a small circle.
    center = total / 2.0
    tick = min(2.5, diameter / 4.0)
    painter.drawLine(
        QPointF(center - tick, center), QPointF(center + tick, center)
    )
    painter.drawLine(
        QPointF(center, center - tick), QPointF(center, center + tick)
    )
    painter.end()

    # With a device pixel ratio set, Qt expects the hot spot in
    # device-independent pixels.
    return QCursor(pixmap, total // 2, total // 2)


class BaseInteractiveSelector:
    """Base class for interactive phasor selectors."""

    selection_applied_signal: Signal = Signal(np.ndarray)

    def __init__(
        self,
        ax: plt.Axes,
        canvas_widget: PhasorCanvasWidget,
        name: str,
    ):
        self.ax = ax
        self.canvas_widget = canvas_widget
        self.name = name
        self._data: np.ndarray | None = None
        self._selector = None
        self._class_value: int = 1
        self._selected_indices: np.ndarray | None = None
        self._last_geometry: SelectionGeometry | None = None
        self._cids: list[int] = []

    @property
    def data(self) -> np.ndarray | None:
        return self._data

    @data.setter
    def data(self, value: np.ndarray | None):
        self._data = value

    @property
    def class_value(self) -> int:
        return self._class_value

    @class_value.setter
    def class_value(self, value: int):
        self._class_value = int(value)

    @property
    def selected_indices(self) -> np.ndarray | None:
        return self._selected_indices

    @selected_indices.setter
    def selected_indices(self, value: np.ndarray | None):
        self._selected_indices = value

    @property
    def last_geometry(self) -> SelectionGeometry | None:
        return self._last_geometry

    def cursor(self) -> QCursor:
        """Mouse cursor to show while this selector is active."""
        return QCursor(Qt.CrossCursor)

    def _connect(self, event_name: str, handler: Callable) -> None:
        """Register a canvas callback owned (and later freed) by self."""
        self._cids.append(
            self.canvas_widget.canvas.mpl_connect(event_name, handler)
        )

    def _disconnect_all(self) -> None:
        """Drop every canvas callback registered by this selector."""
        canvas = getattr(self.canvas_widget, "canvas", None)
        for cid in self._cids:
            with contextlib.suppress(Exception):
                canvas.mpl_disconnect(cid)
        self._cids = []

    def remove(self):
        """Disconnect and clear the selector widget from axes."""
        self._disconnect_all()
        if self._selector is not None:
            with contextlib.suppress(Exception):
                self._selector.clear()
            with contextlib.suppress(Exception):
                self._selector.disconnect_events()
            self._selector = None

    def apply_selection(self):
        """Apply current selection to active artist and notify listeners."""
        if self._selected_indices is None or len(self._selected_indices) == 0:
            return

        artist = self.canvas_widget.active_artist_object
        if artist is not None and artist.data is not None:
            color_indices = artist.color_indices
            if color_indices is None or np.isscalar(color_indices):
                color_indices = np.zeros(len(artist.data), dtype=np.uint32)
            else:
                color_indices = np.array(color_indices, copy=True)

            valid_indices = self._selected_indices[
                self._selected_indices < len(color_indices)
            ]
            color_indices[valid_indices] = self._class_value
            artist.color_indices = color_indices

            self.selection_applied_signal.emit(color_indices)
            self.canvas_widget.figure.canvas.draw_idle()


class InteractiveRectangleSelector(BaseInteractiveSelector):
    """Interactive rectangle selector for phasor space."""

    def __init__(self, ax: plt.Axes, canvas_widget: PhasorCanvasWidget):
        super().__init__(ax, canvas_widget, "Interactive Rectangle Selector")

    def create_selector(self):
        self.remove()
        self._selector = RectangleSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
            drag_from_anywhere=True,
            props={
                "facecolor": "#00c18c",
                "edgecolor": "#00c18c",
                "alpha": 0.3,
                "fill": True,
                "linewidth": 2.0,
                "linestyle": "--",
            },
        )
        self._connect("button_press_event", self._on_button_press)

    def on_select(self, eclick, erelease) -> np.ndarray | None:
        if eclick.xdata is None or erelease.xdata is None:
            return None
        xmin = min(eclick.xdata, erelease.xdata)
        xmax = max(eclick.xdata, erelease.xdata)
        ymin = min(eclick.ydata, erelease.ydata)
        ymax = max(eclick.ydata, erelease.ydata)

        self._last_geometry = SelectionGeometry(
            "rectangle", (xmin, xmax, ymin, ymax)
        )
        points = (
            self._data
            if self._data is not None
            else getattr(self.canvas_widget.active_artist_object, "data", None)
        )
        if points is None or len(points) == 0:
            return None

        inside = self._last_geometry.contains_points(points)
        self._selected_indices = np.flatnonzero(inside)
        return self._selected_indices

    def _on_button_press(self, event):
        if event.button == 3:
            self.apply_selection()


class InteractiveEllipseSelector(BaseInteractiveSelector):
    """Interactive ellipse selector for phasor space."""

    def __init__(self, ax: plt.Axes, canvas_widget: PhasorCanvasWidget):
        super().__init__(ax, canvas_widget, "Interactive Ellipse Selector")

    def create_selector(self):
        self.remove()
        self._selector = EllipseSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
            drag_from_anywhere=True,
            props={
                "facecolor": "#00c18c",
                "edgecolor": "#00c18c",
                "alpha": 0.3,
                "fill": True,
                "linewidth": 2.0,
                "linestyle": "--",
            },
        )
        self._connect("button_press_event", self._on_button_press)

    def on_select(self, eclick, erelease) -> np.ndarray | None:
        if eclick.xdata is None or erelease.xdata is None:
            return None
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        rx = abs(x2 - x1) / 2.0
        ry = abs(y2 - y1) / 2.0

        self._last_geometry = SelectionGeometry("ellipse", (cx, cy, rx, ry))
        points = (
            self._data
            if self._data is not None
            else getattr(self.canvas_widget.active_artist_object, "data", None)
        )
        if points is None or len(points) == 0:
            return None

        inside = self._last_geometry.contains_points(points)
        self._selected_indices = np.flatnonzero(inside)
        return self._selected_indices

    def _on_button_press(self, event):
        if event.button == 3:
            self.apply_selection()


class InteractiveLassoSelector(BaseInteractiveSelector):
    """Interactive free-form lasso selector for phasor space."""

    def __init__(self, ax: plt.Axes, canvas_widget: PhasorCanvasWidget):
        super().__init__(ax, canvas_widget, "Interactive Lasso Selector")

    def create_selector(self):
        self.remove()
        self._selector = LassoSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            props={
                "color": "#00c18c",
                "linewidth": 2.0,
                "linestyle": "--",
            },
        )

    def on_select(self, vertices: Any) -> np.ndarray | None:
        if vertices is None or len(vertices) < 3:
            return None
        path = mplPath(vertices)
        self._last_geometry = SelectionGeometry("lasso", path)

        points = (
            self._data
            if self._data is not None
            else getattr(self.canvas_widget.active_artist_object, "data", None)
        )
        if points is None or len(points) == 0:
            return None

        inside = self._last_geometry.contains_points(points)
        self._selected_indices = np.flatnonzero(inside)
        self.apply_selection()
        return self._selected_indices


class InteractiveBrushSelector(BaseInteractiveSelector):
    """Free-hand brush that paints the active class onto the phasor plot.

    The same implementation backs the eraser: erasing is painting with the
    class value ``0`` (unassigned) instead of the class currently selected
    in the Manual Selection tab.

    The brush size is expressed in screen pixels so that it stays visually
    constant while zooming, and is converted to data units on every event.
    Granularity follows the active artist: histogram bins are painted whole,
    so the overlay always matches what the user sees under the cursor, while
    a scatter plot is painted point by point.

    While the mouse is held down only the plot overlay is refreshed; the
    napari layers and the statistics are updated once, on release.
    """

    BRUSH_COLOR = "#00c18c"
    ERASER_COLOR = "#ff5c5c"

    def __init__(
        self,
        ax: plt.Axes,
        canvas_widget: PhasorCanvasWidget,
        erase: bool = False,
    ):
        super().__init__(
            ax,
            canvas_widget,
            (
                "Interactive Eraser Selector"
                if erase
                else "Interactive Brush Selector"
            ),
        )
        self.erase = bool(erase)
        self._size_px = float(DEFAULT_BRUSH_SIZE_PX)
        self._painting = False
        self._last_point: tuple[float, float] | None = None
        self._artist: Any | None = None
        self._points: np.ndarray | None = None
        self._working: np.ndarray | None = None
        self._stroke_segments: list[tuple[float, ...]] = []
        self._bin_lookup: tuple[Any, ...] | None = None
        self._grid: np.ndarray | None = None

    @property
    def size_px(self) -> float:
        """Brush diameter in screen pixels."""
        return self._size_px

    @size_px.setter
    def size_px(self, value: float):
        self._size_px = float(np.clip(float(value), 1.0, 96.0))
        if self.canvas_widget.active_selector is self:
            self.canvas_widget.canvas.setCursor(self.cursor())

    @property
    def is_eraser(self) -> bool:
        """True when this tool clears the selection instead of painting it."""
        return self.erase

    @property
    def paint_value(self) -> int:
        """Class value this tool writes under the cursor."""
        return 0 if self.erase else int(self._class_value)

    def cursor(self) -> QCursor:
        color = self.ERASER_COLOR if self.erase else self.BRUSH_COLOR
        return _make_brush_cursor(
            self._size_px, color, self.canvas_widget.canvas
        )

    def create_selector(self):
        self.remove()
        self._connect("button_press_event", self._on_press)
        self._connect("motion_notify_event", self._on_motion)
        self._connect("button_release_event", self._on_release)

    def remove(self):
        self._painting = False
        self._last_point = None
        self._artist = None
        self._points = None
        self._working = None
        self._bin_lookup = None
        self._grid = None
        super().remove()

    def _radii_data(self) -> tuple[float, float]:
        """Return the brush radii converted from screen to data units."""
        inverse = self.ax.transData.inverted()
        x0, y0 = inverse.transform((0.0, 0.0))
        x1, y1 = inverse.transform((self._size_px / 2.0, self._size_px / 2.0))
        rx = abs(float(x1 - x0))
        ry = abs(float(y1 - y0))
        if not np.isfinite(rx) or rx <= 0:
            rx = 1e-12
        if not np.isfinite(ry) or ry <= 0:
            ry = 1e-12
        return rx, ry

    def _on_press(self, event):
        if event.button != 1 or event.inaxes is not self.ax:
            return
        if getattr(self.canvas_widget.toolbar, "mode", ""):
            return

        artist = self.canvas_widget.active_artist_object
        points = self._data
        if points is None:
            points = getattr(artist, "data", None)
        if artist is None or points is None or len(points) == 0:
            return

        self._begin_stroke(artist, np.asarray(points, dtype=float))
        self._painting = True
        self._last_point = (event.xdata, event.ydata)
        self._paint_segment(event.xdata, event.ydata, event.xdata, event.ydata)

    def _on_motion(self, event):
        if not self._painting:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._last_point is None:
            self._last_point = (event.xdata, event.ydata)
        x0, y0 = self._last_point
        self._paint_segment(x0, y0, event.xdata, event.ydata)
        self._last_point = (event.xdata, event.ydata)

    def _on_release(self, event):
        if not self._painting:
            return
        if event.button not in (1, None):
            return
        self._painting = False
        self._last_point = None
        self._finish_stroke()

    def _begin_stroke(self, artist: Any, points: np.ndarray):
        """Snapshot the current selection so the stroke can extend it."""
        self._artist = artist
        self._points = points
        self._stroke_segments = []
        self._bin_lookup = None
        self._grid = None

        indices = getattr(artist, "color_indices", None)
        indices = None if indices is None else np.asarray(indices)
        if (
            indices is None
            or indices.ndim == 0
            or len(indices) != len(points)
            or indices.dtype.kind not in "iu"
        ):
            working = np.zeros(len(points), dtype=np.uint32)
        else:
            working = np.array(indices, copy=True)
        self._working = working

        histogram = getattr(artist, "histogram", None)
        if histogram is not None and hasattr(artist, "_render_overlay_grid"):
            self._build_bin_lookup(histogram, points, working)

    def _build_bin_lookup(
        self, histogram: tuple, points: np.ndarray, working: np.ndarray
    ):
        """Index the plotted points by histogram bin for the whole stroke.

        Sorting once on mouse press turns every later dab into a lookup of
        the points inside the touched bins, instead of a scan over the whole
        dataset.
        """
        _, x_edges, y_edges = histogram
        nx = len(x_edges) - 1
        ny = len(y_edges) - 1
        if nx < 1 or ny < 1:
            return

        x_idx = np.digitize(points[:, 0], x_edges) - 1
        y_idx = np.digitize(points[:, 1], y_edges) - 1
        # ``np.histogram2d`` puts values on the upper edge in the last bin.
        x_idx[points[:, 0] == x_edges[-1]] = nx - 1
        y_idx[points[:, 1] == y_edges[-1]] = ny - 1

        inside = (x_idx >= 0) & (x_idx < nx) & (y_idx >= 0) & (y_idx < ny)
        if not np.any(inside):
            return

        point_ids = np.flatnonzero(inside)
        bin_ids = (x_idx[point_ids] * ny + y_idx[point_ids]).astype(np.int64)
        order = np.argsort(bin_ids, kind="stable")
        sorted_points = point_ids[order]
        starts = np.searchsorted(bin_ids[order], np.arange(nx * ny + 1))

        grid = np.zeros((nx, ny), dtype=np.int32)
        values = working[point_ids].astype(np.int32)
        painted = values > 0
        if np.any(painted):
            np.maximum.at(
                grid,
                (x_idx[point_ids][painted], y_idx[point_ids][painted]),
                values[painted],
            )

        centers_x, centers_y = np.meshgrid(
            0.5 * (x_edges[:-1] + x_edges[1:]),
            0.5 * (y_edges[:-1] + y_edges[1:]),
            indexing="ij",
        )
        self._bin_lookup = (
            sorted_points,
            starts,
            nx,
            ny,
            centers_x.ravel(),
            centers_y.ravel(),
            x_edges,
            y_edges,
        )
        self._grid = grid

    def _bin_at(self, x: float, y: float) -> int | None:
        """Return the flat bin index under ``(x, y)``, or None if outside."""
        if self._bin_lookup is None:
            return None
        _, _, nx, ny, _, _, x_edges, y_edges = self._bin_lookup
        ix = int(np.digitize(x, x_edges)) - 1
        iy = int(np.digitize(y, y_edges)) - 1
        if x == x_edges[-1]:
            ix = nx - 1
        if y == y_edges[-1]:
            iy = ny - 1
        if 0 <= ix < nx and 0 <= iy < ny:
            return ix * ny + iy
        return None

    def _paint_segment(self, x0: float, y0: float, x1: float, y1: float):
        """Paint the swept footprint between two cursor positions."""
        if self._working is None or self._artist is None:
            return
        rx, ry = self._radii_data()
        self._stroke_segments.append((x0, y0, x1, y1, rx, ry))
        value = self.paint_value

        if self._bin_lookup is not None:
            sorted_points, starts, nx, ny, cx, cy, _, _ = self._bin_lookup
            hit = _capsule_mask(cx, cy, x0, y0, x1, y1, rx, ry)
            # A brush narrower than a bin would otherwise cover no centre at
            # all, so the bins under the two ends always count as touched.
            for px, py in ((x0, y0), (x1, y1)):
                flat = self._bin_at(px, py)
                if flat is not None:
                    hit[flat] = True

            bins = np.flatnonzero(hit)
            # Empty bins hold no points, and painting them would colour the
            # overlay where the plot shows nothing.
            bins = bins[starts[bins + 1] > starts[bins]]
            if len(bins) == 0:
                return
            self._working[
                np.concatenate(
                    [sorted_points[starts[b] : starts[b + 1]] for b in bins]
                )
            ] = value
            self._grid[bins // ny, bins % ny] = value
            self._artist._color_indices = self._working
            self._artist._render_overlay_grid(self._grid)
        else:
            hit = _capsule_mask(
                self._points[:, 0],
                self._points[:, 1],
                x0,
                y0,
                x1,
                y1,
                rx,
                ry,
            )
            if not np.any(hit):
                return
            self._working[hit] = value
            self._artist._color_indices = self._working
            if hasattr(self._artist, "_colorize"):
                self._artist._colorize(self._working)

        self.canvas_widget.figure.canvas.draw_idle()

    def _finish_stroke(self):
        """Commit the stroke to the artist and notify the rest of the app."""
        artist = self._artist
        working = self._working
        segments = self._stroke_segments

        self._artist = None
        self._points = None
        self._working = None
        self._bin_lookup = None
        self._grid = None
        self._stroke_segments = []

        if artist is None or working is None or not segments:
            return

        self._last_geometry = SelectionGeometry(
            "brush", np.asarray(segments, dtype=float)
        )
        self._selected_indices = np.flatnonzero(working == self.paint_value)
        artist.color_indices = working
        self.selection_applied_signal.emit(working)
        self.canvas_widget.figure.canvas.draw_idle()


class Histogram2D:
    """High-performance 2D Histogram artist with fast vectorized overlay coloring."""

    data_changed_signal: Signal = Signal(np.ndarray)
    color_indices_changed_signal: Signal = Signal(object)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other.upper() in ("HISTOGRAM2D", "HISTOGRAM_2D")
        return super().__eq__(other)

    __hash__ = object.__hash__

    def __init__(
        self,
        ax: plt.Axes,
        bins: Union[int, tuple[int, int]] = 100,
        histogram_colormap: Any = plt.cm.magma,
        overlay_colormap: Any = default_overlay_cmap_first_transparent,
        cmin: int = 1,
    ):
        self.ax = ax
        self._data: np.ndarray | None = None
        self._bins = bins
        self._cmin = cmin
        self._histogram_colormap = histogram_colormap
        self._overlay_colormap = overlay_colormap
        self._histogram_norm_method = "linear"
        self._color_indices: np.ndarray | None = None
        self._fixed_counts_range: tuple[float, float] | None = None
        self._histogram: tuple[np.ndarray, np.ndarray, np.ndarray] | None = (
            None
        )
        self._visible = True
        self._overlay_visible = True

        self._mpl_artists: dict[str, Any] = {}

    @property
    def _napari_phasors_fixed_counts_range(
        self,
    ) -> tuple[float, float] | None:
        return self._fixed_counts_range

    @_napari_phasors_fixed_counts_range.setter
    def _napari_phasors_fixed_counts_range(
        self, value: tuple[float, float] | None
    ):
        self._fixed_counts_range = value

    @property
    def data(self) -> np.ndarray | None:
        return self._data

    @data.setter
    def data(self, value: np.ndarray | None):
        self._data = value
        if value is not None and len(value) > 0:
            self.data_changed_signal.emit(value)
            self._refresh(force_redraw=False)

    @property
    def bins(self) -> Union[int, tuple[int, int]]:
        return self._bins

    @bins.setter
    def bins(self, value: Union[int, tuple[int, int]]):
        self._bins = value
        if self._data is not None and len(self._data) > 0:
            self._refresh(force_redraw=False)

    @property
    def cmin(self) -> int:
        return self._cmin

    @cmin.setter
    def cmin(self, value: int):
        self._cmin = value

    @property
    def histogram(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        return self._histogram

    @property
    def histogram_colormap(self) -> Any:
        return self._histogram_colormap

    @histogram_colormap.setter
    def histogram_colormap(self, value: Any):
        self._histogram_colormap = value
        if self._histogram is not None:
            self._refresh(force_redraw=False)

    @property
    def overlay_colormap(self) -> Any:
        return self._overlay_colormap

    @overlay_colormap.setter
    def overlay_colormap(self, value: Any):
        self._overlay_colormap = value
        if self._color_indices is not None:
            self._colorize(self._color_indices)

    @property
    def histogram_color_normalization_method(self) -> str:
        return self._histogram_norm_method

    @histogram_color_normalization_method.setter
    def histogram_color_normalization_method(self, value: str):
        self._histogram_norm_method = value
        if self._histogram is not None:
            self._refresh(force_redraw=False)

    @property
    def color_indices(self) -> np.ndarray | None:
        return self._color_indices

    @color_indices.setter
    def color_indices(self, value: np.ndarray | None):
        self._color_indices = value
        self.color_indices_changed_signal.emit(value)
        self._colorize(value)

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = bool(value)
        img = self._mpl_artists.get("histogram_image")
        if img is not None:
            img.set_visible(self._visible)
        overlay = self._mpl_artists.get("overlay_histogram_image")
        if overlay is not None:
            overlay.set_visible(self._visible and self._overlay_visible)
        if self.ax.figure and self.ax.figure.canvas:
            self.ax.figure.canvas.draw_idle()

    @property
    def overlay_visible(self) -> bool:
        return self._overlay_visible

    @overlay_visible.setter
    def overlay_visible(self, value: bool):
        self._overlay_visible = bool(value)
        overlay = self._mpl_artists.get("overlay_histogram_image")
        if overlay is not None:
            overlay.set_visible(self._visible and self._overlay_visible)
        if self.ax.figure and self.ax.figure.canvas:
            self.ax.figure.canvas.draw_idle()

    def _remove_artists(self, keys: list[str] | None = None):
        if keys is None:
            self._color_indices = None
            self._histogram = None
        to_remove = (
            keys
            if keys is not None
            else ["histogram_image", "overlay_histogram_image"]
        )
        for key in to_remove:
            artist = self._mpl_artists.pop(key, None)
            if artist is not None:
                with contextlib.suppress(Exception):
                    artist.remove()

    def _get_normalization(
        self, values: np.ndarray | None = None, is_overlay: bool = False
    ) -> Normalize:
        if is_overlay:
            return Normalize(vmin=0, vmax=1)

        counts = (
            values
            if values is not None
            else (self._histogram[0] if self._histogram else None)
        )
        if counts is None:
            return Normalize(vmin=0, vmax=1)

        if self._fixed_counts_range is not None:
            vmin, vmax = (
                float(self._fixed_counts_range[0]),
                float(self._fixed_counts_range[1]),
            )
        else:
            vmin = float(self._cmin)
            vmax = (
                float(np.nanmax(counts)) if np.any(counts > 0) else vmin + 1.0
            )

        if self._histogram_norm_method == "log":
            vmin = max(vmin, 0.01)
            vmax = max(vmax, vmin * 1.0001)
            return LogNorm(vmin=vmin, vmax=vmax)
        else:
            if vmax <= vmin:
                vmax = vmin + 1.0
            return Normalize(vmin=vmin, vmax=vmax)

    def _refresh(self, force_redraw: bool = True):
        if self._data is None or len(self._data) == 0:
            self._remove_artists()
            return

        x = self._data[:, 0]
        y = self._data[:, 1]
        counts, x_edges, y_edges = np.histogram2d(x, y, bins=self._bins)
        self._histogram = (counts, x_edges, y_edges)

        norm = self._get_normalization(counts, is_overlay=False)

        cmap = (
            self._histogram_colormap
            if isinstance(self._histogram_colormap, mcolors.Colormap)
            else plt.get_cmap(self._histogram_colormap)
        )

        counts_t = counts.T
        rgba = cmap(norm(counts_t))
        rgba[counts_t < self._cmin, 3] = 0.0

        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        img = self._mpl_artists.get("histogram_image")
        if img is None or img.axes != self.ax:
            self._mpl_artists["histogram_image"] = self.ax.imshow(
                rgba,
                extent=extent,
                origin="lower",
                zorder=1,
                interpolation="nearest",
                aspect="auto",
                visible=self._visible,
            )
        else:
            img.set_data(rgba)
            img.set_extent(extent)
            img.set_visible(self._visible)

        if self._color_indices is not None:
            self._colorize(self._color_indices)

    def _colorize(self, indices: np.ndarray | None):
        self._render_overlay_grid(self._overlay_grid(indices))

    def _overlay_grid(self, indices: np.ndarray | None) -> np.ndarray | None:
        """Reduce per-point class values to the per-bin grid that is drawn.

        A bin takes the highest class value among the points it holds, and
        None means there is nothing to overlay at all.
        """
        if indices is None or self._data is None or self._histogram is None:
            return None

        indices = np.asarray(indices)
        if indices.ndim == 0:
            if indices == 0:
                return None
            indices = np.full(len(self._data), indices, dtype=np.int32)
        elif len(indices) != len(self._data):
            return None

        non_zero = indices > 0
        if not np.any(non_zero):
            return None

        _, x_edges, y_edges = self._histogram
        nx = len(x_edges) - 1
        ny = len(y_edges) - 1

        x = self._data[:, 0][non_zero]
        y = self._data[:, 1][non_zero]
        vals = indices[non_zero].astype(np.int32)

        x_idx = np.digitize(x, x_edges) - 1
        y_idx = np.digitize(y, y_edges) - 1

        valid = (x_idx >= 0) & (x_idx < nx) & (y_idx >= 0) & (y_idx < ny)
        if not np.any(valid):
            return None

        grid = np.zeros((nx, ny), dtype=np.int32)
        np.maximum.at(grid, (x_idx[valid], y_idx[valid]), vals[valid])
        return grid

    def _render_overlay_grid(self, grid: np.ndarray | None):
        """Draw the categorical overlay image for a per-bin class ``grid``.

        Kept separate from :meth:`_overlay_grid` so tools that already know
        which bins they touched, such as the brush, can refresh the overlay
        without rebuilding it from every plotted point.
        """
        if grid is None or self._histogram is None or not np.any(grid):
            if "overlay_histogram_image" in self._mpl_artists:
                self._remove_artists(["overlay_histogram_image"])
            return

        _, x_edges, y_edges = self._histogram

        overlay_cmap = (
            self._overlay_colormap
            if isinstance(self._overlay_colormap, mcolors.Colormap)
            else plt.get_cmap(self._overlay_colormap)
        )

        grid_t = grid.T
        if isinstance(overlay_cmap, mcolors.ListedColormap):
            clipped_grid = np.clip(grid_t, 0, overlay_cmap.N - 1)
            overlay_rgba = overlay_cmap(clipped_grid)
        else:
            norm = Normalize(vmin=0, vmax=max(int(np.max(grid_t)), 1))
            overlay_rgba = overlay_cmap(norm(grid_t))
        overlay_rgba[grid_t == 0, 3] = 0.0

        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        overlay_img = self._mpl_artists.get("overlay_histogram_image")
        if overlay_img is None or overlay_img.axes != self.ax:
            self._mpl_artists["overlay_histogram_image"] = self.ax.imshow(
                overlay_rgba,
                extent=extent,
                origin="lower",
                zorder=2,
                interpolation="nearest",
                aspect="auto",
                visible=self._visible and self._overlay_visible,
            )
        else:
            overlay_img.set_data(overlay_rgba)
            overlay_img.set_extent(extent)
            overlay_img.set_visible(self._visible and self._overlay_visible)


Histogram2DArtist = Histogram2D


class Scatter:
    """Scatter artist for phasor space."""

    data_changed_signal: Signal = Signal(np.ndarray)
    color_indices_changed_signal: Signal = Signal(object)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other.upper() == "SCATTER"
        return super().__eq__(other)

    __hash__ = object.__hash__

    def __init__(
        self,
        ax: plt.Axes,
        size: float = 20.0,
        alpha: float = 1.0,
        color: Any = "#1f77b4",
        overlay_colormap: Any = default_overlay_cmap,
    ):
        self.ax = ax
        self._data: np.ndarray | None = None
        self._size = size
        self._alpha = alpha
        self._color = color
        self._overlay_colormap = overlay_colormap
        self._color_indices: np.ndarray | None = None
        self._visible = True

        self._mpl_artists: dict[str, Any] = {}

    @property
    def data(self) -> np.ndarray | None:
        return self._data

    @data.setter
    def data(self, value: np.ndarray | None):
        self._data = value
        if value is not None and len(value) > 0:
            self.data_changed_signal.emit(value)
            self._refresh(force_redraw=False)

    @property
    def size(self) -> float:
        return self._size

    @size.setter
    def size(self, value: float):
        self._size = float(value)
        scatter = self._mpl_artists.get("scatter")
        if scatter is not None:
            scatter.set_sizes([self._size])
            if self.ax.figure and self.ax.figure.canvas:
                self.ax.figure.canvas.draw_idle()

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = float(value)
        scatter = self._mpl_artists.get("scatter")
        if scatter is not None:
            scatter.set_alpha(self._alpha)
            if self.ax.figure and self.ax.figure.canvas:
                self.ax.figure.canvas.draw_idle()

    @property
    def color(self) -> Any:
        return self._color

    @color.setter
    def color(self, value: Any):
        self._color = value
        self._colorize(self._color_indices)
        if self.ax.figure and self.ax.figure.canvas:
            self.ax.figure.canvas.draw_idle()

    @property
    def overlay_colormap(self) -> Any:
        return self._overlay_colormap

    @overlay_colormap.setter
    def overlay_colormap(self, value: Any):
        self._overlay_colormap = value
        self._colorize(self._color_indices)

    @property
    def color_indices(self) -> np.ndarray | None:
        return self._color_indices

    @color_indices.setter
    def color_indices(self, value: np.ndarray | None):
        self._color_indices = value
        self.color_indices_changed_signal.emit(value)
        self._colorize(value)

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = bool(value)
        scatter = self._mpl_artists.get("scatter")
        if scatter is not None:
            scatter.set_visible(self._visible)
        if self.ax.figure and self.ax.figure.canvas:
            self.ax.figure.canvas.draw_idle()

    def _remove_artists(self, keys: list[str] | None = None):
        if keys is None:
            self._color_indices = None
        scatter = self._mpl_artists.pop("scatter", None)
        if scatter is not None:
            with contextlib.suppress(Exception):
                scatter.remove()

    def _refresh(self, force_redraw: bool = True):
        if self._data is None or len(self._data) == 0:
            self._remove_artists()
            return

        saved_indices = self._color_indices
        scatter = self._mpl_artists.get("scatter")
        if scatter is None or scatter.axes != self.ax or force_redraw:
            self._remove_artists()
            self._color_indices = saved_indices
            self._mpl_artists["scatter"] = self.ax.scatter(
                self._data[:, 0],
                self._data[:, 1],
                s=self._size,
                alpha=self._alpha,
                c=[self._color],
                edgecolors="none",
                linewidths=0,
                zorder=1,
                visible=self._visible,
            )
        else:
            scatter.set_offsets(self._data)
            scatter.set_visible(self._visible)

        self._colorize(self._color_indices)

    def _colorize(self, indices: np.ndarray | None):
        scatter = self._mpl_artists.get("scatter")
        if scatter is None or self._data is None:
            return

        scatter.set_edgecolor("none")
        scatter.set_linewidth(0)

        if indices is None:
            scatter.set_facecolor(self._color)
            return

        indices = np.asarray(indices)
        if indices.ndim == 0:
            if indices == 0:
                scatter.set_facecolor(self._color)
                return
            indices = np.full(len(self._data), indices, dtype=np.int32)
        elif len(indices) != len(self._data):
            scatter.set_facecolor(self._color)
            return

        cmap = (
            self._overlay_colormap
            if isinstance(self._overlay_colormap, mcolors.Colormap)
            else plt.get_cmap(self._overlay_colormap)
        )
        if isinstance(cmap, mcolors.ListedColormap):
            clipped = np.clip(indices, 0, cmap.N - 1)
            colors = np.array(cmap(clipped))
        else:
            max_idx = max(int(np.nanmax(indices)), 1)
            norm = Normalize(vmin=0, vmax=max_idx)
            colors = np.array(cmap(norm(indices)))

        zero_mask = indices == 0
        if np.any(zero_mask):
            base_rgba = mcolors.to_rgba(self._color)
            if colors.ndim > 1 and colors.shape[1] == 4:
                sample_zero = (
                    cmap(0)
                    if isinstance(cmap, mcolors.ListedColormap)
                    else colors[zero_mask][0]
                )
                if sample_zero[3] == 0 or np.all(zero_mask):
                    colors[zero_mask] = base_rgba

        scatter.set_facecolor(colors)
        scatter.set_edgecolor("none")
        scatter.set_linewidth(0)


ScatterArtist = Scatter


class Contour:
    """Contour plot artist for phasor space with single and multi-group support."""

    data_changed_signal: Signal = Signal(np.ndarray)
    color_indices_changed_signal: Signal = Signal(object)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other.upper() in ("CONTOUR", "CONTOUR_PLOT", "CONTOURPLOT")
        return super().__eq__(other)

    __hash__ = object.__hash__

    def __init__(
        self,
        ax: plt.Axes,
        bins: Union[int, tuple[int, int]] = 150,
        levels: Union[int, list[float], np.ndarray] = 5,
        linewidths: float = 1.5,
        colormap: Any = "jet",
        log_norm: bool = True,
    ):
        self.ax = ax
        self._data: np.ndarray | None = None
        self._bins = bins
        self._levels = levels
        self._linewidths = linewidths
        self._colormap = colormap
        self._log_norm = log_norm
        self._visible = True

        self._contour_collections: list[Any] = []
        self._histogram: tuple[np.ndarray, np.ndarray, np.ndarray] | None = (
            None
        )

        # Multi-group / multi-layer support
        self._grouped_data: dict[Any, tuple[np.ndarray, np.ndarray]] | None = (
            None
        )
        self._group_styles: dict[Any, dict[str, Any]] | None = None
        self._group_names: dict[Any, str] | None = None
        self._show_legend: bool = True

    @property
    def data(self) -> np.ndarray | None:
        return self._data

    @data.setter
    def data(self, value: np.ndarray | None):
        self._data = value
        self._grouped_data = None
        if value is not None and len(value) > 0:
            self.data_changed_signal.emit(value)
            self._refresh(force_redraw=False)
        else:
            self._remove_artists()

    @property
    def bins(self) -> Union[int, tuple[int, int]]:
        return self._bins

    @bins.setter
    def bins(self, value: Union[int, tuple[int, int]]):
        self._bins = value
        if (
            self._data is not None and len(self._data) > 0
        ) or self._grouped_data:
            self._refresh(force_redraw=False)

    @property
    def levels(self) -> Union[int, list[float], np.ndarray]:
        return self._levels

    @levels.setter
    def levels(self, value: Union[int, list[float], np.ndarray]):
        self._levels = value
        if (
            self._data is not None and len(self._data) > 0
        ) or self._grouped_data:
            self._refresh(force_redraw=False)

    @property
    def linewidths(self) -> float:
        return self._linewidths

    @linewidths.setter
    def linewidths(self, value: float):
        self._linewidths = float(value)
        if (
            self._data is not None and len(self._data) > 0
        ) or self._grouped_data:
            self._refresh(force_redraw=False)

    @property
    def colormap(self) -> Any:
        return self._colormap

    @colormap.setter
    def colormap(self, value: Any):
        self._colormap = value
        if (
            self._data is not None and len(self._data) > 0
        ) or self._grouped_data:
            self._refresh(force_redraw=False)

    @property
    def log_norm(self) -> bool:
        return self._log_norm

    @log_norm.setter
    def log_norm(self, value: bool):
        self._log_norm = bool(value)
        if (
            self._data is not None and len(self._data) > 0
        ) or self._grouped_data:
            self._refresh(force_redraw=False)

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = bool(value)
        for cs in self._contour_collections:
            with contextlib.suppress(Exception):
                cs.set_visible(self._visible)
            if hasattr(cs, "collections"):
                for col in cs.collections:
                    with contextlib.suppress(Exception):
                        col.set_visible(self._visible)
        legend = self.ax.get_legend()
        if legend is not None:
            with contextlib.suppress(Exception):
                legend.set_visible(self._visible)
        if self.ax.figure and self.ax.figure.canvas:
            self.ax.figure.canvas.draw_idle()

    @property
    def color_indices(self) -> np.ndarray | None:
        return None

    @color_indices.setter
    def color_indices(self, value: Any):
        pass

    @property
    def overlay_colormap(self) -> Any:
        return None

    @overlay_colormap.setter
    def overlay_colormap(self, value: Any):
        pass

    @property
    def histogram(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        return self._histogram

    def _tag_contour_set(self, cs_obj: Any, label: str | None = None):
        with contextlib.suppress(Exception):
            if hasattr(cs_obj, "collections"):
                for col in cs_obj.collections:
                    col.set_label("contour_plot_element")
            else:
                cs_obj.set_label("contour_plot_element")
            if label and hasattr(cs_obj, "collections") and cs_obj.collections:
                cs_obj.collections[0].set_label(label)

    def _remove_artists(self, keys: list[str] | None = None):
        for cs in self._contour_collections:
            with contextlib.suppress(Exception):
                cs.remove()
            if hasattr(cs, "collections"):
                for col in cs.collections:
                    with contextlib.suppress(Exception):
                        col.remove()
        self._contour_collections.clear()

        # Clean up any lingering tagged contour collections on axes
        for artist in list(self.ax.collections):
            if artist.get_label() == "contour_plot_element":
                with contextlib.suppress(Exception):
                    artist.remove()

        legend = self.ax.get_legend()
        if legend is not None:
            with contextlib.suppress(Exception):
                legend.remove()

        self._histogram = None

    def _compute_grid(self, x_all: np.ndarray, y_all: np.ndarray):
        """Compute the 2D bin edges and centers once on the pooled dataset."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        if (xlim[0] == 0.0 and xlim[1] == 1.0) and (
            ylim[0] == 0.0 and ylim[1] == 1.0
        ):
            xlim = (-0.1, 1.1)
            ylim = (-0.1, 0.7)

        if len(x_all) > 0 and len(y_all) > 0:
            xlim = (
                min(float(xlim[0]), float(np.amin(x_all))),
                max(float(xlim[1]), float(np.amax(x_all))),
            )
            ylim = (
                min(float(ylim[0]), float(np.amin(y_all))),
                max(float(ylim[1]), float(np.amax(y_all))),
            )

        if (
            isinstance(self._bins, (tuple, list))
            and len(self._bins) == 2
            and isinstance(self._bins[0], np.ndarray)
        ):
            h_total, xedges, yedges = np.histogram2d(
                x_all, y_all, bins=self._bins
            )
            xcenters = xedges[:-1] + ((xedges[1:] - xedges[:-1]) / 2.0)
            ycenters = yedges[:-1] + ((yedges[1:] - yedges[:-1]) / 2.0)
            return h_total, xedges, yedges, xcenters, ycenters

        aspect = (xlim[1] - xlim[0]) / max(ylim[1] - ylim[0], 1e-6)
        if isinstance(self._bins, (tuple, list)):
            bins_xy = (int(self._bins[0]), int(self._bins[1]))
        elif aspect > 1:
            bins_xy = (int(self._bins), max(int(self._bins / aspect), 1))
        else:
            bins_xy = (max(int(self._bins * aspect), 1), int(self._bins))

        h_total, xedges, yedges = np.histogram2d(
            x_all, y_all, bins=bins_xy, range=[xlim, ylim]
        )
        xcenters = xedges[:-1] + ((xedges[1:] - xedges[:-1]) / 2.0)
        ycenters = yedges[:-1] + ((yedges[1:] - yedges[:-1]) / 2.0)
        return h_total, xedges, yedges, xcenters, ycenters

    def _compute_clean_levels(
        self, h_grid: np.ndarray, vmax: float | None = None
    ) -> Any:
        if isinstance(self._levels, (list, tuple, np.ndarray)):
            return self._levels

        vmax = vmax if vmax is not None else np.nanmax(h_grid)
        if np.isnan(vmax) or vmax <= 0:
            return None
        if vmax <= 1:
            return np.array([1.0])

        # Clean noise: discard sparse 1-count Poisson noise by starting at max(2.0, vmax * 0.05)
        vmin = max(2.0, float(vmax) * 0.05)
        if vmax <= vmin:
            vmin = max(1.0, float(vmax) * 0.5)

        n_levels = max(int(self._levels), 2)
        if self._log_norm:
            levs = np.logspace(np.log10(vmin), np.log10(vmax), n_levels)
        else:
            levs = np.linspace(vmin, vmax, n_levels)
        return np.unique(levs)

    def _refresh(self, force_redraw: bool = True):
        self._remove_artists()
        if self._grouped_data:
            self._render_grouped_data()
            return

        if self._data is None or len(self._data) == 0:
            return

        x = self._data[:, 0]
        y = self._data[:, 1]
        h_total, xedges, yedges, xcenters, ycenters = self._compute_grid(x, y)
        self._histogram = (h_total, xedges, yedges)

        h_draw = h_total.astype(float)
        h_draw[h_draw <= 0] = np.nan

        levels = self._compute_clean_levels(h_draw)
        if levels is None:
            return

        if isinstance(self._colormap, mcolors.Colormap):
            cmap = self._colormap
        else:
            from ._utils import resolve_colormap_by_name

            cmap = resolve_colormap_by_name(self._colormap) or plt.get_cmap(
                self._colormap
            )

        cs = self.ax.contour(
            xcenters,
            ycenters,
            h_draw.T,
            levels=levels,
            linewidths=self._linewidths,
            cmap=cmap,
            norm="log" if self._log_norm else None,
        )
        self._tag_contour_set(cs, None)
        with contextlib.suppress(Exception):
            cs.set_visible(self._visible)
        self._contour_collections.append(cs)

        if self.ax.figure and self.ax.figure.canvas:
            self.ax.figure.canvas.draw_idle()

    def set_grouped_data(
        self,
        grouped_dict: dict[Any, tuple[np.ndarray, np.ndarray]],
        styles_dict: dict[Any, dict[str, Any]] | None = None,
        group_names: dict[Any, str] | None = None,
        show_legend: bool = True,
    ):
        """Render multi-group / multi-layer contours on a shared, pooled bin grid."""
        self._remove_artists()
        self._data = None
        if not grouped_dict:
            self._grouped_data = None
            return

        self._grouped_data = grouped_dict
        self._group_styles = styles_dict or {}
        self._group_names = group_names or {}
        self._show_legend = show_legend
        self._render_grouped_data()

    def _render_grouped_data(self):
        if not self._grouped_data:
            return

        valid_groups = {
            k: (np.asarray(gx), np.asarray(gy))
            for k, (gx, gy) in self._grouped_data.items()
            if len(gx) > 0 and len(gy) > 0
        }
        if not valid_groups:
            return

        # 1. Pool all points together to determine the common grid
        all_x = np.concatenate([gx for gx, _ in valid_groups.values()])
        all_y = np.concatenate([gy for _, gy in valid_groups.values()])
        h_total, xedges, yedges, xcenters, ycenters = self._compute_grid(
            all_x, all_y
        )
        self._histogram = (h_total, xedges, yedges)

        from ._utils import (
            ColormapLegendHandler,
            ColormapLegendProxy,
            make_solid_contour_cmap,
            normalize_rgb,
            resolve_colormap_by_name,
        )

        legend_handles = []
        legend_labels = []

        default_cmap = (
            self._colormap
            if isinstance(self._colormap, mcolors.Colormap)
            else resolve_colormap_by_name(self._colormap)
            or plt.get_cmap("jet")
        )

        group_keys = sorted(
            valid_groups.keys(), key=lambda k: (isinstance(k, str), k)
        )
        for idx, gid in enumerate(group_keys):
            gx, gy = valid_groups[gid]
            # 2. Bin each group onto the shared common grid
            h_g, _, _ = np.histogram2d(gx, gy, bins=[xedges, yedges])
            h_draw = h_g.astype(float)
            h_draw[h_draw <= 0] = np.nan

            vmax = np.nanmax(h_draw)
            levels = self._compute_clean_levels(h_draw, vmax)
            if levels is None:
                continue

            style = (self._group_styles or {}).get(gid, {})
            style_mode = style.get("mode", "colormap")

            if style_mode == "colormap":
                cmap_name = style.get("colormap")
                cmap = (
                    resolve_colormap_by_name(cmap_name)
                    if cmap_name
                    else default_cmap
                )
                if cmap is None:
                    cmap = default_cmap
                legend_handle = ColormapLegendProxy(
                    cmap,
                    self._linewidths,
                    style="categorical",
                    n_colors=max(
                        int(
                            self._levels
                            if isinstance(self._levels, int)
                            else len(self._levels)
                        ),
                        2,
                    ),
                )
            else:
                color = style.get("color")
                if color is None:
                    tab10 = plt.cm.tab10.colors
                    color = tab10[idx % len(tab10)]
                norm_color = normalize_rgb(color)
                cmap = make_solid_contour_cmap(f"solid_{gid}", norm_color)
                legend_handle = ColormapLegendProxy(
                    cmap,
                    self._linewidths,
                    style="categorical",
                    n_colors=max(
                        int(
                            self._levels
                            if isinstance(self._levels, int)
                            else len(self._levels)
                        ),
                        2,
                    ),
                )

            cs = self.ax.contour(
                xcenters,
                ycenters,
                h_draw.T,
                levels=levels,
                linewidths=self._linewidths,
                cmap=cmap,
                norm="log" if self._log_norm else None,
            )
            label = (self._group_names or {}).get(gid, str(gid))
            self._tag_contour_set(cs, label)
            with contextlib.suppress(Exception):
                cs.set_visible(self._visible)
            self._contour_collections.append(cs)

            legend_handles.append(legend_handle)
            legend_labels.append(label)

        if self._show_legend and legend_handles:
            self.ax.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc="upper right",
                frameon=False,
                handler_map={ColormapLegendProxy: ColormapLegendHandler()},
            )
            legend = self.ax.get_legend()
            if legend is not None:
                with contextlib.suppress(Exception):
                    legend.set_visible(self._visible)

        if self.ax.figure and self.ax.figure.canvas:
            self.ax.figure.canvas.draw_idle()


ContourArtist = Contour


class PhasorNavigationToolbar(NavigationToolbar2QT):
    """Custom navigation toolbar emitting Qt signals when Pan or Zoom are toggled."""

    zoom_toggled_signal: Signal = Signal(bool)
    pan_toggled_signal: Signal = Signal(bool)

    def __init__(
        self,
        canvas: FigureCanvasQTAgg,
        parent: QWidget | None = None,
        viewer: napari.Viewer | None = None,
    ):
        super().__init__(canvas, parent)
        self.viewer = viewer
        self.setIconSize(QSize(28, 28))
        for action in list(self.actions()):
            if action.text() in ("Subplots", "Customize"):
                self.removeAction(action)
        self._replace_toolbar_icons()

    def _get_path_to_icon(self) -> Path:
        icon_root = Path(__file__).parent / "icons"
        if self._napari_theme_has_light_bg():
            return icon_root / "black"
        return icon_root / "white"

    def _napari_theme_has_light_bg(self) -> bool:
        viewer = self.viewer
        if viewer is None and self.parentWidget() is not None:
            viewer = getattr(self.parentWidget(), "viewer", None)
        if viewer is not None:
            try:
                import napari.utils.theme

                theme_name = getattr(viewer, "theme", "dark")
                theme = napari.utils.theme.get_theme(theme_name)
                if hasattr(theme, "background") and hasattr(
                    theme.background, "as_hsl_tuple"
                ):
                    _, _, bg_lightness = theme.background.as_hsl_tuple()
                    return bg_lightness > 0.5
                if hasattr(theme, "type"):
                    return theme.type == "light"
            except Exception:  # noqa: BLE001
                pass
        return False

    def _replace_toolbar_icons(self) -> None:
        """Modify toolbar icons to match the napari theme."""
        icon_dir = self._get_path_to_icon()
        for action in self.actions():
            text = action.text()
            if text == "Pan":
                action.setToolTip(
                    "Pan/Zoom: Left button pans; Right button zooms; "
                    "Click once to activate; Click again to deactivate"
                )
            elif text == "Zoom":
                action.setToolTip(
                    "Zoom to rectangle; Click once to activate; "
                    "Click again to deactivate"
                )
            if len(text) > 0:
                icon_path = icon_dir / f"{text}.png"
                if icon_path.exists():
                    action.setIcon(QIcon(str(icon_path)))

    def _update_buttons_checked(self) -> None:
        """Update toggle tool icons when selected/unselected."""
        super()._update_buttons_checked()
        icon_dir = self._get_path_to_icon()

        if "pan" in self._actions:
            pan_action = self._actions["pan"]
            if pan_action.isChecked():
                checked_path = icon_dir / "Pan_checked.png"
                if checked_path.exists():
                    pan_action.setIcon(QIcon(str(checked_path)))
            else:
                normal_path = icon_dir / "Pan.png"
                if normal_path.exists():
                    pan_action.setIcon(QIcon(str(normal_path)))

        if "zoom" in self._actions:
            zoom_action = self._actions["zoom"]
            if zoom_action.isChecked():
                checked_path = icon_dir / "Zoom_checked.png"
                if checked_path.exists():
                    zoom_action.setIcon(QIcon(str(checked_path)))
            else:
                normal_path = icon_dir / "Zoom.png"
                if normal_path.exists():
                    zoom_action.setIcon(QIcon(str(normal_path)))

    def zoom(self, *args):
        super().zoom(*args)
        self.zoom_toggled_signal.emit(self.mode == "zoom rect")

    def pan(self, *args):
        super().pan(*args)
        self.pan_toggled_signal.emit(self.mode == "pan/zoom")


class SelectionToolbarWidget(QWidget):
    """Toolbar holding the exclusive selection tool buttons."""

    #: Button name paired with the shape drawn on its icon.
    TOOLS: tuple[tuple[str, str, str], ...] = (
        ("LASSO", "lasso", "Lasso selection tool"),
        ("ELLIPSE", "ellipse", "Ellipse selection tool"),
        ("RECTANGLE", "rectangle", "Rectangle selection tool"),
        ("BRUSH", "brush", "Brush tool"),
        ("ERASER", "eraser", "Eraser tool"),
    )

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.setLayout(layout)

        self.setStyleSheet(
            "QToolButton {"
            "  border: 1px solid transparent;"
            "  border-radius: 4px;"
            "  padding: 2px;"
            "  background: transparent;"
            "}"
            "QToolButton:hover {"
            "  border: 1px solid rgba(128, 128, 128, 0.4);"
            "  background-color: rgba(128, 128, 128, 0.15);"
            "}"
            "QToolButton:checked {"
            "  border: 1px solid rgba(0, 193, 140, 0.9);"
            "  background-color: rgba(0, 193, 140, 0.25);"
            "}"
        )

        self.buttons: dict[str, QToolButton] = {}

        for name, shape, tooltip in self.TOOLS:
            self._add_button(name, tooltip, shape)

    def _add_button(self, name: str, tooltip: str, shape: str):
        btn = QToolButton(self)
        btn.setToolTip(tooltip)
        btn.setCheckable(True)
        btn.setIcon(_make_selector_icon(shape))
        btn.setIconSize(QSize(20, 20))
        btn.setFixedSize(28, 28)
        btn.setCursor(Qt.PointingHandCursor)
        self.layout().addWidget(btn)
        self.buttons[name] = btn

    def update_theme(self, has_light_bg: bool = False):
        """Update selector icons to match theme background."""
        normal_color = "#3e3f41" if has_light_bg else "#ffffff"
        for name, shape, _tooltip in self.TOOLS:
            if name in self.buttons:
                self.buttons[name].setIcon(
                    _make_selector_icon(shape, normal_color=normal_color)
                )


class PhasorCanvasWidget(QWidget):
    """Main plotting canvas and selector container widget for napari-phasors.

    Provides a 100% compatible API surface for the previous `biaplotter.plotter.CanvasWidget`.
    """

    artist_changed_signal: Signal = Signal(str)
    selector_changed_signal: Signal = Signal(str)
    show_color_overlay_signal: Signal = Signal(bool)

    def __init__(
        self,
        napari_viewer: napari.Viewer,
        parent: QWidget | None = None,
        highlight_enabled: bool = False,
    ):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.highlight_enabled = highlight_enabled

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.toolbar = PhasorNavigationToolbar(
            self.canvas, parent=self, viewer=self.viewer
        )
        self.layout().addWidget(self.toolbar)

        self.selection_tools_layout = QHBoxLayout()
        self.selection_tools_layout.setContentsMargins(0, 0, 0, 0)
        self.selection_tools_layout.setSpacing(0)

        # Selection tools are now located in the Manual Selection tab.
        # self.selection_toolbar is kept headless for backward compatibility
        # and is intentionally not added to self.layout().
        self.selection_toolbar = SelectionToolbarWidget()

        has_light_bg = self.toolbar._napari_theme_has_light_bg()
        self.selection_toolbar.update_theme(has_light_bg)

        if (
            self.viewer is not None
            and hasattr(self.viewer, "events")
            and hasattr(self.viewer.events, "theme")
        ):
            self.viewer.events.theme.connect(self._on_napari_theme_changed)

        class _ClassSpinboxStub:
            def __init__(self):
                self.value = 1

        self.class_spinbox = _ClassSpinboxStub()

        self.layout().addWidget(self.canvas, 1)

        self.artists: dict[str, Any] = {
            "HISTOGRAM2D": Histogram2DArtist(self.axes),
            "SCATTER": ScatterArtist(self.axes),
            "CONTOUR": ContourArtist(self.axes),
        }
        self._active_artist_name: str = "HISTOGRAM2D"

        self.selectors: dict[str, BaseInteractiveSelector] = {
            "RECTANGLE": InteractiveRectangleSelector(self.axes, self),
            "ELLIPSE": InteractiveEllipseSelector(self.axes, self),
            "LASSO": InteractiveLassoSelector(self.axes, self),
            "BRUSH": InteractiveBrushSelector(self.axes, self),
            "ERASER": InteractiveBrushSelector(self.axes, self, erase=True),
        }
        self._active_selector_name: str | None = None

        for name, btn in self.selection_toolbar.buttons.items():
            btn.clicked.connect(
                lambda checked, n=name: self._on_selector_button_clicked(n)
            )

        self.toolbar.pan_toggled_signal.connect(self._on_navigation_toggled)
        self.toolbar.zoom_toggled_signal.connect(self._on_navigation_toggled)

        if hasattr(self.viewer, "bind_key"):
            self.viewer.bind_key("Escape", self._on_escape, overwrite=True)

        self._apply_theme()

    @property
    def active_artist(self) -> Any | None:
        if self._active_artist_name:
            return self.artists.get(self._active_artist_name)
        return None

    @active_artist.setter
    def active_artist(self, value: Any | None):
        if value is None or value == "" or value == "NONE":
            self._active_artist_name = ""
            for artist in self.artists.values():
                artist.visible = False
            self.figure.canvas.draw_idle()
            return

        target_name = None
        if isinstance(value, str):
            target_name = value.upper()
        else:
            for k, artist in self.artists.items():
                if artist is value:
                    target_name = k
                    break

        if target_name not in self.artists:
            self._active_artist_name = ""
            for artist in self.artists.values():
                artist.visible = False
            self.figure.canvas.draw_idle()
            return

        if target_name == self._active_artist_name:
            for k, artist in self.artists.items():
                artist.visible = k == target_name
            return

        self._active_artist_name = target_name

        for k, artist in self.artists.items():
            artist.visible = k == target_name

        self.artist_changed_signal.emit(target_name)
        self.figure.canvas.draw_idle()

    @property
    def active_artist_object(self) -> Any | None:
        return self.active_artist

    @property
    def brush_size(self) -> float:
        """Diameter, in screen pixels, shared by the brush and eraser."""
        brush = self.selectors.get("BRUSH")
        return brush.size_px if brush is not None else DEFAULT_BRUSH_SIZE_PX

    @brush_size.setter
    def brush_size(self, value: float):
        for name in ("BRUSH", "ERASER"):
            selector = self.selectors.get(name)
            if selector is not None:
                selector.size_px = value

    @property
    def active_selector(self) -> Any | None:
        if self._active_selector_name:
            return self.selectors.get(self._active_selector_name)
        return None

    @active_selector.setter
    def active_selector(self, value: Union[str, Any] | None):
        if value is None:
            self._deactivate_all_selectors()
            return

        target_name = None
        if isinstance(value, str):
            target_name = value.upper()
        elif hasattr(value, "name"):
            for k in self.selectors:
                if k in value.name.upper():
                    target_name = k
                    break

        if target_name in self.selectors:
            self._activate_selector(target_name)
        else:
            self._deactivate_all_selectors()

    def _activate_selector(self, name: str):
        if self.toolbar.mode:
            if self.toolbar.mode == "pan/zoom":
                self.toolbar.pan()
            elif self.toolbar.mode == "zoom rect":
                self.toolbar.zoom()

        for k, selector in self.selectors.items():
            if k == name:
                selector.create_selector()
                btn = self.selection_toolbar.buttons.get(k)
                if btn is not None and not btn.isChecked():
                    btn.blockSignals(True)
                    btn.setChecked(True)
                    btn.blockSignals(False)
            else:
                selector.remove()
                btn = self.selection_toolbar.buttons.get(k)
                if btn is not None and btn.isChecked():
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)

        self._active_selector_name = name
        self.canvas.setCursor(self.selectors[name].cursor())
        self.selector_changed_signal.emit(name)

    def _deactivate_all_selectors(self):
        for k, selector in self.selectors.items():
            selector.remove()
            btn = self.selection_toolbar.buttons.get(k)
            if btn is not None and btn.isChecked():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
        self._active_selector_name = None
        self.canvas.setCursor(QCursor(Qt.ArrowCursor))
        self.selector_changed_signal.emit("")

    # Backward-compatibility aliases
    _deactivate_and_remove_all_selectors = _deactivate_all_selectors
    _remove_all_selectors = _deactivate_all_selectors

    def _on_selector_button_clicked(self, name: str):
        btn = self.selection_toolbar.buttons.get(name)
        if btn is not None and btn.isChecked():
            self._activate_selector(name)
        else:
            self._deactivate_all_selectors()

    def _on_navigation_toggled(self, active: bool):
        if active:
            self._deactivate_all_selectors()

    def _on_escape(self, event=None):
        self._deactivate_all_selectors()
        self.figure.canvas.draw_idle()

    def _is_click_inside_axes(self, event) -> bool:
        if event is None or event.xdata is None or event.ydata is None:
            return False
        return event.inaxes == self.axes

    def active_selection_region(
        self,
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        selector = self.active_selector
        if selector is None or selector.last_geometry is None:
            return None
        geom = selector.last_geometry
        return lambda points: geom.contains_points(points)

    def _apply_theme(self):
        try:
            from napari.utils.theme import get_theme

            theme = get_theme(getattr(self.viewer, "theme", "dark"))
            bg_color = theme.canvas.as_hex()
            text_color = theme.text.as_hex()
        except (
            AttributeError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            bg_color = "#262930"
            text_color = "white"

        self.figure.patch.set_facecolor(bg_color)
        self.axes.set_facecolor(bg_color)
        for spine in self.axes.spines.values():
            spine.set_color(text_color)
        self.axes.tick_params(colors=text_color, which="both")
        self.axes.xaxis.label.set_color(text_color)
        self.axes.yaxis.label.set_color(text_color)

    def _on_napari_theme_changed(self, event: Any = None):
        """Update toolbar icons and canvas styling when napari theme changes."""
        has_light_bg = (
            self.toolbar._napari_theme_has_light_bg()
            if hasattr(self, "toolbar")
            and hasattr(self.toolbar, "_napari_theme_has_light_bg")
            else False
        )
        if hasattr(self, "toolbar") and hasattr(
            self.toolbar, "_replace_toolbar_icons"
        ):
            self.toolbar._replace_toolbar_icons()
        if hasattr(self, "selection_toolbar") and hasattr(
            self.selection_toolbar, "update_theme"
        ):
            self.selection_toolbar.update_theme(has_light_bg)
        self._apply_theme()
        self.figure.canvas.draw_idle()


# Backward-compatibility alias
CanvasWidget = PhasorCanvasWidget
