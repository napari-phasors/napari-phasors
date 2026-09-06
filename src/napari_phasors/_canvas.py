"""Native Matplotlib canvas and interactive selection tools for napari-phasors.

Replaces the unmaintained ``biaplotter`` and ``nap-plot-tools`` packages with a
clean, high-performance, and maintainable implementation tailored to phasor
analysis.
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
from qtpy.QtCore import QRectF, QSize, Qt
from qtpy.QtGui import QColor, QCursor, QIcon, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QHBoxLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari


# ---------------------------------------------------------------------------
# Default Categorical Colormaps (replaces nap-plot-tools dependency)
# ---------------------------------------------------------------------------

CAT10_MOD_HEX_COLORS = [
    "#e6e6fa",  # First color (e.g. background / class 0)
    "#ff7f0e",
    "#1f77b4",
    "#2ca02c",
    "#9400d3",
    "#afeeee",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#ccebc5",
    "#ffed6f",
    "#0054b6",
    "#6aa866",
    "#ffbfff",
    "#8d472a",
    "#417239",
    "#d48fd0",
    "#8b7e32",
    "#7989dc",
    "#f1d200",
    "#a1e9f6",
    "#924c28",
    "#dc797e",
    "#b86e85",
    "#79ea30",
    "#4723b9",
    "#3de658",
    "#de3ce7",
]


def _build_cat10_mod_colormaps() -> tuple[ListedColormap, ListedColormap]:
    """Create opaque and first-color-transparent versions of categorical colormap."""
    rgba_colors = [to_rgba(c) for c in CAT10_MOD_HEX_COLORS]
    opaque_cmap = ListedColormap(rgba_colors, name="cat10_modified")

    trans_colors = list(rgba_colors)
    trans_colors[0] = (
        rgba_colors[0][0],
        rgba_colors[0][1],
        rgba_colors[0][2],
        0.0,
    )
    first_trans_cmap = ListedColormap(
        trans_colors, name="cat10_modified_first_transparent"
    )
    return opaque_cmap, first_trans_cmap


cat10_mod_cmap, cat10_mod_cmap_first_transparent = _build_cat10_mod_colormaps()


# ---------------------------------------------------------------------------
# Selection Geometry
# ---------------------------------------------------------------------------


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

        return np.zeros(len(points), dtype=bool)


# ---------------------------------------------------------------------------
# Selection Toolbar Icons (Vector rendering with QPainter)
# ---------------------------------------------------------------------------


def _render_selector_pixmap(shape: str, color: str, size: int = 24) -> QPixmap:
    """Render a pixmap for a selector shape in a given color."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    pen = QPen(QColor(color))
    pen.setWidthF(1.8)
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

    painter.end()
    return pixmap


def _make_selector_icon(
    shape: str,
    normal_color: str = "#ffffff",
    checked_color: str = "#00c18c",
    size: int = 24,
) -> QIcon:
    """Render a crisp vector QIcon for rectangle, ellipse, or lasso selector with normal and checked states."""
    icon = QIcon()
    pixmap_off = _render_selector_pixmap(shape, normal_color, size)
    pixmap_on = _render_selector_pixmap(shape, checked_color, size)
    icon.addPixmap(pixmap_off, QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(pixmap_on, QIcon.Mode.Normal, QIcon.State.On)
    return icon


# ---------------------------------------------------------------------------
# Interactive Selectors
# ---------------------------------------------------------------------------


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

    def remove(self):
        """Disconnect and clear the selector widget from axes."""
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
        self.canvas_widget.canvas.mpl_connect(
            "button_press_event", self._on_button_press
        )

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
        self.canvas_widget.canvas.mpl_connect(
            "button_press_event", self._on_button_press
        )

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


# ---------------------------------------------------------------------------
# Artists
# ---------------------------------------------------------------------------


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
        overlay_colormap: Any = cat10_mod_cmap_first_transparent,
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
        if indices is None or self._data is None or self._histogram is None:
            self._remove_artists(["overlay_histogram_image"])
            return

        indices = np.asarray(indices)
        if indices.ndim == 0:
            if indices == 0:
                self._remove_artists(["overlay_histogram_image"])
                return
            indices = np.full(len(self._data), indices, dtype=np.int32)
        elif len(indices) != len(self._data):
            self._remove_artists(["overlay_histogram_image"])
            return

        non_zero = indices > 0
        if not np.any(non_zero):
            self._remove_artists(["overlay_histogram_image"])
            return

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
            self._remove_artists(["overlay_histogram_image"])
            return

        grid = np.zeros((nx, ny), dtype=np.int32)
        np.maximum.at(grid, (x_idx[valid], y_idx[valid]), vals[valid])

        overlay_cmap = (
            self._overlay_colormap
            if isinstance(self._overlay_colormap, mcolors.Colormap)
            else plt.get_cmap(self._overlay_colormap)
        )

        grid_t = grid.T
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
        overlay_colormap: Any = cat10_mod_cmap,
    ):
        self.ax = ax
        self._data: np.ndarray | None = None
        self._size = size
        self._alpha = alpha
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
        scatter = self._mpl_artists.pop("scatter", None)
        if scatter is not None:
            with contextlib.suppress(Exception):
                scatter.remove()

    def _refresh(self, force_redraw: bool = True):
        if self._data is None or len(self._data) == 0:
            self._remove_artists()
            return

        scatter = self._mpl_artists.get("scatter")
        if scatter is None or scatter.axes != self.ax or force_redraw:
            self._remove_artists()
            self._mpl_artists["scatter"] = self.ax.scatter(
                self._data[:, 0],
                self._data[:, 1],
                s=self._size,
                alpha=self._alpha,
                zorder=1,
                visible=self._visible,
            )
        else:
            scatter.set_offsets(self._data)
            scatter.set_visible(self._visible)

        if self._color_indices is not None:
            self._colorize(self._color_indices)

    def _colorize(self, indices: np.ndarray | None):
        scatter = self._mpl_artists.get("scatter")
        if scatter is None or self._data is None:
            return

        if indices is None:
            scatter.set_color("#1f77b4")
            return

        indices = np.asarray(indices)
        if indices.ndim == 0:
            if indices == 0:
                scatter.set_color("#1f77b4")
                return
            indices = np.full(len(self._data), indices, dtype=np.int32)
        elif len(indices) != len(self._data):
            scatter.set_color("#1f77b4")
            return

        cmap = (
            self._overlay_colormap
            if isinstance(self._overlay_colormap, mcolors.Colormap)
            else plt.get_cmap(self._overlay_colormap)
        )
        max_idx = max(int(np.nanmax(indices)), 1)
        norm = Normalize(vmin=0, vmax=max_idx)
        colors = cmap(norm(indices))
        scatter.set_color(colors)


ScatterArtist = Scatter


# ---------------------------------------------------------------------------
# Navigation Toolbar with Signals
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Selection Toolbar Widget
# ---------------------------------------------------------------------------


class SelectionToolbarWidget(QWidget):
    """Toolbar holding exclusive selection tool buttons (Lasso, Ellipse, Rectangle)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.setLayout(layout)

        self.buttons: dict[str, QToolButton] = {}

        self._add_button("LASSO", "Lasso selection tool", "lasso")
        self._add_button("ELLIPSE", "Ellipse selection tool", "ellipse")
        self._add_button("RECTANGLE", "Rectangle selection tool", "rectangle")

    def _add_button(self, name: str, tooltip: str, shape: str):
        btn = QToolButton(self)
        btn.setToolTip(tooltip)
        btn.setCheckable(True)
        btn.setIcon(_make_selector_icon(shape))
        btn.setIconSize(QSize(20, 20))
        btn.setCursor(Qt.PointingHandCursor)
        self.layout().addWidget(btn)
        self.buttons[name] = btn

    def update_theme(self, has_light_bg: bool = False):
        """Update selector icons to match theme background."""
        normal_color = "#3e3f41" if has_light_bg else "#ffffff"
        for name, shape in (
            ("LASSO", "lasso"),
            ("ELLIPSE", "ellipse"),
            ("RECTANGLE", "rectangle"),
        ):
            if name in self.buttons:
                self.buttons[name].setIcon(
                    _make_selector_icon(shape, normal_color=normal_color)
                )


# ---------------------------------------------------------------------------
# Phasor Canvas Widget
# ---------------------------------------------------------------------------


class PhasorCanvasWidget(QWidget):
    """Main plotting canvas and selector container widget for napari-phasors.

    Provides a 100% compatible API surface for the previous `biaplotter.plotter.CanvasWidget`,
    while eliminating all upstream monkey patches, memory leaks, and performance bottlenecks.
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
        self.selection_tools_layout.setContentsMargins(4, 2, 4, 2)
        self.selection_tools_layout.setSpacing(6)

        self.selection_toolbar = SelectionToolbarWidget(self)
        self.selection_tools_layout.addWidget(self.selection_toolbar)

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

        self.layout().addLayout(self.selection_tools_layout)
        self.layout().addWidget(self.canvas, 1)

        self.artists: dict[str, Any] = {
            "HISTOGRAM2D": Histogram2DArtist(self.axes),
            "SCATTER": ScatterArtist(self.axes),
        }
        self._active_artist_name: str = "HISTOGRAM2D"

        self.selectors: dict[str, BaseInteractiveSelector] = {
            "RECTANGLE": InteractiveRectangleSelector(self.axes, self),
            "ELLIPSE": InteractiveEllipseSelector(self.axes, self),
            "LASSO": InteractiveLassoSelector(self.axes, self),
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
        self.canvas.setCursor(QCursor(Qt.CrossCursor))
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
        except (AttributeError, KeyError, RuntimeError, TypeError):
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
