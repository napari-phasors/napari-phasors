"""Time-lapse (stack) support for the phasor plot and its histogram.

Phasor data is stored per layer as ``G``/``S`` arrays shaped
``(harmonic, *layer.data.shape)``.  For a time-lapse acquisition
``layer.data`` has one or more leading non-spatial axes (e.g. ``(T, Y, X)``),
and every plot consumer in the plugin historically flattened *all* of them,
pooling every timepoint into a single phasor cloud.

This module adds the notion of a *current frame*:

* :class:`FrameContext` is the single source of truth for which frame is
  displayed.  It stays in sync with the napari dims sliders in both
  directions, so dragging (or playing) the viewer slider moves the phasor
  plot and vice versa.
* :class:`TimelapseControlBar` is the compact row of controls shown beneath
  the phasor plot.  It is hidden entirely when the selected layers have no
  non-spatial axis, so 2-D workflows are unaffected.
* :class:`AnimationExportDialog` and :func:`export_animation` render the
  phasor plot and/or the 1-D histogram frame by frame into an animated GIF.
* :func:`build_frame_statistics_rows` produces the per-frame statistics rows
  used by the CSV exporters.

"""

import contextlib

import numpy as np
from napari.utils import notifications
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ._utils import compute_dataset_statistics

#: Every frame is pooled into a single phasor cloud (historic behaviour).
POOLED = "pooled"
#: Only the samples of the current frame are displayed and summarised.
CURRENT = "current"

#: Human-readable labels for the two modes, used by the mode combobox.
MODE_LABELS = {
    POOLED: "All timepoints",
    CURRENT: "Current timepoint",
}


def stack_axes(layer):
    """Return the non-spatial (stack) axes of *layer*.

    The last two axes of an image layer are always spatial; anything before
    them is a stack axis the user can step through (time, Z, channel, ...).

    Parameters
    ----------
    layer : napari.layers.Image
        Layer to inspect.

    Returns
    -------
    list of int
        Layer-axis indices, empty for plain 2-D data.
    """
    data = getattr(layer, "data", None)
    if data is None:
        return []
    return list(range(max(0, data.ndim - 2)))


class FrameContext(QObject):
    """Track which frame of a time-lapse the phasor plot is showing.

    The context owns the display *mode* (pooled or per-frame), the layer axis
    treated as the frame axis, and the current frame index.  It mirrors the
    napari dims slider so that the viewer and the phasor plot never disagree
    about which timepoint is on screen.

    Parameters
    ----------
    viewer : napari.Viewer
        Viewer whose dims sliders drive (and are driven by) the frame index.
    layers_provider : callable
        Zero-argument callable returning the currently selected image layers.
        Passed as a callable rather than a list so the context always sees
        the live selection.
    parent : QObject, optional
        Qt parent.
    """

    #: Emitted with the new frame index whenever the current frame changes.
    frameChanged = Signal(int)
    #: Emitted with :data:`POOLED` or :data:`CURRENT` when the mode changes.
    modeChanged = Signal(str)
    #: Emitted with the new layer axis when the frame axis changes.
    axisChanged = Signal(int)

    def __init__(self, viewer, layers_provider, parent=None):
        """Create a pooled-by-default context bound to *viewer*."""
        super().__init__(parent)
        self.viewer = viewer
        self._layers_provider = layers_provider
        self._mode = POOLED
        self._axis = 0
        self._index = 0
        self._n_frames = 1
        # Guards the two-way dims sync against feedback loops.
        self._syncing = False

        with contextlib.suppress(AttributeError, RuntimeError):
            self.viewer.dims.events.current_step.connect(
                self._on_dims_current_step
            )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def mode(self):
        """Current display mode, either :data:`POOLED` or :data:`CURRENT`."""
        return self._mode

    @mode.setter
    def mode(self, value):
        """Set the display mode, emitting :attr:`modeChanged` on a change."""
        value = CURRENT if value == CURRENT else POOLED
        if value == self._mode:
            return
        self._mode = value
        if value == CURRENT:
            self.sync_from_dims()
        self.modeChanged.emit(value)

    @property
    def is_per_frame(self):
        """True when only the current frame should be displayed."""
        return self._mode == CURRENT

    @property
    def axis(self):
        """Layer axis treated as the frame axis."""
        return self._axis

    @axis.setter
    def axis(self, value):
        """Set the frame axis, resyncing the index from the viewer."""
        value = int(value)
        if value == self._axis:
            return
        self._axis = value
        self.refresh_bounds()
        self.sync_from_dims()
        self.axisChanged.emit(value)

    @property
    def index(self):
        """Current frame index along :attr:`axis`."""
        return self._index

    @index.setter
    def index(self, value):
        """Set the current frame and push it to the napari dims slider."""
        value = int(np.clip(int(value), 0, max(0, self._n_frames - 1)))
        if value == self._index:
            return
        self._index = value
        self.push_to_dims()
        self.frameChanged.emit(value)

    @property
    def n_frames(self):
        """Number of frames along :attr:`axis` for the current selection."""
        return self._n_frames

    # ------------------------------------------------------------------
    # Layer / viewer introspection
    # ------------------------------------------------------------------

    def _layers(self):
        """Return the selected layers that carry array data."""
        try:
            layers = list(self._layers_provider() or [])
        except (AttributeError, RuntimeError, TypeError):
            return []
        return [
            layer
            for layer in layers
            if getattr(layer, "data", None) is not None
        ]

    def reference_layer(self):
        """Return the selected layer with the most dimensions, or None.

        Using the deepest layer means the axis choices stay available even
        when a plain 2-D layer is selected alongside a stack.
        """
        layers = self._layers()
        if not layers:
            return None
        return max(layers, key=lambda layer: layer.data.ndim)

    def available_axes(self):
        """Return the frame axes offered for the current selection."""
        layer = self.reference_layer()
        if layer is None:
            return []
        return stack_axes(layer)

    def axis_label(self, axis):
        """Return a display name for *axis*, preferring napari's dims label.

        Falls back to ``"Axis <n>"`` when the viewer has no meaningful label
        for that axis (napari defaults to labels like ``"-3"``).
        """
        world_axis = self._world_axis(axis)
        if world_axis is not None:
            with contextlib.suppress(AttributeError, IndexError, TypeError):
                label = str(self.viewer.dims.axis_labels[world_axis])
                if label and not label.lstrip("-").isdigit():
                    return label
        return f"Axis {axis}"

    def _world_axis(self, axis=None):
        """Map a layer axis to the corresponding napari world axis.

        napari right-aligns a layer's axes onto the world dims, so the offset
        is ``viewer.dims.ndim - layer.data.ndim``.
        """
        axis = self._axis if axis is None else axis
        layer = self.reference_layer()
        if layer is None:
            return None
        try:
            offset = self.viewer.dims.ndim - layer.data.ndim
        except (AttributeError, RuntimeError):
            return None
        world_axis = offset + axis
        if world_axis < 0 or world_axis >= self.viewer.dims.ndim:
            return None
        return world_axis

    def refresh_bounds(self):
        """Recompute :attr:`n_frames` and clamp the index into range.

        Returns
        -------
        bool
            True when the current selection has a usable frame axis.
        """
        axes = self.available_axes()
        if not axes:
            self._n_frames = 1
            self._index = 0
            return False

        if self._axis not in axes:
            self._axis = axes[0]

        layer = self.reference_layer()
        n_frames = int(layer.data.shape[self._axis])

        world_axis = self._world_axis()
        if world_axis is not None:
            with contextlib.suppress(AttributeError, IndexError, TypeError):
                n_frames = max(
                    n_frames, int(self.viewer.dims.nsteps[world_axis])
                )

        self._n_frames = max(1, n_frames)
        self._index = int(np.clip(self._index, 0, self._n_frames - 1))
        return True

    # ------------------------------------------------------------------
    # napari dims synchronisation
    # ------------------------------------------------------------------

    def disconnect_viewer(self):
        """Stop listening to the viewer's dims events (called on teardown)."""
        with contextlib.suppress(
            AttributeError, RuntimeError, TypeError, ValueError
        ):
            self.viewer.dims.events.current_step.disconnect(
                self._on_dims_current_step
            )

    def _on_dims_current_step(self, event=None):
        """Adopt the viewer's slider position as the current frame."""
        if self._syncing:
            return
        self.sync_from_dims()

    def sync_from_dims(self):
        """Pull the current frame index from the napari dims slider."""
        world_axis = self._world_axis()
        if world_axis is None:
            return
        try:
            step = int(self.viewer.dims.current_step[world_axis])
        except (AttributeError, IndexError, TypeError):
            return

        self.refresh_bounds()
        step = int(np.clip(step, 0, max(0, self._n_frames - 1)))
        if step == self._index:
            return
        self._index = step
        self.frameChanged.emit(step)

    def push_to_dims(self):
        """Move the napari dims slider to the current frame index."""
        world_axis = self._world_axis()
        if world_axis is None:
            return
        self._syncing = True
        try:
            self.viewer.dims.set_current_step(world_axis, self._index)
        except (AttributeError, IndexError, RuntimeError, TypeError):
            pass
        finally:
            self._syncing = False

    # ------------------------------------------------------------------
    # Sample filtering
    # ------------------------------------------------------------------

    def axis_for_shape(self, shape):
        """Return the frame axis valid for *shape*, or None.

        The axis must exist and must not be one of the two trailing spatial
        axes of the array. Unlike :meth:`frame_mask` this ignores the display
        mode and the current index, so callers can iterate every frame of an
        array (e.g. to compute a range shared by all of them).
        """
        ndim = len(shape)
        if ndim < 3:
            return None
        axis = self._axis
        if axis < 0 or axis >= ndim - 2:
            return None
        if shape[axis] <= 1:
            return None
        return axis

    def _usable_axis_for(self, shape):
        """Return the frame axis to slice *shape* with, or None."""
        if not self.is_per_frame:
            return None
        return self.axis_for_shape(shape)

    def frame_mask(self, shape):
        """Return a broadcastable mask selecting the current frame.

        Parameters
        ----------
        shape : tuple of int
            Shape of the array to be filtered (spatial shape of a layer).

        Returns
        -------
        np.ndarray or None
            A boolean array broadcastable against *shape* that is True only
            on the current frame, or None when no filtering applies (pooled
            mode, 2-D data, or an axis that does not exist for this shape).
        """
        axis = self._usable_axis_for(shape)
        if axis is None:
            return None
        index = int(np.clip(self._index, 0, shape[axis] - 1))
        broadcast_shape = [1] * len(shape)
        broadcast_shape[axis] = shape[axis]
        return np.arange(shape[axis]).reshape(broadcast_shape) == index

    def flat_frame_mask(self, shape):
        """Return the current-frame mask flattened to match ``array.ravel()``.

        Parameters
        ----------
        shape : tuple of int
            Shape of the array whose ravelled form should be filtered.

        Returns
        -------
        np.ndarray or None
            1-D boolean array of ``np.prod(shape)`` elements, or None when no
            filtering applies.
        """
        mask = self.frame_mask(shape)
        if mask is None:
            return None
        return np.broadcast_to(mask, shape).ravel()

    def filter_valid(self, valid, shape=None):
        """Restrict a validity mask to the current frame.

        Parameters
        ----------
        valid : np.ndarray
            Boolean mask, either shaped like the data or already ravelled.
        shape : tuple of int, optional
            Data shape, required when *valid* is already flat.

        Returns
        -------
        np.ndarray
            *valid* itself in pooled mode, otherwise a copy restricted to the
            current frame.
        """
        shape = tuple(valid.shape) if shape is None else tuple(shape)
        if valid.ndim == 1 and len(shape) > 1:
            mask = self.flat_frame_mask(shape)
        else:
            mask = self.frame_mask(shape)
        if mask is None:
            return valid
        return valid & mask

    def slice_array(self, array):
        """Return *array* reduced to the current frame.

        Used for the per-layer scalar arrays that feed the 1-D histogram,
        which are shaped exactly like ``layer.data``.

        Parameters
        ----------
        array : np.ndarray
            Array shaped like the layer data.

        Returns
        -------
        np.ndarray
            The current frame in per-frame mode, otherwise *array* unchanged.
        """
        array = np.asarray(array)
        axis = self._usable_axis_for(array.shape)
        if axis is None:
            return array
        index = int(np.clip(self._index, 0, array.shape[axis] - 1))
        return np.take(array, index, axis=axis)

    def state_key(self):
        """Return a hashable snapshot of the frame state, for cache keys."""
        if not self.is_per_frame:
            return (POOLED,)
        return (CURRENT, self._axis, self._index)


def slice_datasets(frame_context, datasets):
    """Restrict a ``{name: array}`` mapping to the current frame.

    Convenience wrapper used by the analysis tabs, which all feed the 1-D
    histogram one array per layer shaped like ``layer.data``.

    Parameters
    ----------
    frame_context : FrameContext or None
        Context to slice with. ``None`` (or pooled mode) returns *datasets*
        unchanged, so callers need no branching of their own.
    datasets : dict
        ``{name: np.ndarray}`` mapping.

    Returns
    -------
    dict
        Either *datasets* itself or a new mapping of current-frame slices.
    """
    if frame_context is None or not frame_context.is_per_frame:
        return datasets
    return {
        name: frame_context.slice_array(data)
        for name, data in datasets.items()
    }


class TimelapseControlBar(QWidget):
    """Compact controls for how a time-lapse is shown in the phasor plot.

    Deliberately minimal: a mode selector, a frame-axis selector (only when
    the data has more than one stack axis) and an export button. Stepping
    through frames, playback and the frame readout are left to napari's own
    dimension slider, which this bar's :class:`FrameContext` stays in sync
    with — duplicating them here would just be a second set of controls for
    the same state. The whole bar hides itself when the current layer
    selection has no stack axis.

    Parameters
    ----------
    frame_context : FrameContext
        Context this bar drives.
    parent : QWidget, optional
        Qt parent.
    """

    #: Emitted when the user clicks the "Export…" button.
    exportRequested = Signal()

    def __init__(self, frame_context, parent=None):
        """Build the control row bound to *frame_context*."""
        super().__init__(parent)
        self.frame_context = frame_context
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.mode_label = QLabel("Frames:")
        layout.addWidget(self.mode_label)

        self.mode_combobox = QComboBox()
        self.mode_combobox.addItem(MODE_LABELS[POOLED], POOLED)
        self.mode_combobox.addItem(MODE_LABELS[CURRENT], CURRENT)
        self.mode_combobox.setToolTip(
            "Show every timepoint pooled into one phasor plot, or only the "
            "timepoint currently displayed in the viewer."
        )
        layout.addWidget(self.mode_combobox)

        self.axis_label = QLabel("Axis:")
        layout.addWidget(self.axis_label)
        self.axis_combobox = QComboBox()
        self.axis_combobox.setToolTip(
            "Which dimension to step through when showing one timepoint."
        )
        layout.addWidget(self.axis_combobox)

        self.export_button = QPushButton("Export Animation…")
        self.export_button.setToolTip(
            "Export the phasor plot and/or histogram as an animated GIF"
        )
        layout.addWidget(self.export_button)

        layout.addStretch(1)

        self.mode_combobox.currentIndexChanged.connect(self._on_mode_changed)
        self.axis_combobox.currentIndexChanged.connect(self._on_axis_changed)
        self.export_button.clicked.connect(self.exportRequested)

        frame_context.modeChanged.connect(self._on_context_mode_changed)

        self.refresh()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self):
        """Rebuild the axis choices from the context.

        Hides the whole bar when the selection has no stack axis, which keeps
        the plain 2-D layout untouched.
        """
        ctx = self.frame_context
        axes = ctx.available_axes()

        if not axes:
            self.setVisible(False)
            return

        self.setVisible(True)
        ctx.refresh_bounds()

        self._updating = True
        try:
            self.axis_combobox.clear()
            for axis in axes:
                self.axis_combobox.addItem(ctx.axis_label(axis), axis)
            axis_index = axes.index(ctx.axis) if ctx.axis in axes else 0
            self.axis_combobox.setCurrentIndex(axis_index)
            # A single stack axis needs no picker.
            show_axis_picker = len(axes) > 1
            self.axis_label.setVisible(show_axis_picker)
            self.axis_combobox.setVisible(show_axis_picker)

            mode_index = self.mode_combobox.findData(ctx.mode)
            if mode_index >= 0:
                self.mode_combobox.setCurrentIndex(mode_index)
        finally:
            self._updating = False

        self._update_enabled_state()

    def _update_enabled_state(self):
        """Animations can only be exported frame by frame."""
        self.export_button.setEnabled(self.frame_context.is_per_frame)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_mode_changed(self, _index):
        """Push the selected mode into the context."""
        if self._updating:
            return
        self.frame_context.mode = self.mode_combobox.currentData()

    def _on_axis_changed(self, _index):
        """Push the selected frame axis into the context."""
        if self._updating:
            return
        axis = self.axis_combobox.currentData()
        if axis is not None:
            self.frame_context.axis = int(axis)
            self.refresh()

    def _on_context_mode_changed(self, _mode):
        """Mirror an externally driven mode change onto the combobox."""
        self.refresh()


class AnimationExportDialog(QDialog):
    """Ask what to render into the exported GIF.

    Parameters
    ----------
    n_frames : int
        Number of frames available in the current stack.
    histogram_available : bool
        Whether a 1-D histogram with data is currently shown.
    fps : int, optional
        Initial frame rate, by default 5.
    parent : QWidget, optional
        Qt parent.
    """

    def __init__(self, n_frames, histogram_available, fps=5, parent=None):
        """Build the export options dialog."""
        super().__init__(parent)
        self.setWindowTitle("Export Time-lapse Animation")

        layout = QFormLayout(self)

        self.phasor_checkbox = QCheckBox("Phasor plot")
        self.phasor_checkbox.setChecked(True)
        layout.addRow(self.phasor_checkbox)

        self.histogram_checkbox = QCheckBox("Histogram")
        self.histogram_checkbox.setEnabled(histogram_available)
        self.histogram_checkbox.setChecked(False)
        if not histogram_available:
            self.histogram_checkbox.setToolTip(
                "No histogram with data is currently displayed."
            )
        layout.addRow(self.histogram_checkbox)

        self.first_spinbox = QSpinBox()
        self.first_spinbox.setRange(1, max(1, n_frames))
        self.first_spinbox.setValue(1)
        layout.addRow("First frame:", self.first_spinbox)

        self.last_spinbox = QSpinBox()
        self.last_spinbox.setRange(1, max(1, n_frames))
        self.last_spinbox.setValue(max(1, n_frames))
        layout.addRow("Last frame:", self.last_spinbox)

        self.fps_spinbox = QDoubleSpinBox()
        self.fps_spinbox.setRange(0.5, 60.0)
        self.fps_spinbox.setSingleStep(0.5)
        self.fps_spinbox.setValue(float(fps))
        layout.addRow("Frame rate (fps):", self.fps_spinbox)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_options(self):
        """Return the chosen options as a dictionary.

        Returns
        -------
        dict
            Keys ``include_phasor``, ``include_histogram``, ``frames``
            (a list of 0-based frame indices) and ``fps``.
        """
        first = self.first_spinbox.value() - 1
        last = self.last_spinbox.value() - 1
        if last < first:
            first, last = last, first
        return {
            "include_phasor": self.phasor_checkbox.isChecked(),
            "include_histogram": self.histogram_checkbox.isChecked(),
            "frames": list(range(first, last + 1)),
            "fps": self.fps_spinbox.value(),
        }


def figure_to_rgb(figure):
    """Render a Matplotlib figure to an ``(H, W, 3)`` uint8 array.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to render. Its canvas is drawn synchronously.

    Returns
    -------
    np.ndarray or None
        RGB image, or None when the figure has no usable canvas.
    """
    canvas = getattr(figure, "canvas", None)
    if canvas is None:
        return None
    canvas.draw()
    if hasattr(canvas, "buffer_rgba"):
        buffer = np.asarray(canvas.buffer_rgba())
    elif hasattr(canvas, "tostring_rgb"):  # pragma: no cover - old mpl
        width, height = canvas.get_width_height()
        buffer = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(
            height, width, 3
        )
    else:  # pragma: no cover - unexpected backend
        return None
    return np.ascontiguousarray(buffer[..., :3])


def _pad_to(image, height, width, fill=255):
    """Centre *image* inside a ``(height, width)`` canvas of colour *fill*."""
    if image.shape[0] == height and image.shape[1] == width:
        return image
    padded = np.full((height, width, 3), fill, dtype=np.uint8)
    top = (height - image.shape[0]) // 2
    left = (width - image.shape[1]) // 2
    padded[top : top + image.shape[0], left : left + image.shape[1]] = image
    return padded


def combine_frames(images):
    """Stack rendered figures side by side into a single frame.

    Images of differing heights are padded (not scaled) so that no frame is
    resampled.

    Parameters
    ----------
    images : list of np.ndarray
        RGB arrays to combine, left to right.

    Returns
    -------
    np.ndarray or None
        The combined frame, or None when *images* is empty.
    """
    images = [image for image in images if image is not None]
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    height = max(image.shape[0] for image in images)
    padded = [_pad_to(image, height, image.shape[1]) for image in images]
    return np.hstack(padded)


def export_animation(path, frames, fps):
    """Write rendered *frames* to an animated GIF.

    ``imageio`` is imported lazily — it ships with napari but is not a hard
    dependency of this plugin, mirroring how ``_batch_analysis`` handles it.

    Parameters
    ----------
    path : str
        Output file path.
    frames : list of np.ndarray
        RGB frames, all of the same shape.
    fps : float
        Frame rate of the resulting animation.

    Returns
    -------
    bool
        True on success, False when the file could not be written (a napari
        error notification is shown in that case).
    """
    if not frames:
        notifications.show_error("No frames were rendered; nothing to export.")
        return False

    try:
        import imageio.v3 as iio
    except ImportError:
        notifications.show_error(
            "Exporting animations requires the 'imageio' package. "
            "Install it with: pip install imageio"
        )
        return False

    height = max(frame.shape[0] for frame in frames)
    width = max(frame.shape[1] for frame in frames)
    padded = [_pad_to(frame, height, width) for frame in frames]

    try:
        iio.imwrite(
            path,
            np.stack(padded),
            duration=1000.0 / max(0.5, float(fps)),
            loop=0,
        )
    except (OSError, ValueError, TypeError) as exc:
        notifications.show_error(f"Could not write the animation: {exc}")
        return False
    return True


def build_frame_statistics_rows(
    datasets, frame_context, bin_edges=None, bin_centers=None
):
    """Compute per-frame statistics rows for a set of named datasets.

    Each dataset is sliced along the frame axis and summarised with the same
    maths the on-screen statistics table uses
    (:func:`napari_phasors._utils.compute_dataset_statistics`), so exported
    numbers always match what is displayed.

    Parameters
    ----------
    datasets : dict
        ``{name: np.ndarray}`` mapping labels to arrays shaped like the
        corresponding layer data.
    frame_context : FrameContext
        Provides the frame axis. The context's own index is not used — every
        frame is visited.
    bin_edges, bin_centers : np.ndarray, optional
        Histogram binning used for the centre-of-mass column. When omitted,
        the centre of mass falls back to the mean.

    Returns
    -------
    list of dict
        One row per (frame, dataset) with keys ``Frame``, ``Name``,
        ``Center of Mass``, ``Mean``, ``Median`` and ``Std Dev``.
    """
    rows = []
    for name, data in datasets.items():
        array = np.asarray(data)
        axis = None
        if array.ndim >= 3:
            candidate = frame_context.axis
            if 0 <= candidate < array.ndim - 2 and array.shape[candidate] > 1:
                axis = candidate

        if axis is None:
            stats = compute_dataset_statistics(
                array, bin_centers=bin_centers, bin_edges=bin_edges
            )
            rows.append({"Frame": 0, "Name": name, **stats})
            continue

        for frame in range(array.shape[axis]):
            stats = compute_dataset_statistics(
                np.take(array, frame, axis=axis),
                bin_centers=bin_centers,
                bin_edges=bin_edges,
            )
            rows.append({"Frame": frame, "Name": name, **stats})

    rows.sort(key=lambda row: (row["Frame"], str(row["Name"])))
    return rows
