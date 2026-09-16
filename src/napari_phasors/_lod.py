"""Level-of-detail control for large phasor images.

A phasor image with tens of millions of pixels is slow in every direction at
once: filtering it takes seconds, the 2D histogram behind the phasor plot has
to bin every pixel, and each analysis tab walks the whole array again. Yet at
a zoomed-out view the screen cannot show more than a couple of million pixels
anyway, so most of that work is discarded by the display.

This module trades resolution for speed the way a map application does. A
layer is shown and analysed at a coarse bin factor while it is viewed as a
whole, and when the user zooms in, the *visible region only* is recomputed at
a finer factor. Zoom in far enough and that region reaches full resolution.
Because :mod:`~napari_phasors._binning` bins photon-weighted, a coarse level
is not an approximation of the phasor data: it is exactly the phasor of the
summed signal over each block.

The full-resolution arrays are never discarded. They stay in the
:class:`~napari_phasors._binning.PhasorPyramid`, which is what makes zooming
reversible: any level, and full resolution, can be rebuilt at any time.

What changes when a level is applied
------------------------------------

The pristine ``original_mean`` / ``G_original`` / ``S_original`` entries in
the layer's metadata are swapped for the current level's arrays, and the
layer's stored filter and threshold are re-run on top of them. Everything
downstream -- the phasor plot, the filter tab, the components, FRET and
mapping tabs -- reads those same entries, so all of it follows the level
automatically without needing to know this module exists.

``scale`` and ``translate`` are adjusted so the layer keeps covering the same
ground in world coordinates. A level swap therefore does not move the image
under the camera, and napari's own zoom is left alone.
"""

import contextlib

import numpy as np
from napari.utils.notifications import show_error
from qtpy.QtCore import QObject, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QToggleSwitch

from ._binning import (
    DEFAULT_PIXEL_BUDGET,
    MAX_BIN_FACTOR,
    PhasorPyramid,
    bin_factor_for_shape,
)
from ._utils import (
    assign_filter_and_threshold,
    compute_filter_and_threshold,
    make_section,
)

__all__ = [
    "PhasorLod",
    "LodManager",
    "LodSettingsWidget",
    "layer_supports_lod",
    "manual_bin_factors",
    "viewer_camera",
    "viewer_canvas_size",
]

#: Metadata key holding a layer's :class:`PhasorLod`.
METADATA_KEY = "lod"

#: How long the camera must stay still before a refinement runs, in
#: milliseconds. Long enough that a continuous pinch-zoom or scroll only
#: triggers one recompute at the end of the gesture.
REFINE_DEBOUNCE_MS = 250

#: Combo-box entry meaning "pick the factor from the pixel budget".
AUTO_BINNING = "Auto"


def manual_bin_factors(max_factor=MAX_BIN_FACTOR):
    """Return the bin factors a user may pick, coarsest last.

    Powers of two from ``2`` up to *max_factor*: the same ladder
    :func:`~napari_phasors._binning.bin_factor_for_shape` chooses from, so a
    manual choice can never ask for a level ``Auto`` could not have produced.
    ``1`` is not offered: full resolution is what turning binning off gives.
    """
    factors = []
    factor = 2
    while factor <= int(max_factor):
        factors.append(factor)
        factor *= 2
    return tuple(factors)


def layer_supports_lod(layer):
    """Return whether *layer* carries the full-resolution phasor arrays."""
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    return all(
        key in metadata
        for key in ("original_mean", "G_original", "S_original")
    )


def viewer_camera(viewer):
    """Return *viewer*'s camera, across napari versions.

    napari 0.9 moved the camera to ``viewer.scene.camera``. ``viewer.camera``
    still resolves there, but only by way of a deprecation warning, so the
    new location is tried first.

    Returns
    -------
    object or None
        ``None`` for a viewer that exposes no camera at all.
    """
    camera = getattr(getattr(viewer, "scene", None), "camera", None)
    if camera is not None:
        return camera
    return getattr(viewer, "camera", None)


def viewer_canvas_size(viewer):
    """Return the canvas ``(height, width)`` in pixels, or ``None``.

    napari 0.9 replaced the private ``viewer._canvas_size`` with
    ``viewer.canvas.size``. Both follow the NumPy height-by-width order.
    Viewers older than 0.9 have no ``canvas`` attribute at all, so there is
    nothing to confuse the modern lookup with.

    Returns
    -------
    tuple of float or None
        ``None`` when the size cannot be determined, which is the caller's
        signal to skip refinement rather than guess a viewport.
    """
    size = getattr(getattr(viewer, "canvas", None), "size", None)
    if size is None:
        size = getattr(viewer, "_canvas_size", None)
    if not size or len(size) < 2:
        return None
    return float(size[0]), float(size[1])


def _bin_mask(mask, factor):
    """Downsample a label mask by *factor*, keeping any labelled block.

    Block-maximum rather than an average or a mode: when zoomed out, a bin
    that contains any selected pixel stays selected. That errs towards
    showing data rather than hiding it, which is the safer mistake for a
    mask whose whole purpose is to mark a region of interest.
    """
    factor = max(1, int(factor))
    if factor == 1:
        return mask

    height, width = mask.shape[-2:]
    pad_y = (-height) % factor
    pad_x = (-width) % factor
    if pad_y or pad_x:
        pad = [(0, 0)] * (mask.ndim - 2) + [(0, pad_y), (0, pad_x)]
        mask = np.pad(mask, pad, mode="constant", constant_values=0)

    *lead, height, width = mask.shape
    return (
        mask.reshape(*lead, height // factor, factor, width // factor, factor)
        .max(axis=-3)
        .max(axis=-1)
    )


class PhasorLod:
    """Level-of-detail state for one phasor layer.

    Parameters
    ----------
    layer : napari.layers.Image
        A layer carrying full-resolution ``original_mean``, ``G_original``
        and ``S_original`` metadata.

    Attributes
    ----------
    factor : int
        Bin factor currently applied. ``1`` is full resolution.
    origin : tuple of int
        Top-left corner of the currently shown region, in full-resolution
        pixels. ``(0, 0)`` when the whole image is shown.
    """

    def __init__(self, layer):
        if not layer_supports_lod(layer):
            raise ValueError(
                f"Layer {getattr(layer, 'name', '?')!r} has no phasor data "
                "to build a level-of-detail pyramid from."
            )
        metadata = layer.metadata
        self.layer = layer
        self.pyramid = PhasorPyramid(
            metadata["original_mean"],
            metadata["G_original"],
            metadata["S_original"],
        )
        self.full_mask = metadata.get("mask")
        self.base_scale = np.asarray(layer.scale, dtype=float).copy()
        self.base_translate = np.asarray(layer.translate, dtype=float).copy()
        self.factor = 1
        self.origin = (0, 0)
        self._region = None

    @property
    def full_shape(self):
        """Full-resolution spatial shape, ``(Y, X)``."""
        return self.pyramid.shape

    @property
    def is_full_detail(self):
        """Whether the whole image is currently shown at full resolution."""
        return self.factor == 1 and self._region is None

    def suggested_factor(self, budget=DEFAULT_PIXEL_BUDGET):
        """Return the bin factor this layer would auto-select."""
        return bin_factor_for_shape(self.pyramid.full_shape, budget)

    def detach(self):
        """Restore the layer to full resolution and drop cached levels."""
        self.apply(1, region=None)
        self.pyramid.clear()

    def apply(self, factor, region=None):
        """Show the layer at *factor*, optionally cropped to *region*.

        Parameters
        ----------
        factor : int
            Bin factor. ``1`` is full resolution.
        region : tuple of int, optional
            ``(row_start, row_stop, col_start, col_stop)`` in
            full-resolution pixels. ``None`` shows the whole image.

        Returns
        -------
        bool
            Whether anything changed. ``False`` means the requested level was
            already on screen and no work was done.
        """
        factor = max(1, int(factor))

        if region is None:
            arrays = self.pyramid.level(factor)
            origin = (0, 0)
            bounds = None
        else:
            bounds = self.pyramid.region_bounds(factor, *region)
            arrays, origin = self.pyramid.region(factor, *region)

        if factor == self.factor and bounds == self._region:
            return False

        mean, real, imag = arrays
        metadata = self.layer.metadata
        metadata["original_mean"] = mean
        metadata["G_original"] = real
        metadata["S_original"] = imag

        if self.full_mask is not None:
            mask = self.full_mask
            if bounds is not None:
                mask = mask[..., bounds[0] : bounds[1], bounds[2] : bounds[3]]
            metadata["mask"] = _bin_mask(mask, factor)

        self.factor = factor
        self.origin = origin
        self._region = bounds

        self._reapply_filter()
        self._place_in_world()
        return True

    def refine_to(self, region, budget=DEFAULT_PIXEL_BUDGET):
        """Show *region* at the finest factor its pixel budget allows.

        Parameters
        ----------
        region : tuple of int
            ``(row_start, row_stop, col_start, col_stop)`` in
            full-resolution pixels.
        budget : int, optional
            Pixel budget for the refined region.

        Returns
        -------
        bool
            Whether the displayed level changed.
        """
        row_start, row_stop, col_start, col_stop = region
        factor = self.pyramid.factor_for_region(
            row_start, row_stop, col_start, col_stop, budget=budget
        )

        # Once the request covers the whole image there is nothing to crop
        # to, and showing it as a region would only pin the layer to a
        # translate it does not need.
        height, width = self.full_shape
        covers_all = (
            row_start <= 0
            and col_start <= 0
            and row_stop >= height
            and col_stop >= width
        )
        return self.apply(factor, region=None if covers_all else region)

    def _reapply_filter(self):
        """Re-run the layer's stored filter and threshold at this level.

        The working ``G``/``S``/``data`` arrays have to be rebuilt from the
        new level, otherwise they would keep the previous level's shape and
        every consumer would disagree with the layer about its own size.
        """
        settings = self.layer.metadata.get("settings", {}) or {}
        filter_settings = settings.get("filter", {}) or {}

        arrays = compute_filter_and_threshold(
            self.layer,
            threshold=settings.get("threshold"),
            threshold_upper=settings.get("threshold_upper"),
            filter_method=filter_settings.get("method"),
            size=filter_settings.get("size"),
            repeat=filter_settings.get("repeat"),
            sigma=filter_settings.get("sigma"),
            levels=filter_settings.get("levels"),
        )
        assign_filter_and_threshold(
            self.layer,
            arrays,
            threshold=settings.get("threshold"),
            threshold_upper=settings.get("threshold_upper"),
            threshold_method=settings.get("threshold_method"),
            filter_method=filter_settings.get("method"),
            size=filter_settings.get("size"),
            repeat=filter_settings.get("repeat"),
            sigma=filter_settings.get("sigma"),
            levels=filter_settings.get("levels"),
        )

    def _place_in_world(self):
        """Keep the layer covering the same world extent at any level.

        Binning by ``f`` makes each pixel ``f`` times wider, and a region
        starts at its own offset, so scale and translate move together to
        cancel both out. Only the two spatial axes are touched; any leading
        axis keeps the scale it was given.
        """
        scale = self.base_scale.copy()
        scale[-2:] = scale[-2:] * self.factor

        translate = self.base_translate.copy()
        translate[-2:] = (
            translate[-2:] + np.asarray(self.origin) * self.base_scale[-2:]
        )

        self.layer.scale = scale
        self.layer.translate = translate


class LodManager(QObject):
    """Drives level of detail for phasor layers as the camera moves.

    Attaching a layer builds its pyramid and, when *auto* is set, drops it to
    a bin factor that brings it under the pixel budget. While enabled, the
    manager watches the camera and refines whichever region is on screen once
    the view settles.

    Camera events are debounced through a :class:`~qtpy.QtCore.QTimer` that is
    a child of this object, so a continuous zoom gesture costs one recompute
    at the end rather than one per event, and Qt cancels the pending timer
    when the manager is destroyed.

    Parameters
    ----------
    viewer : napari.Viewer or napari.components.ViewerModel
        Viewer whose camera drives refinement.
    parent : QObject, optional
        Qt parent.
    budget : int, optional
        Pixel budget per view. Smaller means coarser levels and more speed.

    Notes
    -----
    Whoever owns a manager must call :meth:`disconnect` from its
    ``closeEvent``, the same rule every other widget in this plugin follows
    for viewer event connections.
    """

    def __init__(self, viewer, parent=None, budget=DEFAULT_PIXEL_BUDGET):
        super().__init__(parent)
        self.viewer = viewer
        self.budget = int(budget)
        self._lods = {}
        self._enabled = False
        self._refining = False

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REFINE_DEBOUNCE_MS)
        self._timer.timeout.connect(self._on_timeout)

    # -- layer registration --------------------------------------------
    @property
    def layers(self):
        """Return the layers currently under level-of-detail control."""
        return list(self._lods)

    def lod_for(self, layer):
        """Return *layer*'s :class:`PhasorLod`, or ``None``."""
        return self._lods.get(layer)

    def attach(self, layer, auto=True):
        """Put *layer* under level-of-detail control.

        Parameters
        ----------
        layer : napari.layers.Image
            Layer to manage. Ignored if it carries no phasor arrays.
        auto : bool, optional
            Immediately drop to the factor that brings the layer under the
            pixel budget. A layer already small enough is left alone.

        Returns
        -------
        PhasorLod or None
            The layer's state, or ``None`` if it cannot be managed.
        """
        if not layer_supports_lod(layer):
            return None

        lod = self._lods.get(layer)
        if lod is None:
            lod = PhasorLod(layer)
            self._lods[layer] = lod

        if auto:
            factor = lod.suggested_factor(self.budget)
            if factor > 1:
                lod.apply(factor)
        return lod

    def detach(self, layer):
        """Restore *layer* to full resolution and stop managing it."""
        lod = self._lods.pop(layer, None)
        if lod is not None:
            lod.detach()

    def detach_all(self):
        """Restore every managed layer to full resolution."""
        for layer in list(self._lods):
            self.detach(layer)

    # -- enable / disable ----------------------------------------------
    @property
    def enabled(self):
        """Whether camera-driven refinement is active."""
        return self._enabled

    def set_enabled(self, enabled):
        """Turn camera-driven refinement on or off."""
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if enabled:
            self._connect_camera()
        else:
            self._disconnect_camera()

    def _connect_camera(self):
        camera = viewer_camera(self.viewer)
        if camera is None:
            return
        camera.events.zoom.connect(self._on_camera_moved)
        camera.events.center.connect(self._on_camera_moved)

    def _disconnect_camera(self):
        camera = viewer_camera(self.viewer)
        if camera is None:
            return
        for event in (camera.events.zoom, camera.events.center):
            with contextlib.suppress(TypeError, ValueError, RuntimeError):
                event.disconnect(self._on_camera_moved)

    def disconnect(self):
        """Stop the timer and drop every camera connection.

        Call this from the owning widget's ``closeEvent``.
        """
        self._timer.stop()
        self._disconnect_camera()
        self._enabled = False

    # -- refinement ----------------------------------------------------
    def _on_camera_moved(self, event=None):
        """Restart the debounce window; the view is still moving."""
        if self._refining or not self._lods:
            return
        self._timer.start()

    def _on_timeout(self):
        self.refine_now()

    def refine_now(self):
        """Refine every managed layer to the region currently on screen.

        Returns
        -------
        list
            The layers whose displayed level actually changed.
        """
        if self._refining:
            return []

        self._refining = True
        changed = []
        try:
            for layer, lod in list(self._lods.items()):
                region = self.visible_region(lod)
                if region is None:
                    continue
                if lod.refine_to(region, budget=self.budget):
                    changed.append(layer)
        finally:
            self._refining = False
        return changed

    def visible_region(self, lod, margin=0.1):
        """Return the on-screen rectangle in a layer's full-resolution pixels.

        Parameters
        ----------
        lod : PhasorLod
            Layer state, which knows the layer's untransformed placement.
        margin : float, optional
            Fraction of the viewport to read in beyond each edge, so a small
            pan does not immediately force another recompute.

        Returns
        -------
        tuple of int or None
            ``(row_start, row_stop, col_start, col_stop)``, clipped to the
            image, or ``None`` if the camera cannot be interpreted.
        """
        camera = viewer_camera(self.viewer)
        if camera is None:
            return None

        zoom = float(getattr(camera, "zoom", 0) or 0)
        if zoom <= 0:
            return None

        canvas = viewer_canvas_size(self.viewer)
        if canvas is None:
            return None
        canvas_height, canvas_width = canvas

        center = np.asarray(camera.center, dtype=float)
        if center.size < 2:
            return None

        # `zoom` is canvas pixels per world unit, so the visible extent in
        # world units is the canvas size divided by it.
        half_height = (canvas_height / zoom) / 2.0 * (1.0 + margin)
        half_width = (canvas_width / zoom) / 2.0 * (1.0 + margin)

        scale = lod.base_scale[-2:]
        translate = lod.base_translate[-2:]
        if not np.all(scale):
            return None

        # World coordinates back to full-resolution pixel indices.
        row_start = (center[-2] - half_height - translate[0]) / scale[0]
        row_stop = (center[-2] + half_height - translate[0]) / scale[0]
        col_start = (center[-1] - half_width - translate[1]) / scale[1]
        col_stop = (center[-1] + half_width - translate[1]) / scale[1]

        height, width = lod.full_shape
        row_start = int(max(0, np.floor(row_start)))
        col_start = int(max(0, np.floor(col_start)))
        row_stop = int(min(height, np.ceil(row_stop)))
        col_stop = int(min(width, np.ceil(col_stop)))

        if row_stop <= row_start or col_stop <= col_start:
            return None
        return row_start, row_stop, col_start, col_stop


class LodSettingsWidget(QWidget):
    """The "Level of Detail" section of the Plot Settings tab.

    Binning is not a filtering option: applying a level swaps the arrays that
    *every* consumer of a layer reads (see the module docstring), so the plot,
    the filter tab and each analysis tab all follow it. The controls therefore
    live with the other plot-wide settings rather than inside one analysis tab.

    How much to bin is a choice, not only a speed setting. Because binning is
    photon-weighted, a binned level *is* the phasor of the summed signal, so
    "bin 4x4" is a legitimate way to buy photons per pixel in a dim dataset.
    ``Auto`` keeps the original behaviour -- pick whatever factor brings the
    image under the manager's pixel budget -- while a fixed factor pins the
    level. Because zoom refinement exists to change the factor as you zoom, it
    would silently undo a fixed choice, so it is offered only under ``Auto``;
    the same goes for the "Full resolution" button, which would otherwise
    leave the level disagreeing with the combo box.

    Parameters
    ----------
    viewer : napari.Viewer or napari.components.ViewerModel
        Viewer whose camera drives refinement.
    parent : QWidget, optional
        The :class:`~napari_phasors.plotter.PlotterWidget` that owns this
        section. It supplies the layer selection and is asked to rebuild the
        phasor data after a level change.

    Notes
    -----
    The owner must call :meth:`close` from its own ``closeEvent`` so the
    manager's camera connections are released.
    """

    def __init__(self, viewer, parent=None):
        """Build the section and wire its controls to the manager."""
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent

        # Built on first use by the `lod_manager` property, so a session that
        # never bins anything never allocates one.
        self._lod_manager = None

        self._setup_ui()
        # Connected after the hierarchy is fully built: connecting signals on
        # ``superqt`` toggles mid-construction segfaults PySide6.
        self._connect_controls()

    def _setup_ui(self):
        """Assemble the titled section box holding the controls."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        detail_box, detail_box_layout = make_section("Level of Detail")

        self.lod_checkbox = QToggleSwitch("Bin large images")
        self.lod_checkbox.setToolTip(
            "Bin very large images so the phasor plot, filtering and every "
            "analysis tab work on fewer pixels. Binning is photon-weighted, "
            "so a binned image is exactly the phasor of the summed signal. "
            "The full-resolution data is kept and can be restored at any "
            "time."
        )
        detail_box_layout.addWidget(self.lod_checkbox)

        binning_layout = QHBoxLayout()
        binning_layout.addWidget(QLabel("Binning:"))
        self.binning_combobox = QComboBox()
        self.binning_combobox.addItem(AUTO_BINNING)
        for factor in manual_bin_factors():
            self.binning_combobox.addItem(f"{factor}x{factor}")
        self.binning_combobox.setCurrentText(AUTO_BINNING)
        self.binning_combobox.setToolTip(
            "How much to bin. \"Auto\" picks the coarsest factor that still "
            "brings the image under the pixel budget; a fixed factor bins by "
            "exactly that much and stays there. Binning is photon-weighted, "
            "so binning NxN gives the phasor of the summed signal over each "
            "block -- more photons per point, less spatial detail."
        )
        self.binning_combobox.setEnabled(False)
        binning_layout.addWidget(self.binning_combobox)
        binning_layout.addStretch()
        detail_box_layout.addLayout(binning_layout)

        self.lod_zoom_checkbox = QToggleSwitch("Refine on zoom")
        self.lod_zoom_checkbox.setToolTip(
            "When you zoom in, recompute the visible region at finer detail, "
            "down to full resolution once the region is small enough. "
            "Available only while binning is set to \"Auto\", since it would "
            "otherwise undo a fixed bin factor as soon as you zoomed."
        )
        self.lod_zoom_checkbox.setEnabled(False)
        detail_box_layout.addWidget(self.lod_zoom_checkbox)

        detail_status_layout = QHBoxLayout()
        self.lod_status_label = QLabel("Full resolution")
        detail_status_layout.addWidget(self.lod_status_label)
        detail_status_layout.addStretch()
        self.lod_full_button = QPushButton("Full resolution")
        self.lod_full_button.setToolTip(
            "Restore this layer to full resolution, leaving binning enabled. "
            "Available only while binning is set to \"Auto\"; with a fixed "
            "factor, turn binning off to get back to full resolution."
        )
        self.lod_full_button.setEnabled(False)
        detail_status_layout.addWidget(self.lod_full_button)
        detail_box_layout.addLayout(detail_status_layout)

        layout.addWidget(detail_box)

    def _connect_controls(self):
        """Wire the level-of-detail controls to their manager."""
        self.lod_checkbox.toggled.connect(self._on_lod_toggled)
        self.binning_combobox.currentTextChanged.connect(
            self._on_binning_changed
        )
        self.lod_zoom_checkbox.toggled.connect(self._on_lod_zoom_toggled)
        self.lod_full_button.clicked.connect(self._on_lod_full_clicked)

    @property
    def lod_manager(self):
        """Return the level-of-detail manager, creating it on first use.

        Built lazily so that a session which never touches a large image never
        allocates one, and so no camera connection exists until the user asks
        for binning.
        """
        if self._lod_manager is None:
            self._lod_manager = LodManager(self.viewer, parent=self)
        return self._lod_manager

    def _selected_layers(self):
        """Return the layers the plotter currently has selected."""
        if self.parent_widget is None:
            return []
        return self.parent_widget.get_selected_layers()

    def selected_factor(self):
        """Return the fixed bin factor chosen, or ``None`` for ``Auto``."""
        text = self.binning_combobox.currentText()
        if text == AUTO_BINNING:
            return None
        return int(text.split("x")[0])

    def _update_control_availability(self):
        """Enable each control for the mode the section is now in.

        Zoom refinement and the full-resolution button both move the level out
        from under a fixed factor, so they are offered only under ``Auto``.
        """
        checked = self.lod_checkbox.isChecked()
        is_auto = self.selected_factor() is None

        self.binning_combobox.setEnabled(checked)
        self.lod_zoom_checkbox.setEnabled(checked and is_auto)
        self.lod_full_button.setEnabled(checked and is_auto)

        if not is_auto and self.lod_zoom_checkbox.isChecked():
            # Turning it off here also drops the manager's camera connection,
            # via `_on_lod_zoom_toggled`.
            self.lod_zoom_checkbox.setChecked(False)

    def _apply_current_level(self):
        """Put every managed layer on the level the controls ask for."""
        manager = self.lod_manager
        manual = self.selected_factor()
        for layer in manager.layers:
            lod = manager.lod_for(layer)
            if lod is None:
                continue
            factor = (
                manual
                if manual is not None
                else lod.suggested_factor(manager.budget)
            )
            lod.apply(factor)

    def _on_lod_toggled(self, checked):
        """Bin (or restore) every selected layer."""
        self._update_control_availability()

        manager = self.lod_manager
        if not checked:
            manager.set_enabled(False)
            self.lod_zoom_checkbox.setChecked(False)
            manager.detach_all()
            self._update_lod_status()
            self._refresh_after_lod_change()
            return

        layers = self._selected_layers()
        if not layers:
            show_error(
                "Please select at least one image layer with phasor features."
            )
            self.lod_checkbox.setChecked(False)
            return

        # Attached without a level so the chosen mode -- not `attach`'s own
        # budget rule -- decides what every layer ends up showing.
        for layer in layers:
            manager.attach(layer, auto=False)
        self._apply_current_level()
        self._update_lod_status()
        self._refresh_after_lod_change()

    def _on_binning_changed(self, _text=None):
        """Re-level every managed layer for the newly chosen amount."""
        self._update_control_availability()
        if not self.lod_checkbox.isChecked():
            return
        self._apply_current_level()
        self._update_lod_status()
        self._refresh_after_lod_change()

    def _on_lod_zoom_toggled(self, checked):
        """Start or stop following the camera."""
        manager = self.lod_manager
        manager.set_enabled(checked)
        if checked:
            manager.refine_now()
            self._update_lod_status()
            self._refresh_after_lod_change()

    def _on_lod_full_clicked(self):
        """Drop back to full resolution without leaving binning enabled."""
        manager = self.lod_manager
        for layer in manager.layers:
            lod = manager.lod_for(layer)
            if lod is not None:
                lod.apply(1)
        self._update_lod_status()
        self._refresh_after_lod_change()

    def _update_lod_status(self):
        """Describe the detail currently on screen."""
        manager = self._lod_manager
        if manager is None or not manager.layers:
            self.lod_status_label.setText("Full resolution")
            return

        factors = sorted(
            {
                manager.lod_for(layer).factor
                for layer in manager.layers
                if manager.lod_for(layer) is not None
            }
        )
        if factors == [1]:
            self.lod_status_label.setText("Full resolution")
        elif len(factors) == 1:
            self.lod_status_label.setText(f"Binned {factors[0]}x{factors[0]}")
        else:
            self.lod_status_label.setText(
                "Binned "
                + ", ".join(f"{factor}x{factor}" for factor in factors)
            )

    def _refresh_after_lod_change(self):
        """Rebuild the plot and every tab for the new detail level."""
        if self.parent_widget is None:
            return
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            self.parent_widget.refresh_phasor_data()

    def closeEvent(self, event):
        """Release the manager's camera connections before closing.

        Leaving them attached to a closed widget is the usual cause of PySide6
        teardown crashes in this plugin.
        """
        if self._lod_manager is not None:
            with contextlib.suppress(RuntimeError, TypeError, ValueError):
                self._lod_manager.disconnect()
        event.accept()
