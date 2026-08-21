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
from qtpy.QtCore import QObject, QTimer

from ._binning import (
    DEFAULT_PIXEL_BUDGET,
    PhasorPyramid,
    bin_factor_for_shape,
)
from ._utils import (
    assign_filter_and_threshold,
    compute_filter_and_threshold,
)

__all__ = ["PhasorLod", "LodManager", "layer_supports_lod"]

#: Metadata key holding a layer's :class:`PhasorLod`.
METADATA_KEY = "lod"

#: How long the camera must stay still before a refinement runs, in
#: milliseconds. Long enough that a continuous pinch-zoom or scroll only
#: triggers one recompute at the end of the gesture.
REFINE_DEBOUNCE_MS = 250


def layer_supports_lod(layer):
    """Return whether *layer* carries the full-resolution phasor arrays."""
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    return all(
        key in metadata
        for key in ("original_mean", "G_original", "S_original")
    )


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
        camera = getattr(self.viewer, "camera", None)
        if camera is None:
            return
        camera.events.zoom.connect(self._on_camera_moved)
        camera.events.center.connect(self._on_camera_moved)

    def _disconnect_camera(self):
        camera = getattr(self.viewer, "camera", None)
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
        camera = getattr(self.viewer, "camera", None)
        if camera is None:
            return None

        zoom = float(getattr(camera, "zoom", 0) or 0)
        if zoom <= 0:
            return None

        canvas = getattr(self.viewer, "_canvas_size", None)
        if not canvas or len(canvas) < 2:
            return None
        canvas_height, canvas_width = float(canvas[0]), float(canvas[1])

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
