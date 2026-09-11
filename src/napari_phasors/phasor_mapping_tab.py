import contextlib
import copy
import warnings
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from napari.layers import Image
from napari.utils.notifications import show_error, show_warning
from phasorpy.lifetime import (
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_from_polar, phasor_to_polar
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QColor, QDoubleValidator
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from superqt import QRangeSlider, QToggleSwitch

from ._mapping_filters import (
    MAPPING_METRICS,
    MappingFilterList,
    baseline_arrays,
    combined_mask,
    compute_metric,
    get_filters,
    kept_fraction,
    rebuild_layer_from_filters,
    requires_frequency,
    select_harmonic,
    set_filters,
)
from ._parallel import parallel_map
from ._settings_store import chain_merges, merge_keyed_path
from ._timelapse import slice_datasets
from ._utils import (
    LIFETIME_OUTPUT_TYPES,
    AutoUpdateMixin,
    HistogramWidget,
    analysis_section_stylesheet,
    create_mpl_colormap_from_qcolor,
    create_settings_note_label,
    layer_colormap_from_settings,
    layer_colormap_to_settings,
    make_section,
    populate_colormap_combobox,
    resolve_colormap_by_name,
    resolve_napari_layer_colormap,
    set_settings_note,
    setup_primary_button,
)

if TYPE_CHECKING:
    import napari


# Default grid resolution used when a precomputed mesh grid is not supplied.
_DEFAULT_MESH_RESOLUTION = 300
# Default opacity of the phase/modulation mesh overlay. The control exposes
# its complement (transparency), which is how the rest of the plugin words it.
DEFAULT_MESH_ALPHA = 0.45
_MAPPING_OUTPUT_METADATA_KEY = 'phasor_mapping_output'
_MAPPING_OUTPUT_TYPES = (
    "Apparent Phase Lifetime",
    "Apparent Modulation Lifetime",
    "Normal Lifetime",
    "Phase",
    "Modulation",
)


def _phasor_to_lifetime(kind, real, imag, frequency):
    """Return the *kind* lifetime (ns) of phasor coordinates.

    ``kind`` is one of ``LIFETIME_OUTPUT_TYPES`` and ``frequency`` is the
    effective frequency in MHz (the laser frequency times the harmonic).
    Non-physical coordinates give negative or non-finite lifetimes; they are
    returned as-is so callers can decide how to treat them.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "Normal Lifetime":
            return phasor_to_normal_lifetime(real, imag, frequency=frequency)
        phase_lifetime, modulation_lifetime = phasor_to_apparent_lifetime(
            real, imag, frequency=frequency
        )
    if kind == "Apparent Phase Lifetime":
        return phase_lifetime
    return modulation_lifetime


def compute_lifetime_mesh_field(p_grid, m_grid, kind, frequency):
    """Return the *kind* lifetime (ns) at every cell of a polar mesh grid.

    The lifetime is computed with the same phasorpy function as the output
    map, so a mesh cell and a pixel at the same phasor coordinates always
    carry the same value. Iso-lifetime lines are rays from the origin for the
    apparent phase lifetime (the phase mesh in ns), circles around the origin
    for the apparent modulation lifetime (the modulation mesh in ns), and rays
    from the semicircle centre (0.5, 0) for the normal lifetime.
    """
    with np.errstate(invalid="ignore"):
        real, imag = phasor_from_polar(p_grid, m_grid)
    return _phasor_to_lifetime(kind, real, imag, frequency)


def lifetime_mesh_upper_bound(frequency):
    """Return the largest lifetime (ns) a lifetime mesh range defaults to.

    Two modulation periods at *frequency* (MHz): the same bound the output
    range falls back to when lifetimes run away (the apparent phase lifetime
    diverges as G approaches 0).
    """
    return 2e3 / frequency


def lifetime_mesh_range_from_phasors(kind, real, imag, frequency):
    """Return the ``(min, max)`` range in ns of the *kind* lifetimes of data.

    Non-physical (negative or non-finite) lifetimes are ignored and both ends
    are capped at :func:`lifetime_mesh_upper_bound`. Returns ``None`` when no
    lifetime is valid.
    """
    upper = lifetime_mesh_upper_bound(frequency)
    lifetimes = np.asarray(
        _phasor_to_lifetime(kind, real, imag, frequency), dtype=float
    )
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes >= 0)]
    if not lifetimes.size:
        return None
    return min(float(lifetimes.min()), upper), min(
        float(lifetimes.max()), upper
    )


def compute_phasor_mesh_mask(
    p_grid,
    m_grid,
    *,
    semicircle=True,
    phase_range=None,
    modulation_range=None,
    clip_semicircle=False,
    lifetime_grid=None,
    lifetime_range=None,
):
    """Return the boolean mask of grid cells *excluded* from a phasor mesh.

    A cell is masked (``True``) when it falls outside the requested phase,
    modulation or lifetime range, or -- when ``clip_semicircle`` is set in
    semicircle mode -- outside the universal semicircle. ``p_grid``/``m_grid``
    are the phase and modulation arrays of a coordinate grid (see
    :func:`draw_phasor_mesh`). When ``lifetime_grid`` is given, cells with a
    non-finite lifetime are masked too, and so -- in semicircle mode -- are
    cells below the diameter. Lifetimes mirror across it, but the phase and
    modulation meshes (whose phase starts at 0) never draw that region, and
    the lifetime meshes follow suit.
    """
    mask = np.zeros(np.shape(p_grid), dtype=bool)
    if phase_range is not None:
        phase_min, phase_max = phase_range
        mask |= ~((p_grid >= phase_min) & (p_grid <= phase_max))
    if modulation_range is not None:
        mod_min, mod_max = modulation_range
        mask |= ~((m_grid >= mod_min) & (m_grid <= mod_max))
    if lifetime_grid is not None:
        with np.errstate(invalid="ignore"):
            keep = np.isfinite(lifetime_grid)
            if lifetime_range is not None:
                lifetime_min, lifetime_max = lifetime_range
                keep &= (lifetime_grid >= lifetime_min) & (
                    lifetime_grid <= lifetime_max
                )
            if semicircle:
                keep &= p_grid >= 0
        mask |= ~keep
    if clip_semicircle and semicircle:
        with np.errstate(invalid="ignore"):
            # Bound by the arc (m <= cos(phase)) and by the diameter (s >= 0).
            # In semicircle mode the phase comes straight from ``atan2`` so
            # points below the diameter have a negative phase; clipping them
            # stops the mesh from bleeding past the bottom of the semicircle.
            mask |= (m_grid > np.cos(p_grid)) | (p_grid < 0)
    return mask


def _resolve_mesh_blur_sigma(ax, resolution, target_px=1.2, floor=1.5):
    """Return the alpha-mask blur sigma (grid-cell units) for a mesh edge.

    The mesh's edge anti-aliasing works by Gaussian-blurring the boolean
    exclusion mask, then letting Matplotlib's Lanczos resampling downscale
    that to display pixels. Sigma has to be picked in *display-pixel* terms
    (``target_px``) and converted to grid cells via ``resolution /
    display_px``, not left as a flat constant: at a fixed sigma, a
    high-resolution grid (this widget scales it up to 1280 for on-screen
    sharpness) has cells much smaller than a display pixel, so the blur band
    becomes sub-pixel and Lanczos-resamples into a ringing/dashed edge
    instead of a smooth fade -- most visible on shallow-angle mesh
    boundaries (e.g. a phase limit near the real axis). ``floor`` keeps a
    minimum blur even when the grid is coarser than the display (e.g. the
    default low-resolution grid), where the ratio alone would undershoot.
    """
    display_px = 0.0
    with contextlib.suppress(Exception):
        bbox = ax.get_window_extent()
        display_px = max(float(bbox.width), float(bbox.height))
    if display_px <= 0:
        return max(floor, target_px * resolution / _DEFAULT_MESH_RESOLUTION)
    return max(floor, target_px * resolution / display_px)


def draw_phasor_mesh(
    ax,
    kind,
    *,
    semicircle=True,
    colormap="jet",
    alpha=0.45,
    alpha_map=None,
    phase_range=None,
    modulation_range=None,
    clip_semicircle=False,
    vmin=None,
    vmax=None,
    resolution=_DEFAULT_MESH_RESOLUTION,
    x_limits=None,
    y_limits=None,
    p_grid=None,
    m_grid=None,
    mask=None,
    extent=None,
    frequency=None,
    lifetime_range=None,
    lifetime_grid=None,
):
    """Draw a phase/modulation/lifetime mesh field behind the phasor data.

    This is the single, viewer-independent implementation shared by the
    interactive Phasor Mapping tab and the headless Batch Analysis export, so
    both produce an identical-looking mesh. It only needs a Matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    kind : {"Phase", "Modulation", "Apparent Phase Lifetime", \
"Apparent Modulation Lifetime", "Normal Lifetime"}
        Quantity used to color the mesh. Lifetime kinds need ``frequency``
        (or a precomputed ``lifetime_grid``).
    semicircle : bool
        Whether the plot is in semicircle (vs. full-quadrant) geometry. Used
        for the default grid extent, phase wrapping and the semicircle clip.
    colormap : str or matplotlib.colors.Colormap
        Colormap (name or object) for the mesh.
    alpha : float
        Overall mesh opacity (ignored when ``alpha_map`` is given).
    alpha_map : numpy.ndarray, optional
        Precomputed per-pixel alpha array (already scaled). When omitted a
        smoothed alpha map is derived from ``mask`` and ``alpha``.
    phase_range, modulation_range : tuple of float, optional
        ``(min, max)`` ranges restricting which mesh cells are shown.
    clip_semicircle : bool
        Hide mesh cells outside the universal semicircle (semicircle mode only).
    vmin, vmax : float, optional
        Color scaling limits. Default to the active range for ``kind`` (or the
        visible field's min/max when no range is given).
    resolution : int
        Grid resolution used when ``p_grid``/``m_grid`` are not supplied.
    x_limits, y_limits : tuple of float, optional
        Grid extent used when computing the grid. Default to sensible bounds
        for the current geometry.
    p_grid, m_grid : numpy.ndarray, optional
        Precomputed phase/modulation grids (e.g. cached by the caller). When
        omitted they are computed from the limits and ``resolution``.
    mask : numpy.ndarray, optional
        Precomputed exclusion mask (see :func:`compute_phasor_mesh_mask`).
    extent : list of float, optional
        ``[xmin, xmax, ymin, ymax]`` matching the grid. Required if grids are
        supplied; otherwise derived from the limits.
    frequency : float, optional
        Effective frequency in MHz (laser frequency times harmonic), used to
        compute a lifetime mesh when ``lifetime_grid`` is not supplied.
    lifetime_range : tuple of float, optional
        ``(min, max)`` lifetime range in ns restricting which cells of a
        lifetime mesh are shown.
    lifetime_grid : numpy.ndarray, optional
        Precomputed lifetime grid (see :func:`compute_lifetime_mesh_field`).

    Returns
    -------
    matplotlib.image.AxesImage
        The drawn mesh image.
    """
    if p_grid is None or m_grid is None or extent is None:
        if x_limits is None or y_limits is None:
            if semicircle:
                x_limits = (-0.05, 1.05)
                y_limits = (-0.05, 0.7)
            else:
                x_limits = (-1.05, 1.05)
                y_limits = (-1.05, 1.05)
        x_fine = np.linspace(x_limits[0], x_limits[1], resolution)
        y_fine = np.linspace(y_limits[0], y_limits[1], resolution)
        grid_x, grid_y = np.meshgrid(x_fine, y_fine)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p_grid, m_grid = phasor_to_polar(grid_x, grid_y)
        if not semicircle:
            with np.errstate(invalid="ignore"):
                p_grid = np.mod(p_grid, 2.0 * np.pi)
        extent = [x_limits[0], x_limits[1], y_limits[0], y_limits[1]]

    is_lifetime = kind in LIFETIME_OUTPUT_TYPES
    if is_lifetime and lifetime_grid is None:
        if frequency is None:
            raise ValueError(f"A frequency is required to draw a {kind} mesh.")
        lifetime_grid = compute_lifetime_mesh_field(
            p_grid, m_grid, kind, frequency
        )

    if mask is None:
        mask = compute_phasor_mesh_mask(
            p_grid,
            m_grid,
            semicircle=semicircle,
            phase_range=phase_range,
            modulation_range=modulation_range,
            clip_semicircle=clip_semicircle,
            lifetime_grid=lifetime_grid if is_lifetime else None,
            lifetime_range=lifetime_range,
        )

    if is_lifetime:
        field = lifetime_grid
    else:
        field = p_grid if kind == "Phase" else m_grid
    # The exterior of the mesh is hidden purely through ``alpha_map`` (0 outside
    # the visible region). We deliberately keep the *colour* field fully valid
    # everywhere rather than masking it: a masked colour field renders excluded
    # cells as the "bad" colour, and blending those transparent-black cells into
    # the feathered edge under RGBA interpolation produces a grey halo / shadow
    # around the outline. Filling non-finite values keeps the edge colours clean.
    field = np.nan_to_num(np.asarray(field, dtype=float), nan=0.0)

    if isinstance(colormap, str):
        cmap = resolve_colormap_by_name(colormap)
        if cmap is None:
            cmap = resolve_colormap_by_name("viridis")
    else:
        cmap = colormap
    mesh_cmap = cmap.copy()
    mesh_cmap.set_bad((0, 0, 0, 0))

    if alpha_map is None:
        sigma = _resolve_mesh_blur_sigma(ax, resolution)
        alpha_base = gaussian_filter(
            (~mask).astype(float), sigma=sigma, mode="nearest"
        )
        alpha_map = np.clip(alpha_base, 0.0, 1.0) * alpha

    if vmin is None or vmax is None:
        if kind == "Phase" and phase_range is not None:
            vmin, vmax = phase_range
        elif kind == "Modulation" and modulation_range is not None:
            vmin, vmax = modulation_range
        elif is_lifetime and lifetime_range is not None:
            vmin, vmax = lifetime_range
        else:
            with np.errstate(invalid="ignore"):
                visible = np.asarray(field)[~mask]
                vmin = float(np.nanmin(visible)) if visible.size else 0.0
                vmax = float(np.nanmax(visible)) if visible.size else 1.0

    mesh_imshow_kwargs = {
        "extent": extent,
        "origin": "lower",
        "cmap": mesh_cmap,
        "vmin": vmin,
        "vmax": vmax,
        "interpolation": "lanczos",
        "zorder": 0.1,  # Behind all phasor artists
        "alpha": alpha_map,
        "aspect": "auto",
    }
    try:
        image = ax.imshow(
            field,
            interpolation_stage="rgba",
            **mesh_imshow_kwargs,
        )
    except TypeError:
        image = ax.imshow(field, **mesh_imshow_kwargs)
    # Restore a 1:1 data aspect; ``aspect="auto"`` on imshow otherwise stretches
    # the axes and distorts the phasor plot.
    ax.set_aspect(1, adjustable="box")
    return image


class PhasorMappingWidget(AutoUpdateMixin, QWidget):
    """Widget to calculate and display phasor mapping outputs.

    Supports lifetime-derived outputs and direct phasor outputs:
    Apparent Phase Lifetime, Apparent Modulation Lifetime,
    Normal Lifetime, Phase, and Modulation.
    """

    outputTypeChanged = Signal(str)
    """Signal emitted with the new output type name when the mapping output changes."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Build the phasor mapping tab for *viewer*."""
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent
        self.frequency = None
        self.lifetime_data = None
        self.lifetime_data_original = None  # Store original unclipped data
        self.per_layer_lifetime_data = {}  # {layer_name: data}
        self.per_layer_lifetime_data_original = {}  # {layer_name: data}
        self.current_metric_data = None
        self.current_metric_data_original = None
        self.per_layer_metric_data = {}
        self.per_layer_metric_data_original = {}
        self.lifetime_layer = (
            None  # Reference to first layer for backward compatibility
        )
        self.lifetime_layers = []  # List of all lifetime layers
        self.metric_layers = []  # List of output layers for current metric
        self.current_output_type = "Apparent Phase Lifetime"
        self._overlay_imshow = None
        self._mesh_overlay_imshow = None
        self._phase_colormap_name = "cool"
        self._modulation_colormap_name = "PiYG"
        # Output types whose colormap the user picked in the combobox since
        # the last run. That explicit pick is applied on the next run; every
        # other run keeps the colormap the output layers already have.
        self._pending_combobox_colormaps = set()
        self._coloring_paused_by_tab = False
        self.min_lifetime = None
        self.max_lifetime = None
        self.lifetime_colormap = None
        self.colormap_contrast_limits = None
        self.colormap_gamma = 1.0
        self.lifetime_type = None
        self.lifetime_range_factor = (
            1000  # Factor to convert to integer for slider
        )
        self._updating_contrast_limits = (
            False  # Flag to track contrast limits updates
        )
        self._updating_settings = False  # Flag to prevent recursive updates
        self._needs_update = False  # Deferred update flag
        self._has_calculated_output = False
        self._output_refresh_timer = QTimer(self)
        self._output_refresh_timer.setSingleShot(True)
        self._output_refresh_timer.setInterval(50)
        self._output_refresh_timer.timeout.connect(self._refresh_active_output)
        self._updating_linked_layers = (
            False  # Flag to prevent recursive layer updates
        )
        self.phase_range_factor = 100
        self.modulation_range_factor = 100
        self.lifetime_mesh_range_factor = 100
        # Set while this tab rewrites the phasor arrays from the filter
        # stack, so the refresh it triggers does not recurse back into it.
        self._applying_mapping_filter = False
        self._axes_limit_callback_cids = []
        self._mesh_axes_update_timer = QTimer(self)
        self._mesh_axes_update_timer.setSingleShot(True)
        self._mesh_axes_update_timer.setInterval(75)
        self._mesh_axes_update_timer.timeout.connect(
            self._apply_mesh_after_axes_change
        )
        self._mesh_grid_cache = {}
        self._mesh_grid_cache_order = []
        self._mesh_grid_cache_max_entries = 8
        self._mesh_alpha_cache = {}
        self._mesh_alpha_cache_order = []
        self._mesh_alpha_cache_max_entries = 8

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create content widget for scroll area
        content_widget = QWidget()
        self.main_layout = QVBoxLayout(content_widget)

        # Set scroll area content
        scroll_area.setWidget(content_widget)

        # Create main layout for this widget
        main_widget_layout = QVBoxLayout(self)
        main_widget_layout.addWidget(scroll_area)
        main_widget_layout.setStretch(0, 1)
        self.setStyleSheet(analysis_section_stylesheet())

        # Output section -----------------------------------------------------
        output_box, output_box_layout = make_section("Output")

        # Output mode (always visible, top row)
        self.output_mode_widget = QWidget()
        output_mode_layout = QHBoxLayout(self.output_mode_widget)
        output_mode_layout.setContentsMargins(0, 0, 0, 0)
        output_mode_layout.addWidget(QLabel("Parameter to Analyze: "))
        self.output_mode_combobox = QComboBox()
        self.output_mode_combobox.addItems(["Lifetime", "Phase", "Modulation"])
        self.output_mode_combobox.setCurrentText("Lifetime")
        output_mode_layout.addWidget(self.output_mode_combobox, 1)
        output_box_layout.addWidget(self.output_mode_widget)

        # Lifetime output selector (visible only for Lifetime mode)
        self.lifetime_output_widget = QWidget()
        lifetime_type_layout = QHBoxLayout(self.lifetime_output_widget)
        lifetime_type_layout.setContentsMargins(0, 0, 0, 0)
        lifetime_type_layout.addWidget(QLabel("Select lifetime to display: "))
        self.lifetime_type_combobox = QComboBox()
        self.lifetime_type_combobox.addItems(
            [
                "Apparent Phase Lifetime",
                "Apparent Modulation Lifetime",
                "Normal Lifetime",
            ]
        )
        self.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
        lifetime_type_layout.addWidget(self.lifetime_type_combobox, 1)
        output_box_layout.addWidget(self.lifetime_output_widget)

        # Frequency input (visible only for Lifetime mode)
        self.frequency_widget = QWidget()
        frequency_layout = QHBoxLayout(self.frequency_widget)
        frequency_layout.setContentsMargins(0, 0, 0, 0)
        frequency_layout.addWidget(QLabel("Frequency (MHz): "))
        self.frequency_input = QLineEdit()
        self.frequency_input.setValidator(QDoubleValidator())
        frequency_layout.addWidget(self.frequency_input)
        output_box_layout.addWidget(self.frequency_widget)
        self.main_layout.addWidget(output_box)

        # Filter section -----------------------------------------------------
        # Each criterion is its own card. Chaining "set a range, press apply"
        # steps made the pixels that vanished between two steps impossible to
        # account for; a visible list of the criteria currently in force, each
        # removable on its own, is what makes the result readable.
        filter_box, filter_box_layout = make_section("Filter")
        self.filter_box = filter_box

        self.filter_intro_label = QLabel(
            "Discard pixels whose value falls outside a range."
        )
        self.filter_intro_label.setWordWrap(True)
        self.filter_intro_label.setToolTip(
            "A pixel is kept only when it satisfies every enabled filter. Each one is measured on the unfiltered data, so the order you add them in does not matter and removing one restores exactly the pixels it hid."
        )
        filter_box_layout.addWidget(self.filter_intro_label)

        self.filter_list = MappingFilterList(MAPPING_METRICS)
        self.filter_list.set_editable_metrics(MAPPING_METRICS)
        self.filter_list.set_params_provider(self._new_filter_params)
        self.filter_list.set_harmonic_provider(self._current_harmonic)
        self.filter_list.set_bounds_provider(self._filter_bounds_for)
        self.filter_list.filtersChanged.connect(self._on_filters_changed)
        filter_box_layout.addWidget(self.filter_list)

        # Coloring section ---------------------------------------------------
        # Only relevant for Phase/Modulation output; hidden for Lifetime (see
        # _sync_mode_widgets).
        coloring_box, coloring_box_layout = make_section("Coloring")
        self.coloring_box = coloring_box

        # 2D phasor custom-color controls (visible for Phase/Modulation modes)
        self.colormap_widget = QWidget()
        colormap_layout = QHBoxLayout(self.colormap_widget)
        colormap_layout.setContentsMargins(0, 0, 0, 0)
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combobox = QComboBox()
        populate_colormap_combobox(
            self.colormap_combobox,
            include_select_color=True,
            selected=self._phase_colormap_name,
        )
        colormap_layout.addWidget(self.colormap_combobox, 1)

        self.custom_color_button = QPushButton()
        self.custom_color_button.setFixedSize(22, 22)
        self.custom_color_button.setToolTip("Select custom color")
        self.custom_color_button.setVisible(False)
        self.custom_color_button.clicked.connect(self._on_custom_color_clicked)
        colormap_layout.addWidget(self.custom_color_button)

        coloring_box_layout.addWidget(self.colormap_widget)

        self.coloring_checkbox_widget = QWidget()
        coloring_checkbox_layout = QHBoxLayout(self.coloring_checkbox_widget)
        coloring_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.apply_2d_colormap_checkbox = QToggleSwitch(
            "Apply colormap to 2D Histogram"
        )
        self.apply_2d_colormap_checkbox.onColor = QColor(
            "#27ae60"
        )  # Nice Green
        coloring_checkbox_layout.addWidget(self.apply_2d_colormap_checkbox)
        coloring_checkbox_layout.addStretch(1)
        coloring_box_layout.addWidget(self.coloring_checkbox_widget)
        self.main_layout.addWidget(coloring_box)

        # Connect signals
        self.lifetime_type_combobox.currentTextChanged.connect(
            self._on_lifetime_type_changed
        )
        self.output_mode_combobox.currentTextChanged.connect(
            self._on_output_mode_changed
        )
        self.colormap_combobox.currentTextChanged.connect(
            self._on_colormap_combobox_changed
        )
        self.apply_2d_colormap_checkbox.toggled.connect(
            self._on_apply_2d_colormap_checkbox_changed
        )
        self.frequency_input.editingFinished.connect(
            self._on_frequency_changed
        )

        # NOTE: The widget is created here but NOT added to this tab's layout.
        # PlotterWidget wraps it in a HistogramDockWidget and docks it separately.
        self.histogram_widget = HistogramWidget(
            xlabel="Lifetime (ns)",
            ylabel="Pixel count",
            bins=150,
            default_colormap_name="plasma",
            range_slider_enabled=True,
            range_label_prefix="Lifetime range (ns)",
            range_factor=self.lifetime_range_factor,
            viewer=self.viewer,
            parent=self,
        )

        # Convenience aliases so the rest of the class (and tests) can
        # keep referring to the same attribute names.
        self.lifetime_range_slider = self.histogram_widget.range_slider
        self.lifetime_range_label = self.histogram_widget.range_label
        self.lifetime_min_edit = self.histogram_widget.range_min_edit
        self.lifetime_max_edit = self.histogram_widget.range_max_edit

        # Connect the histogram widget's rangeChanged signal
        self.histogram_widget.rangeChanged.connect(
            self._on_range_changed_from_histogram
        )
        self._configure_histogram_labels_for_output(
            self.lifetime_type_combobox.currentText()
        )
        self.histogram_widget.update_data(np.array([]))

        # Mesh Overlay Settings (at the end of the tab)
        self.mesh_overlay_group, mesh_overlay_group_layout = make_section(
            "Mesh overlay"
        )

        # Toggle for background mesh
        self.mesh_overlay_checkbox = QToggleSwitch("Show mesh overlay")
        self.mesh_overlay_checkbox.onColor = QColor("#27ae60")
        self.mesh_overlay_checkbox.setToolTip(
            "Show a mesh of the selected phase, modulation or lifetime "
            "behind the phasor data"
        )
        mesh_overlay_group_layout.addWidget(self.mesh_overlay_checkbox)

        # Toggle to clip mesh to semicircle
        self.mesh_clip_semicircle_checkbox = QToggleSwitch(
            "Clip mesh to semicircle"
        )
        self.mesh_clip_semicircle_checkbox.onColor = QColor("#27ae60")
        self.mesh_clip_semicircle_checkbox.setToolTip(
            "Only show the background mesh inside the universal semicircle"
        )
        self.mesh_clip_semicircle_checkbox.setVisible(False)

        # Toggle for colorbar
        self.mesh_colorbar_checkbox = QToggleSwitch("Show colorbar")
        self.mesh_colorbar_checkbox.onColor = QColor("#27ae60")
        self.mesh_colorbar_checkbox.setToolTip(
            "Show a colorbar for the phase, modulation or lifetime mesh/plot"
        )
        self.mesh_colorbar_checkbox.setVisible(False)

        # Row for Transparency, Clip Toggle, and Colorbar Toggle
        self.mesh_controls_widget = QWidget()
        mesh_controls_layout = QHBoxLayout(self.mesh_controls_widget)
        mesh_controls_layout.setContentsMargins(0, 0, 0, 0)

        mesh_controls_layout.addWidget(QLabel("Transparency:"))
        self.mesh_transparency_spinbox = QDoubleSpinBox()
        self.mesh_transparency_spinbox.setRange(0.0, 0.99)
        self.mesh_transparency_spinbox.setSingleStep(0.05)
        self.mesh_transparency_spinbox.setDecimals(2)
        self.mesh_transparency_spinbox.setValue(1.0 - DEFAULT_MESH_ALPHA)
        self.mesh_transparency_spinbox.setToolTip(
            "Transparency of the phase/modulation mesh overlay"
        )
        self.mesh_transparency_spinbox.setFixedWidth(60)
        mesh_controls_layout.addWidget(self.mesh_transparency_spinbox)

        mesh_controls_layout.addSpacing(10)
        mesh_controls_layout.addWidget(self.mesh_clip_semicircle_checkbox)
        mesh_controls_layout.addSpacing(10)
        mesh_controls_layout.addWidget(self.mesh_colorbar_checkbox)
        mesh_controls_layout.addStretch(1)

        mesh_overlay_group_layout.addWidget(self.mesh_controls_widget)

        # Phase range controls
        self.phase_range_container = QWidget()
        ph_cnt_layout = QVBoxLayout(self.phase_range_container)
        ph_cnt_layout.setContentsMargins(0, 5, 0, 0)

        ph_row = QHBoxLayout()
        ph_row.addWidget(QLabel("Phase range (rad):"))
        self.phase_min_edit = QLineEdit("0.00")
        self.phase_max_edit = QLineEdit("1.60")
        self.phase_min_edit.setValidator(QDoubleValidator())
        self.phase_max_edit.setValidator(QDoubleValidator())
        self.phase_min_edit.setFixedWidth(50)
        self.phase_max_edit.setFixedWidth(50)
        self.phase_min_edit.setAlignment(Qt.AlignCenter)
        self.phase_max_edit.setAlignment(Qt.AlignCenter)
        ph_row.addWidget(self.phase_min_edit)
        ph_row.addWidget(QLabel("to"))
        ph_row.addWidget(self.phase_max_edit)
        self.phase_auto_btn = QPushButton("Auto")
        self.phase_auto_btn.clicked.connect(self._on_mesh_auto_clicked)
        ph_row.addWidget(self.phase_auto_btn)
        ph_row.addStretch(1)
        ph_cnt_layout.addLayout(ph_row)
        self.phase_range_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.phase_range_slider.setRange(0, 628)
        self.phase_range_slider.setValue((0, 628))
        ph_cnt_layout.addWidget(self.phase_range_slider)
        mesh_overlay_group_layout.addWidget(self.phase_range_container)

        # Modulation range controls
        self.modulation_range_container = QWidget()
        mod_cnt_layout = QVBoxLayout(self.modulation_range_container)
        mod_cnt_layout.setContentsMargins(0, 5, 0, 0)

        mod_row = QHBoxLayout()
        mod_row.addWidget(QLabel("Modulation range:"))
        self.modulation_min_edit = QLineEdit("0.00")
        self.modulation_max_edit = QLineEdit("1.00")
        self.modulation_min_edit.setValidator(QDoubleValidator())
        self.modulation_max_edit.setValidator(QDoubleValidator())
        self.modulation_min_edit.setFixedWidth(50)
        self.modulation_max_edit.setFixedWidth(50)
        self.modulation_min_edit.setAlignment(Qt.AlignCenter)
        self.modulation_max_edit.setAlignment(Qt.AlignCenter)
        mod_row.addWidget(self.modulation_min_edit)
        mod_row.addWidget(QLabel("to"))
        mod_row.addWidget(self.modulation_max_edit)
        self.modulation_auto_btn = QPushButton("Auto")
        self.modulation_auto_btn.clicked.connect(self._on_mesh_auto_clicked)
        mod_row.addWidget(self.modulation_auto_btn)
        mod_row.addStretch(1)
        mod_cnt_layout.addLayout(mod_row)
        self.modulation_range_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.modulation_range_slider.setRange(0, 100)
        self.modulation_range_slider.setValue((0, 100))
        mod_cnt_layout.addWidget(self.modulation_range_slider)
        mesh_overlay_group_layout.addWidget(self.modulation_range_container)

        # Lifetime range controls (Lifetime mode). Each lifetime type keeps its
        # own range, in ns; it sets the width of the mesh band: an angular
        # wedge for the apparent phase and normal lifetimes, a ring for the
        # apparent modulation lifetime.
        self.lifetime_mesh_range_container = QWidget()
        lt_cnt_layout = QVBoxLayout(self.lifetime_mesh_range_container)
        lt_cnt_layout.setContentsMargins(0, 5, 0, 0)

        lt_row = QHBoxLayout()
        lt_row.addWidget(QLabel("Lifetime range (ns):"))
        self.lifetime_mesh_min_edit = QLineEdit("0.00")
        self.lifetime_mesh_max_edit = QLineEdit("10.00")
        self.lifetime_mesh_min_edit.setValidator(QDoubleValidator())
        self.lifetime_mesh_max_edit.setValidator(QDoubleValidator())
        self.lifetime_mesh_min_edit.setFixedWidth(50)
        self.lifetime_mesh_max_edit.setFixedWidth(50)
        self.lifetime_mesh_min_edit.setAlignment(Qt.AlignCenter)
        self.lifetime_mesh_max_edit.setAlignment(Qt.AlignCenter)
        lt_row.addWidget(self.lifetime_mesh_min_edit)
        lt_row.addWidget(QLabel("to"))
        lt_row.addWidget(self.lifetime_mesh_max_edit)
        self.lifetime_mesh_auto_btn = QPushButton("Auto")
        self.lifetime_mesh_auto_btn.clicked.connect(self._on_mesh_auto_clicked)
        lt_row.addWidget(self.lifetime_mesh_auto_btn)
        lt_row.addStretch(1)
        lt_cnt_layout.addLayout(lt_row)
        self.lifetime_mesh_range_slider = QRangeSlider(
            Qt.Orientation.Horizontal
        )
        self.lifetime_mesh_range_slider.setRange(0, 1000)
        self.lifetime_mesh_range_slider.setValue((0, 1000))
        lt_cnt_layout.addWidget(self.lifetime_mesh_range_slider)
        self.lifetime_mesh_range_container.setToolTip(
            "Lifetimes shown by the mesh. Wider ranges widen the wedge "
            "(apparent phase and normal lifetimes) or the ring (apparent "
            "modulation lifetime)."
        )
        mesh_overlay_group_layout.addWidget(self.lifetime_mesh_range_container)

        self.main_layout.addWidget(self.mesh_overlay_group)

        # Cautions about settings and frequencies a Calculate would change.
        self._settings_note = create_settings_note_label(self)
        self.main_layout.addWidget(self._settings_note)

        # Add Calculate button in its own row (at the bottom of this tab)
        self.calculate_lifetime_button = QPushButton("Calculate Output")
        self.calculate_lifetime_button.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self._refresh_calculate_button = setup_primary_button(
            self.calculate_lifetime_button,
            self._mapping_validation,
            self._on_calculate_lifetime_clicked,
            ready_tooltip="Calculate and display the selected output for all "
            "selected layers.",
        )
        self.main_layout.addWidget(self.calculate_lifetime_button)

        self.main_layout.addWidget(
            self._build_autoupdate_toggle(
                self.calculate_lifetime_button,
                self._mapping_validation,
                self._autoupdate_calculate_output,
                "Recalculate the selected output automatically whenever the "
                "output type, the frequency, the layer selection, or the "
                "filtered/calibrated phasor data change.",
            )
        )
        self.main_layout.addWidget(self.filter_box)
        self.main_layout.addStretch(1)

        # Re-evaluate the button whenever a required input changes.
        self.frequency_input.textChanged.connect(
            lambda _=None: self._refresh_action_buttons()
        )
        self.output_mode_combobox.currentTextChanged.connect(
            lambda _=None: self._on_mapping_input_changed()
        )
        self.lifetime_type_combobox.currentTextChanged.connect(
            lambda _=None: self._on_mapping_input_changed()
        )
        # Autoupdate follows the *committed* frequency (Enter or focus-out),
        # not every keystroke: "8" is a valid frequency on the way to "80".
        self.frequency_input.editingFinished.connect(
            self._on_mapping_input_changed
        )

        # Connect signals for mesh overlay
        self.mesh_overlay_checkbox.toggled.connect(
            self._on_mesh_overlay_toggled
        )
        self.mesh_clip_semicircle_checkbox.toggled.connect(
            self._on_mesh_clip_toggled
        )
        self.mesh_colorbar_checkbox.toggled.connect(
            self._on_mesh_colorbar_toggled
        )
        self.phase_range_slider.valueChanged.connect(
            self._on_phase_slider_changed
        )
        self.phase_min_edit.editingFinished.connect(
            self._on_phase_edits_changed
        )
        self.phase_max_edit.editingFinished.connect(
            self._on_phase_edits_changed
        )
        self.modulation_range_slider.valueChanged.connect(
            self._on_modulation_slider_changed
        )
        self.modulation_min_edit.editingFinished.connect(
            self._on_modulation_edits_changed
        )
        self.modulation_max_edit.editingFinished.connect(
            self._on_modulation_edits_changed
        )
        self.mesh_transparency_spinbox.valueChanged.connect(
            self._on_mesh_transparency_changed
        )
        self.lifetime_mesh_range_slider.valueChanged.connect(
            self._on_lifetime_mesh_slider_changed
        )
        self.lifetime_mesh_min_edit.editingFinished.connect(
            self._on_lifetime_mesh_edits_changed
        )
        self.lifetime_mesh_max_edit.editingFinished.connect(
            self._on_lifetime_mesh_edits_changed
        )

        self._sync_mode_widgets()
        self._update_calculate_button_text()
        self.update_apply_2d_text()

        # Connect to plot type changes in the parent widget
        if self.parent_widget is not None:
            self.parent_widget.plotter_inputs_widget.plot_type_combobox.currentTextChanged.connect(
                self.update_apply_2d_text
            )
            self.parent_widget.plotter_inputs_widget.semi_circle_checkbox.toggled.connect(
                self._on_plot_geometry_mode_toggled
            )
            self._connect_axes_limit_callbacks()

    def _connect_axes_limit_callbacks(self):
        """Subscribe to the plot's x/y limit changes, if not already subscribed."""
        if self.parent_widget is None:
            return
        axes = getattr(self.parent_widget.canvas_widget, 'axes', None)
        if axes is None:
            return
        if self._axes_limit_callback_cids:
            return
        self._axes_limit_callback_cids = [
            axes.callbacks.connect(
                'xlim_changed', self._on_axes_limits_changed
            ),
            axes.callbacks.connect(
                'ylim_changed', self._on_axes_limits_changed
            ),
        ]

    def _disconnect_axes_limit_callbacks(self):
        """Unsubscribe from the plot's x/y limit changes."""
        if self.parent_widget is None:
            return
        axes = getattr(self.parent_widget.canvas_widget, 'axes', None)
        if axes is None:
            return
        for cid in self._axes_limit_callback_cids:
            with contextlib.suppress(Exception):
                axes.callbacks.disconnect(cid)
        self._axes_limit_callback_cids = []

    def _on_axes_limits_changed(self, _axes):
        """Debounce a mesh overlay refresh when the plot is panned or zoomed."""
        if self._coloring_paused_by_tab:
            return
        if self.mesh_overlay_checkbox.isChecked():
            self._mesh_axes_update_timer.start()

    def _apply_mesh_after_axes_change(self):
        """Re-apply the mesh overlay once panning or zooming has settled.

        Also the debounced end of lifetime-mesh refreshes driven by the output
        layers' contrast limits, which change continuously while dragged.
        """
        if self._coloring_paused_by_tab:
            return
        self._refresh_mesh_overlay_if_needed()

    def update_apply_2d_text(self):
        """Update the checkbox text based on the current plot type."""
        plot_type = getattr(self.parent_widget, 'plot_type', 'HISTOGRAM2D')
        if plot_type == 'SCATTER':
            suffix = "Scatter plot"
        elif plot_type == 'CONTOUR':
            suffix = "Contour plot"
        elif plot_type == 'NONE':
            suffix = "Plot"
        else:
            suffix = "2D Histogram"
        self.apply_2d_colormap_checkbox.setText(f"Apply colormap to {suffix}")

    def _on_custom_color_clicked(self):
        """Open a color dialog to select a custom color."""
        from qtpy.QtWidgets import QColorDialog

        current_color = QColor()
        # Try to parse the current button color string
        style = self.custom_color_button.styleSheet()
        if "background-color: " in style:
            try:
                rgb_str = style.split("background-color: ")[1].split(";")[0]
                if rgb_str.startswith("rgb("):
                    r, g, b = map(int, rgb_str[4:-1].split(","))
                    current_color.setRgb(r, g, b)
            except (ValueError, IndexError):
                pass

        color = QColorDialog.getColor(
            current_color, self, "Select Custom Color"
        )
        if color.isValid():
            self._set_custom_color(color)
            # Find which color name we are updating
            # In 'solid' mode, we treat the color as a single-color colormap
            # or we just apply it. For now, we'll store it as a name or similar.
            # But the existing system uses colormap names.
            # If "Select color..." is active, we need to handle it.
            self._on_colormap_combobox_changed("Select color...")

    def _set_custom_color(self, color):
        """Set the custom color on the button and update internal state."""
        self.custom_color_button.setStyleSheet(
            f"background-color: {color.name()};"
        )
        self._custom_color = color

    def _update_calculate_button_text(self):
        """Label the calculate button after the selected output mode."""
        mode = self.output_mode_combobox.currentText()
        if mode == "Lifetime":
            self.calculate_lifetime_button.setText("Display Lifetime Map")
        elif mode == "Phase":
            self.calculate_lifetime_button.setText("Display Phase Map")
        else:
            self.calculate_lifetime_button.setText("Display Modulation Map")

    def _sync_mode_widgets(self):
        """Show only the controls that apply to the selected output mode."""
        is_lifetime_mode = (
            self.output_mode_combobox.currentText() == "Lifetime"
        )
        pw = self.parent_widget
        self.lifetime_output_widget.setVisible(is_lifetime_mode)
        self.frequency_widget.setVisible(is_lifetime_mode)
        # Hide the whole Coloring section for Lifetime; show it (and its
        # controls) for Phase/Modulation.
        self.coloring_box.setVisible(not is_lifetime_mode)
        self.colormap_widget.setVisible(not is_lifetime_mode)
        self.coloring_checkbox_widget.setVisible(not is_lifetime_mode)

        # The mesh applies to every mode; Lifetime swaps the phase and
        # modulation ranges for a single lifetime range.
        show_ranges = self.mesh_overlay_checkbox.isChecked()
        self.mesh_controls_widget.setVisible(show_ranges)
        self.phase_range_container.setVisible(
            show_ranges and not is_lifetime_mode
        )
        self.modulation_range_container.setVisible(
            show_ranges and not is_lifetime_mode
        )
        self.lifetime_mesh_range_container.setVisible(
            show_ranges and is_lifetime_mode
        )
        self.mesh_clip_semicircle_checkbox.setVisible(
            self._is_semicircle_mode()
        )
        self.mesh_colorbar_checkbox.setVisible(True)
        if not is_lifetime_mode:
            self._update_phase_slider_bounds_from_plot_mode()
        elif not show_ranges and pw is not None:
            # Lifetime mode only colours the mesh; without it a colorbar
            # left over from Phase/Modulation describes nothing on screen.
            pw._remove_mapping_colorbar()

        # A new filter defaults to the quantity currently on screen, which is
        # almost always the one the user is reasoning about.
        output_type = self._get_selected_output_type()
        if output_type in MAPPING_METRICS:
            self.filter_list.set_current_metric(output_type)

        self._refresh_action_buttons()

    def _is_semicircle_mode(self) -> bool:
        """Return whether the plot shows the universal semicircle.

        Defaults to True when the plotter cannot be queried, since the
        semicircle is the default geometry.
        """
        if self.parent_widget is None:
            return True
        with contextlib.suppress(AttributeError):
            return bool(self.parent_widget.toggle_semi_circle)
        with contextlib.suppress(AttributeError):
            return not bool(
                self.parent_widget.plotter_inputs_widget.semi_circle_checkbox.isChecked()
            )
        return True

    def _phase_max_allowed(self) -> float:
        """Return the largest phase value the sliders may reach, in radians."""
        return 2.0 * np.pi

    def _update_phase_slider_bounds_from_plot_mode(self):
        """Rescale the phase slider to the current plot geometry's phase range."""
        max_phase = self._phase_max_allowed()
        max_phase_i = int(max_phase * self.phase_range_factor)
        min_phase_i, cur_max_phase_i = self.phase_range_slider.value()
        min_phase_i = max(0, min(min_phase_i, max_phase_i))
        cur_max_phase_i = max(min_phase_i, min(cur_max_phase_i, max_phase_i))

        self._updating_settings = True
        try:
            self.phase_range_slider.setRange(0, max_phase_i)
            self.phase_range_slider.setValue((min_phase_i, cur_max_phase_i))
            self.phase_min_edit.setText(
                f"{min_phase_i / self.phase_range_factor:.2f}"
            )
            self.phase_max_edit.setText(
                f"{cur_max_phase_i / self.phase_range_factor:.2f}"
            )
        finally:
            self._updating_settings = False

    def _initialize_mesh_ranges_from_current_data(self):
        """Set the phase and modulation sliders to span the data's own range.

        Used when the layer has no stored mesh ranges to restore.
        """
        pw = self.parent_widget
        if pw is None:
            return
        features = pw.get_merged_features()
        if features is None:
            return
        g_flat, s_flat = features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            phase_values, modulation_values = phasor_to_polar(g_flat, s_flat)

        if not self._is_semicircle_mode():
            with np.errstate(invalid='ignore'):
                phase_values = np.mod(phase_values, 2.0 * np.pi)

        phase_max_allowed = self._phase_max_allowed()
        with np.errstate(invalid='ignore'):
            phase_valid = phase_values[np.isfinite(phase_values)]
            modulation_valid = modulation_values[
                np.isfinite(modulation_values)
            ]

        if phase_valid.size > 0:
            phase_min = float(np.nanmin(phase_valid))
            phase_max = float(np.nanmax(phase_valid))
            phase_min = max(0.0, min(phase_min, phase_max_allowed))
            phase_max = max(phase_min, min(phase_max, phase_max_allowed))
        else:
            phase_min, phase_max = 0.0, phase_max_allowed

        if modulation_valid.size > 0:
            mod_min = float(np.nanmin(modulation_valid))
            mod_max = float(np.nanmax(modulation_valid))
            mod_min = max(0.0, mod_min)
            mod_max = max(mod_min, mod_max)
        else:
            mod_min, mod_max = 0.0, 1.0

        phase_max_i = int(phase_max_allowed * self.phase_range_factor)
        phase_min_i = int(phase_min * self.phase_range_factor)
        phase_val_max_i = int(phase_max * self.phase_range_factor)

        mod_slider_max = max(1.0, mod_max)
        mod_slider_max_i = int(mod_slider_max * self.modulation_range_factor)
        mod_min_i = int(mod_min * self.modulation_range_factor)
        mod_max_i = int(mod_max * self.modulation_range_factor)

        phase_min_i = max(0, min(phase_min_i, phase_max_i))
        phase_val_max_i = max(phase_min_i, min(phase_val_max_i, phase_max_i))
        mod_min_i = max(0, min(mod_min_i, mod_slider_max_i))
        mod_max_i = max(mod_min_i, min(mod_max_i, mod_slider_max_i))

        self._updating_settings = True
        try:
            self.phase_range_slider.setRange(0, phase_max_i)
            self.phase_range_slider.setValue((phase_min_i, phase_val_max_i))
            self.phase_min_edit.setText(f"{phase_min:.2f}")
            self.phase_max_edit.setText(
                f"{phase_val_max_i / self.phase_range_factor:.2f}"
            )

            self.modulation_range_slider.setRange(0, mod_slider_max_i)
            self.modulation_range_slider.setValue((mod_min_i, mod_max_i))
            self.modulation_min_edit.setText(f"{mod_min:.2f}")
            self.modulation_max_edit.setText(
                f"{mod_max_i / self.modulation_range_factor:.2f}"
            )
        finally:
            self._updating_settings = False

        self._persist_current_mesh_ranges_to_metadata()

    def _refresh_action_buttons(self):
        """Refresh the ready/blocked styling on primary action buttons."""
        if (
            hasattr(self, '_refresh_calculate_button')
            and self._refresh_calculate_button is not None
        ):
            self._refresh_calculate_button()
        if hasattr(self, 'filter_list'):
            self._refresh_filter_add_button()

    def _phase_wraps(self) -> bool:
        """Return whether phase filters are measured on ``[0, 2pi)``."""
        return not self._is_semicircle_mode()

    def _filter_frequency(self, layer=None):
        """Return the frequency metric filters should be evaluated at.

        The tab's own input wins; a layer that was analysed earlier keeps the
        frequency it was analysed with, so a stack restored from metadata
        still means what it meant when it was created.
        """
        frequency = self._parse_positive_frequency(
            self.frequency_input.text().strip()
        )
        if layer is not None and hasattr(
            self.parent_widget, 'layer_frequency'
        ):
            # A non-primary layer is measured at its own stored frequency.
            own = self.parent_widget.layer_frequency(layer, frequency)
            if own is not None:
                return own
        if frequency is not None:
            return frequency
        if layer is not None:
            stored = layer.metadata.get('settings', {}).get('frequency')
            if stored is None:
                stored = layer.metadata.get('frequency')
            return self._parse_positive_frequency(stored)
        return None

    def _get_base_phasor_arrays(self, layer):
        """Return *layer*'s phasor arrays before any metric filter.

        This is the baseline every criterion is measured against: the
        intensity threshold, the median/wavelet filter and the mask are all
        applied, the metric filters are not. Measuring against the filtered
        arrays instead is what would make a stack order-dependent, since each
        new criterion would only ever see what the previous ones left.
        """
        return baseline_arrays(layer, self._layer_filter_params(layer))

    def _layer_filter_params(self, layer):
        """Return *layer*'s own intensity filter/threshold parameters."""
        if self.parent_widget is None:
            return {}
        return self.parent_widget._filter_params_from_settings(layer)

    def _compute_metric_for_layer(self, layer, metric, harmonic, arrays=None):
        """Return *metric* evaluated on *layer*'s unfiltered baseline."""
        mean, real, imag = (
            arrays
            if arrays is not None
            else self._get_base_phasor_arrays(layer)
        )
        if mean is None:
            return None
        plane_real, plane_imag = select_harmonic(
            real, imag, layer.metadata.get('harmonics'), harmonic, mean.ndim
        )
        return compute_metric(
            metric,
            plane_real,
            plane_imag,
            harmonic=harmonic,
            frequency=self._filter_frequency(layer),
            wrap_phase=self._phase_wraps(),
        )

    def _new_filter_params(self, metric, layer=None):
        """Return the extra parameters to freeze into a new criterion."""
        params = {}
        if requires_frequency(metric):
            frequency = self._filter_frequency(layer)
            if frequency is not None:
                params['frequency'] = frequency
        return params

    def _current_harmonic(self):
        """Return the harmonic a new criterion should be measured on."""
        return getattr(self.parent_widget, 'harmonic', 1) or 1

    def _filter_add_blocked_reason(self):
        """Return why a filter cannot be added right now, else ``None``."""
        if not self._filter_layers():
            return "Select at least one image layer with phasor features."
        metric = self.filter_list.current_metric()
        if requires_frequency(metric) and self._filter_frequency() is None:
            return "Enter the frequency (MHz) before filtering on a lifetime."
        return None

    def _refresh_filter_add_button(self):
        """Explain on the button itself when a filter cannot be added yet."""
        reason = self._filter_add_blocked_reason()
        self.filter_list.add_button.setEnabled(reason is None)
        if reason is not None:
            self.filter_list.add_button.setToolTip(reason)
        else:
            self.filter_list.add_button.setToolTip(
                "Add a filter on the selected quantity."
            )

    def _filter_layers(self):
        """Return the layers the filter stack is written to."""
        if self.parent_widget is None:
            return []
        try:
            return list(self.parent_widget.get_selected_layers())
        except (AttributeError, RuntimeError):
            return []

    def _primary_filter_layer(self):
        """Return the layer whose stack the cards show, or ``None``."""
        layers = self._filter_layers()
        return layers[0] if layers else None

    def _sync_filter_ui(self):
        """Show the stack stored on the primary layer, and refresh its ranges."""
        layer = self._primary_filter_layer()
        if layer is None:
            self.filter_list.set_filters([])
            self.filter_list.set_filter_stats({}, "")
            return
        self.filter_list.set_filters(get_filters(layer))
        output_type = self._get_selected_output_type()
        if output_type in MAPPING_METRICS and not self.filter_list.filters():
            # An empty list should offer the quantity the user is looking at.
            self.filter_list.set_current_metric(output_type)
        self._refresh_filter_bounds()
        self._refresh_filter_stats()
        self._refresh_filter_add_button()

    def _filter_bounds_for(self, metric, arrays=None):
        """Return the ``(low, high)`` data range of *metric*, or ``None``.

        Measured on the *unfiltered* baseline of the primary layer, so a
        filter can never shrink the range the next filter is offered -- the
        trap that made the old single-range control feel like it was hiding
        data.
        """
        layer = self._primary_filter_layer()
        if layer is None or metric not in MAPPING_METRICS:
            return None
        harmonic = getattr(self.parent_widget, 'harmonic', 1) or 1
        values = self._compute_metric_for_layer(
            layer, metric, harmonic, arrays=arrays
        )
        if values is None:
            return None
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None
        return float(finite.min()), float(finite.max())

    def _refresh_filter_bounds(self):
        """Widen each card's slider to the full data range of its metric."""
        layer = self._primary_filter_layer()
        if layer is None:
            return
        metrics = {f['metric'] for f in self.filter_list.filters()}
        current = self.filter_list.current_metric()
        if current:
            metrics.add(current)
        metrics &= set(MAPPING_METRICS)
        arrays = self._get_base_phasor_arrays(layer)
        for metric in metrics:
            bounds = self._filter_bounds_for(metric, arrays=arrays)
            if bounds is not None:
                self.filter_list.set_metric_bounds(metric, *bounds)

    def _refresh_filter_stats(self):
        """Report what each criterion, and the stack as a whole, keeps."""
        filters = self.filter_list.filters()
        layer = self._primary_filter_layer()
        if layer is None or not filters:
            self.filter_list.set_filter_stats({}, "")
            return
        mean, real, imag = self._get_base_phasor_arrays(layer)
        harmonics = layer.metadata.get('harmonics')
        stats = {}
        for entry in filters:
            single = dict(entry, enabled=True)
            mask = combined_mask([single], mean, real, imag, harmonics)
            fraction = kept_fraction(mask, mean)
            prefix = "" if entry['enabled'] else "off · "
            stats[entry['id']] = f"{prefix}keeps {fraction:.1%} of the pixels"
        total_mask = combined_mask(filters, mean, real, imag, harmonics)
        active = sum(1 for f in filters if f['enabled'])
        kept = kept_fraction(total_mask, mean)
        # Kept short so it fits the dock on one line; the sentence it stands
        # for is the tooltip.
        summary = f"{active} of {len(filters)} on · {kept:.1%} kept"
        detail = (
            f"{active} of {len(filters)} filters are active, and together "
            f"they keep {kept:.1%} of the measured pixels of {layer.name}."
        )
        self.filter_list.set_filter_stats(stats, summary, detail=detail)

    def _rebuild_layer_from_filters(self, layer, filters, on_error=None):
        """Rewrite *layer*'s phasor arrays from its baseline plus *filters*."""
        rebuild_layer_from_filters(
            layer,
            filters,
            filter_params=self._layer_filter_params(layer),
            on_error=on_error,
        )

    def _on_filters_changed(self, filters):
        """Persist the edited stack and rebuild everything downstream of it."""
        self._apply_filter_stack(filters)

    def _apply_filter_stack(self, filters=None, layers=None):
        """Write *filters* to *layers* and re-derive their phasor data.

        Everything the tab shows -- the phasor plot, the output maps, the
        histogram and the statistics table -- is derived from the layer's G/S
        arrays, so rebuilding those from the stack is the only step needed to
        keep all four in sync.
        """
        if self.parent_widget is None:
            return
        layers = self._filter_layers() if layers is None else list(layers)
        if not layers:
            return
        if filters is None:
            filters = self.filter_list.filters()

        problems = []
        self._applying_mapping_filter = True
        try:
            for layer in layers:
                stored = set_filters(layer, filters)
                self._rebuild_layer_from_filters(
                    layer, stored, on_error=problems.append
                )
            self.parent_widget.refresh_phasor_data()
            self._recalculate_after_filter_change()
        finally:
            self._applying_mapping_filter = False

        self._sync_filter_ui()
        for message in dict.fromkeys(problems):
            show_warning(message)

    def _recalculate_after_filter_change(self):
        """Refresh the output maps and the histogram for the new stack."""
        if self._has_calculated_output:
            self._calculate_and_display_output(show_warnings=False)
        else:
            self.plot_lifetime_histogram()

    def _reapply_filter_stack(self, selected_layers):
        """Re-derive the phasor arrays of layers that carry a filter stack.

        Called by the plotter after something else rewrote the arrays from
        the originals (a mask assignment, an imported analysis), so that the
        criteria on screen and the pixels on screen still agree.
        """
        for layer in selected_layers or []:
            filters = get_filters(layer)
            if not filters:
                continue
            self._rebuild_layer_from_filters(layer, filters)

    def _on_plot_geometry_mode_toggled(self, _checked):
        """Callback when the plot switches between semicircle and full polar."""
        self._update_phase_slider_bounds_from_plot_mode()
        self._sync_mode_widgets()
        if self.mesh_overlay_checkbox.isChecked():
            settings = self._get_current_layer_mapping_settings(create=False)
            self._sync_mesh_ranges(settings)
            self._persist_current_mesh_ranges_to_metadata()
        self.reapply_if_active()

    def _refresh_mesh_overlay_if_needed(self):
        """Re-apply the mesh overlay when it is enabled for the current output."""
        if self.mesh_overlay_checkbox.isChecked():
            self._apply_output_coloring(self._get_selected_output_type())

    def _apply_output_coloring(self, output_type: str):
        """Redraw the plot colouring and mesh overlay for *output_type*."""
        if output_type in LIFETIME_OUTPUT_TYPES:
            self._apply_lifetime_mesh(output_type)
        else:
            self._apply_histogram_coloring(output_type)

    def _sync_mesh_ranges(self, settings):
        """Restore every mesh range from *settings*, else fit it to the data."""
        if not self._restore_mesh_ranges_from_settings(settings):
            self._initialize_mesh_ranges_from_current_data()
        self._sync_lifetime_mesh_range(settings)

    def _on_mesh_auto_clicked(self):
        """Fit the displayed mesh range(s) to the data and redraw the mesh."""
        if self._get_selected_output_type() in LIFETIME_OUTPUT_TYPES:
            self._initialize_lifetime_mesh_range_from_current_data()
            self._sync_lifetime_mesh_range_to_histogram()
        else:
            self._initialize_mesh_ranges_from_current_data()
        self._refresh_mesh_overlay_if_needed()

    def get_selected_output_display_name(self) -> str:
        """Return the selected output's user-facing name.

        The three lifetime variants all collapse to "Lifetime"; other
        outputs use their own name.
        """
        output_type = self._get_selected_output_type()
        if output_type in {
            "Apparent Phase Lifetime",
            "Apparent Modulation Lifetime",
            "Normal Lifetime",
        }:
            return "Lifetime"
        return output_type

    def _get_default_lifetime_settings(self):
        """Get default settings dictionary for lifetime parameters."""
        return {
            'lifetime_type': 'Apparent Phase Lifetime',
            'lifetime_range_min': None,
            'lifetime_range_max': None,
            'output_type': 'Apparent Phase Lifetime',
            'range_min': None,
            'range_max': None,
            'output_ranges': {},
            # {output_type: layer_colormap_to_settings(...)}
            'output_colormaps': {},
            'mesh_overlay_enabled': False,
            'mesh_clip_semicircle_enabled': False,
            'mesh_colorbar_enabled': False,
            'mesh_alpha': DEFAULT_MESH_ALPHA,
            'mesh_phase_min': None,
            'mesh_phase_max': None,
            'mesh_modulation_min': None,
            'mesh_modulation_max': None,
            # {lifetime output type: [min_ns, max_ns]}
            'mesh_lifetime_ranges': {},
        }

    def _get_current_layer_mapping_settings(self, create: bool = False):
        """Return the primary layer's mapping settings, or None if unavailable.

        Set ``create`` to add a default settings entry when the layer has
        none yet.
        """
        layer_name = self.parent_widget.get_primary_layer_name()
        if not layer_name or layer_name not in self.viewer.layers:
            return None
        layer = self.viewer.layers[layer_name]
        return self._get_phasor_mapping_settings(layer, create=create)

    def _persist_current_mesh_ranges_to_metadata(self):
        """Save the mesh slider ranges and toggles to the layer's metadata.

        Does nothing while the widgets are being updated programmatically,
        so restoring settings cannot write them straight back.
        """
        if self._updating_settings:
            return
        phase_min_i, phase_max_i = self.phase_range_slider.value()
        mod_min_i, mod_max_i = self.modulation_range_slider.value()
        self._update_lifetime_setting_in_metadata(
            'mesh_phase_min', phase_min_i / self.phase_range_factor
        )
        self._update_lifetime_setting_in_metadata(
            'mesh_phase_max', phase_max_i / self.phase_range_factor
        )
        self._update_lifetime_setting_in_metadata(
            'mesh_modulation_min', mod_min_i / self.modulation_range_factor
        )
        self._update_lifetime_setting_in_metadata(
            'mesh_modulation_max', mod_max_i / self.modulation_range_factor
        )
        self._update_lifetime_setting_in_metadata(
            'mesh_clip_semicircle_enabled',
            self.mesh_clip_semicircle_checkbox.isChecked(),
        )
        self._update_lifetime_setting_in_metadata(
            'mesh_colorbar_enabled',
            self.mesh_colorbar_checkbox.isChecked(),
        )

    def _restore_mesh_ranges_from_settings(self, settings) -> bool:
        """Apply stored mesh ranges to the sliders.

        Returns False when *settings* is missing or incomplete, leaving the
        widgets untouched so the caller can fall back to the data ranges.
        """
        if settings is None:
            return False

        phase_min = settings.get('mesh_phase_min')
        phase_max = settings.get('mesh_phase_max')
        mod_min = settings.get('mesh_modulation_min')
        mod_max = settings.get('mesh_modulation_max')
        clip_semi = settings.get('mesh_clip_semicircle_enabled', False)
        show_colorbar = settings.get('mesh_colorbar_enabled', False)
        if any(v is None for v in (phase_min, phase_max, mod_min, mod_max)):
            return False

        phase_max_allowed = self._phase_max_allowed()
        phase_min = float(np.clip(float(phase_min), 0.0, phase_max_allowed))
        phase_max = float(
            np.clip(float(phase_max), phase_min, phase_max_allowed)
        )
        mod_min = max(0.0, float(mod_min))
        mod_max = max(mod_min, float(mod_max))

        phase_slider_max_i = int(phase_max_allowed * self.phase_range_factor)
        phase_min_i = int(phase_min * self.phase_range_factor)
        phase_max_i = int(phase_max * self.phase_range_factor)

        mod_slider_max = max(1.0, mod_max)
        mod_slider_max_i = int(mod_slider_max * self.modulation_range_factor)
        mod_min_i = int(mod_min * self.modulation_range_factor)
        mod_max_i = int(mod_max * self.modulation_range_factor)

        self.phase_range_slider.setRange(0, phase_slider_max_i)
        self.phase_range_slider.setValue((phase_min_i, phase_max_i))
        self.phase_min_edit.setText(f"{phase_min:.2f}")
        self.phase_max_edit.setText(f"{phase_max:.2f}")

        self.modulation_range_slider.setRange(0, mod_slider_max_i)
        self.modulation_range_slider.setValue((mod_min_i, mod_max_i))
        self.modulation_min_edit.setText(f"{mod_min:.2f}")
        self.modulation_max_edit.setText(f"{mod_max:.2f}")
        self.mesh_clip_semicircle_checkbox.setChecked(bool(clip_semi))
        self.mesh_colorbar_checkbox.setChecked(bool(show_colorbar))
        return True

    @staticmethod
    def _output_requires_frequency(output_type: str) -> bool:
        """Return whether *output_type* can only be computed with a frequency."""
        return output_type in LIFETIME_OUTPUT_TYPES

    @staticmethod
    def _parse_positive_frequency(text):
        """Return a finite positive frequency, or None for invalid text."""
        try:
            frequency = float(text)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(frequency) or frequency <= 0:
            return None
        return frequency

    def _get_selected_output_type(self) -> str:
        """Return the selected output type, resolving the lifetime sub-type."""
        mode = self.output_mode_combobox.currentText()
        if mode == "Lifetime":
            return self.lifetime_type_combobox.currentText()
        return mode

    def _get_selected_source_names(self) -> set[str]:
        """Return names checked in the plotter's Phasor Layers selector."""
        if self.parent_widget is None:
            return set()
        try:
            return {
                layer.name
                for layer in self.parent_widget.get_selected_layers()
            }
        except (AttributeError, RuntimeError):
            return set()

    def _mapping_output_info(self, layer):
        """Return ``(output_type, source_name)`` for a Mapping output layer."""
        if not isinstance(layer, Image):
            return None
        tag = layer.metadata.get(_MAPPING_OUTPUT_METADATA_KEY)
        if isinstance(tag, dict):
            output_type = tag.get('output_type')
            source_name = tag.get('source_layer')
            if output_type in _MAPPING_OUTPUT_TYPES and source_name:
                return output_type, source_name
        for output_type in _MAPPING_OUTPUT_TYPES:
            prefix = f"{output_type}: "
            if layer.name.startswith(prefix):
                source_name = layer.name[len(prefix) :]
                if source_name in self.viewer.layers:
                    source_layer = self.viewer.layers[source_name]
                    if (
                        isinstance(source_layer, Image)
                        and 'G' in source_layer.metadata
                        and 'S' in source_layer.metadata
                    ):
                        return output_type, source_name
        return None

    def _mapping_output_layers(self, output_type=None, selected_only=False):
        """Return Mapping outputs keyed by source name."""
        selected_names = (
            self._get_selected_source_names() if selected_only else None
        )
        result = {}
        for layer in self.viewer.layers:
            info = self._mapping_output_info(layer)
            if info is None:
                continue
            layer_output_type, source_name = info
            if output_type is not None and layer_output_type != output_type:
                continue
            if (
                selected_names is not None
                and source_name not in selected_names
            ):
                continue
            existing = result.get(source_name)
            layer_is_tagged = isinstance(
                layer.metadata.get(_MAPPING_OUTPUT_METADATA_KEY), dict
            )
            existing_is_tagged = existing is not None and isinstance(
                existing.metadata.get(_MAPPING_OUTPUT_METADATA_KEY), dict
            )
            if existing is None or (
                layer_is_tagged and not existing_is_tagged
            ):
                result[source_name] = layer
        return result

    def _set_metric_layers(self, layers):
        """Replace the active Mapping layer registry without duplicate events."""
        for layer in self.metric_layers:
            with contextlib.suppress(
                AttributeError, RuntimeError, TypeError, ValueError
            ):
                layer.events.colormap.disconnect(self._on_colormap_changed)
                layer.events.contrast_limits.disconnect(
                    self._on_colormap_changed
                )
                layer.events.gamma.disconnect(self._on_colormap_changed)

        self.metric_layers = list(layers)
        self.lifetime_layers = list(self.metric_layers)
        self.lifetime_layer = (
            self.metric_layers[0] if self.metric_layers else None
        )

        for layer in self.metric_layers:
            with contextlib.suppress(
                AttributeError, RuntimeError, TypeError, ValueError
            ):
                layer.events.colormap.disconnect(self._on_colormap_changed)
                layer.events.contrast_limits.disconnect(
                    self._on_colormap_changed
                )
                layer.events.gamma.disconnect(self._on_colormap_changed)
            layer.events.colormap.connect(self._on_colormap_changed)
            layer.events.contrast_limits.connect(self._on_colormap_changed)
            layer.events.gamma.connect(self._on_colormap_changed)

        if self.lifetime_layer is not None:
            self.lifetime_colormap = self.lifetime_layer.colormap.colors
            self.colormap_contrast_limits = self.lifetime_layer.contrast_limits
            self.colormap_gamma = self.lifetime_layer.gamma

    def _sync_mapping_output_visibility(self):
        """Show Mapping outputs only when their source layer is selected."""
        selected_names = self._get_selected_source_names()
        self._updating_linked_layers = True
        try:
            for layer in self.viewer.layers:
                info = self._mapping_output_info(layer)
                if info is None:
                    continue
                _, source_name = info
                desired_visible = source_name in selected_names
                if layer.visible != desired_visible:
                    layer.visible = desired_visible
        finally:
            self._updating_linked_layers = False

    def on_layer_selection_changed(self):
        """Refresh Mapping outputs after the Phasor Layers selection changes."""
        self._sync_mapping_output_visibility()
        output_type = self._get_selected_output_type()
        selected_layers = self._mapping_output_layers(
            output_type, selected_only=True
        )
        self._set_metric_layers(selected_layers.values())
        self.plot_lifetime_histogram()

        if not selected_layers:
            self._clear_2d_coloring()

    @staticmethod
    def _get_output_colormap_name(output_type: str) -> str:
        """Return the default colormap name to use for *output_type*."""
        if output_type == "Phase":
            return "cool"
        if output_type == "Modulation":
            return "PiYG"
        return "plasma"

    def _configure_histogram_labels_for_output(self, output_type: str):
        """Relabel the histogram axis and range slider to *output_type*'s units."""
        if output_type in {
            "Apparent Phase Lifetime",
            "Apparent Modulation Lifetime",
            "Normal Lifetime",
        }:
            self.histogram_widget.xlabel = "Lifetime (ns)"
            self.histogram_widget._range_label_prefix = "Lifetime range (ns)"
            return
        if output_type == "Phase":
            self.histogram_widget.xlabel = "Phase (rad)"
            self.histogram_widget._range_label_prefix = "Phase range (rad)"
            return
        self.histogram_widget.xlabel = "Modulation"
        self.histogram_widget._range_label_prefix = "Modulation range"

    def _on_output_mode_changed(self, mode: str):
        """Callback when the output mode combobox is changed.

        Re-syncs the dependent controls, colormap, histogram labels and 2D
        colouring, and records the new output type in the layer metadata.
        """
        is_lifetime_mode = mode == "Lifetime"
        self.lifetime_type_combobox.setEnabled(is_lifetime_mode)
        self._sync_mode_widgets()
        self._update_calculate_button_text()
        self.colormap_combobox.blockSignals(True)
        try:
            if mode == "Phase":
                self.colormap_combobox.setCurrentText(
                    self._phase_colormap_name
                )
            elif mode == "Modulation":
                self.colormap_combobox.setCurrentText(
                    self._modulation_colormap_name
                )
        finally:
            self.colormap_combobox.blockSignals(False)
        self.custom_color_button.setVisible(
            self.colormap_combobox.currentText() == "Select color..."
        )
        output_type = self._get_selected_output_type()
        self.current_output_type = output_type
        self._set_frequency_input_enabled(
            self._output_requires_frequency(output_type)
        )
        self._configure_histogram_labels_for_output(output_type)
        self._sync_filter_ui()
        self.outputTypeChanged.emit(output_type)
        is_reactive_transition = (
            self._has_calculated_output and not self._updating_settings
        )
        if is_reactive_transition or output_type not in {
            "Phase",
            "Modulation",
        }:
            self._clear_2d_coloring()
        else:
            can_apply_coloring = (
                self.apply_2d_colormap_checkbox.isChecked()
                or self.mesh_overlay_checkbox.isChecked()
                or self.mesh_colorbar_checkbox.isChecked()
            )
            if can_apply_coloring:
                self._apply_histogram_coloring(output_type)
            else:
                self._clear_2d_coloring()
        if (
            not self._updating_settings
            and output_type in LIFETIME_OUTPUT_TYPES
            and self.mesh_overlay_checkbox.isChecked()
        ):
            self._sync_lifetime_mesh_range()
            self._apply_lifetime_mesh(output_type)
        if not self._updating_settings:
            self._update_lifetime_setting_in_metadata(
                'output_type', output_type
            )
        self._schedule_active_output_refresh()

    def _on_colormap_combobox_changed(self, name: str):
        """Callback when the colormap combobox is changed.

        The choice is remembered per output type, so Phase and Modulation
        keep their own colormaps.
        """
        self.custom_color_button.setVisible(name == "Select color...")
        output_type = self._get_selected_output_type()

        if name == "Select color..." and not hasattr(self, "_custom_color"):
            # Set a default initial color if none exists
            self._set_custom_color(QColor(255, 0, 0))  # Default to red

        if output_type == "Phase":
            self._phase_colormap_name = name
        elif output_type == "Modulation":
            self._modulation_colormap_name = name
        if output_type in {"Phase", "Modulation"}:
            self._pending_combobox_colormaps.add(output_type)

        if output_type in {"Phase", "Modulation"} and (
            self.apply_2d_colormap_checkbox.isChecked()
            or self.mesh_overlay_checkbox.isChecked()
            or self.mesh_colorbar_checkbox.isChecked()
        ):
            actual_cmap_to_apply = self._resolve_layer_colormap(name)

            matching_layers = [
                layer
                for layer in self.metric_layers
                if (
                    (info := self._mapping_output_info(layer)) is not None
                    and info[0] == output_type
                )
            ]
            for layer in matching_layers:
                if layer in self.viewer.layers:
                    layer.colormap = actual_cmap_to_apply
            if matching_layers:
                self._apply_histogram_coloring(output_type)

    def _resolve_layer_colormap(self, cmap_name: str):
        """Return a napari-compatible colormap value for image layers."""
        if not hasattr(self, "_custom_color"):
            self._set_custom_color(QColor(255, 0, 0))

        resolved = resolve_napari_layer_colormap(
            cmap_name,
            custom_color=self._custom_color,
            sentinel="Select color...",
        )
        return cmap_name if resolved is None else resolved

    def _on_apply_2d_colormap_checkbox_changed(self, checked):
        """Callback when the "apply colormap to plot" checkbox is toggled."""
        output_type = self._get_selected_output_type()
        if output_type not in {"Phase", "Modulation"}:
            self._clear_2d_coloring()
            return
        if checked or self.mesh_overlay_checkbox.isChecked():
            self._apply_histogram_coloring(output_type)
        else:
            self._clear_2d_coloring()

    def _clear_2d_coloring(self):
        """Remove the mapping overlay and colorbar, restoring the density plot."""
        self._remove_overlay()
        pw = self.parent_widget
        if pw is not None:
            pw._remove_mapping_colorbar()
            self._set_histogram_density_visible(pw, True)
            if getattr(pw, 'plot_type', None) == 'SCATTER':
                pw.refresh_current_plot()
            elif getattr(pw, 'plot_type', None) == 'CONTOUR':
                contour_collections = getattr(pw, '_contour_collections', [])
                for cs in contour_collections:
                    if hasattr(cs, 'collections') and len(cs.collections) > 0:
                        for col in cs.collections:
                            col.set_visible(True)
                    elif hasattr(cs, 'set_visible'):
                        cs.set_visible(True)
            pw.canvas_widget.figure.canvas.draw_idle()

    def _restore_plot_coloring_state_without_mesh(self):
        """Restore non-mesh plot rendering while keeping mesh logic independent."""
        pw = self.parent_widget
        if pw is None:
            return
        self._set_histogram_density_visible(pw, True)
        if getattr(pw, 'plot_type', None) == 'SCATTER':
            pw.refresh_current_plot()
        elif getattr(pw, 'plot_type', None) == 'CONTOUR':
            contour_collections = getattr(pw, '_contour_collections', [])
            for cs in contour_collections:
                if hasattr(cs, 'collections') and len(cs.collections) > 0:
                    for col in cs.collections:
                        col.set_visible(True)
                elif hasattr(cs, 'set_visible'):
                    cs.set_visible(True)
        pw.canvas_widget.figure.canvas.draw_idle()

    def _get_phasor_mapping_settings(self, layer, create: bool = False):
        """Return *layer*'s mapping settings, including unsaved edits.

        Read-only: edits go through :meth:`_stage_mapping_values`, a run
        through :meth:`_commit_mapping_settings`. Layers written before the
        ``phasor_mapping`` key existed keep them under ``lifetime``. With
        ``create``, a layer without any gets a fresh default dict (which is
        not stored).
        """
        if layer is None:
            return None
        if self.parent_widget is not None and hasattr(
            self.parent_widget, 'layer_settings'
        ):
            settings_container = self.parent_widget.layer_settings(layer)
        else:
            settings_container = layer.metadata.get('settings') or {}
        mapping_settings = settings_container.get('phasor_mapping')
        if mapping_settings is None:
            mapping_settings = settings_container.get('lifetime')
        if mapping_settings is None and create:
            mapping_settings = self._get_default_lifetime_settings()
        return mapping_settings

    def _stage_mapping_values(self, updates):
        """Keep *updates* as unsaved mapping settings of the primary layer.

        They are stored in the layers when Calculate runs (see
        :meth:`_commit_mapping_settings`).
        """
        if self.parent_widget is None:
            return
        primary = self.parent_widget.get_primary_layer()
        if primary is None:
            return
        block = copy.deepcopy(
            self._get_phasor_mapping_settings(primary, create=True)
        )
        block.update(updates)
        self.parent_widget.stage_setting('phasor_mapping', block)

    def _update_lifetime_setting_in_metadata(self, key, value):
        """Keep an edited mapping setting as the primary's unsaved setting."""
        if self._updating_settings:
            return

        updates = {key: value}
        if key == 'output_type':
            if value in LIFETIME_OUTPUT_TYPES:
                updates['lifetime_type'] = value
        elif key == 'range_min':
            updates['lifetime_range_min'] = value
        elif key == 'range_max':
            updates['lifetime_range_max'] = value
        self._stage_mapping_values(updates)

    def _collect_mapping_settings(self):
        """Return the mapping settings a Calculate would store.

        The primary layer's settings (with its unsaved edits) completed by
        what the controls show, so a layer that never had any gets the
        parameters it is actually analysed with.
        """
        primary = (
            self.parent_widget.get_primary_layer()
            if self.parent_widget is not None
            else None
        )
        block = copy.deepcopy(
            self._get_phasor_mapping_settings(primary, create=True)
            if primary is not None
            else self._get_default_lifetime_settings()
        )
        output_type = self._get_selected_output_type()
        block['output_type'] = output_type
        if output_type in LIFETIME_OUTPUT_TYPES:
            block['lifetime_type'] = output_type
        block['mesh_overlay_enabled'] = bool(
            self.mesh_overlay_checkbox.isChecked()
        )
        block['mesh_alpha'] = self._mesh_alpha()
        block['mesh_clip_semicircle_enabled'] = bool(
            self.mesh_clip_semicircle_checkbox.isChecked()
        )
        block['mesh_colorbar_enabled'] = bool(
            self.mesh_colorbar_checkbox.isChecked()
        )
        return block

    @staticmethod
    def _mapping_merge_rule(output_type):
        """Return how a run for *output_type* merges into stored settings.

        Ranges and colormaps are kept per output type; a run only replaces
        the entries of the output it computed.
        """
        return chain_merges(
            *(
                merge_keyed_path((key,), [output_type])
                for key in (
                    'output_ranges',
                    'output_colormaps',
                    'mesh_lifetime_ranges',
                )
            )
        )

    @staticmethod
    def _sync_legacy_alias(layers):
        """Point the legacy ``lifetime`` key at ``phasor_mapping``."""
        for layer in layers:
            settings = layer.metadata.get('settings') or {}
            if 'phasor_mapping' in settings:
                settings['lifetime'] = settings['phasor_mapping']

    def _commit_mapping_settings(self, layers):
        """Store the run's mapping settings in every analysed layer."""
        if self.parent_widget is None or not layers:
            return
        for layer in layers:
            # Layers from before the ``phasor_mapping`` key: merge into the
            # settings they have under the legacy name.
            settings = layer.metadata.get('settings') or {}
            if 'phasor_mapping' not in settings and isinstance(
                settings.get('lifetime'), dict
            ):
                settings['phasor_mapping'] = settings['lifetime']
        block = self._collect_mapping_settings()
        self.parent_widget.commit_analysis_settings(
            {'phasor_mapping': block},
            layers=layers,
            merge={
                'phasor_mapping': self._mapping_merge_rule(
                    block['output_type']
                )
            },
        )
        self._sync_legacy_alias(layers)

    def _analysed_layers(self):
        """Return the source layers of the output layers on display."""
        layers = []
        for output_layer in self.metric_layers:
            info = self._mapping_output_info(output_layer)
            if info is None or info[1] not in self.viewer.layers:
                continue
            source = self.viewer.layers[info[1]]
            if source not in layers:
                layers.append(source)
        return layers

    def _refresh_settings_note(self):
        """Caution about settings and frequencies a Calculate would change."""
        note = getattr(self, '_settings_note', None)
        if note is None or self.parent_widget is None:
            return
        if getattr(self, '_needs_update', False):
            # The controls still show another layer; refreshed on restore.
            return
        block = self._collect_mapping_settings()
        rule = self._mapping_merge_rule(block['output_type'])
        messages = [
            self.parent_widget.settings_overwrite_message(
                'phasor_mapping_tab',
                values={'phasor_mapping': block, 'lifetime': block},
                merge={'phasor_mapping': rule, 'lifetime': rule},
                keys=['phasor_mapping', 'lifetime'],
                action="Calculating",
            )
        ]
        if self._output_requires_frequency(block['output_type']):
            messages += self.parent_widget.frequency_note_messages(
                self.frequency_input.text()
            )
        set_settings_note(note, messages)

    def _restore_combobox_colormaps(self, settings):
        """Show the stored Phase / Modulation colormaps in the combobox.

        A combobox pick made for another layer must not carry over, so any
        pending one is dropped. Colormaps the combobox does not list (a
        picked solid colour) are left to the stored settings, which the next
        run applies to the layers directly.
        """
        self._pending_combobox_colormaps.clear()
        stored = settings.get('output_colormaps') or {}
        for output_type, attr in (
            ("Phase", "_phase_colormap_name"),
            ("Modulation", "_modulation_colormap_name"),
        ):
            entry = stored.get(output_type)
            name = (
                entry.get('colormap_name') if isinstance(entry, dict) else None
            )
            if name and self.colormap_combobox.findText(name) >= 0:
                setattr(self, attr, name)

        name = {
            "Phase": self._phase_colormap_name,
            "Modulation": self._modulation_colormap_name,
        }.get(self.output_mode_combobox.currentText())
        if name is None:
            return
        self.colormap_combobox.blockSignals(True)
        try:
            self.colormap_combobox.setCurrentText(name)
        finally:
            self.colormap_combobox.blockSignals(False)
        self.custom_color_button.setVisible(name == "Select color...")

    def _restore_lifetime_settings_from_metadata(self):
        """Restore all lifetime settings from the current layer's metadata."""
        layer_name = self.parent_widget.get_primary_layer_name()
        if not layer_name or layer_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[layer_name]

        frequency = self.parent_widget.layer_settings(layer).get('frequency')
        self._updating_settings = True
        try:
            if frequency is not None:
                self.frequency_input.setText(str(frequency))
            else:
                self.frequency_input.clear()
        finally:
            self._updating_settings = False

        settings = self._get_phasor_mapping_settings(layer, create=False)
        if settings is None:
            self._updating_settings = True
            try:
                self.output_mode_combobox.setCurrentText('Lifetime')
                self.lifetime_type_combobox.setCurrentText(
                    'Apparent Phase Lifetime'
                )
                self._configure_histogram_labels_for_output(
                    'Apparent Phase Lifetime'
                )
                self.lifetime_range_slider.setValue((0, 100))
                self.lifetime_min_edit.setText('0.0')
                self.lifetime_max_edit.setText('100.0')
                self.lifetime_range_label.setText(
                    f'{self.histogram_widget._range_label_prefix}:'
                )
                self._set_frequency_input_enabled(True)
                self.mesh_overlay_checkbox.blockSignals(True)
                try:
                    self.mesh_overlay_checkbox.setChecked(False)
                finally:
                    self.mesh_overlay_checkbox.blockSignals(False)
                self.mesh_transparency_spinbox.blockSignals(True)
                try:
                    self.mesh_transparency_spinbox.setValue(
                        1.0 - DEFAULT_MESH_ALPHA
                    )
                finally:
                    self.mesh_transparency_spinbox.blockSignals(False)
                self._sync_mode_widgets()
                self._sync_filter_ui()
                self._clear_2d_coloring()
                self.histogram_widget.update_data(np.array([]))
            finally:
                self._updating_settings = False
            return

        self._updating_settings = True
        try:
            self._restore_combobox_colormaps(settings)
            output_type = settings.get('output_type') or settings.get(
                'lifetime_type',
                'Apparent Phase Lifetime',
            )
            if output_type:
                if output_type in {
                    "Apparent Phase Lifetime",
                    "Apparent Modulation Lifetime",
                    "Normal Lifetime",
                }:
                    self.output_mode_combobox.setCurrentText("Lifetime")
                    self.lifetime_type_combobox.setCurrentText(output_type)
                else:
                    self.output_mode_combobox.setCurrentText(output_type)
            self._sync_mode_widgets()
            self._set_frequency_input_enabled(
                self._output_requires_frequency(
                    self._get_selected_output_type()
                )
            )

            mesh_enabled = bool(settings.get('mesh_overlay_enabled', False))
            mesh_alpha = float(settings.get('mesh_alpha', DEFAULT_MESH_ALPHA))
            mesh_alpha = float(np.clip(mesh_alpha, 0.0, 1.0))

            self.mesh_overlay_checkbox.blockSignals(True)
            try:
                self.mesh_overlay_checkbox.setChecked(mesh_enabled)
            finally:
                self.mesh_overlay_checkbox.blockSignals(False)

            self.mesh_transparency_spinbox.blockSignals(True)
            try:
                self.mesh_transparency_spinbox.setValue(1.0 - mesh_alpha)
            finally:
                self.mesh_transparency_spinbox.blockSignals(False)

            self._update_phase_slider_bounds_from_plot_mode()
            self._sync_mesh_ranges(settings)
            self._sync_filter_ui()
            self._sync_mode_widgets()

        finally:
            self._updating_settings = False

    def _on_frequency_changed(self):
        """Handle frequency input changes."""
        frequency_text = self.frequency_input.text().strip()
        output_type = self._get_selected_output_type()

        if self._updating_settings or not self._output_requires_frequency(
            output_type
        ):
            return
        frequency = self._parse_positive_frequency(frequency_text)
        if frequency is None:
            self.frequency = None
            self._clear_current_output_display()
            if frequency_text:
                show_error("Invalid frequency value. Enter a positive number.")
            return
        self.frequency = frequency
        self._calculate_and_display_output(show_warnings=False)

    def _set_frequency_input_enabled(self, enabled: bool):
        """Enable or disable the frequency input field."""
        self.frequency_input.setEnabled(enabled)

    def _on_range_changed_from_histogram(self, min_float, max_float):
        """Bridge between HistogramWidget.rangeChanged and the lifetime logic."""
        min_val = int(min_float * self.lifetime_range_factor)
        max_val = int(max_float * self.lifetime_range_factor)
        self._on_lifetime_range_changed((min_val, max_val))

    def _on_lifetime_range_changed(self, value):
        """Callback when lifetime range slider changes - updates all lifetime layers."""
        min_val, max_val = value
        min_lifetime = min_val / self.lifetime_range_factor
        max_lifetime = max_val / self.lifetime_range_factor
        output_type = self._get_selected_output_type()

        if not self._updating_settings:
            self._update_tab_sliders_from_range(min_lifetime, max_lifetime)

        if self.current_metric_data_original is not None:
            self.current_metric_data = np.clip(
                self.current_metric_data_original, min_lifetime, max_lifetime
            )
            self.lifetime_data = self.current_metric_data

        selected_layers = self.parent_widget.get_selected_layers()
        output_layers = self._mapping_output_layers(output_type)

        self._updating_contrast_limits = True
        self._updating_linked_layers = True
        try:
            for layer in selected_layers:
                derived_data = layer.metadata.get('derived_data', {})
                if output_type not in derived_data:
                    continue

                output_data_dict = derived_data[output_type]
                if self.parent_widget.harmonic not in output_data_dict:
                    continue

                output_values = output_data_dict[self.parent_widget.harmonic]
                clipped_lifetime = np.clip(
                    output_values, min_lifetime, max_lifetime
                )

                lifetime_layer = output_layers.get(layer.name)
                if lifetime_layer is not None:
                    # Filtered pixels are already NaN in the phasor arrays
                    # these values were computed from, and np.clip leaves NaN
                    # alone, so the display range cannot resurrect them.
                    lifetime_layer.data = clipped_lifetime
                    lifetime_layer.contrast_limits = [
                        min_lifetime,
                        max_lifetime,
                    ]

            self.colormap_contrast_limits = [min_lifetime, max_lifetime]
        finally:
            self._updating_contrast_limits = False
            self._updating_linked_layers = False

        if not self._updating_settings:
            self._store_output_range(min_lifetime, max_lifetime)

        self._apply_lifetime_range_change(min_val, max_val)

    def calculate_output_data(self):
        """Calculate selected output (lifetime/phase/modulation) for all selected layers."""
        if not self.parent_widget.has_phasor_data():
            return

        output_type = self._get_selected_output_type()
        self.current_output_type = output_type

        frequency_text = self.frequency_input.text().strip()
        if self._output_requires_frequency(output_type) and not frequency_text:
            show_warning("Enter frequency")
            return

        base_frequency = None
        if self._output_requires_frequency(output_type):
            base_frequency = float(frequency_text)
            self.frequency = base_frequency

        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return

        all_output_data = []
        per_layer_data = {}

        # Every layer's output map is independent, and the phasor conversions
        # release the GIL, so they are computed in a thread pool. Values that
        # come from the UI are read once here, on this thread, because the
        # workers must not touch Qt.
        harmonic = self.parent_widget.harmonic
        requires_frequency = self._output_requires_frequency(output_type)
        semicircle_mode = self._is_semicircle_mode()
        # Each layer is analysed at the frequency it was acquired with: the
        # primary at the entered one, the others at their stored one.
        layer_frequencies = {}
        if requires_frequency:
            for layer in selected_layers:
                layer_frequency = (
                    self.parent_widget.layer_frequency(layer, base_frequency)
                    if hasattr(self.parent_widget, 'layer_frequency')
                    else base_frequency
                )
                layer_frequencies[layer.name] = layer_frequency * harmonic

        def compute_output(layer):
            """Return one layer's output map, or ``None``.

            Pure array work (NumPy's error state is thread-local), so this is
            safe to run in a worker thread.
            """
            effective_frequency = layer_frequencies.get(layer.name)
            g_array = layer.metadata.get("G")
            s_array = layer.metadata.get("S")
            harmonics = layer.metadata.get("harmonics")

            if g_array is None or s_array is None:
                return None

            if harmonics is not None and g_array.ndim > layer.data.ndim:
                try:
                    harmonics_array = np.atleast_1d(harmonics)
                    harmonic_index = np.where(harmonics_array == harmonic)[0][
                        0
                    ]
                    real = g_array[harmonic_index]
                    imag = s_array[harmonic_index]
                except IndexError:
                    return None
            else:
                real = g_array
                imag = s_array

            if requires_frequency:
                with np.errstate(divide='ignore', invalid='ignore'):
                    if output_type == "Normal Lifetime":
                        output_values = phasor_to_normal_lifetime(
                            real, imag, frequency=effective_frequency
                        )
                    else:
                        phase_lifetime, modulation_lifetime = (
                            phasor_to_apparent_lifetime(
                                real, imag, frequency=effective_frequency
                            )
                        )
                        if output_type == "Apparent Phase Lifetime":
                            output_values = np.clip(
                                phase_lifetime, a_min=0, a_max=None
                            )
                        else:
                            output_values = np.clip(
                                modulation_lifetime, a_min=0, a_max=None
                            )
                with np.errstate(invalid='ignore'):
                    output_values[output_values < 0] = 0
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    phase_values, modulation_values = phasor_to_polar(
                        real, imag
                    )
                if output_type == "Phase" and not semicircle_mode:
                    # Full polar mode expects phase in [0, 2pi].
                    with np.errstate(invalid='ignore'):
                        phase_values = np.mod(phase_values, 2.0 * np.pi)
                output_values = (
                    phase_values
                    if output_type == "Phase"
                    else modulation_values
                )
            return output_values

        outputs = parallel_map(
            compute_output, selected_layers, on_error="collect"
        )

        for layer, output_values in zip(selected_layers, outputs, strict=True):
            if isinstance(output_values, BaseException):
                show_error(
                    f"{output_type} failed for {layer.name}: {output_values}"
                )
                continue
            if output_values is None:
                continue

            if 'derived_data' not in layer.metadata:
                layer.metadata['derived_data'] = {}
            if output_type not in layer.metadata['derived_data']:
                layer.metadata['derived_data'][output_type] = {}
            layer.metadata['derived_data'][output_type][
                harmonic
            ] = output_values

            all_output_data.append(output_values)
            per_layer_data[layer.name] = output_values

        if not all_output_data:
            return

        merged_output = np.concatenate(
            [data.flatten() for data in all_output_data]
        )
        self.current_metric_data_original = merged_output
        self.current_metric_data = self.current_metric_data_original.copy()
        self.per_layer_metric_data_original = {
            k: v.copy() for k, v in per_layer_data.items()
        }
        self.per_layer_metric_data = {
            k: v.copy() for k, v in per_layer_data.items()
        }

        if self._output_requires_frequency(output_type):
            self.lifetime_data_original = self.current_metric_data_original
            self.lifetime_data = self.current_metric_data
            self.per_layer_lifetime_data_original = (
                self.per_layer_metric_data_original
            )
            self.per_layer_lifetime_data = self.per_layer_metric_data

    def calculate_lifetimes(self):
        """Backward-compatible alias for unified output calculation."""
        self.calculate_output_data()
        self._update_lifetime_range_slider()

    def _update_lifetime_range_slider(self):
        """Update the lifetime range slider based on the calculated lifetime data."""
        if (
            self.current_metric_data_original is None
            and self.lifetime_data_original is not None
        ):
            self.current_metric_data_original = self.lifetime_data_original
            self.current_metric_data = self.lifetime_data
            self.per_layer_metric_data = self.per_layer_lifetime_data
            self.per_layer_metric_data_original = (
                self.per_layer_lifetime_data_original
            )

        if self.current_metric_data_original is None:
            return
        output_type = self._get_selected_output_type()
        if (
            self._output_requires_frequency(output_type)
            and self.frequency is None
        ):
            return

        effective_frequency = None
        if self._output_requires_frequency(output_type):
            effective_frequency = self.frequency * self.parent_widget.harmonic

        flattened_data = self.current_metric_data_original.flatten()
        valid_data = flattened_data[
            ~np.isnan(flattened_data) & np.isfinite(flattened_data)
        ]
        if self._output_requires_frequency(output_type):
            valid_data = valid_data[valid_data > 0]

        self._configure_histogram_labels_for_output(output_type)

        if len(valid_data) == 0:
            self.min_lifetime = 0.0
            self.max_lifetime = (
                10.0 if self._output_requires_frequency(output_type) else 1.0
            )
            min_slider_val = 0
            max_slider_val = int(
                self.max_lifetime * self.lifetime_range_factor
            )
        else:
            if output_type == "Phase" and not self._is_semicircle_mode():
                # In full polar mode, initialize the display range to 0..2pi
                # so users get the expected 0..360 degree domain.
                self.min_lifetime = 0.0
                self.max_lifetime = 2.0 * np.pi
                min_slider_val = 0
                max_slider_val = int(
                    self.max_lifetime * self.lifetime_range_factor
                )
                self.current_metric_data_original = np.mod(
                    self.current_metric_data_original, 2.0 * np.pi
                )
                self.current_metric_data = (
                    self.current_metric_data_original.copy()
                )
                for name, data in self.per_layer_metric_data_original.items():
                    wrapped = np.mod(data, 2.0 * np.pi)
                    self.per_layer_metric_data_original[name] = wrapped
                    self.per_layer_metric_data[name] = wrapped.copy()
            else:
                self.min_lifetime = np.min(valid_data)
                self.max_lifetime = np.max(valid_data)

                if (
                    self._output_requires_frequency(output_type)
                    and effective_frequency is not None
                    and (
                        not np.isfinite(self.min_lifetime)
                        or not np.isfinite(self.max_lifetime)
                        or self.max_lifetime > (2e3 / effective_frequency)
                        or self.min_lifetime < 0
                    )
                ):
                    self.min_lifetime = 0.0
                    self.max_lifetime = (
                        2e3 / effective_frequency
                    )  # 2 periods in ns
                    min_slider_val = 0
                    max_slider_val = int(
                        self.max_lifetime * self.lifetime_range_factor
                    )
                else:
                    if self.min_lifetime >= self.max_lifetime:
                        self.max_lifetime = self.min_lifetime + 1.0
                    min_slider_val = int(
                        self.min_lifetime * self.lifetime_range_factor
                    )
                    max_slider_val = int(
                        self.max_lifetime * self.lifetime_range_factor
                    )
        self.lifetime_range_slider.setRange(0, max_slider_val)
        self.lifetime_range_slider.setValue((min_slider_val, max_slider_val))

        self.lifetime_range_label.setText(
            f"{self.histogram_widget._range_label_prefix}:"
        )

        self.lifetime_min_edit.setText(f"{self.min_lifetime:.2f}")
        self.lifetime_max_edit.setText(f"{self.max_lifetime:.2f}")

    def plot_lifetime_histogram(self):
        """Plot the histogram of the merged lifetime data from all selected layers."""
        selected_names = self._get_selected_source_names()
        if not selected_names or self.parent_widget.harmonic is None:
            self.histogram_widget.clear()
            return

        output_type = self._get_selected_output_type()
        output_layers = self._mapping_output_layers(
            output_type, selected_only=True
        )
        if output_layers:
            layers = list(output_layers.values())
            if layers != self.metric_layers:
                self._set_metric_layers(layers)
            named = {layer.name: layer.data for layer in layers}
            # Groups live on the analysed image layer, not on the derived
            # output layer, so every tab sees the same grouping.
            sources = {
                layer.name: source for source, layer in output_layers.items()
            }
        elif not self._has_calculated_output:
            # Keep a narrow fallback for calculations that have populated
            # per-layer arrays but have not created the viewer layers yet.
            named = {
                f"{output_type}: {name}": data
                for name, data in (self.per_layer_metric_data or {}).items()
                if name in selected_names
            }
            sources = {
                f"{output_type}: {name}": name
                for name in (self.per_layer_metric_data or {})
                if name in selected_names
            }
        else:
            named = {}
            sources = {}

        self.histogram_widget.set_dataset_sources(sources)

        if not named:
            self.histogram_widget.clear()
            return

        self.histogram_widget.update_colormap(
            colormap_colors=self.lifetime_colormap,
            contrast_limits=self.colormap_contrast_limits,
            gamma=self.colormap_gamma,
        )

        # In per-frame mode only the displayed timepoint is summarised. The
        # range slider keeps using the pooled extent (set elsewhere) so the
        # contrast limits don't jump around while playing.
        named = self._slice_datasets_for_frame(named)
        if len(named) > 1:
            self.histogram_widget.update_multi_data(named)
        elif named:
            label, data = next(iter(named.items()))
            self.histogram_widget.update_data(data, label=label)

        if output_type in {"Phase", "Modulation"}:
            self._apply_histogram_coloring(output_type)

    def _frame_context(self):
        """Return the plotter's time-lapse frame context, if available."""
        return getattr(self.parent_widget, 'frame_context', None)

    def _slice_datasets_for_frame(self, datasets):
        """Restrict per-layer datasets to the displayed time-lapse frame.

        The un-sliced arrays are handed to the histogram widget as well, so
        the statistics dock can export per-timepoint numbers.
        """
        frame_context = self._frame_context()
        self.histogram_widget.set_frame_source(frame_context, datasets)
        return slice_datasets(frame_context, datasets)

    def refresh_for_frame_change(self):
        """Re-feed the histogram after the displayed frame changed."""
        if self.current_metric_data is None:
            return
        self.plot_lifetime_histogram()

    def rename_layer(self, old_name: str, new_name: str):
        """Update internal dictionaries and rename derived layers when a layer is renamed."""
        for dict_attr in [
            'per_layer_metric_data_original',
            'per_layer_metric_data',
            'per_layer_lifetime_data_original',
            'per_layer_lifetime_data',
        ]:
            if hasattr(self, dict_attr):
                dict_obj = getattr(self, dict_attr)
                if dict_obj is not None and old_name in dict_obj:
                    dict_obj[new_name] = dict_obj.pop(old_name)

        for output_layer in list(self.viewer.layers):
            tag = output_layer.metadata.get(_MAPPING_OUTPUT_METADATA_KEY)
            if (
                isinstance(tag, dict)
                and tag.get('source_layer') == old_name
                and tag.get('output_type') in _MAPPING_OUTPUT_TYPES
            ):
                output_type = tag['output_type']
            else:
                output_type = next(
                    (
                        candidate
                        for candidate in _MAPPING_OUTPUT_TYPES
                        if output_layer.name == f"{candidate}: {old_name}"
                    ),
                    None,
                )
            if output_type is None:
                continue

            old_output_name = output_layer.name
            output_layer.metadata[_MAPPING_OUTPUT_METADATA_KEY] = {
                'source_layer': new_name,
                'output_type': output_type,
            }
            if old_output_name == f"{output_type}: {old_name}":
                output_layer.name = f"{output_type}: {new_name}"
            if output_layer.name != old_output_name:
                self.histogram_widget.rename_dataset(
                    old_output_name, output_layer.name
                )

    def create_output_layers(self):
        """Create or update output layers for all selected layers."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return

        output_type = self._get_selected_output_type()
        if output_type in {"Phase", "Modulation"}:
            cmap_name = self._resolve_layer_colormap(
                self.colormap_combobox.currentText()
            )
        else:
            cmap_name = self._get_output_colormap_name(output_type)
        use_combobox_pick = output_type in self._pending_combobox_colormaps
        self._pending_combobox_colormaps.discard(output_type)

        self._set_metric_layers([])
        created_layers = []
        existing_outputs = self._mapping_output_layers(output_type)

        for layer in selected_layers:
            derived_data = layer.metadata.get('derived_data', {})
            if output_type not in derived_data:
                continue

            output_data_dict = derived_data[output_type]
            if self.parent_widget.harmonic not in output_data_dict:
                continue

            output_values = output_data_dict[self.parent_widget.harmonic]

            output_layer_name = f"{output_type}: {layer.name}"

            min_val, max_val = self.lifetime_range_slider.value()
            min_lifetime = min_val / self.lifetime_range_factor
            max_lifetime = max_val / self.lifetime_range_factor
            cl_max = (
                max_lifetime
                if max_lifetime > min_lifetime
                else min_lifetime + 1.0
            )
            clipped_output = np.clip(output_values, min_lifetime, cl_max)

            output_layer = existing_outputs.get(layer.name)
            colormap, gamma = self._output_layer_colormap(
                layer,
                output_type,
                output_layer,
                cmap_name,
                use_default=use_combobox_pick,
            )
            output_metadata = {
                'source_layer': layer.name,
                'output_type': output_type,
            }
            if output_layer is None:
                selected_output_layer = Image(
                    clipped_output,
                    name=output_layer_name,
                    scale=layer.scale,
                    colormap=colormap,
                    contrast_limits=[min_lifetime, cl_max],
                    metadata={_MAPPING_OUTPUT_METADATA_KEY: output_metadata},
                )
                output_layer = self.viewer.add_layer(selected_output_layer)
            else:
                output_layer.data = clipped_output
                output_layer.scale = layer.scale
                output_layer.colormap = colormap
                output_layer.contrast_limits = [
                    min_lifetime,
                    cl_max,
                ]
                output_layer.metadata[_MAPPING_OUTPUT_METADATA_KEY] = (
                    output_metadata
                )
            if gamma is not None:
                output_layer.gamma = gamma
            self._store_output_colormap(layer, output_type, output_layer)
            created_layers.append(output_layer)

        self._set_metric_layers(created_layers)

    def _output_layer_colormap(
        self, source_layer, output_type, output_layer, default, use_default
    ):
        """Return ``(colormap, gamma)`` for *source_layer*'s output layer.

        The colormap stored in the source layer's settings wins, so a
        colormap the user set on the layer survives running the analysis
        again, and one copied over with the settings is applied. Without a
        stored one, an existing output layer keeps its own; only a new layer
        gets *default*. With *use_default* (the user has just picked a
        colormap in the tab) *default* is used regardless, keeping the gamma.
        """
        stored = self._stored_output_colormap(source_layer, output_type)
        colormap = None
        if not use_default:
            colormap = layer_colormap_from_settings(stored)
            if colormap is None and output_layer is not None:
                colormap = output_layer.colormap
        if colormap is None:
            colormap = default

        gamma = stored.get('gamma') if stored else None
        if gamma is None and output_layer is not None:
            gamma = output_layer.gamma
        return colormap, gamma

    def _stored_output_colormap(self, source_layer, output_type):
        """Return the colormap entry stored for *output_type*, or None."""
        settings = self._get_phasor_mapping_settings(source_layer)
        if not settings:
            return None
        entry = (settings.get('output_colormaps') or {}).get(output_type)
        return entry if isinstance(entry, dict) else None

    def _store_output_colormap(self, source_layer, output_type, output_layer):
        """Save *output_layer*'s colormap into *source_layer*'s settings.

        The colormap belongs to an output that already exists, so it is
        stored right away rather than kept as an unsaved edit.
        """
        if self._updating_settings or self.parent_widget is None:
            return
        self.parent_widget.settings_store.update_committed(
            [source_layer],
            'phasor_mapping',
            ('output_colormaps', output_type),
            layer_colormap_to_settings(
                output_layer.colormap, output_layer.gamma
            ),
        )
        self._sync_legacy_alias([source_layer])

    def create_lifetime_layer(self):
        """Backward-compatible alias for output layer creation."""
        self.create_output_layers()

    def _on_colormap_changed(self, event):
        """Callback whenever the colormap or contrast limits change on any lifetime layer - sync all layers."""
        if getattr(self, '_updating_contrast_limits', False) or getattr(
            self, '_updating_linked_layers', False
        ):
            return

        source_layer = event.source
        new_colormap = source_layer.colormap
        new_contrast_limits = source_layer.contrast_limits
        new_gamma = source_layer.gamma

        # Update stored values
        self.lifetime_colormap = new_colormap.colors
        self.colormap_contrast_limits = new_contrast_limits
        self.colormap_gamma = new_gamma

        output_type = self._get_selected_output_type()
        if output_type in {"Phase", "Modulation"}:
            cmap_name = new_colormap.name
            self.colormap_combobox.blockSignals(True)
            try:
                idx = self.colormap_combobox.findText(cmap_name)
                if idx >= 0:
                    self.colormap_combobox.setCurrentIndex(idx)
            finally:
                self.colormap_combobox.blockSignals(False)
            if output_type == "Phase":
                self._phase_colormap_name = cmap_name
            elif output_type == "Modulation":
                self._modulation_colormap_name = cmap_name
        # The layer now says which colormap is wanted; an older combobox pick
        # must not override it on the next run.
        self._pending_combobox_colormaps.discard(output_type)

        # Update all other lifetime layers to match
        self._updating_linked_layers = True
        try:
            for layer in self.metric_layers:
                if layer != source_layer and layer in self.viewer.layers:
                    layer.colormap = new_colormap
                    layer.contrast_limits = new_contrast_limits
                    layer.gamma = new_gamma
        finally:
            self._updating_linked_layers = False

        # Remember the colormap with each analysed layer's settings, so a
        # new run — and a copy of the settings — keeps it.
        for layer in self.metric_layers:
            info = self._mapping_output_info(layer)
            if info is None or info[1] not in self.viewer.layers:
                continue
            self._store_output_colormap(
                self.viewer.layers[info[1]], info[0], source_layer
            )

        self.histogram_widget.update_colormap(
            colormap_colors=self.lifetime_colormap,
            contrast_limits=self.colormap_contrast_limits,
            gamma=self.colormap_gamma,
        )

        if output_type in {"Phase", "Modulation"} and (
            self.apply_2d_colormap_checkbox.isChecked()
            or self.mesh_overlay_checkbox.isChecked()
        ):
            self._apply_histogram_coloring(output_type)
        elif (
            output_type in LIFETIME_OUTPUT_TYPES
            and self.mesh_overlay_checkbox.isChecked()
        ):
            # Contrast limits change continuously while dragged in napari.
            self._mesh_axes_update_timer.start()

    def _mapping_validation(self):
        """Return ``None`` if the output can be computed, else the missing msg."""
        if not self.parent_widget.get_selected_layers():
            return "Select at least one image layer with phasor features."
        output_type = self._get_selected_output_type()
        if self._output_requires_frequency(output_type):
            frequency_text = self.frequency_input.text().strip()
            if not frequency_text:
                return "Enter the frequency (MHz)."
            if self._parse_positive_frequency(frequency_text) is None:
                return "Enter a valid positive frequency (MHz)."
        return None

    def _on_image_layer_changed(self):
        """Callback whenever the image layer with phasor features changes.

        This only restores UI state from metadata - it does NOT run calculations.
        User must click "Calculate" button to run lifetime analysis.
        """
        self._teardown_on_layer_change()
        self._restore_on_layer_change()
        if hasattr(self, '_refresh_calculate_button'):
            self._refresh_calculate_button()

    def _teardown_on_layer_change(self):
        """Immediate cleanup: disconnect signals and clear data."""
        self._output_refresh_timer.stop()
        layer_name = self.parent_widget.get_primary_layer_name()
        if not layer_name:
            self.histogram_widget.update_data(np.array([]))
            # Disconnect events from all lifetime layers
            for layer in self.metric_layers:
                if layer in self.viewer.layers:
                    with contextlib.suppress(Exception):
                        layer.events.colormap.disconnect(
                            self._on_colormap_changed
                        )
                        layer.events.contrast_limits.disconnect(
                            self._on_colormap_changed
                        )
                        layer.events.gamma.disconnect(
                            self._on_colormap_changed
                        )

            self.lifetime_data = None
            self.lifetime_data_original = None
            self.current_metric_data = None
            self.current_metric_data_original = None
            self.per_layer_lifetime_data = {}
            self.per_layer_lifetime_data_original = {}
            self.per_layer_metric_data = {}
            self.per_layer_metric_data_original = {}
            self.lifetime_layer = None
            self.lifetime_layers = []
            self.metric_layers = []

            self._clear_2d_coloring()
            self.histogram_widget.clear()

    def _restore_on_layer_change(self):
        """Deferred restore: update UI state from metadata."""
        if getattr(self, '_applying_mapping_filter', False):
            return
        self._output_refresh_timer.stop()
        self._needs_update = False

        layer_name = self.parent_widget.get_primary_layer_name()
        if layer_name:
            self._restore_lifetime_settings_from_metadata()
            self._sync_mode_widgets()
            self._sync_filter_ui()
            self._set_frequency_input_enabled(
                self._output_requires_frequency(
                    self._get_selected_output_type()
                )
            )
            self._clear_2d_coloring()

        # Refresh the primary button's ready/blocked style. This runs on both
        # the direct layer-change path and the deferred tab-switch path, so the
        # button reflects the current validation state (e.g. a selected layer
        # with the frequency already filled from metadata) instead of keeping a
        # stale blocked (grey) style.
        self._refresh_action_buttons()
        self._refresh_settings_note()

    def _on_mapping_input_changed(self):
        """Re-evaluate the Calculate button after an input changed."""
        self._refresh_action_buttons()
        self.request_autoupdate()

    def _autoupdate_calculate_output(self):
        """Recompute the output for an autoupdate, without popping warnings.

        An automatic run is not a user action, so a transiently invalid state
        (a layer whose data a filter is still rewriting, say) must not raise
        a dialog the user did not ask for.

        Output-mode changes also arm the debounced refresh timer (which only
        runs once an output exists); computing here makes that pending tick
        redundant, so it is dropped rather than repeating the work.
        """
        if getattr(self, '_applying_mapping_filter', False):
            return
        self._output_refresh_timer.stop()
        if self._calculate_and_display_output(show_warnings=False):
            self._has_calculated_output = True

    def _on_calculate_lifetime_clicked(self):
        """Callback when Calculate button is clicked.

        Runs the lifetime calculation and creates/updates lifetime layers.
        """
        if self._calculate_and_display_output(show_warnings=True):
            self._has_calculated_output = True

    def _clear_current_output_display(self):
        """Clear stale data for the currently selected Mapping output."""
        self.current_metric_data = None
        self.current_metric_data_original = None
        self.per_layer_metric_data = {}
        self.per_layer_metric_data_original = {}
        self._set_metric_layers([])
        self.histogram_widget.clear()
        self._clear_2d_coloring()

    def _calculate_and_display_output(self, *, show_warnings):
        """Calculate and render the selected Mapping output.

        Returns
        -------
        bool
            True when output data and layers were created.
        """
        validation_message = self._mapping_validation()
        if validation_message is not None:
            if show_warnings:
                warning = (
                    "Enter frequency"
                    if validation_message == "Enter the frequency (MHz)."
                    else validation_message
                )
                show_warning(warning)
            self._clear_current_output_display()
            return False

        selected_layers = self.parent_widget.get_selected_layers()
        output_type = self._get_selected_output_type()
        frequency = self.frequency_input.text().strip()

        self.current_metric_data = None
        self.current_metric_data_original = None
        self.per_layer_metric_data = {}
        self.per_layer_metric_data_original = {}
        self.calculate_output_data()
        if self.current_metric_data is None:
            self._clear_current_output_display()
            return False

        if not self._updating_settings:
            # What was just computed becomes the analysed layers' settings,
            # before the output layers are built from them.
            analysed_layers = [
                layer
                for layer in selected_layers
                if layer.name in self.per_layer_metric_data
            ]
            self._commit_mapping_settings(analysed_layers)
            if self._output_requires_frequency(output_type):
                self.parent_widget.commit_frequency(analysed_layers, frequency)

        self._update_lifetime_range_slider()
        self.create_output_layers()
        settings = self._get_current_layer_mapping_settings(create=False)
        self._sync_mesh_ranges(settings)

        self._sync_filter_ui()

        self._restore_lifetime_range_from_metadata()
        self._on_lifetime_range_changed(self.lifetime_range_slider.value())

        if self.current_metric_data is not None:
            self.plot_lifetime_histogram()

        if output_type not in {"Phase", "Modulation"}:
            self._clear_2d_coloring()
            self._refresh_mesh_overlay_if_needed()

        return bool(self.metric_layers)

    def _schedule_active_output_refresh(self):
        """Coalesce Mapping control changes after the first calculation."""
        if (
            self._updating_settings
            or not self._has_calculated_output
            or getattr(self.parent_widget, '_is_closing', False)
        ):
            return
        self._output_refresh_timer.start()

    def _refresh_active_output(self):
        """Recalculate Mapping output after a reactive control change."""
        if (
            self._updating_settings
            or not self._has_calculated_output
            or getattr(self.parent_widget, '_is_closing', False)
        ):
            return
        self._calculate_and_display_output(show_warnings=False)

    def _on_lifetime_type_changed(self, text):
        """Callback when lifetime type combobox selection changes.

        This only updates the setting in metadata - it does NOT run
        calculations. Picking another lifetime asks for a different analysis,
        not a different rendering of the one already on screen, so the user
        clicks Calculate (or turns Autoupdate on) to run it. A refresh armed
        by an earlier control change is dropped for the same reason: it would
        compute the newly picked lifetime nobody asked for yet.
        """
        output_type = self._get_selected_output_type()
        self.current_output_type = output_type
        self._update_calculate_button_text()
        self._set_frequency_input_enabled(
            self._output_requires_frequency(output_type)
        )
        self._configure_histogram_labels_for_output(output_type)
        if output_type in MAPPING_METRICS:
            self.filter_list.set_current_metric(output_type)
        self._sync_filter_ui()
        self.outputTypeChanged.emit(output_type)
        if not self._updating_settings:
            self._update_lifetime_setting_in_metadata('lifetime_type', text)
            self._update_lifetime_setting_in_metadata(
                'output_type', output_type
            )
            # The mesh is only a guide drawn from the frequency, so unlike
            # the output map it follows the new lifetime type right away.
            if (
                output_type in LIFETIME_OUTPUT_TYPES
                and self.mesh_overlay_checkbox.isChecked()
            ):
                self._sync_lifetime_mesh_range()
                self._apply_lifetime_mesh(output_type)
        self._output_refresh_timer.stop()

    def _store_output_range(self, min_value, max_value):
        """Remember the displayed range of the current output type.

        ``range_min``/``range_max`` are a single slot shared by every output,
        so on their own they let one output's range clip another's map. The
        per-output copy kept here is what the restore reads back.

        The range applies to the output layers of every analysed layer, so
        it is stored in all of them right away.
        """
        layers = self._analysed_layers()
        if not layers or self.parent_widget is None:
            return
        store = self.parent_widget.settings_store
        min_value, max_value = float(min_value), float(max_value)
        store.update_committed(
            layers,
            'phasor_mapping',
            ('output_ranges', self._get_selected_output_type()),
            [min_value, max_value],
        )
        for key, value in (
            ('range_min', min_value),
            ('range_max', max_value),
            ('lifetime_range_min', min_value),
            ('lifetime_range_max', max_value),
        ):
            store.update_committed(layers, 'phasor_mapping', (key,), value)
        self._sync_legacy_alias(layers)

    def _saved_range_for_current_output(self, settings):
        """Return the stored ``(min, max)`` for the current output, or None.

        Falls back to the shared ``range_min``/``range_max`` slot only when no
        per-output range has ever been stored for this layer, which is how
        settings written before this key existed (or imported from a settings
        file) are still honoured.
        """
        output_type = self._get_selected_output_type()
        ranges = settings.get('output_ranges')
        if isinstance(ranges, dict) and ranges:
            stored = ranges.get(output_type)
            if (
                isinstance(stored, (list, tuple))
                and len(stored) == 2
                and stored[0] is not None
                and stored[1] is not None
            ):
                return float(stored[0]), float(stored[1])
            return None

        min_val = settings.get('range_min', settings.get('lifetime_range_min'))
        max_val = settings.get('range_max', settings.get('lifetime_range_max'))
        if min_val is None or max_val is None:
            return None
        return float(min_val), float(max_val)

    def _restore_lifetime_range_from_metadata(self):
        """Restore lifetime range from metadata after calculation."""
        layer_name = self.parent_widget.get_primary_layer_name()
        if not layer_name or layer_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[layer_name]
        settings = self._get_phasor_mapping_settings(layer, create=False)
        if settings is None:
            return

        saved = self._saved_range_for_current_output(settings)
        if saved is None:
            return
        min_val, max_val = saved

        if (
            self.min_lifetime is not None
            and self.max_lifetime is not None
            and min_val >= self.min_lifetime
            and max_val <= self.max_lifetime
        ):

            min_slider = int(min_val * self.lifetime_range_factor)
            max_slider = int(max_val * self.lifetime_range_factor)

            self._updating_settings = True
            try:
                self.lifetime_range_slider.setValue((min_slider, max_slider))
                self.lifetime_min_edit.setText(f"{min_val:.2f}")
                self.lifetime_max_edit.setText(f"{max_val:.2f}")
                self.lifetime_range_label.setText(
                    f"{self.histogram_widget._range_label_prefix}:"
                )
            finally:
                self._updating_settings = False

            self._apply_lifetime_range_change(min_slider, max_slider)

    def _apply_lifetime_range_change(self, min_slider, max_slider):
        """Apply lifetime range change for histogram without updating layers (layers updated in _on_lifetime_range_changed)."""
        min_lifetime = min_slider / self.lifetime_range_factor
        max_lifetime = max_slider / self.lifetime_range_factor

        # Only update the histogram data (merged/flattened), not the individual layers
        if self.current_metric_data_original is not None:
            self.current_metric_data = np.clip(
                self.current_metric_data_original,
                min_lifetime,
                max_lifetime,
            )
            self.lifetime_data = self.current_metric_data
            # Also clip per-layer data for multi-layer histogram modes
            for name, orig in self.per_layer_metric_data_original.items():
                self.per_layer_metric_data[name] = np.clip(
                    orig, min_lifetime, max_lifetime
                )
            self.per_layer_lifetime_data = self.per_layer_metric_data
            self.plot_lifetime_histogram()

        # The lifetime mesh is coloured with the output layers' contrast
        # limits, which this range just changed; coalesce slider drags.
        if (
            self._get_selected_output_type() in LIFETIME_OUTPUT_TYPES
            and self.mesh_overlay_checkbox.isChecked()
        ):
            self._mesh_axes_update_timer.start()

    @staticmethod
    def _set_histogram_density_visible(pw, visible: bool):
        """Show or hide the biaplotter histogram density image."""
        hist_artist = pw.canvas_widget.artists.get("HISTOGRAM2D")
        if hist_artist is None:
            return

        if (
            visible
            and getattr(pw, 'plot_type', 'HISTOGRAM2D') != 'HISTOGRAM2D'
        ):
            return

        img = hist_artist._mpl_artists.get("histogram_image")
        if img is not None:
            img.set_visible(visible)

    def _remove_overlay(self):
        """Remove both the plot and mesh overlays from the phasor plot."""
        self._remove_plot_overlay()
        self._remove_mesh_overlay()

    def _remove_plot_overlay(self):
        """Remove the colormapped plot overlay and its clipping patch."""
        if getattr(self, '_overlay_imshow', None) is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self._overlay_imshow.remove()
            self._overlay_imshow = None
        if getattr(self, '_overlay_clip_patch', None) is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self._overlay_clip_patch.remove()
            self._overlay_clip_patch = None

    def _remove_mesh_overlay(self):
        """Remove the mesh overlay image from the phasor plot."""
        if getattr(self, '_mesh_overlay_imshow', None) is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self._mesh_overlay_imshow.remove()
            self._mesh_overlay_imshow = None

    def _get_mesh_grid_resolution(self, ax) -> int:
        """Return the mesh sampling resolution suited to *ax*'s on-screen size."""
        # Sample the mesh grid well above the on-screen pixel size so the mesh
        # outline (the curved semicircle / range boundaries) stays finely
        # detailed - especially when zoomed out, where a coarse grid spreads
        # few samples over a wide data extent and the edge looks rough. Grids
        # are cached per view, so a higher resolution only costs on a new
        # zoom/pan level (~25 ms at the cap, within the redraw debounce).
        with contextlib.suppress(Exception):
            bbox = ax.get_window_extent()
            target = int(max(float(bbox.width), float(bbox.height)) * 2.0)
            return int(np.clip(target, 640, 1280))
        return 1000

    def _make_mesh_grid_cache_key(self, ax, resolution: int):
        """Return the cache key identifying a mesh grid for the current view."""
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        return (
            round(float(x_min), 5),
            round(float(x_max), 5),
            round(float(y_min), 5),
            round(float(y_max), 5),
            bool(self._is_semicircle_mode()),
            int(resolution),
        )

    def _get_mesh_polar_grid(self, ax, resolution: int):
        """Return the phase/modulation grid covering *ax*'s current view.

        Grids are cached per view and evicted least-recently-used first, so
        panning back to a previous zoom level avoids recomputing them.
        """
        key = self._make_mesh_grid_cache_key(ax, resolution)
        cached = self._mesh_grid_cache.get(key)
        if cached is not None:
            with contextlib.suppress(ValueError):
                self._mesh_grid_cache_order.remove(key)
            self._mesh_grid_cache_order.append(key)
            return cached

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_fine = np.linspace(x_min, x_max, resolution)
        y_fine = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_fine, y_fine)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p_grid, m_grid = phasor_to_polar(X, Y)

        if not self._is_semicircle_mode():
            with np.errstate(invalid='ignore'):
                p_grid = np.mod(p_grid, 2.0 * np.pi)

        grid = {
            'p_grid': p_grid,
            'm_grid': m_grid,
            'extent': [x_min, x_max, y_min, y_max],
        }
        self._mesh_grid_cache[key] = grid
        self._mesh_grid_cache_order.append(key)

        while (
            len(self._mesh_grid_cache_order)
            > self._mesh_grid_cache_max_entries
        ):
            stale_key = self._mesh_grid_cache_order.pop(0)
            self._mesh_grid_cache.pop(stale_key, None)

        return grid

    def _get_mesh_alpha_map(
        self, mesh_mask, alpha_key, mesh_alpha: float, resolution: int, ax
    ):
        """Return the mesh's blurred alpha map, scaled by *mesh_alpha*.

        The blurred base is cached under *alpha_key* so changing only the
        opacity re-uses it instead of re-running the Gaussian filter.
        """
        cached = self._mesh_alpha_cache.get(alpha_key)
        if cached is not None:
            with contextlib.suppress(ValueError):
                self._mesh_alpha_cache_order.remove(alpha_key)
            self._mesh_alpha_cache_order.append(alpha_key)
            return cached * mesh_alpha

        # See ``_resolve_mesh_blur_sigma`` for why sigma must be resolved
        # from the actual display size rather than left as a flat constant.
        sigma = _resolve_mesh_blur_sigma(ax, resolution)
        alpha_base = gaussian_filter(
            (~mesh_mask).astype(float), sigma=sigma, mode="nearest"
        )
        alpha_base = np.clip(alpha_base, 0.0, 1.0)
        self._mesh_alpha_cache[alpha_key] = alpha_base
        self._mesh_alpha_cache_order.append(alpha_key)

        while (
            len(self._mesh_alpha_cache_order)
            > self._mesh_alpha_cache_max_entries
        ):
            stale_key = self._mesh_alpha_cache_order.pop(0)
            self._mesh_alpha_cache.pop(stale_key, None)

        return alpha_base * mesh_alpha

    def _get_clim_from_metric_layers(self):
        """Return the first metric layer's contrast limits, or (None, None)."""
        for layer in self.metric_layers:
            if layer in self.viewer.layers:
                with contextlib.suppress(
                    AttributeError, TypeError, ValueError
                ):
                    vmin, vmax = layer.contrast_limits
                    return float(vmin), float(vmax)
        return None, None

    def _apply_histogram_coloring(self, output_type: str):
        """Colour the phasor plot by *output_type* and draw the mesh overlay.

        Only "Phase" and "Modulation" are colourable; any other output type
        returns without touching the plot.
        """
        if output_type not in {"Phase", "Modulation"}:
            return
        pw = self.parent_widget
        if pw is None or getattr(pw, 'plot_type', 'HISTOGRAM2D') == 'NONE':
            return

        apply_plot_coloring = self.apply_2d_colormap_checkbox.isChecked()
        show_mesh = self.mesh_overlay_checkbox.isChecked()

        if not show_mesh:
            self._remove_mesh_overlay()
        if not apply_plot_coloring:
            self._remove_plot_overlay()

        features = pw.get_merged_features()
        if features is None:
            canvas_widget = getattr(pw, "canvas_widget", None)
            figure = getattr(canvas_widget, "figure", None)
            canvas = getattr(figure, "canvas", None)
            if canvas is not None and hasattr(canvas, "draw_idle"):
                canvas.draw_idle()
            return
        g_flat, s_flat = features

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            phase, modulation = phasor_to_polar(g_flat, s_flat)

        if not self._is_semicircle_mode():
            with np.errstate(invalid='ignore'):
                phase = np.mod(phase, 2.0 * np.pi)

        values = phase if output_type == "Phase" else modulation

        cmap_name = self.colormap_combobox.currentText()
        if cmap_name == "Select color..." and hasattr(self, "_custom_color"):
            cmap = create_mpl_colormap_from_qcolor(self._custom_color)
        else:
            cmap = resolve_colormap_by_name(cmap_name)
            if cmap is None:
                cmap = resolve_colormap_by_name("viridis")
        vmin, vmax = self._get_clim_from_metric_layers()
        if vmin is None or vmax is None:
            if output_type == "Phase":
                vmin, vmax = self.phase_range_slider.value()
                vmin /= self.phase_range_factor
                vmax /= self.phase_range_factor
            elif output_type == "Modulation":
                vmin, vmax = self.modulation_range_slider.value()
                vmin /= self.modulation_range_factor
                vmax /= self.modulation_range_factor
            else:
                with np.errstate(invalid='ignore'):
                    vmin = float(np.nanmin(values))
                    vmax = float(np.nanmax(values))

        if show_mesh:
            ax = pw.canvas_widget.axes
            resolution = self._get_mesh_grid_resolution(ax)
            mesh_grid = self._get_mesh_polar_grid(ax, resolution)
            p_grid = mesh_grid['p_grid']
            m_grid = mesh_grid['m_grid']

            phase_min_i, phase_max_i = self.phase_range_slider.value()
            mod_min_i, mod_max_i = self.modulation_range_slider.value()
            phase_min = phase_min_i / self.phase_range_factor
            phase_max = phase_max_i / self.phase_range_factor
            mod_min = mod_min_i / self.modulation_range_factor
            mod_max = mod_max_i / self.modulation_range_factor

            clip_semicircle = (
                self.mesh_clip_semicircle_checkbox.isChecked()
                and self._is_semicircle_mode()
            )
            mesh_mask = compute_phasor_mesh_mask(
                p_grid,
                m_grid,
                semicircle=self._is_semicircle_mode(),
                phase_range=(phase_min, phase_max),
                modulation_range=(mod_min, mod_max),
                clip_semicircle=clip_semicircle,
            )

            extent = mesh_grid['extent']
            mesh_alpha = self._mesh_alpha()
            alpha_key = (
                *self._make_mesh_grid_cache_key(ax, resolution),
                int(phase_min_i),
                int(phase_max_i),
                int(mod_min_i),
                int(mod_max_i),
                bool(self.mesh_clip_semicircle_checkbox.isChecked()),
            )
            mesh_alpha_map = self._get_mesh_alpha_map(
                mesh_mask,
                alpha_key,
                mesh_alpha,
                resolution,
                ax,
            )

            self._remove_mesh_overlay()

            self._mesh_overlay_imshow = draw_phasor_mesh(
                ax,
                output_type,
                semicircle=self._is_semicircle_mode(),
                colormap=cmap,
                alpha_map=mesh_alpha_map,
                vmin=vmin,
                vmax=vmax,
                p_grid=p_grid,
                m_grid=m_grid,
                mask=mesh_mask,
                extent=extent,
            )
            pw.canvas_widget.figure.canvas.draw_idle()

            if not apply_plot_coloring:
                self._restore_plot_coloring_state_without_mesh()
                if not self.mesh_colorbar_checkbox.isChecked():
                    pw._remove_mapping_colorbar()
                else:
                    self._update_mapping_colorbar(
                        cmap, vmin, vmax, output_type
                    )
                return

        if not apply_plot_coloring:
            self._restore_plot_coloring_state_without_mesh()
            if not self.mesh_colorbar_checkbox.isChecked():
                pw._remove_mapping_colorbar()
            else:
                self._update_mapping_colorbar(cmap, vmin, vmax, output_type)
            return

        # If reaching here, apply_plot_coloring is True
        if self.mesh_colorbar_checkbox.isChecked():
            self._update_mapping_colorbar(cmap, vmin, vmax, output_type)
        else:
            pw._remove_mapping_colorbar()

        if pw.plot_type == 'SCATTER':
            self._remove_plot_overlay()
            scatter_artist = pw.canvas_widget.artists.get("SCATTER")
            if scatter_artist is not None:
                sc = scatter_artist._mpl_artists.get("scatter")
                if sc is not None:
                    sc.set_array(values)
                    sc.set_cmap(cmap)
                    sc.set_clim(vmin, vmax)
                    pw.canvas_widget.figure.canvas.draw_idle()
            return

        if pw.plot_type == 'CONTOUR':
            ax = pw.canvas_widget.axes
            range_xlim = ax.get_xlim()
            range_ylim = ax.get_ylim()

            x_fine = np.linspace(range_xlim[0], range_xlim[1], 500)
            y_fine = np.linspace(range_ylim[0], range_ylim[1], 500)
            X, Y = np.meshgrid(x_fine, y_fine)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p_grid, m_grid = phasor_to_polar(X, Y)

            stat_display = p_grid if output_type == "Phase" else m_grid
            extent = [
                range_xlim[0],
                range_xlim[1],
                range_ylim[0],
                range_ylim[1],
            ]

            import matplotlib.patches as mpatches
            import matplotlib.path as mpath

            # Get paths and hide contours first
            contour_collections = getattr(pw, '_contour_collections', [])
            min_paths = []
            if contour_collections:
                for cs in contour_collections:
                    if hasattr(cs, 'collections') and len(cs.collections) > 0:
                        # Matplotlib < 3.8: collections is a list of LineCollections per level
                        for col in cs.collections:
                            for p in col.get_paths():
                                if (
                                    p.vertices is not None
                                    and len(p.vertices) > 0
                                ):
                                    vertices = p.vertices
                                    codes = p.codes
                                    if codes is None:
                                        codes = np.full(
                                            len(vertices), mpath.Path.LINETO
                                        )
                                        codes[0] = mpath.Path.MOVETO
                                    min_paths.append((vertices, codes))
                            col.set_visible(False)
                    elif hasattr(cs, 'get_paths'):
                        # Matplotlib >= 3.8: get_paths() returns a list of Paths, one per level
                        for p in cs.get_paths():
                            if p.vertices is not None and len(p.vertices) > 0:
                                vertices = p.vertices
                                codes = p.codes
                                if codes is None:
                                    codes = np.full(
                                        len(vertices), mpath.Path.LINETO
                                    )
                                    codes[0] = mpath.Path.MOVETO
                                min_paths.append((vertices, codes))
                        cs.set_visible(False)

            if not min_paths:
                # If no contours found or they don't have paths, don't show the mesh overlay
                self._remove_plot_overlay()
                self._set_histogram_density_visible(
                    pw, pw.plot_type == 'HISTOGRAM2D'
                )
                pw.canvas_widget.figure.canvas.draw_idle()
                return

            self._remove_plot_overlay()
            self._set_histogram_density_visible(pw, False)

            self._overlay_imshow = ax.imshow(
                stat_display,
                extent=extent,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
                zorder=1.5,
                alpha=1.0,
                aspect="auto",
            )

            all_v = np.concatenate([v for v, c in min_paths])
            all_c = np.concatenate([c for v, c in min_paths])
            compound_path = mpath.Path(all_v, all_c)
            patch = mpatches.PathPatch(
                compound_path,
                transform=ax.transData,
                facecolor='none',
                edgecolor='none',
            )
            ax.add_patch(patch)
            self._overlay_imshow.set_clip_path(patch)
            self._overlay_clip_patch = patch

            ax.set_aspect(1, adjustable="box")
            pw.canvas_widget.figure.canvas.draw_idle()
            return

        hist_artist = pw.canvas_widget.artists.get("HISTOGRAM2D")
        if hist_artist is None or hist_artist.histogram is None:
            return
        H, x_edges, y_edges = hist_artist.histogram

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stat, _, _, _ = binned_statistic_2d(
                g_flat,
                s_flat,
                values,
                statistic="median",
                bins=[x_edges, y_edges],
            )

        mask = np.isnan(H) | (H <= 0)
        zorder = 3
        if pw.plot_type == 'CONTOUR':
            zorder = 1.5
            contour_collections = getattr(pw, '_contour_collections', [])
            if contour_collections:
                min_levels = []
                for cs in contour_collections:
                    if hasattr(cs, 'levels') and len(cs.levels) > 0:
                        min_levels.append(cs.levels[0])
                if min_levels:
                    lowest_level = min(min_levels)
                    mask = lowest_level > H

        stat[mask] = np.nan
        stat_display = stat.T
        if np.all(np.isnan(stat_display)):
            return

        # Ensure vmin, vmax match the calculated stat range if none was available
        with np.errstate(invalid='ignore'):
            if (
                vmin is None
                or vmax is None
                or not np.isfinite(vmin)
                or not np.isfinite(vmax)
            ):
                vmin = float(np.nanmin(stat_display))
                vmax = float(np.nanmax(stat_display))

        ax = pw.canvas_widget.axes
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        self._remove_plot_overlay()
        self._set_histogram_density_visible(pw, False)

        self._overlay_imshow = ax.imshow(
            stat_display,
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            zorder=zorder,
            alpha=1.0,
            aspect="auto",
        )
        ax.set_aspect(1, adjustable="box")
        pw.canvas_widget.figure.canvas.draw_idle()

    def reapply_if_active(self):
        """Redraw the histogram and colouring unless the tab is hidden."""
        if self._coloring_paused_by_tab:
            self._clear_2d_coloring()
            return

        self.plot_lifetime_histogram()

        output_type = self._get_selected_output_type()
        if (
            output_type in LIFETIME_OUTPUT_TYPES
            and self.mesh_overlay_checkbox.isChecked()
        ):
            self._apply_lifetime_mesh(output_type)
        elif output_type not in {"Phase", "Modulation"}:
            self._clear_2d_coloring()

    def on_tab_visibility_changed(self, is_visible: bool):
        """Pause plot colouring while the tab is hidden, restoring it on show."""
        self._coloring_paused_by_tab = not is_visible
        if not is_visible:
            self._clear_2d_coloring()
            return
        self.reapply_if_active()

    def closeEvent(self, event):
        """Clean up signal connections before closing."""
        self._output_refresh_timer.stop()
        self._mesh_axes_update_timer.stop()
        self._disconnect_axes_limit_callbacks()

        # Disconnect all lifetime layer events
        for layer in self.metric_layers:
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                layer.events.colormap.disconnect(self._on_colormap_changed)
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                layer.events.contrast_limits.disconnect(
                    self._on_colormap_changed
                )
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                layer.events.gamma.disconnect(self._on_colormap_changed)

        event.accept()

    def _on_mesh_overlay_toggled(self, checked):
        """Handle mesh overlay toggle."""
        if self._updating_settings:
            return

        self._sync_mode_widgets()
        if checked:
            settings = self._get_current_layer_mapping_settings(create=False)
            self._sync_mesh_ranges(settings)

        self._update_lifetime_setting_in_metadata(
            'mesh_overlay_enabled', bool(checked)
        )

        output_type = self._get_selected_output_type()
        if output_type in LIFETIME_OUTPUT_TYPES:
            if checked and self._mesh_frequency() is None:
                show_warning(
                    "Enter the frequency (MHz) to draw the lifetime mesh."
                )
            self._apply_lifetime_mesh(output_type)
        elif checked or self.apply_2d_colormap_checkbox.isChecked():
            self._apply_histogram_coloring(output_type)
        else:
            self._clear_2d_coloring()

    def _on_mesh_clip_toggled(self, checked):
        """Handle mesh clipping toggle."""
        if self._updating_settings:
            return

        self._update_lifetime_setting_in_metadata(
            'mesh_clip_semicircle_enabled', bool(checked)
        )
        self._refresh_mesh_overlay_if_needed()

    def _update_mapping_colorbar(self, cmap, vmin, vmax, output_type):
        """Update the plotter colorbar for phase or modulation."""
        pw = self.parent_widget
        if pw is None:
            return
        norm = Normalize(vmin=vmin, vmax=vmax)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        if output_type in LIFETIME_OUTPUT_TYPES:
            label = "Lifetime (ns)"
        elif output_type == "Phase":
            label = "Phase (rad)"
        else:
            label = "Modulation"
        # Access the private method to update the colorbar in the plotter
        pw._update_mapping_colorbar(mappable=mappable, label=label)

    def _on_mesh_colorbar_toggled(self, checked):
        """Handle mesh colorbar toggle."""
        if self._updating_settings:
            return

        self._update_lifetime_setting_in_metadata(
            'mesh_colorbar_enabled', bool(checked)
        )

        output_type = self._get_selected_output_type()
        if output_type in LIFETIME_OUTPUT_TYPES:
            self._apply_lifetime_mesh(output_type)
        elif output_type in {"Phase", "Modulation"}:
            if checked:
                self._apply_histogram_coloring(output_type)
            elif self.parent_widget is not None:
                self.parent_widget._remove_mapping_colorbar()

    def _mesh_alpha(self):
        """Return the mesh opacity from the transparency control."""
        return round(1.0 - float(self.mesh_transparency_spinbox.value()), 10)

    def _on_mesh_transparency_changed(self, _value):
        """Refresh mesh overlay when its transparency changes."""
        if not self._updating_settings:
            self._update_lifetime_setting_in_metadata(
                'mesh_alpha', self._mesh_alpha()
            )
        self._refresh_mesh_overlay_if_needed()

    def _on_phase_slider_changed(self, value):
        """Handle phase range slider change from tab."""
        if self._updating_settings:
            return
        min_v, max_v = value
        min_f = min_v / self.phase_range_factor
        max_f = max_v / self.phase_range_factor
        self.phase_min_edit.setText(f"{min_f:.2f}")
        self.phase_max_edit.setText(f"{max_f:.2f}")

        if self.output_mode_combobox.currentText() == "Phase":
            self._sync_range_to_histogram(min_f, max_f)
        self._persist_current_mesh_ranges_to_metadata()
        self._refresh_mesh_overlay_if_needed()

    def _on_phase_edits_changed(self):
        """Handle phase range line edit change from tab."""
        if self._updating_settings:
            return
        try:
            min_f = float(self.phase_min_edit.text())
            max_f = float(self.phase_max_edit.text())
            min_v = int(min_f * self.phase_range_factor)
            max_v = int(max_f * self.phase_range_factor)
            self.phase_range_slider.setValue((min_v, max_v))
            # slider valueChanged will trigger sync, persist, and refresh
        except ValueError:
            pass

    def _on_modulation_slider_changed(self, value):
        """Handle modulation range slider change from tab."""
        if self._updating_settings:
            return
        min_v, max_v = value
        min_f = min_v / self.modulation_range_factor
        max_f = max_v / self.modulation_range_factor
        self.modulation_min_edit.setText(f"{min_f:.2f}")
        self.modulation_max_edit.setText(f"{max_f:.2f}")

        if self.output_mode_combobox.currentText() == "Modulation":
            self._sync_range_to_histogram(min_f, max_f)
        self._persist_current_mesh_ranges_to_metadata()
        self._refresh_mesh_overlay_if_needed()

    def _on_modulation_edits_changed(self):
        """Handle modulation range line edit change from tab."""
        if self._updating_settings:
            return
        try:
            min_f = float(self.modulation_min_edit.text())
            max_f = float(self.modulation_max_edit.text())
            min_v = int(min_f * self.modulation_range_factor)
            max_v = int(max_f * self.modulation_range_factor)
            self.modulation_range_slider.setValue((min_v, max_v))
            # slider valueChanged will trigger sync, persist, and refresh
        except ValueError:
            pass

    # Lifetime mesh ------------------------------------------------------

    def _mesh_frequency(self):
        """Return the lifetime mesh's effective frequency (MHz), or ``None``.

        That is the tab's frequency times the displayed harmonic, the
        frequency the lifetime output map is computed at.
        """
        frequency = self._parse_positive_frequency(
            self.frequency_input.text().strip()
        )
        if frequency is None:
            return None
        harmonic = getattr(self.parent_widget, 'harmonic', 1) or 1
        return frequency * harmonic

    def _lifetime_mesh_range(self):
        """Return the lifetime mesh ``(min, max)`` range in ns."""
        min_i, max_i = self.lifetime_mesh_range_slider.value()
        factor = self.lifetime_mesh_range_factor
        return min_i / factor, max_i / factor

    def _set_lifetime_mesh_range(
        self, lifetime_min, lifetime_max, slider_max=None
    ):
        """Show ``[lifetime_min, lifetime_max]`` ns on the lifetime controls.

        The slider's end is *slider_max* (its current end when omitted),
        widened when needed so the range is never clamped.
        """
        factor = self.lifetime_mesh_range_factor
        lifetime_min = max(0.0, float(lifetime_min))
        lifetime_max = max(lifetime_min, float(lifetime_max))
        if slider_max is None:
            slider_max = self.lifetime_mesh_range_slider.maximum() / factor
        slider_max = max(float(slider_max), lifetime_max)

        was_updating = self._updating_settings
        self._updating_settings = True
        try:
            self.lifetime_mesh_range_slider.setRange(
                0, int(round(slider_max * factor))
            )
            self.lifetime_mesh_range_slider.setValue(
                (
                    int(round(lifetime_min * factor)),
                    int(round(lifetime_max * factor)),
                )
            )
            self.lifetime_mesh_min_edit.setText(f"{lifetime_min:.2f}")
            self.lifetime_mesh_max_edit.setText(f"{lifetime_max:.2f}")
        finally:
            self._updating_settings = was_updating

    def _initialize_lifetime_mesh_range_from_current_data(self):
        """Fit the lifetime mesh range to the plotted data's lifetimes.

        Needs a frequency; without one the controls are left untouched.
        """
        output_type = self._get_selected_output_type()
        frequency = self._mesh_frequency()
        pw = self.parent_widget
        if (
            pw is None
            or frequency is None
            or output_type not in LIFETIME_OUTPUT_TYPES
        ):
            return
        slider_max = lifetime_mesh_upper_bound(frequency)
        lifetime_range = None
        features = pw.get_merged_features()
        if features is not None:
            lifetime_range = lifetime_mesh_range_from_phasors(
                output_type, *features, frequency
            )
        lifetime_min, lifetime_max = lifetime_range or (0.0, slider_max)
        self._set_lifetime_mesh_range(lifetime_min, lifetime_max, slider_max)
        self._persist_lifetime_mesh_range_to_metadata()

    def _restore_lifetime_mesh_range_from_settings(self, settings) -> bool:
        """Apply the stored range of the selected lifetime type.

        Returns False when there is none, leaving the controls untouched.
        """
        output_type = self._get_selected_output_type()
        if settings is None or output_type not in LIFETIME_OUTPUT_TYPES:
            return False
        ranges = settings.get('mesh_lifetime_ranges')
        stored = ranges.get(output_type) if isinstance(ranges, dict) else None
        if (
            not isinstance(stored, (list, tuple))
            or len(stored) != 2
            or None in stored
        ):
            return False
        frequency = self._mesh_frequency()
        slider_max = (
            lifetime_mesh_upper_bound(frequency)
            if frequency is not None
            else None
        )
        self._set_lifetime_mesh_range(stored[0], stored[1], slider_max)
        return True

    def _sync_lifetime_mesh_range(self, settings=None):
        """Restore the selected lifetime type's mesh range, else fit it."""
        if self.parent_widget is None:
            return
        if settings is None:
            settings = self._get_current_layer_mapping_settings(create=False)
        if not self._restore_lifetime_mesh_range_from_settings(settings):
            self._initialize_lifetime_mesh_range_from_current_data()

    def _persist_lifetime_mesh_range_to_metadata(self):
        """Save the lifetime mesh range under the selected lifetime type."""
        if self._updating_settings or self.parent_widget is None:
            return
        output_type = self._get_selected_output_type()
        if output_type not in LIFETIME_OUTPUT_TYPES:
            return
        settings = self._get_current_layer_mapping_settings(create=True)
        if settings is None:
            return
        ranges = settings.get('mesh_lifetime_ranges')
        ranges = dict(ranges) if isinstance(ranges, dict) else {}
        ranges[output_type] = [float(v) for v in self._lifetime_mesh_range()]
        self._stage_mapping_values({'mesh_lifetime_ranges': ranges})

    def _on_lifetime_mesh_slider_changed(self, value):
        """Handle a lifetime mesh range slider change."""
        if self._updating_settings:
            return
        factor = self.lifetime_mesh_range_factor
        self.lifetime_mesh_min_edit.setText(f"{value[0] / factor:.2f}")
        self.lifetime_mesh_max_edit.setText(f"{value[1] / factor:.2f}")
        self._sync_lifetime_mesh_range_to_histogram()
        self._persist_lifetime_mesh_range_to_metadata()
        self._refresh_mesh_overlay_if_needed()

    def _on_lifetime_mesh_edits_changed(self):
        """Handle a lifetime mesh range line edit change."""
        if self._updating_settings:
            return
        try:
            lifetime_min = float(self.lifetime_mesh_min_edit.text())
            lifetime_max = float(self.lifetime_mesh_max_edit.text())
        except ValueError:
            return
        # A value typed past the slider's end widens it rather than being
        # clamped to it.
        self._set_lifetime_mesh_range(lifetime_min, lifetime_max)
        self._sync_lifetime_mesh_range_to_histogram()
        self._persist_lifetime_mesh_range_to_metadata()
        self._refresh_mesh_overlay_if_needed()

    def _sync_lifetime_mesh_range_to_histogram(self):
        """Apply the lifetime mesh range to the histogram and output maps.

        The two ranges are linked, as the phase mesh range and the histogram
        are in Phase mode: the mesh band is the displayed lifetime range, and
        ``_update_tab_sliders_from_range`` carries the reverse direction.
        """
        if self._get_selected_output_type() in LIFETIME_OUTPUT_TYPES:
            self._sync_range_to_histogram(*self._lifetime_mesh_range())

    def _lifetime_mesh_colormap(self, output_type):
        """Return ``(cmap, vmin, vmax)`` for the *output_type* lifetime mesh.

        The mesh takes the output layers' own colormap and contrast limits,
        so a mesh cell and a pixel with the same lifetime share a colour.
        Without a matching layer (nothing calculated yet, or another lifetime
        on screen) the default lifetime colormap spans the mesh range.
        """
        for layer in self.metric_layers:
            info = self._mapping_output_info(layer)
            if (
                info is None
                or info[0] != output_type
                or layer not in self.viewer.layers
            ):
                continue
            with contextlib.suppress(AttributeError, TypeError, ValueError):
                vmin, vmax = (float(v) for v in layer.contrast_limits)
                cmap = LinearSegmentedColormap.from_list(
                    layer.colormap.name, layer.colormap.colors
                )
                return cmap, vmin, vmax
        cmap = resolve_colormap_by_name(
            self._get_output_colormap_name(output_type)
        ) or resolve_colormap_by_name("viridis")
        vmin, vmax = self._lifetime_mesh_range()
        return cmap, vmin, vmax

    @staticmethod
    def _lifetime_mesh_field(mesh_grid, output_type, frequency):
        """Return *mesh_grid*'s lifetime field, cached on the grid entry.

        Only the latest lifetime type/frequency is kept per grid: those
        change rarely, while the view (and so the grid) changes constantly.
        """
        key = (output_type, float(frequency))
        cached = mesh_grid.get('lifetime')
        if cached is not None and cached[0] == key:
            return cached[1]
        field = compute_lifetime_mesh_field(
            mesh_grid['p_grid'], mesh_grid['m_grid'], output_type, frequency
        )
        mesh_grid['lifetime'] = (key, field)
        return field

    def _apply_lifetime_mesh(self, output_type: str):
        """Draw the *output_type* lifetime mesh behind the phasor data.

        Only the mesh is drawn: unlike Phase/Modulation, Lifetime mode does
        not recolour the phasor data itself. Without the mesh toggle or a
        frequency the mesh and its colorbar are removed.
        """
        if output_type not in LIFETIME_OUTPUT_TYPES:
            return
        pw = self.parent_widget
        if pw is None or getattr(pw, 'plot_type', 'HISTOGRAM2D') == 'NONE':
            return
        frequency = self._mesh_frequency()
        if not self.mesh_overlay_checkbox.isChecked() or frequency is None:
            self._remove_mesh_overlay()
            pw._remove_mapping_colorbar()
            return

        ax = pw.canvas_widget.axes
        resolution = self._get_mesh_grid_resolution(ax)
        mesh_grid = self._get_mesh_polar_grid(ax, resolution)
        p_grid = mesh_grid['p_grid']
        m_grid = mesh_grid['m_grid']
        lifetime_grid = self._lifetime_mesh_field(
            mesh_grid, output_type, frequency
        )

        semicircle = self._is_semicircle_mode()
        clip_semicircle = (
            self.mesh_clip_semicircle_checkbox.isChecked() and semicircle
        )
        mesh_mask = compute_phasor_mesh_mask(
            p_grid,
            m_grid,
            semicircle=semicircle,
            clip_semicircle=clip_semicircle,
            lifetime_grid=lifetime_grid,
            lifetime_range=self._lifetime_mesh_range(),
        )
        alpha_key = (
            *self._make_mesh_grid_cache_key(ax, resolution),
            output_type,
            float(frequency),
            *self.lifetime_mesh_range_slider.value(),
            clip_semicircle,
        )
        mesh_alpha_map = self._get_mesh_alpha_map(
            mesh_mask, alpha_key, self._mesh_alpha(), resolution, ax
        )
        cmap, vmin, vmax = self._lifetime_mesh_colormap(output_type)

        self._remove_mesh_overlay()
        self._mesh_overlay_imshow = draw_phasor_mesh(
            ax,
            output_type,
            semicircle=semicircle,
            colormap=cmap,
            alpha_map=mesh_alpha_map,
            vmin=vmin,
            vmax=vmax,
            p_grid=p_grid,
            m_grid=m_grid,
            mask=mesh_mask,
            extent=mesh_grid['extent'],
            lifetime_grid=lifetime_grid,
        )
        if self.mesh_colorbar_checkbox.isChecked():
            self._update_mapping_colorbar(cmap, vmin, vmax, output_type)
        else:
            pw._remove_mapping_colorbar()
        pw.canvas_widget.figure.canvas.draw_idle()

    def _sync_range_to_histogram(self, min_f, max_f):
        """Sync tab slider range changes to HistogramWidget and update layers."""
        # Use HistogramWidget's factor for its slider
        factor = self.lifetime_range_factor
        self.histogram_widget.range_slider.blockSignals(True)
        try:
            self.histogram_widget.set_range(min_f, max_f)
            # Re-run current range logic to update napari layers
            self._on_lifetime_range_changed(
                (int(min_f * factor), int(max_f * factor))
            )
        finally:
            self.histogram_widget.range_slider.blockSignals(False)

    def _update_tab_sliders_from_range(self, min_f, max_f):
        """Update tab sliders when range changes from HistogramWidget."""
        output_type = self._get_selected_output_type()
        self._updating_settings = True
        try:
            if output_type == "Phase":
                min_v = int(min_f * self.phase_range_factor)
                max_v = int(max_f * self.phase_range_factor)
                self.phase_range_slider.setValue((min_v, max_v))
                self.phase_min_edit.setText(f"{min_f:.2f}")
                self.phase_max_edit.setText(f"{max_f:.2f}")
            elif output_type == "Modulation":
                min_v = int(min_f * self.modulation_range_factor)
                max_v = int(max_f * self.modulation_range_factor)
                self.modulation_range_slider.setValue((min_v, max_v))
                self.modulation_min_edit.setText(f"{min_f:.2f}")
                self.modulation_max_edit.setText(f"{max_f:.2f}")
            elif output_type in LIFETIME_OUTPUT_TYPES:
                self._set_lifetime_mesh_range(min_f, max_f)
        finally:
            self._updating_settings = False
        if output_type in LIFETIME_OUTPUT_TYPES:
            self._persist_lifetime_mesh_range_to_metadata()
