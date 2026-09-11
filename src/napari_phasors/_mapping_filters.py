"""Metric filters that invalidate phasor coordinates outside a value range.

The Filter tab restricts an image by *intensity*. The filters defined here
restrict it by any quantity **derived** from the phasor coordinates -- an
apparent lifetime, the phase or modulation, a FRET efficiency -- by setting
the phasor coordinates of every pixel outside the requested range to NaN.

The whole point of this module is that these filters form one **flat,
declarative stack**, not a chain of edits:

* Every criterion is stored as a plain dict in
  ``layer.metadata['settings']['mapping_filters']``. The stack is the single
  source of truth; the arrays are always recomputed from it.
* Each criterion's metric is evaluated on the *baseline* arrays (the output
  of the intensity threshold, the median/wavelet filter and the mask), never
  on the partially-NaN arrays another criterion produced. The result is the
  logical AND of the criteria, so the stack is order-independent and adding a
  fourth filter can never silently discard the first three.
* Because nothing is applied in place, removing or disabling a criterion
  restores exactly the pixels it had hidden.

:class:`MappingFilterList` is the Qt counterpart: one card per criterion, so
what the data shows and what the user sees listed cannot drift apart. It is
kept in this module so the two halves are edited together, but every function
above it is Qt-free and safe to call from a worker thread.
"""

import uuid

import numpy as np
from phasorpy.lifetime import (
    phasor_from_fret_donor,
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_nearest_neighbor, phasor_to_polar
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

#: Key under ``layer.metadata['settings']`` holding the filter stack.
MAPPING_FILTERS_KEY = 'mapping_filters'
#: Key of the single-filter format this stack replaced, still read on load.
LEGACY_FILTER_KEY = 'mapping_filter'

APPARENT_PHASE_LIFETIME = "Apparent Phase Lifetime"
APPARENT_MODULATION_LIFETIME = "Apparent Modulation Lifetime"
NORMAL_LIFETIME = "Normal Lifetime"
PHASE = "Phase"
MODULATION = "Modulation"
FRET_EFFICIENCY = "FRET efficiency"

#: Metrics that can only be evaluated with an excitation frequency.
LIFETIME_METRICS = frozenset(
    {APPARENT_PHASE_LIFETIME, APPARENT_MODULATION_LIFETIME, NORMAL_LIFETIME}
)
#: Metrics offered by the Phasor Mapping tab, in menu order.
MAPPING_METRICS = (
    APPARENT_PHASE_LIFETIME,
    APPARENT_MODULATION_LIFETIME,
    NORMAL_LIFETIME,
    PHASE,
    MODULATION,
)
#: Every metric a stored criterion may name.
ALL_METRICS = MAPPING_METRICS + (FRET_EFFICIENCY,)

#: Unit shown after a criterion's range, per metric ("" when dimensionless).
METRIC_UNITS = {
    APPARENT_PHASE_LIFETIME: "ns",
    APPARENT_MODULATION_LIFETIME: "ns",
    NORMAL_LIFETIME: "ns",
    PHASE: "rad",
    MODULATION: "",
    FRET_EFFICIENCY: "",
}

#: Range used for a criterion's slider before any data has been measured.
METRIC_FALLBACK_RANGE = {
    APPARENT_PHASE_LIFETIME: (0.0, 10.0),
    APPARENT_MODULATION_LIFETIME: (0.0, 10.0),
    NORMAL_LIFETIME: (0.0, 10.0),
    PHASE: (0.0, float(2.0 * np.pi)),
    MODULATION: (0.0, 1.0),
    FRET_EFFICIENCY: (0.0, 1.0),
}

#: Keeps pixels *inside* the range; the opposite punches the range out.
KEEP = 'keep'
EXCLUDE = 'exclude'

#: Efficiencies sampled along the FRET donor trajectory, matching the Fret tab.
_FRET_EFFICIENCY_SAMPLES = 500


def metric_unit(metric):
    """Return the unit string shown after *metric*'s range."""
    return METRIC_UNITS.get(metric, "")


def metric_fallback_range(metric):
    """Return the range to show for *metric* when no data has been measured."""
    return METRIC_FALLBACK_RANGE.get(metric, (0.0, 1.0))


def requires_frequency(metric):
    """Return whether *metric* can only be computed with a frequency."""
    return metric in LIFETIME_METRICS


def format_range(metric, minimum, maximum, mode=KEEP):
    """Return a compact human-readable description of one criterion."""
    unit = metric_unit(metric)
    suffix = f" {unit}" if unit else ""
    verb = "outside" if mode == EXCLUDE else ""
    body = f"{minimum:g} – {maximum:g}{suffix}"
    return f"{verb} {body}".strip()


def new_filter(
    metric,
    minimum,
    maximum,
    harmonic=1,
    *,
    mode=KEEP,
    enabled=True,
    params=None,
    wrap_phase=False,
):
    """Return a new criterion dict for *metric* over ``[minimum, maximum]``."""
    return {
        'id': uuid.uuid4().hex,
        'metric': metric,
        'harmonic': int(harmonic),
        'min': float(minimum),
        'max': float(maximum),
        'mode': EXCLUDE if mode == EXCLUDE else KEEP,
        'enabled': bool(enabled),
        'wrap_phase': bool(wrap_phase),
        'params': dict(params) if params else {},
    }


def _coerce_filter(raw):
    """Return *raw* as a valid criterion dict, or ``None`` if unusable.

    Stored settings survive an OME-TIFF round-trip and can be hand-edited, so
    anything that is not a complete, finite range over a known metric is
    dropped rather than allowed to silently blank an image.
    """
    if not isinstance(raw, dict):
        return None
    metric = raw.get('metric', raw.get('output_type'))
    if metric not in ALL_METRICS:
        return None
    try:
        minimum = float(raw['min'])
        maximum = float(raw['max'])
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(minimum) and np.isfinite(maximum)):
        return None
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    try:
        harmonic = int(raw.get('harmonic', 1))
    except (TypeError, ValueError):
        harmonic = 1
    params = raw.get('params')
    entry = new_filter(
        metric,
        minimum,
        maximum,
        harmonic,
        mode=raw.get('mode', KEEP),
        enabled=raw.get('enabled', True),
        params=params if isinstance(params, dict) else None,
        wrap_phase=raw.get('wrap_phase', False),
    )
    stored_id = raw.get('id')
    if isinstance(stored_id, str) and stored_id:
        entry['id'] = stored_id
    return entry


def normalize_filters(raw):
    """Return *raw* as a clean list of criteria, dropping unusable entries."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return [f for f in (_coerce_filter(item) for item in raw) if f]


def get_filters(layer):
    """Return the filter stack stored on *layer*.

    Layers written before the stack existed carry a single ``mapping_filter``
    dict instead; it is read as a one-entry stack so an old project keeps the
    filter it was saved with.
    """
    return filters_from_settings(layer.metadata.get('settings'))


def filters_from_settings(settings):
    """Return the filter stack stored in a ``settings`` dict.

    The layer-free half of :func:`get_filters`, for the reader, which has
    the settings before any layer exists.
    """
    settings = settings or {}
    if MAPPING_FILTERS_KEY in settings:
        return normalize_filters(settings.get(MAPPING_FILTERS_KEY))
    legacy = settings.get(LEGACY_FILTER_KEY)
    if legacy is None:
        legacy = (settings.get('phasor_mapping') or {}).get(LEGACY_FILTER_KEY)
    return normalize_filters(legacy)


def set_filters(layer, filters):
    """Store *filters* on *layer*, replacing any previous stack."""
    settings = layer.metadata.setdefault('settings', {})
    normalized = normalize_filters(filters)
    if normalized:
        settings[MAPPING_FILTERS_KEY] = normalized
    else:
        settings.pop(MAPPING_FILTERS_KEY, None)
    # The single-filter format is write-only history from here on: leaving a
    # copy behind would resurrect a deleted criterion on the next load.
    settings.pop(LEGACY_FILTER_KEY, None)
    mapping_settings = settings.get('phasor_mapping')
    if isinstance(mapping_settings, dict):
        mapping_settings.pop(LEGACY_FILTER_KEY, None)
    return normalized


def has_filters(layer):
    """Return whether *layer* carries at least one enabled criterion."""
    return any(f['enabled'] for f in get_filters(layer))


def select_harmonic(real, imag, harmonics, harmonic, mean_ndim):
    """Return the ``(real, imag)`` plane for *harmonic*.

    Falls back to the first harmonic when the requested one was never
    computed, which is what every other tab does with a stale selection.
    """
    if real is None or imag is None:
        return None, None
    if real.ndim <= mean_ndim:
        return real, imag
    values = (
        np.atleast_1d(harmonics) if harmonics is not None else np.array([1])
    )
    match = np.where(values == harmonic)[0]
    index = int(match[0]) if len(match) else 0
    if index >= real.shape[0]:
        index = 0
    return real[index], imag[index]


def compute_metric(
    metric,
    real,
    imag,
    *,
    harmonic=1,
    frequency=None,
    wrap_phase=False,
    params=None,
):
    """Return *metric* evaluated pixel-wise on the ``(real, imag)`` plane.

    Pure array work -- NumPy's error state is thread-local -- so this is safe
    to call from a worker thread.

    Parameters
    ----------
    metric : str
        One of :data:`ALL_METRICS`.
    real, imag : np.ndarray
        A single harmonic's phasor coordinates (see :func:`select_harmonic`).
    harmonic : int, optional
        Harmonic the coordinates belong to; scales the frequency.
    frequency : float, optional
        Excitation frequency in MHz. Required by the lifetime metrics.
    wrap_phase : bool, optional
        Wrap the phase into ``[0, 2pi)`` instead of ``(-pi, pi]``, matching
        the full-polar plot mode.
    params : dict, optional
        Extra metric parameters. ``FRET efficiency`` reads its donor
        trajectory from here.

    Returns
    -------
    np.ndarray or None
        The metric map, or ``None`` when it cannot be computed.
    """
    if real is None or imag is None:
        return None
    params = params or {}

    if metric in LIFETIME_METRICS:
        if frequency is None:
            frequency = params.get('frequency')
        if frequency is None or not np.isfinite(frequency) or frequency <= 0:
            return None
        effective = float(frequency) * max(int(harmonic), 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            if metric == NORMAL_LIFETIME:
                values = phasor_to_normal_lifetime(
                    real, imag, frequency=effective
                )
            else:
                phase_lifetime, modulation_lifetime = (
                    phasor_to_apparent_lifetime(
                        real, imag, frequency=effective
                    )
                )
                values = (
                    phase_lifetime
                    if metric == APPARENT_PHASE_LIFETIME
                    else modulation_lifetime
                )
        values = np.asarray(values, dtype=float)
        with np.errstate(invalid='ignore'):
            values[values < 0] = 0
        return values

    if metric in (PHASE, MODULATION):
        with np.errstate(divide='ignore', invalid='ignore'):
            phase_values, modulation_values = phasor_to_polar(real, imag)
        if metric == MODULATION:
            return np.asarray(modulation_values, dtype=float)
        phase_values = np.asarray(phase_values, dtype=float)
        if wrap_phase:
            with np.errstate(invalid='ignore'):
                phase_values = np.mod(phase_values, 2.0 * np.pi)
        return phase_values

    if metric == FRET_EFFICIENCY:
        return _compute_fret_efficiency(real, imag, harmonic, params)

    return None


def _compute_fret_efficiency(real, imag, harmonic, params):
    """Return the FRET efficiency map for the given donor trajectory.

    The trajectory is rebuilt from the parameters stored with the criterion
    rather than read off the Fret tab, so the filter keeps meaning the same
    thing after the tab's sliders have moved on.
    """
    frequency = params.get('frequency')
    donor_lifetime = params.get('donor_lifetime')
    if frequency is None or donor_lifetime is None:
        return None
    try:
        frequency = float(frequency)
        donor_lifetime = float(donor_lifetime)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(frequency) or frequency <= 0:
        return None
    if not np.isfinite(donor_lifetime) or donor_lifetime <= 0:
        return None

    def optional(key, default):
        """Return a stored trajectory parameter, or phasorpy's default."""
        value = params.get(key)
        return default if value is None else value

    efficiencies = np.linspace(0, 1, _FRET_EFFICIENCY_SAMPLES)
    neighbor_real, neighbor_imag = phasor_from_fret_donor(
        frequency=frequency * max(int(harmonic), 1),
        donor_lifetime=donor_lifetime,
        fret_efficiency=efficiencies,
        donor_background=optional('donor_background', 0.0),
        background_real=optional('background_real', 0.0),
        background_imag=optional('background_imag', 0.0),
        donor_fretting=optional('donor_fretting', 1.0),
    )
    values = np.asarray(
        phasor_nearest_neighbor(
            real,
            imag,
            neighbor_real,
            neighbor_imag,
            values=efficiencies,
        ),
        dtype=float,
    )
    # A pixel with no phasor coordinate has no efficiency; the nearest
    # neighbour search would otherwise hand it whichever sample it landed on.
    invalid = ~np.isfinite(real) | ~np.isfinite(imag)
    if invalid.any():
        values = np.where(invalid, np.nan, values)
    return values


def range_mask(values, minimum, maximum, mode=KEEP):
    """Return the "drop this pixel" mask for *values* against one range.

    NaN is always dropped: a pixel with no measurable metric cannot satisfy a
    criterion about that metric.
    """
    values = np.asarray(values, dtype=float)
    with np.errstate(invalid='ignore'):
        missing = ~np.isfinite(values)
        inside = (values >= minimum) & (values <= maximum)
    if mode == EXCLUDE:
        return missing | inside
    return missing | ~inside


def combined_mask(filters, mean, real, imag, harmonics, *, on_error=None):
    """Return the "drop this pixel" mask for the whole stack, or ``None``.

    Every enabled criterion is evaluated on the arrays as given -- the
    caller's baseline -- and the masks are OR-ed, which makes the stack a
    logical AND of the criteria and independent of the order they were added
    in. ``None`` means nothing is filtered.

    Parameters
    ----------
    filters : sequence of dict
        The stack, as returned by :func:`get_filters`.
    mean, real, imag : np.ndarray
        Baseline phasor arrays. *real* and *imag* may carry a leading
        harmonic axis.
    harmonics : np.ndarray or None
        Harmonic values matching that axis.
    on_error : callable, optional
        Called with a message for every criterion that could not be
        evaluated (a lifetime filter with no frequency, say). Those criteria
        are skipped rather than treated as excluding everything.
    """
    if mean is None:
        return None
    mask = None
    for entry in filters:
        if not entry['enabled']:
            continue
        plane_real, plane_imag = select_harmonic(
            real, imag, harmonics, entry['harmonic'], mean.ndim
        )
        values = compute_metric(
            entry['metric'],
            plane_real,
            plane_imag,
            harmonic=entry['harmonic'],
            frequency=entry['params'].get('frequency'),
            wrap_phase=entry['wrap_phase'],
            params=entry['params'],
        )
        if values is None or values.shape != mean.shape:
            if on_error is not None:
                on_error(
                    f"Skipped the {entry['metric']} filter: it cannot be "
                    "evaluated on this layer."
                )
            continue
        entry_mask = range_mask(
            values, entry['min'], entry['max'], entry['mode']
        )
        mask = entry_mask if mask is None else (mask | entry_mask)
    return mask


def apply_mask_to_arrays(mask, mean, real, imag):
    """Return copies of the arrays with *mask*'s pixels set to NaN."""
    if mask is None:
        return mean, real, imag
    if mean is not None:
        mean = np.where(mask, np.nan, mean)
    if real is not None and imag is not None:
        expanded = (
            mask[np.newaxis, ...]
            if mean is not None and real.ndim > mean.ndim
            else mask
        )
        real = np.where(expanded, np.nan, real)
        imag = np.where(expanded, np.nan, imag)
    return mean, real, imag


def apply_filters_to_arrays(
    filters, mean, real, imag, harmonics, *, on_error=None
):
    """Return ``(mean, real, imag, mask)`` with the stack applied.

    The arrays are treated as the baseline; nothing already NaN is restored.
    """
    mask = combined_mask(
        filters, mean, real, imag, harmonics, on_error=on_error
    )
    mean, real, imag = apply_mask_to_arrays(mask, mean, real, imag)
    return mean, real, imag, mask


def apply_layer_filters(layer, arrays, *, harmonics=None, on_error=None):
    """Apply *layer*'s stored stack to its freshly computed baseline arrays.

    Called at the one point where the phasor arrays are written back from the
    baseline, so a criterion can never be lost to an intensity threshold, a
    mask edit or a re-read of the layer.
    """
    mean, real, imag = arrays
    filters = get_filters(layer)
    if not filters:
        return mean, real, imag
    if harmonics is None:
        harmonics = layer.metadata.get('harmonics')
    mean, real, imag, _ = apply_filters_to_arrays(
        filters, mean, real, imag, harmonics, on_error=on_error
    )
    return mean, real, imag


def baseline_arrays(layer, filter_params=None):
    """Return *layer*'s phasor arrays with every step *but* the metric filters.

    That is the intensity threshold, the median/wavelet filter and the mask
    applied to the untouched originals. Every criterion is measured against
    these arrays, which is what keeps the stack a set of independent
    conditions rather than a chain where each one only sees the leftovers of
    the last.
    """
    from ._utils import compute_filter_and_threshold

    if 'G_original' in layer.metadata and 'original_mean' in layer.metadata:
        params = dict(filter_params or {})
        params.pop('threshold_method', None)
        return compute_filter_and_threshold(
            layer, **params, apply_mapping_filters=False
        )
    mean = layer.metadata.get('original_mean')
    if mean is None:
        mean = layer.data
    real = layer.metadata.get('G_original', layer.metadata.get('G'))
    imag = layer.metadata.get('S_original', layer.metadata.get('S'))
    return (
        mean.copy() if mean is not None else None,
        real.copy() if real is not None else None,
        imag.copy() if imag is not None else None,
    )


def rebuild_layer_from_filters(
    layer, filters=None, *, filter_params=None, on_error=None
):
    """Rewrite *layer*'s phasor arrays from its baseline plus *filters*.

    Returns the "dropped pixel" mask, or ``None`` when nothing was filtered.
    """
    if filters is None:
        filters = get_filters(layer)
    mean, real, imag = baseline_arrays(layer, filter_params)
    if mean is None:
        return None
    mean, real, imag, mask = apply_filters_to_arrays(
        filters,
        mean,
        real,
        imag,
        layer.metadata.get('harmonics'),
        on_error=on_error,
    )
    if real is not None:
        layer.metadata['G'] = real
    if imag is not None:
        layer.metadata['S'] = imag
    layer.data = mean
    layer.refresh()
    return mask


def kept_fraction(mask, mean=None):
    """Return the share of measurable pixels a mask keeps, in ``[0, 1]``.

    Pixels that were already NaN before filtering are left out of both sides
    of the ratio, so the number answers "how much of my data does this
    criterion keep" rather than "how much of the frame".
    """
    if mask is None:
        return 1.0
    if mean is not None:
        measurable = np.isfinite(mean)
        total = int(measurable.sum())
        if total == 0:
            return 0.0
        return float((measurable & ~mask).sum()) / total
    total = int(mask.size)
    if total == 0:
        return 0.0
    return float((~mask).sum()) / total


def describe_filters(filters):
    """Return a one-line summary of the enabled criteria in *filters*."""
    enabled = [f for f in filters if f['enabled']]
    if not enabled:
        return "No filters applied."
    parts = [
        f"{f['metric']} {format_range(f['metric'], f['min'], f['max'], f['mode'])}"
        for f in enabled
    ]
    return " · ".join(parts)


FILTER_CARD_STYLE = (
    "QFrame#mappingFilterCard {"
    "  border: 1px solid rgba(128, 128, 128, 0.3);"
    "  border-radius: 4px;"
    "  background-color: rgba(255, 255, 255, 0.02);"
    "}"
    "QFrame#mappingFilterCard:hover {"
    "  border: 1px solid rgba(128, 128, 128, 0.55);"
    "}"
    'QFrame#mappingFilterCard[muted="true"] {'
    "  background-color: rgba(128, 128, 128, 0.05);"
    "  border: 1px solid rgba(128, 128, 128, 0.2);"
    "}"
    'QFrame#mappingFilterCard[muted="true"] QLabel {'
    "  color: rgba(160, 160, 160, 0.6);"
    "}"
    "QPushButton#mappingFilterRemoveBtn {"
    "  background: transparent;"
    "  border: none;"
    "  color: rgba(200, 200, 200, 0.7);"
    "  font-size: 15px;"
    "  font-weight: bold;"
    "  padding: 0px;"
    "  border-radius: 3px;"
    "}"
    "QPushButton#mappingFilterRemoveBtn:hover {"
    "  color: #ff5555;"
    "  background: rgba(255, 85, 85, 0.15);"
    "}"
    "QLabel#mappingFilterStat {"
    "  color: rgba(180, 180, 180, 0.75);"
    "}"
    "QLabel#mappingFilterForeign {"
    "  color: rgba(180, 180, 180, 0.75);"
    "  font-style: italic;"
    "}"
)

_ADD_TOOLTIP = (
    "Add a filter. Pixels outside its range keep no phasor coordinates, so "
    "they disappear from the plot, the maps, the histogram and the "
    "statistics at once."
)
_METRIC_TOOLTIP = (
    "Quantity this filter is evaluated on. It does not have to be the "
    "quantity currently displayed."
)
_MODE_TOOLTIP = (
    "Keep: only pixels inside the range survive.\n"
    "Exclude: pixels inside the range are removed instead."
)
_ENABLE_TOOLTIP = (
    "Turn this filter off without deleting it. The pixels it hid come back "
    "immediately."
)


class _FilterCard(QFrame):
    """One criterion, shown as an editable card."""

    changed = Signal(str)
    """Emitted with this card's filter id after the user edited it."""

    removeRequested = Signal(str)
    """Emitted with this card's filter id when its × is clicked."""

    metricChangeRequested = Signal(str, str)
    """Emitted with the filter id and the metric picked in its selector."""

    def __init__(
        self,
        entry,
        *,
        metrics=None,
        editable=True,
        removable=True,
        parent=None,
    ):
        super().__init__(parent)
        self.entry = dict(entry)
        self.scale = 1000
        self._updating = False
        self._editable = bool(editable)
        self.setObjectName("mappingFilterCard")
        self.setStyleSheet(FILTER_CARD_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)

        self.enabled_check = QCheckBox()
        self.enabled_check.setChecked(self.entry['enabled'])
        self.enabled_check.setToolTip(_ENABLE_TOOLTIP)
        header.addWidget(self.enabled_check)

        # The metric is chosen on the card itself. A card that can only ever
        # be one metric (the Fret tab's, or another tab's criterion listed
        # here) shows it as plain text instead of a one-item selector.
        offered = list(metrics or [])
        if self.entry['metric'] not in offered:
            offered = [self.entry['metric']]
        self.metric_combobox = QComboBox()
        self.metric_combobox.addItems(offered)
        self.metric_combobox.setCurrentText(self.entry['metric'])
        self.metric_combobox.setVisible(len(offered) > 1)
        header.addWidget(self.metric_combobox, 1)

        self.metric_label = QLabel(self.entry['metric'])
        self.metric_label.setStyleSheet("font-weight: 600;")
        self.metric_label.setVisible(len(offered) == 1)
        header.addWidget(self.metric_label, 1)
        self._refresh_metric_tooltip()

        self.mode_combobox = QComboBox()
        self.mode_combobox.addItems(["Keep", "Exclude"])
        self.mode_combobox.setCurrentIndex(
            1 if self.entry['mode'] == EXCLUDE else 0
        )
        self.mode_combobox.setToolTip(_MODE_TOOLTIP)
        self.mode_combobox.setMaximumWidth(90)
        header.addWidget(self.mode_combobox)

        self.remove_button = QPushButton("×")
        self.remove_button.setObjectName("mappingFilterRemoveBtn")
        self.remove_button.setFixedSize(22, 22)
        self.remove_button.setToolTip("Remove this filter.")
        self.remove_button.setVisible(removable)
        header.addWidget(self.remove_button)
        layout.addLayout(header)

        range_row = QHBoxLayout()
        range_row.setContentsMargins(0, 0, 0, 0)
        range_row.setSpacing(6)

        self.min_edit = QLineEdit()
        self.min_edit.setValidator(QDoubleValidator())
        self.min_edit.setFixedWidth(58)
        self.min_edit.setAlignment(Qt.AlignCenter)
        range_row.addWidget(self.min_edit)

        self.range_slider = QRangeSlider(Qt.Orientation.Horizontal)
        range_row.addWidget(self.range_slider, 1)

        self.max_edit = QLineEdit()
        self.max_edit.setValidator(QDoubleValidator())
        self.max_edit.setFixedWidth(58)
        self.max_edit.setAlignment(Qt.AlignCenter)
        range_row.addWidget(self.max_edit)

        self.unit_label = QLabel()
        range_row.addWidget(self.unit_label)
        self._refresh_unit()
        layout.addLayout(range_row)

        self.stat_label = QLabel("")
        self.stat_label.setObjectName("mappingFilterStat")
        layout.addWidget(self.stat_label)

        low, high = metric_fallback_range(self.entry['metric'])
        self.set_bounds(
            min(low, self.entry['min']), max(high, self.entry['max'])
        )

        self.setEditable(editable)

        self.enabled_check.toggled.connect(self._on_enabled_toggled)
        self.metric_combobox.currentTextChanged.connect(
            self._on_metric_selected
        )
        self.mode_combobox.currentIndexChanged.connect(self._on_mode_changed)
        self.range_slider.valueChanged.connect(self._on_slider_changed)
        self.range_slider.sliderReleased.connect(self._commit)
        self.min_edit.editingFinished.connect(self._on_edits_changed)
        self.max_edit.editingFinished.connect(self._on_edits_changed)
        self.remove_button.clicked.connect(
            lambda: self.removeRequested.emit(self.entry['id'])
        )

        # Dragging a handle repaints the maps and the histogram, so the edit
        # is only published once the value has settled.
        self._commit_timer = QTimer(self)
        self._commit_timer.setSingleShot(True)
        self._commit_timer.setInterval(150)
        self._commit_timer.timeout.connect(self._commit)

    # -- state -----------------------------------------------------------
    @property
    def filter_id(self):
        """Return the id of the criterion this card edits."""
        return self.entry['id']

    def _refresh_metric_tooltip(self):
        """Say on hover which harmonic the metric is measured on."""
        tooltip = f"{_METRIC_TOOLTIP}\nMeasured on harmonic {self.entry['harmonic']}."
        self.metric_combobox.setToolTip(tooltip)
        self.metric_label.setToolTip(
            f"Measured on harmonic {self.entry['harmonic']}."
        )

    def _refresh_unit(self):
        """Show the current metric's unit after the range, if it has one."""
        unit = metric_unit(self.entry['metric'])
        self.unit_label.setText(unit)
        self.unit_label.setVisible(bool(unit))

    def setEditable(self, editable):
        """Show the card read-only, keeping only enable and remove usable.

        Used for criteria another tab owns: they are still listed (and can
        still be switched off or deleted from here) but their range is edited
        where it was defined.
        """
        self._editable = bool(editable)
        for widget in (
            self.min_edit,
            self.max_edit,
            self.range_slider,
            self.mode_combobox,
            self.metric_combobox,
        ):
            widget.setEnabled(self._editable)
        self._refresh_muted()

    def _refresh_muted(self):
        """Grey the card while it is read-only or switched off.

        A criterion that is off hides nothing, so it must not look like the
        ones that do.
        """
        muted = not self._editable or not self.entry['enabled']
        if self.property("muted") == muted:
            return
        self.setProperty("muted", muted)
        self.style().unpolish(self)
        self.style().polish(self)

    def apply_entry(self, entry):
        """Show *entry* on this card without publishing it back."""
        self.entry = dict(entry)
        self._updating = True
        try:
            self.metric_combobox.blockSignals(True)
            self.metric_combobox.setCurrentText(self.entry['metric'])
            self.metric_combobox.blockSignals(False)
            self.metric_label.setText(self.entry['metric'])
        finally:
            self._updating = False
        self._refresh_unit()
        self._refresh_metric_tooltip()
        self.set_bounds(self.entry['min'], self.entry['max'])

    def set_bounds(self, low, high):
        """Fit the slider to ``[low, high]``, widened to the current range."""
        low = min(float(low), self.entry['min'])
        high = max(float(high), self.entry['max'])
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = metric_fallback_range(self.entry['metric'])
        if high <= low:
            high = low + 1.0
        self._updating = True
        try:
            self.range_slider.setRange(
                int(np.floor(low * self.scale)),
                int(np.ceil(high * self.scale)),
            )
            self.range_slider.setValue(
                (
                    int(round(self.entry['min'] * self.scale)),
                    int(round(self.entry['max'] * self.scale)),
                )
            )
            self._refresh_edits()
        finally:
            self._updating = False

    def set_stat(self, text):
        """Set the small line under the range showing what the filter keeps."""
        self.stat_label.setText(text)

    def set_enable_blocked(self, reason):
        """Forbid switching the filter *on* while *reason* is set.

        Switching an active filter off stays possible, so a filter can never
        be stuck hiding pixels.
        """
        blocked = bool(reason) and not self.entry['enabled']
        self.enabled_check.setEnabled(not blocked)
        self.enabled_check.setToolTip(reason if blocked else _ENABLE_TOOLTIP)

    def _refresh_edits(self):
        """Rewrite the two number boxes from the criterion's current range."""
        self.min_edit.setText(f"{self.entry['min']:.2f}")
        self.max_edit.setText(f"{self.entry['max']:.2f}")

    # -- signals ---------------------------------------------------------
    def _on_enabled_toggled(self, checked):
        """Record the enable switch and publish the change immediately."""
        self.entry['enabled'] = bool(checked)
        self._refresh_muted()
        self._commit()

    def _on_metric_selected(self, metric):
        """Ask the list to re-seed this criterion for *metric*."""
        if self._updating or not metric or metric == self.entry['metric']:
            return
        self.metricChangeRequested.emit(self.entry['id'], metric)

    def _on_mode_changed(self, index):
        """Record keep/exclude and publish the change immediately."""
        self.entry['mode'] = EXCLUDE if index == 1 else KEEP
        self._commit()

    def _on_slider_changed(self, value):
        """Follow a dragged handle, deferring the expensive redraw."""
        if self._updating:
            return
        low, high = value
        self.entry['min'] = low / self.scale
        self.entry['max'] = high / self.scale
        self._refresh_edits()
        self._commit_timer.start()

    def _on_edits_changed(self):
        """Read both number boxes, then move the slider to match."""
        if self._updating:
            return
        try:
            low = float(self.min_edit.text())
            high = float(self.max_edit.text())
        except ValueError:
            self._refresh_edits()
            return
        if low > high:
            low, high = high, low
        self.entry['min'] = low
        self.entry['max'] = high
        self.set_bounds(low, high)
        self._commit()

    def _commit(self):
        """Publish this card's edit to the owning list."""
        self._commit_timer.stop()
        self.changed.emit(self.entry['id'])


class MappingFilterList(QWidget):
    """The filter stack, shown as a list of cards above an "add" button.

    The widget owns no data of its own: it renders whatever stack it is given
    and emits :attr:`filtersChanged` with the edited stack, so the tab that
    owns the layer stays the single writer.

    With ``single=True`` the list holds exactly one criterion on the first
    metric -- the Fret tab's efficiency filter. There is nothing to add or
    remove: the card is always shown and its check box turns it on and off.
    Until the user touches it, that card is a placeholder and the stack it
    reports is empty.
    """

    filtersChanged = Signal(list)
    """Emitted with the full, edited stack whenever the user changes it."""

    def __init__(
        self, metrics, *, parent=None, add_label="Add filter", single=False
    ):
        super().__init__(parent)
        self._metrics = list(metrics)
        self._single = bool(single)
        self._placeholder = False
        self._filters = []
        self._cards = {}
        self._bounds = {}
        self._rebuilding = False
        self._params_provider = None
        self._harmonic_provider = None
        self._bounds_provider = None
        self._enable_blocked_reason = None
        self._current_metric = self._metrics[0] if self._metrics else None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(4)
        layout.addWidget(self._cards_container)

        self.empty_label = QLabel(
            "No filters yet. Every filter you add is listed here and can be "
            "switched off or removed."
        )
        self.empty_label.setWordWrap(True)
        self.empty_label.setObjectName("mappingFilterForeign")
        layout.addWidget(self.empty_label)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("mappingFilterStat")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.add_button = QPushButton(f"+ {add_label}")
        self.add_button.setToolTip(_ADD_TOOLTIP)
        self.add_button.clicked.connect(self._on_add_clicked)
        layout.addWidget(self.add_button)

        if self._single:
            self.add_button.setVisible(False)
            self.summary_label.setVisible(False)
            self._set_placeholder()
            self._rebuild_cards()
        self._refresh_chrome()

    # -- public API ------------------------------------------------------
    def filters(self):
        """Return a copy of the stack currently displayed."""
        if self._placeholder:
            return []
        return [dict(f) for f in self._filters]

    def set_filters(self, filters):
        """Rebuild the cards from *filters* without emitting a change.

        A stack equal to the one on screen is ignored. Applying a filter
        writes the stack back to the layer and the tab re-reads it from
        there, so without this the card whose slider is being dragged would
        be destroyed and rebuilt under the pointer on every tick.
        """
        normalized = normalize_filters(filters)
        if self._single:
            normalized = normalized[:1]
            if not normalized and self._placeholder:
                return
        if normalized == self._filters and not self._placeholder:
            return
        self._rebuilding = True
        try:
            self._placeholder = False
            self._filters = normalized
            if self._single and not self._filters:
                self._set_placeholder()
            self._rebuild_cards()
        finally:
            self._rebuilding = False

    def set_metric_bounds(self, metric, low, high):
        """Record the data range of *metric* and widen its cards' sliders."""
        if low is None or high is None:
            return
        low = float(low)
        high = float(high)
        if not np.isfinite(low) or not np.isfinite(high):
            return
        self._bounds[metric] = (low, high)
        for entry, card in zip(
            self._filters, self._ordered_cards(), strict=True
        ):
            if entry['metric'] == metric:
                card.set_bounds(low, high)

    def bounds_for(self, metric):
        """Return the recorded data range of *metric*, or its fallback."""
        return self._bounds.get(metric, metric_fallback_range(metric))

    def set_filter_stats(self, stats, summary="", detail=""):
        """Show per-card kept fractions and the stack's overall summary.

        *detail* is added above the criteria list in the summary's tooltip,
        which is where anything too long for one line belongs -- the dock is
        narrow and the figure is what the line exists to show.
        """
        for entry, card in zip(
            self._filters, self._ordered_cards(), strict=True
        ):
            text = stats.get(entry['id'], "")
            card.set_stat(text)
        if summary:
            self.summary_label.setText(summary)
        tooltip = describe_filters(self.filters())
        self.summary_label.setToolTip(
            f"{detail}\n{tooltip}" if detail else tooltip
        )

    def current_metric(self):
        """Return the metric a new filter starts on."""
        return self._current_metric

    def set_current_metric(self, metric):
        """Start new filters on *metric* if it is one of the offered ones."""
        if metric in self._metrics:
            self._current_metric = metric

    def set_editable_metrics(self, metrics):
        """Restrict full editing to *metrics*; others are shown read-only."""
        self._editable_metrics = set(metrics)
        for entry, card in zip(
            self._filters, self._ordered_cards(), strict=True
        ):
            card.setEditable(entry['metric'] in self._editable_metrics)

    def set_params_provider(self, provider):
        """Set the callable supplying a new criterion's frozen parameters.

        Called as ``provider(metric)`` and expected to return a dict. It is
        how a criterion captures the state it must be reproducible from --
        the excitation frequency, or the whole FRET donor trajectory.
        """
        self._params_provider = provider

    def set_harmonic_provider(self, provider):
        """Set the callable returning the harmonic a new criterion applies to."""
        self._harmonic_provider = provider

    def set_bounds_provider(self, provider):
        """Set the callable measuring a metric's data range.

        Called as ``provider(metric)``; returns ``(low, high)`` or ``None``.
        Used whenever a criterion is (re)seeded on a metric, so it starts out
        spanning the data rather than a generic default.
        """
        self._bounds_provider = provider

    def set_enable_blocked(self, reason):
        """Forbid switching a filter on while *reason* is set (``None`` clears)."""
        self._enable_blocked_reason = reason
        for card in self._cards.values():
            card.set_enable_blocked(reason)

    def add_filter(self, entry):
        """Append *entry* to the stack and publish the new stack."""
        coerced = _coerce_filter(entry)
        if coerced is None:
            return None
        self._filters.append(coerced)
        self._rebuild_cards()
        self._emit()
        return coerced

    # -- internals -------------------------------------------------------
    def _ordered_cards(self):
        """Return the cards in stack order."""
        return [self._cards[f['id']] for f in self._filters]

    def _seed_range(self, metric):
        """Return the range a criterion on *metric* should start with."""
        if self._bounds_provider is not None:
            measured = self._bounds_provider(metric)
            if measured is not None:
                low, high = measured
                self._bounds[metric] = (float(low), float(high))
                return float(low), float(high)
        return self.bounds_for(metric)

    def _seed_params(self, metric):
        """Return the frozen parameters for a criterion on *metric*."""
        if self._params_provider is None:
            return None
        return self._params_provider(metric)

    def _new_entry(self, metric, **kwargs):
        """Return a fresh criterion on *metric* spanning its data."""
        low, high = self._seed_range(metric)
        harmonic = (
            self._harmonic_provider()
            if self._harmonic_provider is not None
            else 1
        )
        return new_filter(
            metric,
            low,
            high,
            harmonic or 1,
            params=self._seed_params(metric),
            **kwargs,
        )

    def _set_placeholder(self):
        """Show the single-mode card, switched off, until it is used."""
        self._filters = [self._new_entry(self._metrics[0], enabled=False)]
        self._placeholder = True

    def _rebuild_cards(self):
        """Recreate every card so the list matches the stack exactly."""
        editable_metrics = getattr(self, '_editable_metrics', None)
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._cards = {}
        for entry in self._filters:
            editable = (
                True
                if editable_metrics is None
                else entry['metric'] in editable_metrics
            )
            card = _FilterCard(
                entry,
                metrics=self._metrics if editable else None,
                editable=editable,
                removable=not self._single,
            )
            low, high = self.bounds_for(entry['metric'])
            card.set_bounds(low, high)
            card.set_enable_blocked(self._enable_blocked_reason)
            card.changed.connect(self._on_card_changed)
            card.removeRequested.connect(self._on_card_removed)
            card.metricChangeRequested.connect(self._on_metric_change)
            self._cards_layout.addWidget(card)
            self._cards[entry['id']] = card
        self._refresh_chrome()

    def _refresh_chrome(self):
        """Update the placeholder text and the summary line."""
        has_any = bool(self._filters)
        self.empty_label.setVisible(not has_any and not self._single)
        if not has_any:
            self.summary_label.setText("")

    def _on_add_clicked(self):
        """Add a filter on the displayed metric, spanning its whole data."""
        metric = self.current_metric()
        if metric is None:
            return
        self.add_filter(self._new_entry(metric))

    def _on_metric_change(self, filter_id, metric):
        """Re-seed one criterion for a newly chosen metric and publish it.

        The old range means nothing in the new metric's units, so the range
        restarts at the new metric's full data span.
        """
        card = self._cards.get(filter_id)
        if card is None:
            return
        low, high = self._seed_range(metric)
        entry = dict(
            card.entry,
            metric=metric,
            min=low,
            max=high,
            params=dict(self._seed_params(metric) or {}),
        )
        card.apply_entry(entry)
        self._replace(entry)
        self._emit()

    def _replace(self, entry):
        """Swap the stored copy of *entry* for the edited one."""
        for index, current in enumerate(self._filters):
            if current['id'] == entry['id']:
                self._filters[index] = dict(entry)
                return

    def _on_card_changed(self, filter_id):
        """Copy one card's edited values back into the stack."""
        card = self._cards.get(filter_id)
        if card is None:
            return
        # A criterion created before its parameters existed (the Fret tab's
        # card, shown before a donor lifetime is entered) picks them up on
        # its first edit. Filling them in here, rather than in the tab, keeps
        # the stack the tab writes back identical to the one on screen, so
        # the card being edited is not rebuilt under the pointer.
        if not card.entry['params']:
            card.entry['params'] = dict(
                self._seed_params(card.entry['metric']) or {}
            )
        self._replace(card.entry)
        self._emit()

    def _on_card_removed(self, filter_id):
        """Delete one criterion and publish the shortened stack."""
        self._filters = [f for f in self._filters if f['id'] != filter_id]
        self._rebuild_cards()
        self._emit()

    def _emit(self):
        """Publish the stack, unless the list is being rebuilt from outside."""
        # Any user edit turns the single-mode placeholder into a real filter.
        self._placeholder = False
        self._refresh_chrome()
        if not self._rebuilding:
            self.filtersChanged.emit(self.filters())
