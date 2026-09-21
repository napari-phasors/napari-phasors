"""Metric filters that invalidate phasor coordinates outside a value range.

The Filter tab restricts an image by *intensity*. The filters defined here
restrict it by any quantity **derived** from the phasor coordinates -- an
apparent lifetime, the phase or modulation, a FRET efficiency, the fraction
of a component -- by setting the phasor coordinates of every pixel outside
the requested range to NaN.

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
what the data shows and what the user sees listed cannot drift apart.
:class:`ComponentFilterList` is the same idea for the Components tab, where
the criteria are not added one by one but follow the components themselves.
Both are kept in this module so the halves are edited together, but every
function above them is Qt-free and safe to call from a worker thread.
"""

import functools
import uuid

import numpy as np
from phasorpy.component import phasor_component_fit, phasor_component_fraction
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
COMPONENT_FRACTION = "Component fraction"

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
ALL_METRICS = MAPPING_METRICS + (FRET_EFFICIENCY, COMPONENT_FRACTION)

#: Unit shown after a criterion's range, per metric ("" when dimensionless).
METRIC_UNITS = {
    APPARENT_PHASE_LIFETIME: "ns",
    APPARENT_MODULATION_LIFETIME: "ns",
    NORMAL_LIFETIME: "ns",
    PHASE: "rad",
    MODULATION: "",
    FRET_EFFICIENCY: "",
    COMPONENT_FRACTION: "",
}

#: Range used for a criterion's slider before any data has been measured.
METRIC_FALLBACK_RANGE = {
    APPARENT_PHASE_LIFETIME: (0.0, 10.0),
    APPARENT_MODULATION_LIFETIME: (0.0, 10.0),
    NORMAL_LIFETIME: (0.0, 10.0),
    PHASE: (0.0, float(2.0 * np.pi)),
    MODULATION: (0.0, 1.0),
    FRET_EFFICIENCY: (0.0, 1.0),
    COMPONENT_FRACTION: (0.0, 1.0),
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


def filter_display_name(entry):
    """Return the name a criterion is listed under.

    Every criterion but a component fraction is fully described by its
    metric. A component fraction is not: three of them on one layer would
    otherwise all read "Component fraction", so the component's own name is
    used instead and the quantity is left to the tooltip.
    """
    if entry.get('metric') != COMPONENT_FRACTION:
        return entry.get('metric', "")
    name = (entry.get('params') or {}).get('component_name')
    if name:
        return str(name)
    index = (entry.get('params') or {}).get('component_index')
    if index is None:
        return COMPONENT_FRACTION
    return f"Component {int(index) + 1}"


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


class MetricContext:
    """The arrays a metric may need beyond the single harmonic plane.

    Most metrics are a function of one ``(real, imag)`` plane alone. A
    component fraction is not: a component fit consumes the mean image and,
    above two components, several harmonics at once. Rather than widen
    :func:`compute_metric` with four more arguments, the caller hands it this
    little bundle of the baseline arrays.

    It doubles as a per-call memo. One component fit yields *every*
    component's fraction at once, so filtering a four-component image on
    three of its fractions has to fit it once, not three times.
    """

    def __init__(self, mean=None, real=None, imag=None, harmonics=None):
        self.mean = mean
        self.real = real
        self.imag = imag
        self.harmonics = harmonics
        self._fractions = {}

    def fractions(self, key, compute):
        """Return the cached fraction maps for *key*, computing them once."""
        if key not in self._fractions:
            self._fractions[key] = compute()
        return self._fractions[key]


def _hashable(value):
    """Return *value* as something usable in a dict key."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    return value


def has_component_positions(params):
    """Return whether *params* carries usable component positions.

    Stored settings come back from an OME-TIFF as arrays rather than lists,
    and a bare truth test on an array raises, so emptiness is measured
    rather than asked -- and measuring it can raise in its own turn on a
    hand-edited, ragged value, which is no better than empty.
    """
    if not params:
        return False
    real = params.get('component_real')
    imag = params.get('component_imag')
    if real is None or imag is None:
        return False
    try:
        return bool(np.size(real)) and bool(np.size(imag))
    except (TypeError, ValueError):
        # Ragged, or not array-like at all: not positions a fit can use.
        return False


def _component_fit_planes(params, context):
    """Return ``(mean, real, imag)`` sliced to the harmonics a fit needs.

    ``None`` whenever the layer cannot supply them -- a two-harmonic fit
    stored on a layer that only carries the first harmonic, say -- so the
    criterion is skipped instead of blanking the image.
    """
    if context is None or context.mean is None:
        return None
    mean, real, imag = context.mean, context.real, context.imag
    if real is None or imag is None:
        return None
    wanted = params.get('harmonics') or [params.get('harmonic', 1)]
    if len(wanted) > 1 and real.ndim <= mean.ndim:
        return None
    planes_real = []
    planes_imag = []
    for value in wanted:
        try:
            harmonic = int(value)
        except (TypeError, ValueError):
            return None
        plane_real, plane_imag = select_harmonic(
            real, imag, context.harmonics, harmonic, mean.ndim
        )
        if plane_real is None or plane_real.shape != mean.shape:
            return None
        planes_real.append(plane_real)
        planes_imag.append(plane_imag)
    if len(planes_real) == 1:
        return mean, planes_real[0], planes_imag[0]
    return mean, np.stack(planes_real), np.stack(planes_imag)


def _compute_component_fraction(real, imag, params, context):
    """Return the fraction map of one component, or ``None``.

    The component positions are read from the criterion's own parameters
    rather than from the Components tab, so a filter keeps meaning the same
    thing after the components on screen have been moved -- the same
    contract the FRET efficiency filter has with its donor trajectory.
    """
    if not has_component_positions(params):
        return None
    component_real = params['component_real']
    component_imag = params['component_imag']
    try:
        index = int(params.get('component_index', 0))
    except (TypeError, ValueError):
        return None

    if params.get('analysis_type') == 'Linear Projection':
        if real is None or imag is None or index not in (0, 1):
            return None
        with np.errstate(divide='ignore', invalid='ignore'):
            fraction = np.asarray(
                phasor_component_fraction(
                    real, imag, component_real, component_imag
                ),
                dtype=float,
            )
        # The second component of a projection has no fit of its own: what is
        # left of the pixel once the first component is accounted for is, by
        # definition, the second.
        return fraction if index == 0 else 1.0 - fraction

    planes = _component_fit_planes(params, context)
    if planes is None:
        return None
    key = (
        'fit',
        _hashable(component_real),
        _hashable(component_imag),
        _hashable(params.get('harmonics')),
    )
    fractions = context.fractions(
        key,
        lambda: _safe_component_fit(*planes, component_real, component_imag),
    )
    if fractions is None or index >= len(fractions):
        return None
    return np.asarray(fractions[index], dtype=float)


def _safe_component_fit(mean, real, imag, component_real, component_imag):
    """Return every component's fraction map, or ``None`` if the fit fails.

    A stored criterion can outlive the data it was made for (fewer harmonics
    after a re-read, a component count the arrays no longer support), and a
    filter that cannot be evaluated must be skipped, not raised through the
    redraw that triggered it.
    """
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            fractions = phasor_component_fit(
                mean, real, imag, component_real, component_imag
            )
    except Exception:  # noqa: BLE001 - see docstring
        return None
    if isinstance(fractions, np.ndarray) and fractions.shape == mean.shape:
        return [fractions]
    return list(fractions)


def compute_metric(
    metric,
    real,
    imag,
    *,
    harmonic=1,
    frequency=None,
    wrap_phase=False,
    params=None,
    context=None,
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
        trajectory from here, ``Component fraction`` its component
        positions.
    context : MetricContext, optional
        The baseline arrays a metric may need beyond one harmonic plane.
        Required by ``Component fraction`` above a linear projection.

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

    if metric == COMPONENT_FRACTION:
        return _compute_component_fraction(real, imag, params, context)

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


def combined_mask(
    filters, mean, real, imag, harmonics, *, on_error=None, context=None
):
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
    context : MetricContext, optional
        Context to evaluate the criteria against, so that a caller making
        several calls over the same arrays -- measuring what each criterion
        keeps, say -- pays for one component fit rather than one per call.
    """
    if mean is None:
        return None
    if context is None:
        context = MetricContext(mean, real, imag, harmonics)
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
            context=context,
        )
        if values is None or values.shape != mean.shape:
            if on_error is not None:
                on_error(
                    f"Skipped the {filter_display_name(entry)} filter: it "
                    "cannot be evaluated on this layer."
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
    filters, mean, real, imag, harmonics, *, on_error=None, context=None
):
    """Return ``(mean, real, imag, mask)`` with the stack applied.

    The arrays are treated as the baseline; nothing already NaN is restored.
    """
    mask = combined_mask(
        filters,
        mean,
        real,
        imag,
        harmonics,
        on_error=on_error,
        context=context,
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
    layer,
    filters=None,
    *,
    filter_params=None,
    on_error=None,
    arrays=None,
    context=None,
):
    """Rewrite *layer*'s phasor arrays from its baseline plus *filters*.

    Returns the "dropped pixel" mask, or ``None`` when nothing was filtered.

    *arrays* and *context* are the baseline and the measuring context a
    caller has already derived for this layer. Re-deriving the baseline
    re-runs the median filter over the whole image, so a caller that holds
    one -- the tab that is about to measure the very same arrays again for
    its cards and its labels layers -- hands it over rather than paying for
    it twice.
    """
    if filters is None:
        filters = get_filters(layer)
    borrowed = arrays is not None
    mean, real, imag = (
        arrays if borrowed else baseline_arrays(layer, filter_params)
    )
    if mean is None:
        return None
    mean, real, imag, mask = apply_filters_to_arrays(
        filters,
        mean,
        real,
        imag,
        layer.metadata.get('harmonics'),
        on_error=on_error,
        context=context,
    )
    # With nothing to mask the arrays come back untouched, so a borrowed
    # baseline would be handed to the layer itself and the caller's copy
    # would then follow every later edit of it.
    if borrowed and mask is None:
        mean = None if mean is None else mean.copy()
        real = None if real is None else real.copy()
        imag = None if imag is None else imag.copy()
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


def serialize_filter_applies(method):
    """Never run a tab's filter apply inside another of its own.

    Applying a stack re-derives the phasor arrays, re-runs the tab's
    analysis and adds or removes napari layers, all of which spin the Qt
    event loop. The user's next edit -- a slider they released, a number
    they typed -- is then delivered *inside* the apply it interrupted, and
    the two rebuild the same card lists over each other, leaving a list
    whose criteria and cards disagree.

    So a nested call is not run: it is remembered and replayed once the
    outer one has finished, which is both correct (the last edit wins) and
    cheaper (one rebuild instead of two interleaved). Replay is a loop, not
    recursion -- the flag is already cleared -- and each pass consumes one
    pending stack, so it ends as soon as the user stops editing.
    """

    @functools.wraps(method)
    def apply_filters(self, filters=None, layers=None):
        if getattr(self, '_filter_apply_busy', False):
            self._pending_filter_apply = (filters, layers)
            return
        self._filter_apply_busy = True
        try:
            method(self, filters, layers)
        finally:
            self._filter_apply_busy = False
        pending = getattr(self, '_pending_filter_apply', None)
        if pending is not None:
            self._pending_filter_apply = None
            apply_filters(self, *pending)

    return apply_filters


def describe_filters(filters):
    """Return a one-line summary of the enabled criteria in *filters*."""
    enabled = [f for f in filters if f['enabled']]
    if not enabled:
        return "No filters applied."
    parts = [
        f"{filter_display_name(f)} "
        f"{format_range(f['metric'], f['min'], f['max'], f['mode'])}"
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


class FilterCard(QFrame):
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
        #: The criterion as last published, so an edit that lands back on it
        #: does not rebuild everything downstream for no change.
        self._published = None
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

        self.metric_label = QLabel(filter_display_name(self.entry))
        self._accent_color = None
        self._refresh_metric_label_style()
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
        """Say on hover what is measured, and on which harmonic."""
        tooltip = f"{_METRIC_TOOLTIP}\nMeasured on harmonic {self.entry['harmonic']}."
        self.metric_combobox.setToolTip(tooltip)
        measured = f"Measured on harmonic {self.entry['harmonic']}."
        # A card named after a component says nowhere else that the number it
        # tests is a fraction.
        if filter_display_name(self.entry) != self.entry['metric']:
            measured = f"{self.entry['metric']}. {measured}"
        self.metric_label.setToolTip(measured)

    def _refresh_metric_label_style(self):
        """Draw the card's title, tinted with its accent colour if it has one."""
        style = "font-weight: 600;"
        if self._accent_color:
            style += f" color: {self._accent_color};"
        self.metric_label.setStyleSheet(style)

    def set_accent_color(self, color):
        """Tint the card's title, or clear the tint with ``None``.

        Used by the Components tab so a fraction filter is recognisable as
        belonging to the component drawn in that colour on the phasor plot.
        """
        self._accent_color = color or None
        self._refresh_metric_label_style()

    @property
    def accent_color(self):
        """Return the colour this card is tinted with, if any."""
        return self._accent_color

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
        # Shown, not published: the next edit publishes whatever it makes of
        # this, even if that is the value the card last sent itself.
        self._published = None
        self._updating = True
        try:
            self.metric_combobox.blockSignals(True)
            self.metric_combobox.setCurrentText(self.entry['metric'])
            self.metric_combobox.blockSignals(False)
            self.metric_label.setText(filter_display_name(self.entry))
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
        """Publish this card's edit to the owning list, if it is a change.

        A handle dragged with a pause in it is published twice -- once by
        the timer, once again when it is released -- and each publication
        re-derives the phasor arrays of every selected image. The second one
        cannot change anything the first did not, so it is dropped.
        """
        self._commit_timer.stop()
        published = dict(self.entry)
        if published == self._published:
            return
        self._published = published
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
        for entry, card in self._card_pairs():
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
        for entry, card in self._card_pairs():
            card.set_stat(stats.get(entry['id'], ""))
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
        for entry, card in self._card_pairs():
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
    def _card_pairs(self):
        """Return ``(criterion, card)`` for every criterion that has a card.

        A criterion with no card means the list is part-way through a
        rebuild -- a refresh that arrived while another was in flight, which
        applying a stack makes possible because it pumps the event loop. The
        rebuild ends with a refresh of its own, so the missing card is
        skipped rather than raised out of the redraw that asked for it.
        """
        return [
            (entry, self._cards[entry['id']])
            for entry in self._filters
            if entry['id'] in self._cards
        ]

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
            card = FilterCard(
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


_COMPONENT_LIST_EMPTY = (
    "Define at least two components and run an analysis to filter on their "
    "fractions."
)
_COMPONENT_ENABLE_TOOLTIP = (
    "Keep only the pixels whose fraction of this component falls inside the "
    "range. They lose their phasor coordinates everywhere else, exactly as "
    "an intensity threshold would."
)


class ComponentFilterList(QWidget):
    """One fraction-range card per component, feeding the same filter stack.

    Unlike :class:`MappingFilterList` there is nothing to add or to remove:
    the components are given, and each one gets exactly one card. A card the
    user has never touched is a *placeholder* -- switched off, and left out
    of the stack this widget reports -- so merely defining components never
    writes a filter onto a layer.

    The widget owns no data of its own either: it renders the components and
    criteria it is given and emits :attr:`filtersChanged` with the edited
    component criteria, leaving the tab that owns the layer the single
    writer.
    """

    filtersChanged = Signal(list)
    """Emitted with this widget's criteria whenever the user changes one."""

    def __init__(self, parent=None):
        super().__init__(parent)
        #: ``[(component_index, display_name, colour)]``, in card order.
        self._components = []
        #: ``{component_index: criterion}`` for every card on screen.
        self._entries = {}
        #: ``{component_index: (low, high)}`` measured data range.
        self._bounds = {}
        #: Indices whose card is still an untouched placeholder.
        self._placeholders = set()
        self._cards = {}
        self._params_provider = None
        self._harmonic_provider = None
        self._bounds_provider = None
        self._enable_blocked_reason = None
        self._rebuilding = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(4)
        layout.addWidget(self._cards_container)

        self.empty_label = QLabel(_COMPONENT_LIST_EMPTY)
        self.empty_label.setWordWrap(True)
        self.empty_label.setObjectName("mappingFilterForeign")
        layout.addWidget(self.empty_label)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("mappingFilterStat")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self._refresh_chrome()

    # -- public API ------------------------------------------------------
    def components(self):
        """Return the components currently carrying a card."""
        return list(self._components)

    def set_components(self, components):
        """Show one card per entry of *components*.

        Parameters
        ----------
        components : sequence of tuple
            ``(component_index, display_name, colour)``. A component that
            already has a criterion keeps it, renamed to the name given here
            so the card follows a rename immediately.
        """
        normalized = [
            (int(index), str(name), color) for index, name, color in components
        ]
        # Rebuilding the cards destroys the one the user may be dragging a
        # handle on, and this is called on every redraw, so an unchanged set
        # of components must leave the list exactly as it is.
        if normalized == self._components and all(
            index in self._entries for index, _name, _color in normalized
        ):
            return
        indices = {index for index, _name, _color in normalized}
        # A component that is gone takes its criterion with it; leaving it in
        # the stack would hide pixels with no card left to explain them.
        for index in list(self._entries):
            if index not in indices:
                del self._entries[index]
                self._placeholders.discard(index)
        self._components = normalized
        for index, name, _color in normalized:
            entry = self._entries.get(index)
            if entry is None:
                self._entries[index] = self._new_entry(index, name)
                self._placeholders.add(index)
            else:
                entry['params'] = dict(entry['params'], component_name=name)
        self._rebuild_cards()

    def set_filters(self, filters):
        """Adopt the stored component criteria without emitting a change.

        A stack equal to the one on screen is ignored, so the card whose
        slider is being dragged is not rebuilt under the pointer every time
        the edit is written back to the layer and read out again.
        """
        stored = {}
        for entry in normalize_filters(filters):
            if entry['metric'] != COMPONENT_FRACTION:
                continue
            index = entry['params'].get('component_index')
            if index is None:
                continue
            stored[int(index)] = entry
        # A criterion is compared and adopted under the component's *current*
        # name: a rename that has not reached the layer yet must show on the
        # card, and must not make the stack look changed on every sync.
        stored = {
            index: self._with_name(entry, index)
            for index, entry in stored.items()
        }
        if stored == {
            index: entry
            for index, entry in self._entries.items()
            if index not in self._placeholders
        }:
            return
        self._rebuilding = True
        try:
            for index, _name, _color in self._components:
                entry = stored.get(index)
                if entry is None:
                    if index not in self._placeholders:
                        name = self._name_for(index)
                        self._entries[index] = self._new_entry(index, name)
                        self._placeholders.add(index)
                else:
                    self._entries[index] = entry
                    self._placeholders.discard(index)
            self._rebuild_cards()
        finally:
            self._rebuilding = False

    def filters(self):
        """Return the criteria the user has actually defined, in card order."""
        return [
            dict(self._entries[index])
            for index, _name, _color in self._components
            if index in self._entries and index not in self._placeholders
        ]

    def set_filter_stats(self, stats, summary="", detail=""):
        """Show per-card kept fractions and the whole list's summary line."""
        for index, card in self._cards.items():
            entry = self._entries.get(index)
            card.set_stat(stats.get(entry['id'], "") if entry else "")
        self.summary_label.setText(summary)
        tooltip = describe_filters(self.filters())
        self.summary_label.setToolTip(
            f"{detail}\n{tooltip}" if detail else tooltip
        )

    def set_enable_blocked(self, reason):
        """Forbid switching a filter on while *reason* is set (``None`` clears)."""
        self._enable_blocked_reason = reason
        for card in self._cards.values():
            card.set_enable_blocked(reason)

    def set_params_provider(self, provider):
        """Set the callable freezing a component's definition into a criterion.

        Called as ``provider(component_index)`` and expected to return a
        dict: the analysis type, the component positions and the harmonics
        they were taken on. It is what lets a criterion keep meaning the same
        thing after the components on screen have moved.
        """
        self._params_provider = provider

    def set_harmonic_provider(self, provider):
        """Set the callable returning the harmonic a new criterion applies to."""
        self._harmonic_provider = provider

    def set_bounds_provider(self, provider):
        """Set the callable measuring a component's fraction range.

        Called as ``provider(component_index)``; returns ``(low, high)`` or
        ``None``. A linear projection's fractions sit in ``[0, 1]``, but a
        component fit solves a system that is not constrained that way and
        routinely produces fractions below 0 or above 1. Measuring the range
        rather than assuming it is what lets those pixels be filtered on at
        all.
        """
        self._bounds_provider = provider

    def bounds_for(self, index):
        """Return the recorded fraction range of a component, or a fallback."""
        return self._bounds.get(
            index, metric_fallback_range(COMPONENT_FRACTION)
        )

    def set_component_bounds(self, index, low, high):
        """Record a component's fraction range and widen its card's slider."""
        if low is None or high is None:
            return
        low = float(low)
        high = float(high)
        if not np.isfinite(low) or not np.isfinite(high):
            return
        self._bounds[index] = (low, high)
        card = self._cards.get(index)
        if card is not None:
            card.set_bounds(low, high)

    def refresh_params(self):
        """Re-freeze every criterion's parameters from the current components.

        Returns ``True`` when anything changed. A fraction the user can see
        on screen and a fraction a filter tests must be the same number, so
        moving a component has to move its filter with it.
        """
        changed = False
        for index, entry in self._entries.items():
            params = self._seed_params(index)
            # Positions that cannot be read right now are not an update:
            # overwriting a working criterion with them would leave it
            # hiding pixels by a rule it can no longer evaluate.
            if not has_component_positions(params):
                continue
            if entry['params'] == params:
                continue
            entry['params'] = params
            changed = True
        if changed:
            self._rebuild_cards()
        return changed

    # -- internals -------------------------------------------------------
    def _with_name(self, entry, index):
        """Return *entry* named after the component at *index*."""
        return dict(
            entry,
            params=dict(entry['params'], component_name=self._name_for(index)),
        )

    def _name_for(self, index):
        """Return the display name of the component at *index*."""
        for candidate, name, _color in self._components:
            if candidate == index:
                return name
        return f"Component {index + 1}"

    def _color_for(self, index):
        """Return the colour the component at *index* is drawn in."""
        for candidate, _name, color in self._components:
            if candidate == index:
                return color
        return None

    def _seed_range(self, index):
        """Return the range a criterion on this component should start with."""
        if self._bounds_provider is not None:
            measured = self._bounds_provider(index)
            if measured is not None:
                low, high = float(measured[0]), float(measured[1])
                if np.isfinite(low) and np.isfinite(high):
                    self._bounds[index] = (low, high)
                    return low, high
        return self.bounds_for(index)

    def _seed_params(self, index):
        """Return the frozen parameters of the component at *index*."""
        if self._params_provider is None:
            return {}
        return dict(self._params_provider(index) or {})

    def _new_entry(self, index, name):
        """Return a fresh, switched-off criterion over the whole 0-1 range."""
        harmonic = (
            self._harmonic_provider()
            if self._harmonic_provider is not None
            else 1
        )
        params = self._seed_params(index)
        params.setdefault('component_index', index)
        params['component_name'] = name
        low, high = self._seed_range(index)
        return new_filter(
            COMPONENT_FRACTION,
            low,
            high,
            harmonic or 1,
            enabled=False,
            params=params,
        )

    def _rebuild_cards(self):
        """Recreate every card so the list matches the components exactly."""
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._cards = {}
        for index, name, color in self._components:
            entry = self._entries[index]
            card = FilterCard(entry, metrics=None, removable=False)
            # The card is the component's *filter*, not the component itself.
            card.metric_label.setText(f"{name} Filter")
            card.set_bounds(*self.bounds_for(index))
            card.set_accent_color(color)
            card.enabled_check.setToolTip(_COMPONENT_ENABLE_TOOLTIP)
            card.set_enable_blocked(self._enable_blocked_reason)
            card.changed.connect(self._on_card_changed)
            self._cards_layout.addWidget(card)
            self._cards[index] = card
        self._refresh_chrome()

    def _refresh_chrome(self):
        """Show the placeholder text only while there is nothing to filter."""
        has_any = bool(self._components)
        self.empty_label.setVisible(not has_any)
        if not has_any:
            self.summary_label.setText("")

    def _index_of_card(self, filter_id):
        """Return the component index the card editing *filter_id* belongs to."""
        for index, entry in self._entries.items():
            if entry['id'] == filter_id:
                return index
        return None

    def _on_card_changed(self, filter_id):
        """Copy one card's edited values back and publish the new stack."""
        index = self._index_of_card(filter_id)
        if index is None:
            return
        card = self._cards.get(index)
        if card is None:
            return
        # A criterion created before the analysis ran carries no component
        # positions yet; it picks them up on its first edit, so the stack the
        # tab writes back stays identical to the one on screen and the card
        # being edited is not rebuilt under the pointer.
        if not has_component_positions(card.entry['params']):
            params = self._seed_params(index)
            if params:
                params['component_index'] = index
                params['component_name'] = self._name_for(index)
                card.entry['params'] = params
        self._entries[index] = dict(card.entry)
        self._placeholders.discard(index)
        if not self._rebuilding:
            self.filtersChanged.emit(self.filters())
