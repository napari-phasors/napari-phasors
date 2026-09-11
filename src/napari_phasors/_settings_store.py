"""Per-layer analysis settings: unsaved edits, commits and overwrite checks.

``layer.metadata['settings']`` records the parameters of the analyses that
were actually run on a layer, so a layer (or the OME-TIFF it is exported to)
can be reproduced later. Three rules keep that record honest when several
layers are selected at once:

* The tabs show the *primary* layer's parameters. Edits made there are kept
  as in-memory *drafts* of that layer until an analysis runs; they never
  reach the metadata (or an exported file) on their own, and switching the
  primary layer away and back brings them back.
* Running an analysis *commits* that analysis' parameters -- and only that
  analysis' -- to every layer it ran on, replacing their previous values.
  The other analyses stored on those layers are left untouched.
* Parameters kept per harmonic (or per output type) are merged instead of
  replaced: a run only replaces the entries it actually used.

:class:`LayerSettingsStore` holds the drafts and performs the commits; the
helpers below compare settings and build the merge rules.
"""

from __future__ import annotations

import copy
import math
import weakref
from collections.abc import Callable, Iterable, Mapping

import numpy as np

#: Top-level ``settings`` keys owned by each analysis (tab).
ANALYSIS_SETTINGS_KEYS = {
    "settings_tab": [
        "harmonic",
        "semi_circle",
        "white_background",
        "plot_type",
        "colormap",
        "histogram_style",
        "histogram_color",
        "number_of_bins",
        "log_scale",
        "marker_size",
        "marker_alpha",
        "marker_color",
        "contour_levels",
        "contour_linewidth",
        "contour_display_mode",
        "contour_layer_colors",
        "contour_group_assignments",
        "contour_group_colors",
        "contour_group_names",
        "contour_multi_layer_colormap",
        "contour_merged_style",
        "contour_merged_color",
        "contour_layer_styles",
        "contour_group_styles",
        "contour_show_legend",
        "contour_single_style",
        "contour_single_colormap",
        "contour_single_color",
        "phasor_center_enabled",
        "phasor_center_method",
        "phasor_center_color",
        "phasor_center_size",
        "phasor_center_alpha",
        "phasor_center_display_mode",
        "phasor_center_layer_colors",
        "phasor_center_group_assignments",
        "phasor_center_group_colors",
        "phasor_center_group_names",
        "timelapse_mode",
        "timelapse_axis",
    ],
    "calibration_tab": [
        "calibrated",
        "calibration_phase",
        "calibration_modulation",
        "calibration_reference",
    ],
    "filter_tab": [
        "filter",
        "threshold",
        "threshold_upper",
        "threshold_method",
    ],
    # The metric filter stack rides with the Phasor Mapping tab: it is one
    # ordered object, even when some of its criteria are on the FRET
    # efficiency.
    "phasor_mapping_tab": [
        "phasor_mapping",
        "lifetime",
        "mapping_filters",
    ],
    "fret_tab": ["fret"],
    "components_tab": ["component_analysis"],
    "selection_tab": ["selections"],
}

#: Human-readable name of each analysis, used in the overwrite notes.
ANALYSIS_LABELS = {
    "settings_tab": "plot settings",
    "calibration_tab": "Calibration",
    "filter_tab": "Filter",
    "phasor_mapping_tab": "Phasor Mapping",
    "fret_tab": "FRET",
    "components_tab": "Components",
    "selection_tab": "Selection",
}

#: How many layer names an overwrite note lists before summarising.
_MAX_NAMED_LAYERS = 4


def _normalize_key(key):
    """Return *key* as the string JSON turns it into.

    Harmonic-keyed dicts hold ``int`` keys in memory but ``str`` keys once
    they went through an OME-TIFF round-trip, and both must compare equal.
    """
    return str(key)


def settings_equal(a, b) -> bool:
    """Return whether two settings values are equivalent.

    Tolerates the differences an OME-TIFF round-trip introduces -- tuples
    read back as lists, ``int`` dict keys read back as strings, NumPy arrays
    read back as nested lists -- and float round-off.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)
        except (TypeError, ValueError):
            return False
        if a_arr.shape != b_arr.shape:
            return False
        if a_arr.dtype.kind in "fc" or b_arr.dtype.kind in "fc":
            return bool(np.allclose(a_arr, b_arr, equal_nan=True))
        return bool(np.array_equal(a_arr, b_arr))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        a_items = {_normalize_key(k): v for k, v in a.items()}
        b_items = {_normalize_key(k): v for k, v in b.items()}
        if a_items.keys() != b_items.keys():
            return False
        return all(settings_equal(a_items[k], b_items[k]) for k in a_items)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            settings_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or (
            isinstance(a, (bool, np.bool_))
            and isinstance(b, (bool, np.bool_))
            and bool(a) == bool(b)
        )
    if isinstance(a, (int, float, np.number)) and isinstance(
        b, (int, float, np.number)
    ):
        a_f, b_f = float(a), float(b)
        if math.isnan(a_f) and math.isnan(b_f):
            return True
        return math.isclose(a_f, b_f, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def replace_keyed_entries(old, new, keys) -> dict:
    """Return *old* with the entries for *keys* taken from *new*.

    Used for dicts keyed by harmonic (or output type): entries *new* does not
    carry for a run key are dropped, every other entry of *old* is kept.
    Keys are matched by their string form, see :func:`_normalize_key`.
    """
    wanted = {_normalize_key(k) for k in keys}
    merged = {
        k: copy.deepcopy(v)
        for k, v in (old or {}).items()
        if _normalize_key(k) not in wanted
    }
    for k, v in (new or {}).items():
        if _normalize_key(k) in wanted:
            merged[k] = copy.deepcopy(v)
    return merged


def merge_keyed_path(path: Iterable[str], keys) -> Callable:
    """Return a merge rule replacing only *keys* of the dict at *path*.

    The rest of the block is replaced by the new value as usual; only the
    dict found by following *path* inside the block keeps the old entries
    for keys the run did not use.
    """
    path = tuple(path)

    def merge(old, new):
        if not isinstance(new, Mapping):
            return new
        merged = copy.deepcopy(dict(new))
        old_sub = _get_path(old, path)
        new_sub = _get_path(new, path)
        _set_path(
            merged,
            path,
            replace_keyed_entries(
                old_sub if isinstance(old_sub, Mapping) else {},
                new_sub if isinstance(new_sub, Mapping) else {},
                keys,
            ),
        )
        return merged

    return merge


def chain_merges(*merges: Callable) -> Callable:
    """Return a merge rule applying *merges* one after the other."""

    def merge(old, new):
        for rule in merges:
            new = rule(old, new)
        return new

    return merge


def _get_path(container, path):
    """Return the value at *path* inside nested dicts, or ``None``."""
    value = container
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _set_path(container, path, value):
    """Set *value* at *path* inside nested dicts, creating them as needed."""
    target = container
    for key in path[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            child = {}
            target[key] = child
        target = child
    target[path[-1]] = value


def format_layer_list(names) -> str:
    """Return *names* as a short comma-separated list for a note."""
    names = list(names)
    if len(names) <= _MAX_NAMED_LAYERS:
        return ", ".join(names)
    shown = ", ".join(names[:_MAX_NAMED_LAYERS])
    return f"{shown} and {len(names) - _MAX_NAMED_LAYERS} more"


class LayerSettingsStore:
    """Hold unsaved per-layer edits and commit analyses to layer metadata.

    Drafts are keyed by the layer object (weakly), so they follow a renamed
    layer and disappear with a deleted one. A draft replaces the whole
    top-level settings value it is stored under.

    Parameters
    ----------
    on_change : callable, optional
        Called without arguments after drafts or committed settings change,
        e.g. to refresh the overwrite notes.
    """

    def __init__(self, on_change: Callable | None = None):
        self._drafts = weakref.WeakKeyDictionary()
        self._on_change = on_change

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @staticmethod
    def committed(layer) -> dict:
        """Return *layer*'s stored settings (not a copy; do not mutate)."""
        if layer is None:
            return {}
        return layer.metadata.get("settings") or {}

    def drafts(self, layer) -> dict:
        """Return *layer*'s unsaved edits (not a copy; do not mutate)."""
        if layer is None:
            return {}
        return self._drafts.get(layer, {})

    def effective(self, layer) -> dict:
        """Return *layer*'s settings with its unsaved edits applied.

        The result is a new top-level dict, but its values are shared with
        the metadata and the drafts, so it must be treated as read-only.
        """
        merged = dict(self.committed(layer))
        merged.update(self.drafts(layer))
        return merged

    def get(self, layer, key, default=None):
        """Return the effective value of *key* for *layer*."""
        drafts = self.drafts(layer)
        if key in drafts:
            return drafts[key]
        return self.committed(layer).get(key, default)

    def has_draft(self, layer, keys=None) -> bool:
        """Return whether *layer* has unsaved edits (for any of *keys*)."""
        drafts = self.drafts(layer)
        if keys is None:
            return bool(drafts)
        return any(key in drafts for key in keys)

    # ------------------------------------------------------------------
    # Drafts
    # ------------------------------------------------------------------

    def set_draft(self, layer, key, value, *, notify=True):
        """Record *value* as the unsaved value of *key* on *layer*.

        A draft equal to the stored value is dropped instead, so restoring
        the widgets from a layer never leaves drafts behind.
        """
        if layer is None:
            return
        committed = self.committed(layer)
        drafts = self._drafts.setdefault(layer, {})
        if key in committed and settings_equal(committed[key], value):
            drafts.pop(key, None)
        else:
            drafts[key] = copy.deepcopy(value)
        if not drafts:
            self._drafts.pop(layer, None)
        if notify:
            self._notify()

    def draft_block(self, layer, key, default_factory: Callable = dict):
        """Return the mutable draft dict stored under *key* for *layer*.

        The first call copies the stored value (or ``default_factory()``) so
        edits can be made in place; :meth:`notify` should be called after.
        """
        drafts = self._drafts.setdefault(layer, {})
        if key not in drafts:
            committed = self.committed(layer).get(key)
            drafts[key] = (
                copy.deepcopy(committed)
                if isinstance(committed, dict)
                else default_factory()
            )
        return drafts[key]

    def set_draft_path(self, layer, key, path, value, default_factory=dict):
        """Set *value* at *path* inside the draft dict stored under *key*."""
        if layer is None:
            return
        block = self.draft_block(layer, key, default_factory)
        _set_path(block, tuple(path), copy.deepcopy(value))
        self._drop_if_unchanged(layer, key)
        self._notify()

    def settle_draft(self, layer, key):
        """Finish an in-place edit of the draft from :meth:`draft_block`.

        Drops the draft when the edit made it match the metadata again, and
        notifies the owner.
        """
        if layer is not None:
            self._drop_if_unchanged(layer, key)
        self._notify()

    def discard_drafts(self, layers, keys=None, *, notify=True):
        """Forget the unsaved edits of *layers* (only *keys* if given)."""
        for layer in layers:
            drafts = self._drafts.get(layer)
            if drafts is None:
                continue
            if keys is None:
                drafts.clear()
            else:
                for key in keys:
                    drafts.pop(key, None)
            if not drafts:
                self._drafts.pop(layer, None)
        if notify:
            self._notify()

    def _drop_if_unchanged(self, layer, key):
        """Drop the draft of *key* on *layer* when it matches the metadata."""
        drafts = self._drafts.get(layer)
        if drafts is None or key not in drafts:
            return
        committed = self.committed(layer)
        if key in committed and settings_equal(committed[key], drafts[key]):
            del drafts[key]
            if not drafts:
                self._drafts.pop(layer, None)

    # ------------------------------------------------------------------
    # Committing
    # ------------------------------------------------------------------

    def commit(self, layers, values: Mapping, merge: Mapping | None = None):
        """Write *values* into the settings of every layer in *layers*.

        Each value is deep-copied per layer, so layers never share a dict.
        ``merge`` maps a key to ``rule(old_value, new_value) -> value`` for
        keys that must keep part of the old value (per-harmonic entries).
        The layers' drafts for the committed keys are discarded.
        """
        merge = merge or {}
        for layer in layers:
            settings = layer.metadata.setdefault("settings", {})
            for key, value in values.items():
                new_value = copy.deepcopy(value)
                rule = merge.get(key)
                if rule is not None and key in settings:
                    new_value = rule(settings[key], new_value)
                settings[key] = new_value
        self.discard_drafts(layers, list(values), notify=False)
        self._notify()

    def update_committed(self, layers, key, path, value):
        """Set *value* at *path* inside the stored *key* dict of *layers*.

        Meant for changes to an analysis' outputs after it ran (a colormap
        or display range), which apply to every layer it ran on. Existing
        drafts of *key* get the same change, so they do not undo it.
        """
        path = tuple(path)
        for layer in layers:
            settings = layer.metadata.setdefault("settings", {})
            block = settings.get(key)
            if not isinstance(block, dict):
                block = {}
                settings[key] = block
            _set_path(block, path, copy.deepcopy(value))
            draft = self.drafts(layer).get(key)
            if isinstance(draft, dict):
                _set_path(draft, path, copy.deepcopy(value))
                self._drop_if_unchanged(layer, key)
        self._notify()

    def overwritten_layers(self, layers, keys, values, merge=None):
        """Return the layers whose stored *keys* a commit would change.

        *values* holds the values that would be committed; a key missing
        from it means they are unknown (the tab shows unsaved defaults), in
        which case any stored value counts as overwritten. Layers that have
        none of *keys* stored lose nothing and are never returned.
        """
        merge = merge or {}
        result = []
        for layer in layers:
            stored = self.committed(layer)
            for key in keys:
                if key not in stored or stored[key] in (None, {}, []):
                    continue
                if key not in values:
                    result.append(layer)
                    break
                new_value = values[key]
                rule = merge.get(key)
                if rule is not None:
                    new_value = rule(stored[key], copy.deepcopy(new_value))
                if not settings_equal(stored[key], new_value):
                    result.append(layer)
                    break
        return result

    def notify(self):
        """Tell the owner that drafts changed after an in-place edit."""
        self._notify()

    def _notify(self):
        if self._on_change is not None:
            self._on_change()
