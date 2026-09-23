"""Tests for the metric filter stack and its card list."""

import numpy as np
import pytest
from napari.layers import Image

from napari_phasors._mapping_filters import (
    ALL_METRICS,
    APPARENT_MODULATION_LIFETIME,
    APPARENT_PHASE_LIFETIME,
    COMPONENT_FRACTION,
    EXCLUDE,
    FRET_EFFICIENCY,
    KEEP,
    MAPPING_METRICS,
    MODULATION,
    NORMAL_LIFETIME,
    PHASE,
    ComponentFilterList,
    MappingFilterList,
    MetricContext,
    apply_filters_to_arrays,
    apply_layer_filters,
    apply_mask_to_arrays,
    baseline_arrays,
    combined_mask,
    compute_metric,
    describe_filters,
    filter_display_name,
    format_range,
    get_filters,
    has_filters,
    kept_fraction,
    metric_fallback_range,
    metric_unit,
    new_filter,
    normalize_filters,
    range_mask,
    rebuild_layer_from_filters,
    requires_frequency,
    select_harmonic,
    serialize_filter_applies,
    set_filters,
)


def _layer(name="layer", harmonics=None, multi=False):
    """Return an image layer carrying simple, fully finite phasor arrays."""
    mean = np.ones((4, 4), dtype=float)
    real = np.linspace(0.1, 0.9, 16).reshape(4, 4)
    imag = np.linspace(0.05, 0.45, 16).reshape(4, 4)
    if multi:
        real = np.stack([real, real / 2])
        imag = np.stack([imag, imag / 2])
        harmonics = np.array([1, 2]) if harmonics is None else harmonics
    return Image(
        mean,
        name=name,
        metadata={
            'G': real.copy(),
            'S': imag.copy(),
            'G_original': real.copy(),
            'S_original': imag.copy(),
            'original_mean': mean.copy(),
            'harmonics': harmonics,
            'settings': {},
        },
    )


# ---------------------------------------------------------------- metadata


def test_metric_metadata_helpers():
    """Units, fallback ranges and frequency needs are reported per metric."""
    assert metric_unit(NORMAL_LIFETIME) == "ns"
    assert metric_unit(PHASE) == "rad"
    assert metric_unit(MODULATION) == ""
    assert metric_unit("nonsense") == ""
    assert metric_fallback_range(MODULATION) == (0.0, 1.0)
    assert metric_fallback_range("nonsense") == (0.0, 1.0)
    assert metric_fallback_range(PHASE)[1] == pytest.approx(2 * np.pi)
    assert requires_frequency(NORMAL_LIFETIME)
    assert requires_frequency(APPARENT_MODULATION_LIFETIME)
    assert not requires_frequency(PHASE)
    assert not requires_frequency(FRET_EFFICIENCY)


def test_format_range_names_the_unit_and_the_mode():
    """A criterion reads as a range, and says so when it excludes one."""
    assert format_range(NORMAL_LIFETIME, 0.5, 3.25) == "0.5 – 3.25 ns"
    assert format_range(MODULATION, 0.2, 0.8) == "0.2 – 0.8"
    assert format_range(PHASE, 0.0, 1.0, EXCLUDE) == "outside 0 – 1 rad"


def test_describe_filters_lists_only_the_enabled_criteria():
    """The summary names the criteria actually in force."""
    assert describe_filters([]) == "No filters applied."
    filters = [
        new_filter(NORMAL_LIFETIME, 1.0, 2.0),
        new_filter(MODULATION, 0.1, 0.2, enabled=False),
    ]
    text = describe_filters(filters)
    assert NORMAL_LIFETIME in text
    assert MODULATION not in text


# ------------------------------------------------------------ (de)serialise


def test_new_filter_normalises_its_inputs():
    """A criterion is a plain, JSON-safe dict with a stable id."""
    entry = new_filter(PHASE, 0, 1, harmonic=2.0, mode="nonsense")
    assert entry['metric'] == PHASE
    assert entry['harmonic'] == 2
    assert entry['mode'] == KEEP
    assert entry['enabled'] is True
    assert entry['params'] == {}
    assert isinstance(entry['id'], str) and entry['id']
    assert new_filter(PHASE, 0, 1, mode=EXCLUDE)['mode'] == EXCLUDE


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not a dict",
        {'metric': 'Unknown metric', 'min': 0, 'max': 1},
        {'metric': PHASE, 'min': 0},
        {'metric': PHASE, 'min': 0, 'max': 'x'},
        {'metric': PHASE, 'min': np.nan, 'max': 1},
        {'metric': PHASE, 'min': 0, 'max': np.inf},
    ],
)
def test_unusable_entries_are_dropped(raw):
    """Junk in the stored settings never reaches the arrays."""
    assert normalize_filters([raw]) == []


def test_normalize_filters_repairs_what_it_can():
    """A reversed range, an odd harmonic and a bad params blob are fixed."""
    (entry,) = normalize_filters(
        [
            {
                'metric': NORMAL_LIFETIME,
                'min': 3.0,
                'max': 1.0,
                'harmonic': 'x',
                'params': 'not a dict',
                'id': 'keep-me',
            }
        ]
    )
    assert (entry['min'], entry['max']) == (1.0, 3.0)
    assert entry['harmonic'] == 1
    assert entry['params'] == {}
    assert entry['id'] == 'keep-me'


def test_normalize_filters_accepts_a_bare_dict_and_rejects_scalars():
    """A single criterion is read as a one-entry stack."""
    assert len(normalize_filters({'metric': PHASE, 'min': 0, 'max': 1})) == 1
    assert normalize_filters(None) == []
    assert normalize_filters(42) == []


def test_get_filters_reads_the_legacy_single_filter_format():
    """A layer saved before the stack existed keeps its one filter."""
    layer = _layer()
    layer.metadata['settings']['mapping_filter'] = {
        'output_type': NORMAL_LIFETIME,
        'min': 1.0,
        'max': 2.0,
        'harmonic': 1,
    }
    (entry,) = get_filters(layer)
    assert entry['metric'] == NORMAL_LIFETIME
    assert (entry['min'], entry['max']) == (1.0, 2.0)

    nested = _layer()
    nested.metadata['settings']['phasor_mapping'] = {
        'mapping_filter': {'metric': MODULATION, 'min': 0.1, 'max': 0.9}
    }
    assert get_filters(nested)[0]['metric'] == MODULATION


def test_set_filters_retires_the_legacy_keys():
    """Storing a stack removes the old key, so a deletion cannot come back."""
    layer = _layer()
    layer.metadata['settings']['mapping_filter'] = {
        'metric': MODULATION,
        'min': 0.1,
        'max': 0.9,
    }
    layer.metadata['settings']['phasor_mapping'] = {
        'mapping_filter': {'metric': PHASE, 'min': 0, 'max': 1}
    }
    set_filters(layer, [])
    assert 'mapping_filter' not in layer.metadata['settings']
    assert 'mapping_filter' not in layer.metadata['settings']['phasor_mapping']
    assert get_filters(layer) == []
    assert not has_filters(layer)


def test_set_filters_round_trips_and_reports_enabled_state():
    """What is stored is what comes back, and only enabled entries count."""
    layer = _layer()
    stored = set_filters(layer, [new_filter(MODULATION, 0.2, 0.8)])
    assert len(stored) == 1
    assert get_filters(layer) == stored
    assert has_filters(layer)

    set_filters(layer, [new_filter(MODULATION, 0.2, 0.8, enabled=False)])
    assert not has_filters(layer)
    assert len(get_filters(layer)) == 1


# ------------------------------------------------------------------ metrics


def test_select_harmonic_picks_the_requested_plane():
    """A stacked array is indexed by harmonic value, not by position."""
    real = np.stack([np.zeros((2, 2)), np.ones((2, 2))])
    imag = real.copy()
    harmonics = np.array([1, 3])

    picked, _ = select_harmonic(real, imag, harmonics, 3, 2)
    assert np.all(picked == 1)

    # A harmonic that was never computed falls back to the first plane.
    picked, _ = select_harmonic(real, imag, harmonics, 7, 2)
    assert np.all(picked == 0)
    # So does an index the array is too short for.
    picked, _ = select_harmonic(real, imag, np.array([1, 3, 5]), 5, 2)
    assert np.all(picked == 0)
    # Without a harmonic axis the arrays are returned untouched.
    flat = np.ones((2, 2))
    assert select_harmonic(flat, flat, None, 1, 2)[0] is flat
    assert select_harmonic(None, None, None, 1, 2) == (None, None)
    # No harmonics recorded: the single stored plane is used.
    picked, _ = select_harmonic(real, imag, None, 1, 2)
    assert np.all(picked == 0)


@pytest.mark.parametrize(
    "metric",
    [APPARENT_PHASE_LIFETIME, APPARENT_MODULATION_LIFETIME, NORMAL_LIFETIME],
)
def test_lifetime_metrics_need_a_positive_frequency(metric):
    """A lifetime with no usable frequency is not guessed at."""
    real = np.full((2, 2), 0.5)
    imag = np.full((2, 2), 0.3)
    assert compute_metric(metric, real, imag) is None
    assert compute_metric(metric, real, imag, frequency=0) is None
    assert compute_metric(metric, real, imag, frequency=np.nan) is None
    values = compute_metric(metric, real, imag, frequency=80.0)
    assert values.shape == (2, 2)
    assert np.all(values >= 0)


def test_lifetime_metric_reads_the_frequency_from_params():
    """A criterion carrying its own frequency does not need one passed in."""
    real = np.full((2, 2), 0.5)
    imag = np.full((2, 2), 0.3)
    from_params = compute_metric(
        NORMAL_LIFETIME, real, imag, params={'frequency': 80.0}
    )
    assert from_params == pytest.approx(
        compute_metric(NORMAL_LIFETIME, real, imag, frequency=80.0)
    )


def test_lifetime_metric_scales_with_the_harmonic():
    """Harmonic n is measured at n times the excitation frequency."""
    real = np.full((2, 2), 0.5)
    imag = np.full((2, 2), 0.3)
    first = compute_metric(NORMAL_LIFETIME, real, imag, frequency=80.0)
    second = compute_metric(
        NORMAL_LIFETIME, real, imag, harmonic=2, frequency=80.0
    )
    assert second == pytest.approx(
        compute_metric(NORMAL_LIFETIME, real, imag, frequency=160.0)
    )
    assert not np.allclose(first, second)


def test_negative_apparent_lifetimes_are_clamped_to_zero():
    """A phasor outside the semicircle cannot report a negative lifetime."""
    real = np.array([[1.4]])
    imag = np.array([[-0.4]])
    values = compute_metric(
        APPARENT_PHASE_LIFETIME, real, imag, frequency=80.0
    )
    assert values[0, 0] == 0


def test_phase_and_modulation_metrics():
    """Phase optionally wraps to [0, 2pi); modulation never does."""
    real = np.array([[0.5, -0.5]])
    imag = np.array([[0.5, -0.5]])
    unwrapped = compute_metric(PHASE, real, imag)
    wrapped = compute_metric(PHASE, real, imag, wrap_phase=True)
    assert unwrapped[0, 1] < 0
    assert np.all(wrapped >= 0)
    assert wrapped[0, 1] == pytest.approx(unwrapped[0, 1] + 2 * np.pi)

    modulation = compute_metric(MODULATION, real, imag)
    assert modulation[0, 0] == pytest.approx(np.hypot(0.5, 0.5))


def test_compute_metric_rejects_missing_input_and_unknown_metrics():
    """Nothing to measure, or nothing known to measure, yields None."""
    real = np.full((2, 2), 0.5)
    assert compute_metric(PHASE, None, real) is None
    assert compute_metric("Unknown", real, real) is None


def test_fret_efficiency_metric():
    """The efficiency is rebuilt from the criterion's own donor trajectory."""
    real = np.array([[0.5, np.nan]])
    imag = np.array([[0.3, np.nan]])
    params = {'frequency': 80.0, 'donor_lifetime': 4.2}
    values = compute_metric(FRET_EFFICIENCY, real, imag, params=params)
    assert values.shape == (1, 2)
    assert 0.0 <= values[0, 0] <= 1.0
    assert np.isnan(values[0, 1])


@pytest.mark.parametrize(
    "params",
    [
        {},
        {'frequency': 80.0},
        {'donor_lifetime': 4.2},
        {'frequency': 'x', 'donor_lifetime': 4.2},
        {'frequency': 0.0, 'donor_lifetime': 4.2},
        {'frequency': 80.0, 'donor_lifetime': -1.0},
    ],
)
def test_fret_efficiency_metric_needs_a_full_trajectory(params):
    """An incomplete donor trajectory produces no efficiency at all."""
    real = np.full((2, 2), 0.5)
    assert compute_metric(FRET_EFFICIENCY, real, real, params=params) is None


# ------------------------------------------------------------------- masks


def test_range_mask_keeps_and_excludes():
    """A criterion selects inside its range, or outside it, and drops NaN."""
    values = np.array([0.0, 0.5, 1.0, np.nan, np.inf])
    keep = range_mask(values, 0.4, 0.9)
    assert list(keep) == [True, False, True, True, True]
    exclude = range_mask(values, 0.4, 0.9, EXCLUDE)
    assert list(exclude) == [False, True, False, True, True]


def test_combined_mask_is_the_and_of_the_criteria():
    """Two criteria keep only what satisfies both, in either order."""
    layer = _layer()
    mean, real, imag = (
        layer.data,
        layer.metadata['G'],
        layer.metadata['S'],
    )
    a = new_filter(MODULATION, 0.0, 0.6)
    b = new_filter(PHASE, 0.0, 0.5)
    forward = combined_mask([a, b], mean, real, imag, None)
    backward = combined_mask([b, a], mean, real, imag, None)
    assert np.array_equal(forward, backward)
    only_a = combined_mask([a], mean, real, imag, None)
    assert np.array_equal(
        forward, only_a | combined_mask([b], mean, real, imag, None)
    )
    assert forward.sum() >= only_a.sum()


def test_combined_mask_skips_disabled_and_unusable_criteria():
    """A disabled criterion does nothing; an unusable one is reported."""
    layer = _layer()
    mean, real, imag = layer.data, layer.metadata['G'], layer.metadata['S']

    disabled = new_filter(MODULATION, 0.0, 0.1, enabled=False)
    assert combined_mask([disabled], mean, real, imag, None) is None

    problems = []
    # A lifetime criterion with no frequency cannot be evaluated.
    unusable = new_filter(NORMAL_LIFETIME, 1.0, 2.0)
    assert (
        combined_mask(
            [unusable], mean, real, imag, None, on_error=problems.append
        )
        is None
    )
    assert problems and NORMAL_LIFETIME in problems[0]

    assert combined_mask([disabled], None, real, imag, None) is None


def test_combined_mask_skips_a_metric_of_the_wrong_shape():
    """A metric that does not line up with the image is not applied."""
    layer = _layer()
    mean = np.ones((3, 3))
    problems = []
    entry = new_filter(MODULATION, 0.0, 0.1)
    assert (
        combined_mask(
            [entry],
            mean,
            layer.metadata['G'],
            layer.metadata['S'],
            None,
            on_error=problems.append,
        )
        is None
    )
    assert problems


def test_apply_mask_to_arrays_handles_both_layouts():
    """The mask is broadcast over the harmonic axis when there is one."""
    mask = np.array([[True, False]])
    mean = np.ones((1, 2))
    flat = np.ones((1, 2))
    _, real, imag = apply_mask_to_arrays(mask, mean, flat, flat)
    assert np.isnan(real[0, 0]) and real[0, 1] == 1

    stacked = np.ones((2, 1, 2))
    _, real, imag = apply_mask_to_arrays(mask, mean, stacked, stacked.copy())
    assert np.all(np.isnan(real[:, 0, 0]))
    assert np.all(real[:, 0, 1] == 1)

    assert apply_mask_to_arrays(None, mean, flat, flat)[0] is mean
    assert apply_mask_to_arrays(mask, None, None, None) == (None, None, None)


def test_kept_fraction_ignores_pixels_that_were_never_measurable():
    """The share is of the measured pixels, not of the whole frame."""
    mean = np.array([1.0, 1.0, np.nan, np.nan])
    mask = np.array([False, True, False, True])
    assert kept_fraction(mask, mean) == pytest.approx(0.5)
    assert kept_fraction(mask) == pytest.approx(0.5)
    assert kept_fraction(None) == 1.0
    assert kept_fraction(np.array([True]), np.array([np.nan])) == 0.0
    assert kept_fraction(np.zeros(0, dtype=bool)) == 0.0


# ----------------------------------------------------------------- applying


def test_apply_filters_to_arrays_nans_the_dropped_pixels():
    """Filtered pixels lose their coordinates, in the mean and in G/S alike."""
    layer = _layer()
    entry = new_filter(MODULATION, 0.0, 0.4)
    mean, real, imag, mask = apply_filters_to_arrays(
        [entry],
        layer.data.copy(),
        layer.metadata['G'].copy(),
        layer.metadata['S'].copy(),
        None,
    )
    assert mask.any()
    assert np.array_equal(np.isnan(mean), mask)
    assert np.array_equal(np.isnan(real), mask)
    assert np.array_equal(np.isnan(imag), mask)


def test_apply_layer_filters_is_a_no_op_without_a_stack():
    """A layer with no criteria gets its arrays back untouched."""
    layer = _layer()
    arrays = (layer.data, layer.metadata['G'], layer.metadata['S'])
    assert apply_layer_filters(layer, arrays) == arrays


def test_baseline_arrays_ignores_the_stack():
    """The baseline is what the criteria are measured against, unfiltered."""
    layer = _layer()
    set_filters(layer, [new_filter(MODULATION, 0.0, 0.2)])
    rebuild_layer_from_filters(layer)
    assert np.isnan(layer.metadata['G']).any()

    mean, real, _ = baseline_arrays(layer)
    assert not np.isnan(real).any()
    assert not np.isnan(mean).any()


def test_baseline_arrays_falls_back_to_the_stored_arrays():
    """A layer without the originals still yields something to measure."""
    layer = _layer()
    del layer.metadata['original_mean']
    mean, real, imag = baseline_arrays(layer)
    assert mean.shape == layer.data.shape
    assert real.shape == layer.metadata['G'].shape

    bare = Image(np.ones((2, 2)), name="bare", metadata={'settings': {}})
    mean, real, imag = baseline_arrays(bare)
    assert mean.shape == (2, 2)
    assert real is None and imag is None


def test_rebuild_layer_from_filters_round_trips():
    """Removing a criterion restores exactly the pixels it had hidden."""
    layer = _layer()
    original = layer.metadata['G'].copy()

    set_filters(layer, [new_filter(MODULATION, 0.0, 0.3)])
    mask = rebuild_layer_from_filters(layer)
    assert mask.any()
    assert np.isnan(layer.metadata['G']).sum() == mask.sum()

    set_filters(layer, [])
    assert rebuild_layer_from_filters(layer) is None
    np.testing.assert_allclose(layer.metadata['G'], original)


def test_rebuild_layer_from_filters_handles_a_multi_harmonic_layer():
    """Every harmonic plane loses the same pixels."""
    layer = _layer(multi=True)
    set_filters(layer, [new_filter(MODULATION, 0.0, 0.3, harmonic=2)])
    mask = rebuild_layer_from_filters(layer)
    assert mask.shape == layer.data.shape
    for plane in layer.metadata['G']:
        assert np.array_equal(np.isnan(plane), mask)


class _NoDataLayer:
    """Stand-in for a layer whose arrays have gone missing entirely."""

    data = None
    metadata = {'settings': {}}


def test_rebuild_layer_from_filters_without_data_is_a_no_op():
    """A layer with nothing to rebuild reports no mask."""
    assert rebuild_layer_from_filters(_NoDataLayer(), []) is None


# ---------------------------------------------------------------- card list


def test_filter_list_add_edit_and_remove(qtbot):
    """The card list is the stack: adding, editing and removing publish it."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    published = []
    widget.filtersChanged.connect(published.append)

    assert widget.empty_label.isVisibleTo(widget)

    widget.set_current_metric(MODULATION)
    widget.set_metric_bounds(MODULATION, 0.0, 1.0)
    widget._on_add_clicked()

    assert len(published[-1]) == 1
    assert not widget.empty_label.isVisibleTo(widget)
    entry = widget.filters()[0]
    assert entry['metric'] == MODULATION
    assert (entry['min'], entry['max']) == (0.0, 1.0)

    card = widget._cards[entry['id']]
    # The metric is chosen on the card, from every metric the list offers.
    assert card.metric_combobox.isVisibleTo(card)
    assert card.metric_combobox.currentText() == MODULATION
    assert [
        card.metric_combobox.itemText(i)
        for i in range(card.metric_combobox.count())
    ] == list(MAPPING_METRICS)

    card.min_edit.setText("0.25")
    card.max_edit.setText("0.75")
    card._on_edits_changed()
    edited = widget.filters()[0]
    assert (edited['min'], edited['max']) == (0.25, 0.75)

    widget._on_card_removed(entry['id'])
    assert widget.filters() == []
    assert published[-1] == []


def test_add_button_sits_below_the_cards(qtbot):
    """New filters are added from the bottom of the list; nothing clears all."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    layout = widget.layout()
    assert layout.indexOf(widget.add_button) > layout.indexOf(
        widget._cards_container
    )
    assert layout.indexOf(widget.add_button) == layout.count() - 1
    assert not hasattr(widget, 'clear_button')


def test_card_metric_selector_reseeds_the_criterion(qtbot):
    """Switching a card's metric restarts its range and parameters."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget.set_bounds_provider({PHASE: (0.1, 1.2)}.get)
    widget.set_params_provider(
        lambda metric: (
            {'frequency': 80.0} if requires_frequency(metric) else {}
        )
    )
    widget.set_current_metric(MODULATION)
    widget._on_add_clicked()
    card = next(iter(widget._cards.values()))
    filter_id = card.filter_id
    published = []
    widget.filtersChanged.connect(published.append)

    card.metric_combobox.setCurrentText(PHASE)
    (entry,) = published[-1]
    assert entry['id'] == filter_id
    assert entry['metric'] == PHASE
    assert (entry['min'], entry['max']) == (0.1, 1.2)
    assert entry['params'] == {}
    assert card.unit_label.text() == "rad"
    # The card is updated in place, not rebuilt under the pointer.
    assert next(iter(widget._cards.values())) is card

    # A metric the provider cannot measure starts on its fallback range.
    card.metric_combobox.setCurrentText(NORMAL_LIFETIME)
    (entry,) = published[-1]
    assert (entry['min'], entry['max']) == metric_fallback_range(
        NORMAL_LIFETIME
    )
    assert entry['params'] == {'frequency': 80.0}

    count = len(published)
    card._on_metric_selected(NORMAL_LIFETIME)
    widget._on_metric_change("no-such-card", PHASE)
    assert len(published) == count


def test_filter_list_set_filters_does_not_rebuild_an_identical_stack(qtbot):
    """Re-reading the stack must not destroy the card being edited."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget._on_add_clicked()
    card = next(iter(widget._cards.values()))

    widget.set_filters(widget.filters())
    assert next(iter(widget._cards.values())) is card

    widget.set_filters([])
    assert widget._cards == {}


def test_filter_list_reports_stats_and_summary(qtbot):
    """Per-card and overall figures are shown where the user is looking."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget._on_add_clicked()
    entry = widget.filters()[0]

    widget.set_filter_stats(
        {entry['id']: "keeps 40.0%"},
        "1 of 1 active",
        detail="Measured on some layer.",
    )
    card = widget._cards[entry['id']]
    assert card.stat_label.text() == "keeps 40.0%"
    assert widget.summary_label.text() == "1 of 1 active"
    # Anything too long for the line lives in the tooltip instead.
    assert "Measured on some layer." in widget.summary_label.toolTip()
    assert entry['metric'] in widget.summary_label.toolTip()

    widget.set_filter_stats({}, "1 of 1 active")
    assert widget.summary_label.toolTip() == describe_filters(widget.filters())


def test_filter_list_providers_seed_a_new_criterion(qtbot):
    """The harmonic and frozen parameters come from the owning tab."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget.set_params_provider(lambda metric: {'frequency': 80.0})
    widget.set_harmonic_provider(lambda: 3)
    widget.set_current_metric(NORMAL_LIFETIME)
    widget._on_add_clicked()

    entry = widget.filters()[0]
    assert entry['params'] == {'frequency': 80.0}
    assert entry['harmonic'] == 3


def test_filter_list_add_rejects_an_unusable_entry(qtbot):
    """A malformed criterion is refused rather than half-added."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    assert widget.add_filter({'metric': 'nonsense'}) is None
    assert widget.filters() == []


def test_single_mode_shows_one_fixed_card(qtbot):
    """The Fret tab's list is one always-present card: no add, no remove."""
    widget = MappingFilterList([FRET_EFFICIENCY], single=True)
    qtbot.addWidget(widget)
    (card,) = widget._cards.values()

    assert not widget.add_button.isVisibleTo(widget)
    assert not widget.summary_label.isVisibleTo(widget)
    assert not widget.empty_label.isVisibleTo(widget)
    assert not card.remove_button.isVisibleTo(card)
    assert not card.metric_combobox.isVisibleTo(card)
    assert card.metric_label.text() == FRET_EFFICIENCY
    assert widget.current_metric() == FRET_EFFICIENCY
    widget.set_current_metric(MODULATION)
    assert widget.current_metric() == FRET_EFFICIENCY

    # Until it is touched the card is a placeholder: the stack is empty.
    assert card.entry['enabled'] is False
    assert widget.filters() == []
    widget.set_filters([])
    assert next(iter(widget._cards.values())) is card


def test_single_mode_placeholder_becomes_a_real_filter(qtbot):
    """Switching the card on publishes it, with the tab's parameters."""
    widget = MappingFilterList([FRET_EFFICIENCY], single=True)
    qtbot.addWidget(widget)
    widget.set_params_provider(
        lambda metric: {'frequency': 80.0, 'donor_lifetime': 4.2}
    )
    published = []
    widget.filtersChanged.connect(published.append)
    card = next(iter(widget._cards.values()))

    card.enabled_check.setChecked(True)
    (entry,) = published[-1]
    assert entry['enabled'] is True
    assert entry['params'] == {'frequency': 80.0, 'donor_lifetime': 4.2}

    # Reading the stored stack back does not rebuild the card.
    widget.set_filters(published[-1])
    assert next(iter(widget._cards.values())) is card

    # An emptied stack brings the placeholder back.
    widget.set_filters([])
    assert widget.filters() == []
    assert len(widget._cards) == 1

    # Only one criterion is ever shown.
    widget.set_filters(
        [
            new_filter(FRET_EFFICIENCY, 0.1, 0.5),
            new_filter(FRET_EFFICIENCY, 0.6, 0.9),
        ]
    )
    assert len(widget._cards) == 1
    assert widget.filters()[0]['max'] == 0.5


def test_enable_blocked_only_prevents_switching_on(qtbot):
    """A filter can always be switched off, but only on when it can run."""
    widget = MappingFilterList([FRET_EFFICIENCY], single=True)
    qtbot.addWidget(widget)
    widget.set_enable_blocked("Enter a donor lifetime.")
    card = next(iter(widget._cards.values()))
    assert not card.enabled_check.isEnabled()
    assert card.enabled_check.toolTip() == "Enter a donor lifetime."

    widget.set_enable_blocked(None)
    assert card.enabled_check.isEnabled()

    widget.set_filters([new_filter(FRET_EFFICIENCY, 0.0, 1.0)])
    widget.set_enable_blocked("Enter a donor lifetime.")
    card = next(iter(widget._cards.values()))
    assert card.enabled_check.isEnabled()


def test_empty_metric_list_adds_nothing(qtbot):
    """A list offering no metric has nothing to add."""
    empty = MappingFilterList([])
    qtbot.addWidget(empty)
    assert empty.current_metric() is None
    empty._on_add_clicked()
    assert empty.filters() == []


def test_filter_list_marks_foreign_criteria_read_only(qtbot):
    """A criterion another tab owns is listed, but edited where it was made."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget.set_editable_metrics(MAPPING_METRICS)
    widget.set_filters(
        [
            new_filter(MODULATION, 0.1, 0.9),
            new_filter(FRET_EFFICIENCY, 0.2, 0.8),
        ]
    )
    mine, foreign = (widget._cards[f['id']] for f in widget.filters())
    assert mine.range_slider.isEnabled()
    assert not foreign.range_slider.isEnabled()
    # Switching it off or deleting it stays possible from either tab.
    assert foreign.enabled_check.isEnabled()
    assert foreign.remove_button.isEnabled()


def test_filter_list_bounds_only_widen(qtbot):
    """A card's slider always still spans the range the criterion asks for."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget.set_filters([new_filter(MODULATION, -2.0, 5.0)])
    card = next(iter(widget._cards.values()))
    widget.set_metric_bounds(MODULATION, 0.0, 1.0)
    assert card.range_slider.minimum() <= -2 * card.scale
    assert card.range_slider.maximum() >= 5 * card.scale

    widget.set_metric_bounds(MODULATION, None, 1.0)
    widget.set_metric_bounds(MODULATION, np.nan, 1.0)
    assert widget.bounds_for(MODULATION) == (0.0, 1.0)
    assert widget.bounds_for(PHASE) == metric_fallback_range(PHASE)


def test_card_toggles_and_mode_publish_immediately(qtbot):
    """Enable and keep/exclude take effect on the click, not on a later apply."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget._on_add_clicked()
    published = []
    widget.filtersChanged.connect(published.append)
    card = next(iter(widget._cards.values()))

    card.enabled_check.setChecked(False)
    assert published[-1][0]['enabled'] is False

    card.mode_combobox.setCurrentIndex(1)
    assert published[-1][0]['mode'] == EXCLUDE
    card.mode_combobox.setCurrentIndex(0)
    assert published[-1][0]['mode'] == KEEP


def test_card_slider_defers_until_the_value_settles(qtbot):
    """Dragging updates the boxes at once but publishes once, when released."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget.set_metric_bounds(MODULATION, 0.0, 1.0)
    widget.set_current_metric(MODULATION)
    widget._on_add_clicked()
    published = []
    widget.filtersChanged.connect(published.append)
    card = next(iter(widget._cards.values()))

    card._on_slider_changed((250, 750))
    assert card.min_edit.text() == "0.25"
    assert published == []

    card._commit()
    assert published[-1][0]['min'] == pytest.approx(0.25)


def test_card_rejects_unparsable_text_and_reorders_a_backwards_range(qtbot):
    """Typed nonsense is undone; a reversed range is silently swapped."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget.set_metric_bounds(MODULATION, 0.0, 1.0)
    widget.set_current_metric(MODULATION)
    widget._on_add_clicked()
    card = next(iter(widget._cards.values()))

    card.min_edit.setText("not a number")
    card._on_edits_changed()
    assert card.min_edit.text() == "0.00"

    card.min_edit.setText("0.90")
    card.max_edit.setText("0.10")
    card._on_edits_changed()
    assert (card.entry['min'], card.entry['max']) == (0.10, 0.90)


def test_typed_range_moves_the_handles_and_keeps_the_slider_range(qtbot):
    """Typing a range moves the handles; the slider widens only if needed."""
    widget = ComponentFilterList()
    qtbot.addWidget(widget)
    widget.set_components([(0, "A", "#ff0000")])
    widget.set_component_bounds(0, 0.0, 1.0)
    card = widget._cards[0]
    slider = card.range_slider

    card.min_edit.setText("0.30")
    card.max_edit.setText("0.60")
    card.min_edit.editingFinished.emit()
    assert (slider.minimum(), slider.maximum()) == (0, 1000)
    assert slider.value() == (300, 600)

    card.max_edit.setText("1.50")
    card.max_edit.editingFinished.emit()
    assert (slider.minimum(), slider.maximum()) == (0, 1500)
    assert slider.value() == (300, 1500)


def test_card_ignores_edits_while_the_list_is_writing_to_it(qtbot):
    """Programmatic updates do not echo back as user edits."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget._on_add_clicked()
    card = next(iter(widget._cards.values()))
    published = []
    widget.filtersChanged.connect(published.append)

    card._updating = True
    card._on_slider_changed((0, 10))
    card._on_edits_changed()
    card._updating = False
    assert published == []


def test_card_bounds_survive_a_degenerate_range(qtbot):
    """A zero-width or non-finite range still produces a usable slider."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget.set_filters([new_filter(MODULATION, 0.5, 0.5)])
    card = next(iter(widget._cards.values()))
    card.set_bounds(0.5, 0.5)
    assert card.range_slider.maximum() > card.range_slider.minimum()
    card.set_bounds(np.nan, np.nan)
    assert card.range_slider.maximum() > card.range_slider.minimum()


def test_apply_layer_filters_reads_the_harmonics_off_the_layer():
    """The harmonic axis is resolved from the layer when none is passed in."""
    layer = _layer(multi=True)
    set_filters(layer, [new_filter(MODULATION, 0.0, 0.3, harmonic=2)])
    mean, real, _ = apply_layer_filters(
        layer,
        (
            layer.data.copy(),
            layer.metadata['G'].copy(),
            layer.metadata['S'].copy(),
        ),
    )
    assert np.isnan(mean).any()
    assert np.array_equal(np.isnan(real[0]), np.isnan(mean))


def test_set_editable_metrics_applies_to_the_cards_already_shown(qtbot):
    """Restricting editing after the fact re-styles the existing cards."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget.set_filters(
        [
            new_filter(MODULATION, 0.1, 0.9),
            new_filter(FRET_EFFICIENCY, 0.2, 0.8),
        ]
    )
    widget.set_editable_metrics([FRET_EFFICIENCY])
    mine, foreign = (widget._cards[f['id']] for f in widget.filters())
    assert not mine.range_slider.isEnabled()
    assert foreign.range_slider.isEnabled()


def test_card_changed_for_an_unknown_id_is_ignored(qtbot):
    """A signal from a card that has already gone away changes nothing."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget._on_add_clicked()
    published = []
    widget.filtersChanged.connect(published.append)
    widget._on_card_changed("no-such-card")
    assert published == []


def test_a_switched_off_card_reads_as_switched_off(qtbot):
    """Muting is what tells a filter that hides nothing from one that does."""
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget._on_add_clicked()
    card = next(iter(widget._cards.values()))
    assert card.property("muted") is False

    card.enabled_check.setChecked(False)
    assert card.property("muted") is True

    card.enabled_check.setChecked(True)
    assert card.property("muted") is False

    # A read-only card stays muted whether or not it is enabled.
    card.setEditable(False)
    assert card.property("muted") is True
    card.enabled_check.setChecked(False)
    assert card.property("muted") is True


# ------------------------------------------------------- component fractions


def _projection_params(index=0, **overrides):
    """Return the parameters of a two-component linear-projection criterion."""
    params = {
        'analysis_type': 'Linear Projection',
        'component_index': index,
        'component_name': f"Component {index + 1}",
        'component_real': [0.1, 0.9],
        'component_imag': [0.05, 0.45],
        'harmonics': [1],
    }
    params.update(overrides)
    return params


def _fit_params(index=0, **overrides):
    """Return the parameters of a two-component single-harmonic fit."""
    params = {
        'analysis_type': 'Component Fit',
        'component_index': index,
        'component_name': f"Component {index + 1}",
        'component_real': [0.1, 0.9],
        'component_imag': [0.05, 0.45],
        'harmonics': [1],
    }
    params.update(overrides)
    return params


def _arrays(layer):
    """Return ``(mean, real, imag)`` straight off a test layer."""
    return (
        np.asarray(layer.data, dtype=float),
        layer.metadata['G'],
        layer.metadata['S'],
    )


def test_component_fraction_is_offered_like_any_other_metric():
    """The new metric carries the same metadata as the ones before it."""
    assert COMPONENT_FRACTION in ALL_METRICS
    assert COMPONENT_FRACTION not in MAPPING_METRICS
    assert metric_unit(COMPONENT_FRACTION) == ""
    assert metric_fallback_range(COMPONENT_FRACTION) == (0.0, 1.0)
    assert not requires_frequency(COMPONENT_FRACTION)


def test_a_fraction_criterion_is_listed_under_its_component():
    """Three fractions on one layer must not all read the same."""
    assert filter_display_name(new_filter(MODULATION, 0, 1)) == MODULATION
    named = new_filter(
        COMPONENT_FRACTION, 0, 1, params={'component_name': "Free NADH"}
    )
    assert filter_display_name(named) == "Free NADH"
    numbered = new_filter(
        COMPONENT_FRACTION, 0, 1, params={'component_index': 2}
    )
    assert filter_display_name(numbered) == "Component 3"
    bare = new_filter(COMPONENT_FRACTION, 0, 1)
    assert filter_display_name(bare) == COMPONENT_FRACTION
    assert filter_display_name({}) == ""
    assert "Free NADH" in describe_filters([named])


def test_linear_projection_fraction_matches_phasorpy():
    """The metric is phasorpy's own projection, not a re-derivation of it."""
    from phasorpy.component import phasor_component_fraction

    layer = _layer()
    _mean, real, imag = _arrays(layer)
    expected = phasor_component_fraction(real, imag, [0.1, 0.9], [0.05, 0.45])
    values = compute_metric(
        COMPONENT_FRACTION, real, imag, params=_projection_params(0)
    )
    np.testing.assert_allclose(values, expected)

    # The second component is whatever the first one does not account for.
    other = compute_metric(
        COMPONENT_FRACTION, real, imag, params=_projection_params(1)
    )
    np.testing.assert_allclose(other, 1.0 - np.asarray(expected))


@pytest.mark.parametrize(
    "params",
    [
        {},
        _projection_params(component_real=None),
        _projection_params(component_imag=[]),
        _projection_params(component_index="second"),
        # A projection only ever yields two fractions.
        _projection_params(index=2),
    ],
)
def test_an_unusable_fraction_criterion_computes_nothing(params):
    """A criterion that cannot be evaluated is skipped, not guessed at."""
    layer = _layer()
    _mean, real, imag = _arrays(layer)
    assert (
        compute_metric(COMPONENT_FRACTION, real, imag, params=params) is None
    )


def test_component_fit_fraction_matches_phasorpy():
    """A fit criterion returns that component's map out of the whole fit."""
    from phasorpy.component import phasor_component_fit

    layer = _layer()
    mean, real, imag = _arrays(layer)
    context = MetricContext(mean, real, imag, layer.metadata['harmonics'])
    expected = phasor_component_fit(mean, real, imag, [0.1, 0.9], [0.05, 0.45])
    for index in (0, 1):
        values = compute_metric(
            COMPONENT_FRACTION,
            real,
            imag,
            params=_fit_params(index),
            context=context,
        )
        np.testing.assert_allclose(values, expected[index])

    # A component the fit never produced has no map.
    assert (
        compute_metric(
            COMPONENT_FRACTION,
            real,
            imag,
            params=_fit_params(7),
            context=context,
        )
        is None
    )


def test_a_fit_is_computed_once_for_every_component_filtered_on():
    """One fit yields every fraction, so three criteria must not fit thrice."""
    import napari_phasors._mapping_filters as module

    layer = _layer()
    mean, real, imag = _arrays(layer)
    calls = []
    original = module.phasor_component_fit

    def counting_fit(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    module.phasor_component_fit = counting_fit
    try:
        mask = combined_mask(
            [
                new_filter(
                    COMPONENT_FRACTION, 0.1, 0.9, params=_fit_params(0)
                ),
                new_filter(
                    COMPONENT_FRACTION, 0.1, 0.9, params=_fit_params(1)
                ),
            ],
            mean,
            real,
            imag,
            layer.metadata['harmonics'],
        )
    finally:
        module.phasor_component_fit = original

    assert mask is not None
    assert len(calls) == 1


def test_a_failing_fit_skips_the_criterion():
    """A fit that cannot run hides no pixels and raises nothing."""
    import napari_phasors._mapping_filters as module

    layer = _layer()
    mean, real, imag = _arrays(layer)
    original = module.phasor_component_fit

    def exploding_fit(*args, **kwargs):
        raise ValueError("no fit here")

    module.phasor_component_fit = exploding_fit
    try:
        problems = []
        mask = combined_mask(
            [new_filter(COMPONENT_FRACTION, 0.1, 0.9, params=_fit_params(0))],
            mean,
            real,
            imag,
            layer.metadata['harmonics'],
            on_error=problems.append,
        )
    finally:
        module.phasor_component_fit = original

    assert mask is None
    assert problems and "Component 1" in problems[0]


def test_a_single_map_fit_is_read_as_one_component():
    """phasorpy returning one array, not a sequence, still names component 1."""
    import napari_phasors._mapping_filters as module

    layer = _layer()
    mean, real, imag = _arrays(layer)
    original = module.phasor_component_fit
    module.phasor_component_fit = lambda m, r, i, cg, cs: np.zeros_like(m)
    try:
        values = compute_metric(
            COMPONENT_FRACTION,
            real,
            imag,
            params=_fit_params(0),
            context=MetricContext(mean, real, imag, None),
        )
    finally:
        module.phasor_component_fit = original
    np.testing.assert_allclose(values, np.zeros_like(mean))


def test_a_fit_criterion_without_its_arrays_is_skipped():
    """A fit needs the mean image; a plane on its own is not enough."""
    layer = _layer()
    mean, real, imag = _arrays(layer)
    assert (
        compute_metric(COMPONENT_FRACTION, real, imag, params=_fit_params(0))
        is None
    )
    assert (
        compute_metric(
            COMPONENT_FRACTION,
            real,
            imag,
            params=_fit_params(0),
            context=MetricContext(None, real, imag, None),
        )
        is None
    )
    assert (
        compute_metric(
            COMPONENT_FRACTION,
            real,
            imag,
            params=_fit_params(0),
            context=MetricContext(mean, None, None, None),
        )
        is None
    )


def test_a_two_harmonic_fit_needs_two_harmonics_in_the_data():
    """A criterion the layer can no longer support hides nothing."""
    single = _layer()
    mean, real, imag = _arrays(single)
    params = _fit_params(
        0,
        harmonics=[1, 2],
        component_real=[[0.1, 0.9], [0.05, 0.45]],
        component_imag=[[0.05, 0.45], [0.02, 0.2]],
    )
    assert (
        compute_metric(
            COMPONENT_FRACTION,
            real,
            imag,
            params=params,
            context=MetricContext(mean, real, imag, None),
        )
        is None
    )
    # An unreadable harmonic number is treated the same way.
    assert (
        compute_metric(
            COMPONENT_FRACTION,
            real,
            imag,
            params=_fit_params(0, harmonics=["first"]),
            context=MetricContext(mean, real, imag, None),
        )
        is None
    )


def test_a_two_harmonic_fit_uses_both_planes():
    """With both harmonics present the criterion evaluates on the stack."""
    layer = _layer(multi=True)
    mean, real, imag = _arrays(layer)
    params = _fit_params(
        0,
        harmonics=[1, 2],
        component_real=[[0.1, 0.9], [0.05, 0.45]],
        component_imag=[[0.05, 0.45], [0.02, 0.2]],
    )
    values = compute_metric(
        COMPONENT_FRACTION,
        real[0],
        imag[0],
        params=params,
        context=MetricContext(mean, real, imag, layer.metadata['harmonics']),
    )
    assert values is not None
    assert values.shape == mean.shape


def test_a_fraction_criterion_blanks_the_pixels_outside_its_range():
    """The whole point: filtered pixels lose their phasor coordinates."""
    layer = _layer()
    criterion = new_filter(
        COMPONENT_FRACTION, 0.0, 0.4, params=_projection_params(0)
    )
    set_filters(layer, [criterion])
    mask = rebuild_layer_from_filters(layer)
    assert mask is not None and mask.any()
    assert np.isnan(layer.data[mask]).all()
    assert np.isfinite(layer.data[~mask]).all()

    # Removing it restores exactly the pixels it had hidden.
    set_filters(layer, [])
    rebuild_layer_from_filters(layer)
    assert np.isfinite(layer.data).all()


# ------------------------------------------------- the component filter list


def _component_list(
    qtbot,
    components=((0, "Component 1", "#ff00ff"), (1, "Component 2", "#00ffff")),
):
    """Return a list widget already showing *components*."""
    widget = ComponentFilterList()
    qtbot.addWidget(widget)
    widget.set_params_provider(lambda index: _projection_params(index))
    widget.set_harmonic_provider(lambda: 2)
    widget.set_components(list(components))
    return widget


def test_an_untouched_component_card_is_not_a_filter(qtbot):
    """Defining components must not by itself write filters onto a layer."""
    widget = _component_list(qtbot)
    assert sorted(widget._cards) == [0, 1]
    assert widget.filters() == []
    assert not widget.empty_label.isVisible()

    empty = ComponentFilterList()
    qtbot.addWidget(empty)
    empty.set_components([])
    assert empty.components() == []
    assert empty.summary_label.text() == ""


def test_editing_a_component_card_publishes_the_criterion(qtbot):
    """A card becomes a real criterion the moment the user touches it."""
    widget = _component_list(qtbot)
    published = []
    widget.filtersChanged.connect(published.append)

    card = widget._cards[1]
    card.enabled_check.setChecked(True)

    assert len(published) == 1
    (entry,) = published[-1]
    assert entry['metric'] == COMPONENT_FRACTION
    assert entry['harmonic'] == 2
    assert entry['enabled'] is True
    assert entry['params']['component_index'] == 1
    assert entry['params']['component_name'] == "Component 2"
    assert widget.filters() == published[-1]

    # The other card is still a placeholder.
    assert len(widget.filters()) == 1


def test_a_card_picks_up_its_component_on_its_first_edit(qtbot):
    """A criterion made before the analysis ran is completed when used."""
    widget = ComponentFilterList()
    qtbot.addWidget(widget)
    provider = {'params': {}}
    widget.set_params_provider(lambda index: provider['params'])
    widget.set_components([(0, "Component 1", None)])
    assert widget._cards[0].entry['params'].get('component_real') is None

    provider['params'] = _projection_params(0)
    widget._cards[0].enabled_check.setChecked(True)
    stored = widget.filters()[0]
    assert stored['params']['component_real'] == [0.1, 0.9]
    assert stored['params']['component_index'] == 0


def test_a_signal_from_a_vanished_card_changes_nothing(qtbot):
    """A card that has already gone away cannot publish a criterion."""
    widget = _component_list(qtbot)
    published = []
    widget.filtersChanged.connect(published.append)
    widget._on_card_changed("no-such-card")
    assert published == []

    # An entry with no card of its own is ignored too.
    widget._entries[9] = new_filter(COMPONENT_FRACTION, 0, 1)
    widget._on_card_changed(widget._entries[9]['id'])
    assert published == []


def test_the_list_follows_the_components(qtbot):
    """Renaming keeps a criterion, removing takes it away."""
    widget = _component_list(qtbot)
    widget._cards[0].enabled_check.setChecked(True)
    assert widget.filters()[0]['params']['component_name'] == "Component 1"

    widget.set_components([(0, "Free NADH", "#ff00ff"), (1, "Bound", "#0f0")])
    assert widget._cards[0].metric_label.text() == "Free NADH Filter"
    assert widget.filters()[0]['params']['component_name'] == "Free NADH"

    # An unchanged set of components leaves the cards exactly as they are.
    card = widget._cards[0]
    widget.set_components([(0, "Free NADH", "#ff00ff"), (1, "Bound", "#0f0")])
    assert widget._cards[0] is card

    widget.set_components([(0, "Free NADH", "#ff00ff")])
    assert sorted(widget._cards) == [0]
    assert len(widget.filters()) == 1


def test_stored_criteria_are_adopted_under_the_current_names(qtbot):
    """A rename that has not reached the layer yet still shows on the card."""
    widget = _component_list(qtbot)
    stored = new_filter(
        COMPONENT_FRACTION,
        0.2,
        0.6,
        params=_projection_params(1, component_name="Old name"),
    )
    widget.set_filters([stored, new_filter(MODULATION, 0, 1)])

    assert widget._cards[1].metric_label.text() == "Component 2 Filter"
    (adopted,) = widget.filters()
    assert adopted['id'] == stored['id']
    assert adopted['min'] == 0.2
    assert adopted['params']['component_name'] == "Component 2"

    # Handing the same stack back is a no-op, so a card being dragged is not
    # rebuilt under the pointer.
    card = widget._cards[1]
    widget.set_filters([stored])
    assert widget._cards[1] is card

    # A criterion that is gone turns its card back into a placeholder.
    widget.set_filters([])
    assert widget.filters() == []
    assert widget._cards[1] is not card


def test_criteria_without_a_component_index_are_ignored(qtbot):
    """Only a criterion that names its component can be shown on a card."""
    widget = _component_list(qtbot)
    widget.set_filters([new_filter(COMPONENT_FRACTION, 0, 1)])
    assert widget.filters() == []


def test_component_card_stats_and_summary(qtbot):
    """Each card reports what it keeps, and the list what they keep together."""
    widget = _component_list(qtbot)
    widget._cards[0].enabled_check.setChecked(True)
    entry = widget.filters()[0]

    widget.set_filter_stats(
        {entry['id']: "keeps 40.0% of the pixels"},
        "1 of 1 on · 40.0% kept",
        detail="Something longer",
    )
    assert widget._cards[0].stat_label.text() == "keeps 40.0% of the pixels"
    assert widget._cards[1].stat_label.text() == ""
    assert widget.summary_label.text() == "1 of 1 on · 40.0% kept"
    assert "Something longer" in widget.summary_label.toolTip()

    widget.set_filter_stats({})
    assert widget._cards[0].stat_label.text() == ""


def test_a_component_filter_can_be_blocked_from_being_switched_on(qtbot):
    """A criterion that cannot be evaluated says so on the card itself."""
    widget = _component_list(qtbot)
    widget.set_enable_blocked("Run the analysis first.")
    card = widget._cards[0]
    assert not card.enabled_check.isEnabled()
    assert card.enabled_check.toolTip() == "Run the analysis first."

    widget.set_enable_blocked(None)
    assert card.enabled_check.isEnabled()


def test_refresh_params_follows_the_components(qtbot):
    """A criterion tests the fraction the user can see, not an older one."""
    widget = ComponentFilterList()
    qtbot.addWidget(widget)
    current = {'params': _projection_params(0)}
    widget.set_params_provider(lambda index: current['params'])
    widget.set_components([(0, "Component 1", None)])
    widget._cards[0].enabled_check.setChecked(True)

    assert widget.refresh_params() is False

    current['params'] = _projection_params(0, component_real=[0.3, 0.7])
    assert widget.refresh_params() is True
    assert widget.filters()[0]['params']['component_real'] == [0.3, 0.7]

    # Without a provider there is nothing to refresh from.
    widget.set_params_provider(None)
    assert widget.refresh_params() is False


def test_a_component_card_is_tinted_and_cannot_be_removed(qtbot):
    """The card reads as the component it belongs to, and stays put."""
    widget = _component_list(qtbot)
    card = widget._cards[0]
    assert not card.remove_button.isVisible()
    assert "#ff00ff" in card.metric_label.styleSheet()
    assert COMPONENT_FRACTION in card.metric_label.toolTip()

    card.set_accent_color(None)
    assert "color:" not in card.metric_label.styleSheet()


def test_unknown_components_fall_back_to_their_number(qtbot):
    """A stale index still names something rather than raising."""
    widget = _component_list(qtbot)
    assert widget._name_for(7) == "Component 8"
    assert widget._color_for(7) is None
    assert widget._color_for(0) == "#ff00ff"


def test_component_positions_survive_an_array_round_trip():
    """Positions read back as arrays key the fit cache just as lists do."""
    layer = _layer()
    mean, real, imag = _arrays(layer)
    params = _fit_params(
        0,
        component_real=np.array([0.1, 0.9]),
        component_imag=np.array([0.05, 0.45]),
    )
    values = compute_metric(
        COMPONENT_FRACTION,
        real,
        imag,
        params=params,
        context=MetricContext(mean, real, imag, None),
    )
    assert values is not None and values.shape == mean.shape


def test_a_fit_criterion_on_mismatched_arrays_is_skipped():
    """Phasor coordinates that are not the shape of the image fit nothing."""
    mean = np.ones((4, 4))
    real = np.ones((3, 3))
    assert (
        compute_metric(
            COMPONENT_FRACTION,
            real,
            real,
            params=_fit_params(0),
            context=MetricContext(mean, real, real, None),
        )
        is None
    )


def test_ragged_component_positions_are_no_positions_at_all():
    """Hand-edited settings must not raise out of the redraw that read them."""
    from napari_phasors._mapping_filters import has_component_positions

    assert not has_component_positions(None)
    assert not has_component_positions({})
    assert not has_component_positions({'component_real': [0.1, 0.9]})
    assert not has_component_positions(
        {'component_real': [], 'component_imag': []}
    )
    assert not has_component_positions(
        {
            'component_real': [[0.1, 0.9], [0.2]],
            'component_imag': [[0.05, 0.45], [0.1]],
        }
    )
    assert has_component_positions(
        {'component_real': [0.1, 0.9], 'component_imag': [0.05, 0.45]}
    )


# ------------------------------------------- refreshes during a rebuild


def test_a_list_caught_mid_rebuild_refreshes_instead_of_raising(qtbot):
    """Applying a stack pumps the event loop, so a refresh can land early.

    Regression: a refresh that arrived while the cards were being rebuilt
    paired criteria with cards that did not exist yet and raised a KeyError
    out of the redraw that asked for it.
    """
    widget = MappingFilterList(MAPPING_METRICS)
    qtbot.addWidget(widget)
    widget._on_add_clicked()
    entry = widget.filters()[0]

    # Exactly the state a rebuild is in between clearing the cards and
    # recreating them.
    widget._cards = {}

    assert widget._card_pairs() == []
    widget.set_filter_stats({entry['id']: "keeps 50.0% of the pixels"}, "1 on")
    widget.set_metric_bounds(entry['metric'], 0.0, 5.0)
    widget.set_editable_metrics(MAPPING_METRICS)
    assert widget.summary_label.text() == "1 on"

    # Once the rebuild finishes the card is paired again.
    widget._rebuild_cards()
    assert [e['id'] for e, _card in widget._card_pairs()] == [entry['id']]
    widget.set_filter_stats({entry['id']: "keeps 50.0% of the pixels"})
    card = widget._cards[entry['id']]
    assert card.stat_label.text() == "keeps 50.0% of the pixels"


def test_a_nested_apply_is_replayed_rather_than_interleaved():
    """The user's next edit must not rebuild on top of the one in flight."""

    class Tab:
        def __init__(self):
            self.applied = []

        @serialize_filter_applies
        def _apply_filter_stack(self, filters=None, layers=None):
            self.applied.append(filters)
            if filters == "first":
                # What a queued editingFinished does once the long apply
                # spins the event loop.
                self._apply_filter_stack("second")
                self._apply_filter_stack("third")

    tab = Tab()
    tab._apply_filter_stack("first")
    # The outer apply ran once, and only the last interrupting edit was
    # replayed after it — not both, and not inside it.
    assert tab.applied == ["first", "third"]

    # Nothing pending leaves the next apply alone.
    tab.applied.clear()
    tab._apply_filter_stack("later")
    assert tab.applied == ["later"]


def test_one_context_serves_a_whole_refresh():
    """Measuring what each criterion keeps must not refit once per criterion."""
    import napari_phasors._mapping_filters as module

    layer = _layer()
    mean, real, imag = _arrays(layer)
    filters = [
        new_filter(COMPONENT_FRACTION, 0.1, 0.9, params=_fit_params(0)),
        new_filter(COMPONENT_FRACTION, 0.1, 0.9, params=_fit_params(1)),
    ]
    context = MetricContext(mean, real, imag, layer.metadata['harmonics'])

    calls = []
    original = module.phasor_component_fit

    def counting_fit(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    module.phasor_component_fit = counting_fit
    try:
        for entry in filters:
            combined_mask(
                [entry],
                mean,
                real,
                imag,
                layer.metadata['harmonics'],
                context=context,
            )
        combined_mask(
            filters,
            mean,
            real,
            imag,
            layer.metadata['harmonics'],
            context=context,
        )
    finally:
        module.phasor_component_fit = original

    assert len(calls) == 1


# --------------------------------------------- component fraction ranges


def test_a_component_card_opens_on_the_measured_fraction_range(qtbot):
    """A fit is unconstrained, so 0-1 is an assumption, not a fact."""
    widget = ComponentFilterList()
    qtbot.addWidget(widget)
    widget.set_params_provider(lambda index: _fit_params(index))
    widget.set_bounds_provider(lambda index: (-1.4, 2.1))
    widget.set_components([(0, "Component 1", None)])

    assert widget.bounds_for(0) == (-1.4, 2.1)
    entry = widget._entries[0]
    assert entry['min'] == pytest.approx(-1.4)
    assert entry['max'] == pytest.approx(2.1)
    card = widget._cards[0]
    assert card.range_slider.minimum() / card.scale == pytest.approx(-1.4)
    assert card.range_slider.maximum() / card.scale == pytest.approx(2.1)


def test_component_card_bounds_can_be_widened_after_the_fact(qtbot):
    """A re-run that moves the fractions moves the range they are picked from."""
    widget = ComponentFilterList()
    qtbot.addWidget(widget)
    widget.set_components([(0, "Component 1", None)])
    # Nothing measured: the 0-1 fallback, as a projection would give.
    assert widget.bounds_for(0) == (0.0, 1.0)

    widget.set_component_bounds(0, -0.5, 1.8)
    card = widget._cards[0]
    assert widget.bounds_for(0) == (-0.5, 1.8)
    assert card.range_slider.minimum() / card.scale == pytest.approx(-0.5)
    assert card.range_slider.maximum() / card.scale == pytest.approx(1.8)

    # Unmeasurable bounds are ignored rather than collapsing the slider.
    widget.set_component_bounds(0, None, 1.0)
    widget.set_component_bounds(0, np.nan, np.inf)
    widget.set_component_bounds(9, 0.0, 1.0)
    assert widget.bounds_for(0) == (-0.5, 1.8)

    # A provider that cannot measure falls back to what was recorded.
    widget.set_bounds_provider(lambda index: None)
    assert widget._seed_range(0) == (-0.5, 1.8)
    widget.set_bounds_provider(lambda index: (np.nan, 1.0))
    assert widget._seed_range(0) == (-0.5, 1.8)


def test_refresh_params_keeps_a_criterion_it_cannot_re_read(qtbot):
    """Positions that cannot be read now must not overwrite working ones."""
    widget = ComponentFilterList()
    qtbot.addWidget(widget)
    current = {'params': _projection_params(0)}
    widget.set_params_provider(lambda index: current['params'])
    widget.set_components([(0, "Component 1", None)])
    widget._cards[0].enabled_check.setChecked(True)
    assert widget.filters()[0]['params']['component_real'] == [0.1, 0.9]

    # The components are gone from the tab, so the provider can only report
    # the identifying keys. Adopting those would leave the criterion hiding
    # pixels by a rule it could no longer evaluate.
    current['params'] = {'component_index': 0, 'component_name': "Component 1"}
    assert widget.refresh_params() is False
    assert widget.filters()[0]['params']['component_real'] == [0.1, 0.9]
