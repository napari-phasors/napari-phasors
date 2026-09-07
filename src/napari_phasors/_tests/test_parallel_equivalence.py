"""Parallel and sequential must produce the same answer, everywhere.

Every helper in :mod:`napari_phasors._parallel` exists on the promise that
turning parallelism on changes only how long something takes. This module is
where that promise is checked, entry point by entry point: each test runs the
same call twice -- once with the plugin-wide switch off, once with it on --
and asserts the two results are *identical*, not merely close. Identical
means equal values, equal dtypes, and NaNs in exactly the same places.

Two fixtures make the matrix affordable:

``force_split`` drops :data:`MIN_PARALLEL_PIXELS` to one and pins the worker
count, so a 240-row test array takes the band-splitting path that a real
image only reaches above a megapixel. Without it the small arrays these
tests use would quietly skip the very code they are meant to cover, and the
matrix would pass while testing nothing.

``split_spy`` closes the same hole from the other side: it counts how often
the band helpers really did cut something up, so a test can assert that the
parallel run was parallel rather than passing because both runs quietly took
the same sequential path.
"""

import contextlib
import itertools
import os

import numpy as np
import pytest
from napari.layers import Image

from napari_phasors import _parallel
from napari_phasors._parallel import (
    parallel_filter_median,
    parallel_phasor_from_signal,
    parallel_rowwise,
)

# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


@contextlib.contextmanager
def parallelism(enabled):
    """Run the block with the plugin-wide parallel switch in a known state."""
    previous = _parallel.parallel_enabled()
    _parallel.set_parallel_enabled(enabled)
    try:
        yield
    finally:
        _parallel.set_parallel_enabled(previous)


@pytest.fixture(autouse=True)
def clear_worker_env(monkeypatch):
    """A ``NAPARI_PHASORS_WORKERS`` in the environment would rewrite the matrix."""
    monkeypatch.delenv("NAPARI_PHASORS_WORKERS", raising=False)


@pytest.fixture(autouse=True)
def restore_parallel_switch():
    """Leave the process-wide switch exactly as this module found it."""
    previous = _parallel.parallel_enabled()
    previous_fraction = _parallel.memory_fraction()
    yield
    _parallel.set_parallel_enabled(previous)
    _parallel.set_memory_fraction(previous_fraction)


@pytest.fixture
def force_split(monkeypatch):
    """Make small arrays take the band-splitting path, on any machine.

    Splitting normally waits for a megapixel and for more than one core.
    Both are pinned here so the matrix below runs in milliseconds and still
    exercises multi-band, multi-worker code on a single-core CI box.
    """
    monkeypatch.setattr(_parallel, "MIN_PARALLEL_PIXELS", 1)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)


@pytest.fixture
def split_spy(monkeypatch):
    """Count how often the band helpers actually split something.

    Guards against a matrix that passes because both runs silently took the
    same sequential path.
    """
    counter = {"bands": 0, "streams": 0}

    real_bands = _parallel.parallel_bands
    real_stream = _parallel.parallel_stream

    def counting_bands(size, func, **kwargs):
        bounds = _parallel.band_bounds(
            size,
            workers=kwargs.get("workers"),
            halo=kwargs.get("halo", 0),
            min_band=kwargs.get("min_band", 1),
            max_band=kwargs.get("max_band"),
        )
        if len(bounds) > 1:
            counter["bands"] += 1
        return real_bands(size, func, **kwargs)

    def counting_stream(func, items, **kwargs):
        items = list(items)
        if (
            len(items) > 1
            and _parallel.default_workers(len(items), kwargs.get("workers"))
            > 1
        ):
            counter["streams"] += 1
        return real_stream(func, items, **kwargs)

    monkeypatch.setattr(_parallel, "parallel_bands", counting_bands)
    monkeypatch.setattr(_parallel, "parallel_stream", counting_stream)
    return counter


def assert_same(got, want, path="result"):
    """Assert two results are identical, recursing through the containers.

    Stricter than ``allclose`` on purpose: a parallel run that returns the
    right numbers in the wrong dtype, or that loses a NaN, is a bug even
    when the values round-trip.
    """
    if isinstance(want, BaseException) or isinstance(got, BaseException):
        assert type(got) is type(want), f"{path}: {got!r} != {want!r}"
        assert str(got) == str(want), f"{path}: {got!r} != {want!r}"
        return
    if want is None:
        assert got is None, f"{path}: expected None, got {got!r}"
        return
    if isinstance(want, dict):
        assert isinstance(got, dict), f"{path}: expected a dict"
        assert set(got) == set(want), f"{path}: keys differ"
        for key in want:
            assert_same(got[key], want[key], f"{path}[{key!r}]")
        return
    if isinstance(want, (list, tuple)):
        assert type(got) is type(want), f"{path}: container type differs"
        assert len(got) == len(want), f"{path}: length differs"
        for index, (g, w) in enumerate(zip(got, want, strict=True)):
            assert_same(g, w, f"{path}[{index}]")
        return
    if isinstance(want, np.ndarray) or isinstance(got, np.ndarray):
        got_array = np.asarray(got)
        want_array = np.asarray(want)
        assert (
            got_array.dtype == want_array.dtype
        ), f"{path}: dtype {got_array.dtype} != {want_array.dtype}"
        assert (
            got_array.shape == want_array.shape
        ), f"{path}: shape {got_array.shape} != {want_array.shape}"
        assert np.array_equal(
            got_array, want_array, equal_nan=True
        ), f"{path}: values differ"
        return
    if isinstance(want, float) and np.isnan(want):
        assert np.isnan(got), f"{path}: expected NaN, got {got!r}"
        return
    assert got == want, f"{path}: {got!r} != {want!r}"


def both_ways(call):
    """Return ``(sequential, parallel)`` results of the same call.

    The sequential run goes first so a bug in the parallel path cannot
    poison the reference through shared mutable inputs.
    """
    with parallelism(False):
        sequential = call()
    with parallelism(True):
        parallel = call()
    return sequential, parallel


def assert_identical_both_ways(call, path="result"):
    """Run *call* with parallelism off and on and assert the results match."""
    sequential, parallel = both_ways(call)
    assert_same(parallel, sequential, path)
    return sequential


# --------------------------------------------------------------------------
# array fixtures
# --------------------------------------------------------------------------

MEAN_DTYPES = (np.float32, np.float64, np.uint16, np.int32)
GS_DTYPES = (np.float32, np.float64)
LAYOUTS = ("plain", "harmonic", "stack")


def make_phasor_arrays(layout, mean_dtype, gs_dtype, nan_fraction, seed):
    """Return ``(mean, real, imag, skip_axis)`` for one matrix cell.

    ``plain`` is a single 2-D image, ``harmonic`` adds a leading harmonic
    axis to ``real``/``imag`` only, and ``stack`` adds a leading slice axis
    to all three -- the three shapes the filter tab actually hands to
    phasorpy.
    """
    rng = np.random.default_rng(seed)
    if layout == "stack":
        mean_shape, skip_axis = (3, 240, 48), (0,)
    else:
        mean_shape, skip_axis = (240, 48), None
    gs_shape = (2,) + mean_shape if layout == "harmonic" else mean_shape

    mean = (rng.random(mean_shape) * 500).astype(mean_dtype)
    real = rng.random(gs_shape).astype(gs_dtype)
    imag = rng.random(gs_shape).astype(gs_dtype)
    if nan_fraction:
        real[rng.random(gs_shape) < nan_fraction] = np.nan
        imag[rng.random(gs_shape) < nan_fraction] = np.nan
        if np.issubdtype(np.dtype(mean_dtype), np.floating):
            mean[rng.random(mean_shape) < nan_fraction] = np.nan
    return mean, real, imag, skip_axis


# --------------------------------------------------------------------------
# 1. median filter -- the one kernel that needs a halo
# --------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("repeat", (1, 2, 3))
@pytest.mark.parametrize("size", (3, 5, 7))
def test_median_filter_matrix(force_split, split_spy, layout, repeat, size):
    """Every dtype, NaN and layout combination filters to the same bits.

    The halo is ``repeat * (size // 2)`` rows, so this sweep is what proves
    the halo is big enough at every ``repeat``/``size`` the UI allows -- and
    that band boundaries never leak into a kept row.
    """
    for seed, (mean_dtype, gs_dtype, nan_fraction) in enumerate(
        itertools.product(MEAN_DTYPES, GS_DTYPES, (0.0, 0.3)), start=1
    ):
        arrays = make_phasor_arrays(
            layout, mean_dtype, gs_dtype, nan_fraction, seed
        )
        label = (
            f"{layout}/{np.dtype(mean_dtype).name}/"
            f"{np.dtype(gs_dtype).name}/nan={nan_fraction}"
        )

        def filter_this_cell(arrays=arrays):
            mean, real, imag, skip_axis = arrays
            return parallel_filter_median(
                mean,
                real,
                imag,
                repeat=repeat,
                size=size,
                skip_axis=skip_axis,
            )

        assert_identical_both_ways(filter_this_cell, label)
    assert split_spy["bands"] > 0, "no cell actually split into bands"


def test_median_filter_matches_phasorpy_directly(force_split):
    """The parallel path also matches phasorpy itself, not just itself-off."""
    from phasorpy.filter import phasor_filter_median

    for layout, mean_dtype, gs_dtype in itertools.product(
        LAYOUTS, MEAN_DTYPES, GS_DTYPES
    ):
        mean, real, imag, skip_axis = make_phasor_arrays(
            layout, mean_dtype, gs_dtype, 0.25, seed=7
        )
        want = phasor_filter_median(
            mean, real, imag, repeat=2, size=5, skip_axis=skip_axis
        )
        with parallelism(True):
            got = parallel_filter_median(
                mean, real, imag, repeat=2, size=5, skip_axis=skip_axis
            )
        assert_same(got, want, f"{layout}/{np.dtype(mean_dtype).name}")


@pytest.mark.parametrize("size", (1, 2, 4))
@pytest.mark.parametrize("repeat", (0, 1))
def test_median_filter_degenerate_parameters(force_split, size, repeat):
    """A no-op or even-sized kernel behaves the same either way.

    ``size=1`` and ``repeat=0`` both mean "don't filter", and an even
    ``size`` gives a zero halo for ``size=2`` -- the cases where the halo
    rule has nothing to work with and the split must stand down.
    """
    mean, real, imag, _ = make_phasor_arrays(
        "plain", np.float32, np.float32, 0.2, seed=11
    )
    assert_identical_both_ways(
        lambda: parallel_filter_median(
            mean, real, imag, repeat=repeat, size=size
        ),
        f"size={size} repeat={repeat}",
    )


def test_median_filter_rows_fewer_than_the_halo(force_split):
    """An image shorter than one halo cannot be split and must not try."""
    mean, real, imag, _ = make_phasor_arrays(
        "plain", np.float32, np.float32, 0.0, seed=12
    )
    short = (mean[:4], real[:4], imag[:4])
    assert_identical_both_ways(
        lambda: parallel_filter_median(*short, repeat=3, size=7),
        "4 rows",
    )


# --------------------------------------------------------------------------
# 2. phasor transform -- point-wise, no halo
# --------------------------------------------------------------------------


TRANSFORM_CASES = (
    ((64, 40, 24), 0),
    ((40, 64, 24), 1),
    ((40, 24, 64), -1),
    ((40, 24, 64), 2),
    ((2, 64, 30, 24), 1),
)


@pytest.mark.parametrize("shape,axis", TRANSFORM_CASES)
@pytest.mark.parametrize("harmonic", (None, 1, [1, 2], [1, 2, 3]))
@pytest.mark.parametrize("dtype", (np.uint16, np.float32, np.float64))
def test_phasor_transform_matrix(
    force_split, split_spy, shape, axis, harmonic, dtype
):
    """Splitting the signal gives the same phasor for every layout tried.

    The split axis is chosen from the spatial axes, and the results are
    concatenated back on an axis index derived from where the histogram and
    harmonic axes sit -- arithmetic that has to hold for a histogram axis
    that is leading, trailing or in the middle.
    """
    rng = np.random.default_rng(3)
    signal = (rng.random(shape) * 100).astype(dtype)
    assert_identical_both_ways(
        lambda: parallel_phasor_from_signal(
            signal, axis=axis, harmonic=harmonic
        ),
        f"{shape}/axis={axis}/harmonic={harmonic}/{np.dtype(dtype).name}",
    )
    assert split_spy["bands"] > 0


def test_phasor_transform_leaves_unsplittable_input_alone(force_split):
    """A named axis or a non-ndarray carries metadata the bands would lose."""
    import xarray as xr

    rng = np.random.default_rng(4)
    cube = xr.DataArray(
        (rng.random((64, 40, 24)) * 10).astype(np.float32),
        dims=("H", "Y", "X"),
    )
    assert_identical_both_ways(
        lambda: parallel_phasor_from_signal(cube, axis="H", harmonic=1),
        "named axis",
    )
    assert_identical_both_ways(
        lambda: parallel_phasor_from_signal(cube, axis=0, harmonic=1),
        "DataArray with an integer axis",
    )


def test_phasor_transform_two_dimensional_signal(force_split):
    """A signal with a single spatial axis has nothing to split."""
    rng = np.random.default_rng(5)
    signal = (rng.random((64, 400)) * 10).astype(np.float32)
    assert_identical_both_ways(
        lambda: parallel_phasor_from_signal(signal, axis=0, harmonic=[1, 2]),
        "2-D signal",
    )


# --------------------------------------------------------------------------
# 3. component fit and the generic row-wise splitter
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_harmonics,n_components", ((1, 2), (1, 3), (2, 3)))
@pytest.mark.parametrize("nan_fraction", (0.0, 0.3))
def test_component_fit_matrix(
    force_split, split_spy, n_harmonics, n_components, nan_fraction
):
    """Fitting fractions band by band gives the same fractions."""
    from napari_phasors.components_tab import _fit_components

    rng = np.random.default_rng(6)
    shape = (240, 48)
    mean = rng.random(shape) * 100
    real = rng.random((n_harmonics,) + shape) * 0.5 + 0.2
    imag = rng.random((n_harmonics,) + shape) * 0.3 + 0.1
    if nan_fraction:
        real[rng.random(real.shape) < nan_fraction] = np.nan
        imag[rng.random(imag.shape) < nan_fraction] = np.nan
    if n_harmonics == 1:
        real, imag = real[0], imag[0]
        component_g = rng.random(n_components)
        component_s = rng.random(n_components)
    else:
        component_g = rng.random((n_harmonics, n_components))
        component_s = rng.random((n_harmonics, n_components))

    assert_identical_both_ways(
        lambda: _fit_components(mean, real, imag, component_g, component_s),
        f"{n_harmonics}h/{n_components}c/nan={nan_fraction}",
    )
    assert split_spy["bands"] > 0


def test_rowwise_reassembles_every_return_shape(force_split):
    """Single arrays, tuples and leading axes all come back unchanged."""
    rng = np.random.default_rng(8)
    a = rng.random((240, 48))
    b = rng.random((240, 48))
    stacked = rng.random((3, 240, 48))

    assert_identical_both_ways(
        lambda: parallel_rowwise(lambda x, y: x * 2 + y, a, b), "single"
    )
    assert_identical_both_ways(
        lambda: parallel_rowwise(lambda x: (x + 1, x - 1, x * 0), a), "tuple"
    )
    assert_identical_both_ways(
        lambda: parallel_rowwise(lambda x: np.cumsum(x, axis=-1), stacked),
        "leading axes",
    )


# --------------------------------------------------------------------------
# 4. filter + threshold across layers
# --------------------------------------------------------------------------


def make_layer(name, shape=(240, 48), n_harmonics=2, seed=0):
    """Return an Image layer carrying the phasor metadata the plugin expects."""
    rng = np.random.default_rng(seed)
    mean = (rng.random(shape) * 500).astype(np.float32)
    real = rng.random((n_harmonics,) + shape).astype(np.float32)
    imag = rng.random((n_harmonics,) + shape).astype(np.float32)
    real[rng.random(real.shape) < 0.1] = np.nan
    layer = Image(mean.copy(), name=name)
    layer.metadata.update(
        {
            "original_mean": mean,
            "G_original": real,
            "S_original": imag,
            "G": real.copy(),
            "S": imag.copy(),
            "harmonics": np.arange(1, n_harmonics + 1),
            "settings": {},
        }
    )
    return layer


def layer_state(layer):
    """Return the arrays and settings a filter run is expected to write."""
    return {
        "data": np.asarray(layer.data),
        "G": layer.metadata["G"],
        "S": layer.metadata["S"],
        "settings": dict(layer.metadata["settings"]),
    }


FILTER_CASES = (
    {"filter_method": None, "threshold": None},
    {"filter_method": None, "threshold": 100.0},
    {"filter_method": None, "threshold": 100.0, "threshold_upper": 400.0},
    {"filter_method": "median", "size": 3, "repeat": 1, "threshold": 50.0},
    {"filter_method": "median", "size": 5, "repeat": 3, "threshold": None},
    {"filter_method": "median", "size": 7, "repeat": 2, "threshold": 200.0},
    {
        "filter_method": "wavelet",
        "sigma": 2.0,
        "levels": 1,
        "threshold": 50.0,
    },
)


@pytest.mark.parametrize("params", FILTER_CASES)
@pytest.mark.parametrize("n_layers", (1, 3))
def test_filter_and_threshold_layers_matrix(
    force_split, split_spy, params, n_layers
):
    """Filtering N layers in a pool matches filtering them one at a time.

    The sequential reference is the single-layer entry point the filter tab
    used before the fan-out existed, so this also pins the refactor that
    split it into compute and assign halves.
    """
    from napari_phasors._utils import (
        apply_filter_and_threshold,
        apply_filter_and_threshold_to_layers,
    )

    full = dict(params)
    full.setdefault("threshold_upper", None)
    full.setdefault("harmonics", np.array([1, 2]))

    with parallelism(False):
        expected = []
        for index in range(n_layers):
            layer = make_layer(f"seq{index}", seed=index)
            apply_filter_and_threshold(layer, **full)
            expected.append(layer_state(layer))

    with parallelism(True):
        layers = [make_layer(f"par{i}", seed=i) for i in range(n_layers)]
        errors = apply_filter_and_threshold_to_layers(
            [(layer, full) for layer in layers]
        )
        assert not any(isinstance(e, BaseException) for e in errors)
        got = [layer_state(layer) for layer in layers]

    for index, (have, want) in enumerate(zip(got, expected, strict=True)):
        assert_same(have, want, f"layer {index}")


def test_filter_layers_reports_one_failure_without_losing_the_others():
    """A layer that cannot be filtered fails alone, in both modes."""
    from napari_phasors._utils import apply_filter_and_threshold_to_layers

    params = {"filter_method": "median", "size": 3, "repeat": 1}

    def run():
        good_a = make_layer("a", seed=1)
        broken = make_layer("b", seed=2)
        del broken.metadata["G_original"]
        good_b = make_layer("c", seed=3)
        layers = [good_a, broken, good_b]
        errors = apply_filter_and_threshold_to_layers(
            [(layer, params) for layer in layers]
        )
        return [
            type(error).__name__ if isinstance(error, BaseException) else None
            for error in errors
        ], [layer_state(good_a), layer_state(good_b)]

    sequential, parallel = both_ways(run)
    assert sequential[0] == [None, "KeyError", None]
    assert_same(parallel[0], sequential[0], "error slots")
    assert_same(parallel[1], sequential[1], "surviving layers")


# --------------------------------------------------------------------------
# 5. readers and writers
# --------------------------------------------------------------------------


def test_stack_reader_matrix(force_split, monkeypatch):
    """A stack reads to the same arrays however many threads decode it."""
    from napari_phasors import _reader as reader_module

    def fake(path, reader_options=None, harmonics=None):
        index = int(os.path.basename(path).split(".")[0])
        rng = np.random.default_rng(index)
        mean = (rng.random((12, 10)) * 100).astype(np.float32)
        real = rng.random((2, 12, 10)).astype(np.float32)
        imag = rng.random((2, 12, 10)).astype(np.float32)
        meta = {
            "original_mean": mean.copy(),
            "settings": {"channel": 0},
            "summed_signal": np.arange(4) + index,
            "G": real,
            "S": imag,
            "G_original": real.copy(),
            "S_original": imag.copy(),
            "harmonics": [1, 2],
        }
        return [(mean, {"name": "f Intensity Image", "metadata": meta})]

    monkeypatch.setattr(reader_module, "raw_file_reader", fake)
    paths = [f"d/{i}.lsm" for i in range(6)]

    def to_comparable(layers):
        out = []
        for data, kwargs in layers:
            meta = kwargs["metadata"]
            out.append(
                {
                    "data": np.asarray(data),
                    "name": kwargs["name"],
                    "original_mean": meta["original_mean"],
                    "G": meta["G"],
                    "S": meta["S"],
                    "G_original": meta["G_original"],
                    "S_original": meta["S_original"],
                    "stack_files": meta["stack_files"],
                    "summed_signal": meta["summed_signal"],
                }
            )
        return out

    assert_identical_both_ways(
        lambda: to_comparable(reader_module.raw_file_stack_reader(paths)),
        "stack",
    )


@pytest.mark.parametrize(
    "breakage", ("channels", "shape", "phasor_shape", "dtype")
)
def test_stack_reader_rejects_mismatched_files_the_same_way(
    force_split, monkeypatch, breakage
):
    """Every mismatch is caught identically with and without threads."""
    from napari_phasors import _reader as reader_module

    def fake(path, reader_options=None, harmonics=None):
        odd = path.endswith("2.lsm")
        shape = (12, 10)
        harmonics = 2
        dtype = np.float32
        channels = 1
        if odd:
            if breakage == "channels":
                channels = 2
            elif breakage == "shape":
                shape = (11, 10)
            elif breakage == "phasor_shape":
                harmonics = 3
            elif breakage == "dtype":
                dtype = np.float64
        mean = np.ones(shape, dtype=np.float32)
        real = np.ones((harmonics,) + shape, dtype=dtype)
        layers = []
        for _ in range(channels):
            layers.append(
                (
                    mean,
                    {
                        "name": "f Intensity Image",
                        "metadata": {
                            "original_mean": mean.copy(),
                            "settings": {},
                            "summed_signal": None,
                            "G": real,
                            "S": real,
                            "G_original": real,
                            "S_original": real,
                            "harmonics": list(range(1, harmonics + 1)),
                        },
                    },
                )
            )
        return layers

    monkeypatch.setattr(reader_module, "raw_file_reader", fake)
    paths = [f"d/{i}.lsm" for i in range(4)]
    assert_identical_both_ways(
        lambda: reader_module.raw_file_stack_reader(paths), breakage
    )


def test_ome_tiff_export_matrix(force_split, tmp_path):
    """Exported OME-TIFFs carry the same phasor data and the same settings.

    Compared by reading the files back rather than by their bytes: phasorpy
    stamps a fresh UUID into the OME header on every write, so two files
    written from identical arrays are never byte-identical -- not even two
    sequential writes.
    """
    import re

    from phasorpy.io import phasor_from_ometiff

    from napari_phasors._writer import write_ome_tiff

    uuid_attribute = re.compile(r'UUID="[^"]*"')

    def run(subdir):
        target = tmp_path / subdir
        target.mkdir()
        layers = [
            make_layer(f"L{i}", shape=(12, 10), seed=i) for i in range(3)
        ]
        paths = sorted(write_ome_tiff(str(target / "out.ome.tif"), layers))
        read_back = []
        for path in paths:
            mean, real, imag, attrs = phasor_from_ometiff(path)
            read_back.append(
                {
                    "name": os.path.basename(path),
                    "mean": np.asarray(mean),
                    "real": np.asarray(real),
                    "imag": np.asarray(imag),
                    "description": uuid_attribute.sub(
                        'UUID=""', str(attrs.get("description", ""))
                    ),
                }
            )
        return read_back

    with parallelism(False):
        sequential = run("seq")
    with parallelism(True):
        parallel = run("par")
    assert sequential, "nothing was exported"
    assert_same(parallel, sequential, "exported OME-TIFFs")


def test_csv_export_matrix(force_split, tmp_path):
    """CSV exports match too, including which pixels are dropped as NaN."""
    from napari_phasors._writer import export_layer_as_csv

    def run(subdir):
        target = tmp_path / subdir
        target.mkdir()
        layers = [
            make_layer(f"L{i}", shape=(12, 10), seed=i) for i in range(3)
        ]
        paths = export_layer_as_csv(str(target / "out.csv"), layers)
        return [os.path.basename(p) for p in sorted(paths)], [
            (tmp_path / subdir / os.path.basename(p)).read_text()
            for p in sorted(paths)
        ]

    with parallelism(False):
        sequential = run("seq")
    with parallelism(True):
        parallel = run("par")
    assert_same(parallel[0], sequential[0], "written names")
    assert parallel[1] == sequential[1], "file contents differ"


# --------------------------------------------------------------------------
# 6. the analysis tabs
# --------------------------------------------------------------------------


def add_phasor_layers(viewer, n_layers, harmonic=(1, 2)):
    """Add *n_layers* synthetic phasor layers and return their names."""
    from napari_phasors._synthetic_generator import (
        make_intensity_layer_with_phasors,
        make_raw_flim_data,
    )

    names = []
    for index in range(n_layers):
        raw = make_raw_flim_data(shape=(20, 12), n_time_bins=64)
        layer = make_intensity_layer_with_phasors(
            raw, harmonic=list(harmonic), name=f"img{index}"
        )
        viewer.add_layer(layer)
        names.append(layer.name)
    return names


def select_layers(plotter, names):
    """Check *names* in the plotter's layer selector and apply the change now.

    The plotter debounces selection changes behind a timer; the tabs read
    ``get_selected_layers()``, so the timer has to be short-circuited or the
    analysis would run against an empty selection.
    """
    plotter.image_layers_checkable_combobox.setCheckedItems(list(names))
    plotter._layer_selection_timer.stop()
    plotter._process_layer_selection_change()


def analysis_layer_data(viewer, exclude):
    """Return the data of every layer the analysis added, keyed by name."""
    return {
        layer.name: np.asarray(layer.data)
        for layer in viewer.layers
        if layer.name not in exclude
    }


@pytest.mark.parametrize("n_layers", (1, 3))
def test_fret_efficiency_matrix(force_split, make_viewer_model, n_layers):
    """FRET efficiency maps are identical with and without the fan-out."""
    from napari_phasors.plotter import PlotterWidget

    def run():
        viewer = make_viewer_model()
        plotter = PlotterWidget(viewer)
        sources = add_phasor_layers(viewer, n_layers)
        select_layers(plotter, sources)
        widget = plotter.fret_tab
        widget.donor_line_edit.setText("2.0")
        widget.frequency_input.setText("80")
        widget.calculate_fret_efficiency()
        result = analysis_layer_data(viewer, set(sources))
        plotter.deleteLater()
        return result

    sequential, parallel = both_ways(run)
    assert len(sequential) == n_layers, "the FRET run produced no maps"
    assert_same(parallel, sequential, "FRET efficiency")


@pytest.mark.parametrize("n_layers", (1, 3))
@pytest.mark.parametrize(
    "output_type",
    ("Phase", "Modulation", "Normal Lifetime", "Apparent Phase Lifetime"),
)
def test_phasor_mapping_matrix(
    force_split, make_viewer_model, n_layers, output_type
):
    """Every mapping output type matches its sequential run."""
    from napari_phasors.plotter import PlotterWidget

    def run():
        viewer = make_viewer_model()
        plotter = PlotterWidget(viewer)
        sources = add_phasor_layers(viewer, n_layers)
        select_layers(plotter, sources)
        widget = plotter.phasor_mapping_tab
        # Type before frequency: setting the frequency triggers a
        # calculation of its own, so choosing the type first keeps the run
        # from also producing maps of the default type.
        widget.lifetime_type_combobox.setCurrentText(output_type)
        widget.frequency_input.setText("80.0")
        widget._on_frequency_changed()
        widget._on_calculate_lifetime_clicked()
        result = analysis_layer_data(viewer, set(sources))
        plotter.deleteLater()
        return result

    sequential, parallel = both_ways(run)
    assert len(sequential) >= n_layers, f"{output_type} produced no maps"
    assert_same(parallel, sequential, output_type)


@pytest.mark.parametrize("n_layers", (1, 3))
@pytest.mark.parametrize("analysis", ("Linear Projection", "Component Fit"))
def test_components_analysis_matrix(
    force_split, make_viewer_model, n_layers, analysis
):
    """Component fractions match whether the layers are fitted in a pool."""
    from napari_phasors.plotter import PlotterWidget

    def run():
        viewer = make_viewer_model()
        plotter = PlotterWidget(viewer)
        sources = add_phasor_layers(viewer, n_layers)
        select_layers(plotter, sources)
        widget = plotter.components_tab
        plotter.tab_widget.setCurrentWidget(widget)
        widget.analysis_type_combo.setCurrentText(analysis)
        for index, (g, s) in enumerate(((0.2, 0.1), (0.8, 0.5))):
            widget.components[index].g_edit.setText(str(g))
            widget.components[index].s_edit.setText(str(s))
            widget._on_component_coords_changed(index)
        widget._run_analysis()
        result = analysis_layer_data(viewer, set(sources))
        plotter.deleteLater()
        return result

    sequential, parallel = both_ways(run)
    assert len(sequential) >= n_layers, f"{analysis} produced no fraction maps"
    assert_same(parallel, sequential, analysis)
