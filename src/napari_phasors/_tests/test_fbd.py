"""Tests for :mod:`napari_phasors._fbd` and the FBD import widget."""

import os
import warnings
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest
from phasorpy.io import phasor_to_simfcs_referenced
from phasorpy.io import signal_from_fbd as phasorpy_signal_from_fbd

from napari_phasors._fbd import (
    DEFAULT_CANDIDATES,
    IOTECH,
    FbdReconstructionSettings,
    _best_line_start,
    _intensity_image,
    _is_iotech,
    _nominal_dwell_time,
    find_reference_file,
    image_correlation,
    iotech_laser_factor,
    match_reference_settings,
    read_reference_image,
    signal_from_fbd,
)
from napari_phasors._reader import extension_mapping
from napari_phasors._tests.test_data_utils import get_test_file_path
from napari_phasors._widget import FbdWidget, _parse_optional

FBD_FILE = "test_file$EI0S.fbd"
MATCHED_LINE_START = 60
"""Line start used to build the synthetic reference images below."""


@contextmanager
def quiet():
    """Silence the decode warnings fbdfile emits while refining settings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


@pytest.fixture(scope="module")
def fbd_file():
    """Path of the FLIMbox test file."""
    return get_test_file_path(FBD_FILE)


@pytest.fixture(scope="module")
def reference_image(fbd_file):
    """Intensity image of a reconstruction with known settings.

    Built from the file itself so a match is exact and does not depend on
    a reference file that would have to be downloaded.
    """
    with quiet():
        signal = signal_from_fbd(
            fbd_file,
            frame=-1,
            channel=0,
            laser_factor=IOTECH,
            scanner_line_start=MATCHED_LINE_START,
        )
    return np.asarray(signal).sum(-1).astype(np.float64)


@pytest.fixture(scope="module")
def reference_file(tmp_path_factory, reference_image):
    """SimFCS R64 file holding :func:`reference_image`."""
    path = str(tmp_path_factory.mktemp("r64") / "test_file_ch1_h1_h2.r64")
    real = np.full(reference_image.shape, 0.5, dtype=np.float32)
    phasor_to_simfcs_referenced(
        path, reference_image.astype(np.float32), real, real
    )
    return path


class FakeFbdFile:
    """Minimal in-memory stand-in for :class:`fbdfile.FbdFile`.

    Records the ``refine`` value it is asked to reconstruct with, so tests
    can assert what the wrapper forwarded without decoding a real file.
    """

    #: ``(T, C, Y, X, H)`` payload returned by :meth:`asimage`.
    shape = (3, 2, 5, 9, 4)

    def __init__(self, filename, /, *, laser_factor=-1.0, **kwargs):
        self.filename = filename
        self.kwargs = kwargs
        self.refine_calls = []
        self.frame_size = 5
        self.scanner_line_start = kwargs.get("scanner_line_start", 2)
        self.laser_frequency = 40e6
        self.harmonics = 2
        self.pmax = 128
        self.pixel_dwell_time = 32.0
        self.laser_factor = 1.00001 if laser_factor < 0 else laser_factor
        self.header = {
            "laser_factor": 1.00001,
            "line_time": 12928.0,
            "line_length": 404,
        }
        self.fbf = {"firmware": "fake"}
        self.fbs = {"settings": "fake"}
        size = int(np.prod(self.shape))
        self.data = np.arange(size, dtype=np.uint16).reshape(self.shape) % 97

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def asimage(
        self, *, integrate_frames=1, square_frame=True, refine=None, **kwargs
    ):
        """Return the payload, cropped like the real reader would."""
        self.refine_calls.append(refine)
        data = self.data
        if integrate_frames:
            data = data.sum(axis=0, keepdims=True, dtype=np.uint16)
        if square_frame:
            start = self.scanner_line_start
            data = data[
                ..., : self.frame_size, start : start + self.frame_size, :
            ]
        return data


@pytest.fixture
def fake_fbdfile(monkeypatch):
    """Replace ``fbdfile.FbdFile`` with :class:`FakeFbdFile`."""
    import fbdfile

    created = []

    def factory(filename, /, **kwargs):
        fake = FakeFbdFile(filename, **kwargs)
        created.append(fake)
        return fake

    monkeypatch.setattr(fbdfile, "FbdFile", factory)
    return created


# -- signal_from_fbd -------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"frame": -1, "channel": None, "keepdims": False},
        {"frame": -1, "channel": 0},
        {"frame": 0, "channel": 1, "keepdims": True},
    ],
)
def test_matches_phasorpy_by_default(fbd_file, kwargs):
    """Without overrides, output is identical to phasorpy's reader."""
    with quiet():
        ours = signal_from_fbd(fbd_file, **kwargs)
        theirs = phasorpy_signal_from_fbd(fbd_file, **kwargs)
    assert ours.dims == theirs.dims
    assert np.array_equal(ours.values, theirs.values)
    assert ours.attrs["frequency"] == theirs.attrs["frequency"]
    assert ours.attrs["harmonic"] == theirs.attrs["harmonic"]
    assert np.allclose(ours.coords["H"], theirs.coords["H"])


def test_explicit_laser_factor_is_honored(fbd_file):
    """An explicit laser_factor disables refining instead of being lost."""
    factor = 0.995968862745098
    with quiet():
        signal = signal_from_fbd(
            fbd_file, frame=-1, channel=0, laser_factor=factor
        )
        refined = signal_from_fbd(
            fbd_file, frame=-1, channel=0, laser_factor=factor, refine=True
        )
    assert signal.attrs["laser_factor"] == factor
    # phasorpy's reader always refines, so it cannot honor the value
    assert refined.attrs["laser_factor"] != factor
    assert not np.array_equal(signal.values, refined.values)


def test_scanner_line_start_shifts_image(fbd_file):
    """scanner_line_start is forwarded to FbdFile and shifts the image."""
    with quiet():
        default = signal_from_fbd(fbd_file, frame=-1, channel=0)
        shifted = signal_from_fbd(
            fbd_file,
            frame=-1,
            channel=0,
            refine=False,
            scanner_line_start=int(default.attrs["scanner_line_start"]) + 5,
        )
    assert shifted.attrs["scanner_line_start"] == (
        default.attrs["scanner_line_start"] + 5
    )
    assert shifted.shape == default.shape
    assert not np.array_equal(shifted.values, default.values)


def test_out_of_bounds(fbd_file):
    """Frame and channel indices are validated against the decoded data."""
    with quiet():
        with pytest.raises(IndexError, match="channel"):
            signal_from_fbd(fbd_file, channel=99)
        with pytest.raises(IndexError, match="frame"):
            signal_from_fbd(fbd_file, frame=9999)


@pytest.mark.parametrize(
    ("kwargs", "refine"),
    [
        ({}, True),
        ({"laser_factor": 0.5}, False),
        ({"laser_factor": IOTECH}, False),
        ({"refine": None}, None),
        ({"laser_factor": 0.5, "refine": True}, True),
    ],
)
def test_refine_default(fake_fbdfile, kwargs, refine):
    """``refine`` is only defaulted to True without an explicit factor."""
    signal_from_fbd("fake.fbd", **kwargs)
    assert fake_fbdfile[0].refine_calls == [refine]


def test_all_metadata_is_forwarded(fake_fbdfile):
    """Header, firmware, and FBS settings all reach the signal attrs."""
    signal = signal_from_fbd("fake.fbd", frame=-1, channel=0)
    assert signal.attrs["flimbox_header"]["laser_factor"] == 1.00001
    assert signal.attrs["flimbox_firmware"] == {"firmware": "fake"}
    assert signal.attrs["flimbox_settings"] == {"settings": "fake"}
    assert signal.attrs["frequency"] == 40.0
    assert signal.attrs["scanner_line_start"] == 2


@pytest.mark.parametrize(
    ("kwargs", "dims"),
    [
        ({"frame": -1, "channel": None}, ("C", "Y", "X", "H")),
        ({"frame": -1, "channel": None, "keepdims": True}, "TCYXH"),
        ({"frame": -1, "channel": 1, "keepdims": True}, "TCYXH"),
        ({"frame": None, "channel": 1}, ("T", "Y", "X", "H")),
        ({"frame": 1, "channel": None, "keepdims": True}, "TCYXH"),
    ],
)
def test_axes_reduction(fake_fbdfile, kwargs, dims):
    """Reduced axes are dropped, or kept as length-1 with ``keepdims``."""
    signal = signal_from_fbd("fake.fbd", **kwargs)
    assert signal.dims == tuple(dims)


def test_single_channel_file_drops_channel_axis(fake_fbdfile):
    """A one-channel file loses its channel axis when ``channel`` is None."""
    with patch.object(FakeFbdFile, "shape", (1, 1, 5, 9, 4)):
        signal = signal_from_fbd("fake.fbd", frame=-1, channel=None)
    assert signal.dims == ("Y", "X", "H")


# -- iotech_laser_factor ---------------------------------------------------


def test_iotech_laser_factor(fbd_file):
    """Correction factor targets SimFCS's units per scanner sample."""
    from fbdfile import FbdFile

    with quiet():
        with FbdFile(fbd_file) as fbd:
            if fbd.header is None:
                pytest.skip("test file has no binary header")
            factor = iotech_laser_factor(fbd)
            pmax = fbd.pmax
            phase_max = pmax * fbd.harmonics
            nominal_dwell_time = float(fbd.header["line_time"]) / int(
                fbd.header["line_length"]
            )
            expected = (
                float(fbd.header["laser_factor"])
                * (nominal_dwell_time / fbd.pixel_dwell_time)
                * (phase_max / (phase_max - 1))
                / (pmax / (pmax - 1))
            )
        # the resulting units per sample must not depend on which dwell-time
        # table the installed fbdfile release carries
        with FbdFile(fbd_file, laser_factor=factor) as fbd:
            units_per_sample = fbd.units_per_sample
        with FbdFile(fbd_file) as fbd:
            target = (
                nominal_dwell_time
                * 1e-6
                * (phase_max / (phase_max - 1))
                * fbd.laser_frequency
                * float(fbd.header["laser_factor"])
            )
    assert factor == pytest.approx(expected)
    assert units_per_sample == pytest.approx(target)


def test_iotech_sentinel(fbd_file):
    """``laser_factor='iotech'`` derives the factor from the header."""
    from fbdfile import FbdFile

    with quiet():
        with FbdFile(fbd_file) as fbd:
            expected = iotech_laser_factor(fbd)
        signal = signal_from_fbd(
            fbd_file, frame=-1, channel=0, laser_factor=IOTECH
        )
        explicit = signal_from_fbd(
            fbd_file, frame=-1, channel=0, laser_factor=expected
        )
    assert signal.attrs["laser_factor"] == pytest.approx(expected)
    assert np.array_equal(signal.values, explicit.values)


def test_bad_laser_factor_string(fbd_file):
    """Only the documented sentinel is accepted as a string factor."""
    with pytest.raises(ValueError, match="iotech"):
        signal_from_fbd(fbd_file, laser_factor="nonsense")
    with pytest.raises(ValueError, match="iotech"):
        _is_iotech("nope")
    assert _is_iotech(IOTECH) is True
    assert _is_iotech(-1.0) is False


def test_iotech_laser_factor_without_header():
    """A file without a binary header cannot supply a correction factor."""

    class _NoHeader:
        header = None

    with pytest.raises(ValueError, match="no binary header"):
        iotech_laser_factor(_NoHeader())


def test_iotech_laser_factor_needs_phase_bins():
    """A file with a single phase bin cannot be corrected."""

    class _OneBin:
        header = {"laser_factor": 1.0}
        pmax = 1
        harmonics = 1

    with pytest.raises(ValueError, match="too few phase bins"):
        iotech_laser_factor(_OneBin())


def test_iotech_laser_factor_needs_dwell_time():
    """A file with no usable dwell time cannot be corrected."""

    class _NoDwell:
        header = {"laser_factor": 1.0, "line_time": 0.0, "line_length": 0}
        pmax = 128
        harmonics = 2
        pixel_dwell_time = 0.0

    with pytest.raises(ValueError, match="non-positive dwell time"):
        iotech_laser_factor(_NoDwell())


@pytest.mark.parametrize(
    "header",
    [
        {},  # no line timing at all
        {"line_time": 0.0, "line_length": 404},  # not a usable duration
        {"line_time": 12928.0, "line_length": 0},  # not a usable length
    ],
)
def test_nominal_dwell_time_falls_back(header):
    """Without usable line timing, the file's own dwell time is used."""

    class _Fbd:
        pass

    fbd = _Fbd()
    fbd.header = header
    assert _nominal_dwell_time(fbd, 31.875) == 31.875


# -- reference helpers -----------------------------------------------------


def test_find_reference_file(tmp_path):
    """The first-channel companion is preferred over the others."""
    fbd = tmp_path / "acquisition$EI0S.fbd"
    fbd.write_bytes(b"")
    assert find_reference_file(fbd) is None

    (tmp_path / "acquisition$EI0S_000__ch2_h1_h2.R64").write_bytes(b"")
    (tmp_path / "acquisition$EI0S_000__ch1_h1_h2.R64").write_bytes(b"")
    (tmp_path / "acquisition$EI0S_notes.txt").write_bytes(b"")
    (tmp_path / "other$EI0S_000__ch1_h1_h2.R64").write_bytes(b"")

    found = find_reference_file(str(fbd))
    assert os.path.basename(found) == "acquisition$EI0S_000__ch1_h1_h2.R64"


def test_find_reference_file_without_first_channel(tmp_path):
    """Any companion is returned when no first-channel file exists."""
    fbd = tmp_path / "acquisition.fbd"
    fbd.write_bytes(b"")
    (tmp_path / "acquisition_b.ref").write_bytes(b"")
    (tmp_path / "acquisition_a.ref").write_bytes(b"")
    assert os.path.basename(find_reference_file(fbd)) == "acquisition_a.ref"


def test_find_reference_file_without_directory(tmp_path):
    """A path in a missing directory has no companion."""
    assert find_reference_file(tmp_path / "gone" / "acquisition.fbd") is None
    assert find_reference_file(tmp_path / ".fbd") is None


def test_read_reference_image(reference_file, reference_image):
    """The mean intensity image is read back from an R64 file."""
    image = read_reference_image(reference_file)
    assert image.shape == reference_image.shape
    assert np.allclose(image, reference_image)


def test_read_reference_image_rejects_stacks(monkeypatch):
    """A file holding more than one image cannot be used as a reference."""
    monkeypatch.setattr(
        "phasorpy.io.phasor_from_simfcs_referenced",
        lambda *args, **kwargs: (np.zeros((2, 4, 4)),) * 4,
    )
    with pytest.raises(ValueError, match="3-dimensional"):
        read_reference_image("stack.r64")


def test_image_correlation():
    """Pearson correlation, with NaN for images that carry no signal."""
    image = np.arange(12.0).reshape(3, 4)
    assert image_correlation(image, image) == pytest.approx(1.0)
    assert image_correlation(image, -image) == pytest.approx(-1.0)
    assert image_correlation(image, 2.0 * image + 7.0) == pytest.approx(1.0)
    assert np.isnan(image_correlation(image, np.ones((3, 4))))


# -- match_reference_settings ---------------------------------------------


def test_match_reference_settings_recovers_line_start(
    fbd_file, reference_image
):
    """The scan recovers the settings the reference was built with."""
    with quiet():
        settings = match_reference_settings(fbd_file, reference_image)
    assert isinstance(settings, FbdReconstructionSettings)
    assert settings.scanner_line_start == MATCHED_LINE_START
    assert settings.laser_factor == IOTECH
    assert settings.refine is False
    assert settings.correlation == pytest.approx(1.0)
    assert settings.laser_factor_value == pytest.approx(0.996, abs=0.01)


def test_match_reference_settings_round_trip(fbd_file, reference_image):
    """The returned options reproduce the reference reconstruction."""
    with quiet():
        settings = match_reference_settings(fbd_file, reference_image)
        signal = signal_from_fbd(
            fbd_file, frame=-1, channel=0, **settings.as_reader_options()
        )
    assert np.array_equal(
        np.asarray(signal).sum(-1).astype(np.float64), reference_image
    )


def test_match_reference_settings_from_file(fbd_file, reference_file):
    """A reference given as a path is read as a SimFCS referenced file."""
    with quiet():
        settings = match_reference_settings(fbd_file, reference_file)
    assert settings.scanner_line_start == MATCHED_LINE_START
    assert settings.correlation == pytest.approx(1.0, abs=1e-6)


def test_match_reference_settings_reports_progress(fbd_file, reference_image):
    """Progress is reported once per candidate plus a final call."""
    seen = []
    with quiet():
        match_reference_settings(
            fbd_file,
            reference_image,
            progress=lambda done, total: seen.append((done, total)),
            line_start_range=(MATCHED_LINE_START, MATCHED_LINE_START),
        )
    total = len(DEFAULT_CANDIDATES)
    assert seen == [(i, total) for i in range(total + 1)]


def test_match_reference_settings_line_start_range(fbd_file, reference_image):
    """The scan honors an explicit line start range."""
    with quiet():
        settings = match_reference_settings(
            fbd_file, reference_image, line_start_range=(20, 30)
        )
    assert 20 <= settings.scanner_line_start <= 30
    assert settings.correlation < 1.0

    with quiet(), pytest.raises(ValueError, match="no candidate settings"):
        match_reference_settings(
            fbd_file, reference_image, line_start_range=(500, 600)
        )


def test_match_reference_settings_all_channels(fbd_file):
    """``channel=None`` matches the sum of all detector channels."""
    with quiet():
        signal = signal_from_fbd(
            fbd_file,
            frame=-1,
            channel=None,
            laser_factor=IOTECH,
            scanner_line_start=MATCHED_LINE_START,
        )
        summed = np.asarray(signal).sum(axis=(0, -1)).astype(np.float64)
        settings = match_reference_settings(
            fbd_file, summed, channel=None, candidates=[(IOTECH, False)]
        )
    assert settings.scanner_line_start == MATCHED_LINE_START
    assert settings.correlation == pytest.approx(1.0)


def test_match_reference_settings_single_frame(fbd_file):
    """A non-negative frame index matches that frame only."""
    with quiet():
        signal = signal_from_fbd(
            fbd_file,
            frame=2,
            channel=0,
            laser_factor=IOTECH,
            scanner_line_start=MATCHED_LINE_START,
        )
        settings = match_reference_settings(
            fbd_file,
            np.asarray(signal).sum(-1).astype(np.float64),
            frame=2,
            candidates=[(IOTECH, False)],
        )
    assert settings.scanner_line_start == MATCHED_LINE_START
    assert settings.correlation == pytest.approx(1.0)


def test_match_reference_settings_rejects_bad_reference(fbd_file):
    """The reference must be a single two-dimensional image."""
    with pytest.raises(ValueError, match="two-dimensional"):
        match_reference_settings(fbd_file, np.zeros((2, 4, 4)))


def test_match_reference_settings_needs_candidates(fbd_file):
    """An empty candidate list is a programming error, not a bad match."""
    with pytest.raises(ValueError, match="no candidate reconstruction"):
        match_reference_settings(fbd_file, np.zeros((4, 4)), candidates=[])


def test_match_reference_settings_reports_every_failure(fbd_file):
    """When no candidate can be evaluated, each reason is reported."""
    with quiet(), pytest.raises(ValueError) as excinfo:
        match_reference_settings(
            fbd_file,
            np.zeros((4, 4)),
            candidates=[(IOTECH, False), (-1.0, False)],
        )
    message = str(excinfo.value)
    assert "no candidate settings could be evaluated" in message
    assert message.count("laser_factor=") == 2
    assert "4 pixels wide" in message


def test_intensity_image_bounds():
    """Frame and channel indices are validated before slicing."""
    data = np.ones((2, 2, 3, 4, 5), dtype=np.uint16)
    assert _intensity_image(data, -1, None, 1).shape == (3, 4)
    assert _intensity_image(data, 1, 0, 0).shape == (3, 4)
    with pytest.raises(IndexError, match="frame"):
        _intensity_image(data, 5, 0, 0)
    with pytest.raises(IndexError, match="channel"):
        _intensity_image(data, 0, 5, 0)


def test_best_line_start_validation():
    """Mismatched geometry is reported instead of scanned."""
    image = np.zeros((4, 10))
    with pytest.raises(ValueError, match="reconstructs 4 pixel wide"):
        _best_line_start(image, np.zeros((4, 6)), 4, None)
    with pytest.raises(ValueError, match="wider than"):
        _best_line_start(np.zeros((4, 3)), np.zeros((4, 4)), 4, None)
    with pytest.raises(ValueError, match="no rows in common"):
        _best_line_start(np.zeros((0, 10)), np.zeros((4, 4)), 4, None)
    with pytest.raises(ValueError, match="is empty"):
        _best_line_start(image, np.zeros((4, 4)), 4, (9, 3))
    with pytest.raises(ValueError, match="could not be correlated"):
        _best_line_start(image, np.zeros((4, 4)), 4, None)


def test_best_line_start_finds_the_peak():
    """The window with the highest correlation wins."""
    image = np.zeros((4, 10))
    image[:, 3:7] = np.arange(16.0).reshape(4, 4)
    start, score = _best_line_start(image, image[:, 3:7], 4, None)
    assert (start, score) == (3, pytest.approx(1.0))


def test_reconstruction_settings_as_reader_options():
    """The named tuple maps onto the reader's keyword arguments."""
    settings = FbdReconstructionSettings(51, IOTECH, False, 0.9, 0.996)
    assert settings.as_reader_options() == {
        "laser_factor": IOTECH,
        "scanner_line_start": 51,
        "refine": False,
    }


# -- reader wiring ---------------------------------------------------------


def test_reader_forwards_reconstruction_settings(fbd_file):
    """The .fbd entry accepts the settings phasorpy's reader cannot."""
    with quiet():
        signal = extension_mapping["raw"][".fbd"](
            fbd_file,
            {
                "frame": -1,
                "channel": 0,
                "laser_factor": IOTECH,
                "scanner_line_start": MATCHED_LINE_START,
                "refine": False,
            },
        )
    assert signal.attrs["scanner_line_start"] == MATCHED_LINE_START
    assert signal.attrs["laser_factor"] == pytest.approx(0.996, abs=0.01)


# -- FbdWidget -------------------------------------------------------------


def test_parse_optional():
    """Optional line-edit values tolerate empty and partial input."""
    assert _parse_optional("51", int) == 51
    assert _parse_optional(" 0.5 ", float) == 0.5
    assert _parse_optional("", int) is None
    assert _parse_optional("   ", int) is None
    assert _parse_optional(None, int) is None
    assert _parse_optional("-", float) is None


def test_fbd_widget_option_defaults(make_viewer_model, qtbot, fbd_file):
    """The new fields start out neutral, so defaults are unchanged."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)

    assert widget.iotech_laser_factor.isChecked() is False
    assert widget.scanner_line_start.text() == ""
    assert widget.refine.currentText() == "Auto"

    options = {}
    widget._apply_fbd_options(options)
    assert options == {"laser_factor": -1.0}


def test_fbd_widget_iotech_toggle(make_viewer_model, qtbot, fbd_file):
    """Checking the box derives the factor and greys out the manual one."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)

    with patch.object(widget, "_update_signal_plot") as update:
        widget.iotech_laser_factor.setChecked(True)
    update.assert_called_once()
    assert widget.laser_factor.isEnabled() is False

    options = {}
    widget._apply_fbd_options(options)
    assert options["laser_factor"] == IOTECH

    with patch.object(widget, "_update_signal_plot"):
        widget.iotech_laser_factor.setChecked(False)
    assert widget.laser_factor.isEnabled() is True


@pytest.mark.parametrize(
    ("index", "refine"),
    [(1, True), (2, None), (3, False)],
)
def test_fbd_widget_refine_modes(
    make_viewer_model, qtbot, fbd_file, index, refine
):
    """Every explicit refine mode reaches the reader; "Auto" does not."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)

    options = {}
    widget._apply_fbd_options(options)
    assert "refine" not in options

    with patch.object(widget, "_update_signal_plot"):
        widget.refine.setCurrentIndex(index)
    options = {}
    widget._apply_fbd_options(options)
    assert options["refine"] is refine


def test_fbd_widget_options_and_signature(make_viewer_model, qtbot, fbd_file):
    """Every field reaches the reader and invalidates the preview cache."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)
    baseline = widget._preview_signature()

    widget.laser_factor.setText("0.5")
    widget.scanner_line_start.setText("51")
    with patch.object(widget, "_update_signal_plot"):
        widget.refine.setCurrentIndex(3)

    options = {}
    widget._apply_fbd_options(options)
    assert options == {
        "laser_factor": 0.5,
        "scanner_line_start": 51,
        "refine": False,
    }
    assert widget._extra_preview_signature() == (
        "laser_factor",
        "0.5",
        "iotech_laser_factor",
        False,
        "scanner_line_start",
        "51",
        "refine",
        3,
    )
    assert widget._preview_signature() != baseline


def test_fbd_widget_extra_kwargs_win(make_viewer_model, qtbot, fbd_file):
    """A name typed in "Additional kwargs" overrides the dedicated field."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)
    widget.scanner_line_start.setText("51")
    widget.add_kwarg_btn.click()
    key_edit, val_edit, _ = widget.kwargs_widgets[0]
    key_edit.setText("scanner_line_start")
    val_edit.setText("74")

    options = {}
    widget._apply_fbd_options(options)
    assert options["scanner_line_start"] == 74


def test_fbd_widget_on_click_applies_options(
    make_viewer_model, qtbot, fbd_file
):
    """The transform runs with the settings shown in the widget."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)
    widget.iotech_laser_factor.blockSignals(True)
    widget.iotech_laser_factor.setChecked(True)
    widget.iotech_laser_factor.blockSignals(False)
    widget.scanner_line_start.setText("51")

    reader_options = {"frame": -1, "channel": 0}
    with patch(
        "napari_phasors._widget.AdvancedOptionsWidget._on_click"
    ) as on_click:
        widget._on_click(fbd_file, reader_options, [1])
    on_click.assert_called_once_with(fbd_file, reader_options, [1])
    assert reader_options["laser_factor"] == IOTECH
    assert reader_options["scanner_line_start"] == 51


def test_fbd_widget_match_reference(
    make_viewer_model, qtbot, fbd_file, reference_file
):
    """Matching a reference file fills every reconstruction field."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)

    with (
        quiet(),
        patch(
            "napari_phasors._widget.QFileDialog.getOpenFileName",
            return_value=(reference_file, ""),
        ),
    ):
        widget.match_reference_btn.click()

    assert widget.iotech_laser_factor.isChecked() is True
    assert widget.laser_factor.isEnabled() is False
    assert widget.scanner_line_start.text() == str(MATCHED_LINE_START)
    assert widget.refine.currentText() == "Never"
    assert "line start 60" in widget.match_reference_label.text()
    assert "r = 1.0000" in widget.match_reference_label.text()
    assert widget.match_reference_btn.isEnabled()

    options = {}
    widget._apply_fbd_options(options)
    assert options == {
        "laser_factor": IOTECH,
        "scanner_line_start": MATCHED_LINE_START,
        "refine": False,
    }


def test_fbd_widget_match_reference_cancelled(
    make_viewer_model, qtbot, fbd_file
):
    """Cancelling the file dialog leaves the fields untouched."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)

    with (
        patch(
            "napari_phasors._widget.QFileDialog.getOpenFileName",
            return_value=("", ""),
        ),
        patch("napari_phasors._widget.match_reference_settings") as match,
    ):
        widget.match_reference_btn.click()
    match.assert_not_called()
    assert widget.scanner_line_start.text() == ""
    assert widget.match_reference_label.text() == ""


def test_fbd_widget_match_reference_starts_at_companion(
    make_viewer_model, qtbot, fbd_file, tmp_path
):
    """The dialog opens on the companion reference file when there is one."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)
    companion = str(tmp_path / "companion_ch1.r64")

    with (
        patch(
            "napari_phasors._widget.find_reference_file",
            return_value=companion,
        ),
        patch(
            "napari_phasors._widget.QFileDialog.getOpenFileName",
            return_value=("", ""),
        ) as dialog,
    ):
        widget.match_reference_btn.click()
    assert dialog.call_args[0][2] == companion

    with (
        patch("napari_phasors._widget.find_reference_file", return_value=None),
        patch(
            "napari_phasors._widget.QFileDialog.getOpenFileName",
            return_value=("", ""),
        ) as dialog,
    ):
        widget.match_reference_btn.click()
    assert dialog.call_args[0][2] == os.path.dirname(fbd_file)


def test_fbd_widget_match_reference_failure(
    make_viewer_model, qtbot, fbd_file
):
    """A failed match is reported and leaves the button usable."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)

    with (
        patch(
            "napari_phasors._widget.QFileDialog.getOpenFileName",
            return_value=("missing.r64", ""),
        ),
        patch(
            "napari_phasors._widget.match_reference_settings",
            side_effect=ValueError("boom"),
        ),
        patch("napari_phasors._widget.show_error") as show,
    ):
        widget.match_reference_btn.click()

    assert "boom" in show.call_args[0][0]
    assert widget.match_reference_btn.isEnabled()
    assert widget.match_reference_label.text() == ""


def test_fbd_widget_match_reference_uses_current_options(
    make_viewer_model, qtbot, fbd_file
):
    """The match runs on the frame and channel the widget is showing."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)
    widget.reader_options["frame"] = 2
    assert widget._match_channel() == 0  # "All channels" -> first channel
    widget.reader_options["channel"] = 1
    assert widget._match_channel() == 1

    with (
        patch(
            "napari_phasors._widget.QFileDialog.getOpenFileName",
            return_value=("reference.r64", ""),
        ),
        patch(
            "napari_phasors._widget.match_reference_settings",
            return_value=FbdReconstructionSettings(7, -1.0, True, 0.5, 1.0),
        ) as match,
    ):
        widget.match_reference_btn.click()

    assert match.call_args[0] == (widget.path, "reference.r64")
    assert match.call_args[1]["frame"] == 2
    assert match.call_args[1]["channel"] == 1
    # a numeric factor is written to the field instead of deriving it
    assert widget.iotech_laser_factor.isChecked() is False
    assert widget.laser_factor.text() == "-1"
    assert widget.refine.currentText() == "Always"


def test_fbd_widget_match_progress_updates_description(
    make_viewer_model, qtbot, fbd_file
):
    """The activity progress reports how many candidates were tried."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)

    class _Progress:
        descriptions = []

        def set_description(self, text):
            self.descriptions.append(text)

    progress = _Progress()
    widget._report_match_progress(progress, 1, 3)
    assert progress.descriptions == [
        "Matching FBD reconstruction to reference (1/3)"
    ]


def test_fbd_widget_apply_unknown_refine(make_viewer_model, qtbot, fbd_file):
    """A refine value with no combobox entry falls back to "Auto"."""
    widget = FbdWidget(make_viewer_model(), path=fbd_file)
    with patch.object(widget, "_update_signal_plot"):
        widget._apply_matched_settings(
            FbdReconstructionSettings(9, 0.5, "unknown", 0.25, 0.5)
        )
    assert widget.refine.currentIndex() == 0
    assert widget.laser_factor.text() == "0.5"
    assert widget.scanner_line_start.text() == "9"
