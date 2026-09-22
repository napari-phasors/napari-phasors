"""Tests for pixel-size calibration: reading it, exporting it, overriding it.

Covers the whole path a pixel size takes: out of a raw file's coordinates or
an OME-TIFF's physical sizes, onto the napari layer as ``scale``/``units``,
back into an exported OME-TIFF, and the manual override for formats that
store no pixel size at all.
"""

import json
import logging
import sys
import types

import numpy as np
import pytest
import tifffile
import xarray as xr
from phasorpy.io import signal_from_lsm, signal_from_ptu

import napari_phasors._reader as reader_module
from napari_phasors._tests.test_data_utils import get_test_file_path
from napari_phasors._widget import (
    AdvancedOptionsWidget,
    PtuWidget,
    _try_get_z_spacing_from_ome_tiff,
)
from napari_phasors._writer import _resolution_tags, write_ome_tiff

CZI_SCALING_XML = """<?xml version="1.0"?>
<ImageDocument><Metadata><Scaling><Items>
  <Distance Id="X"><Value>2.5E-07</Value></Distance>
  <Distance Id="Y"><Value>2.5E-07</Value></Distance>
  <Distance Id="T"><Value>1.0E-03</Value></Distance>
</Items></Scaling></Metadata></ImageDocument>
"""


def _write_ome(path, physical_sizes, shape=(4, 4), axes="YX"):
    """Write a small OME-TIFF carrying *physical_sizes* in its OME-XML."""
    tifffile.imwrite(
        str(path),
        np.zeros(shape, dtype=np.float32),
        photometric="minisblack",
        metadata={"axes": axes, **physical_sizes},
    )
    return str(path)


# --- physical_sizes_from_ome_tiff -------------------------------------


def test_physical_sizes_read_from_ome_xml(tmp_path):
    """Physical sizes come back keyed by axis, in micrometers."""
    path = _write_ome(
        tmp_path / "sizes.ome.tif",
        {
            "PhysicalSizeX": 0.25,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.5,
            "PhysicalSizeYUnit": "µm",
        },
    )
    assert reader_module.physical_sizes_from_ome_tiff(path) == {
        "Y": 0.5,
        "X": 0.25,
    }


@pytest.mark.parametrize(
    ("unit", "expected"),
    [("nm", 1e-3), ("mm", 1e3), ("m", 1e6), ("cm", 1e4)],
)
def test_physical_sizes_converted_from_declared_unit(tmp_path, unit, expected):
    """A size declared in another unit is converted, not assumed to be um."""
    path = _write_ome(
        tmp_path / f"{unit}.ome.tif",
        {"PhysicalSizeX": 1.0, "PhysicalSizeXUnit": unit},
    )
    sizes = reader_module.physical_sizes_from_ome_tiff(path)
    assert sizes["X"] == pytest.approx(expected)


def test_physical_sizes_default_unit_is_micrometers(tmp_path):
    """A size with no unit attribute is taken as micrometers."""
    path = str(tmp_path / "nounit.ome.tif")
    tifffile.imwrite(
        path,
        np.zeros((4, 4), dtype=np.float32),
        description=(
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Image><Pixels PhysicalSizeX="0.75"/></Image></OME>'
        ),
    )
    assert reader_module.physical_sizes_from_ome_tiff(path) == {"X": 0.75}


def test_physical_sizes_unknown_unit_is_skipped(tmp_path):
    """A unit outside the conversion table yields no size for that axis."""
    path = _write_ome(
        tmp_path / "weird.ome.tif",
        {
            "PhysicalSizeX": 1.0,
            "PhysicalSizeXUnit": "furlong",
            "PhysicalSizeY": 2.0,
            "PhysicalSizeYUnit": "um",
        },
    )
    assert reader_module.physical_sizes_from_ome_tiff(path) == {"Y": 2.0}


def test_physical_sizes_non_positive_is_skipped(tmp_path):
    """A zero or negative physical size is not a calibration."""
    path = _write_ome(
        tmp_path / "zero.ome.tif",
        {"PhysicalSizeX": 0.0, "PhysicalSizeXUnit": "um"},
    )
    assert reader_module.physical_sizes_from_ome_tiff(path) == {}


def test_physical_sizes_without_ome_metadata(tmp_path):
    """A plain TIFF carries no OME-XML and yields nothing."""
    path = str(tmp_path / "plain.tif")
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.float32))
    assert reader_module.physical_sizes_from_ome_tiff(path) == {}


def test_physical_sizes_with_no_image_element(tmp_path):
    """OME-XML without an Image element yields nothing."""
    path = str(tmp_path / "empty.ome.tif")
    tifffile.imwrite(
        path,
        np.zeros((4, 4), dtype=np.float32),
        description=(
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            "</OME>"
        ),
    )
    assert reader_module.physical_sizes_from_ome_tiff(path) == {}


def test_ome_attr_unwraps_a_text_node():
    """``xml2dict`` sometimes wraps an attribute as a ``#text`` node."""
    container = {"@PhysicalSizeX": {"#text": "0.5", "@Unit": "um"}}
    assert reader_module._ome_attr(container, "PhysicalSizeX") == "0.5"


def test_physical_sizes_with_a_single_image_element(tmp_path):
    """One Image comes back as a dict rather than a list of one."""
    path = str(tmp_path / "single.ome.tif")
    tifffile.imwrite(
        path,
        np.zeros((4, 4), dtype=np.float32),
        description=(
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Image><Pixels PhysicalSizeY="1.5"/></Image></OME>'
        ),
    )
    assert reader_module.physical_sizes_from_ome_tiff(path) == {"Y": 1.5}


def test_physical_sizes_of_missing_file_is_empty():
    """Calibration is best effort: an unreadable file is not an error."""
    assert reader_module.physical_sizes_from_ome_tiff("nope.ome.tif") == {}


def test_physical_sizes_of_unparsable_value(tmp_path):
    """A non-numeric physical size is discarded rather than raised."""
    path = str(tmp_path / "bad.ome.tif")
    tifffile.imwrite(
        path,
        np.zeros((4, 4), dtype=np.float32),
        description=(
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Image><Pixels PhysicalSizeX="abc"/></Image></OME>'
        ),
    )
    assert reader_module.physical_sizes_from_ome_tiff(path) == {}


# --- pixel_size_um ----------------------------------------------------


def test_pixel_size_from_signal_coordinates():
    """Spatial coordinates in metres become micrometers."""
    signal = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("Z", "Y", "X"),
        coords={
            "Z": np.array([0.0, 1e-6]),
            "Y": np.arange(3) * 2e-7,
            "X": np.arange(4) * 5e-8,
        },
    )
    sizes = reader_module.pixel_size_um(signal, ".ptu")
    assert sizes["Z"] == pytest.approx(1.0)
    assert sizes["Y"] == pytest.approx(0.2)
    assert sizes["X"] == pytest.approx(0.05)


def test_pixel_size_from_attrs_takes_precedence():
    """A reader that knows its pixel size says so in ``attrs``."""
    signal = xr.DataArray(
        np.zeros((3, 4)),
        dims=("Y", "X"),
        coords={"Y": np.arange(3) * 1e-6, "X": np.arange(4) * 1e-6},
        attrs={"pixel_size_um": {"y": 9.0, "x": 9.0}},
    )
    assert reader_module.pixel_size_um(signal, ".ptu") == {"Y": 9.0, "X": 9.0}


def test_pixel_size_of_uncalibrated_format_is_empty():
    """A format with no entry in the unit table stays uncalibrated."""
    signal = xr.DataArray(
        np.zeros((3, 4)),
        dims=("Y", "X"),
        coords={"Y": np.arange(3) * 1e-6, "X": np.arange(4) * 1e-6},
    )
    assert reader_module.pixel_size_um(signal, ".fbd") == {}


def test_pixel_size_of_plain_array_is_empty():
    """A numpy array carries neither coordinates nor attributes."""
    assert reader_module.pixel_size_um(np.zeros((3, 4)), ".ptu") == {}


def test_pixel_size_needs_two_coordinates():
    """A single coordinate gives no step to measure."""
    signal = xr.DataArray(
        np.zeros((1, 4)),
        dims=("Y", "X"),
        coords={"Y": np.array([0.0]), "X": np.arange(4) * 1e-6},
    )
    assert reader_module.pixel_size_um(signal, ".ptu") == {"X": 1.0}


def test_pixel_size_ignores_constant_coordinates():
    """Coordinates that do not advance are not a pixel size."""
    signal = xr.DataArray(
        np.zeros((3, 3)),
        dims=("Y", "X"),
        coords={"Y": np.zeros(3), "X": np.zeros(3)},
    )
    assert reader_module.pixel_size_um(signal, ".ptu") == {}


def test_pixel_size_ignores_non_numeric_coordinates():
    """Text coordinates are skipped instead of raising."""
    signal = xr.DataArray(
        np.zeros((2, 4)),
        dims=("Y", "X"),
        coords={"Y": ["a", "b"], "X": np.arange(4) * 1e-6},
    )
    assert reader_module.pixel_size_um(signal, ".ptu") == {"X": 1.0}


# --- _set_scale_and_units ---------------------------------------------


def test_set_scale_aligns_axis_names_to_the_right():
    """Y and X land on the trailing axes however many axes precede them."""
    add_kwargs = {}
    reader_module._set_scale_and_units(
        add_kwargs, 4, ("Y", "X"), {"Y": 0.5, "X": 0.25}
    )
    assert add_kwargs["scale"] == (1.0, 1.0, 0.5, 0.25)
    assert add_kwargs["units"] == ("", "", "um", "um")


def test_set_scale_ignores_names_beyond_the_data():
    """More axis names than axes: the extra leading names are dropped."""
    add_kwargs = {}
    reader_module._set_scale_and_units(
        add_kwargs, 2, ("Z", "Y", "X"), {"Z": 3.0, "Y": 0.5, "X": 0.25}
    )
    assert add_kwargs["scale"] == (0.5, 0.25)


def test_set_scale_without_any_match_leaves_defaults():
    """A size for an axis the layer does not have sets nothing at all."""
    add_kwargs = {}
    reader_module._set_scale_and_units(add_kwargs, 2, ("Y", "X"), {"Z": 3.0})
    assert add_kwargs == {}


@pytest.mark.parametrize(
    ("ndim", "names", "sizes"),
    [(2, ("Y", "X"), {}), (0, ("Y", "X"), {"X": 1.0}), (2, None, {"X": 1.0})],
)
def test_set_scale_guards(ndim, names, sizes):
    """Nothing to place, nowhere to place it, or no names: no keys added."""
    add_kwargs = {}
    reader_module._set_scale_and_units(add_kwargs, ndim, names, sizes)
    assert add_kwargs == {}


# --- raw readers ------------------------------------------------------


def test_ptu_layer_is_calibrated_from_the_file(caplog):
    """A real PTU's ``ImgHdr_PixResol`` reaches the layer as a scale."""
    caplog.set_level(logging.ERROR, logger="ptufile")
    path = get_test_file_path("test_file.ptu")
    expected = reader_module.pixel_size_um(
        signal_from_ptu(path, frame=-1, keepdims=False), ".ptu"
    )
    assert expected["X"] == pytest.approx(0.1463261569)

    layers = reader_module.raw_file_reader(path)
    scale = layers[0][1]["scale"]
    assert scale == pytest.approx((expected["Y"], expected["X"]))
    assert layers[0][1]["units"] == ("um", "um")


def test_lsm_layer_is_calibrated_from_the_file():
    """A real LSM's voxel size reaches the layer as a scale."""
    path = get_test_file_path("test_file.lsm")
    expected = reader_module.pixel_size_um(signal_from_lsm(path), ".lsm")
    layers = reader_module.raw_file_reader(path)
    assert layers[0][1]["scale"] == pytest.approx(
        (expected["Y"], expected["X"])
    )
    assert layers[0][1]["units"] == ("um", "um")


def test_fbd_layer_has_no_scale():
    """FBD stores no pixel size, so the layer keeps napari's default."""
    layers = reader_module.raw_file_reader(
        get_test_file_path("test_file$EI0S.fbd"), {"frame": -1, "channel": 0}
    )
    assert "scale" not in layers[0][1]


# --- processed reader -------------------------------------------------


def _patch_processed(monkeypatch, mean, real, attrs):
    monkeypatch.setitem(
        reader_module.extension_mapping["processed"],
        ".ome.tif",
        lambda path, opts: (mean, real, real, attrs),
    )


def test_processed_reader_reads_xy_from_ome_xml(monkeypatch, tmp_path):
    """Physical sizes written by any software calibrate the layer."""
    path = _write_ome(
        tmp_path / "xy.ome.tif",
        {
            "PhysicalSizeX": 0.3,
            "PhysicalSizeXUnit": "um",
            "PhysicalSizeY": 0.4,
            "PhysicalSizeYUnit": "um",
        },
    )
    attrs = {"harmonic": [1], "dims": ["Y", "X"]}
    _patch_processed(monkeypatch, np.ones((4, 4)), np.zeros((1, 4, 4)), attrs)
    layers = reader_module.processed_file_reader(path)
    assert layers[0][1]["scale"] == (0.4, 0.3)
    assert layers[0][1]["units"] == ("um", "um")


def test_processed_reader_prefers_ome_xml_over_settings(monkeypatch, tmp_path):
    """What the file declares wins over a spacing a past export recorded."""
    path = _write_ome(
        tmp_path / "zconflict.ome.tif",
        {"PhysicalSizeZ": 7.0, "PhysicalSizeZUnit": "um"},
        shape=(3, 4, 4),
        axes="ZYX",
    )
    settings = {"z_spacing_um": 2.5}
    attrs = {
        "description": json.dumps(
            {"napari_phasors_settings": json.dumps(settings)}
        ),
        "harmonic": [1],
        "dims": ["Z", "Y", "X"],
    }
    _patch_processed(
        monkeypatch, np.ones((3, 4, 4)), np.zeros((1, 3, 4, 4)), attrs
    )
    layers = reader_module.processed_file_reader(path)
    assert layers[0][1]["scale"][0] == 7.0


def test_processed_reader_falls_back_to_settings_z_spacing(monkeypatch):
    """With no OME-XML sizes, the exported settings still calibrate Z."""
    settings = {"z_spacing_um": 2.5}
    attrs = {
        "description": json.dumps(
            {"napari_phasors_settings": json.dumps(settings)}
        ),
        "harmonic": [1],
        "dims": ["Z", "Y", "X"],
    }
    _patch_processed(
        monkeypatch, np.ones((3, 4, 4)), np.zeros((1, 3, 4, 4)), attrs
    )
    layers = reader_module.processed_file_reader("missing.ome.tif")
    assert layers[0][1]["scale"] == (2.5, 1.0, 1.0)
    assert layers[0][1]["units"] == ("um", "", "")


def test_processed_reader_without_dims_assumes_trailing_zyx(monkeypatch):
    """No axis labels: Z, Y and X are taken to be the trailing axes."""
    settings = {"z_spacing_um": 2.0}
    attrs = {
        "description": json.dumps(
            {"napari_phasors_settings": json.dumps(settings)}
        ),
        "harmonic": [1],
    }
    _patch_processed(
        monkeypatch, np.ones((3, 4, 4)), np.zeros((1, 3, 4, 4)), attrs
    )
    layers = reader_module.processed_file_reader("missing.ome.tif")
    assert layers[0][1]["scale"] == (2.0, 1.0, 1.0)


def test_processed_reader_non_positive_z_spacing_is_ignored(monkeypatch):
    """A zero z spacing is not a calibration."""
    settings = {"z_spacing_um": 0}
    attrs = {
        "description": json.dumps(
            {"napari_phasors_settings": json.dumps(settings)}
        ),
        "harmonic": [1],
        "dims": ["Z", "Y", "X"],
    }
    _patch_processed(
        monkeypatch, np.ones((3, 4, 4)), np.zeros((1, 3, 4, 4)), attrs
    )
    layers = reader_module.processed_file_reader("missing.ome.tif")
    assert "scale" not in layers[0][1]


# --- CZI mosaics ------------------------------------------------------


class _FakeCzi:
    """Minimal stand-in exposing only the metadata a scaling read needs."""

    def __init__(self, xml=CZI_SCALING_XML):
        self._xml = xml

    def metadata(self):
        if self._xml is None:
            raise ValueError("no metadata")
        return self._xml


def test_czi_pixel_size_from_scaling():
    """CZI records its scaling in metres, one Distance per axis."""
    sizes = reader_module._czi_pixel_size_um(_FakeCzi())
    assert sizes == {"X": pytest.approx(0.25), "Y": pytest.approx(0.25)}


def test_czi_pixel_size_without_metadata():
    """A file whose metadata cannot be read stays uncalibrated."""
    assert reader_module._czi_pixel_size_um(_FakeCzi(None)) == {}


def test_czi_pixel_size_ignores_non_positive_distance():
    """A zero distance is not a pixel size."""
    xml = CZI_SCALING_XML.replace(
        "2.5E-07</Value></Distance>", "0</Value></Distance>", 1
    )
    assert "X" not in reader_module._czi_pixel_size_um(_FakeCzi(xml))


def _install_fake_czifile(monkeypatch, scaling_xml=CZI_SCALING_XML):
    """Register a one-tile fake ``czifile`` module with CZI scaling."""

    class _Segment:
        def __init__(self, values):
            self._values = values

        def data(self):
            return self._values

    class _Entry:
        dims = ("H", "C", "Y", "X")

        def __init__(self, plane):
            self.start = (plane, plane, 0, 0)
            self.shape = (1, 1, 4, 4)
            self.mosaic_index = 0
            self._values = np.full((4, 4), plane + 1, dtype=np.uint16)

        def read_segment_data(self, czi):
            return _Segment(self._values)

    class _CziFile:
        def __init__(self, path):
            self.filtered_subblock_directory = [_Entry(i) for i in range(4)]

        def metadata(self):
            return scaling_xml

        def close(self):
            pass

    module = types.ModuleType("czifile")
    module.CziFile = _CziFile
    monkeypatch.setitem(sys.modules, "czifile", module)


def test_czi_mosaic_tile_carries_pixel_size(monkeypatch):
    """A decoded tile advertises its pixel size for the reader to pick up."""
    _install_fake_czifile(monkeypatch)
    with reader_module.CziMosaic("m.czi") as mosaic:
        assert mosaic.pixel_size_um == {
            "X": pytest.approx(0.25),
            "Y": pytest.approx(0.25),
        }
        tile = mosaic.read_tile(0)
    assert tile.attrs["pixel_size_um"]["X"] == pytest.approx(0.25)


def test_czi_mosaic_binning_enlarges_pixels(monkeypatch):
    """Binning by two makes each pixel twice as wide."""
    _install_fake_czifile(monkeypatch)
    with reader_module.CziMosaic("m.czi") as mosaic:
        tile = mosaic.read_tile(0, binning=2)
    assert tile.attrs["pixel_size_um"]["X"] == pytest.approx(0.5)


def test_czi_mosaic_tile_without_scaling(monkeypatch):
    """A CZI with no scaling produces tiles with no calibration attribute."""
    _install_fake_czifile(monkeypatch, scaling_xml="<ImageDocument/>")
    with reader_module.CziMosaic("m.czi") as mosaic:
        tile = mosaic.read_tile(0)
    assert tile.attrs == {}


def test_stitched_layer_inherits_tile_calibration():
    """The mosaic is calibrated by the tiles it was blended from."""
    from napari_phasors._stitching import TileGeometry, TilePlacement

    tile = (
        np.ones((4, 4), dtype=np.float32),
        np.zeros((1, 4, 4), dtype=np.float32),
        np.zeros((1, 4, 4), dtype=np.float32),
    )
    template = {
        "name": "t Intensity [Phasor]",
        "metadata": {"harmonics": [1], "settings": {}},
        "scale": (0.25, 0.25),
        "units": ("um", "um"),
    }
    tile_set = reader_module.TileSet(
        ["a.czi", "b.czi"], (4, 4), [[tile, tile]], [template]
    )
    geometry = TileGeometry(
        tile_shape=(4, 4),
        placements=[
            TilePlacement(path="a.czi", row=0, col=0),
            TilePlacement(path="b.czi", row=0, col=1),
        ],
    )
    layers = tile_set.stitch(geometry)
    assert layers[0][1]["scale"] == (0.25, 0.25)
    assert layers[0][1]["units"] == ("um", "um")


# --- export -----------------------------------------------------------


def test_resolution_tags_from_physical_sizes():
    """Physical sizes in micrometers become pixels per centimeter."""
    tags = _resolution_tags({"PhysicalSizeX": 0.5, "PhysicalSizeY": 0.25})
    assert tags["resolution"] == (pytest.approx(2e4), pytest.approx(4e4))
    assert tags["resolutionunit"] == "CENTIMETER"


@pytest.mark.parametrize(
    "metadata",
    [{}, {"PhysicalSizeX": 0.5}, {"PhysicalSizeX": 0.5, "PhysicalSizeY": 0}],
)
def test_resolution_tags_need_both_axes(metadata):
    """Half a calibration is no calibration."""
    assert _resolution_tags(metadata) == {}


def _phasor_layer(scale):
    from napari.layers import Image

    mean = np.ones((4, 4), dtype=np.float32)
    phasor = np.zeros((1, 4, 4), dtype=np.float32)
    return Image(
        mean,
        name="calibrated",
        scale=scale,
        metadata={
            "original_mean": mean,
            "G_original": phasor,
            "S_original": phasor,
            "G": phasor,
            "S": phasor,
            "harmonics": [1],
            "settings": {},
        },
    )


def test_export_writes_tiff_resolution_tags(tmp_path):
    """Fiji's own TIFF reader calibrates from the resolution tags."""
    path = write_ome_tiff(
        str(tmp_path / "res.ome.tif"), _phasor_layer((0.5, 0.25))
    )[0]
    with tifffile.TiffFile(path) as tif:
        tags = tif.pages[0].tags
        x_res = tags["XResolution"].value
        assert x_res[0] / x_res[1] == pytest.approx(4e4)
        assert tags["ResolutionUnit"].value == 3  # CENTIMETER


def test_export_without_phasor_data_writes_resolution_tags(tmp_path):
    """The plain-image export branch calibrates the same way."""
    from napari.layers import Image

    layer = Image(
        np.ones((4, 4), dtype=np.float32), name="plain", scale=(0.5, 0.25)
    )
    path = write_ome_tiff(str(tmp_path / "plain.ome.tif"), layer)[0]
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].tags["ResolutionUnit"].value == 3


def test_pixel_size_round_trips_through_export(tmp_path):
    """Export then import returns the layer to the same pixel size."""
    path = write_ome_tiff(
        str(tmp_path / "trip.ome.tif"), _phasor_layer((0.4, 0.3))
    )[0]
    layers = reader_module.processed_file_reader(path)
    assert layers[0][1]["scale"] == pytest.approx((0.4, 0.3))
    assert layers[0][1]["units"] == ("um", "um")


def test_z_spacing_helper_uses_ome_physical_size(tmp_path):
    """The import dialog's z readout goes through the shared OME parser."""
    path = _write_ome(
        tmp_path / "zread.ome.tif",
        {"PhysicalSizeZ": 3.5, "PhysicalSizeZUnit": "um"},
        shape=(2, 4, 4),
        axes="ZYX",
    )
    assert _try_get_z_spacing_from_ome_tiff(path) == pytest.approx(3.5)


# --- manual override in the import dialog -----------------------------


def test_import_dialog_offers_a_pixel_size_field(
    make_viewer_model, qtbot, caplog
):
    """Formats that store no pixel size can still be calibrated by hand."""
    caplog.set_level(logging.ERROR, logger="ptufile")
    widget = PtuWidget(
        make_viewer_model(), path=get_test_file_path("test_file.ptu")
    )
    qtbot.addWidget(widget)
    widget._sync_pixel_size_widget()
    assert widget._pixel_size_edit is not None
    # Empty means "use whatever the file says".
    assert widget._get_pixel_size() is None

    widget._pixel_size_edit.setText("0.8")
    assert widget._get_pixel_size() == pytest.approx(0.8)

    # Building the row twice does not add a second one.
    row = widget._pixel_size_layout
    widget._sync_pixel_size_widget()
    assert widget._pixel_size_layout is row


def test_pixel_size_field_overrides_the_file(make_viewer_model, qtbot, caplog):
    """A typed pixel size wins over the one read from the file."""
    caplog.set_level(logging.ERROR, logger="ptufile")
    viewer = make_viewer_model()
    widget = PtuWidget(viewer, path=get_test_file_path("test_file.ptu"))
    qtbot.addWidget(widget)
    widget._sync_pixel_size_widget()
    widget._pixel_size_edit.setText("2.0")
    widget._on_click(widget.path, dict(widget.reader_options), [1])

    layer = viewer.layers[-1]
    assert tuple(layer.scale) == pytest.approx((2.0, 2.0))


def test_get_pixel_size_without_the_field():
    """A widget whose row was never built reports no override."""
    widget = AdvancedOptionsWidget.__new__(AdvancedOptionsWidget)
    widget._pixel_size_edit = None
    assert widget._get_pixel_size() is None


def test_calibration_row_appends_without_a_shape_preview(
    make_viewer_model, qtbot, caplog
):
    """With no shape preview to sit above, the row goes at the end."""
    from qtpy.QtWidgets import QHBoxLayout

    caplog.set_level(logging.ERROR, logger="ptufile")
    widget = PtuWidget(
        make_viewer_model(), path=get_test_file_path("test_file.ptu")
    )
    qtbot.addWidget(widget)
    del widget.shape_preview_label
    before = widget.mainLayout.count()
    widget._insert_calibration_row(QHBoxLayout())
    assert widget.mainLayout.count() == before + 1


def test_grouped_import_applies_the_pixel_size(
    make_viewer_model, qtbot, caplog
):
    """Each file of a grouped import is calibrated with the typed size."""
    caplog.set_level(logging.ERROR, logger="ptufile")
    viewer = make_viewer_model()
    path = get_test_file_path("test_file.ptu")
    widget = PtuWidget(viewer, path=path)
    qtbot.addWidget(widget)
    widget._sync_pixel_size_widget()
    widget._pixel_size_edit.setText("1.5")
    widget._grouped_file_paths = [path, path]
    widget._on_click(path, dict(widget.reader_options), [1])

    assert len(viewer.layers) == 2
    for layer in viewer.layers:
        assert tuple(layer.scale) == pytest.approx((1.5, 1.5))


def test_stacked_import_applies_pixel_size_and_z_spacing(
    make_viewer_model, qtbot, caplog
):
    """A 3D stack takes the typed z spacing and XY size together."""
    caplog.set_level(logging.ERROR, logger="ptufile")
    viewer = make_viewer_model()
    path = get_test_file_path("test_file.ptu")
    widget = PtuWidget(viewer, path=path)
    qtbot.addWidget(widget)
    widget._sync_pixel_size_widget()
    widget._pixel_size_edit.setText("0.6")
    widget._multi_file_paths = [path, path]
    widget._stack_z_spacing = 4.0
    widget._on_click(path, dict(widget.reader_options), [1])

    layer = viewer.layers[-1]
    assert layer.data.ndim == 3
    assert tuple(layer.scale) == pytest.approx((4.0, 0.6, 0.6))
    assert tuple(str(u) for u in layer.units) == (
        "micrometer",
        "micrometer",
        "micrometer",
    )
