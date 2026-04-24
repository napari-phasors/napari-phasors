"""Pure function tests for _writer.py -- no Qt, no napari viewer.

These tests cover uncovered lines using unittest.mock for napari
layer objects and synthetic numpy data.
"""

import csv
import json
import os
from unittest.mock import MagicMock

import numpy as np

from napari_phasors._writer import (
    _extract_z_spacing_um,
    export_layer_as_csv,
    export_layer_as_image,
    write_ome_tiff,
)

# ------------------------------------------------------------------ #
#  _extract_z_spacing_um                                              #
# ------------------------------------------------------------------ #


class TestExtractZSpacingUm:

    def test_2d_data_returns_none(self):
        layer = MagicMock()
        data = np.zeros((10, 10))
        assert _extract_z_spacing_um(layer, data, {}) is None

    def test_3d_data_no_axis_labels_defaults_to_z0(self):
        layer = MagicMock()
        layer.axis_labels = None
        layer.scale = (2.0, 1.0, 1.0)
        data = np.zeros((4, 10, 10))
        result = _extract_z_spacing_um(layer, data, {})
        assert result == 2.0

    def test_3d_data_with_z_axis_label(self):
        layer = MagicMock()
        layer.axis_labels = ['Z', 'Y', 'X']
        layer.scale = (3.5, 1.0, 1.0)
        data = np.zeros((4, 10, 10))
        result = _extract_z_spacing_um(layer, data, {})
        assert result == 3.5

    def test_z_label_at_nonzero_index(self):
        layer = MagicMock()
        layer.axis_labels = ['T', 'Z', 'Y', 'X']
        layer.scale = (1.0, 2.75, 1.0, 1.0)
        data = np.zeros((2, 4, 10, 10))
        result = _extract_z_spacing_um(layer, data, {})
        assert result == 2.75

    def test_no_z_label_4d_returns_none(self):
        layer = MagicMock()
        layer.axis_labels = ['T', 'C', 'Y', 'X']
        layer.scale = (1.0, 1.0, 1.0, 1.0)
        data = np.zeros((2, 3, 10, 10))
        result = _extract_z_spacing_um(layer, data, {})
        assert result is None

    def test_scale_zero_returns_none(self):
        layer = MagicMock()
        layer.axis_labels = ['Z', 'Y', 'X']
        layer.scale = (0.0, 1.0, 1.0)
        data = np.zeros((4, 10, 10))
        result = _extract_z_spacing_um(layer, data, {})
        assert result is None

    def test_negative_scale_returns_none(self):
        layer = MagicMock()
        layer.axis_labels = ['Z', 'Y', 'X']
        layer.scale = (-1.0, 1.0, 1.0)
        data = np.zeros((4, 10, 10))
        result = _extract_z_spacing_um(layer, data, {})
        assert result is None

    def test_scale_none_falls_back_to_settings(self):
        layer = MagicMock()
        layer.axis_labels = None
        layer.scale = None
        data = np.zeros((4, 10, 10))
        metadata = {'settings': {'z_spacing_um': 1.5}}
        result = _extract_z_spacing_um(layer, data, metadata)
        assert result == 1.5

    def test_settings_fallback_invalid_returns_none(self):
        layer = MagicMock()
        layer.axis_labels = None
        layer.scale = None
        data = np.zeros((4, 10, 10))
        metadata = {'settings': {'z_spacing_um': 'bad'}}
        result = _extract_z_spacing_um(layer, data, metadata)
        assert result is None

    def test_settings_fallback_zero_returns_none(self):
        layer = MagicMock()
        layer.axis_labels = None
        layer.scale = None
        data = np.zeros((4, 10, 10))
        metadata = {'settings': {'z_spacing_um': 0}}
        result = _extract_z_spacing_um(layer, data, metadata)
        assert result is None

    def test_no_settings_key_returns_none(self):
        layer = MagicMock()
        layer.axis_labels = None
        layer.scale = None
        data = np.zeros((4, 10, 10))
        result = _extract_z_spacing_um(layer, data, {})
        assert result is None

    def test_stack_files_uses_axis_0(self):
        layer = MagicMock()
        layer.axis_labels = None
        layer.scale = (5.0, 1.0, 1.0)
        data = np.zeros((4, 10, 10))
        metadata = {'stack_files': ['a.lsm', 'b.lsm']}
        result = _extract_z_spacing_um(layer, data, metadata)
        assert result == 5.0


# ------------------------------------------------------------------ #
#  export_layer_as_image (with Mock)                                  #
# ------------------------------------------------------------------ #


class TestExportLayerAsImage:

    def _make_mock_layer(self, data, cmap_name='gray'):
        layer = MagicMock()
        layer.data = data
        cmap_mock = MagicMock()
        cmap_mock.name = cmap_name
        cmap_mock.colors = np.array(
            [[0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.float32
        )
        layer.colormap = cmap_mock
        layer.contrast_limits = (0.0, 1.0)
        return layer

    def test_2d_png_export(self, tmp_path):
        data = np.random.rand(10, 12).astype(np.float32)
        layer = self._make_mock_layer(data)
        path = str(tmp_path / 'test.png')
        export_layer_as_image(path, layer)
        assert os.path.exists(path)

    def test_2d_no_colorbar(self, tmp_path):
        data = np.random.rand(10, 12).astype(np.float32)
        layer = self._make_mock_layer(data)
        path = str(tmp_path / 'test_nocb.png')
        export_layer_as_image(path, layer, include_colorbar=False)
        assert os.path.exists(path)

    def test_3d_data_uses_current_step(self, tmp_path):
        data = np.random.rand(3, 10, 12).astype(np.float32)
        layer = self._make_mock_layer(data)
        path = str(tmp_path / 'test3d.png')
        export_layer_as_image(path, layer, current_step=(1,))
        assert os.path.exists(path)

    def test_3d_data_no_current_step(self, tmp_path):
        data = np.random.rand(3, 10, 12).astype(np.float32)
        layer = self._make_mock_layer(data)
        path = str(tmp_path / 'test3d_default.png')
        export_layer_as_image(path, layer)
        assert os.path.exists(path)

    def test_jpeg_output(self, tmp_path):
        data = np.random.rand(10, 12).astype(np.float32)
        layer = self._make_mock_layer(data)
        path = str(tmp_path / 'test.jpg')
        export_layer_as_image(path, layer)
        assert os.path.exists(path)

    def test_jpeg_extension(self, tmp_path):
        data = np.random.rand(10, 12).astype(np.float32)
        layer = self._make_mock_layer(data)
        path = str(tmp_path / 'test.jpeg')
        export_layer_as_image(path, layer)
        assert os.path.exists(path)

    def test_colormap_without_colors_attr(self, tmp_path):
        data = np.random.rand(8, 8).astype(np.float32)
        layer = MagicMock()
        layer.data = data
        cmap_mock = MagicMock(spec=['name'])
        cmap_mock.name = 'gray'
        layer.colormap = cmap_mock
        layer.contrast_limits = (0.0, 1.0)
        path = str(tmp_path / 'test_no_colors.png')
        export_layer_as_image(path, layer)
        assert os.path.exists(path)

    def test_colormap_unknown_name(self, tmp_path):
        data = np.random.rand(8, 8).astype(np.float32)
        layer = MagicMock()
        layer.data = data
        cmap_mock = MagicMock(spec=[])
        cmap_mock.name = 'nonexistent_cmap_xyz'
        layer.colormap = cmap_mock
        layer.contrast_limits = (0.0, 1.0)
        path = str(tmp_path / 'test_fallback.png')
        export_layer_as_image(path, layer)
        assert os.path.exists(path)

    def test_4d_data_with_current_step(self, tmp_path):
        data = np.random.rand(2, 3, 10, 12).astype(np.float32)
        layer = self._make_mock_layer(data)
        path = str(tmp_path / 'test4d.png')
        export_layer_as_image(path, layer, current_step=(1, 2))
        assert os.path.exists(path)


# ------------------------------------------------------------------ #
#  export_layer_as_csv                                                #
# ------------------------------------------------------------------ #


class TestExportLayerAsCsv:

    def test_phasor_data_single_harmonic(self, tmp_path):
        layer = MagicMock()
        shape = (4, 4)
        g = np.random.rand(*shape).astype(np.float32)
        s = np.random.rand(*shape).astype(np.float32)
        layer.metadata = {
            'G': g,
            'S': s,
            'G_original': g.copy(),
            'S_original': s.copy(),
            'harmonics': np.array([1]),
        }
        path = str(tmp_path / 'phasor.csv')
        export_layer_as_csv(path, layer)
        assert os.path.exists(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0
        assert 'harmonic' in rows[0]
        assert 'G' in rows[0]
        assert 'S' in rows[0]
        assert 'dim_0' in rows[0]
        assert 'dim_1' in rows[0]

    def test_phasor_data_multi_harmonic(self, tmp_path):
        layer = MagicMock()
        shape = (3, 3)
        g = np.random.rand(2, *shape).astype(np.float32)
        s = np.random.rand(2, *shape).astype(np.float32)
        layer.metadata = {
            'G': g,
            'S': s,
            'G_original': g.copy(),
            'S_original': s.copy(),
            'harmonics': np.array([1, 2]),
        }
        path = str(tmp_path / 'phasor_multi.csv')
        export_layer_as_csv(path, layer)
        assert os.path.exists(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        harmonics_seen = {row['harmonic'] for row in rows}
        assert '1' in harmonics_seen
        assert '2' in harmonics_seen

    def test_phasor_data_nan_pixels_excluded(self, tmp_path):
        layer = MagicMock()
        g = np.array([[np.nan, 0.5], [0.3, np.nan]])
        s = np.array([[np.nan, 0.4], [0.2, np.nan]])
        layer.metadata = {
            'G': g,
            'S': s,
            'G_original': g.copy(),
            'S_original': s.copy(),
            'harmonics': np.array([1]),
        }
        path = str(tmp_path / 'phasor_nan.csv')
        export_layer_as_csv(path, layer)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

    def test_raw_data_2d(self, tmp_path):
        layer = MagicMock()
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        layer.data = data
        layer.metadata = {}
        path = str(tmp_path / 'raw_2d.csv')
        export_layer_as_csv(path, layer)
        assert os.path.exists(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 4
        assert 'y' in rows[0]
        assert 'x' in rows[0]
        assert 'value' in rows[0]

    def test_raw_data_3d(self, tmp_path):
        layer = MagicMock()
        data = np.ones((2, 3, 4), dtype=np.float32)
        layer.data = data
        layer.metadata = {}
        path = str(tmp_path / 'raw_3d.csv')
        export_layer_as_csv(path, layer)
        assert os.path.exists(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 24
        assert 'dim_0' in rows[0]
        assert 'dim_1' in rows[0]
        assert 'dim_2' in rows[0]
        assert 'value' in rows[0]


# ------------------------------------------------------------------ #
#  write_ome_tiff edge cases                                          #
# ------------------------------------------------------------------ #


class TestWriteOmeTiffEdgeCases:

    def test_path_without_extension_appended(self, tmp_path):
        from napari_phasors._synthetic_generator import (
            make_intensity_layer_with_phasors,
            make_raw_flim_data,
        )

        raw = make_raw_flim_data(shape=(2, 5))
        layer = make_intensity_layer_with_phasors(raw)
        path = str(tmp_path / 'test_no_ext')
        result = write_ome_tiff(
            path,
            [
                (
                    layer.data,
                    {'metadata': layer.metadata},
                )
            ],
        )
        assert result == [path + '.ome.tif']
        assert os.path.exists(path + '.ome.tif')

    def test_write_with_selections_removes_manual(self, tmp_path):
        from napari_phasors._synthetic_generator import (
            make_intensity_layer_with_phasors,
            make_raw_flim_data,
        )

        raw = make_raw_flim_data(shape=(2, 5))
        layer = make_intensity_layer_with_phasors(raw)
        layer.metadata['settings']['selections'] = {
            'manual_selections': {
                1: np.ones((2, 5)),
            },
            'circular_cursors': [],
        }
        path = str(tmp_path / 'test_selections.ome.tif')
        write_ome_tiff(
            path,
            [
                (
                    layer.data,
                    {'metadata': layer.metadata},
                )
            ],
        )
        assert os.path.exists(path)
        import html

        from phasorpy.io import phasor_from_ometiff

        _, _, _, attrs = phasor_from_ometiff(path, harmonic='all')
        desc_str = html.unescape(attrs['description'])
        desc = json.loads(desc_str)
        settings = json.loads(desc['napari_phasors_settings'])
        assert 'manual_selections' not in settings.get('selections', {})

    def test_write_non_phasor_3d(self, tmp_path):
        from napari.layers import Image

        data = np.random.rand(4, 10, 10).astype(np.float32)
        layer = Image(data, name='stack')
        layer.metadata = {}
        path = str(tmp_path / 'non_phasor_3d.ome.tif')
        result = write_ome_tiff(path, layer)
        assert os.path.exists(path)
        assert result == [path]

    def test_write_non_phasor_4d(self, tmp_path):
        from napari.layers import Image

        data = np.random.rand(2, 3, 10, 10).astype(np.float32)
        layer = Image(data, name='4d')
        layer.metadata = {}
        path = str(tmp_path / 'non_phasor_4d.ome.tif')
        write_ome_tiff(path, layer)
        assert os.path.exists(path)

    def test_write_with_summed_signal(self, tmp_path):
        from napari_phasors._synthetic_generator import (
            make_intensity_layer_with_phasors,
            make_raw_flim_data,
        )

        raw = make_raw_flim_data(shape=(2, 5))
        layer = make_intensity_layer_with_phasors(raw)
        layer.metadata['summed_signal'] = np.array([1.0, 2.0, 3.0])
        path = str(tmp_path / 'summed.ome.tif')
        write_ome_tiff(
            path,
            [
                (
                    layer.data,
                    {'metadata': layer.metadata},
                )
            ],
        )
        assert os.path.exists(path)

    def test_write_with_xy_scale(self, tmp_path):
        from napari.layers import Image

        data = np.random.rand(10, 12).astype(np.float32)
        layer = Image(data, name='scaled')
        layer.metadata = {
            'original_mean': data.copy(),
            'G_original': np.zeros((1, 10, 12)),
            'S_original': np.zeros((1, 10, 12)),
            'harmonics': [1],
            'settings': {},
        }
        layer.scale = (0.5, 0.25)
        path = str(tmp_path / 'xy_scale.ome.tif')
        write_ome_tiff(path, layer)
        assert os.path.exists(path)

    def test_write_non_phasor_no_settings(self, tmp_path):
        from napari.layers import Image

        data = np.random.rand(10, 10).astype(np.float32)
        layer = Image(data, name='plain')
        layer.metadata = {}
        path = str(tmp_path / 'plain.ome.tif')
        result = write_ome_tiff(path, layer)
        assert os.path.exists(path)
        assert result == [path]
