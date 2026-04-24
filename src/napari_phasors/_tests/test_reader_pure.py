"""Pure function tests for _reader.py — no Qt, no napari viewer.

These tests cover uncovered lines in _reader.py using monkeypatching,
synthetic data, and unittest.mock.
"""

import json

import numpy as np
import pytest
import xarray as xr

import napari_phasors._reader as reader_module
from napari_phasors._reader import (
    _get_filename_extension,
    _parse_and_call_io_function,
    napari_get_reader,
)

# ------------------------------------------------------------------ #
#  _get_filename_extension                                            #
# ------------------------------------------------------------------ #


class TestGetFilenameExtension:
    """Tests for _get_filename_extension (line 821+)."""

    def test_simple_extension(self):
        name, ext = _get_filename_extension('/data/img.tif')
        assert name == 'img'
        assert ext == '.tif'

    def test_double_extension_ome_tif(self):
        name, ext = _get_filename_extension('/data/scan.ome.tif')
        assert name == 'scan'
        assert ext == '.ome.tif'

    def test_double_extension_ome_tiff(self):
        name, ext = _get_filename_extension('/path/to/file.ome.tiff')
        assert name == 'file'
        assert ext == '.ome.tiff'

    def test_no_extension(self):
        name, ext = _get_filename_extension('/data/README')
        assert name == 'README'
        assert ext == ''

    def test_uppercase_extension_lowered(self):
        name, ext = _get_filename_extension('/data/IMG.TIF')
        assert ext == '.tif'

    def test_mixed_case_extension(self):
        name, ext = _get_filename_extension('C:\\Users\\img.Ptu')
        assert ext == '.ptu'

    def test_dotted_filename(self):
        name, ext = _get_filename_extension('/data/my.file.name.sdt')
        assert name == 'my'
        assert ext == '.file.name.sdt'

    def test_hidden_file_unix(self):
        name, ext = _get_filename_extension('/data/.hidden')
        assert name == ''
        assert ext == '.hidden'

    def test_path_with_spaces(self):
        name, ext = _get_filename_extension('/my folder/my file.lsm')
        assert name == 'my file'
        assert ext == '.lsm'


# ------------------------------------------------------------------ #
#  _parse_and_call_io_function                                        #
# ------------------------------------------------------------------ #


class TestParseAndCallIoFunction:
    """Tests for _parse_and_call_io_function (line 769+)."""

    def test_call_with_no_options_uses_defaults(self):
        """Defaults from args_defaults are used when reader_options
        is None."""

        def fake_reader(path, frame=-1, keepdims=False):
            return (path, frame, keepdims)

        result = _parse_and_call_io_function(
            '/data/test.ptu',
            fake_reader,
            {'frame': (-1, False), 'keepdims': (False, False)},
            reader_options=None,
        )
        assert result == ('/data/test.ptu', -1, False)

    def test_reader_options_override_defaults(self):
        """User-supplied reader_options override defaults."""

        def fake_reader(path, frame=-1, keepdims=False):
            return (path, frame, keepdims)

        result = _parse_and_call_io_function(
            '/data/test.ptu',
            fake_reader,
            {'frame': (-1, False), 'keepdims': (False, False)},
            reader_options={'frame': 5},
        )
        assert result == ('/data/test.ptu', 5, False)

    def test_required_arg_missing_raises(self):
        """Missing required argument raises ValueError."""

        def fake_reader(path, channel=0):
            return channel

        with pytest.raises(ValueError, match="Required argument"):
            _parse_and_call_io_function(
                '/data/f.fbd',
                fake_reader,
                {'channel': (None, True)},
                reader_options=None,
            )

    def test_invalid_arg_raises(self):
        """Argument not in the function signature raises ValueError."""

        def fake_reader(path):
            return path

        with pytest.raises(ValueError, match="Invalid argument"):
            _parse_and_call_io_function(
                '/data/f.tif',
                fake_reader,
                {},
                reader_options={'bogus': 42},
            )

    def test_reader_options_empty_dict(self):
        """Empty reader_options dict still applies defaults."""

        def fake_reader(path, index=0):
            return index

        result = _parse_and_call_io_function(
            '/data/f.sdt',
            fake_reader,
            {'index': (7, False)},
            reader_options={},
        )
        assert result == 7

    def test_all_options_provided(self):
        """All options provided via reader_options, no defaults needed."""

        def fake_reader(path, a=0, b=0):
            return a + b

        result = _parse_and_call_io_function(
            'f.tif',
            fake_reader,
            {'a': (0, False), 'b': (0, False)},
            reader_options={'a': 3, 'b': 4},
        )
        assert result == 7


# ------------------------------------------------------------------ #
#  napari_get_reader — routing logic                                  #
# ------------------------------------------------------------------ #


class TestNapariGetReader:
    """Tests for napari_get_reader routing (line 150+)."""

    def test_empty_list_returns_none(self):
        """Empty list shows error and returns None (line 181-183)."""
        result = napari_get_reader([])
        assert result is None

    def test_single_item_list_unwraps(self):
        """A single-element list is treated like a single path
        (line 207)."""
        reader = napari_get_reader(['test.lsm'])
        assert callable(reader)

    def test_unsupported_extension_returns_none(self):
        """Unsupported extension returns None (line 225-226)."""
        result = napari_get_reader('test.xyz')
        assert result is None

    def test_raw_extension_returns_callable(self):
        reader = napari_get_reader('test.ptu')
        assert callable(reader)

    def test_processed_extension_returns_callable(self):
        reader = napari_get_reader('test.ome.tif')
        assert callable(reader)

    def test_processed_r64_returns_callable(self):
        reader = napari_get_reader('data.r64')
        assert callable(reader)

    def test_processed_ref_returns_callable(self):
        reader = napari_get_reader('data.ref')
        assert callable(reader)

    def test_processed_ifli_returns_callable(self):
        reader = napari_get_reader('data.ifli')
        assert callable(reader)

    def test_multi_file_mixed_extensions_returns_none(self):
        """Mixed extensions in multi-file list returns None
        (lines 188-192)."""
        result = napari_get_reader(
            ['a.ptu', 'b.lsm'],
        )
        assert result is None

    def test_multi_file_processed_extension_returns_none(self):
        """Multi-file with processed extension returns None
        (lines 202-205)."""
        result = napari_get_reader(
            ['a.ome.tif', 'b.ome.tif'],
        )
        assert result is None

    def test_multi_file_raw_returns_stack_reader(self, monkeypatch):
        """Multi-file with raw extension dispatches to stack reader
        (lines 195-200)."""
        sentinel = [(np.zeros((2, 2, 2)), {'name': 'stack'})]

        def fake_stack(paths, reader_options=None, harmonics=None):
            return sentinel

        monkeypatch.setattr(reader_module, 'raw_file_stack_reader', fake_stack)
        reader = napari_get_reader(['a.lsm', 'b.lsm'])
        assert callable(reader)
        result = reader(['a.lsm', 'b.lsm'])
        assert result is sentinel

    def test_ambiguous_extension_returns_callable(self):
        """Ambiguous extensions (.lif, .json) return a callable
        (lines 209-215)."""
        reader = napari_get_reader('test.lif')
        assert callable(reader)

        reader_json = napari_get_reader('test.json')
        assert callable(reader_json)

    def test_harmonics_forwarded(self, monkeypatch):
        """harmonics parameter is forwarded through the lambda."""
        captured = {}

        def fake_raw(path, reader_options=None, harmonics=None):
            captured['harmonics'] = harmonics
            return [(np.zeros((2, 2)), {'name': 'x', 'metadata': {}})]

        monkeypatch.setattr(reader_module, 'raw_file_reader', fake_raw)
        reader = napari_get_reader('test.ptu', harmonics=[1, 3])
        reader('test.ptu')
        assert captured['harmonics'] == [1, 3]


# ------------------------------------------------------------------ #
#  raw_file_reader — SDT branch (lines 289-304)                      #
# ------------------------------------------------------------------ #


class TestRawFileReaderSdt:
    """Cover SDT multi-index reading branch."""

    def test_sdt_multi_index_stacking(self, monkeypatch):
        """SDT reader iterates over index values and stacks on C
        (lines 289-304)."""
        shape = (4, 4, 16)  # Y, X, H
        arr0 = xr.DataArray(
            np.random.rand(*shape).astype(np.float32),
            dims=('Y', 'X', 'H'),
        )
        arr1 = xr.DataArray(
            np.random.rand(*shape).astype(np.float32),
            dims=('Y', 'X', 'H'),
        )
        call_count = [0]

        def fake_sdt(path, reader_options):
            idx = reader_options.get('index', 0)
            call_count[0] += 1
            if idx == 0:
                return arr0
            elif idx == 1:
                return arr1
            else:
                raise IndexError('no more')

        monkeypatch.setitem(
            reader_module.extension_mapping['raw'],
            '.sdt',
            fake_sdt,
        )

        layers = reader_module.raw_file_reader('scan.sdt')
        # Two channels → two layers, each with shape (4, 4)
        assert len(layers) == 2
        assert layers[0][0].shape == (4, 4)
        assert layers[1][0].shape == (4, 4)
        assert call_count[0] == 3  # 0, 1, then IndexError

    def test_sdt_shape_mismatch_raises(self, monkeypatch):
        """SDT reader raises when stacked shapes don't match
        (lines 300-303)."""
        arr0 = xr.DataArray(
            np.ones((4, 4, 8), dtype=np.float32),
            dims=('Y', 'X', 'H'),
        )
        arr1 = xr.DataArray(
            np.ones((3, 3, 8), dtype=np.float32),
            dims=('Y', 'X', 'H'),
        )

        def fake_sdt(path, reader_options):
            idx = reader_options.get('index', 0)
            if idx == 0:
                return arr0
            elif idx == 1:
                return arr1
            raise IndexError

        monkeypatch.setitem(
            reader_module.extension_mapping['raw'],
            '.sdt',
            fake_sdt,
        )

        with pytest.raises(AssertionError, match='Shapes'):
            reader_module.raw_file_reader('scan.sdt')


# ------------------------------------------------------------------ #
#  raw_file_reader — multichannel colormaps (lines 439-448)           #
# ------------------------------------------------------------------ #


class TestRawFileReaderColormaps:
    """Cover colormap assignment for 2-channel and 3+-channel."""

    def _make_multichannel_data(self, n_channels):
        """Create a fake xr.DataArray with C channels."""
        shape = (4, 4, n_channels, 8)
        return xr.DataArray(
            np.random.rand(*shape).astype(np.float32),
            dims=('Y', 'X', 'C', 'H'),
            coords={'C': list(range(n_channels))},
        )

    def test_two_channels_get_magenta_green(self, monkeypatch):
        """Two-channel files get MAGENTA_GREEN colormaps
        (lines 441-443)."""
        data = self._make_multichannel_data(2)
        monkeypatch.setitem(
            reader_module.extension_mapping['raw'],
            '.ptu',
            lambda path, ro: data,
        )
        layers = reader_module.raw_file_reader('test.ptu')
        assert len(layers) == 2
        for layer in layers:
            assert 'colormap' in layer[1]
            assert layer[1]['blending'] == 'additive'

    def test_three_plus_channels_get_cymrgb(self, monkeypatch):
        """Three or more channels get CYMRGB colormaps
        (lines 446-448)."""
        data = self._make_multichannel_data(3)
        monkeypatch.setitem(
            reader_module.extension_mapping['raw'],
            '.ptu',
            lambda path, ro: data,
        )
        layers = reader_module.raw_file_reader('test.ptu')
        assert len(layers) == 3
        for layer in layers:
            assert 'colormap' in layer[1]
            assert layer[1]['blending'] == 'additive'


# ------------------------------------------------------------------ #
#  raw_file_reader — frequency from attrs (line 316)                  #
# ------------------------------------------------------------------ #


class TestRawFileReaderFrequency:
    """Cover frequency extraction from xarray attrs (line 316)."""

    def test_frequency_stored_in_settings(self, monkeypatch):
        """Frequency from xarray attrs is stored in settings."""
        data = xr.DataArray(
            np.random.rand(4, 4, 8).astype(np.float32),
            dims=('Y', 'X', 'H'),
            attrs={'frequency': 80.0},
        )
        monkeypatch.setitem(
            reader_module.extension_mapping['raw'],
            '.lsm',
            lambda path, ro: data,
        )
        layers = reader_module.raw_file_reader('test.lsm')
        assert layers[0][1]['metadata']['settings']['frequency'] == 80.0


# ------------------------------------------------------------------ #
#  raw_file_stack_reader (line 453+)                                  #
# ------------------------------------------------------------------ #


class TestRawFileStackReader:
    """Tests for raw_file_stack_reader covering lines 483-612."""

    def _make_single_layer(
        self, shape=(4, 4), harmonics=None, channel_label=0
    ):
        """Helper: create a single-layer result mimicking raw_file_reader."""
        if harmonics is None:
            harmonics = [1, 2]
        mean = np.random.rand(*shape).astype(np.float32)
        g = np.random.rand(len(harmonics), *shape).astype(np.float32)
        s = np.random.rand(len(harmonics), *shape).astype(np.float32)
        summed = np.random.rand(8).astype(np.float32)
        return [
            (
                mean,
                {
                    'name': (f'file Intensity Image: Channel {channel_label}'),
                    'metadata': {
                        'original_mean': mean.copy(),
                        'settings': {'channel': channel_label},
                        'summed_signal': summed,
                        'G': g,
                        'S': s,
                        'G_original': g.copy(),
                        'S_original': s.copy(),
                        'harmonics': harmonics,
                    },
                },
            )
        ]

    def test_empty_paths_returns_empty(self):
        """Empty paths list returns [] (line 483-485)."""
        result = reader_module.raw_file_stack_reader([])
        assert result == []

    def test_mixed_extensions_returns_empty(self):
        """Mixed extensions returns [] (lines 492-496)."""
        result = reader_module.raw_file_stack_reader(['a.ptu', 'b.lsm'])
        assert result == []

    def test_stacks_single_channel_files(self, monkeypatch, tmp_path):
        """Stacking two single-channel files produces 3D arrays
        (lines 520-610)."""
        layer_a = self._make_single_layer(shape=(4, 4))
        layer_b = self._make_single_layer(shape=(4, 4))

        call_idx = [0]

        def fake_raw(path, reader_options=None, harmonics=None):
            result = [layer_a, layer_b][call_idx[0]]
            call_idx[0] += 1
            return result

        monkeypatch.setattr(reader_module, 'raw_file_reader', fake_raw)

        paths = [
            str(tmp_path / 'slice_01.lsm'),
            str(tmp_path / 'slice_02.lsm'),
        ]
        result = reader_module.raw_file_stack_reader(paths)

        assert len(result) == 1
        stacked_data, kwargs = result[0]
        assert stacked_data.shape == (2, 4, 4)
        meta = kwargs['metadata']
        assert meta['G'].shape == (2, 2, 4, 4)
        assert meta['S'].shape == (2, 2, 4, 4)
        assert 'stack_files' in meta
        assert len(meta['stack_files']) == 2

    def test_shape_mismatch_returns_empty(self, monkeypatch, tmp_path):
        """Shape mismatch across files returns [] (lines 533-538)."""
        layer_a = self._make_single_layer(shape=(4, 4))
        layer_b = self._make_single_layer(shape=(3, 3))
        call_idx = [0]

        def fake_raw(path, reader_options=None, harmonics=None):
            result = [layer_a, layer_b][call_idx[0]]
            call_idx[0] += 1
            return result

        monkeypatch.setattr(reader_module, 'raw_file_reader', fake_raw)

        paths = [
            str(tmp_path / 'a.lsm'),
            str(tmp_path / 'b.lsm'),
        ]
        result = reader_module.raw_file_stack_reader(paths)
        assert result == []

    def test_channel_count_mismatch_returns_empty(self, monkeypatch, tmp_path):
        """Channel count mismatch across files returns []
        (lines 510-517)."""
        layer_a = self._make_single_layer(shape=(4, 4))
        # Simulate second file producing 2 channels
        layer_b = self._make_single_layer(shape=(4, 4))
        layer_b2 = self._make_single_layer(shape=(4, 4), channel_label=1)
        call_idx = [0]

        def fake_raw(path, reader_options=None, harmonics=None):
            if call_idx[0] == 0:
                call_idx[0] += 1
                return layer_a
            else:
                return layer_b + layer_b2

        monkeypatch.setattr(reader_module, 'raw_file_reader', fake_raw)

        paths = [
            str(tmp_path / 'a.lsm'),
            str(tmp_path / 'b.lsm'),
        ]
        result = reader_module.raw_file_stack_reader(paths)
        assert result == []

    def test_stacks_2d_phasor_arrays(self, monkeypatch, tmp_path):
        """2D G/S arrays (no harmonic dim) are stacked on axis 0
        (lines 567-571)."""
        shape = (4, 4)
        mean = np.random.rand(*shape).astype(np.float32)
        g = np.random.rand(*shape).astype(np.float32)  # 2D
        s = np.random.rand(*shape).astype(np.float32)

        def make_layer():
            return [
                (
                    mean.copy(),
                    {
                        'name': 'file Intensity Image',
                        'metadata': {
                            'original_mean': mean.copy(),
                            'settings': {},
                            'summed_signal': np.ones(4),
                            'G': g.copy(),
                            'S': s.copy(),
                            'G_original': g.copy(),
                            'S_original': s.copy(),
                            'harmonics': [1],
                        },
                    },
                )
            ]

        call_idx = [0]

        def fake_raw(path, reader_options=None, harmonics=None):
            call_idx[0] += 1
            return make_layer()

        monkeypatch.setattr(reader_module, 'raw_file_reader', fake_raw)

        paths = [
            str(tmp_path / 'a.lsm'),
            str(tmp_path / 'b.lsm'),
        ]
        result = reader_module.raw_file_stack_reader(paths)
        assert len(result) == 1
        meta = result[0][1]['metadata']
        # 2D G stacked along axis 0 → (2, 4, 4)
        assert meta['G'].shape == (2, 4, 4)

    def test_preserves_colormap_blending(self, monkeypatch, tmp_path):
        """Colormap and blending from first file are preserved
        (lines 605-608)."""
        shape = (4, 4)
        mean = np.random.rand(*shape).astype(np.float32)
        g = np.random.rand(2, *shape).astype(np.float32)
        s = np.random.rand(2, *shape).astype(np.float32)

        def make_layer():
            return [
                (
                    mean.copy(),
                    {
                        'name': 'file Intensity Image',
                        'colormap': 'magenta',
                        'blending': 'additive',
                        'metadata': {
                            'original_mean': mean.copy(),
                            'settings': {},
                            'summed_signal': np.ones(4),
                            'G': g.copy(),
                            'S': s.copy(),
                            'G_original': g.copy(),
                            'S_original': s.copy(),
                            'harmonics': [1, 2],
                        },
                    },
                )
            ]

        call_idx = [0]

        def fake_raw(path, reader_options=None, harmonics=None):
            call_idx[0] += 1
            return make_layer()

        monkeypatch.setattr(reader_module, 'raw_file_reader', fake_raw)

        paths = [
            str(tmp_path / 'a.ptu'),
            str(tmp_path / 'b.ptu'),
        ]
        result = reader_module.raw_file_stack_reader(paths)
        assert result[0][1]['colormap'] == 'magenta'
        assert result[0][1]['blending'] == 'additive'


# ------------------------------------------------------------------ #
#  processed_file_reader (line 615+)                                  #
# ------------------------------------------------------------------ #


class TestProcessedFileReader:
    """Tests for processed_file_reader covering lines 615-766."""

    def _make_phasor_result(
        self,
        shape=(8, 8),
        harmonics=None,
        settings=None,
        description=True,
    ):
        """Build a fake return value for phasor_from_ometiff."""
        if harmonics is None:
            harmonics = np.array([1, 2])
        n_h = len(harmonics)
        mean = np.random.rand(*shape).astype(np.float32)
        real = np.random.rand(n_h, *shape).astype(np.float32)
        imag = np.random.rand(n_h, *shape).astype(np.float32)
        attrs = {'harmonic': harmonics}
        if description:
            if settings is None:
                settings = {}
            settings_json = json.dumps(settings)
            desc = json.dumps({'napari_phasors_settings': settings_json})
            attrs['description'] = desc
        return mean, real, imag, attrs

    def test_basic_read(self, monkeypatch):
        """Basic processed read without settings (line 647+)."""
        mean, real, imag, attrs = self._make_phasor_result()
        monkeypatch.setitem(
            reader_module.extension_mapping['processed'],
            '.ome.tif',
            lambda path, ro: (mean, real, imag, attrs),
        )
        layers = reader_module.processed_file_reader('test.ome.tif')
        assert len(layers) == 1
        data, kw = layers[0]
        assert data.shape == (8, 8)
        assert 'G' in kw['metadata']
        assert 'S' in kw['metadata']

    def test_no_description_in_attrs(self, monkeypatch):
        """No description in attrs → empty settings (line 663-664)."""
        mean, real, imag, _ = self._make_phasor_result(description=False)
        attrs = {'harmonic': np.array([1, 2])}
        monkeypatch.setitem(
            reader_module.extension_mapping['processed'],
            '.ome.tif',
            lambda path, ro: (mean, real, imag, attrs),
        )
        layers = reader_module.processed_file_reader('test.ome.tif')
        settings = layers[0][1]['metadata']['settings']
        assert settings == {}

    def test_frequency_in_attrs(self, monkeypatch):
        """Frequency in attrs is stored in settings (line 666)."""
        mean, real, imag, attrs = self._make_phasor_result()
        attrs['frequency'] = 80.0
        monkeypatch.setitem(
            reader_module.extension_mapping['processed'],
            '.ome.tif',
            lambda path, ro: (mean, real, imag, attrs),
        )
        layers = reader_module.processed_file_reader('test.ome.tif')
        assert layers[0][1]['metadata']['settings']['frequency'] == 80.0

    def test_calibrated_flag_converted_to_bool(self, monkeypatch):
        """calibrated flag is converted to bool (lines 661-662)."""
        mean, real, imag, attrs = self._make_phasor_result(
            settings={'calibrated': 1}
        )
        monkeypatch.setitem(
            reader_module.extension_mapping['processed'],
            '.ome.tif',
            lambda path, ro: (mean, real, imag, attrs),
        )
        layers = reader_module.processed_file_reader('test.ome.tif')
        val = layers[0][1]['metadata']['settings']['calibrated']
        assert val is True
        assert isinstance(val, bool)

    def test_description_too_large_raises(self, monkeypatch):
        """Description exceeding 256 KB raises ValueError
        (line 657-658)."""
        huge_settings = {'big': 'x' * (512 * 512 + 1)}
        settings_json = json.dumps(huge_settings)
        desc = json.dumps({'napari_phasors_settings': settings_json})
        mean = np.zeros((4, 4), dtype=np.float32)
        real = np.zeros((1, 4, 4), dtype=np.float32)
        imag = np.zeros((1, 4, 4), dtype=np.float32)
        attrs = {
            'harmonic': np.array([1]),
            'description': desc,
        }
        monkeypatch.setitem(
            reader_module.extension_mapping['processed'],
            '.ome.tif',
            lambda path, ro: (mean, real, imag, attrs),
        )
        with pytest.raises(ValueError, match='too large'):
            reader_module.processed_file_reader('test.ome.tif')

    def test_filter_and_threshold_applied(self, monkeypatch):
        """Filter + threshold settings trigger processing
        (lines 679-727)."""
        settings = {
            'filter': {
                'method': 'median',
                'size': 3,
                'repeat': 1,
            },
            'threshold': 0.01,
        }
        mean, real, imag, attrs = self._make_phasor_result(
            shape=(8, 8), settings=settings
        )
        monkeypatch.setitem(
            reader_module.extension_mapping['processed'],
            '.ome.tif',
            lambda path, ro: (mean, real, imag, attrs),
        )
        layers = reader_module.processed_file_reader('test.ome.tif')
        meta = layers[0][1]['metadata']
        # Originals preserved
        assert 'G_original' in meta
        assert 'S_original' in meta

    def test_threshold_upper_applied(self, monkeypatch):
        """threshold_upper setting triggers processing
        (lines 694-699)."""
        settings = {'threshold': 0.01, 'threshold_upper': 0.9}
        mean, real, imag, attrs = self._make_phasor_result(
            shape=(8, 8), settings=settings
        )
        monkeypatch.setitem(
            reader_module.extension_mapping['processed'],
            '.ome.tif',
            lambda path, ro: (mean, real, imag, attrs),
        )
        layers = reader_module.processed_file_reader('test.ome.tif')
        meta = layers[0][1]['metadata']
        assert meta['settings']['threshold_upper'] == 0.9

    def test_z_spacing_sets_scale(self, monkeypatch):
        """z_spacing_um in settings sets scale on 3D data
        (lines 750-763)."""
        settings = {'z_spacing_um': 2.5}
        mean, real, imag, attrs = self._make_phasor_result(
            shape=(3, 8, 8), settings=settings
        )
        monkeypatch.setitem(
            reader_module.extension_mapping['processed'],
            '.ome.tif',
            lambda path, ro: (mean, real, imag, attrs),
        )
        layers = reader_module.processed_file_reader('test.ome.tif')
        kw = layers[0][1]
        assert 'scale' in kw
        assert kw['scale'][0] == 2.5

    def test_dims_in_attrs_sets_axis_labels(self, monkeypatch):
        """dims in attrs sets axis_labels (line 746)."""
        mean, real, imag, attrs = self._make_phasor_result(shape=(8, 8))
        attrs['dims'] = 'YX'
        monkeypatch.setitem(
            reader_module.extension_mapping['processed'],
            '.ome.tif',
            lambda path, ro: (mean, real, imag, attrs),
        )
        layers = reader_module.processed_file_reader('test.ome.tif')
        assert layers[0][1]['axis_labels'] == ('Y', 'X')

    def test_axes_in_attrs_sets_axis_labels(self, monkeypatch):
        """axes in attrs sets axis_labels (line 747)."""
        mean, real, imag, attrs = self._make_phasor_result(shape=(8, 8))
        attrs['axes'] = 'YX'
        monkeypatch.setitem(
            reader_module.extension_mapping['processed'],
            '.ome.tif',
            lambda path, ro: (mean, real, imag, attrs),
        )
        layers = reader_module.processed_file_reader('test.ome.tif')
        assert layers[0][1]['axis_labels'] == ('Y', 'X')

    def test_z_spacing_with_axis_labels(self, monkeypatch):
        """z_spacing_um with Z in axis_labels finds correct axis
        (lines 753-758)."""
        settings = {'z_spacing_um': 1.5}
        mean, real, imag, attrs = self._make_phasor_result(
            shape=(3, 8, 8), settings=settings
        )
        attrs['dims'] = 'ZYX'
        monkeypatch.setitem(
            reader_module.extension_mapping['processed'],
            '.ome.tif',
            lambda path, ro: (mean, real, imag, attrs),
        )
        layers = reader_module.processed_file_reader('test.ome.tif')
        kw = layers[0][1]
        assert kw['scale'] == (1.5, 1.0, 1.0)

    def test_harmonics_default_all(self, monkeypatch):
        """When harmonics is None, defaults to 'all' (line 646-647)."""
        captured = {}

        def fake_reader(path, ro):
            captured['harmonic'] = ro.get('harmonic')
            mean = np.zeros((4, 4), dtype=np.float32)
            real = np.zeros((1, 4, 4), dtype=np.float32)
            imag = np.zeros((1, 4, 4), dtype=np.float32)
            attrs = {'harmonic': np.array([1])}
            return mean, real, imag, attrs

        monkeypatch.setitem(
            reader_module.extension_mapping['processed'],
            '.ome.tif',
            fake_reader,
        )
        reader_module.processed_file_reader('test.ome.tif', harmonics=None)
        assert captured['harmonic'] == 'all'
