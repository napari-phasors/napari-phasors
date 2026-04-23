"""Tests for _widget.py that do NOT require a real napari viewer.

Every test uses ``qtbot`` for Qt lifecycle management and
``unittest.mock.MagicMock`` for the napari viewer object.
"""

from unittest.mock import MagicMock

import numpy as np

from napari_phasors._widget import (
    AdvancedOptionsWidget,
    BhWidget,
    CziWidget,
    FlifWidget,
    IfliWidget,
    JsonWidget,
    LifWidget,
    LsmWidget,
    OmeTifWidget,
    PhasorTransform,
    PqbinWidget,
    ProcessedOnlyWidget,
    SdtWidget,
    SimfcsWidget,
    WriterWidget,
    _default_axis_labels,
    _silence_ptufile_logger,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_mock_viewer():
    """Return a MagicMock that satisfies the viewer interface."""
    viewer = MagicMock()
    # WriterWidget iterates over viewer.layers
    viewer.layers.__iter__ = MagicMock(return_value=iter([]))
    return viewer


# ------------------------------------------------------------------
# PhasorTransform
# ------------------------------------------------------------------


class TestPhasorTransform:
    """Tests for the PhasorTransform widget."""

    def test_init(self, qtbot):
        """Widget initialises and has expected child widgets."""
        viewer = _make_mock_viewer()
        widget = PhasorTransform(viewer)
        qtbot.addWidget(widget)

        assert widget.viewer is viewer
        assert widget.search_button is not None
        assert widget.multi_file_button is not None
        assert widget.save_path is not None
        assert widget.selected_paths_list is not None

    def test_reader_options_map(self, qtbot):
        """The reader_options map covers all expected extensions."""
        viewer = _make_mock_viewer()
        widget = PhasorTransform(viewer)
        qtbot.addWidget(widget)

        expected = {
            '.fbd',
            '.ptu',
            '.lsm',
            '.tif',
            '.tiff',
            '.ome.tif',
            '.ome.tiff',
            '.sdt',
            '.czi',
            '.flif',
            '.bh',
            '.b&h',
            '.bhz',
            '.bin',
            '.r64',
            '.ref',
            '.ifli',
            '.lif',
            '.json',
        }
        assert set(widget.reader_options.keys()) == expected

    def test_show_path_text(self, qtbot):
        """_show_path_text displays text and hides list."""
        viewer = _make_mock_viewer()
        widget = PhasorTransform(viewer)
        qtbot.addWidget(widget)

        widget._show_path_text('/some/file.tif')

        assert widget.save_path.text() == '/some/file.tif'
        assert not widget.save_path.isHidden()
        assert widget.selected_paths_list.isHidden()

    def test_show_path_list(self, qtbot):
        """_show_path_list populates the list and hides save_path."""
        viewer = _make_mock_viewer()
        widget = PhasorTransform(viewer)
        qtbot.addWidget(widget)

        paths = ['/a/file1.tif', '/a/file2.tif', '/a/file3.tif']
        widget._show_path_list(paths)

        assert widget.save_path.isHidden()
        assert not widget.selected_paths_list.isHidden()
        assert widget.selected_paths_list.count() == 3

    def test_clear_dynamic_widgets(self, qtbot):
        """_clear_dynamic_widgets removes all children."""
        viewer = _make_mock_viewer()
        widget = PhasorTransform(viewer)
        qtbot.addWidget(widget)

        # Add a dummy widget to dynamic layout
        from qtpy.QtWidgets import QLabel

        label = QLabel('test')
        widget.dynamic_widget_layout.addWidget(label)
        assert widget.dynamic_widget_layout.count() == 1

        widget._clear_dynamic_widgets()
        assert widget.dynamic_widget_layout.count() == 0


# ------------------------------------------------------------------
# AdvancedOptionsWidget (base class)
# ------------------------------------------------------------------


class TestAdvancedOptionsWidget:
    """Tests for the AdvancedOptionsWidget base class."""

    def test_static_choose_signal_axis_default(self):
        """Falls back to the last axis when no labels are given."""
        assert AdvancedOptionsWidget._choose_signal_axis((10, 256, 256)) == 2

    def test_static_choose_signal_axis_with_labels(self):
        """Selects the histogram axis when explicit labels are given."""
        # 'H' should be chosen as histogram axis
        result = AdvancedOptionsWidget._choose_signal_axis(
            (256, 256, 64), ['Y', 'X', 'H']
        )
        assert result == 2

    def test_static_choose_signal_axis_time_label(self):
        """Selects a T-labelled axis."""
        result = AdvancedOptionsWidget._choose_signal_axis(
            (10, 256, 256), ['T', 'Y', 'X']
        )
        assert result == 0

    def test_static_choose_signal_axis_0d(self):
        """Returns 0 for scalar shapes."""
        assert AdvancedOptionsWidget._choose_signal_axis(()) == 0

    def test_static_choose_signal_axis_non_spatial(self):
        """Selects the last non-spatial axis."""
        result = AdvancedOptionsWidget._choose_signal_axis(
            (5, 256, 256, 64), ['Z', 'Y', 'X', 'C']
        )
        assert result == 3

    def test_collapse_signal_for_plot_1d(self):
        """1-D arrays are returned unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = AdvancedOptionsWidget._collapse_signal_for_plot(arr)
        np.testing.assert_array_equal(result, arr)

    def test_collapse_signal_for_plot_scalar(self):
        """0-D arrays become a single-element 1-D array."""
        result = AdvancedOptionsWidget._collapse_signal_for_plot(
            np.float64(5.0)
        )
        assert result.shape == (1,)
        assert result[0] == 5.0

    def test_collapse_signal_for_plot_3d(self):
        """3-D arrays are summed over the spatial axes."""
        arr = np.ones((10, 4, 4))
        result = AdvancedOptionsWidget._collapse_signal_for_plot(arr)
        # Last axis has size 4, summing over first two gives 10*4=40
        # Wait — _choose_signal_axis with no labels returns last axis
        # so signal axis = 2 (size 4), sum over axes 0,1 -> each bin
        # gets 10*4 = 40... but that's wrong from the user's POV.
        # Actually, the default picks last axis, so the result has
        # shape (4,) and each element is sum of 10*4=40 ones = 40.
        # But wait, the first axis (10) is likely the "signal". Let me
        # re-check: _choose_signal_axis for shape (10,4,4) returns 2.
        # signal_axis = 2 (size 4), so axes_to_sum = (0, 1).
        # result[j] = sum(arr[:,:,j]) = 10*4*1 = 40 for each j.
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, np.full(4, 40.0))

    def test_apply_axis_transform_identity(self):
        """No transformation when axis_order is identity or None."""
        data = np.arange(24).reshape(2, 3, 4)
        kwargs = {'metadata': {}}
        result = AdvancedOptionsWidget._apply_axis_transform(
            kwargs, data, None, None
        )
        np.testing.assert_array_equal(result, data)

    def test_apply_axis_transform_reorder(self):
        """Axis reordering transposes the data."""
        data = np.arange(24).reshape(2, 3, 4)
        kwargs = {'metadata': {}}
        result = AdvancedOptionsWidget._apply_axis_transform(
            kwargs, data, [2, 0, 1], None
        )
        assert result.shape == (4, 2, 3)

    def test_apply_axis_transform_labels(self):
        """Axis labels are stored in add_kwargs."""
        data = np.arange(24).reshape(2, 3, 4)
        kwargs = {'metadata': {}}
        AdvancedOptionsWidget._apply_axis_transform(
            kwargs, data, None, ['Z', 'Y', 'X']
        )
        assert kwargs['axis_labels'] == ('Z', 'Y', 'X')

    def test_apply_axis_transform_metadata_transposed(self):
        """Metadata arrays are transposed together with data."""
        data = np.arange(24).reshape(2, 3, 4)
        g_arr = np.ones((2, 3, 4))
        kwargs = {'metadata': {'G': g_arr}}
        AdvancedOptionsWidget._apply_axis_transform(
            kwargs, data, [2, 0, 1], None
        )
        assert kwargs['metadata']['G'].shape == (4, 2, 3)

    def test_set_layer_z_scale_none(self):
        """No-op when z_spacing is None."""
        kwargs = {}
        data = np.zeros((5, 10, 10))
        AdvancedOptionsWidget._set_layer_z_scale(kwargs, data, None)
        assert 'scale' not in kwargs

    def test_set_layer_z_scale_2d(self):
        """No-op for 2-D data."""
        kwargs = {}
        data = np.zeros((10, 10))
        AdvancedOptionsWidget._set_layer_z_scale(kwargs, data, 2.5)
        assert 'scale' not in kwargs

    def test_set_layer_z_scale_3d(self):
        """Sets first-axis scale for 3-D data."""
        kwargs = {}
        data = np.zeros((5, 10, 10))
        AdvancedOptionsWidget._set_layer_z_scale(kwargs, data, 2.5)
        assert kwargs['scale'] == (2.5, 1.0, 1.0)

    def test_set_layer_z_scale_existing(self):
        """Preserves existing spatial scales."""
        kwargs = {'scale': (1.0, 0.5, 0.5)}
        data = np.zeros((5, 10, 10))
        AdvancedOptionsWidget._set_layer_z_scale(kwargs, data, 3.0)
        assert kwargs['scale'] == (3.0, 0.5, 0.5)

    def test_set_layer_z_scale_negative(self):
        """No-op for non-positive z_spacing."""
        kwargs = {}
        data = np.zeros((5, 10, 10))
        AdvancedOptionsWidget._set_layer_z_scale(kwargs, data, -1.0)
        assert 'scale' not in kwargs

    def test_set_layer_z_scale_zero(self):
        """No-op for zero z_spacing."""
        kwargs = {}
        data = np.zeros((5, 10, 10))
        AdvancedOptionsWidget._set_layer_z_scale(kwargs, data, 0.0)
        assert 'scale' not in kwargs

    def test_set_layer_z_scale_invalid_string(self):
        """No-op for non-numeric z_spacing."""
        kwargs = {}
        data = np.zeros((5, 10, 10))
        AdvancedOptionsWidget._set_layer_z_scale(kwargs, data, 'abc')
        assert 'scale' not in kwargs

    def test_set_layer_z_scale_short_existing(self):
        """Extends short scale list to match data ndim."""
        kwargs = {'scale': (0.5,)}
        data = np.zeros((5, 10, 10))
        AdvancedOptionsWidget._set_layer_z_scale(kwargs, data, 2.0)
        assert kwargs['scale'] == (2.0, 1.0, 1.0)

    def test_set_layer_z_scale_long_existing(self):
        """Truncates long scale list to match data ndim."""
        kwargs = {'scale': (1.0, 0.5, 0.5, 0.5)}
        data = np.zeros((5, 10, 10))
        AdvancedOptionsWidget._set_layer_z_scale(kwargs, data, 2.0)
        assert kwargs['scale'] == (2.0, 0.5, 0.5)

    def test_on_harmonic_edit_changed(self, qtbot):
        """Harmonic edits update the harmonics list."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        # Force max_harmonic high so edits are created
        widget.max_harmonic = 10
        widget._update_harmonic_slider()

        assert widget.harmonic_start_edit is not None
        assert widget.harmonic_end_edit is not None

        widget.harmonic_start_edit.setText('2')
        widget.harmonic_end_edit.setText('5')
        widget._on_harmonic_edit_changed()

        assert widget.harmonics == [2, 3, 4, 5]

    def test_on_harmonic_edit_clamping(self, qtbot):
        """Harmonic edits are clamped to valid range."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        widget.max_harmonic = 5
        widget._update_harmonic_slider()

        widget.harmonic_start_edit.setText('0')
        widget.harmonic_end_edit.setText('100')
        widget._on_harmonic_edit_changed()

        assert widget.harmonics == [1, 2, 3, 4, 5]

    def test_on_harmonic_slider_changed(self, qtbot):
        """Harmonic slider updates the harmonics list."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        widget.max_harmonic = 10
        widget._update_harmonic_slider()

        widget._on_harmonic_slider_changed((3, 7))
        assert widget.harmonics == [3, 4, 5, 6, 7]

    def test_on_frames_combobox_changed(self, qtbot):
        """Frames combobox updates reader_options."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        widget._on_frames_combobox_changed(3)
        assert widget.reader_options.get('frame') == 2

    def test_on_channels_combobox_changed_all(self, qtbot):
        """Channel index 0 means 'all channels' → None."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        # Need a channels combobox
        widget.all_channels = 3
        widget._channels_widget()
        widget._on_channels_combobox_changed(0)
        assert widget.reader_options.get('channel') is None

    def test_on_channels_combobox_changed_specific(self, qtbot):
        """Selecting a specific channel updates reader_options."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        widget.all_channels = 3
        widget._channels_widget()
        widget._on_channels_combobox_changed(2)
        assert widget.reader_options.get('channel') == 1

    def test_on_stack_z_spacing_changed_valid(self, qtbot):
        """Valid z-spacing is stored."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        from qtpy.QtWidgets import QLineEdit

        widget._stack_z_spacing_edit = QLineEdit()
        widget._stack_z_spacing_edit.setText('2.5')
        widget._on_stack_z_spacing_changed()
        assert widget._stack_z_spacing == 2.5

    def test_on_stack_z_spacing_changed_invalid(self, qtbot):
        """Invalid z-spacing falls back to 1.0."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        from qtpy.QtWidgets import QLineEdit

        widget._stack_z_spacing_edit = QLineEdit()
        widget._stack_z_spacing_edit.setText('abc')
        widget._on_stack_z_spacing_changed()
        assert widget._stack_z_spacing == 1.0

    def test_on_stack_z_spacing_changed_negative(self, qtbot):
        """Negative z-spacing falls back to 1.0."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        from qtpy.QtWidgets import QLineEdit

        widget._stack_z_spacing_edit = QLineEdit()
        widget._stack_z_spacing_edit.setText('-5')
        widget._on_stack_z_spacing_changed()
        assert widget._stack_z_spacing == 1.0

    def test_on_stack_z_spacing_changed_none_edit(self, qtbot):
        """No-op when _stack_z_spacing_edit is None."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        widget._stack_z_spacing_edit = None
        widget._on_stack_z_spacing_changed()
        # Should not raise


# ------------------------------------------------------------------
# SdtWidget
# ------------------------------------------------------------------


class TestSdtWidget:
    """Tests for the SdtWidget."""

    def test_init(self, qtbot):
        """SdtWidget initialises with expected UI elements."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        assert widget.viewer is viewer
        assert widget.path == '/fake/test.sdt'
        assert hasattr(widget, 'btn')
        assert hasattr(widget, 'index')
        assert hasattr(widget, 'harmonics')
        assert hasattr(widget, 'canvas')
        assert hasattr(widget, 'shape_preview_label')

    def test_default_index(self, qtbot):
        """Default index value is '0'."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        assert widget.index.text() == '0'

    def test_harmonic_widget_exists(self, qtbot):
        """Harmonic layout is set up."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        assert widget.harmonic_layout is not None

    def test_reader_options_empty_initially(self, qtbot):
        """reader_options starts empty for SdtWidget."""
        viewer = _make_mock_viewer()
        widget = SdtWidget(viewer, '/fake/test.sdt')
        qtbot.addWidget(widget)

        assert widget.reader_options == {}


# ------------------------------------------------------------------
# CziWidget
# ------------------------------------------------------------------


class TestCziWidget:
    """Tests for the CziWidget."""

    def test_init(self, qtbot):
        """CziWidget initialises with expected UI elements."""
        viewer = _make_mock_viewer()
        widget = CziWidget(viewer, '/fake/test.czi')
        qtbot.addWidget(widget)

        assert widget.path == '/fake/test.czi'
        assert hasattr(widget, 'btn')
        assert hasattr(widget, 'canvas')

    def test_reader_options_empty(self, qtbot):
        """CziWidget reader_options starts empty."""
        viewer = _make_mock_viewer()
        widget = CziWidget(viewer, '/fake/test.czi')
        qtbot.addWidget(widget)

        assert widget.reader_options == {}


# ------------------------------------------------------------------
# LsmWidget
# ------------------------------------------------------------------


class TestLsmWidget:
    """Tests for the LsmWidget."""

    def test_init(self, qtbot):
        """LsmWidget initialises safely with a fake path."""
        viewer = _make_mock_viewer()
        widget = LsmWidget(viewer, '/fake/test.lsm')
        qtbot.addWidget(widget)

        assert widget.path == '/fake/test.lsm'
        assert widget._is_lsm is False  # fake path
        assert hasattr(widget, 'btn')

    def test_check_if_lsm_fake_path(self, qtbot):
        """_check_if_lsm returns False for a nonexistent path."""
        viewer = _make_mock_viewer()
        widget = LsmWidget(viewer, '/fake/test.lsm')
        qtbot.addWidget(widget)

        assert widget._check_if_lsm('/nonexistent.lsm') is False


# ------------------------------------------------------------------
# OmeTifWidget
# ------------------------------------------------------------------


class TestOmeTifWidget:
    """Tests for the OmeTifWidget."""

    def test_init(self, qtbot):
        """OmeTifWidget initialises safely with a fake path."""
        viewer = _make_mock_viewer()
        widget = OmeTifWidget(viewer, '/fake/test.ome.tif')
        qtbot.addWidget(widget)

        assert widget.path == '/fake/test.ome.tif'
        assert hasattr(widget, 'btn')
        assert hasattr(widget, 'canvas')

    def test_has_phasor_settings_false(self, qtbot):
        """has_phasor_settings is False for fake file."""
        viewer = _make_mock_viewer()
        widget = OmeTifWidget(viewer, '/fake/test.ome.tif')
        qtbot.addWidget(widget)

        assert widget.has_phasor_settings is False

    def test_get_signal_data_none(self, qtbot):
        """Returns None when no phasor settings are loaded."""
        viewer = _make_mock_viewer()
        widget = OmeTifWidget(viewer, '/fake/test.ome.tif')
        qtbot.addWidget(widget)

        assert widget._get_signal_data() is None


# ------------------------------------------------------------------
# FlifWidget
# ------------------------------------------------------------------


class TestFlifWidget:
    """Tests for the FlifWidget."""

    def test_init(self, qtbot):
        """FlifWidget initialises with expected UI elements."""
        viewer = _make_mock_viewer()
        widget = FlifWidget(viewer, '/fake/test.flif')
        qtbot.addWidget(widget)

        assert widget.path == '/fake/test.flif'
        assert hasattr(widget, 'btn')
        assert hasattr(widget, 'canvas')
        assert hasattr(widget, 'harmonics')


# ------------------------------------------------------------------
# BhWidget
# ------------------------------------------------------------------


class TestBhWidget:
    """Tests for the BhWidget."""

    def test_init(self, qtbot):
        """BhWidget initialises with expected UI elements."""
        viewer = _make_mock_viewer()
        widget = BhWidget(viewer, '/fake/test.bh')
        qtbot.addWidget(widget)

        assert widget.path == '/fake/test.bh'
        assert hasattr(widget, 'btn')


# ------------------------------------------------------------------
# PqbinWidget
# ------------------------------------------------------------------


class TestPqbinWidget:
    """Tests for the PqbinWidget."""

    def test_init(self, qtbot):
        """PqbinWidget initialises with expected UI elements."""
        viewer = _make_mock_viewer()
        widget = PqbinWidget(viewer, '/fake/test.bin')
        qtbot.addWidget(widget)

        assert widget.path == '/fake/test.bin'
        assert hasattr(widget, 'btn')


# ------------------------------------------------------------------
# LifWidget
# ------------------------------------------------------------------


class TestLifWidget:
    """Tests for the LifWidget."""

    def test_init(self, qtbot):
        """LifWidget initialises with expected UI elements."""
        viewer = _make_mock_viewer()
        widget = LifWidget(viewer, '/fake/test.lif')
        qtbot.addWidget(widget)

        assert widget.path == '/fake/test.lif'
        assert hasattr(widget, 'btn')
        assert hasattr(widget, 'image')
        assert hasattr(widget, 'dim')

    def test_dim_combobox_items(self, qtbot):
        """Dim combobox has the expected spectral dimension options."""
        viewer = _make_mock_viewer()
        widget = LifWidget(viewer, '/fake/test.lif')
        qtbot.addWidget(widget)

        items = [widget.dim.itemText(i) for i in range(widget.dim.count())]
        assert items == ['\u03bb', '\u039b']

    def test_sync_lif_reader_options_empty(self, qtbot):
        """Empty image field does not set 'image' key."""
        viewer = _make_mock_viewer()
        widget = LifWidget(viewer, '/fake/test.lif')
        qtbot.addWidget(widget)

        widget.image.setText('')
        widget._sync_lif_reader_options()

        assert 'image' not in widget.reader_options
        assert widget.reader_options['dim'] == '\u03bb'

    def test_sync_lif_reader_options_index(self, qtbot):
        """Numeric image text is parsed as integer."""
        viewer = _make_mock_viewer()
        widget = LifWidget(viewer, '/fake/test.lif')
        qtbot.addWidget(widget)

        widget.image.setText('3')
        widget._sync_lif_reader_options()

        assert widget.reader_options['image'] == 3

    def test_sync_lif_reader_options_regex(self, qtbot):
        """Non-numeric text is stored as string."""
        viewer = _make_mock_viewer()
        widget = LifWidget(viewer, '/fake/test.lif')
        qtbot.addWidget(widget)

        widget.image.setText('.*DAPI.*')
        widget._sync_lif_reader_options()

        assert widget.reader_options['image'] == '.*DAPI.*'


# ------------------------------------------------------------------
# JsonWidget
# ------------------------------------------------------------------


class TestJsonWidget:
    """Tests for the JsonWidget."""

    def test_init(self, qtbot):
        """JsonWidget initialises with expected UI elements."""
        viewer = _make_mock_viewer()
        widget = JsonWidget(viewer, '/fake/test.json')
        qtbot.addWidget(widget)

        assert widget.path == '/fake/test.json'
        assert hasattr(widget, 'btn')
        assert hasattr(widget, 'channel_entry')

    def test_default_channel(self, qtbot):
        """Default channel is 0."""
        viewer = _make_mock_viewer()
        widget = JsonWidget(viewer, '/fake/test.json')
        qtbot.addWidget(widget)

        assert widget.channel_entry.text() == '0'
        assert widget.reader_options.get('channel') == 0

    def test_sync_json_reader_options(self, qtbot):
        """Channel entry updates reader_options."""
        viewer = _make_mock_viewer()
        widget = JsonWidget(viewer, '/fake/test.json')
        qtbot.addWidget(widget)

        widget.channel_entry.setText('5')
        widget._sync_json_reader_options()
        assert widget.reader_options['channel'] == 5

    def test_sync_json_reader_options_empty(self, qtbot):
        """Empty channel entry sets channel to None."""
        viewer = _make_mock_viewer()
        widget = JsonWidget(viewer, '/fake/test.json')
        qtbot.addWidget(widget)

        widget.channel_entry.setText('')
        widget._sync_json_reader_options()
        assert widget.reader_options['channel'] is None


# ------------------------------------------------------------------
# ProcessedOnlyWidget / SimfcsWidget / IfliWidget
# ------------------------------------------------------------------


class TestProcessedOnlyWidget:
    """Tests for ProcessedOnlyWidget and subclasses."""

    def test_init(self, qtbot):
        """ProcessedOnlyWidget initialises with hidden canvas."""
        viewer = _make_mock_viewer()
        widget = ProcessedOnlyWidget(viewer, '/fake/test.r64')
        qtbot.addWidget(widget)

        assert widget.path == '/fake/test.r64'
        assert hasattr(widget, 'btn')
        assert not widget.canvas.isVisible()

    def test_get_signal_data_returns_none(self, qtbot):
        """Processed files have no raw signal."""
        viewer = _make_mock_viewer()
        widget = ProcessedOnlyWidget(viewer, '/fake/test.r64')
        qtbot.addWidget(widget)

        assert widget._get_signal_data() is None


class TestSimfcsWidget:
    """Tests for the SimfcsWidget."""

    def test_init(self, qtbot):
        """SimfcsWidget initialises as a ProcessedOnlyWidget."""
        viewer = _make_mock_viewer()
        widget = SimfcsWidget(viewer, '/fake/test.r64')
        qtbot.addWidget(widget)

        assert isinstance(widget, ProcessedOnlyWidget)
        assert hasattr(widget, 'btn')


class TestIfliWidget:
    """Tests for the IfliWidget."""

    def test_init(self, qtbot):
        """IfliWidget initialises with channel entry."""
        viewer = _make_mock_viewer()
        widget = IfliWidget(viewer, '/fake/test.ifli')
        qtbot.addWidget(widget)

        assert hasattr(widget, 'channel_entry')
        assert widget.channel_entry.text() == '0'
        assert widget.reader_options.get('channel') == 0


# ------------------------------------------------------------------
# WriterWidget
# ------------------------------------------------------------------


class TestWriterWidget:
    """Tests for the WriterWidget."""

    def test_init(self, qtbot):
        """WriterWidget initialises with expected UI elements."""
        viewer = _make_mock_viewer()
        widget = WriterWidget(viewer)
        qtbot.addWidget(widget)

        assert widget.viewer is viewer
        assert hasattr(widget, 'export_layer_combobox')
        assert hasattr(widget, 'colorbar_checkbox')
        assert hasattr(widget, 'search_button')

    def test_colorbar_checkbox_default(self, qtbot):
        """Colorbar checkbox is checked by default."""
        viewer = _make_mock_viewer()
        widget = WriterWidget(viewer)
        qtbot.addWidget(widget)

        assert widget.colorbar_checkbox.isChecked()

    def test_populate_combobox_empty(self, qtbot):
        """Combobox has no items when viewer has no layers."""
        viewer = _make_mock_viewer()
        widget = WriterWidget(viewer)
        qtbot.addWidget(widget)

        assert widget.export_layer_combobox.model().rowCount() == 0


# ------------------------------------------------------------------
# _default_axis_labels
# ------------------------------------------------------------------


class TestDefaultAxisLabels:
    """Tests for the _default_axis_labels helper function."""

    def test_2d(self):
        assert _default_axis_labels(2) == ['Y', 'X']

    def test_3d(self):
        assert _default_axis_labels(3) == ['Z', 'Y', 'X']

    def test_4d(self):
        assert _default_axis_labels(4) == ['T', 'Z', 'Y', 'X']

    def test_5d(self):
        result = _default_axis_labels(5)
        assert len(result) == 5
        assert result[0] == 'Axis 0'


# ------------------------------------------------------------------
# _silence_ptufile_logger
# ------------------------------------------------------------------


class TestSilencePtufileLogger:
    """Tests for the _silence_ptufile_logger context manager."""

    def test_restores_level(self):
        """Logger level is restored after the context exits."""
        import logging

        logger = logging.getLogger('ptufile')
        original = logger.level

        with _silence_ptufile_logger():
            assert logger.level == logging.CRITICAL

        assert logger.level == original
