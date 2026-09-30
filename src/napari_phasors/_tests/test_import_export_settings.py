import json
from unittest.mock import Mock, patch

import numpy as np

from napari_phasors._synthetic_generator import make_raw_flim_data
from napari_phasors._tests.test_plotter import create_image_layer_with_phasors
from napari_phasors.plotter import PlotterWidget


def create_layer_with_custom_settings(frequency=80.0):
    """Create a layer with custom settings for testing."""
    layer = create_image_layer_with_phasors()

    # Set custom settings
    layer.metadata['settings'] = {
        'frequency': frequency,
        'harmonic': 2,
        'semi_circle': False,
        'white_background': False,
        'plot_type': 'SCATTER',
        'colormap': 'viridis',
        'number_of_bins': 200,
        'log_scale': True,
        'calibrated': True,
        'calibration_phase': 0.5,
        'calibration_modulation': 0.9,
    }

    return layer


def create_ome_tiff_with_settings(
    tmp_path, filename="test_phasors.ome.tif", include_settings=True
):
    """Create a test OME-TIFF file with or without napari-phasors settings."""
    from phasorpy import io
    from phasorpy.phasor import phasor_from_signal

    # Create test data
    time_constants = [1, 2, 3]
    raw_flim_data = make_raw_flim_data(
        time_constants=time_constants, shape=(32, 32)
    )

    # Create settings dictionary
    settings = {
        'frequency': 80.0,
        'harmonic': 2,
        'semi_circle': True,
        'white_background': True,
        'plot_type': 'HISTOGRAM2D',
        'colormap': 'plasma',
        'number_of_bins': 180,
        'log_scale': False,
        'calibrated': False,
    }

    filepath = tmp_path / filename

    phasor = phasor_from_signal(raw_flim_data)

    # Write OME-TIFF with or without settings in description
    if include_settings:
        description = {'napari_phasors_settings': json.dumps(settings)}
        io.phasor_to_ometiff(
            filepath,
            *phasor,
            frequency=settings['frequency'],
            description=json.dumps(description),
        )
    else:
        io.phasor_to_ometiff(
            filepath,
            *phasor,
            frequency=settings['frequency'],
        )

    return filepath, settings if include_settings else None


def test_import_settings_from_another_layer(make_viewer_model, qtbot):
    """Any phasor layer, including one checked for the plot, can be the
    source; the chosen analyses are copied, calibration and filters are
    applied to the target's phasors."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_layer_with_custom_settings()
    layer3 = create_image_layer_with_phasors()
    layer1.name = "layer1"
    layer2.name = "layer2"
    layer3.name = "layer3"
    threshold_value = float(np.nanmax(layer1.metadata['original_mean']) * 0.9)
    layer3.metadata['settings'] = {
        'filter': {
            'method': 'median',
            'size': 3,
            'repeat': 1,
        },
        'threshold': threshold_value,
        'threshold_upper': None,
        'threshold_method': 'Manual',
    }
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    viewer.add_layer(layer3)

    plotter = PlotterWidget(viewer)
    plotter.image_layer_with_phasor_features_combobox.setCurrentText("layer1")
    # layer1 is checked for visualization but must still be offered as a
    # source layer to copy settings from.
    plotter.image_layers_checkable_combobox.setCheckedItems(["layer1"])
    assert plotter.harmonic == 1
    assert plotter.plot_type == 'HISTOGRAM2D'

    with (
        patch('napari_phasors.plotter.QDialog') as mock_dialog,
        patch('napari_phasors.plotter.QVBoxLayout'),
        patch('napari_phasors.plotter.QLabel'),
        patch('napari_phasors.plotter.QDialogButtonBox'),
        patch.object(
            plotter, '_show_import_dialog', return_value=[]
        ) as mock_show_import_dialog,
        patch('napari_phasors.plotter.QComboBox') as mock_combo,
    ):
        mock_dialog.Accepted = 1
        mock_dialog_instance = Mock()
        mock_dialog_instance.exec = Mock(return_value=mock_dialog.Accepted)
        mock_dialog.return_value = mock_dialog_instance
        mock_combo_instance = Mock()
        mock_combo_instance.currentText = Mock(return_value="layer2")
        mock_combo.return_value = mock_combo_instance

        plotter._import_settings_from_layer()

        mock_dialog.assert_called_once()
        mock_dialog_instance.exec.assert_called_once()
        mock_combo_instance.addItems.assert_called_with(
            ['layer1', 'layer2', 'layer3']
        )
        # The first dialog was accepted, so the second one was shown.
        mock_show_import_dialog.assert_called_once()

    # Only frequency.
    plotter._copy_metadata_from_layer("layer2", ['frequency'])
    assert layer1.metadata['settings']['frequency'] == 80.0

    # Frequency and plot settings.
    plotter._copy_metadata_from_layer("layer2", ['frequency', 'settings_tab'])
    assert layer1.metadata['settings']['frequency'] == 80.0
    assert layer1.metadata['settings']['harmonic'] == 2

    # Importing calibration settings applies the phasor transform.
    g_original_before = layer1.metadata['G_original'].copy()
    s_original_before = layer1.metadata['S_original'].copy()
    plotter._copy_metadata_from_layer("layer2", ['calibration_tab'])
    assert layer1.metadata['settings'].get('calibrated')
    assert layer1.metadata['settings'].get('calibration_phase') == 0.5
    assert layer1.metadata['settings'].get('calibration_modulation') == 0.9
    assert not np.allclose(layer1.metadata['G_original'], g_original_before)
    assert not np.allclose(layer1.metadata['S_original'], s_original_before)

    # Everything at once.
    plotter._copy_metadata_from_layer(
        "layer2", ['frequency', 'settings_tab', 'calibration_tab']
    )
    assert layer1.metadata['settings']['frequency'] == 80.0
    assert layer1.metadata['settings']['harmonic'] == 2
    assert layer1.metadata['settings']['plot_type'] == 'SCATTER'
    assert layer1.metadata['settings']['colormap'] == 'viridis'

    # Importing filter settings applies them to the target layer, drawn
    # as a density plot of the few pixels the threshold leaves.
    plotter.plot_type = 'HISTOGRAM2D'
    g_before = layer1.metadata['G'].copy()
    data_before = layer1.data.copy()
    plotter._copy_metadata_from_layer("layer3", ['filter_tab'])
    assert layer1.metadata['settings']['filter']['method'] == 'median'
    assert layer1.metadata['settings']['threshold'] == threshold_value
    assert np.isnan(layer1.data).any()
    assert not np.allclose(layer1.metadata['G'], g_before, equal_nan=True)
    assert not np.allclose(layer1.data, data_before, equal_nan=True)


def test_import_dialogs_without_layers(make_viewer_model, qtbot):
    """The import buttons exist; the settings dialog returns nothing when
    rejected and offers every analysis by default; and the layer dialog
    offers no source and goes no further when there is no phasor layer."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    assert plotter.import_from_file_button.text() == "OME-TIFF File"

    with (
        patch('napari_phasors.plotter.QDialog') as mock_dialog_class,
        patch('napari_phasors.plotter.QVBoxLayout'),
    ):
        mock_dialog_instance = Mock()
        mock_dialog_instance.exec = Mock(return_value=0)  # Rejected
        mock_dialog_class.return_value = mock_dialog_instance
        assert plotter._show_import_dialog() == []
        mock_dialog_class.assert_called_once()
        mock_dialog_instance.exec.assert_called_once()

    for accepted, kwargs in (
        (1, {}),
        (0, {"default_checked": ["settings_tab"]}),
    ):
        with (
            patch('napari_phasors.plotter.QDialog') as mock_dialog,
            patch('napari_phasors.plotter.QVBoxLayout'),
            patch('napari_phasors.plotter.QLabel'),
            patch('napari_phasors.plotter.QToggleSwitch') as mock_checkbox,
            patch('napari_phasors.plotter.QDialogButtonBox'),
        ):
            mock_dialog_instance = Mock()
            mock_dialog_instance.exec = Mock(return_value=accepted)
            mock_dialog.return_value = mock_dialog_instance
            mock_cb_instance = Mock()
            mock_cb_instance.isChecked = Mock(return_value=True)
            mock_checkbox.return_value = mock_cb_instance

            result = plotter._show_import_dialog(**kwargs)
            assert isinstance(result, list)
            # One toggle per analysis that can be imported.
            assert mock_checkbox.call_count > 0

    with (
        patch('napari_phasors.plotter.QDialog') as mock_dialog,
        patch('napari_phasors.plotter.QVBoxLayout'),
        patch('napari_phasors.plotter.QLabel'),
        patch('napari_phasors.plotter.QDialogButtonBox'),
        patch.object(plotter, '_show_import_dialog') as mock_show_import,
        patch('napari_phasors.plotter.QComboBox') as mock_combo,
    ):
        mock_dialog.Accepted = 1
        mock_dialog_instance = Mock()
        mock_dialog_instance.exec = Mock(return_value=mock_dialog.Accepted)
        mock_dialog.return_value = mock_dialog_instance
        mock_combo_instance = Mock()
        # With no phasor layers the combobox is empty and currentText
        # returns an empty string.
        mock_combo_instance.currentText = Mock(return_value="")
        mock_combo.return_value = mock_combo_instance

        plotter._import_settings_from_layer()

        mock_dialog.assert_called_once()
        mock_dialog_instance.exec.assert_called_once()
        mock_combo_instance.addItems.assert_called_with([])
        mock_show_import.assert_not_called()


def test_import_settings_from_an_ome_tiff(make_viewer_model, qtbot, tmp_path):
    """Settings are read from an OME-TIFF chosen in a file dialog, all or
    only the selected ones; a cancelled dialog changes nothing, and a file
    without settings or an invalid one only warns."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer1.name = "layer1"
    viewer.add_layer(layer1)
    plotter = PlotterWidget(viewer)

    # Plot settings are initialized in the layer metadata.
    assert 'harmonic' in layer1.metadata['settings']
    assert 'semi_circle' in layer1.metadata['settings']
    assert 'plot_type' in layer1.metadata['settings']

    def choose(path):
        return patch(
            'napari_phasors.plotter.QFileDialog.getOpenFileName',
            return_value=(str(path), ""),
        )

    # The button opens the file dialog; cancelling it changes nothing.
    settings_before = dict(layer1.metadata.get("settings", {}))
    with choose("") as mock_dialog:
        plotter.import_from_file_button.click()
        mock_dialog.assert_called_once()
        plotter._import_settings_from_file()
    assert dict(layer1.metadata.get("settings", {})) == settings_before

    # An invalid file only warns.
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("not a valid OME-TIFF")
    with (
        choose(invalid),
        patch(
            'napari_phasors.plotter.notifications.WarningNotification'
        ) as mock_warning,
    ):
        plotter._import_settings_from_file()
        mock_warning.assert_called_once()

    # A file without settings or frequency warns instead of asking.
    from phasorpy import io
    from phasorpy.phasor import phasor_from_signal

    raw_flim_data = make_raw_flim_data(
        time_constants=[1, 2, 3], shape=(32, 32)
    )
    no_settings = tmp_path / "no_settings.ome.tif"
    io.phasor_to_ometiff(no_settings, *phasor_from_signal(raw_flim_data))
    with (
        choose(no_settings),
        patch(
            'napari_phasors.plotter.notifications.WarningNotification'
        ) as mock_warning,
        patch.object(plotter, '_show_import_dialog') as mock_show_import,
    ):
        plotter._import_settings_from_file()
        mock_warning.assert_called_once()
        mock_show_import.assert_not_called()

    filepath, expected_settings = create_ome_tiff_with_settings(tmp_path)

    # Only the selected settings.
    with (
        choose(filepath),
        patch.object(
            plotter,
            '_show_import_dialog',
            return_value=['frequency', 'settings_tab'],
        ),
    ):
        plotter._import_settings_from_file()
    assert (
        layer1.metadata['settings']['frequency']
        == expected_settings['frequency']
    )

    # Everything.
    with (
        choose(filepath),
        patch.object(
            plotter,
            '_show_import_dialog',
            return_value=[
                'frequency',
                'settings_tab',
                'calibration_tab',
                'filter_tab',
                'phasor_mapping_tab',
                'fret_tab',
                'components_tab',
            ],
        ),
    ):
        plotter._import_settings_from_file()
    for key in ('frequency', 'harmonic', 'plot_type', 'colormap'):
        assert layer1.metadata['settings'][key] == expected_settings[key]


# Settings Metadata tests
def test_plot_settings_round_trip_through_metadata(make_viewer_model, qtbot):
    """A layer's stored plot settings are restored into the plotter, and a
    changed setting is written back."""
    viewer = make_viewer_model()
    layer = create_layer_with_custom_settings()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    assert plotter.harmonic == 2
    assert plotter.plot_type == 'SCATTER'
    assert plotter.histogram_colormap == 'viridis'
    assert not plotter.toggle_semi_circle
    assert not plotter.white_background

    plotter.harmonic = 3
    assert layer.metadata['settings']['harmonic'] == 3
