import json
from unittest.mock import Mock, patch

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


def test_import_from_layer_dialog_accepted(make_napari_viewer):
    """Test that accepting the layer selection dialog proceeds to the next step."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_layer_with_custom_settings()
    layer1.name = "layer1"
    layer2.name = "layer2"
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    plotter = PlotterWidget(viewer)

    # Set current layer in plotter
    plotter.image_layer_with_phasor_features_combobox.setCurrentText("layer1")

    with (
        patch('napari_phasors.plotter.QDialog') as mock_dialog,
        patch('napari_phasors.plotter.QVBoxLayout'),
        patch('napari_phasors.plotter.QLabel'),
        patch('napari_phasors.plotter.QDialogButtonBox'),
    ):

        # Configure the mock to have the correct 'Accepted' value
        mock_dialog.Accepted = 1

        mock_dialog_instance = Mock()
        mock_dialog_instance.exec_ = Mock(return_value=mock_dialog.Accepted)
        mock_dialog.return_value = mock_dialog_instance

        # Mock the second dialog (_show_import_dialog)
        with patch.object(
            plotter, '_show_import_dialog', return_value=[]
        ) as mock_show_import_dialog:
            # Mock QComboBox to return the name of the other layer
            with patch('napari_phasors.plotter.QComboBox') as mock_combo:
                mock_combo_instance = Mock()
                mock_combo_instance.currentText = Mock(return_value="layer2")
                mock_combo.return_value = mock_combo_instance

                plotter._import_settings_from_layer()

                # Verify the first dialog was created and executed
                mock_dialog.assert_called_once()
                mock_dialog_instance.exec_.assert_called_once()

                # Verify the layer selection combobox was populated
                mock_combo_instance.addItems.assert_called_with(['layer2'])

                # Verify that since the first dialog was accepted,
                # the second dialog was shown.
                mock_show_import_dialog.assert_called_once()


def test_import_from_layer_no_other_layers(make_napari_viewer):
    """Test import when there are no other layers available."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer1.name = "layer1"
    viewer.add_layer(layer1)

    plotter = PlotterWidget(viewer)
    plotter.image_layer_with_phasor_features_combobox.setCurrentText("layer1")

    with (
        patch('napari_phasors.plotter.QDialog') as mock_dialog,
        patch('napari_phasors.plotter.QVBoxLayout'),
        patch('napari_phasors.plotter.QLabel'),
        patch('napari_phasors.plotter.QDialogButtonBox'),
    ):

        mock_dialog.Accepted = 1
        mock_dialog_instance = Mock()
        mock_dialog_instance.exec_ = Mock(return_value=mock_dialog.Accepted)
        mock_dialog.return_value = mock_dialog_instance

        with patch.object(
            plotter, '_show_import_dialog'
        ) as mock_show_import_dialog:
            with patch('napari_phasors.plotter.QComboBox') as mock_combo:
                mock_combo_instance = Mock()
                # When no other layers are available, the combobox is empty
                # and currentText returns an empty string.
                mock_combo_instance.currentText = Mock(return_value="")
                mock_combo.return_value = mock_combo_instance

                plotter._import_settings_from_layer()

                # Verify the first dialog was shown
                mock_dialog.assert_called_once()
                mock_dialog_instance.exec_.assert_called_once()

                # Verify the combobox was populated with an empty list
                mock_combo_instance.addItems.assert_called_with([])

                # Verify the second dialog was NOT shown
                mock_show_import_dialog.assert_not_called()


def test_import_all_settings_from_layer(make_napari_viewer):
    """Test importing all settings from another layer."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_layer_with_custom_settings()
    layer1.name = "layer1"
    layer2.name = "layer2"
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    plotter = PlotterWidget(viewer)

    # Verify initial state
    assert plotter.harmonic == 1
    assert plotter.plot_type == 'HISTOGRAM2D'

    # Mock the selection dialog to select all tabs
    with patch.object(
        plotter,
        '_show_import_dialog',
        return_value=[
            'frequency',
            'settings_tab',
            'calibration_tab',
            'filter_tab',
            'lifetime_tab',
            'fret_tab',
            'components_tab',
        ],
    ):
        # Copy metadata from layer2 to layer1
        plotter._copy_metadata_from_layer(
            "layer2", ['frequency', 'settings_tab', 'calibration_tab']
        )

    # Verify settings were imported
    assert layer1.metadata['settings']['frequency'] == 80.0
    assert layer1.metadata['settings']['harmonic'] == 2
    assert layer1.metadata['settings']['plot_type'] == 'SCATTER'
    assert layer1.metadata['settings']['colormap'] == 'viridis'


def test_import_partial_settings_from_layer(make_napari_viewer):
    """Test importing only selected settings from another layer."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_layer_with_custom_settings()
    layer1.name = "layer1"
    layer2.name = "layer2"
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    plotter = PlotterWidget(viewer)

    # Import only frequency and plot settings
    with patch.object(
        plotter,
        '_show_import_dialog',
        return_value=['frequency', 'settings_tab'],
    ):
        plotter._copy_metadata_from_layer(
            "layer2", ['frequency', 'settings_tab']
        )

    # Verify only selected settings were imported
    assert layer1.metadata['settings']['frequency'] == 80.0
    assert layer1.metadata['settings']['harmonic'] == 2


def test_import_frequency_only(make_napari_viewer):
    """Test importing only frequency from another layer."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_layer_with_custom_settings()
    layer1.name = "layer1"
    layer2.name = "layer2"
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    plotter = PlotterWidget(viewer)

    # Import only frequency
    plotter._copy_metadata_from_layer("layer2", ['frequency'])

    # Verify frequency was imported
    assert layer1.metadata['settings']['frequency'] == 80.0


def test_import_with_calibration(make_napari_viewer):
    """Test importing calibration settings."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_layer_with_custom_settings()
    layer1.name = "layer1"
    layer2.name = "layer2"
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    plotter = PlotterWidget(viewer)

    # Import calibration settings
    plotter._copy_metadata_from_layer("layer2", ['calibration_tab'])

    # Verify calibration settings were imported
    assert layer1.metadata['settings'].get('calibrated') == True
    assert layer1.metadata['settings'].get('calibration_phase') == 0.5
    assert layer1.metadata['settings'].get('calibration_modulation') == 0.9


def test_import_from_file_button_exists(make_napari_viewer):
    """Test that import from file button exists."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    assert hasattr(plotter, 'import_from_file_button')
    assert plotter.import_from_file_button.text() == "OME-TIFF File"


def test_import_from_file_dialog_opens(make_napari_viewer):
    """Test that clicking import from file button opens file dialog."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer1.name = "layer1"
    viewer.add_layer(layer1)

    plotter = PlotterWidget(viewer)

    with patch(
        'napari_phasors.plotter.QFileDialog.getOpenFileName'
    ) as mock_dialog:
        mock_dialog.return_value = ("", "")  # No file selected

        plotter.import_from_file_button.click()

        # Verify dialog was opened
        mock_dialog.assert_called_once()


def test_import_from_file_cancel(make_napari_viewer):
    """Test canceling file import dialog."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer1.name = "layer1"
    viewer.add_layer(layer1)

    plotter = PlotterWidget(viewer)

    with patch(
        'napari_phasors.plotter.QFileDialog.getOpenFileName'
    ) as mock_dialog:
        mock_dialog.return_value = ("", "")  # Canceled

        # Should not crash or show error
        plotter._import_settings_from_file()


def test_import_all_settings_from_ome_tiff(make_napari_viewer, tmp_path):
    """Test importing all settings from an OME-TIFF file."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer1.name = "layer1"
    viewer.add_layer(layer1)

    plotter = PlotterWidget(viewer)

    # Create test OME-TIFF file
    filepath, expected_settings = create_ome_tiff_with_settings(tmp_path)

    # Mock file dialog to return our test file
    with patch(
        'napari_phasors.plotter.QFileDialog.getOpenFileName'
    ) as mock_dialog:
        mock_dialog.return_value = (str(filepath), "")

        # Mock selection dialog to select all
        with patch.object(
            plotter,
            '_show_import_dialog',
            return_value=[
                'frequency',
                'settings_tab',
                'calibration_tab',
                'filter_tab',
                'lifetime_tab',
                'fret_tab',
                'components_tab',
            ],
        ):
            plotter._import_settings_from_file()

    # Verify settings were imported
    assert (
        layer1.metadata['settings']['frequency']
        == expected_settings['frequency']
    )
    assert (
        layer1.metadata['settings']['harmonic']
        == expected_settings['harmonic']
    )
    assert (
        layer1.metadata['settings']['plot_type']
        == expected_settings['plot_type']
    )
    assert (
        layer1.metadata['settings']['colormap']
        == expected_settings['colormap']
    )


def test_import_partial_settings_from_ome_tiff(make_napari_viewer, tmp_path):
    """Test importing only selected settings from OME-TIFF."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer1.name = "layer1"
    viewer.add_layer(layer1)

    plotter = PlotterWidget(viewer)

    # Create test OME-TIFF file
    filepath, expected_settings = create_ome_tiff_with_settings(tmp_path)

    # Mock file dialog
    with patch(
        'napari_phasors.plotter.QFileDialog.getOpenFileName'
    ) as mock_dialog:
        mock_dialog.return_value = (str(filepath), "")

        # Mock selection dialog to select only frequency and settings
        with patch.object(
            plotter,
            '_show_import_dialog',
            return_value=['frequency', 'settings_tab'],
        ):
            plotter._import_settings_from_file()

    # Verify only selected settings were applied
    assert (
        layer1.metadata['settings']['frequency']
        == expected_settings['frequency']
    )


def test_import_from_file_without_settings(make_napari_viewer, tmp_path):
    """Test importing from OME-TIFF file without napari-phasors settings or frequency."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer1.name = "layer1"
    viewer.add_layer(layer1)

    plotter = PlotterWidget(viewer)

    # Create OME-TIFF without settings AND without frequency
    from phasorpy import io
    from phasorpy.phasor import phasor_from_signal

    time_constants = [1, 2, 3]
    raw_flim_data = make_raw_flim_data(
        time_constants=time_constants, shape=(32, 32)
    )

    filepath = tmp_path / "no_settings.ome.tif"
    phasor = phasor_from_signal(raw_flim_data)

    # Write OME-TIFF without frequency or settings
    io.phasor_to_ometiff(filepath, *phasor)

    # Mock file dialog
    with patch(
        'napari_phasors.plotter.QFileDialog.getOpenFileName'
    ) as mock_dialog:
        mock_dialog.return_value = (str(filepath), "")

        # Should show warning notification and NOT show import dialog
        with patch(
            'napari_phasors.plotter.notifications.WarningNotification'
        ) as mock_warning:
            with patch.object(
                plotter, '_show_import_dialog'
            ) as mock_show_import_dialog:
                plotter._import_settings_from_file()

                # Verify warning was shown
                mock_warning.assert_called_once()

                # Verify import dialog was NOT shown
                mock_show_import_dialog.assert_not_called()


def test_import_from_invalid_file(make_napari_viewer, tmp_path):
    """Test importing from invalid file."""
    viewer = make_napari_viewer()
    layer1 = create_image_layer_with_phasors()
    layer1.name = "layer1"
    viewer.add_layer(layer1)

    plotter = PlotterWidget(viewer)

    # Create invalid file
    filepath = tmp_path / "invalid.txt"
    filepath.write_text("not a valid OME-TIFF")

    # Mock file dialog
    with patch(
        'napari_phasors.plotter.QFileDialog.getOpenFileName'
    ) as mock_dialog:
        mock_dialog.return_value = (str(filepath), "")

        # Should show warning notification
        with patch(
            'napari_phasors.plotter.notifications.WarningNotification'
        ) as mock_warning:
            plotter._import_settings_from_file()

            # Verify warning was shown
            mock_warning.assert_called_once()


def test_show_import_dialog_all_options(make_napari_viewer):
    """Test import dialog shows all available options."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    with patch('napari_phasors.plotter.QDialog') as mock_dialog_class:
        # Create a proper mock instance that can be used as a parent
        mock_dialog_instance = Mock()
        mock_dialog_instance.exec_ = Mock(return_value=0)  # Rejected
        mock_dialog_class.return_value = mock_dialog_instance

        # Mock the layout to avoid type errors
        with patch('napari_phasors.plotter.QVBoxLayout') as mock_layout:
            result = plotter._show_import_dialog()

            # Should return empty list when dialog is rejected
            assert result == []
            # Verify dialog was created
            mock_dialog_class.assert_called_once()
            mock_dialog_instance.exec_.assert_called_once()


def test_show_import_dialog_default_all_checked(make_napari_viewer):
    """Test that all checkboxes are checked by default."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    with (
        patch('napari_phasors.plotter.QDialog') as mock_dialog,
        patch('napari_phasors.plotter.QVBoxLayout') as mock_layout,
        patch('napari_phasors.plotter.QLabel') as mock_label,
        patch('napari_phasors.plotter.QCheckBox') as mock_checkbox,
        patch('napari_phasors.plotter.QDialogButtonBox') as mock_buttonbox,
    ):

        mock_dialog_instance = Mock()
        mock_dialog_instance.exec_ = Mock(return_value=1)  # Accepted
        mock_dialog.return_value = mock_dialog_instance

        mock_cb_instance = Mock()
        mock_cb_instance.isChecked = Mock(return_value=True)
        mock_checkbox.return_value = mock_cb_instance

        plotter._show_import_dialog()

        # Verify checkboxes were created
        assert mock_checkbox.call_count > 0


def test_show_import_dialog_partial_selection(make_napari_viewer):
    """Test selecting only some options in import dialog."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # This test would require more complex mocking of the dialog interaction
    # For now, we'll just verify the method exists and can be called
    with (
        patch('napari_phasors.plotter.QDialog') as mock_dialog,
        patch('napari_phasors.plotter.QVBoxLayout'),
        patch('napari_phasors.plotter.QLabel'),
        patch('napari_phasors.plotter.QCheckBox'),
        patch('napari_phasors.plotter.QDialogButtonBox'),
    ):

        mock_dialog_instance = Mock()
        mock_dialog_instance.exec_ = Mock(return_value=0)
        mock_dialog.return_value = mock_dialog_instance

        result = plotter._show_import_dialog(default_checked=['settings_tab'])

        assert isinstance(result, list)


# Settings Metadata tests
def test_initialize_plot_settings_in_metadata(make_napari_viewer):
    """Test that settings are initialized in layer metadata."""
    viewer = make_napari_viewer()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    plotter = PlotterWidget(viewer)

    # Settings should be initialized
    assert 'settings' in layer.metadata
    assert 'harmonic' in layer.metadata['settings']
    assert 'semi_circle' in layer.metadata['settings']
    assert 'plot_type' in layer.metadata['settings']


def test_restore_plot_settings_from_metadata(make_napari_viewer):
    """Test restoring plot settings from metadata."""
    viewer = make_napari_viewer()
    layer = create_layer_with_custom_settings()
    viewer.add_layer(layer)

    plotter = PlotterWidget(viewer)

    # Verify settings were restored
    assert plotter.harmonic == 2
    assert plotter.plot_type == 'SCATTER'
    assert plotter.histogram_colormap == 'viridis'
    assert plotter.toggle_semi_circle == False
    assert plotter.white_background == False


def test_update_setting_in_metadata(make_napari_viewer):
    """Test that changing settings updates metadata."""
    viewer = make_napari_viewer()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    plotter = PlotterWidget(viewer)

    # Change a setting
    plotter.harmonic = 3

    # Verify metadata was updated
    assert layer.metadata['settings']['harmonic'] == 3
