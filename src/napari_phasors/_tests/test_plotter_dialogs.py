from napari_phasors.plotter import (
    ContourLayerSettingsDialog,
    MaskAssignmentDialog,
)


def test_contour_layer_settings_dialog(qtbot):
    layer_labels = ["Layer1", "Layer2"]
    dialog = ContourLayerSettingsDialog(
        display_mode="Merged",
        merged_colormap="turbo",
        merged_style="colormap",
        show_legend=True,
        grouped_color_mode="Use colormap",
        layer_labels=layer_labels,
        group_assignments={"Layer1": 1, "Layer2": 2},
        layer_colors={"Layer1": (1, 0, 0), "Layer2": (0, 1, 0)},
        group_names={1: "G1", 2: "G2"},
    )
    qtbot.addWidget(dialog)

    assert dialog.get_display_mode() == "Merged"
    assert dialog.get_individual_color_mode() == "Use colormap"
    assert dialog.get_grouped_color_mode() == "Use colormap"
    assert dialog.get_merged_colormap() == "turbo"
    assert dialog.get_merged_style() == "colormap"
    assert dialog.get_show_legend() is True

    # Test layer styles
    styles = dialog.get_layer_styles()
    assert "Layer1" in styles
    assert styles["Layer1"]["colormap"] == "turbo"

    # Test group names
    group_names = dialog.get_group_names()
    assert group_names[1] == "G1"
    assert group_names[2] == "G2"

    # Test removing group
    dialog._on_remove_group(dialog._group_row_data[1]["container"])
    assert len(dialog._group_row_data) == 1

    # Test adding group
    dialog._on_add_group()
    assert len(dialog._group_row_data) == 2

    # Test combobox change
    from unittest.mock import MagicMock, patch

    mock_color = MagicMock()
    mock_color.isValid.return_value = True
    mock_color.getRgbF.return_value = (1.0, 1.0, 1.0, 1.0)
    with patch(
        "qtpy.QtWidgets.QColorDialog.getColor", return_value=mock_color
    ):
        dialog._on_merged_colormap_changed("Select color...")
    assert dialog.get_merged_style() == "solid"


def test_mask_assignment_dialog(qtbot):
    from unittest.mock import MagicMock

    mask1, mask2 = MagicMock(), MagicMock()
    mask1.name = "Mask1"
    mask2.name = "Mask2"
    dialog = MaskAssignmentDialog(
        image_layer_names=["Layer1", "Layer2"],
        mask_layers=[mask1, mask2],
        current_assignments={"Layer1": "Mask1"},
    )
    qtbot.addWidget(dialog)

    assignments = dialog.get_assignments()
    assert assignments["Layer1"] == "Mask1"
    assert assignments["Layer2"] == "None"

    dialog._apply_all_combo.setCurrentText("None")
    dialog._apply_to_all()
    assignments = dialog.get_assignments()
    assert assignments["Layer1"] == "None"


def test_import_settings_from_layer_and_file(make_viewer_model, qtbot):
    from unittest.mock import patch

    import numpy as np

    from napari_phasors.plotter import PlotterWidget

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    viewer.add_image(
        np.random.random((10, 10)),
        name="Layer1",
        metadata={
            "G": np.array([1]),
            "S": np.array([1]),
            "G_original": 1,
            "S_original": 1,
        },
    )
    viewer.add_image(
        np.random.random((10, 10)),
        name="Layer2",
        metadata={
            "G": np.array([1]),
            "S": np.array([1]),
            "G_original": 1,
            "S_original": 1,
        },
    )

    with (
        patch("napari_phasors.plotter.QDialog.exec", return_value=1),
        patch(
            "napari_phasors.plotter.QComboBox.currentText",
            return_value="Layer2",
        ),
        patch.object(
            plotter, "_show_import_dialog", return_value=["settings_tab"]
        ),
        patch.object(plotter, "_copy_metadata_from_layer") as mock_copy,
    ):
        plotter._import_settings_from_layer()
        mock_copy.assert_called_once_with("Layer2", ["settings_tab"])

    with (
        patch(
            "napari_phasors.plotter.QFileDialog.getOpenFileName",
            return_value=("dummy.ome.tif", ""),
        ),
        patch(
            "phasorpy.io.phasor_from_ometiff",
            return_value=(
                None,
                None,
                None,
                {
                    "frequency": 80,
                    "description": '{"napari_phasors_settings": "{\\"key\\": \\"value\\"}"}',
                },
            ),
        ),
        patch.object(
            plotter, "_show_import_dialog", return_value=["settings_tab"]
        ),
        patch.object(plotter, "_apply_imported_settings") as mock_apply,
    ):
        plotter._import_settings_from_file()
        mock_apply.assert_called_once()


def test_prompt_user_for_analysis_import(make_viewer_model, qtbot):
    from unittest.mock import patch

    from napari_phasors.plotter import PlotterWidget

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    source_settings = {"frequency": 80, "calibrated": True}
    with patch("napari_phasors.plotter.QDialog.exec", return_value=1):
        selected = plotter._show_import_dialog(
            source_settings=source_settings,
            default_checked=["calibration_tab"],
        )
        # All available tabs for these settings are checked by default, plus frequency if requested
        assert isinstance(selected, list)


def test_phasor_center_settings_dialog(qtbot):
    from napari_phasors.plotter import PhasorCenterLayerSettingsDialog

    dialog = PhasorCenterLayerSettingsDialog(
        display_mode="Grouped",
        layer_labels=["L1", "L2"],
        group_assignments={"L1": 1, "L2": 1},
        group_names={1: "Group A"},
    )
    qtbot.addWidget(dialog)

    assert dialog.get_display_mode() == "Grouped"
    assert dialog.get_center_method() == "mean"

    dialog._size_spinbox.setValue(10)
    assert dialog.get_marker_size() == 10

    dialog._alpha_spinbox.setValue(0.5)
    assert dialog.get_alpha() == 0.5

    group_assignments = dialog.get_group_assignments()
    assert group_assignments["L1"] == 1
    assert group_assignments["L2"] == 1

    names = dialog.get_group_names()
    assert names[1] == "Group A"

    dialog._on_add_group()
    assert len(dialog._group_row_data) == 2

    dialog._on_remove_group(dialog._group_row_data[1]["container"])
    assert len(dialog._group_row_data) == 1
