from unittest.mock import MagicMock, patch

import numpy as np
from napari.layers import Image
from qtpy.QtCore import QEvent
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
)

from napari_phasors._mapping_filters import new_filter, set_filters
from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from napari_phasors._tests.test_plotter import (  # noqa: E501
    create_image_layer_with_phasors,
)
from napari_phasors._utils import apply_filter_and_threshold
from napari_phasors.plotter import (
    MaskAssignmentDialog,
    PlotterWidget,
    _apply_label_colors_to_combo,
)


def test_applying_and_restoring_a_mask_on_one_layer(make_viewer_model):
    """A Labels or Shapes mask blanks G/S outside it, optionally inverted,
    and restoring brings the original coordinates back."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Before any mask exists the selector reads "None" and Invert is off.
    assert isinstance(plotter.mask_layer_combobox, QComboBox)
    assert isinstance(plotter.mask_layer_label, QLabel)
    assert plotter.mask_layer_combobox.currentText() == "None"
    assert isinstance(plotter.mask_invert_checkbox, QCheckBox)
    assert not plotter.mask_invert_checkbox.isChecked()
    assert not plotter.mask_invert_checkbox.isEnabled()

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    original_g = layer.metadata["G"].copy()
    original_s = layer.metadata["S"].copy()
    shape = _make_mask_shape(layer)
    half = shape[0] // 2

    def g_2d():
        g = layer.metadata["G"]
        return g[0] if g.ndim == 3 else g

    # An empty mask selects nothing, so it is ignored; inverted, it keeps
    # every pixel.
    empty = viewer.add_labels(np.zeros(shape, dtype=int), name="empty")
    plotter._apply_mask_to_phasor_data(empty, layer, invert=False)
    np.testing.assert_array_equal(layer.metadata["G"], original_g)
    assert "mask" not in layer.metadata
    plotter._apply_mask_to_phasor_data(empty, layer, invert=True)
    np.testing.assert_array_equal(
        np.isnan(layer.metadata["G"]), np.isnan(original_g)
    )

    # A bottom-half mask blanks the top half; inverted, the bottom half.
    mask_data = np.zeros(shape, dtype=int)
    mask_data[half:, :] = 1
    labels_layer = viewer.add_labels(mask_data, name="test_mask")
    plotter._apply_mask_to_phasor_data(labels_layer, layer, invert=False)
    assert "mask" in layer.metadata
    assert np.isnan(layer.metadata["S"]).sum() > 0
    assert np.isnan(g_2d()[:half, :]).all()
    assert not np.isnan(g_2d()[half:, :]).all()

    plotter._restore_original_phasor_data(layer)
    np.testing.assert_array_almost_equal(layer.metadata["G"], original_g)
    np.testing.assert_array_almost_equal(layer.metadata["S"], original_s)

    plotter._apply_mask_to_phasor_data(labels_layer, layer, invert=True)
    assert not np.isnan(g_2d()[:half, :]).all()
    assert np.isnan(g_2d()[half:, :]).all()
    plotter._restore_original_phasor_data(layer)

    # A Shapes layer is rasterised into a mask.
    rect = np.array(
        [[0, 0], [0, shape[1]], [shape[0], shape[1]], [shape[0], 0]]
    )
    shapes_layer = viewer.add_shapes(
        [rect], shape_type="polygon", name="shape_mask"
    )
    plotter._apply_mask_to_phasor_data(shapes_layer, layer)
    assert "mask" in layer.metadata

    # Applying without a label subset drops one stored earlier.
    plotter._apply_mask_array_to_phasor_data(mask_data, layer, labels=[1])
    assert "mask_labels" in layer.metadata
    plotter._apply_mask_array_to_phasor_data(mask_data, layer, labels=None)
    assert "mask_labels" not in layer.metadata


def test_renaming_layers_keeps_the_selectors_in_sync(make_viewer_model):
    """Renamed image and mask layers stay selected and assigned under their
    new names, including an image that only gains phasors later."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    raw = Image(np.ones((10, 10)), name="raw_image")
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    viewer.add_layer(raw)
    labels_layer = viewer.add_labels(
        np.ones(_make_mask_shape(layer1), dtype=int),
        name="mask_before_rename",
    )
    plotter = PlotterWidget(viewer)
    combobox = plotter.image_layers_checkable_combobox

    # A layer without phasors is not offered, but its renames are tracked
    # so it shows up under its current name once it has them.
    assert "raw_image" not in combobox.allItems()
    raw.name = "renamed_raw_image"
    raw.metadata = {
        "G": np.ones((10, 10)),
        "S": np.ones((10, 10)),
        "G_original": np.ones((10, 10)),
        "S_original": np.ones((10, 10)),
        "harmonics": [1],
    }
    plotter.reset_layer_choices()
    assert "renamed_raw_image" in combobox.allItems()

    # Renaming a selected layer keeps it selected under the new name.
    combobox.setCheckedItems(["renamed_raw_image"])
    raw.name = "final_image_name"
    assert plotter.get_selected_layer_names() == ["final_image_name"]
    assert "final_image_name" in combobox.allItems()
    assert "renamed_raw_image" not in combobox.allItems()

    combobox.setCheckedItems([layer1.name])
    old_name = layer1.name
    layer1.name = "renamed_image_layer"
    assert plotter.get_selected_layer_names() == ["renamed_image_layer"]
    assert "renamed_image_layer" in combobox.allItems()
    assert old_name not in combobox.allItems()

    # Renaming the mask layer updates the selector and the assignment.
    plotter.mask_layer_combobox.setCurrentText("mask_before_rename")
    assert plotter.mask_layer_combobox.currentText() == "mask_before_rename"
    assert plotter._mask_assignments.get(layer1.name) == "mask_before_rename"
    labels_layer.name = "mask_after_rename"
    combo_items = [
        plotter.mask_layer_combobox.itemText(i)
        for i in range(plotter.mask_layer_combobox.count())
    ]
    assert "mask_after_rename" in combo_items
    assert "mask_before_rename" not in combo_items
    assert plotter.mask_layer_combobox.currentText() == "mask_after_rename"
    assert plotter._mask_assignments.get(layer1.name) == "mask_after_rename"

    # Re-running UI sync must not reset selection to None.
    plotter._update_mask_ui_mode()
    assert plotter.mask_layer_combobox.currentText() == "mask_after_rename"


def test_assigning_masks_to_several_layers(make_viewer_model):
    """Each selected layer gets its own mask and Invert flag, the summary
    counts the masked layers, and a repainted mask is re-applied only to
    the layers using it."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # With no layer selected a repainted mask has nothing to update.
    stray = viewer.add_labels(np.ones((5, 5), dtype=int), name="stray")
    plotter._on_mask_data_changed(type("Event", (), {"source": stray})())
    viewer.layers.remove(stray)

    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    shape = _make_mask_shape(layer1)
    half = shape[0] // 2
    bottom = np.zeros(shape, dtype=int)
    bottom[half:, :] = 1
    top = np.zeros(shape, dtype=int)
    top[:half, :] = 1
    mask_bottom = viewer.add_labels(bottom, name="mask_bottom")
    mask_top = viewer.add_labels(top, name="mask_top")

    def g_2d(layer):
        g = layer.metadata["G"]
        return g[0] if g.ndim == 3 else g

    def repaint(mask_layer):
        event = type("Event", (), {"source": mask_layer})()
        plotter._on_mask_data_changed(event)

    # The summary button is the whole row, whatever the selection count;
    # the editor controls live in the popover, not in the row.
    combobox = plotter.image_layers_checkable_combobox
    for names in ([layer1.name], [layer1.name, layer2.name], [layer1.name]):
        combobox.setCheckedItems(names)
        assert not plotter.mask_summary_button.isHidden()
        assert not plotter.mask_button_label.isHidden()
    assert plotter.mask_layer_combobox.parent() is plotter.mask_editor_popover
    assert plotter.mask_invert_checkbox.parent() is plotter.mask_editor_popover

    # Several selected layers go to the per-layer assignment dialog.
    combobox.setCheckedItems([layer1.name, layer2.name])
    with (
        patch.object(plotter, '_open_mask_assignment_dialog') as dialog,
        patch.object(plotter, '_show_mask_editor_popover') as popover,
    ):
        plotter.mask_summary_button.click()
    dialog.assert_called_once()
    popover.assert_not_called()

    # Nothing assigned: the summary says so and a repaint changes nothing.
    plotter._mask_assignments = {}
    plotter._update_mask_summary_text()
    assert plotter._mask_summary_full_text == "None"
    repaint(mask_bottom)
    assert "mask" not in layer1.metadata
    assert "mask" not in layer2.metadata

    # One layer masked: counted, and the only one a repaint touches.
    plotter._mask_assignments = {layer1.name: mask_bottom.name}
    assert plotter.get_mask_for_layer(layer1.name) == mask_bottom.name
    assert plotter.get_mask_for_layer(layer2.name) == "None"
    plotter._update_mask_summary_text()
    assert plotter._mask_summary_full_text == "1 of 2 layers"
    assert "1 of 2 selected layers masked" in (
        plotter.mask_summary_button.toolTip()
    )
    repaint(mask_bottom)
    assert "mask" in layer1.metadata
    assert "mask" not in layer2.metadata

    plotter._mask_assignments = {
        layer1.name: mask_bottom.name,
        layer2.name: mask_bottom.name,
    }
    plotter._update_mask_summary_text()
    assert plotter._mask_summary_full_text == "2 of 2 layers"

    # Distinct masks per layer.
    plotter._apply_mask_assignments(
        {layer1.name: mask_bottom.name, layer2.name: mask_top.name}
    )
    assert np.isnan(g_2d(layer1)[:half, :]).all()
    assert not np.isnan(g_2d(layer1)[half:, :]).all()
    assert np.isnan(g_2d(layer2)[half:, :]).all()
    assert not np.isnan(g_2d(layer2)[:half, :]).all()
    assert plotter._mask_assignments[layer1.name] == mask_bottom.name
    assert plotter._mask_assignments[layer2.name] == mask_top.name

    # The same mask, inverted for one layer only.
    plotter._apply_mask_assignments(
        {layer1.name: mask_bottom.name, layer2.name: mask_bottom.name},
        invert_assignments={layer1.name: False, layer2.name: True},
    )
    assert np.isnan(g_2d(layer1)[:half, :]).all()
    assert np.isnan(g_2d(layer2)[half:, :]).all()

    # Accepting the dialog applies its Invert choices too.
    mock_dialog = MagicMock()
    mock_dialog.exec.return_value = 1  # QDialog.Accepted
    mock_dialog.get_assignments.return_value = {
        layer1.name: mask_top.name,
        layer2.name: "None",
    }
    mock_dialog.get_invert_assignments.return_value = {
        layer1.name: True,
        layer2.name: False,
    }
    with patch(
        "napari_phasors.plotter.MaskAssignmentDialog",
        return_value=mock_dialog,
    ):
        plotter._open_mask_assignment_dialog()
    assert plotter._mask_invert_assignments.get(layer1.name, False)

    # Assigning "None" removes a layer's mask.
    combobox.setCheckedItems([layer1.name])
    plotter._apply_mask_assignments({layer1.name: "None"})
    assert 'mask' not in layer1.metadata


def test_mask_assignment_dialog_set_all_to():
    """Rows start from the current assignments (unknown masks fall back to
    "None"), and "Set all to" mirrors the shared mask, or shows a
    placeholder while the rows differ."""

    def open_dialog(current, names=("layer_A", "layer_B")):
        return MaskAssignmentDialog(
            image_layer_names=list(names),
            mask_layer_names=["mask_1", "mask_2"],
            current_assignments=current,
            parent=None,
        )

    dialog = open_dialog(
        {"layer_A": "mask_1", "layer_C": "mask_2", "layer_D": "deleted"},
        names=("layer_A", "layer_B", "layer_C", "layer_D"),
    )
    assert dialog.get_assignments() == {
        "layer_A": "mask_1",
        "layer_B": "None",
        "layer_C": "mask_2",
        "layer_D": "None",
    }
    # Differing masks always show the placeholder, never a stale value.
    assert dialog._apply_all_combo.currentText() == "Select ..."
    dialog.close()

    # Selecting the placeholder must not clobber the per-layer choices.
    dialog = open_dialog({"layer_A": "mask_1", "layer_B": "None"})
    assert dialog._apply_all_combo.currentText() == "Select ..."
    assert dialog.get_assignments() == {"layer_A": "mask_1", "layer_B": "None"}
    dialog.close()

    # Matching rows show their shared mask, live as the rows are edited.
    dialog = open_dialog({"layer_A": "mask_1", "layer_B": "mask_1"})
    assert dialog._apply_all_combo.currentText() == "mask_1"
    dialog._combos["layer_B"].setCurrentText("mask_2")
    assert dialog._apply_all_combo.currentText() == "Select ..."
    dialog._combos["layer_B"].setCurrentText("mask_1")
    assert dialog._apply_all_combo.currentText() == "mask_1"
    dialog.close()

    # Picking a mask in "Set all to" assigns it to every row.
    dialog = open_dialog(None)
    dialog._apply_all_combo.setCurrentText("mask_2")
    assert dialog.get_assignments() == {
        "layer_A": "mask_2",
        "layer_B": "mask_2",
    }
    dialog.close()


def test_mask_assignment_dialog_auto_assign():
    """Auto-assign matches each layer to its mask by name and leaves near
    misses alone; without masks it is disabled and a no-op."""
    dialog = MaskAssignmentDialog(
        image_layer_names=["layer_A", "layer_B"],
        mask_layer_names=[],
        parent=None,
    )
    assert not dialog.auto_assign_button.isEnabled()
    # The guard holds even if the disabled button is bypassed.
    dialog._on_auto_assign()
    assert dialog.get_assignments() == {"layer_A": "None", "layer_B": "None"}
    dialog.close()

    images = [
        "embryo_1.ptu Intensity [Phasor]",
        "embryo_2.ptu Intensity [Phasor]",
        "unmatched_image.ptu Intensity [Phasor]",
    ]
    dialog = MaskAssignmentDialog(
        image_layer_names=images,
        mask_layer_names=["embryo_2_segmentation", "embryo_1", "random_mask"],
        parent=None,
    )
    assert dialog.auto_assign_button.isEnabled()
    for combo in dialog._combos.values():
        assert combo.currentText() == "None"
    dialog.auto_assign_button.click()
    assignments = dialog.get_assignments()
    assert assignments[images[0]] == "embryo_1"
    assert assignments[images[1]] == "embryo_2_segmentation"
    # No name match: left alone rather than given a near-miss mask.
    assert assignments[images[2]] == "None"
    dialog.close()

    # A layer whose only sibling mask belongs to another image stays None.
    image = "2026-05-03_control.lsm Intensity [Phasor]"
    dialog = MaskAssignmentDialog(
        image_layer_names=[image],
        mask_layer_names=["2026-05-03_treated_mask"],
        parent=None,
    )
    dialog.auto_assign_button.click()
    assert dialog.get_assignments() == {image: "None"}
    dialog.close()

    # Auto-assign goes through the combo, so the row's widgets follow: the
    # Invert checkbox is only meaningful once a mask is assigned.
    image = "sample.tif Intensity [Phasor]"
    dialog = MaskAssignmentDialog(
        image_layer_names=[image],
        mask_layer_names=["sample_mask"],
        parent=None,
    )
    invert = dialog._invert_checks[image]
    assert not invert.isEnabled()
    dialog.auto_assign_button.click()
    assert dialog.get_assignments() == {image: "sample_mask"}
    assert invert.isEnabled()
    dialog.close()


def _make_mask_shape(layer):
    """Get the spatial shape for creating a mask from a layer."""
    G = layer.metadata["G"]
    return G.shape[1:] if G.ndim == 3 else G.shape


# --- Feature 1: Invert Mask (single-layer mode) ---


def test_mask_selector_with_a_labels_layer(make_viewer_model):
    """Choosing a Labels mask enables Invert and lists its labels, all
    ticked and coloured; narrowing, inverting and repainting keep G/S in
    step, and "None" resets everything."""
    _, plotter, image_layer, labels_layer, labels_data = (
        _setup_plotter_with_labels(make_viewer_model)
    )
    name = image_layer.name
    combo = plotter.mask_labels_combobox

    def masked():
        g = image_layer.metadata["G"]
        return np.isnan(g[0] if g.ndim == 3 else g)

    assert plotter.mask_invert_checkbox.isEnabled()

    # Every label is ticked by default, stored as "no label filter".
    assert not combo.isHidden()
    assert combo.allItems() == ["1", "2", "3"]
    assert combo.checkedItems() == combo.allItems()
    assert plotter._mask_label_assignments.get(name) is None
    combo.selectAll()
    assert combo.lineEdit().text() == ""
    assert combo.lineEdit().placeholderText() == "All Labels"
    assert plotter._mask_label_assignments.get(name, "MISSING") is None

    # Each item is painted in its label's colour.
    expected_bg = QColor(160, 160, 160, 160)
    for i, label in enumerate([1, 2, 3]):
        item = combo.model().item(i)
        r, g, b = (int(c * 255) for c in labels_layer.get_color(label)[:3])
        assert item.foreground().color() == QColor(r, g, b)
        assert item.background().color() == expected_bg

    # The row carries the combobox's select all/none buttons; with every
    # label checked, "all" is a no-op and "none" is not.
    assert plotter.mask_labels_select_buttons is combo.select_all_buttons
    assert not combo._select_all_button.isEnabled()
    assert combo._select_none_button.isEnabled()
    combo.deselectAll()
    assert combo.lineEdit().text() == "No labels"
    assert combo._select_all_button.isEnabled()
    assert not combo._select_none_button.isEnabled()
    combo._select_all_button.click()
    assert combo.checkedItems() == combo.allItems()
    assert not combo._select_all_button.isEnabled()

    # A newly painted label is added, keeping the ticked ones ticked.
    labels_data = labels_data.copy()
    labels_data[: labels_data.shape[0] // 2, :] = 4
    labels_layer.data = labels_data
    plotter._refresh_mask_labels_combobox(labels_layer)
    assert combo.allItems() == ["1", "2", "3", "4"]
    assert "1" in combo.checkedItems()

    combo.setCheckedItems(["1", "3"])
    assert combo.lineEdit().text() == "2 labels selected"

    # A single label keeps only its pixels.
    combo.setCheckedItems(["2"])
    assert plotter._mask_label_assignments[name] == [2]
    assert combo.checkedItems() == ["2"]
    np.testing.assert_array_equal(masked(), labels_data != 2)

    # A refresh with the same labels leaves the selection alone.
    plotter._refresh_mask_labels_combobox(labels_layer)
    assert combo.checkedItems() == ["2"]

    # Inverting keeps the label subset and flips which pixels survive.
    plotter.mask_invert_checkbox.setChecked(True)
    assert plotter.mask_invert_checkbox.isChecked()
    assert plotter._mask_invert_assignments[name] is True
    assert plotter._mask_label_assignments[name] == [2]
    assert combo.checkedItems() == ["2"]
    np.testing.assert_array_equal(masked(), labels_data == 2)

    # Clearing the mask resets Invert and drops the stored label subset.
    assert image_layer.metadata["mask_labels"] == [2]
    plotter.mask_layer_combobox.setCurrentText("None")
    assert not plotter.mask_invert_checkbox.isChecked()
    assert not plotter.mask_invert_checkbox.isEnabled()
    assert "mask_labels" not in image_layer.metadata
    assert name not in plotter._mask_label_assignments


# --- Feature 1: Invert Mask (multi-layer mode) ---


def test_mask_assignment_dialog_invert_controls():
    """Each row's Invert box is enabled once the row has a mask, and
    "Invert All" toggles every enabled row and follows them."""
    # No masks assigned: every row is off and "Invert All" is disabled.
    dialog = MaskAssignmentDialog(
        image_layer_names=["img1", "img2"],
        mask_layer_names=["mask1"],
        current_assignments={},
    )
    assert dialog.get_invert_assignments() == {"img1": False, "img2": False}
    dialog.close()

    dialog = MaskAssignmentDialog(
        image_layer_names=["img1", "img2"],
        mask_layer_names=["mask1"],
        current_assignments={"img1": "None", "img2": "None"},
    )
    assert not dialog.invert_all_check.isEnabled()
    assert not dialog.invert_all_check.isChecked()

    # Assign mask to img1 -> Invert All becomes enabled
    dialog._combos["img1"].setCurrentText("mask1")
    assert dialog.invert_all_check.isEnabled()
    assert not dialog.invert_all_check.isChecked()

    # Click Invert All -> only img1 is inverted (img2 is disabled)
    dialog.invert_all_check.click()
    assert dialog.invert_all_check.isChecked()
    assert dialog._invert_checks["img1"].isChecked()
    assert not dialog._invert_checks["img2"].isChecked()
    assert not dialog._invert_checks["img2"].isEnabled()
    assert dialog.get_invert_assignments() == {"img1": True, "img2": False}

    # img2 is newly enabled and unchecked, so Invert All reflects that
    dialog._combos["img2"].setCurrentText("mask1")
    assert not dialog.invert_all_check.isChecked()

    # Click Invert All -> both are now checked
    dialog.invert_all_check.click()
    assert dialog.invert_all_check.isChecked()
    assert dialog._invert_checks["img1"].isChecked()
    assert dialog._invert_checks["img2"].isChecked()
    dialog.close()

    # Rows that start assigned and not inverted.
    dialog = MaskAssignmentDialog(
        image_layer_names=["img1", "img2"],
        mask_layer_names=["mask1"],
        current_assignments={"img1": "mask1", "img2": "mask1"},
        current_invert_assignments={"img1": False, "img2": False},
    )
    assert dialog.invert_all_check.isEnabled()
    assert not dialog.invert_all_check.isChecked()

    dialog.invert_all_check.click()
    assert dialog.invert_all_check.isChecked()
    assert dialog.get_invert_assignments() == {"img1": True, "img2": True}

    # Unchecking one row unchecks Invert All; re-checking it restores it.
    dialog._invert_checks["img1"].setChecked(False)
    assert not dialog.invert_all_check.isChecked()
    assert dialog._invert_checks["img2"].isChecked()
    dialog._invert_checks["img1"].setChecked(True)
    assert dialog.invert_all_check.isChecked()

    # Click Invert All again to uncheck -> both become unchecked
    dialog.invert_all_check.click()
    assert not dialog.invert_all_check.isChecked()
    assert not dialog._invert_checks["img1"].isChecked()
    assert not dialog._invert_checks["img2"].isChecked()

    # The row callbacks fire while the dialog is still being built, before
    # the checkbox exists: a no-op rather than an AttributeError.
    saved = dialog.invert_all_check
    del dialog.invert_all_check
    dialog._sync_invert_all_check()
    dialog.invert_all_check = saved
    dialog.close()


def test_deleting_a_mask_layer_unmasks_the_layers_using_it(make_viewer_model):
    """Removing a mask layer clears its mask in single and multi-layer mode."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    mask = _make_spatial_mask(layer1)

    # One layer selected: the mask selector falls back to "None".
    plotter.image_layers_checkable_combobox.setCheckedItems([layer1.name])
    single = viewer.add_labels(mask, name="single_mask")
    plotter.mask_layer_combobox.setCurrentText(single.name)
    assert 'mask' in layer1.metadata
    viewer.layers.remove(single)
    assert plotter.mask_layer_combobox.currentText() == "None"
    assert 'mask' not in layer1.metadata

    # Several layers selected: each re-applies what is left of its
    # assignment, which for the deleted mask is nothing.
    plotter.image_layers_checkable_combobox.setCheckedItems([layer2.name])
    shared = viewer.add_labels(mask, name="shared_mask")
    plotter.mask_layer_combobox.setCurrentText(shared.name)
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )
    assert plotter._mask_assignments[layer2.name] == shared.name
    viewer.layers.remove(shared)
    assert plotter._mask_assignments == {}
    assert 'mask' not in layer1.metadata
    assert 'mask' not in layer2.metadata

    # Regression: a mask assigned per layer through the assignment dialog
    # is not the one the selector shows, and deleting it used to leave the
    # layers masked.
    per_layer = viewer.add_labels(mask, name="per_layer_mask")
    plotter._apply_mask_assignments(
        {layer1.name: per_layer.name, layer2.name: per_layer.name}
    )
    assert plotter.mask_layer_combobox.currentText() == "None"
    assert np.isnan(layer1.metadata['G']).any()
    viewer.layers.remove(per_layer)
    assert plotter._mask_assignments == {}
    for layer in (layer1, layer2):
        assert 'mask' not in layer.metadata
        np.testing.assert_array_equal(
            np.isnan(layer.metadata['G']),
            np.isnan(layer.metadata['G_original']),
        )


def test_restoring_a_stored_mask_recreates_its_layer(make_viewer_model):
    """A stored mask with no matching Labels/Shapes layer gets a new one."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    mask = _make_spatial_mask(layer)
    layer.metadata['mask'] = mask
    # A Shapes layer is rasterised for the comparison, but does not match.
    viewer.add_shapes(
        [np.array([[0, 0], [0, 3], [3, 3], [3, 0]])],
        shape_type="polygon",
        name="unrelated",
    )

    plotter._restore_plot_settings_from_metadata()

    restored = f"Restored Mask: {layer.name}"
    assert restored in viewer.layers
    np.testing.assert_array_equal(viewer.layers[restored].data, mask)
    assert plotter.mask_layer_combobox.currentText() == restored


def _setup_plotter_with_labels(make_viewer_model, n_labels=3):
    """Return (viewer, plotter, image_layer, labels_layer, labels_data)."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    image_layer = create_image_layer_with_phasors()
    viewer.add_layer(image_layer)
    plotter.image_layers_checkable_combobox.setCheckedItems([image_layer.name])

    G = image_layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    h, w = shape
    labels_data = np.zeros(shape, dtype=int)
    for idx in range(n_labels):
        col_start = idx * (w // n_labels)
        col_end = (idx + 1) * (w // n_labels) if idx < n_labels - 1 else w
        labels_data[:, col_start:col_end] = idx + 1

    labels_layer = viewer.add_labels(labels_data, name="lbl")
    plotter.mask_layer_combobox.setCurrentText("lbl")
    return viewer, plotter, image_layer, labels_layer, labels_data


# ---------------------------------------------------------------------------
# 1. Default all-checked state when mask layer is first selected
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Display text for all selection states
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. Normalization: all-checked is canonically equal to empty list
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. _extract_phasor_arrays_from_layer: mask_labels and mask_invert
# ---------------------------------------------------------------------------


def test_extract_phasor_arrays_honours_mask_labels_and_invert(
    make_viewer_model,
):
    """The stored mask keeps pixels by label subset (None: every label,
    []: no masking), optionally inverted; without a mask nothing changes."""
    from napari_phasors._utils import _extract_phasor_arrays_from_layer

    image_layer = create_image_layer_with_phasors()
    make_viewer_model().add_layer(image_layer)

    mean, _, _, _ = _extract_phasor_arrays_from_layer(image_layer)
    np.testing.assert_array_equal(mean, image_layer.metadata["original_mean"])
    assert not np.any(np.isnan(mean))

    shape = _make_mask_shape(image_layer)
    third = shape[1] // 3
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, :third] = 1
    mask_data[:, third : 2 * third] = 2
    unmasked = np.zeros(shape, dtype=bool)
    image_layer.metadata["mask"] = mask_data
    for labels, invert, expected_nan in (
        (None, False, mask_data <= 0),
        ([], False, unmasked),
        ([1], False, mask_data != 1),
        ([1], True, mask_data == 1),
        (None, True, mask_data > 0),
        ([], True, unmasked),
    ):
        image_layer.metadata["mask_labels"] = labels
        image_layer.metadata["mask_invert"] = invert
        _, real, _, _ = _extract_phasor_arrays_from_layer(image_layer)
        np.testing.assert_array_equal(
            np.isnan(real[0]),
            expected_nan,
            err_msg=f"labels={labels}, invert={invert}",
        )


def test_masking_a_single_harmonic_layer(make_viewer_model):
    """G/S without a leading harmonic axis are masked directly, not per row
    as if a harmonic axis existed, both when extracted and when stored."""
    from napari_phasors._utils import _extract_phasor_arrays_from_layer

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    image_layer = create_image_layer_with_phasors(harmonic=1)
    viewer.add_layer(image_layer)

    G = image_layer.metadata["G"]
    assert G.ndim == image_layer.data.ndim
    mask_data = np.zeros(G.shape, dtype=int)
    mask_data[:, : G.shape[1] // 2] = 1  # left half label 1
    mask_data[:, G.shape[1] // 2 :] = 2  # right half label 2
    image_layer.metadata["mask"] = mask_data
    image_layer.metadata["mask_labels"] = None  # all labels valid
    image_layer.metadata["mask_invert"] = False

    _, real, imag, _ = _extract_phasor_arrays_from_layer(image_layer)
    assert real.shape == G.shape
    assert imag.shape == G.shape
    np.testing.assert_array_equal(np.isnan(real), mask_data <= 0)

    plotter._apply_mask_array_to_phasor_data(
        _make_spatial_mask(image_layer), image_layer
    )
    assert np.isnan(image_layer.metadata["G"]).sum() > 0
    assert np.isnan(image_layer.metadata["S"]).sum() > 0


# ---------------------------------------------------------------------------
# 5. MaskAssignmentDialog: defaults and get_label_assignments normalisation
# ---------------------------------------------------------------------------


def test_mask_assignment_dialog_label_selection(make_viewer_model):
    """Picking a Labels mask for a row lists its labels, all ticked and
    coloured like the layer, and all ticked is reported as None."""
    from napari_phasors._utils import CheckableComboBox

    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    shape = _make_mask_shape(layer1)
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1
    mask_data[:, shape[1] // 2 :] = 2
    labels_layer = viewer.add_labels(mask_data, name="lbl")
    expected_bg = QColor(160, 160, 160, 160)

    def assert_coloured(combo, labels):
        for i, label in enumerate(labels):
            item = combo.model().item(i)
            rgba = labels_layer.get_color(label)
            r, g, b = (int(c * 255) for c in rgba[:3])
            assert item.foreground().color() == QColor(r, g, b)
            assert item.background().color() == expected_bg

    dialog = MaskAssignmentDialog(
        image_layer_names=[layer1.name, layer2.name],
        mask_layer_names=["None", "lbl"],
        mask_layers=[labels_layer],
        current_assignments={layer1.name: "None", layer2.name: "None"},
        current_label_assignments={},
        current_invert_assignments={},
        parent=None,
    )
    dialog._combos[layer1.name].setCurrentText("lbl")
    label_combo = dialog._label_combos.get(layer1.name)
    assert label_combo is not None
    assert not label_combo.isHidden()
    assert label_combo.allItems() == ["1", "2"]
    assert label_combo.checkedItems() == ["1", "2"]
    assert_coloured(label_combo, [1, 2])
    dialog.close()

    # A row that starts on the mask is populated too.
    dialog = MaskAssignmentDialog(
        image_layer_names=[layer1.name],
        mask_layer_names=["None", "lbl"],
        mask_layers=[labels_layer],
        current_assignments={layer1.name: "lbl"},
        current_label_assignments={layer1.name: None},
        current_invert_assignments={},
        parent=None,
    )
    label_combo = dialog._label_combos.get(layer1.name)
    assert label_combo is not None
    label_combo.selectAll()
    assert dialog.get_label_assignments().get(layer1.name, "MISSING") is None
    dialog.close()

    # The shared helper colours any checkable combobox the same way.
    combo = CheckableComboBox(
        placeholder="All Labels",
        enable_primary_layer=False,
        unit="labels",
        no_selection_text="No labels",
    )
    combo.addItems(["1", "2"])
    _apply_label_colors_to_combo(
        combo, labels_layer, np.unique(labels_layer.data)
    )
    assert_coloured(combo, [1, 2])


def _make_spatial_mask(layer, fill=1):
    """Build a labels mask matching a phasor layer's spatial dimensions."""
    g_data = layer.metadata["G"]
    mask_shape = g_data.shape[1:] if g_data.ndim == 3 else g_data.shape
    mask = np.zeros(mask_shape, dtype=int)
    mask[mask_shape[0] // 2 :, :] = fill
    return mask


def test_copying_masking_between_layers(make_viewer_model):
    """Copying masking mirrors the source's mask, Invert flag and labels onto
    the target, clears it for an unmasked source, recreates a layer for a
    mask left without one, and skips masks that do not fit."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    source_layer = create_image_layer_with_phasors()
    target_layer = create_image_layer_with_phasors()
    viewer.add_layer(source_layer)
    viewer.add_layer(target_layer)
    combobox = plotter.image_layers_checkable_combobox
    shape = _make_mask_shape(source_layer)

    def warning_patch():
        return patch(
            "napari_phasors.plotter.notifications.WarningNotification"
        )

    # An empty Shapes layer is skipped when looking for a matching mask.
    empty_shapes = viewer.add_shapes([], name="empty_shapes")
    assert (
        plotter._find_mask_layer_for(
            _make_spatial_mask(target_layer), target_layer
        )
        is None
    )
    viewer.layers.remove(empty_shapes)

    # A stored mask whose shape differs from the target is not copied.
    source_layer.metadata["mask"] = np.ones((3, 3), dtype=int)
    source_layer.metadata["mask_invert"] = False
    with warning_patch() as warn:
        copied = plotter._copy_mask_from_layer(source_layer, target_layer)
    assert copied is False
    assert "mask" not in target_layer.metadata
    warn.assert_called_once()

    # A stored mask with no Labels/Shapes layer left is applied as an array
    # and given a new layer to represent it.
    orphan = _make_spatial_mask(source_layer, fill=2)
    source_layer.metadata["mask"] = orphan
    before_layers = set(viewer.layers)
    assert plotter._copy_mask_from_layer(source_layer, target_layer) is True
    (created,) = set(viewer.layers) - before_layers
    assert created.name.startswith("Restored Mask:")
    assert plotter._mask_assignments[target_layer.name] == created.name
    np.testing.assert_array_equal(target_layer.metadata["mask"], orphan)
    del source_layer.metadata["mask"], source_layer.metadata["mask_invert"]

    # An unmasked source removes the target's mask.
    combobox.setCheckedItems([target_layer.name])
    original_g = target_layer.metadata["G_original"]
    assert plotter._copy_mask_from_layer(source_layer, target_layer) is False
    assert "mask" not in target_layer.metadata
    assert "mask_invert" not in target_layer.metadata
    np.testing.assert_array_almost_equal(
        target_layer.metadata["G"], original_g
    )

    # A masked source passes its mask, Invert flag and labels on.
    src_mask = viewer.add_labels(
        _make_spatial_mask(source_layer), name="src_mask"
    )
    combobox.setCheckedItems([source_layer.name])
    plotter._apply_mask_to_phasor_data(
        src_mask, source_layer, invert=True, labels=[1]
    )
    assert plotter._copy_mask_from_layer(source_layer, target_layer) is True
    np.testing.assert_array_equal(
        target_layer.metadata["mask"], source_layer.metadata["mask"]
    )
    assert target_layer.metadata["mask_invert"] is True
    assert target_layer.metadata["mask_labels"] == [1]
    assert np.isnan(target_layer.metadata["G"]).sum() > 0
    assert plotter._mask_assignments[target_layer.name] == "src_mask"
    assert plotter._mask_invert_assignments[target_layer.name] is True
    assert plotter._mask_label_assignments[target_layer.name] == [1]

    # Regression: importing masking replaces the target's own mask
    # selection, invert flag and NaN pattern with the source's.
    plotter._apply_mask_to_phasor_data(src_mask, source_layer, invert=True)
    source_nan = np.isnan(source_layer.metadata["G"])
    tgt_arr = np.zeros(shape, dtype=int)
    tgt_arr[:, : shape[1] // 2] = 1
    tgt_mask = viewer.add_labels(tgt_arr, name="tgt_mask")
    combobox.setCheckedItems([target_layer.name])
    plotter._apply_mask_to_phasor_data(tgt_mask, target_layer)
    plotter.mask_layer_combobox.setCurrentText("tgt_mask")
    plotter._copy_metadata_from_layer(
        source_layer.name, selected_tabs=["masking"]
    )
    assert plotter._mask_assignments[target_layer.name] == "src_mask"
    assert plotter.mask_layer_combobox.currentText() == "src_mask"
    assert plotter._mask_invert_assignments[target_layer.name] is True
    np.testing.assert_array_equal(
        target_layer.metadata["mask"], source_layer.metadata["mask"]
    )
    np.testing.assert_array_equal(
        np.isnan(target_layer.metadata["G"]), source_nan
    )

    # The matching Labels layer is pixel-bound, unlike Shapes, so it is
    # skipped for a target of a different size.
    other_shape = (shape[0] + 2, shape[1] + 2)
    other = Image(
        np.zeros(other_shape),
        name="other",
        metadata={
            "G": np.zeros(other_shape),
            "S": np.zeros(other_shape),
            "G_original": np.zeros(other_shape),
            "S_original": np.zeros(other_shape),
            "original_mean": np.zeros(other_shape),
        },
    )
    viewer.add_layer(other)
    with warning_patch() as warn:
        copied = plotter._copy_mask_from_layer(source_layer, other)
    assert copied is False
    warn.assert_called_once()


def test_copy_mask_from_layer_shapes_mask_different_image_sizes(
    make_viewer_model,
):
    """A Shapes mask copies across images of different sizes.

    Regression: copying the source's rasterized mask array failed the
    size check when the target had different dimensions, silently leaving the
    target's own mask. Copying the mask *layer* re-rasterizes the shapes to the
    target's shape instead.
    """
    from napari_phasors._synthetic_generator import (
        make_intensity_layer_with_phasors,
        make_raw_flim_data,
    )

    def _layer(shape, name):
        raw = make_raw_flim_data(
            time_constants=[0.1, 1, 2, 3, 4, 5, 10], shape=shape
        )
        layer = make_intensity_layer_with_phasors(raw, harmonic=[1, 2, 3])
        layer.name = name
        return layer

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    source_layer = _layer((10, 10), "SOURCE")
    target_layer = _layer((6, 8), "TARGET")  # different dimensions
    viewer.add_layer(source_layer)
    viewer.add_layer(target_layer)

    shapes_src = viewer.add_shapes(
        [np.array([[0, 0], [5, 0], [5, 5], [0, 5]])],
        shape_type="polygon",
        name="shapes_src",
    )
    shapes_tgt = viewer.add_shapes(
        [np.array([[3, 4], [6, 4], [6, 8], [3, 8]])],
        shape_type="polygon",
        name="shapes_tgt",
    )

    plotter.image_layers_checkable_combobox.setCheckedItems(["SOURCE"])
    plotter._apply_mask_to_phasor_data(shapes_src, source_layer)
    plotter.image_layers_checkable_combobox.setCheckedItems(["TARGET"])
    plotter._apply_mask_to_phasor_data(shapes_tgt, target_layer)
    plotter.mask_layer_combobox.setCurrentText("shapes_tgt")

    plotter._copy_metadata_from_layer("SOURCE", selected_tabs=["masking"])

    # Target now uses the source's Shapes mask, rasterized to its own shape.
    assert plotter._mask_assignments[target_layer.name] == "shapes_src"
    assert plotter.mask_layer_combobox.currentText() == "shapes_src"
    expected = shapes_src.to_labels(labels_shape=target_layer.data.shape)
    np.testing.assert_array_equal(target_layer.metadata["mask"], expected)

    plotter.deleteLater()


def test_copy_masking_to_several_layers_keeps_invert(make_viewer_model):
    """Copying an inverted mask onto several layers keeps the invert flag.

    Regression: restoring the plot settings re-applied the primary layer's
    mask to every selected layer with the (unsynced) invert checkbox,
    silently turning the invert off.
    """

    def _layer(name):
        raw = make_raw_flim_data(
            time_constants=[0.1, 1, 2, 3, 4, 5, 10], shape=(10, 10)
        )
        layer = make_intensity_layer_with_phasors(raw, harmonic=[1, 2, 3])
        layer.name = name
        return layer

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    source, first, second = _layer("SRC"), _layer("T1"), _layer("T2")
    for layer in (source, first, second):
        viewer.add_layer(layer)
    mask = np.zeros((10, 10), dtype=int)
    mask[:, :5] = 1
    viewer.add_labels(mask, name="mask")

    combobox = plotter.image_layers_checkable_combobox
    combobox.setCheckedItems(["SRC"])
    plotter.mask_layer_combobox.setCurrentText("mask")
    plotter.mask_invert_checkbox.setChecked(True)
    source_nan = np.isnan(source.metadata["G"])

    combobox.setCheckedItems(["T1", "T2"])
    plotter._copy_metadata_from_layer("SRC", selected_tabs=["masking"])

    for target in (first, second):
        assert target.metadata["mask_invert"] is True
        assert plotter._mask_invert_assignments[target.name] is True
        np.testing.assert_array_equal(
            np.isnan(target.metadata["G"]), source_nan
        )

    plotter.deleteLater()


# -- Coverage gaps: masking toggle in the import dialog --------------------


def test_import_dialog_and_parallel_hint(make_viewer_model):
    """The import dialog offers Masking (checked by default) only when a mask
    is available, and the parallel hint quantifies the memory budget only
    while the budget is switched on."""
    from napari_phasors import _parallel

    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    source_settings = {"frequency": 80}
    with patch("napari_phasors.plotter.QDialog.exec", return_value=1):
        assert "masking" in plotter._show_import_dialog(
            source_settings=source_settings, mask_available=True
        )
        assert "masking" not in plotter._show_import_dialog(
            source_settings=source_settings,
            mask_available=True,
            default_checked=["settings_tab"],
        )
        assert "masking" not in plotter._show_import_dialog(
            source_settings=source_settings, mask_available=False
        )

    previous = _parallel.memory_budget_enabled()
    previous_items = _parallel.parallel_items_enabled()
    try:
        plotter.parallel_items_checkbox.setChecked(True)
        plotter.memory_budget_checkbox.setChecked(True)
        plotter._update_parallel_processing_hint()
        assert "sized to fit" in plotter.parallel_processing_hint.text()

        plotter.memory_budget_checkbox.setChecked(False)
        plotter._update_parallel_processing_hint()
        assert "sized to fit" not in plotter.parallel_processing_hint.text()
        assert "at once" in plotter.parallel_processing_hint.text()
    finally:
        _parallel.set_memory_budget_enabled(previous)
        _parallel.set_parallel_items_enabled(previous_items)


def test_reapplying_filters_after_a_mask_change(make_viewer_model):
    """Only layers with processing of their own are re-filtered, with the
    parameters stored for them, and a failure names the layer."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # An upper bound alone, or a metric filter alone, is worth reapplying.
    layer = create_image_layer_with_phasors()
    layer.metadata['settings'] = {}
    assert not plotter._has_filter_or_threshold_settings(layer)
    layer.metadata['settings'] = {'threshold_upper': 5.0}
    assert plotter._has_filter_or_threshold_settings(layer)
    layer.metadata['settings'] = {}
    set_filters(layer, [new_filter("Modulation", 0.1, 0.9)])
    assert plotter._has_filter_or_threshold_settings(layer)

    # Wavelet, median and an unknown method each yield usable parameters.
    layer = create_image_layer_with_phasors()
    layer.metadata['harmonics'] = np.array([1, 2])
    layer.metadata['settings']['filter'] = {
        'method': 'wavelet',
        'sigma': 3.0,
        'levels': 2,
    }
    params = plotter._filter_params_from_settings(layer)
    assert params['filter_method'] == 'wavelet'
    assert params['sigma'] == 3.0
    assert params['levels'] == 2
    assert params['harmonics'] is not None
    # Harmonics that wavelet filtering cannot handle drop the method.
    layer.metadata['harmonics'] = np.array([1, 5])
    params = plotter._filter_params_from_settings(layer)
    assert params['filter_method'] is None
    assert params['harmonics'] is None
    layer.metadata['settings']['filter'] = {'method': 'median', 'repeat': 0}
    assert plotter._filter_params_from_settings(layer)['filter_method'] is None
    layer.metadata['settings']['filter'] = {'method': 'something else'}
    assert plotter._filter_params_from_settings(layer)['filter_method'] is None

    primary = plotter.image_layer_with_phasor_features_combobox

    # Layers with no processing of their own are left untouched.
    plain = create_image_layer_with_phasors()
    plain.name = "plain"
    viewer.add_layer(plain)
    primary.setCurrentText("plain")
    before = plain.metadata['G'].copy()
    plotter._reapply_filter_and_threshold([])
    plotter._reapply_filter_and_threshold([plain])
    np.testing.assert_array_equal(plain.metadata['G'], before)

    # A layer whose filtering raises is named in a single error message.
    explodes = create_image_layer_with_phasors()
    explodes.name = "explodes"
    viewer.add_layer(explodes)
    primary.setCurrentText("explodes")
    explodes.metadata['settings']['threshold'] = 0.0
    errors = []
    with (
        patch(
            "napari_phasors.plotter.notifications.show_error", errors.append
        ),
        patch(
            "napari_phasors.plotter.apply_filter_and_threshold_to_layers",
            lambda pairs, **kwargs: [RuntimeError("boom") for _ in pairs],
        ),
    ):
        plotter._reapply_filter_and_threshold([explodes])
    assert errors and "explodes" in errors[0] and "boom" in errors[0]

    # An imported stack reaches the arrays without the Filter tab's help.
    imported = create_image_layer_with_phasors()
    imported.name = "imported"
    viewer.add_layer(imported)
    primary.setCurrentText("imported")
    set_filters(imported, [new_filter("Modulation", 0.0, 0.05)])
    plotter._apply_imported_analyses([imported], ["phasor_mapping_tab"])
    assert np.isnan(imported.metadata['G']).any()


# -- Coverage gaps: _apply_mask_array_to_phasor_data branches ---------------


# -- Coverage gaps: _copy_mask_from_layer / _find_mask_layer_for branches --


def test_mask_labels_split_histogram_and_statistics(make_viewer_model, qtbot):
    """Masking with several labels lets each label be analysed on its own."""
    viewer = make_viewer_model()
    image_layer = create_image_layer_with_phasors()
    viewer.add_layer(image_layer)
    plotter = PlotterWidget(viewer)
    try:
        mask_data = np.zeros(image_layer.data.shape, dtype=int)
        mask_data[: mask_data.shape[0] // 2, :] = 1
        mask_data[mask_data.shape[0] // 2 :, :] = 2
        viewer.add_labels(mask_data, name="two_labels")

        plotter.image_layers_checkable_combobox.setCheckedItems(
            [image_layer.name]
        )
        plotter._process_layer_selection_change()
        # Assign the mask through the UI so the metadata is written by the
        # same code path the user goes through.
        plotter.mask_layer_combobox.setCurrentText("two_labels")
        # Every label ticked is stored as "no label filter" at all.
        assert 'mask_labels' not in image_layer.metadata

        mapping_tab = plotter.phasor_mapping_tab
        mapping_tab.frequency_input.setText("80.0")
        mapping_tab._on_calculate_lifetime_clicked()

        histogram = mapping_tab.histogram_widget
        stats = plotter._statistics_stack.widget(
            plotter._phasor_map_stats_page_idx
        )
        assert histogram.mask_label_split_available()
        assert stats.layer_stats_table.rowCount() == 1

        histogram.split_by_mask_labels = True

        names = list(histogram._datasets)
        assert len(names) == 2
        assert all("label" in name for name in names)
        # Both curves come from the analysed image layer, so grouping and
        # every other per-layer feature still sees one layer.
        assert histogram._group_source_names() == [image_layer.name]
        table = stats.layer_stats_table
        assert table.rowCount() == 2
        assert [table.item(row, 0).text() for row in range(2)] == names

        # Deselecting a label leaves nothing to separate.
        plotter.mask_labels_combobox.setCheckedItems(["1"])
        plotter._on_mask_labels_changed()
        mapping_tab._on_calculate_lifetime_clicked()
        assert not histogram.mask_label_split_available()
        assert not histogram.mask_label_split_active()
        assert len(histogram._datasets) == 1
    finally:
        plotter.close()


def _layer_with_larger_phasors():
    """A phasor layer big enough for median filtering to change values."""
    raw_flim_data = make_raw_flim_data(
        shape=(16, 16), time_constants=[0.1, 1, 2, 3, 4, 5, 10]
    )
    return make_intensity_layer_with_phasors(raw_flim_data, harmonic=[1, 2, 3])


def test_threshold_without_filter_survives_mask(make_viewer_model):
    """Applying a mask must not undo a threshold applied without a filter.

    Regression: the reapply after masking required *both* a filter and a
    lower threshold in the settings, so a threshold-only layer was left with
    the restored, unthresholded data.
    """
    viewer = make_viewer_model()
    layer = _layer_with_larger_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    shape = _make_mask_shape(layer)
    mask_data = np.zeros(shape, dtype=int)
    mask_data[2:12, 2:12] = 1
    viewer.add_labels(mask_data, name="mask")
    plotter.reset_layer_choices()

    filter_tab = plotter.filter_tab
    filter_tab.threshold_method_combobox.setCurrentText("Manual")
    lower, upper = filter_tab.threshold_slider.value()
    filter_tab.threshold_slider.setValue((lower + (upper - lower) // 2, upper))
    filter_tab.apply_button_clicked()
    assert "filter" not in layer.metadata["settings"]
    assert layer.metadata["settings"]["threshold"] is not None

    plotter.mask_layer_combobox.setCurrentText("mask")

    # More pixels are dropped than the mask alone would drop, i.e. the
    # threshold is still in effect on top of the mask.
    assert np.isnan(layer.data).sum() > int((mask_data <= 0).sum())


def test_filter_without_threshold_survives_mask(make_viewer_model):
    """Applying a mask must not undo a filter applied without a threshold."""
    viewer = make_viewer_model()
    layer = _layer_with_larger_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    shape = _make_mask_shape(layer)
    mask_data = np.zeros(shape, dtype=int)
    mask_data[2:12, 2:12] = 1
    viewer.add_labels(mask_data, name="mask")
    plotter.reset_layer_choices()

    filter_tab = plotter.filter_tab
    filter_tab.filter_method_combobox.setCurrentText("Median")
    filter_tab.median_filter_spinbox.setValue(3)
    filter_tab.median_filter_repetition_spinbox.setValue(1)
    filter_tab.threshold_method_combobox.setCurrentText("None")
    filter_tab.apply_button_clicked()
    assert layer.metadata["settings"]["threshold"] is None
    assert not np.allclose(
        layer.metadata["G"], layer.metadata["G_original"], equal_nan=True
    )

    plotter.mask_layer_combobox.setCurrentText("mask")

    inside = mask_data > 0
    assert not np.allclose(
        layer.metadata["G"][..., inside],
        layer.metadata["G_original"][..., inside],
        equal_nan=True,
    )


def test_mask_does_not_move_phasor_coordinates(make_viewer_model):
    """Masking must not change the phasor coordinates of the kept pixels.

    Regression: the mask was applied to the original arrays *before*
    filtering, so every kept pixel was re-filtered against NaN neighbours and
    moved. A phasor-cursor selection is scattered across the image, so nearly
    all of its pixels border NaN and ended up back at their unfiltered
    positions — visibly outside the cursor that selected them.
    """
    rng = np.random.default_rng(0)
    raw = make_raw_flim_data(
        shape=(32, 32), time_constants=[0.1, 0.5, 1, 2, 3, 4, 5, 10]
    )
    # Poisson noise makes the median filter actually move the coordinates.
    raw = rng.poisson(raw * 50).astype(float)
    layer = make_intensity_layer_with_phasors(raw, harmonic=[1, 2])

    viewer = make_viewer_model()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    filter_tab = plotter.filter_tab
    filter_tab.filter_method_combobox.setCurrentText("Median")
    filter_tab.median_filter_spinbox.setValue(3)
    filter_tab.median_filter_repetition_spinbox.setValue(1)
    filter_tab.apply_button_clicked()

    # A circular cursor drawn on the filtered cloud, as the selection tab does.
    g_shown, s_shown = layer.metadata["G"][0], layer.metadata["S"][0]
    g_c, s_c = np.nanmedian(g_shown), np.nanmedian(s_shown)
    distances = np.sqrt((g_shown - g_c) ** 2 + (s_shown - s_c) ** 2)
    # A cursor tight enough that a moved pixel escapes it.
    radius = float(np.nanpercentile(distances, 25))
    inside = distances <= radius
    assert inside.sum() > 10

    viewer.add_labels(inside.astype(int), name="Cursor Selection")
    plotter.reset_layer_choices()
    plotter.mask_layer_combobox.setCurrentText("Cursor Selection")

    g_after, s_after = layer.metadata["G"][0], layer.metadata["S"][0]
    # Exactly the selected pixels survive, at exactly the coordinates the
    # cursor selected them at — so every plotted point is inside the cursor.
    assert np.array_equal(~np.isnan(g_after), inside)
    np.testing.assert_allclose(g_after[inside], g_shown[inside])
    np.testing.assert_allclose(s_after[inside], s_shown[inside])
    distance = np.sqrt(
        (g_after[inside] - g_c) ** 2 + (s_after[inside] - s_c) ** 2
    )
    assert distance.max() <= radius


def _noisy_phasor_layer(name, seed):
    """A phasor layer with noise, so filter parameters visibly matter."""
    rng = np.random.default_rng(seed)
    raw = make_raw_flim_data(
        shape=(32, 32), time_constants=[0.1, 0.5, 1, 2, 3, 4, 5, 10]
    )
    raw = rng.poisson(raw * 50).astype(float)
    return make_intensity_layer_with_phasors(raw, harmonic=[1, 2], name=name)


def test_multi_layer_masks_reapply_each_layers_own_settings(make_viewer_model):
    """Masking several layers must not clobber their individual settings.

    Regression: the re-apply after a mask change went through the Filter tab's
    apply button, which reads the *widgets* — populated from the primary layer
    only — and wrote those values to every selected layer. Layers filtered
    differently (e.g. several OME-TIFFs read back with their own stored
    settings) were re-filtered with the primary layer's parameters, so their
    phasors moved out of the cursor that had selected them.
    """
    viewer = make_viewer_model()
    layer_a = _noisy_phasor_layer("A", 0)
    layer_b = _noisy_phasor_layer("B", 1)
    viewer.add_layer(layer_a)
    viewer.add_layer(layer_b)
    plotter = PlotterWidget(viewer)

    # Each layer arrives with its own filter, as when read back from file.
    apply_filter_and_threshold(
        layer_a,
        threshold=0.0,
        threshold_method="Manual",
        filter_method="median",
        size=3,
        repeat=1,
    )
    apply_filter_and_threshold(
        layer_b,
        threshold=0.0,
        threshold_method="Manual",
        filter_method="median",
        size=7,
        repeat=3,
    )

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer_a.name, layer_b.name]
    )
    plotter._process_layer_selection_change()

    # A circular cursor per layer, drawn on what that layer displays.
    cursors, selections = {}, {}
    for layer in (layer_a, layer_b):
        g, s = layer.metadata["G"][0], layer.metadata["S"][0]
        g_c, s_c = np.nanmedian(g), np.nanmedian(s)
        distance = np.sqrt((g - g_c) ** 2 + (s - s_c) ** 2)
        radius = float(np.nanpercentile(distance, 25))
        cursors[layer.name] = (g_c, s_c, radius)
        selections[layer.name] = distance <= radius
        viewer.add_labels(
            selections[layer.name].astype(int), name=f"sel {layer.name}"
        )
    plotter.reset_layer_choices()

    plotter._apply_mask_assignments(
        {
            layer_a.name: f"sel {layer_a.name}",
            layer_b.name: f"sel {layer_b.name}",
        }
    )

    for layer, size, repeat in ((layer_a, 3, 1), (layer_b, 7, 3)):
        # Each layer kept its own filter parameters ...
        filter_settings = layer.metadata["settings"]["filter"]
        assert (filter_settings["size"], filter_settings["repeat"]) == (
            size,
            repeat,
        )
        # ... so every plotted point is still inside that layer's cursor.
        g_c, s_c, radius = cursors[layer.name]
        inside = selections[layer.name]
        g, s = layer.metadata["G"][0], layer.metadata["S"][0]
        distance = np.sqrt((g[inside] - g_c) ** 2 + (s[inside] - s_c) ** 2)
        assert np.array_equal(~np.isnan(g), inside)
        assert distance.max() <= radius


# ---------------------------------------------------------------------------
# Mask summary button and editor popover
# ---------------------------------------------------------------------------


def _plotter_with_label_mask(make_viewer_model, n_layers=1):
    """Build a plotter with ``n_layers`` phasor layers and a 3-label mask."""
    viewer = make_viewer_model()
    layers = []
    for _ in range(n_layers):
        layer = create_image_layer_with_phasors()
        viewer.add_layer(layer)
        layers.append(layer)
    plotter = PlotterWidget(viewer)
    shape = _make_mask_shape(layers[0])
    data = np.zeros(shape, dtype=int)
    width = shape[-1]
    data[..., : width // 3] = 1
    data[..., width // 3 : 2 * width // 3] = 2
    data[..., 2 * width // 3 :] = 3
    mask = viewer.add_labels(data, name="cells")
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer.name for layer in layers]
    )
    return viewer, plotter, layers, mask


def test_mask_summary_button_and_editor(make_viewer_model):
    """The summary button names the mask, its label subset and inversion,
    fits its text to its width, and opens the editor popover."""
    _, plotter, layers, mask = _plotter_with_label_mask(make_viewer_model)
    button = plotter.mask_summary_button
    popover = plotter.mask_editor_popover

    # With no mask the button reads 'None' and says what it is for.
    assert plotter._mask_summary_full_text == "None"
    assert "restrict the analysis" in button.toolTip()

    # The 'Labels' caption hides and shows with the combobox it names.
    plotter._set_mask_labels_visible(True)
    assert not plotter.mask_labels_label.isHidden()
    assert not plotter.mask_labels_container.isHidden()
    plotter._set_mask_labels_visible(False)
    assert plotter.mask_labels_label.isHidden()
    assert plotter.mask_labels_container.isHidden()

    # One selected layer edits its own mask in the popover.
    with (
        patch.object(plotter, '_open_mask_assignment_dialog') as dialog,
        patch.object(plotter, '_show_mask_editor_popover') as show_popover,
    ):
        button.click()
    dialog.assert_not_called()
    show_popover.assert_called_once()

    # The popover is at least as wide as the button and sits below it.
    # Never actually show it: a Qt.Popup takes a global input grab, which
    # would wedge the machine running the suite if this test died mid-way.
    button.resize(240, 24)
    with patch.object(type(popover), 'show') as show:
        plotter._show_mask_editor_popover()
    show.assert_called_once()
    assert popover.minimumWidth() >= button.width()
    assert popover.pos() == button.mapToGlobal(button.rect().bottomLeft())

    # Selecting a mask puts its name on the button.
    plotter.mask_layer_combobox.setCurrentText(mask.name)
    assert plotter._mask_summary_full_text == "cells"
    assert "Click to edit" in button.toolTip()

    # An unassigned layer keeps the mask already showing in the editor.
    plotter._mask_assignments.pop(layers[0].name, None)
    plotter._update_mask_ui_mode()
    assert plotter.mask_layer_combobox.currentText() == mask.name
    assert plotter._mask_summary_full_text == "cells"

    # Labels are mentioned only when they actually narrow the mask.
    plotter.mask_labels_combobox.selectAll()
    assert plotter._mask_summary_full_text == "cells"
    plotter.mask_labels_combobox.setCheckedItems(["1", "2"])
    assert plotter._mask_summary_full_text == "cells · 2 labels"
    plotter.mask_labels_combobox.setCheckedItems(["1"])
    assert plotter._mask_summary_full_text == "cells · 1 label"

    # Invert shows up in the summary, alongside any label narrowing.
    plotter.mask_labels_combobox.selectAll()
    plotter.mask_invert_checkbox.setChecked(True)
    assert plotter._mask_summary_full_text == "cells · inverted"
    plotter.mask_labels_combobox.setCheckedItems(["3"])
    assert plotter._mask_summary_full_text == "cells · 1 label · inverted"

    # Re-syncing the editor restores the labels stored for that layer.
    plotter._mask_label_assignments[layers[0].name] = [2, 3]
    plotter._update_mask_ui_mode()
    assert plotter.mask_labels_combobox.checkedItems() == ["2", "3"]

    # A summary too wide for the button is elided, never clipped.
    full_text = "a very long mask layer name indeed"
    plotter._mask_summary_full_text = full_text
    button.resize(60, 24)
    plotter._elide_mask_summary()
    assert button.text() != full_text
    assert "…" in button.text()
    button.resize(400, 24)
    plotter._elide_mask_summary()
    assert button.text() == full_text

    # Resizing or showing the button re-fits the summary to it. Show
    # matters because the first summary is set while the row is still
    # unlaid-out and the button has no meaningful width.
    for event_type in (QEvent.Resize, QEvent.Show):
        button.resize(400, 24)
        plotter._elide_mask_summary()
        button.resize(60, 24)
        plotter.eventFilter(button, QEvent(event_type))
        assert "…" in button.text()

    # A button with no width yet falls back to the untruncated text.
    plotter._mask_summary_full_text = "cells"
    button.resize(0, 24)
    plotter._elide_mask_summary()
    assert button.text() == "cells"
