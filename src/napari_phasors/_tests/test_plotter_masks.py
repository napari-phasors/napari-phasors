from unittest.mock import MagicMock, patch

import numpy as np
from napari.layers import Image
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
)

from napari_phasors._tests.test_plotter import (  # noqa: E501
    create_image_layer_with_phasors,
)
from napari_phasors.plotter import (
    MaskAssignmentDialog,
    PlotterWidget,
    _apply_label_colors_to_combo,
)


def test_phasor_plotter_mask_layer_ui_initialization(make_viewer_model):
    """Test mask layer UI components exist and are initialized correctly."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Test mask layer combobox and label exist
    assert hasattr(plotter, 'mask_layer_combobox')
    assert hasattr(plotter, 'mask_layer_label')
    assert isinstance(plotter.mask_layer_combobox, QComboBox)
    assert isinstance(plotter.mask_layer_label, QLabel)

    # Test initial state - should have "None" as default
    assert plotter.mask_layer_combobox.currentText() == "None"


def test_phasor_plotter_apply_mask_to_phasor_data(make_viewer_model):
    """Test that applying a mask sets G and S values outside mask to NaN."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Add image layer with phasors
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    # Get original G and S shape
    G_original = intensity_image_layer.metadata["G"]

    # Create a mask - need to match the spatial dimensions of G and S
    if G_original.ndim == 3:
        # Multi-harmonic: shape is (n_harmonics, height, width)
        mask_shape = G_original.shape[1:]
    else:
        # Single harmonic: shape is (height, width)
        mask_shape = G_original.shape

    mask_data = np.zeros(mask_shape, dtype=int)
    mask_data[mask_shape[0] // 2 :, :] = 1  # Mask in bottom half
    labels_layer = viewer.add_labels(mask_data, name="test_mask")

    # Apply mask
    plotter._apply_mask_to_phasor_data(labels_layer, intensity_image_layer)

    # Check that mask was stored in metadata
    assert 'mask' in intensity_image_layer.metadata

    # Check that G and S values outside mask are now NaN
    current_g = intensity_image_layer.metadata["G"]
    current_s = intensity_image_layer.metadata["S"]
    assert np.isnan(current_g).sum() > 0
    assert np.isnan(current_s).sum() > 0


def test_phasor_plotter_restore_original_phasor_data(make_viewer_model):
    """Test restoring original phasor data removes mask effects."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Add image layer with phasors
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)

    # Get original data
    original_g = intensity_image_layer.metadata["G"].copy()
    original_s = intensity_image_layer.metadata["S"].copy()

    # Get the shape for the mask
    if original_g.ndim == 3:
        mask_shape = original_g.shape[1:]
    else:
        mask_shape = original_g.shape

    # Apply mask to modify data
    mask_data = np.zeros(mask_shape, dtype=int)
    mask_data[mask_shape[0] // 2 :, :] = 1
    labels_layer = viewer.add_labels(mask_data, name="test_mask")
    plotter._apply_mask_to_phasor_data(labels_layer, intensity_image_layer)

    # Verify data was modified (some values are NaN)
    assert np.isnan(intensity_image_layer.metadata["G"]).sum() > 0

    # Restore original data
    plotter._restore_original_phasor_data(intensity_image_layer)

    # Verify data was restored (no NaN values in original)
    np.testing.assert_array_almost_equal(
        intensity_image_layer.metadata["G"], original_g
    )
    np.testing.assert_array_almost_equal(
        intensity_image_layer.metadata["S"], original_s
    )


def test_mask_layer_rename_updates_combobox_and_assignments(
    make_viewer_model,
):
    """Regression test for mask-layer rename synchronization in the plotter."""
    viewer = make_viewer_model()

    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    # Build a mask compatible with phasor spatial dimensions.
    g_data = layer1.metadata["G"]
    mask_shape = g_data.shape[1:] if g_data.ndim == 3 else g_data.shape
    mask = np.ones(mask_shape, dtype=int)
    labels_layer = viewer.add_labels(mask, name="mask_before_rename")

    plotter = PlotterWidget(viewer)

    # Single-layer mode: selecting a mask updates the combobox and assignments.
    plotter.image_layers_checkable_combobox.setCheckedItems([layer1.name])
    plotter.mask_layer_combobox.setCurrentText("mask_before_rename")
    assert plotter.mask_layer_combobox.currentText() == "mask_before_rename"
    assert plotter._mask_assignments.get(layer1.name) == "mask_before_rename"

    # Renaming the mask layer should propagate to UI and assignments.
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


def test_image_layer_rename_updates_combobox(make_viewer_model):
    """Test that renaming an image layer keeps it selected and updates the name in the checkable combobox."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)

    plotter = PlotterWidget(viewer)

    # Select the layer
    plotter.image_layers_checkable_combobox.setCheckedItems([layer1.name])
    assert plotter.get_selected_layer_names() == [layer1.name]

    # Rename the layer
    old_name = layer1.name
    new_name = "renamed_image_layer"
    layer1.name = new_name

    assert plotter.get_selected_layer_names() == [new_name]
    assert new_name in plotter.image_layers_checkable_combobox.allItems()
    assert old_name not in plotter.image_layers_checkable_combobox.allItems()


def test_image_layer_rename_without_initial_phasors(make_viewer_model):
    """Test that renaming an image layer that starts without phasor metadata correctly tracks rename when it later gets phasors."""
    viewer = make_viewer_model()
    # Create image layer without phasors
    layer1 = Image(np.ones((10, 10)), name="raw_image")
    viewer.add_layer(layer1)

    plotter = PlotterWidget(viewer)

    # It should not be in the combobox yet
    assert (
        "raw_image" not in plotter.image_layers_checkable_combobox.allItems()
    )

    # Rename the layer before it has phasor metadata
    layer1.name = "renamed_raw_image"

    # Set phasor metadata to simulate calibration/calculation
    layer1.metadata = {
        "G": np.ones((10, 10)),
        "S": np.ones((10, 10)),
        "G_original": np.ones((10, 10)),
        "S_original": np.ones((10, 10)),
        "harmonics": [1],
    }

    # Trigger reset_layer_choices (e.g. simulation of tab change or manual refresh)
    plotter.reset_layer_choices()

    # It should now be in the combobox with the renamed name
    assert (
        "renamed_raw_image"
        in plotter.image_layers_checkable_combobox.allItems()
    )

    # If we select it
    plotter.image_layers_checkable_combobox.setCheckedItems(
        ["renamed_raw_image"]
    )

    # Rename it again (while it has phasor metadata and is selected)
    layer1.name = "final_image_name"

    # It should keep being selected and update the name in the combobox
    assert plotter.get_selected_layer_names() == ["final_image_name"]
    assert (
        "final_image_name"
        in plotter.image_layers_checkable_combobox.allItems()
    )
    assert (
        "renamed_raw_image"
        not in plotter.image_layers_checkable_combobox.allItems()
    )


def test_mask_ui_switches_to_button_when_multiple_layers_selected(
    make_viewer_model,
):
    """Test that selecting multiple layers switches mask UI from combobox to button."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    # With only one layer selected (default primary), combobox should be visible.
    # Use not isHidden() because the plotter is not rendered in a window during tests,
    # so isVisible() would be False for all widgets regardless of their setVisible() state.
    plotter.image_layers_checkable_combobox.setCheckedItems([layer1.name])
    assert not plotter.mask_layer_combobox.isHidden()
    assert not plotter.mask_layer_label.isHidden()
    assert plotter.mask_assign_button.isHidden()

    # Select both layers - should switch to button mode
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )
    assert plotter.mask_layer_combobox.isHidden()
    assert plotter.mask_layer_label.isHidden()
    assert not plotter.mask_assign_button.isHidden()

    # Back to single selection - should revert to combobox mode
    plotter.image_layers_checkable_combobox.setCheckedItems([layer1.name])
    assert not plotter.mask_layer_combobox.isHidden()
    assert plotter.mask_assign_button.isHidden()


def test_mask_assignment_dialog_get_assignments(make_viewer_model):
    """Test that MaskAssignmentDialog returns correct per-layer assignments."""
    from napari_phasors.plotter import MaskAssignmentDialog

    image_names = ["layer_A", "layer_B", "layer_C"]
    mask_names = ["mask_1", "mask_2"]
    current = {"layer_A": "mask_1", "layer_C": "mask_2"}

    dialog = MaskAssignmentDialog(
        image_layer_names=image_names,
        mask_layer_names=mask_names,
        current_assignments=current,
        parent=None,
    )

    # Initial assignments should reflect current_assignments
    assignments = dialog.get_assignments()
    assert assignments["layer_A"] == "mask_1"
    assert assignments["layer_B"] == "None"
    assert assignments["layer_C"] == "mask_2"


def test_mask_assignment_dialog_apply_all(make_viewer_model):
    """Test that 'Set all to' applies the same mask to every layer in the dialog."""
    from napari_phasors.plotter import MaskAssignmentDialog

    image_names = ["layer_A", "layer_B"]
    mask_names = ["mask_1", "mask_2"]

    dialog = MaskAssignmentDialog(
        image_layer_names=image_names,
        mask_layer_names=mask_names,
        parent=None,
    )

    # Trigger apply-all by changing the combo
    dialog._apply_all_combo.setCurrentText("mask_2")

    assignments = dialog.get_assignments()
    assert assignments["layer_A"] == "mask_2"
    assert assignments["layer_B"] == "mask_2"


def test_apply_mask_assignments_different_masks_per_layer(make_viewer_model):
    """Test that _apply_mask_assignments applies distinct masks to each layer."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    # Select both layers
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )

    # Build matching mask shapes from each layer's G data
    def make_mask(layer, fill_row_half):
        G = layer.metadata["G"]
        shape = G.shape[1:] if G.ndim == 3 else G.shape
        mask = np.zeros(shape, dtype=int)
        if fill_row_half:
            mask[shape[0] // 2 :, :] = 1  # bottom half
        else:
            mask[: shape[0] // 2, :] = 1  # top half
        return mask

    mask_data_1 = make_mask(layer1, fill_row_half=True)
    mask_data_2 = make_mask(layer2, fill_row_half=False)

    labels_layer1 = viewer.add_labels(mask_data_1, name="mask_bottom")
    labels_layer2 = viewer.add_labels(mask_data_2, name="mask_top")

    # Apply different masks to each layer
    assignments = {
        layer1.name: labels_layer1.name,
        layer2.name: labels_layer2.name,
    }
    plotter._apply_mask_assignments(assignments)

    # Both layers should now have masks stored
    assert 'mask' in layer1.metadata
    assert 'mask' in layer2.metadata

    # layer1 mask covers bottom half, so top half pixels should be NaN
    g1 = layer1.metadata["G"]
    if g1.ndim == 3:
        g1 = g1[0]
    assert np.isnan(g1[: g1.shape[0] // 2, :]).all()  # top half is NaN
    assert not np.isnan(g1[g1.shape[0] // 2 :, :]).all()  # bottom half kept

    # layer2 mask covers top half, so bottom half pixels should be NaN
    g2 = layer2.metadata["G"]
    if g2.ndim == 3:
        g2 = g2[0]
    assert np.isnan(g2[g2.shape[0] // 2 :, :]).all()  # bottom half is NaN
    assert not np.isnan(g2[: g2.shape[0] // 2, :]).all()  # top half kept

    # _mask_assignments should only contain non-None entries
    assert plotter._mask_assignments[layer1.name] == labels_layer1.name
    assert plotter._mask_assignments[layer2.name] == labels_layer2.name


def test_apply_mask_assignments_none_removes_mask(make_viewer_model):
    """Test that assigning 'None' removes an existing mask from a layer."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter.image_layers_checkable_combobox.setCheckedItems([layer.name])

    G = layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.ones(shape, dtype=int)
    labels_layer = viewer.add_labels(mask_data, name="full_mask")

    # First apply a mask
    plotter._apply_mask_to_phasor_data(labels_layer, layer)
    assert 'mask' in layer.metadata

    # Now assign None to remove it
    plotter._apply_mask_assignments({layer.name: "None"})

    assert 'mask' not in layer.metadata


def test_get_mask_for_layer_multi_mode(make_viewer_model):
    """Test get_mask_for_layer returns per-layer assignment when multiple layers selected."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    G = layer1.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.ones(shape, dtype=int)
    labels_layer = viewer.add_labels(mask_data, name="my_mask")

    # Select both layers so we're in multi mode
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )

    # Assign mask only to layer1
    plotter._mask_assignments = {layer1.name: labels_layer.name}

    assert plotter.get_mask_for_layer(layer1.name) == labels_layer.name
    assert plotter.get_mask_for_layer(layer2.name) == "None"


def test_mask_assign_button_text_updates_with_count(make_viewer_model):
    """Test that the assign button text shows how many layers have masks assigned."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )

    # No masks assigned yet
    plotter._mask_assignments = {}
    plotter._update_mask_assign_button_text()
    assert plotter.mask_assign_button.text() == "Assign Masks..."

    # One mask assigned
    G = layer1.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    labels_layer = viewer.add_labels(np.ones(shape, dtype=int), name="m")
    plotter._mask_assignments = {layer1.name: labels_layer.name}
    plotter._update_mask_assign_button_text()
    assert "1/2" in plotter.mask_assign_button.text()

    # Both masks assigned
    plotter._mask_assignments = {
        layer1.name: labels_layer.name,
        layer2.name: labels_layer.name,
    }
    plotter._update_mask_assign_button_text()
    assert "2/2" in plotter.mask_assign_button.text()


def _make_mask_shape(layer):
    """Get the spatial shape for creating a mask from a layer."""
    G = layer.metadata["G"]
    return G.shape[1:] if G.ndim == 3 else G.shape


# --- Feature 1: Invert Mask (single-layer mode) ---


def test_invert_mask_checkbox_exists_and_default(make_viewer_model):
    """Test that the invert mask checkbox exists and is unchecked."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    assert hasattr(plotter, 'mask_invert_checkbox')
    assert isinstance(plotter.mask_invert_checkbox, QCheckBox)
    assert not plotter.mask_invert_checkbox.isChecked()


def test_invert_mask_checkbox_disabled_when_no_mask(
    make_viewer_model,
):
    """Test that invert checkbox is disabled when mask is 'None'."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    assert not plotter.mask_invert_checkbox.isEnabled()


def test_invert_mask_checkbox_enabled_when_mask_selected(
    make_viewer_model,
):
    """Test invert checkbox enables when a mask layer is selected."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    shape = _make_mask_shape(layer)
    mask_data = np.ones(shape, dtype=int)
    viewer.add_labels(mask_data, name="test_mask")

    plotter.mask_layer_combobox.setCurrentText("test_mask")

    assert plotter.mask_invert_checkbox.isEnabled()


def test_invert_mask_applies_inverted_logic(make_viewer_model):
    """Test that invert param applies mask with inverted logic."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    shape = _make_mask_shape(layer)
    # Mask bottom half (label=1), top half (label=0)
    mask_data = np.zeros(shape, dtype=int)
    mask_data[shape[0] // 2 :, :] = 1
    labels_layer = viewer.add_labels(mask_data, name="test_mask")

    # Normal mask: top half -> NaN, bottom half -> kept
    plotter._apply_mask_to_phasor_data(labels_layer, layer, invert=False)
    g = layer.metadata["G"]
    g_2d = g[0] if g.ndim == 3 else g
    assert np.isnan(g_2d[: shape[0] // 2, :]).all()
    assert not np.isnan(g_2d[shape[0] // 2 :, :]).all()

    # Restore and apply inverted mask: bottom half -> NaN,
    # top half -> kept
    plotter._restore_original_phasor_data(layer)
    plotter._apply_mask_to_phasor_data(labels_layer, layer, invert=True)
    g_inv = layer.metadata["G"]
    g_inv_2d = g_inv[0] if g_inv.ndim == 3 else g_inv
    assert not np.isnan(g_inv_2d[: shape[0] // 2, :]).all()
    assert np.isnan(g_inv_2d[shape[0] // 2 :, :]).all()


def test_invert_mask_resets_on_none(make_viewer_model):
    """Test that invert checkbox resets when mask is set to None."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    shape = _make_mask_shape(layer)
    mask_data = np.ones(shape, dtype=int)
    viewer.add_labels(mask_data, name="test_mask")

    # Select mask, enable invert
    plotter.mask_layer_combobox.setCurrentText("test_mask")
    plotter.mask_invert_checkbox.setChecked(True)
    assert plotter.mask_invert_checkbox.isChecked()

    # Set mask to None - invert should reset
    plotter.mask_layer_combobox.setCurrentText("None")
    assert not plotter.mask_invert_checkbox.isChecked()
    assert not plotter.mask_invert_checkbox.isEnabled()


def test_invert_mask_empty_mask_keeps_all_pixels(
    make_viewer_model,
):
    """Test invert with empty mask (all zeros): all pixels kept."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    original_g = layer.metadata["G"].copy()
    shape = _make_mask_shape(layer)
    mask_data = np.zeros(shape, dtype=int)
    labels_layer = viewer.add_labels(mask_data, name="empty_mask")

    # Inverted empty mask should keep all pixels
    plotter._apply_mask_to_phasor_data(labels_layer, layer, invert=True)
    np.testing.assert_array_equal(
        np.isnan(layer.metadata["G"]),
        np.isnan(original_g),
    )


# --- Feature 1: Invert Mask (multi-layer mode) ---


def test_mask_assignment_dialog_has_invert_option(
    make_viewer_model,
):
    """Test MaskAssignmentDialog includes invert option."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    shape = _make_mask_shape(layer1)
    viewer.add_labels(np.ones(shape, dtype=int), name="mask1")

    dialog = MaskAssignmentDialog(
        image_layer_names=[layer1.name, layer2.name],
        mask_layer_names=["mask1"],
        current_assignments={},
    )

    # Dialog should support invert assignments
    invert_assignments = dialog.get_invert_assignments()
    assert isinstance(invert_assignments, dict)
    for name in [layer1.name, layer2.name]:
        assert name in invert_assignments
        assert invert_assignments[name] is False


def test_apply_mask_assignments_with_invert(make_viewer_model):
    """Test per-layer invert via _apply_mask_assignments."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )

    shape = _make_mask_shape(layer1)
    mask_data = np.zeros(shape, dtype=int)
    mask_data[shape[0] // 2 :, :] = 1
    viewer.add_labels(mask_data, name="half_mask")

    # Apply same mask, invert only layer2
    assignments = {
        layer1.name: "half_mask",
        layer2.name: "half_mask",
    }
    invert_assignments = {
        layer1.name: False,
        layer2.name: True,
    }
    plotter._apply_mask_assignments(
        assignments, invert_assignments=invert_assignments
    )

    # layer1: normal mask — top half NaN
    g1 = layer1.metadata["G"]
    g1_2d = g1[0] if g1.ndim == 3 else g1
    assert np.isnan(g1_2d[: shape[0] // 2, :]).all()

    # layer2: inverted mask — bottom half NaN
    g2 = layer2.metadata["G"]
    g2_2d = g2[0] if g2.ndim == 3 else g2
    assert np.isnan(g2_2d[shape[0] // 2 :, :]).all()


def test_mask_assignment_dialog_fallback_to_none(
    make_viewer_model,
):
    """Test dialog sets 'None' when current assignment is invalid."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    shape = _make_mask_shape(layer)
    viewer.add_labels(np.ones(shape, dtype=int), name="mask1")

    # Pass an assignment that references a non-existent mask
    dialog = MaskAssignmentDialog(
        image_layer_names=[layer.name],
        mask_layer_names=["mask1"],
        current_assignments={layer.name: "deleted_mask"},
    )

    assignments = dialog.get_assignments()
    assert assignments[layer.name] == "None"


def test_apply_mask_shapes_layer(make_viewer_model):
    """Test _apply_mask_to_phasor_data with a Shapes layer."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    shape = _make_mask_shape(layer)
    # Create a shapes layer with a rectangle covering the image
    rect = np.array(
        [[0, 0], [0, shape[1]], [shape[0], shape[1]], [shape[0], 0]]
    )
    shapes_layer = viewer.add_shapes(
        [rect], shape_type="polygon", name="shape_mask"
    )

    plotter._apply_mask_to_phasor_data(shapes_layer, layer)

    assert "mask" in layer.metadata


def test_apply_mask_non_invert_empty_labels_returns(
    make_viewer_model,
):
    """Test non-invert empty Labels mask early returns."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)
    plotter = PlotterWidget(viewer)

    original_g = layer.metadata["G"].copy()
    shape = _make_mask_shape(layer)
    empty_labels = viewer.add_labels(np.zeros(shape, dtype=int), name="empty")

    # Non-invert + empty mask should early return (no changes)
    plotter._apply_mask_to_phasor_data(empty_labels, layer, invert=False)
    np.testing.assert_array_equal(layer.metadata["G"], original_g)
    assert "mask" not in layer.metadata


def test_on_mask_data_changed_no_selected_layers(
    make_viewer_model,
):
    """Test _on_mask_data_changed returns early with no layers."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    shape = (5, 5)
    labels_layer = viewer.add_labels(np.ones(shape, dtype=int), name="mask")

    # No image layers selected — should return early
    event = type("Event", (), {"source": labels_layer})()
    plotter._on_mask_data_changed(event)


def test_on_mask_data_changed_multi_layer_branch(
    make_viewer_model,
):
    """Test _on_mask_data_changed multi-layer path."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )

    shape = _make_mask_shape(layer1)
    mask_data = np.ones(shape, dtype=int)
    labels_layer = viewer.add_labels(mask_data, name="shared_mask")

    # Assign mask only to layer1
    plotter._mask_assignments = {layer1.name: "shared_mask"}

    # Trigger mask data change — should only affect layer1
    event = type("Event", (), {"source": labels_layer})()
    plotter._on_mask_data_changed(event)

    assert "mask" in layer1.metadata


def test_on_mask_data_changed_multi_layer_no_affected(
    make_viewer_model,
):
    """Test _on_mask_data_changed multi-layer with no affected."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )

    shape = _make_mask_shape(layer1)
    labels_layer = viewer.add_labels(
        np.ones(shape, dtype=int), name="unassigned_mask"
    )

    # No layer has this mask assigned
    plotter._mask_assignments = {}

    event = type("Event", (), {"source": labels_layer})()
    plotter._on_mask_data_changed(event)

    # Should return early — no masks applied
    assert "mask" not in layer1.metadata
    assert "mask" not in layer2.metadata


def test_open_mask_assignment_dialog_applies_invert(
    make_viewer_model,
):
    """Test _open_mask_assignment_dialog passes invert state."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    plotter = PlotterWidget(viewer)

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [layer1.name, layer2.name]
    )

    shape = _make_mask_shape(layer1)
    viewer.add_labels(np.ones(shape, dtype=int), name="mask1")

    # Mock dialog to simulate user accepting with invert
    mock_dialog = MagicMock()
    mock_dialog.exec.return_value = 1  # QDialog.Accepted
    mock_dialog.get_assignments.return_value = {
        layer1.name: "mask1",
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

    # layer1 should have inverted mask applied
    assert plotter._mask_invert_assignments.get(layer1.name, False)


def test_mask_labels_combobox_interaction(make_viewer_model):
    """Test selecting labels in mask_labels_combobox correctly applies and preserves labels."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Add image layer with phasors
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [intensity_image_layer.name]
    )

    # Create a labels layer with multiple label IDs (e.g. 1, 2, 3)
    G_original = intensity_image_layer.metadata["G"]
    shape = G_original.shape[1:] if G_original.ndim == 3 else G_original.shape
    labels_data = np.zeros(shape, dtype=int)
    # Divide image into 3 vertical bands of labels
    h, w = shape
    labels_data[:, : w // 3] = 1
    labels_data[:, w // 3 : 2 * (w // 3)] = 2
    labels_data[:, 2 * (w // 3) :] = 3

    viewer.add_labels(labels_data, name="test_labels_mask")

    # Select the mask layer in the plotter combobox
    plotter.mask_layer_combobox.setCurrentText("test_labels_mask")

    # Verify that the mask labels combobox is visible and populated with labels '1', '2', '3'
    assert not plotter.mask_labels_combobox.isHidden()
    assert plotter.mask_labels_combobox.allItems() == ["1", "2", "3"]

    # Initially no labels are checked in metadata assignment (defaults to all labels > 0, which is represented by None)
    assert (
        plotter._mask_label_assignments.get(intensity_image_layer.name) is None
    )

    # Check label '2'
    plotter.mask_labels_combobox.setCheckedItems(["2"])

    # Verify that the label assignment was saved
    assert plotter._mask_label_assignments[intensity_image_layer.name] == [2]

    # Verify that the checked items in the combobox were preserved (i.e. '2' is still checked)
    assert plotter.mask_labels_combobox.checkedItems() == ["2"]

    # Verify that G/S data is masked to only label '2'
    # Pixels where label is 1 or 3 should be NaN, pixels where label is 2 should be preserved
    g_data = intensity_image_layer.metadata["G"]
    if g_data.ndim == 3:
        g_data = g_data[0]

    np.testing.assert_array_equal(np.isnan(g_data), (labels_data != 2))

    # Toggling the invert checkbox should invert the selection and preserve the label selection '2'
    plotter.mask_invert_checkbox.setChecked(True)
    assert plotter._mask_invert_assignments[intensity_image_layer.name] is True
    assert plotter._mask_label_assignments[intensity_image_layer.name] == [2]
    assert plotter.mask_labels_combobox.checkedItems() == ["2"]

    # Inverted mask with label '2' selected means pixels where label is 2 should be NaN, others (1, 3) preserved
    g_data_inverted = intensity_image_layer.metadata["G"]
    if g_data_inverted.ndim == 3:
        g_data_inverted = g_data_inverted[0]
    np.testing.assert_array_equal(
        np.isnan(g_data_inverted), (labels_data == 2)
    )


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


def test_mask_labels_combobox_defaults_to_all_checked(make_viewer_model):
    """mask_labels_combobox should have all labels checked by default."""
    _, plotter, _, _, _ = _setup_plotter_with_labels(make_viewer_model)

    assert not plotter.mask_labels_combobox.isHidden()
    all_items = plotter.mask_labels_combobox.allItems()
    checked = plotter.mask_labels_combobox.checkedItems()
    assert all_items == ["1", "2", "3"]
    assert checked == all_items, "All label items should be checked by default"


def test_mask_labels_combobox_has_external_all_none_labels(make_viewer_model):
    """mask_labels_container should contain 'All' and 'None' clickable labels."""
    _, plotter, _, _, _ = _setup_plotter_with_labels(make_viewer_model)

    assert (
        plotter.mask_labels_select_all.text()
        == '<a href="all" style="color: gray;">All</a>'
    )
    assert (
        plotter.mask_labels_select_none.text()
        == '<a href="none" style="color: gray;">None</a>'
    )


# ---------------------------------------------------------------------------
# 2. Display text for all selection states
# ---------------------------------------------------------------------------


def test_mask_labels_display_all_labels_when_all_checked(make_viewer_model):
    """When all labels are checked, combobox should show 'All Labels' placeholder."""
    _, plotter, _, _, _ = _setup_plotter_with_labels(make_viewer_model)

    combo = plotter.mask_labels_combobox
    combo.selectAll()
    assert combo.lineEdit().text() == ""
    assert combo.lineEdit().placeholderText() == "All Labels"


def test_mask_labels_display_no_labels_when_none_checked(make_viewer_model):
    """When no labels are checked, combobox should show 'No labels'."""
    _, plotter, _, _, _ = _setup_plotter_with_labels(make_viewer_model)

    combo = plotter.mask_labels_combobox
    combo.deselectAll()
    assert combo.lineEdit().text() == "No labels"


def test_mask_labels_display_count_for_partial_selection(make_viewer_model):
    """When some (but not all) labels are checked, combobox shows 'N labels selected'."""
    _, plotter, _, _, _ = _setup_plotter_with_labels(make_viewer_model)

    combo = plotter.mask_labels_combobox
    combo.setCheckedItems(["1", "3"])
    assert combo.lineEdit().text() == "2 labels selected"


# ---------------------------------------------------------------------------
# 3. Normalization: all-checked is canonically equal to empty list
# ---------------------------------------------------------------------------


def test_mask_label_assignment_is_empty_when_all_checked(make_viewer_model):
    """When all labels are checked the assignment stored should be None (all-labels)."""
    _, plotter, image_layer, _, _ = _setup_plotter_with_labels(
        make_viewer_model
    )
    # Ensure all are checked (default)
    plotter.mask_labels_combobox.selectAll()
    # The on_mask_labels_changed handler normalises all-checked → None
    assignment = plotter._mask_label_assignments.get(
        image_layer.name, "MISSING"
    )
    assert assignment is None, "All-labels selection should be stored as None"


def test_mask_label_assignment_is_specific_when_partial(make_viewer_model):
    """A partial label selection should be stored as the exact list of selected labels."""
    _, plotter, image_layer, _, _ = _setup_plotter_with_labels(
        make_viewer_model
    )
    plotter.mask_labels_combobox.setCheckedItems(["2"])
    assignment = plotter._mask_label_assignments.get(image_layer.name, None)
    assert assignment == [2]


# ---------------------------------------------------------------------------
# 4. _extract_phasor_arrays_from_layer: mask_labels and mask_invert
# ---------------------------------------------------------------------------


def test_extract_phasor_arrays_no_mask(make_viewer_model):
    """Without a mask in metadata, arrays are returned unchanged."""
    from napari_phasors._utils import _extract_phasor_arrays_from_layer

    image_layer = create_image_layer_with_phasors()
    make_viewer_model().add_layer(image_layer)

    mean, real, imag, harmonics = _extract_phasor_arrays_from_layer(
        image_layer
    )
    np.testing.assert_array_equal(mean, image_layer.metadata["original_mean"])
    assert not np.any(np.isnan(mean))


def test_extract_phasor_arrays_with_mask_all_labels(make_viewer_model):
    """mask_labels=None treats all non-zero pixels as valid."""
    from napari_phasors._utils import _extract_phasor_arrays_from_layer

    image_layer = create_image_layer_with_phasors()
    viewer = make_viewer_model()
    viewer.add_layer(image_layer)

    G = image_layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1  # left half label 1
    mask_data[:, shape[1] // 2 :] = 2  # right half label 2

    image_layer.metadata["mask"] = mask_data
    image_layer.metadata["mask_labels"] = None  # all labels valid
    image_layer.metadata["mask_invert"] = False

    _, real, _, _ = _extract_phasor_arrays_from_layer(image_layer)
    # Background (0) pixels → NaN; label pixels → valid
    expected_nan = mask_data <= 0
    np.testing.assert_array_equal(np.isnan(real[0]), expected_nan)


def test_extract_phasor_arrays_with_mask_no_labels(make_viewer_model):
    """mask_labels=[] treats all pixels as valid (no masking applied)."""
    from napari_phasors._utils import _extract_phasor_arrays_from_layer

    image_layer = create_image_layer_with_phasors()
    viewer = make_viewer_model()
    viewer.add_layer(image_layer)

    G = image_layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1
    mask_data[:, shape[1] // 2 :] = 2

    image_layer.metadata["mask"] = mask_data
    image_layer.metadata["mask_labels"] = []  # no labels selected -> no mask
    image_layer.metadata["mask_invert"] = False

    _, real, _, _ = _extract_phasor_arrays_from_layer(image_layer)
    # No pixels should be NaN (unmasked)
    assert not np.any(np.isnan(real[0]))


def test_extract_phasor_arrays_with_specific_mask_labels(make_viewer_model):
    """mask_labels=[1] makes only pixels with label 1 valid; label 2 → NaN."""
    from napari_phasors._utils import _extract_phasor_arrays_from_layer

    image_layer = create_image_layer_with_phasors()
    viewer = make_viewer_model()
    viewer.add_layer(image_layer)

    G = image_layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1
    mask_data[:, shape[1] // 2 :] = 2

    image_layer.metadata["mask"] = mask_data
    image_layer.metadata["mask_labels"] = [1]
    image_layer.metadata["mask_invert"] = False

    _, real, _, _ = _extract_phasor_arrays_from_layer(image_layer)
    # Only label-1 pixels are valid; label-2 and background → NaN
    expected_nan = mask_data != 1
    np.testing.assert_array_equal(np.isnan(real[0]), expected_nan)


def test_extract_phasor_arrays_with_inverted_mask_labels(make_viewer_model):
    """mask_invert=True with mask_labels=[1] makes label-1 pixels NaN."""
    from napari_phasors._utils import _extract_phasor_arrays_from_layer

    image_layer = create_image_layer_with_phasors()
    viewer = make_viewer_model()
    viewer.add_layer(image_layer)

    G = image_layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1
    mask_data[:, shape[1] // 2 :] = 2

    image_layer.metadata["mask"] = mask_data
    image_layer.metadata["mask_labels"] = [1]
    image_layer.metadata["mask_invert"] = True

    _, real, _, _ = _extract_phasor_arrays_from_layer(image_layer)
    # Inverted: label-1 pixels → NaN; label-2 (and background) → valid
    expected_nan = np.isin(mask_data, [1])
    np.testing.assert_array_equal(np.isnan(real[0]), expected_nan)


def test_extract_phasor_arrays_inverted_all_labels(make_viewer_model):
    """mask_invert=True with mask_labels=None inverts the all-non-zero rule."""
    from napari_phasors._utils import _extract_phasor_arrays_from_layer

    image_layer = create_image_layer_with_phasors()
    viewer = make_viewer_model()
    viewer.add_layer(image_layer)

    G = image_layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1

    image_layer.metadata["mask"] = mask_data
    image_layer.metadata["mask_labels"] = None
    image_layer.metadata["mask_invert"] = True

    _, real, _, _ = _extract_phasor_arrays_from_layer(image_layer)
    # Inverted all-labels: non-zero pixels → NaN; background (0) → valid
    expected_nan = mask_data > 0
    np.testing.assert_array_equal(np.isnan(real[0]), expected_nan)


def test_extract_phasor_arrays_inverted_no_labels(make_viewer_model):
    """mask_invert=True with mask_labels=[] leaves all pixels unmasked."""
    from napari_phasors._utils import _extract_phasor_arrays_from_layer

    image_layer = create_image_layer_with_phasors()
    viewer = make_viewer_model()
    viewer.add_layer(image_layer)

    G = image_layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1

    image_layer.metadata["mask"] = mask_data
    image_layer.metadata["mask_labels"] = []
    image_layer.metadata["mask_invert"] = True

    _, real, _, _ = _extract_phasor_arrays_from_layer(image_layer)
    # No pixels should be NaN (unmasked)
    assert not np.any(np.isnan(real[0]))


# ---------------------------------------------------------------------------
# 5. MaskAssignmentDialog: defaults and get_label_assignments normalisation
# ---------------------------------------------------------------------------


def test_mask_assignment_dialog_defaults_all_checked(make_viewer_model):
    """MaskAssignmentDialog label comboboxes default to all labels checked."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    layer2 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)

    G = layer1.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1
    mask_data[:, shape[1] // 2 :] = 2
    labels_layer = viewer.add_labels(mask_data, name="labels_for_dialog")

    dialog = MaskAssignmentDialog(
        image_layer_names=[layer1.name, layer2.name],
        mask_layer_names=["None", "labels_for_dialog"],
        mask_layers=[labels_layer],
        current_assignments={layer1.name: "None", layer2.name: "None"},
        current_label_assignments={},
        current_invert_assignments={},
        parent=None,
    )

    # Trigger the label combobox for layer1 by selecting the labels layer
    dialog._combos[layer1.name].setCurrentText("labels_for_dialog")

    label_combo = dialog._label_combos.get(layer1.name)
    assert label_combo is not None
    assert not label_combo.isHidden(), "Label combo should be visible"
    # All label items should be checked by default
    all_items = label_combo.allItems()
    checked = label_combo.checkedItems()
    assert all_items == ["1", "2"], "Expected labels 1 and 2"
    assert checked == all_items, "All labels should be checked by default"

    dialog.close()


def test_mask_assignment_dialog_get_label_assignments_normalises_all_checked(
    make_viewer_model,
):
    """get_label_assignments normalises all-checked state to None."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)

    G = layer1.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1
    mask_data[:, shape[1] // 2 :] = 2
    labels_layer = viewer.add_labels(mask_data, name="lbl2")

    dialog = MaskAssignmentDialog(
        image_layer_names=[layer1.name],
        mask_layer_names=["None", "lbl2"],
        mask_layers=[labels_layer],
        current_assignments={layer1.name: "lbl2"},
        current_label_assignments={layer1.name: None},
        current_invert_assignments={},
        parent=None,
    )

    # The combo for layer1 already starts at "lbl2"; label_combo is populated
    label_combo = dialog._label_combos.get(layer1.name)
    assert label_combo is not None
    # Make sure all items are checked (default)
    label_combo.selectAll()
    assignments = dialog.get_label_assignments()
    assert (
        assignments.get(layer1.name, "MISSING") is None
    ), "All-checked labels should normalise to None in get_label_assignments"

    dialog.close()


def test_mask_assignment_dialog_label_items_colored_by_layer(
    make_viewer_model,
):
    """Label combo items get foreground color and background matching the Labels layer."""
    viewer = make_viewer_model()
    layer1 = create_image_layer_with_phasors()
    viewer.add_layer(layer1)

    G = layer1.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1
    mask_data[:, shape[1] // 2 :] = 2
    labels_layer = viewer.add_labels(mask_data, name="colored_labels")

    dialog = MaskAssignmentDialog(
        image_layer_names=[layer1.name],
        mask_layer_names=["None", "colored_labels"],
        mask_layers=[labels_layer],
        current_assignments={layer1.name: "None"},
        current_label_assignments={},
        current_invert_assignments={},
        parent=None,
    )

    # Trigger mask selection to populate label combo with colored items
    dialog._combos[layer1.name].setCurrentText("colored_labels")

    label_combo = dialog._label_combos[layer1.name]
    model = label_combo.model()
    offset = label_combo._header_count

    expected_bg = QColor(160, 160, 160, 160)
    for i, lbl in enumerate([1, 2]):
        item = model.item(offset + i)
        assert item is not None, f"Item for label {lbl} is missing"
        rgba = labels_layer.get_color(lbl)
        assert rgba is not None, f"Labels layer has no color for label {lbl}"
        r, g, b = (int(c * 255) for c in rgba[:3])
        assert item.foreground().color() == QColor(
            r, g, b
        ), f"Label {lbl}: wrong foreground color"
        assert (
            item.background().color() == expected_bg
        ), f"Label {lbl}: wrong background color"

    dialog.close()


def test_single_layer_mask_label_items_colored_by_layer(make_viewer_model):
    """mask_labels_combobox items get label colors in single-layer mode."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    G = layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1
    mask_data[:, shape[1] // 2 :] = 2
    labels_layer = viewer.add_labels(mask_data, name="single_colored_labels")

    plotter = PlotterWidget(viewer)
    # Select the image layer so single-layer mode is active
    plotter.image_layers_checkable_combobox.setCheckedItems([layer.name])

    # Selecting the labels layer triggers _on_mask_layer_changed
    plotter.mask_layer_combobox.setCurrentText("single_colored_labels")

    combo = plotter.mask_labels_combobox
    model = combo.model()
    offset = combo._header_count

    expected_bg = QColor(160, 160, 160, 160)
    for i, lbl in enumerate([1, 2]):
        item = model.item(offset + i)
        assert item is not None, f"Item for label {lbl} is missing"
        rgba = labels_layer.get_color(lbl)
        assert rgba is not None, f"Labels layer has no color for label {lbl}"
        r, g, b = (int(c * 255) for c in rgba[:3])
        assert item.foreground().color() == QColor(
            r, g, b
        ), f"Label {lbl}: wrong foreground color in single-layer mode"
        assert (
            item.background().color() == expected_bg
        ), f"Label {lbl}: wrong background color in single-layer mode"


def test_apply_label_colors_to_combo(make_viewer_model):
    """_apply_label_colors_to_combo sets foreground and background on items."""
    from napari_phasors._utils import CheckableComboBox

    viewer = make_viewer_model()
    mask_data = np.array([[0, 1], [2, 3]], dtype=int)
    labels_layer = viewer.add_labels(mask_data, name="helper_test_labels")

    combo = CheckableComboBox(
        placeholder="All Labels",
        enable_primary_layer=False,
        unit="labels",
        show_select_all_none=False,
        no_selection_text="No labels",
    )
    unique_labels = np.unique(labels_layer.data)
    valid_labels = [str(lbl) for lbl in unique_labels if lbl > 0]
    combo.addItems(valid_labels)

    _apply_label_colors_to_combo(combo, labels_layer, unique_labels)

    expected_bg = QColor(160, 160, 160, 160)
    offset = combo._header_count
    for i, lbl in enumerate([1, 2, 3]):
        item = combo.model().item(offset + i)
        assert item is not None
        rgba = labels_layer.get_color(lbl)
        if rgba is not None:
            r, g, b = (int(c * 255) for c in rgba[:3])
            assert item.foreground().color() == QColor(r, g, b)
            assert item.background().color() == expected_bg


def test_refresh_mask_labels_combobox_adds_new_label(make_viewer_model):
    """_refresh_mask_labels_combobox adds a new label and preserves selection."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    G = layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1
    labels_layer = viewer.add_labels(mask_data, name="refresh_labels")

    plotter = PlotterWidget(viewer)
    plotter.image_layers_checkable_combobox.setCheckedItems([layer.name])
    plotter.mask_layer_combobox.setCurrentText("refresh_labels")

    combo = plotter.mask_labels_combobox
    assert combo.allItems() == ["1"], "Should start with only label 1"
    # Check only label 1 (simulates user deselecting nothing — all checked)
    combo.selectAll()

    # Add a new label to the layer data
    new_data = mask_data.copy()
    new_data[: shape[0] // 2, shape[1] // 2 :] = 2
    labels_layer.data = new_data

    plotter._refresh_mask_labels_combobox(labels_layer)

    assert combo.allItems() == ["1", "2"], "Label 2 should be added"
    # New label 2 should be unchecked (previously_checked only had "1")
    checked = combo.checkedItems()
    assert "1" in checked, "Previously checked label 1 should remain checked"


def test_refresh_mask_labels_combobox_noop_when_unchanged(make_viewer_model):
    """_refresh_mask_labels_combobox does nothing when label set is unchanged."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    G = layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1
    mask_data[:, shape[1] // 2 :] = 2
    labels_layer = viewer.add_labels(mask_data, name="noop_labels")

    plotter = PlotterWidget(viewer)
    plotter.image_layers_checkable_combobox.setCheckedItems([layer.name])
    plotter.mask_layer_combobox.setCurrentText("noop_labels")

    combo = plotter.mask_labels_combobox
    assert combo.allItems() == ["1", "2"]
    combo.setCheckedItems(["1"])  # user selects only label 1

    # Call refresh with the same data — should be a no-op
    plotter._refresh_mask_labels_combobox(labels_layer)

    assert combo.checkedItems() == ["1"], "Selection should be unchanged"


def test_on_mask_data_changed_refreshes_label_combo(make_viewer_model):
    """Paint event on Labels layer updates mask_labels_combobox items."""
    viewer = make_viewer_model()
    layer = create_image_layer_with_phasors()
    viewer.add_layer(layer)

    G = layer.metadata["G"]
    shape = G.shape[1:] if G.ndim == 3 else G.shape
    mask_data = np.zeros(shape, dtype=int)
    mask_data[:, : shape[1] // 2] = 1
    labels_layer = viewer.add_labels(mask_data, name="paint_event_labels")

    plotter = PlotterWidget(viewer)
    plotter.image_layers_checkable_combobox.setCheckedItems([layer.name])
    plotter.mask_layer_combobox.setCurrentText("paint_event_labels")

    combo = plotter.mask_labels_combobox
    assert combo.allItems() == ["1"]

    # Simulate painting a new label by modifying data and calling the handler
    new_data = mask_data.copy()
    new_data[: shape[0] // 2, shape[1] // 2 :] = 2
    labels_layer.data = new_data

    # Fire the paint-event handler directly
    plotter._refresh_mask_labels_combobox(labels_layer)

    assert (
        "2" in combo.allItems()
    ), "New label 2 should appear after data change"


def _make_spatial_mask(layer, fill=1):
    """Build a labels mask matching a phasor layer's spatial dimensions."""
    g_data = layer.metadata["G"]
    mask_shape = g_data.shape[1:] if g_data.ndim == 3 else g_data.shape
    mask = np.zeros(mask_shape, dtype=int)
    mask[mask_shape[0] // 2 :, :] = fill
    return mask


def test_copy_mask_from_layer_copies_and_applies_mask(make_viewer_model):
    """_copy_mask_from_layer mirrors the source mask onto the target layer."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    source_layer = create_image_layer_with_phasors()
    target_layer = create_image_layer_with_phasors()
    viewer.add_layer(source_layer)
    viewer.add_layer(target_layer)

    # Mask the source layer (invert + a label selection so we can assert both
    # are carried over to the target).
    mask = _make_spatial_mask(source_layer)
    mask_layer = viewer.add_labels(mask, name="src_mask")
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [source_layer.name]
    )
    plotter._apply_mask_to_phasor_data(
        mask_layer, source_layer, invert=True, labels=[1]
    )

    copied = plotter._copy_mask_from_layer(source_layer, target_layer)

    assert copied is True
    np.testing.assert_array_equal(
        target_layer.metadata["mask"], source_layer.metadata["mask"]
    )
    assert target_layer.metadata["mask_invert"] is True
    assert target_layer.metadata["mask_labels"] == [1]
    # Mask was actually applied to the target's phasor coordinates.
    assert np.isnan(target_layer.metadata["G"]).sum() > 0
    # The per-layer assignment points at the source's mask layer.
    assert plotter._mask_assignments[target_layer.name] == "src_mask"
    assert plotter._mask_invert_assignments[target_layer.name] is True
    assert plotter._mask_label_assignments[target_layer.name] == [1]

    plotter.deleteLater()


def test_copy_mask_from_unmasked_source_clears_target_mask(make_viewer_model):
    """Copying masking from an unmasked source removes the target's mask."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    source_layer = create_image_layer_with_phasors()
    target_layer = create_image_layer_with_phasors()
    viewer.add_layer(source_layer)
    viewer.add_layer(target_layer)

    # Give the target an existing mask; the source has none.
    mask = _make_spatial_mask(target_layer)
    mask_layer = viewer.add_labels(mask, name="tgt_mask")
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [target_layer.name]
    )
    plotter._apply_mask_to_phasor_data(mask_layer, target_layer)
    assert "mask" in target_layer.metadata

    original_g = target_layer.metadata["G_original"]
    copied = plotter._copy_mask_from_layer(source_layer, target_layer)

    assert copied is False
    assert "mask" not in target_layer.metadata
    assert "mask_invert" not in target_layer.metadata
    # Phasor data restored to its unmasked original.
    np.testing.assert_array_almost_equal(
        target_layer.metadata["G"], original_g
    )

    plotter.deleteLater()


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


def test_copy_mask_from_layer_shape_mismatch_skips(make_viewer_model):
    """A mask whose shape differs from the target is not copied."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    source_layer = create_image_layer_with_phasors()
    target_layer = create_image_layer_with_phasors()
    viewer.add_layer(source_layer)
    viewer.add_layer(target_layer)

    # Stash a deliberately mismatched mask on the source.
    source_layer.metadata["mask"] = np.ones((3, 3), dtype=int)
    source_layer.metadata["mask_invert"] = False

    with patch(
        "napari_phasors.plotter.notifications.WarningNotification"
    ) as warn:
        copied = plotter._copy_mask_from_layer(source_layer, target_layer)

    assert copied is False
    assert "mask" not in target_layer.metadata
    warn.assert_called_once()

    plotter.deleteLater()


def test_copy_metadata_from_layer_with_masking_tab(make_viewer_model):
    """Selecting 'masking' in the import dialog copies the mask to targets."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    source_layer = create_image_layer_with_phasors()
    target_layer = create_image_layer_with_phasors()
    viewer.add_layer(source_layer)
    viewer.add_layer(target_layer)

    mask = _make_spatial_mask(source_layer)
    mask_layer = viewer.add_labels(mask, name="src_mask")
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [source_layer.name]
    )
    plotter._apply_mask_to_phasor_data(mask_layer, source_layer)

    # Now select the target and import with masking enabled.
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [target_layer.name]
    )
    plotter._copy_metadata_from_layer(
        source_layer.name, selected_tabs=["masking"]
    )

    assert "mask" in target_layer.metadata
    np.testing.assert_array_equal(
        target_layer.metadata["mask"], source_layer.metadata["mask"]
    )

    plotter.deleteLater()


def test_copy_metadata_from_layer_switches_selected_mask(make_viewer_model):
    """Regression: copying masking replaces the target's own mask selection.

    The target starts with its own mask; after importing masking from a source
    that uses a *different* (inverted) mask, the target's selected mask layer,
    invert flag, and NaN pattern must all match the source — not its own
    previous mask.
    """
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    source_layer = create_image_layer_with_phasors()
    target_layer = create_image_layer_with_phasors()
    viewer.add_layer(source_layer)
    viewer.add_layer(target_layer)

    src_mask = viewer.add_labels(
        _make_spatial_mask(source_layer), name="src_mask"
    )
    # A different mask for the target (left half rather than bottom half).
    g = target_layer.metadata["G"]
    shape = g.shape[1:] if g.ndim == 3 else g.shape
    tgt_arr = np.zeros(shape, dtype=int)
    tgt_arr[:, : shape[1] // 2] = 1
    tgt_mask = viewer.add_labels(tgt_arr, name="tgt_mask")

    # Apply the inverted source mask to the source.
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [source_layer.name]
    )
    plotter._apply_mask_to_phasor_data(src_mask, source_layer, invert=True)
    source_nan = np.isnan(source_layer.metadata["G"])

    # Give the target its own (non-inverted) mask and select it.
    plotter.image_layers_checkable_combobox.setCheckedItems(
        [target_layer.name]
    )
    plotter._apply_mask_to_phasor_data(tgt_mask, target_layer)
    plotter.mask_layer_combobox.setCurrentText("tgt_mask")

    plotter._copy_metadata_from_layer(
        source_layer.name, selected_tabs=["masking"]
    )

    # Selection now points at the source's mask, not the target's old one.
    assert plotter._mask_assignments[target_layer.name] == "src_mask"
    assert plotter.mask_layer_combobox.currentText() == "src_mask"
    assert plotter._mask_invert_assignments[target_layer.name] is True
    # Invert was honored: the NaN pattern matches the inverted source mask.
    np.testing.assert_array_equal(
        np.isnan(target_layer.metadata["G"]), source_nan
    )

    plotter.deleteLater()
