"""Tests for level-of-detail control of large phasor layers."""

import numpy as np
import pytest

from napari_phasors._lod import (
    LodManager,
    PhasorLod,
    layer_supports_lod,
)


def make_lod_layer(viewer, size=64, scale=(1.0, 1.0), settings=None):
    """Add an image layer carrying full-resolution phasor arrays."""
    rng = np.random.default_rng(0)
    mean = rng.random((size, size)).astype(np.float32) * 100
    real = rng.random((2, size, size)).astype(np.float32)
    imag = rng.random((2, size, size)).astype(np.float32)
    return viewer.add_image(
        mean.copy(),
        name="phasor",
        scale=scale,
        metadata={
            "original_mean": mean,
            "G_original": real,
            "S_original": imag,
            "G": real.copy(),
            "S": imag.copy(),
            "harmonics": np.array([1, 2]),
            "settings": settings if settings is not None else {},
        },
    )


def test_layer_supports_lod(make_viewer_model):
    viewer = make_viewer_model()
    layer = make_lod_layer(viewer)
    assert layer_supports_lod(layer)

    plain = viewer.add_image(np.zeros((4, 4)))
    assert not layer_supports_lod(plain)


def test_lod_rejects_a_layer_without_phasor_data(make_viewer_model):
    viewer = make_viewer_model()
    plain = viewer.add_image(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="no phasor data"):
        PhasorLod(plain)


def test_applying_a_level_reshapes_every_phasor_array(make_viewer_model):
    """The working arrays must follow the level, or consumers disagree."""
    viewer = make_viewer_model()
    layer = make_lod_layer(viewer, size=64)
    lod = PhasorLod(layer)

    assert lod.apply(4) is True
    assert layer.data.shape == (16, 16)
    assert layer.metadata["original_mean"].shape == (16, 16)
    assert layer.metadata["G"].shape == (2, 16, 16)
    assert layer.metadata["S"].shape == (2, 16, 16)
    assert layer.metadata["G_original"].shape == (2, 16, 16)


def test_reapplying_the_same_level_is_a_no_op(make_viewer_model):
    viewer = make_viewer_model()
    lod = PhasorLod(make_lod_layer(viewer))
    lod.apply(2)
    assert lod.apply(2) is False


def test_world_extent_survives_a_level_change(make_viewer_model):
    """A level swap must not move the image under the camera."""
    viewer = make_viewer_model()
    layer = make_lod_layer(viewer, size=64)
    lod = PhasorLod(layer)

    full = layer.extent.world.copy()
    lod.apply(4)
    coarse = layer.extent.world

    # Extents agree to within one coarse pixel (napari measures between pixel
    # centres, so the last bin's width shows up as a small difference).
    assert np.allclose(coarse, full, atol=4)


def test_physical_scale_is_multiplied_not_replaced(make_viewer_model):
    """A layer with micron spacing must keep it across levels."""
    viewer = make_viewer_model()
    layer = make_lod_layer(viewer, size=64, scale=(0.5, 0.5))
    lod = PhasorLod(layer)

    lod.apply(4)
    assert tuple(layer.scale) == (2.0, 2.0)
    lod.apply(1)
    assert tuple(layer.scale) == (0.5, 0.5)


def test_region_holds_the_real_pixels_and_sits_in_the_right_place(
    make_viewer_model,
):
    viewer = make_viewer_model()
    layer = make_lod_layer(viewer, size=64, scale=(0.5, 0.5))
    source = layer.metadata["original_mean"].copy()
    lod = PhasorLod(layer)

    lod.apply(1, region=(16, 32, 24, 48))

    assert lod.origin == (16, 24)
    assert np.array_equal(
        layer.metadata["original_mean"], source[16:32, 24:48]
    )
    # translate is the region origin expressed in world units.
    assert tuple(layer.translate) == (8.0, 12.0)


def test_zooming_in_earns_full_resolution(make_viewer_model):
    """Refining a small enough region reaches factor 1."""
    viewer = make_viewer_model()
    layer = make_lod_layer(viewer, size=64)
    lod = PhasorLod(layer)
    lod.apply(4)

    lod.refine_to((0, 8, 0, 8), budget=1024)
    assert lod.factor == 1


def test_refining_to_the_whole_image_drops_the_crop(make_viewer_model):
    viewer = make_viewer_model()
    layer = make_lod_layer(viewer, size=64)
    lod = PhasorLod(layer)
    lod.apply(1, region=(0, 16, 0, 16))

    lod.refine_to((0, 64, 0, 64), budget=64)
    assert lod.origin == (0, 0)
    assert tuple(layer.translate) == (0.0, 0.0)


def test_stored_filter_is_rerun_at_the_new_level(make_viewer_model):
    """Thresholded-out pixels must stay out after a level change."""
    viewer = make_viewer_model()
    layer = make_lod_layer(
        viewer,
        size=64,
        settings={
            "threshold": 50.0,
            "threshold_upper": None,
            "threshold_method": "Manual",
        },
    )
    lod = PhasorLod(layer)
    lod.apply(2)

    below = layer.metadata["original_mean"] < 50.0
    assert below.any(), "test needs some pixels under the threshold"
    assert np.isnan(layer.metadata["G"][0][below]).all()


def test_detach_restores_full_resolution(make_viewer_model):
    viewer = make_viewer_model()
    layer = make_lod_layer(viewer, size=64)
    source = layer.metadata["original_mean"].copy()
    lod = PhasorLod(layer)

    lod.apply(4, region=(16, 48, 16, 48))
    lod.detach()

    assert lod.factor == 1
    assert layer.data.shape == (64, 64)
    assert np.array_equal(layer.metadata["original_mean"], source)
    assert tuple(layer.translate) == (0.0, 0.0)
    assert lod.pyramid.available_factors() == []


def test_mask_is_downsampled_alongside_the_data(make_viewer_model):
    """A mask must keep matching the arrays it filters."""
    viewer = make_viewer_model()
    layer = make_lod_layer(viewer, size=64)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[0, 0] = 1
    layer.metadata["mask"] = mask

    lod = PhasorLod(layer)
    lod.apply(4)

    assert layer.metadata["mask"].shape == (16, 16)
    # Block-max: a block holding any labelled pixel stays labelled.
    assert layer.metadata["mask"][0, 0] == 1


class TestLodManager:
    """Camera-driven refinement."""

    def test_attach_auto_bins_only_large_layers(self, make_viewer_model):
        viewer = make_viewer_model()
        layer = make_lod_layer(viewer, size=64)
        manager = LodManager(viewer, budget=256)

        lod = manager.attach(layer, auto=True)
        assert lod.factor > 1

        manager.detach_all()
        assert lod.factor == 1

    def test_attach_leaves_small_layers_alone(self, make_viewer_model):
        viewer = make_viewer_model()
        layer = make_lod_layer(viewer, size=16)
        manager = LodManager(viewer, budget=10_000_000)
        assert manager.attach(layer, auto=True).factor == 1

    def test_attach_ignores_layers_without_phasor_data(
        self, make_viewer_model
    ):
        viewer = make_viewer_model()
        plain = viewer.add_image(np.zeros((4, 4)))
        manager = LodManager(viewer)
        assert manager.attach(plain) is None
        assert manager.layers == []

    def test_attaching_twice_reuses_the_pyramid(self, make_viewer_model):
        viewer = make_viewer_model()
        layer = make_lod_layer(viewer)
        manager = LodManager(viewer)
        assert manager.attach(layer) is manager.attach(layer)
        assert manager.lod_for(layer) is not None

    def test_visible_region_tracks_the_camera(self, make_viewer_model):
        viewer = make_viewer_model()
        layer = make_lod_layer(viewer, size=256)
        manager = LodManager(viewer)
        lod = manager.attach(layer, auto=False)

        viewer.camera.zoom = 0.01
        wide = manager.visible_region(lod)
        viewer.camera.zoom = 20.0
        viewer.camera.center = (0, 128, 128)
        narrow = manager.visible_region(lod)

        assert wide == (0, 256, 0, 256)
        assert narrow is not None
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_visible_region_is_none_without_a_usable_camera(
        self, make_viewer_model
    ):
        viewer = make_viewer_model()
        layer = make_lod_layer(viewer)
        manager = LodManager(viewer)
        lod = manager.attach(layer, auto=False)

        viewer.camera.zoom = 0
        assert manager.visible_region(lod) is None

    def test_refine_now_follows_the_camera(self, make_viewer_model):
        viewer = make_viewer_model()
        layer = make_lod_layer(viewer, size=256)
        manager = LodManager(viewer, budget=4096)
        manager.attach(layer, auto=True)
        coarse = manager.lod_for(layer).factor

        viewer.camera.zoom = 40.0
        viewer.camera.center = (0, 128, 128)
        changed = manager.refine_now()

        assert changed == [layer]
        assert manager.lod_for(layer).factor < coarse

    def test_enabling_connects_and_disconnecting_releases(
        self, make_viewer_model
    ):
        viewer = make_viewer_model()
        manager = LodManager(viewer)

        manager.set_enabled(True)
        assert manager.enabled
        manager.set_enabled(True)  # idempotent

        manager.disconnect()
        assert not manager.enabled

    def test_camera_movement_is_debounced(self, make_viewer_model):
        """A zoom gesture must schedule one recompute, not one per event."""
        viewer = make_viewer_model()
        layer = make_lod_layer(viewer, size=128)
        manager = LodManager(viewer)
        manager.attach(layer, auto=False)
        manager.set_enabled(True)

        for zoom in (2.0, 3.0, 4.0, 5.0):
            viewer.camera.zoom = zoom

        assert manager._timer.isActive()
        manager.disconnect()
        assert not manager._timer.isActive()


class TestFilterTabLodControls:
    """The Level of Detail section wired into the filter tab."""

    @staticmethod
    def _widget(make_viewer_model, size=64):
        from napari_phasors.plotter import PlotterWidget

        viewer = make_viewer_model()
        layer = make_lod_layer(viewer, size=size)
        parent = PlotterWidget(viewer)
        return viewer, layer, parent, parent.filter_tab

    def test_manager_is_built_lazily(self, make_viewer_model):
        """A tab that never bins anything holds no camera connection."""
        _, _, _, tab = self._widget(make_viewer_model)
        assert tab._lod_manager is None
        assert tab.lod_manager is not None
        assert tab.lod_manager is tab.lod_manager

    def test_controls_start_disabled(self, make_viewer_model):
        _, _, _, tab = self._widget(make_viewer_model)
        assert not tab.lod_checkbox.isChecked()
        assert not tab.lod_zoom_checkbox.isEnabled()
        assert not tab.lod_full_button.isEnabled()
        assert tab.lod_status_label.text() == "Full resolution"

    def test_enabling_bins_the_selected_layer(self, make_viewer_model):
        _, layer, _, tab = self._widget(make_viewer_model, size=64)
        tab.lod_manager.budget = 256

        tab.lod_checkbox.setChecked(True)

        assert tab.lod_zoom_checkbox.isEnabled()
        assert tab.lod_full_button.isEnabled()
        assert tab.lod_manager.lod_for(layer).factor > 1
        assert "Binned" in tab.lod_status_label.text()

    def test_disabling_restores_full_resolution(self, make_viewer_model):
        _, layer, _, tab = self._widget(make_viewer_model, size=64)
        tab.lod_manager.budget = 256

        tab.lod_checkbox.setChecked(True)
        tab.lod_checkbox.setChecked(False)

        assert layer.data.shape == (64, 64)
        assert tab.lod_status_label.text() == "Full resolution"
        assert not tab.lod_manager.enabled

    def test_full_resolution_button_keeps_binning_enabled(
        self, make_viewer_model
    ):
        _, layer, _, tab = self._widget(make_viewer_model, size=64)
        tab.lod_manager.budget = 256
        tab.lod_checkbox.setChecked(True)

        tab.lod_full_button.click()

        assert layer.data.shape == (64, 64)
        assert tab.lod_checkbox.isChecked()
        assert tab.lod_status_label.text() == "Full resolution"

    def test_enabling_without_a_selection_warns_and_reverts(
        self, make_viewer_model, monkeypatch
    ):
        from napari_phasors.plotter import PlotterWidget

        viewer = make_viewer_model()
        parent = PlotterWidget(viewer)
        tab = parent.filter_tab

        errors = []
        monkeypatch.setattr(
            "napari_phasors.filter_tab.show_error", errors.append
        )
        tab.lod_checkbox.setChecked(True)

        assert errors
        assert not tab.lod_checkbox.isChecked()

    def test_zoom_toggle_drives_the_camera_connection(self, make_viewer_model):
        _, _, _, tab = self._widget(make_viewer_model, size=64)
        tab.lod_manager.budget = 256
        tab.lod_checkbox.setChecked(True)

        tab.lod_zoom_checkbox.setChecked(True)
        assert tab.lod_manager.enabled

        tab.lod_zoom_checkbox.setChecked(False)
        assert not tab.lod_manager.enabled

    def test_close_releases_the_camera_connection(self, make_viewer_model):
        """Leaving connections on a closed widget crashes PySide6 teardown."""
        _, _, _, tab = self._widget(make_viewer_model, size=64)
        tab.lod_manager.budget = 256
        tab.lod_checkbox.setChecked(True)
        tab.lod_zoom_checkbox.setChecked(True)

        tab.close()

        assert not tab._lod_manager.enabled
        assert not tab._lod_manager._timer.isActive()
