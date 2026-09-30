from unittest.mock import patch

import numpy as np

from napari_phasors._tests.test_plotter import (  # noqa: E501
    create_image_layer_with_phasors,
)
from napari_phasors._utils import analysis_layer_name
from napari_phasors.plotter import (
    PlotterWidget,
)


def test_tab_changes_show_only_that_tabs_artists(make_viewer_model):
    """Changing tab hides every tab's artists and shows the new tab's:
    components and FRET toggle their own artists, other tabs neither."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Real tab changes: the tab left is hidden and the tab entered shown.
    # The exact call counts vary with initialization and signal handling.
    with (
        patch.object(plotter, '_set_components_visibility') as mock_comp_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):
        plotter.tab_widget.setCurrentIndex(4)  # Components
        assert mock_comp_vis.call_count >= 2
        assert mock_fret_vis.call_count >= 1
        assert mock_comp_vis.call_args_list[-1][0][0]
        for call in mock_fret_vis.call_args_list:
            assert not call[0][0]

        mock_comp_vis.reset_mock()
        mock_fret_vis.reset_mock()
        plotter.tab_widget.setCurrentIndex(6)  # FRET
        assert mock_comp_vis.call_count >= 1
        assert mock_fret_vis.call_count >= 2
        for call in mock_comp_vis.call_args_list:
            assert not call[0][0]
        assert mock_fret_vis.call_args_list[-1][0][0]

    # _on_tab_changed hides all artists, shows the tab's and redraws, for
    # every tab.
    with (
        patch.object(plotter, '_hide_all_tab_artists') as mock_hide_all,
        patch.object(plotter, '_show_tab_artists') as mock_show_tab,
        patch.object(
            plotter.canvas_widget.figure.canvas, 'draw_idle'
        ) as mock_draw,
    ):
        for i in range(plotter.tab_widget.count()):
            mock_hide_all.reset_mock()
            mock_show_tab.reset_mock()
            mock_draw.reset_mock()
            plotter._on_tab_changed(i)
            mock_hide_all.assert_called_once()
            mock_show_tab.assert_called_once_with(plotter.tab_widget.widget(i))
            assert mock_draw.call_count >= 1

    with (
        patch.object(plotter, '_set_components_visibility') as mock_comp_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):
        # _hide_all_tab_artists only ever hides.
        plotter._hide_all_tab_artists()
        mock_comp_vis.assert_called_with(False)
        mock_fret_vis.assert_called_with(False)
        for call in (
            mock_comp_vis.call_args_list + mock_fret_vis.call_args_list
        ):
            assert not call[0][0]

        for tab, shown in (
            (plotter.components_tab, mock_comp_vis),
            (plotter.fret_tab, mock_fret_vis),
            (plotter.settings_tab, None),
        ):
            mock_comp_vis.reset_mock()
            mock_fret_vis.reset_mock()
            plotter._show_tab_artists(tab)
            for mock in (mock_comp_vis, mock_fret_vis):
                if mock is shown:
                    mock.assert_called_once_with(True)
                else:
                    mock.assert_not_called()

    # The tab-specific methods forward to the tab's own artists, and let
    # an error there propagate.
    for tab, method in (
        (plotter.components_tab, plotter._set_components_visibility),
        (plotter.fret_tab, plotter._set_fret_visibility),
    ):
        with patch.object(tab, 'set_artists_visible') as mock_set_visible:
            method(True)
            mock_set_visible.assert_called_once_with(True)
            mock_set_visible.reset_mock()
            method(False)
            mock_set_visible.assert_called_once_with(False)

        with patch.object(
            tab,
            'set_artists_visible',
            side_effect=Exception("Mock error in set_artists_visible"),
        ):
            try:
                method(True)
            except Exception as e:  # noqa: BLE001
                assert "Mock error in set_artists_visible" in str(e)


def test_tab_and_selection_mode_without_layers(make_viewer_model):
    """The tab widget drives the artist visibility, the selection mode can
    be switched, and the visibility helpers tolerate missing tabs, all
    without any layer."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Verify signal wiring by observing side effects from _on_tab_changed.
    with patch.object(plotter, '_hide_all_tab_artists') as mock_hide_all:
        plotter.tab_widget.setCurrentIndex(2)  # Change to filter tab
        assert mock_hide_all.call_count >= 1

    # Initially in circular cursor mode (index 0); manual is index 2.
    selection_widget = plotter.selection_tab
    assert selection_widget.selection_mode_combobox.currentIndex() == 0
    assert not selection_widget.is_manual_selection_mode()
    selection_widget.selection_mode_combobox.setCurrentText("Manual Selection")
    assert selection_widget.is_manual_selection_mode()
    selection_widget.selection_mode_combobox.setCurrentText("Cursor Selection")
    assert not selection_widget.is_manual_selection_mode()

    # The visibility helpers handle a missing components or FRET tab.
    for name, method in (
        ("components_tab", plotter._set_components_visibility),
        ("fret_tab", plotter._set_fret_visibility),
    ):
        original = getattr(plotter, name)
        delattr(plotter, name)
        try:
            method(True)
            method(False)
        except AttributeError as err:
            raise AssertionError(
                f"{method.__name__} should handle a missing {name} gracefully"
            ) from err
        finally:
            setattr(plotter, name, original)

    plotter.deleteLater()


def test_selection_toolbar_follows_the_selection_mode(make_viewer_model):
    """The manual-selection toolbar shows only in manual mode on the
    Selection tab; cursors and manual selections hide each other's layers,
    and leaving manual mode clears the plot colours."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)
    selection_widget = plotter.selection_tab
    mode_combobox = selection_widget.selection_mode_combobox

    def show_selection_tab_and_expect(visible):
        with patch.object(plotter, '_set_selection_visibility') as mock_vis:
            plotter._show_tab_artists(selection_widget)
            mock_vis.assert_called_once_with(visible)

    # Initially in circular cursor mode, toolbar should be hidden
    assert mode_combobox.currentIndex() == 0
    assert not selection_widget.is_manual_selection_mode()
    show_selection_tab_and_expect(False)

    # Manual selection mode shows it.
    mode_combobox.setCurrentText("Manual Selection")
    assert selection_widget.is_manual_selection_mode()
    show_selection_tab_and_expect(True)

    # Switching back to cursors hides it straight away.
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        mode_combobox.setCurrentText("Cursor Selection")
        mock_vis.assert_called_once_with(False)

    # The same on the Selection tab itself.
    plotter.tab_widget.setCurrentIndex(3)
    assert not selection_widget.is_manual_selection_mode()
    show_selection_tab_and_expect(False)
    mode_combobox.setCurrentText("Manual Selection")
    show_selection_tab_and_expect(True)

    # Switching modes within the tab ends with the matching visibility.
    for mode, visible in (
        ("Cursor Selection", False),
        ("Manual Selection", True),
        ("Cursor Selection", False),
    ):
        with patch.object(plotter, '_set_selection_visibility') as mock_vis:
            mode_combobox.setCurrentText(mode)
            assert mock_vis.call_count >= 1
            assert bool(mock_vis.call_args_list[-1][0][0]) is visible

    # Cursors are shown and the toolbar hidden in cursor mode; in manual
    # mode cursors are hidden via layer visibility, not this method.
    with (
        patch.object(plotter, '_set_selection_visibility') as mock_manual_vis,
        patch.object(
            plotter, '_set_selection_cursors_visibility'
        ) as mock_circular_vis,
    ):
        plotter._show_tab_artists(selection_widget)
        mock_manual_vis.assert_called_with(False)
        mock_circular_vis.assert_called_with(True)
    with patch.object(plotter, '_set_selection_visibility') as mock_manual_vis:
        mode_combobox.setCurrentText("Manual Selection")
        plotter._show_tab_artists(selection_widget)
        mock_manual_vis.assert_called_with(True)

    # Leaving the Selection tab hides the toolbar.
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        plotter.tab_widget.setCurrentIndex(4)
        mock_vis.assert_called_with(False)

    # A cursor selection and a manual selection hide each other's layers.
    mode_combobox.setCurrentText("Cursor Selection")
    assert mode_combobox.currentIndex() == 0
    assert selection_widget.stacked_widget.currentIndex() == 0
    circular_widget = selection_widget.cursor_selection_widget
    circular_widget._add_cursor()
    circular_widget._apply_selection()
    circular_layer_name = analysis_layer_name(
        "Cursor Selection", intensity_image_layer.name
    )
    assert circular_layer_name in [layer.name for layer in viewer.layers]
    circular_layer = viewer.layers[circular_layer_name]
    assert circular_layer.visible is True

    mode_combobox.setCurrentText("Manual Selection")
    assert selection_widget.stacked_widget.currentIndex() == 2
    assert selection_widget.is_manual_selection_mode()
    assert circular_layer.visible is False

    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    selection_widget.manual_selection_changed(manual_selection)
    manual_layer_name = analysis_layer_name(
        "MANUAL SELECTION #1", intensity_image_layer.name
    )
    assert manual_layer_name in [layer.name for layer in viewer.layers]
    manual_layer = viewer.layers[manual_layer_name]
    assert manual_layer.visible is True

    # Leaving manual mode clears the plot colours: plot is redrawn with
    # selection_id_data=None.
    with patch.object(plotter, 'plot') as mock_plot:
        mode_combobox.setCurrentText("Cursor Selection")
        mock_plot.assert_called_once_with(selection_id_data=None)
    assert not selection_widget.is_manual_selection_mode()
    assert circular_layer.visible is True
    assert manual_layer.visible is False

    # Circular cursor visibility method is only called in circular cursor mode
    # When in manual mode, circular cursors are hidden via layer visibility, not this method


def test_deferred_tab_update_after_primary_layer_removed(make_viewer_model):
    """Deferred tab updates must survive the primary layer being removed.

    The FRET and Phasor Mapping tabs resolve their layer by name. Once the
    plotter is closed its viewer connections are gone, so removing a layer
    leaves the combobox reporting a stale name while the layer is no longer
    in ``viewer.layers``. A pending update for a non-visible tab then runs
    against that stale name and used to raise ``KeyError``.
    """
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    plotter.image_layers_checkable_combobox.setCheckedItems(
        [intensity_image_layer.name]
    )
    plotter._process_layer_selection_change()

    # Keep the analysis tabs non-visible so their updates stay deferred.
    plotter.tab_widget.setCurrentWidget(plotter.settings_tab)
    plotter.close()
    viewer.layers.remove(intensity_image_layer)

    # The stale name is what makes this a regression test: the guards, not
    # an empty selection, are what keep the lookups from raising.
    stale_name = plotter.get_primary_layer_name()
    assert stale_name == intensity_image_layer.name
    assert stale_name not in viewer.layers
    assert plotter.fret_tab._needs_update
    assert plotter.phasor_mapping_tab._needs_update

    plotter.tab_widget.setCurrentWidget(plotter.fret_tab)
    plotter.tab_widget.setCurrentWidget(plotter.phasor_mapping_tab)

    # The individual guarded lookups, exercised directly.
    plotter.fret_tab._update_fret_setting_in_metadata('donor_lifetime', 1.0)
    plotter.fret_tab._restore_fret_settings_from_metadata()

    mapping_tab = plotter.phasor_mapping_tab
    assert mapping_tab._get_current_layer_mapping_settings() is None
    assert mapping_tab._get_current_layer_mapping_settings(create=True) is None
    mapping_tab._restore_lifetime_settings_from_metadata()
    mapping_tab._restore_lifetime_range_from_metadata()
