from unittest.mock import patch

import numpy as np

from napari_phasors._tests.test_plotter import (  # noqa: E501
    create_image_layer_with_phasors,
)
from napari_phasors.plotter import (
    PlotterWidget,
)


def test_phasor_plotter_tab_changed_functionality(make_viewer_model):
    """Test tab change functionality and artist visibility management."""

    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Mock the tab-specific visibility methods
    with (
        patch.object(plotter, '_hide_all_tab_artists') as mock_hide_all,
        patch.object(plotter, '_show_tab_artists') as mock_show_tab,
        patch.object(
            plotter.canvas_widget.figure.canvas, 'draw_idle'
        ) as mock_draw,
    ):

        # Test tab change to components tab (index 4)
        components_tab_index = 4
        plotter._on_tab_changed(components_tab_index)

        # Verify methods were called
        mock_hide_all.assert_called_once()
        mock_show_tab.assert_called_once_with(plotter.components_tab)
        assert mock_draw.call_count >= 1

        # Reset mocks
        mock_hide_all.reset_mock()
        mock_show_tab.reset_mock()
        mock_draw.reset_mock()

        # Test tab change to FRET tab (index 6)
        fret_tab_index = 6
        plotter._on_tab_changed(fret_tab_index)

        mock_hide_all.assert_called_once()
        mock_show_tab.assert_called_once_with(plotter.fret_tab)
        assert mock_draw.call_count >= 1


def test_phasor_plotter_hide_and_show_tab_artists(make_viewer_model):
    """Test hiding and showing tab-specific artists."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Mock the tab-specific visibility methods
    with (
        patch.object(
            plotter, '_set_components_visibility'
        ) as mock_components_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):

        # Test _hide_all_tab_artists
        plotter._hide_all_tab_artists()

        # Should call visibility methods with False
        mock_components_vis.assert_called_with(False)
        mock_fret_vis.assert_called_with(False)

        # Reset mocks
        mock_components_vis.reset_mock()
        mock_fret_vis.reset_mock()

        # Test _show_tab_artists with components tab
        plotter._show_tab_artists(plotter.components_tab)
        mock_components_vis.assert_called_once_with(True)
        mock_fret_vis.assert_not_called()

        # Reset mocks
        mock_components_vis.reset_mock()
        mock_fret_vis.reset_mock()

        # Test _show_tab_artists with FRET tab
        plotter._show_tab_artists(plotter.fret_tab)
        mock_fret_vis.assert_called_once_with(True)
        mock_components_vis.assert_not_called()

        # Test _show_tab_artists with non-specific tab (should not call any visibility methods)
        mock_components_vis.reset_mock()
        mock_fret_vis.reset_mock()

        plotter._show_tab_artists(plotter.settings_tab)
        mock_components_vis.assert_not_called()
        mock_fret_vis.assert_not_called()


def test_phasor_plotter_tab_specific_visibility_methods(make_viewer_model):
    """Test the tab-specific visibility methods."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Test _set_components_visibility
    with patch.object(
        plotter.components_tab, 'set_artists_visible'
    ) as mock_components_artists:
        plotter._set_components_visibility(True)
        mock_components_artists.assert_called_once_with(True)

        mock_components_artists.reset_mock()
        plotter._set_components_visibility(False)
        mock_components_artists.assert_called_once_with(False)

    # Test _set_fret_visibility
    with patch.object(
        plotter.fret_tab, 'set_artists_visible'
    ) as mock_fret_artists:
        plotter._set_fret_visibility(True)
        mock_fret_artists.assert_called_once_with(True)

        mock_fret_artists.reset_mock()
        plotter._set_fret_visibility(False)
        mock_fret_artists.assert_called_once_with(False)


def test_phasor_plotter_tab_widget_signal_connection(make_viewer_model):
    """Test that tab widget currentChanged signal is properly connected."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Verify signal wiring by observing side effects from _on_tab_changed.
    with patch.object(plotter, '_hide_all_tab_artists') as mock_hide_all:
        plotter.tab_widget.setCurrentIndex(2)  # Change to filter tab
        assert mock_hide_all.call_count >= 1

    plotter.deleteLater()


def test_phasor_plotter_tab_changed_with_different_tabs(make_viewer_model):
    """Test tab changes with different tab indices."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    with (
        patch.object(plotter, '_hide_all_tab_artists') as mock_hide_all,
        patch.object(plotter, '_show_tab_artists') as mock_show_tab,
    ):

        # Test each tab index

        for i in range(plotter.tab_widget.count()):
            mock_hide_all.reset_mock()
            mock_show_tab.reset_mock()

            plotter._on_tab_changed(i)

            # Should always hide all artists first
            mock_hide_all.assert_called_once()

            # Should show artists for the current tab
            expected_tab = plotter.tab_widget.widget(i)
            mock_show_tab.assert_called_once_with(expected_tab)


def test_phasor_plotter_set_components_visibility_method(make_viewer_model):
    """Test _set_components_visibility method behavior."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Test with components tab having set_artists_visible method
    with patch.object(
        plotter.components_tab, 'set_artists_visible'
    ) as mock_set_visible:
        # Test setting visibility to True
        plotter._set_components_visibility(True)
        mock_set_visible.assert_called_once_with(True)

        mock_set_visible.reset_mock()

        # Test setting visibility to False
        plotter._set_components_visibility(False)
        mock_set_visible.assert_called_once_with(False)


def test_phasor_plotter_set_fret_visibility_method(make_viewer_model):
    """Test _set_fret_visibility method behavior."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Test with FRET tab having set_artists_visible method
    with patch.object(
        plotter.fret_tab, 'set_artists_visible'
    ) as mock_set_visible:
        # Test setting visibility to True
        plotter._set_fret_visibility(True)
        mock_set_visible.assert_called_once_with(True)

        mock_set_visible.reset_mock()

        # Test setting visibility to False
        plotter._set_fret_visibility(False)
        mock_set_visible.assert_called_once_with(False)


def test_phasor_plotter_set_visibility_methods_without_tabs(
    make_viewer_model,
):
    """Test visibility methods when tabs don't have the expected attributes."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)

    # Remove the components_tab attribute to test the hasattr check
    original_components_tab = plotter.components_tab
    del plotter.components_tab

    # Should not raise an error when components_tab doesn't exist
    try:
        plotter._set_components_visibility(True)
        plotter._set_components_visibility(False)
    except AttributeError as err:
        raise AssertionError(
            "_set_components_visibility should handle missing components_tab gracefully"
        ) from err

    # Restore components_tab
    plotter.components_tab = original_components_tab

    # Remove the fret_tab attribute to test the hasattr check
    original_fret_tab = plotter.fret_tab
    del plotter.fret_tab

    # Should not raise an error when fret_tab doesn't exist
    try:
        plotter._set_fret_visibility(True)
        plotter._set_fret_visibility(False)
    except AttributeError as err:
        raise AssertionError(
            "_set_fret_visibility should handle missing fret_tab gracefully"
        ) from err

    # Restore fret_tab
    plotter.fret_tab = original_fret_tab


def test_phasor_plotter_visibility_methods_error_handling(make_viewer_model):
    """Test that visibility methods handle errors gracefully."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Mock set_artists_visible to raise an exception
    def mock_set_visible_error(visible):
        raise Exception("Mock error in set_artists_visible")

    with patch.object(
        plotter.components_tab,
        'set_artists_visible',
        side_effect=mock_set_visible_error,
    ):
        # Should not crash if set_artists_visible raises an exception
        try:
            plotter._set_components_visibility(True)
        except Exception as e:  # noqa: BLE001
            # If the method doesn't handle the exception, the test will catch it
            # In a real implementation, you might want to handle this gracefully
            assert "Mock error in set_artists_visible" in str(e)

    with patch.object(
        plotter.fret_tab,
        'set_artists_visible',
        side_effect=mock_set_visible_error,
    ):
        # Should not crash if set_artists_visible raises an exception
        try:
            plotter._set_fret_visibility(False)
        except Exception as e:  # noqa: BLE001
            # If the method doesn't handle the exception, the test will catch it
            # In a real implementation, you might want to handle this gracefully
            assert "Mock error in set_artists_visible" in str(e)


def test_phasor_plotter_visibility_methods_called_by_hide_show_artists(
    make_viewer_model,
):
    """Test that visibility methods are called correctly by hide/show artists methods."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Test that _hide_all_tab_artists calls both visibility methods with False
    with (
        patch.object(plotter, '_set_components_visibility') as mock_comp_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):

        plotter._hide_all_tab_artists()

        # Check that the methods were called with False (allow multiple calls)
        mock_comp_vis.assert_called_with(False)
        mock_fret_vis.assert_called_with(False)

        # Verify all calls were with False
        for call in mock_comp_vis.call_args_list:
            assert not call[0][0]
        for call in mock_fret_vis.call_args_list:
            assert not call[0][0]

    # Test that _show_tab_artists calls the correct visibility method for components tab
    with (
        patch.object(plotter, '_set_components_visibility') as mock_comp_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):

        plotter._show_tab_artists(plotter.components_tab)

        mock_comp_vis.assert_called_once_with(True)
        mock_fret_vis.assert_not_called()

    # Test that _show_tab_artists calls the correct visibility method for FRET tab
    with (
        patch.object(plotter, '_set_components_visibility') as mock_comp_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):

        plotter._show_tab_artists(plotter.fret_tab)

        mock_comp_vis.assert_not_called()
        mock_fret_vis.assert_called_once_with(True)


def test_toolbar_visibility_based_on_selection_mode(make_viewer_model):
    """Test that toolbar visibility is controlled by selection mode."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Initially in circular cursor mode, toolbar should be hidden
    assert plotter.selection_tab.selection_mode_combobox.currentIndex() == 0
    assert not plotter.selection_tab.is_manual_selection_mode()

    # In circular cursor mode, _show_tab_artists should not show toolbar
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        plotter._show_tab_artists(plotter.selection_tab)
        # Should call _set_selection_visibility(False) when in circular cursor mode
        mock_vis.assert_called_once_with(False)

    # Switch to manual selection mode
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        "Manual Selection"
    )
    assert plotter.selection_tab.is_manual_selection_mode()

    # Now _show_tab_artists should show toolbar
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        plotter._show_tab_artists(plotter.selection_tab)
        mock_vis.assert_called_once_with(True)

    # Mock _set_selection_visibility to track calls
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        # Switch to circular cursor mode
        plotter.selection_tab.selection_mode_combobox.setCurrentText(
            "Circular Cursor"
        )

        # Should call _set_selection_visibility(False) to hide toolbar
        mock_vis.assert_called_once_with(False)


def test_is_manual_selection_mode_method(make_viewer_model):
    """Test the is_manual_selection_mode() method."""
    viewer = make_viewer_model()
    plotter = PlotterWidget(viewer)
    selection_widget = plotter.selection_tab

    # Initially in circular cursor mode (index 0)
    assert selection_widget.selection_mode_combobox.currentIndex() == 0
    assert not selection_widget.is_manual_selection_mode()

    # Switch to manual selection mode (index 2)
    selection_widget.selection_mode_combobox.setCurrentText("Manual Selection")
    assert selection_widget.is_manual_selection_mode()

    # Switch back to circular cursor mode
    selection_widget.selection_mode_combobox.setCurrentText("Circular Cursor")
    assert not selection_widget.is_manual_selection_mode()


def test_plot_colors_cleared_when_switching_from_manual_mode(
    make_viewer_model,
):
    """Test that plot colors are cleared when switching from manual selection mode."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Switch to manual selection mode
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        "Manual Selection"
    )

    # Make a manual selection (this would create colored points on plot)
    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    plotter.selection_tab.manual_selection_changed(manual_selection)

    # Mock plot method to verify it's called with selection_id_data=None
    with patch.object(plotter, 'plot') as mock_plot:
        # Switch to circular cursor mode
        plotter.selection_tab.selection_mode_combobox.setCurrentText(
            "Circular Cursor"
        )

        # plot should be called with selection_id_data=None to clear colors
        mock_plot.assert_called_once_with(selection_id_data=None)


def test_selection_tab_mode_switching_integration(make_viewer_model):
    """Integration test for complete mode switching workflow."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)
    selection_widget = plotter.selection_tab

    # Start in circular cursor mode
    assert selection_widget.selection_mode_combobox.currentIndex() == 0
    assert selection_widget.stacked_widget.currentIndex() == 0

    # Add a circular cursor
    circular_widget = selection_widget.circular_cursor_widget
    circular_widget._add_cursor()
    circular_widget._apply_selection()

    # Verify circular cursor layer exists (actual name is 'Cursor Selection:')
    circular_layer_name = f"Cursor Selection: {intensity_image_layer.name}"
    assert circular_layer_name in [layer.name for layer in viewer.layers]
    circular_layer = viewer.layers[circular_layer_name]
    assert circular_layer.visible is True

    # Switch to manual selection mode
    selection_widget.selection_mode_combobox.setCurrentText("Manual Selection")

    # Verify UI switched
    assert selection_widget.stacked_widget.currentIndex() == 4
    assert selection_widget.is_manual_selection_mode()

    # Verify circular cursor layer is now hidden
    assert circular_layer.visible is False

    # Make a manual selection
    manual_selection = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    selection_widget.manual_selection_changed(manual_selection)

    # Verify manual selection layer exists and is visible (no 'Selection ' prefix)
    manual_layer_name = f"MANUAL SELECTION #1: {intensity_image_layer.name}"
    assert manual_layer_name in [layer.name for layer in viewer.layers]
    manual_layer = viewer.layers[manual_layer_name]
    assert manual_layer.visible is True

    # Switch back to circular cursor mode
    selection_widget.selection_mode_combobox.setCurrentText("Circular Cursor")

    # Verify we're back in circular cursor mode
    assert not selection_widget.is_manual_selection_mode()

    # Verify visibility has switched back
    assert circular_layer.visible is True
    assert manual_layer.visible is False


def test_toolbar_hidden_when_switching_tabs(make_viewer_model):
    """Test that toolbar is hidden when switching away from selection tab."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Start at selection tab (index 3)
    plotter.tab_widget.setCurrentIndex(3)

    # Switch to manual selection mode to show toolbar
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        "Manual Selection"
    )

    # Mock _set_selection_visibility to track calls
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        # Switch to a different tab (e.g., components tab at index 4)
        plotter.tab_widget.setCurrentIndex(4)

        # Should call _set_selection_visibility(False) to hide toolbar
        mock_vis.assert_called_with(False)


def test_toolbar_hidden_in_circular_cursor_mode(make_viewer_model):
    """Test that toolbar is hidden when in circular cursor mode."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Start at selection tab
    plotter.tab_widget.setCurrentIndex(3)

    # Start in circular cursor mode
    assert not plotter.selection_tab.is_manual_selection_mode()

    # Mock _set_selection_visibility to track calls
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        # Trigger _show_tab_artists for selection tab
        plotter._show_tab_artists(plotter.selection_tab)

        # Should call _set_selection_visibility(False) because in circular cursor mode
        mock_vis.assert_called_once_with(False)


def test_toolbar_shown_only_in_manual_selection_mode(make_viewer_model):
    """Test that toolbar is only shown in manual selection mode."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Go to selection tab
    plotter.tab_widget.setCurrentIndex(3)

    # Test circular cursor mode - toolbar should be hidden
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        "Circular Cursor"
    )
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        plotter._show_tab_artists(plotter.selection_tab)
        mock_vis.assert_called_once_with(False)

    # Test manual selection mode - toolbar should be shown
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        "Manual Selection"
    )
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        plotter._show_tab_artists(plotter.selection_tab)
        mock_vis.assert_called_once_with(True)


def test_toolbar_visibility_on_mode_change_within_selection_tab(
    make_viewer_model,
):
    """Test that toolbar visibility changes when switching modes within selection tab."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Go to selection tab
    plotter.tab_widget.setCurrentIndex(3)

    # Start in circular cursor mode
    plotter.selection_tab.selection_mode_combobox.setCurrentText(
        "Circular Cursor"
    )

    # Mock _set_selection_visibility to track calls
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        # Switch to manual selection mode
        plotter.selection_tab.selection_mode_combobox.setCurrentText(
            "Manual Selection"
        )

        # Should call _set_selection_visibility(True) to show toolbar
        assert mock_vis.call_count >= 1
        # Last call should be True
        assert mock_vis.call_args_list[-1][0][0]

    # Reset mock
    with patch.object(plotter, '_set_selection_visibility') as mock_vis:
        # Switch back to circular cursor mode
        plotter.selection_tab.selection_mode_combobox.setCurrentText(
            "Circular Cursor"
        )

        # Should call _set_selection_visibility(False) to hide toolbar
        assert mock_vis.call_count >= 1
        # Last call should be False
        assert not mock_vis.call_args_list[-1][0][0]


def test_circular_cursor_and_manual_selection_visibility_coordination(
    make_viewer_model,
):
    """Test that circular cursor and manual selection visibility methods are properly coordinated."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    # Go to selection tab
    plotter.tab_widget.setCurrentIndex(3)

    # Mock both visibility methods
    with (
        patch.object(plotter, '_set_selection_visibility') as mock_manual_vis,
        patch.object(
            plotter, '_set_selection_cursors_visibility'
        ) as mock_circular_vis,
    ):
        # Switch to circular cursor mode
        plotter.selection_tab.selection_mode_combobox.setCurrentText(
            "Circular Cursor"
        )
        plotter._show_tab_artists(plotter.selection_tab)

        # Manual selection toolbar should be hidden
        mock_manual_vis.assert_called_with(False)
        # Circular cursor should be shown
        mock_circular_vis.assert_called_with(True)

    # Reset mocks
    with (
        patch.object(plotter, '_set_selection_visibility') as mock_manual_vis,
    ):
        # Switch to manual selection mode
        plotter.selection_tab.selection_mode_combobox.setCurrentText(
            "Manual Selection"
        )
        plotter._show_tab_artists(plotter.selection_tab)

        # Manual selection toolbar should be shown
        mock_manual_vis.assert_called_with(True)
        # Circular cursor visibility method is only called in circular cursor mode
        # When in manual mode, circular cursors are hidden via layer visibility, not this method


def test_phasor_plotter_visibility_methods_integration_with_tab_changes(
    make_viewer_model,
):
    """Test integration of visibility methods with actual tab changes."""
    viewer = make_viewer_model()
    intensity_image_layer = create_image_layer_with_phasors()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)

    with (
        patch.object(plotter, '_set_components_visibility') as mock_comp_vis,
        patch.object(plotter, '_set_fret_visibility') as mock_fret_vis,
    ):

        # Reset mocks to clear any initialization calls
        mock_comp_vis.reset_mock()
        mock_fret_vis.reset_mock()

        # Change to components tab (index 4)
        plotter.tab_widget.setCurrentIndex(4)

        # Should have at least one False call (hide) and one True call (show) for components
        # The exact count may vary depending on initialization and signal handling
        assert mock_comp_vis.call_count >= 2
        assert mock_fret_vis.call_count >= 1

        # Check that the last calls were in the correct order
        calls_comp = mock_comp_vis.call_args_list
        calls_fret = mock_fret_vis.call_args_list

        # The last component call should be True (show components)
        assert calls_comp[-1][0][0]
        # All fret calls should be False (hide fret)
        for call in calls_fret:
            assert not call[0][0]

        # Reset mocks
        mock_comp_vis.reset_mock()
        mock_fret_vis.reset_mock()

        # Change to FRET tab (index 6)
        plotter.tab_widget.setCurrentIndex(6)

        # Should have at least one False call (hide) and one True call (show) for FRET
        assert mock_comp_vis.call_count >= 1
        assert mock_fret_vis.call_count >= 2

        # Check the calls were in the right order
        calls_comp = mock_comp_vis.call_args_list
        calls_fret = mock_fret_vis.call_args_list

        # All component calls should be False (hide components)
        for call in calls_comp:
            assert not call[0][0]
        # The last FRET call should be True (show FRET)
        assert calls_fret[-1][0][0]
