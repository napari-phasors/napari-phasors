def _test_phasor_plotter_canvas_click_handling(make_napari_viewer):
    """Test canvas click event handling."""
    viewer = make_napari_viewer()
    plotter = PlotterWidget(viewer)

    # Test that canvas click handler exists
    assert hasattr(plotter, '_on_canvas_click')

    # Create a mock event
    class MockEvent:
        def __init__(self, button=1, xdata=0.5, ydata=0.3, inaxes=None):
            self.button = button
            self.xdata = xdata
            self.ydata = ydata
            self.inaxes = inaxes

    # Test left click with valid coordinates
    event = MockEvent(
        button=1, xdata=0.5, ydata=0.3, inaxes=plotter.canvas_widget.axes
    )
    result = plotter._on_canvas_click(event)
    assert result == (0.5, 0.3)

    # Test click outside axes
    event = MockEvent(button=1, xdata=0.5, ydata=0.3, inaxes=None)
    result = plotter._on_canvas_click(event)
    assert result == (None, None)

    # Test right click
    event = MockEvent(
        button=3, xdata=0.5, ydata=0.3, inaxes=plotter.canvas_widget.axes
    )
    result = plotter._on_canvas_click(event)
    assert result == (None, None)

    # Test click with invalid coordinates
    event = MockEvent(
        button=1, xdata=None, ydata=None, inaxes=plotter.canvas_widget.axes
    )
    result = plotter._on_canvas_click(event)
    assert result == (None, None)