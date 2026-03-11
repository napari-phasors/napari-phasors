self.histogram_widget.update_multi_data(per_layer_data)
else:
    self.histogram_widget.update_data(first_layer.data)

self.histogram_widget.show()

def closeEvent(self, event):
    """Clean up signal connections before closing."""
    # Disconnect parent widget signal if present
    if hasattr(self, 'parent_widget') and self.parent_widget:
        with contextlib.suppress(ValueError, AttributeError):
            self.parent_widget.harmonic_spinbox.valueChanged.disconnect(
                self._on_harmonic_changed
            )

    event.accept()

# Fix for #202 – cursors visibility when components tab is active
def set_cursors_visible(self):
    try:
        self.viewer.cursor = QtCore.Qt.ArrowCursor
    except AttributeError:
        pass
