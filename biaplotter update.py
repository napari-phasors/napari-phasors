def apply_selection(self):
        """Applies the selection to the data, updating the colors."""
        if self._selected_indices is None or len(self._selected_indices) == 0:
            self._selected_indices = None
            return

        # Ensure the overlay_colormap of the active artist is set to cat10_mod_cmap if needed
        if not self._active_artist.overlay_colormap.cmap.name.startswith(
            "cat10"
        ):
            # Clear previous color indices to remove previous feature coloring
            self._active_artist.color_indices = 0
            if isinstance(self._active_artist, Scatter):
                self._active_artist.overlay_colormap = cat10_mod_cmap
            elif isinstance(self._active_artist, Histogram2D):
                self._active_artist.overlay_colormap = (
                    cat10_mod_cmap_first_transparent
                )

        # Update color indices for the selected indices
        color_indices = self._active_artist.color_indices
        
        # Handle the case where color_indices is None
        if color_indices is None:
            # Initialize color_indices with zeros if it's None
            # Get data length from the active artist
            if hasattr(self._active_artist, 'data') and self._active_artist.data is not None:
                data_length = len(self._active_artist.data)
                color_indices = np.zeros(data_length, dtype=np.int8)
                self._active_artist.color_indices = color_indices
            else:
                # If we can't determine data length, we can't proceed
                self._selected_indices = None
                return
        
        # Now safely assign the selection values
        color_indices[self._selected_indices] = self._class_value
        self._active_artist.color_indices = color_indices

        # Emit signal and reset selected indices
        self.selection_applied_signal.emit(color_indices)
        self._selected_indices = None
        # Remove selector and create a new one
        self.remove()
        self.create_selector()