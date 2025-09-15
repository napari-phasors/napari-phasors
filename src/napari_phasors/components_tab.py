from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_error
from phasorpy.component import phasor_component_fraction
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari


class ComponentsWidget(QWidget):
    """Widget to perform component analysis on phasor coordinates."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent

        # Initialize dot and line references
        self.component1_dot = None
        self.component2_dot = None
        self.component1_text = None  # Text annotation for component 1
        self.component2_text = None  # Text annotation for component 2
        self.component_line = None
        self.fractions_layer = None
        self.fractions_colormap = None
        self.colormap_contrast_limits = None

        # Dragging state
        self.dragging_component = None
        self.press_event = None
        self.drag_events_connected = False

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface for the components widget."""
        layout = QVBoxLayout()

        # Component 1 section
        comp1_layout = QHBoxLayout()
        comp1_layout.addWidget(QLabel("Component 1:"))
        self.comp1_name_edit = QLineEdit()
        self.comp1_name_edit.setPlaceholderText("Component name (optional)")
        self.comp1_name_edit.textChanged.connect(
            self._on_component1_name_changed
        )
        comp1_layout.addWidget(self.comp1_name_edit)
        self.my_button = QPushButton("Select Component 1")
        self.my_button.clicked.connect(self.on_first_button_clicked)
        comp1_layout.addWidget(self.my_button)

        # Component 1 coordinates
        coord1_layout = QHBoxLayout()
        coord1_layout.addWidget(QLabel("G:"))
        self.first_edit1 = QLineEdit()
        self.first_edit1.setPlaceholderText("Real coordinate")
        self.first_edit1.editingFinished.connect(
            self._on_component1_coords_changed
        )
        coord1_layout.addWidget(self.first_edit1)
        coord1_layout.addWidget(QLabel("S:"))
        self.first_edit2 = QLineEdit()
        self.first_edit2.setPlaceholderText("Imaginary coordinate")
        self.first_edit2.editingFinished.connect(
            self._on_component1_coords_changed
        )
        coord1_layout.addWidget(self.first_edit2)

        # Component 2 section
        comp2_layout = QHBoxLayout()
        comp2_layout.addWidget(QLabel("Component 2:"))
        self.comp2_name_edit = QLineEdit()
        self.comp2_name_edit.setPlaceholderText("Component name (optional)")
        self.comp2_name_edit.textChanged.connect(
            self._on_component2_name_changed
        )
        comp2_layout.addWidget(self.comp2_name_edit)
        self.second_button = QPushButton("Select Component 2")
        self.second_button.clicked.connect(self.on_second_button_clicked)
        comp2_layout.addWidget(self.second_button)

        # Component 2 coordinates
        coord2_layout = QHBoxLayout()
        coord2_layout.addWidget(QLabel("G:"))
        self.second_edit1 = QLineEdit()
        self.second_edit1.setPlaceholderText("Real coordinate")
        self.second_edit1.editingFinished.connect(
            self._on_component2_coords_changed
        )
        coord2_layout.addWidget(self.second_edit1)
        coord2_layout.addWidget(QLabel("S:"))
        self.second_edit2 = QLineEdit()
        self.second_edit2.setPlaceholderText("Imaginary coordinate")
        self.second_edit2.editingFinished.connect(
            self._on_component2_coords_changed
        )
        coord2_layout.addWidget(self.second_edit2)

        # Calculate button
        self.calculate_button = QPushButton("Analyze")
        self.calculate_button.clicked.connect(self.on_calculate_button_clicked)

        # Add all layouts to main layout
        layout.addLayout(comp1_layout)
        layout.addLayout(coord1_layout)
        layout.addLayout(comp2_layout)
        layout.addLayout(coord2_layout)
        layout.addWidget(self.calculate_button)
        layout.addStretch()  # Add stretch to push everything to top

        self.setLayout(layout)

    def get_all_artists(self):
        """Return a list of all matplotlib artists created by this widget."""
        artists = []
        if self.component1_dot is not None:
            artists.append(self.component1_dot)
        if self.component2_dot is not None:
            artists.append(self.component2_dot)
        if self.component1_text is not None:
            artists.append(self.component1_text)
        if self.component2_text is not None:
            artists.append(self.component2_text)
        if self.component_line is not None:
            artists.append(self.component_line)
        return artists

    def set_artists_visible(self, visible):
        """Set visibility of all artists created by this widget."""
        for artist in self.get_all_artists():
            if hasattr(artist, 'set_visible'):
                artist.set_visible(visible)

    def _on_component1_coords_changed(self):
        """Handle changes to component 1 coordinates in line edits."""
        # Allow updates even if component doesn't exist yet
        try:
            x = float(self.first_edit1.text())
            y = float(self.first_edit2.text())

            if self.component1_dot is not None:
                # Update existing component position
                self.component1_dot.set_data([x], [y])

                # Update text position if it exists
                if self.component1_text is not None:
                    self.component1_text.set_position((x, y))

                # Redraw line between components
                self.draw_line_between_components()
            else:
                # Create new component if coordinates are valid
                self._create_component1_at_coordinates(x, y)

        except ValueError:
            # Invalid input, ignore
            pass

    def _on_component2_coords_changed(self):
        """Handle changes to component 2 coordinates in line edits."""
        # Allow updates even if component doesn't exist yet
        try:
            x = float(self.second_edit1.text())
            y = float(self.second_edit2.text())

            if self.component2_dot is not None:
                # Update existing component position
                self.component2_dot.set_data([x], [y])

                # Update text position if it exists
                if self.component2_text is not None:
                    self.component2_text.set_position((x, y))

                # Redraw line between components
                self.draw_line_between_components()
            else:
                # Create new component if coordinates are valid
                self._create_component2_at_coordinates(x, y)

        except ValueError:
            # Invalid input, ignore
            pass

    def _create_component1_at_coordinates(self, x, y):
        """Create component 1 at the specified coordinates."""
        if self.parent_widget is None:
            return

        # Get component color
        component1_color, _ = self._get_component_colors()

        # Plot the dot on the canvas
        ax = self.parent_widget.canvas_widget.figure.gca()
        self.component1_dot = ax.plot(
            x,
            y,
            'o',
            color=component1_color,
            markersize=8,
            label='Component 1',
        )[0]

        # Add text if name is provided
        name = self.comp1_name_edit.text().strip()
        if name:
            self.component1_text = ax.annotate(
                name,
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                color='black',  # Remove bbox outline
            )

        # Make components draggable
        self._make_components_draggable()

        # Refresh the canvas
        self.parent_widget.canvas_widget.canvas.draw_idle()

        # Draw line if both components exist
        self.draw_line_between_components()

    def _create_component2_at_coordinates(self, x, y):
        """Create component 2 at the specified coordinates."""
        if self.parent_widget is None:
            return

        # Get component color
        _, component2_color = self._get_component_colors()

        # Plot the dot on the canvas
        ax = self.parent_widget.canvas_widget.figure.gca()
        self.component2_dot = ax.plot(
            x,
            y,
            'o',
            color=component2_color,
            markersize=8,
            label='Component 2',
        )[0]

        # Add text if name is provided
        name = self.comp2_name_edit.text().strip()
        if name:
            self.component2_text = ax.annotate(
                name,
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                color='black',  # Remove bbox outline
            )

        # Make components draggable
        self._make_components_draggable()

        # Refresh the canvas
        self.parent_widget.canvas_widget.canvas.draw_idle()

        # Draw line if both components exist
        self.draw_line_between_components()

    def _on_component1_name_changed(self):
        """Handle changes to component 1 name."""
        if self.component1_dot is None:
            return

        name = self.comp1_name_edit.text().strip()

        # Remove existing text if it exists
        if self.component1_text is not None:
            self.component1_text.remove()
            self.component1_text = None

        # Add new text if name is not empty
        if name:
            x, y = self.component1_dot.get_data()
            ax = self.parent_widget.canvas_widget.figure.gca()
            self.component1_text = ax.annotate(
                name,
                (x[0], y[0]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                color='black',
            )

        # Refresh the canvas
        self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_component2_name_changed(self):
        """Handle changes to component 2 name."""
        if self.component2_dot is None:
            return

        name = self.comp2_name_edit.text().strip()

        # Remove existing text if it exists
        if self.component2_text is not None:
            self.component2_text.remove()
            self.component2_text = None

        # Add new text if name is not empty
        if name:
            x, y = self.component2_dot.get_data()
            ax = self.parent_widget.canvas_widget.figure.gca()
            self.component2_text = ax.annotate(
                name,
                (x[0], y[0]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                color='black',  # Remove bbox outline
            )

        # Refresh the canvas
        self.parent_widget.canvas_widget.canvas.draw_idle()

    def _get_component_colors(self):
        """Get colors for components based on the colormap ends."""
        if (
            hasattr(self, 'fractions_colormap')
            and self.fractions_colormap is not None
        ):
            # Get contrast limits
            if (
                hasattr(self, 'colormap_contrast_limits')
                and self.colormap_contrast_limits is not None
            ):
                vmin, vmax = self.colormap_contrast_limits
            else:
                vmin, vmax = 0, 1

            # Component 1 represents fraction = 1.0 (pure component 1)
            # Component 2 represents fraction = 0.0 (pure component 2)
            # Map these values to colormap indices
            if vmax > vmin:
                # Normalize the fraction values to colormap indices
                component1_idx = int(
                    ((1.0 - vmin) / (vmax - vmin))
                    * (len(self.fractions_colormap) - 1)
                )
                component2_idx = int(
                    ((0.0 - vmin) / (vmax - vmin))
                    * (len(self.fractions_colormap) - 1)
                )

                # Clamp indices to valid range
                component1_idx = max(
                    0, min(len(self.fractions_colormap) - 1, component1_idx)
                )
                component2_idx = max(
                    0, min(len(self.fractions_colormap) - 1, component2_idx)
                )

                component1_color = self.fractions_colormap[component1_idx]
                component2_color = self.fractions_colormap[component2_idx]
            else:
                # Fallback if vmax <= vmin
                component1_color = self.fractions_colormap[-1]
                component2_color = self.fractions_colormap[0]

            return component1_color, component2_color
        else:
            # Default colors if no colormap is available
            return 'red', 'blue'

    def _update_component_colors(self):
        """Update the colors of the component dots to match colormap ends."""
        component1_color, component2_color = self._get_component_colors()

        if self.component1_dot is not None:
            self.component1_dot.set_color(component1_color)

        if self.component2_dot is not None:
            self.component2_dot.set_color(component2_color)

        # Refresh the canvas
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def draw_line_between_components(self):
        """Draw a line between component 1 and component 2 if both exist."""
        if self.component1_dot is None or self.component2_dot is None:
            return

        # Remove existing line if it exists
        if self.component_line is not None:
            try:
                self.component_line.remove()
            except (ValueError, AttributeError):
                # Handle case where line was already removed
                pass
            self.component_line = None

        try:
            # Get coordinates from the dots
            x1, y1 = self.component1_dot.get_data()
            x2, y2 = self.component2_dot.get_data()

            # Plot a simple line if no fractions layer exists yet
            ax = self.parent_widget.canvas_widget.figure.gca()
            if self.fractions_layer is None:
                self.component_line = ax.plot(
                    [x1[0], x2[0]], [y1[0], y2[0]], 'k', linewidth=2, alpha=0.7
                )[0]
            else:
                # Create colormap bar between components
                self._draw_colormap_line(ax, x1[0], y1[0], x2[0], y2[0])
                # Update component colors to match colormap
                self._update_component_colors()

            # Refresh the canvas
            self.parent_widget.canvas_widget.canvas.draw_idle()

        except Exception as e:
            show_error(f"Error drawing line: {str(e)}")

    def _draw_colormap_line(self, ax, x1, y1, x2, y2):
        """Draw a colormap bar between two components."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap
        import numpy as np

        # Calculate line properties
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            return

        # Create points array for the line
        t_values = np.linspace(0, 1, 500)
        
        trajectory_real = x1 + t_values * dx
        trajectory_imag = y1 + t_values * dy

        # Higher density_factor = more segments = smoother appearance
        density_factor = 2  # Adjust this value to control detail level
        num_segments = min(
            len(trajectory_real) * density_factor, len(trajectory_real) - 1
        )  # Number of color segments

        # Get colormap from stored colors or fallback
        if (
            hasattr(self, 'fractions_colormap')
            and self.fractions_colormap is not None
        ):
            colormap = ListedColormap(self.fractions_colormap)
        else:
            colormap = plt.cm.plasma  # Use plasma as fallback

        # Get the actual contrast limits from the fractions layer
        if (
            hasattr(self, 'colormap_contrast_limits')
            and self.colormap_contrast_limits is not None
        ):
            vmin, vmax = self.colormap_contrast_limits
        elif self.fractions_layer is not None:
            vmin, vmax = self.fractions_layer.contrast_limits
        else:
            vmin, vmax = 0, 1

        # Create line segments
        segments = []
        colors = []

        for i in range(num_segments):
            # Get indices for this segment with overlap
            start_idx = int(i * (len(trajectory_real) - 1) / num_segments)
            end_idx = int((i + 1) * (len(trajectory_real) - 1) / num_segments)
            
            # Ensure we don't go out of bounds
            end_idx = min(end_idx, len(trajectory_real) - 1)
            
            # For segments after the first, start slightly before to overlap
            if i > 0:
                start_idx = max(0, start_idx - 1)

            # Create line segment
            segment = [(trajectory_real[start_idx], trajectory_imag[start_idx]),
                      (trajectory_real[end_idx], trajectory_imag[end_idx])]
            segments.append(segment)

            # Calculate fraction value for this segment
            # Component 1 (start) = fraction 1.0, Component 2 (end) = fraction 0.0
            t = start_idx / (len(trajectory_real) - 1) if len(trajectory_real) > 1 else 0
            fraction_value = 1.0 - t  # Linear from 1.0 to 0.0
            colors.append(fraction_value)

        # Create line collection
        lc = LineCollection(segments, cmap=colormap, linewidths=4)
        lc.set_array(np.array(colors))
        lc.set_clim(vmin, vmax)

        # Add to axes
        self.component_line = ax.add_collection(lc)

    def _on_colormap_changed(self, event):
        """Handle changes to the colormap of the fractions layer."""
        if (
            self.fractions_layer is not None
            and self.component_line is not None
        ):
            layer = event.source
            self.fractions_colormap = layer.colormap.colors
            self.colormap_contrast_limits = layer.contrast_limits

            # Redraw the colormap line with updated colormap
            self.draw_line_between_components()

    def _on_contrast_limits_changed(self, event):
        """Handle changes to the contrast limits of the fractions layer."""
        if (
            self.fractions_layer is not None
            and self.component_line is not None
        ):
            layer = event.source
            self.colormap_contrast_limits = layer.contrast_limits

            # Redraw the colormap line with updated contrast limits
            self.draw_line_between_components()

    def _make_components_draggable(self):
        """Make component dots draggable."""
        if self.parent_widget is None or self.drag_events_connected:
            return

        # Connect drag events
        self.parent_widget.canvas_widget.canvas.mpl_connect(
            'button_press_event', self._on_press
        )
        self.parent_widget.canvas_widget.canvas.mpl_connect(
            'button_release_event', self._on_release
        )
        self.parent_widget.canvas_widget.canvas.mpl_connect(
            'motion_notify_event', self._on_motion
        )
        self.drag_events_connected = True

    def _on_press(self, event):
        """Handle mouse press for dragging components."""
        if event.inaxes is None:
            return

        # Check if we clicked on a component dot
        if (
            self.component1_dot is not None
            and self.component1_dot.contains(event)[0]
        ):
            self.dragging_component = 1
            self.press_event = event
        elif (
            self.component2_dot is not None
            and self.component2_dot.contains(event)[0]
        ):
            self.dragging_component = 2
            self.press_event = event

    def _on_motion(self, event):
        """Handle mouse motion for dragging components."""
        if self.dragging_component is None or event.inaxes is None:
            return

        # Update component position
        x, y = event.xdata, event.ydata

        if self.dragging_component == 1 and self.component1_dot is not None:
            self.component1_dot.set_data([x], [y])
            self.first_edit1.setText(f"{x:.6f}")
            self.first_edit2.setText(f"{y:.6f}")
            # Update text position if it exists
            self._on_component1_name_changed()
        elif self.dragging_component == 2 and self.component2_dot is not None:
            self.component2_dot.set_data([x], [y])
            self.second_edit1.setText(f"{x:.6f}")
            self.second_edit2.setText(f"{y:.6f}")
            # Update text position if it exists
            self._on_component2_name_changed()

        # Redraw line between components
        self.draw_line_between_components()

    def _on_release(self, event):
        """Handle mouse release for dragging components."""
        self.dragging_component = None
        self.press_event = None

    def on_first_button_clicked(self):
        """Function called when the first button is clicked."""
        # Check if parent widget exists
        if self.parent_widget is None:
            show_error("Parent widget not available")
            return

        # Remove existing dot and text if they exist
        if self.component1_dot is not None:
            self.component1_dot.remove()
            self.component1_dot = None
        if self.component1_text is not None:
            self.component1_text.remove()
            self.component1_text = None
        # Remove line as well since component 1 changed
        if self.component_line is not None:
            try:
                self.component_line.remove()
            except (ValueError, AttributeError):
                pass
            self.component_line = None
        self.parent_widget.canvas_widget.canvas.draw_idle()

        # Change button text to indicate waiting state
        original_text = self.my_button.text()
        self.my_button.setText("Click on plot...")
        self.my_button.setEnabled(False)

        # Disconnect original click handler
        if hasattr(self.parent_widget, 'click_cid'):
            self.parent_widget.canvas_widget.canvas.mpl_disconnect(
                self.parent_widget.click_cid
            )

        # Create a new click handler for component 1
        def handle_first_component_click(event):
            if not event.inaxes:
                return

            # Get coordinates
            x, y = event.xdata, event.ydata

            try:
                # Update the first component line edits
                self.first_edit1.setText(f"{x:.6f}")
                self.first_edit2.setText(f"{y:.6f}")

                # Get component color
                component1_color, _ = self._get_component_colors()

                # Plot the dot on the canvas
                ax = self.parent_widget.canvas_widget.figure.gca()
                self.component1_dot = ax.plot(
                    x,
                    y,
                    'o',
                    color=component1_color,
                    markersize=8,
                    label='Component 1',
                )[0]

                # Add text if name is provided
                name = self.comp1_name_edit.text().strip()
                if name:
                    self.component1_text = ax.annotate(
                        name,
                        (x, y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=10,
                        color='black',  # Remove bbox outline
                    )

                # Make components draggable immediately
                self._make_components_draggable()

                # Refresh the canvas to show the dot immediately
                self.parent_widget.canvas_widget.canvas.draw_idle()

                # Draw line if both components exist
                self.draw_line_between_components()

            except Exception as e:
                show_error(f"Error setting coordinates: {str(e)}")
            finally:
                # Disconnect temporary handler
                self.parent_widget.canvas_widget.canvas.mpl_disconnect(
                    self.temp_click_cid
                )

                # Restore button state
                self.my_button.setText(original_text)
                self.my_button.setEnabled(True)

        # Connect temporary click handler
        self.temp_click_cid = (
            self.parent_widget.canvas_widget.canvas.mpl_connect(
                'button_press_event', handle_first_component_click
            )
        )

    def on_second_button_clicked(self):
        """Function called when the second button is clicked."""
        # Check if parent widget exists
        if self.parent_widget is None:
            show_error("Parent widget not available")
            return

        # Remove existing dot and text if they exist
        if self.component2_dot is not None:
            self.component2_dot.remove()
            self.component2_dot = None
        if self.component2_text is not None:
            self.component2_text.remove()
            self.component2_text = None
        # Remove line as well since component 2 changed
        if self.component_line is not None:
            try:
                self.component_line.remove()
            except (ValueError, AttributeError):
                pass
            self.component_line = None
        self.parent_widget.canvas_widget.canvas.draw_idle()

        # Change button text to indicate waiting state
        original_text = self.second_button.text()
        self.second_button.setText("Click on plot...")
        self.second_button.setEnabled(False)

        # Disconnect original click handler
        if hasattr(self.parent_widget, 'click_cid'):
            self.parent_widget.canvas_widget.canvas.mpl_disconnect(
                self.parent_widget.click_cid
            )

        # Create a new click handler for component 2
        def handle_second_component_click(event):
            if not event.inaxes:
                return

            # Get coordinates
            x, y = event.xdata, event.ydata

            try:
                # Update the second component line edits
                self.second_edit1.setText(f"{x:.6f}")
                self.second_edit2.setText(f"{y:.6f}")

                # Get component color
                _, component2_color = self._get_component_colors()

                # Plot the dot on the canvas
                ax = self.parent_widget.canvas_widget.figure.gca()
                self.component2_dot = ax.plot(
                    x,
                    y,
                    'o',
                    color=component2_color,
                    markersize=8,
                    label='Component 2',
                )[0]

                # Add text if name is provided
                name = self.comp2_name_edit.text().strip()
                if name:
                    self.component2_text = ax.annotate(
                        name,
                        (x, y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=10,
                        color='black',  # Remove bbox outline
                    )

                # Make components draggable immediately
                self._make_components_draggable()

                # Refresh the canvas to show the dot immediately
                self.parent_widget.canvas_widget.canvas.draw_idle()

                # Draw line if both components exist
                self.draw_line_between_components()

            except Exception as e:
                show_error(f"Error setting coordinates: {str(e)}")
            finally:
                # Disconnect temporary handler
                self.parent_widget.canvas_widget.canvas.mpl_disconnect(
                    self.temp_click_cid
                )

                # Restore button state
                self.second_button.setText(original_text)
                self.second_button.setEnabled(True)

        # Connect temporary click handler
        self.temp_click_cid = (
            self.parent_widget.canvas_widget.canvas.mpl_connect(
                'button_press_event', handle_second_component_click
            )
        )

    def on_calculate_button_clicked(self):
        """Function called when the calculate button is clicked."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        if self.component1_dot is None or self.component2_dot is None:
            return

        component_real = (
            self.component1_dot.get_data()[0][0],
            self.component2_dot.get_data()[0][0],
        )
        component_imag = (
            self.component1_dot.get_data()[1][0],
            self.component2_dot.get_data()[1][0],
        )

        phasor_data = (
            self.parent_widget._labels_layer_with_phasor_features.features
        )
        harmonic_mask = phasor_data['harmonic'] == self.parent_widget.harmonic
        real = phasor_data.loc[harmonic_mask, 'G']
        imag = phasor_data.loc[harmonic_mask, 'S']

        fractions = phasor_component_fraction(
            np.array(real), np.array(imag), component_real, component_imag
        )
        fractions = fractions.reshape(
            self.parent_widget._labels_layer_with_phasor_features.data.shape
        )

        fractions_layer_name = f"Component 1 fractions: {self.parent_widget.image_layer_with_phasor_features_combobox.currentText()}"
        selected_fractions_layer = Image(
            fractions,
            name=fractions_layer_name,
            scale=self.parent_widget._labels_layer_with_phasor_features.scale,
            colormap='plasma',
            contrast_limits=(0, 1),
        )

        # Check if the layer is in the viewer before attempting to remove it
        if fractions_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[fractions_layer_name])

        self.fractions_layer = self.viewer.add_layer(selected_fractions_layer)
        self.fractions_colormap = self.fractions_layer.colormap.colors
        self.colormap_contrast_limits = self.fractions_layer.contrast_limits

        # Connect to both colormap and contrast limits events
        self.fractions_layer.events.colormap.connect(self._on_colormap_changed)
        self.fractions_layer.events.contrast_limits.connect(
            self._on_contrast_limits_changed
        )

        # Redraw the line with colormap after creating the fractions layer
        self.draw_line_between_components()
