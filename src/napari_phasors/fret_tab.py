import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from napari.layers import Image
from napari.utils.notifications import show_error
from phasorpy.lifetime import phasor_from_fret_donor
from phasorpy.phasor import phasor_center, phasor_nearest_neighbor
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ._utils import update_frequency_in_metadata


class FretWidget(QWidget):
    """Widget to perform FLIM FRET analysis."""

    def __init__(self, viewer, parent=None):
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent
        self.frequency = 80.0  # Default frequency in MHz
        self.donor_lifetime = 2.0  # Default donor lifetime in ns
        self._fret_efficiencies = np.linspace(0, 1, 500)
        self.current_donor_line = None
        self.fret_layer = None
        self.colormap_contrast_limits = None
        self.fret_colormap = None
        self.use_colormap = True
        self.colormap_density_factor = (
            5  # Controls trajectory colormap detail level
        )

        # Initialize parameters
        self.donor_background = 0.1
        self.background_real = 0.1
        self.background_imag = 0.1
        self.donor_fretting_proportion = 1.0

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface for the FRET widget with a scroll area."""
        from qtpy.QtWidgets import QScrollArea

        # Create the scroll area and the content widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # Donor lifetime and frequency
        self.donor_line_edit = QLineEdit()
        self.donor_line_edit.setPlaceholderText("Donor Lifetime (ns)")
        self.donor_line_edit.setValidator(QDoubleValidator())
        self.donor_line_edit.textChanged.connect(self._on_parameters_changed)
        layout.addWidget(QLabel("Donor Lifetime (ns):"))
        layout.addWidget(self.donor_line_edit)

        self.frequency_input = QLineEdit()
        self.frequency_input.setPlaceholderText("Frequency (MHz)")
        self.frequency_input.setValidator(QDoubleValidator())
        self.frequency_input.textChanged.connect(self._on_parameters_changed)
        layout.addWidget(QLabel("Frequency (MHz):"))
        layout.addWidget(self.frequency_input)

        # Background slider (0 to 1)
        layout.addWidget(QLabel("Donor Background:"))
        self.background_slider = QSlider(Qt.Horizontal)
        self.background_slider.setMinimum(0)
        self.background_slider.setMaximum(100)
        self.background_slider.setValue(10)  # 0.1 default
        self.background_slider.valueChanged.connect(
            self._on_background_slider_changed
        )

        self.background_label = QLabel("0.10")
        background_layout = QHBoxLayout()
        background_layout.addWidget(self.background_slider)
        background_layout.addWidget(self.background_label)
        layout.addLayout(background_layout)

        # Background position line edits
        layout.addWidget(QLabel("Background Position:"))
        bg_pos_layout = QHBoxLayout()

        bg_pos_layout.addWidget(QLabel("Real:"))
        self.background_real_edit = QLineEdit()
        self.background_real_edit.setPlaceholderText("0.1")
        self.background_real_edit.setValidator(QDoubleValidator())
        self.background_real_edit.setText("0.1")
        self.background_real_edit.textChanged.connect(
            self._on_parameters_changed
        )
        bg_pos_layout.addWidget(self.background_real_edit)

        bg_pos_layout.addWidget(QLabel("Imag:"))
        self.background_imag_edit = QLineEdit()
        self.background_imag_edit.setPlaceholderText("0.1")
        self.background_imag_edit.setValidator(QDoubleValidator())
        self.background_imag_edit.setText("0.1")
        self.background_imag_edit.textChanged.connect(
            self._on_parameters_changed
        )
        bg_pos_layout.addWidget(self.background_imag_edit)

        self.calculate_bg_button = QPushButton("Get from image")
        self.calculate_bg_button.clicked.connect(
            self._calculate_background_position
        )
        bg_pos_layout.addWidget(self.calculate_bg_button)

        layout.addLayout(bg_pos_layout)

        # Donor fretting proportion slider (0 to 1)
        layout.addWidget(QLabel("Proportion of Donors Fretting:"))
        self.fretting_slider = QSlider(Qt.Horizontal)
        self.fretting_slider.setMinimum(0)
        self.fretting_slider.setMaximum(100)
        self.fretting_slider.setValue(100)  # 1.0 default
        self.fretting_slider.valueChanged.connect(
            self._on_fretting_slider_changed
        )

        self.fretting_label = QLabel("1.00")
        fretting_layout = QHBoxLayout()
        fretting_layout.addWidget(self.fretting_slider)
        fretting_layout.addWidget(self.fretting_label)
        layout.addLayout(fretting_layout)

        # Colormap checkbox
        self.colormap_checkbox = QCheckBox(
            "Overlay colormap on donor trajectory"
        )
        self.colormap_checkbox.setChecked(True)  # Default checked
        self.colormap_checkbox.stateChanged.connect(
            self._on_colormap_checkbox_changed
        )
        layout.addWidget(self.colormap_checkbox)

        # Plot button
        self.calculate_fret_efficiency_button = QPushButton(
            "Calculate FRET efficiency"
        )
        self.calculate_fret_efficiency_button.clicked.connect(
            self.calculate_fret_efficiency
        )
        layout.addWidget(self.calculate_fret_efficiency_button)

        layout.addStretch()

        # Set the content widget as the scroll area's widget
        scroll_area.setWidget(content_widget)

        # Set the scroll area as the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)

    def get_all_artists(self):
        """Return a list of all matplotlib artists created by this widget."""
        artists = []
        if self.current_donor_line is not None:
            artists.append(self.current_donor_line)
        return artists

    def set_artists_visible(self, visible):
        """Set visibility of all artists created by this widget."""
        for artist in self.get_all_artists():
            if hasattr(artist, 'set_visible'):
                artist.set_visible(visible)

    def _on_background_slider_changed(self):
        """Handle background slider value change."""
        value = self.background_slider.value() / 100.0
        self.donor_background = value
        self.background_label.setText(f"{value:.2f}")
        self._on_parameters_changed()

    def _on_fretting_slider_changed(self):
        """Handle fretting proportion slider value change."""
        value = self.fretting_slider.value() / 100.0
        self.donor_fretting_proportion = value
        self.fretting_label.setText(f"{value:.2f}")
        self._on_parameters_changed()

    def _on_colormap_checkbox_changed(self):
        """Handle colormap checkbox state change."""
        self.use_colormap = self.colormap_checkbox.isChecked()
        self.plot_donor_trajectory()

    def _on_parameters_changed(self):
        """Update plot when any parameter changes."""
        if (
            self.donor_line_edit.text()
            and self.frequency_input.text()
            and self.background_real_edit.text()
            and self.background_imag_edit.text()
        ):
            try:
                self.frequency = (
                    float(self.frequency_input.text().strip())
                    * self.parent_widget.harmonic
                )
                self.donor_lifetime = float(self.donor_line_edit.text())
                self.background_real = float(self.background_real_edit.text())
                self.background_imag = float(self.background_imag_edit.text())
                self.plot_donor_trajectory()
            except ValueError:
                pass

    def _calculate_background_position(self):
        """Calculate the background position from image layer."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        labels_layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        if not labels_layer_name:
            return

        phasor_data = (
            self.parent_widget._labels_layer_with_phasor_features.features
        )
        harmonic_mask = phasor_data['harmonic'] == self.parent_widget.harmonic
        real = phasor_data.loc[harmonic_mask, 'G']
        imag = phasor_data.loc[harmonic_mask, 'S']
        mean = np.empty_like(real)

        _, center_real, center_imag = phasor_center(mean, real, imag)

        self.background_real_edit.setText(f"{center_real:.2f}")
        self.background_imag_edit.setText(f"{center_imag:.2f}")

    def plot_donor_trajectory(self):
        """Plot the donor trajectory with current parameters."""
        try:
            if self.current_donor_line is not None:
                try:
                    self.current_donor_line.remove()
                except ValueError:
                    pass
                self.current_donor_line = None

            if not (
                self.donor_line_edit.text() and self.frequency_input.text()
            ):
                return

            self.frequency = (
                float(self.frequency_input.text().strip())
                * self.parent_widget.harmonic
            )
            self.donor_lifetime = float(self.donor_line_edit.text().strip())
            self.background_real = float(
                self.background_real_edit.text().strip()
            )
            self.background_imag = float(
                self.background_imag_edit.text().strip()
            )

            # Calculate donor trajectory
            donor_trajectory_real, donor_trajectory_imag = (
                phasor_from_fret_donor(
                    self.frequency,
                    self.donor_lifetime,
                    fret_efficiency=self._fret_efficiencies,
                    donor_background=self.donor_background,
                    background_imag=self.background_imag,
                    background_real=self.background_real,
                    donor_fretting=self.donor_fretting_proportion,
                )
            )

            # Get the current axes
            ax = self.parent_widget.canvas_widget.figure.gca()

            if self.fret_layer is not None and self.use_colormap:
                # Create colormap trajectory
                self._draw_colormap_trajectory(
                    ax, donor_trajectory_real, donor_trajectory_imag
                )
            else:
                # Plot simple green line
                self.current_donor_line = ax.plot(
                    donor_trajectory_real,
                    donor_trajectory_imag,
                    color='green',
                    linewidth=2,
                    label='Donor Trajectory',
                )[0]

            # Refresh the canvas
            self.parent_widget.canvas_widget.canvas.draw_idle()

        except Exception as e:
            show_error(f"Error drawing line: {str(e)}")

    def _draw_colormap_trajectory(self, ax, trajectory_real, trajectory_imag):
        """Draw a colormap trajectory line."""
        num_segments = min(
            len(trajectory_real) * self.colormap_density_factor,
            len(trajectory_real) - 1,
        )  # Number of color segments

        # Get colormap from stored colors or fallback
        if hasattr(self, 'fret_colormap') and self.fret_colormap is not None:
            colormap = ListedColormap(self.fret_colormap)
        else:
            colormap = plt.cm.turbo  # Use turbo as fallback

        # Get the actual contrast limits from the FRET layer
        if (
            hasattr(self, 'colormap_contrast_limits')
            and self.colormap_contrast_limits is not None
        ):
            vmin, vmax = self.colormap_contrast_limits
        elif self.fret_layer is not None:
            vmin, vmax = self.fret_layer.contrast_limits
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
            segment = [
                (trajectory_real[start_idx], trajectory_imag[start_idx]),
                (trajectory_real[end_idx], trajectory_imag[end_idx]),
            ]
            segments.append(segment)

            # FRET efficiency value for this segment (0 to 1)
            fret_value = self._fret_efficiencies[start_idx]
            colors.append(fret_value)

        # Create line collection
        lc = LineCollection(segments, cmap=colormap, linewidths=3)
        lc.set_array(np.array(colors))
        lc.set_clim(vmin, vmax)

        # Add to axes
        self.current_donor_line = ax.add_collection(lc)

    def _on_colormap_changed(self, event):
        """Handle changes to the colormap of the FRET layer."""
        if self.fret_layer is not None:
            layer = event.source
            self.fret_colormap = layer.colormap.colors
            self.colormap_contrast_limits = layer.contrast_limits

            # Redraw the donor trajectory with updated colormap
            self.plot_donor_trajectory()

    def _on_contrast_limits_changed(self, event):
        """Handle changes to the contrast limits of the FRET layer."""
        if self.fret_layer is not None:
            layer = event.source
            self.colormap_contrast_limits = layer.contrast_limits

            # Redraw the donor trajectory with updated contrast limits
            self.plot_donor_trajectory()

    def calculate_fret_efficiency(self):
        """Calculate FRET efficiency based on donor intensities."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        labels_layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        if not labels_layer_name:
            return

        sample_layer = self.viewer.layers[labels_layer_name]
        phasor_data = (
            self.parent_widget._labels_layer_with_phasor_features.features
        )
        harmonic_mask = phasor_data['harmonic'] == self.parent_widget.harmonic
        real = phasor_data.loc[harmonic_mask, 'G']
        imag = phasor_data.loc[harmonic_mask, 'S']

        # Get trajectory data depending on the type of line object
        if hasattr(self.current_donor_line, 'get_xydata'):
            # Regular line plot
            neighbor_real, neighbor_imag = (
                self.current_donor_line.get_xydata().T
            )
        else:
            # LineCollection - use the original trajectory data
            donor_trajectory_real, donor_trajectory_imag = (
                phasor_from_fret_donor(
                    self.frequency,
                    self.donor_lifetime,
                    fret_efficiency=self._fret_efficiencies,
                    donor_background=self.donor_background,
                    background_imag=self.background_imag,
                    background_real=self.background_real,
                    donor_fretting=self.donor_fretting_proportion,
                )
            )
            neighbor_real, neighbor_imag = (
                donor_trajectory_real,
                donor_trajectory_imag,
            )

        fret_efficiency = phasor_nearest_neighbor(
            np.array(real),
            np.array(imag),
            neighbor_real,
            neighbor_imag,
            values=self._fret_efficiencies,
        )
        fret_efficiency = fret_efficiency.reshape(
            self.parent_widget._labels_layer_with_phasor_features.data.shape
        )

        fret_layer_name = f"FRET efficiency: {labels_layer_name}"
        selected_fret_layer = Image(
            fret_efficiency,
            name=fret_layer_name,
            scale=self.parent_widget._labels_layer_with_phasor_features.scale,
            colormap='plasma',
            contrast_limits=(0, 1),  # Force contrast limits to 0-1
        )

        # Check if the layer is in the viewer before attempting to remove it
        if fret_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[fret_layer_name])

        self.fret_layer = self.viewer.add_layer(selected_fret_layer)
        self.fret_colormap = self.fret_layer.colormap.colors
        self.fret_layer.events.colormap.connect(self._on_colormap_changed)
        self.colormap_contrast_limits = (0, 1)  # Set to 0-1 by default
        self.fret_layer.events.contrast_limits.connect(
            self._on_contrast_limits_changed
        )

        # Redraw the trajectory with the new colormap
        self.plot_donor_trajectory()

        update_frequency_in_metadata(
            sample_layer,
            self.frequency_input.text().strip(),
        )
