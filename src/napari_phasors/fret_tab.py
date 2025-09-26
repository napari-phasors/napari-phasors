import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from napari.layers import Image
from napari.utils.notifications import show_error
from phasorpy.lifetime import phasor_from_fret_donor
from phasorpy.phasor import phasor_center, phasor_nearest_neighbor
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ._utils import update_frequency_in_metadata


class FretWidget(QWidget):
    """Widget to perform FLIM FRET analysis."""

    def __init__(self, viewer, parent=None):
        """Initialize the FretWidget."""
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent
        self.frequency = 80.0
        self.donor_lifetime = 2.0
        self._fret_efficiencies = np.linspace(0, 1, 500)
        self.current_donor_line = None
        self.fret_layer = None
        self.colormap_contrast_limits = None
        self.fret_colormap = None
        self.use_colormap = True
        self.colormap_density_factor = (
            5  # Controls trajectory colormap detail level
        )
        self.current_donor_circle = None
        self.current_background_circle = None

        # Initialize parameters
        self.donor_background = 0.1
        self.background_real = 0.0
        self.background_imag = 0.0
        self.donor_fretting_proportion = 1.0

        # Store background positions for different harmonics
        self.background_positions_by_harmonic = {}

        # Track current harmonic
        self.current_harmonic = 1

        # Setup UI
        self.setup_ui()

        # Connect to layer events to update background combobox
        self.viewer.layers.events.inserted.connect(self._on_layer_changed)
        self.viewer.layers.events.removed.connect(self._on_layer_changed)

    def setup_ui(self):
        """Set up the user interface for the FRET widget with a scroll area."""

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

        bg_pos_layout.addWidget(QLabel("G:"))
        self.background_real_edit = QLineEdit()
        self.background_real_edit.setPlaceholderText("Real coordinate")
        self.background_real_edit.setValidator(QDoubleValidator())
        self.background_real_edit.setText("0.1")
        self.background_real_edit.textChanged.connect(
            self._on_background_position_changed
        )
        bg_pos_layout.addWidget(self.background_real_edit)

        bg_pos_layout.addWidget(QLabel("S:"))
        self.background_imag_edit = QLineEdit()
        self.background_imag_edit.setPlaceholderText("Imaginary coordinate")
        self.background_imag_edit.setValidator(QDoubleValidator())
        self.background_imag_edit.setText("0.1")
        self.background_imag_edit.textChanged.connect(
            self._on_background_position_changed
        )
        bg_pos_layout.addWidget(self.background_imag_edit)

        layout.addLayout(bg_pos_layout)

        # Background image selection
        bg_image_layout = QHBoxLayout()
        bg_image_layout.addWidget(
            QLabel("Calculate background position from image:")
        )
        self.background_image_combobox = QComboBox()
        self.background_image_combobox.currentIndexChanged.connect(
            self._calculate_background_position
        )
        bg_image_layout.addWidget(self.background_image_combobox)
        layout.addLayout(bg_image_layout)

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

        # Initialize background combobox
        self._update_background_combobox()

    def get_all_artists(self):
        """Return a list of all matplotlib artists created by this widget."""
        artists = []
        if self.current_donor_line is not None:
            artists.append(self.current_donor_line)
        if self.current_donor_circle is not None:
            artists.append(self.current_donor_circle)
        if self.current_background_circle is not None:
            artists.append(self.current_background_circle)
        return artists

    def set_artists_visible(self, visible):
        """Set visibility of all artists created by this widget."""
        for artist in self.get_all_artists():
            if hasattr(artist, 'set_visible'):
                artist.set_visible(visible)

    def _on_harmonic_changed(self):
        """Handle harmonic changes from the parent widget."""
        if self.parent_widget is None:
            return

        new_harmonic = self.parent_widget.harmonic

        if self.current_harmonic != new_harmonic:
            self._store_current_background_position()

        self.current_harmonic = new_harmonic

        self._load_background_position_for_harmonic(new_harmonic)

        self.plot_donor_trajectory()

    def _store_current_background_position(self):
        """Store the current background position for the current harmonic."""
        if (
            self.background_real_edit.text()
            and self.background_imag_edit.text()
        ):
            try:
                real = float(self.background_real_edit.text().strip())
                imag = float(self.background_imag_edit.text().strip())
                self.background_positions_by_harmonic[
                    self.current_harmonic
                ] = {'real': real, 'imag': imag}
            except ValueError:
                pass

    def _load_background_position_for_harmonic(self, harmonic):
        """Load background position for the specified harmonic."""
        if harmonic in self.background_positions_by_harmonic:
            stored = self.background_positions_by_harmonic[harmonic]
            self.background_real_edit.setText(f"{stored['real']:.3f}")
            self.background_imag_edit.setText(f"{stored['imag']:.3f}")
            self.background_real = stored['real']
            self.background_imag = stored['imag']
        else:
            self.background_real_edit.setText("0.0")
            self.background_imag_edit.setText("0.0")
            self.background_real = 0.0
            self.background_imag = 0.0

    def _on_background_position_changed(self):
        """Handle manual changes to background position fields."""
        self._store_current_background_position()
        self._on_parameters_changed()

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
                base_frequency = float(self.frequency_input.text().strip())
                self.frequency = base_frequency * self.parent_widget.harmonic
                self.donor_lifetime = float(self.donor_line_edit.text())
                self.background_real = float(self.background_real_edit.text())
                self.background_imag = float(self.background_imag_edit.text())
                self.plot_donor_trajectory()
            except ValueError:
                pass

    def _update_background_combobox(self):
        """Update the background image combobox with available layers."""
        current_text = self.background_image_combobox.currentText()
        self.background_image_combobox.blockSignals(True)
        self.background_image_combobox.clear()

        self.background_image_combobox.addItem("None")

        from napari.layers import Image

        layer_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and "phasor_features_labels_layer" in layer.metadata.keys()
        ]

        self.background_image_combobox.addItems(layer_names)

        if current_text in layer_names:
            index = self.background_image_combobox.findText(current_text)
            if index >= 0:
                self.background_image_combobox.setCurrentIndex(index)
        elif current_text == "None" or current_text == "":
            self.background_image_combobox.setCurrentIndex(0)

        self.background_image_combobox.blockSignals(False)

    def _on_layer_changed(self):
        """Handle when layers are added/removed to update background combobox."""
        self._update_background_combobox()

    def _calculate_background_position(self):
        """Calculate the background position from selected background image layer for all harmonics."""
        background_layer_name = self.background_image_combobox.currentText()
        if not background_layer_name or background_layer_name == "None":
            return

        try:
            background_layer = self.viewer.layers[background_layer_name]
            background_phasor_layer = background_layer.metadata[
                "phasor_features_labels_layer"
            ]
        except (KeyError, AttributeError):
            return

        phasor_data = background_phasor_layer.features

        if 'harmonic' not in phasor_data.columns:
            return

        unique_harmonics = phasor_data['harmonic'].unique()

        for harmonic in unique_harmonics:
            harmonic_mask = phasor_data['harmonic'] == harmonic
            real = phasor_data.loc[harmonic_mask, 'G']
            imag = phasor_data.loc[harmonic_mask, 'S']

            if len(real) == 0 or len(imag) == 0:
                continue

            try:
                _, center_real, center_imag = phasor_center(
                    background_layer.data.flatten(), real, imag
                )

                self.background_positions_by_harmonic[harmonic] = {
                    'real': center_real,
                    'imag': center_imag,
                }
            except Exception:
                continue

        current_harmonic = self.parent_widget.harmonic
        if current_harmonic in self.background_positions_by_harmonic:
            stored = self.background_positions_by_harmonic[current_harmonic]
            self.background_real_edit.setText(f"{stored['real']:.3f}")
            self.background_imag_edit.setText(f"{stored['imag']:.3f}")
            self.background_real = stored['real']
            self.background_imag = stored['imag']
        else:
            self.background_real_edit.setText("0.0")
            self.background_imag_edit.setText("0.0")
            self.background_real = 0.0
            self.background_imag = 0.0

        self.plot_donor_trajectory()

    def plot_donor_trajectory(self):
        """Plot the donor trajectory with current parameters."""
        try:
            if self.current_donor_line is not None:
                try:
                    self.current_donor_line.remove()
                except ValueError:
                    pass
                self.current_donor_line = None

            if self.current_donor_circle is not None:
                try:
                    self.current_donor_circle.remove()
                except ValueError:
                    pass
                self.current_donor_circle = None

            if self.current_background_circle is not None:
                try:
                    self.current_background_circle.remove()
                except ValueError:
                    pass
                self.current_background_circle = None

            if not (
                self.donor_line_edit.text() and self.frequency_input.text()
            ):
                return

            base_frequency = float(self.frequency_input.text().strip())
            self.frequency = base_frequency * self.parent_widget.harmonic
            self.donor_lifetime = float(self.donor_line_edit.text().strip())
            self.background_real = float(
                self.background_real_edit.text().strip()
            )
            self.background_imag = float(
                self.background_imag_edit.text().strip()
            )

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

            ax = self.parent_widget.canvas_widget.figure.gca()

            if self.fret_layer is not None and self.use_colormap:
                if (
                    hasattr(self, 'fret_colormap')
                    and self.fret_colormap is not None
                ):
                    colormap = LinearSegmentedColormap.from_list(
                        "fret_interp", self.fret_colormap, N=256
                    )
                else:
                    colormap = plt.cm.turbo

                donor_color = colormap(0.0)[:3]
                background_color = colormap(1.0)[:3]
            else:
                donor_color = 'dimgray'
                background_color = 'dimgray'

            trajectory_zorder = 10
            dot_zorder = 11

            if self.fret_layer is not None and self.use_colormap:
                self._draw_colormap_trajectory(
                    ax,
                    donor_trajectory_real,
                    donor_trajectory_imag,
                    zorder=trajectory_zorder,
                )
            else:
                self.current_donor_line = ax.plot(
                    donor_trajectory_real,
                    donor_trajectory_imag,
                    color='dimgray',
                    linewidth=3,
                    label='Donor Trajectory',
                    zorder=trajectory_zorder,
                )[0]

            circle_radius = 0.02

            donor_circle = plt.Circle(
                (donor_trajectory_real[0], donor_trajectory_imag[0]),
                circle_radius,
                fill=True,
                facecolor=donor_color,
                linewidth=1,
                zorder=dot_zorder,
            )
            self.current_donor_circle = ax.add_patch(donor_circle)

            background_circle = plt.Circle(
                (donor_trajectory_real[-1], donor_trajectory_imag[-1]),
                circle_radius,
                fill=True,
                facecolor=background_color,
                linewidth=1,
                zorder=dot_zorder,
            )
            self.current_background_circle = ax.add_patch(background_circle)

            self.parent_widget.canvas_widget.canvas.draw_idle()

        except Exception as e:
            show_error(f"Error drawing line: {str(e)}")

    def _draw_colormap_trajectory(
        self, ax, trajectory_real, trajectory_imag, zorder=10
    ):
        """Draw a colormap trajectory line."""
        num_segments = min(
            len(trajectory_real) * self.colormap_density_factor,
            len(trajectory_real) - 1,
        )

        if hasattr(self, 'fret_colormap') and self.fret_colormap is not None:
            colormap = LinearSegmentedColormap.from_list(
                "fret_interp", self.fret_colormap, N=256
            )
        else:
            colormap = plt.cm.turbo

        if (
            hasattr(self, 'colormap_contrast_limits')
            and self.colormap_contrast_limits is not None
        ):
            vmin, vmax = self.colormap_contrast_limits
        elif self.fret_layer is not None:
            vmin, vmax = self.fret_layer.contrast_limits
        else:
            vmin, vmax = 0, 1

        segments = []
        colors = []

        for i in range(num_segments):
            start_idx = int(i * (len(trajectory_real) - 1) / num_segments)
            end_idx = int((i + 1) * (len(trajectory_real) - 1) / num_segments)

            end_idx = min(end_idx, len(trajectory_real) - 1)

            if i > 0:
                start_idx = max(0, start_idx - 1)

            segment = [
                (trajectory_real[start_idx], trajectory_imag[start_idx]),
                (trajectory_real[end_idx], trajectory_imag[end_idx]),
            ]
            segments.append(segment)

            fret_value = self._fret_efficiencies[start_idx]
            colors.append(fret_value)

        lc = LineCollection(
            segments, cmap=colormap, linewidths=3, zorder=zorder
        )
        lc.set_array(np.array(colors))
        lc.set_clim(vmin, vmax)

        self.current_donor_line = ax.add_collection(lc)

    def _on_colormap_changed(self, event):
        """Handle changes to the colormap of the FRET layer."""
        if self.fret_layer is not None:
            layer = event.source
            self.fret_colormap = layer.colormap.colors
            self.colormap_contrast_limits = layer.contrast_limits

            self.plot_donor_trajectory()

    def _on_contrast_limits_changed(self, event):
        """Handle changes to the contrast limits of the FRET layer."""
        if self.fret_layer is not None:
            layer = event.source
            self.colormap_contrast_limits = layer.contrast_limits

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

        if hasattr(self.current_donor_line, 'get_xydata'):
            neighbor_real, neighbor_imag = (
                self.current_donor_line.get_xydata().T
            )
        else:
            base_frequency = float(self.frequency_input.text().strip())
            effective_frequency = base_frequency * self.parent_widget.harmonic

            donor_trajectory_real, donor_trajectory_imag = (
                phasor_from_fret_donor(
                    effective_frequency,
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
            colormap='viridis',
            contrast_limits=(0, 1),
        )

        if fret_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[fret_layer_name])

        self.fret_layer = self.viewer.add_layer(selected_fret_layer)
        self.fret_colormap = self.fret_layer.colormap.colors
        self.fret_layer.events.colormap.connect(self._on_colormap_changed)
        self.colormap_contrast_limits = (0, 1)
        self.fret_layer.events.contrast_limits.connect(
            self._on_contrast_limits_changed
        )

        self.plot_donor_trajectory()

        update_frequency_in_metadata(
            sample_layer,
            self.frequency_input.text().strip(),
        )
