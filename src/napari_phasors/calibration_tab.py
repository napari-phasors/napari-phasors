import contextlib
import copy
from html import escape
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_error
from phasorpy.lifetime import phasor_from_lifetime, polar_from_reference_phasor
from phasorpy.phasor import phasor_center, phasor_transform
from qtpy.QtCore import QRectF, QSize, Qt
from qtpy.QtGui import QAbstractTextDocumentLayout, QPalette, QTextDocument
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from ._utils import (
    REFERENCE_LIFETIMES_SOURCE,
    analysis_section_stylesheet,
    apply_filter_and_threshold,
    create_settings_note_label,
    make_section,
    reference_lifetimes,
    set_settings_note,
    setup_primary_button,
)

#: Settings a calibration stores in each calibrated layer.
_CALIBRATION_KEYS = (
    "calibrated",
    "calibration_phase",
    "calibration_modulation",
    "calibration_reference",
)

if TYPE_CHECKING:
    import napari

#: Qt.ItemDataRole slot used to stash the HTML label rendered by
#: ``_RichTextItemDelegate``, kept separate from the plain-text
#: ``Qt.DisplayRole`` so ``currentText``/``itemText`` stay unaffected.
_HTML_LABEL_ROLE = Qt.UserRole + 1


class _RichTextItemDelegate(QStyledItemDelegate):
    """Item delegate that renders an item's HTML label, if any, in a popup.

    Only affects the dropdown popup list; the combobox's own closed-state
    display always shows the plain-text ``Qt.DisplayRole``.
    """

    def paint(self, painter, option, index):
        """Draw the item, rendering its HTML label in the themed text color."""
        html = index.data(_HTML_LABEL_ROLE)
        if html is None:
            super().paint(painter, option, index)
            return

        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        style = (
            options.widget.style() if options.widget else QApplication.style()
        )

        doc = QTextDocument()
        doc.setHtml(html)

        options.text = ""
        style.drawControl(
            QStyle.CE_ItemViewItem, options, painter, options.widget
        )

        painter.save()
        text_rect = style.subElementRect(
            QStyle.SE_ItemViewItemText, options, options.widget
        )
        painter.translate(text_rect.topLeft())
        doc.setTextWidth(text_rect.width())

        # QTextDocument defaults to black text, ignoring the item's palette
        # (e.g. white text in napari's dark theme). Drawing through the
        # document layout with an explicit paint context, rather than
        # doc.drawContents(), lets the correct themed color come through.
        ctx = QAbstractTextDocumentLayout.PaintContext()
        text_color = options.palette.color(
            QPalette.HighlightedText
            if options.state & QStyle.State_Selected
            else QPalette.Text
        )
        ctx.palette.setColor(QPalette.Text, text_color)
        ctx.clip = QRectF(0, 0, text_rect.width(), text_rect.height())
        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sizeHint(self, option, index):
        """Return the size the item's rendered HTML label needs."""
        html = index.data(_HTML_LABEL_ROLE)
        if html is None:
            return super().sizeHint(option, index)

        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        doc = QTextDocument()
        doc.setHtml(html)
        doc.setTextWidth(max(options.rect.width(), 1))
        return QSize(int(doc.idealWidth()), int(doc.size().height()))


class CalibrationWidget(QWidget):
    """Widget to calibrate a FLIM image layer using a calibration image."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        """Initialize the calibration widget."""
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent
        # Set while the inputs are filled from a layer's settings, so that
        # filling them is not mistaken for an edit.
        self._restoring_settings = False

        # Build the calibration controls (formerly loaded from a .ui file).
        self.calibration_widget = self._build_calibration_widget()

        # Apply the shared section styling and wire the primary action as a
        # validated button (greyed out with a tooltip while inputs are missing).
        self.calibration_widget.setStyleSheet(analysis_section_stylesheet())
        self._refresh_calibrate_button = setup_primary_button(
            self.calibration_widget.calibrate_push_button,
            self._calibrate_validation,
            self._on_click,
            ready_tooltip="Calibrate the selected layer(s).",
        )

        # Re-evaluate the button whenever a required input changes.
        self.calibration_widget.frequency_input.textChanged.connect(
            lambda _=None: self._refresh_calibrate_button()
        )
        self.calibration_widget.lifetime_line_edit_widget.textChanged.connect(
            lambda _=None: self._refresh_calibrate_button()
        )

        # Fill the lifetime edit when a reference fluorophore is picked, and
        # clear the selection back to the placeholder if the user then edits
        # the lifetime by hand (so the shown fluorophore never contradicts it).
        self.calibration_widget.fluorophore_combobox.currentIndexChanged.connect(
            self._on_fluorophore_selected
        )
        self.calibration_widget.lifetime_line_edit_widget.textEdited.connect(
            self._on_lifetime_edited
        )
        self.calibration_widget.calibration_layer_combobox.currentTextChanged.connect(
            lambda _=None: self._refresh_calibrate_button()
        )

        # The reference inputs are unsaved settings of the primary layer
        # until Calibrate stores them with each calibrated layer.
        self.calibration_widget.calibration_layer_combobox.currentTextChanged.connect(
            self._stage_reference
        )
        self.calibration_widget.lifetime_line_edit_widget.textChanged.connect(
            self._stage_reference
        )
        self.calibration_widget.frequency_input.textChanged.connect(
            lambda _=None: self._refresh_settings_note()
        )

        # Connect layer events to populate combobox and update button state
        self.viewer.layers.events.inserted.connect(self._populate_comboboxes)
        self.viewer.layers.events.removed.connect(self._populate_comboboxes)

        # Connect to update button state when layer selection changes
        if hasattr(
            self.parent_widget, 'image_layer_with_phasor_features_combobox'
        ):
            self.parent_widget.image_layer_with_phasor_features_combobox.currentTextChanged.connect(
                self._update_button_state
            )

        # Populate combobox
        self._populate_comboboxes()

        # Create scroll area and add calibration widget to it
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.calibration_widget)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(scroll_area)
        self.setLayout(mainLayout)

    def _build_calibration_widget(self):
        """Build the calibration controls programmatically.

        Returns a container ``QWidget`` exposing the same named children the
        tab logic relies on (``calibration_layer_combobox``,
        ``frequency_input``, ``lifetime_line_edit_widget`` and
        ``calibrate_push_button``).
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Calibration reference -------------------------------------------
        # Borderless, no section title: just a bold label + the layer combobox.
        reference_box = QWidget()
        reference_layout = QVBoxLayout(reference_box)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_grid = QGridLayout()
        reference_layout.addLayout(reference_grid)
        widget.calibration_layer_label_widget = QLabel("Calibration Layer:")
        widget.calibration_layer_label_widget.setStyleSheet(
            "font-weight: 600;"
        )
        widget.calibration_layer_combobox = QComboBox()
        reference_grid.addWidget(widget.calibration_layer_label_widget, 0, 0)
        reference_grid.addWidget(widget.calibration_layer_combobox, 0, 1)
        layout.addWidget(reference_box)

        # Reference parameters --------------------------------------------
        parameters_box, parameters_layout = make_section(
            "Reference parameters"
        )
        parameters_grid = QGridLayout()
        parameters_layout.addLayout(parameters_grid)
        widget.frequency_label_widget = QLabel("Frequency (MHz):")
        widget.frequency_input = QLineEdit()
        widget.fluorophore_label_widget = QLabel("Reference fluorophore:")
        widget.fluorophore_combobox = QComboBox()
        widget.lifetime_label_widget = QLabel("Lifetime (ns):")
        widget.lifetime_line_edit_widget = QLineEdit()
        parameters_grid.addWidget(widget.frequency_label_widget, 0, 0)
        parameters_grid.addWidget(widget.frequency_input, 0, 1)
        parameters_grid.addWidget(widget.fluorophore_label_widget, 1, 0)
        parameters_grid.addWidget(widget.fluorophore_combobox, 1, 1)
        parameters_grid.addWidget(widget.lifetime_label_widget, 2, 0)
        parameters_grid.addWidget(widget.lifetime_line_edit_widget, 2, 1)
        layout.addWidget(parameters_box)

        self._populate_fluorophore_combobox(widget.fluorophore_combobox)

        # Cautions about the selected layers' frequencies.
        widget.settings_note = create_settings_note_label(widget)
        layout.addWidget(widget.settings_note)

        widget.calibrate_push_button = QPushButton("Calibrate")
        layout.addWidget(widget.calibrate_push_button)

        layout.addStretch(1)
        return widget

    def _populate_fluorophore_combobox(self, combobox):
        """Fill the reference-fluorophore combobox from the known lifetimes.

        The first entry is a placeholder; each fluorophore entry stores its
        lifetime (ns) as item data so selecting it can fill the lifetime edit.
        """
        combobox.setItemDelegate(_RichTextItemDelegate(combobox))

        combobox.addItem("Select fluorophore (optional)", None)
        for entry in reference_lifetimes():
            name = escape(entry['name'])
            rest = f"({escape(entry['solvent'])}): {entry['lifetime']:g} ns"
            combobox.addItem(f"{entry['name']} {rest}", entry["lifetime"])
            combobox.setItemData(
                combobox.count() - 1,
                f"<b>{name}</b> {rest}",
                _HTML_LABEL_ROLE,
            )
        # Long fluorophore labels shouldn't force the combobox (and thus the
        # whole tab) to widen; let it size to a modest content length and
        # shrink/grow with the layout instead. The full label is still shown
        # via the tooltip-less native elided text and the dropdown popup.
        combobox.setMinimumContentsLength(8)
        combobox.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        combobox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combobox.setToolTip(
            "Pick a reference fluorophore to fill in its known lifetime.\n"
            f"Source: {REFERENCE_LIFETIMES_SOURCE}"
        )

    def _on_fluorophore_selected(self, index):
        """Populate the lifetime edit with the selected fluorophore lifetime."""
        lifetime = self.calibration_widget.fluorophore_combobox.itemData(index)
        if lifetime is None:
            return
        lifetime_edit = self.calibration_widget.lifetime_line_edit_widget
        # Setting the text programmatically must not reset the combobox, so
        # guard against the ``_on_lifetime_edited`` handler (it only reacts to
        # user edits, but stay explicit).
        self._setting_lifetime_from_combo = True
        try:
            lifetime_edit.setText(f"{lifetime:g}")
        finally:
            self._setting_lifetime_from_combo = False

    def _on_lifetime_edited(self, _text=None):
        """Reset the fluorophore selection when the lifetime is edited by hand."""
        if getattr(self, '_setting_lifetime_from_combo', False):
            return
        combobox = self.calibration_widget.fluorophore_combobox
        if combobox.currentIndex() != 0:
            combobox.blockSignals(True)
            combobox.setCurrentIndex(0)
            combobox.blockSignals(False)

    def _populate_comboboxes(self, event=None):
        """Populate calibration layer combobox with image layers."""
        if getattr(self, '_populating_comboboxes', False):
            return

        self._populating_comboboxes = True

        try:
            current_text = (
                self.calibration_widget.calibration_layer_combobox.currentText()
            )

            self.calibration_widget.calibration_layer_combobox.blockSignals(
                True
            )

            self.calibration_widget.calibration_layer_combobox.clear()
            image_layers = [
                layer
                for layer in self.viewer.layers
                if isinstance(layer, Image)
                and 'G' in layer.metadata
                and 'S' in layer.metadata
            ]

            layer_names = [layer.name for layer in image_layers]
            self.calibration_widget.calibration_layer_combobox.addItems(
                layer_names
            )

            if current_text in layer_names:
                index = self.calibration_widget.calibration_layer_combobox.findText(
                    current_text
                )
                if index >= 0:
                    self.calibration_widget.calibration_layer_combobox.setCurrentIndex(
                        index
                    )

            self.calibration_widget.calibration_layer_combobox.blockSignals(
                False
            )

            for layer in image_layers:
                with contextlib.suppress(TypeError, ValueError):
                    layer.events.name.disconnect(self._populate_comboboxes)
                layer.events.name.connect(self._populate_comboboxes)

        finally:
            self._populating_comboboxes = False

    def _on_image_layer_changed(self):
        """Update button state when the selected image layer changes."""
        self._restore_reference_from_primary()
        self._update_button_state()
        self._refresh_settings_note()

    def _has_settings_store(self):
        """Return whether the parent keeps per-layer settings."""
        return self.parent_widget is not None and hasattr(
            self.parent_widget, "settings_store"
        )

    def _collect_reference(self):
        """Return the calibration reference entered in the inputs."""
        try:
            lifetime = float(
                self.calibration_widget.lifetime_line_edit_widget.text()
            )
        except ValueError:
            lifetime = None
        name = self.calibration_widget.calibration_layer_combobox.currentText()
        return {
            "reference_layer": name or None,
            "reference_lifetime": lifetime,
        }

    def _stage_reference(self, *_):
        """Keep the entered reference as the primary's unsaved setting."""
        if self._restoring_settings or not self._has_settings_store():
            return
        self.parent_widget.stage_setting(
            "calibration_reference", self._collect_reference()
        )

    def _restore_reference_from_primary(self):
        """Show the reference the primary layer was calibrated (or edited) with.

        Inputs are left alone when the primary layer has no reference, so the
        values being typed for a new calibration are kept.
        """
        if not self._has_settings_store():
            return
        primary = self.parent_widget.get_primary_layer()
        if primary is None:
            return
        reference = self.parent_widget.layer_settings(primary).get(
            "calibration_reference"
        )
        if not isinstance(reference, dict):
            return
        self._restoring_settings = True
        try:
            name = reference.get("reference_layer")
            combobox = self.calibration_widget.calibration_layer_combobox
            if name and combobox.findText(name) >= 0:
                combobox.setCurrentText(name)
            lifetime = reference.get("reference_lifetime")
            if lifetime is not None:
                self.calibration_widget.lifetime_line_edit_widget.setText(
                    f"{float(lifetime):g}"
                )
        finally:
            self._restoring_settings = False

    def _refresh_settings_note(self):
        """Caution about the frequencies the selected layers are calibrated at."""
        messages = []
        if self._has_settings_store():
            selected_layers = self.parent_widget.get_selected_layers()
            # Nothing is calibrated while the button uncalibrates.
            if selected_layers and not any(
                self._is_layer_calibrated(layer) for layer in selected_layers
            ):
                messages = self.parent_widget.frequency_note_messages(
                    self.calibration_widget.frequency_input.text()
                )
        set_settings_note(self.calibration_widget.settings_note, messages)

    def _update_button_state(self):
        """Update button text and state based on current layer's calibration status."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            self.calibration_widget.calibrate_push_button.setText("Calibrate")
            self._refresh_calibrate_button()
            return

        # Check if any selected layer is calibrated
        any_calibrated = any(
            self._is_layer_calibrated(layer) for layer in selected_layers
        )

        if any_calibrated:
            self.calibration_widget.calibrate_push_button.setText(
                "Uncalibrate"
            )
        else:
            self.calibration_widget.calibrate_push_button.setText("Calibrate")

        self._refresh_calibrate_button()

    def _calibrate_validation(self):
        """Return ``None`` if calibration can run, else the missing-input msg.

        Uncalibration (when a selected layer is already calibrated) needs no
        inputs, so the button is always ready in that case.
        """
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return "Select at least one image layer with phasor features."
        if any(self._is_layer_calibrated(layer) for layer in selected_layers):
            return None
        if (
            not self.calibration_widget.calibration_layer_combobox.currentText()
        ):
            return "Select a calibration layer."
        if not self.calibration_widget.frequency_input.text().strip():
            return "Enter the frequency (MHz)."
        if (
            not self.calibration_widget.lifetime_line_edit_widget.text().strip()
        ):
            return "Enter the reference lifetime (ns)."
        return None

    def _on_click(self):
        """Handle calibration button click for all selected layers."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            show_error("Select sample and calibration layers")
            return

        calibration_name = (
            self.calibration_widget.calibration_layer_combobox.currentText()
        )

        if calibration_name == "":
            show_error("Select sample and calibration layers")
            return

        any_calibrated = any(
            self._is_layer_calibrated(layer) for layer in selected_layers
        )

        if any_calibrated:
            calibrated_layers = [
                layer
                for layer in selected_layers
                if self._is_layer_calibrated(layer)
            ]
            for layer in calibrated_layers:
                self._uncalibrate_layer(layer.name)
        else:
            calibrated_layers = []
            for layer in selected_layers:
                result = self._calibrate_layer(layer.name, calibration_name)
                if result is not False:
                    calibrated_layers.append(layer)
            if calibrated_layers and self._has_settings_store():
                self.parent_widget.commit_frequency(
                    calibrated_layers,
                    self.calibration_widget.frequency_input.text(),
                )

        self._update_button_state()
        self._refresh_settings_note()
        self.parent_widget.plot()

    def _is_layer_calibrated(self, sample_layer):
        """Check if a layer is already calibrated."""
        settings = sample_layer.metadata.get("settings", {})
        return settings.get("calibrated", False)

    def _calibrate_layer(self, sample_name, calibration_name):
        """Calibrate a layer using the specified calibration layer."""
        sample_layer = self.viewer.layers[sample_name]
        calibration_layer = self.viewer.layers[calibration_name]

        calibration_was_calibrated = False
        saved_calibration = None
        if self._is_layer_calibrated(calibration_layer):
            reply = QMessageBox.question(
                self,
                'Calibration Layer Already Calibrated',
                f'The calibration layer "{calibration_name}" is already calibrated.\n\n'
                'Would you like to use the uncalibrated data as reference?\n\n'
                'Yes: Use original uncalibrated data\n'
                'No: Use current calibrated data\n'
                'Cancel: Abort calibration',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )

            if reply == QMessageBox.Cancel:
                return False

            if reply == QMessageBox.Yes:
                calibration_was_calibrated = True
                # Uncalibrating drops the stored parameters; keep them so the
                # reference layer can be calibrated again afterwards.
                saved_calibration = self._calibration_settings(
                    calibration_layer
                )
                self._uncalibrate_layer(calibration_name)
                calibration_layer = self.viewer.layers[calibration_name]

        frequency, lifetime = self._get_and_validate_inputs()
        if frequency is None or lifetime is None:
            if calibration_was_calibrated:
                self._restore_calibration(calibration_name, saved_calibration)
            return False
        if self._has_settings_store():
            # A layer is calibrated at the frequency it was acquired with.
            frequency = self.parent_widget.layer_frequency(
                sample_layer, frequency
            )

        sample_phasor_data, harmonics = self._get_phasor_data(sample_layer)
        calibration_phasor_data, calibration_harmonics = self._get_phasor_data(
            calibration_layer
        )

        if not np.array_equal(harmonics, calibration_harmonics):
            show_error(
                "Harmonics in sample and calibration layers do not match"
            )
            if calibration_was_calibrated:
                self._restore_calibration(calibration_name, saved_calibration)
            return False

        phi_zero, mod_zero = self._calculate_calibration_parameters(
            calibration_layer,
            calibration_phasor_data,
            calibration_harmonics,
            frequency,
            harmonics,
            lifetime,
        )

        try:
            values = {
                "calibration_phase": np.asarray(phi_zero).tolist(),
                "calibration_modulation": np.asarray(mod_zero).tolist(),
                "calibrated": True,
                # Everything needed to repeat the calibration.
                "calibration_reference": {
                    "reference_layer": calibration_name,
                    "reference_lifetime": lifetime,
                    "frequency": frequency,
                },
            }
            if self._has_settings_store():
                self.parent_widget.commit_analysis_settings(
                    values, layers=[sample_layer]
                )
            else:
                sample_layer.metadata.setdefault("settings", {}).update(values)

            self._apply_phasor_transformation(sample_name, phi_zero, mod_zero)

            self._apply_existing_filters_and_thresholds(sample_layer)
        finally:
            if calibration_was_calibrated:
                self._restore_calibration(calibration_name, saved_calibration)

    @staticmethod
    def _calibration_settings(layer):
        """Return a copy of the calibration settings stored in *layer*."""
        settings = layer.metadata.get("settings") or {}
        return {
            key: copy.deepcopy(settings[key])
            for key in _CALIBRATION_KEYS
            if key in settings
        }

    def _restore_calibration(self, layer_name, saved=None):
        """Restore calibration to a layer using stored parameters.

        ``saved`` holds the settings kept before the layer was uncalibrated
        (see :meth:`_calibration_settings`); uncalibrating removes them from
        the layer, so without it there is nothing to restore.
        """
        layer = self.viewer.layers[layer_name]
        settings = layer.metadata.setdefault("settings", {})
        if saved:
            settings.update(copy.deepcopy(saved))

        phi_zero = settings.get("calibration_phase")
        mod_zero = settings.get("calibration_modulation")

        if phi_zero is not None and mod_zero is not None:
            if isinstance(phi_zero, list):
                phi_zero = np.array(phi_zero)
            if isinstance(mod_zero, list):
                mod_zero = np.array(mod_zero)

            self._apply_phasor_transformation(layer_name, phi_zero, mod_zero)
            settings["calibrated"] = True
            self._apply_existing_filters_and_thresholds(layer)

    def _uncalibrate_layer(self, sample_name):
        """Uncalibrate a layer."""
        if sample_name == "":
            return

        sample_layer = self.viewer.layers[sample_name]
        sample_metadata = sample_layer.metadata
        settings = sample_metadata.get("settings", {})

        phi_zero = settings.get("calibration_phase")
        mod_zero = settings.get("calibration_modulation")

        if phi_zero is None or mod_zero is None:
            show_error("Layer is not calibrated")
            return

        phi_zero_inv, mod_zero_inv = self._invert_calibration_parameters(
            phi_zero, mod_zero
        )

        self._apply_phasor_transformation(
            sample_name, phi_zero_inv, mod_zero_inv
        )

        settings["calibrated"] = False
        settings.pop("calibration_phase", None)
        settings.pop("calibration_modulation", None)
        settings.pop("calibration_reference", None)

        self._apply_existing_filters_and_thresholds(sample_layer)

    def _get_and_validate_inputs(self):
        """Get and validate frequency and lifetime inputs."""
        frequency = self.calibration_widget.frequency_input.text().strip()
        lifetime = (
            self.calibration_widget.lifetime_line_edit_widget.text().strip()
        )

        if frequency == "":
            show_error("Enter frequency")
            return None, None
        if lifetime == "":
            show_error("Enter reference lifetime")
            return None, None

        return float(frequency), float(lifetime)

    def _get_phasor_data(self, layer):
        """Get phasor data and harmonics from a layer."""
        phasor_data = {
            "G_original": layer.metadata.get("G_original"),
            "S_original": layer.metadata.get("S_original"),
            "G": layer.metadata.get("G"),
            "S": layer.metadata.get("S"),
        }
        harmonics = layer.metadata.get("harmonics")
        return phasor_data, harmonics

    def _calculate_calibration_parameters(
        self,
        calibration_layer,
        calibration_phasor_data,
        calibration_harmonics,
        frequency,
        harmonics,
        lifetime,
    ):
        """Calculate calibration phase and modulation parameters."""
        calibration_mean = calibration_layer.metadata["original_mean"]

        calibration_g = calibration_phasor_data["G_original"]
        calibration_s = calibration_phasor_data["S_original"]

        _, measured_re, measured_im = phasor_center(
            calibration_mean, calibration_g, calibration_s
        )

        harmonics_array = np.atleast_1d(harmonics)

        known_re, known_im = phasor_from_lifetime(
            frequency * harmonics_array, lifetime
        )
        phi_zero, mod_zero = polar_from_reference_phasor(
            measured_re, measured_im, known_re, known_im
        )

        return phi_zero, mod_zero

    def _invert_calibration_parameters(self, phi_zero, mod_zero):
        """Invert calibration parameters for uncalibration."""
        if isinstance(phi_zero, list):
            phi_zero = np.array(phi_zero)
        if isinstance(mod_zero, list):
            mod_zero = np.array(mod_zero)

        if np.ndim(phi_zero) > 0:
            phi_zero_inv = -phi_zero.copy()
            mod_zero_inv = 1.0 / mod_zero.copy()
        else:
            phi_zero_inv = -phi_zero
            mod_zero_inv = 1.0 / mod_zero

        return phi_zero_inv, mod_zero_inv

    def _apply_existing_filters_and_thresholds(self, sample_layer):
        """Apply existing filter and threshold settings if they exist."""
        settings = sample_layer.metadata.get("settings", {})
        filter_settings = settings.get("filter", {})

        # Build kwargs dict with only non-None values
        kwargs = {}
        for key, value in [
            ("filter_method", filter_settings.get("method")),
            ("size", filter_settings.get("size")),
            ("repeat", filter_settings.get("repeat")),
            ("sigma", filter_settings.get("sigma")),
            ("levels", filter_settings.get("levels")),
            ("threshold", settings.get("threshold")),
            ("threshold_upper", settings.get("threshold_upper")),
            ("threshold_method", settings.get("threshold_method")),
        ]:
            if value is not None:
                kwargs[key] = value

        if kwargs:
            apply_filter_and_threshold(sample_layer, **kwargs)

    def _apply_phasor_transformation(self, sample_name, phi_zero, mod_zero):
        """Apply phasor transformation with given correction parameters."""
        sample_layer = self.viewer.layers[sample_name]
        sample_metadata = sample_layer.metadata

        harmonics = sample_metadata.get("harmonics")
        g_original = sample_metadata["G_original"]
        s_original = sample_metadata["S_original"]
        g_current = sample_metadata["G"]
        s_current = sample_metadata["S"]

        if isinstance(phi_zero, list):
            phi_zero = np.array(phi_zero)
        if isinstance(mod_zero, list):
            mod_zero = np.array(mod_zero)

        harmonics = np.atleast_1d(harmonics)

        if g_original.ndim > 1 and len(harmonics) > 1:
            spatial_dims = g_original.ndim - 1
            expand_shape = (slice(None),) + (None,) * spatial_dims

            if np.ndim(phi_zero) > 0:
                phi_zero_expanded = phi_zero[expand_shape]
                mod_zero_expanded = mod_zero[expand_shape]
            else:
                phi_zero_expanded = phi_zero
                mod_zero_expanded = mod_zero
        else:
            phi_zero_expanded = phi_zero
            mod_zero_expanded = mod_zero

        real_original, imag_original = phasor_transform(
            g_original,
            s_original,
            phi_zero_expanded,
            mod_zero_expanded,
        )

        real, imag = phasor_transform(
            g_current,
            s_current,
            phi_zero_expanded,
            mod_zero_expanded,
        )

        sample_metadata["G_original"] = real_original
        sample_metadata["S_original"] = imag_original
        sample_metadata["G"] = real
        sample_metadata["S"] = imag

        if self.parent_widget:
            self.parent_widget.refresh_phasor_data()

    def closeEvent(self, event):
        """Clean up signal connections before closing."""
        # Disconnect viewer events
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self.viewer.layers.events.inserted.disconnect(
                self._populate_comboboxes
            )
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            self.viewer.layers.events.removed.disconnect(
                self._populate_comboboxes
            )

        # Disconnect parent widget signal if present
        if hasattr(self, 'parent_widget') and hasattr(
            self.parent_widget, 'image_layer_with_phasor_features_combobox'
        ):
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                self.parent_widget.image_layer_with_phasor_features_combobox.currentTextChanged.disconnect(
                    self._update_button_state
                )

        # Disconnect layer name change events
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            for layer in self.viewer.layers:
                with contextlib.suppress(TypeError, ValueError):
                    layer.events.name.disconnect(self._populate_comboboxes)

        event.accept()
