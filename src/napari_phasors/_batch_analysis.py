"""Batch Analysis widget.

This module provides :class:`BatchAnalysisWidget`, a dockable widget that
applies the same phasor analysis pipeline to every supported file found in a
folder and exports the results to disk.

The widget runs *headlessly*: it reuses the plugin's reader
(:func:`napari_phasors._reader.napari_get_reader`), the separable computation
functions (:func:`napari_phasors._utils.apply_filter_and_threshold`,
:func:`napari_phasors._utils.compute_calibration_parameters`,
:func:`napari_phasors._utils.apply_calibration_correction`) and the writer
functions (:func:`napari_phasors._writer.write_ome_tiff`,
:func:`napari_phasors._writer.export_layer_as_csv`) without instantiating the
interactive analysis tabs.
"""

import ast
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_error, show_info
from phasorpy.component import phasor_component_fraction
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._reader import extension_mapping, napari_get_reader
from ._utils import (
    CollapsibleSection,
    apply_calibration_correction,
    apply_filter_and_threshold,
    compute_calibration_parameters,
    read_ome_tiff_settings,
)
from ._writer import (
    export_layer_as_csv,
    export_layer_as_image,
    write_ome_tiff,
)

if TYPE_CHECKING:
    import napari


# Per-format read parameters exposed as typed inputs in the batch widget.
# Mirrors the ``args_defaults`` declared for each reader in
# ``napari_phasors._reader.extension_mapping``. Each spec is
# ``(arg_name, kind, default)`` where ``kind`` is one of
# ``"int"``, ``"bool"``, ``"int_or_none"``, ``"str"`` or ``"str_or_none"``.
BATCH_READER_PARAMS = {
    ".ptu": [("frame", "int", -1), ("keepdims", "bool", False)],
    ".fbd": [
        ("frame", "int", -1),
        ("keepdims", "bool", False),
        ("channel", "int_or_none", None),
    ],
    ".lif": [("image", "int_or_none", None), ("dim", "str", "λ")],
    ".json": [("channel", "int", 0), ("dtype", "str_or_none", None)],
    ".ifli": [("channel", "int", 0)],
}


def supported_extensions():
    """Return the sorted set of file extensions the plugin can read.

    Longer (more specific) extensions such as ``.ome.tif`` are placed before
    their shorter counterparts so that callers matching by suffix can pick the
    most specific match first.
    """
    exts = set(extension_mapping["raw"].keys()) | set(
        extension_mapping["processed"].keys()
    )
    return sorted(exts, key=len, reverse=True)


def match_extension(filename):
    """Return the supported extension matching ``filename``, or ``None``.

    Matching is done by longest suffix so that e.g. ``image.ome.tif`` resolves
    to ``.ome.tif`` rather than ``.tif``.
    """
    lower = filename.lower()
    for ext in supported_extensions():
        if lower.endswith(ext):
            return ext
    return None


def scan_folder(folder, recursive):
    """Scan ``folder`` for supported files grouped by extension.

    Parameters
    ----------
    folder : str
        Directory to scan.
    recursive : bool
        Whether to descend into subfolders.

    Returns
    -------
    dict
        Mapping of extension -> sorted list of absolute file paths.
    """
    found = {}
    if recursive:
        walker = ((root, files) for root, _dirs, files in os.walk(folder))
    else:
        try:
            entries = os.listdir(folder)
        except OSError:
            return found
        files = [f for f in entries if os.path.isfile(os.path.join(folder, f))]
        walker = [(folder, files)]

    for root, files in walker:
        for name in files:
            ext = match_extension(name)
            if ext is not None:
                found.setdefault(ext, []).append(os.path.join(root, name))
    for ext in found:
        found[ext].sort()
    return found


def parse_harmonics(text):
    """Parse a harmonics string into ``list[int]``, ``"all"`` or ``None``.

    Accepts comma-separated integers (e.g. ``"1, 2"``), the literal ``"all"``
    or an empty string (returns ``None``, letting the reader use its default).
    """
    text = text.strip().lower()
    if not text:
        return None
    if text == "all":
        return "all"
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values or None


@dataclass
class BatchPipeline:
    """Resolved, headless analysis steps applied to every file in a batch.

    Steps are applied in order: calibration, filter/threshold, then component
    analysis. ``calibration`` is either ``{"phi_zero": ..., "mod_zero": ...}``
    (pre-computed correction parameters) or ``None``.
    """

    calibration: dict | None = None
    filter: dict | None = None
    components: dict | None = None
    warnings: list = field(default_factory=list)


def apply_pipeline(layer, pipeline):
    """Apply ``pipeline`` to ``layer`` in place and return extra output layers.

    Parameters
    ----------
    layer : napari.layers.Image
        Layer with phasor metadata to process.
    pipeline : BatchPipeline
        The resolved analysis steps.

    Returns
    -------
    list of napari.layers.Image
        Additional layers produced by the analyses (e.g. component fraction
        images) that should also be exported. Empty if none.
    """
    extra_layers = []

    if pipeline.calibration is not None:
        apply_calibration_correction(
            layer,
            pipeline.calibration["phi_zero"],
            pipeline.calibration["mod_zero"],
        )

    if pipeline.filter is not None:
        apply_filter_and_threshold(layer, **pipeline.filter)

    if pipeline.components is not None:
        frac_layer = _apply_component_fraction(layer, pipeline.components)
        if frac_layer is not None:
            extra_layers.append(frac_layer)

    return extra_layers


def _select_harmonic_arrays(layer, harmonic):
    """Return ``(real, imag)`` for ``harmonic`` from a layer's phasor data."""
    g_array = layer.metadata.get("G")
    s_array = layer.metadata.get("S")
    if g_array is None or s_array is None:
        return None, None
    harmonics = layer.metadata.get("harmonics")
    if harmonics is not None:
        harmonics_array = np.atleast_1d(harmonics)
        idx = np.where(harmonics_array == harmonic)[0]
        if g_array.ndim == layer.data.ndim + 1 and idx.size > 0:
            return g_array[idx[0]], s_array[idx[0]]
    return g_array, s_array


def _apply_component_fraction(layer, components):
    """Run a 2-component linear-projection fraction analysis on ``layer``.

    Returns a new :class:`napari.layers.Image` holding the fraction image for
    the first component, or ``None`` if the layer lacks usable phasor data.
    """
    harmonic = components["harmonic"]
    real, imag = _select_harmonic_arrays(layer, harmonic)
    if real is None:
        return None

    component_real = components["component_real"]
    component_imag = components["component_imag"]

    fraction = phasor_component_fraction(
        real, imag, component_real, component_imag
    )

    name = f"{components['name']} fractions: {layer.name}"
    return Image(np.asarray(fraction), name=name)


class BatchReadOptionsWidget(QWidget):
    """Collect reader options for a single file format (no signal preview).

    Renders typed inputs for the chosen extension's known parameters (from
    :data:`BATCH_READER_PARAMS`) plus a dynamic set of "additional kwargs"
    rows for anything else, mirroring the dynamic-kwargs feature of the
    interactive Custom Import widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._extension = None
        self._param_widgets = {}
        self._kwargs_rows = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._form_container = QWidget()
        self._form = QFormLayout(self._form_container)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._form_container)

        # Dynamic "additional kwargs" rows.
        self._kwargs_container = QWidget()
        self._kwargs_layout = QVBoxLayout(self._kwargs_container)
        self._kwargs_layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._kwargs_container)

        add_kwarg_button = QPushButton("+ Add reader keyword argument")
        add_kwarg_button.clicked.connect(lambda: self._add_kwarg_row())
        self._layout.addWidget(add_kwarg_button)

    def set_extension(self, extension):
        """Rebuild the typed inputs for ``extension``."""
        self._extension = extension
        # Clear existing typed inputs.
        while self._form.rowCount():
            self._form.removeRow(0)
        self._param_widgets = {}

        for arg, kind, default in BATCH_READER_PARAMS.get(extension, []):
            widget = self._make_param_widget(kind, default)
            self._param_widgets[arg] = (widget, kind)
            self._form.addRow(QLabel(f"{arg}:"), widget)

    def _make_param_widget(self, kind, default):
        if kind == "int":
            w = QSpinBox()
            w.setRange(-1, 1_000_000)
            w.setValue(int(default))
            return w
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
            return w
        # int_or_none, str, str_or_none -> free text line edit.
        w = QLineEdit()
        if default is not None:
            w.setText(str(default))
        return w

    def _add_kwarg_row(self, key="", value=""):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        key_edit = QLineEdit()
        key_edit.setPlaceholderText("key")
        key_edit.setText(key)
        val_edit = QLineEdit()
        val_edit.setPlaceholderText("value")
        val_edit.setText(value)
        remove = QPushButton("✕")
        remove.setFixedWidth(28)
        layout.addWidget(key_edit)
        layout.addWidget(val_edit)
        layout.addWidget(remove)
        self._kwargs_layout.addWidget(row)
        entry = (row, key_edit, val_edit)
        self._kwargs_rows.append(entry)
        remove.clicked.connect(lambda: self._remove_kwarg_row(entry))

    def _remove_kwarg_row(self, entry):
        row, _, _ = entry
        if entry in self._kwargs_rows:
            self._kwargs_rows.remove(entry)
        row.setParent(None)
        row.deleteLater()

    def get_reader_options(self):
        """Return the collected reader options as a dict."""
        options = {}
        for arg, (widget, kind) in self._param_widgets.items():
            if kind == "int":
                options[arg] = widget.value()
            elif kind == "bool":
                options[arg] = widget.isChecked()
            else:
                text = widget.text().strip()
                if kind == "int_or_none":
                    options[arg] = int(text) if text else None
                elif kind == "str_or_none":
                    options[arg] = text if text else None
                else:  # str
                    options[arg] = text
        for _row, key_edit, val_edit in self._kwargs_rows:
            key = key_edit.text().strip()
            val_str = val_edit.text().strip()
            if not key:
                continue
            try:
                value = ast.literal_eval(val_str)
            except (ValueError, TypeError, SyntaxError, MemoryError):
                value = val_str
            options[key] = value
        return options


class BatchAnalysisWidget(QWidget):
    """Apply a phasor analysis pipeline to every supported file in a folder."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self._floated = False
        self._scanned = {}  # extension -> list[path]
        self._input_folder = None

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Batch Analysis</b>"))
        intro = QLabel(
            "Apply the same reading and analysis pipeline to every supported "
            "file in a folder, then export the results."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._build_input_section(layout)
        self._build_read_section(layout)
        self._build_pipeline_section(layout)
        self._build_export_section(layout)
        self._build_run_section(layout)
        layout.addStretch()

        self.viewer.layers.events.inserted.connect(
            lambda e: self._populate_layer_comboboxes()
        )
        self.viewer.layers.events.removed.connect(
            lambda e: self._populate_layer_comboboxes()
        )
        self._populate_layer_comboboxes()

    # -- UI construction ---------------------------------------------------

    def _build_input_section(self, layout):
        section = CollapsibleSection(
            "1. Input folder", initially_collapsed=False
        )
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        self.select_folder_button = QPushButton("Select input folder")
        self.select_folder_button.clicked.connect(self._on_select_folder)
        row.addWidget(self.select_folder_button)
        self.folder_label = QLabel("<i>No folder selected</i>")
        self.folder_label.setWordWrap(True)
        row.addWidget(self.folder_label, 1)
        body_layout.addLayout(row)

        self.subfolders_checkbox = QCheckBox("Include subfolders")
        self.subfolders_checkbox.stateChanged.connect(lambda _: self._rescan())
        body_layout.addWidget(self.subfolders_checkbox)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("File format:"))
        self.format_combobox = QComboBox()
        self.format_combobox.currentIndexChanged.connect(
            self._on_format_changed
        )
        format_row.addWidget(self.format_combobox, 1)
        body_layout.addLayout(format_row)

        section.add_widget(body)
        layout.addWidget(section)

    def _build_read_section(self, layout):
        self.read_section = CollapsibleSection(
            "2. Read parameters", initially_collapsed=False
        )
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        harmonics_row = QHBoxLayout()
        harmonics_row.addWidget(QLabel("Harmonics:"))
        self.harmonics_edit = QLineEdit("1, 2")
        self.harmonics_edit.setToolTip(
            "Comma-separated harmonics (e.g. '1, 2'), 'all', or empty for the "
            "reader default."
        )
        harmonics_row.addWidget(self.harmonics_edit, 1)
        body_layout.addLayout(harmonics_row)

        self.read_options_widget = BatchReadOptionsWidget()
        body_layout.addWidget(self.read_options_widget)

        self.read_section.add_widget(body)
        layout.addWidget(self.read_section)

    def _build_pipeline_section(self, layout):
        section = CollapsibleSection(
            "3. Analysis pipeline", initially_collapsed=False
        )
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        mode_row = QHBoxLayout()
        self.manual_mode_radio = QRadioButton("Configure manually")
        self.manual_mode_radio.setChecked(True)
        self.copy_mode_radio = QRadioButton("Copy settings from reference")
        self.manual_mode_radio.toggled.connect(self._on_pipeline_mode_changed)
        mode_row.addWidget(self.manual_mode_radio)
        mode_row.addWidget(self.copy_mode_radio)
        mode_row.addStretch()
        body_layout.addLayout(mode_row)

        self._build_manual_pipeline(body_layout)
        self._build_copy_pipeline(body_layout)
        self._on_pipeline_mode_changed()

        section.add_widget(body)
        layout.addWidget(section)

    def _build_manual_pipeline(self, parent_layout):
        self.manual_container = QWidget()
        manual_layout = QVBoxLayout(self.manual_container)
        manual_layout.setContentsMargins(0, 0, 0, 0)

        # Filter & threshold ------------------------------------------------
        self.filter_enable = QCheckBox("Apply filter & threshold")
        filter_section = CollapsibleSection("Filter & threshold")
        filter_body = QWidget()
        f_layout = QFormLayout(filter_body)
        f_layout.setContentsMargins(0, 0, 0, 0)

        self.filter_method_combo = QComboBox()
        self.filter_method_combo.addItems(["None", "Median", "Wavelet"])
        f_layout.addRow("Filter method:", self.filter_method_combo)

        self.median_size_spin = QSpinBox()
        self.median_size_spin.setRange(2, 99)
        self.median_size_spin.setValue(3)
        f_layout.addRow("Median kernel size:", self.median_size_spin)

        self.median_repeat_spin = QSpinBox()
        self.median_repeat_spin.setRange(1, 100)
        self.median_repeat_spin.setValue(1)
        f_layout.addRow("Median repetitions:", self.median_repeat_spin)

        self.wavelet_sigma_spin = QDoubleSpinBox()
        self.wavelet_sigma_spin.setRange(0.1, 10.0)
        self.wavelet_sigma_spin.setSingleStep(0.1)
        self.wavelet_sigma_spin.setValue(2.0)
        f_layout.addRow("Wavelet sigma:", self.wavelet_sigma_spin)

        self.wavelet_levels_spin = QSpinBox()
        self.wavelet_levels_spin.setRange(1, 10)
        self.wavelet_levels_spin.setValue(1)
        f_layout.addRow("Wavelet levels:", self.wavelet_levels_spin)

        self.threshold_method_combo = QComboBox()
        self.threshold_method_combo.addItems(
            ["None", "Manual", "Otsu", "Li", "Yen"]
        )
        f_layout.addRow("Threshold method:", self.threshold_method_combo)

        self.threshold_min_spin = QDoubleSpinBox()
        self.threshold_min_spin.setRange(0.0, 1_000_000.0)
        self.threshold_min_spin.setDecimals(3)
        f_layout.addRow("Threshold (min):", self.threshold_min_spin)

        self.threshold_max_spin = QDoubleSpinBox()
        self.threshold_max_spin.setRange(0.0, 1_000_000.0)
        self.threshold_max_spin.setDecimals(3)
        self.threshold_max_spin.setSpecialValueText("none")
        f_layout.addRow("Threshold (max):", self.threshold_max_spin)

        filter_section.add_widget(filter_body)
        self.filter_enable.toggled.connect(
            lambda checked: filter_section.set_content_visible(checked)
        )
        manual_layout.addWidget(self.filter_enable)
        manual_layout.addWidget(filter_section)

        # Calibration -------------------------------------------------------
        self.calibration_enable = QCheckBox("Apply calibration")
        calib_section = CollapsibleSection("Calibration")
        calib_body = QWidget()
        c_layout = QFormLayout(calib_body)
        c_layout.setContentsMargins(0, 0, 0, 0)

        self.calib_reference_combo = QComboBox()
        c_layout.addRow("Reference layer:", self.calib_reference_combo)

        calib_file_row = QHBoxLayout()
        self.calib_file_edit = QLineEdit()
        self.calib_file_edit.setPlaceholderText("…or OME-TIFF reference file")
        calib_browse = QPushButton("Browse")
        calib_browse.clicked.connect(self._on_browse_calibration_file)
        calib_file_row.addWidget(self.calib_file_edit, 1)
        calib_file_row.addWidget(calib_browse)
        c_layout.addRow("Reference file:", calib_file_row)

        self.calib_frequency_spin = QDoubleSpinBox()
        self.calib_frequency_spin.setRange(0.0, 1_000_000.0)
        self.calib_frequency_spin.setDecimals(3)
        c_layout.addRow("Frequency (MHz):", self.calib_frequency_spin)

        self.calib_lifetime_spin = QDoubleSpinBox()
        self.calib_lifetime_spin.setRange(0.0, 1_000.0)
        self.calib_lifetime_spin.setDecimals(3)
        c_layout.addRow("Reference lifetime (ns):", self.calib_lifetime_spin)

        calib_section.add_widget(calib_body)
        self.calibration_enable.toggled.connect(
            lambda checked: calib_section.set_content_visible(checked)
        )
        manual_layout.addWidget(self.calibration_enable)
        manual_layout.addWidget(calib_section)

        # Component analysis ------------------------------------------------
        self.components_enable = QCheckBox(
            "Component analysis (2-component linear projection)"
        )
        comp_section = CollapsibleSection("Component analysis")
        comp_body = QWidget()
        comp_layout = QFormLayout(comp_body)
        comp_layout.setContentsMargins(0, 0, 0, 0)

        self.comp_name_edit = QLineEdit("Component 1")
        comp_layout.addRow("Fraction name:", self.comp_name_edit)

        self.comp1_g_spin = QDoubleSpinBox()
        self.comp1_g_spin.setRange(-2.0, 2.0)
        self.comp1_g_spin.setDecimals(4)
        comp_layout.addRow("Component 1 G:", self.comp1_g_spin)
        self.comp1_s_spin = QDoubleSpinBox()
        self.comp1_s_spin.setRange(-2.0, 2.0)
        self.comp1_s_spin.setDecimals(4)
        comp_layout.addRow("Component 1 S:", self.comp1_s_spin)

        self.comp2_g_spin = QDoubleSpinBox()
        self.comp2_g_spin.setRange(-2.0, 2.0)
        self.comp2_g_spin.setDecimals(4)
        comp_layout.addRow("Component 2 G:", self.comp2_g_spin)
        self.comp2_s_spin = QDoubleSpinBox()
        self.comp2_s_spin.setRange(-2.0, 2.0)
        self.comp2_s_spin.setDecimals(4)
        comp_layout.addRow("Component 2 S:", self.comp2_s_spin)

        comp_section.add_widget(comp_body)
        self.components_enable.toggled.connect(
            lambda checked: comp_section.set_content_visible(checked)
        )
        manual_layout.addWidget(self.components_enable)
        manual_layout.addWidget(comp_section)

        parent_layout.addWidget(self.manual_container)

    def _build_copy_pipeline(self, parent_layout):
        self.copy_container = QWidget()
        copy_layout = QVBoxLayout(self.copy_container)
        copy_layout.setContentsMargins(0, 0, 0, 0)

        copy_layout.addWidget(
            QLabel(
                "Copy the analysis settings (frequency, filter/threshold, "
                "calibration) stored in a reference layer or OME-TIFF and "
                "apply them to every file."
            )
        )

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Reference layer:"))
        self.copy_reference_combo = QComboBox()
        source_row.addWidget(self.copy_reference_combo, 1)
        copy_layout.addLayout(source_row)

        file_row = QHBoxLayout()
        self.copy_file_edit = QLineEdit()
        self.copy_file_edit.setPlaceholderText("…or OME-TIFF settings source")
        copy_browse = QPushButton("Browse")
        copy_browse.clicked.connect(self._on_browse_copy_file)
        file_row.addWidget(self.copy_file_edit, 1)
        file_row.addWidget(copy_browse)
        copy_layout.addLayout(file_row)

        parent_layout.addWidget(self.copy_container)

    def _build_export_section(self, layout):
        section = CollapsibleSection("4. Export", initially_collapsed=False)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        out_row = QHBoxLayout()
        self.export_folder_button = QPushButton("Select export folder")
        self.export_folder_button.clicked.connect(self._on_select_export)
        out_row.addWidget(self.export_folder_button)
        self.export_folder_label = QLabel("<i>No folder selected</i>")
        self.export_folder_label.setWordWrap(True)
        out_row.addWidget(self.export_folder_label, 1)
        body_layout.addLayout(out_row)
        self._export_folder = None

        types_row = QHBoxLayout()
        types_row.addWidget(QLabel("Export as:"))
        self.export_ometiff_checkbox = QCheckBox("OME-TIFF")
        self.export_ometiff_checkbox.setChecked(True)
        self.export_csv_checkbox = QCheckBox("CSV")
        self.export_image_checkbox = QCheckBox("Image (PNG)")
        types_row.addWidget(self.export_ometiff_checkbox)
        types_row.addWidget(self.export_csv_checkbox)
        types_row.addWidget(self.export_image_checkbox)
        types_row.addStretch()
        body_layout.addLayout(types_row)

        self.preserve_paths_checkbox = QCheckBox(
            "Preserve relative subfolder structure"
        )
        self.preserve_paths_checkbox.setChecked(True)
        body_layout.addWidget(self.preserve_paths_checkbox)

        suffix_row = QHBoxLayout()
        suffix_row.addWidget(QLabel("Filename suffix:"))
        self.suffix_edit = QLineEdit("_analyzed")
        suffix_row.addWidget(self.suffix_edit, 1)
        body_layout.addLayout(suffix_row)

        self.load_into_viewer_checkbox = QCheckBox(
            "Also load results into the viewer"
        )
        body_layout.addWidget(self.load_into_viewer_checkbox)

        section.add_widget(body)
        layout.addWidget(section)

    def _build_run_section(self, layout):
        self.run_button = QPushButton("Run batch analysis")
        self.run_button.clicked.connect(self.run_batch)
        layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # -- Window behaviour --------------------------------------------------

    def showEvent(self, event):
        """Float the dock widget on first show and center it on screen."""
        super().showEvent(event)
        if not self._floated:
            self._floated = True
            parent = self.parent()
            while parent is not None:
                if isinstance(parent, QDockWidget):
                    parent.setFloating(True)
                    screen = QApplication.primaryScreen().geometry()
                    dw_size = parent.sizeHint()
                    parent.move(
                        screen.center().x() - dw_size.width() // 2,
                        screen.center().y() - dw_size.height() // 2,
                    )
                    break
                parent = parent.parent()

    # -- Callbacks ---------------------------------------------------------

    def _populate_layer_comboboxes(self):
        names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and all(
                key in layer.metadata
                for key in ["G", "S", "G_original", "S_original"]
            )
        ]
        for combo in (
            getattr(self, "calib_reference_combo", None),
            getattr(self, "copy_reference_combo", None),
        ):
            if combo is None:
                continue
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def _on_select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if not folder:
            return
        self._input_folder = folder
        self.folder_label.setText(folder)
        self._rescan()

    def _rescan(self):
        if not self._input_folder:
            return
        self._scanned = scan_folder(
            self._input_folder, self.subfolders_checkbox.isChecked()
        )
        self.format_combobox.blockSignals(True)
        self.format_combobox.clear()
        for ext in sorted(self._scanned):
            count = len(self._scanned[ext])
            self.format_combobox.addItem(f"{ext} ({count} files)", ext)
        self.format_combobox.blockSignals(False)
        if self.format_combobox.count():
            self.format_combobox.setCurrentIndex(0)
            self._on_format_changed()
        else:
            self.status_label.setText(
                "No supported files found in the selected folder."
            )

    def _on_format_changed(self):
        ext = self.format_combobox.currentData()
        if ext:
            self.read_options_widget.set_extension(ext)

    def _on_pipeline_mode_changed(self):
        manual = self.manual_mode_radio.isChecked()
        self.manual_container.setVisible(manual)
        self.copy_container.setVisible(not manual)

    def _on_browse_calibration_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OME-TIFF reference",
            "",
            "OME-TIFF Files (*.ome.tif *.ome.tiff)",
        )
        if path:
            self.calib_file_edit.setText(path)

    def _on_browse_copy_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OME-TIFF settings source",
            "",
            "OME-TIFF Files (*.ome.tif *.ome.tiff)",
        )
        if path:
            self.copy_file_edit.setText(path)

    def _on_select_export(self):
        folder = QFileDialog.getExistingDirectory(self, "Select export folder")
        if folder:
            self._export_folder = folder
            self.export_folder_label.setText(folder)

    # -- Pipeline building -------------------------------------------------

    def _load_reference_layer(self, path, harmonics):
        """Load the first phasor layer from a file as an in-memory Image."""
        reader = napari_get_reader(
            path, reader_options={}, harmonics=harmonics
        )
        if reader is None:
            raise ValueError(f"No reader for reference file: {path}")
        layer_data = reader(path)
        data, add_kw = layer_data[0][0], layer_data[0][1]
        return Image(
            data,
            name=add_kw.get("name", os.path.basename(path)),
            metadata=add_kw.get("metadata", {}),
        )

    def build_pipeline(self, harmonics):
        """Resolve the configured pipeline into a :class:`BatchPipeline`."""
        if self.copy_mode_radio.isChecked():
            return self._build_copy_pipeline_object(harmonics)
        return self._build_manual_pipeline_object(harmonics)

    def _build_manual_pipeline_object(self, harmonics):
        pipeline = BatchPipeline()

        if self.calibration_enable.isChecked():
            ref_layer = None
            file_path = self.calib_file_edit.text().strip()
            if file_path:
                ref_layer = self._load_reference_layer(file_path, harmonics)
            else:
                name = self.calib_reference_combo.currentText()
                if name and name in self.viewer.layers:
                    ref_layer = self.viewer.layers[name]
            if ref_layer is None:
                raise ValueError(
                    "Calibration is enabled but no reference layer or file "
                    "was selected."
                )
            frequency = self.calib_frequency_spin.value()
            lifetime = self.calib_lifetime_spin.value()
            phi_zero, mod_zero = compute_calibration_parameters(
                ref_layer, frequency, lifetime
            )
            pipeline.calibration = {
                "phi_zero": phi_zero,
                "mod_zero": mod_zero,
            }

        if self.filter_enable.isChecked():
            pipeline.filter = self._collect_filter_kwargs()

        if self.components_enable.isChecked():
            harmonic = (
                harmonics[0]
                if isinstance(harmonics, list) and harmonics
                else 1
            )
            pipeline.components = {
                "name": self.comp_name_edit.text().strip() or "Component 1",
                "component_real": (
                    self.comp1_g_spin.value(),
                    self.comp2_g_spin.value(),
                ),
                "component_imag": (
                    self.comp1_s_spin.value(),
                    self.comp2_s_spin.value(),
                ),
                "harmonic": harmonic,
            }

        return pipeline

    def _collect_filter_kwargs(self):
        kwargs = {}
        method = self.filter_method_combo.currentText()
        if method == "Median":
            kwargs["filter_method"] = "median"
            kwargs["size"] = self.median_size_spin.value()
            kwargs["repeat"] = self.median_repeat_spin.value()
        elif method == "Wavelet":
            kwargs["filter_method"] = "wavelet"
            kwargs["sigma"] = self.wavelet_sigma_spin.value()
            kwargs["levels"] = self.wavelet_levels_spin.value()

        threshold_method = self.threshold_method_combo.currentText()
        if threshold_method != "None":
            kwargs["threshold_method"] = threshold_method.lower()
            if threshold_method == "Manual":
                kwargs["threshold"] = self.threshold_min_spin.value()
                if self.threshold_max_spin.value() > 0:
                    kwargs["threshold_upper"] = self.threshold_max_spin.value()
        return kwargs

    def _build_copy_pipeline_object(self, harmonics):
        settings = None
        file_path = self.copy_file_edit.text().strip()
        if file_path:
            settings = read_ome_tiff_settings(file_path)
        else:
            name = self.copy_reference_combo.currentText()
            if name and name in self.viewer.layers:
                settings = self.viewer.layers[name].metadata.get(
                    "settings", {}
                )
        if not settings:
            raise ValueError(
                "Copy-settings mode is selected but no reference layer or "
                "file with settings was provided."
            )

        pipeline = BatchPipeline()

        if settings.get("calibrated") and (
            "calibration_phase" in settings
            and "calibration_modulation" in settings
        ):
            pipeline.calibration = {
                "phi_zero": np.asarray(settings["calibration_phase"]),
                "mod_zero": np.asarray(settings["calibration_modulation"]),
            }

        filter_settings = settings.get("filter") or {}
        filter_kwargs = {}
        if filter_settings.get("method"):
            filter_kwargs["filter_method"] = filter_settings["method"]
            for src, dst in (
                ("size", "size"),
                ("repeat", "repeat"),
                ("sigma", "sigma"),
                ("levels", "levels"),
            ):
                if filter_settings.get(src) is not None:
                    filter_kwargs[dst] = filter_settings[src]
        for key in ("threshold", "threshold_upper", "threshold_method"):
            if settings.get(key) is not None:
                filter_kwargs[key] = settings[key]
        if filter_kwargs:
            pipeline.filter = filter_kwargs

        return pipeline

    # -- Batch execution ---------------------------------------------------

    def run_batch(self):
        """Run the configured pipeline over every file of the chosen format."""
        ext = self.format_combobox.currentData()
        if not ext or ext not in self._scanned:
            show_error("Select an input folder with supported files.")
            return
        files = self._scanned[ext]
        if not files:
            show_error("No files of the selected format were found.")
            return
        if not self._export_folder:
            show_error("Select an export folder.")
            return

        output_types = []
        if self.export_ometiff_checkbox.isChecked():
            output_types.append("ome.tif")
        if self.export_csv_checkbox.isChecked():
            output_types.append("csv")
        if self.export_image_checkbox.isChecked():
            output_types.append("png")
        if not output_types:
            show_error("Select at least one export format.")
            return

        try:
            harmonics = parse_harmonics(self.harmonics_edit.text())
        except ValueError:
            show_error("Invalid harmonics value.")
            return

        reader_options = self.read_options_widget.get_reader_options()

        try:
            pipeline = self.build_pipeline(harmonics)
        except ValueError as exc:
            show_error(str(exc))
            return

        suffix = self.suffix_edit.text()
        preserve = self.preserve_paths_checkbox.isChecked()
        load_into_viewer = self.load_into_viewer_checkbox.isChecked()

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(files))
        self.progress_bar.setValue(0)

        processed = 0
        failed = []
        for index, path in enumerate(files):
            try:
                self._process_file(
                    path,
                    ext,
                    reader_options,
                    harmonics,
                    pipeline,
                    output_types,
                    suffix,
                    preserve,
                    load_into_viewer,
                )
                processed += 1
            except Exception as exc:  # noqa: BLE001
                failed.append((os.path.basename(path), str(exc)))
            self.progress_bar.setValue(index + 1)
            self.status_label.setText(
                f"Processed {index + 1}/{len(files)} files…"
            )
            QApplication.processEvents()

        self.progress_bar.setVisible(False)
        summary = f"Batch complete: {processed}/{len(files)} files processed."
        if failed:
            names = ", ".join(name for name, _ in failed)
            summary += f" Failed: {names}."
            show_error(summary)
        else:
            show_info(summary)
        self.status_label.setText(summary)

    def _process_file(
        self,
        path,
        ext,
        reader_options,
        harmonics,
        pipeline,
        output_types,
        suffix,
        preserve,
        load_into_viewer,
    ):
        reader = napari_get_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
        if reader is None:
            raise ValueError("no reader")

        for layer_data in reader(path):
            data = layer_data[0]
            add_kw = dict(layer_data[1])
            layer = Image(
                data,
                name=add_kw.get("name", os.path.basename(path)),
                metadata=add_kw.get("metadata", {}),
            )
            extra_layers = apply_pipeline(layer, pipeline)

            base_out = self._derive_output_path(path, ext, suffix, preserve)
            self._export_layer(layer, base_out, output_types)
            for extra in extra_layers:
                extra_out = f"{base_out}_{_safe_suffix(extra.name)}"
                self._export_layer(extra, extra_out, output_types)

            if load_into_viewer:
                self.viewer.add_image(
                    layer.data, name=layer.name, metadata=layer.metadata
                )
                for extra in extra_layers:
                    self.viewer.add_image(extra.data, name=extra.name)

    def _derive_output_path(self, src_path, ext, suffix, preserve):
        """Return the output path *without* extension for ``src_path``."""
        filename = os.path.basename(src_path)
        stem = (
            filename[: -len(ext)]
            if filename.lower().endswith(ext)
            else (os.path.splitext(filename)[0])
        )
        if preserve and self._input_folder:
            rel_dir = os.path.relpath(
                os.path.dirname(src_path), self._input_folder
            )
            target_dir = os.path.normpath(
                os.path.join(self._export_folder, rel_dir)
            )
        else:
            target_dir = self._export_folder
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, f"{stem}{suffix}")

    def _export_layer(self, layer, base_path, output_types):
        for out_type in output_types:
            target = f"{base_path}.{out_type}"
            if out_type == "ome.tif":
                write_ome_tiff(target, layer)
            elif out_type == "csv":
                export_layer_as_csv(target, layer)
            elif out_type == "png":
                export_layer_as_image(
                    target,
                    layer,
                    include_colorbar=False,
                )


def _safe_suffix(name):
    """Turn a layer name into a filesystem-safe filename fragment."""
    keep = []
    for ch in name:
        keep.append(ch if (ch.isalnum() or ch in "-_") else "_")
    return "".join(keep)
