import csv

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QComboBox, QLabel

from napari_phasors._utils import (
    CheckableComboBox,
    CollapsibleSection,
    FileOrderDialog,
    HistogramDockWidget,
    HistogramWidget,
    StatisticsDockWidget,
    StatisticsTableWidget,
    _check_state_value,
    _ColormapDelegate,
    _PrimaryLayerDelegate,
    create_colormap_icon,
    populate_colormap_combobox,
)


def test_histogram_widget_update_data_and_clear(qtbot):
    """HistogramWidget should compute bins and reset cleanly."""
    widget = HistogramWidget(bins=8)
    qtbot.addWidget(widget)

    # Include invalid values to verify filtering behavior.
    data = np.array([1.0, 2.0, 3.0, np.nan, np.inf, -1.0, 0.0])
    widget.update_data(data)

    assert widget.counts is not None
    assert widget.bin_edges is not None
    assert widget.bin_centers is not None
    assert len(widget.counts) == 8
    assert 0.0 in widget._raw_valid_data
    assert -1.0 in widget._raw_valid_data
    assert np.all(np.isfinite(widget._raw_valid_data))
    assert widget._settings_button.isEnabled()
    assert widget.save_png_button.isEnabled()
    assert widget.save_csv_button.isEnabled()

    widget.clear()

    assert widget.counts is None
    assert widget.bin_edges is None
    assert widget.bin_centers is None
    assert widget._datasets == {}
    assert widget._raw_valid_data is None
    assert not widget._settings_button.isEnabled()
    assert not widget.save_png_button.isEnabled()
    assert not widget.save_csv_button.isEnabled()


def test_histogram_widget_save_csv(qtbot, tmp_path, monkeypatch):
    """HistogramWidget should export CSV accurately in all modes."""
    from qtpy.QtWidgets import QFileDialog

    widget = HistogramWidget(bins=2)
    qtbot.addWidget(widget)

    # Put some data
    data1 = np.array([1.0, 1.0, 2.0])
    data2 = np.array([1.0, 2.0, 2.0])

    csv_file = tmp_path / "test.csv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(csv_file), ""),
    )

    # 1. Test Merged (default) mode with multiple datasets
    widget.update_multi_data({"Layer 1": data1, "Layer 2": data2})
    widget._display_mode = "Merged"
    widget._save_histogram_csv()

    assert csv_file.exists()

    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[0] == ['Bin Center', 'Mean Counts', 'Std Counts']
    assert len(rows) == 3  # Header + 2 bins

    # 2. Test Grouped mode
    widget._display_mode = "Grouped"
    widget._group_assignments = {"Layer 1": 1, "Layer 2": 1}
    widget._group_names = {1: "MyGroup"}
    widget._save_histogram_csv()

    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[0] == ['Bin Center', 'MyGroup Mean', 'MyGroup Std']
    assert len(rows) == 3

    # 3. Test Individual mode
    widget._display_mode = "Individual layers"
    widget._save_histogram_csv()

    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[0] == ['Bin Center', 'Layer 1', 'Layer 2']
    assert len(rows) == 3


def test_histogram_widget_default_filter_keeps_zero(qtbot):
    """Default filtering should keep zero values (drop only NaN/Inf)."""
    widget = HistogramWidget(bins=5)
    qtbot.addWidget(widget)
    widget.update_data(np.array([0.0, 0.25, np.nan, np.inf]))

    assert widget.counts is not None
    assert np.any(np.isclose(widget._raw_valid_data, 0.0))


def test_histogram_widget_exclude_nonpositive_option(qtbot):
    """Optional non-positive filtering should preserve old behavior."""
    widget = HistogramWidget(bins=5, exclude_nonpositive=True)
    qtbot.addWidget(widget)
    widget.update_data(np.array([-1.0, 0.0, 0.25, np.nan, np.inf]))

    assert widget.counts is not None
    assert np.all(widget._raw_valid_data > 0)
    assert np.allclose(widget._raw_valid_data, np.array([0.25]))


def test_histogram_widget_update_multi_data_autosd(qtbot):
    """Multi-dataset updates should auto-enable SD shading when needed."""
    widget = HistogramWidget(bins=10)
    qtbot.addWidget(widget)

    datasets = {
        "Layer A": np.array([1, 2, 3, 4], dtype=float),
        "Layer B": np.array([2, 3, 4, 5], dtype=float),
    }
    widget.update_multi_data(datasets)

    assert set(widget._datasets.keys()) == {"Layer A", "Layer B"}
    assert set(widget._counts_per_dataset.keys()) == {"Layer A", "Layer B"}
    assert widget._show_sd is True


def test_statistics_table_widget_populates_rows(qtbot):
    """StatisticsTableWidget should populate rows for each dataset."""
    table = StatisticsTableWidget()
    qtbot.addWidget(table)

    datasets = {
        "Layer A": np.array([1.0, 2.0, 3.0]),
        "Layer B": np.array([2.0, 4.0, 6.0]),
    }
    bin_edges = np.linspace(1.0, 6.0, 6)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    table.update_statistics(
        datasets, bin_centers=bin_centers, bin_edges=bin_edges
    )

    assert table.rowCount() == 2
    assert table.columnCount() == len(StatisticsTableWidget.COLUMNS)
    row_names = {table.item(row, 0).text() for row in range(table.rowCount())}
    assert row_names == {"Layer A", "Layer B"}


def test_statistics_dock_widget_updates_for_single_and_grouped_data(qtbot):
    """StatisticsDockWidget should toggle sections by histogram mode/data."""
    histogram_widget = HistogramWidget(bins=12)
    qtbot.addWidget(histogram_widget)
    stats_dock = StatisticsDockWidget(histogram_widget)
    qtbot.addWidget(stats_dock)

    histogram_widget.update_data(np.array([1.0, 2.0, 3.0, 4.0]))
    assert not stats_dock.layer_stats_section.isHidden()
    assert stats_dock.group_stats_section.isHidden()
    assert stats_dock.export_csv_button.isEnabled()
    assert stats_dock.layer_stats_table.rowCount() == 1

    histogram_widget.update_multi_data(
        {
            "Layer A": np.array([1.0, 2.0, 3.0]),
            "Layer B": np.array([2.0, 3.0, 4.0]),
        }
    )
    histogram_widget.display_mode = "Grouped"
    histogram_widget._group_assignments = {"Layer A": 1, "Layer B": 2}
    histogram_widget._group_names = {1: "Group 1", 2: "Group 2"}
    stats_dock._update_statistics()

    assert not stats_dock.group_stats_section.isHidden()
    assert stats_dock.group_stats_table.rowCount() == 2


def test_histogram_dock_widget_links_statistics_dock(qtbot):
    """HistogramDockWidget should store a linked statistics dock."""
    histogram_widget = HistogramWidget()
    qtbot.addWidget(histogram_widget)
    hist_dock = HistogramDockWidget(histogram_widget)
    qtbot.addWidget(hist_dock)
    stats_dock = StatisticsDockWidget(histogram_widget)
    qtbot.addWidget(stats_dock)

    hist_dock.link_statistics_dock(stats_dock)

    assert hist_dock._stats_dock is stats_dock


def test_phasor_center_statistics_widget_update(qtbot):
    """PhasorCenterStatisticsWidget table should populate and be formatted correctly."""
    from napari_phasors.plotter import PhasorCenterStatisticsWidget

    widget = PhasorCenterStatisticsWidget()
    qtbot.addWidget(widget)

    # name -> (G, S, phase, mod)
    data = {"Layer X": (0.5, 0.4, 45.0, 0.7071)}
    widget.update_centers(data)

    table = widget._layer_table
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == "Layer X"
    assert table.item(0, 1).text() == "0.500000"
    assert table.item(0, 2).text() == "0.400000"
    assert table.item(0, 3).text() == "45.0000"
    assert table.item(0, 4).text() == "0.707100"


def test_histogram_widget_respects_range_slider_limits(qtbot):
    """HistogramWidget axes should match the range slider even if data is outside."""
    # Use explicit range_factor for easy testing
    widget = HistogramWidget(range_slider_enabled=True, range_factor=100)
    qtbot.addWidget(widget)

    # Data from 0 to 1
    data = np.array([0.1, 0.5, 0.9])

    # Set range slider to 2 to 3 (outside data) using set_range helper
    # We must also update the slider's absolute limits (slider_min/max)
    # otherwise it will be clipped to the default 0-100 (which is 1.0 with factor 100)
    widget.set_range(2.0, 3.0, slider_min=0.0, slider_max=5.0)
    widget.update_data(data)

    # X-axis should be exactly 2 to 3
    xlim = widget.ax.get_xlim()
    assert xlim[0] == 2.0
    assert xlim[1] == 3.0


# ------------------------------------------------------------------
# _check_state_value
# ------------------------------------------------------------------


def test_check_state_value_int():
    """_check_state_value returns int for plain int values."""
    assert _check_state_value(2) == 2
    assert _check_state_value(0) == 0


def test_check_state_value_qt_enum():
    """_check_state_value handles Qt.Checked / Qt.Unchecked."""
    assert _check_state_value(Qt.Checked) == int(Qt.Checked)
    assert _check_state_value(Qt.Unchecked) == int(Qt.Unchecked)


# ------------------------------------------------------------------
# CheckableComboBox
# ------------------------------------------------------------------


def test_checkable_combobox_add_items(qtbot):
    """Adding items populates the model."""
    combo = CheckableComboBox()
    qtbot.addWidget(combo)

    combo.addItem('Layer A')
    combo.addItem('Layer B')
    combo.addItem('Layer C')

    assert combo.model().rowCount() == 3
    assert combo.allItems() == ['Layer A', 'Layer B', 'Layer C']


def test_checkable_combobox_add_items_batch(qtbot):
    """addItems populates the model in batch."""
    combo = CheckableComboBox()
    qtbot.addWidget(combo)

    combo.addItems(['L1', 'L2', 'L3'])
    assert combo.allItems() == ['L1', 'L2', 'L3']


def test_checkable_combobox_check_uncheck(qtbot):
    """Checking and unchecking items updates checkedItems."""
    combo = CheckableComboBox()
    qtbot.addWidget(combo)

    combo.addItem('A')
    combo.addItem('B')
    combo.addItem('C')

    combo.setItemCheckState(0, Qt.Checked)
    combo.setItemCheckState(2, Qt.Checked)

    assert combo.checkedItems() == ['A', 'C']

    combo.setItemCheckState(0, Qt.Unchecked)
    assert combo.checkedItems() == ['C']


def test_checkable_combobox_checked_items_empty(qtbot):
    """No items checked by default."""
    combo = CheckableComboBox()
    qtbot.addWidget(combo)

    combo.addItems(['X', 'Y'])
    assert combo.checkedItems() == []


def test_checkable_combobox_clear(qtbot):
    """Clearing removes all items."""
    combo = CheckableComboBox()
    qtbot.addWidget(combo)

    combo.addItems(['A', 'B'])
    combo.clear()

    assert combo.model().rowCount() == 0
    assert combo.allItems() == []
    assert combo.checkedItems() == []


def test_checkable_combobox_select_all(qtbot):
    """selectAll checks every item."""
    combo = CheckableComboBox()
    qtbot.addWidget(combo)

    combo.addItems(['A', 'B', 'C'])
    combo.selectAll()

    assert combo.checkedItems() == ['A', 'B', 'C']


def test_checkable_combobox_deselect_all(qtbot):
    """deselectAll unchecks every item."""
    combo = CheckableComboBox()
    qtbot.addWidget(combo)

    combo.addItems(['A', 'B', 'C'])
    combo.selectAll()
    combo.deselectAll()

    assert combo.checkedItems() == []


def test_checkable_combobox_set_checked_items(qtbot):
    """setCheckedItems checks only specified items."""
    combo = CheckableComboBox()
    qtbot.addWidget(combo)

    combo.addItems(['A', 'B', 'C', 'D'])
    combo.setCheckedItems(['B', 'D'])

    assert combo.checkedItems() == ['B', 'D']


def test_checkable_combobox_item_check_state(qtbot):
    """itemCheckState returns the correct state."""
    combo = CheckableComboBox()
    qtbot.addWidget(combo)

    combo.addItem('X')
    assert combo.itemCheckState(0) == Qt.Unchecked

    combo.setItemCheckState(0, Qt.Checked)
    assert combo.itemCheckState(0) == Qt.Checked


def test_checkable_combobox_primary_layer(qtbot):
    """Primary layer is set to the first checked item."""
    combo = CheckableComboBox(enable_primary_layer=True)
    qtbot.addWidget(combo)

    combo.addItems(['A', 'B', 'C'])
    combo.setCheckedItems(['B', 'C'])

    assert combo.getPrimaryLayer() == 'B'


def test_checkable_combobox_set_primary_layer(qtbot):
    """setPrimaryLayer changes the primary to a checked item."""
    combo = CheckableComboBox(enable_primary_layer=True)
    qtbot.addWidget(combo)

    combo.addItems(['A', 'B', 'C'])
    combo.setCheckedItems(['A', 'B', 'C'])
    combo.setPrimaryLayer('C')

    assert combo.getPrimaryLayer() == 'C'


def test_checkable_combobox_primary_layer_unchecked_fallback(qtbot):
    """Primary layer falls back to first checked if current is unchecked."""
    combo = CheckableComboBox(enable_primary_layer=True)
    qtbot.addWidget(combo)

    combo.addItems(['A', 'B', 'C'])
    combo.setCheckedItems(['A', 'B'])
    combo.setPrimaryLayer('A')

    # Uncheck 'A' — primary should fall back
    combo.setItemCheckState(0, Qt.Unchecked)
    assert combo.getPrimaryLayer() == 'B'


def test_checkable_combobox_primary_layer_disabled(qtbot):
    """Without primary layer, display shows count only."""
    combo = CheckableComboBox(enable_primary_layer=False)
    qtbot.addWidget(combo)

    combo.addItems(['A', 'B', 'C'])
    combo.setCheckedItems(['A', 'B', 'C'])

    text = combo.lineEdit().text()
    assert '3' in text
    assert 'selected' in text.lower()


def test_checkable_combobox_display_text_single(qtbot):
    """Single selection shows the item name directly."""
    combo = CheckableComboBox()
    qtbot.addWidget(combo)

    combo.addItems(['Alpha', 'Beta'])
    combo.setCheckedItems(['Alpha'])

    assert combo.lineEdit().text() == 'Alpha'


def test_checkable_combobox_display_text_empty(qtbot):
    """No selection shows placeholder."""
    combo = CheckableComboBox(placeholder='Pick...')
    qtbot.addWidget(combo)

    combo.addItems(['A', 'B'])
    assert combo.lineEdit().text() == ''
    assert combo.lineEdit().placeholderText() == 'Pick...'


def test_checkable_combobox_show_hide_popup(qtbot):
    """showPopup / hidePopup update popup visibility tracking."""
    combo = CheckableComboBox()
    qtbot.addWidget(combo)
    combo.addItems(['A'])

    combo.showPopup()
    assert combo._popup_visible is True

    combo.hidePopup()
    assert combo._popup_visible is False


def test_checkable_combobox_primary_layer_changed_signal(qtbot):
    """primaryLayerChanged signal is emitted on primary change."""
    combo = CheckableComboBox(enable_primary_layer=True)
    qtbot.addWidget(combo)
    combo.addItems(['A', 'B', 'C'])

    received = []
    combo.primaryLayerChanged.connect(lambda name: received.append(name))

    combo.setCheckedItems(['A', 'B'])
    combo.setPrimaryLayer('B')

    assert 'B' in received


def test_checkable_combobox_selection_changed_signal(qtbot):
    """selectionChanged signal fires on check state change."""
    combo = CheckableComboBox()
    qtbot.addWidget(combo)
    combo.addItems(['A', 'B'])

    received = []
    combo.selectionChanged.connect(lambda: received.append(True))

    combo.setItemCheckState(0, Qt.Checked)
    assert len(received) > 0


# ------------------------------------------------------------------
# FileOrderDialog
# ------------------------------------------------------------------


def test_file_order_dialog_init(qtbot):
    """FileOrderDialog shows files in order."""
    paths = ['/a/file1.tif', '/a/file2.tif', '/a/file3.tif']
    dialog = FileOrderDialog(paths)
    qtbot.addWidget(dialog)

    assert dialog.file_list.count() == 3
    assert dialog.get_ordered_paths() == paths


def test_file_order_dialog_move_up(qtbot):
    """Move up reorders the file list."""
    paths = ['/a/file1.tif', '/a/file2.tif', '/a/file3.tif']
    dialog = FileOrderDialog(paths)
    qtbot.addWidget(dialog)

    # Select second item and move it up
    dialog.file_list.setCurrentRow(1)
    dialog._move_up()

    result = dialog.get_ordered_paths()
    assert result == ['/a/file2.tif', '/a/file1.tif', '/a/file3.tif']


def test_file_order_dialog_move_down(qtbot):
    """Move down reorders the file list."""
    paths = ['/a/file1.tif', '/a/file2.tif', '/a/file3.tif']
    dialog = FileOrderDialog(paths)
    qtbot.addWidget(dialog)

    # Select first item and move it down
    dialog.file_list.setCurrentRow(0)
    dialog._move_down()

    result = dialog.get_ordered_paths()
    assert result == ['/a/file2.tif', '/a/file1.tif', '/a/file3.tif']


def test_file_order_dialog_move_up_first_item(qtbot):
    """Moving the first item up is a no-op."""
    paths = ['/a/file1.tif', '/a/file2.tif']
    dialog = FileOrderDialog(paths)
    qtbot.addWidget(dialog)

    dialog.file_list.setCurrentRow(0)
    dialog._move_up()

    assert dialog.get_ordered_paths() == paths


def test_file_order_dialog_move_down_last_item(qtbot):
    """Moving the last item down is a no-op."""
    paths = ['/a/file1.tif', '/a/file2.tif']
    dialog = FileOrderDialog(paths)
    qtbot.addWidget(dialog)

    dialog.file_list.setCurrentRow(1)
    dialog._move_down()

    assert dialog.get_ordered_paths() == paths


def test_file_order_dialog_z_spacing_default(qtbot):
    """Default z spacing is 1.0."""
    dialog = FileOrderDialog(['/a/file.tif'])
    qtbot.addWidget(dialog)

    assert dialog.get_z_spacing() == 1.0


def test_file_order_dialog_z_spacing_custom(qtbot):
    """Custom initial z spacing is used."""
    dialog = FileOrderDialog(['/a/file.tif'], initial_z_spacing=2.5)
    qtbot.addWidget(dialog)

    assert dialog.get_z_spacing() == 2.5


def test_file_order_dialog_z_spacing_invalid(qtbot):
    """Invalid z spacing text falls back to 1.0."""
    dialog = FileOrderDialog(['/a/file.tif'])
    qtbot.addWidget(dialog)

    dialog.z_spacing_edit.setText('abc')
    assert dialog.get_z_spacing() == 1.0


def test_file_order_dialog_z_spacing_negative(qtbot):
    """Negative z spacing falls back to 1.0."""
    dialog = FileOrderDialog(['/a/file.tif'])
    qtbot.addWidget(dialog)

    dialog.z_spacing_edit.setText('-5')
    assert dialog.get_z_spacing() == 1.0


def test_file_order_dialog_axis_labels(qtbot):
    """Axis labels are parsed from the edit field."""
    dialog = FileOrderDialog(['/a/file.tif'], estimated_shape=(5, 256, 256))
    qtbot.addWidget(dialog)

    dialog.axis_labels_edit.setText('Z, Y, X')
    labels = dialog.get_axis_labels()
    assert labels == ['Z', 'Y', 'X']


def test_file_order_dialog_axis_labels_empty(qtbot):
    """Empty axis labels returns None."""
    dialog = FileOrderDialog(['/a/file.tif'])
    qtbot.addWidget(dialog)

    dialog.axis_labels_edit.setText('')
    assert dialog.get_axis_labels() is None


def test_file_order_dialog_axis_order(qtbot):
    """get_axis_order returns None (not configurable)."""
    dialog = FileOrderDialog(['/a/file.tif'])
    qtbot.addWidget(dialog)

    assert dialog.get_axis_order() is None


def test_file_order_dialog_estimated_shape(qtbot):
    """Estimated shape is displayed in the label."""
    dialog = FileOrderDialog(['/a/file.tif'], estimated_shape=(3, 256, 256))
    qtbot.addWidget(dialog)

    assert '(3, 256, 256)' in dialog.shape_label.text()


def test_file_order_dialog_default_axis_labels_3d(qtbot):
    """Default axis labels are Z, Y, X for 3D shape."""
    dialog = FileOrderDialog(['/a/file.tif'], estimated_shape=(5, 128, 128))
    qtbot.addWidget(dialog)

    assert dialog.axis_labels_edit.text() == 'Z, Y, X'


def test_file_order_dialog_default_axis_labels_2d(qtbot):
    """Default axis labels are Y, X for 2D shape."""
    dialog = FileOrderDialog(['/a/file.tif'], estimated_shape=(128, 128))
    qtbot.addWidget(dialog)

    assert dialog.axis_labels_edit.text() == 'Y, X'


# ------------------------------------------------------------------
# CollapsibleSection
# ------------------------------------------------------------------


def test_collapsible_section_initially_collapsed(qtbot):
    """Content is hidden when initially_collapsed=True."""
    section = CollapsibleSection(title='Test', initially_collapsed=True)
    qtbot.addWidget(section)

    assert section._content.isHidden()
    assert not section._toggle_button.isChecked()
    assert '\u25b6' in section._toggle_button.text()


def test_collapsible_section_initially_expanded(qtbot):
    """Content is not hidden when initially_collapsed=False."""
    section = CollapsibleSection(title='Test', initially_collapsed=False)
    qtbot.addWidget(section)

    assert not section._content.isHidden()
    assert section._toggle_button.isChecked()
    assert '\u25bc' in section._toggle_button.text()


def test_collapsible_section_toggle(qtbot):
    """Clicking the toggle button expands/collapses content."""
    section = CollapsibleSection(title='Test', initially_collapsed=True)
    qtbot.addWidget(section)

    # Expand
    section._toggle_button.setChecked(True)
    section._on_toggle()
    assert not section._content.isHidden()

    # Collapse
    section._toggle_button.setChecked(False)
    section._on_toggle()
    assert section._content.isHidden()


def test_collapsible_section_add_widget(qtbot):
    """add_widget adds a child to the content area."""
    section = CollapsibleSection(title='Test')
    qtbot.addWidget(section)

    label = QLabel('Hello')
    section.add_widget(label)

    assert section._content_layout.count() == 1


def test_collapsible_section_set_content_visible(qtbot):
    """set_content_visible programmatically expands/collapses."""
    section = CollapsibleSection(title='Test', initially_collapsed=True)
    qtbot.addWidget(section)

    section.set_content_visible(True)
    assert not section._content.isHidden()
    assert section._toggle_button.isChecked()

    section.set_content_visible(False)
    assert section._content.isHidden()
    assert not section._toggle_button.isChecked()


def test_collapsible_section_title(qtbot):
    """Title text appears in the toggle button."""
    section = CollapsibleSection(title='My Section')
    qtbot.addWidget(section)

    assert 'My Section' in section._toggle_button.text()


# ------------------------------------------------------------------
# create_colormap_icon / populate_colormap_combobox
# ------------------------------------------------------------------


def test_create_colormap_icon_valid(qtbot):
    """create_colormap_icon returns a QIcon for a known colormap."""
    icon = create_colormap_icon('viridis')
    assert isinstance(icon, QIcon)
    assert not icon.isNull()


def test_create_colormap_icon_unknown(qtbot):
    """create_colormap_icon returns an empty icon for unknown name."""
    icon = create_colormap_icon('nonexistent_cmap_xyz')
    assert isinstance(icon, QIcon)


def test_create_colormap_icon_none(qtbot):
    """create_colormap_icon handles None gracefully."""
    icon = create_colormap_icon(None)
    assert isinstance(icon, QIcon)


def test_populate_colormap_combobox(qtbot):
    """populate_colormap_combobox fills a QComboBox with colormaps."""
    combo = QComboBox()
    qtbot.addWidget(combo)

    populate_colormap_combobox(combo, include_select_color=True)

    assert combo.count() > 1
    assert combo.itemText(0) == 'Select color...'


def test_populate_colormap_combobox_no_select_color(qtbot):
    """populate_colormap_combobox without 'Select color...' option."""
    combo = QComboBox()
    qtbot.addWidget(combo)

    populate_colormap_combobox(combo, include_select_color=False)

    assert combo.count() > 0
    assert combo.itemText(0) != 'Select color...'


def test_populate_colormap_combobox_selected(qtbot):
    """populate_colormap_combobox selects a specific colormap."""
    combo = QComboBox()
    qtbot.addWidget(combo)

    populate_colormap_combobox(
        combo,
        include_select_color=True,
        available_colormaps=['viridis', 'plasma'],
        selected='plasma',
    )

    assert combo.currentText() == 'plasma'


def test_populate_colormap_combobox_custom_list(qtbot):
    """populate_colormap_combobox with explicit colormap list."""
    combo = QComboBox()
    qtbot.addWidget(combo)

    populate_colormap_combobox(
        combo,
        include_select_color=False,
        available_colormaps=['viridis', 'plasma'],
    )

    assert combo.count() == 2
    texts = [combo.itemText(i) for i in range(combo.count())]
    assert texts == ['viridis', 'plasma']


# ------------------------------------------------------------------
# _ColormapDelegate
# ------------------------------------------------------------------


def test_colormap_delegate_init(qtbot):
    """_ColormapDelegate can be instantiated."""
    combo = QComboBox()
    qtbot.addWidget(combo)
    delegate = _ColormapDelegate(combo)
    assert delegate is not None


# ------------------------------------------------------------------
# _PrimaryLayerDelegate
# ------------------------------------------------------------------


def test_primary_layer_delegate_init(qtbot):
    """_PrimaryLayerDelegate can be instantiated."""
    combo = QComboBox()
    qtbot.addWidget(combo)
    delegate = _PrimaryLayerDelegate(combo, enable_primary_layer=True)
    assert delegate is not None
    assert delegate._enable_primary_layer is True


def test_primary_layer_delegate_disabled(qtbot):
    """_PrimaryLayerDelegate can be created with primary disabled."""
    combo = QComboBox()
    qtbot.addWidget(combo)
    delegate = _PrimaryLayerDelegate(combo, enable_primary_layer=False)
    assert delegate._enable_primary_layer is False
