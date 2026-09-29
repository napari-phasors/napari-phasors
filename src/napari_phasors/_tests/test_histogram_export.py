"""Tests for the histogram export dialog and the legend location option."""

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.legend import Legend
from PIL import Image
from qtpy.QtWidgets import QDialog, QFileDialog

from napari_phasors._utils import (
    LEGEND_POSITIONS,
    HistogramExportDialog,
    HistogramSettingsDialog,
    HistogramWidget,
    _number_or,
    default_legend_position,
    normalize_legend_location,
)


def _histogram(qtbot, mode="Individual layers"):
    """A histogram with two datasets, so a legend has something to list."""
    widget = HistogramWidget(bins=20)
    qtbot.addWidget(widget)
    rng = np.random.default_rng(0)
    widget.update_multi_data(
        {
            "first": rng.normal(1.0, 0.2, 500),
            "second": rng.normal(2.0, 0.3, 500),
        }
    )
    widget._display_mode = mode
    widget._render()
    return widget


def _select(combo, data):
    """Choose the entry of *combo* carrying *data*."""
    index = combo.findData(data)
    assert index >= 0, data
    combo.setCurrentIndex(index)


def _legend_texts(legend):
    return [text.get_text() for text in legend.texts]


# ---------------------------------------------------------------------------
# Legend location
# ---------------------------------------------------------------------------


def test_normalize_legend_location_falls_back_to_defaults():
    """Unusable stored placements and positions become the defaults."""
    assert normalize_legend_location("inside", "lower left") == (
        "inside",
        "lower left",
    )
    assert normalize_legend_location("outside", "bottom") == (
        "outside",
        "bottom",
    )
    # A position of the other placement is not valid for this one.
    assert normalize_legend_location("outside", "lower left") == (
        "outside",
        default_legend_position("outside"),
    )
    assert normalize_legend_location(None, None) == (
        "inside",
        default_legend_position("inside"),
    )
    assert normalize_legend_location("sideways", "top") == (
        "inside",
        default_legend_position("inside"),
    )


def test_settings_dialog_has_no_aspect_ratio_option(qtbot):
    """The square / rectangle choice moved to the export dialog."""
    dlg = HistogramSettingsDialog()
    qtbot.addWidget(dlg)
    assert not hasattr(dlg, "aspect_ratio_combo")

    widget = _histogram(qtbot)
    assert not hasattr(widget, "_aspect_ratio")
    # The plot always fills its canvas.
    assert widget.ax.get_box_aspect() is None


def test_settings_dialog_legend_location_defaults(qtbot):
    """The legend starts inside, at the upper right."""
    dlg = HistogramSettingsDialog(
        display_mode="Individual layers", show_legend=True
    )
    qtbot.addWidget(dlg)

    assert dlg.get_legend_placement() == "inside"
    assert dlg.get_legend_position() == "upper right"
    positions = [
        dlg.legend_position_combo.itemData(i)
        for i in range(dlg.legend_position_combo.count())
    ]
    assert positions == [key for _, key in LEGEND_POSITIONS["inside"]]


def test_settings_dialog_legend_location_from_arguments(qtbot):
    """The dialog opens on the stored placement and position."""
    dlg = HistogramSettingsDialog(
        display_mode="Individual layers",
        show_legend=True,
        legend_placement="outside",
        legend_position="bottom",
    )
    qtbot.addWidget(dlg)
    assert dlg.get_legend_placement() == "outside"
    assert dlg.get_legend_position() == "bottom"
    assert dlg.legend_position_combo.count() == len(
        LEGEND_POSITIONS["outside"]
    )

    # A position that does not exist for the placement is not shown.
    dlg = HistogramSettingsDialog(
        legend_placement="outside", legend_position="lower left"
    )
    qtbot.addWidget(dlg)
    assert dlg.get_legend_position() == default_legend_position("outside")


def test_settings_dialog_placement_change_lists_matching_positions(qtbot):
    """Switching inside/outside swaps the positions, keeping a shared one."""
    dlg = HistogramSettingsDialog(
        display_mode="Individual layers",
        show_legend=True,
        legend_position="lower left",
    )
    qtbot.addWidget(dlg)
    assert dlg.get_legend_position() == "lower left"

    _select(dlg.legend_placement_combo, "outside")
    assert dlg.get_legend_placement() == "outside"
    assert [
        dlg.legend_position_combo.itemData(i)
        for i in range(dlg.legend_position_combo.count())
    ] == [key for _, key in LEGEND_POSITIONS["outside"]]
    # "lower left" is not an outside position: the default is used.
    assert dlg.get_legend_position() == default_legend_position("outside")

    _select(dlg.legend_position_combo, "top")
    _select(dlg.legend_placement_combo, "inside")
    assert dlg.get_legend_position() == default_legend_position("inside")


def test_settings_dialog_legend_controls_follow_the_legend(qtbot):
    """The location controls are live only while a legend is drawn."""
    dlg = HistogramSettingsDialog(
        display_mode="Individual layers", show_legend=True
    )
    qtbot.addWidget(dlg)

    def controls_enabled():
        return (
            dlg.legend_placement_combo.isEnabled()
            and dlg.legend_position_combo.isEnabled()
            and dlg._legend_location_label.isEnabled()
        )

    assert controls_enabled()

    dlg.legend_checkbox.setChecked(False)
    assert not dlg.legend_placement_combo.isEnabled()
    assert not dlg.legend_position_combo.isEnabled()
    assert not dlg._legend_location_label.isEnabled()

    dlg.legend_checkbox.setChecked(True)
    assert controls_enabled()

    # Merged mode draws no legend, so the checkbox and location are off.
    dlg.mode_combo.setCurrentText("Merged")
    assert not controls_enabled()
    dlg.mode_combo.setCurrentText("Grouped")
    assert controls_enabled()


@pytest.mark.parametrize("position", ["upper left", "lower right", "center"])
def test_legend_inside_uses_the_chosen_corner(qtbot, position):
    """An inside legend is an axes legend at the chosen position."""
    widget = _histogram(qtbot)
    widget._legend_position = position
    widget._render()

    legend = widget.ax.get_legend()
    assert legend is not None
    assert legend._loc == Legend.codes[position]
    assert _legend_texts(legend) == ["first", "second"]
    assert widget.fig.legends == []


@pytest.mark.parametrize(
    "position, ncol",
    [("right", 1), ("top", 2), ("bottom", 2)],
)
def test_legend_outside_is_a_figure_legend(qtbot, position, ncol):
    """An outside legend sits on the figure, beside/above/below the axes."""
    widget = _histogram(qtbot)
    widget._legend_placement = "outside"
    widget._legend_position = position
    widget._render()

    assert widget.ax.get_legend() is None
    assert len(widget.fig.legends) == 1
    legend = widget.fig.legends[0]
    assert _legend_texts(legend) == ["first", "second"]
    assert legend._ncols == ncol

    widget.fig.canvas.draw()
    legend_box = legend.get_window_extent()
    axes_box = widget.ax.get_window_extent()
    if position == "right":
        assert legend_box.x0 >= axes_box.x1
    elif position == "top":
        assert legend_box.y0 >= axes_box.y1
    else:
        assert legend_box.y1 <= axes_box.y0


def test_legend_outside_does_not_accumulate_or_linger(qtbot):
    """Re-rendering, hiding, moving inside or clearing removes the legend."""
    widget = _histogram(qtbot)
    widget._legend_placement = "outside"
    widget._legend_position = "right"

    for _ in range(3):
        widget._render()
    assert len(widget.fig.legends) == 1

    widget._show_legend = False
    widget._render()
    assert widget.fig.legends == []

    widget._show_legend = True
    widget._render()
    assert len(widget.fig.legends) == 1

    widget._legend_placement = "inside"
    widget._legend_position = "upper right"
    widget._render()
    assert widget.fig.legends == []
    assert widget.ax.get_legend() is not None

    widget._legend_placement = "outside"
    widget._render()
    assert len(widget.fig.legends) == 1
    widget.clear()
    assert widget.fig.legends == []


def test_legend_with_nothing_to_list_is_skipped(qtbot):
    """No curve carries a label, so no legend (and no warning) is drawn."""
    widget = HistogramWidget(bins=5)
    qtbot.addWidget(widget)
    widget._draw_legend()
    assert widget.ax.get_legend() is None
    assert widget.fig.legends == []


def test_legend_placement_applies_to_grouped_and_series_curves(qtbot):
    """Grouped curves and per-series merged curves honour the placement."""
    widget = _histogram(qtbot, mode="Grouped")
    widget._group_assignments = {"first": 1, "second": 2}
    widget._legend_placement = "outside"
    widget._legend_position = "top"
    widget._render()
    assert len(widget.fig.legends) == 1
    assert len(_legend_texts(widget.fig.legends[0])) == 2

    merged = HistogramWidget(bins=20)
    qtbot.addWidget(merged)
    merged.set_dataset_series({"a": "Series A", "b": "Series B"})
    merged.update_multi_data(
        {
            "a": np.random.default_rng(1).normal(1.0, 0.2, 300),
            "b": np.random.default_rng(2).normal(2.0, 0.2, 300),
        }
    )
    merged._legend_placement = "outside"
    merged._legend_position = "bottom"
    merged._render()
    assert len(merged.fig.legends) == 1
    assert _legend_texts(merged.fig.legends[0]) == ["Series A", "Series B"]


def test_settings_dialog_applies_legend_location(qtbot, monkeypatch):
    """Accepting the settings moves the legend, and reopening shows it."""
    widget = _histogram(qtbot)
    seen = {}

    def fake_exec(self):
        seen["opened_on"] = (
            self.get_legend_placement(),
            self.get_legend_position(),
        )
        _select(self.legend_placement_combo, "outside")
        _select(self.legend_position_combo, "bottom")
        return QDialog.Accepted

    monkeypatch.setattr(HistogramSettingsDialog, "exec", fake_exec)
    widget._open_settings_dialog()

    assert seen["opened_on"] == ("inside", "upper right")
    assert widget._legend_placement == "outside"
    assert widget._legend_position == "bottom"
    assert len(widget.fig.legends) == 1

    widget._open_settings_dialog()
    assert seen["opened_on"] == ("outside", "bottom")


# ---------------------------------------------------------------------------
# Export dialog
# ---------------------------------------------------------------------------


def _dialog(qtbot, widget=None, options=None):
    widget = widget or _histogram(qtbot)
    dlg = HistogramExportDialog(widget, options=options)
    qtbot.addWidget(dlg)
    return dlg


def test_number_or_accepts_only_finite_numbers():
    assert _number_or("2.5", 1.0) == 2.5
    assert _number_or(3, 1.0) == 3.0
    assert _number_or(None, 1.0) == 1.0
    assert _number_or("many", 1.0) == 1.0
    assert _number_or(float("nan"), 1.0) == 1.0
    assert _number_or(float("inf"), 1.0) == 1.0


def test_export_dialog_defaults(qtbot):
    """The dialog starts on a 300 DPI PNG the shape of the plot on screen."""
    widget = _histogram(qtbot)
    dlg = _dialog(qtbot, widget)

    assert dlg.export_format() == "png"
    assert dlg.aspect_combo.currentText() == "As shown"
    assert dlg.dpi() == 300
    assert dlg.unit_combo.currentText() == "cm"
    assert dlg.width_spin.value() == 12.0

    width_in, height_in = dlg.size_inches()
    assert width_in == pytest.approx(12.0 / 2.54)
    shown_width, shown_height = widget.fig.get_size_inches()
    assert width_in / height_in == pytest.approx(shown_width / shown_height)
    assert dlg.size_pixels() == (
        round(width_in * 300),
        round(height_in * 300),
    )
    # The height follows the width unless the ratio is custom.
    assert not dlg.height_spin.isEnabled()
    assert dlg.save_button.isEnabled()


def test_export_dialog_offers_every_format(qtbot):
    dlg = _dialog(qtbot)
    formats = [
        dlg.format_combo.itemData(i) for i in range(dlg.format_combo.count())
    ]
    assert formats == ["png", "svg", "jpg", "csv"]


@pytest.mark.parametrize(
    "aspect, ratio",
    [
        ("Square (1:1)", 1.0),
        ("4:3", 4 / 3),
        ("3:2", 3 / 2),
        ("16:9", 16 / 9),
        ("2:1", 2.0),
    ],
)
def test_export_dialog_aspect_ratio_sets_the_height(qtbot, aspect, ratio):
    """A fixed ratio derives the height from the width."""
    dlg = _dialog(qtbot)
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText(aspect))

    width_in, height_in = dlg.size_inches()
    assert width_in / height_in == pytest.approx(ratio)
    assert not dlg.height_spin.isEnabled()
    # The disabled height box shows the derived value.
    assert dlg.height_spin.value() == pytest.approx(height_in * 2.54, abs=0.06)


def test_export_dialog_custom_aspect_uses_the_height(qtbot):
    dlg = _dialog(qtbot)
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("Custom"))
    assert dlg.height_spin.isEnabled()

    dlg.width_spin.setValue(10.0)
    dlg.height_spin.setValue(5.0)
    width_in, height_in = dlg.size_inches()
    assert width_in == pytest.approx(10.0 / 2.54)
    assert height_in == pytest.approx(5.0 / 2.54)


def test_export_dialog_unit_change_keeps_the_physical_size(qtbot):
    dlg = _dialog(qtbot)
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("Custom"))
    dlg.width_spin.setValue(10.0)
    dlg.height_spin.setValue(5.0)
    before = dlg.size_inches()

    dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("in"))
    assert dlg.width_spin.value() == pytest.approx(10.0 / 2.54, abs=0.06)
    assert dlg.size_inches() == pytest.approx(before, abs=0.06)

    dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("cm"))
    assert dlg.size_inches() == pytest.approx(before, abs=0.1)


def test_export_dialog_size_label_and_dpi(qtbot):
    """The label states the pixel size; an SVG states its physical size."""
    dlg = _dialog(qtbot)
    dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("in"))
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("Square (1:1)"))
    dlg.width_spin.setValue(4.0)
    dlg.dpi_spin.setValue(200)

    assert dlg.size_pixels() == (800, 800)
    assert dlg.size_label.text() == "800 × 800 px"

    _select(dlg.format_combo, "svg")
    assert "vector" in dlg.size_label.text()
    assert "10.2 × 10.2 cm" in dlg.size_label.text()
    assert dlg.dpi_spin.isEnabled()


def test_export_dialog_csv_hides_the_image_options(qtbot):
    dlg = _dialog(qtbot)
    assert dlg._image_options.isVisibleTo(dlg)
    assert dlg.preview_label.isVisibleTo(dlg)

    _select(dlg.format_combo, "csv")
    assert dlg.export_format() == "csv"
    assert not dlg._image_options.isVisibleTo(dlg)
    assert not dlg.preview_label.isVisibleTo(dlg)
    assert "table" in dlg.preview_note.text()
    assert dlg.save_button.isEnabled()

    _select(dlg.format_combo, "png")
    assert dlg._image_options.isVisibleTo(dlg)
    assert dlg.preview_label.isVisibleTo(dlg)


def test_export_dialog_refuses_an_image_too_large_to_render(qtbot):
    dlg = _dialog(qtbot)
    dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("in"))
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("Square (1:1)"))
    dlg.width_spin.setValue(100.0)
    dlg.dpi_spin.setValue(300)

    assert dlg.size_pixels() == (30000, 30000)
    assert "too large" in dlg.size_label.text()
    assert not dlg.save_button.isEnabled()

    dlg.dpi_spin.setValue(50)
    assert dlg.save_button.isEnabled()
    assert "too large" not in dlg.size_label.text()


def test_export_dialog_preview_has_the_export_proportions(qtbot):
    """The preview is the histogram drawn at the export aspect ratio."""
    dlg = _dialog(qtbot)
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("Custom"))
    dlg.width_spin.setValue(16.0)
    dlg.height_spin.setValue(4.0)

    pixmap = dlg.preview_label.pixmap()
    assert not pixmap.isNull()
    max_width, max_height = dlg.PREVIEW_SIZE
    assert pixmap.width() <= max_width
    assert pixmap.height() <= max_height
    assert pixmap.width() / pixmap.height() == pytest.approx(4.0, rel=0.02)

    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("Square (1:1)"))
    pixmap = dlg.preview_label.pixmap()
    assert pixmap.width() == pytest.approx(pixmap.height(), abs=2)


def test_export_dialog_preview_note_describes_the_background(qtbot):
    widget = _histogram(qtbot)
    dlg = _dialog(qtbot, widget)
    assert "transparent" in dlg.preview_note.text()

    # JPG has no transparency.
    _select(dlg.format_combo, "jpg")
    assert "transparent" not in dlg.preview_note.text()

    _select(dlg.format_combo, "svg")
    assert "transparent" in dlg.preview_note.text()

    dlg.white_bg_checkbox.setChecked(True)
    _select(dlg.format_combo, "png")
    assert "transparent" not in dlg.preview_note.text()
    assert dlg.is_opaque()


def test_export_dialog_preview_leaves_the_plot_untouched(qtbot):
    """Drawing the preview does not resize or restyle the on-screen plot."""
    widget = _histogram(qtbot)
    size = widget.fig.get_size_inches().copy()
    alpha = widget.fig.patch.get_alpha()

    dlg = _dialog(qtbot, widget)
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("16:9"))
    _select(dlg.format_combo, "jpg")

    assert widget.fig.get_size_inches() == pytest.approx(size)
    assert widget.fig.patch.get_alpha() == alpha


def test_export_dialog_options_round_trip(qtbot):
    dlg = _dialog(qtbot)
    _select(dlg.format_combo, "svg")
    dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("in"))
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("Custom"))
    dlg.width_spin.setValue(6.0)
    dlg.height_spin.setValue(3.5)
    dlg.dpi_spin.setValue(150)
    options = dlg.options()
    assert options == {
        "format": "svg",
        "aspect": "Custom",
        "unit": "in",
        "width": 6.0,
        "height": 3.5,
        "dpi": 150,
        "text_size": 6.0,
        "tick_size": 7.0,
    }

    again = _dialog(qtbot, options=options)
    assert again.options() == options
    assert again.size_inches() == pytest.approx(dlg.size_inches())


def test_export_dialog_ignores_invalid_options(qtbot):
    """Stale or garbled options fall back to the defaults."""
    dlg = _dialog(
        qtbot,
        options={
            "format": "bmp",
            "aspect": "portrait",
            "unit": "parsec",
            "width": "wide",
            "height": float("nan"),
            "dpi": None,
        },
    )
    assert dlg.export_format() == "png"
    assert dlg.aspect_combo.currentText() == "As shown"
    assert dlg.unit_combo.currentText() == "cm"
    assert dlg.width_spin.value() == 12.0
    assert dlg.dpi() == 300


def test_export_dialog_default_width_in_inches(qtbot):
    dlg = _dialog(qtbot, options={"unit": "in"})
    assert dlg.unit_combo.currentText() == "in"
    assert dlg.width_spin.value() == 5.0


def test_export_dialog_buttons(qtbot):
    dlg = _dialog(qtbot)
    dlg.show()
    dlg.save_button.click()
    assert dlg.result() == QDialog.Accepted

    dlg = _dialog(qtbot)
    dlg.show()
    dlg.reject()
    assert dlg.result() == QDialog.Rejected


# ---------------------------------------------------------------------------
# Saving from the histogram widget
# ---------------------------------------------------------------------------


def _accept_export(
    monkeypatch, fmt, *, aspect="Square (1:1)", then=None, **settings
):
    """Make the export dialog "accept" with the given choices.

    ``then`` is called with the dialog afterwards, to set further controls.
    """
    seen = {}

    def fake_exec(self):
        seen["dialog"] = self
        seen["initial"] = self.options()
        _select(self.format_combo, fmt)
        self.unit_combo.setCurrentIndex(self.unit_combo.findText("in"))
        self.aspect_combo.setCurrentIndex(self.aspect_combo.findText(aspect))
        self.width_spin.setValue(settings.get("width", 4.0))
        self.dpi_spin.setValue(settings.get("dpi", 100))
        if then is not None:
            then(self)
        return QDialog.Accepted

    monkeypatch.setattr(HistogramExportDialog, "exec", fake_exec)
    return seen


def _save_to(monkeypatch, path):
    """Answer the file dialog with *path* (or a cancel when ``None``)."""
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *a, **k: ("" if path is None else str(path), ""),
    )


def test_export_png_has_the_requested_size(qtbot, tmp_path, monkeypatch):
    widget = _histogram(qtbot)
    _accept_export(monkeypatch, "png", width=4.0, dpi=100)
    out = tmp_path / "hist.png"
    _save_to(monkeypatch, out)

    widget._open_export_dialog()

    with Image.open(out) as image:
        assert image.format == "PNG"
        assert image.size == (400, 400)
        # Transparent unless the white background is on.
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 0


def test_export_png_with_white_background_is_opaque(
    qtbot, tmp_path, monkeypatch
):
    widget = _histogram(qtbot)
    widget._white_background = True
    _accept_export(monkeypatch, "png")
    out = tmp_path / "white.png"
    _save_to(monkeypatch, out)

    widget._open_export_dialog()

    with Image.open(out) as image:
        assert image.convert("RGBA").getpixel((0, 0)) == (255, 255, 255, 255)


def test_export_png_aspect_ratio_changes_the_shape(
    qtbot, tmp_path, monkeypatch
):
    widget = _histogram(qtbot)
    _accept_export(monkeypatch, "png", aspect="16:9", width=8.0, dpi=100)
    out = tmp_path / "wide.png"
    _save_to(monkeypatch, out)

    widget._open_export_dialog()

    with Image.open(out) as image:
        assert image.size == (800, 450)


def test_export_jpg_is_opaque(qtbot, tmp_path, monkeypatch):
    """JPG has no transparency, so the background is filled white."""
    widget = _histogram(qtbot)
    _accept_export(monkeypatch, "jpg", width=4.0, dpi=100)
    out = tmp_path / "hist.jpg"
    _save_to(monkeypatch, out)

    widget._open_export_dialog()

    with Image.open(out) as image:
        assert image.format == "JPEG"
        assert image.size == (400, 400)
        assert image.mode == "RGB"
        assert all(value > 240 for value in image.getpixel((0, 0)))


def test_export_svg_is_a_vector_image(qtbot, tmp_path, monkeypatch):
    widget = _histogram(qtbot)
    _accept_export(monkeypatch, "svg", width=4.0)
    out = tmp_path / "hist.svg"
    _save_to(monkeypatch, out)

    widget._open_export_dialog()

    text = out.read_text()
    assert "<svg" in text
    # 4 inches at 72 pt per inch.
    assert 'width="288pt"' in text


def test_export_adds_the_missing_file_suffix(qtbot, tmp_path, monkeypatch):
    widget = _histogram(qtbot)
    for fmt, suffix in (("png", ".png"), ("svg", ".svg"), ("jpg", ".jpg")):
        _accept_export(monkeypatch, fmt, dpi=50)
        _save_to(monkeypatch, tmp_path / f"noext_{fmt}")
        widget._open_export_dialog()
        assert (tmp_path / f"noext_{fmt}{suffix}").exists()


def test_export_keeps_an_accepted_suffix(qtbot, tmp_path, monkeypatch):
    """Upper case and ``.jpeg`` are not turned into a doubled suffix."""
    widget = _histogram(qtbot)
    _accept_export(monkeypatch, "jpg", dpi=50)
    _save_to(monkeypatch, tmp_path / "photo.JPEG")
    widget._open_export_dialog()
    assert (tmp_path / "photo.JPEG").exists()

    _accept_export(monkeypatch, "png", dpi=50)
    _save_to(monkeypatch, tmp_path / "Figure.PNG")
    widget._open_export_dialog()
    assert (tmp_path / "Figure.PNG").exists()
    assert not (tmp_path / "Figure.PNG.png").exists()


def test_export_cancelled_file_dialog_writes_nothing(
    qtbot, tmp_path, monkeypatch
):
    widget = _histogram(qtbot)
    _accept_export(monkeypatch, "png", dpi=50)
    _save_to(monkeypatch, None)
    widget._open_export_dialog()
    assert list(tmp_path.iterdir()) == []


def test_export_rejected_dialog_does_nothing(qtbot, tmp_path, monkeypatch):
    widget = _histogram(qtbot)
    monkeypatch.setattr(
        HistogramExportDialog, "exec", lambda self: QDialog.Rejected
    )

    def fail(*args, **kwargs):
        raise AssertionError("nothing should be saved")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fail)
    widget._open_export_dialog()
    assert widget._export_options is None


def test_export_csv_goes_through_the_csv_writer(qtbot, tmp_path, monkeypatch):
    widget = _histogram(qtbot)
    _accept_export(monkeypatch, "csv")
    out = tmp_path / "hist.csv"
    _save_to(monkeypatch, out)

    widget._open_export_dialog()

    header = out.read_text().splitlines()[0]
    assert header == "Bin Center,first,second"


def test_export_options_are_remembered(qtbot, tmp_path, monkeypatch):
    """The next export starts from what the last one used."""
    widget = _histogram(qtbot)
    seen = _accept_export(monkeypatch, "svg", aspect="3:2", width=6.0, dpi=150)
    _save_to(monkeypatch, tmp_path / "first.svg")
    widget._open_export_dialog()
    assert seen["initial"]["format"] == "png"  # the defaults the first time

    assert widget._export_options["format"] == "svg"
    assert widget._export_options["aspect"] == "3:2"
    assert widget._export_options["dpi"] == 150

    _save_to(monkeypatch, tmp_path / "second.svg")
    widget._open_export_dialog()
    assert seen["initial"] == widget._export_options


def test_export_restores_the_plot_it_was_drawn_from(
    qtbot, tmp_path, monkeypatch
):
    """Exporting leaves the on-screen figure as it found it."""
    widget = _histogram(qtbot)
    size = widget.fig.get_size_inches().copy()
    grey = to_rgba("grey")

    _accept_export(monkeypatch, "png", aspect="16:9", width=8.0, dpi=50)
    _save_to(monkeypatch, tmp_path / "hist.png")
    widget._open_export_dialog()

    assert widget.fig.get_size_inches() == pytest.approx(size)
    assert widget.fig.patch.get_alpha() == 0
    assert widget.ax.spines["left"].get_edgecolor() == grey


def test_export_restores_the_plot_when_saving_fails(qtbot, monkeypatch):
    widget = _histogram(qtbot)
    size = widget.fig.get_size_inches().copy()

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(widget.fig, "savefig", boom)
    with pytest.raises(OSError, match="disk full"):
        widget._export_figure("ignored.png", "png", 8.0, 2.0, 100)

    assert widget.fig.get_size_inches() == pytest.approx(size)
    assert widget.ax.spines["left"].get_edgecolor() == to_rgba("grey")


def test_export_keeps_an_outside_legend_inside_the_image(
    qtbot, tmp_path, monkeypatch
):
    """The exported image has room for a legend placed outside the plot."""
    widget = _histogram(qtbot)
    widget._legend_placement = "outside"
    widget._legend_position = "right"
    widget._render()

    _accept_export(monkeypatch, "png", aspect="2:1", width=6.0, dpi=100)
    out = tmp_path / "legend.png"
    _save_to(monkeypatch, out)
    widget._open_export_dialog()

    with Image.open(out) as image:
        assert image.size == (600, 300)
        alpha = np.asarray(image)[..., 3]
    # Something is drawn in the right-hand margin (the legend), and the
    # column at the very edge is blank.
    right_margin = alpha[:, int(600 * 0.8) : -2]
    assert right_margin.max() > 0
    assert alpha[:, -1].max() == 0
    assert len(widget.fig.legends) == 1


# ---------------------------------------------------------------------------
# Size in pixels
# ---------------------------------------------------------------------------


def test_export_dialog_size_in_pixels(qtbot):
    """With pixels as the unit the size boxes are the image size itself."""
    dlg = _dialog(qtbot)
    dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("px"))
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("Custom"))
    dlg.width_spin.setValue(1499)
    dlg.height_spin.setValue(777)

    assert dlg.width_spin.decimals() == 0
    assert dlg.size_pixels() == (1499, 777)
    assert dlg.size_label.text() == "1499 × 777 px"
    # The physical size follows the DPI.
    assert dlg.size_inches() == pytest.approx((1499 / 300, 777 / 300))
    dlg.dpi_spin.setValue(150)
    assert dlg.size_pixels() == (1499, 777)
    assert dlg.size_inches() == pytest.approx((1499 / 150, 777 / 150))


def test_export_dialog_pixel_height_follows_the_aspect_ratio(qtbot):
    dlg = _dialog(qtbot)
    dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("px"))
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("3:2"))
    dlg.width_spin.setValue(1500)

    assert dlg.size_pixels() == (1500, 1000)
    assert dlg.height_spin.value() == 1000
    assert not dlg.height_spin.isEnabled()


def test_export_dialog_converts_sizes_to_and_from_pixels(qtbot):
    """Changing to pixels keeps the physical size at the current DPI."""
    dlg = _dialog(qtbot)
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("Custom"))
    dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("in"))
    dlg.width_spin.setValue(5.0)
    dlg.height_spin.setValue(2.5)

    dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("px"))
    assert (dlg.width_spin.value(), dlg.height_spin.value()) == (1500, 750)
    assert dlg.width_spin.maximum() == 20000

    dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("cm"))
    assert dlg.width_spin.value() == pytest.approx(12.7)
    assert dlg.width_spin.decimals() == 1
    assert dlg.width_spin.maximum() == 200


def test_export_dialog_pixel_size_limit(qtbot):
    dlg = _dialog(qtbot)
    dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("px"))
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("Square (1:1)"))
    dlg.width_spin.setValue(20000)
    assert not dlg.save_button.isEnabled()
    dlg.width_spin.setValue(5000)
    assert dlg.save_button.isEnabled()


def test_export_dialog_pixel_options_round_trip(qtbot):
    dlg = _dialog(qtbot, options={"unit": "px"})
    assert dlg.width_spin.value() == 1500
    assert dlg.height_spin.value() == 750
    dlg.width_spin.setValue(2000)
    again = _dialog(qtbot, options=dlg.options())
    assert again.unit_combo.currentText() == "px"
    assert again.width_spin.value() == 2000


@pytest.mark.parametrize("width, height", [(1499, 777), (801, 333)])
@pytest.mark.parametrize("fmt", ["png", "jpg"])
def test_export_has_exactly_the_requested_pixels(
    qtbot, tmp_path, monkeypatch, fmt, width, height
):
    """An odd pixel size is not truncated by a floating point remainder."""
    widget = _histogram(qtbot)

    def in_pixels(dlg):
        dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("px"))
        dlg.width_spin.setValue(width)
        dlg.height_spin.setValue(height)

    _accept_export(monkeypatch, fmt, aspect="Custom", dpi=300, then=in_pixels)
    out = tmp_path / f"px.{fmt}"
    _save_to(monkeypatch, out)
    widget._open_export_dialog()

    with Image.open(out) as image:
        assert image.size == (width, height)


# ---------------------------------------------------------------------------
# Text and tick sizes
# ---------------------------------------------------------------------------


def test_export_dialog_text_sizes_start_from_the_plot(qtbot):
    widget = _histogram(qtbot)
    widget._label_fontsize = 8
    widget._tick_fontsize = 9
    dlg = _dialog(qtbot, widget)
    assert dlg.text_size_spin.value() == 8
    assert dlg.tick_size_spin.value() == 9

    dlg = _dialog(qtbot)
    assert dlg.text_size_spin.value() == HistogramWidget.DEFAULT_LABEL_FONTSIZE
    assert dlg.tick_size_spin.value() == HistogramWidget.DEFAULT_TICK_FONTSIZE


def test_export_style_applies_only_while_exporting(qtbot):
    """Export text sizes, legend and background apply, then are undone."""
    widget = _histogram(qtbot)
    style = {
        "white_background": True,
        "text_size": 14,
        "tick_size": 11,
        "legend_location": ("outside", "top"),
    }
    with widget._export_style(style):
        assert widget.ax.xaxis.label.get_fontsize() == 14
        assert widget.ax.yaxis.label.get_fontsize() == 14
        assert widget.ax.xaxis.get_ticklabels()[0].get_fontsize() == 11
        assert widget.ax.yaxis.get_ticklabels()[0].get_fontsize() == 11
        assert widget._white_background is True
        assert len(widget.fig.legends) == 1
        assert widget.fig.legends[0].texts[0].get_fontsize() == 13
        assert widget.ax.get_legend() is None

    assert widget.ax.xaxis.label.get_fontsize() == 6
    assert widget.ax.xaxis.get_ticklabels()[0].get_fontsize() == 7
    assert widget._white_background is False
    assert widget.fig.legends == []
    assert widget.ax.get_legend()._loc == Legend.codes["upper right"]


def test_export_style_without_changes_does_not_redraw(qtbot, monkeypatch):
    widget = _histogram(qtbot)
    calls = []
    monkeypatch.setattr(widget, "_render", lambda: calls.append(1))
    with widget._export_style(None):
        pass
    with widget._export_style({"text_size": 6, "tick_size": 7}):
        pass
    assert calls == []


def test_export_style_is_undone_when_saving_fails(qtbot, monkeypatch):
    widget = _histogram(qtbot)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(widget.fig, "savefig", boom)
    with pytest.raises(OSError, match="disk full"):
        widget._export_figure(
            "ignored.png",
            "png",
            4.0,
            2.0,
            100,
            style={"text_size": 20, "white_background": True},
        )
    assert widget._label_fontsize == 6
    assert widget._white_background is False


def test_export_text_size_changes_the_image(qtbot, tmp_path, monkeypatch):
    """Larger text takes more of the image, and the plot keeps its sizes."""
    widget = _histogram(qtbot)
    small = tmp_path / "small.png"
    large = tmp_path / "large.png"
    for out, size in ((small, 6.0), (large, 16.0)):

        def set_sizes(dlg, size=size):
            dlg.text_size_spin.setValue(size)
            dlg.tick_size_spin.setValue(size)

        _accept_export(monkeypatch, "png", dpi=100, then=set_sizes)
        _save_to(monkeypatch, out)
        widget._open_export_dialog()

    def inked(path):
        with Image.open(path) as image:
            return int((np.asarray(image)[..., 3] > 0).sum())

    assert inked(large) > inked(small)
    assert widget.ax.xaxis.label.get_fontsize() == 6
    assert widget.ax.xaxis.get_ticklabels()[0].get_fontsize() == 7
    # The sizes are remembered for the next export.
    assert widget._export_options["text_size"] == 16.0
    assert widget._export_options["tick_size"] == 16.0


def test_export_dialog_style_reports_the_choices(qtbot):
    dlg = _dialog(qtbot)
    dlg.text_size_spin.setValue(9.5)
    dlg.tick_size_spin.setValue(8.0)
    dlg.white_bg_checkbox.setChecked(True)
    _select(dlg.legend_placement_combo, "outside")
    _select(dlg.legend_position_combo, "bottom")

    assert dlg.style() == {
        "white_background": True,
        "text_size": 9.5,
        "tick_size": 8.0,
        "legend_location": ("outside", "bottom"),
    }


# ---------------------------------------------------------------------------
# White background
# ---------------------------------------------------------------------------


def test_export_dialog_white_background_starts_from_the_plot(qtbot):
    widget = _histogram(qtbot)
    assert not _dialog(qtbot, widget).white_bg_checkbox.isChecked()
    widget._white_background = True
    assert _dialog(qtbot, widget).white_bg_checkbox.isChecked()


def test_export_white_background_toggle_is_export_only(
    qtbot, tmp_path, monkeypatch
):
    """Ticking it fills the file, and leaves the plot's own setting alone."""
    widget = _histogram(qtbot)
    _accept_export(
        monkeypatch,
        "png",
        dpi=50,
        then=lambda dlg: dlg.white_bg_checkbox.setChecked(True),
    )
    out = tmp_path / "white.png"
    _save_to(monkeypatch, out)
    widget._open_export_dialog()

    with Image.open(out) as image:
        assert image.convert("RGBA").getpixel((0, 0)) == (255, 255, 255, 255)
    assert widget._white_background is False
    assert widget.fig.patch.get_alpha() == 0


def test_export_white_background_can_be_switched_off(
    qtbot, tmp_path, monkeypatch
):
    widget = _histogram(qtbot)
    widget._white_background = True
    _accept_export(
        monkeypatch,
        "png",
        dpi=50,
        then=lambda dlg: dlg.white_bg_checkbox.setChecked(False),
    )
    out = tmp_path / "clear.png"
    _save_to(monkeypatch, out)
    widget._open_export_dialog()

    with Image.open(out) as image:
        assert image.getpixel((0, 0))[3] == 0
    assert widget._white_background is True


def test_export_dialog_options_leave_out_background_and_legend(qtbot):
    dlg = _dialog(qtbot)
    dlg.white_bg_checkbox.setChecked(True)
    assert "white_background" not in dlg.options()
    assert "legend_location" not in dlg.options()


# ---------------------------------------------------------------------------
# Legend location in the export dialog
# ---------------------------------------------------------------------------


def test_widget_reports_whether_it_has_a_legend(qtbot):
    widget = _histogram(qtbot)
    assert widget.has_legend()
    widget._legend_placement = "outside"
    widget._render()
    assert widget.has_legend()
    widget._show_legend = False
    widget._render()
    assert not widget.has_legend()


def test_export_dialog_legend_location_starts_from_the_plot(qtbot):
    widget = _histogram(qtbot)
    widget._legend_placement = "outside"
    widget._legend_position = "bottom"
    widget._render()
    dlg = _dialog(qtbot, widget)

    assert dlg.legend_placement_combo.isEnabled()
    assert dlg.legend_position_combo.isEnabled()
    assert dlg.legend_placement_combo.currentData() == "outside"
    assert dlg.legend_position_combo.currentData() == "bottom"


def test_export_dialog_legend_placement_lists_matching_positions(qtbot):
    dlg = _dialog(qtbot)
    assert dlg.legend_position_combo.currentData() == "upper right"
    assert dlg.legend_position_combo.count() == len(LEGEND_POSITIONS["inside"])

    _select(dlg.legend_placement_combo, "outside")
    assert dlg.legend_position_combo.count() == len(
        LEGEND_POSITIONS["outside"]
    )
    assert dlg.legend_position_combo.currentData() == "right"

    _select(dlg.legend_position_combo, "top")
    assert dlg.style()["legend_location"] == ("outside", "top")


def test_export_dialog_legend_controls_explain_when_unavailable(qtbot):
    """With no legend on the plot, the controls are off and say why."""
    widget = _histogram(qtbot, mode="Merged")
    assert not widget.has_legend()
    dlg = _dialog(qtbot, widget)

    for control in (
        dlg._legend_label,
        dlg.legend_placement_combo,
        dlg.legend_position_combo,
    ):
        assert not control.isEnabled()
        assert "Show legend" in control.toolTip()


def test_export_dialog_legend_choice_moves_the_preview_legend(qtbot):
    """The preview is drawn with the chosen legend location."""
    widget = _histogram(qtbot)
    dlg = _dialog(qtbot, widget)
    before = dlg.preview_label.pixmap().toImage()

    _select(dlg.legend_placement_combo, "outside")
    after = dlg.preview_label.pixmap().toImage()
    assert before != after
    # The plot on screen has not moved.
    assert widget.ax.get_legend() is not None
    assert widget.fig.legends == []


def test_export_legend_location_is_export_only(qtbot, tmp_path, monkeypatch):
    """The saved image has an outside legend; the plot keeps its own."""
    widget = _histogram(qtbot)

    def outside_right(dlg):
        _select(dlg.legend_placement_combo, "outside")
        _select(dlg.legend_position_combo, "right")

    _accept_export(
        monkeypatch, "png", aspect="2:1", width=6.0, then=outside_right
    )
    out = tmp_path / "outside.png"
    _save_to(monkeypatch, out)
    widget._open_export_dialog()

    with Image.open(out) as image:
        alpha = np.asarray(image)[..., 3]
    assert alpha[:, int(alpha.shape[1] * 0.8) : -2].max() > 0
    assert widget._legend_placement == "inside"
    assert widget.fig.legends == []
    assert widget.ax.get_legend() is not None
