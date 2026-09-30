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


def test_settings_dialog_legend_location(qtbot):
    """The settings dialog has no aspect ratio option (the plot always fills
    its canvas); its legend location starts inside at the upper right or
    on the stored placement, lists the positions of the chosen placement,
    and is live only while a legend is drawn."""
    dlg = HistogramSettingsDialog()
    qtbot.addWidget(dlg)
    assert not hasattr(dlg, "aspect_ratio_combo")
    widget = _histogram(qtbot)
    assert not hasattr(widget, "_aspect_ratio")
    assert widget.ax.get_box_aspect() is None

    def positions(dlg):
        return [
            dlg.legend_position_combo.itemData(i)
            for i in range(dlg.legend_position_combo.count())
        ]

    dlg = HistogramSettingsDialog(
        display_mode="Individual layers", show_legend=True
    )
    qtbot.addWidget(dlg)
    assert dlg.get_legend_placement() == "inside"
    assert dlg.get_legend_position() == "upper right"
    assert positions(dlg) == [key for _, key in LEGEND_POSITIONS["inside"]]

    # The location controls are live only while a legend is drawn.
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

    # The dialog opens on the stored placement and position.
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

    # Switching inside/outside swaps the positions, keeping a shared one.
    dlg = HistogramSettingsDialog(
        display_mode="Individual layers",
        show_legend=True,
        legend_position="lower left",
    )
    qtbot.addWidget(dlg)
    assert dlg.get_legend_position() == "lower left"
    _select(dlg.legend_placement_combo, "outside")
    assert dlg.get_legend_placement() == "outside"
    assert positions(dlg) == [key for _, key in LEGEND_POSITIONS["outside"]]
    # "lower left" is not an outside position: the default is used.
    assert dlg.get_legend_position() == default_legend_position("outside")
    _select(dlg.legend_position_combo, "top")
    _select(dlg.legend_placement_combo, "inside")
    assert dlg.get_legend_position() == default_legend_position("inside")


def test_legend_placement_on_the_plot(qtbot, monkeypatch):
    """An inside legend is an axes legend at the chosen corner; an outside
    one is a single figure legend beside, above or below the axes, for
    individual, grouped and per-series curves; and the settings dialog
    moves it."""
    widget = _histogram(qtbot)
    for position in ("upper left", "lower right", "center"):
        widget._legend_position = position
        widget._render()
        legend = widget.ax.get_legend()
        assert legend is not None
        assert legend._loc == Legend.codes[position]
        assert _legend_texts(legend) == ["first", "second"]
        assert widget.fig.legends == []

    widget._legend_placement = "outside"
    for position, ncol in (("right", 1), ("top", 2), ("bottom", 2)):
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

    # Re-rendering, hiding, moving inside or clearing removes the legend.
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

    # No curve carries a label, so no legend (and no warning) is drawn.
    empty = HistogramWidget(bins=5)
    qtbot.addWidget(empty)
    empty._draw_legend()
    assert empty.ax.get_legend() is None
    assert empty.fig.legends == []

    # Grouped curves and per-series merged curves honour the placement.
    grouped = _histogram(qtbot, mode="Grouped")
    grouped._group_assignments = {"first": 1, "second": 2}
    grouped._legend_placement = "outside"
    grouped._legend_position = "top"
    grouped._render()
    assert len(grouped.fig.legends) == 1
    assert len(_legend_texts(grouped.fig.legends[0])) == 2

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

    # Accepting the settings moves the legend, and reopening shows it.
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


def test_export_dialog_sizes(qtbot):
    """The dialog starts on a 300 DPI PNG the shape of the plot on screen;
    a fixed aspect ratio derives the height, units convert without changing
    the physical size, the label states the pixel (or, for SVG, physical)
    size, and images too large to render are refused."""
    widget = _histogram(qtbot)
    dlg = _dialog(qtbot, widget)

    def set_unit(unit):
        dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText(unit))

    def set_aspect(aspect):
        dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText(aspect))

    assert [
        dlg.format_combo.itemData(i) for i in range(dlg.format_combo.count())
    ] == ["png", "svg", "jpg", "csv"]
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

    # A fixed ratio derives the height, shown in the disabled height box.
    for aspect, ratio in (
        ("Square (1:1)", 1.0),
        ("4:3", 4 / 3),
        ("3:2", 3 / 2),
        ("16:9", 16 / 9),
        ("2:1", 2.0),
    ):
        set_aspect(aspect)
        width_in, height_in = dlg.size_inches()
        assert width_in / height_in == pytest.approx(ratio)
        assert not dlg.height_spin.isEnabled()
        assert dlg.height_spin.value() == pytest.approx(
            height_in * 2.54, abs=0.06
        )

    # A custom ratio uses the height, and a unit change keeps the size.
    set_aspect("Custom")
    assert dlg.height_spin.isEnabled()
    dlg.width_spin.setValue(10.0)
    dlg.height_spin.setValue(5.0)
    before = dlg.size_inches()
    assert before == pytest.approx((10.0 / 2.54, 5.0 / 2.54))
    set_unit("in")
    assert dlg.width_spin.value() == pytest.approx(10.0 / 2.54, abs=0.06)
    assert dlg.size_inches() == pytest.approx(before, abs=0.06)
    set_unit("cm")
    assert dlg.size_inches() == pytest.approx(before, abs=0.1)

    # The preview is the histogram drawn at the export aspect ratio.
    dlg.width_spin.setValue(16.0)
    dlg.height_spin.setValue(4.0)
    pixmap = dlg.preview_label.pixmap()
    assert not pixmap.isNull()
    max_width, max_height = dlg.PREVIEW_SIZE
    assert pixmap.width() <= max_width
    assert pixmap.height() <= max_height
    assert pixmap.width() / pixmap.height() == pytest.approx(4.0, rel=0.02)
    set_aspect("Square (1:1)")
    pixmap = dlg.preview_label.pixmap()
    assert pixmap.width() == pytest.approx(pixmap.height(), abs=2)

    # The label states the pixel size; an SVG states its physical size.
    set_unit("in")
    dlg.width_spin.setValue(4.0)
    dlg.dpi_spin.setValue(200)
    assert dlg.size_pixels() == (800, 800)
    assert dlg.size_label.text() == "800 × 800 px"
    _select(dlg.format_combo, "svg")
    assert "vector" in dlg.size_label.text()
    assert "10.2 × 10.2 cm" in dlg.size_label.text()
    assert dlg.dpi_spin.isEnabled()
    _select(dlg.format_combo, "png")

    # An image too large to render is refused.
    dlg.width_spin.setValue(100.0)
    dlg.dpi_spin.setValue(300)
    assert dlg.size_pixels() == (30000, 30000)
    assert "too large" in dlg.size_label.text()
    assert not dlg.save_button.isEnabled()
    dlg.dpi_spin.setValue(50)
    assert dlg.save_button.isEnabled()
    assert "too large" not in dlg.size_label.text()

    # Changing to pixels keeps the physical size at the current DPI.
    dlg.dpi_spin.setValue(300)
    set_aspect("Custom")
    dlg.width_spin.setValue(5.0)
    dlg.height_spin.setValue(2.5)
    set_unit("px")
    assert (dlg.width_spin.value(), dlg.height_spin.value()) == (1500, 750)
    assert dlg.width_spin.maximum() == 20000
    set_unit("cm")
    assert dlg.width_spin.value() == pytest.approx(12.7)
    assert dlg.width_spin.decimals() == 1
    assert dlg.width_spin.maximum() == 200

    # With pixels as the unit the size boxes are the image size itself,
    # and the physical size follows the DPI.
    set_unit("px")
    dlg.width_spin.setValue(1499)
    dlg.height_spin.setValue(777)
    assert dlg.width_spin.decimals() == 0
    assert dlg.size_pixels() == (1499, 777)
    assert dlg.size_label.text() == "1499 × 777 px"
    assert dlg.size_inches() == pytest.approx((1499 / 300, 777 / 300))
    dlg.dpi_spin.setValue(150)
    assert dlg.size_pixels() == (1499, 777)
    assert dlg.size_inches() == pytest.approx((1499 / 150, 777 / 150))

    # A fixed ratio derives the pixel height too.
    set_aspect("3:2")
    dlg.width_spin.setValue(1500)
    assert dlg.size_pixels() == (1500, 1000)
    assert dlg.height_spin.value() == 1000
    assert not dlg.height_spin.isEnabled()

    # The pixel size is limited as well.
    set_aspect("Square (1:1)")
    dlg.width_spin.setValue(20000)
    assert not dlg.save_button.isEnabled()
    dlg.width_spin.setValue(5000)
    assert dlg.save_button.isEnabled()


def test_export_dialog_appearance(qtbot):
    """CSV hides the image options; the background is opaque for a white
    background or JPG; the text sizes and white background start from the
    plot; and drawing the preview leaves the plot untouched."""
    widget = _histogram(qtbot)
    size = widget.fig.get_size_inches().copy()
    alpha = widget.fig.patch.get_alpha()
    dlg = _dialog(qtbot, widget)

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

    assert not dlg.is_opaque()
    # JPG has no transparency.
    _select(dlg.format_combo, "jpg")
    assert dlg.is_opaque()
    _select(dlg.format_combo, "svg")
    assert not dlg.is_opaque()
    dlg.white_bg_checkbox.setChecked(True)
    _select(dlg.format_combo, "png")
    assert dlg.is_opaque()

    # Drawing the preview does not resize or restyle the on-screen plot.
    dlg.aspect_combo.setCurrentIndex(dlg.aspect_combo.findText("16:9"))
    _select(dlg.format_combo, "jpg")
    assert widget.fig.get_size_inches() == pytest.approx(size)
    assert widget.fig.patch.get_alpha() == alpha

    assert dlg.text_size_spin.value() == HistogramWidget.DEFAULT_LABEL_FONTSIZE
    assert dlg.tick_size_spin.value() == HistogramWidget.DEFAULT_TICK_FONTSIZE
    assert not _dialog(qtbot, widget).white_bg_checkbox.isChecked()

    widget._label_fontsize = 8
    widget._tick_fontsize = 9
    widget._white_background = True
    dlg = _dialog(qtbot, widget)
    assert dlg.text_size_spin.value() == 8
    assert dlg.tick_size_spin.value() == 9
    assert dlg.white_bg_checkbox.isChecked()


def test_export_dialog_options(qtbot):
    """The dialog's options round-trip (in physical units or pixels), fall
    back to the defaults when stale or garbled, leave out the per-export
    background and legend, and its buttons accept or reject."""
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

    # The white background and legend location apply to one export only.
    dlg.white_bg_checkbox.setChecked(True)
    assert "white_background" not in dlg.options()
    assert "legend_location" not in dlg.options()

    # ...and are reported, with the text sizes, as the export style.
    dlg.text_size_spin.setValue(9.5)
    dlg.tick_size_spin.setValue(8.0)
    _select(dlg.legend_placement_combo, "outside")
    _select(dlg.legend_position_combo, "bottom")
    assert dlg.style() == {
        "white_background": True,
        "text_size": 9.5,
        "tick_size": 8.0,
        "legend_location": ("outside", "bottom"),
    }

    # Stale or garbled options fall back to the defaults.
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

    dlg = _dialog(qtbot, options={"unit": "in"})
    assert dlg.unit_combo.currentText() == "in"
    assert dlg.width_spin.value() == 5.0

    # "As shown" takes the height from the plot's own size, which differs
    # between machines, so the shape is fixed here.
    dlg = _dialog(qtbot, options={"unit": "px", "aspect": "Custom"})
    assert dlg.width_spin.value() == 1500
    assert dlg.height_spin.value() == 750
    dlg.width_spin.setValue(2000)
    again = _dialog(qtbot, options=dlg.options())
    assert again.unit_combo.currentText() == "px"
    assert again.width_spin.value() == 2000

    dlg.show()
    dlg.save_button.click()
    assert dlg.result() == QDialog.Accepted
    again.show()
    again.reject()
    assert again.result() == QDialog.Rejected


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


def test_export_writes_each_format(qtbot, tmp_path, monkeypatch):
    """PNG (transparent), JPG (opaque) and SVG (vector) are written at the
    requested size and shape, down to an odd number of pixels, and CSV goes
    through the CSV writer."""
    widget = _histogram(qtbot)

    def export(fmt, name, **kwargs):
        _accept_export(monkeypatch, fmt, **kwargs)
        out = tmp_path / name
        _save_to(monkeypatch, out)
        widget._open_export_dialog()
        return out

    with Image.open(export("png", "hist.png", width=4.0, dpi=100)) as image:
        assert image.format == "PNG"
        assert image.size == (400, 400)
        # Transparent unless the white background is on.
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 0

    out = export("png", "wide.png", aspect="16:9", width=8.0, dpi=100)
    with Image.open(out) as image:
        assert image.size == (800, 450)

    # JPG has no transparency, so the background is filled white.
    with Image.open(export("jpg", "hist.jpg", width=4.0, dpi=100)) as image:
        assert image.format == "JPEG"
        assert image.size == (400, 400)
        assert image.mode == "RGB"
        assert all(value > 240 for value in image.getpixel((0, 0)))

    text = export("svg", "hist.svg", width=4.0).read_text()
    assert "<svg" in text
    # 4 inches at 72 pt per inch.
    assert 'width="288pt"' in text

    # An odd pixel size is not truncated by a floating point remainder.
    for fmt in ("png", "jpg"):
        for width, height in ((1499, 777), (801, 333)):

            def in_pixels(dlg, width=width, height=height):
                dlg.unit_combo.setCurrentIndex(dlg.unit_combo.findText("px"))
                dlg.width_spin.setValue(width)
                dlg.height_spin.setValue(height)

            out = export(
                fmt,
                f"px_{width}.{fmt}",
                aspect="Custom",
                dpi=300,
                then=in_pixels,
            )
            with Image.open(out) as image:
                assert image.size == (width, height)

    header = export("csv", "hist.csv").read_text().splitlines()[0]
    assert header == "Bin Center,first,second"


def test_export_style_is_export_only(qtbot, tmp_path, monkeypatch):
    """Text sizes, legend location and background chosen for an export
    apply to the file only; the plot is restored afterwards (even when
    saving fails), and the options are remembered for the next export."""
    from unittest.mock import patch

    widget = _histogram(qtbot)
    size = widget.fig.get_size_inches().copy()
    grey = to_rgba("grey")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    # The style applies inside the context only.
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

    # Without any change nothing is redrawn.
    calls = []
    with patch.object(widget, "_render", lambda: calls.append(1)):
        with widget._export_style(None):
            pass
        with widget._export_style({"text_size": 6, "tick_size": 7}):
            pass
    assert calls == []

    # A failed save still restores the plot and undoes the style.
    with patch.object(widget.fig, "savefig", boom):
        with pytest.raises(OSError, match="disk full"):
            widget._export_figure("ignored.png", "png", 8.0, 2.0, 100)
        with pytest.raises(OSError, match="disk full"):
            widget._export_figure(
                "ignored.png",
                "png",
                4.0,
                2.0,
                100,
                style={"text_size": 20, "white_background": True},
            )
    assert widget.fig.get_size_inches() == pytest.approx(size)
    assert widget.ax.spines["left"].get_edgecolor() == grey
    assert widget._label_fontsize == 6
    assert widget._white_background is False

    # The next export starts from what the last one used.
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

    # Exporting leaves the on-screen figure as it found it.
    _accept_export(monkeypatch, "png", aspect="16:9", width=8.0, dpi=50)
    _save_to(monkeypatch, tmp_path / "hist.png")
    widget._open_export_dialog()
    assert widget.fig.get_size_inches() == pytest.approx(size)
    assert widget.fig.patch.get_alpha() == 0
    assert widget.ax.spines["left"].get_edgecolor() == grey

    # Ticking the white background fills the file only.
    _accept_export(
        monkeypatch,
        "png",
        dpi=50,
        then=lambda dlg: dlg.white_bg_checkbox.setChecked(True),
    )
    _save_to(monkeypatch, tmp_path / "white_once.png")
    widget._open_export_dialog()
    with Image.open(tmp_path / "white_once.png") as image:
        assert image.convert("RGBA").getpixel((0, 0)) == (255, 255, 255, 255)
    assert widget._white_background is False
    assert widget.fig.patch.get_alpha() == 0

    # The saved image can have an outside legend; the plot keeps its own.
    def outside_right(dlg):
        _select(dlg.legend_placement_combo, "outside")
        _select(dlg.legend_position_combo, "right")

    _accept_export(
        monkeypatch, "png", aspect="2:1", width=6.0, then=outside_right
    )
    _save_to(monkeypatch, tmp_path / "outside.png")
    widget._open_export_dialog()
    with Image.open(tmp_path / "outside.png") as image:
        alpha = np.asarray(image)[..., 3]
    assert alpha[:, int(alpha.shape[1] * 0.8) : -2].max() > 0
    assert widget._legend_placement == "inside"
    assert widget.fig.legends == []
    assert widget.ax.get_legend() is not None

    # The exported image has room for a legend the plot places outside.
    widget._legend_placement = "outside"
    widget._legend_position = "right"
    widget._render()
    _accept_export(monkeypatch, "png", aspect="2:1", width=6.0, dpi=100)
    _save_to(monkeypatch, tmp_path / "legend.png")
    widget._open_export_dialog()
    with Image.open(tmp_path / "legend.png") as image:
        assert image.size == (600, 300)
        alpha = np.asarray(image)[..., 3]
    # Something is drawn in the right-hand margin (the legend), and the
    # column at the very edge is blank.
    assert alpha[:, int(600 * 0.8) : -2].max() > 0
    assert alpha[:, -1].max() == 0
    assert len(widget.fig.legends) == 1
    widget._legend_placement = "inside"
    widget._legend_position = "upper right"
    widget._render()

    # Larger text takes more of the image, and the plot keeps its sizes.
    small = tmp_path / "small.png"
    large = tmp_path / "large.png"
    for out, text_size in ((small, 6.0), (large, 16.0)):

        def set_sizes(dlg, text_size=text_size):
            dlg.text_size_spin.setValue(text_size)
            dlg.tick_size_spin.setValue(text_size)

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

    # A plot with a white background exports opaque, unless unticked for
    # that export, which leaves the plot's own setting alone.
    widget._white_background = True
    _accept_export(monkeypatch, "png")
    _save_to(monkeypatch, tmp_path / "white.png")
    widget._open_export_dialog()
    with Image.open(tmp_path / "white.png") as image:
        assert image.convert("RGBA").getpixel((0, 0)) == (255, 255, 255, 255)
    _accept_export(
        monkeypatch,
        "png",
        dpi=50,
        then=lambda dlg: dlg.white_bg_checkbox.setChecked(False),
    )
    _save_to(monkeypatch, tmp_path / "clear.png")
    widget._open_export_dialog()
    with Image.open(tmp_path / "clear.png") as image:
        assert image.getpixel((0, 0))[3] == 0
    assert widget._white_background is True


def test_export_file_names_and_cancelling(qtbot, tmp_path, monkeypatch):
    """A rejected dialog or a cancelled file dialog writes nothing; a
    missing suffix is added, but upper case and .jpeg are kept."""
    widget = _histogram(qtbot)

    monkeypatch.setattr(
        HistogramExportDialog, "exec", lambda self: QDialog.Rejected
    )

    def fail(*args, **kwargs):
        raise AssertionError("nothing should be saved")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fail)
    widget._open_export_dialog()
    assert widget._export_options is None

    _accept_export(monkeypatch, "png", dpi=50)
    _save_to(monkeypatch, None)
    widget._open_export_dialog()
    assert list(tmp_path.iterdir()) == []

    for fmt, suffix in (("png", ".png"), ("svg", ".svg"), ("jpg", ".jpg")):
        _accept_export(monkeypatch, fmt, dpi=50)
        _save_to(monkeypatch, tmp_path / f"noext_{fmt}")
        widget._open_export_dialog()
        assert (tmp_path / f"noext_{fmt}{suffix}").exists()

    _accept_export(monkeypatch, "jpg", dpi=50)
    _save_to(monkeypatch, tmp_path / "photo.JPEG")
    widget._open_export_dialog()
    assert (tmp_path / "photo.JPEG").exists()
    _accept_export(monkeypatch, "png", dpi=50)
    _save_to(monkeypatch, tmp_path / "Figure.PNG")
    widget._open_export_dialog()
    assert (tmp_path / "Figure.PNG").exists()
    assert not (tmp_path / "Figure.PNG.png").exists()


# ---------------------------------------------------------------------------
# Size in pixels
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Text and tick sizes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# White background
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Legend location in the export dialog
# ---------------------------------------------------------------------------


def test_export_dialog_legend_location(qtbot):
    """The dialog's legend location starts from the plot, lists the
    positions of the chosen placement, moves the preview's legend but not
    the plot's, and is off (saying why) when the plot has no legend."""
    widget = _histogram(qtbot)
    assert widget.has_legend()
    dlg = _dialog(qtbot, widget)
    assert dlg.legend_position_combo.currentData() == "upper right"
    assert dlg.legend_position_combo.count() == len(LEGEND_POSITIONS["inside"])

    # The preview is drawn with the chosen legend location.
    before = dlg.preview_label.pixmap().toImage()
    _select(dlg.legend_placement_combo, "outside")
    after = dlg.preview_label.pixmap().toImage()
    assert before != after
    # The plot on screen has not moved.
    assert widget.ax.get_legend() is not None
    assert widget.fig.legends == []
    assert dlg.legend_position_combo.count() == len(
        LEGEND_POSITIONS["outside"]
    )
    assert dlg.legend_position_combo.currentData() == "right"
    _select(dlg.legend_position_combo, "top")
    assert dlg.style()["legend_location"] == ("outside", "top")

    widget._legend_placement = "outside"
    widget._legend_position = "bottom"
    widget._render()
    assert widget.has_legend()
    dlg = _dialog(qtbot, widget)
    assert dlg.legend_placement_combo.isEnabled()
    assert dlg.legend_position_combo.isEnabled()
    assert dlg.legend_placement_combo.currentData() == "outside"
    assert dlg.legend_position_combo.currentData() == "bottom"

    widget._show_legend = False
    widget._render()
    assert not widget.has_legend()

    # With no legend on the plot, the controls are off and say why.
    merged = _histogram(qtbot, mode="Merged")
    assert not merged.has_legend()
    dlg = _dialog(qtbot, merged)
    for control in (
        dlg._legend_label,
        dlg.legend_placement_combo,
        dlg.legend_position_combo,
    ):
        assert not control.isEnabled()
        assert "Show legend" in control.toolTip()
