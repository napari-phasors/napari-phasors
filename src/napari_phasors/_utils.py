"""
This module contains utility functions used by other modules.

"""

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon
from napari.layers import Image
from phasorpy.filter import (
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_threshold,
)
from qtpy.QtCore import QRect, QSize, Qt, Signal
from qtpy.QtGui import (
    QColor,
    QDoubleValidator,
    QFont,
    QFontMetrics,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
    QStandardItem,
    QStandardItemModel,
)
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import gaussian_filter1d
from superqt import QRangeSlider

if TYPE_CHECKING:
    import napari


def threshold_otsu(data, nbins=256):
    """Calculate Otsu's threshold for the given data.

    Otsu's method finds the threshold that minimizes the weighted
    within-class variance, which is equivalent to maximizing the
    between-class variance.

    Parameters
    ----------
    data : np.ndarray
        1D array of values (NaN values should be removed beforehand).
    nbins : int, optional
        Number of histogram bins. Default is 256.

    Returns
    -------
    float
        The optimal threshold value.

    References
    ----------
    .. [1] Otsu, N., "A Threshold Selection Method from Gray-Level
           Histograms", IEEE Transactions on Systems, Man, and
           Cybernetics, vol. 9, no. 1, pp. 62-66, 1979.
    """
    data = np.asarray(data, dtype=float).ravel()

    if data.size == 0:
        return 0.0

    min_val, max_val = data.min(), data.max()
    if min_val == max_val:
        return float(min_val)

    counts, bin_edges = np.histogram(
        data, bins=nbins, range=(min_val, max_val)
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Class probabilities and means for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]

    # Avoid division by zero
    mean1 = np.cumsum(counts * bin_centers) / np.maximum(weight1, 1)
    mean2 = np.cumsum((counts * bin_centers)[::-1])[::-1] / np.maximum(
        weight2, 1
    )

    # Between-class variance
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance)
    return float(bin_centers[idx])


def threshold_li(data, initial_guess=None, tolerance=None):
    """Calculate Li's minimum cross-entropy threshold.

    Li's iterative method finds the threshold that minimizes the
    cross-entropy between the foreground and background distributions.

    This implementation matches scikit-image's ``threshold_li`` by
    shifting the data so that the minimum is zero before iterating,
    then adding the shift back to the final threshold.

    Parameters
    ----------
    data : np.ndarray
        1D array of values (NaN values should be removed beforehand).
    initial_guess : float, optional
        Starting threshold. If None, uses the mean of the data.
    tolerance : float, optional
        Convergence tolerance. If None, uses half the smallest
        difference between unique values.

    Returns
    -------
    float
        The optimal threshold value.

    References
    ----------
    .. [1] Li, C.H. and Lee, C.K., "Minimum Cross Entropy Thresholding",
           Pattern Recognition, vol. 26, no. 4, pp. 617-625, 1993.
    .. [2] Li, C.H. and Tam, P.K.S., "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding", Pattern Recognition Letters,
           vol. 19, no. 8, pp. 771-776, 1998.
    """
    data = np.asarray(data, dtype=float).ravel()

    if data.size == 0:
        return 0.0

    min_val, max_val = data.min(), data.max()
    if min_val == max_val:
        return float(min_val)

    # Shift data so that minimum is 0 (Li's method requires positive values
    # and the shift affects the threshold due to the logarithm).
    image_min = float(min_val)
    data = data - image_min

    if tolerance is None:
        sorted_unique = np.unique(data)
        if len(sorted_unique) > 1:
            diffs = np.diff(sorted_unique)
            tolerance = diffs[diffs > 0].min() / 2.0
        else:
            return float(min_val)

    # Initialise with the convention used by scikit-image:
    # t_next holds the *candidate* threshold, t_curr the *previous* one.
    if initial_guess is None:
        t_next = float(np.mean(data))
    else:
        t_next = float(initial_guess) - image_min

    t_curr = -2 * tolerance  # ensure first iteration always runs

    while abs(t_next - t_curr) > tolerance:
        t_curr = t_next
        foreground = data > t_curr
        mean_fore = np.mean(data[foreground])
        mean_back = np.mean(data[~foreground])

        if mean_back == 0.0:
            break

        t_next = (mean_back - mean_fore) / (
            np.log(mean_back) - np.log(mean_fore)
        )

    return float(t_next + image_min)


def threshold_yen(data, nbins=256):
    """Calculate Yen's threshold.

    Yen's method maximizes the correlation between the original and
    thresholded images in terms of their entropy.

    Parameters
    ----------
    data : np.ndarray
        1D array of values (NaN values should be removed beforehand).
    nbins : int, optional
        Number of histogram bins. Default is 256.

    Returns
    -------
    float
        The optimal threshold value.

    References
    ----------
    .. [1] Yen, J.C., Chang, F.J., and Chang, S., "A New Criterion for
           Automatic Multilevel Thresholding", IEEE Transactions on Image
           Processing, vol. 4, no. 3, pp. 370-378, 1995.
    """
    data = np.asarray(data, dtype=float).ravel()

    if data.size == 0:
        return 0.0

    min_val, max_val = data.min(), data.max()
    if min_val == max_val:
        return float(min_val)

    counts, bin_edges = np.histogram(
        data, bins=nbins, range=(min_val, max_val)
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Normalize to probabilities
    total = counts.sum()
    if total == 0:
        return float(min_val)

    pmf = counts.astype(float) / total

    # Cumulative sums
    P1 = np.cumsum(pmf)  # P(class 1)
    P1_sq = np.cumsum(pmf**2)  # sum of p_i^2 for class 1
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]  # sum of p_j^2 for class 2

    # Yen's criterion (Eq. 4 in [1]):
    #   crit = log( (P1*(1-P1))^2 / (P1_sq * P2_sq) )
    crit = np.log(
        ((P1_sq[:-1] * P2_sq[1:]) ** -1) * (P1[:-1] * (1.0 - P1[:-1])) ** 2
    )

    idx = np.argmax(crit)
    return float(bin_centers[idx])


def validate_harmonics_for_wavelet(harmonics):
    """Validate that harmonics have their double or half correspondent.

    Parameters
    ----------
    harmonics : array-like
        Array of harmonic values

    Returns
    -------
    bool
        True if harmonics are valid for wavelet filtering, False otherwise
    """
    harmonics = np.atleast_1d(harmonics)

    for harmonic in harmonics:
        # Check if double or half exists
        has_double = (harmonic * 2) in harmonics
        has_half = (harmonic / 2) in harmonics

        if not (has_double or has_half):
            return False

    return True


def _extract_phasor_arrays_from_layer(
    layer: Image, harmonics: np.ndarray = None
):
    """Extract phasor arrays from layer metadata.

    Parameters
    ----------
    layer : napari.layers.Image
        Napari image layer with phasor features.
    harmonics : np.ndarray, optional
        Harmonic values. If None, will be extracted from layer.

    Returns
    -------
    tuple
        (mean, real, imag, harmonics) arrays
    """
    mean = layer.metadata['original_mean'].copy()

    if harmonics is None:
        harmonics = layer.metadata.get('harmonics')

    harmonics = np.atleast_1d(harmonics)

    real = layer.metadata['G_original'].copy()
    imag = layer.metadata['S_original'].copy()

    # Apply mask if present in metadata
    if 'mask' in layer.metadata:
        mask = layer.metadata['mask']
        # Apply mask: set values to NaN where mask <= 0
        mask_invalid = mask <= 0
        mean = np.where(mask_invalid, np.nan, mean)
        for h in range(len(harmonics)):
            real[h] = np.where(mask_invalid, np.nan, real[h])
            imag[h] = np.where(mask_invalid, np.nan, imag[h])

    return mean, real, imag, harmonics


def _apply_filter_and_threshold_to_phasor_arrays(
    mean: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    harmonics: np.ndarray,
    *,
    threshold: float = None,
    threshold_upper: float = None,
    filter_method: str = None,
    size: int = None,
    repeat: int = None,
    sigma: float = None,
    levels: int = None,
):
    """Apply filter and threshold to phasor arrays.

    Parameters
    ----------
    mean : np.ndarray
        Mean intensity array.
    real : np.ndarray
        Real part of phasor (G).
    imag : np.ndarray
        Imaginary part of phasor (S).
    harmonics : np.ndarray
        Harmonic values.
    threshold : float, optional
        Lower threshold value for the mean value to be applied to G and S.
        If None, no lower threshold is applied.
    threshold_upper : float, optional
        Upper threshold value for the mean value to be applied to G and S.
        If None, no upper threshold is applied.
    filter_method : str, optional
        Filter method. Options are 'median' or 'wavelet'.
        If None, no filter is applied.
    size : int, optional
        Size of the median filter. Only used if filter_method is 'median'.
    repeat : int, optional
        Number of times to apply the median filter. Only used if filter_method is 'median'.
    sigma : float, optional
        Sigma parameter for wavelet filter. Only used if filter_method is 'wavelet'.
    levels : int, optional
        Number of levels for wavelet filter. Only used if filter_method is 'wavelet'.

    Returns
    -------
    tuple
        (mean, real, imag) filtered and thresholded arrays
    """
    if filter_method == "median" and repeat is not None and repeat > 0:
        mean, real, imag = phasor_filter_median(
            mean,
            real,
            imag,
            repeat=repeat,
            size=size if size is not None else 3,
        )
    elif filter_method == "wavelet" and validate_harmonics_for_wavelet(
        harmonics
    ):
        mean, real, imag = phasor_filter_pawflim(
            mean,
            real,
            imag,
            sigma=sigma if sigma is not None else 1.0,
            levels=levels if levels is not None else 3,
            harmonic=harmonics,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean, real, imag = phasor_threshold(
            mean, real, imag, mean_min=threshold, mean_max=threshold_upper
        )

    return mean, real, imag


def apply_filter_and_threshold(
    layer: Image,
    /,
    *,
    threshold: float = None,
    threshold_upper: float = None,
    threshold_method: str = None,
    filter_method: str = None,
    size: int = None,
    repeat: int = None,
    sigma: float = None,
    levels: int = None,
    harmonics: np.ndarray = None,
):
    """Apply filter to an image layer.

    Parameters
    ----------
    layer : napari.layers.Image
        Napari image layer with phasor features.
    threshold : float, optional
        Lower threshold value for the mean value to be applied to G and S.
        If None, no lower threshold is applied.
    threshold_upper : float, optional
        Upper threshold value for the mean value to be applied to G and S.
        If None, no upper threshold is applied.
    threshold_method : str, optional
        Threshold method used. If None, no threshold method is saved.
    filter_method : str, optional
        Filter method. Options are 'median' or 'wavelet'.
        If None, no filter is applied.
    size : int, optional
        Size of the median filter. Only used if filter_method is 'median'.
    repeat : int, optional
        Number of times to apply the median filter. Only used if filter_method is 'median'.
    sigma : float, optional
        Sigma parameter for wavelet filter. Only used if filter_method is 'wavelet'.
    levels : int, optional
        Number of levels for wavelet filter. Only used if filter_method is 'wavelet'.
    harmonics : np.ndarray, optional
        Harmonic values for wavelet filter. If None, will be extracted from layer.

    """
    mean, real, imag, harmonics = _extract_phasor_arrays_from_layer(
        layer, harmonics
    )

    mean, real, imag = _apply_filter_and_threshold_to_phasor_arrays(
        mean,
        real,
        imag,
        harmonics,
        threshold=threshold,
        threshold_upper=threshold_upper,
        filter_method=filter_method,
        size=size,
        repeat=repeat,
        sigma=sigma,
        levels=levels,
    )

    layer.metadata['G'] = real
    layer.metadata['S'] = imag
    layer.data = mean

    if "settings" not in layer.metadata:
        layer.metadata["settings"] = {}

    # Only save filter settings if a filter was actually applied
    if filter_method is not None:
        layer.metadata["settings"]["filter"] = {}
        layer.metadata["settings"]["filter"]["method"] = filter_method

        if filter_method == "median":
            if size is not None:
                layer.metadata["settings"]["filter"]["size"] = size
            if repeat is not None:
                layer.metadata["settings"]["filter"]["repeat"] = repeat
        elif filter_method == "wavelet":
            if sigma is not None:
                layer.metadata["settings"]["filter"]["sigma"] = sigma
            if levels is not None:
                layer.metadata["settings"]["filter"]["levels"] = levels

    layer.metadata["settings"]["threshold"] = threshold
    layer.metadata["settings"]["threshold_upper"] = threshold_upper
    layer.metadata["settings"]["threshold_method"] = threshold_method
    layer.refresh()

    return


def colormap_to_dict(colormap, num_colors=10, exclude_first=True):
    """
    Converts a matplotlib colormap into a dictionary of RGBA colors.

    Parameters
    ----------
    colormap : matplotlib.colors.Colormap
        The colormap to convert.
    num_colors : int, optional
        The number of colors in the colormap, by default 10.
    exclude_first : bool, optional
        Whether to exclude the first color in the colormap, by default True.

    Returns
    -------
    color_dict: dict
        A dictionary with keys as positive integers and values as RGBA colors.
    """
    color_dict = {}
    start = 0
    if exclude_first:
        start = 1
    for i in range(start, num_colors + start):
        pos = i / (num_colors - 1)
        color = colormap(pos)
        color_dict[i + 1 - start] = color
    color_dict[None] = (0, 0, 0, 0)
    return color_dict


def update_frequency_in_metadata(
    image_layer: "napari.layers.Image",
    frequency: float,
):
    """Update the frequency in the layer metadata."""
    if "settings" not in image_layer.metadata:
        image_layer.metadata["settings"] = {}
    image_layer.metadata["settings"]["frequency"] = frequency


class _PrimaryLayerDelegate(QStyledItemDelegate):
    """Custom delegate that renders items with a "Set as primary" action or
    a "Primary layer" indicator on the right side of each row."""

    # Role used to store whether an item is the primary layer
    PRIMARY_ROLE = Qt.UserRole + 100

    def __init__(self, parent=None, enable_primary_layer=True):
        super().__init__(parent)
        self._action_font = QFont()
        self._action_font.setUnderline(True)
        self._label_font = QFont()
        self._label_font.setItalic(True)
        self._hovered_index = None
        self._enable_primary_layer = enable_primary_layer

    def sizeHint(self, option, index):
        base = super().sizeHint(option, index)
        return QSize(base.width(), max(base.height(), 24))

    def paint(self, painter, option, index):
        painter.save()

        self.initStyleOption(option, index)
        style = option.widget.style() if option.widget else QComboBox().style()
        style.drawPrimitive(
            QStyle.PE_PanelItemViewItem, option, painter, option.widget
        )

        rect = option.rect
        is_primary = index.data(self.PRIMARY_ROLE)

        check_state = index.data(Qt.CheckStateRole)
        cb_option = QStyleOptionButton()
        cb_size = 16
        cb_margin = 4
        cb_option.rect = QRect(
            rect.left() + cb_margin,
            rect.top() + (rect.height() - cb_size) // 2,
            cb_size,
            cb_size,
        )
        if check_state == Qt.Checked:
            cb_option.state = QStyle.State_On | QStyle.State_Enabled
        else:
            cb_option.state = QStyle.State_Off | QStyle.State_Enabled
        style.drawControl(QStyle.CE_CheckBox, cb_option, painter)

        text_left = rect.left() + cb_margin + cb_size + cb_margin

        right_margin = 8
        label_text = None
        label_color = None

        is_hovered = (
            index.row() == getattr(self._hovered_index, 'row', lambda: -1)()
            if self._hovered_index
            else False
        )

        if self._enable_primary_layer:
            if is_primary:
                label_text = "Primary layer"
                painter.setFont(self._label_font)
                label_color = QColor(255, 100, 100)
            elif is_hovered:
                if check_state == Qt.Checked:
                    label_text = "Set as primary"
                    painter.setFont(self._action_font)
                    label_color = QColor(180, 180, 220)

        label_rect = QRect()
        if label_text:
            fm_label = QFontMetrics(painter.font())
            label_width = fm_label.horizontalAdvance(label_text) + 4
            label_rect = QRect(
                rect.right() - label_width - right_margin,
                rect.top(),
                label_width,
                rect.height(),
            )
            painter.setPen(QPen(label_color))
            painter.drawText(
                label_rect, Qt.AlignVCenter | Qt.AlignRight, label_text
            )
        else:
            label_rect = QRect(rect.right(), rect.top(), 0, rect.height())

        name = index.data(Qt.DisplayRole) or ""
        text_right = label_rect.left() - 6
        available = text_right - text_left
        painter.setFont(option.font)
        fm_name = QFontMetrics(option.font)
        elided = fm_name.elidedText(name, Qt.ElideRight, max(available, 30))

        if option.state & QStyle.State_Selected:
            painter.setPen(QPen(option.palette.highlightedText().color()))
        else:
            painter.setPen(QPen(option.palette.text().color()))

        name_rect = QRect(text_left, rect.top(), available, rect.height())
        painter.drawText(name_rect, Qt.AlignVCenter | Qt.AlignLeft, elided)

        painter.restore()

    def labelRect(self, option, index):
        """Return the QRect of the right-side label for hit testing."""
        if not self._enable_primary_layer:
            return QRect()

        rect = option.rect
        is_primary = index.data(self.PRIMARY_ROLE)

        label_text = None
        font = None

        if is_primary:
            label_text = "Primary layer"
            font = self._label_font
        else:
            check_state = index.data(Qt.CheckStateRole)
            is_hovered = (
                index.row()
                == getattr(self._hovered_index, 'row', lambda: -1)()
                if self._hovered_index
                else False
            )
            if is_hovered and check_state == Qt.Checked:
                label_text = "Set as primary"
                font = self._action_font

        if not label_text:
            return QRect()

        fm = QFontMetrics(font)
        label_width = fm.horizontalAdvance(label_text) + 4
        right_margin = 8
        return QRect(
            rect.right() - label_width - right_margin,
            rect.top(),
            label_width,
            rect.height(),
        )


class CheckableComboBox(QComboBox):
    """A ComboBox with checkable items for multi-selection.

    Displays selected items and emits selectionChanged signal when items
    are checked/unchecked.  The *primary* layer (used for metadata and
    settings) can be set by clicking "Set as primary" next to any
    checked item in the dropdown.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    enable_primary_layer : bool, optional
        Whether to enable primary layer functionality (default: True).
    """

    selectionChanged = Signal()
    primaryLayerChanged = Signal(str)  # Emits the new primary layer name

    def __init__(self, parent=None, enable_primary_layer=True):
        super().__init__(parent)
        self.setModel(QStandardItemModel(self))
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Select Layers...")

        self._enable_primary_layer = enable_primary_layer

        # Custom delegate with "Set as primary" support
        self._delegate = _PrimaryLayerDelegate(self, enable_primary_layer)
        self.setItemDelegate(self._delegate)

        # Track primary layer name
        self._primary_layer_name = ""

        # Connect model signals
        self.model().dataChanged.connect(self._on_data_changed)

        # Track if we're inside the popup
        self._popup_visible = False

        # Track the last known primary for change detection
        self._last_emitted_primary = ""

        # Track star icon action for multi-selection display
        self._star_action = None

        # Make the line edit clickable to open popup
        self.lineEdit().installEventFilter(self)
        # Prevent cursor positioning in line edit
        self.lineEdit().setFocusPolicy(Qt.NoFocus)

        # Install event filter on view to handle item clicks and hover
        self.view().viewport().installEventFilter(self)
        self.view().setMouseTracking(True)

    def selectAll(self):
        """Check all items (emits one selectionChanged)."""
        self.blockSignals(True)
        for i in range(self.model().rowCount()):
            self.model().item(i).setCheckState(Qt.Checked)
        self.blockSignals(False)
        self._refresh_primary_and_notify()

    def deselectAll(self):
        """Uncheck all items (emits one selectionChanged)."""
        self.blockSignals(True)
        for i in range(self.model().rowCount()):
            self.model().item(i).setCheckState(Qt.Unchecked)
        self.blockSignals(False)
        self._refresh_primary_and_notify()

    def eventFilter(self, obj, event):
        """Filter events to make line edit clickable and handle item clicks."""
        if obj == self.lineEdit():
            if event.type() == event.MouseButtonRelease:
                if not self.view().isVisible():
                    self.showPopup()
                return True
            elif event.type() == event.MouseButtonPress:
                return True
        elif obj == self.view().viewport():
            if event.type() == event.MouseMove:
                index = self.view().indexAt(event.pos())
                old_hover = self._delegate._hovered_index
                self._delegate._hovered_index = (
                    index if index.isValid() else None
                )
                if old_hover != self._delegate._hovered_index:
                    self.view().viewport().update()
                return False
            elif event.type() == event.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                if index.isValid():
                    vis_rect = self.view().visualRect(index)
                    option = QStyleOptionViewItem()
                    option.rect = vis_rect
                    label_rect = self._delegate.labelRect(option, index)

                    if (
                        self._enable_primary_layer
                        and label_rect.contains(event.pos())
                        and not label_rect.isEmpty()
                    ):
                        is_primary = index.data(
                            _PrimaryLayerDelegate.PRIMARY_ROLE
                        )
                        if not is_primary:
                            item = self.model().itemFromIndex(index)
                            if item:
                                if item.checkState() != Qt.Checked:
                                    item.setCheckState(Qt.Checked)
                                self._set_primary_by_name(item.text())
                        return True
                    else:
                        item = self.model().itemFromIndex(index)
                        if item:
                            current_state = item.checkState()
                            new_state = (
                                Qt.Unchecked
                                if current_state == Qt.Checked
                                else Qt.Checked
                            )
                            item.setCheckState(new_state)
                        return True
            elif event.type() == event.Leave:
                if self._delegate._hovered_index is not None:
                    self._delegate._hovered_index = None
                    self.view().viewport().update()
        return super().eventFilter(obj, event)

    def addItem(self, text, checked=False):
        """Add a checkable item to the combobox."""
        item = QStandardItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(
            Qt.Checked if checked else Qt.Unchecked, Qt.CheckStateRole
        )
        item.setData(False, _PrimaryLayerDelegate.PRIMARY_ROLE)
        self.model().appendRow(item)
        if checked and not self._primary_layer_name:
            self._set_primary_by_name(text, emit=False)

    def addItems(self, texts):
        """Add multiple items to the combobox."""
        for text in texts:
            self.addItem(text)

    def clear(self):
        """Clear all items."""
        self.model().clear()
        self._primary_layer_name = ""
        self._last_emitted_primary = ""
        self._update_display_text()

    def checkedItems(self):
        """Return list of checked item texts in list order (top to bottom)."""
        checked = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item and item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked

    def allItems(self):
        """Return list of all item texts in list order."""
        return [
            self.model().item(i).text()
            for i in range(self.model().rowCount())
            if self.model().item(i)
        ]

    def getPrimaryLayer(self):
        """Return the name of the primary layer."""
        checked = self.checkedItems()
        if self._primary_layer_name not in checked:
            self._primary_layer_name = checked[0] if checked else ""
            self._sync_primary_role()
        return self._primary_layer_name

    def setPrimaryLayer(self, name):
        """Set the primary layer by name.

        The layer must already be checked.
        """
        checked = self.checkedItems()
        if name in checked:
            self._set_primary_by_name(name)

    def setCheckedItems(self, texts):
        """Set which items are checked by their text."""
        # Check if signals are already blocked by parent
        signals_were_blocked = self.signalsBlocked()

        if not signals_were_blocked:
            self.blockSignals(True)

        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item.text() in texts:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

        if not signals_were_blocked:
            self.blockSignals(False)

        if texts and self._primary_layer_name not in texts:
            self._primary_layer_name = texts[0]
        elif not texts:
            self._primary_layer_name = ""
        self._sync_primary_role()

        # Only emit signals if they weren't blocked by parent
        if not signals_were_blocked:
            self._refresh_primary_and_notify()

    def _set_primary_by_name(self, name, emit=True):
        """Set the primary layer and update role data on all items."""
        old = self._primary_layer_name
        self._primary_layer_name = name
        self._sync_primary_role()
        self._update_display_text()
        if emit and old != name:
            self._last_emitted_primary = name
            self.primaryLayerChanged.emit(name)

    def _sync_primary_role(self):
        """Update the PRIMARY_ROLE on every item to match _primary_layer_name."""
        model = self.model()
        for i in range(model.rowCount()):
            item = model.item(i)
            if item:
                item.setData(
                    item.text() == self._primary_layer_name,
                    _PrimaryLayerDelegate.PRIMARY_ROLE,
                )

    def _on_data_changed(self, topLeft, bottomRight, roles):
        """Handle item check state changes."""
        if Qt.CheckStateRole in roles:
            self._refresh_primary_and_notify()

    def _refresh_primary_and_notify(self):
        """Recalculate primary, emit appropriate signals."""
        checked = self.checkedItems()
        old_primary = self._last_emitted_primary

        if self._primary_layer_name not in checked:
            self._primary_layer_name = checked[0] if checked else ""
        elif not self._primary_layer_name and checked:
            self._primary_layer_name = checked[0]

        self._sync_primary_role()
        self._update_display_text()

        new_primary = self._primary_layer_name
        if not self.signalsBlocked():
            if old_primary != new_primary:
                self._last_emitted_primary = new_primary
                self.primaryLayerChanged.emit(new_primary)
            self.selectionChanged.emit()

    def _create_star_icon(self):
        """Create a star icon for the line edit using Qt rendering."""
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor(255, 200, 0)))  # Gold color
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "★")
        painter.end()
        return QIcon(pixmap)

    def _update_display_text(self):
        """Update the display text to show primary layer and selection count."""
        checked = self.checkedItems()
        line_edit = self.lineEdit()

        if self._star_action:
            line_edit.removeAction(self._star_action)
            self._star_action = None

        if not checked:
            line_edit.setText("")
            line_edit.setPlaceholderText("Select Layers...")
        elif len(checked) == 1:
            line_edit.setText(checked[0])
        else:
            if self._enable_primary_layer:
                star_icon = self._create_star_icon()
                self._star_action = line_edit.addAction(
                    star_icon, QLineEdit.LeadingPosition
                )
                primary = self._primary_layer_name or checked[0]
                others = len(checked) - 1
                suffix = "selected layer" if others == 1 else "selected layers"
                line_edit.setText(f"{primary}  + {others} {suffix}")
            else:
                # Without primary layer, just show count
                suffix = "layers" if len(checked) > 1 else "layer"
                line_edit.setText(f"{len(checked)} {suffix} selected")

    def showPopup(self):
        """Show the popup and track visibility."""
        self._popup_visible = True
        super().showPopup()

    def hidePopup(self):
        """Hide the popup and clear hover state."""
        self._popup_visible = False
        self._delegate._hovered_index = None
        super().hidePopup()

    def itemCheckState(self, index):
        """Get the check state of item at index."""
        item = self.model().item(index)
        return item.checkState() if item else Qt.Unchecked

    def setItemCheckState(self, index, state):
        """Set the check state of item at index."""
        item = self.model().item(index)
        if item:
            item.setCheckState(state)


class HistogramSettingsDialog(QDialog):
    """Dialog for histogram visualization settings.

    Provides controls for:
    - Display mode: Merged / Individual layers / Grouped.
    - Toggling SD shading (for Merged and Grouped modes).
    - Central-tendency vertical line (Mean / Median / Center of mass).
    - Show / hide legend.
    - Per-layer colour selection (Individual layers mode).
    - Group assignment and per-group colour (Grouped mode).

    Parameters
    ----------
    display_mode : str
        Initial display mode.
    show_sd : bool
        Initial state of the *Show standard deviation* checkbox.
    central_tendency : str
        Initial central-tendency line selection.
    show_legend : bool
        Initial state of the *Show legend* checkbox.
    layer_labels : list of str, optional
        Layer names for group assignment.
    group_assignments : dict, optional
        ``{label: group_int}`` initial group assignments.
    layer_colors : dict, optional
        ``{label: (r, g, b)}`` initial per-layer colours (0-1 floats).
    group_colors : dict, optional
        ``{group_id: (r, g, b)}`` initial per-group colours (0-1 floats).
    group_names : dict, optional
        ``{group_id: str}`` initial per-group display names.
    parent : QWidget, optional
        Parent widget.
    """

    DISPLAY_MODES = ("Merged", "Individual layers", "Grouped")
    CENTRAL_TENDENCY_OPTIONS = (
        "None",
        "Center of mass",
        "Mean",
        "Median",
    )
    MAX_GROUPS = 6

    def __init__(
        self,
        display_mode: str = "Merged",
        show_sd: bool = False,
        central_tendency: str = "None",
        show_legend: bool = False,
        layer_labels: list = None,
        group_assignments: dict = None,
        layer_colors: dict = None,
        group_colors: dict = None,
        group_names: dict = None,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Histogram Settings")
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)

        # --- Display mode ---
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Display mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(self.DISPLAY_MODES))
        self.mode_combo.setCurrentText(display_mode)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # --- Show SD ---
        self.sd_checkbox = QCheckBox("Show standard deviation")
        self.sd_checkbox.setChecked(show_sd)
        layout.addWidget(self.sd_checkbox)

        # --- Central tendency ---
        ct_layout = QHBoxLayout()
        ct_layout.addWidget(QLabel("Show line:"))
        self.central_tendency_combo = QComboBox()
        self.central_tendency_combo.addItems(
            list(self.CENTRAL_TENDENCY_OPTIONS)
        )
        self.central_tendency_combo.setCurrentText(central_tendency)
        ct_layout.addWidget(self.central_tendency_combo)
        layout.addLayout(ct_layout)

        # --- Show legend ---
        self.legend_checkbox = QCheckBox("Show legend")
        self.legend_checkbox.setChecked(show_legend)
        layout.addWidget(self.legend_checkbox)

        # --- White background ---
        self.white_bg_checkbox = QCheckBox("White background")
        self.white_bg_checkbox.setChecked(False)
        layout.addWidget(self.white_bg_checkbox)

        # --- Smooth curves ---
        self.smooth_checkbox = QCheckBox("Smooth curves")
        self.smooth_checkbox.setChecked(True)
        layout.addWidget(self.smooth_checkbox)

        # --- Layer colours (Individual layers mode) ---
        default_tab10 = plt.cm.tab10.colors

        self._layer_section = QWidget()
        layer_sec_layout = QVBoxLayout(self._layer_section)
        layer_sec_layout.setContentsMargins(0, 0, 0, 0)
        layer_sec_layout.addWidget(QLabel("Layer colours:"))
        self._layer_color_buttons = {}
        if layer_labels:
            for idx, label in enumerate(layer_labels):
                row = QHBoxLayout()
                name_lbl = QLabel(label)
                name_lbl.setMaximumWidth(200)
                row.addWidget(name_lbl)
                if layer_colors and label in layer_colors:
                    color = layer_colors[label]
                else:
                    color = default_tab10[idx % len(default_tab10)][:3]
                btn = QPushButton()
                btn.setFixedSize(24, 24)
                self._set_btn_color(btn, color)
                btn.clicked.connect(lambda checked, b=btn: self._pick_color(b))
                row.addWidget(btn)
                layer_sec_layout.addLayout(row)
                self._layer_color_buttons[label] = btn
        layout.addWidget(self._layer_section)

        # --- Group section (Grouped mode) ---
        self._group_section = QWidget()
        group_layout = QVBoxLayout(self._group_section)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.addWidget(QLabel("Groups:"))

        # Scroll area to hold group rows
        self._group_rows_widget = QWidget()
        self._group_rows_layout = QVBoxLayout(self._group_rows_widget)
        self._group_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._group_rows_layout.setSpacing(4)
        group_layout.addWidget(self._group_rows_widget)

        # Store group row data: list of dicts with keys:
        #   'container', 'name_edit', 'color_btn', 'layer_combo'
        self._group_row_data = []
        self._layer_labels = layer_labels or []

        # Populate groups from existing assignments
        if group_assignments and layer_labels:
            # Infer groups from assignments
            groups_seen = {}
            for label, gid in group_assignments.items():
                groups_seen.setdefault(gid, []).append(label)
            for gid in sorted(groups_seen):
                default_c = default_tab10[(gid - 1) % len(default_tab10)][:3]
                gc = (
                    group_colors[gid]
                    if group_colors and gid in group_colors
                    else default_c
                )
                gname = (
                    group_names.get(gid, f"Group {gid}")
                    if group_names
                    else f"Group {gid}"
                )
                self._add_group_row(
                    name=gname,
                    color=gc,
                    checked_layers=groups_seen[gid],
                )
        else:
            # Start with one empty group
            gc = default_tab10[0][:3]
            self._add_group_row(
                name="Group 1",
                color=gc,
                checked_layers=layer_labels or [],
            )

        # Add group button
        add_group_btn = QPushButton("+ Add Group")
        add_group_btn.setMaximumWidth(120)
        add_group_btn.clicked.connect(self._on_add_group)
        group_layout.addWidget(add_group_btn)

        layout.addWidget(self._group_section)

        # Show / hide sections based on mode
        self._update_ui_for_mode(display_mode)
        self.mode_combo.currentTextChanged.connect(self._update_ui_for_mode)

        # --- OK / Cancel ---
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_btn_color(btn, color):
        """Set a button's background from an (r, g, b) 0-1 float tuple."""
        r = int(color[0] * 255)
        g = int(color[1] * 255)
        b = int(color[2] * 255)
        btn.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border: 1px solid grey;"
        )
        btn._color = tuple(color[:3])

    def _pick_color(self, btn):
        """Open a colour picker for *btn*."""
        cur = btn._color
        initial = QColor.fromRgbF(cur[0], cur[1], cur[2])
        chosen = QColorDialog.getColor(initial, self)
        if chosen.isValid():
            rgb = (chosen.redF(), chosen.greenF(), chosen.blueF())
            self._set_btn_color(btn, rgb)

    def _update_ui_for_mode(self, mode: str) -> None:
        """Show/hide controls depending on the selected mode."""
        is_grouped = mode == "Grouped"
        is_individual = mode == "Individual layers"
        self._group_section.setVisible(is_grouped)
        self._layer_section.setVisible(is_individual)
        # SD only meaningful for Merged / Grouped
        self.sd_checkbox.setEnabled(not is_individual)
        # Legend only meaningful for Individual / Grouped
        self.legend_checkbox.setEnabled(is_individual or is_grouped)

    # ------------------------------------------------------------------
    # Group row management
    # ------------------------------------------------------------------

    def _add_group_row(
        self,
        name: str = "Group",
        color=None,
        checked_layers: list = None,
    ) -> None:
        """Add a new group row to the group section.

        Parameters
        ----------
        name : str
            Default name for the group.
        color : tuple, optional
            (r, g, b) colour in 0-1 floats. If None, auto-assigned.
        checked_layers : list, optional
            Layer names to pre-check.
        """
        default_tab10 = plt.cm.tab10.colors
        idx = len(self._group_row_data)
        if color is None:
            color = default_tab10[idx % len(default_tab10)][:3]

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        # Name edit
        name_edit = QLineEdit(name)
        name_edit.setMaximumWidth(120)
        name_edit.setPlaceholderText("Name")
        row_layout.addWidget(name_edit)

        # Color button
        color_btn = QPushButton()
        color_btn.setFixedSize(24, 24)
        self._set_btn_color(color_btn, color)
        color_btn.clicked.connect(
            lambda checked, b=color_btn: self._pick_color(b)
        )
        row_layout.addWidget(color_btn)

        # Checkable combobox for layer selection
        layer_combo = CheckableComboBox(
            placeholder="Select layers...", parent=self
        )
        layer_combo.addItems(self._layer_labels)
        if checked_layers:
            layer_combo.setCheckedItems(checked_layers)
        row_layout.addWidget(layer_combo, 1)

        # Remove button
        remove_btn = QPushButton("\u2212")  # minus sign
        remove_btn.setFixedSize(24, 24)
        remove_btn.setToolTip("Remove this group")
        remove_btn.clicked.connect(lambda: self._on_remove_group(row_widget))
        row_layout.addWidget(remove_btn)

        self._group_rows_layout.addWidget(row_widget)
        self._group_row_data.append(
            {
                "container": row_widget,
                "name_edit": name_edit,
                "color_btn": color_btn,
                "layer_combo": layer_combo,
            }
        )

    def _on_add_group(self) -> None:
        """Slot for the *Add Group* button."""
        idx = len(self._group_row_data) + 1
        self._add_group_row(name=f"Group {idx}")

    def _on_remove_group(self, row_widget: QWidget) -> None:
        """Remove a group row by its container widget."""
        if len(self._group_row_data) <= 1:
            return  # always keep at least one group
        for i, data in enumerate(self._group_row_data):
            if data["container"] is row_widget:
                self._group_rows_layout.removeWidget(row_widget)
                row_widget.deleteLater()
                self._group_row_data.pop(i)
                break

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_group_assignments(self) -> dict:
        """Return ``{label: group_int}`` from the dialog.

        Groups are numbered starting from 1 in the order they appear.
        A layer that is checked in multiple groups is assigned to the
        first group that contains it.
        """
        assignments = {}
        for gid_zero, row in enumerate(self._group_row_data):
            gid = gid_zero + 1
            for layer in row["layer_combo"].checkedItems():
                if layer not in assignments:
                    assignments[layer] = gid
        return assignments

    def get_group_names(self) -> dict:
        """Return ``{group_id: str}`` from the dialog."""
        return {
            i + 1: row["name_edit"].text() or f"Group {i + 1}"
            for i, row in enumerate(self._group_row_data)
        }

    def get_layer_colors(self) -> dict:
        """Return ``{label: (r, g, b)}`` from the dialog."""
        return {
            label: btn._color
            for label, btn in self._layer_color_buttons.items()
        }

    def get_group_colors(self) -> dict:
        """Return ``{group_id: (r, g, b)}`` from the dialog."""
        return {
            i + 1: row["color_btn"]._color
            for i, row in enumerate(self._group_row_data)
        }


class HistogramWidget(QWidget):
    """Reusable 1D histogram widget with colormap-synced colored bars.

    This widget wraps a Matplotlib figure that renders a 1D histogram
    with bars colored according to a given colormap and contrast limits.
    It is designed to be embedded in any tab that needs to display a
    histogram of scalar data (e.g. lifetime, concentration, FRET efficiency).

    When *range_slider_enabled* is ``True`` the widget also displays a
    range slider together with min / max line-edits that allow the user
    to clip the displayed / stored data.  The ``rangeChanged`` signal is
    emitted whenever the effective range changes (min, max as floats).

    Parameters
    ----------
    xlabel : str, optional
        Label for the x-axis, by default ``"Value"``.
    ylabel : str, optional
        Label for the y-axis, by default ``"Pixel count"``.
    bins : int, optional
        Number of histogram bins, by default 300.
    default_colormap_name : str, optional
        Name of the Matplotlib colormap to use as fallback when no explicit
        colormap colors are provided, by default ``"plasma"``.
    canvas_height : int, optional
        Fixed pixel height of the canvas, by default 150.
    range_slider_enabled : bool, optional
        If ``True``, show a range slider with min / max edits above the
        histogram plot, by default ``False``.
    range_label_prefix : str, optional
        Prefix for the range label, e.g. ``"Lifetime range (ns)"``.
        Only used when *range_slider_enabled* is ``True``.
    range_factor : int, optional
        Multiplicative factor to convert float range values to integer
        slider positions, by default 1000.
    parent : QWidget, optional
        Parent widget.
    """

    # Emitted as (min_float, max_float) whenever the range changes.
    rangeChanged = Signal(float, float)
    # Emitted whenever the underlying data or display settings change.
    dataChanged = Signal()

    def __init__(
        self,
        xlabel: str = "Value",
        ylabel: str = "Pixel count",
        bins: int = 150,
        default_colormap_name: str = "plasma",
        canvas_height: int = 150,
        range_slider_enabled: bool = False,
        range_label_prefix: str = "Range",
        range_factor: int = 1000,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.bins = bins
        self.default_colormap_name = default_colormap_name

        # Histogram state
        self.counts = None
        self.bin_edges = None
        self.bin_centers = None

        # Multi-layer state
        self._datasets = {}  # {label: valid_1d_array}
        self._counts_per_dataset = {}  # {label: counts on common bins}
        self._previous_dataset_count = (
            0  # Track transitions for auto-enabling SD
        )

        # Colormap state (set externally)
        self.colormap_colors = None  # Nx4 array of RGBA colors
        self.contrast_limits = None  # [vmin, vmax]

        # Raw pooled data (for central tendency computation)
        self._raw_valid_data = None

        # Display settings
        self._display_mode = (
            "Merged"  # "Merged", "Individual layers", "Grouped"
        )
        self._show_sd = False
        self._group_assignments = {}  # {label: group_int}
        self._group_names = {}  # {group_id: str}
        self._central_tendency = "None"
        self._show_legend = True
        self._layer_colors = {}  # {label: (r,g,b)}
        self._group_colors = {}  # {group_id: (r,g,b)}
        self._white_background = False
        self._smooth_curves = True

        # Range slider state
        self._range_slider_enabled = range_slider_enabled
        self._range_label_prefix = range_label_prefix
        self.range_factor = range_factor
        self._slider_being_dragged = False

        # Build UI
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Optional range slider section
        if self._range_slider_enabled:
            self.range_label = QLabel(
                f"{self._range_label_prefix}: 0.0 - 100.0"
            )
            layout.addWidget(self.range_label)

            edit_layout = QHBoxLayout()
            self.range_min_edit = QLineEdit("0.0")
            self.range_max_edit = QLineEdit("100.0")
            self.range_min_edit.setValidator(QDoubleValidator())
            self.range_max_edit.setValidator(QDoubleValidator())
            edit_layout.addWidget(QLabel("Min:"))
            edit_layout.addWidget(self.range_min_edit)
            edit_layout.addWidget(QLabel("Max:"))
            edit_layout.addWidget(self.range_max_edit)
            layout.addLayout(edit_layout)

            self.range_slider = QRangeSlider(Qt.Orientation.Horizontal)
            self.range_slider.setRange(0, 100)
            self.range_slider.setValue((0, 100))
            self.range_slider.setBarMovesAllHandles(False)

            self.range_slider.valueChanged.connect(self._on_range_label_update)
            self.range_slider.sliderPressed.connect(self._on_slider_pressed)
            self.range_slider.sliderReleased.connect(self._on_slider_released)
            layout.addWidget(self.range_slider)

            self.range_min_edit.editingFinished.connect(
                self._on_range_min_edit
            )
            self.range_max_edit.editingFinished.connect(
                self._on_range_max_edit
            )

        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(
            figsize=(8, 4), constrained_layout=True
        )
        self._style_axes()

        canvas = FigureCanvas(self.fig)
        canvas.setFixedHeight(canvas_height)
        canvas.setSizePolicy(
            canvas.sizePolicy().Expanding, canvas.sizePolicy().Fixed
        )
        layout.addWidget(canvas)

        # Settings and export controls in one row
        controls_layout = QHBoxLayout()

        self._settings_button = QPushButton("Histogram Settings…")
        self._settings_button.setMaximumWidth(150)
        self._settings_button.clicked.connect(self._open_settings_dialog)
        controls_layout.addWidget(self._settings_button)

        controls_layout.addStretch()

        self.export_csv_button = QPushButton("Export Table CSV")
        self.export_csv_button.setMinimumWidth(140)
        self.export_csv_button.clicked.connect(self._export_table_csv)
        controls_layout.addWidget(self.export_csv_button)

        self.save_png_button = QPushButton("Save Histogram as PNG")
        self.save_png_button.setMinimumWidth(180)
        self.save_png_button.clicked.connect(self._save_histogram_png)
        controls_layout.addWidget(self.save_png_button)

        layout.addLayout(controls_layout)

        # Start hidden
        self.hide()

    def set_range(
        self, min_val: float, max_val: float, *, slider_max: float = None
    ) -> None:
        """Programmatically set the range slider position.

        Parameters
        ----------
        min_val : float
            Minimum value.
        max_val : float
            Maximum value.
        slider_max : float, optional
            If given, also update the slider's maximum to
            ``int(slider_max * range_factor)``.
        """
        if not self._range_slider_enabled:
            return
        if slider_max is not None:
            self.range_slider.setRange(0, int(slider_max * self.range_factor))
        min_s = int(min_val * self.range_factor)
        max_s = int(max_val * self.range_factor)
        self.range_slider.setValue((min_s, max_s))
        self.range_min_edit.setText(f"{min_val:.2f}")
        self.range_max_edit.setText(f"{max_val:.2f}")
        self.range_label.setText(
            f"{self._range_label_prefix}: {min_val:.2f} - {max_val:.2f}"
        )

    def get_range(self) -> tuple:
        """Return ``(min_float, max_float)`` from the slider."""
        if not self._range_slider_enabled:
            return (0.0, 0.0)
        lo, hi = self.range_slider.value()
        return lo / self.range_factor, hi / self.range_factor

    def _on_range_label_update(self, value):
        """Update label + edits while dragging (no heavy work)."""
        lo, hi = value
        lo_f = lo / self.range_factor
        hi_f = hi / self.range_factor
        self.range_label.setText(
            f"{self._range_label_prefix}: {lo_f:.2f} - {hi_f:.2f}"
        )
        self.range_min_edit.setText(f"{lo_f:.2f}")
        self.range_max_edit.setText(f"{hi_f:.2f}")

    def _on_slider_pressed(self):
        self._slider_being_dragged = True

    def _on_slider_released(self):
        self._slider_being_dragged = False
        lo, hi = self.range_slider.value()
        self.rangeChanged.emit(lo / self.range_factor, hi / self.range_factor)

    def _on_range_min_edit(self):
        try:
            lo = float(self.range_min_edit.text())
            hi = float(self.range_max_edit.text())
        except ValueError:
            return
        if lo >= hi:
            hi = lo + 0.01
        lo_s = int(lo * self.range_factor)
        hi_s = int(hi * self.range_factor)
        self.range_slider.setValue((lo_s, hi_s))
        self.rangeChanged.emit(lo, hi)

    def _on_range_max_edit(self):
        try:
            lo = float(self.range_min_edit.text())
            hi = float(self.range_max_edit.text())
        except ValueError:
            return
        if hi <= lo:
            lo = hi - 0.01 if hi > 0.01 else 0.0
        lo_s = int(lo * self.range_factor)
        hi_s = int(hi * self.range_factor)
        self.range_slider.setValue((lo_s, hi_s))
        self.rangeChanged.emit(lo, hi)

    def _export_table_csv(self):
        """Export statistics table to CSV - delegates to dock widget."""
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, '_export_table_csv_impl'):
                parent._export_table_csv_impl()
                return
            parent = parent.parent()

    def _save_histogram_png(self):
        """Save the histogram as a high-DPI PNG image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Histogram as PNG",
            "",
            "PNG Files (*.png)",
        )

        if not file_path:
            return

        if not file_path.endswith('.png'):
            file_path += '.png'

        self._style_axes(export_mode=True)
        self.fig.canvas.draw_idle()

        use_transparent = not self._white_background
        self.fig.savefig(
            file_path,
            dpi=300,
            bbox_inches='tight',
            transparent=use_transparent,
            facecolor='white' if self._white_background else 'none',
        )

        self._style_axes(export_mode=False)
        self.fig.canvas.draw_idle()

    def _open_settings_dialog(self):
        """Open the histogram settings dialog."""
        layer_labels = list(self._datasets.keys()) if self._datasets else None
        dlg = HistogramSettingsDialog(
            display_mode=self._display_mode,
            show_sd=self._show_sd,
            central_tendency=self._central_tendency,
            show_legend=self._show_legend,
            layer_labels=layer_labels,
            group_assignments=self._group_assignments,
            layer_colors=self._layer_colors,
            group_colors=self._group_colors,
            group_names=self._group_names,
            parent=self,
        )
        dlg.white_bg_checkbox.setChecked(self._white_background)
        dlg.smooth_checkbox.setChecked(self._smooth_curves)

        if dlg.exec_() == QDialog.Accepted:
            self._display_mode = dlg.mode_combo.currentText()
            self._show_sd = dlg.sd_checkbox.isChecked()
            self._central_tendency = dlg.central_tendency_combo.currentText()
            self._show_legend = dlg.legend_checkbox.isChecked()
            self._white_background = dlg.white_bg_checkbox.isChecked()
            self._smooth_curves = dlg.smooth_checkbox.isChecked()
            if dlg._group_row_data:
                self._group_assignments = dlg.get_group_assignments()
                self._group_colors = dlg.get_group_colors()
                self._group_names = dlg.get_group_names()
            if dlg._layer_color_buttons:
                self._layer_colors = dlg.get_layer_colors()
            if self.counts is not None:
                self._render()
            self.dataChanged.emit()

    def update_data(self, data: np.ndarray) -> None:
        """Compute histogram from *data* and render.

        Values that are NaN, non-positive, or non-finite are excluded
        before computing the histogram.  This is the single-dataset
        entry point; multi-layer features are disabled.

        Parameters
        ----------
        data : np.ndarray
            Scalar data array (any shape – will be flattened internally).
        """
        flat = np.asarray(data).ravel()
        valid = flat[~np.isnan(flat) & (flat > 0) & np.isfinite(flat)]

        if len(valid) == 0:
            self.ax.clear()
            self.ax.text(
                0.5,
                0.5,
                "No valid data",
                transform=self.ax.transAxes,
                ha="center",
            )
            self.fig.canvas.draw_idle()
            self.show()
            return

        self._datasets = {"Layer": valid}
        self._raw_valid_data = valid
        self._previous_dataset_count = 0

        self.counts, self.bin_edges = np.histogram(valid, bins=self.bins)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        self._counts_per_dataset = {"Layer": self.counts}

        self._render()
        self.show()
        self.dataChanged.emit()

    def update_multi_data(self, datasets: dict) -> None:
        """Compute histograms from multiple datasets and render.

        Each dataset (one per layer) is stored individually so that
        *Individual layers*, *Grouped*, and *Merged + SD* display modes
        can operate on per-layer counts.

        Parameters
        ----------
        datasets : dict
            ``{label: np.ndarray}`` mapping layer names to their scalar
            data arrays.  Arrays will be flattened and filtered.
        """
        self._datasets = {}
        for label, data in datasets.items():
            flat = np.asarray(data).ravel()
            valid = flat[~np.isnan(flat) & (flat > 0) & np.isfinite(flat)]
            if len(valid) > 0:
                self._datasets[label] = valid

        if not self._datasets:
            self.ax.clear()
            self.ax.text(
                0.5,
                0.5,
                "No valid data",
                transform=self.ax.transAxes,
                ha="center",
            )
            self.fig.canvas.draw_idle()
            self.show()
            return

        all_valid = np.concatenate(list(self._datasets.values()))
        self._raw_valid_data = all_valid
        self.counts, self.bin_edges = np.histogram(all_valid, bins=self.bins)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        self._counts_per_dataset = {}
        for label, valid in self._datasets.items():
            counts, _ = np.histogram(valid, bins=self.bin_edges)
            self._counts_per_dataset[label] = counts

        current_count = len(self._datasets)
        if current_count > 1 and self._previous_dataset_count <= 1:
            self._show_sd = True
        self._previous_dataset_count = current_count

        self._render()
        self.show()
        self.dataChanged.emit()

    def update_colormap(
        self,
        colormap_colors: np.ndarray = None,
        contrast_limits: list = None,
    ) -> None:
        """Update the colormap / contrast limits and re-render.

        Parameters
        ----------
        colormap_colors : np.ndarray, optional
            Nx4 RGBA array that defines the colormap.
        contrast_limits : list, optional
            ``[vmin, vmax]`` for the normalisation.
        """
        self.colormap_colors = colormap_colors
        self.contrast_limits = contrast_limits
        if self.counts is not None:
            self._render()

    def clear(self) -> None:
        """Clear the histogram and hide the widget."""
        self.counts = None
        self.bin_edges = None
        self.bin_centers = None
        self._datasets = {}
        self._counts_per_dataset = {}
        self._raw_valid_data = None
        self._previous_dataset_count = 0
        self.ax.clear()
        self.fig.canvas.draw_idle()
        self.hide()
        self.dataChanged.emit()

    @property
    def display_mode(self) -> str:
        """Current display mode."""
        return self._display_mode

    @display_mode.setter
    def display_mode(self, value: str):
        self._display_mode = value
        if self.counts is not None:
            self._render()

    @property
    def show_sd(self) -> bool:
        """Whether SD shading is enabled."""
        return self._show_sd

    @show_sd.setter
    def show_sd(self, value: bool):
        self._show_sd = value
        if self.counts is not None:
            self._render()

    @property
    def white_background(self) -> bool:
        """Whether white background is enabled."""
        return self._white_background

    @white_background.setter
    def white_background(self, value: bool):
        self._white_background = value
        self._style_axes()
        if self.counts is not None:
            self._render()
        self.dataChanged.emit()

    def _style_axes(self, export_mode: bool = False) -> None:
        """Apply consistent styling to the axes and figure.

        Parameters
        ----------
        export_mode : bool, optional
            If True, use black colors suitable for export, by default False.
        """
        if export_mode:
            if self._white_background:
                self.ax.patch.set_facecolor('white')
                self.ax.patch.set_alpha(1)
                self.fig.patch.set_facecolor('white')
                self.fig.patch.set_alpha(1)
            else:
                self.ax.patch.set_alpha(0)
                self.fig.patch.set_alpha(0)
            color = 'black'
        else:
            if self._white_background:
                self.ax.patch.set_facecolor('white')
                self.ax.patch.set_alpha(1)
                self.fig.patch.set_facecolor('white')
                self.fig.patch.set_alpha(1)
            else:
                # self.ax.patch.set_facecolor("#828A99")
                self.ax.patch.set_alpha(0)
                self.fig.patch.set_alpha(0)
            color = 'grey'

        for spine in self.ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1)
        self.ax.set_ylabel(self.ylabel, fontsize=6, color=color)
        self.ax.set_xlabel(self.xlabel, fontsize=6, color=color)
        for which in ("major", "minor"):
            self.ax.tick_params(
                axis="x", which=which, labelsize=7, colors=color
            )
            self.ax.tick_params(
                axis="y", which=which, labelsize=7, colors=color
            )

    def _get_cmap_and_norm(self):
        """Return (cmap, norm) from current colormap state."""
        if self.colormap_colors is None or self.contrast_limits is None:
            cmap = plt.cm.get_cmap(self.default_colormap_name)
            norm = plt.Normalize(
                vmin=(
                    np.min(self.bin_centers)
                    if len(self.bin_centers) > 0
                    else 0
                ),
                vmax=(
                    np.max(self.bin_centers)
                    if len(self.bin_centers) > 0
                    else 1
                ),
            )
        else:
            cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", self.colormap_colors
            )
            norm = plt.Normalize(
                vmin=self.contrast_limits[0],
                vmax=self.contrast_limits[1],
            )
        return cmap, norm

    def _smooth_curve(self, y, sigma=2, upsample=5):
        """Return (x_fine, y_fine) with Gaussian-smoothed, upsampled data.

        Parameters
        ----------
        y : np.ndarray
            Y-values corresponding to ``self.bin_centers``.
        sigma : float
            Gaussian smoothing sigma (in bins).
        upsample : int
            Upsampling factor for the output grid.

        Returns
        -------
        x_fine : np.ndarray
        y_fine : np.ndarray
        """
        if self._smooth_curves:
            y_smooth = gaussian_filter1d(y.astype(float), sigma=sigma)
            x_fine = np.linspace(
                self.bin_centers[0],
                self.bin_centers[-1],
                len(self.bin_centers) * upsample,
            )
            y_fine = np.interp(x_fine, self.bin_centers, y_smooth)
        else:
            x_fine = self.bin_centers
            y_fine = y.astype(float)
        return x_fine, y_fine

    @staticmethod
    def _compute_central_tendency(
        data: np.ndarray,
        method: str,
        bin_centers: np.ndarray = None,
        bin_edges: np.ndarray = None,
    ):
        """Return a scalar central-tendency value.

        Parameters
        ----------
        data : np.ndarray
            1-D array of valid values.
        method : str
            ``"Mean"``, ``"Median"``, or ``"Center of mass"``.
        bin_centers, bin_edges : np.ndarray, optional
            Needed only for ``"Center of mass"``.
        """
        if data is None or len(data) == 0:
            return None
        if method == "Mean":
            return float(np.mean(data))
        if method == "Median":
            return float(np.median(data))
        if method == "Center of mass":
            if bin_centers is None or bin_edges is None:
                return float(np.mean(data))
            counts, _ = np.histogram(data, bins=bin_edges)
            total = counts.sum()
            if total == 0:
                return None
            return float(np.sum(bin_centers * counts) / total)
        return None

    def _draw_central_tendency_lines(self) -> None:
        """Draw vertical lines at the selected central-tendency statistic."""
        choice = self._central_tendency
        if choice == "None":
            return

        default_colors = plt.cm.tab10.colors
        n_datasets = len(self._counts_per_dataset)

        if n_datasets > 1 and self._display_mode == "Individual layers":
            for idx, (label, valid) in enumerate(self._datasets.items()):
                default_c = default_colors[idx % len(default_colors)][:3]
                color = self._layer_colors.get(label, default_c)
                val = self._compute_central_tendency(
                    valid, choice, self.bin_centers, self.bin_edges
                )
                if val is not None:
                    self.ax.axvline(
                        val, color=color, ls="--", lw=2, alpha=0.85
                    )
        elif n_datasets > 1 and self._display_mode == "Grouped":
            groups: dict[int, list] = {}
            for label, valid in self._datasets.items():
                g = self._group_assignments.get(label, 1)
                groups.setdefault(g, []).append(valid)
            for _gidx, (gid, data_list) in enumerate(sorted(groups.items())):
                default_c = default_colors[(gid - 1) % len(default_colors)][:3]
                color = self._group_colors.get(gid, default_c)
                pooled = np.concatenate(data_list)
                val = self._compute_central_tendency(
                    pooled, choice, self.bin_centers, self.bin_edges
                )
                if val is not None:
                    self.ax.axvline(
                        val, color=color, ls="--", lw=2, alpha=0.85
                    )
        else:
            if self._raw_valid_data is not None:
                val = self._compute_central_tendency(
                    self._raw_valid_data,
                    choice,
                    self.bin_centers,
                    self.bin_edges,
                )
                if val is not None:
                    self.ax.axvline(
                        val, color="white", ls="--", lw=2, alpha=0.85
                    )

    def _render(self) -> None:
        """Re-draw the histogram using the active display mode."""
        self.ax.clear()

        n_datasets = len(self._counts_per_dataset)

        if n_datasets >= 1:
            if self._display_mode == "Individual layers":
                self._render_individual()
            elif self._display_mode == "Grouped":
                self._render_grouped()
            else:
                self._render_merged()
        else:
            self._render_bars()

        self._draw_central_tendency_lines()
        self._style_axes()
        self.fig.canvas.draw_idle()

    def _render_bars(self) -> None:
        """Render the standard colormap-colored bar histogram."""
        cmap, norm = self._get_cmap_and_norm()
        x = self.bin_centers
        y = self.counts.astype(float)
        self._fill_gradient(x, y, np.zeros_like(y), cmap, norm, alpha=0.7)

    def _fill_gradient(
        self,
        x: np.ndarray,
        y_upper: np.ndarray,
        y_lower: np.ndarray,
        cmap,
        norm,
        *,
        alpha: float = 0.8,
    ) -> None:
        """Fill between *y_lower* and *y_upper* with a smooth colormap gradient.

        Uses a single ``imshow`` call clipped to a polygon, which is
        dramatically faster than hundreds of individual ``fill_between``
        calls and produces a perfectly seamless gradient.
        """
        if len(x) < 2:
            return

        y_max = float(np.max(y_upper))
        y_min = float(np.min(y_lower))
        if y_max <= y_min:
            y_max = y_min + 1

        n_pixels = 256
        gradient_values = np.linspace(
            float(x[0]), float(x[-1]), n_pixels
        ).reshape(1, -1)
        extent = [float(x[0]), float(x[-1]), y_min, y_max * 1.02]

        im = self.ax.imshow(
            gradient_values,
            aspect="auto",
            extent=extent,
            origin="lower",
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            interpolation="bilinear",
        )

        verts_x = np.concatenate([x, x[::-1]])
        verts_y = np.concatenate([y_upper, y_lower[::-1]])
        verts = np.column_stack([verts_x, verts_y])
        clip_poly = MplPolygon(verts, closed=True, transform=self.ax.transData)
        im.set_clip_path(clip_poly)

        self.ax.set_xlim(float(x[0]), float(x[-1]))
        self.ax.set_ylim(0, y_max * 1.05)

    def _draw_gradient_line(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cmap,
        norm,
        *,
        linewidth: float = 2,
    ) -> None:
        """Draw a line colored by a smooth colormap gradient.

        Uses a ``LineCollection`` for efficient per-segment coloring.
        """
        from matplotlib.collections import LineCollection

        if len(x) < 2:
            return
        points = np.column_stack([x, y]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = cmap(norm(x[:-1]))
        lc = LineCollection(segments, colors=colors, linewidths=linewidth)
        self.ax.add_collection(lc)

    def _render_merged(self) -> None:
        """Render merged histogram with optional SD shading.

        Uses ``imshow`` with polygon clipping for seamless color-mapped
        gradient fills, and ``LineCollection`` for efficient line rendering.
        """
        cmap, norm = self._get_cmap_and_norm()
        n = len(self._counts_per_dataset)

        if self._show_sd and n > 1:
            all_counts = np.array(
                list(self._counts_per_dataset.values()), dtype=float
            )
            mean_counts = np.mean(all_counts, axis=0)
            std_counts = np.std(all_counts, axis=0, ddof=1)
            lower = np.maximum(mean_counts - std_counts, 0)
            upper = mean_counts + std_counts

            x_fine, mean_fine = self._smooth_curve(mean_counts)
            _, lower_fine = self._smooth_curve(lower)
            _, upper_fine = self._smooth_curve(upper)

            self._fill_gradient(
                x_fine, upper_fine, lower_fine, cmap, norm, alpha=0.35
            )
            self._draw_gradient_line(
                x_fine, mean_fine, cmap, norm, linewidth=2
            )
        elif self._show_sd and n == 1:
            counts = list(self._counts_per_dataset.values())[0]
            x_fine, y_fine = self._smooth_curve(counts)
            self._draw_gradient_line(x_fine, y_fine, cmap, norm, linewidth=2)
            self.ax.set_xlim(float(x_fine[0]), float(x_fine[-1]))
            self.ax.set_ylim(0, float(np.max(y_fine)) * 1.05)
        else:
            if n > 1:
                all_counts = np.array(
                    list(self._counts_per_dataset.values()), dtype=float
                )
                mean_counts = np.mean(all_counts, axis=0)
            else:
                mean_counts = list(self._counts_per_dataset.values())[0]

            x_fine, mean_fine = self._smooth_curve(mean_counts)

            self._fill_gradient(
                x_fine,
                mean_fine,
                np.zeros_like(mean_fine),
                cmap,
                norm,
                alpha=0.8,
            )
            self._draw_gradient_line(
                x_fine, mean_fine, cmap, norm, linewidth=2
            )

    def _render_individual(self) -> None:
        """Render each dataset as a smooth outline."""
        default_colors = plt.cm.tab10.colors
        for idx, (label, counts) in enumerate(
            self._counts_per_dataset.items()
        ):
            default_c = default_colors[idx % len(default_colors)][:3]
            color = self._layer_colors.get(label, default_c)
            x_fine, y_fine = self._smooth_curve(counts)
            self.ax.plot(
                x_fine,
                y_fine,
                color=color,
                linewidth=2,
                label=label,
            )
        if self._show_legend and self._counts_per_dataset:
            self.ax.legend(fontsize=5, loc="upper right")

    def _render_grouped(self) -> None:
        """Render grouped histograms with smooth curves and optional SD."""
        default_colors = plt.cm.tab10.colors

        groups: dict[int, list[tuple[str, np.ndarray]]] = {}
        for label, counts in self._counts_per_dataset.items():
            g = self._group_assignments.get(label, 1)
            groups.setdefault(g, []).append((label, counts))

        for _gidx, (group_id, members) in enumerate(sorted(groups.items())):
            default_c = default_colors[(group_id - 1) % len(default_colors)][
                :3
            ]
            color = self._group_colors.get(group_id, default_c)
            all_counts = np.array([c for _, c in members], dtype=float)
            mean_counts = np.mean(all_counts, axis=0)

            x_fine, mean_fine = self._smooth_curve(mean_counts)
            group_label = self._group_names.get(group_id, f"Group {group_id}")
            self.ax.plot(
                x_fine,
                mean_fine,
                color=color,
                linewidth=2,
                label=group_label,
            )

            if self._show_sd and len(members) > 1:
                std_counts = np.std(all_counts, axis=0, ddof=1)
                lower = np.maximum(mean_counts - std_counts, 0)
                upper = mean_counts + std_counts
                _, lower_fine = self._smooth_curve(lower)
                _, upper_fine = self._smooth_curve(upper)
                self.ax.fill_between(
                    x_fine,
                    lower_fine,
                    upper_fine,
                    color=color,
                    alpha=0.25,
                    linewidth=0,
                )

        if self._show_legend and groups:
            self.ax.legend(fontsize=5, loc="upper right")


class CollapsibleSection(QWidget):
    """A collapsible section with a clickable header and hideable content.

    Parameters
    ----------
    title : str
        Header text displayed on the toggle button.
    initially_collapsed : bool, optional
        Whether the content starts hidden, by default ``True``.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, title="Section", initially_collapsed=True, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Clickable header with disclosure triangle
        self._toggle_button = QPushButton()
        self._toggle_button.setStyleSheet(
            "QPushButton { text-align: left; border: none; padding: 4px; "
            "font-weight: bold; color: grey; }"
        )
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(not initially_collapsed)
        self._toggle_button.clicked.connect(self._on_toggle)
        layout.addWidget(self._toggle_button)

        # Content area
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._content)

        self._title = title
        self._update_button_text()
        self._content.setVisible(not initially_collapsed)

    def _update_button_text(self):
        """Update the button text with a disclosure triangle."""
        arrow = "\u25BC" if self._toggle_button.isChecked() else "\u25B6"
        self._toggle_button.setText(f"{arrow} {self._title}")

    def _on_toggle(self):
        """Toggle content visibility when the header is clicked."""
        self._content.setVisible(self._toggle_button.isChecked())
        self._update_button_text()

    def add_widget(self, widget):
        """Add a widget to the collapsible content area."""
        self._content_layout.addWidget(widget)

    def set_content_visible(self, visible):
        """Programmatically expand or collapse the section."""
        self._toggle_button.setChecked(visible)
        self._content.setVisible(visible)
        self._update_button_text()


class StatisticsTableWidget(QTableWidget):
    """Table showing statistics (Mean, Median, Center of Mass, Std Dev).

    Each row corresponds to a named dataset (layer or group).

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    """

    COLUMNS = ["Name", "Center of Mass", "Mean", "Median", "Std Dev"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionMode(QTableWidget.ExtendedSelection)
        self.setSelectionBehavior(QTableWidget.SelectItems)
        self.verticalHeader().setVisible(False)
        self.setMaximumHeight(200)
        self.setStyleSheet(
            "QTableWidget { background: transparent; color: grey; "
            "gridline-color: #555; }"
            "QHeaderView::section { background: transparent; color: grey; "
            "border: 1px solid #555; font-size: 10px; }"
            "QTableWidget::item:selected { background: #4a6fa5; color: white; }"
        )
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def keyPressEvent(self, event):
        """Handle Ctrl+C / Ctrl+A keyboard shortcuts."""
        if event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_C:
                self._copy_selection()
                return
            if event.key() == Qt.Key_A:
                self.selectAll()
                return
        super().keyPressEvent(event)

    def _copy_selection(self, include_headers=False):
        """Copy selected cells (or entire table if nothing selected) to clipboard."""
        selected = self.selectedItems()
        if not selected:
            rows = range(self.rowCount())
            cols = range(self.columnCount())
            sel_indices = {(r, c) for r in rows for c in cols}
        else:
            sel_indices = {(item.row(), item.column()) for item in selected}

        if not sel_indices:
            return

        min_row = min(r for r, c in sel_indices)
        max_row = max(r for r, c in sel_indices)
        min_col = min(c for r, c in sel_indices)
        max_col = max(c for r, c in sel_indices)

        lines = []
        if include_headers:
            headers = [
                self.horizontalHeaderItem(c).text()
                for c in range(min_col, max_col + 1)
            ]
            lines.append("\t".join(headers))

        for row in range(min_row, max_row + 1):
            row_data = []
            for col in range(min_col, max_col + 1):
                if (row, col) in sel_indices:
                    item = self.item(row, col)
                    row_data.append(item.text() if item else "")
                else:
                    row_data.append("")
            lines.append("\t".join(row_data))

        QApplication.clipboard().setText("\n".join(lines))

    def _show_context_menu(self, pos):
        """Show right-click context menu."""
        menu = QMenu(self)
        copy_action = menu.addAction("Copy")
        copy_with_headers_action = menu.addAction("Copy with Headers")
        menu.addSeparator()
        select_all_action = menu.addAction("Select All")

        action = menu.exec_(self.viewport().mapToGlobal(pos))
        if action == copy_action:
            self._copy_selection(include_headers=False)
        elif action == copy_with_headers_action:
            self._copy_selection(include_headers=True)
        elif action == select_all_action:
            self.selectAll()

    def update_statistics(self, datasets, bin_centers=None, bin_edges=None):
        """Update table rows from a ``{name: 1-D array}`` mapping.

        Parameters
        ----------
        datasets : dict
            ``{label: np.ndarray}`` mapping names to data arrays.
        bin_centers : np.ndarray, optional
            Bin centres for center-of-mass computation.
        bin_edges : np.ndarray, optional
            Bin edges for center-of-mass computation.
        """
        self.setRowCount(len(datasets))
        for row, (name, data) in enumerate(datasets.items()):
            flat = np.asarray(data).ravel()
            valid = flat[~np.isnan(flat) & np.isfinite(flat)]

            if len(valid) > 0:
                mean_val = float(np.mean(valid))
                median_val = float(np.median(valid))
                std_val = float(np.std(valid))

                if bin_centers is not None and bin_edges is not None:
                    counts, _ = np.histogram(valid, bins=bin_edges)
                    total = counts.sum()
                    com_val = (
                        float(np.sum(bin_centers * counts) / total)
                        if total > 0
                        else float("nan")
                    )
                else:
                    com_val = mean_val
            else:
                mean_val = median_val = com_val = std_val = float("nan")

            self.setItem(row, 0, QTableWidgetItem(str(name)))
            self.setItem(row, 1, QTableWidgetItem(f"{mean_val:.4f}"))
            self.setItem(row, 2, QTableWidgetItem(f"{median_val:.4f}"))
            self.setItem(row, 3, QTableWidgetItem(f"{com_val:.4f}"))
            self.setItem(row, 4, QTableWidgetItem(f"{std_val:.4f}"))

    def update_group_statistics(
        self,
        datasets,
        group_assignments,
        group_names=None,
        bin_centers=None,
        bin_edges=None,
    ):
        """Update table rows with per-group pooled statistics.

        Parameters
        ----------
        datasets : dict
            ``{label: np.ndarray}`` mapping layer names to data arrays.
        group_assignments : dict
            ``{label: group_int}`` mapping layer names to group IDs.
        group_names : dict, optional
            ``{group_int: str}`` mapping group IDs to display names.
        bin_centers : np.ndarray, optional
            Bin centres for center-of-mass computation.
        bin_edges : np.ndarray, optional
            Bin edges for center-of-mass computation.
        """
        if group_names is None:
            group_names = {}
        groups = {}
        for label, data in datasets.items():
            gid = group_assignments.get(label, 1)
            groups.setdefault(gid, []).append(data)

        pooled_datasets = {}
        for gid in sorted(groups):
            pooled = np.concatenate(groups[gid])
            name = group_names.get(gid, f"Group {gid}")
            pooled_datasets[name] = pooled

        self.update_statistics(pooled_datasets, bin_centers, bin_edges)


class ResponsiveFormContainer(QWidget):
    """Container that arranges form rows in 1 or 2 columns based on width.

    Each *row* is a pair of ``(label_widget, field_widget)`` or a single
    *full-span* widget.  When the container's width is below
    ``width_threshold`` the rows are stacked vertically (single column).
    When it is at or above the threshold the rows are arranged in a
    two-column grid so that two label–field pairs sit side by side.

    Usage::

        form = ResponsiveFormContainer(width_threshold=450)
        form.add_row(QLabel("Frequency:"), frequency_input)
        form.add_row(QLabel("Type:"), type_combobox)
        form.add_full_span_widget(calculate_button)

    Parameters
    ----------
    width_threshold : int, optional
        Width in pixels at which to switch from 1 to 2 columns.
        Default is 450.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, width_threshold: int = 450, parent: QWidget = None):
        super().__init__(parent)
        self._width_threshold = width_threshold
        self._rows = []  # list of (label_widget | None, field_widget)
        self._current_columns = 1
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(4)

    # -- public API ----------------------------------------------------------

    @property
    def width_threshold(self) -> int:
        return self._width_threshold

    @width_threshold.setter
    def width_threshold(self, value: int):
        self._width_threshold = value
        self._relayout()

    def add_row(self, label_widget: QWidget, field_widget: QWidget):
        """Add a label + field pair."""
        self._rows.append((label_widget, field_widget))
        self._relayout()

    def add_full_span_widget(self, widget: QWidget):
        """Add a widget that always takes the full width."""
        self._rows.append((None, widget))
        self._relayout()

    def add_layout_as_row(self, label_widget: QWidget, layout):
        """Add a label + layout pair (wraps the layout in a QWidget)."""
        wrapper = QWidget()
        wrapper.setLayout(layout)
        self.add_row(label_widget, wrapper)

    def add_full_span_layout(self, layout):
        """Add a layout that always takes the full width."""
        wrapper = QWidget()
        wrapper.setLayout(layout)
        self.add_full_span_widget(wrapper)

    # -- layout logic --------------------------------------------------------

    def resizeEvent(self, event):
        """Re-evaluate column count when width changes."""
        super().resizeEvent(event)
        new_cols = 2 if self.width() >= self._width_threshold else 1
        if new_cols != self._current_columns:
            self._current_columns = new_cols
            self._relayout()

    def _relayout(self):
        """Remove all items from the grid and re-add with the current column mode."""
        # Detach all widgets without deleting them
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        cols = self._current_columns
        # Each "logical column" is 2 grid columns: label col + field col
        # So 1 column mode uses grid cols 0-1, 2 column mode uses 0-1 and 2-3
        grid_row = 0
        col_pair = 0  # 0 or 1 (which logical column we're filling)

        for label_w, field_w in self._rows:
            if label_w is None:
                # Full-span widget: finish current row if partially filled
                if col_pair != 0:
                    grid_row += 1
                    col_pair = 0
                self._grid.addWidget(field_w, grid_row, 0, 1, cols * 2)
                field_w.setParent(self)
                field_w.show()
                grid_row += 1
            else:
                base_col = col_pair * 2
                label_w.setParent(self)
                field_w.setParent(self)
                self._grid.addWidget(label_w, grid_row, base_col)
                self._grid.addWidget(field_w, grid_row, base_col + 1)
                label_w.show()
                field_w.show()
                col_pair += 1
                if col_pair >= cols:
                    grid_row += 1
                    col_pair = 0

        # Make field columns stretch equally
        for c in range(cols):
            self._grid.setColumnStretch(c * 2, 0)  # label: no stretch
            self._grid.setColumnStretch(c * 2 + 1, 1)  # field: stretch


class HistogramDockWidget(QWidget):
    """Dockable container that wraps a :class:`HistogramWidget` with
    collapsible statistics tables underneath.

    This widget is intended to be added as a separate napari dock widget
    so that it can be detached into its own window.

    Parameters
    ----------
    histogram_widget : HistogramWidget
        The histogram widget to wrap.  Ownership is transferred to this
        container (Qt reparents the widget).
    title : str, optional
        Human-readable title, by default ``"Histogram & Statistics"``.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        histogram_widget: "HistogramWidget",
        title: str = "Histogram & Statistics",
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self._title = title
        self.histogram_widget = histogram_widget

        self.setMinimumHeight(300)
        self.setMinimumWidth(300)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(histogram_widget)

        self.layer_stats_section = CollapsibleSection(
            "Layer Statistics", initially_collapsed=True
        )
        self.layer_stats_table = StatisticsTableWidget()
        self.layer_stats_section.add_widget(self.layer_stats_table)
        layout.addWidget(self.layer_stats_section)

        self.group_stats_section = CollapsibleSection(
            "Group Statistics", initially_collapsed=True
        )
        self.group_stats_table = StatisticsTableWidget()
        self.group_stats_section.add_widget(self.group_stats_table)
        layout.addWidget(self.group_stats_section)
        self.group_stats_section.setVisible(False)

        layout.addStretch()

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        histogram_widget.dataChanged.connect(self._update_statistics)

    def _update_statistics(self):
        """Recompute the statistics tables from the histogram's data."""
        hw = self.histogram_widget

        has_multi = bool(hw._datasets)
        has_single = (
            hw._raw_valid_data is not None and len(hw._raw_valid_data) > 0
        )

        if has_multi:
            self.layer_stats_table.update_statistics(
                hw._datasets, hw.bin_centers, hw.bin_edges
            )
            self.layer_stats_section.setVisible(True)

            if hw._display_mode == "Grouped" and hw._group_assignments:
                self.group_stats_table.update_group_statistics(
                    hw._datasets,
                    hw._group_assignments,
                    group_names=hw._group_names,
                    bin_centers=hw.bin_centers,
                    bin_edges=hw.bin_edges,
                )
                self.group_stats_section.setVisible(True)
            else:
                self.group_stats_section.setVisible(False)
        elif has_single:
            self.layer_stats_table.update_statistics(
                {"Data": hw._raw_valid_data}, hw.bin_centers, hw.bin_edges
            )
            self.layer_stats_section.setVisible(True)
            self.group_stats_section.setVisible(False)
        else:
            self.layer_stats_section.setVisible(False)
            self.group_stats_section.setVisible(False)

    def _export_table_csv_impl(self):
        """Export the visible statistics table(s) to CSV file(s)."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Statistics as CSV",
            "",
            "CSV Files (*.csv)",
        )

        if not file_path:
            return

        if not file_path.endswith('.csv'):
            file_path += '.csv'

        if self.layer_stats_section.isVisible():
            self._write_table_to_csv(self.layer_stats_table, file_path)

        if self.group_stats_section.isVisible():
            group_file = file_path.replace('.csv', '_groups.csv')
            self._write_table_to_csv(self.group_stats_table, group_file)

    def _write_table_to_csv(self, table: QTableWidget, file_path: str):
        """Write a QTableWidget to a CSV file.

        Parameters
        ----------
        table : QTableWidget
            The table widget to export.
        file_path : str
            Path to the output CSV file.
        """
        import csv

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            headers = []
            for col in range(table.columnCount()):
                headers.append(table.horizontalHeaderItem(col).text())
            writer.writerow(headers)

            for row in range(table.rowCount()):
                row_data = []
                for col in range(table.columnCount()):
                    item = table.item(row, col)
                    row_data.append(item.text() if item else '')
                writer.writerow(row_data)
