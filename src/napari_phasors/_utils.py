"""
This module contains utility functions used by other modules.

"""

import warnings
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from phasorpy.filter import (
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_threshold,
)
from qtpy.QtCore import QRect, QSize, Qt, Signal
from qtpy.QtGui import (
    QColor,
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
    QComboBox,
    QLineEdit,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QStyleOptionViewItem,
)

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
