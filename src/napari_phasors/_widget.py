"""
This module contains widgets to:

    - Transform FLIM and hyperspectral images into phasor space

"""

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QPushButton,
    QWidget,
    QTreeView,
    QDirModel,
    QComboBox,
    QLabel,
    QVBoxLayout,
    QLineEdit,
    QCompleter,
)
from qtpy.QtGui import QDoubleValidator
from ._reader import napari_get_reader, _get_filename_extension

if TYPE_CHECKING:
    import napari


class PhasorTransform(QWidget):
    """Widget to transform FLIM and hyperspectral images into phasor space."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Initialize the widget.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            Napari viewer instance.

        """
        super().__init__()
        self.viewer = viewer

        # Create main layout
        main_layout = QVBoxLayout(self)

        # Create search tree
        search_tree = QTreeView()
        model = QDirModel()
        search_tree.setModel(model)
        search_tree.setColumnHidden(1, True)
        search_tree.setColumnHidden(2, True)
        search_tree.setColumnHidden(3, True)
        main_layout.addWidget(search_tree)

        # Create layout for dynamic widgets
        self.dynamic_widget_layout = QVBoxLayout()
        main_layout.addLayout(self.dynamic_widget_layout)

        # Set up callbacks whenever the selection changes
        selection = search_tree.selectionModel()
        selection.currentChanged.connect(
            lambda current: self._on_change(current, model)
        )

        # Define reader options (example)
        self.reader_options = {
            ".fbd": FbdWidget,
            ".ptu": PtuWidget,
            ".lsm": LsmWidget,
            ".tif": LsmWidget,
        }

    def _on_change(self, current, model):
        """Callback whenever the selection changes."""
        path = model.filePath(current)
        _, extension = _get_filename_extension(path)
        if extension in self.reader_options:
            # Clear existing widgets
            for i in reversed(range(self.dynamic_widget_layout.count())):
                widget = self.dynamic_widget_layout.takeAt(i).widget()
                widget.deleteLater()

            # Create new widgets based on extension
            create_widget_class = self.reader_options[extension]
            new_widget = create_widget_class(self.viewer, path)
            self.dynamic_widget_layout.addWidget(new_widget)


class AdvancedOptionsWidget(QWidget):
    """Base class for advanced options widgets."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__()
        self.viewer = viewer
        self.path = path
        self.options = None
        self.initUI()

    def initUI(self):
        """Initialize the user interface."""
        # Initial layout
        self.mainLayout = QVBoxLayout()

        # Channel selection
        self.mainLayout.addWidget(QLabel("Channels: "))
        self.channels = QComboBox()
        self.channels.addItems(["All channels"])
        for channel in range(self.all_channels):
            self.channels.addItem(str(channel))
        self.channels.setCurrentIndex(0)
        self.channels.currentIndexChanged.connect(
            self._on_channels_combobox_changed
        )
        self.mainLayout.addWidget(self.channels)

        # Frames selection
        self.mainLayout.addWidget(QLabel("Frames: "))
        self.frames = QComboBox()
        self.frames.addItems(["Average all frames"])
        for frame in range(self.all_frames):
            self.frames.addItem(str(frame))
        self.frames.setCurrentIndex(0)
        self.frames.currentIndexChanged.connect(
            self._on_frames_combobox_changed
        )
        self.mainLayout.addWidget(self.frames)
        self.setLayout(self.mainLayout)

    def _on_frames_combobox_changed(self, index):
        """Callback whenever the frames combobox changes."""
        self.options["frame"] = index - 1

    def _on_channels_combobox_changed(self, index):
        """Callback whenever the channels combobox changes."""
        if index == 0:
            self.options["channel"] = None
        else:
            self.options["channel"] = index - 1

    def _on_click(self, path, options):
        """Callback whenever the calculate phasor button is clicked."""
        reader = napari_get_reader(path, options=options)
        for layer in reader(path):
            self.viewer.add_image(
                layer[0], name=layer[1]["name"], metadata=layer[1]["metadata"]
            )


class FbdWidget(AdvancedOptionsWidget):
    """Widget for FLIMbox FBD files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        import lfdfiles

        with lfdfiles.FlimboxFbd(path) as fbd:
            self.all_frames = len(fbd.frames(None)[1])
            self.all_channels = fbd.channels
        super().__init__(viewer, path)
        self.options = {"frame": -1, "channel": None}

    def initUI(self):
        super().initUI()

        # Laser factor
        self.mainLayout.addWidget(QLabel("Laser Factor: "))
        self.laser_factor = QLineEdit()
        self.laser_factor.setToolTip(
            "Most probable laser factors are: 0.00022, 2.50012, 2.50016"
        )
        self.laser_factor.setValidator(QDoubleValidator())
        laser_factor_completer = QCompleter(["0.00022", "2.50012", "2.50016"])
        self.laser_factor.setCompleter(laser_factor_completer)
        self.mainLayout.addWidget(self.laser_factor)

        # Calculate phasor button
        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(self.path, self.options)
        )
        self.mainLayout.addWidget(self.btn)

    def _on_click(self, path, options):
        """Callback whenever the calculate phasor button is clicked."""
        if self.laser_factor.text():
            options["laser_factor"] = float(self.laser_factor.text())
        super()._on_click(path, options)


class PtuWidget(AdvancedOptionsWidget):
    """Widget for PicoQuant PTU files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        import ptufile

        with ptufile.PtuFile(path) as ptu:
            self.all_frames = ptu.shape[0]
            self.all_channels = ptu.shape[-2]
        super().__init__(viewer, path)
        self.options = {"frame": -1, "channel": None}

    def initUI(self):
        """Initialize the user interface."""
        super().initUI()

        # dtime
        self.mainLayout.addWidget(QLabel("dtime: "))
        self.dtime = QLineEdit()
        self.dtime.setToolTip(
            "Specifies number of bins in image histogram."
            "If 0 (default), return number of bins in one period."
            "If < 0, integrate delay time axis."
            "If > 0, return up to specified bin."
        )
        self.mainLayout.addWidget(self.dtime)

        # Calculate phasor button
        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(self.path, self.options)
        )
        self.mainLayout.addWidget(self.btn)

    def _on_click(self, path, options):
        """Callback whenever the calculate phasor button is clicked."""
        if self.dtime.text():
            options["dtime"] = float(self.dtime.text())
        super()._on_click(path, options)


class LsmWidget(AdvancedOptionsWidget):
    """Widget for Zeiss LSM files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__(viewer, path)

    def initUI(self):
        """Initialize the user interface."""
        # LSM does not have channels and frames
        self.mainLayout = QVBoxLayout()

        # Calculate phasor button
        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(self.path, self.options)
        )
        self.mainLayout.addWidget(self.btn)
        self.setLayout(self.mainLayout)
