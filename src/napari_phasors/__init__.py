try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from qtpy import API_NAME

from ._reader import napari_get_reader
from ._sample_data import (
    convallaria_FLIM_sample_data,
    embryo_FLIM_sample_data,
    paramecium_HSI_sample_data,
)
from ._widget import PhasorTransform, WriterWidget
from ._writer import export_layer_as_csv, export_layer_as_image, write_ome_tiff
from .plotter import PlotterWidget

__all__ = (
    "napari_get_reader",
    "write_ome_tiff",
    "export_layer_as_csv",
    "export_layer_as_image",
    "convallaria_FLIM_sample_data",
    "embryo_FLIM_sample_data",
    "paramecium_HSI_sample_data",
    "PhasorTransform",
    "PlotterWidget",
    "WriterWidget",
)

if API_NAME.lower().startswith("pyside"):
    raise RuntimeError(
        f"napari-phasors is currently only compatible with PyQt5 or PyQt6 (detected {API_NAME}). "
        "Please install a PyQt backend (e.g., 'pip install PyQt6') and set your "
        "NAPARI_QT_API environment variable to 'pyqt6' or 'pyqt5'."
    )
