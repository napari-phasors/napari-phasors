try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader
from ._sample_data import (
    convallaria_FLIM_sample_data,
    embryo_FLIM_sample_data,
    paramecium_HSI_sample_data,
)
from ._widget import PhasorTransform, WriterWidget
from ._writer import write_ome_tiff
from .plotter import PlotterWidget

__all__ = (
    "napari_get_reader",
    "write_ome_tiff",
    "convallaria_FLIM_sample_data",
    "embryo_FLIM_sample_data",
    "paramecium_HSI_sample_data",
    "PhasorTransform",
    "PlotterWidget",
    "WriterWidget",
)
