__version__ = "0.0.3"

from ._reader import napari_get_reader
from ._sample_data import (
    convallaria_FLIM_sample_data,
    embryo_FLIM_sample_data,
    paramecium_HSI_sample_data,
)
from ._widget import (
    CalibrationWidget,
    LifetimeWidget,
    PhasorTransform,
    WriterWidget,
)
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
    "CalibrationWidget",
    "WriterWidget",
    "LifetimeWidget",
)
