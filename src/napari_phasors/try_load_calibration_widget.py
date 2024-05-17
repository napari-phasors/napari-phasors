import napari
from napari_phasors._widget import CalibrationWidget
import numpy as np

viewer = napari.Viewer()

data = np.random.rand(100, 100)
labels = np.random.randint(0, 2, (100, 100))
viewer.add_image(data)
viewer.add_labels(labels)

calibration_widget = CalibrationWidget(viewer)
viewer.window.add_dock_widget(calibration_widget)
napari.run()