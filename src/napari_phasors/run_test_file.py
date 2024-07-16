import napari
from napari_phasors._widget import PhasorTransform

viewer = napari.Viewer()


# file_path = '/Users/bruno/Documents/UBA/phasorpy/test_data/FBDfiles-DIVER/BUENOS/convallaria_000$EI0S.fbd'
# calibration_path = '/Users/bruno/Documents/UBA/phasorpy/test_data/FBDfiles-DIVER/BUENOS/RH110CALIBRATION_000$EI0S.fbd'

# viewer.open(file_path, plugin='napari-phasors')
# viewer.open(calibration_path, plugin='napari-phasors')

transform_widget = PhasorTransform(viewer)
viewer.window.add_dock_widget(transform_widget)
napari.run()