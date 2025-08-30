import numpy as np
from biaplotter.artists import Scatter

# Create a simple scatter artist
artist = Scatter

# Simulate some data
x_data = np.random.rand(100)
y_data = np.random.rand(100)
artist.set_data(x_data, y_data)

# Simulate the error condition: color_indices is None
artist.color_indices = None

# This reproduces the error from biaplotter/selectors.py line 594
# TypeError: 'NoneType' object does not support item assignment

selected_indices = np.array([0, 1, 2])
class_value = 1
artist.color_indices[selected_indices] = class_value