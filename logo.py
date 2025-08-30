#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import FancyBboxPatch

shape_mask_path = '/Users/bruno/Downloads/logo-silhouette-192.png'

# Generate sample phasor data
np.random.seed(42)
n_points = 100000

real = np.concatenate([
    np.random.normal(0.03, 0.025, 30000),
    np.random.normal(0.15, 0.03, 60000),
])
imag = np.concatenate([
    np.random.normal(0.16, 0.025, 30000),
    np.random.normal(0.03, 0.03, 60000)
])

# Use fixed ranges for consistent aspect ratio
bins_x = np.linspace(-0.1, 0.3, 97)
bins_y = np.linspace(-0.1, 0.3, 97)
hist, xedges, yedges = np.histogram2d(real, imag, bins=[bins_x, bins_y], density=True)

# Load the image with alpha channel
mask_img = Image.open(shape_mask_path).convert('RGBA')
mask_array = np.array(mask_img)


# Use the alpha channel as the mask (where alpha > 0 means visible/solid pixels)
alpha_channel = mask_array[:,:,3]

# Flip the mask vertically and horizontally to match histogram orientation
alpha_channel = np.flipud(alpha_channel)
# alpha_channel = np.fliplr(alpha_channel)  # Add horizontal flip

mask_binary = alpha_channel > 0  # True where the silhouette is

# Resize if needed
if mask_binary.shape != hist.shape:
    mask_img_alpha = Image.fromarray(alpha_channel).resize((hist.shape[1], hist.shape[0]), Image.Resampling.LANCZOS)
    mask_binary = np.array(mask_img_alpha) > 0

hist = hist.T  # Transpose to match the orientation of the mask

# Apply mask with transparent background
masked_hist = np.where(mask_binary, hist, np.nan)  # Use NaN for background

# Adjust histogram position and size
# Shift to left-bottom and make smaller
shift_x = -0.02  # Move left
shift_y = -0.02  # Move down
scale_factor = 0.95  # Make smaller (80% of original size)

# Adjust the extent for the shifted and scaled histogram
hist_width = xedges[-1] - xedges[0]
hist_height = yedges[-1] - yedges[0]

new_width = hist_width * scale_factor
new_height = hist_height * scale_factor

# Calculate new extent with shift and scale
new_xmin = xedges[0] + shift_x
new_xmax = new_xmin + new_width
new_ymin = yedges[0] + shift_y
new_ymax = new_ymin + new_height

adjusted_extent = [new_xmin, new_xmax, new_ymin, new_ymax]

# Smooth the mask for smoother contours
from scipy.ndimage import gaussian_filter

# Display the result
plt.figure(figsize=(10, 10))

# Set the figure background color
plt.gcf().patch.set_facecolor('#564b69')
plt.gca().set_facecolor('#564b69')


# Calculate center of the histogram data
center_x = (xedges[0] + xedges[-1]) / 2
center_y = (yedges[0] + yedges[-1]) / 2

# Add unit circle centered at histogram center
theta = np.linspace(0, 2*np.pi, 1000)
radius = 0.2  # Change this value to control the circle size

# Get the colormap and set bad color to transparent
cmap = plt.cm.turbo.copy()
cmap.set_bad(color='none', alpha=0)  # Transparent background

# Draw circle and lines FIRST, but mask them to only show in background
circle_x = center_x + radius * np.cos(theta)
circle_y = center_y + radius * np.sin(theta)

# Create coordinate arrays for masking - use adjusted extent
x_coords = np.linspace(adjusted_extent[0], adjusted_extent[1], mask_binary.shape[1])
y_coords = np.linspace(adjusted_extent[2], adjusted_extent[3], mask_binary.shape[0])

# Add radial lines, masked similarly
# radial_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
# for angle in radial_angles:
#     line_x = np.linspace(center_x, center_x + radius * np.cos(angle), 50)
#     line_y = np.linspace(center_y, center_y + radius * np.sin(angle), 50)
    
#     # Only plot segments in background
#     for i in range(len(line_x)-1):
#         x_idx = np.argmin(np.abs(x_coords - line_x[i]))
#         y_idx = np.argmin(np.abs(y_coords - line_y[i]))
        
#         # Add bounds checking
#         if (0 <= x_idx < mask_binary.shape[1] and 0 <= y_idx < mask_binary.shape[0] and 
#             not mask_binary[y_idx, x_idx]):
#             plt.plot([line_x[i], line_x[i+1]], [line_y[i], line_y[i+1]], 
#                     color="#a59678", linewidth=3, alpha=1, linestyle=':')

# Plot circle segments only where mask is False (background)
for i in range(len(theta)-1):
    # Get circle segment coordinates
    x_seg = [circle_x[i], circle_x[i+1]]
    y_seg = [circle_y[i], circle_y[i+1]]
    
    # Check if segment midpoint is in background
    mid_x = (x_seg[0] + x_seg[1]) / 2
    mid_y = (y_seg[0] + y_seg[1]) / 2
    
    # Find closest indices in mask
    x_idx = np.argmin(np.abs(x_coords - mid_x))
    y_idx = np.argmin(np.abs(y_coords - mid_y))
    
    # Add bounds checking and only plot if in background (inverse mask)
    if (0 <= x_idx < mask_binary.shape[1] and 0 <= y_idx < mask_binary.shape[0] and 
        not mask_binary[y_idx, x_idx]):
        plt.plot(x_seg, y_seg, color="#a59678", linewidth=15, alpha=1)


# Add semicircle centered at (0.5, 0) with radius 0.5 - RELATIVE to the unit circle center
semicircle_offset_x = 0.5 * radius  # Scale the offset by the display radius
semicircle_offset_y = 0.0 * radius  # Scale the offset by the display radius
semicircle_radius = 0.5 * radius    # Scale the semicircle radius by the display radius

# Calculate actual center coordinates relative to the unit circle center
semicircle_center_x = center_x + semicircle_offset_x
semicircle_center_y = center_y + semicircle_offset_y

# Create theta for semicircle (upper half)
semicircle_theta = np.linspace(0, np.pi, 500)  # 0 to π for upper semicircle
semicircle_x = semicircle_center_x + semicircle_radius * np.cos(semicircle_theta)
semicircle_y = semicircle_center_y + semicircle_radius * np.sin(semicircle_theta)

# Plot semicircle segments only where mask is False (background)
for i in range(len(semicircle_theta)-1):
    # Get semicircle segment coordinates
    x_seg = [semicircle_x[i], semicircle_x[i+1]]
    y_seg = [semicircle_y[i], semicircle_y[i+1]]
    
    # Check if segment midpoint is in background
    mid_x = (x_seg[0] + x_seg[1]) / 2
    mid_y = (y_seg[0] + y_seg[1]) / 2
    
    # Find closest indices in mask
    x_idx = np.argmin(np.abs(x_coords - mid_x))
    y_idx = np.argmin(np.abs(y_coords - mid_y))
    
    # Add bounds checking and only plot if in background (inverse mask)
    if (0 <= x_idx < mask_binary.shape[1] and 0 <= y_idx < mask_binary.shape[0] and 
        not mask_binary[y_idx, x_idx]):
        plt.plot(x_seg, y_seg, color="#a59678", linewidth=15, alpha=1)


# Plot the histogram AFTER the masked circle and lines
im = plt.imshow(masked_hist, extent=adjusted_extent, 
                origin='lower', cmap=cmap, aspect='equal')

# Use the SAME mask that was used for masking the histogram
mask_smooth = gaussian_filter(mask_binary.astype(float), sigma=5)

# Create an expanded mask for the outer contour by dilating the original mask
from scipy.ndimage import binary_dilation, binary_erosion
expanded_mask = binary_dilation(mask_binary, iterations=1)  # Expand the mask
expanded_mask_smooth = gaussian_filter(expanded_mask.astype(float), sigma=5)

# Add smoothed contour line with custom color (outer contour) - now expanded
mask_extent = adjusted_extent  # Use the adjusted extent for contours too

# Add inner contour with color #a59678
# Create a smaller mask for the inner contour by eroding the original mask
# inner_mask = binary_erosion(mask_binary, iterations=1)  # Adjust iterations to control inner contour distance
inner_mask_smooth = gaussian_filter(mask_binary.astype(float), sigma=5)

plt.contour(inner_mask_smooth, levels=[0.5], colors='#a59678', linewidths=22, 
           extent=mask_extent, origin='lower', alpha=1)
plt.contour(expanded_mask_smooth, levels=[0.5], colors='#26283d', linewidths=15, 
           extent=mask_extent, origin='lower', alpha=1)

# Remove axes, spines, ticks, and labels
# Remove axes, spines, ticks, and labels
plt.axis('off')

ax = plt.gca()

# Center the data in the plot by setting appropriate limits
data_range = 0.43  # Adjust this to control spacing around your content
ax.set_xlim(center_x - data_range/2, center_x + data_range/2)
ax.set_ylim(center_y - data_range/2, center_y + data_range/2)

# Use axes coordinates for the rounded rectangle so it fills the entire export
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# Create a gradient colormap from bottom-left to top-right
gradient_colors = ['#f4f3f5', '#564b69']
n_bins = 256
cmap_gradient = LinearSegmentedColormap.from_list('gradient', gradient_colors, N=n_bins)

# Create the gradient background
# First create a meshgrid for the gradient
x_grad = np.linspace(0, 1, 1000)
y_grad = np.linspace(0, 1, 1000)
X_grad, Y_grad = np.meshgrid(x_grad, y_grad)

# Create diagonal gradient (bottom-left to top-right)
base_gradient = (X_grad + 1 - Y_grad) / 2

# Create a gradient where upper diagonal is mostly dark, lower has sharp transition
# Split the gradient: upper half mostly dark, lower half sharp transition
gradient_data = np.where(base_gradient > 0.5, 
                        1,  # Upper: 70% to 100% dark
                        np.power(base_gradient * 2, 0.4))  # This creates the correct diagonal

# Create the rounded rectangle FIRST to use as a clip mask
rounded_rect_mask = FancyBboxPatch(
    (0, 0), 1, 1,  # Full axes coordinates (0,0 to 1,1)
    transform=ax.transAxes,  # Use axes coordinate system
    boxstyle="round,pad=0.002,rounding_size=0.12",
    facecolor='none',  # Temporary color for the mask
    edgecolor='none',
    mutation_aspect=1.0,
    clip_on=False
)

# Add the gradient as an image behind everything, clipped to the rounded rectangle
im_gradient = plt.imshow(gradient_data, extent=[0, 1, 0, 1], cmap=cmap_gradient, 
                        aspect='equal', transform=ax.transAxes, zorder=-200, alpha=1)
im_gradient.set_clip_path(rounded_rect_mask)

# Create the rounded rectangle border
rounded_rect_border = FancyBboxPatch(
    (0, 0), 1, 1,  # Full axes coordinates (0,0 to 1,1)
    transform=ax.transAxes,  # Use axes coordinate system
    boxstyle="round,pad=0.002,rounding_size=0.12",
    facecolor='none',        # Transparent face to show gradient
    edgecolor='#26283d',
    linewidth=20,
    mutation_aspect=1.0,
    zorder=-50,              # Above gradient but below other content
    clip_on=False
)
ax.add_patch(rounded_rect_border)

ax.text(0.5, -0.05,  # x=0.5 (center), y=-0.08 (below the box)
        'napari-phasors', 
        transform=ax.transAxes,  # Use axes coordinates
        fontsize=66,             # Adjust size as needed
        color='#26283d',         # Same color as box background
        ha='center',             # Horizontal alignment: center
        va='top',                # Vertical alignment: top
        fontweight='bold',       # Make it bold
        fontfamily='DejaVu Sans',
        ) # Clean font

# Save as PNG with smaller pad_inches
plt.savefig('phasor_logo.png', 
            dpi=300,                    # High resolution
            bbox_inches='tight',        # Tight bounding box
            pad_inches=0.1,             # Adjust this for border spacing
            facecolor='#564b69',       # Match background
            edgecolor='none')

# Save as SVG with background
plt.savefig('phasor_logo.svg', 
            bbox_inches='tight',        # Tight bounding box
            pad_inches=0.1,             # Adjust this for border spacing
            facecolor='#564b69',       # Match background
            edgecolor='none',
            format='svg')

# Now create versions with transparent background
plt.gcf().patch.set_facecolor('none')
plt.gca().set_facecolor('none')

# Save PNG with transparent background
plt.savefig('phasor_logo_transparent.png', 
            dpi=300,                    # High resolution
            bbox_inches='tight',        # Tight bounding box
            pad_inches=0.2,             # Smaller padding (was 0.1)
            transparent=True,           # Transparent background
            facecolor='none')

# Save SVG with transparent background
plt.savefig('phasor_logo_transparent.svg', 
            bbox_inches='tight',        # Tight bounding box
            pad_inches=0.2,             # Smaller padding
            transparent=True,           # Transparent background
            facecolor='none',
            format='svg')
# %%
