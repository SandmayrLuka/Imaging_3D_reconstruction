print("The surface reflects light uniformly in all directions. The position of the light source is known. The surface has uniform reflectivity. The camera projects rays orthogonally onto the image plane.")
print("This implementation assumes uniform lighting and simple geometry.")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def remove_white_background(image_path, threshold=250):
    """
    This function is for displaying the masked image.

    Removing white background from an image by applying threshold.
    Pixels above the threshold intensity are set to black, so the removed background is displayed as black.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    mask = image < threshold # mask where intensity below threshold (high values lighter) ... mask over non-background values

    masked_image = image.copy()
    masked_image[~mask] = 0  # pixels above threshold set to black (black = background)

    return masked_image

def save_white_background_mask(image_path, threshold=250, mask_save_path="mask.png"):
    """
    This function is for creates and saves a mask of the white-background so it can later be applied to the heatmap.

    Returns:
    - mask: Boolean mask of white-background pixels. (True = white)
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = image >= threshold # above threshold detected as white ... mask over background values

    ### mask_image = (mask * 255).astype(np.uint8)  # Convert boolean mask to uint8 (0 or 255)
    ### cv2.imwrite(mask_save_path, mask_image)

    return mask

def apply_mask_to_heatmap(heatmap, mask):
    """
    Applies the mask of the white background to the heatmap. Heigth is set to 0 for the background (masked) pixels.
    
    Input:
    - heatmap: heatmap array
    - mask: white-background mask (boolean). (True = background)
    
    Returns:
    - masked_heatmap: Heatmap with masked (background) pixels set to 0.
    """
    
    masked_heatmap = heatmap.copy()
    masked_heatmap[mask] = 0 # Set values in the heatmap where the mask is True to 0

    return masked_heatmap

def shape_from_shading(image_path, light_source=[0, 0, 1]):
    """
    Create heatmap with shape-from-shading
    
    light source direction examples:
    - [0, 0, 1] (light coming from above, along positive Z-axis)
    - [0, 0, -1] (light coming from below, along negative Z-axis)
    - [-1, 0, 0] (light coming from the left, along negative X-axis)
    - [1, 0, 0] (light coming from the right, along positive X-axis)
    - [0, 1, 0] (light coming directly from the front along Y-axis)
    - [0, -1, 0] (light coming directly from the back along Y-axis)
    - [-1, -1, 1] (light coming from the top-left corner at an angle)
    - [1, -1, 1] (light coming from the top-right corner at an angle)
    - [-1, 1, 1] (light coming from the bottom-left corner at an angle)

    Computes the partial derivatives of the image intensity to approximate the slopes of the surface in the x and y direction. N=(Nx, Ny, Nz) needs to have a unit magnitude that means sqrt(Nx^2+Ny^2+Nz^2)=1. Use this to compute Nz from  Nx, Ny: Nz = sqrt(1-Nx^2-Ny^2). (assumes smooth surface so we don't get negative values in the sqr)

    Compute the cumulative sum of values along the axis.

    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 # Load and normalize image (greyscale) to [0,1] scale | [astype(np.float32) ... float point convertion for math-operations]
    # rows, cols = image.shape

    L = np.array(light_source) # light source direction ([0, 0, 1] ... top-down)
    L = L / np.linalg.norm(L) # Normalize to unit-vector (by deviding by length np.linalg.norm(L))
    print("nom light:", L)

    # np.gradient(image) computes the partial derivatives of the image intensity to approximate the slopes of the surface in the x and y direction
    # Nx=dI/dx ... rate of change of intensity in the horizontal direction
    # Ny=dI/dy ... rate of change of intensity in the vertical direction

    Nx, Ny = np.gradient(image) # surface normals (Nx, Ny, Nz)

    # N=(Nx, Ny, Nz) needs to have a unit magnitude that means sqrt(Nx^2+Ny^2+Nz^2)=1
    # use this to compute Nz from  Nx, Ny: Nz = sqrt(1-Nx^2-Ny^2)
    # !!! assumes smooth surface so we don't get negative values in the sqr

    Nz = np.sqrt(1 - Nx**2 - Ny**2)  # Assumption of Lambertian surface

    N = np.stack((Nx, Ny, Nz), axis=-1) # combining Nx, Ny, Nz into N (surface normals at each pixel)
    N = N / np.linalg.norm(N, axis=-1, keepdims=True) # Normalize

    # np.cumsum computes the cumulative sum of values along a specified axis
    # np.cumsum(Nx, axis=1) ... Integrates the gradient in the x-direction along rows
    # np.cumsum(Ny, axis=1) ... Integrates the gradient in the y-direction along cols
    # the sum of those two combines the integrations to approximate the height

    height_map = np.cumsum(Nx * L[0], axis=1) + np.cumsum(Ny * L[1], axis=0) # Compute height map by integration

    # Compute the dot product between the surface normal and light source direction
    ### height_map = np.sum(N * L, axis=-1)  # Dot product of N and L

    # astype(np.uint8) ... for the display we need int values (convert back)
    height_map_normalized = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # Normalize depth map to display (scales height to [0, 255] => 8-bit image)
    return height_map, height_map_normalized

def smooth_heatmap(heatmap, kernel_size=5, sigma=1):
    """
    Smooths the heatmap using a Gaussian blur to reduce distortions along object (and shadow) borders.
    """
    # Ensure kernel size is odd and not 0 or negative
    if kernel_size  <= 0:
        kernel_size = 1

    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply Gaussian blur
    smoothed_heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), sigma)
    return smoothed_heatmap

def merge_heightmap_with_color(image_path, height_map, crop=10, scale=1):
    """
    Creates a 3D visualization by merging the cropped height map with the color information from the image.
    
    Input:
    - crop: Number of pixels to crop from each side of the image and height map.
    - scale: Scale factor for the height map (exaggerate depth for visualization).
    """

    color_image = cv2.imread(image_path)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    if color_image.shape[:2] != height_map.shape: # ensure the color image matches the size of the height map
        color_image = cv2.resize(color_image, (height_map.shape[1], height_map.shape[0]))

    height_map_cropped = height_map[crop:-crop, crop:-crop] # crop to remove edge distortions
    color_image_cropped = color_image[crop:-crop, crop:-crop] # crop to remove edge distortions

    # grid coordinates for 3D surface
    x = np.arange(0, height_map_cropped.shape[1])
    y = np.arange(0, height_map_cropped.shape[0])
    x, y = np.meshgrid(x, y)

    z = height_map_cropped * scale # scale the cropped height map (ex. exaggerate depth, scale down depth)

    # Plot the 3D surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use the cropped color image as the surface texture
    ax.plot_surface(
        x, y, z,
        facecolors=color_image_cropped / 255.0,
        rstride=1, cstride=1, linewidth=0, antialiased=True
    )

    # Add labels and adjust view
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Height")
    ax.set_title("3D Visualization with Cropped Height Map and Color Information layover")
    ax.view_init(elev=45, azim=120)  # view

    plt.show()

# Example usage

#img_path = "/Users/sarahtrausner/Desktop/Master/Semester 3/Imaging beyond consumer cameras/PS/Tests/surface_reflection/dome.png"
img_path = "C:/Users/User/dome.png"
#img_path = "C:/Users/User/cone.png"

# Displaying mask
image_with_no_background = remove_white_background(img_path)
# Display the result
plt.imshow(image_with_no_background, cmap='gray')
plt.title('Image with White Background Removed')
plt.axis('off')
plt.show()

# Creating heatmap with shape_from_shading
array_height_map, display_height_map = shape_from_shading(img_path, [0.8, 0.5, 0.8]) # the pictures seem to be flipped !!!
# cv2.imwrite("depth_map.jpg", display_height_map)
print(array_height_map)
print(display_height_map)
# Display heat map
plt.imshow(display_height_map, cmap='hot')  # Use 'hot' colormap
plt.colorbar(label='Depth (height)')
plt.title('Heat Map of Height')
plt.axis('off')  # Turn off axes for better visualization
plt.show()

# Masking the heatmap
mask_save_path = "white_background_mask.png"
white_background_mask = save_white_background_mask(img_path, threshold=250, mask_save_path=mask_save_path)
masked_height_map = apply_mask_to_heatmap(array_height_map, white_background_mask)

# normalize non-masked pixels
non_masked_values = masked_height_map[~white_background_mask]
min_val, max_val = non_masked_values.min(), non_masked_values.max()

"""
# Normalize valid pixels to range [0, max_range] with a linear function
max_range = 10
normalized_heatmap = np.zeros_like(masked_height_map, dtype=np.uint8)  # Initialize all as 0
normalized_heatmap[~white_background_mask] = ((masked_height_map[~white_background_mask] - min_val) / (max_val - min_val) * max_range).astype(np.uint8)
"""

# Normalize valid pixels to range [0, max_range] with a non-linear function (gamma transformation)
max_range = 10 # max 255
gamma = 0.4  # Gamma < 1 compresses high values more
normalized_heatmap = np.zeros_like(masked_height_map, dtype=np.uint8)  # Initialize all as 0
# Scale values to [0, 1], apply gamma correction, and scale back to [0, max_range]
normalized_values = (masked_height_map[~white_background_mask] - min_val) / (max_val - min_val)
gamma_corrected_values = np.power(normalized_values, gamma)  # Apply gamma correction
normalized_heatmap[~white_background_mask] = (gamma_corrected_values * max_range).astype(np.uint8)


# smooth the normalized heatmap (get rid of distorting on object brorders and the borders of shadows)
smoothed_heatmap = smooth_heatmap(normalized_heatmap, kernel_size=1, sigma=1)



# Display the masked heatmap
plt.imshow(masked_height_map, cmap='hot')
plt.colorbar(label='Depth (height)')
plt.title('Masked Heat Map of Height - masked height map')
plt.axis('off')
plt.show()
# Display the normalized masked heatmap
plt.imshow(normalized_heatmap, cmap='hot')
plt.colorbar(label='Depth (height)')
plt.title('Masked Heat Map of Height - normalized masked height map')
plt.axis('off')
plt.show()
# Display the smoothed normalized masked heatmap
plt.imshow(smoothed_heatmap, cmap='hot')
plt.colorbar(label='Depth (height)')
plt.title('Masked Heat Map of Height - smoothed normalized masked height map')
plt.axis('off')
plt.show()

#merge_heightmap_with_color(img_path, masked_height_map, crop=10, scale=10)
#merge_heightmap_with_color(img_path, normalized_heatmap, crop=10, scale=10)
merge_heightmap_with_color(img_path, smoothed_heatmap, crop=10, scale=10)
