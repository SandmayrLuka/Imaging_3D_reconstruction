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

def determine_taper_directions(edges):
    """
    By looking at the extracted edges the direktion of the taker is determined.

    Returns:
    - taper_direction_x: +1 (left-to-right left smaller) or -1 (right-to-left right smaller).
    - taper_direction_y: +1 (top-to-bottom top smaller) or -1 (bottom-to-top bottom smaller).

    Guide: (taper_direction_x, taper_direction_y)
    - (1,1) ... closest part bottom-right
    - (-1,-1) ... closest part top-left
    - (-1,1) ... closest part bottom-left
    - (1,-1) ... closest part top-right
    """

    # Display the edges
    # plt.figure(figsize=(6, 4))
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Map")
    plt.axis('off')
    plt.show()

    rows, cols = edges.shape
    
    # mean position (split once horizontally and once vertically) of edges [list(set(.)) ... removes duplicates]
    # vertical lines (dev picture in left and right - bec. we look along rows and count line pixels we see fewer counts for short vertical lines and higher ones for longer vertical lines)
    left_edges = np.count_nonzero(list(set(np.nonzero(edges[:, :cols // 2])[0]))) # edges[:, :cols // 2] selects the left half of the edge map (all rows and first half of the columns), np.nonzero returns the position of the edges ([0]... only the row indices)
    right_edges = np.count_nonzero(list(set(np.nonzero(edges[:, cols // 2:])[0])))
    # horizontal lines
    top_edges = np.count_nonzero(list(set(np.nonzero(edges[:rows // 2, :])[1])))
    bottom_edges = np.count_nonzero(list(set(np.nonzero(edges[rows // 2:, :])[1])))

    taper_direction_x = 1 if right_edges > left_edges else -1 # horizontal tapering
    taper_direction_y = 1 if bottom_edges > top_edges else -1 # vertical tapering

    # TODO @Sarah: hier hann man auch noch eine Gewichtung einbauen (taper_strength zusätzlich für beide mitgeben)

    return taper_direction_x, taper_direction_y

def shape_from_taper(image_path, max_depth=255, taper_factor_x=1, taper_factor_y=1):
    """
    Returns depth map (heatmap) based on tapering directions (including mixing).

    - max_depth is the maximum depth value for the heatmap (for display in greyscale 255) [can be used for regulations of hight of 3D model]
    - taper_factor_x: Strength of horizontal tapering.
    - taper_factor_y: Strength of vertical tapering.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)  # Canny edge detector -> detect edges

    taper_direction_x, taper_direction_y = determine_taper_directions(edges)
    print(f"Taper Directions: Horizontal={taper_direction_x}, Vertical={taper_direction_y}")
    print(f"Taper Factor: Horizontal={taper_factor_x}, Vertical={taper_factor_y}")
    print(taper_factor_x * taper_direction_x)
    print(taper_factor_y * taper_direction_y)


    rows, cols = image.shape
    print(f"Dimensions: rows={rows}, cols={cols}")
    depth_map = np.zeros_like(image, dtype=np.float32)


    # Compute depth values based on tapering (high value ... close to us -> big depth)
    for y in range(rows):
        for x in range(cols):
            depth_map[y, x] = 1 - (taper_factor_x * taper_direction_x * x) / cols - (taper_factor_y * taper_direction_y * y) / rows

    # Normalize depth map
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return depth_map_normalized

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

#img_path = "/Users/sarahtrausner/Desktop/Master/Semester 3/Imaging beyond consumer cameras/PS/Tests/tapert_surface/block9.png"
img_path = "C:/Users/User/block9.png"

# Displaying mask
image_with_no_background = remove_white_background(img_path)
# Display the result
plt.imshow(image_with_no_background, cmap='gray')
plt.title('Image with White Background Removed')
plt.axis('off')
plt.show()

# Creating heatmap with shape_from_shading
height_map = shape_from_taper(img_path, [0.8, 0.5, 0.8]) # the pictures seem to be flipped !!!
# cv2.imwrite("depth_map.jpg", display_height_map)
print(height_map)
# Display heat map
plt.imshow(height_map, cmap='hot')  # Use 'hot' colormap
plt.colorbar(label='Depth (height)')
plt.title('Heat Map of Height')
plt.axis('off')  # Turn off axes for better visualization
plt.show()

# Masking the heatmap
mask_save_path = "white_background_mask.png"
white_background_mask = save_white_background_mask(img_path, threshold=250, mask_save_path=mask_save_path)
masked_height_map = apply_mask_to_heatmap(height_map, white_background_mask)

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

merge_heightmap_with_color(img_path, masked_height_map, crop=10, scale=10)
merge_heightmap_with_color(img_path, normalized_heatmap, crop=10, scale=10)
merge_heightmap_with_color(img_path, smoothed_heatmap, crop=10, scale=10)
