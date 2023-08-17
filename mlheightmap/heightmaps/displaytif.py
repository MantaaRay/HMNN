from pyproj import Transformer
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import Window
import numpy as np

import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import Window
import numpy as np
import glob
import os

def get_center_coords(a, b, left, bottom, right, top):
    # Calculate the center coordinates
    center_x_coord = left + (right - left) * a
    center_y_coord = top + (bottom - top) * b

    return center_x_coord, center_y_coord

def get_coords_lerp(a, b, left, bottom, right, top, tiff_crs):
    center_x_coord, center_y_coord = get_center_coords(a, b, left, bottom, right, top)

    # Define a transformer from the TIFF's CRS to WGS 84
    transformer = Transformer.from_crs(tiff_crs, 'EPSG:4326')

    # Transform the coordinates
    latitude, longitude = transformer.transform(center_x_coord, center_y_coord)

    return latitude, longitude

def display_tiff_subset(tiff_path, top_left_pixel, bottom_right_pixel):
    with rasterio.open(tiff_path) as src:
        # Define the window to read based on the top-left and bottom-right pixel coordinates
        window = Window.from_slices((top_left_pixel[1], bottom_right_pixel[1]), (top_left_pixel[0], bottom_right_pixel[0]))

        # Read the specified window from the first band (index 1) of the TIFF
        image_data = src.read(1, window=window)

        # Check for a NoData value and replace it with NaN
        nodata_value = src.nodatavals[0]
        if nodata_value is not None:
            image_data[image_data == nodata_value] = np.nan

        # Apply a scaling factor if needed
        scaling_factor = src.scales[0]
        if scaling_factor != 1:
            image_data = image_data * scaling_factor

        # Get the transform for the window
        window_transform = rasterio.windows.transform(window, src.transform)
        
        # Get the bounds of the cropped region
        left, bottom, right, top = rasterio.windows.bounds(window, src.transform)

        tl_latitude, tl_longitude = get_coords_lerp(0, 0, left, bottom, right, top, src.crs)
        br_latitude, br_longitude = get_coords_lerp(1, 1, left, bottom, right, top, src.crs)

        # Get the coordinates and resolution in meters
        top_left_coord = [tl_latitude, tl_longitude]
        bottom_right_coord = [br_latitude, br_longitude]
        resolution = src.res[0]

        # Display the image using matplotlib
        plt.imshow(image_data, cmap='terrain')  # You can change the colormap if needed
        plt.colorbar(label='Elevation')
        plt.title(f'TIFF Image Subset\nResolution: {resolution} meters/pixel')
        plt.xlabel(f'Longitude (meters)\n{top_left_coord[0]} to {bottom_right_coord[0]}')
        plt.ylabel(f'Latitude (meters)\n{bottom_right_coord[1]} to {top_left_coord[1]}')
        plt.show()
        
def display_fot_subsets(directory_path, top_left_pixel, bottom_right_pixel, grid_size=(2, 2)):
    tiff_files = glob.glob(os.path.join(directory_path, '*.tif'))
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))

    for idx, tiff_path in enumerate(tiff_files):
        row = idx // grid_size[1]
        col = idx % grid_size[1]

        with rasterio.open(tiff_path) as src:
            # Define the window to read based on the top-left and bottom-right pixel coordinates
            window = Window.from_slices((top_left_pixel[1], bottom_right_pixel[1]), (top_left_pixel[0], bottom_right_pixel[0]))

            # Read the specified window from the first band (index 1) of the TIFF
            image_data = src.read(1, window=window)

            # Check for a NoData value and replace it with NaN
            nodata_value = src.nodatavals[0]
            if nodata_value is not None:
                image_data[image_data == nodata_value] = np.nan

            # Apply a scaling factor if needed
            scaling_factor = src.scales[0]
            if scaling_factor != 1:
                image_data = image_data * scaling_factor

            # Display the image using matplotlib
            axes[row, col].imshow(image_data, cmap='terrain')  # You can change the colormap if needed
            axes[row, col].set_title(os.path.basename(tiff_path))

    plt.show()

# Example usage:
tiff_path = r'mlheightmap\featuregathering\bigtiffs\tahoebasin\hh_38119G81.tif'
top_left_pixel = (0, 0)  # Example coordinates for the top-left pixel
bottom_right_pixel = (10000, 10000)  # Example coordinates for the bottom-right pixel
# display_tiff_subset(tiff_path, top_left_pixel, bottom_right_pixel)
fot_path = r"mlheightmap\featuregathering\tiffs\california"
display_fot_subsets(fot_path, top_left_pixel, bottom_right_pixel, (2, 8))
