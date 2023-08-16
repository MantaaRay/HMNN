import rasterio
from rasterio.windows import Window
import numpy as np
import random

def crop_tiff(tiff_path, output_path, crop_size=(4000, 4000), max_attempts=1000):
    with rasterio.open(tiff_path) as src:
        # Get the NoData value for the TIFF
        nodata_value = src.nodatavals[0]

        # Define the bounds for the random selection
        max_x = src.width - crop_size[1]
        max_y = src.height - crop_size[0]

        # Attempt to find a valid crop
        for attempt in range(max_attempts):
            # Pick a random point within the bounds
            left = random.randint(0, max_x)
            top = random.randint(0, max_y)

            # Define the window to read
            window = Window(left, top, crop_size[1], crop_size[0])

            # Read the window from the first band (index 1) of the TIFF
            image_data = src.read(1, window=window)

            # Check if the window contains any NoData values
            if nodata_value is not None and (image_data == nodata_value).any():
                continue

            # Create a new TIFF file with the cropped data
            with rasterio.open(output_path, 'w', driver='GTiff',
                               height=crop_size[0], width=crop_size[1],
                               count=1, dtype=image_data.dtype,
                               crs=src.crs, transform=src.window_transform(window)) as dst:
                dst.write(image_data, 1)

            print(f"Successfully cropped region found at attempt {attempt + 1}.")
            return

    print(f"No valid cropped region found without NoData values after {max_attempts} attempts.")

# Example usage:
tiff_path = 'heightmaps/testing/ca13_guo_test.tif'
output_path = 'heightmaps/testing/cropped.tif'
crop_tiff(tiff_path, output_path)
