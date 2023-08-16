from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from pathlib import Path
import numpy as np
from scipy.interpolate import griddata
import random


def get_center_coords(left, bottom, right, top, tiff_crs):
    # Calculate the center coordinates
    center_x_coord = (left + right) / 2
    center_y_coord = (top + bottom) / 2

    # Define a transformer from the TIFF's CRS to WGS 84
    transformer = Transformer.from_crs(tiff_crs, 'EPSG:4326')

    # Transform the coordinates
    latitude, longitude = transformer.transform(center_x_coord, center_y_coord)

    return latitude, longitude

def save_tiles_as_tiff(tiles, save_directory):
    save_path = Path(save_directory)
    save_path.mkdir(exist_ok=True)

    for idx, (tile_data, tile_meta) in enumerate(tiles):
        tile_path = save_path / f'tile_{idx}.tif'
        with rasterio.open(tile_path, 'w', **tile_meta) as dst:
            dst.write(tile_data)
            
def has_discontinuity(data, threshold=100):
    # Calculate the differences between adjacent pixels
    diffs = np.abs(np.diff(data))

    # Flatten the differences
    flattened_diffs = diffs.flatten()

    # Find the differences that exceed the threshold
    large_diffs = flattened_diffs[flattened_diffs > threshold]

    # Print the top 10 highest differences
    top_10_diffs = sorted(large_diffs, reverse=True)[:10]
    print(f"Top 10 highest differences: {top_10_diffs}")

    return len(large_diffs) > 0






def extract_tiles_from_large_tiff(tiff_path, tile_size=4000, overlap=1500, target_resolution=1.0, max_nodata_percentage=5, randomness=.7, min_random=.3):
    tiles = []

    with rasterio.open(tiff_path) as src:
        # Check and Adjust Resolution
        current_resolution = src.res[0]
        if current_resolution != target_resolution:
            scaling_factor = current_resolution / target_resolution
            new_shape = (int(src.height * scaling_factor), int(src.width * scaling_factor))
            resampled_data = src.read(
                out_shape=new_shape,
                resampling=Resampling.bilinear
            )
            transform = src.transform * src.transform.scale(
                (src.width / resampled_data.shape[-1]),
                (src.height / resampled_data.shape[-2])
            )
        else:
            resampled_data = src.read()
            transform = src.transform

        nodata_value = src.nodatavals[0]  # Get the nodata value for the TIFF file
        
        if (nodata_value != None):
            # Find the coordinates of the nodata and non-nodata values
            nodata_coords = np.column_stack(np.where(resampled_data[0] == nodata_value))
            non_nodata_coords = np.column_stack(np.where(resampled_data[0] != nodata_value))

            # Interpolate the nodata values from the non-nodata values
            interpolated_values = griddata(non_nodata_coords, resampled_data[0][resampled_data[0] != nodata_value], nodata_coords, method='nearest')
            resampled_data[0][nodata_coords[:, 0], nodata_coords[:, 1]] = interpolated_values
        
        



        # Iterate Through the Large TIFF to Create Tiles
        rows, cols = resampled_data.shape[1:]

        i = 0
        while i + tile_size <= cols:
            j = 0
            while j + tile_size <= rows:
                window = rasterio.windows.Window(j, i, tile_size, tile_size)
                tile_data = resampled_data[:, i:i+tile_size, j:j+tile_size]

                # Save the tile data and metadata
                tile_transform = rasterio.windows.transform(window, transform)
                tile_meta = src.meta.copy()
                tile_meta.update({
                    'height': tile_size,
                    'width': tile_size,
                    'transform': tile_transform
                })

                tiles.append((tile_data, tile_meta))

                # Generate a random offset for the next tile
                random_offset = random.randint(np.round(min_random * tile_size), np.round(randomness * tile_size))
                j += random_offset

            # Generate a random offset for the next row of tiles
            random_offset = random.randint(np.round(min_random * tile_size), np.round(randomness * tile_size))
            i += random_offset
        
    tiles_to_return = []
    for index, (tile_data, tile_meta) in enumerate(tiles):
        # ... (same code as before for NoData handling)

        # Create a tile name using the index
        tile_name = f"tile_{index}"

        if has_discontinuity(tile_data):
            print(f"Discarding tile {tile_name} due to discontinuity")
            continue
        else:
            tiles_to_return.append((tile_data, tile_meta))

    # # Handle NoData values outside the window loop
    # for tile_data, tile_meta in tiles:
    #     # Calculate the percentage of nodata values
    #     nodata_percentage = (tile_data == nodata_value).sum() / (tile_size * tile_size) * 100

    #     # Skip the tile if the percentage of nodata values is above the threshold
    #     if nodata_percentage > max_nodata_percentage:
    #         continue

    #     # Find the coordinates of the nodata and non-nodata values
    #     nodata_coords = np.column_stack(np.where(tile_data[0] == nodata_value))
    #     non_nodata_coords = np.column_stack(np.where(tile_data[0] != nodata_value))

    #     # Interpolate the nodata values from the non-nodata values
    #     interpolated_values = griddata(non_nodata_coords, tile_data[0][tile_data[0] != nodata_value], nodata_coords, method='nearest')
    #     tile_data[0][nodata_coords[:, 0], nodata_coords[:, 1]] = interpolated_values

    return tiles_to_return


# Example usage:
tiff_path = 'featuregathering/bigtiffs/gedemsa_1_2.tif'
tiles = extract_tiles_from_large_tiff(tiff_path, min_random=0, randomness=.5)

# You can now save these tiles as individual TIFF files or use them as needed.

# Example usage:
save_directory = 'featuregathering/tiffs'
save_tiles_as_tiff(tiles, save_directory)