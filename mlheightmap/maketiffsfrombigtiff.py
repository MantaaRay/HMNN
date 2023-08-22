import os
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from pathlib import Path
import numpy as np
from scipy.interpolate import griddata
import random
from pathlib import Path
import geopy
from geopy.geocoders import Nominatim
import time
import googlemaps

from getapikeys import get_gmaps_key



def get_center_coords(left, bottom, right, top, tiff_crs):
    # Calculate the center coordinates
    center_x_coord = (left + right) / 2
    center_y_coord = (top + bottom) / 2

    # Define a transformer from the TIFF's CRS to WGS 84
    transformer = Transformer.from_crs(tiff_crs, 'EPSG:4326')

    # Transform the coordinates
    latitude, longitude = transformer.transform(center_x_coord, center_y_coord)

    return latitude, longitude

def get_tilemeta_center_coords(tile_meta):
    # Get the transform from the metadata
    transform = tile_meta['transform']

    # Calculate the bounds
    left = transform.c
    top = transform.f
    right = left + transform.a * tile_meta['width']
    bottom = top + transform.e * tile_meta['height']

    # Use the get_center_coords function to calculate the center coordinates
    return get_center_coords(left, bottom, right, top, tile_meta['crs'])

def get_tiff_coords_from_transform(transform, width, height, crs):
    # Calculate the bounds
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height

    return get_center_coords(left, bottom, right, top, crs)


# def get_tilemeta_center_coords(tile_meta):
#     # Get the bounds of the cropped region
#     left, bottom, right, top = tile_meta['bounds']

#     return get_center_coords(left, bottom, right, top, tile_meta['crs'])

def create_tiff_filename(coords, folder_path):
    # Use Geopy to look up the location based on the coordinates
    google_maps_api_key = get_gmaps_key()
    gmaps = googlemaps.Client(key=google_maps_api_key)

    # Reverse geocode the coordinates
    reverse_geocode_results = gmaps.reverse_geocode(coords)
    # Initialize variables to hold the country and state names
    country_name = None
    state_name = None

    # Iterate through the results
    for result in reverse_geocode_results:
        # Iterate through the address components to find the country and state names
        for component in result['address_components']:
            if 'country' in component['types']:
                country_name = component['long_name']
            if 'administrative_area_level_1' in component['types']:
                state_name = component['long_name']

            if country_name and state_name:
                break
        if country_name and state_name:
            break

    # Use the state name if the country is the United States, otherwise use the country name
    feature_name = state_name if country_name == 'United States' else country_name

    if feature_name is None:
        raise ValueError("Feature name not found in geocoding result")

    # Replace spaces and special characters with underscores
    feature_name = feature_name.replace(" ", "_")


    # Check for existing files with the same name and add a numerical suffix if needed
    folder = Path(folder_path)
    suffix = 0
    filename = f"{feature_name}.tif"
    while (folder / filename).exists():
        suffix += 1
        filename = f"{feature_name}_{suffix}.tif"
        
    time.sleep(.1)  # Be nice to the Nominatim server

    return filename

def save_tiles_as_tiff(tiles, save_directory):
    save_path = Path(save_directory)
    save_path.mkdir(exist_ok=True)

    for idx, (tile_data, tile_meta) in enumerate(tiles):
        # Skip empty tile data arrays
        if tile_data.size == 0:
            print(f"Skipping empty tile at index {idx}.")
            continue
        
        # Get the NoData value from the metadata
        nodata_value = tile_meta.get('nodata', None)

        # Log the minimum and maximum values, and the NoData value
        print(f"Tile {idx}: Min value = {tile_data.min()}, Max value = {tile_data.max()}, NoData value = {nodata_value}")

        # Check if the tile data contains any NoData values
        if nodata_value is not None and np.any(tile_data == nodata_value):
            print(f"Skipping tile {idx} due to NoData values.")
            continue

        # Create the filename based on the tile metadata
        file_name = create_tiff_filename(get_tilemeta_center_coords(tile_meta), save_directory)
        tile_path = save_path / file_name
        
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
        
        # if (nodata_value != None):
        #     # Find the coordinates of the nodata and non-nodata values
        #     nodata_coords = np.column_stack(np.where(resampled_data[0] == nodata_value))
        #     non_nodata_coords = np.column_stack(np.where(resampled_data[0] != nodata_value))

        #     # Interpolate the nodata values from the non-nodata values
        #     interpolated_values = griddata(non_nodata_coords, resampled_data[0][resampled_data[0] != nodata_value], nodata_coords, method='nearest')
        #     resampled_data[0][nodata_coords[:, 0], nodata_coords[:, 1]] = interpolated_values
        
        



        # Iterate Through the Large TIFF to Create Tiles
        rows, cols = resampled_data.shape[1:]

        i = 0
        while i + tile_size <= cols:
            j = 0
            while j + tile_size <= rows:
                window = rasterio.windows.Window(j, i, tile_size, tile_size)
                tile_data = resampled_data[:, i:i+tile_size, j:j+tile_size]

                # Check the percentage of NoData values in the tile BEFORE interpolation
                # Interpolation code
                if nodata_value != None:
                    original_nodata_percentage = 100 * np.sum(tile_data[0] == nodata_value) / (tile_size * tile_size)
                    if original_nodata_percentage > max_nodata_percentage:
                        print(f"Skipping tile at ({i}, {j}) due to high original NoData percentage: {original_nodata_percentage}%")
                        j += tile_size - overlap  # Move to the next window
                        continue

                    # Find the coordinates of the nodata and non-nodata values
                    nodata_coords = np.column_stack(np.where(tile_data[0] == nodata_value))
                    non_nodata_coords = np.column_stack(np.where(tile_data[0] != nodata_value))

                    # Interpolate the nodata values from the non-nodata values
                    interpolated_values = griddata(non_nodata_coords, tile_data[0][tile_data[0] != nodata_value], nodata_coords, method='nearest')
                    tile_data[0][nodata_coords[:, 0], nodata_coords[:, 1]] = interpolated_values

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
            
    return tiles_to_return

def extract_tiles_from_large_tiffs(folder_path, *args, **kwargs):
    all_tiles = []
    folder = Path(folder_path)
    for tiff_file in folder.glob('*.tif'):
        tiles = extract_tiles_from_large_tiff(str(tiff_file), *args, **kwargs)
        all_tiles.extend(tiles)
    return all_tiles


# Example usage:
# tiff_path = 'featuregathering/bigtiffs/gedemsa_1_2.tif'
# tiles = extract_tiles_from_large_tiff(tiff_path, min_random=0, randomness=.5)

# bigtiff_folder_path = 'mlheightmap/featuregathering/bigtiffs/testing things 1'
# tiles = extract_tiles_from_large_tiffs(bigtiff_folder_path, tile_size=4000, target_resolution = 1, min_random=.3, randomness=.7)

# # You can now save these tiles as individual TIFF files or use them as needed.

# # Example usage:
# save_directory = 'mlheightmap/featuregathering/tiffs/testing'
# save_tiles_as_tiff(tiles, save_directory)

from scipy.ndimage import zoom
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.widgets as widgets


def normalize_tiff_band(band, nodata_value=None):
    # Normalize the non-nodata pixels
    if nodata_value is not None:
        # Identify the non-nodata pixels
        non_nodata_mask = band != nodata_value
        band_min = band[non_nodata_mask].min()
        band_max = band[non_nodata_mask].max()
        
        # Clip values to prevent overflow issues
        band = np.clip(band, band_min, band_max)
        
        normalized_band = (band - band_min) / (band_max - band_min)
        
        # Set nodata pixels to 0 in the normalized band
        normalized_band[~non_nodata_mask] = 0
        
        return normalized_band
    else:
        return band
      
def scale_tiff_band(src, band_index=0, target_resolution=1):
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
        
    return resampled_data[0], transform

def set_nodatavals_to_nan(band, nodata_value):
    # Set the nodata values to NaN
    band[band == nodata_value] = np.nan
    
    return band

def find_valid_patches(offset_y, offset_x, src_or_band, patch_size, overlap_x, overlap_y, max_nodata_percentage):
    if isinstance(src_or_band, str):
        with rasterio.open(src_or_band) as src:
            band = src.read(1)
    else:
        band = src_or_band

    height, width = band.shape
    valid_patches = []
    invalid_patches = []

    for y in range(offset_y, height - patch_size + 1, patch_size - overlap_y):
        for x in range(offset_x, width - patch_size + 1, patch_size - overlap_x):
            patch = band[y:y+patch_size, x:x+patch_size]
            nodata_count = np.count_nonzero(np.isnan(patch))
            patch_size_total = patch_size * patch_size
            nodata_percentage = (nodata_count / patch_size_total) * 100
            print(f"Patch at ({y}, {x}): {nodata_percentage}% nodata")

            if nodata_percentage <= max_nodata_percentage:
                valid_patches.append((y, x))
            else:
                invalid_patches.append((y, x))
    return valid_patches, invalid_patches

from scipy.interpolate import griddata

def interpolate_nan_values(band):
    # Get coordinates where the data is not NaN
    valid_x, valid_y = np.where(np.isnan(band) == False)
    valid_values = band[valid_x, valid_y]

    # Get coordinates where the data is NaN
    nan_x, nan_y = np.where(np.isnan(band))

    # Interpolate using the valid values
    interpolated_values = griddata((valid_x, valid_y), valid_values, (nan_x, nan_y), method='nearest')

    # Replace the NaN values with the interpolated values
    band[nan_x, nan_y] = interpolated_values

    return band

from rasterio.crs import CRS
from rasterio.transform import Affine


def save_patches(scaled_band, metadata, valid_patches, patch_size, output_folder, target_resolution):
    resolution = metadata['resolution']
    transform = metadata['transform']
    nodatavalue = metadata['nodatavalue']
    crs = metadata['crs']
    # Iterate through the coordinates of valid patches
    for i, (start_y, start_x) in enumerate(valid_patches):
        # Extract the patch
        end_y = start_y + patch_size
        end_x = start_x + patch_size
        # patch = src.read(1, window=((start_y, start_y+patch_size), (start_x, start_x+patch_size)))
        patch = scaled_band[start_y:end_y, start_x:end_x]

        # Apply interpolation if needed
        patch = interpolate_nan_values(patch)
        
        # nodata_value = src.nodatavals[0]
        num_nans_before = np.count_nonzero(np.isnan(patch))
        num_nodata_before = np.count_nonzero(patch == nodatavalue)
        
        print(f'Saving patch {i} with {num_nans_before} NaN values and {num_nodata_before} nodata values before interpolation.')
        print(f'Number of nodata values in original tiff: {np.count_nonzero(scaled_band == nodatavalue)}')

        # Update the transform for this patch
        scaling_factor =  target_resolution / resolution[0]
        # new_transform = transform * Affine.scale(scaling_factor, scaling_factor)
        new_transform = transform * Affine.translation(start_x * scaling_factor, start_y * scaling_factor)
        
        # Count the number of NaN values within the patch
        num_nans_after = np.count_nonzero(np.isnan(patch))

        # Count the number of nodata values within the patch
        num_nodata_after = np.count_nonzero(patch == nodatavalue)

        print(f'Saving patch {i} with {num_nans_after} NaN values and {num_nodata_after} nodata values after interpolation.')

        coords = get_tiff_coords_from_transform(new_transform, patch_size, patch_size, crs)
        name = create_tiff_filename(coords, output_folder)
        
        # Write to a new file
        output_file = os.path.join(output_folder, name)
        with rasterio.open(output_file, 'w', driver='GTiff', height=patch_size, width=patch_size, count=1,
                           dtype=str(patch.dtype), crs=crs, transform=new_transform) as dst:
            dst.write(patch, 1)

        # Here, you can add code to write or calculate any additional metadata you need.

import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

metadata_dict = None
scaled_band = None
valid_patches = None
new_patch_size = None

def display_scaled_down_tiff(opened_tiff, target_resolution, ax1, ax2, fig, root, canvas):
    global metadata_dict, scaled_band, valid_patches, new_patch_size
    band = opened_tiff.read(1)
    nodata_value = opened_tiff.nodatavals[0]
    band = set_nodatavals_to_nan(band, nodata_value)
    # normalized_band = normalize_tiff_band(band, nodatavalue)
    current_resolution = opened_tiff.res[0]
    scaled_band, _ = scale_tiff_band(opened_tiff, target_resolution=target_resolution)
    # normalized_scaled_band = normalize_tiff_band(scaled_band, nodatavalue)
    
    scaled_band = set_nodatavals_to_nan(scaled_band, nodata_value)
    scaled_nodata_count = np.count_nonzero(np.isnan(scaled_band))
    print(f"Scaled NaN count: {scaled_nodata_count}")
    
    
    # Display the original and scaled-down images
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    ax1.imshow(band, cmap='terrain')
    ax1.set_title(f'Original Image (resolution: {current_resolution}m/pixel))')
    ax1.axis('off')
    
    patch_size = 1000
    overlap = 300
    
    # Draw a rectangle on the scaled-down image
    # rect = Rectangle((100, 100), 1000, 1000, linewidth=2, edgecolor='red', facecolor='none')
    
    # valid_patches = find_valid_patches(scaled_band, patch_size, overlap)
    # print("Number of valid patches:", len(valid_patches))
    # print("Example patch locations:", valid_patches[:5])
    
    # cmap = mcolors.LinearSegmentedColormap.from_list('RedBlue', [(1, 0, 0), (0, 0, 1)])
    
    
    ax2.imshow(scaled_band, cmap='terrain')
    # for patch in valid_patches[:6]:
    #     rect = Rectangle((patch[1], patch[0]), patch_size, patch_size, linewidth=2, edgecolor='red', facecolor='none', alpha=0.5)
    #     ax2.add_patch(rect)
    ax2.set_title(f'Scaled-down Image (resolution: {target_resolution}m/pixel))')
    ax2.axis('off')
    
    # patch_size_slider_ax = plt.axes([0.2, 0.02, 0.65, 0.03])
    # overlap_slider_ax = plt.axes([0.2, 0.07, 0.65, 0.03])
    # update_button_ax = plt.axes([0.8, 0.15, 0.1, 0.04])

    
    # patch_size_slider = widgets.Slider(patch_size_slider_ax, 'Patch Size', valmin=100, valmax=2000, valinit=patch_size)
    # overlap_slider = widgets.Slider(overlap_slider_ax, 'Overlap', valmin=0, valmax=1000, valinit=overlap)
    # update_button = widgets.Button(update_button_ax, 'Update')
    
    # interpolate_button_ax = plt.axes([0.8, 0.08, 0.1, 0.04])
    # interpolate_button = widgets.Button(interpolate_button_ax, 'Interpolate')

    # Create a button and text box in your UI
    # Add this function to handle saving
    
    num_nodata_values = np.count_nonzero(scaled_band == nodata_value)
    print(f"Number of nodata values: {num_nodata_values}")
    
    new_patch_size = patch_size
    def update(val):
        new_patch_size = int(patch_size_slider.val)
        new_overlap = int(overlap_slider.val)
        
        ax2.clear()
        ax2.imshow(scaled_band, cmap='terrain')
        
        global valid_patches
        valid_patches = find_valid_patches(scaled_band, new_patch_size, new_overlap, 5)
        for patch in valid_patches:
            rect = Rectangle((patch[1], patch[0]), new_patch_size, new_patch_size, linewidth=2, edgecolor='red', facecolor='none', alpha=0.5)
            ax2.add_patch(rect)
        
        ax2.set_title(f'Scaled-down Image (resolution: {target_resolution}m/pixel))')
        ax2.axis('off')
        
        patches_text = ax2.text(0.02, 0.98, 'Test', transform=ax2.transAxes, color='white', ha='left', va='top')

        new_patches_text = f'Patches: {len(valid_patches)}'
        print(f"New patches text: {new_patches_text}")
        patches_text.set_text(new_patches_text)
        plt.draw()
        
    def update_button_clicked(event):
        update(None)
    
    # update_button.on_clicked(update_button_clicked)
    
    def start_save_patches():
        save_path = save_path_entry.get()
        print(f'Saving patches to {save_path}...')
        metadata_dict = {
        'resolution': opened_tiff.res,
        'transform': opened_tiff.transform,
        'nodatavalue': opened_tiff.nodatavals[0],
        'crs': opened_tiff.crs
        }
        save_patches(scaled_band, metadata_dict, valid_patches, new_patch_size, save_path, target_resolution)
        print(f'Saved patches to {save_path}')

    def browse():
        folder_selected = filedialog.askdirectory()
        save_path_entry.delete(0, tk.END)
        save_path_entry.insert(0, folder_selected)

    # Create a Tkinter window
    
    metadata_dict = {
    'resolution': opened_tiff.res,
    'transform': opened_tiff.transform,
    'nodatavalue': opened_tiff.nodatavals[0],
    'crs': opened_tiff.crs
    }

    # # Create a matplotlib figure and a canvas to embed it in the Tkinter window
    # # canvas = FigureCanvasTkAgg(fig, master=root)
    # canvas_widget = canvas.get_tk_widget()
    # canvas_widget.pack()
    
    # # Add a button to save patches
    # save_button = tk.Button(root, text='Save Patches', command=start_save_patches)
    # save_button.pack()

    # # Add an entry field for the save path
    # save_path_label = tk.Label(root, text='Save Path:')
    # save_path_label.pack()
    # save_path_entry = tk.Entry(root)
    # save_path_entry.pack()

    # # Add a browse button to select the save path
    # browse_button = tk.Button(root, text='Browse', command=browse)
    # browse_button.pack()

    # root.mainloop()

    
    # plt.show()
    
    
global show_invalid_patches
show_invalid_patches = False
    
def display_tiffs_in_folder(folder_path, target_resolution):
    # List all TIFF files in the folder
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.tif')]
    current_file_index = 0

    # Create the plot outside of the display_tiff function
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    root = tk.Tk()
    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()
    
    # # Add sliders for patch size and overlap
    # patch_size_slider = tk.Scale(root, from_=100, to=2000, orient='horizontal', label='Patch Size')
    # patch_size_slider.pack()



    def on_slider_change(val):
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, str(val))

    def on_entry_change():
        try:
            value = int(entry_widget.get())
            patch_size_slider.set(value)
        except ValueError:
            pass
    # Create the patch size slider
    patch_size_slider = tk.Scale(root, from_=100, to=2000, orient=tk.HORIZONTAL, label="Patch Size", command=on_slider_change)
    patch_size_slider.set(1000)
    patch_size_slider.pack()
    overlap_x_slider = tk.Scale(root, from_=0, to=1000, orient='horizontal', label='Overlap X')
    overlap_x_slider.pack()
    overlap_x_slider.set(300)  
    overlap_y_slider = tk.Scale(root, from_=0, to=1000, orient='horizontal', label='Overlap Y')
    overlap_y_slider.pack()
    overlap_y_slider.set(300) # Set initial value for overlap_slider
    offset_x_slider = tk.Scale(root, from_=0, to=1000, orient='horizontal', label='Offset X')
    offset_x_slider.pack()
    offset_x_slider.set(0)     # Set initial value for overlap_slider
    offset_y_slider = tk.Scale(root, from_=0, to=1000, orient='horizontal', label='Offset Y')
    offset_y_slider.pack()
    offset_y_slider.set(0)     # Set initial value for overlap_slider

    # Create an Entry widget for manual input
    entry_widget = tk.Entry(root)
    entry_widget.pack()
    entry_widget.bind('<Return>', lambda event: on_entry_change())

    # Create a label to display the number of valid patches
    patches_label = tk.Label(root, text="")
    patches_label.pack()

    def draw_valid_patches(patch_size, overlap_x, overlap_y):
        global scaled_band, valid_patches
        # Clear existing patches on ax2
        ax2.clear()
        # Compute valid patches
        offset_y = offset_y_slider.get()
        offset_x = offset_x_slider.get()
        valid_patches, invalid_patches = find_valid_patches(offset_y, offset_x, scaled_band, patch_size, overlap_x, overlap_y, 5)
        # Draw the patches
        for patch in valid_patches:
            rect = Rectangle((patch[1], patch[0]), patch_size, patch_size, linewidth=2, edgecolor='red', facecolor='none', alpha=0.5)
            ax2.add_patch(rect)
        if show_invalid_patches:  
            for patch in invalid_patches:
                rect = Rectangle((patch[1], patch[0]), patch_size, patch_size, linewidth=2, edgecolor='blue', facecolor='none', alpha=0.5)
                ax2.add_patch(rect)
        # Redraw the scaled_band and other necessary parts of ax2
        ax2.imshow(scaled_band, cmap='terrain')
        ax2.set_title(f'Scaled-down Image (resolution: {target_resolution}m/pixel))')
        ax2.axis('off')
        # Update the number of patches label
        patches_label.config(text=f'Number of valid patches: {len(valid_patches)}')


    def update():
        # Get the values from the sliders
        patch_size = patch_size_slider.get()
        overlap_x = overlap_x_slider.get()
        overlap_y = overlap_y_slider.get()
        # Call the function to redraw patches
        draw_valid_patches(patch_size, overlap_x, overlap_y)
        # Update the canvas to reflect changes
        canvas.draw()

    def process_file(file_path):
        ax1.clear()
        ax2.clear()
        with rasterio.open(file_path) as opened_tiff:
            display_scaled_down_tiff(opened_tiff, target_resolution, ax1, ax2, fig, root, canvas)
        
        print(f"Processing file: {file_path}")
        patch_size = patch_size_slider.get()
        overlap_x = overlap_x_slider.get()
        overlap_y = overlap_y_slider.get()

        # Call the function to draw the valid patches again
        draw_valid_patches(patch_size, overlap_x, overlap_y)
        file_name_label.config(text=f"Current File Name: {os.path.basename(files[current_file_index])}")
        file_index_label.config(text=f"Current File Index: {current_file_index+1}/{len(files)}")
        # Force a redraw of the canvas to update the plot
        canvas.draw()

    # Add an update button
    update_button = tk.Button(root, text='Update', command=update)
    update_button.pack()
    
    def toggle_invalid_patches():
        global show_invalid_patches
        show_invalid_patches = not show_invalid_patches
        update()
    
    # Add an update button
    toggle_invalid_patches_button = tk.Button(root, text='Toggle Invalid Patches', command=toggle_invalid_patches)
    toggle_invalid_patches_button.pack()
    
    # Create a label to display the current file name
    file_name_label = tk.Label(root, text="")
    file_name_label.pack()

    # Create a label to display the current file index
    file_index_label = tk.Label(root, text="")
    file_index_label.pack()
        
    # Function to navigate to the next file
    def next_file():
        nonlocal current_file_index
        current_file_index = (current_file_index + 1) % len(files)
        # Update the labels with the current file name and index
        process_file(files[current_file_index])

    # Function to navigate to the last file
    def last_file():
        nonlocal current_file_index
        current_file_index = (current_file_index - 1) if current_file_index > 0 else len(files) - 1
        process_file(files[current_file_index])
        
        

    
    def start_save_patches():
        save_path = save_path_entry.get()
        print(f'Saving patches to {save_path}...')
        # metadata_dict = {
        #     'resolution': opened_tiff.res,
        #     'transform': opened_tiff.transform,
        #     'nodatavalue': opened_tiff.nodatavals[0],
        #     'crs': opened_tiff.crs
        # }
        global metadata_dict, valid_patches
        patch_size = patch_size_slider.get()
        save_patches(scaled_band, metadata_dict, valid_patches, patch_size, save_path, target_resolution)
        print(f'Saved patches to {save_path}')

    def browse():
        folder_selected = filedialog.askdirectory()
        save_path_entry.delete(0, tk.END)
        save_path_entry.insert(0, folder_selected)
    
        # Add the widgets that don't change between files here
    save_button = tk.Button(root, text='Save Patches', command=start_save_patches)
    save_button.pack()
    save_path_label = tk.Label(root, text='Save Path:')
    save_path_label.pack()
    save_path_entry = tk.Entry(root)
    save_path_entry.pack()
    browse_button = tk.Button(root, text='Browse', command=browse)
    browse_button.pack()

    # Button for next file
    next_button = tk.Button(root, text='Next File', command=next_file)
    next_button.pack()

    # Button for last file
    last_button = tk.Button(root, text='Last File', command=last_file)
    last_button.pack()

    # Initialize the plot with the current file
    process_file(files[current_file_index])

    root.mainloop()

# tiff_path = 'mlheightmap/featuregathering/bigtiffs/testing things 2/fm449_459_OR08_Wallick_OR08_Wallick_hh.tif'
# with rasterio.open(tiff_path) as src:
#     display_scaled_down_tiff(src, 4)

tiff_folder_path = 'mlheightmap/featuregathering/bigtiffs/first batch'
display_tiffs_in_folder(tiff_folder_path, 3)
    
# def tkinter_test():
#     def save_patches():
#         save_path = save_path_entry.get()
#         print(f'Saving patches to {save_path}')
#         # Your code to save patches...

#     def browse():
#         save_path = filedialog.askdirectory()
#         save_path_entry.delete(0, tk.END)
#         save_path_entry.insert(0, save_path)

#     # Create a Tkinter window
#     root = tk.Tk()

#     # Create a matplotlib figure and a canvas to embed it in the Tkinter window
#     figure, ax = plt.subplots(figsize=(5, 3))
#     canvas = FigureCanvasTkAgg(figure, master=root)
#     canvas_widget = canvas.get_tk_widget()
#     canvas_widget.pack()

#     # Add a button to save patches
#     save_button = tk.Button(root, text='Save Patches', command=save_patches)
#     save_button.pack()

#     # Add an entry field for the save path
#     save_path_label = tk.Label(root, text='Save Path:')
#     save_path_label.pack()
#     save_path_entry = tk.Entry(root)
#     save_path_entry.pack()

#     # Add a browse button to select the save path
#     browse_button = tk.Button(root, text='Browse', command=browse)
#     browse_button.pack()

#     root.mainloop()
    
# tkinter_test()