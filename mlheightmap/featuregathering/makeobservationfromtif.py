from pyproj import Transformer
import rasterio
import requests
import numpy as np
from skimage.transform import resize
from scipy.spatial import Delaunay
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import concurrent
from rasterio.enums import Resampling
from scipy.interpolate import griddata


def calculate_chunk_rugosity(heightmap_chunk):
    rows, cols = heightmap_chunk.shape
    total_surface_area = 0
    total_plane_area = 0

    for i in range(rows - 1):
        for j in range(cols - 1):
            dh_dx = heightmap_chunk[i, j + 1] - heightmap_chunk[i, j]
            dh_dy = heightmap_chunk[i + 1, j] - heightmap_chunk[i, j]
            dA = np.sqrt(1 + dh_dx**2 + dh_dy**2)
            total_surface_area += dA
            total_plane_area += 1

    return total_surface_area, total_plane_area

def calculate_rugosity(heightmap, chunk_size=100):
    rows, cols = heightmap.shape
    total_surface_area = 0
    total_plane_area = 0
    
    total_chunks = ((rows - 1) // chunk_size + 1) * ((cols - 1) // chunk_size + 1)
    processed_chunks = 0

    # Create a thread pool
    with ThreadPoolExecutor() as executor:
        # Split the heightmap into chunks and submit each chunk to the thread pool
        futures = []
        for i in range(0, rows - 1, chunk_size):
            for j in range(0, cols - 1, chunk_size):
                chunk = heightmap[i:i+chunk_size, j:j+chunk_size]
                futures.append(executor.submit(calculate_chunk_rugosity, chunk))

        # Collect the results as they become available
        for future in concurrent.futures.as_completed(futures):
            surface_area, plane_area = future.result()
            total_surface_area += surface_area
            total_plane_area += plane_area
            
            processed_chunks += 1
            percent_done = (processed_chunks / total_chunks) * 100
            print(f"{percent_done:.2f}% done")

    # Calculate the rugosity as the ratio of the actual surface area to the plane area
    rugosity = total_surface_area / total_plane_area

    return rugosity


def get_center_coords(left, bottom, right, top, tiff_crs):
    # Calculate the center coordinates
    center_x_coord = (left + right) / 2
    center_y_coord = (top + bottom) / 2

    # Define a transformer from the TIFF's CRS to WGS 84
    transformer = Transformer.from_crs(tiff_crs, 'EPSG:4326')

    # Transform the coordinates
    latitude, longitude = transformer.transform(center_x_coord, center_y_coord)

    return latitude, longitude

def get_climate_data(latitude, longitude):
    # URL for the Open-Meteo API (you may need to adjust the endpoint and parameters)
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date=2022-01-01&end_date=2023-01-01&hourly=temperature_2m,precipitation"

    # Make the API request
    response = requests.get(url)
    data = response.json()

    # Extract the temperature and precipitation data
    temperatures = np.array(data['hourly']['temperature_2m'])
    precipitation = np.array(data['hourly']['precipitation'])

    # Calculate the mean annual temperature (in Celsius)
    mat_kelvin = temperatures.mean() + 273.15

    # Calculate the mean annual precipitation (in mm)
    map_mm = precipitation.sum()

    return mat_kelvin, map_mm

def crop_tiff(tiff_input, size=(4000, 4000)):
    # Open the TIFF file or use the provided rasterio dataset object
    if isinstance(tiff_input, str):
        src = rasterio.open(tiff_input)
    else:
        src = tiff_input

    # Get the nodata value for the TIFF file
    nodata_value = src.nodatavals[0]

    # Pick a random starting point for the crop
    start_x = np.random.randint(0, src.width - size[0])
    start_y = np.random.randint(0, src.height - size[1])

    # Read the specified window
    window = rasterio.windows.Window(start_x, start_y, size[0], size[1])
    cropped_data = src.read(1, window=window)

    max_nodata_percentage=5
    # Calculate the percentage of nodata values
    nodata_percentage = (cropped_data == nodata_value).sum() / (size[0] * size[1]) * 100

    # If the percentage of nodata values is above the threshold, retry
    if nodata_percentage > max_nodata_percentage:
        print(f"Nodata percentage: {nodata_percentage:.2f}% - Retrying...")
        return crop_tiff(tiff_path, size, max_nodata_percentage)

    # Find the coordinates of the nodata and non-nodata values
    nodata_coords = np.column_stack(np.where(cropped_data == nodata_value))
    non_nodata_coords = np.column_stack(np.where(cropped_data != nodata_value))


    # Interpolate the nodata values from the non-nodata values
    interpolated_values = griddata(non_nodata_coords, cropped_data[cropped_data != nodata_value], nodata_coords, method='nearest')
    cropped_data[nodata_coords[:, 0], nodata_coords[:, 1]] = interpolated_values


    # Get the bounds of the cropped region
    left, bottom, right, top = rasterio.windows.bounds(window, src.transform)

    # Get the center coordinates of the cropped region
    latitude, longitude = get_center_coords(left, bottom, right, top, src.crs)

    # Close the rasterio object if it was opened in this function
    if isinstance(tiff_input, str):
        src.close()

    return cropped_data, latitude, longitude

def resample_tiff_to_resolution(tiff_path, target_resolution=1.0):
    with rasterio.open(tiff_path) as src:
        # Get the current resolution
        current_resolution = src.res[0]

        # Calculate the scaling factor to achieve target resolution
        scaling_factor = current_resolution / target_resolution

        # Resample the TIFF to the desired resolution
        new_shape = (int(src.height * scaling_factor), int(src.width * scaling_factor))
        resampled_data = src.read(
            out_shape=new_shape,
            resampling=Resampling.bilinear
        )

        # Update the transform
        transform = src.transform * src.transform.scale(
            (src.width / resampled_data.shape[-1]),
            (src.height / resampled_data.shape[-2])
        )

        return resampled_data, transform, src.crs, src.meta

def make_observation_from_tif(tiff_path, save_path='featuregathering/observations/observation.npz'):
    # Resample the TIFF to 1m/pixel resolution
    resampled_data, transform, crs, meta = resample_tiff_to_resolution(tiff_path)

    # Update the metadata with the new shape and transform
    meta.update({
        'height': resampled_data.shape[1],
        'width': resampled_data.shape[2],
        'transform': transform
    })

    # Create a temporary rasterio Dataset with the resampled data
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**meta) as src:
            src.write(resampled_data)

            # Crop the resampled TIFF file to 4000x4000 and get the center coordinates
            elevation_data, latitude, longitude = crop_tiff(src)

    # Get the climate data
    mat, map_mm = get_climate_data(latitude, longitude)

    # Calculate mean elevation and standard deviation of elevation
    E_mean = elevation_data.mean()
    E_stdev = elevation_data.std()
    
    # Calculate rugosity
    rugosity = calculate_rugosity(elevation_data)

    # Save the features and observation as a NumPy file
    np.savez(save_path, E_mean=E_mean, E_stdev=E_stdev, mat=mat, map_mm=map_mm, rugosity=rugosity, observation=elevation_data)

    print(f"Observation saved to {save_path}")

# Example usage:
tiff_path = 'heightmaps/testing/arra10.tif'
make_observation_from_tif(tiff_path, 'featuregathering/observations/observation2.npz')

# TODO 'free values' featuers adjustable for creative control
# add roughness
# future: river points as free values?