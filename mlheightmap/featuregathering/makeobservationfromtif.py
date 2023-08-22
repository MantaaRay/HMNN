import glob
import os
from pathlib import Path
from matplotlib import pyplot as plt
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


# def calculate_surface_area(A, B, C, D, resolution):
#     # Convert the corner heights to 3D coordinates
#     A_3d = np.array([0, 0, A])
#     B_3d = np.array([resolution, 0, B])
#     C_3d = np.array([resolution, resolution, C])
#     D_3d = np.array([0, resolution, D])

#     # Vectors for two sides of each triangle
#     vec1_triangle1 = B_3d - A_3d
#     vec2_triangle1 = D_3d - A_3d
#     vec1_triangle2 = B_3d - C_3d
#     vec2_triangle2 = D_3d - C_3d

#     # Cross products for each triangle
#     cross_product_triangle1 = np.cross(vec1_triangle1, vec2_triangle1)
#     cross_product_triangle2 = np.cross(vec1_triangle2, vec2_triangle2)

#     # Surface areas for each triangle (half the magnitude of the cross product)
#     area_triangle1 = np.linalg.norm(cross_product_triangle1) / 2
#     area_triangle2 = np.linalg.norm(cross_product_triangle2) / 2

#     # Total surface area
#     total_surface_area = area_triangle1 + area_triangle2

#     return total_surface_area

# def calculate_chunk_rugosity(heightmap_chunk):
#     rows, cols = heightmap_chunk.shape
#     total_surface_area = 0
#     total_plane_area = 0

#     for i in range(rows - 1):
#         for j in range(cols - 1):
#             # dh_dx = heightmap_chunk[i, j + 1] - heightmap_chunk[i, j]
#             # dh_dy = heightmap_chunk[i + 1, j] - heightmap_chunk[i, j]
#             # dA = np.sqrt(16 + dh_dx**2 + dh_dy**2)
#             surface_area = calculate_surface_area(heightmap_chunk[i, j], heightmap_chunk[i, j + 1], heightmap_chunk[i + 1, j + 1], heightmap_chunk[i + 1, j], 4)
#             total_surface_area += surface_area
#             total_plane_area += 16

    # return total_surface_area, total_plane_area
from scipy.spatial import Delaunay, ConvexHull
from shapely.geometry import Polygon
# import meshplex


def calculate_chunk_rugosity(chunk, chunk_offset):
    rows, cols = chunk.shape
    resolution = 4  # 4 meters per pixel
    surface_area = 0

    # Create points for triangulation
    points = []
    for i in range(rows):
        for j in range(cols):
            x = (i + chunk_offset[0]) * resolution
            y = (j + chunk_offset[1]) * resolution
            points.append((x, y, chunk[i, j]))

    # Perform Delaunay triangulation
    tri = Delaunay([point[:2] for point in points])

    # Iterate through the triangles, calculating the area
    for simplex in tri.simplices:
        # Extract the points for each simplex (triangle) in the triangulation
        triangle_points = [points[index][:2] for index in simplex]
        # Create a Shapely polygon from the triangle points
        polygon = Polygon(triangle_points)
        # Use Shapely's area method to calculate the area
        surface_area += polygon.area

    return surface_area

import logging
from scipy.spatial.distance import euclidean


def add_newell_cross_v3_v3v3(n, v1, v2):
    n[0] += (v1[1] - v2[1]) * (v1[2] + v2[2])
    n[1] += (v1[2] - v2[2]) * (v1[0] + v2[0])
    n[2] += (v1[0] - v2[0]) * (v1[1] + v2[1])

def calculate_triangle_area(v1, v2, v3):
    n = np.zeros(3)
    add_newell_cross_v3_v3v3(n, v1, v2)
    add_newell_cross_v3_v3v3(n, v2, v3)
    add_newell_cross_v3_v3v3(n, v3, v1)
    return np.linalg.norm(n) * 0.5

from scipy.signal import convolve2d

def calculate_roughness(heightmap, resolution=4):
    # rows, cols = heightmap.shape
    # surface_area = 0
    # plane_area = 0

    # # Iterate through the heightmap and calculate the area of each grid cell
    # for i in range(rows - 1):
    #     for j in range(cols - 1):
    #         # Define the four vertices of the grid cell
    #         vertices = [
    #             np.array([i * resolution, j * resolution, heightmap[i, j]]),
    #             np.array([(i + 1) * resolution, j * resolution, heightmap[i + 1, j]]),
    #             np.array([(i + 1) * resolution, (j + 1) * resolution, heightmap[i + 1, j + 1]]),
    #             np.array([i * resolution, (j + 1) * resolution, heightmap[i, j + 1]])
    #         ]

    #         # Divide the grid cell into two triangles and calculate their area
    #         area_triangle1 = calculate_triangle_area(vertices[0], vertices[1], vertices[2])
    #         area_triangle2 = calculate_triangle_area(vertices[0], vertices[2], vertices[3])

    #         # Add the area of the triangles to the total surface area
    #         surface_area += area_triangle1 + area_triangle2
    #         plane_area += resolution * resolution  # each grid cell's flat area

    #     percent_done = (i / (rows - 1)) * 100
    #     if i % 100 == 0:  # Print every 100 rows processed
    #         print(f"{percent_done:.2f}% done")

    # # Calculate the rugosity as the ratio of the actual surface area to the plane area
    # rugosity = surface_area / plane_area
    
    # print(f"Rugosity: {rugosity:.2f}")

    # return rugosity
    
    # Sample heightmap
    
    # Sobel operator for calculating slope
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Convolve with Sobel operator to get gradients
    gradient_x = convolve2d(heightmap, sobel_x, mode='same', boundary='symm')
    gradient_y = convolve2d(heightmap, sobel_y, mode='same', boundary='symm')

    # Calculate slope (magnitude of gradient)
    slope = np.sqrt(gradient_x**2 + gradient_y**2)

    # Calculate average roughness
    percentile_90_roughness = np.median(slope)

    print(f"The roughness is {percentile_90_roughness}")

    return percentile_90_roughness



def get_center_coords(left, bottom, right, top, tiff_crs, src):
    # Calculate the center coordinates
    center_x_coord = (left + right) / 2
    center_y_coord = (top + bottom) / 2
    left, bottom, right, top = src.bounds

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
    latitude, longitude = get_center_coords(left, bottom, right, top, src.crs, src)

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
    # # Resample the TIFF to 1m/pixel resolution
    # resampled_data, transform, crs, meta = resample_tiff_to_resolution(tiff_path)

    # # Update the metadata with the new shape and transform
    # meta.update({
    #     'height': resampled_data.shape[1],
    #     'width': resampled_data.shape[2],
    #     'transform': transform
    # })

    # # Create a temporary rasterio Dataset with the resampled data
    # with rasterio.MemoryFile() as memfile:
    #     with memfile.open(**meta) as src:
    #         src.write(resampled_data)

    #         # Crop the resampled TIFF file to 4000x4000 and get the center coordinates
    #         elevation_data, latitude, longitude = crop_tiff(src)
    
    # Open the TIFF file
    with rasterio.open(tiff_path) as src:
        # Read the elevation data
        elevation_data = src.read(1)

        # Get the bounds of the region
        left, bottom, right, top = src.bounds

        # Get the center coordinates of the region
        latitude, longitude = get_center_coords(left, bottom, right, top, src.crs, src)


    # Get the climate data
    mat, map_mm = get_climate_data(latitude, longitude)

    # Calculate mean elevation and standard deviation of elevation
    E_mean = elevation_data.mean()
    E_stdev = elevation_data.std()
    
    # Calculate rugosity
    roughness = calculate_roughness(elevation_data)

    # Concatenate all scalar features into a single array
    input_features = np.array([E_mean, E_stdev, mat, map_mm, roughness])

    # # Save the features and observation as a NumPy file
    # np.savez(save_path, input_features=input_features, target=elevation_data)
    input_features = {'e_mean': E_mean, 'e_stdev': E_stdev, 'mat': mat, 'map': map_mm, 'roughness': roughness}
    return {'features': input_features, 'target': elevation_data}


    print(f"Observation saved to {save_path}")
    
def make_observation_from_tiffs(input_folder, output_folder, master_save_path=None):
    all_observations = []

    # Use glob to find all the .tif files in the input folder
    tif_files = glob.glob(str(Path(input_folder) / '*.tif'))

    # Iterate through all the matched files
    for input_path in tif_files:
        filename = Path(input_path).name
        observation = make_observation_from_tif(input_path)

        if master_save_path is None:
            output_filename = 'obs_' + Path(filename).stem + '.npz'
            output_path = str(Path(output_folder) / output_filename)
            print(f"Processed {filename}, saved to {output_path}")
            print(f"Features: {observation['features']}, targets: {observation['target'].shape}")
            np.savez(output_path, **observation)
        else:
            all_observations.append(observation)
            print(f"Features: {observation['features']}, targets: {observation['target'].shape}")

    if master_save_path is not None:
        # Save all observations in a master NPZ file
        np.savez(master_save_path, observations=all_observations)
        print(f"Master observation file saved to {master_save_path}")
    else:
        print("All files processed successfully.")


# Example usage:
tiff_path = 'heightmaps/testing/arra10.tif'
# make_observation_from_tif(tiff_path, 'featuregathering/observations/observation2.npz')

input_folder_path = 'mlheightmap/featuregathering/tiffs/testsavepatches/'
output_folder_path = 'mlheightmap/featuregathering/observations/'
make_observation_from_tiffs(input_folder_path, output_folder_path, master_save_path=output_folder_path + 'observations.npz')


# TODO 'free values' featuers adjustable for creative control
# add roughness
# future: river points as free values?