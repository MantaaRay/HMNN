from concurrent.futures import ThreadPoolExecutor
import random
import time
import googlemaps
from matplotlib.widgets import Button, Slider
import numpy as np
import matplotlib.pyplot as plt
import math
import geopandas as gpd
from shapely.geometry import Point
from getapikeys import get_gmaps_key


def meters_to_degrees(meters, latitude):
    return meters / (111.32 * 1000 * math.cos(math.radians(latitude)))

def get_elevation_data(api_key, lat, lon, size, resolution_meters):
    gmaps = googlemaps.Client(key=api_key)
    elevations = np.zeros((size, size))
    resolution_degrees = meters_to_degrees(resolution_meters, lat)
    locations = []
    half_size = size // 2
    center_offset_meters = half_size * resolution_meters
    center_offset_degrees = meters_to_degrees(center_offset_meters, lat)

    for i in range(size):
        for j in range(size):
            lat_offset = (i - half_size) * resolution_degrees
            lon_offset = (j - half_size) * resolution_degrees
            location = (lat + lat_offset, lon + lon_offset)
            locations.append(location)

    # Split locations into batches of 512
    batches = [locations[i:i + 512] for i in range(0, len(locations), 512)]
    num_requests = len(batches)
    print(f"Number of requests to be made: {num_requests}")
    continue_prompt = input(f"Continue with {num_requests} requests? y/n: ")
    if continue_prompt.lower() != 'y':
        print("Exiting.")
        return None

    batch_size = 512
    for batch_index, batch in enumerate(batches):
        results = gmaps.elevation(batch)
        for i, result in enumerate(results):
            overall_index = batch_index * batch_size + i
            row = overall_index // size
            col = overall_index % size
            elevations[row, col] = result['elevation']

    return elevations

def plot_elevation_data(elevations):
    plt.imshow(elevations, cmap='terrain')
    plt.colorbar(label='Elevation (meters)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Elevation Map')
    plt.show()
    
def is_point_on_land_within_radius(point, gdf_land, spatial_index):
    # Use the spatial index to find possible matches
    possible_matches_index = list(spatial_index.intersection(point.bounds))
    possible_matches = gdf_land.iloc[possible_matches_index]

    # Check if the point is within any land geometry
    return any(possible_matches.contains(point))

def generate_random_point_on_land_within_radius(center_point, radius, gdf_land, spatial_index):
    center_x, center_y = center_point.x, center_point.y
    angle = random.uniform(0, 360)
    distance = random.uniform(0, radius)
    random_x = center_x + distance * math.cos(math.radians(angle))
    random_y = center_y + distance * math.sin(math.radians(angle))
    random_point = Point(random_x, random_y)
    if is_point_on_land_within_radius(random_point, gdf_land, spatial_index):
        return random_point
    return None

def generate_random_points_on_land_within_radius(num_points, center_lat, center_lon, radius, shapefile_path):
    start_time = time.time()

    # Read the shapefile that represents land
    gdf_land = gpd.read_file(shapefile_path)

    # Simplify the geometries to speed up spatial queries
    gdf_land['geometry'] = gdf_land['geometry'].simplify(tolerance=0.01)

    # Create a spatial index
    spatial_index = gdf_land.sindex

    center_point = Point(center_lon, center_lat)

    # Use a thread pool to generate random points on land
    random_points_on_land = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_random_point_on_land_within_radius, center_point, radius, gdf_land, spatial_index) for _ in range(num_points)]
        for future in futures:
            point = future.result()
            if point:
                random_points_on_land.append(point)

    end_time = time.time()
    print(f"Total time to generate points: {end_time - start_time} seconds")

    return random_points_on_land
    
def is_point_on_land(point, gdf_land, spatial_index):
    # Use the spatial index to find possible matches
    possible_matches_index = list(spatial_index.intersection(point.bounds))
    possible_matches = gdf_land.iloc[possible_matches_index]

    # Check if the point is within any land geometry
    return any(possible_matches.contains(point))

def generate_random_point_on_land(bounds, gdf_land, spatial_index):
    min_x, min_y, max_x, max_y = bounds
    while True:
        random_x = random.uniform(min_x, max_x)
        random_y = random.uniform(min_y, max_y)
        random_point = Point(random_x, random_y)
        if is_point_on_land(random_point, gdf_land, spatial_index):
            return random_point
    
def generate_random_points_on_land(num_points, shapefile_path):
    start_time = time.time()

    # Read the shapefile that represents land
    gdf_land = gpd.read_file(shapefile_path)

    # Simplify the geometries to speed up spatial queries
    gdf_land['geometry'] = gdf_land['geometry'].simplify(tolerance=0.01)

    # Create a spatial index
    spatial_index = gdf_land.sindex

    # Get the bounds of the land shapefile
    bounds = gdf_land.total_bounds

    # Use a thread pool to generate random points on land
    with ThreadPoolExecutor() as executor:
        random_points_on_land = list(executor.map(
            lambda _: generate_random_point_on_land(bounds, gdf_land, spatial_index),
            range(num_points)
        ))

    end_time = time.time()
    total_point_generation_time = end_time - start_time
    print(f"Total time to generate points: {total_point_generation_time} seconds")
    print(f"Average time to generate a point: {total_point_generation_time / num_points} seconds")
    return random_points_on_land
    

def plot_random_points_on_land(num_points, shapefile_path):
    # Generate random points on land
    random_points_on_land = generate_random_points_on_land_within_radius(num_points, 36.57, -118.29, 5, shapefile_path)

    # Read the shapefile that represents land
    gdf_land = gpd.read_file(shapefile_path)

    # Create a GeoDataFrame for the random points
    gdf_random_points = gpd.GeoDataFrame(geometry=random_points_on_land)

    # Plot the land
    plot_start_time = time.time()
    ax = gdf_land.plot(color='lightgreen')

    # Plot the random points as markers
    gdf_random_points.plot(ax=ax, marker='o', color='red', markersize=5)

    plot_end_time = time.time()
    print(f"Time to plot: {plot_end_time - plot_start_time} seconds")

    # Show the plot
    plt.show()
    
def plot_land_and_points(gdf_land, points, ax):
    gdf_land.plot(ax=ax, color='lightgrey')
    gdf_points = gpd.GeoDataFrame(geometry=points)
    gdf_points.plot(ax=ax, marker='o', color='red', markersize=5)

def toggle_view(event):
    global zoomed_in
    zoomed_in = not zoomed_in
    update_view()
    
def update_radius(val):
    update_view()
    
def update_view():
    ax.clear()
    plot_land_and_points(gdf_land, random_points_on_land, ax)
    if zoomed_in:
        ax.set_xlim(center_lon - zoom_radius.val, center_lon + zoom_radius.val)
        ax.set_ylim(center_lat - zoom_radius.val, center_lat + zoom_radius.val)
    plt.draw()

# Parameters
api_key = get_gmaps_key()
lat = 36.5775  # Center latitude
lon = -118.2949  # Center longitude
size = 3000  # Size of the image (100x100)
resolution_meters = 1  # Resolution in meters

# # Get elevation data
# elevations = get_elevation_data(api_key, lat, lon, size, resolution_meters)

# # Plot the elevation data
# if elevations is not None:
#     plot_elevation_data(elevations)

# Read the shapefile
# shapefile_path = 'land10m.shp'
# gdf = gpd.read_file(shapefile_path)

# Plot the shapefile
# gdf.plot()

# Customize the plot (optional)
# plt.title('My Shapefile')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')

# plt.scatter(50, -120, marker='o', color='red')


# Show the plot
# plt.show()
# plot_random_points_on_land(100, shapefile_path)

# Example usage
shapefile_path = 'land10m.shp'
center_lat = 36.7783
center_lon = -119.4179

# Read the shapefile that represents land
gdf_land = gpd.read_file(shapefile_path)

# Generate random points on land (using the previous code)
random_points_on_land = generate_random_points_on_land_within_radius(100, center_lat, center_lon, 1, shapefile_path)

# Create the plot
fig, ax = plt.subplots()
plot_land_and_points(gdf_land, random_points_on_land, ax)

# Create a button to toggle the view
zoomed_in = False
button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
button = Button(button_ax, 'Toggle View')
button.on_clicked(toggle_view)

# Create a slider to set the zoom radius
slider_ax = plt.axes([0.2, 0.05, 0.5, 0.03])
zoom_radius = Slider(slider_ax, 'Zoom Radius', 0.1, 10, valinit=1)
zoom_radius.on_changed(update_radius)

plt.show()