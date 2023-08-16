import requests
import json
import numpy as np
import matplotlib.pyplot as plt

def get_elevation(latitude, longitude):
    url = f"https://epqs.nationalmap.gov/v1/json?x={longitude}&y={latitude}&wkid=4326&units=Feet&includeDate=false"
    response = requests.get(url)
    if response.status_code == 200:
        elevation_data = json.loads(response.text)
        return elevation_data['value']
    else:
        return None

def create_heightmap(min_lat, max_lat, min_lon, max_lon, x_pixels, y_pixels):
    lat_values = np.linspace(min_lat, max_lat, y_pixels)
    lon_values = np.linspace(min_lon, max_lon, x_pixels)
    heightmap = np.zeros((y_pixels, x_pixels))
    total_points = x_pixels * y_pixels

    for i, lat in enumerate(lat_values):
        for j, lon in enumerate(lon_values):
            elevation = get_elevation(lat, lon)
            if elevation is not None:
                heightmap[i, j] = elevation
            current_point = i * x_pixels + j + 1
            progress_percent = (current_point / total_points) * 100
            print(f"Progress: {progress_percent:.2f}% done")

    return heightmap

def plot_heightmap(heightmap):
    plt.imshow(heightmap, cmap='terrain')
    plt.colorbar(label='Elevation (feet)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Heightmap')
    plt.show()

# Define the bounding box and number of pixels
min_lat = 37.7
max_lat = 37.8
min_lon = -122.5
max_lon = -122.4
x_pixels = 20   
y_pixels = 20

# Create the heightmap
heightmap = create_heightmap(min_lat, max_lat, min_lon, max_lon, x_pixels, y_pixels)

# Plot the heightmap
plot_heightmap(heightmap)
