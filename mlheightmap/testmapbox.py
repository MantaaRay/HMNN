import requests
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

import math

def lat_lon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x_tile, y_tile


def get_mapbox_elevation_tile(x, y, z):
    access_token = 'pk.eyJ1IjoiYmlndHJlZXRoZXNlY29uZCIsImEiOiJjbGw3YWZwbG0wdWJ2M2VtejRpMWkxMGVrIn0.86-30o_u7GDgFlupq65o8w'
    url = f"https://api.mapbox.com/v4/mapbox.mapbox-terrain-dem-v1/{z}/{x}/{y}.pngraw?access_token={access_token}"
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        data = np.array(image)
        max_red = np.max(data[:,:,0])
        max_green = np.max(data[:,:,1])
        max_blue = np.max(data[:,:,2])
        print(f"Highest red value: {max_red}")
        print(f"Highest green value: {max_green}")
        print(f"Highest blue value: {max_blue}")
        print(f"Elevation at highest rgb: {elevation_from_rgb(max_red, max_green, max_blue)}")
        elevation_from_rgb_vectorized = np.vectorize(elevation_from_rgb)
        elevations = elevation_from_rgb_vectorized(data[:,:,0], data[:,:,1], data[:,:,2])
        
        return elevations
    else:
        return None
    
def elevation_from_rgb(r, g, b):
    return -10000 + (r * 256 * 256 + g * 256 + b) * .1

def plot_heightmap(heightmap):
    plt.imshow(heightmap, cmap='gray')
    plt.colorbar(label='Elevation (meters)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Heightmap')
    plt.show()

def get_mapbox_tile(x, y, z):
    access_token = 'pk.eyJ1IjoiYmlndHJlZXRoZXNlY29uZCIsImEiOiJjbGw3YWZwbG0wdWJ2M2VtejRpMWkxMGVrIn0.86-30o_u7GDgFlupq65o8w'
    url = f"https://api.mapbox.com/v4/mapbox.mapbox-terrain-dem-v1/{z}/{x}/{y}.png?access_token={access_token}"
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        return image
    else:
        return None

def plot_mapbox_tile(image):
    plt.imshow(image, cmap="gray")
    plt.axis('off') # To turn off axis
    plt.show()

# Example tile coordinates
# x = 3826
# y = 6127
# z = 14

lat = 36.5593 
lon = -118.2598
zoom = 14

x_tile, y_tile = lat_lon_to_tile(lat, lon, zoom)
print(f"Tile coordinates: X={x_tile}, Y={y_tile}")

# Fetch the elevation tile from Mapbox
elevation_tile = get_mapbox_elevation_tile(x_tile, y_tile, zoom)
# tile_image = get_mapbox_tile(x_tile, y_tile, zoom)
# tile_image_data = np.array(tile_image)


# Plot the heightmap using Matplotlib
if elevation_tile is not None:
    # max_red = np.max(tile_image_data[:,:,0])
    # max_green = np.max(tile_image_data[:,:,1])
    # max_blue = np.max(tile_image_data[:,:,2])
    # print(f"Highest red value: {max_red}")
    # print(f"Highest green value: {max_green}")
    # print(f"Highest blue value: {max_blue}")
    # plot_mapbox_tile(tile_image)
    plot_heightmap(elevation_tile)
else:
    print("Failed to fetch elevation data.")
