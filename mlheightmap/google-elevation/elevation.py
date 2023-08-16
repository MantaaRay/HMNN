import googlemaps
from matplotlib import pyplot as plt
import numpy as np
import math

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
