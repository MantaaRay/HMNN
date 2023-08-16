import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from shapely.geometry import box


def distance_to_nearest_lake(latitude, longitude, tree, lake_points):
    location = Point(longitude, latitude)
    distance, _ = tree.query((location.y, location.x))
    return distance

# Load the lakes shapefile
lakes_shapefile_path = 'lakes shapefiles/ne_10m_lakes.shp'
lakes = gpd.read_file(lakes_shapefile_path)

# Extract the coordinates of the lake points
lake_points = []
for lake in lakes.geometry:
    if isinstance(lake, Polygon):
        lake_points.extend(lake.exterior.coords)
    else:  # MultiPolygon
        for polygon in lake.geoms:
            lake_points.extend(polygon.exterior.coords)

lake_points = np.array([(point[1], point[0]) for point in lake_points])


# Build a spatial index for the lake points
tree = cKDTree(lake_points)

# Define the grid resolution
resolution = .1

# Define the bounds for the continental U.S.
latitude_bounds = (24.396308, 49.384358)
longitude_bounds = (-125.000000, -66.934570)

# Create the grid covering the continental U.S.
latitudes = np.arange(latitude_bounds[0], latitude_bounds[1], resolution)
longitudes = np.arange(longitude_bounds[0], longitude_bounds[1], resolution)
distance_raster = np.zeros((len(latitudes), len(longitudes)))

# Calculate the distance to the nearest lake for each point in the grid
total_points = len(latitudes) * len(longitudes)
processed_points = 0

for i, latitude in enumerate(latitudes):
    for j, longitude in enumerate(longitudes):
        distance_raster[i, j] = distance_to_nearest_lake(latitude, longitude, tree, lake_points)
        processed_points += 1
        if processed_points % 1000 == 0:
            percent_done = (processed_points / total_points) * 100
            print(f"{percent_done:.2f}% done")

# Normalize the distances
distance_raster = (distance_raster - distance_raster.min()) / (distance_raster.max() - distance_raster.min())

# Plot the normalized distances using Matplotlib
plt.imshow(distance_raster, extent=[longitude_bounds[0], longitude_bounds[1], latitude_bounds[0], latitude_bounds[1]], cmap='gray', origin='lower')
plt.colorbar(label='Normalized Distance to Nearest Lake')
plt.title('Distance to Nearest Lake in the U.S.')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

bounding_box = box(longitude_bounds[0], latitude_bounds[0], longitude_bounds[1], latitude_bounds[1])

lakes_clipped = lakes[lakes.geometry.intersects(bounding_box)]


# Overlay the lakes shapefile (make sure to filter or clip it to the U.S. bounds)
lakes_clipped.plot(ax=plt.gca(), color='none', edgecolor='blue')

plt.show()
