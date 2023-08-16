from elevation import get_elevation_data, plot_elevation_data
from plotting import ElevationPlotter
from matplotlib.widgets import Button, Slider
from random_points import generate_random_points_on_land_within_radius
from random_points import generate_random_points_on_land
# from plotting import plot_land_and_points, toggle_view, update_radius, update_view
import geopandas as gpd
import matplotlib.pyplot as plt
from getapikeys import get_gmaps_key

# Parameters
api_key = get_gmaps_key()
# center_lat = 36.7783
# center_lon = -119.4179

# # Read the shapefile that represents land
# shapefile_path = 'land10m.shp'
# gdf_land = gpd.read_file(shapefile_path)

# # Generate random points on land
# random_points_on_land = generate_random_points_on_land_within_radius(100, center_lat, center_lon, 1, shapefile_path)

# # Create the plot
# fig, ax = plt.subplots()
# plot_land_and_points(gdf_land, random_points_on_land, ax)

# Create a button to toggle the view
shapefile_path = 'land10m.shp'
center_lat = 36.7783
center_lon = -119.4179
gdf_land = gpd.read_file(shapefile_path)
num_points = 512
# random_points_on_land_within_radius = generate_random_points_on_land_within_radius(num_points, center_lat, center_lon, 1, shapefile_path)
# random_points_on_land = generate_random_points_on_land(num_points, shapefile_path)

# plotter = ElevationPlotter(gdf_land, random_points_on_land, center_lat, center_lon)

resolution = 30
size = 20

elevation_data = get_elevation_data(api_key, center_lat, center_lon, resolution, size)
plot_elevation_data(elevation_data)
