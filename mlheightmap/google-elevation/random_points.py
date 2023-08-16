from concurrent.futures import ThreadPoolExecutor
import random
import time
import geopandas as gpd
from shapely.geometry import Point
import math

def is_point_on_land_within_radius(point, gdf_land, spatial_index):
    possible_matches_index = list(spatial_index.intersection(point.bounds))
    possible_matches = gdf_land.iloc[possible_matches_index]
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
    gdf_land = gpd.read_file(shapefile_path)
    gdf_land['geometry'] = gdf_land['geometry'].simplify(tolerance=0.01)
    spatial_index = gdf_land.sindex
    center_point = Point(center_lon, center_lat)
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
    possible_matches_index = list(spatial_index.intersection(point.bounds))
    possible_matches = gdf_land.iloc[possible_matches_index]
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
    gdf_land = gpd.read_file(shapefile_path)
    gdf_land['geometry'] = gdf_land['geometry'].simplify(tolerance=0.01)
    spatial_index = gdf_land.sindex
    bounds = gdf_land.total_bounds
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
