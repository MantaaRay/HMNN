import googlemaps
import requests
from random_points import generate_random_points_on_land
from random_points import generate_random_points_on_land_within_radius
from shapely.geometry import Point
import polyline
from getapikeys import get_gmaps_key

def get_elevation_data(points, api_key="YOUR_API_KEY"):
    # gmaps = googlemaps.Client(key=api_key)
    locations = [(point.y, point.x) for point in points]
    encoded = polyline.encode(locations)
    locations_encoded = "enc:" + encoded

    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations=enc:{encoded}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        elevation_data = response.json()
        return elevation_data['results']
    else:
        print(f"Error fetching elevation data: {response.text}")
        return None
    return elevation_data

# Example usage:
random_points_on_land = generate_random_points_on_land_within_radius(100, 36.7783, -119.4179, .01, "land10m.shp")
elevation_data = get_elevation_data(random_points_on_land, api_key=get_gmaps_key())
# Sort by resolution in descending order
sorted_elevation_data = sorted(elevation_data, key=lambda x: x['resolution'])

# Print the top fifty highest resolution points
for result in sorted_elevation_data[:50]:
    print(f"Location: {result['location']}, Elevation: {result['elevation']}, Resolution: {result['resolution']}")

# encoded = polyline.encode([(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)])
# decoded = polyline.decode(encoded)
# print(f"Encoded: {encoded}, decoded: {decoded}")