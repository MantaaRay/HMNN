import rasterio
from pyproj import Transformer

def get_center_coords(tiff_path):
    with rasterio.open(tiff_path) as dataset:
        # Get the bounds of the TIFF file
        left = dataset.bounds[0]
        bottom = dataset.bounds[1]
        right = dataset.bounds[2]
        top = dataset.bounds[3]

        # Calculate the center coordinates
        center_x_coord = (left + right) / 2
        center_y_coord = (top + bottom) / 2

        # Get the CRS of the TIFF file
        tiff_crs = dataset.crs

        # Define a transformer from the TIFF's CRS to WGS 84
        transformer = Transformer.from_crs(tiff_crs, 'EPSG:4326')

        # Transform the coordinates
        latitude, longitude = transformer.transform(center_x_coord, center_y_coord)

        return latitude, longitude

# Example usage:
tiff_path = 'heightmaps/testing/ca13_guo_test.tif'
center_coords = get_center_coords(tiff_path)
print(f"Center coordinates of the TIFF file (latitude, longitude): {center_coords}")
