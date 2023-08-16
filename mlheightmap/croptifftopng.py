from PIL import Image
import rasterio
import numpy as np

def crop_tiff(tiff_path, size=(4000, 4000)):
    with rasterio.open(tiff_path) as src:
        # Get the nodata value for the TIFF file
        nodata_value = src.nodatavals[0]

        # Pick a random starting point for the crop
        start_x = np.random.randint(0, src.width - size[0])
        start_y = np.random.randint(0, src.height - size[1])

        # Read the specified window
        window = rasterio.windows.Window(start_x, start_y, size[0], size[1])
        cropped_data = src.read(1, window=window)

        # Check for nodata values and retry if found
        if (cropped_data == nodata_value).any():
            return crop_tiff(tiff_path, size)

        # Get the bounds of the cropped region
        left, bottom, right, top = rasterio.windows.bounds(window, src.transform)

        # Get the center coordinates of the cropped region
        # latitude, longitude = get_center_coords(left, bottom, right, top, src.crs)

        return cropped_data

def save_as_png(tiff_path, png_path, size=(4000, 4000)):
    # Crop the TIFF file using the provided function
    cropped_data = crop_tiff(tiff_path, size)

    # Normalize the cropped data to the range [0, 255]
    normalized_data = ((cropped_data - cropped_data.min()) / (cropped_data.max() - cropped_data.min()) * 255).astype(np.uint8)

    # Create a PIL Image object from the normalized data
    image = Image.fromarray(normalized_data)

    # Save the image as a PNG file
    image.save(png_path)

    print(f"Saved cropped and normalized image to {png_path}")

# Example usage:
tiff_path = 'heightmaps/testing/arra10.tif'
png_path = 'croppedtiffarra10.png'
save_as_png(tiff_path, png_path)
