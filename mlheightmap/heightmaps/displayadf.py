from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

# Path to the TIFF file
file_path = 'heightmaps/testing/ca13_guo_test.tif'

# Open the file as a GDAL dataset
dataset = gdal.Open(file_path)

# Get the raster band (assuming a single-band raster)
band = dataset.GetRasterBand(1)

# Get the NoData value from the band
nodata_value = band.GetNoDataValue()

# Read the data into a NumPy array
array = band.ReadAsArray()

# Close the dataset
dataset = None

# Create a mask for NoData values
mask = array != nodata_value

# Apply the mask to create a masked array
masked_array = np.ma.array(array, mask=~mask)

# Print some statistics to understand the data
print("Min:", masked_array.min())
print("Max:", masked_array.max())
print("Mean:", masked_array.mean())
print("Std Dev:", masked_array.std())

# Normalize the data to the range [0, 1] for visualization
normalized_array = (masked_array - masked_array.min()) / (masked_array.max() - masked_array.min())

# Plot the normalized data
plt.imshow(normalized_array, cmap='gray')
plt.colorbar()
plt.show()
