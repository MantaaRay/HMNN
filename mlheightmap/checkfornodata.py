import rasterio
import numpy as np

def check_nodata_in_tiff(tiff_path):
    with rasterio.open(tiff_path) as src:
        # Read the NoData value from the file's metadata
        nodata_value = src.nodata
        
        # Read the entire file
        data = src.read()
        
        if nodata_value is not None:
            # Check if the data contains the NoData value
            contains_nodata = np.any(data == nodata_value)
            print(f"The TIFF file at {tiff_path} {'contains' if contains_nodata else 'does not contain'} NoData pixels.")
        else:
            print(f"NoData value is not defined for the TIFF file at {tiff_path}.")

tiff_file_path = 'mlheightmap/featuregathering/tiffs/testing/Alaska_1.tif'  # Change this to the path of your TIFF file
check_nodata_in_tiff(tiff_file_path)