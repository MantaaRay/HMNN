import os
import shutil
import time
import rasterio
import matplotlib.pyplot as plt
import numpy as np


def get_folder_size(folder_path):
    return sum(os.path.getsize(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)))


def delete_files(folder_path, extension, target_size_gb=2):
    # Delete all files with the specified extension
    for filename in os.listdir(folder_path):
        if filename.endswith(extension):
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Deleted {file_path}")

    # Check folder size and delete .img files if needed
    target_size_bytes = target_size_gb * 1024 * 1024 * 1024
    img_files = [f for f in os.listdir(folder_path) if f.endswith('.img')]

    for file_to_delete in img_files:
        file_path = os.path.join(folder_path, file_to_delete)

        # Open the .img file and check for anomalies
        with rasterio.open(file_path) as src:
            # Read a subset of the data (e.g., 10x10 window)
            subset = src.read(1, window=rasterio.windows.Window(0, 0, 10, 10))
            unique_values = np.unique(subset[subset != src.nodata]) # Exclude NoData values

            # If all non-NoData values are the same, delete the file
            if len(unique_values) == 1:
                os.remove(file_path)
                print(f"Deleted anomaly {file_path}")
                continue

        # If the file is not an anomaly, delete it only if the folder size exceeds the target
        if get_folder_size(folder_path) > target_size_bytes:
            os.remove(file_path)
            print(f"Deleted {file_path}")

    print("All files processed successfully.")

        
def display_img(img_path):
    # Open the .img file using rasterio
    with rasterio.open(img_path) as src:
        # Read the first band (assuming the data you want is in the first band)
        img_data = src.read(1)

        # Display the data using matplotlib
        plt.imshow(img_data, cmap='gray') # You can change the colormap as needed
        plt.colorbar(label='Value')
        plt.title('IMG File Visualization')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

# img_path =r"C:\Users\gathr\OneDrive\Desktop\downloladign bulk opentopogarphy data\data\TAHOE_hh\hh_38120G12.img" # Replace with the path to your .img file
# display_img(img_path)


def convert_imgs_to_tiffs(img_folder):
    # Create a folder called 'tiffs' inside the given folder
    tiff_folder = os.path.join(img_folder, 'tiffs')
    os.makedirs(tiff_folder, exist_ok=True)

    # Iterate through all the .img files in the given folder
    for filename in os.listdir(img_folder):
        if filename.endswith('.img'):
            img_path = os.path.join(img_folder, filename)
            
            # Open the .img file
            with rasterio.open(img_path) as src:
                # Read the data and metadata
                img_data = src.read()
                meta = src.meta

                # Update the metadata to reflect the new file format
                meta.update(driver='GTiff')

                # Create the corresponding .tif filename
                tiff_filename = os.path.splitext(filename)[0] + '.tif'
                tiff_path = os.path.join(tiff_folder, tiff_filename)

                # Write the .tif file with the same data and metadata
                with rasterio.open(tiff_path, 'w', **meta) as dst:
                    dst.write(img_data)

                print(f"Converted {filename} to {tiff_filename}")

    print("All files converted successfully.")
    
def delete_anomalies(img_folder):
    # Iterate through all the .img files in the given folder
    for filename in os.listdir(img_folder):
        if filename.endswith('.img'):
            img_path = os.path.join(img_folder, filename)

            # Open the .img file and check for anomalies
            with rasterio.open(img_path) as src:
                # Read a subset of the data (e.g., 10x10 window)
                subset = src.read(1, window=rasterio.windows.Window(0, 0, 10, 10))
                unique_values = np.unique(subset[subset != src.nodata]) # Exclude NoData values

                # If all non-NoData values are the same, delete the file
                if len(unique_values) == 1:
                    src.close()  # Explicitly close the file
                    time.sleep(1)  # Wait for 1 second
                    os.remove(img_path)
                    print(f"Deleted anomaly {filename}")

    print("Anomalies deleted successfully.")

img_folder = 'path/to/your/folder' # Replace with the path to your folder of .img files
# convert_imgs_to_tiffs(img_folder)


folder_path = r"C:\Users\gathr\OneDrive\Desktop\downloladign bulk opentopogarphy data\data\tahoe basin files"  # Replace with the path to your folder
delete_anomalies(folder_path)
convert_imgs_to_tiffs(folder_path)