import tkinter as tk
import numpy as np
from selenium import webdriver
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
from selenium.webdriver.common.by import By


# Initialize Selenium WebDriver
driver = webdriver.Chrome()
driver.get("https://portal.opentopography.org/datasets?")

# Define functions for zooming and centering
def zoom_in():
    js_code = "map.getView().setZoom(map.getView().getZoom() + 1);"
    driver.execute_script(js_code)

def zoom_out():
    js_code = "map.getView().setZoom(map.getView().getZoom() - 1);"
    driver.execute_script(js_code)

def get_center():
    js_code = "return ol.proj.transform(map.getView().getCenter(), 'EPSG:3857', 'EPSG:4326');"
    center = driver.execute_script(js_code)
    center_label.config(text="Center: " + str(center))

def set_center():
    longitude = float(longitude_entry.get())
    latitude = float(latitude_entry.get())
    js_code = f"map.getView().setCenter(ol.proj.transform([{longitude}, {latitude}], 'EPSG:4326', 'EPSG:3857'));"
    driver.execute_script(js_code)

def view_map():
    # Find the element with ID "map"
    map_element = driver.find_element(By.ID, 'map')

    # Take a screenshot of the map element and save it to a file
    map_element.screenshot('map.png')

    # Open the image with PIL
    image = Image.open('map.png')

    # Define the cropping dimensions
    left = 150
    top = 75
    right = image.width - 300
    bottom = image.height - 50

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    # Resize the image to lower resolution (you can adjust the size as needed)
    resized_image = cropped_image.resize((int(cropped_image.width / 2), int(cropped_image.height / 2)))

    # Convert the image to an array for processing
    image_array = np.array(resized_image)

    # Define the red value threshold (you can adjust this value as needed)
    red_threshold = 225

    # Create a mask where the red values are greater than the threshold
    red_mask = image_array[:,:,0] > red_threshold

    # Highlight the pixels where the red values are greater than the threshold (e.g., set them to blue)
    image_array[red_mask, 0] = 0
    image_array[red_mask, 1] = 0
    image_array[red_mask, 2] = 255

    # Create a PIL image from the modified array
    processed_image = Image.fromarray(image_array)

    # Display the processed image with Matplotlib
    plt.imshow(processed_image)
    plt.axis('off') # To turn off axes
    plt.show()

# Create GUI
root = tk.Tk()
root.title("OpenTopography Control")

# Zoom buttons
zoom_in_button = tk.Button(root, text="Zoom In", command=zoom_in)
zoom_in_button.pack()
zoom_out_button = tk.Button(root, text="Zoom Out", command=zoom_out)
zoom_out_button.pack()

# Get center button
center_label = tk.Label(root, text="Center:")
center_label.pack()
get_center_button = tk.Button(root, text="Get Center", command=get_center)
get_center_button.pack()

# Set center inputs and button
longitude_label = tk.Label(root, text="Longitude:")
longitude_label.pack()
longitude_entry = tk.Entry(root)
longitude_entry.pack()
latitude_label = tk.Label(root, text="Latitude:")
latitude_label.pack()
latitude_entry = tk.Entry(root)
latitude_entry.pack()
set_center_button = tk.Button(root, text="Set Center", command=set_center)
set_center_button.pack()

# View map button
view_map_button = tk.Button(root, text="View Map", command=view_map)
view_map_button.pack()

root.mainloop()
