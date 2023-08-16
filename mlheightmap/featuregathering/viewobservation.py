import numpy as np
import matplotlib.pyplot as plt

def open_and_display_observation(observation_path='observation.npz'):
    # Load the observation from the .npz file
    observation_data = np.load(observation_path)

    # Extract the features and label
    E_mean = observation_data['E_mean']
    E_stdev = observation_data['E_stdev']
    mat = observation_data['mat']
    map_mm = observation_data['map_mm']
    rugosity = observation_data['rugosity']  # Extract the rugosity value
    elevation_data = observation_data['observation']

    # Display the elevation data as an image
    plt.imshow(elevation_data, cmap='gray')
    plt.colorbar(label='Elevation')
    plt.title('Elevation Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Display the features as text on the plot
    feature_text = f"Mean Elevation: {E_mean}\nStandard Deviation of Elevation: {E_stdev}\nMean Annual Temperature: {mat} K\nMean Annual Precipitation: {map_mm} mm\nRugosity: {rugosity}"  # Include rugosity in the text
    plt.text(0, 0, feature_text, fontsize=9, color='white', backgroundcolor='black', verticalalignment='top')

    plt.show()

# Example usage:
open_and_display_observation('featuregathering/observations/observation2.npz')
