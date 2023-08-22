import glob
import os
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
    
def open_and_display_observations(folder_path):
    # Get all the .npz files in the folder
    observation_files = glob.glob(os.path.join(folder_path, '*.npz'))
    
    observation_files = observation_files[:9]  # Limit the number of observations to 9

    # Determine the number of rows and columns for the subplot layout
    num_files = len(observation_files)
    num_cols = int(np.ceil(np.sqrt(num_files)))
    num_rows = int(np.ceil(num_files / num_cols))

    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Flatten the axes array if there's only one row
    if num_rows == 1:
        axes = axes.flatten()

    # Iterate through the observation files and plot each one
    for i, observation_path in enumerate(observation_files):
        # Load the observation from the .npz file
        try:
            observation_data = np.load(observation_path)
            elevation_data = observation_data['observation']
            E_mean = observation_data['E_mean']
            E_stdev = observation_data['E_stdev']
            mat = observation_data['mat']
            map_mm = observation_data['map_mm']
            roughness = observation_data['roughness']
        except:
            print("Failed to load observation data from " + observation_path)
            return

        # Get the current subplot axis
        ax = axes[i // num_cols, i % num_cols] if num_rows > 1 else axes[i]

        # Display the elevation data as an image
        im = ax.imshow(elevation_data, cmap='gray')
        ax.set_title(os.path.basename(observation_path))

        # Add a colorbar to the current subplot
        plt.colorbar(im, ax=ax, label='Elevation')

        # Add the feature text to the current subplot
        feature_text = f"Mean Elevation: {E_mean}\nStandard Deviation of Elevation: {E_stdev}\nMean Annual Temperature: {mat} K\nMean Annual Precipitation: {map_mm} mm\nRoughness: {roughness}"
        ax.text(0.5, 1.15, feature_text, fontsize=9, horizontalalignment='center', transform=ax.transAxes, backgroundcolor='white')

    plt.tight_layout()

    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_random_observations(observations):
    # Select 10 random observations
    selected_observations = random.sample(observations, 10)

    # Set up a grid for plotting
    fig, axes = plt.subplots(5, 2, figsize=(12, 24))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate through the selected observations and axes
    for observation, ax in zip(selected_observations, axes):
        # Extract the features and target
        features = observation['features']
        target = observation['target']

        # Create a title with the features
        feature_str = '\n'.join([f'{key}: {value:.2f}' for key, value in features.items()])
        ax.set_title(feature_str)

        # Plot the heightmap
        im = ax.imshow(target, cmap='terrain')

        # Add a colorbar to show elevation levels
        plt.colorbar(im, ax=ax)

    # Adjust the layout
    plt.tight_layout()
    plt.show()


# Example usage:
# open_and_display_observation('featuregathering/observations/observation2.npz')
observations_folder_path = 'mlheightmap/featuregathering/observations/'
# open_and_display_observations(observations_folder_path)
with np.load('mlheightmap/featuregathering/observations/observations.npz', allow_pickle=True) as data:
    observations = data['observations'].tolist()
    plot_random_observations(observations)

