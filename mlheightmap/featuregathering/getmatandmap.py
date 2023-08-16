import requests
import numpy as np

def get_climate_data(latitude, longitude):
    # URL for the Open-Meteo API (you may need to adjust the endpoint and parameters)
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date=2022-01-01&end_date=2023-01-01&hourly=temperature_2m,precipitation"

    # Make the API request
    response = requests.get(url)
    data = response.json()

    # Extract the temperature and precipitation data
    temperatures = np.array(data['hourly']['temperature_2m'])
    precipitation = np.array(data['hourly']['precipitation'])

    # Calculate the mean annual temperature (in Celsius)
    mat_kelvin = temperatures.mean() + 273.15

    # Calculate the mean annual precipitation (in mm)
    map_mm = precipitation.sum()

    return mat_kelvin, map_mm

latitude = 40.730610
longitude = -73.935242
mat, map = get_climate_data(latitude, longitude)

print(f"Mean Annual Temperature (MAT): {mat} K")
print(f"Mean Annual Precipitation (MAP): {map} mm")
