import requests
from getapikeys import get_gpxz_key


API_KEY = get_gpxz_key()

response = requests.get(
    "https://api.gpxz.io/v1/elevation/point?lat=40.4&lon=-122.7",
    headers={"x-api-key": API_KEY},
)
print(response.json())