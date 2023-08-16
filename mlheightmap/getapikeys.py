import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('mlheightmap/sensitivedata.csv')

def get_gmaps_key():
    # Find the row where the key_column_name equals the key you're looking for
    key_to_search = "googlemapsapikey"
    row = df[df['name'] == key_to_search]

    # Retrieve the value from the value_column_name
    value = row['data'].values[0]
    
    print(f"The value for key {key_to_search} is {value}")

    return value;
    
def get_gpxz_key():
    # Find the row where the key_column_name equals the key you're looking for
    key_to_search = "gpxzkey"
    row = df[df['name'] == key_to_search]

    # Retrieve the value from the value_column_name
    value = row['data'].values[0]

    print(f"The value for key {key_to_search} is {value}")
    
    return value
    
get_gmaps_key()
get_gpxz_key()

