# Tried to change the projection of the coordinates, did not succeed

'''
Creator: Mari Eldegard Heie
Date: August 15th 2025
Purpose: Retrieve weather data from the archive of
Meterologisk institutt.

The user provides a set of locations (specified in latitude and longitude),
in a .csv file and a range of dates to retrieve hourly data for.

The data retrieved is from the closest observation points in Metrologisk
institutt's dataset, which has a resolution of 1km. The retrieved data includes
the exact location which the data was retrieved from (closest latitude and
longitude to the latitude and longitude if the location specified by the user).

NOTE: Due to an error in xarray, you may have to run this files several times,
for a few dates at a time, because the program becomes very slow after
opening many data files. Running this file several times, for a few timepoints
at a time, avoids this issue. To load the data in batches, adjust the start and
end dates as below "User Input" as well as the batch number. Eventually you
will have many data files named WeatherDataX that you may combine to a single
dataset using concat_batched_data.py.

NOTE: Sometimes, weather data might be missing for a single timepoint in the
archive of Metrologisk Institutt. Running this script will create a file
MissingWeatherDataX (for each batch X) telling you which timepoints, if any,
are missing data.

NOTE: If there is any file named WeatherData.csv in a subfolder "data" in of
the directory of this file,running this file will overwrite that file.
'''

import sys
import time
from datetime import date, timedelta

import geopandas as gpd
import pandas as pd
import numpy as np

import xarray as xr
from pyproj import CRS

# USER INPUT: Path to .csv file containing latitude and longitude for locations to retrieve weather data for
# The file should contain one column describing the locations' latitude (labeled Latitude),
# and one column describing the locations' longitude (labeled Longitude),
# If the locations also have a column labeled "Name", this will be inluded in the output data
filepath = "C:/Users/Maris/Documents/data/Coordinates 2024 CM.csv"
df = pd.read_csv(filepath)


if "Latitude" not in df.columns or "Longitude" not in df.columns:
    print("Input file is missing required column 'Latitude'.")
    sys.exit()

#ADDITION
# Create GeoDataFrame assuming WGS84
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")

# Define original CRS (Lambert Conformal Conic)
lcc_crs = CRS.from_proj4("+proj=lcc +lat_0=63 +lon_0=15 +lat_1=63 +lat_2=63 +no_defs +R=6.371e+06")

#print(gdf)
#print(gdf.crs)

# 3. Transform to LCC CRS
gdf_lcc = gdf.to_crs(lcc_crs)
#print(gdf_lcc)



# Extrac new latitude and longitude to run extract data
gdf_lcc['Longitude'] = gdf_lcc.geometry.x
gdf_lcc['Latitude'] = gdf_lcc.geometry.y
#gdf_lcc['Longitude']
#print(gdf_lcc['Name'])

new_df = {
    'Longitude':[gdf_lcc['Longitude']],
    'Latitude': [gdf_lcc['Latitude']],
    'Name': [gdf_lcc['Name']] }
new_df2 = pd.DataFrame(new_df) #what does this do? This makes a wrong shaped df
new_df = gdf_lcc[['Longitude', 'Latitude', 'Name']].copy()
#print(new_df)

df = new_df
#print (df)

#FROM HERE OLD CODE
# USER INPUT: Range of dates to retrieve data for
start_date = date(2024, 6, 14)  # First date: 2024 May 31st
end_date = date(2024, 6, 21)  # 2024 November 2nd (last included date November 1st)
n_batch = 1  # Increase for with each run to avoid overwriting previously loaded batch
print(f"Fetching weather data from {start_date} to {end_date}")


# Define functions for later use
def daterange(start_date: date, end_date: date):
    """Retrun all dates we want to to retrieve data for."""
    n_days = int((end_date - start_date).days)
    for n in range(n_days):
        yield start_date + timedelta(n)


def find_file(
        year: int = 2024,
        month: int = 1,
        day: int = 1,
        hour: int = 0
) -> str:
    """
    Return file from MET Nordic archive for a given time.
    Adjust filepath to select which archive to retrieve data from (may change
    over time).
    """
    _date = f"{year}/{month:02d}/{day:02d}"
    timestamp = f"{year}{month:02d}{day:02d}T{hour:02d}Z"
    datapath = f"https://thredds.met.no/thredds/dodsC/metpparchive/{_date}/met_analysis_1_0km_nordic_{timestamp}.nc"
    return datapath

def find_index_of_closest_datapoint(
        location_index: int,
        lat: np.ndarray[float],
        lon: np.ndarray[float],
) -> tuple[int]:
    """
    Find the index of the weather datapoint closest to our target location
    using euclidian distance.
    """
    target = [df["Latitude"][location_index], df["Longitude"][location_index]]

    lat_lon_grid = np.stack((lat, lon), axis=-1)  # combine lat/lon grids to 3D grid
    points = lat_lon_grid.reshape(-1, 2)  # Reshape to 2D (flatten the last axis)

    # Calculate squared distances to avoid computing square root (it preserves the order of distances)
    squared_distances = np.sum((points - target) ** 2, axis=1)

    # Find the index of the minimum distance
    closest_index = np.argmin(squared_distances)

    # Convert the index back to the original grid coordinates (m, n)
    closest_point = points[closest_index]

    # Find index of closest point in the longitude (and latitude) grid
    # NOTE: a possible error that might occur here is that multiple points in
    # the grid may have the same longitude, causing more than one point to be
    # returned. This sometimes happens when using the latitude instead of the
    # longitude grid (but this issue have not been observed when using 'lon').
    indices = np.where(lon == closest_point[1])

    return int(indices[0][0]), int(indices[1][0])



# get every filepath for every date and every hour from 5pm to 6am
dates = [
    date_.strftime("%Y/%m/%d")
    for date_ in daterange(start_date, end_date)
]
hours = [i for i in range(0, 7)] + [e for e in range(18, 24)]

all_ncfiles = [
    (
        find_file(int(_date[:4]), int(_date[5:7]), int(_date[8:10]), hour),
        int(_date[:4]),
        int(_date[5:7]),
        int(_date[8:10]),
        hour
    )
    for _date in dates for hour in hours
]

if int((end_date - start_date).days) * 24 != len(all_ncfiles):
    print("Warning: number of files retrieved does not match number of hours \
          between start/end date.")

# find the (2d) index of the closest data point to each of our locations
# OBS: assumption: lat/lon grids are constant, i.e. the weather observation
# points are constant
try:
    nc = xr.open_dataset(all_ncfiles[0][0], engine="netcdf4")  # open first file
except OSError:
    print()
    print(f"Could not load {all_ncfiles[0][0]}.")
    print("Check if xarray is installed, and whether the file exsists.")
    print("You may find information on which files are accessible at \
          https://github.com/metno/NWPdocs/wiki/MET-Nordic-dataset. \n")
    sys.exit()

lat = np.asarray(nc.latitude)  # grid
lon = np.asarray(nc.longitude)  # grid

if np.shape(lat) != np.shape(lon):
    print("WARNING: Grids for latitude and longitude does not match.")

closest_datapoints = [
    find_index_of_closest_datapoint(location_index, lat, lon)
    for location_index in range(df.shape[0])
]

# Create dataframe to store weather data for the given locations
# See https://github.com/metno/NWPdocs/wiki/MET-Nordic-dataset for variable information.
variables = [
    'name',
    'location_latitude',
    'location_longitude',
    'nearest_latitude',
    'nearest_longitude',
    'year',
    'month',
    'day',
    'hour',
    'air_temperature_2m',
    'air_pressure_at_sea_level',
    'cloud_area_fraction',
    'relative_humidity_2m',
    'precipitation_amount',
    'wind_speed_10m',
    'wind_direction_10m',
]
if "Name" in df.columns:
    data = pd.DataFrame(columns=variables)
else:
    data = pd.DataFrame(columns=variables[1:])

# Store the dates with missing data
missing_data = []
n_missing_data = 0

# Run through all locations in all retrieved files to get data for every location at every hour
for time_index in range(len(all_ncfiles)):
    start_time = time.time()

    year = all_ncfiles[time_index][1]
    month = all_ncfiles[time_index][2]
    day = all_ncfiles[time_index][3]
    hour = all_ncfiles[time_index][4]

    try:
        nc = xr.open_dataset(all_ncfiles[time_index][0])
        print(f"Fetching data from {all_ncfiles[time_index][0]}.")

        for location_index, closest_datapoint in enumerate(closest_datapoints):
            name = df["Name"][location_index]
            location_latitude = df["Latitude"][location_index]
            location_longitude = df["Longitude"][location_index]

            nearest_latitude = lat[closest_datapoint]
            nearest_longitude = lon[closest_datapoint]
            air_temperature_2m = nc.air_temperature_2m.values[0][closest_datapoint] - 272.15
            air_pressure_at_sea_level = nc.air_pressure_at_sea_level.values[0][closest_datapoint]
            cloud_area_fraction = nc.cloud_area_fraction.values[0][closest_datapoint]
            relative_humidity_2m = nc.relative_humidity_2m.values[0][closest_datapoint]
            precipitation_amount = nc.precipitation_amount.values[0][closest_datapoint]
            wind_speed_10m = nc.wind_speed_10m.values[0][closest_datapoint]
            wind_direction_10m = nc.wind_direction_10m.values[0][closest_datapoint]

            new_row = [
                name,
                location_latitude,
                location_longitude,
                nearest_latitude,
                nearest_longitude,
                year,
                month,
                day,
                hour,
                air_temperature_2m,
                air_pressure_at_sea_level,
                cloud_area_fraction,
                relative_humidity_2m,
                precipitation_amount,
                wind_speed_10m,
                wind_direction_10m
            ]

            data.loc[len(data)] = new_row

        time_spent = time.time() - start_time
        print(f"Data retrieved in {time_spent:.4f} seconds.")
        print(f"Estimated remaining time: {time_spent * (len(all_ncfiles) - time_index) / 3600:.1f} hours.")

        # Load data that's fetched so far (overwrites "data/WeatherData.csv")
        data.to_csv(f"data/WeatherData{n_batch}.csv")

    except:
        print(f"Warning: Could not load {all_ncfiles[time_index][0]}.")
        missing_data.append(f"{year}/{month}/{day}/{hour}")
        n_missing_data += 1
        continue

with open(f"data/MissingWeatherData{n_batch}", "w") as outfile:
    outfile.write(f"Missing data for {n_missing_data} timepoints:\n")
    for missing in missing_data:
        outfile.write(missing)

### NOTE: If dividing the data into batches, do not use the commented lines,
###       rather, you should concat the your batches, making sure there is no
###       overlapping data before adding all the data to a single file
# group data by location
# if "name" in data.columns:
#     data = data.sort_values(by="name")

# Load data (overwrites "data/WeatherData.csv")
# data.to_csv("data/WeatherData.csv")
