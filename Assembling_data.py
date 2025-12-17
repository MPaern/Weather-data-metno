"""
Creator: Mari Eldegard Heie
Date: August 15th 2025
Purpose:
This file concats several batches of weather data from different files into a
single dataframe with weather data. Make sure the filepaths match the filenames
of your stored batches of data.

Note that this files does not make any changes to the data, it only concats
the data. If there is overlapping data in the loaded files, this overlap
will also be in the output file. For example, if two of the files to be
combined both contain weather data for aug 8th 2024, the output file will have
weather data for aug 8th 2024 repeated twice. Thus, you may want to make sure
that there is no overlap in the datasets before executing this script.
"""

import pandas as pd

file_paths = [f"C:/Users/Maris/PycharmProjects/PythonProject/.venv/Scripts/data/WeatherData{n}.csv" for n in range(1, 10)] #range 1 bigger than actual number

dfs = []
for file_path in file_paths:
    df = pd.read_csv(file_path)
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)

# group data by location
if "name" in all_data.columns:
    all_data = all_data.sort_values(
        by=["name", "year", "month", "day", "hour"]
        )

# Load data (overwrites "data/WeatherData.csv")
all_data.to_csv("data/WeatherData.csv")
