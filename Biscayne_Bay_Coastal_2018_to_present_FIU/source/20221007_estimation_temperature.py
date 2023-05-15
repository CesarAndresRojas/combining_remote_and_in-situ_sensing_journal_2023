import datetime
import json
import math
import os

from netCDF4 import Dataset
from shapely.geometry import Polygon, shape
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import rasterio
import rioxarray
import xarray

import geopandas as gpd
import helpers as hlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ['USE_PYGEOS'] = '0'

def extract_water_mask(bqa):
    # Initialize the water and cloud masks as arrays of zeros with the same shape as the BQA band
    water_mask = np.zeros(bqa.shape, dtype=bool)
    cloud_mask = np.zeros(bqa.shape, dtype=bool)
    
    # Loop through each pixel in the BQA band
    for i in range(bqa.shape[0]):
        bqa_value = bqa[i]
        # print(bqa_value)
        # Check if the first two bits of the BQA band indicate a clear water pixel
        if bqa_value[8] == '1':
            water_mask[i] = True
        # Check if the first two bits of the BQA band indicate a cloud or cloud shadow pixel
        elif bqa_value[12] == '1':
            cloud_mask[i] = True
    
    return (water_mask, cloud_mask)


f = '../data/level_3/pixEx_L8-9_OLI_100722_SURFACE_WATER_QUALITY_006-010_NetCDF_measurements.csv'
f2 = '../data/level_3/pixEx_L8-9_TIRS_100722_SURFACE_WATER_QUALITY_006-010_NetCDF_measurements.csv'

colocated_oli_df = pd.read_csv(f, sep=',', low_memory=False).dropna()
colocated_tirs_df = pd.read_csv(f2, sep=',', low_memory=False).dropna()
df = colocated_oli_df
df2 = colocated_tirs_df

# Cast the column containing the BQA data to binary bits
df['QA_PIXEL_binary'] = df['QA_PIXEL'].apply(lambda x: np.binary_repr(x, width=16))
bqa_list = df['QA_PIXEL_binary'].values

# Pass the BQA column of the dataframe to the function
(water_mask, cloud_mask) = extract_water_mask(bqa_list)

# Add the returned lists as new columns in the dataframe
df2['water_mask'] = water_mask
df2['cloud_mask'] = cloud_mask

col = 'Temp °C'
print('')
hlp.print_stats(df2, col)

ocean_surface_df = df2.loc[(df2['water_mask'] == 1) & (df2['Temp °C'] >= 0)]
cols = ['st10', 'st11']
x = np.array(ocean_surface_df[cols]).reshape(-1, 2)
y = np.array(ocean_surface_df['Temp °C'])

STD_regr = hlp.create_2d_linear_regression_model(x, y, True,['st10','st11'], 'In-situ Temp °C', 'Remote Sensing & In-situ Data Co-location', 'temp')

# Open the netCDF file
file_name = 'L8_OLI_2022_10_23_15_50_38_015042'
file_name2 = 'L8_TIRS_2022_10_23_15_50_38_015042'
xr = rioxarray.open_rasterio('../data/level_2/remote_sensing/landsat_8/' + file_name + '/' + file_name + '_L2R.nc', masked=True)
xr2 = rioxarray.open_rasterio('../data/level_2/remote_sensing/landsat_8/' + file_name + '/' + file_name2 + '_ST.nc', masked=True)

geojson_file_name = "../locations/Biscayne_Bay_Campus_Pier_Square_ESPG32617.json"

with open(geojson_file_name, "r") as f:
    geojson = json.load(f)
geometry = geojson["features"][0]["geometry"]

# Set the CRS on the xarray objects
# xr.rio.set_crs("EPSG:32617")
# xr2.rio.set_crs("EPSG:32617")

# Write the CRS information to the netCDF files
# xr.rio.write_crs()
# xr2.rio.write_crs()

# Set the spatial resolution to 10 meters
new_res = (10, 10)

# Resample the data to the new resolution
xr_resampled = xr.rio.reproject(xr.rio.crs, resolution=new_res)
xr2_resampled = xr2.rio.reproject(xr2.rio.crs, resolution=new_res)

# Use the clip method from rioxarray to extract the pixels within the polygon
cropped_xr = xr_resampled.rio.clip([geometry])
cropped_xr2 = xr2_resampled.rio.clip([geometry])

df = cropped_xr.to_dataframe()
df2 = cropped_xr2.to_dataframe()
xr.close()
xr2.close()
cropped_xr.close()
cropped_xr2.close()

# Cast the column containing the BQA data to binary bits
df['QA_PIXEL'] = df['QA_PIXEL'].astype(np.uint16)
df['QA_PIXEL_binary'] = df['QA_PIXEL'].apply(lambda x: np.binary_repr(x, width=16))
bqa_list = df['QA_PIXEL_binary'].values

# Pass the BQA column of the dataframe to the function
(water_mask, cloud_mask) = extract_water_mask(bqa_list)

# Add the returned lists as new columns in the dataframe
df2['water_mask'] = water_mask
df2['cloud_mask'] = cloud_mask
df2['STempD'] = STD_regr.predict(df2[cols].fillna('0', inplace=False).values.reshape(-1, 2))
df2.to_csv('../data/level_4/20221007_STempD.csv')

ocean_surface_df = df2.loc[(df2['water_mask'] == 1)]
print('Maximum Value:')
print(ocean_surface_df['STempD'].max())
print('Minimum Value:')
print(ocean_surface_df['STempD'].min())
hlp.plot_2d_scatter_with_colorbar(ocean_surface_df['lon'], ocean_surface_df['lat'], ocean_surface_df['STempD'], 'Temperature (°C)', 'Satellite Derived Temperature', 'autumn_r','temp', 58)
