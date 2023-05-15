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

f = '../data/level_3/pixEx_S2A-B_MSI_100722_SURFACE_WATER_QUALITY_006-010_NetCDF_measurements.csv'
colocated_depth_df = pd.read_csv(f, sep=',', low_memory=False).dropna()
col = 'ODO mg/L'
df = colocated_depth_df
print('')
hlp.print_stats(df, col)

df['water_mask'] = np.where(df['rhos_833'] > 0.05, 0, 1)
# Calculate Sentinel Water Mask:
# swm = (df['B02'] + df['B03']) / (df['B08'] + df['B11']) #Sentinel-2
swm = (df['rhos_492'] + df['rhos_560']) / (df['rhos_833'] + df['rhos_1614']) #S2A
df['SWM'] = swm

ocean_surface_df = df.loc[(df['water_mask'] == 1) & (df['ODO mg/L'] >= 0)]
x = np.array(ocean_surface_df['SWM']).reshape(-1, 1)
y = np.array(ocean_surface_df['ODO mg/L'])

SDoD_regr = hlp.create_linear_regression_model(x, y, min(x), max(x), True,'Sentinel Water Mask', 'In-situ ODO mg/L', 'Remote Sensing & In-situ Data Co-location', 'do')

# Open the netCDF file
file_name = 'S2B_MSI_2022_10_22_16_06_12_T17RNJ'
xr = rioxarray.open_rasterio('../data/level_2/remote_sensing/sentinel_2/' + file_name + '/' + file_name + '_L2R.nc', masked=True)

geojson_file_name = "../locations/Biscayne_Bay_Campus_Pier_Square_ESPG32617.json"

with open(geojson_file_name, "r") as f:
    geojson = json.load(f)
geometry = geojson["features"][0]["geometry"]

# Use the clip method from rioxarray to extract the pixels within the polygon
cropped_xr = xr.rio.clip([geometry])

xr.close()
df = cropped_xr.to_dataframe()
cropped_xr.close()

df['water_mask'] = np.where(df['rhos_833'] > 0.05, 0, 1)

# Calculate SWM
swm = (df['rhos_492'] + df['rhos_559']) / (df['rhos_833'] + df['rhos_1610']) #S2B
df['SWM'] = swm
df['SDoD'] = SDoD_regr.predict(df['SWM'].values.reshape(-1, 1))
df.to_csv('../data/level_4/20221007_SDoD.csv')

ocean_surface_df = df.loc[(df['water_mask'] == 1)]
print('Maximum Value:')
print(ocean_surface_df['SDoD'].max())
print('Minimum Value:')
print(ocean_surface_df['SDoD'].min())

hlp.plot_2d_scatter_with_colorbar(ocean_surface_df['lon'], ocean_surface_df['lat'], ocean_surface_df['SDoD'], 'ODO (mg/L)', 'Satellite Derived Dissolved Oxygen', 'winter_r', 'do')
