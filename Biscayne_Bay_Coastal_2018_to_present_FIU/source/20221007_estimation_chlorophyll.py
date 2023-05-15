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
colocated_chlorophyll_df = pd.read_csv(f, sep=',', low_memory=False).dropna()
# colocated_chlorophyll_df = colocated_chlorophyll_df[colocated_chlorophyll_df['Chlorophyll ug/L'] > 0]
col = 'Chlorophyll ug/L'
df = colocated_chlorophyll_df
print('')
hlp.print_stats(df, col)


df['water_mask'] = np.where(df['rhos_833'] > 0.05, 0, 1)
df['NDCI'] = (df['rhos_704'] - df['rhos_665']) / (df['rhos_704'] + df['rhos_665'])

ocean_surface_df = df.loc[(df['water_mask'] == 1) & (df['Chlorophyll ug/L'] >= 0)]
x = np.array(ocean_surface_df['NDCI']).reshape(-1, 1)
y = np.array(ocean_surface_df['Chlorophyll ug/L'])

SCD_regr = hlp.create_linear_regression_model(x, y, min(x), max(x), True,'Normalized Difference Chlorophyll Index', 'In-situ Chlorophyll ug/L', 'Remote Sensing & In-situ Data Co-location', 'chla')

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

df['NDCI'] = (df['rhos_704'] - df['rhos_665']) / (df['rhos_704'] + df['rhos_665'])
df['SCD'] = SCD_regr.predict(df['NDCI'].fillna('0', inplace=False).values.reshape(-1, 1))
df.to_csv('../data/level_4/20221007_SCD.csv')

ocean_surface_df = df.loc[(df['water_mask'] == 1)]
print('Maximum Value:')
print(ocean_surface_df['SCD'].max())
print('Minimum Value:')
print(ocean_surface_df['SCD'].min())
hlp.plot_2d_scatter_with_colorbar(ocean_surface_df['lon'], ocean_surface_df['lat'], ocean_surface_df['SCD'], 'Chlorophyll-a (ug/l)', 'Satellite Derived Chlorophyll-a', 'summer_r','chla')
