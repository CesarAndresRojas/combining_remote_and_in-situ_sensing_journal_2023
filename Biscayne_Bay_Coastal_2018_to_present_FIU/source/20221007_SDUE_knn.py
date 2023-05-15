import datetime
import json
import math
import os
import sys

from netCDF4 import Dataset
from shapely.geometry import Polygon, shape
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import rasterio
import rioxarray
import xarray

import geopandas as gpd
import helpers as hlp
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import water_parameter_ranges as constants

os.environ['USE_PYGEOS'] = '0'

insitu_f = '../data/level_1/in-situ/isolated/10072022/100722_SURFACE_WATER_QUALITY_006-010.csv'
insitu_df = pd.read_csv(insitu_f, sep=',', low_memory=False).dropna()

water_column_f_list = ['../data/level_1/in-situ/isolated/10072022/100722_WATER_QUALITY_002.csv',
'../data/level_1/in-situ/isolated/10072022/100722_WATER_QUALITY_003.csv',
'../data/level_1/in-situ/isolated/10072022/100722_WATER_QUALITY_004.csv',
'../data/level_1/in-situ/isolated/10072022/100722_WATER_QUALITY_005.csv']

navigation_chart_f = '../data/level_3/pixEx_S2A-B_MSI_100722_NOAA_CHART_11467_NetCDF_measurements.csv'
navigation_chart_df = pd.read_csv(navigation_chart_f, sep=',', low_memory=False).dropna()
navigation_chart_df['Total Water Column (m)'] = navigation_chart_df['Total Water Column (ft)'] * 0.3048

SBD_f = '../data/level_4/20221007_SBD.csv'
SDoD_f = '../data/level_4/20221007_SDoD.csv'
SCD_f = '../data/level_4/20221007_SCD.csv'
STempD_f = '../data/level_4/20221007_STempD.csv'
STurbD_f = '../data/level_4/20221007_STurbD.csv'

SBD_df = pd.read_csv(SBD_f, sep=',', low_memory=False).dropna()
SDoD_df = pd.read_csv(SDoD_f, sep=',', low_memory=False).dropna()
SCD_df = pd.read_csv(SCD_f, sep=',', low_memory=False).dropna()
STempD_df = pd.read_csv(STempD_f, sep=',', low_memory=False).dropna()
STurbD_df = pd.read_csv(STurbD_f, sep=',', low_memory=False).dropna()

df = SBD_df
df['SDoD'] = SDoD_df['SDoD']
df['SCD'] = SCD_df['SCD']
df['STempD'] = STempD_df['STempD']
df['water_mask2'] = STempD_df['water_mask']
df['STurbD'] = STurbD_df['STurbD']

df['water_mask'] = np.where(df['rhos_833'] > 0.05, 0, 1)

print('Number of rows')
print(df['water_mask'].value_counts())
print('Number of non water pixels')
print(df[df['water_mask'] == 0].index)
print('Number of water pixels')
print(df[df['water_mask'] == 1].index)

print("Water mask for first pixel:")
print(df['water_mask'][0])

print("Water mask for last pixel:")
print(df['lat'][len(df) - 1])
print(df['lon'][len(df) - 1])
print(df['water_mask'][len(df) - 1])

# Extract row and column information
df['x'] = df['x'].subtract(df['x'].min()).divide(10).astype(int)
df['y'] = df['y'].subtract(df['y'].min()).divide(10).astype(int)

print(df.pivot_table(values=['lat', 'lon', 'SBD', 'SDoD', 'SCD', 'STempD', 'STurbD', 'water_mask'], index=['x', 'y'], fill_value=0))

# Remove outliers

# Bathymetry meters
min_depth = navigation_chart_df['Total Water Column (m)'].min()
max_depth = navigation_chart_df['Total Water Column (m)'].max()

# Dissolved Oxygen mg/L
min_DO = insitu_df['ODO mg/L'].min()
max_DO = insitu_df['ODO mg/L'].max()

# Chlorophyll-a ug/L
min_chla = insitu_df['Chlorophyll ug/L'].min()
max_chla = insitu_df['Chlorophyll ug/L'].max()

# Temperature C
min_temp = insitu_df['Temp °C'].min()
max_temp = insitu_df['Temp °C'].max()

# Turbidity FNU
min_turbidity = insitu_df['Turbidity FNU'].min()
max_turbidity = insitu_df['Turbidity FNU'].max()

df.loc[df['SBD'] < constants.extreme_min_depth, 'SBD'] = constants.extreme_min_depth
df.loc[df['SDoD'] < constants.extreme_min_do, 'SDoD'] = constants.extreme_min_do
df.loc[df['SCD'] < constants.extreme_min_chla, 'SCD'] = constants.extreme_min_chla
df.loc[df['STempD'] < constants.extreme_min_temp, 'STempD'] = constants.extreme_min_temp
df.loc[df['STurbD'] < constants.extreme_min_turbidity, 'STurbD'] = constants.extreme_min_turbidity

df.loc[df['SBD'] > max_depth, 'SBD'] = max_depth + 1

# max_chlorophyll = colocated_chlorophyll_df['Chlorophyll ug/L'].max()
# pier_df['SCD'][pier_df['SCD'] > max_chlorophyll] = max_chlorophyll

# Setup 3-D world

# Extract row and column information
rowIDs = df['x']
colIDs = df['y']

# Setup image array and set values into it from "grumpiness" column
A = np.zeros((rowIDs.max() + 1, colIDs.max() + 1, 9))
A[rowIDs, colIDs] = df[['lat', 'lon', 'SBD', 'SDoD', 'SCD', 'STempD', 'STurbD', 'water_mask', 'water_mask2']]
print(A[rowIDs.max()][colIDs.max()])
# ODO,CHLA, TEMP, TURB
n_neighbors_list = [
[2, 2, 2, 2],
[2, 2, 2, 2],
[2, 2, 2, 2],
[2, 2, 2, 2]
]

#model_list = [hlp.create_knn_regression_models(water_column_f, n_neighbors, [True,True,True,True]) for water_column_f, n_neighbors in zip(water_column_f_list, n_neighbors_list)]
model_list = [hlp.create_knn_regression_models(water_column_f, n_neighbors, [True,True,True,True], str(i) +'_') for i, (water_column_f, n_neighbors) in enumerate(zip(water_column_f_list, n_neighbors_list))]

# max_depth = math.ceil(pier_df['SBD'].max())
global_max_depth = constants.extreme_max_depth
B = np.zeros((rowIDs.max() + 1, colIDs.max() + 1, global_max_depth, 4))
for i in range(rowIDs.max() + 1):
    for j in range(colIDs.max() + 1):
        water_mask = A[i][j][7]
        water_mask2 = A[i][j][8]
        if water_mask == 0:
            continue
        lat = A[i][j][0]
        lon = A[i][j][1]
        max_depth = math.ceil(A[i][j][2])
        surface_do = A[i][j][3]
        surface_chla = A[i][j][4]
        surface_temp = A[i][j][5]
        surface_turb = A[i][j][6]
        scaler = MinMaxScaler(feature_range=(0, 1))
        depth_scale = scaler.fit_transform([[i] for i in range(max_depth)])

        model_lat, model_lon, SDoD_knn, SCD_knn, STempD_knn, STurbD_knn = hlp.get_closest_model(lat, lon, model_list)
        
        y_intercept_do = SDoD_knn.predict([[0]])[0]
        y_intercept_chla = SCD_knn.predict([[0]])[0]
        y_intercept_temp = STempD_knn.predict([[0]])[0]
        y_intercept_turb = STurbD_knn.predict([[0]])[0]
        
        for k in range(global_max_depth):
            if max_depth > 0 and k < max_depth: 
                input_list = [[k]]
                # input_scaled = scaler.transform(input_list)
                knn_input = [[depth_scale[k][0]]]
                
                predicted_do = SDoD_knn.predict(knn_input)[0] - y_intercept_do + surface_do
                predicted_do = np.maximum(predicted_do, constants.extreme_min_do)

                predicted_chla = SDoD_knn.predict(knn_input)[0] - y_intercept_chla + surface_chla
                predicted_chla = np.maximum(predicted_chla, constants.extreme_min_chla)

                if water_mask2 == 1:
                    predicted_temp = STempD_knn.predict(knn_input)[0] - y_intercept_temp + surface_temp
                    predicted_temp = np.maximum(predicted_temp, constants.extreme_min_temp)
                else:
                    predicted_temp = 0

                predicted_turb = STurbD_knn.predict(knn_input)[0] - y_intercept_turb + surface_turb
                predicted_turb = np.maximum(predicted_turb, constants.extreme_min_turbidity)

                B[i][j][k] = [predicted_do, predicted_chla, predicted_temp, predicted_turb]

print('Bottom right pixel')
print(B[-1][0])
print(A[-1][0])
predicted_temp = B[:, :, :, 2]
data = predicted_temp[-1][10]
print(data)
my_np_array = np.array(B)
np.save('../data/level_4/20221007_SDUE.npy', my_np_array)
