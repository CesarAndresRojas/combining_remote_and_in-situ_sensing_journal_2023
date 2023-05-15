import datetime
import json
import math
import os
import sys

from matplotlib import cm
from matplotlib.ticker import LinearLocator
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

import helpers as hlp
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import water_parameter_ranges as constants

os.environ['USE_PYGEOS'] = '0'

insitu_f = '../data/level_1/in-situ/isolated/10072022/100722_SURFACE_WATER_QUALITY_006-010.csv'
insitu_df = pd.read_csv(insitu_f, sep=',', low_memory=False).dropna()

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
A = np.zeros((rowIDs.max() + 1, colIDs.max() + 1, 8))
A[rowIDs, colIDs] = df[['lat', 'lon', 'SBD', 'SDoD', 'SCD', 'STempD', 'STurbD', 'water_mask' ]]
print(A[rowIDs.max()][colIDs.max()])

X = np.zeros((rowIDs.max()+1,colIDs.max()+1))
Y = np.zeros((rowIDs.max()+1,colIDs.max()+1))
Z = np.zeros((rowIDs.max()+1,colIDs.max()+1))
for i in range(rowIDs.max()+1):
    for j in range(colIDs.max()+1):
        X[i][j] = A[i][j][1]
        Y[i][j] = A[i][j][0]
        if A[i][j][7] == 1:
            Z[i][j] = A[i][j][2]*-1
        else:
            #Z[i][j] = float('NaN')
            Z[i][j] = 0
        #Z[i][j] = A[i][j][3]*-1
print(Z.shape)
# Set font family to Arial
#plt.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
#X = df['Longitude']
#Y = df['Latitude']
#X, Y = np.meshgrid(X, Y)
#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)
#Z = C

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.ocean,linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-6, 1)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# Set the format of the z tick labels to an integer format
ax.zaxis.set_major_formatter('{:.0f}'.format)
#ax.xaxis.set_major_formatter('{x:.4f}')
#ax.yaxis.set_major_formatter('{x:.4f}')

#ax.set_xlabel('Longitude',fontsize=30,labelpad=-1)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel("Depth (m)", labelpad=-5)

#plt.rc('xtick', labelsize=25)
#plt.rc('ytick', labelsize=25)
# change the fontsize of axes lables
#lt.rc('axes', labelsize=30)

my_xticks = ax.get_xticklabels()
for tick in my_xticks:
    #plt.setp(tick, visible=False,fontsize=20)
    plt.setp(tick, visible=False)
plt.setp(my_xticks[1], visible=True)
plt.setp(my_xticks[-2], visible=True)

my_yticks = ax.get_yticklabels()
for tick in my_yticks:
    plt.setp(tick, visible=False)
plt.setp(my_yticks[1], visible=True)
plt.setp(my_yticks[-2], visible=True)

my_zticks = ax.get_zticklabels()
for tick in my_zticks:
    plt.setp(tick, visible=False)
plt.setp(my_zticks[0], visible=True)
plt.setp(my_zticks[-1], visible=True)

# Add a color bar which maps values to colors.
fig.colorbar(surf, location='top', shrink=0.6, pad=-0.15, aspect=20, orientation='horizontal')#.ax.tick_params(labelsize=20)
#plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

plt.title('Satelite Derived Bathymetry')
plt.savefig('../data/level_4/20221007_SDB.pdf', bbox_inches='tight')
plt.savefig('../data/level_4/20221007_SDB.png', bbox_inches='tight')
plt.savefig('../data/level_4/20221007_SDB.svg', bbox_inches='tight')
plt.show()

#Load data and mask
data =  np.load('../data/level_4/20221007_SDUE.npy')

#data[i][j][k] = [predicted_do, predicted_chla, predicted_temp, predicted_turb]

shape = data.shape
print(shape)
