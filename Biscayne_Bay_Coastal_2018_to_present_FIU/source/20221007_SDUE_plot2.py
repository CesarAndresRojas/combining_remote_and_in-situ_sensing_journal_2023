import json
import os

import cv2
import matplotlib

import helpers as hlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import water_parameter_ranges as constants

# Load data and mask
# data[i][j][k] = [predicted_do, predicted_chla, predicted_temp, predicted_turb]
sdue = np.load('../data/level_4/20221007_SDUE.npy')

# Split data into 4 separate 3D arrays
predicted_do = sdue[:, :, :, 0]
predicted_chla = sdue[:, :, :, 1]
predicted_temp = sdue[:, :, :, 2]
predicted_turb = sdue[:, :, :, 3]

predictions = [
predicted_do,
predicted_chla,
predicted_temp,
predicted_turb
]
titles = [
'3-D Satellite Derived Dissolved Oxygen',
'3-D Satellite Derived Chlorophyll-a',
'3-D Satellite Derived Temperature',
'3-D Satellite Derived Turbidity'
]

units = [
'ODO (mg/L)',
'Chlorophyll-a (ug/l)',
'Temperature (°C)',
'Turbidity FNU'
]

typical_minimums = [
constants.typical_min_do,
constants.typical_min_chla,
constants.typical_min_temp,
constants.typical_min_turbidity
]

typical_maximums = [
constants.typical_max_do,
constants.typical_max_chla,
constants.typical_max_temp,
constants.typical_max_turbidity
]

extreme_minimums = [
constants.extreme_min_do,
constants.extreme_min_chla,
constants.extreme_min_temp,
constants.extreme_min_turbidity
]

extreme_maximums = [
constants.extreme_max_do,
constants.extreme_max_chla,
constants.extreme_max_temp,
constants.extreme_max_turbidity
]

file_names = [
'do',
'chla',
'temp',
'turb'
]


print(predictions[1][predictions[1]>0])
for i in range(len(predictions)):
    data = predictions[i]
    mask = data > 0
    mean = np.mean(data[mask])
    #print(data[mask])
    vmin = np.maximum(data[mask].min(), typical_minimums[i])
    vmax = np.minimum(data[mask].max(), typical_maximums[i])
    print(vmin)
    print(vmax)
    hlp.plot_sdue(data,vmin,vmax,4,titles[i],units[i], file_names[i])
